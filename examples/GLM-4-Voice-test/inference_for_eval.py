# -*- coding: utf-8 -*-
import os
import sys
import json
import torch
import torchaudio
import argparse
import uuid
from transformers import AutoTokenizer, WhisperFeatureExtractor, AutoModel
from speech_tokenizer.modeling_whisper import WhisperVQEncoder
from speech_tokenizer.utils import extract_speech_token
from flow_inference import AudioDecoder
from datasets import load_dataset, load_from_disk
import logging
from tqdm import tqdm
import warnings
import librosa
import jsonlines

warnings.filterwarnings("ignore")
sys.path.insert(0, "./cosyvoice")
sys.path.insert(0, "/data/ruiqi.yan/omni_models/GLM-4-Voice/third_party/Matcha-TTS")


def load_models(args):
    # Load models
    glm_tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True
    )
    whisper_model = (
        WhisperVQEncoder.from_pretrained(args.tokenizer_path).eval().to(args.device)
    )
    feature_extractor = WhisperFeatureExtractor.from_pretrained(args.tokenizer_path)
    audio_decoder = AudioDecoder(
        config_path=os.path.join(args.flow_path, "config.yaml"),
        flow_ckpt_path=os.path.join(args.flow_path, "flow.pt"),
        hift_ckpt_path=os.path.join(args.flow_path, "hift.pt"),
        device=args.device,
    )
    return glm_tokenizer, whisper_model, feature_extractor, audio_decoder


def process_audio(audio, whisper_model, feature_extractor):
    audio_tokens = extract_speech_token(whisper_model, feature_extractor, [audio])[0]
    if len(audio_tokens) == 0:
        raise ValueError("No audio tokens extracted")
    audio_tokens = "".join([f"<|audio_{x}|>" for x in audio_tokens])
    return "<|begin_of_audio|>" + audio_tokens + "<|end_of_audio|>"


def generate_response(
    prompt, glm_tokenizer, glm_model, temperature, top_p, max_new_token
):
    inputs = glm_tokenizer([prompt], return_tensors="pt").to(args.device)
    outputs = glm_model.generate(
        **inputs, max_new_tokens=max_new_token, temperature=temperature, top_p=top_p
    )
    return outputs


def save_results(text, audio, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    text_path = os.path.join(output_dir, "output.txt")
    audio_path = os.path.join(output_dir, "output.wav")
    with open(text_path, "w", encoding="utf-8") as f:
        f.write(text)
    audio = audio.squeeze().cpu()
    torchaudio.save(audio_path, audio.unsqueeze(0), 22050, format="wav")


def main(args):
    # set log
    output_dir = args.output_dir
    output_audio_dir = os.path.join(output_dir, "audio")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    if not os.path.exists(output_audio_dir):
        os.makedirs(output_audio_dir, exist_ok=True)
    val_log = os.path.join(output_dir, "val_log")
    pred_text = os.path.join(output_dir, "pred_text.jsonl")
    question_text = os.path.join(output_dir, "question_text.jsonl")
    gt_text = os.path.join(output_dir, "gt_text.jsonl")
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(name)s][%(levelname)s] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.FileHandler(val_log), logging.StreamHandler()],
    )

    # get model
    logging.info("<========loading model========>")
    glm_tokenizer, whisper_model, feature_extractor, audio_decoder = load_models(args)
    glm_model = (
        AutoModel.from_pretrained(args.model_path, trust_remote_code=True)
        .eval()
        .to(args.device)
    )

    # get dataset
    logging.info(
        f"<========get dataset {args.val_data_name} from {args.val_data_path}========>"
    )
    data_list = []
    if args.manifest_format == "datasets":
        ds = load_dataset(args.val_data_path, args.val_data_name)
        if args.val_data_name == "sd-qa":
            data_list = ds["usa"]
        else:
            data_list = ds["test"]
    elif args.manifest_format == "jsonl":
        with open(args.val_data_path, "r") as f:
            for data in jsonlines.Reader(f):
                data_list.append(data)

    # do inference
    with jsonlines.open(pred_text, mode="w") as pt, jsonlines.open(
        gt_text, mode="w"
    ) as gt, jsonlines.open(question_text, mode="w") as qt:
        for step, s in enumerate(
            tqdm(
                data_list,
                total=len(data_list),
                desc=f"evaluating on {args.val_data_name}",
            )
        ):
            if args.manifest_format == "datasets":
                audio_raw, sr, source_text = (
                    s["audio"]["array"],
                    s["audio"]["sampling_rate"],
                    s["prompt"],
                )
                if "reference" in s:
                    target_text = s["reference"]
                else:
                    target_text = s["prompt"]
            elif args.manifest_format == "jsonl":
                source_wav = os.path.join(
                    os.path.dirname(args.val_data_path), s["source_wav"]
                )
                source_text = s["source_text"]
                if "target_text" in s.keys():
                    target_text = s["target_text"]
                else:
                    target_text = s["source_text"]
                audio_raw, sr = librosa.load(source_wav)
            audio_raw = torch.from_numpy(audio_raw).unsqueeze(0)
            user_input = process_audio(
                (audio_raw, sr), whisper_model, feature_extractor
            )
            system_prompt = "User will provide you with a speech instruction. Do it step by step. First, think about the instruction and respond in a interleaved manner, with 13 text token followed by 26 audio tokens. "
            prompt = f"<|system|>\n{system_prompt}<|user|>\n{user_input}<|assistant|>streaming_transcription\n"
            response = generate_response(
                prompt,
                glm_tokenizer,
                glm_model,
                args.temperature,
                args.top_p,
                args.max_new_token,
            )
            response_list = response[0].tolist()

            text_tokens, audio_tokens = [], []
            audio_offset = glm_tokenizer.convert_tokens_to_ids("<|audio_0|>")
            end_of_question_tokens = glm_tokenizer.encode("streaming_transcription\n")[
                -5:
            ]

            start_index = next(
                i
                for i in range(len(response_list) - len(end_of_question_tokens) + 1)
                if response_list[i : i + len(end_of_question_tokens)]
                == end_of_question_tokens
            ) + len(end_of_question_tokens)

            for token_id in response[0][start_index:]:
                if token_id >= audio_offset:
                    audio_tokens.append(token_id - audio_offset)
                else:
                    text_tokens.append(token_id)

            tts_token = torch.tensor(audio_tokens, device=args.device).unsqueeze(0)
            tts_speech, _ = audio_decoder.token2wav(
                tts_token, uuid=str(uuid.uuid4()), finalize=True
            )
            complete_text = glm_tokenizer.decode(
                text_tokens, spaces_between_special_tokens=False
            )
            complete_text = complete_text.split("<|user|>")[0].strip()
            logging.info(f"input text: {source_text}")
            logging.info(f"output text: {complete_text}")

            qt.write({str(step).zfill(4): source_text})
            pt.write({str(step).zfill(4): complete_text})
            if isinstance(target_text, list):
                gt.write({str(step).zfill(4): " / ".join(target_text)})
            else:
                gt.write({str(step).zfill(4): target_text})

            audio_path = os.path.join(output_audio_dir, f"{step:04d}.wav")
            audio = tts_speech.squeeze().cpu()
            torchaudio.save(audio_path, audio.unsqueeze(0), 22050, format="wav")
        # save_results(complete_text, tts_speech.squeeze(), args.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-mode", type=str, choices=["audio", "text"], required=True
    )
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--val_data_name", type=str, required=True)
    parser.add_argument("--val_data_path", type=str, required=True)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=0.8)
    parser.add_argument("--max_new_token", type=int, default=2000)
    parser.add_argument(
        "--flow-path", type=str, default="../GLM-4-Voice/glm-4-voice-decoder"
    )
    parser.add_argument("--model-path", type=str, default="THUDM/glm-4-voice-9b")
    parser.add_argument(
        "--tokenizer-path", type=str, default="THUDM/glm-4-voice-tokenizer"
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--manifest_format",
        type=str,
        choices=["datasets", "jsonl"],
        help="validation dataset format, must be datasets (from hf) or jsonl",
    )
    args = parser.parse_args()

    main(args)
