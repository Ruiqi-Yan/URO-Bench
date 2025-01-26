# -*- coding: utf-8 -*-
import os
import sys
import json
import torch
import torchaudio
import argparse
import uuid
import codecs
from transformers import AutoTokenizer, WhisperFeatureExtractor, AutoModel
from speech_tokenizer.modeling_whisper import WhisperVQEncoder
from speech_tokenizer.utils import extract_speech_token
from flow_inference import AudioDecoder
from tqdm import tqdm

sys.path.insert(0, "./cosyvoice")
sys.path.insert(0, "./third_party/Matcha-TTS")


def load_models(args):
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
    glm_model = (
        AutoModel.from_pretrained(args.model_path, trust_remote_code=True)
        .eval()
        .to(args.device)
    )
    return glm_tokenizer, whisper_model, feature_extractor, audio_decoder, glm_model


def process_audio(audio_path, whisper_model, feature_extractor):
    audio_tokens = extract_speech_token(whisper_model, feature_extractor, [audio_path])[
        0
    ]
    if len(audio_tokens) == 0:
        raise ValueError("No audio tokens extracted")
    audio_tokens = "".join([f"<|audio_{x}|>" for x in audio_tokens])
    return "<|begin_of_audio|>" + audio_tokens + "<|end_of_audio|>"


def generate_response(
    prompt, glm_tokenizer, glm_model, temperature, top_p, max_new_token, device
):
    inputs = glm_tokenizer([prompt], return_tensors="pt").to(device)
    outputs = glm_model.generate(
        **inputs, max_new_tokens=max_new_token, temperature=temperature, top_p=top_p
    )
    return outputs


def parse_response(response, glm_tokenizer, audio_decoder, device):
    response_list = response[0].tolist()
    end_of_question_tokens = glm_tokenizer.encode(
        "streaming_transcription\n", add_special_tokens=False
    )
    start_index = None
    for i in range(len(response_list) - len(end_of_question_tokens), -1, -1):
        if response_list[i : i + len(end_of_question_tokens)] == end_of_question_tokens:
            start_index = i + len(end_of_question_tokens)
            break

    if start_index is None:
        start_index = 0

    answer_tokens = response_list[start_index:]
    audio_offset = glm_tokenizer.convert_tokens_to_ids("<|audio_0|>")
    text_tokens = []
    audio_tokens = []

    for token_id in answer_tokens:
        if token_id >= audio_offset:
            audio_tokens.append(token_id - audio_offset)
        else:
            text_tokens.append(token_id)

    tts_token = torch.tensor(audio_tokens, device=device).unsqueeze(0)
    tts_speech, _ = audio_decoder.token2wav(
        tts_token, uuid=str(uuid.uuid4()), finalize=True
    )

    complete_text = glm_tokenizer.decode(
        text_tokens, spaces_between_special_tokens=False
    )
    if "<|user|>" in complete_text:
        complete_text = complete_text.split("<|user|>")[0].strip()

    return complete_text, tts_speech


def main(args):
    glm_tokenizer, whisper_model, feature_extractor, audio_decoder, glm_model = (
        load_models(args)
    )

    output_jsonl_path = os.path.join(args.output_dir, "output_with_text.jsonl")
    os.makedirs(args.output_dir, exist_ok=True)

    with codecs.open(args.input_jsonl, "r", "utf-8") as fin, open(
        output_jsonl_path, "a", encoding="utf-8"
    ) as fout:
        lines = fin.readlines()
        for line_idx, line in enumerate(tqdm(lines, desc="Processing dialogues")):
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            dialogue_id = data["id"]
            num_round = data["num_round"]
            dialogue = data["dialogue"]

            system_prompt_audio = "User will provide you with a speech instruction. Do it step by step. First, think about the instruction and respond in a interleaved manner, with 13 text token followed by 26 audio tokens."
            system_prompt_text = "User will provide you with a text instruction. Do it step by step. First, think about the instruction and respond in a interleaved manner, with 13 text token followed by 26 audio tokens."

            first_round_mode = "audio" if dialogue[0].get("source_wav") else "text"
            system_prompt = (
                system_prompt_audio
                if first_round_mode == "audio"
                else system_prompt_text
            )

            prompt = f"<|system|>\n{system_prompt}"

            for round_idx, round_data in enumerate(dialogue):
                tqdm.write(f"Processing line {line_idx + 1}, round {round_idx + 1}")
                user_text = round_data.get("source_text", "")
                user_wav = round_data.get("source_wav")

                if user_wav and os.path.exists(user_wav):
                    user_input = process_audio(
                        user_wav, whisper_model, feature_extractor
                    )
                else:
                    user_input = user_text

                prompt += (
                    f"<|user|>\n{user_input}<|assistant|>streaming_transcription\n"
                )
                response = generate_response(
                    prompt,
                    glm_tokenizer,
                    glm_model,
                    args.temperature,
                    args.top_p,
                    args.max_new_token,
                    args.device,
                )
                complete_text, tts_speech = parse_response(
                    response, glm_tokenizer, audio_decoder, args.device
                )
                prompt += complete_text

                id_dir = os.path.join(args.output_dir, str(dialogue_id))
                os.makedirs(id_dir, exist_ok=True)
                audio_path = os.path.join(id_dir, f"chat_{round_data['round']}.wav")
                audio = tts_speech.squeeze().cpu()
                torchaudio.save(audio_path, audio.unsqueeze(0), 22050, format="wav")

                round_data["output_text"] = complete_text

            fout.write(
                json.dumps(
                    {"id": dialogue_id, "num_round": num_round, "dialogue": dialogue},
                    ensure_ascii=False,
                )
                + "\n"
            )


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-jsonl",
        type=str,
        required=True,
        help="Path to the multi-round dialogue JSONL file.",
    )
    parser.add_argument(
        "--output-dir", type=str, required=True, help="Output directory."
    )
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=0.8)
    parser.add_argument("--max_new_token", type=int, default=2000)
    parser.add_argument("--flow-path", type=str, default="./glm-4-voice-decoder")
    parser.add_argument("--model-path", type=str, default="THUDM/glm-4-voice-9b")
    parser.add_argument(
        "--tokenizer-path", type=str, default="THUDM/glm-4-voice-tokenizer"
    )
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
