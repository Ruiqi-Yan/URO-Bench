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

sys.path.insert(0, "./cosyvoice")
sys.path.insert(0, "/data/xiquan.li/GLM-4-Voice/third_party/Matcha-TTS")

def load_models(args):
    # Load models
    glm_tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    whisper_model = WhisperVQEncoder.from_pretrained(args.tokenizer_path).eval().to(args.device)
    feature_extractor = WhisperFeatureExtractor.from_pretrained(args.tokenizer_path)
    audio_decoder = AudioDecoder(
        config_path=os.path.join(args.flow_path, "config.yaml"),
        flow_ckpt_path=os.path.join(args.flow_path, 'flow.pt'),
        hift_ckpt_path=os.path.join(args.flow_path, 'hift.pt'),
        device=args.device
    )
    return glm_tokenizer, whisper_model, feature_extractor, audio_decoder

def process_audio(audio_path, whisper_model, feature_extractor):
    audio_tokens = extract_speech_token(whisper_model, feature_extractor, [audio_path])[0]
    if len(audio_tokens) == 0:
        raise ValueError("No audio tokens extracted")
    audio_tokens = "".join([f"<|audio_{x}|>" for x in audio_tokens])
    return "<|begin_of_audio|>" + audio_tokens + "<|end_of_audio|>"

def generate_response(prompt, glm_tokenizer, glm_model, temperature, top_p, max_new_token):
    inputs = glm_tokenizer([prompt], return_tensors="pt").to(args.device)
    outputs = glm_model.generate(
        **inputs,
        max_new_tokens=max_new_token,
        temperature=temperature,
        top_p=top_p
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
    glm_tokenizer, whisper_model, feature_extractor, audio_decoder = load_models(args)
    glm_model = AutoModel.from_pretrained(args.model_path, trust_remote_code=True).eval().to(args.device)

    if args.input_mode == "audio":
        user_input = process_audio(args.input_path, whisper_model, feature_extractor)
        system_prompt = "User will provide you with a speech instruction. Do it step by step. First, think about the instruction and respond in a interleaved manner, with 13 text token followed by 26 audio tokens. "
    else:
        with open(args.input_path, "r", encoding="utf-8") as f:
            user_input = f.read().strip()
        system_prompt = "User will provide you with a text instruction. Do it step by step. First, think about the instruction and respond in a interleaved manner, with 13 text token followed by 26 audio tokens."

    prompt = f"<|system|>\n{system_prompt}<|user|>\n{user_input}<|assistant|>streaming_transcription\n"
    response = generate_response(prompt, glm_tokenizer, glm_model, args.temperature, args.top_p, args.max_new_token)
    response_list = response[0].tolist()

    text_tokens, audio_tokens = [], []
    audio_offset = glm_tokenizer.convert_tokens_to_ids('<|audio_0|>')
    end_of_question_tokens = glm_tokenizer.encode('streaming_transcription\n')[-5:]

    start_index = next(i for i in range(len(response_list) - len(end_of_question_tokens) + 1)
                       if response_list[i:i+len(end_of_question_tokens)] == end_of_question_tokens) + len(end_of_question_tokens)

    for token_id in response[0][start_index:]:
        if token_id >= audio_offset:
            audio_tokens.append(token_id - audio_offset)
        else:
            text_tokens.append(token_id)

    tts_token = torch.tensor(audio_tokens, device=args.device).unsqueeze(0)
    tts_speech, _ = audio_decoder.token2wav(tts_token, uuid=str(uuid.uuid4()), finalize=True)
    complete_text = glm_tokenizer.decode(text_tokens, spaces_between_special_tokens=False)
    complete_text = complete_text.split("<|user|>")[0].strip()

    save_results(complete_text, tts_speech.squeeze(), args.output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-mode", type=str, choices=["audio", "text"], required=True)
    parser.add_argument("--input-path", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=0.8)
    parser.add_argument("--max_new_token", type=int, default=2000)
    parser.add_argument("--flow-path", type=str, default="../GLM-4-Voice/glm-4-voice-decoder")
    parser.add_argument("--model-path", type=str, default="THUDM/glm-4-voice-9b")
    parser.add_argument("--tokenizer-path", type=str, default="THUDM/glm-4-voice-tokenizer")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    main(args)

# Run the inference script with the following command:
# text input
# python inference.py --input-mode text --input-path test.txt --output-dir single-output-text
# audio input
# python inference.py --input-mode audio --input-path test.wav --output-dir single-output-audio

# debug
# python -m debugpy --listen 5678 --wait-for-client inference.py --input-mode audio --input-path test.wav --output-dir single-output-audio

# 批处理的时候修改 main 函数即可