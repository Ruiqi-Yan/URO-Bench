import os
import torch
import soundfile as sf
import numpy as np

import logging
from argparse import ArgumentParser
import jsonlines

import torch, math
from transformers import (
    MoshiForConditionalGeneration,
    AutoFeatureExtractor,
    AutoTokenizer,
)
import librosa
import time


class VoiceAssistant:
    @torch.no_grad()
    def generate_audio(
        self,
        audio,
        sr,
        max_new_tokens=2048,
    ):
        raise NotImplementedError

    @torch.no_grad()
    def generate_text(
        self,
        text,
    ):
        raise NotImplementedError

    @torch.no_grad()
    def generate_ttft(
        self,
        audio,
    ):
        tmp = time.perf_counter()
        self.generate_audio(audio, max_new_tokens=1)
        return time.perf_counter() - tmp


class MoshiAssistant(VoiceAssistant):
    def __init__(self):
        self.device = "cuda"
        self.dtype = torch.float16
        self.model = MoshiForConditionalGeneration.from_pretrained(
            "kmhf/hf-moshiko",
            device_map=self.device,
            torch_dtype=self.dtype,
            cache_dir="./cache",
        )
        self.tokenizer = AutoTokenizer.from_pretrained("kmhf/hf-moshiko")
        self.feature_extractor = AutoFeatureExtractor.from_pretrained("kmhf/hf-moshiko")

    def generate_audio(
        self,
        audio,
        sr,
        max_new_tokens=2048,
    ):
        audio = librosa.resample(
            audio, orig_sr=sr, target_sr=self.feature_extractor.sampling_rate
        )

        user_input_values = self.feature_extractor(
            raw_audio=audio,
            sampling_rate=self.feature_extractor.sampling_rate,
            return_tensors="pt",
        ).to(device=self.device, dtype=self.dtype)

        # prepare moshi input values - we suppose moshi didn't say anything while the user spoke
        moshi_input_values = torch.zeros_like(user_input_values.input_values)

        ratio = (
            self.model.config.audio_encoder_config.frame_rate
            / self.model.config.sampling_rate
        )

        # prepare moshi input ids - we suppose moshi didn't say anything while the user spoke
        num_tokens = math.ceil(moshi_input_values.shape[-1] * ratio)
        input_ids = (
            torch.ones((1, num_tokens), device=self.device, dtype=torch.int64)
            * self.tokenizer.encode("<pad>")[0]
        )

        output = self.model.generate(
            input_ids=input_ids,
            user_input_values=user_input_values.input_values,
            moshi_input_values=moshi_input_values,
            max_new_tokens=max_new_tokens,
            return_audio_waveforms=True,
        )

        text_tokens = output.sequences.cpu().numpy()
        audio_waveforms = output.audio_sequences.cpu().numpy()

        response = self.tokenizer.batch_decode(text_tokens, skip_special_tokens=True)[0]

        return response, audio_waveforms


def main():
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    # Set log
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(name)s][%(levelname)s] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # inference
    device = "cuda:0"
    output_dir = args.output_dir
    output_audio_dir = os.path.join(output_dir, "audio")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    if not os.path.exists(output_audio_dir):
        os.makedirs(output_audio_dir, exist_ok=True)
    pred_text = os.path.join(output_dir, "pred_text.jsonl")
    question_text = os.path.join(output_dir, "question_text.jsonl")
    gt_text = os.path.join(output_dir, "gt_text.jsonl")

    logging.info("<========loading model========>")
    model = MoshiAssistant()

    logging.info("<========inference starts========>")

    with open(args.dataset, "r") as f, jsonlines.open(
        pred_text, mode="w"
    ) as pt, jsonlines.open(question_text, mode="w") as qt, jsonlines.open(
        gt_text, mode="w"
    ) as gt:
        for step, item in enumerate(jsonlines.Reader(f)):
            input_path = os.path.join(os.path.dirname(args.dataset), item["source_wav"])
            input_text = item["source_text"]
            if "target_text" in item:
                target_text = item["target_text"]
            else:
                target_text = item["source_text"]

            assert os.path.exists(input_path), f"audio file {input_path} not found"
            audio, sr = sf.read(input_path)
            text, audio_waveforms = model.generate_audio(audio=audio, sr=sr)
            sf.write(
                f"{output_audio_dir}/{step:04d}.wav",
                audio_waveforms,
                24000,
            )

            logging.info(f"Input text: {input_text}")
            logging.info(f"Output text: {text}")
            logging.info(f"output audio saved to {output_audio_dir}/{step:04d}.wav")
            pt.write({str(step).zfill(4): text})
            qt.write({str(step).zfill(4): input_text})
            if isinstance(target_text, list):
                gt.write({str(step).zfill(4): " / ".join(target_text)})
            else:
                gt.write({str(step).zfill(4): target_text})


if __name__ == "__main__":
    main()

# python /data/ruiqi.yan/URO-Bench/examples/moshi/inference_for_eval.py \
#         --dataset /data/ruiqi.yan/data/final/alpacaeval_test/test.jsonl \
#         --output_dir /data/ruiqi.yan/omni_models/moshi-test
