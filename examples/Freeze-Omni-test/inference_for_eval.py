from __future__ import print_function

import argparse
import os
import json
import queue
import torch
import yaml
import threading
import struct
import time
import torchaudio
import datetime
import builtins
import math

import soundfile as sf
import numpy as np
import torch.nn.functional as F
import torchaudio.compliance.kaldi as k

from torch.utils.data import DataLoader

from models.pipeline import inferencePipeline
from models.decoder.llm2tts import llm2TTS
from web.parms import GlobalParams
from web.pool import TTSObjectPool

import jsonlines
import logging


def get_args():
    parser = argparse.ArgumentParser(description="Freeze-Omni")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_path", required=True, help="model_path to load")
    parser.add_argument("--llm_path", required=True, help="llm_path to load")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--top_p", type=float, default=0.8)
    parser.add_argument("--temperature", type=float, default=0.7)

    args = parser.parse_args()
    print(args)
    return args


class audioEncoderProcessor:
    def __init__(self, chunk_size=16):
        self.chunk_size = 16
        self.chunk_overlap = 3
        self.feat_dim = 80
        self.frame_size = 400
        self.frame_shift = 160
        self.frame_overlap = self.frame_size - self.frame_shift
        self.CHUNK = self.frame_shift * self.chunk_size
        self.reset()

    def get_chunk_size(self):
        return self.CHUNK

    def reset(self):
        self.input_chunk = torch.zeros(
            [1, self.chunk_size + self.chunk_overlap, self.feat_dim]
        )
        self.input_sample = torch.zeros([1, self.CHUNK + self.frame_overlap, 1])

    def fbank_shift(self, sample_data):
        # fbank feature shift
        self.input_sample[:, : self.frame_overlap, :] = self.input_sample[
            :, -self.frame_overlap :, :
        ].clone()
        self.input_sample[:, self.frame_overlap :, :] = sample_data

    def chunk_data_shift(self, xs):
        # chunk feature shift
        self.input_chunk[:, : self.chunk_overlap, :] = self.input_chunk[
            :, -self.chunk_overlap :, :
        ].clone()
        self.input_chunk[:, self.chunk_overlap :, :] = xs.squeeze(0)

    def process(self, audio: torch.Tensor):
        with torch.no_grad():
            sample_data = torch.tensor(audio).reshape(1, -1, 1)[:, :, :1] * 32768
            self.fbank_shift(sample_data)
            # use kaldi api to compute fbank
            xs = k.fbank(
                waveform=self.input_sample.squeeze(-1),
                dither=0,
                frame_length=25,
                frame_shift=10,
                num_mel_bins=self.feat_dim,
            )
            self.chunk_data_shift(xs)
        return self.input_chunk.clone()


def decoder(
    cur_hidden_state,
    pipeline,
    cur_text,
    tts,
    codec_chunk_size,
    codec_padding_size,
    decoder_topk,
    wav,
):
    hidden_state_output = torch.cat(cur_hidden_state).squeeze(1)
    cur_text_procced = pipeline.post_process(cur_text)
    print("Synthesis: ", [cur_text_procced])
    # fix some bugs
    if len(cur_text_procced) == 0:
        return
    embeddings = pipeline.model.llm_decoder.model.embed_tokens(
        torch.tensor(pipeline.model.tokenizer.encode(cur_text_procced)).cuda()
    )
    for seg in tts.run(
        embeddings.reshape(-1, 896).unsqueeze(0),
        decoder_topk,
        hidden_state_output.reshape(-1, 896).unsqueeze(0),
        codec_chunk_size,
        codec_padding_size,
    ):
        wav.append(seg)


def inference(pipeline, audio_processor, tts, input_wav_path, output_wav_path):
    """
    Perform inference for a speech dialogue system.

    Parameters:
    - pipeline: Speech dialogue pipeline.
    - audio_processor: Processes raw audio data into a format suitable for the pipeline.
    - tts: The speech decoder moudule.
    - input_wav_path: Path to the input wav file.
    - output_wav_path: Path to the output wav file.

    Returns:
    - None
    """
    wav, fs = sf.read(input_wav_path)
    wav = torch.tensor(wav)
    if fs != 16000:
        wav = torchaudio.transforms.Resample(orig_freq=fs, new_freq=16000)(wav.float())
        fs = 16000

    codec_chunk_size = 40
    codec_padding_size = 10
    decoder_topk = 2

    # Satge0: preprocess
    # set system role, stat will be set to 'sl'
    stat = "pre"
    outputs = pipeline.speech_dialogue(
        None, stat=stat, role="You are a helpful assistant."
    )
    chunk_size = audio_processor.get_chunk_size()

    # Satge1: start listen
    # stat will be auto set to 'cl' after Stage1
    wav_input = torch.zeros(math.ceil(wav.shape[0] / chunk_size) * chunk_size)
    # fix some bugs
    try:
        wav_input[: wav.shape[0]] = wav
    except:
        print("error")
        return " "
    for i in range(0, wav_input.shape[0], chunk_size):
        fbank = audio_processor.process(wav_input[i : i + chunk_size])
        outputs = pipeline.speech_dialogue(fbank, **outputs)
        outputs["stat"] = "cl"
    audio_processor.reset()

    outputs["adapter_cache"] = None
    outputs["encoder_cache"] = None
    outputs["pe_index"] = 0
    outputs["stat"] = "ss"

    # Stage3: start speak
    outputs = pipeline.speech_dialogue(None, **outputs)
    cur_hidden_state = []
    cur_hidden_state.append(outputs["hidden_state"])

    whole_text = ""
    last_text = ""
    cur_text = ""
    wav = []
    # Stage4: contiune speak until stat is set to 'sl'
    # use 'stop' to interrupt generation, stat need to be manually set as 'sl'
    stop = False
    while True:
        if len(outputs["past_tokens"]) > 128:
            stop = True
        if stop:
            break
        del outputs["text"]
        del outputs["hidden_state"]
        outputs = pipeline.speech_dialogue(None, **outputs)
        if outputs["stat"] == "cs":
            cur_hidden_state.append(outputs["hidden_state"])
            whole_text += outputs["text"][len(last_text) :]
            cur_text += outputs["text"][len(last_text) :]
            suffix_list = ["。", "：", "？", "！", ".", "?", "!", "\n"]
            if outputs["text"][len(last_text) :].endswith(tuple(suffix_list)):
                # fix string index out of range bugs
                if (
                    outputs["text"][len(last_text) :].endswith(".")
                    and len(last_text) != 0
                    and last_text[-1].isdigit()
                ):
                    pass
                else:
                    if len(cur_hidden_state) > 0:
                        decoder(
                            cur_hidden_state,
                            pipeline,
                            cur_text,
                            tts,
                            codec_chunk_size,
                            codec_padding_size,
                            decoder_topk,
                            wav,
                        )
                        cur_hidden_state = []
                    cur_text = ""
        if outputs["stat"] == "sl":
            break
        # print(outputs['text'])
        last_text = outputs["text"]
    if len(cur_hidden_state) != 0:
        decoder(
            cur_hidden_state,
            pipeline,
            cur_text,
            tts,
            codec_chunk_size,
            codec_padding_size,
            decoder_topk,
            wav,
        )

    # fix some bugs
    if len(wav) == 0:
        print("error")
        return whole_text
    sf.write(output_wav_path, torch.cat(wav, -1).squeeze().float().cpu().numpy(), 24000)
    outputs["stat"] = "sl"
    outputs["last_id"] = None
    print(whole_text)
    return whole_text


if __name__ == "__main__":
    configs = get_args()

    # Set log
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(name)s][%(levelname)s] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # inference
    output_dir = configs.output_dir
    output_audio_dir = os.path.join(output_dir, "audio")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    if not os.path.exists(output_audio_dir):
        os.makedirs(output_audio_dir, exist_ok=True)
    pred_text = os.path.join(output_dir, "pred_text.jsonl")
    question_text = os.path.join(output_dir, "question_text.jsonl")
    gt_text = os.path.join(output_dir, "gt_text.jsonl")

    logging.info("<========loading model========>")
    pipeline = inferencePipeline(configs)
    tts = llm2TTS(configs.model_path)
    audio_processor = audioEncoderProcessor()

    # 做批量处理的时候这里加个循环就好了
    logging.info("<========inference starts========>")
    with open(configs.dataset, "r") as f, jsonlines.open(
        pred_text, mode="w"
    ) as pt, jsonlines.open(question_text, mode="w") as qt, jsonlines.open(
        gt_text, mode="w"
    ) as gt:
        for step, item in enumerate(jsonlines.Reader(f)):
            input_path = os.path.join(
                os.path.dirname(configs.dataset), item["source_wav"]
            )
            output_path = os.path.join(output_audio_dir, str(step).zfill(4) + ".wav")
            input_text = item["source_text"]
            if "target_text" in item:
                target_text = item["target_text"]
            else:
                target_text = item["source_text"]
            output = inference(pipeline, audio_processor, tts, input_path, output_path)
            logging.info(f"Input text: {input_text}")
            logging.info(f"Output text: {output}")
            logging.info(f"output audio saved to {output_audio_dir}/{step:04d}.wav")
            pt.write({str(step).zfill(4): output})
            qt.write({str(step).zfill(4): input_text})
            if isinstance(target_text, list):
                gt.write({str(step).zfill(4): " / ".join(target_text)})
            else:
                gt.write({str(step).zfill(4): target_text})


# python bin/inference.py \
#     --model_path ./checkpoints \
#     --llm_path ./Qwen2-7B-Instruct \
#     --input_wav $input_wav \
#     --output_wav $output_wav \
#     --top_k 20 \
#     --top_p 0.8 \
#     --temperature 0.8
