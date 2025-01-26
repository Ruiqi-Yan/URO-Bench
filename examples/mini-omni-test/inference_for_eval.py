import os
import torch
import soundfile as sf
from snac import SNAC
from litgpt import Tokenizer
from litgpt.generate.base import generate_AA, generate_TA
from litgpt.model import GPT, Config
from lightning.fabric.utilities.load import _lazy_load as lazy_load
from utils.snac_utils import (
    layershift,
    reconscruct_snac,
    reconstruct_tensors,
    get_time_str,
)
from inference_demo import (
    load_audio,
    get_input_ids_whisper,
    get_input_ids_TA,
    download_model,
)
import lightning as L
import whisper
import logging
from argparse import ArgumentParser
import jsonlines

torch.set_printoptions(sci_mode=False)
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# TODO
text_vocabsize = 151936
text_specialtokens = 64
audio_vocabsize = 4096
audio_specialtokens = 64

padded_text_vocabsize = text_vocabsize + text_specialtokens
padded_audio_vocabsize = audio_vocabsize + audio_specialtokens

_eot = text_vocabsize
_pad_t = text_vocabsize + 1
_input_t = text_vocabsize + 2
_answer_t = text_vocabsize + 3
_asr = text_vocabsize + 4

_eoa = audio_vocabsize
_pad_a = audio_vocabsize + 1
_input_a = audio_vocabsize + 2
_answer_a = audio_vocabsize + 3
_split = audio_vocabsize + 4


def load_model(ckpt_dir, device):
    snacmodel = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval().to(device)
    whispermodel = whisper.load_model("small").to(device)
    text_tokenizer = Tokenizer(ckpt_dir)
    fabric = L.Fabric(devices=1, strategy="auto")
    config = Config.from_file(ckpt_dir + "/model_config.yaml")
    config.post_adapter = False

    with fabric.init_module(empty_init=False):
        model = GPT(config)

    model = fabric.setup(model)
    state_dict = lazy_load(ckpt_dir + "/lit_model.pth")
    model.load_state_dict(state_dict, strict=True)
    model.to(device).eval()

    return fabric, model, text_tokenizer, snacmodel, whispermodel


def A1_A2(
    fabric,
    audio_feature,
    input_ids,
    leng,
    model,
    text_tokenizer,
    step,
    snacmodel,
    out_dir,
):
    with fabric.init_tensor():
        model.set_kv_cache(batch_size=1)
    tokenlist = generate_AA(
        model,
        audio_feature,
        input_ids,
        [leng],
        ["A1T2"],
        max_returned_tokens=2048,
        temperature=0.9,
        top_k=1,
        eos_id_a=_eoa,
        eos_id_t=_eot,
        pad_id_t=_pad_t,
        shift=padded_text_vocabsize,
        include_prompt=True,
        generate_text=True,
    )
    audiolist = reconscruct_snac(tokenlist)
    tokenlist = tokenlist[-1]
    if text_vocabsize in tokenlist:
        tokenlist = tokenlist[: tokenlist.index(text_vocabsize)]
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    audio = reconstruct_tensors(audiolist)
    # fix some bugs
    for item in audio:
        item[item > 4095] = 4095
    with torch.inference_mode():
        audio_hat = snacmodel.decode(audio)
    sf.write(
        f"{out_dir}/audio/{step:04d}.wav",
        audio_hat.squeeze().cpu().numpy(),
        24000,
    )
    model.clear_kv_cache()
    return text_tokenizer.decode(torch.tensor(tokenlist)).strip()


def T1_A2(fabric, input_ids, model, text_tokenizer, step, snacmodel, out_dir):
    with fabric.init_tensor():
        model.set_kv_cache(batch_size=1)
    tokenlist = generate_TA(
        model,
        None,
        input_ids,
        None,
        ["T1A2"],
        max_returned_tokens=2048,
        temperature=0.9,
        top_k=1,
        eos_id_a=_eoa,
        eos_id_t=_eot,
        pad_id_t=_pad_t,
        shift=padded_text_vocabsize,
        include_prompt=True,
        generate_text=True,
    )

    audiolist = reconscruct_snac(tokenlist)
    tokenlist = tokenlist[-1]

    if text_vocabsize in tokenlist:
        tokenlist = tokenlist[: tokenlist.index(text_vocabsize)]
    audio = reconstruct_tensors(audiolist)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    with torch.inference_mode():
        audio_hat = snacmodel.decode(audio)
    sf.write(
        f"{out_dir}/audio/{step:04d}.wav",
        audio_hat.squeeze().cpu().numpy(),
        24000,
    )
    model.clear_kv_cache()
    return text_tokenizer.decode(torch.tensor(tokenlist)).strip()


def load_from_checkpoint(device, ckpt_dir=None):
    if ckpt_dir == None:
        ckpt_dir = os.path.join(os.getcwd(), "checkpoint")
    if not os.path.exists(ckpt_dir):
        print(
            f"checkpoint directory {ckpt_dir} not found, downloading from huggingface"
        )
        download_model(ckpt_dir)

    fabric, model, text_tokenizer, snacmodel, whispermodel = load_model(
        ckpt_dir, device
    )
    return fabric, model, text_tokenizer, snacmodel, whispermodel


def main():
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--ckpt_dir", type=str, required=True)
    parser.add_argument("--mode", default="A1A2")
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
    mode = args.mode

    logging.info("<========loading model========>")
    fabric, model, text_tokenizer, snacmodel, whispermodel = load_from_checkpoint(
        device, args.ckpt_dir
    )
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
            if mode == "A1A2":
                assert os.path.exists(input_path), f"audio file {input_path} not found"
                mel, leng = load_audio(input_path)
                audio_feature, input_ids = get_input_ids_whisper(
                    mel, leng, whispermodel, device
                )
                text = A1_A2(
                    fabric,
                    audio_feature,
                    input_ids,
                    leng,
                    model,
                    text_tokenizer,
                    step,
                    snacmodel,
                    output_dir,
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
            elif mode == "T1A2":
                input_ids = get_input_ids_TA(input_text, text_tokenizer)
                text_output = T1_A2(
                    fabric,
                    input_ids,
                    model,
                    text_tokenizer,
                    step,
                    snacmodel,
                    output_dir,
                )
                logging.info(f"Input text: {input_text}")
                logging.info(f"Output text: {text_output}")
                logging.info(f"output audio saved to {output_audio_dir}/{step:04d}.wav")
                pt.write({str(step).zfill(4): text_output})
                qt.write({str(step).zfill(4): input_text})
                if isinstance(target_text, list):
                    gt.write({str(step).zfill(4): " / ".join(target_text)})
                else:
                    gt.write({str(step).zfill(4): target_text})
            else:
                logging.warning(
                    "Invalid mode selected. Please choose either 'A1A2' or 'T1A2'."
                )


if __name__ == "__main__":
    main()

# A1A2 / T1A2
# python inference.py
