import os
import torch
import soundfile as sf
from snac import SNAC
from litgpt import Tokenizer
from litgpt.generate.base import generate_AA, generate_TA
from litgpt.model import GPT, Config
from lightning.fabric.utilities.load import _lazy_load as lazy_load
from utils.snac_utils import layershift, reconscruct_snac, reconstruct_tensors, get_time_str
from inference_demo import load_audio, get_input_ids_whisper, get_input_ids_TA, download_model
import lightning as L
import whisper

torch.set_printoptions(sci_mode=False)

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

def A1_A2(fabric, audio_feature, input_ids, leng, model, text_tokenizer, step, snacmodel, out_dir):
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
    with torch.inference_mode():
        audio_hat = snacmodel.decode(audio)
    sf.write(
        f"{out_dir}/{step:02d}.wav",
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
        f"{out_dir}/{step:02d}.wav",
        audio_hat.squeeze().cpu().numpy(),
        24000,
    )
    model.clear_kv_cache()
    return text_tokenizer.decode(torch.tensor(tokenlist)).strip()

def run_inference(mode, input_path, ckpt_dir="./checkpoint", out_dir=None):
    device = "cuda:0"
    if out_dir is None:
        out_dir = f"./output/{get_time_str()}"
    if not os.path.exists(ckpt_dir):
        print(f"checkpoint directory {ckpt_dir} not found, downloading from huggingface")
        download_model(ckpt_dir)

    fabric, model, text_tokenizer, snacmodel, whispermodel = load_model(ckpt_dir, device)

    if mode == "A1A2":
        assert os.path.exists(input_path), f"audio file {input_path} not found"
        mel, leng = load_audio(input_path)
        audio_feature, input_ids = get_input_ids_whisper(mel, leng, whispermodel, device)
        text = A1_A2(fabric, audio_feature, input_ids, leng, model, text_tokenizer, 0, snacmodel, out_dir)
        print(f"Output text: {text}")
        with open(f"{out_dir}/output.txt", "w") as f:
            f.write(text)
    elif mode == "T1A2":
        input_ids = get_input_ids_TA(input_path, text_tokenizer)
        text_output = T1_A2(fabric, input_ids, model, text_tokenizer, 0, snacmodel, out_dir)
        print(f"Output text: {text_output}")
        with open(f"{out_dir}/output.txt", "w") as f:
            f.write(text_output)
    else:
        print("Invalid mode selected. Please choose either 'A1A2' or 'T1A2'.")

if __name__ == "__main__":
    mode = input("Enter mode (A1A2 for audio input, T1A2 for text input): ").strip()
    if mode == "A1A2":
        input_ = input("Enter the path to the audio file: ").strip()
    elif mode == "T1A2":
        input_ = input("Enter the input text: ").strip()
    else:
        print("Invalid mode selected. Please choose either 'A1A2' or 'T1A2'.")
        exit(1)
    run_inference(mode, input_)

# A1A2 / T1A2
# python inference.py

# python -m debugpy --listen 5678 --wait-for-client inference.py