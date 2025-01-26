import argparse
import os
import json
import torch
import whisper
from tqdm import tqdm
from omni_speech.constants import SPEECH_TOKEN_INDEX, DEFAULT_SPEECH_TOKEN
from omni_speech.conversation import conv_templates, SeparatorStyle
from omni_speech.model.builder import load_pretrained_model
from omni_speech.utils import disable_torch_init
from omni_speech.datasets.preprocess import tokenizer_speech_token
from torch.utils.data import Dataset, DataLoader
import soundfile as sf
from fairseq import utils
from fairseq.models.text_to_speech.vocoder import CodeHiFiGANVocoder
import math 

def split_list(lst, n):
    chunk_size = math.ceil(len(lst) / n)
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]

def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

class CustomDataset(Dataset):
    def __init__(self, questions, tokenizer, model_config, input_type, mel_size):
        self.questions = questions
        self.tokenizer = tokenizer
        self.model_config = model_config
        self.input_type = input_type
        self.mel_size = mel_size

    def __getitem__(self, index):
        item = self.questions[index]
        speech_file = item["speech"]
        qs = item["conversations"][0]["value"]

        conv = conv_templates["llama_3"].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        speech = whisper.load_audio(speech_file)
        if self.input_type == "raw":
            speech = torch.from_numpy(speech)
            if self.model_config.speech_normalize:
                speech = torch.nn.functional.layer_norm(speech, speech.shape)
        elif self.input_type == "mel":
            speech = whisper.pad_or_trim(speech)
            speech = whisper.log_mel_spectrogram(speech, n_mels=self.mel_size).permute(1, 0)

        input_ids = tokenizer_speech_token(prompt, self.tokenizer, return_tensors='pt')

        return input_ids, speech, torch.LongTensor([speech.shape[0]])

    def __len__(self):
        return len(self.questions)

def collate_fn(batch):
    input_ids, speech_tensors, speech_lengths = zip(*batch)
    input_ids = torch.stack(input_ids, dim=0)
    speech_tensors = torch.stack(speech_tensors, dim=0)
    speech_lengths = torch.stack(speech_lengths, dim=0)
    return input_ids, speech_tensors, speech_lengths

def ctc_postprocess(tokens, blank):
    _toks = tokens.squeeze(0).tolist()
    deduplicated_toks = [v for i, v in enumerate(_toks) if i == 0 or v != _toks[i - 1]]
    hyp = [v for v in deduplicated_toks if v != blank]
    hyp = " ".join(list(map(str, hyp)))
    return hyp

def create_data_loader(questions, tokenizer, model_config, input_type, mel_size, batch_size=1, num_workers=4):
    assert batch_size == 1, "batch_size must be 1"
    dataset = CustomDataset(questions, tokenizer, model_config, input_type, mel_size)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, collate_fn=collate_fn)
    return data_loader

def eval_model(args, questions):
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    tokenizer, model, context_len = load_pretrained_model(model_path, args.model_base, is_lora=args.is_lora, s2s=args.s2s)

    data_loader = create_data_loader(questions, tokenizer, model.config, args.input_type, args.mel_size)

    results = []
    for (input_ids, speech_tensor, speech_length), item in tqdm(zip(data_loader, questions), total=len(questions)):
        idx = item["id"]
        try:
            answer = item["conversations"][1]["value"]
        except:
            answer = None
        input_ids = input_ids.to(device='cuda', non_blocking=True)
        speech_tensor = speech_tensor.to(dtype=torch.float16, device='cuda', non_blocking=True)
        speech_length = speech_length.to(device='cuda', non_blocking=True)

        with torch.inference_mode():
            if args.s2s:
                outputs = model.generate(
                    input_ids,
                    speech=speech_tensor,
                    speech_lengths=speech_length,
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    max_new_tokens=args.max_new_tokens,
                    use_cache=True,
                    pad_token_id=128004,
                    streaming_unit_gen=False,
                )
                output_ids, output_units = outputs
            else:
                outputs = model.generate(
                    input_ids,
                    speech=speech_tensor,
                    speech_lengths=speech_length,
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    max_new_tokens=args.max_new_tokens,
                    use_cache=True,
                    pad_token_id=128004,
                )
                output_ids = outputs

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        if args.s2s:
            output_units = ctc_postprocess(output_units, blank=model.config.unit_vocab_size)

        results.append({"question_id": idx, "prediction": outputs, "prediction_units": output_units if args.s2s else None, "answer": answer})
    return results

def generate_waveform(args, prediction_units):
    with open(args.vocoder_cfg) as f:
        vocoder_cfg = json.load(f)
    vocoder = CodeHiFiGANVocoder(args.vocoder, vocoder_cfg)
    if torch.cuda.is_available() and not args.cpu:
        vocoder = vocoder.cuda()

    os.makedirs(args.results_path, exist_ok=True)

    data = [list(map(int, prediction_units.split()))]
    for i, d in enumerate(data):
        x = {"code": torch.LongTensor(d).view(1, -1)}
        x = utils.move_to_cuda(x) if torch.cuda.is_available() and not args.cpu else x
        wav = vocoder(x, args.dur_prediction)
        sf.write(f"{args.results_path}/result_{i}_pred.wav", wav.detach().cpu().numpy(), 16000)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="Llama-3.1-8B-Omni")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--wav-path", type=str, required=True)
    parser.add_argument("--results-path", type=str, required=True)
    parser.add_argument("--conv-mode", type=str, default="llama_3")
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--input_type", type=str, default="mel")
    parser.add_argument("--mel_size", type=int, default=128)
    parser.add_argument("--s2s", action="store_true", default=False)
    parser.add_argument("--is_lora", action="store_true", default=False)
    parser.add_argument("--vocoder", type=str, default="vocoder/g_00500000")
    parser.add_argument("--vocoder-cfg", type=str, default="vocoder/config.json")
    parser.add_argument("--dur-prediction", action="store_true")
    parser.add_argument("--cpu", action="store_true")

    args = parser.parse_args()

    questions = [{
        "id": "input",
        "speech": args.wav_path,
        "conversations": [{"from": "human", "value": "<speech>\nPlease directly answer the questions in the user's speech."}]
    }]

    os.makedirs(args.results_path, exist_ok=True)
    results = eval_model(args, questions)
    for result in results:
        print(f"Prediction: {result['prediction']}")
        with open(os.path.join(args.results_path, f"result_{result['question_id']}_pred.txt"), "w") as f:
            f.write(result['prediction'])
        if args.s2s:
            generate_waveform(args, result['prediction_units'])

if __name__ == "__main__":
    main()

# python inference.py --wav-path <输入wav文件路径> --results-path <结果保存路径> --s2s --dur-prediction

# debug
# python -m debugpy --listen 5678 --wait-for-client inference.py --wav-path test.wav --results-path output  --s2s --dur-prediction

# Run the following command in the terminal:
# python inference.py --wav-path test.wav --results-path output  --s2s --dur-prediction

