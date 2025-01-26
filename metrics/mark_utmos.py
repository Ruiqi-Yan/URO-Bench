import torch
import librosa
from pathlib import Path
from tqdm import tqdm
import os
import jsonlines
import logging


def eval(args):
    predictor = torch.hub.load(
        "tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True
    ).cuda()

    audio_path = args.audio_dir
    audio_files = list(Path(audio_path).rglob("*.wav"))

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, "result_utmos.jsonl")

    logging.info("<------start utmos eval------>")
    with jsonlines.open(output_file, mode="w") as f:
        utmos_result = 0
        for line in tqdm(audio_files):
            wave_name = line.stem
            wave, sr = librosa.load(line, sr=None, mono=True)
            score = predictor(torch.from_numpy(wave).cuda().unsqueeze(0), sr)
            utmos_result += score.item()
            f.write({str(wave_name): score.item()})

        f.write({"final_UTMOS": utmos_result / len(audio_files)})
        logging.info(f"UTMOS: {utmos_result / len(audio_files)}")
        logging.info(f"Results have been saved to {output_file}")
