from tqdm import tqdm
from argparse import ArgumentParser
import logging
import os
import jsonlines


def conclude(dir, non_asr, name_format):
    datasets = {
        "alpacaeval_test": "result_open.jsonl",
        "commoneval_test": "result_open.jsonl",
        "wildchat_test": "result_open.jsonl",
        "storal_test": "result_semi_open.jsonl",
        "summary_test": "result_semi_open.jsonl",
        "truthful_test": "result_semi_open.jsonl",
        "gaokao_test": "result_qa.jsonl",
        "gsm8k_test": "result_qa.jsonl",
        "mlc_test": "result_qa.jsonl",
        "repeat_test": "result_repeat_wer.jsonl",
        "mt_test": "result_multi.jsonl",
    }
    score_sum = 0
    result_num = 0
    utmos_sum = 0
    utmos_num = 0
    wer_sum = 0
    wer_num = 0
    with jsonlines.open(os.path.join(dir, "evaluation.jsonl"), mode="w") as r:
        for dataset, result in datasets.items():
            if name_format is not None:
                if not non_asr:
                    result_dir = os.path.join(
                        dir, f"{name_format}{dataset}/eval_with_asr/{dataset}"
                    )
                else:
                    result_dir = os.path.join(
                        dir, f"{name_format}{dataset}/eval/{dataset}"
                    )
            else:
                if not non_asr:
                    result_dir = os.path.join(dir, f"{dataset}/eval_with_asr/{dataset}")
                else:
                    result_dir = os.path.join(dir, f"{dataset}/eval/{dataset}")
            result_file = os.path.join(result_dir, result)
            utmos_file = os.path.join(result_dir, "result_utmos.jsonl")
            wer_file = os.path.join(result_dir, "result_wer.jsonl")
            if os.path.exists(result_file):
                with open(result_file, "r") as f:
                    num = 0
                    for item in jsonlines.Reader(f):
                        data = item
                        num += 1
                    result_num += 1
                    if dataset == "repeat_test":
                        score_sum += (
                            data["ok_rate"] * (1 - data["final_WER_for_ok_case"]) * 100
                        )
                        r.write(
                            {
                                "samples": num - 1,
                                f"score on {dataset}": data["ok_rate"]
                                * (1 - data["final_WER_for_ok_case"])
                                * 100,
                            }
                        )
                    else:
                        score_sum += data["final_score"]
                        r.write(
                            {
                                "samples": num - 1,
                                f"score on {dataset}": data["final_score"],
                            }
                        )
            if os.path.exists(utmos_file):
                with open(utmos_file, "r") as f:
                    for item in jsonlines.Reader(f):
                        data = item
                    utmos_num += 1
                    utmos_sum += data["final_UTMOS"]
            if os.path.exists(wer_file):
                with open(wer_file, "r") as f:
                    for item in jsonlines.Reader(f):
                        data = item
                    wer_num += 1
                    wer_sum += data["final_WER"]
    final_score = score_sum / result_num if result_num != 0 else 0
    final_utmos = utmos_sum / utmos_num if utmos_num != 0 else 0
    final_wer = wer_sum / wer_num if wer_num != 0 else 0
    return final_score, final_utmos, final_wer


def main():
    parser = ArgumentParser()
    parser.add_argument("--eval_dir", type=str, required=True)
    parser.add_argument("--non_asr", action="store_true")
    parser.add_argument("--name_format", required=False)
    args = parser.parse_args()

    # Set log
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(name)s][%(levelname)s] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # evaluate
    dir = args.eval_dir
    non_asr = args.non_asr
    name_format = args.name_format
    score, utmos, wer = conclude(dir, non_asr, name_format)
    with jsonlines.open(os.path.join(dir, "evaluation.jsonl"), mode="a") as f:
        f.write({"final_score": score})
        f.write({"final_UTMOS": utmos})
        f.write({"final_WER": wer})
    logging.info(f"final_score: {score}, final_UTMOS: {utmos}, final_WER: {wer}")
    logging.info(f"result saved to {os.path.join(dir, 'evaluation.jsonl')}")


if __name__ == "__main__":
    main()
