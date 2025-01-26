# s2s-Benchmark

## Environment Setup

### Slam-Omni
Set up the environment using the following command after setting up the environment for SLAM-LLM:
```bash
# there may be conflicts, but runs well on my machine 
pip install -r requirements.txt
# or
pip install -r requirements.txt --no-dependencies   
```
or you can set up another environment, read [voicebench](VoiceBench/README.md) for more detail. This way, you need to switch your environment between inference and marking.

### Mini-Omni
Use the same environment as [Slam-omni](#slam-omni)

### Llama-Omni
Set up the environment according to [Llama-omni](LLaMA-Omni-test/README.md)

## Datasets
Currently, we support evaluation for 10 datasets.
Model's responses are evaluated in 4 different modes.

### open
alpacaeval_test, commoneval_test, wildchat_test
### semi-open
storal_test, summary_test, truthful_test
### qa
gaokao_test, gsm8k_test, mlc_test
### wer
repeat_test

## Evaluation

### Slam-Omni

#### non-asr mode
In non-asr mode, we directly evaluate the output text of LLM.

Run the following command:
```bash
# choose ${val_data_name}
bash ./scripts/eval/eval.sh
```
or run inference and marking separately
```bash
# choose ${val_data_name}
bash ./scripts/eval/inference_for_eval.sh
conda activate voicebench
bash ./scripts/eval/mark_only.sh
```

#### asr mode
In asr mode, we use [whisper-large-v3](https://github.com/openai/whisper) for asr and evaluate the transcription of the output speech.

Run the following command:
```bash
# choose ${val_data_name}
bash ./scripts/eval/eval_with_asr.sh
```
or run inference and marking separately
```bash
# choose ${val_data_name}
bash ./scripts/eval/inference_for_eval.sh
conda activate voicebench
bash ./scripts/eval/asr_for_eval.sh
```

### Mini-Omni
For non-asr mode, run the following command:
```bash
# choose ${val_data_name}
bash ./scripts/eval/mini-omni-eval.sh
```

For asr mode, just uncomment corresponding code in [mini-omni-eval.sh](scripts/eval/mini-omni-eval.sh)

### Llama-Omni
Attention! You need to switch to your [Llama-Omni environment](#llama-omni)

For non-asr mode, run the following command:
```bash
conda activate llama-omni
# choose ${val_data_name}
bash ./scripts/eval/llama-omni-eval.sh
```

For asr mode, just uncomment corresponding code in [llama-omni-eval.sh](scripts/eval/llama-omni-eval.sh)