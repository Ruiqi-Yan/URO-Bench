#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
export HF_ENDPOINT=https://hf-mirror.com

# name of your SDM, e.g. slam-omni
model_name=

# dir of URO-Bench, e.g. /data/ruiqi.yan/URO-Bench
code_dir=

# dir to record the result of evaluation, e.g. /data/ruiqi.yan/URO-Bench-log/slam-omni-test
log_dir=

# dir of whisper-large-v3, e.g. /data/ruiqi.yan/models/whisper-large-v3
# if your network is OK, just fill in "openai/whisper-large-v3"
whisper_dir=

# dir of downloaded URO-Bench-data, e.g. /data/ruiqi.yan/URO-Bench-data
uro_data_dir=

# dir of conda.sh, e.g. /home/visitor/miniconda3/etc/profile.d/conda.sh
# for the use of switching env automatically
conda_dir=

# name of the env of your SDM, e.g. yrq-omni
sdm_env_name=

# name of the env of URO-Bench, if you follow our guide, this should be "uro"
uro_env_name=uro

# OpenAI api key, e.g.
openai_api_key=

# choose the datasets you want to test on your SDM, single-round
datasets=(
    "AlpacaEval 199 open basic en"
    "CommonEval 200 open basic en"
    "WildchatEval 349 open basic en"
    "StoralEval 201 semi-open basic en"
    "Summary 118 semi-open basic en"
    "TruthfulEval 470 semi-open basic en"
    "GaokaoEval 303 qa basic en"
    "Gsm8kEval 582 qa basic en"
    "MLC 177 qa basic en"
    "Repeat 252 wer basic en"
    "AlpacaEval-zh 200 open basic zh"
    "Claude-zh 273 open basic zh"
    "LCSTS-zh 229 semi-open basic zh"
    "MLC-zh 136 qa basic zh"
    "OpenbookQA-zh 257 qa basic zh"
    "Repeat-zh 210 wer basic zh"
    "CodeSwitching-en 70 semi-open pro en"
    "CodeSwitching-zh 70 semi-open pro zh"
    "GenEmotion-en 54 ge pro en"
    "GenEmotion-zh 43 ge pro zh"
    "GenStyle-en 44 gs pro en"
    "GenStyle-zh 39 gs pro zh"
    "MLCpro-en 91 qa pro en"
    "MLCpro-zh 64 qa pro zh"
    "Safety-en 24 sf pro en"
    "Safety-zh 20 sf pro zh"
    "SRT-en 43 srt pro en"
    "SRT-zh 21 srt pro zh"
    "UnderEmotion-en 137 ue pro en"
    "UnderEmotion-zh 79 ue pro zh"
    "Multilingual 1108 ml pro en"
    "ClothoEval-en 265 qa pro en"
    "MuChoEval-en 311 qa pro en"
)

# choose the datasets you want to test on your SDM, multi-round
multi_datasets=(
    "MtBenchEval-en 190 multi basic en"
    "SpeakerAware-en 55 sa pro en"
    "SpeakerAware-zh 49 sa pro zh"
)