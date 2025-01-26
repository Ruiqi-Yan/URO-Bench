#!/bin/bash
export CUDA_VISIBLE_DEVICES=5
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export LD_LIBRARY_PATH=/home/visitor/miniconda3/envs/yrq-omni/lib:$LD_LIBRARY_PATH
export PYDEVD_WARN_SLOW_RESOLVE_TIMEOUT=2
export CUDA_LAUNCH_BLOCKING=1


code_dir=/data/ruiqi.yan/SLAM-LLM/examples/benchmark
ckpt_dir=/data/ruiqi.yan/models
llm=glm-4-9b-chat-hf

# jsonl dataset
manifest_format=jsonl
# alpacaeval_test, 199
# commoneval_test, 200
# wildchat_test, 349
# storal_test, 201
# summary_test, 118
# truthful_test, 470
# gaokao_test, 303
# gsm8k_test, 582
# mlc_test, 177
# repeat_test, 252
# mt_test, 190
val_data_name="mt_test"
val_data_path=/data/ruiqi.yan/data/final/${val_data_name}/test.jsonl
data_number=190

decode_log=/data/ruiqi.yan/omni_models/${llm}-test/${val_data_name}
audio_prompt=prompt_6
output_dir=$decode_log/eval/${val_data_name}

# -m debugpy --listen 5678 --wait-for-client
python $code_dir/llm-test/inference_multi.py \
        --dataset $val_data_path \
        --modality "audio" \
        --output_dir $decode_log \
        --whisper_path $ckpt_dir/whisper-large-v3 \
        --llm_path $ckpt_dir/$llm

# python $code_dir/asr_for_eval.py \
#         --input_dir $decode_log \
#         --model_dir $ckpt_dir/whisper-large-v3 \
#         --output_dir $decode_log \
#         --number $data_number \
#         --dataset $val_data_path \
#         --multi

# eval mode
# open: alpacaeval_test, commoneval_test, wildchat_test
# semi-open: storal_test, summary_test, truthful_test
# qa: gaokao_test, gsm8k_test, mlc_test
# wer: repeat_test
# multi: mt_test
mode="multi"    # open, semi-open, qa, wer, multi, contrast

python $code_dir/mark.py \
        --mode $mode \
        --question $decode_log/output_text.jsonl \
        --answer $decode_log/output_text.jsonl \
        --output_dir $output_dir \
        --dataset $val_data_name