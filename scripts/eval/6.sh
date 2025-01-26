#!/bin/bash
export CUDA_VISIBLE_DEVICES=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export LD_LIBRARY_PATH=/home/visitor/miniconda3/envs/yrq-omni/lib:$LD_LIBRARY_PATH
export PYDEVD_WARN_SLOW_RESOLVE_TIMEOUT=2
export CUDA_LAUNCH_BLOCKING=1

# code dir
code_dir=/data/ruiqi.yan/SLAM-LLM/examples/benchmark
ckpt_dir=/data/ruiqi.yan/omni_models/LLaMA-Omni-test

# jsonl dataset
manifest_format=jsonl
val_data_name="commoneval"     # alpacaeval，commoneval，sd-qa
val_data_path=/data/ruiqi.yan/data/voicebench_raw/${val_data_name}/test.jsonl

# inference output dir
decode_log=/data/ruiqi.yan/omni_models/LLaMA-Omni-test/${val_data_name}
contrast_dir=/data/ruiqi.yan/omni_models/model/gpu16-btz3-lr5e-4-fp16-epochs10-whisper_small-latency5-group3-s2s_epoch_4_step_1179/s2s_decode_test_trp1.2_arp1.22_seed888_greedy_commoneval
output_dir=$decode_log/eval/${val_data_name}

# eval mode
mode="contrast"    # open, qa, contrast

python $code_dir/mark.py \
        --mode $mode \
        --question $decode_log/question_text \
        --answer $decode_log/pred_text \
        --output_dir $output_dir \
        --dataset $val_data_name \
        --answer_contrast $contrast_dir/pred_text
