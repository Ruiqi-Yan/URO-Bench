#!/bin/bash
export CUDA_VISIBLE_DEVICES=7
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export LD_LIBRARY_PATH=/home/visitor/miniconda3/envs/yrq-omni/lib:$LD_LIBRARY_PATH


# code dir
model_name=Freeze-Omni
code_dir=/data/ruiqi.yan/URO-Bench
ckpt_dir=/data/ruiqi.yan/URO-Bench-log/${model_name}-test
log_dir=/data/ruiqi.yan/URO-Bench-log/${model_name}-test
whisper_dir=/data/ruiqi.yan/models/whisper-large-v3

# all the datasets
datasets=(
    "alpacaeval_test 199 open basic en"
    "commoneval_test 200 open basic en"
    "wildchat_test 349 open basic en"
    "storal_test 201 semi-open basic en"
    "summary_test 118 semi-open basic en"
    "truthful_test 470 semi-open basic en"
    "gaokao_test 303 qa basic en"
    "gsm8k_test 582 qa basic en"
    "mlc_test 177 qa basic en"
    "repeat_test 252 wer basic en"
    "alpacaeval_zh 200 open basic zh"
    "claude_zh 273 open basic zh"
    "lcsts_zh 229 semi-open basic zh"
    "mlc_zh 136 qa basic zh"
    "openbookqa_zh 257 qa basic zh"
    "repeat_zh 210 wer basic zh"
    "codeswitching_en 70 cs pro en"
    "codeswitching_zh 70 cs pro zh"
    "genemotion_en 54 ge pro en"
    "genemotion_zh 43 ge pro zh"
    "genstyle_en 44 gs pro en"
    "genstyle_zh 39 gs pro zh"
    "mlcpro_en 91 qa pro en"
    "mlcpro_zh 64 qa pro zh"
    "safety_en 24 sf pro en"
    "safety_zh 20 sf pro zh"
    "SRT_en 43 srt pro en"
    "SRT_zh 21 srt pro zh"
    "underemotion_en 137 ue pro en"
    "underemotion_zh 79 ue pro zh"
    "multilingual_test 1108 ml pro en"
)

# eval
for pair in "${datasets[@]}"
do
    # get dataset info
    dataset_name=$(echo "$pair" | cut -d' ' -f1)
    sample_number=$(echo "$pair" | cut -d' ' -f2)
    eval_mode=$(echo "$pair" | cut -d' ' -f3)
    level=$(echo "$pair" | cut -d' ' -f4)
    language=$(echo "$pair" | cut -d' ' -f5)
    dataset_path=/data/ruiqi.yan/URO-Bench-data/${level}/${dataset_name}/test.jsonl

    # output dir
    infer_output_dir=${log_dir}/eval/${level}/${dataset_name}
    eval_output_dir=$infer_output_dir/eval_with_asr

    source /home/visitor/miniconda3/etc/profile.d/conda.sh
    # put your env name here, this env depends on the model you are testing
    conda activate yrq-omni
    # inference
    # -m debugpy --listen 5678 --wait-for-client
    python $code_dir/examples/${model_name}-test/inference_for_eval.py \
        --model_path $ckpt_dir/checkpoints \
        --llm_path $ckpt_dir/Qwen2-7B-Instruct \
        --dataset $dataset_path \
        --output_dir $infer_output_dir \
        --top_k 20 \
        --top_p 0.8 \
        --temperature 0.8

    source /home/visitor/miniconda3/etc/profile.d/conda.sh
    conda activate yrq-uro               # put your env name here
    # asr
    python $code_dir/asr_for_eval.py \
        --input_dir $infer_output_dir/audio \
        --model_dir $whisper_dir \
        --output_dir $infer_output_dir \
        --number $sample_number

    # assign scores
    if [[ ${eval_mode} == "open" ]]; then
        python $code_dir/mark.py \
        --mode $eval_mode \
        --question $infer_output_dir/question_text.jsonl \
        --answer $infer_output_dir/asr_text.jsonl \
        --answer_text $infer_output_dir/pred_text.jsonl \
        --output_dir $eval_output_dir \
        --dataset $dataset_name \
        --audio_dir $infer_output_dir/audio
    else
        python $code_dir/mark.py \
        --mode $eval_mode \
        --question $infer_output_dir/question_text.jsonl \
        --answer $infer_output_dir/asr_text.jsonl \
        --answer_text $infer_output_dir/pred_text.jsonl \
        --output_dir $eval_output_dir \
        --dataset $dataset_name \
        --audio_dir $infer_output_dir/audio \
        --reference $infer_output_dir/gt_text.jsonl
    fi

done

# conclusion
python $code_dir/evaluate.py --eval_dir ${log_dir}/eval