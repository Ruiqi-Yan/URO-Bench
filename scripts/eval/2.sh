#!/bin/bash
export CUDA_VISIBLE_DEVICES=6
export HF_ENDPOINT=https://hf-mirror.com


# code dir
model_name=GLM-4-Voice
code_dir=/data/ruiqi.yan/URO-Bench
ckpt_dir=/data/ruiqi.yan/URO-Bench-log/${model_name}-test
log_dir=/data/ruiqi.yan/URO-Bench-log/${model_name}-test
whisper_dir=/data/ruiqi.yan/models/whisper-large-v3

# all the datasets
datasets=(
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
    cd $code_dir/examples/${model_name}-test
    # -m debugpy --listen 5678 --wait-for-client
    python $code_dir/examples/${model_name}-test/inference_for_eval.py \
        --input-mode "audio" \
        --output-dir $infer_output_dir \
        --val_data_path $dataset_path \
        --val_data_name $dataset_name \
        --flow-path $ckpt_dir/glm-4-voice-decoder \
        --model-path $ckpt_dir/glm-4-voice-9b \
        --tokenizer-path $ckpt_dir/glm-4-voice-tokenizer \
        --manifest_format "jsonl"

#     source /home/visitor/miniconda3/etc/profile.d/conda.sh
#     conda activate yrq-uro               # put your env name here
#     # asr
#     python $code_dir/asr_for_eval.py \
#         --input_dir $infer_output_dir/audio \
#         --model_dir $whisper_dir \
#         --output_dir $infer_output_dir \
#         --number $sample_number

#     # assign scores
#     if [[ ${eval_mode} == "open" ]]; then
#         python $code_dir/mark.py \
#         --mode $eval_mode \
#         --question $infer_output_dir/question_text.jsonl \
#         --answer $infer_output_dir/asr_text.jsonl \
#         --answer_text $infer_output_dir/pred_text.jsonl \
#         --output_dir $eval_output_dir \
#         --dataset $dataset_name \
#         --audio_dir $infer_output_dir/audio
#     else
#         python $code_dir/mark.py \
#         --mode $eval_mode \
#         --question $infer_output_dir/question_text.jsonl \
#         --answer $infer_output_dir/asr_text.jsonl \
#         --answer_text $infer_output_dir/pred_text.jsonl \
#         --output_dir $eval_output_dir \
#         --dataset $dataset_name \
#         --audio_dir $infer_output_dir/audio \
#         --reference $infer_output_dir/gt_text.jsonl
#     fi

done

# multi-round eval
# multi-round datasets
manifest_format=jsonl
datasets=(
    "mtpro_en 55 multi pro en"
    "mtpro_zh 49 multi pro zh"
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
    cd $code_dir/examples/${model_name}-test
    # python -m debugpy --listen 5678 --wait-for-client inference_multi-round.py \
    python $code_dir/examples/${model_name}-test/inference_multi-round.py \
        --input-jsonl $dataset_path \
        --output-dir $infer_output_dir \
        --model-path $ckpt_dir/glm-4-voice-9b \
        --tokenizer-path $ckpt_dir/glm-4-voice-tokenizer \
        --flow-path $ckpt_dir/glm-4-voice-decoder
    

#     source /home/visitor/miniconda3/etc/profile.d/conda.sh
#     conda activate yrq-uro               # put your env name here
#     # asr
#     python $code_dir/asr_for_eval.py \
#         --input_dir $infer_output_dir \
#         --model_dir $whisper_dir \
#         --output_dir $infer_output_dir \
#         --number $sample_number \
#         --dataset $dataset_path \
#         --multi

#     # assign scores
#     python $code_dir/mark.py \
#     --mode $eval_mode \
#     --question $infer_output_dir/output_with_text.jsonl \
#     --answer $infer_output_dir/output_with_text.jsonl \
#     --output_dir $eval_output_dir \
#     --dataset $dataset_name \
#     --audio_dir $infer_output_dir

done

# # conclusion
# python $code_dir/evaluate.py --eval_dir ${log_dir}/eval