#!/bin/bash
export CUDA_VISIBLE_DEVICES=6
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export LD_LIBRARY_PATH=/home/visitor/miniconda3/envs/yrq-omni/lib:$LD_LIBRARY_PATH
export PYDEVD_WARN_SLOW_RESOLVE_TIMEOUT=2
export CUDA_LAUNCH_BLOCKING=1


# code dir
model_name=slam-omni
code_dir=/data/ruiqi.yan/URO-Bench
log_dir=/data/ruiqi.yan/URO-Bench-log/${model_name}-test
whisper_dir=/data/ruiqi.yan/models/whisper-large-v3

whisper_size=small  # tiny base small medium large-v3
speech_encoder_path="/data/ruiqi.yan/models/whisper/${whisper_size}.pt"   # different whisper size
llm_path="/data/ruiqi.yan/omni_models/model/Qwen2-0.5B"
codec_decoder_path="/data/ruiqi.yan/models/CosyVoice/pretrained_models/CosyVoice-300M-SFT" # replace this with your own CosyVoice model path

encoder_dim=768  # 384 512 768 896 1024 1280 
mel_size=80      # 80 128 (128 for whisper-large only)
llm_dim=896     # 896 1536 3584 8192  -> 0.5B 1.5B 3.5B 7B

task_type=s2s

# vocabulary settings
code_layer=3            # 1 single semantic code layer   2 3 4 5 6 7 8 group semantic code layers 
total_audio_vocabsize=4160
total_vocabsize=156160  # 152000 + 4160 Sry: Here is not elegant to set the total_vocabsize manually, I may fix it later :)

# code settings
code_type=CosyVoice     # CosyVoice or SNAC
codec_decoder_type=CosyVoice
num_latency_tokens=0    # number of latency tokens (same as the number in training)
do_layershift=false      # if false, tokens in each layers use the same codebook, otherwise, use different codebooks

ckpt_path=/data/ruiqi.yan/URO-Bench-log/slam-omni-test/Qwen2-0.5b-whisper_small-latency0-group3-single-round-English
split=test

load_from_cache_file=false
dataset_sample_seed=888

# model settings
tts_adapter=false
group_decode=true
group_decode_adapter_type=linear

# decode config
text_repetition_penalty=1.2
audio_repetition_penalty=1.2        # default 1.0, set to 1.2 for reduce silence
max_new_tokens=3000                 # 500 for SNAC, 3000 for CosyVoice-single
do_sample=false
top_p=0.9
top_k=50
temperature=1.0
decode_text_only=false
upsampling_factor=1

output_text_only=false
speech_sample_rate=22050    # 22050 for CosyVoice, 24000 for SNAC
inference_online=false
audio_prompt_path=/data/ruiqi.yan/URO-Bench/examples/${model_name}-test/audio_prompt/en/prompt_6.wav      # replace this with your own audio prompt path or our provided audio prompt path
audio_prompt=prompt_6

# all the datasets
manifest_format=jsonl
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

    if [[ ${language} == "en" ]]; then
        ckpt_path=/data/ruiqi.yan/URO-Bench-log/slam-omni-test/Qwen2-0.5b-whisper_small-latency0-group3-single-round-English
    else
        ckpt_path=/data/ruiqi.yan/URO-Bench-log/slam-omni-test/Qwen2-0.5b-whisper_small-latency0-group3-multi-round-Chinese
    fi

    # output dir
    infer_output_dir=${log_dir}/eval/${level}/${dataset_name}
    eval_output_dir=$infer_output_dir/eval_with_asr

    source /home/visitor/miniconda3/etc/profile.d/conda.sh
    # put your env name here, this env depends on the model you are testing
    conda activate yrq-omni

    # inference
    cd /data/ruiqi.yan/URO-Bench-data/${level}/${dataset_name}
    # -m debugpy --listen 5678 --wait-for-client
    python $code_dir/examples/${model_name}-test/inference_s2s.py \
        --config-path "conf" \
        --config-name "prompt.yaml" \
        hydra.run.dir=$ckpt_path \
        ++model_config.llm_name=qwen2-0.5b \
        ++model_config.file=$code_dir/examples/${model_name}-test/model/slam_model_s2s.py:model_factory \
        ++model_config.llm_path=$llm_path \
        ++model_config.llm_dim=$llm_dim \
        ++model_config.encoder_name=whisper \
        ++model_config.encoder_projector_ds_rate=5 \
        ++model_config.encoder_path=$speech_encoder_path \
        ++model_config.encoder_dim=$encoder_dim \
        ++model_config.encoder_projector=linear \
        ++model_config.codec_decoder_path=$codec_decoder_path \
        ++model_config.codec_decode=true \
        ++model_config.vocab_config.code_layer=$code_layer \
        ++model_config.vocab_config.total_audio_vocabsize=$total_audio_vocabsize \
        ++model_config.vocab_config.total_vocabsize=$total_vocabsize \
        ++model_config.code_type=$code_type \
        ++model_config.codec_decoder_type=$codec_decoder_type \
        ++model_config.group_decode=$group_decode \
        ++model_config.group_decode_adapter_type=$group_decode_adapter_type \
        ++dataset_config.dataset=speech_dataset_s2s \
        ++dataset_config.val_data_path=$dataset_path \
        ++dataset_config.file=$code_dir/examples/${model_name}-test/speech_dataset_s2s.py:get_speech_dataset \
        ++dataset_config.train_data_path=$dataset_path \
        ++dataset_config.input_type=mel \
        ++dataset_config.mel_size=$mel_size \
        ++dataset_config.inference_mode=true \
        ++dataset_config.manifest_format=$manifest_format \
        ++dataset_config.load_from_cache_file=$load_from_cache_file \
        ++dataset_config.task_type=$task_type \
        ++dataset_config.seed=$dataset_sample_seed \
        ++dataset_config.vocab_config.code_layer=$code_layer \
        ++dataset_config.vocab_config.total_audio_vocabsize=$total_audio_vocabsize \
        ++dataset_config.vocab_config.total_vocabsize=$total_vocabsize \
        ++dataset_config.code_type=$code_type \
        ++dataset_config.num_latency_tokens=$num_latency_tokens \
        ++dataset_config.do_layershift=$do_layershift \
        ++train_config.model_name=s2s \
        ++train_config.freeze_encoder=true \
        ++train_config.freeze_llm=true \
        ++train_config.freeze_encoder_projector=true \
        ++train_config.freeze_group_decode_adapter=true \
        ++train_config.batching_strategy=custom \
        ++train_config.num_epochs=1 \
        ++train_config.val_batch_size=1 \
        ++train_config.num_workers_dataloader=2 \
        ++train_config.task_type=$task_type \
        ++decode_config.text_repetition_penalty=$text_repetition_penalty \
        ++decode_config.audio_repetition_penalty=$audio_repetition_penalty \
        ++decode_config.max_new_tokens=$max_new_tokens \
        ++decode_config.task_type=$task_type \
        ++decode_config.do_sample=$do_sample \
        ++decode_config.top_p=$top_p \
        ++decode_config.top_k=$top_k \
        ++decode_config.temperature=$temperature \
        ++decode_config.decode_text_only=$decode_text_only \
        ++decode_config.do_layershift=$do_layershift \
        ++decode_log=$infer_output_dir \
        ++decode_config.num_latency_tokens=$num_latency_tokens \
        ++ckpt_path=$ckpt_path/model.pt \
        ++output_text_only=$output_text_only \
        ++inference_online=$inference_online \
        ++speech_sample_rate=$speech_sample_rate \
        ++audio_prompt_path=$audio_prompt_path \
        ++log_config.log_file="/data/ruiqi.yan/exp/s2s/debug/inference.log" \
        ++log_config.wandb_dir="/data/ruiqi.yan/exp/wandb_log"    # put your log file here

    source /home/visitor/miniconda3/etc/profile.d/conda.sh
    conda activate yrq-uro               # put your env name here
    # asr
    python $code_dir/asr_for_eval.py \
        --input_dir $infer_output_dir/pred_audio/$audio_prompt \
        --model_dir $whisper_dir \
        --output_dir $infer_output_dir \
        --language $language \
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
        --dataset_path $dataset_path \
        --language $language \
        --audio_dir $infer_output_dir/pred_audio/$audio_prompt
    else
        python $code_dir/mark.py \
        --mode $eval_mode \
        --question $infer_output_dir/question_text.jsonl \
        --answer $infer_output_dir/asr_text.jsonl \
        --answer_text $infer_output_dir/pred_text.jsonl \
        --output_dir $eval_output_dir \
        --dataset $dataset_name \
        --dataset_path $dataset_path \
        --language $language \
        --audio_dir $infer_output_dir/pred_audio/$audio_prompt \
        --reference $infer_output_dir/gt_text.jsonl
    fi

done

# multi-round eval
# vocabulary settings
code_layer=3                        # 1 single semantic code layer   2 3 4 5 6 7 8 group semantic code layers 
total_audio_vocabsize=4160          # the vocab size of the codec token
llm_vocabsize=152000                # the vocab size of the LLM model (Qwen2 here)
total_vocabsize=$((total_audio_vocabsize + llm_vocabsize))

# code settings
code_type=CosyVoice                 # CosyVoice or SNAC
codec_decoder_type=CosyVoice
num_latency_tokens=0                # number of latency tokens (same as the number in training)
do_layershift=false                 # if false, tokens in each layers use the same codebook, otherwise, use different codebooks

# model settings
group_decode=true
group_decode_adapter_type=linear
whisper_decode=true
use_peft=false

# decode config
text_repetition_penalty=1.2
audio_repetition_penalty=1.2        # default 1.0, set to 1.2 for reduce silence
max_new_tokens=3000                 # 500 for SNAC, 3000 for CosyVoice-single
do_sample=false
top_p=1.0
top_k=0
temperature=1.0
decode_text_only=false
input_text=false

output_text_only=false
speech_sample_rate=22050            # 22050 for CosyVoice, 24000 for SNAC
inference_online=false

multi_round=true
audio_prompt_path=/data/ruiqi.yan/URO-Bench/examples/${model_name}-test/audio_prompt/en/prompt_6.wav      # replace this with your own audio prompt path or our provided audio prompt path

# multi-round datasets
manifest_format=jsonl
datasets=(
    "MtBenchEval-en 190 multi pro en"
    "SpeakerAware-en 55 sa pro en"
    "SpeakerAware-zh 49 sa pro zh"
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

    if [[ ${language} == "en" ]]; then
        ckpt_path=/data/ruiqi.yan/URO-Bench-log/slam-omni-test/Qwen2-0.5b-whisper_small-latency0-group3-multi-round-English
    else
        ckpt_path=/data/ruiqi.yan/URO-Bench-log/slam-omni-test/Qwen2-0.5b-whisper_small-latency0-group3-multi-round-Chinese
    fi

    # output dir
    infer_output_dir=${log_dir}/eval/${level}/${dataset_name}
    eval_output_dir=$infer_output_dir/eval_with_asr

    source /home/visitor/miniconda3/etc/profile.d/conda.sh
    # put your env name here, this env depends on the model you are testing
    conda activate yrq-omni

    # inference
    cd /data/ruiqi.yan/URO-Bench-data/${level}/${dataset_name}
    # -m debugpy --listen 5678 --wait-for-client
    python $code_dir/examples/${model_name}-test/inference_s2s.py \
        --config-path "conf" \
        --config-name "prompt.yaml" \
        hydra.run.dir=$ckpt_path \
        ++model_config.llm_name=qwen2-0.5b \
        ++model_config.file=$code_dir/examples/${model_name}-test/model/slam_model_s2s.py:model_factory \
        ++model_config.llm_path=$llm_path \
        ++model_config.llm_dim=$llm_dim \
        ++model_config.encoder_name=whisper \
        ++model_config.encoder_projector_ds_rate=5 \
        ++model_config.encoder_path=$speech_encoder_path \
        ++model_config.encoder_dim=$encoder_dim \
        ++model_config.encoder_projector=linear \
        ++model_config.codec_decoder_path=$codec_decoder_path \
        ++model_config.codec_decode=true \
        ++model_config.vocab_config.code_layer=$code_layer \
        ++model_config.vocab_config.total_audio_vocabsize=$total_audio_vocabsize \
        ++model_config.vocab_config.total_vocabsize=$total_vocabsize \
        ++model_config.code_type=$code_type \
        ++model_config.codec_decoder_type=$codec_decoder_type \
        ++model_config.group_decode=$group_decode \
        ++model_config.group_decode_adapter_type=$group_decode_adapter_type \
        ++model_config.whisper_decode=$whisper_decode \
        ++dataset_config.dataset=speech_dataset_s2s \
        ++dataset_config.file=$code_dir/examples/${model_name}-test/speech_dataset_s2s.py:get_speech_dataset \
        ++dataset_config.input_type=mel \
        ++dataset_config.mel_size=$mel_size \
        ++dataset_config.inference_mode=true \
        ++dataset_config.task_type=$task_type \
        ++dataset_config.vocab_config.code_layer=$code_layer \
        ++dataset_config.vocab_config.total_audio_vocabsize=$total_audio_vocabsize \
        ++dataset_config.vocab_config.total_vocabsize=$total_vocabsize \
        ++dataset_config.code_type=$code_type \
        ++dataset_config.num_latency_tokens=$num_latency_tokens \
        ++dataset_config.do_layershift=$do_layershift \
        ++train_config.model_name=s2s \
        ++train_config.freeze_encoder=true \
        ++train_config.freeze_llm=true \
        ++train_config.freeze_encoder_projector=true \
        ++train_config.freeze_group_decode_adapter=true \
        ++train_config.batching_strategy=custom \
        ++train_config.num_epochs=1 \
        ++train_config.val_batch_size=1 \
        ++train_config.num_workers_dataloader=2 \
        ++train_config.task_type=$task_type \
        ++train_config.use_peft=$use_peft \
        ++decode_config.text_repetition_penalty=$text_repetition_penalty \
        ++decode_config.audio_repetition_penalty=$audio_repetition_penalty \
        ++decode_config.max_new_tokens=$max_new_tokens \
        ++decode_config.task_type=$task_type \
        ++decode_config.do_sample=$do_sample \
        ++decode_config.top_p=$top_p \
        ++decode_config.top_k=$top_k \
        ++decode_config.temperature=$temperature \
        ++decode_config.decode_text_only=$decode_text_only \
        ++decode_config.do_layershift=$do_layershift \
        ++decode_log=$infer_output_dir \
        ++decode_config.num_latency_tokens=$num_latency_tokens \
        ++ckpt_path=$ckpt_path/model.pt \
        ++output_text_only=$output_text_only \
        ++inference_online=$inference_online \
        ++speech_sample_rate=$speech_sample_rate \
        ++audio_prompt_path=$audio_prompt_path \
        ++multi_round=$multi_round \
        ++batch_input_jsonl=$dataset_path \
        ++log_config.online_output_dir=$infer_output_dir \
        ++log_config.log_file="/data/ruiqi.yan/exp/s2s/debug/inference.log" \
        ++log_config.wandb_dir="/data/ruiqi.yan/exp/wandb_log"    # put your log file here

    source /home/visitor/miniconda3/etc/profile.d/conda.sh
    conda activate yrq-uro               # put your env name here
    # asr
    python $code_dir/asr_for_eval.py \
        --input_dir $infer_output_dir/$audio_prompt \
        --model_dir $whisper_dir \
        --output_dir $infer_output_dir \
        --language $language \
        --number $sample_number \
        --dataset $dataset_path \
        --multi

    # assign scores
    python $code_dir/mark.py \
    --mode $eval_mode \
    --question $infer_output_dir/asr_text.jsonl \
    --answer $infer_output_dir/asr_text.jsonl \
    --answer_text $infer_output_dir/output_with_text.jsonl \
    --output_dir $eval_output_dir \
    --dataset $dataset_name \
    --dataset_path $dataset_path \
    --language $language \
    --audio_dir $infer_output_dir/$audio_prompt

done

# conclusion
python $code_dir/evaluate.py --eval_dir ${log_dir}/eval