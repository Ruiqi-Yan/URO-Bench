#!/bin/bash
export CUDA_VISIBLE_DEVICES=4
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export LD_LIBRARY_PATH=/home/visitor/miniconda3/envs/yrq-omni/lib:$LD_LIBRARY_PATH
export PYDEVD_WARN_SLOW_RESOLVE_TIMEOUT=2
export CUDA_LAUNCH_BLOCKING=1


code_dir=/data/ruiqi.yan/SLAM-LLM/examples/benchmark

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
num_latency_tokens=5    # number of latency tokens (same as the number in training)
do_layershift=false      # if false, tokens in each layers use the same codebook, otherwise, use different codebooks

ckpt_path=/data/ruiqi.yan/omni_models/model_multi/ablation-UltraChat/Qwen2-0.5b-gpu16-btz1-lr5e-4-nofp16-epochs10-whisper_small-latency5-group3-UltraChat-from_pretrain-FFT-s2s_epoch_2_step_17364
split=test

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
val_data_name="alpacaeval_test"
val_data_path=/data/ruiqi.yan/data/final/${val_data_name}/test.jsonl
data_number=199

# huggingface dataset
# manifest_format=datasets
# val_data_path="/data/ruiqi.yan/data/voicebench"
# val_data_name="sd-qa"     # alpacaeval，commoneval，sd-qa
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
audio_prompt_path=/data/ruiqi.yan/SLAM-LLM/examples/s2s/prompt/en/prompt_6.wav       # replace this with your own audio prompt path or our provided audio prompt path
audio_prompt=prompt_6

decode_log=$ckpt_path/s2s_decode_${split}_trp${text_repetition_penalty}_arp${audio_repetition_penalty}_seed${dataset_sample_seed}_greedy_${val_data_name}
if [ "$do_sample" = true ] ; then
    decode_log=$ckpt_path/s2s_decode_${split}_trp${text_repetition_penalty}_arp${audio_repetition_penalty}_seed${dataset_sample_seed}_sampling_topk${top_k}_topp${top_p}_temp${temperature}_${val_data_name}
fi

if [ "$decode_text_only" = true ] ; then
    decode_log=$decode_log"_text_only"
fi

cd /data/ruiqi.yan/data/final/${val_data_name}
# -m debugpy --listen 5678 --wait-for-client
python $code_dir/s2s/inference_s2s.py \
        --config-path "conf" \
        --config-name "prompt.yaml" \
        hydra.run.dir=$ckpt_path \
        ++model_config.llm_name=qwen2-0.5b \
        ++model_config.file=$code_dir/s2s/model/slam_model_s2s.py:model_factory \
        ++model_config.llm_path=$llm_path \
        ++model_config.llm_dim=$llm_dim \
        ++model_config.encoder_name=whisper \
        ++model_config.encoder_projector_ds_rate=5 \
        ++model_config.encoder_path=$speech_encoder_path \
        ++model_config.encoder_dim=$encoder_dim \
        ++model_config.encoder_projector=linear \
        ++model_config.codec_decoder_path=$codec_decoder_path \
        ++model_config.codec_decode=true \
        ++model_config.tts_adapter=$tts_adapter \
        ++model_config.vocab_config.code_layer=$code_layer \
        ++model_config.vocab_config.total_audio_vocabsize=$total_audio_vocabsize \
        ++model_config.vocab_config.total_vocabsize=$total_vocabsize \
        ++model_config.code_type=$code_type \
        ++model_config.codec_decoder_type=$codec_decoder_type \
        ++model_config.group_decode=$group_decode \
        ++model_config.group_decode_adapter_type=$group_decode_adapter_type \
        ++dataset_config.dataset=speech_dataset_s2s \
        ++dataset_config.val_data_path=$val_data_path \
        ++dataset_config.file=$code_dir/s2s/speech_dataset_s2s.py:get_speech_dataset \
        ++dataset_config.train_data_path=$val_data_path \
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
        ++decode_config.upsampling_factor=$upsampling_factor \
        ++decode_config.do_layershift=$do_layershift \
        ++decode_log=$decode_log \
        ++ckpt_path=$ckpt_path/model.pt \
        ++output_text_only=$output_text_only \
        ++inference_online=$inference_online \
        ++speech_sample_rate=$speech_sample_rate \
        ++audio_prompt_path=$audio_prompt_path \
        ++log_config.log_file="/data/ruiqi.yan/exp/s2s/debug/inference.log" \
        ++log_config.wandb_dir="/data/ruiqi.yan/exp/wandb_log"    # put your log_file here

# bash ./examples/s2s/scripts/inference/inference_s2s_cosyvoice_group.sh