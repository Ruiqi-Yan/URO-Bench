export CUDA_VISIBLE_DEVICES=0
export HF_ENDPOINT=https://hf-mirror.com

cd /data/xiquan.li/slam-omni/examples/benchmark/GLM-4-Voice-test
val_data_name=sd-qa   # alpacaeval，commoneval，sd-qa

# -m debugpy --listen 6666 --wait-for-client
python inference_eval.py \
    --input-mode audio --output-dir /data/xiquan.li/exps/glm-4-voice \
    --val_data_path "/data/ruiqi.yan/data/voicebench" \
    --val_data_name $val_data_name
