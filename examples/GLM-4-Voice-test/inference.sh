export CUDA_VISIBLE_DEVICES=3
export HF_ENDPOINT=https://hf-mirror.com

cd /data/xiquan.li/GLM-4-Voice-test

# --m debugpy --listen 6666 --wait-for-client
python inference.py \
    --input-mode audio --input-path  '/data/xiquan.li/mini-omni/data/samples/output1.wav' --output-dir single-output-audio
