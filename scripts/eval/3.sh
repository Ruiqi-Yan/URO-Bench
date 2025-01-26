#!/bin/bash
export CUDA_VISIBLE_DEVICES=7
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export LD_LIBRARY_PATH=/home/visitor/miniconda3/envs/yrq-omni/lib:$LD_LIBRARY_PATH
export PYDEVD_WARN_SLOW_RESOLVE_TIMEOUT=2
export CUDA_LAUNCH_BLOCKING=1

# code dir
code_dir=/data/ruiqi.yan/SLAM-LLM/examples/benchmark
ckpt_dir=/data/ruiqi.yan/omni_models/mini-omni-test/checkpoint

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

# inference output dir
decode_log=/data/ruiqi.yan/omni_models/mini-omni-test/${val_data_name}

# -m debugpy --listen 5678 --wait-for-client
python $code_dir/mini-omni-test/inference_for_eval.py \
        --dataset $val_data_path \
        --output_dir $decode_log \
        --ckpt_dir $ckpt_dir


output_dir=$decode_log/eval_with_asr/${val_data_name}

python $code_dir/asr_for_eval.py \
        --input_dir $decode_log/audio \
        --model_dir "/data/ruiqi.yan/models/whisper-large-v3" \
        --output_dir $decode_log \
        --number $data_number

# eval mode
# open: alpacaeval_test, commoneval_test, wildchat_test
# semi-open: storal_test, summary_test, truthful_test
# qa: gaokao_test, gsm8k_test, mlc_test
# wer: repeat_test
mode="open"    # open, semi-open, qa, contrast

python $code_dir/mark.py \
        --mode $mode \
        --question $decode_log/question_text \
        --answer $decode_log/asr_text \
        --answer_text $decode_log/pred_text \
        --output_dir $output_dir \
        --dataset $val_data_name \
        --audio_dir $decode_log/audio \
        # --reference $decode_log/gt_text

#!/bin/bash
export CUDA_VISIBLE_DEVICES=7
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export LD_LIBRARY_PATH=/home/visitor/miniconda3/envs/yrq-omni/lib:$LD_LIBRARY_PATH
export PYDEVD_WARN_SLOW_RESOLVE_TIMEOUT=2
export CUDA_LAUNCH_BLOCKING=1

# code dir
code_dir=/data/ruiqi.yan/SLAM-LLM/examples/benchmark
ckpt_dir=/data/ruiqi.yan/omni_models/mini-omni-test/checkpoint

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
val_data_name="commoneval_test"
val_data_path=/data/ruiqi.yan/data/final/${val_data_name}/test.jsonl
data_number=200

# inference output dir
decode_log=/data/ruiqi.yan/omni_models/mini-omni-test/${val_data_name}

# -m debugpy --listen 5678 --wait-for-client
python $code_dir/mini-omni-test/inference_for_eval.py \
        --dataset $val_data_path \
        --output_dir $decode_log \
        --ckpt_dir $ckpt_dir


output_dir=$decode_log/eval_with_asr/${val_data_name}

python $code_dir/asr_for_eval.py \
        --input_dir $decode_log/audio \
        --model_dir "/data/ruiqi.yan/models/whisper-large-v3" \
        --output_dir $decode_log \
        --number $data_number

# eval mode
# open: alpacaeval_test, commoneval_test, wildchat_test
# semi-open: storal_test, summary_test, truthful_test
# qa: gaokao_test, gsm8k_test, mlc_test
# wer: repeat_test
mode="open"    # open, semi-open, qa, contrast

python $code_dir/mark.py \
        --mode $mode \
        --question $decode_log/question_text \
        --answer $decode_log/asr_text \
        --answer_text $decode_log/pred_text \
        --output_dir $output_dir \
        --dataset $val_data_name \
        --audio_dir $decode_log/audio \
        # --reference $decode_log/gt_text

#!/bin/bash
export CUDA_VISIBLE_DEVICES=7
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export LD_LIBRARY_PATH=/home/visitor/miniconda3/envs/yrq-omni/lib:$LD_LIBRARY_PATH
export PYDEVD_WARN_SLOW_RESOLVE_TIMEOUT=2
export CUDA_LAUNCH_BLOCKING=1

# code dir
code_dir=/data/ruiqi.yan/SLAM-LLM/examples/benchmark
ckpt_dir=/data/ruiqi.yan/omni_models/mini-omni-test/checkpoint

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
val_data_name="wildchat_test"
val_data_path=/data/ruiqi.yan/data/final/${val_data_name}/test.jsonl
data_number=349

# inference output dir
decode_log=/data/ruiqi.yan/omni_models/mini-omni-test/${val_data_name}

# -m debugpy --listen 5678 --wait-for-client
python $code_dir/mini-omni-test/inference_for_eval.py \
        --dataset $val_data_path \
        --output_dir $decode_log \
        --ckpt_dir $ckpt_dir


output_dir=$decode_log/eval_with_asr/${val_data_name}

python $code_dir/asr_for_eval.py \
        --input_dir $decode_log/audio \
        --model_dir "/data/ruiqi.yan/models/whisper-large-v3" \
        --output_dir $decode_log \
        --number $data_number

# eval mode
# open: alpacaeval_test, commoneval_test, wildchat_test
# semi-open: storal_test, summary_test, truthful_test
# qa: gaokao_test, gsm8k_test, mlc_test
# wer: repeat_test
mode="open"    # open, semi-open, qa, contrast

python $code_dir/mark.py \
        --mode $mode \
        --question $decode_log/question_text \
        --answer $decode_log/asr_text \
        --answer_text $decode_log/pred_text \
        --output_dir $output_dir \
        --dataset $val_data_name \
        --audio_dir $decode_log/audio \
        # --reference $decode_log/gt_text

#!/bin/bash
export CUDA_VISIBLE_DEVICES=7
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export LD_LIBRARY_PATH=/home/visitor/miniconda3/envs/yrq-omni/lib:$LD_LIBRARY_PATH
export PYDEVD_WARN_SLOW_RESOLVE_TIMEOUT=2
export CUDA_LAUNCH_BLOCKING=1

# code dir
code_dir=/data/ruiqi.yan/SLAM-LLM/examples/benchmark
ckpt_dir=/data/ruiqi.yan/omni_models/mini-omni-test/checkpoint

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
val_data_name="storal_test"
val_data_path=/data/ruiqi.yan/data/final/${val_data_name}/test.jsonl
data_number=201

# inference output dir
decode_log=/data/ruiqi.yan/omni_models/mini-omni-test/${val_data_name}

# -m debugpy --listen 5678 --wait-for-client
python $code_dir/mini-omni-test/inference_for_eval.py \
        --dataset $val_data_path \
        --output_dir $decode_log \
        --ckpt_dir $ckpt_dir


output_dir=$decode_log/eval_with_asr/${val_data_name}

python $code_dir/asr_for_eval.py \
        --input_dir $decode_log/audio \
        --model_dir "/data/ruiqi.yan/models/whisper-large-v3" \
        --output_dir $decode_log \
        --number $data_number

# eval mode
# open: alpacaeval_test, commoneval_test, wildchat_test
# semi-open: storal_test, summary_test, truthful_test
# qa: gaokao_test, gsm8k_test, mlc_test
# wer: repeat_test
mode="semi-open"    # open, semi-open, qa, contrast

python $code_dir/mark.py \
        --mode $mode \
        --question $decode_log/question_text \
        --answer $decode_log/asr_text \
        --answer_text $decode_log/pred_text \
        --output_dir $output_dir \
        --dataset $val_data_name \
        --audio_dir $decode_log/audio \
        --reference $decode_log/gt_text

#!/bin/bash
export CUDA_VISIBLE_DEVICES=7
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export LD_LIBRARY_PATH=/home/visitor/miniconda3/envs/yrq-omni/lib:$LD_LIBRARY_PATH
export PYDEVD_WARN_SLOW_RESOLVE_TIMEOUT=2
export CUDA_LAUNCH_BLOCKING=1

# code dir
code_dir=/data/ruiqi.yan/SLAM-LLM/examples/benchmark
ckpt_dir=/data/ruiqi.yan/omni_models/mini-omni-test/checkpoint

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
val_data_name="summary_test"
val_data_path=/data/ruiqi.yan/data/final/${val_data_name}/test.jsonl
data_number=118

# inference output dir
decode_log=/data/ruiqi.yan/omni_models/mini-omni-test/${val_data_name}

# -m debugpy --listen 5678 --wait-for-client
python $code_dir/mini-omni-test/inference_for_eval.py \
        --dataset $val_data_path \
        --output_dir $decode_log \
        --ckpt_dir $ckpt_dir


output_dir=$decode_log/eval_with_asr/${val_data_name}

python $code_dir/asr_for_eval.py \
        --input_dir $decode_log/audio \
        --model_dir "/data/ruiqi.yan/models/whisper-large-v3" \
        --output_dir $decode_log \
        --number $data_number

# eval mode
# open: alpacaeval_test, commoneval_test, wildchat_test
# semi-open: storal_test, summary_test, truthful_test
# qa: gaokao_test, gsm8k_test, mlc_test
# wer: repeat_test
mode="semi-open"    # open, semi-open, qa, contrast

python $code_dir/mark.py \
        --mode $mode \
        --question $decode_log/question_text \
        --answer $decode_log/asr_text \
        --answer_text $decode_log/pred_text \
        --output_dir $output_dir \
        --dataset $val_data_name \
        --audio_dir $decode_log/audio \
        --reference $decode_log/gt_text

#!/bin/bash
export CUDA_VISIBLE_DEVICES=7
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export LD_LIBRARY_PATH=/home/visitor/miniconda3/envs/yrq-omni/lib:$LD_LIBRARY_PATH
export PYDEVD_WARN_SLOW_RESOLVE_TIMEOUT=2
export CUDA_LAUNCH_BLOCKING=1

# code dir
code_dir=/data/ruiqi.yan/SLAM-LLM/examples/benchmark
ckpt_dir=/data/ruiqi.yan/omni_models/mini-omni-test/checkpoint

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
val_data_name="truthful_test"
val_data_path=/data/ruiqi.yan/data/final/${val_data_name}/test.jsonl
data_number=470

# inference output dir
decode_log=/data/ruiqi.yan/omni_models/mini-omni-test/${val_data_name}

# -m debugpy --listen 5678 --wait-for-client
python $code_dir/mini-omni-test/inference_for_eval.py \
        --dataset $val_data_path \
        --output_dir $decode_log \
        --ckpt_dir $ckpt_dir


output_dir=$decode_log/eval_with_asr/${val_data_name}

python $code_dir/asr_for_eval.py \
        --input_dir $decode_log/audio \
        --model_dir "/data/ruiqi.yan/models/whisper-large-v3" \
        --output_dir $decode_log \
        --number $data_number

# eval mode
# open: alpacaeval_test, commoneval_test, wildchat_test
# semi-open: storal_test, summary_test, truthful_test
# qa: gaokao_test, gsm8k_test, mlc_test
# wer: repeat_test
mode="semi-open"    # open, semi-open, qa, contrast

python $code_dir/mark.py \
        --mode $mode \
        --question $decode_log/question_text \
        --answer $decode_log/asr_text \
        --answer_text $decode_log/pred_text \
        --output_dir $output_dir \
        --dataset $val_data_name \
        --audio_dir $decode_log/audio \
        --reference $decode_log/gt_text

#!/bin/bash
export CUDA_VISIBLE_DEVICES=7
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export LD_LIBRARY_PATH=/home/visitor/miniconda3/envs/yrq-omni/lib:$LD_LIBRARY_PATH
export PYDEVD_WARN_SLOW_RESOLVE_TIMEOUT=2
export CUDA_LAUNCH_BLOCKING=1

# code dir
code_dir=/data/ruiqi.yan/SLAM-LLM/examples/benchmark
ckpt_dir=/data/ruiqi.yan/omni_models/mini-omni-test/checkpoint

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
val_data_name="gaokao_test"
val_data_path=/data/ruiqi.yan/data/final/${val_data_name}/test.jsonl
data_number=303

# inference output dir
decode_log=/data/ruiqi.yan/omni_models/mini-omni-test/${val_data_name}

# -m debugpy --listen 5678 --wait-for-client
python $code_dir/mini-omni-test/inference_for_eval.py \
        --dataset $val_data_path \
        --output_dir $decode_log \
        --ckpt_dir $ckpt_dir


output_dir=$decode_log/eval_with_asr/${val_data_name}

python $code_dir/asr_for_eval.py \
        --input_dir $decode_log/audio \
        --model_dir "/data/ruiqi.yan/models/whisper-large-v3" \
        --output_dir $decode_log \
        --number $data_number

# eval mode
# open: alpacaeval_test, commoneval_test, wildchat_test
# semi-open: storal_test, summary_test, truthful_test
# qa: gaokao_test, gsm8k_test, mlc_test
# wer: repeat_test
mode="qa"    # open, semi-open, qa, contrast

python $code_dir/mark.py \
        --mode $mode \
        --question $decode_log/question_text \
        --answer $decode_log/asr_text \
        --answer_text $decode_log/pred_text \
        --output_dir $output_dir \
        --dataset $val_data_name \
        --audio_dir $decode_log/audio \
        --reference $decode_log/gt_text

#!/bin/bash
export CUDA_VISIBLE_DEVICES=7
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export LD_LIBRARY_PATH=/home/visitor/miniconda3/envs/yrq-omni/lib:$LD_LIBRARY_PATH
export PYDEVD_WARN_SLOW_RESOLVE_TIMEOUT=2
export CUDA_LAUNCH_BLOCKING=1

# code dir
code_dir=/data/ruiqi.yan/SLAM-LLM/examples/benchmark
ckpt_dir=/data/ruiqi.yan/omni_models/mini-omni-test/checkpoint

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
val_data_name="gsm8k_test"
val_data_path=/data/ruiqi.yan/data/final/${val_data_name}/test.jsonl
data_number=582

# inference output dir
decode_log=/data/ruiqi.yan/omni_models/mini-omni-test/${val_data_name}

# -m debugpy --listen 5678 --wait-for-client
python $code_dir/mini-omni-test/inference_for_eval.py \
        --dataset $val_data_path \
        --output_dir $decode_log \
        --ckpt_dir $ckpt_dir


output_dir=$decode_log/eval_with_asr/${val_data_name}

python $code_dir/asr_for_eval.py \
        --input_dir $decode_log/audio \
        --model_dir "/data/ruiqi.yan/models/whisper-large-v3" \
        --output_dir $decode_log \
        --number $data_number

# eval mode
# open: alpacaeval_test, commoneval_test, wildchat_test
# semi-open: storal_test, summary_test, truthful_test
# qa: gaokao_test, gsm8k_test, mlc_test
# wer: repeat_test
mode="qa"    # open, semi-open, qa, contrast

python $code_dir/mark.py \
        --mode $mode \
        --question $decode_log/question_text \
        --answer $decode_log/asr_text \
        --answer_text $decode_log/pred_text \
        --output_dir $output_dir \
        --dataset $val_data_name \
        --audio_dir $decode_log/audio \
        --reference $decode_log/gt_text

#!/bin/bash
export CUDA_VISIBLE_DEVICES=7
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export LD_LIBRARY_PATH=/home/visitor/miniconda3/envs/yrq-omni/lib:$LD_LIBRARY_PATH
export PYDEVD_WARN_SLOW_RESOLVE_TIMEOUT=2
export CUDA_LAUNCH_BLOCKING=1

# code dir
code_dir=/data/ruiqi.yan/SLAM-LLM/examples/benchmark
ckpt_dir=/data/ruiqi.yan/omni_models/mini-omni-test/checkpoint

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
val_data_name="mlc_test"
val_data_path=/data/ruiqi.yan/data/final/${val_data_name}/test.jsonl
data_number=177

# inference output dir
decode_log=/data/ruiqi.yan/omni_models/mini-omni-test/${val_data_name}

# -m debugpy --listen 5678 --wait-for-client
python $code_dir/mini-omni-test/inference_for_eval.py \
        --dataset $val_data_path \
        --output_dir $decode_log \
        --ckpt_dir $ckpt_dir


output_dir=$decode_log/eval_with_asr/${val_data_name}

python $code_dir/asr_for_eval.py \
        --input_dir $decode_log/audio \
        --model_dir "/data/ruiqi.yan/models/whisper-large-v3" \
        --output_dir $decode_log \
        --number $data_number

# eval mode
# open: alpacaeval_test, commoneval_test, wildchat_test
# semi-open: storal_test, summary_test, truthful_test
# qa: gaokao_test, gsm8k_test, mlc_test
# wer: repeat_test
mode="qa"    # open, semi-open, qa, contrast

python $code_dir/mark.py \
        --mode $mode \
        --question $decode_log/question_text \
        --answer $decode_log/asr_text \
        --answer_text $decode_log/pred_text \
        --output_dir $output_dir \
        --dataset $val_data_name \
        --audio_dir $decode_log/audio \
        --reference $decode_log/gt_text

#!/bin/bash
export CUDA_VISIBLE_DEVICES=7
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export LD_LIBRARY_PATH=/home/visitor/miniconda3/envs/yrq-omni/lib:$LD_LIBRARY_PATH
export PYDEVD_WARN_SLOW_RESOLVE_TIMEOUT=2
export CUDA_LAUNCH_BLOCKING=1

# code dir
code_dir=/data/ruiqi.yan/SLAM-LLM/examples/benchmark
ckpt_dir=/data/ruiqi.yan/omni_models/mini-omni-test/checkpoint

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
val_data_name="repeat_test"
val_data_path=/data/ruiqi.yan/data/final/${val_data_name}/test.jsonl
data_number=252

# inference output dir
decode_log=/data/ruiqi.yan/omni_models/mini-omni-test/${val_data_name}

# -m debugpy --listen 5678 --wait-for-client
python $code_dir/mini-omni-test/inference_for_eval.py \
        --dataset $val_data_path \
        --output_dir $decode_log \
        --ckpt_dir $ckpt_dir


output_dir=$decode_log/eval_with_asr/${val_data_name}

python $code_dir/asr_for_eval.py \
        --input_dir $decode_log/audio \
        --model_dir "/data/ruiqi.yan/models/whisper-large-v3" \
        --output_dir $decode_log \
        --number $data_number

# eval mode
# open: alpacaeval_test, commoneval_test, wildchat_test
# semi-open: storal_test, summary_test, truthful_test
# qa: gaokao_test, gsm8k_test, mlc_test
# wer: repeat_test
mode="wer"    # open, semi-open, qa, contrast

python $code_dir/mark.py \
        --mode $mode \
        --question $decode_log/question_text \
        --answer $decode_log/asr_text \
        --answer_text $decode_log/pred_text \
        --output_dir $output_dir \
        --dataset $val_data_name \
        --audio_dir $decode_log/audio \
        --reference $decode_log/gt_text