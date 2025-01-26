from huggingface_hub import snapshot_download

# 下载到指定路径
home_dir = "PATH/TO/Freeze-Omni"

local_dir = f"{home_dir}/Qwen2-7B-Instruct"
local_dir_2 = f"{home_dir}/checkpoints"

# 仓库 ID && 下载模型
repo_id = "Qwen/Qwen2-7B-Instruct"
snapshot_download(repo_id=repo_id, local_dir=local_dir)

repo_id_2 = "VITA-MLLM/Freeze-Omni"
snapshot_download(repo_id=repo_id_2, local_dir=local_dir_2)

print(f"模型已下载到 {local_dir}")
