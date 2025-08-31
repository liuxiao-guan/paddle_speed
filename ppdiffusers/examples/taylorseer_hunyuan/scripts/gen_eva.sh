CUDA_VISIBLE_DEVICES=2 python generation_hunyuan_video.py \
--inference_step 50 \
--seed 42 \
--dataset 'vbench' \
--origin


# 设置模型子目录名称
NAME="origin_50steps"

# 设置基础路径
BASE="/root/paddlejob/workspace/env_run/gxl/output/PaddleMIX/inf_speed_hunyuan"
VIDEO_PATH="${BASE}/${NAME}"
SAVE_PATH="${BASE}/eval/${NAME}"

# 第二部分：切换到 vbench_torch 环境运行
source /root/miniconda3/etc/profile.d/conda.sh
conda activate vbench_torch

# 执行评估任务
CUDA_VISIBLE_DEVICES=2 python vbench/run_vbench.py --video_path "$VIDEO_PATH" --save_path "$SAVE_PATH"
CUDA_VISIBLE_DEVICES=2 python vbench/cal_vbench.py --score_dir "$SAVE_PATH"

# CUDA_VISIBLE_DEVICES=2 python evaluation.py \
# --inference_step 50 \
# --seed 124 \
# --training_path /root/paddlejob/workspace/env_run/test_data/coco1k/1k \
# --generation_path /root/paddlejob/workspace/env_run/gxl/output/PaddleMIX/inf_speed_bf16/origin_50steps_coco1k \
# --speed_generation_path /root/paddlejob/workspace/env_run/gxl/output/PaddleMIX/inf_speed_bf16/taylorseer_coco1k \
# --resolution 1024 