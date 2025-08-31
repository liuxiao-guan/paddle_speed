CUDA_VISIBLE_DEVICES=1 python generation_wan_video.py \
--inference_step 50 \
--seed 42 \
--dataset 'vbench' \
--repeat 0 \
--firstblock_predicterror_taylor

# 设置模型子目录名称
NAME="firstpredict_fs5.0_gc5_PE_cnt5_rel0.36_bO3_nofixseed"

# 设置基础路径
BASE="/root/paddlejob/workspace/env_run/gxl/output/PaddleMIX/inf_speed_wan"
VIDEO_PATH="${BASE}/${NAME}"
SAVE_PATH="${BASE}/eval/${NAME}"

# 第二部分：切换到 vbench_torch 环境运行
source /root/miniconda3/etc/profile.d/conda.sh
conda activate vbench_torch


# 执行评估任务
CUDA_VISIBLE_DEVICES=1 python vbench/run_vbench.py --video_path "$VIDEO_PATH" --save_path "$SAVE_PATH"
CUDA_VISIBLE_DEVICES=1 python vbench/cal_vbench.py --score_dir "$SAVE_PATH"