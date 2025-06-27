
# # 设置模型子目录名称
# NAME="taylorseer_fs5_N5"

# # 设置基础路径
# BASE="/root/paddlejob/workspace/env_run/gxl/output/PaddleMIX/inf_speed_wan"
# VIDEO_PATH="${BASE}/${NAME}"
# SAVE_PATH="${BASE}/eval/${NAME}"

# # 执行评估任务
# CUDA_VISIBLE_DEVICES=4 python vbench/run_vbench.py --video_path "$VIDEO_PATH" --save_path "$SAVE_PATH"
# CUDA_VISIBLE_DEVICES=4 python vbench/cal_vbench.py --score_dir "$SAVE_PATH"



# 设置模型子目录名称
NAME="firstpredict_fs5_cnt2_rel0.15_bO2"

# 设置基础路径
BASE="/root/paddlejob/workspace/env_run/gxl/output/PaddleMIX/inf_speed_wan"
VIDEO_PATH="${BASE}/${NAME}"
SAVE_PATH="${BASE}/eval/${NAME}"

# 执行评估任务
CUDA_VISIBLE_DEVICES=3 python vbench/run_vbench.py --video_path "$VIDEO_PATH" --save_path "$SAVE_PATH"
CUDA_VISIBLE_DEVICES=3 python vbench/cal_vbench.py --score_dir "$SAVE_PATH"




# # 设置模型子目录名称
# NAME="firstpredict_fs5_cnt5_rel0.36_bO3"

# # 设置基础路径
# BASE="/root/paddlejob/workspace/env_run/gxl/output/PaddleMIX/inf_speed_wan"
# VIDEO_PATH="${BASE}/${NAME}"
# SAVE_PATH="${BASE}/eval/${NAME}"

# # 执行评估任务
# CUDA_VISIBLE_DEVICES=4 python vbench/run_vbench.py --video_path "$VIDEO_PATH" --save_path "$SAVE_PATH"
# CUDA_VISIBLE_DEVICES=4 python vbench/cal_vbench.py --score_dir "$SAVE_PATH"
