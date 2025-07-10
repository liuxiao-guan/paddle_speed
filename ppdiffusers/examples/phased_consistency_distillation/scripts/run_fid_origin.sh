#!/bin/bash

# 设置真实图片路径
REAL_PATH="/root/paddlejob/workspace/env_run/test_data/coco1k/1k"

# 设置包含多个生成图像子文件夹的根路径
GENERATED_ROOT="/root/paddlejob/workspace/env_run/output/gxl/pcm_eval_results_flux_coco1k"

# 设置 FID 分辨率
RES=512
# 输出日志文件
LOG_FILE="scripts/fid_results_origin.txt"
# echo "FID Results Log" > "$LOG_FILE"  # 清空旧文件并写入表头

# 遍历所有子目录
for GENERATED_PATH in "$GENERATED_ROOT"/*/; do
    # 检查路径是否已经记录过
    if grep -q "$GENERATED_PATH" "$LOG_FILE"; then
        echo "[SKIP] 已记录: $GENERATED_PATH"
        continue
    fi
    echo "Running FID for: $GENERATED_PATH"
    # 执行并获取结果
    FID_OUTPUT=$(python scripts/fid_score.py "$REAL_PATH" "$GENERATED_PATH" --resolution $RES)
    
    # 提取数值（假设结果格式中含有 "FID: xx.xx" 或 "FID score: xx.xx"）
    FID_VALUE=$(echo "$FID_OUTPUT" | grep -i "FID" | grep -oE '[0-9]+\.[0-9]+')

    # 写入日志文件
    echo "$GENERATED_PATH : $FID_VALUE" >> "$LOG_FILE"
done


# # 自定义的生成图片路径列表（每个路径为一个子文件夹）
# GENERATED_LIST=(
#     "/root/paddlejob/workspace/env_run/output/gxl/pcm_eval_results_flux_coco1k/date202506302334_step4_shift3__num100_gs3.5_seed42_h1024_w1024_bs1_sched-deterministic_cp-latest_ls-0.25_promptver-v1"
#     "/root/paddlejob/workspace/env_run/output/gxl/pcm_eval_results_flux_coco1k/date202506301812_step4_shift3__num100_gs3_seed42_h1024_w1024_bs1_sched-deterministic_cp-latest_ls-0.25_promptver-v1"
#     "/root/paddlejob/workspace/env_run/output/gxl/pcm_eval_results_flux_coco1k/date202506292205_step4_shift3__num100_gs2_seed42_h1024_w1024_bs1_sched-deterministic_cp-latest_ls-0.25_promptver-v1"
# )

# # 遍历自定义路径并运行 FID
# for GENERATED_PATH in "${GENERATED_LIST[@]}"; do
#     echo "Running FID for: $GENERATED_PATH"
#     python fid_score.py "$REAL_PATH" "$GENERATED_PATH" --resolution $RES
# done