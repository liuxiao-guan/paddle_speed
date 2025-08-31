CUDA_VISIBLE_DEVICES=0 python generation_bf16_geneval.py \
--inference_step 50 \
--seed 124 \
--dataset 'geneval'   \
--anno_path /root/paddlejob/workspace/env_run/test_data/coco1k \
--firstblock_predicterror_taylor

CUDA_VISIBLE_DEVICES=0 python generation_bf16_geneval.py \
--inference_step 50 \
--seed 124 \
--dataset 'geneval'   \
--anno_path /root/paddlejob/workspace/env_run/test_data/coco1k \
--pab

CUDA_VISIBLE_DEVICES=0 python generation_bf16_geneval.py \
--inference_step 50 \
--seed 124 \
--dataset 'geneval'   \
--anno_path /root/paddlejob/workspace/env_run/test_data/coco1k \
--teacache

# 第二部分：切换到 vbench_torch 环境运行
# source /root/miniconda3/etc/profile.d/conda.sh
# conda activate geneval
# METHOD="taylorseer_N5O2_geneval"
# OUTFILE="/root/paddlejob/workspace/env_run/gxl/output/PaddleMIX/inf_speed_bf16/GenEval/${METHOD}/results.jsonl"
# BASE_IMAGE_PATH="/root/paddlejob/workspace/env_run/gxl/output/PaddleMIX/inf_speed_bf16/GenEval/"


# CUDA_VISIBLE_DEVICES=3  python /root/paddlejob/workspace/env_run/gxl/geneval/evaluation/evaluate_images.py \
#     "${BASE_IMAGE_PATH}${METHOD}" \
#     --outfile "$OUTFILE" \
#     --model-path /root/paddlejob/workspace/env_run/gxl/geneval/ckpt/


# CUDA_VISIBLE_DEVICES=3 python /root/paddlejob/workspace/env_run/gxl/geneval/evaluation/summary_scores.py "$OUTFILE"