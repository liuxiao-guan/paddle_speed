CUDA_VISIBLE_DEVICES=4 python generation_wan_video.py \
--inference_step 50 \
--seed 124 \
--dataset 'vbench' \
--repeat 1 \
--firstblock_predicterror_taylor


CUDA_VISIBLE_DEVICES=4 python generation_wan_video.py \
--inference_step 50 \
--seed 256 \
--dataset 'vbench' \
--repeat 2 \
--firstblock_predicterror_taylor

CUDA_VISIBLE_DEVICES=4 python generation_wan_video.py \
--inference_step 50 \
--seed 257 \
--dataset 'vbench' \
--repeat 3 \
--firstblock_predicterror_taylor


CUDA_VISIBLE_DEVICES=4 python generation_wan_video.py \
--inference_step 50 \
--seed 258 \
--dataset 'vbench' \
--repeat 4 \
--firstblock_predicterror_taylor

# CUDA_VISIBLE_DEVICES=4 python generation_wan_video.py \
# --inference_step 50 \
# --seed 124 \
# --dataset 'vbench' \
# --repeat 1 \
# --taylorseer

# CUDA_VISIBLE_DEVICES=2 python evaluation.py \
# --inference_step 50 \
# --seed 124 \
# --training_path /root/paddlejob/workspace/env_run/test_data/coco1k/1k \
# --generation_path /root/paddlejob/workspace/env_run/gxl/output/PaddleMIX/inf_speed_bf16/origin_50steps_coco1k \
# --speed_generation_path /root/paddlejob/workspace/env_run/gxl/output/PaddleMIX/inf_speed_bf16/taylorseer_coco1k \
# --resolution 1024 