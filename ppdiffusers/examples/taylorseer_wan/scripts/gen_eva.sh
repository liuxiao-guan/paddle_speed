CUDA_VISIBLE_DEVICES=4 python generation_wan_video.py \
--inference_step 50 \
--seed 42 \
--dataset 'vbench' \
--repeat 0 \
--teacache

# CUDA_VISIBLE_DEVICES=1 python generation_wan_video.py \
# --inference_step 50 \
# --seed 42 \
# --dataset 'vbench' \
# --repeat 0 \
# --pab

# CUDA_VISIBLE_DEVICES=1 python generation_wan_video.py \
# --inference_step 50 \
# --seed 42 \
# --dataset 'vbench' \
# --repeat 0 \
# --pab








# CUDA_VISIBLE_DEVICES=2 python evaluation.py \
# --inference_step 50 \
# --seed 124 \
# --training_path /root/paddlejob/workspace/env_run/test_data/coco1k/1k \
# --generation_path /root/paddlejob/workspace/env_run/gxl/output/PaddleMIX/inf_speed_bf16/origin_50steps_coco1k \
# --speed_generation_path /root/paddlejob/workspace/env_run/gxl/output/PaddleMIX/inf_speed_bf16/taylorseer_coco1k \
# --resolution 1024 