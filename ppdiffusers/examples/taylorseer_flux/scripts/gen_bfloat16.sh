

CUDA_VISIBLE_DEVICES=1 python generation_bf16.py \
--inference_step 50 \
--seed 124 \
--dataset 'coco1k'   \
--anno_path /root/paddlejob/workspace/env_run/test_data/coco1k \
--taylorseer

# CUDA_VISIBLE_DEVICES=1 python generation_bf16.py \
# --inference_step 50 \
# --seed 124 \
# --dataset 'DrawBench' \
# --anno_path /root/paddlejob/workspace/env_run/test_data/coco1k \
# --teacache


# CUDA_VISIBLE_DEVICES=4 python generation_bf16.py \
# --inference_step 50 \
# --seed 124 \
# --dataset 'coco1k' \
# --anno_path /root/paddlejob/workspace/env_run/test_data/coco1k \
# --pab

CUDA_VISIBLE_DEVICES=1 python evaluation.py \
--inference_step 50 \
--seed 124 \
--training_path /root/paddlejob/workspace/env_run/test_data/coco1k/1k \
--generation_path /root/paddlejob/workspace/env_run/gxl/output/PaddleMIX/inf_speed_bf16/origin_50steps_coco1k \
--speed_generation_path /root/paddlejob/workspace/env_run/gxl/output/PaddleMIX/inf_speed_bf16/taylorseer_N4O1_coco1k \
--resolution 1024 

# CUDA_VISIBLE_DEVICES=2 python generation_bf16.py \
# --inference_step 50 \
# --seed 124 \
# --dataset 'coco1k' \
# --anno_path /root/paddlejob/workspace/env_run/test_data/coco1k \
# --teacache


# CUDA_VISIBLE_DEVICES=2 python evaluation.py \
# --inference_step 50 \
# --seed 124 \
# --training_path /root/paddlejob/workspace/env_run/test_data/coco1k/1k \
# --generation_path /root/paddlejob/workspace/env_run/gxl/output/PaddleMIX/inf_speed_bf16/origin_50steps_coco1k \
# --speed_generation_path /root/paddlejob/workspace/env_run/gxl/output/PaddleMIX/inf_speed_bf16/teacache0.20_coco1k_nomid_v \
# --resolution 1024 
