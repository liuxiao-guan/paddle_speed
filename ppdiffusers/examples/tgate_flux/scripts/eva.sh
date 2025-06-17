# CUDA_VISIBLE_DEVICES=2  nohup python evaluation.py \
# --inference_step 50 \
# --seed 124 \
# --training_path /root/paddlejob/workspace/env_run/test_data/coco10k/subset \
# --generation_path /root/paddlejob/workspace/env_run/gxl/output/PaddleMIX/inf_speed/origin_50steps \
# --speed_generation_path /root/paddlejob/workspace/env_run/gxl/output/PaddleMIX/inf_speed/tgate_50steps \
# --resolution 1024 > output.log 2>&1 &


# CUDA_VISIBLE_DEVICES=2  nohup python evaluation.py \
# --inference_step 50 \
# --seed 124 \
# --training_path /root/paddlejob/workspace/env_run/test_data/coco10k/subset \
# --generation_path /root/paddlejob/workspace/env_run/gxl/output/PaddleMIX/inf_speed/origin_50steps \
# --speed_generation_path /root/paddlejob/workspace/env_run/gxl/output/PaddleMIX/inf_speed/teacache \
# --resolution 1024 > output_1.log 2>&1 &


# CUDA_VISIBLE_DEVICES=2  nohup python evaluation.py \
# --inference_step 50 \
# --seed 124 \
# --training_path /root/paddlejob/workspace/env_run/test_data/coco10k/subset \
# --generation_path /root/paddlejob/workspace/env_run/gxl/output/PaddleMIX/inf_speed/origin_50steps \
# --speed_generation_path /root/paddlejob/workspace/env_run/gxl/output/PaddleMIX/inf_speed/teacache_pab \
# --resolution 1024 > output_2.log 2>&1 &


# CUDA_VISIBLE_DEVICES=2  nohup python evaluation.py \
# --inference_step 50 \
# --seed 124 \
# --training_path /root/paddlejob/workspace/env_run/test_data/coco1k/1k \
# --generation_path /root/paddlejob/workspace/env_run/gxl/output/PaddleMIX/inf_speed/origin_50steps_coco1k \
# --speed_generation_path /root/paddlejob/workspace/env_run/gxl/output/PaddleMIX/inf_speed/teacache_coco1k \
# --resolution 1024 > output_1.log 2>&1 &

# ## torch
# CUDA_VISIBLE_DEVICES=2  nohup python evaluation.py \
# --inference_step 50 \
# --seed 124 \
# --training_path /root/paddlejob/workspace/env_run/test_data/coco1k/1k \
# --generation_path /root/paddlejob/workspace/env_run/gxl/output/PaddleMIX/inf_speed/torch_origin_50steps_coco1k \
# --speed_generation_path /root/paddlejob/workspace/env_run/gxl/output/PaddleMIX/inf_speed/torch_taylorsteer_50steps_coco1k_N4O2 \
# --resolution 1024 > output_1.log 2>&1 &



CUDA_VISIBLE_DEVICES=3  nohup python evaluation.py \
--inference_step 50 \
--seed 124 \
--training_path /root/paddlejob/workspace/env_run/test_data/coco1k/1k \
--generation_path /root/paddlejob/workspace/env_run/gxl/output/PaddleMIX/inf_speed/origin_50steps_coco1k \
--speed_generation_path /root/paddlejob/workspace/env_run/gxl/output/PaddleMIX/inf_speed/block \
--resolution 1024 > output_2.log 2>&1 &



CUDA_VISIBLE_DEVICES=2 python evaluation.py \
--inference_step 50 \
--seed 124 \
--training_path /root/paddlejob/workspace/env_run/test_data/coco1k/1k \
--generation_path /root/paddlejob/workspace/env_run/gxl/output/PaddleMIX/inf_speed/origin_50steps_coco1k \
--speed_generation_path /root/paddlejob/workspace/env_run/gxl/output/PaddleMIX/inf_speed/tgate_50steps_coco1k \
--resolution 1024 