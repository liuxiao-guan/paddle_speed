CUDA_VISIBLE_DEVICES=3 python generation_bf16.py \
--inference_step 50 \
--seed 124 \
--dataset 'DrawBench' \
--anno_path /root/paddlejob/workspace/env_run/test_data/coco1k \
--otherblock_predicterror_taylor

CUDA_VISIBLE_DEVICES=3 python generation_bf16.py \
--inference_step 50 \
--seed 124 \
--dataset 'coco1k' \
--anno_path /root/paddlejob/workspace/env_run/test_data/coco1k \
--otherblock_predicterror_taylor


CUDA_VISIBLE_DEVICES=3 python evaluation.py \
--inference_step 50 \
--seed 124 \
--training_path /root/paddlejob/workspace/env_run/test_data/coco1k/1k \
--generation_path /root/paddlejob/workspace/env_run/gxl/output/PaddleMIX/inf_speed_bf16/origin_50steps_coco1k \
--speed_generation_path /root/paddlejob/workspace/env_run/gxl/output/PaddleMIX/inf_speed_bf16/fifthblock_predicterror_taylor0.13_coco1k \
--resolution 1024 

# CUDA_VISIBLE_DEVICES=3 python generation_bf16_1.py \
# --inference_step 50 \
# --seed 124 \
# --dataset 'DrawBench'  \
# --anno_path /root/paddlejob/workspace/env_run/test_data/coco1k \
# --firstblock_threshold 0.09 \
# --firstblock_predicterror_taylor

# CUDA_VISIBLE_DEVICES=3 python evaluation.py \
# --inference_step 50 \
# --seed 124 \
# --training_path /root/paddlejob/workspace/env_run/test_data/coco1k/1k \
# --generation_path /root/paddlejob/workspace/env_run/gxl/output/PaddleMIX/inf_speed_bf16/origin_50steps_coco1k \
# --speed_generation_path /root/paddlejob/workspace/env_run/gxl/output/PaddleMIX/inf_speed_bf16/firstblock_predicterror_taylor0.09_coco1k \
# --resolution 1024 

CUDA_VISIBLE_DEVICES=3 python generation_bf16_1.py \
--inference_step 50 \
--seed 124 \
--dataset 'coco1k'   \
--anno_path /root/paddlejob/workspace/env_run/test_data/coco1k \
--firstblock_threshold 0.18 \
--firstblock_predicterror_taylor

# CUDA_VISIBLE_DEVICES=3 python generation_bf16_1.py \
# --inference_step 50 \
# --seed 124 \
# --dataset 'coco1k'  \
# --anno_path /root/paddlejob/workspace/env_run/test_data/coco1k \
# --firstblock_threshold 0.16 \
# --firstblock_predicterror_taylor




CUDA_VISIBLE_DEVICES=3 python evaluation.py \
--inference_step 50 \
--seed 124 \
--training_path /root/paddlejob/workspace/env_run/test_data/coco1k/1k \
--generation_path /root/paddlejob/workspace/env_run/gxl/output/PaddleMIX/inf_speed_bf16/origin_50steps_coco1k \
--speed_generation_path /root/paddlejob/workspace/env_run/gxl/output/PaddleMIX/inf_speed_bf16/firstblock_predicterror_taylor0.18_coco1k \
--resolution 1024
# CUDA_VISIBLE_DEVICES=1 python generation_bf16.py \
# --inference_step 50 \
# --seed 124 \
# --dataset 'coco1k' \
# --anno_path /root/paddlejob/workspace/env_run/test_data/coco1k \
# --firstblock_pre_predicterror_taylor_timede


# CUDA_VISIBLE_DEVICES=5 python generation_bf16_1.py \
# --inference_step 50 \
# --seed 124 \
# --dataset 'coco1k' \
# --anno_path /root/paddlejob/workspace/env_run/test_data/coco1k \
# --pab

# CUDA_VISIBLE_DEVICES=5 python generation_bf16_1.py \
# --inference_step 50 \
# --seed 124 \
# --dataset 'coco1k' \
# --anno_path /root/paddlejob/workspace/env_run/test_data/coco1k \
# --blockdance

# CUDA_VISIBLE_DEVICES=5 python generation_bf16.py \
# --inference_step 50 \
# --seed 124 \
# --dataset 'coco1k' \
# --anno_path /root/paddlejob/workspace/env_run/test_data/coco1k \
# --blockdance







# CUDA_VISIBLE_DEVICES=1 python generation_bf16.py \
# --inference_step 50 \
# --seed 124 \
# --dataset 'DrawBench' \
# --anno_path /root/paddlejob/workspace/env_run/test_data/coco1k \
# --firstblock_pre_predicterror_taylor_timede


# CUDA_VISIBLE_DEVICES=5 python evaluation.py \
# --inference_step 50 \
# --seed 124 \
# --training_path /root/paddlejob/workspace/env_run/test_data/coco1k/1k \
# --generation_path /root/paddlejob/workspace/env_run/gxl/output/PaddleMIX/inf_speed_bf16/origin_50steps_coco1k \
# --speed_generation_path /root/paddlejob/workspace/env_run/gxl/output/PaddleMIX/inf_speed_bf16/taylorseer_N3O3_coco1k \
# --resolution 1024 



# CUDA_VISIBLE_DEVICES=5 python evaluation.py \
# --inference_step 50 \
# --seed 124 \
# --training_path /root/paddlejob/workspace/env_run/test_data/coco1k/1k \
# --generation_path /root/paddlejob/workspace/env_run/gxl/output/PaddleMIX/inf_speed_bf16/origin_50steps_coco1k \
# --speed_generation_path /root/paddlejob/workspace/env_run/gxl/output/PaddleMIX/inf_speed_bf16/pab_R100-950_N4_coco1k \
# --resolution 1024 




# CUDA_VISIBLE_DEVICES=5 python evaluation.py \
# --inference_step 50 \
# --seed 124 \
# --training_path /root/paddlejob/workspace/env_run/test_data/coco1k/1k \
# --generation_path /root/paddlejob/workspace/env_run/gxl/output/PaddleMIX/inf_speed_bf16/origin_50steps_coco1k \
# --speed_generation_path /root/paddlejob/workspace/env_run/gxl/output/PaddleMIX/inf_speed_bf16/blockdance_R900-100_B30-15_N4_coco1k \
# --resolution 1024 



