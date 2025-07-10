# CUDA_VISIBLE_DEVICES=6 python generation_bf16.py \
# --inference_step 50 \
# --seed 124 \
# --dataset 'DrawBench' \
# --anno_path /root/paddlejob/workspace/env_run/test_data/coco1k \
# --origin


# CUDA_VISIBLE_DEVICES=6 python generation_bf16.py \
# --inference_step 50 \
# --seed 42 \
# --dataset 'DrawBench' \
# --anno_path /root/paddlejob/workspace/env_run/test_data/coco1k \
# --taylorseer 



CUDA_VISIBLE_DEVICES=1 python generation_bf16.py \
--inference_step 50 \
--seed 124 \
--dataset 'coco1k' \
--anno_path /root/paddlejob/workspace/env_run/test_data/coco1k \
--teacache


CUDA_VISIBLE_DEVICES=1 python evaluation.py \
--inference_step 50 \
--seed 124 \
--training_path /root/paddlejob/workspace/env_run/test_data/coco1k/1k \
--generation_path /root/paddlejob/workspace/env_run/gxl/output/PaddleMIX/inf_speed_bf16/origin_50steps_coco1k \
--speed_generation_path /root/paddlejob/workspace/env_run/gxl/output/PaddleMIX/inf_speed_bf16/teacache0.15_coco1k \
--resolution 1024 