# CUDA_VISIBLE_DEVICES=4 python generation.py \
# --inference_step 50 \
# --seed 124 \
# --dataset '300Prompt' \
# --anno_path /root/paddlejob/workspace/env_run/test_data/coco1k \
# --hyper_flux

CUDA_VISIBLE_DEVICES=5 python generation_bf16.py \
--inference_step 50 \
--seed 124 \
--dataset 'DrawBench' \
--anno_path /root/paddlejob/workspace/env_run/test_data/coco1k \
--firstblock_predicterror_taylor

CUDA_VISIBLE_DEVICES=5 python generation_bf16.py \
--inference_step 50 \
--seed 124 \
--dataset 'DrawBench' \
--anno_path /root/paddlejob/workspace/env_run/test_data/coco1k \
--timeemb_predicterror_taylor



