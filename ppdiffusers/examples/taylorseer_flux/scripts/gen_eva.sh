CUDA_VISIBLE_DEVICES=5 python generation.py \
--inference_step 50 \
--seed 124 \
--dataset 'coco1k' \
--anno_path /root/paddlejob/workspace/env_run/test_data/coco1k \
--predicterror_taylorseer_block_base