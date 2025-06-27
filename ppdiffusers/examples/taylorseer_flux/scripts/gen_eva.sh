CUDA_VISIBLE_DEVICES=3 python generation.py \
--inference_step 50 \
--seed 124 \
--dataset '300Prompt' \
--anno_path /root/paddlejob/workspace/env_run/test_data/coco1k \
--sanaprint

CUDA_VISIBLE_DEVICES=3 python generation.py \
--inference_step 50 \
--seed 124 \
--dataset 'coco1k' \
--anno_path /root/paddlejob/workspace/env_run/test_data/coco1k \
--sanaprint

