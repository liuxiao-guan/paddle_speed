
CUDA_VISIBLE_DEVICES=1 nohup python generation.py \
--gate_step 25 \
--sp_interval 5 \
--fi_interval 1 \
--warm_up 2 \
--inference_step 50 \
--seed 124 \
--tgate > output_tgate_50steps.log 2>&1 &


CUDA_VISIBLE_DEVICES=2 nohup python generation.py \
--gate_step 25 \
--sp_interval 5 \
--fi_interval 1 \
--warm_up 2 \
--inference_step 50 \
--seed 124 \
--origin > output_50steps.log 2>&1 &

### coco1k

CUDA_VISIBLE_DEVICES=3 nohup python generation.py \
--inference_step 50 \
--seed 124 \
--dataset 'coco1k' \
--anno_path /root/paddlejob/workspace/env_run/test_data/coco1k \
--teacache > output.log 2>&1 &



CUDA_VISIBLE_DEVICES=3 python generation.py \
--inference_step 28 \
--seed 124 \
--dataset '300Prompt' \
--anno_path /root/paddlejob/workspace/env_run/test_data/coco1k \
--taylorsteer 


CUDA_VISIBLE_DEVICES=3  nohup python generation.py \
--inference_step 50 \
--seed 124 \
--dataset '300Prompt' \
--anno_path /root/paddlejob/workspace/env_run/test_data/coco1k \
--taylorsteer > output_coco1k_1.log 2>&1 &


CUDA_VISIBLE_DEVICES=2 nohup python generation.py \
--inference_step 50 \
--seed 124 \
--dataset 'coco1k' \
--anno_path /root/paddlejob/workspace/env_run/test_data/coco1k \
--tay_teactrl_block > output_coco1k_1.log 2>&1 &

CUDA_VISIBLE_DEVICES=3 python generation.py \
--inference_step 50 \
--seed 124 \
--dataset 'coco1k' \
--anno_path /root/paddlejob/workspace/env_run/test_data/coco1k \
--teacache_taylor_predict 


CUDA_VISIBLE_DEVICES=2 python generation.py \
--inference_step 28 \
--seed 124 \
--dataset '300Prompt' \
--anno_path /root/paddlejob/workspace/env_run/test_data/coco1k \
--teacache_taylor_predict 


CUDA_VISIBLE_DEVICES=2 python generation.py \
--inference_step 50 \
--seed 124 \
--dataset '300Prompt' \
--anno_path /root/paddlejob/workspace/env_run/test_data/coco1k \
--origin


CUDA_VISIBLE_DEVICES=3 python generation.py \
--inference_step 50 \
--seed 124 \
--dataset '300Prompt' \
--anno_path /root/paddlejob/workspace/env_run/test_data/coco1k \
--firstblock_taylorseer 


CUDA_VISIBLE_DEVICES=4 python generation.py \
--inference_step 50 \
--seed 124 \
--dataset 'coco1k' \
--anno_path /root/paddlejob/workspace/env_run/test_data/coco1k \
--predicterror_taylorseer_block

CUDA_VISIBLE_DEVICES=4 python generation.py \
--inference_step 28 \
--seed 124 \
--dataset '300Prompt' \
--anno_path /root/paddlejob/workspace/env_run/test_data/coco1k \
--firstblock_taylorseer_block 
