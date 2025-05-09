
CUDA_VISIBLE_DEVICES=1 nohup python generation.py \
--model 'flux' \
--gate_step 25 \
--sp_interval 5 \
--fi_interval 1 \
--warm_up 2 \
--inference_step 50 \
--seed 124 \
--tgate > output_tgate_50steps.log 2>&1 &


CUDA_VISIBLE_DEVICES=2 nohup python generation.py \
--model 'flux' \
--gate_step 25 \
--sp_interval 5 \
--fi_interval 1 \
--warm_up 2 \
--inference_step 50 \
--seed 124 \
--origin > output_50steps.log 2>&1 &

### coco1k

CUDA_VISIBLE_DEVICES=3 nohup python generation.py \
--model 'flux' \
--inference_step 50 \
--seed 124 \
--dataset 'coco1k' \
--anno_path /root/paddlejob/workspace/env_run/test_data/coco1k \
--teacache > output.log 2>&1 &



CUDA_VISIBLE_DEVICES=3 nohup python generation.py \
--model 'flux' \
--inference_step 50 \
--seed 124 \
--dataset 'coco1k' \
--anno_path /root/paddlejob/workspace/env_run/test_data/coco1k \
--origin > output_coco1k.log 2>&1 &
