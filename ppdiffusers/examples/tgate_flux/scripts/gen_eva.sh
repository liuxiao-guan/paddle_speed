CUDA_VISIBLE_DEVICES=6 python generation.py \
--gate_step 25 \
--sp_interval 5 \
--fi_interval 1 \
--warm_up 2 \
--inference_step 50 \
--seed 124 \
--dataset 'coco1k' \
--anno_path /root/paddlejob/workspace/env_run/test_data/coco1k \
--tgate


CUDA_VISIBLE_DEVICES=6 python evaluation.py \
--inference_step 50 \
--seed 124 \
--training_path /root/paddlejob/workspace/env_run/test_data/coco1k/1k \
--generation_path /root/paddlejob/workspace/env_run/gxl/output/PaddleMIX/inf_speed_bf16/origin_50steps_coco1k \
--speed_generation_path /root/paddlejob/workspace/env_run/gxl/output/PaddleMIX/inf_speed_bf16/tgate_50steps_gs25_si5_fi1coco1k \
--resolution 1024 


CUDA_VISIBLE_DEVICES=6 python generation.py \
--gate_step 25 \
--sp_interval 6 \
--fi_interval 4 \
--warm_up 2 \
--inference_step 50 \
--seed 124 \
--dataset 'coco1k' \
--anno_path /root/paddlejob/workspace/env_run/test_data/coco1k \
--tgate


CUDA_VISIBLE_DEVICES=6 python evaluation.py \
--inference_step 50 \
--seed 124 \
--training_path /root/paddlejob/workspace/env_run/test_data/coco1k/1k \
--generation_path /root/paddlejob/workspace/env_run/gxl/output/PaddleMIX/inf_speed_bf16/origin_50steps_coco1k \
--speed_generation_path /root/paddlejob/workspace/env_run/gxl/output/PaddleMIX/inf_speed_bf16/tgate_50steps_gs25_si6_fi4coco1k \
--resolution 1024 


CUDA_VISIBLE_DEVICES=6 python generation.py \
--gate_step 25 \
--sp_interval 20 \
--fi_interval 20 \
--warm_up 2 \
--inference_step 50 \
--seed 124 \
--dataset 'coco1k' \
--anno_path /root/paddlejob/workspace/env_run/test_data/coco1k \
--tgate


CUDA_VISIBLE_DEVICES=6 python evaluation.py \
--inference_step 50 \
--seed 124 \
--training_path /root/paddlejob/workspace/env_run/test_data/coco1k/1k \
--generation_path /root/paddlejob/workspace/env_run/gxl/output/PaddleMIX/inf_speed_bf16/origin_50steps_coco1k \
--speed_generation_path /root/paddlejob/workspace/env_run/gxl/output/PaddleMIX/inf_speed_bf16/tgate_50steps_gs25_si20_fi20coco1k \
--resolution 1024 


