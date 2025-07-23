
CUDA_VISIBLE_DEVICES=2 python clip_score.py \
--clip_score \
--prompt_type  "drawbench" \
--path "/root/paddlejob/workspace/env_run/gxl/output/PaddleMIX/inf_speed_bf16/DrawBench_firstblock_predicterror_taylor0.05" \
--total_eval_number 200 \


CUDA_VISIBLE_DEVICES=2 python clip_score.py \
--clip_score \
--prompt_type  "drawbench" \
--path "/root/paddlejob/workspace/env_run/gxl/output/PaddleMIX/inf_speed_bf16/DrawBench_firstblock_predicterror_taylor0.08" \
--total_eval_number 200 \



