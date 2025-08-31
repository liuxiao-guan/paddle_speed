# python common_metrics/eval.py --gt_video_dir aa --generated_video_dir bb

#python common_metrics/eval.py --gt_video_dir /root/paddlejob/workspace/env_run/gxl/output/PaddleMIX/inf_speed_wan/origin_fs5_50steps --generated_video_dir /root/paddlejob/workspace/env_run/gxl/output/PaddleMIX/inf_speed_wan/pab_N20_B100-950



#python common_metrics/eval.py --gt_video_dir /root/paddlejob/workspace/env_run/gxl/output/PaddleMIX/inf_speed_wan/origin_fs5_50steps --generated_video_dir /root/paddlejob/workspace/env_run/gxl/output/PaddleMIX/inf_speed_wan/pab_N20_B100-950
#python common_metrics/eval.py --gt_video_dir /root/paddlejob/workspace/env_run/gxl/output/PaddleMIX/inf_speed_wan/origin_fs5_50steps --generated_video_dir /root/paddlejob/workspace/env_run/gxl/output/PaddleMIX/inf_speed_wan/taylorseer_fs5_N5

CUDA_VISIBLE_DEVICES=0 python common_metrics/eval.py --gt_video_dir /root/paddlejob/workspace/env_run/gxl/output/PaddleMIX/inf_speed_hunyuan/origin_50steps --generated_video_dir /root/paddlejob/workspace/env_run/gxl/output/PaddleMIX/inf_speed_hunyuan/firstblock_predicterror_taylor0.12BO2 
CUDA_VISIBLE_DEVICES=0 python common_metrics/eval.py --gt_video_dir /root/paddlejob/workspace/env_run/gxl/output/PaddleMIX/inf_speed_hunyuan/origin_50steps --generated_video_dir /root/paddlejob/workspace/env_run/gxl/output/PaddleMIX/inf_speed_hunyuan/taylorseer_N5O1
CUDA_VISIBLE_DEVICES=0 python common_metrics/eval.py --gt_video_dir /root/paddlejob/workspace/env_run/gxl/output/PaddleMIX/inf_speed_hunyuan/origin_50steps --generated_video_dir /root/paddlejob/workspace/env_run/gxl/output/PaddleMIX/inf_speed_hunyuan/pab_456
CUDA_VISIBLE_DEVICES=0 python common_metrics/eval.py --gt_video_dir /root/paddlejob/workspace/env_run/gxl/output/PaddleMIX/inf_speed_hunyuan/origin_50steps --generated_video_dir /root/paddlejob/workspace/env_run/gxl/output/PaddleMIX/inf_speed_hunyuan/teacache0.15