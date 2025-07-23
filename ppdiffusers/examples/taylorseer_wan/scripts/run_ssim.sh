# python common_metrics/eval.py --gt_video_dir aa --generated_video_dir bb

#python common_metrics/eval.py --gt_video_dir /root/paddlejob/workspace/env_run/gxl/output/PaddleMIX/inf_speed_wan/origin_fs5_50steps --generated_video_dir /root/paddlejob/workspace/env_run/gxl/output/PaddleMIX/inf_speed_wan/pab_N20_B100-950



#python common_metrics/eval.py --gt_video_dir /root/paddlejob/workspace/env_run/gxl/output/PaddleMIX/inf_speed_wan/origin_fs5_50steps --generated_video_dir /root/paddlejob/workspace/env_run/gxl/output/PaddleMIX/inf_speed_wan/pab_N20_B100-950
#python common_metrics/eval.py --gt_video_dir /root/paddlejob/workspace/env_run/gxl/output/PaddleMIX/inf_speed_wan/origin_fs5_50steps --generated_video_dir /root/paddlejob/workspace/env_run/gxl/output/PaddleMIX/inf_speed_wan/taylorseer_fs5_N5

CUDA_VISIBLE_DEVICES=1 python common_metrics/eval.py --gt_video_dir /root/paddlejob/workspace/env_run/gxl/output/PaddleMIX/inf_speed_wan/taylorseer_fs5_N5 --generated_video_dir /root/paddlejob/workspace/env_run/gxl/output/PaddleMIX/inf_speed_wan/origin_fs5_50steps
CUDA_VISIBLE_DEVICES=1 python common_metrics/eval.py --gt_video_dir /root/paddlejob/workspace/env_run/gxl/output/PaddleMIX/inf_speed_wan/teacache0.26_fs5 --generated_video_dir /root/paddlejob/workspace/env_run/gxl/output/PaddleMIX/inf_speed_wan/origin_fs5_50steps
CUDA_VISIBLE_DEVICES=3 python common_metrics/eval.py --gt_video_dir /root/paddlejob/workspace/env_run/gxl/output/PaddleMIX/inf_speed_wan/firstpredict_fs5_cnt5_rel0.36_bO3 --generated_video_dir /root/paddlejob/workspace/env_run/gxl/output/PaddleMIX/inf_speed_wan/origin_fs5_50steps
