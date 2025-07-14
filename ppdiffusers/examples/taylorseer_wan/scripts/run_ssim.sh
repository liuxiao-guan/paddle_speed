# python common_metrics/eval.py --gt_video_dir aa --generated_video_dir bb

#python common_metrics/eval.py --gt_video_dir /root/paddlejob/workspace/env_run/gxl/output/PaddleMIX_A800/inf_speed_wan/origin_fs5_50steps --generated_video_dir /root/paddlejob/workspace/env_run/gxl/output/PaddleMIX_A800/inf_speed_wan/pab_N20_B100-950



#python common_metrics/eval.py --gt_video_dir /root/paddlejob/workspace/env_run/gxl/output/PaddleMIX_A800/inf_speed_wan/origin_fs5_50steps --generated_video_dir /root/paddlejob/workspace/env_run/gxl/output/PaddleMIX_A800/inf_speed_wan/pab_N20_B100-950
#python common_metrics/eval.py --gt_video_dir /root/paddlejob/workspace/env_run/gxl/output/PaddleMIX_A800/inf_speed_wan/origin_fs5_50steps --generated_video_dir /root/paddlejob/workspace/env_run/gxl/output/PaddleMIX_A800/inf_speed_wan/taylorseer_fs5_N5

CUDA_VISIBLE_DEVICE=2 python common_metrics/eval.py --gt_video_dir /root/paddlejob/workspace/env_run/gxl/output/PaddleMIX_A800/inf_speed_wan/teacache0.26_fs5 --generated_video_dir /root/paddlejob/workspace/env_run/gxl/output/PaddleMIX_A800/inf_speed_wan/origin_fs5_50steps
