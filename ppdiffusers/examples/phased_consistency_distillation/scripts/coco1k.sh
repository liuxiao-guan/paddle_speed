


LORA_LIST=(
"/root/paddlejob/workspace/env_run/test_data/lora_64_fuyun_PCM_flux_202507122210/paddle_lora_weights.safetensors" 
"/root/paddlejob/workspace/env_run/test_data/lora_64_fuyun_PCM_flux_202507131343/paddle_lora_weights.safetensors" 
"/root/paddlejob/workspace/env_run/test_data/lora_64_fuyun_PCM_flux_202507112228/paddle_lora_weights.safetensors"
)

CUDA_VISIBLE_DEVICES=2 python eval_infer_pcm_flux_coco1k.py  --guidance_scale 3.5 --path_lora ${LORA_LIST[@]}


# CUDA_VISIBLE_DEVICES=4 python eval_infer_pcm_flux_coco1k.py  --guidance_scale 2.5
# CUDA_VISIBLE_DEVICES=4 python eval_infer_pcm_flux_coco1k.py  --guidance_scale 2


# CUDA_VISIBLE_DEVICES=4 python eval_infer_pcm_flux_vis.py  --guidance_scale 3.0
# CUDA_VISIBLE_DEVICES=4 python eval_infer_pcm_flux_vis.py  --guidance_scale 2.5
# CUDA_VISIBLE_DEVICES=4 python eval_infer_pcm_flux_vis.py  --guidance_scale 2




