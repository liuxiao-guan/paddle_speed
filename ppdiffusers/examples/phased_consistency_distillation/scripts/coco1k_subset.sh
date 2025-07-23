LORA_LIST=(
"/root/paddlejob/workspace/env_run/test_data/lora_64_fuyun_PCM_flux_202507112228/checkpoint-3000/paddle_lora_weights.safetensors" 
"/root/paddlejob/workspace/env_run/test_data/lora_64_fuyun_PCM_flux_202507112228/checkpoint-6000/paddle_lora_weights.safetensors" 
"/root/paddlejob/workspace/env_run/test_data/lora_64_fuyun_PCM_flux_202507112228/checkpoint-9000/paddle_lora_weights.safetensors"
"/root/paddlejob/workspace/env_run/test_data/lora_64_fuyun_PCM_flux_202507112228/checkpoint-12000/paddle_lora_weights.safetensors"
"/root/paddlejob/workspace/env_run/test_data/lora_64_fuyun_PCM_flux_202507112228/checkpoint-15000/paddle_lora_weights.safetensors"
)

CUDA_VISIBLE_DEVICES=1 python eval_infer_pcm_flux_coco1k_subset.py  --guidance_scale 3.5 --path_lora ${LORA_LIST[@]}