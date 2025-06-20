import os
import csv
os.environ["USE_PEFT_BACKEND"] = "True"
import paddle
paddle.device.set_device('gpu:0')
import numpy as np
from PIL import Image
from pcm_fm_deterministic_scheduler import PCMFMDeterministicScheduler
from pcm_fm_stochastic_scheduler import PCMFMStochasticScheduler
from ppdiffusers import StableDiffusion3Pipeline

csv_path = "/root/paddlejob/workspace/env_run/gxl/output/PaddleMIX/inf_speed/coco10k/subset.csv"
save_root = "/root/paddlejob/workspace/env_run/yjx/pcm_eval_results_v2"
# path_to_lora = "/root/paddlejob/workspace/env_run/yjx/PaddleMIX/ppdiffusers/examples/phased_consistency_distillation/outputs/lora_64_fuyun_PCM_sd3_202504142205/paddle_lora_weights.safetensors"
# path_to_lora = "/root/paddlejob/workspace/env_run/yjx/PaddleMIX/ppdiffusers/examples/phased_consistency_distillation/outputs/lora_64_fuyun_PCM_sd3_202504161140/paddle_lora_weights.safetensors"
# path_to_lora = "/root/paddlejob/workspace/env_run/yjx/PaddleMIX/ppdiffusers/examples/phased_consistency_distillation/outputs/lora_64_fuyun_PCM_sd3_202504181911/paddle_lora_weights.safetensors"
# path_to_lora = "/root/paddlejob/workspace/env_run/yjx/PaddleMIX/ppdiffusers/examples/phased_consistency_distillation/outputs/lora_64_fuyun_PCM_sd3_202504212000/paddle_lora_weights.safetensors"
# path_to_lora = "/root/paddlejob/workspace/env_run/yjx/PaddleMIX/ppdiffusers/examples/phased_consistency_distillation/outputs/lora_64_fuyun_PCM_sd3_202504251540/paddle_lora_weights.safetensors"

# path_to_lora = "/root/paddlejob/workspace/env_run/yjx/PaddleMIX/ppdiffusers/examples/phased_consistency_distillation/outputs/lora_64_fuyun_PCM_sd3_202504271200/checkpoint-20000/paddle_lora_weights.safetensors"
path_to_lora = "/root/paddlejob/workspace/env_run/yjx/PaddleMIX/ppdiffusers/examples/phased_consistency_distillation/outputs/lora_64_fuyun_PCM_sd3_202505101606/checkpoint-20000/paddle_lora_weights.safetensors"

scheduler_type = "deterministic" # deterministic or stochastic
num_pcm_timesteps = 100
guidance_scale = 1.0
height = 512
width = 512
batch_size = 1

step = 4
shift = 3
seed = 42

if len(path_to_lora.split('/')) == 14:
    checkpoint_version = path_to_lora.split("/")[-2].split("-")[-1]
    checkpoint_date = path_to_lora.split("/")[-3].split("_")[-1]
else:
    checkpoint_version = "latest"
    checkpoint_date = path_to_lora.split("/")[-2].split("_")[-1]

if scheduler_type == "deterministic":
    scheduler = PCMFMDeterministicScheduler(1000, shift, num_pcm_timesteps)
elif scheduler_type == "stochastic":
    scheduler = PCMFMStochasticScheduler(1000, shift, num_pcm_timesteps)
save_dir_name = f"date{checkpoint_date}_step{step}_shift{shift}__num{num_pcm_timesteps}_gs{guidance_scale}_seed{seed}_h{height}_w{width}_bs{batch_size}_sched-{scheduler_type}_cp-{checkpoint_version}"
save_dir = os.path.join(save_root, save_dir_name)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

file_names = []
captions = []
with open(csv_path, 'r', encoding='utf-8') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)  # 跳过标题行
    for row in reader:
        file_names.append(row[1])  # 第二列是file_name
        captions.append(row[2])     # 第三列是caption
# 打印前3条结果验证
print("File Names:", file_names[:3])
print("Captions:", captions[:3])

pipe = StableDiffusion3Pipeline.from_pretrained(
    "stabilityai/stable-diffusion-3-medium-diffusers",
    scheduler=scheduler,
    map_location="cpu",
    paddle_dtype=paddle.float16
)
pipe.load_lora_weights(path_to_lora)

print(
    pipe.text_encoder.dtype,
    pipe.text_encoder_2.dtype,
    pipe.text_encoder_3.dtype,
    pipe.transformer.dtype,
    pipe.vae.dtype,
    pipe.dtype
)

prompts = []
file_names_list = []
for i, (file_name, prompt) in enumerate(zip(file_names, captions)):
    print("Processing image {}/{}".format(i + 1, len(file_names)))
    prompts.append(prompt)
    file_names_list.append(file_name)
    if len(prompts) == batch_size or i == len(file_names)-1:
        if len(prompts) == 1:
            prompts = prompts[0]
        with paddle.no_grad():
            result_image = pipe(
                prompt=prompts,
                negative_prompt="",
                height=height,
                width=width,
                num_inference_steps=step,
                guidance_scale=guidance_scale,
                generator=paddle.Generator().manual_seed(seed),
                joint_attention_kwargs={"scale": 0.125}  # for lora scaling
            ).images
        
        for file_name, img in zip(file_names_list, result_image):
            save_path = os.path.join(save_dir, file_name)
            img.save(save_path)

        prompts = []
        file_names_list = []