import os
import csv

from tqdm import tqdm
os.environ["USE_PEFT_BACKEND"] = "True"
import paddle
paddle.device.set_device('gpu:0')
import numpy as np
from PIL import Image
from pcm_flux_fm_deterministic_scheduler import FLUXPCMFMDeterministicScheduler
# from pcm_fm_stochastic_scheduler import PCMFMStochasticScheduler
from ppdiffusers import FluxPipeline
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--guidance_scale", type=float, default=3.5, help="Set guidance scale")
parser.add_argument(
    "--path_lora", 
    nargs="+",  # 支持多个参数
    required=True,
    help="List of LoRA weight paths"
)

args = parser.parse_args()


# csv_path = "/root/paddlejob/workspace/env_run/output/yjx/coco10k/coco10k/subset.csv"
save_root = "/root/paddlejob/workspace/env_run/output/gxl/pcm_eval_results_flux_coco1k"

# path_to_lora = "/root/paddlejob/workspace/env_run/output/yjx/PaddleMIX/ppdiffusers/examples/phased_consistency_distillation/outputs/lora_64_fuyun_PCM_sd3_202504142205/paddle_lora_weights.safetensors"
# path_to_lora = "/root/paddlejob/workspace/env_run/output/yjx/PaddleMIX/ppdiffusers/examples/phased_consistency_distillation/outputs/lora_64_fuyun_PCM_sd3_202504161140/paddle_lora_weights.safetensors"
# path_to_lora = "/root/paddlejob/workspace/env_run/output/yjx/PaddleMIX/ppdiffusers/examples/phased_consistency_distillation/outputs/lora_64_fuyun_PCM_sd3_202504181911/paddle_lora_weights.safetensors"
# path_to_lora = "/root/paddlejob/workspace/env_run/output/yjx/PaddleMIX/ppdiffusers/examples/phased_consistency_distillation/outputs/lora_64_fuyun_PCM_sd3_202504212000/paddle_lora_weights.safetensors"
# path_to_lora = "/root/paddlejob/workspace/env_run/output/yjx/PaddleMIX/ppdiffusers/examples/phased_consistency_distillation/outputs/lora_64_fuyun_PCM_sd3_202504251540/paddle_lora_weights.safetensors"

# path_to_lora = "/root/paddlejob/workspace/env_run/output/yjx/PaddleMIX/ppdiffusers/examples/phased_consistency_distillation/outputs/lora_64_fuyun_PCM_sd3_202504271200/checkpoint-20000/paddle_lora_weights.safetensors"
# path_to_lora = "/root/paddlejob/workspace/env_run/output/yjx/flux_pcm/PaddleMIX/ppdiffusers/examples/phased_consistency_distillation/outputs/lora_64_fuyun_PCM_flux_202505301409/paddle_lora_weights.safetensors"
# path_to_lora = "/root/paddlejob/workspace/env_run/output/yjx/flux_pcm/PaddleMIX/ppdiffusers/examples/phased_consistency_distillation/outputs/paddle_lora_weights_202506070019.safetensors"

# path_to_lora = "/root/paddlejob/workspace/env_run/test_data/lora_64_fuyun_PCM_flux_202506111424/paddle_lora_weights.safetensors"
path_lora = [
# "/root/paddlejob/workspace/env_run/test_data/lora_64_fuyun_PCM_flux_202506172108/paddle_lora_weights.safetensors",
# "/root/paddlejob/workspace/env_run/test_data/lora_64_fuyun_PCM_flux_202506181652/paddle_lora_weights.safetensors",
# "/root/paddlejob/workspace/env_run/test_data/lora_64_fuyun_PCM_flux_202506192039/paddle_lora_weights.safetensors",
# "/root/paddlejob/workspace/env_run/test_data/lora_64_fuyun_PCM_flux_202506201725/paddle_lora_weights.safetensors",
# "/root/paddlejob/workspace/env_run/test_data/lora_64_fuyun_PCM_flux_202506210928/paddle_lora_weights.safetensors",
"/root/paddlejob/workspace/env_run/test_data/lora_64_fuyun_PCM_flux_202507041033/paddle_lora_weights.safetensors",
"/root/paddlejob/workspace/env_run/test_data/lora_64_fuyun_PCM_flux_202506221639/paddle_lora_weights.safetensors",
"/root/paddlejob/workspace/env_run/test_data/lora_64_fuyun_PCM_flux_202507041009/paddle_lora_weights.safetensors"

]
path_lora= args.path_lora
pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        # scheduler=scheduler,
        map_location="cpu",
        paddle_dtype=paddle.bfloat16
        # paddle_dtype="bfloat16"
    )
for path_to_lora in path_lora:
    prompt_path = "/root/paddlejob/workspace/env_run/test_data/coco1k/coco1k.tsv"

    scheduler_type = "deterministic" # deterministic or stochastic
    num_pcm_timesteps = 100
    guidance_scale = args.guidance_scale
    
    height = 1024
    width = 1024
    batch_size = 1
    lora_scale = 0.25

    step = 4
    shift = 3
    seed = 42
    prompt_version = "v2" if "v2" in prompt_path else "v1"

    if len(path_to_lora.split('/')) == 16:
        checkpoint_version = path_to_lora.split("/")[-2].split("-")[-1]
        checkpoint_date = path_to_lora.split("/")[-3].split("_")[-1]
    else:
        checkpoint_version = "latest"
        checkpoint_date = path_to_lora.split("/")[-2].split("_")[-1]

    if scheduler_type == "deterministic":
        scheduler = FLUXPCMFMDeterministicScheduler(
            1000, 
            shift, 
            num_pcm_timesteps,
            base_image_seq_len=256,
            base_shift=0.5,
            max_image_seq_len=4096,
            max_shift=1.15,
            use_dynamic_shifting=True,
            )
    elif scheduler_type == "stochastic":
        scheduler = PCMFMStochasticScheduler(1000, shift, num_pcm_timesteps)
    save_dir_name = f"date{checkpoint_date}_step{step}_shift{shift}__num{num_pcm_timesteps}_gs{guidance_scale}_seed{seed}_h{height}_w{width}_bs{batch_size}_sched-{scheduler_type}_cp-{checkpoint_version}_ls-{lora_scale}_promptver-{prompt_version}"
    save_dir = os.path.join(save_root, save_dir_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    import pickle
    import pandas as pd
    # 读取 .tsv 文件（tab 分隔）
    df = pd.read_csv(os.path.join("/root/paddlejob/workspace/env_run/test_data/coco1k","coco1k.tsv"), sep="\t")
    # 假设列名为 "prompt"，提取成 list
    captions = df['caption_en'].tolist()
    import gc
    del pipe
    gc.collect()
    paddle.device.cuda.empty_cache()

    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        # scheduler=scheduler,
        map_location="cpu",
        paddle_dtype=paddle.bfloat16
        # paddle_dtype="bfloat16"
    )
    
    pipe.load_lora_weights(path_to_lora)

    print(
        pipe.text_encoder.dtype,
        pipe.text_encoder_2.dtype,
        # pipe.text_encoder_3.dtype,
        pipe.transformer.dtype,
        pipe.vae.dtype,
        pipe.dtype
    )

    for i, prompt in tqdm(enumerate(captions), total=len(captions)):
        # if i > 1:
        #     break
        print(f"{i}: {prompt}")
        with paddle.no_grad():
            result_image = pipe(
                prompt=prompt,
                negative_prompt="",
                height=height,
                width=width,
                num_inference_steps=step,
                guidance_scale=guidance_scale,
                generator=paddle.Generator().manual_seed(seed),
                joint_attention_kwargs={"scale": lora_scale}  # for lora scaling
            ).images[0]
        
        save_path = os.path.join(save_dir, f'{i}.png')
        result_image.save(save_path)