import os
import argparse
import pickle
from typing_extensions import Self
from numpy import imag
import paddle
from ppdiffusers.models.transformer_flux import FluxTransformer2DModel
from tqdm import tqdm

from ppdiffusers import UNet2DConditionModel, LCMScheduler,FluxPipeline,PyramidAttentionBroadcastConfig, apply_pyramid_attention_broadcast
from ppdiffusers.utils import load_image, export_to_video
from teacache_forward import TeaCacheForward
from forwards import FirstBlock_taylor_predict_Forward,FirstBlock_taylor_block_predict_Forward,Taylor_predicterror_Forward,BlockDanceForward,Taylor_predicterror_base_Forward
# from ..taylorseer_flux.forwards.double_transformer_forward import taylorseer_flux_double_block_forward
# from ..taylorseer_flux.forwards.single_transformer_forward import taylorseer_flux_single_block_forward
# from ..taylorseer_flux.forwards.xfuser_flux_forward import xfuser_flux_forward
# from ..taylorseer_flux.forwards.flux_forward import taylorseer_flux_forward





import sys
sys.stdout.isatty = lambda: False
def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of TGATE V2.")
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="the input prompts",
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="the dir of input image to generate video",
    )
    parser.add_argument(
        "--saved_path",
        type=str,
        default='/root/paddlejob/workspace/env_run/gxl/output/PaddleMIX/inf_speed',
        help="the path to save images",
    )
    parser.add_argument(
        "--model",
        type=str,
        default='pixart',
        help="[pixart_alpha,sdxl,lcm_sdxl,lcm_pixart_alpha,svd]",
    )
    parser.add_argument(
        "--gate_step",
        type=int,
        default=10,
        help="When re-using the cross-attention",
    )
    parser.add_argument(
        '--sp_interval',
        type=int,
        default=5,
        help="The time-step interval to cache self attention before gate_step (Semantics-Planning Phase).",
    )
    parser.add_argument(
        '--fi_interval',
        type=int,
        default=1,
        help="The time-step interval to cache self attention after gate_step (Fidelity-Improving Phase).",
    )
    parser.add_argument(
        '--warm_up',
        type=int,
        default=2,
        help="The time step to warm up the model inference",
    )
    parser.add_argument(
        "--inference_step",
        type=int,
        default=25,
        help="total inference steps",
    )
    parser.add_argument(
        '--deepcache', 
        action='store_true', 
        default=False, 
        help='do deep cache',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed for generation. Set for reproducible results.',
    )
    parser.add_argument(
        '--tgate', 
        action='store_true', 
        default=False, 
        help='do add tgate',
    )
    parser.add_argument(
        '--pab', 
        action='store_true', 
        default=False, 
        help='do add pab',
    )
    parser.add_argument(
        '--blockdance', 
        action='store_true', 
        default=False, 
        help='do add blockdance',
    )
    parser.add_argument(
        '--teacache', 
        action='store_true', 
        default=False, 
        help='do add teacache',
    )
    parser.add_argument(
        '--taylorsteer', 
        action='store_true', 
        default=False, 
        help='do add taylorsteer',
    )
    
    parser.add_argument(
        '--firstblock_taylorseer', 
        action='store_true', 
        default=False, 
        help='do add firstblock taylorsteer',
    )
    
    parser.add_argument(
        '--firstblock_taylorseer_block', 
        action='store_true', 
        default=False, 
        help='do add firstblock taylorsteer block',
    )
    parser.add_argument(
        '--predicterror_taylorseer_block', 
        action='store_true', 
        default=False, 
        help='do add predicterror taylorsteer',
    )
    
    parser.add_argument(
        '--predicterror_taylorseer_block_base', 
        action='store_true', 
        default=False, 
        help='do add predicterror taylorseer block base',
    )
    
    parser.add_argument(
        '--sanaprint', 
        action='store_true', 
        default=False, 
        help='do add sanaprint',
    )
    parser.add_argument(
        '--sana', 
        action='store_true', 
        default=False, 
        help='do add sana',
    )
    
    
    parser.add_argument(
        '--hyper_flux', 
        action='store_true', 
        default=False, 
        help='do add hyper_flux',
    )
    parser.add_argument(
        '--flux_dev', 
        action='store_true', 
        default=False, 
        help='do add hyper_flux',
    )
    
    parser.add_argument(
        '--origin', 
        action='store_true', 
        default=False, 
        help='do add origin',
    )
   
    
    parser.add_argument(
        "--anno_path",
        type=str,
        default='/root/paddlejob/workspace/env_run/test_data/coco10k/all_prompts.pkl',
        help="the path of evaluation annotations",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default='coco10k',
        help="the path of evaluation annotations",
    )
    
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    os.makedirs(args.saved_path, exist_ok=True)
    # 获取prompt
    if args.dataset == "coco10k":
        all_prompts = pickle.load(open(args.anno_path, "rb"))
    elif args.dataset == "300Prompt":
        with open('/root/paddlejob/workspace/env_run/gxl/paddle_speed/ppdiffusers/examples/taylorseer_flux/prompts/prompt.txt', 'r', encoding='utf-8') as f:
            all_prompts = [line.strip() for line in f if line.strip()]
    else:
        import pandas as pd
        # 读取 .tsv 文件（tab 分隔）
        df = pd.read_csv(os.path.join(args.anno_path,"coco1k.tsv"), sep="\t")
        assert args.anno_path == "/root/paddlejob/workspace/env_run/test_data/coco1k"
        # 假设列名为 "prompt"，提取成 list
        all_prompts = df['caption_en'].tolist()
    
    # Create generator if seed is provided
    generator = None
    if args.seed is not None:
        generator = paddle.Generator().manual_seed(args.seed)
    # 原始生成的
    if args.origin == True:
        pipe = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev", paddle_dtype=paddle.float16
        )
        if args.dataset == "coco10k":
            saved_path = os.path.join(args.saved_path,"origin_50steps")
        elif args.dataset == "300Prompt":
            saved_path = os.path.join(args.saved_path,"origin_300")
        else:
            saved_path = os.path.join(args.saved_path,"origin_50steps_coco1k")
        os.makedirs(saved_path, exist_ok=True)
        for i, prompt in enumerate(tqdm(all_prompts)):
            image = pipe(
                prompt=prompt,
                height=1024,
                width=1024,
                guidance_scale=3.5,
                max_sequence_length=512,
                num_inference_steps=args.inference_step,
                generator=generator,
            ).images[0]
            image.save(os.path.join(saved_path, f"{i}.png"))
        
    #加入tgate 方法的
    if args.tgate == True :
        pipe = FluxPipeline.from_pretrained(
                "black-forest-labs/FLUX.1-dev", paddle_dtype=paddle.float16)
        pipe = TgateFLUXLoader(pipe)
        saved_path = os.path.join(args.saved_path,"tgate_50steps")
        os.makedirs(saved_path, exist_ok=True)
        for i, prompt in enumerate(tqdm(all_prompts)):
            # if i == 3:
            #     break
            image = pipe.tgate(
                prompt=prompt,
                height=1024,
                width=1024,
                gate_step=args.gate_step,
                sp_interval=args.sp_interval ,
                fi_interval=args.fi_interval,
                warm_up=args.warm_up,
                num_inference_steps=args.inference_step,
                generator=generator,
            ).images[0]
            image.save(os.path.join(saved_path, f"{i}.png"))
    if args.pab == True:
        pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", paddle_dtype=paddle.float16)

        config = PyramidAttentionBroadcastConfig(
            spatial_attention_block_skip_range=14,
            temporal_attention_block_skip_range = 2,
            cross_attention_block_skip_range = 4,
            spatial_attention_timestep_skip_range=(100, 950),
            current_timestep_callback=lambda: pipe._current_timestep,
        )
        apply_pyramid_attention_broadcast(pipe.transformer, config)
        if args.dataset == "coco10k":
            saved_path = os.path.join(args.saved_path,"pab")
        elif args.dataset == "300Prompt":
            saved_path = os.path.join(args.saved_path,"pab_300")
        else:
            saved_path = os.path.join(args.saved_path,"pab_coco1k")
        os.makedirs(saved_path, exist_ok=True)
        for i, prompt in enumerate(tqdm(all_prompts)):
            image = pipe(
                prompt=prompt,
                height=1024,
                width=1024,
                guidance_scale=3.5,
                max_sequence_length=512,
                num_inference_steps=args.inference_step,
                generator=generator,
            ).images[0]
            image.save(os.path.join(saved_path, f"{i}.png"))
    if args.blockdance == True:
        pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", paddle_dtype=paddle.float16)
        FluxTransformer2DModel.forward = BlockDanceForward

        pipe.transformer.previous_block = None
        pipe.transformer.previous_block_encoder = None
        pipe.transformer.previous_single_block = None
        pipe.transformer.step_start = 100
        pipe.transformer.step_end = 950
        pipe.transformer.block_step_single = 30
        pipe.transformer.block_step = 15
        pipe.transformer.block_step_N = 8
        pipe.transformer.count = 0
        if args.dataset == "coco10k":
            saved_path = os.path.join(args.saved_path,"blockdance")
        elif args.dataset == "300Prompt":
            saved_path = os.path.join(args.saved_path,"blockdance_300")
        else:
            saved_path = os.path.join(args.saved_path,"blockdance_R950_B30-15_N8_coco1k")
        os.makedirs(saved_path, exist_ok=True)
        for i, prompt in enumerate(tqdm(all_prompts)):
            image = pipe(
                prompt=prompt,
                height=1024,
                width=1024,
                guidance_scale=3.5,
                max_sequence_length=512,
                num_inference_steps=args.inference_step,
                generator=generator,
            ).images[0]
            image.save(os.path.join(saved_path, f"{i}.png"))

    # 加入teacache 方法的
    if args.teacache == True :

        pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", paddle_dtype=paddle.float16)
        
        FluxTransformer2DModel.forward = TeaCacheForward
        pipe.transformer.enable_teacache = True
        pipe.transformer.cnt = 0
        pipe.transformer.num_steps = 50
        pipe.transformer.rel_l1_thresh = (
            0.25  # 0.25 for 1.5x speedup, 0.4 for 1.8x speedup, 0.6 for 2.0x speedup, 0.8 for 2.25x speedup
        )
        pipe.transformer.accumulated_rel_l1_distance = 0
        pipe.transformer.previous_modulated_input = None
        pipe.transformer.previous_residual = None
        if args.dataset == "coco10k":
            saved_path = os.path.join(args.saved_path,"teacache")
        elif args.dataset == "300Prompt":
            saved_path = os.path.join(args.saved_path,"teacache_300")
        else:
            saved_path = os.path.join(args.saved_path,"teacache_coco1k")
        os.makedirs(saved_path, exist_ok=True)
        for i, prompt in enumerate(tqdm(all_prompts)):
            image = pipe(
                prompt=prompt,
                height=1024,
                width=1024,
                guidance_scale=3.5,
                max_sequence_length=512,
                num_inference_steps=args.inference_step,
                generator=generator,
            ).images[0]
            image.save(os.path.join(saved_path, f"{i}.png"))


    if args.firstblock_taylorseer == True:
        pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", paddle_dtype=paddle.float16)
        #pipeline.enable_model_cpu_offload() #save some VRAM by offloading the model to CPU. Remove this if you have enough GPU power

        # TaylorSeer settings
        pipe.transformer.__class__.num_steps = args.inference_step

        pipe.transformer.__class__.forward = FirstBlock_taylor_predict_Forward

        # for double_transformer_block in pipe.transformer.transformer_blocks:
        #     double_transformer_block.__class__.forward = taylorseer_flux_double_block_forward
            
        # for single_transformer_block in pipe.transformer.single_transformer_blocks:
        #     single_transformer_block.__class__.forward = taylorseer_flux_single_block_forward

        pipe.transformer.enable_teacache = True
        pipe.transformer.cnt = 0
        pipe.transformer.num_steps = 50
        pipe.transformer.residual_diff_threshold = (
            0.14 #0.05  7.6s 
        )
        pipe.transformer.downsample_factor=(1)
        pipe.transformer.accumulated_rel_l1_distance = 0
        pipe.transformer.prev_first_hidden_states_residual = None
        pipe.transformer.previous_residual = None
        if args.dataset == "coco10k":
            saved_path = os.path.join(args.saved_path,"firstblock_taylorseer")
        elif args.dataset == "300Prompt":
            saved_path = os.path.join(args.saved_path,"firstblock_taylorseer0.07_300")
        else:
            saved_path = os.path.join(args.saved_path,"firstblock_taylorseer0.14_coco1k")
        os.makedirs(saved_path, exist_ok=True)
        seed_list = []
        for i, prompt in enumerate(tqdm(all_prompts)):
            # if i==3 or i == 47 or i == 251 or i==292:
                # a =  paddle.get_cuda_rng_state()
                #paddle.save(generator.get_state(), f"generator_state_{i}.pt")
            image = pipe(
                prompt=prompt,
                height=1024,
                width=1024,
                guidance_scale=3.5,
                max_sequence_length=512,
                num_inference_steps=args.inference_step,
                generator=generator,
            ).images[0]
            image.save(os.path.join(saved_path, f"{i}.png"))
    if args.firstblock_taylorseer_block == True:
        pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", paddle_dtype=paddle.float16)
        #pipeline.enable_model_cpu_offload() #save some VRAM by offloading the model to CPU. Remove this if you have enough GPU power

        # TaylorSeer settings
        pipe.transformer.__class__.num_steps = args.inference_step

        pipe.transformer.__class__.forward = FirstBlock_taylor_block_predict_Forward

        # for double_transformer_block in pipe.transformer.transformer_blocks:
        #     double_transformer_block.__class__.forward = taylorseer_flux_double_block_forward
            
        # for single_transformer_block in pipe.transformer.single_transformer_blocks:
        #     single_transformer_block.__class__.forward = taylorseer_flux_single_block_forward

        pipe.transformer.enable_teacache = True
        pipe.transformer.cnt = 0
        pipe.transformer.num_steps = 28

            

        pipe.transformer.residual_diff_threshold = (
            0.09 #0.05  7.6s 
        )
        pipe.transformer.downsample_factor=(1)
        pipe.transformer.accumulated_rel_l1_distance = 0
        pipe.transformer.prev_first_hidden_states_residual = None
        pipe.transformer.previous_residual = None
        pipe.transformer.previous_block_residual = None
        pipe.transformer.previous_block_encoder_residual = None
        pipe.transformer.previous_single_block_residual = None
        pipe.transformer.step_start = 100
        pipe.transformer.step_end = 800
        pipe.transformer.block_step_single = 28
        pipe.transformer.block_step = 13
        pipe.transformer.block_step_N = 2
        pipe.transformer.count = 0

        if args.dataset == "coco10k":
            saved_path = os.path.join(args.saved_path,"firstblock_taylorseer")
        elif args.dataset == "300Prompt":
            saved_path = os.path.join(args.saved_path,"firstblock_taylorseer_block3_300_28")
        else:
            saved_path = os.path.join(args.saved_path,"firstblock_taylorseer0.07_coco1k")
        os.makedirs(saved_path, exist_ok=True)

        for i, prompt in enumerate(tqdm(all_prompts)):
            # if i==3 or i == 47 or i == 251 or i==292:
                # a =  paddle.get_cuda_rng_state()
                #paddle.save(generator.get_state(), f"generator_state_{i}.pt")
            image = pipe(
                prompt=prompt,
                height=1024,
                width=1024,
                guidance_scale=3.5,
                max_sequence_length=512,
                num_inference_steps=args.inference_step,
                generator=generator,
            ).images[0]
            image.save(os.path.join(saved_path, f"{i}.png"))
   
    if args.predicterror_taylorseer_block == True:
        pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", paddle_dtype=paddle.float16)
        pipe.transformer.__class__.forward = Taylor_predicterror_Forward
        pipe.transformer.enable_teacache = True
        pipe.transformer.cnt = 0
        pipe.transformer.num_steps = 50

        pipe.transformer.prev_first_hidden_states_residual = None
        pipe.transformer.previous_residual = None
        pipe.transformer.pre_compute_hidden =None
        pipe.transformer.predict_loss  = None
        pipe.transformer.predict_hidden_states= None
        pipe.transformer.threshold= 0.42
        if args.dataset == "coco10k":
            saved_path = os.path.join(args.saved_path,"predicterror_taylorseer")
        elif args.dataset == "300Prompt":
            saved_path = os.path.join(args.saved_path,"predicterror_taylorseer0.20_300")
        else:
            saved_path = os.path.join(args.saved_path,"predicterror_taylorseer0.42_coco1k")
        os.makedirs(saved_path, exist_ok=True)
        seed_list = []
        for i, prompt in enumerate(tqdm(all_prompts)):
            # if i==3 or i == 47 or i == 251 or i==292:
                # a =  paddle.get_cuda_rng_state()
                #paddle.save(generator.get_state(), f"generator_state_{i}.pt")
            image = pipe(
                prompt=prompt,
                height=1024,
                width=1024,
                guidance_scale=3.5,
                max_sequence_length=512,
                num_inference_steps=args.inference_step,
                generator=generator,
            ).images[0]
            image.save(os.path.join(saved_path, f"{i}.png"))
    if args.predicterror_taylorseer_block_base == True:
        pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", paddle_dtype=paddle.float16)
        #pipeline.enable_model_cpu_offload() #save some VRAM by offloading the model to CPU. Remove this if you have enough GPU power
        pipe.transformer.__class__.forward = Taylor_predicterror_base_Forward
        pipe.transformer.enable_teacache = True
        pipe.transformer.cnt = 0
        pipe.transformer.num_steps = args.inference_step

        pipe.transformer.prev_first_hidden_states_residual = None
        pipe.transformer.previous_residual = None
        pipe.transformer.pre_compute_hidden =None
        pipe.transformer.predict_loss  = None
        pipe.transformer.predict_hidden_states= None
        pipe.transformer.threshold= 0.25
        if args.dataset == "coco10k":
            saved_path = os.path.join(args.saved_path,"predicterror_taylorseer_base")
        elif args.dataset == "300Prompt":
            saved_path = os.path.join(args.saved_path,"predicterror_taylorseer_base0.20_300")
        else:
            saved_path = os.path.join(args.saved_path,"predicterror_taylorseer_base0.25_coco1k")
        os.makedirs(saved_path, exist_ok=True)
        seed_list = []
        for i, prompt in enumerate(tqdm(all_prompts)):
            # if i==3 or i == 47 or i == 251 or i==292:
                # a =  paddle.get_cuda_rng_state()
                #paddle.save(generator.get_state(), f"generator_state_{i}.pt")
            image = pipe(
                prompt=prompt,
                height=1024,
                width=1024,
                guidance_scale=3.5,
                max_sequence_length=512,
                num_inference_steps=args.inference_step,
                generator=generator,
            ).images[0]
            image.save(os.path.join(saved_path, f"{i}.png"))
    if args.sanaprint == True:
        # test sana sprint
        from diffusers import SanaSprintPipeline
        import torch
        pipeline = SanaSprintPipeline.from_pretrained(
            "Efficient-Large-Model/Sana_Sprint_1.6B_1024px_diffusers",
            torch_dtype=torch.bfloat16
        )
        pipeline.to("cuda")
        if args.dataset == "coco10k":
            saved_path = os.path.join(args.saved_path,"predicterror_taylorseer_base")
        elif args.dataset == "300Prompt":
            saved_path = os.path.join(args.saved_path,"sanasprint_300")
        else:
            saved_path = os.path.join(args.saved_path,"sanasprint_coco1k")
        os.makedirs(saved_path, exist_ok=True)
        generator =torch.Generator(device="cpu").manual_seed(0)
        #prompt = "a tiny astronaut hatching from an egg on the moon"
        for i, prompt in enumerate(tqdm(all_prompts)):
            image = pipeline(prompt=prompt, num_inference_steps=2,generator=generator).images[0]
            image.save(os.path.join(saved_path, f"{i}.png"))
    if args.hyper_flux == True:
        import torch
        from diffusers import FluxPipeline
        from huggingface_hub import hf_hub_download
        base_model_id = "black-forest-labs/FLUX.1-dev"
        repo_name = "ByteDance/Hyper-SD"
        # Take 8-steps lora as an example
        ckpt_name = "Hyper-FLUX.1-dev-8steps-lora.safetensors"
        # Load model, please fill in your access tokens since FLUX.1-dev repo is a gated model.
        pipe = FluxPipeline.from_pretrained(base_model_id, token="xxx")
        pipe.load_lora_weights(hf_hub_download(repo_name, ckpt_name))
        pipe.fuse_lora(lora_scale=0.125)
        pipe.to("cuda", dtype=torch.float16)
        if args.dataset == "coco10k":
            saved_path = os.path.join(args.saved_path,"predicterror_taylorseer_base")
        elif args.dataset == "300Prompt":
            saved_path = os.path.join(args.saved_path,"hyper_flux_300")
        else:
            saved_path = os.path.join(args.saved_path,"hyper_flux_coco1k")
        os.makedirs(saved_path, exist_ok=True)
        for i, prompt in enumerate(tqdm(all_prompts)):
            image=pipe(prompt=prompt, num_inference_steps=4, guidance_scale=3.5,generator =torch.Generator(device="cpu").manual_seed(0)).images[0]
            image.save(os.path.join(saved_path, f"{i}.png"))
    if args.sana == True:
        import torch
        from diffusers import SanaPipeline
        
        pipe = SanaPipeline.from_pretrained(
            "Efficient-Large-Model/SANA1.5_1.6B_1024px_diffusers",
            torch_dtype=torch.bfloat16,
        )
        pipe.to("cuda")

        pipe.vae.to(torch.bfloat16)
        pipe.text_encoder.to(torch.bfloat16)
        if args.dataset == "coco10k":
            saved_path = os.path.join(args.saved_path,"sana_coco10k")
        elif args.dataset == "300Prompt":
            args.saved_path = "/root/paddlejob/workspace/env_run/output/gxl/pcm_eval_results_flux_vis"
            
            saved_path = os.path.join(args.saved_path,"sana_300")
        else:
            args.saved_path = "/root/paddlejob/workspace/env_run/output/gxl/pcm_eval_results_flux_coco1k"
            saved_path = os.path.join(args.saved_path,"sana_coco1k")
        os.makedirs(saved_path, exist_ok=True)

        for i, prompt in enumerate(tqdm(all_prompts)):
            image = pipe(
                prompt=prompt,
                height=1024,
                width=1024,
                guidance_scale=4.5,
                num_inference_steps=20,
                generator =torch.Generator(device="cpu").manual_seed(0)
            )[0]
            image[0].save(os.path.join(saved_path, f"{i}.png"))
    if args.flux_dev == True:
        import torch
        from diffusers import FluxPipeline

        pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16)
        pipe = pipe.to("cuda")
        # pipe.enable_model_cpu_offload() #save some VRAM by offloading the model to CPU. Remove this if you have enough GPU power
        if args.dataset == "coco10k":
            saved_path = os.path.join(args.saved_path,"flux-dev_coco10k")
        elif args.dataset == "300Prompt":
            args.saved_path = "/root/paddlejob/workspace/env_run/output/gxl/pcm_eval_results_flux_vis"
            
            saved_path = os.path.join(args.saved_path,"flux-dev_300")
        else:
            args.saved_path = "/root/paddlejob/workspace/env_run/output/gxl/pcm_eval_results_flux_coco1k"
            saved_path = os.path.join(args.saved_path,"flux-dev_coco1k")
        os.makedirs(saved_path, exist_ok=True)
        for i, prompt in enumerate(tqdm(all_prompts)):
            image = pipe(
                prompt,
                height=1024,
                width=1024,
                guidance_scale=3.5,
                num_inference_steps=50,
                max_sequence_length=512,
                generator=torch.Generator("cpu").manual_seed(0)
            ).images[0]
            image.save(os.path.join(saved_path, f"{i}.png"))


    

    