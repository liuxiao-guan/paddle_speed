import os
import argparse
from numpy import imag
import paddle
from tqdm import tqdm
import time
from paddlenlp.transformers import LlamaModel
from paddlenlp.transformers.llama.tokenizer_fast import LlamaTokenizerFast
from ppdiffusers import HunyuanVideoPipeline, HunyuanVideoTransformer3DModel
from ppdiffusers.utils import export_to_video_2
import json
from ppdiffusers import AutoencoderKLWan, WanPipeline,PyramidAttentionBroadcastConfig, apply_pyramid_attention_broadcast
from ppdiffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
from ppdiffusers.utils import export_to_video_2
from forwards import wan_forward,wan_block_forward,wan_pipeline, wan_firstpredict_step_forward,wan_step_pipeline,wan_teacache_forward,wan_step_forward

import time

# from teacache_forward import TeaCacheForward


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
        default='/root/paddlejob/workspace/env_run/gxl/output/PaddleMIX/inf_speed_wan',
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
        '--taylorseer', 
        action='store_true', 
        default=False, 
        help='do add taylorsteer',
    )
    
    
    parser.add_argument(
        '--firstblock_predicterror_taylor', 
        action='store_true', 
        default=False, 
        help='do add predicterror taylorseer block base',
    )
    parser.add_argument(
        '--taylorseer_step', 
        action='store_true', 
        default=False, 
        help='do add taylorseer step ',
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
    parser.add_argument(
        '--repeat',
        type=int,
        default=0,
        help='the count of repeat',
    )
    
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    os.makedirs(args.saved_path, exist_ok=True)
    negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"
    # 获取prompt
    if args.dataset == "vbench":
        with open("/root/paddlejob/workspace/env_run/gxl/paddle_speed/ppdiffusers/examples/taylorseer_hunyuan/vbench/VBench_full_info.json", 'r') as f:
            prompts_data = json.load(f)
        all_prompts = prompts_data[:]
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
        
        # Available models: Wan-AI/Wan2.1-T2V-14B-Diffusers, Wan-AI/Wan2.1-T2V-1.3B-Diffusers
        model_id = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
        vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", paddle_dtype=paddle.float32)
        pipe = WanPipeline.from_pretrained(model_id, vae=vae, paddle_dtype=paddle.bfloat16)

        flow_shift = 5.0  # 5.0 for 720P, 3.0 for 480P
        scheduler = UniPCMultistepScheduler(
            prediction_type="flow_prediction", use_flow_sigmas=True, num_train_timesteps=1000, flow_shift=flow_shift
        )
        pipe.scheduler = scheduler
        if args.dataset == "vbench":
            saved_path = os.path.join(args.saved_path,"origin_fs5_50steps")
        elif args.dataset == "300Prompt":
            saved_path = os.path.join(args.saved_path,"origin_300")
        else:
            saved_path = os.path.join(args.saved_path,"origin_50steps_coco1k")
        
    #加入tgate 方法的
    if args.tgate == True :
        pass
    if args.pab == True:
        model_id = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
        vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", paddle_dtype=paddle.float32)
        pipe = WanPipeline.from_pretrained(model_id, vae=vae, paddle_dtype=paddle.bfloat16)
        config = PyramidAttentionBroadcastConfig(
            spatial_attention_block_skip_range=20,
            spatial_attention_timestep_skip_range=(100, 950),
            current_timestep_callback=lambda: pipe._current_timestep,
        )
        apply_pyramid_attention_broadcast(pipe.transformer, config)
        flow_shift = 5.0  # 5.0 for 720P, 3.0 for 480P
        scheduler = UniPCMultistepScheduler(
            prediction_type="flow_prediction", use_flow_sigmas=True, num_train_timesteps=1000, flow_shift=flow_shift
        )
        pipe.scheduler = scheduler
        if args.dataset == "vbench":
            saved_path = os.path.join(args.saved_path,"pab_N20_B100-950")
        elif args.dataset == "300Prompt":
            saved_path = os.path.join(args.saved_path,"pab_300")
        else:
            saved_path = os.path.join(args.saved_path,"pab_coco1k")
    if args.blockdance == True:
        if args.dataset == "coco10k":
            saved_path = os.path.join(args.saved_path,"blockdance")
        elif args.dataset == "300Prompt":
            saved_path = os.path.join(args.saved_path,"blockdance_300")
        else:
            saved_path = os.path.join(args.saved_path,"blockdance_R950_B30-15_N8_coco1k")
    if args.taylorseer == True:
        # Available models: Wan-AI/Wan2.1-T2V-14B-Diffusers, Wan-AI/Wan2.1-T2V-1.3B-Diffusers
        model_id = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
        vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", paddle_dtype=paddle.float32)
        pipe = WanPipeline.from_pretrained(model_id, vae=vae, paddle_dtype=paddle.bfloat16)

        flow_shift = 5.0  # 5.0 for 720P, 3.0 for 480P
        scheduler = UniPCMultistepScheduler(
            prediction_type="flow_prediction", use_flow_sigmas=True, num_train_timesteps=1000, flow_shift=flow_shift
        )
        pipe.scheduler = scheduler
        pipe.__class__.__call__ = wan_pipeline
        pipe.transformer.__class__.forward = wan_forward

        for double_transformer_block in pipe.transformer.blocks:
            double_transformer_block.__class__.forward = wan_block_forward
        if args.dataset == "coco10k":
            saved_path = os.path.join(args.saved_path,"taylorseer")
        elif args.dataset == "300Prompt":
            saved_path = os.path.join(args.saved_path,"taylorseer_300")
        else:
            saved_path = os.path.join(args.saved_path,"taylorseer_fs5_N5")
    # 加入teacache 方法的
    if args.teacache == True :
        # Available models: Wan-AI/Wan2.1-T2V-14B-Diffusers, Wan-AI/Wan2.1-T2V-1.3B-Diffusers
        model_id = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
        vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", paddle_dtype=paddle.float32)
        pipe = WanPipeline.from_pretrained(model_id, vae=vae, paddle_dtype=paddle.bfloat16)

        flow_shift = 5.0  # 5.0 for 720P, 3.0 for 480P
        scheduler = UniPCMultistepScheduler(
            prediction_type="flow_prediction", use_flow_sigmas=True, num_train_timesteps=1000, flow_shift=flow_shift
        )
        pipe.transformer.__class__.forward = wan_teacache_forward
        pipe.scheduler = scheduler
        # pipe.__class__.generate = t2v_generate
        pipe.transformer.enable_teacache = True

        pipe.transformer.cnt = 0
        pipe.transformer.num_steps = 50*2 # 因为有cfg
        pipe.transformer.teacache_thresh = 0.26 #0.08 0.2 
        pipe.transformer.accumulated_rel_l1_distance_even = 0
        pipe.transformer.accumulated_rel_l1_distance_odd = 0
        pipe.transformer.previous_e0_even = None
        pipe.transformer.previous_e0_odd = None
        pipe.transformer.previous_residual_even = None
        pipe.transformer.previous_residual_odd = None
        pipe.transformer.coefficients= [2.39676752e+03, -1.31110545e+03,  2.01331979e+02, -8.29855975e+00, 1.37887774e-01]
        pipe.transformer.ret_steps=1*2
        pipe.transformer.cutoff_steps=50*2 - 2

        if args.dataset == "vbench":
            saved_path = os.path.join(args.saved_path,"teacache0.26_fs5")
        elif args.dataset == "300Prompt":
            saved_path = os.path.join(args.saved_path,"teacache_300")
        else:
            saved_path = os.path.join(args.saved_path,"teacache_coco1k")
   
    if args.firstblock_predicterror_taylor == True:
        # Available models: Wan-AI/Wan2.1-T2V-14B-Diffusers, Wan-AI/Wan2.1-T2V-1.3B-Diffusers
        model_id = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
        vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", paddle_dtype=paddle.float32)
        pipe = WanPipeline.from_pretrained(model_id, vae=vae, paddle_dtype=paddle.bfloat16)

        flow_shift = 5.0  # 5.0 for 720P, 3.0 for 480P
        scheduler = UniPCMultistepScheduler(
            prediction_type="flow_prediction", use_flow_sigmas=True, num_train_timesteps=1000, flow_shift=flow_shift
        )
        pipe.scheduler = scheduler
        pipe.__class__.__call__ = wan_step_pipeline
        pipe.transformer.__class__.forward = wan_firstpredict_step_forward
        pipe.transformer.cnt = 0
        pipe.transformer.num_steps = 50
        pipe.transformer.predict_loss  = None
        pipe.transformer.threshold= 0.36
        pipe.transformer.should_calc = False

        if args.dataset == "coco10k":
            saved_path = os.path.join(args.saved_path,"firstblock_predicterror_taylor")
        elif args.dataset == "300Prompt":
            saved_path = os.path.join(args.saved_path,"firstblock_predicterror_taylor_300")
        else:
            saved_path = os.path.join(args.saved_path,"firstpredict_fs5_cnt5_rel0.36_bO3")
    if args.taylorseer_step == True:
       # Available models: Wan-AI/Wan2.1-T2V-14B-Diffusers, Wan-AI/Wan2.1-T2V-1.3B-Diffusers
        model_id = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
        vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", paddle_dtype=paddle.float32)
        pipe = WanPipeline.from_pretrained(model_id, vae=vae, paddle_dtype=paddle.bfloat16)

        flow_shift = 5.0  # 5.0 for 720P, 3.0 for 480P
        scheduler = UniPCMultistepScheduler(
            prediction_type="flow_prediction", use_flow_sigmas=True, num_train_timesteps=1000, flow_shift=flow_shift
        )
        pipe.scheduler = scheduler
        pipe.__class__.__call__ = wan_step_pipeline
        pipe.transformer.__class__.forward = wan_step_forward
        if args.dataset == "coco10k":
            saved_path = os.path.join(args.saved_path,"taylorseer_step")
        elif args.dataset == "300Prompt":
            saved_path = os.path.join(args.saved_path,"taylorseer_step_300")
        else:
            saved_path = os.path.join(args.saved_path,"taylorseer_stepN5O1")
    os.makedirs(saved_path, exist_ok=True)
    total_time = 0
    for i, item in enumerate(tqdm(all_prompts)):
        # if i==1:
        #     break
        prompt = item.get("prompt_en", "")
        print(prompt)
        start_time = time.time()
        output = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=480,
            width=832,
            num_frames=81,
            guidance_scale=5.0,
            generator=paddle.Generator().manual_seed(args.seed),
        ).frames[0]
        end = time.time()
        print("Time used:" , end - start_time)
        total_time += (end - start_time)
        export_to_video_2(output, os.path.join(saved_path, f"{prompt}-{args.repeat}.mp4"), fps=16)
    # 平均时间计算
    avg_time = total_time / len(all_prompts)
    file_name = os.path.basename(saved_path)  # -> 'firstblock_taylorseer0.07_300'
    # 构造要写入的字符串
    content = f"{file_name}: Total {total_time:.2f}s, Avg {avg_time:.2f}s/image"
    # 写入到 file_name.txt
    txt_path = f"./output/{file_name}.txt"
    with open(txt_path, "w") as f:
        f.write(content)

    # 也可以 print 看看
    print(f"已写入: {txt_path}")
    print(f"{file_name}: Total {total_time:.2f}s, Avg {avg_time:.2f}s/image")





    # total_time = 0
    # for i, prompt in enumerate(tqdm(all_prompts)):
    #     start_time = time.time()
    #     image = pipe(
    #         prompt=prompt,
    #         height=1024,
    #         width=1024,
    #         guidance_scale=3.5,
    #         max_sequence_length=512,
    #         num_inference_steps=args.inference_step,
    #         generator=generator,
    #     ).images[0]
    #     end = time.time()
    #     total_time += (end - start_time)
    #     image.save(os.path.join(saved_path, f"{i}.png"))
    # # 平均时间计算
    # avg_time = total_time / len(all_prompts)
    # print(f"Pure Generation Time: Total {total_time:.2f}s, Avg {avg_time:.2f}s/image")
        