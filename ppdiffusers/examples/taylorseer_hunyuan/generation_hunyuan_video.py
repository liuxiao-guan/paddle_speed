import os
import argparse
import pickle
from typing_extensions import Self
from numpy import imag
import paddle
from ppdiffusers.models.transformer_flux import FluxTransformer2DModel
from tqdm import tqdm
import time

# from tgate import TgateSDXLLoader, TgateSDLoader,TgateFLUXLoader,TgatePixArtAlphaLoader
from ppdiffusers import StableDiffusionXLPipeline, PixArtAlphaPipeline, StableVideoDiffusionPipeline
from ppdiffusers import UNet2DConditionModel, LCMScheduler,FluxPipeline,PyramidAttentionBroadcastConfig, apply_pyramid_attention_broadcast
from paddlenlp.transformers import LlamaModel
from paddlenlp.transformers.llama.tokenizer_fast import LlamaTokenizerFast
from ppdiffusers import HunyuanVideoPipeline, HunyuanVideoTransformer3DModel
from ppdiffusers.utils import export_to_video,export_to_video_2
from ppdiffusers import DPMSolverMultistepScheduler
from ppdiffusers.utils import load_image, export_to_video
import json
from forwards import taylorhunyuanpipeline,taylorseer_hunyuan_forward, \
taylorseer_hunyuan_double_block_forward,taylorseer_hunyuan_single_block_forward,teacache_forward,\
taylorstepfirstpredicthunyuanpipeline,taylorseer_step_firstpredict_hunyuan_forward


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
        default='/root/paddlejob/workspace/env_run/gxl/output/PaddleMIX/inf_speed_hunyuan',
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
    # 获取prompt
    if args.dataset == "vbench":
        with open("/root/paddlejob/workspace/env_run/gxl/paddle_speed/ppdiffusers/examples/taylorseer_hunyuan/vbench/VBench_full_info.json", 'r') as f:
            prompts_data = json.load(f)
        with open("/root/paddlejob/workspace/env_run/gxl/paddle_speed/ppdiffusers/examples/taylorseer_hunyuan/vbench/hunyuan_all_dimension.txt","r") as f:
            all_aug_prompts=[line.strip() for line in f if line.strip()]
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

    os.environ["SKIP_PARENT_CLASS_CHECK"] = "True"
    model_id = "hunyuanvideo-community/HunyuanVideo"
    transformer = HunyuanVideoTransformer3DModel.from_pretrained(model_id, subfolder="transformer", paddle_dtype=paddle.bfloat16)
    tokenizer = LlamaTokenizerFast.from_pretrained(model_id, subfolder="tokenizer")
    text_encoder = LlamaModel.from_pretrained(model_id, subfolder="text_encoder", dtype="float16")
    # 原始生成的
    if args.origin == True:
        pipe = HunyuanVideoPipeline.from_pretrained(
            model_id,
            transformer = transformer,
            text_encoder = text_encoder,
            tokenizer = tokenizer,
            paddle_dtype=paddle.float16,
            map_location="cpu")
        pipe.vae.enable_tiling()
        pipe.vae.enable_slicing()
        if args.dataset == "vbench":
            saved_path = os.path.join(args.saved_path,"origin_50steps")
        elif args.dataset == "300Prompt":
            saved_path = os.path.join(args.saved_path,"origin_300")
        else:
            saved_path = os.path.join(args.saved_path,"origin_50steps_coco1k")
        
    #加入tgate 方法的
    if args.tgate == True :
        pass
    if args.pab == True:
        pipe = HunyuanVideoPipeline.from_pretrained(
            model_id,
            transformer=transformer,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            paddle_dtype=paddle.float16,
            map_location="cpu",
        )
        config = PyramidAttentionBroadcastConfig(
            spatial_attention_block_skip_range=4,
            temporal_attention_block_skip_range=5,
            cross_attention_block_skip_range=6,
            spatial_attention_timestep_skip_range=(100, 1000),
            current_timestep_callback=lambda: pipe._current_timestep,
        )
        apply_pyramid_attention_broadcast(pipe.transformer, config)
        pipe.vae.enable_tiling()
        pipe.vae.enable_slicing()
        if args.dataset == "coco10k":
            saved_path = os.path.join(args.saved_path,"pab")
        elif args.dataset == "300Prompt":
            saved_path = os.path.join(args.saved_path,"pab_300")
        else:
            saved_path = os.path.join(args.saved_path,"pab_456")
    if args.blockdance == True:
        if args.dataset == "coco10k":
            saved_path = os.path.join(args.saved_path,"blockdance")
        elif args.dataset == "300Prompt":
            saved_path = os.path.join(args.saved_path,"blockdance_300")
        else:
            saved_path = os.path.join(args.saved_path,"blockdance_R950_B30-15_N8_coco1k")
    if args.taylorseer == True:
        if args.dataset == "coco10k":
            saved_path = os.path.join(args.saved_path,"taylorseer")
        elif args.dataset == "300Prompt":
            saved_path = os.path.join(args.saved_path,"taylorseer_300")
        else:
            saved_path = os.path.join(args.saved_path,"taylorseer_N5O1")
        # os.environ["SKIP_PARENT_CLASS_CHECK"] = "True"
        # model_id = "hunyuanvideo-community/HunyuanVideo"
        # transformer = HunyuanVideoTransformer3DModel.from_pretrained(model_id, subfolder="transformer", paddle_dtype=paddle.bfloat16)
        # tokenizer = LlamaTokenizerFast.from_pretrained(model_id, subfolder="tokenizer")
        # text_encoder = LlamaModel.from_pretrained(model_id, subfolder="text_encoder", dtype="float16")
        pipe = HunyuanVideoPipeline.from_pretrained(
            model_id,
            transformer = transformer,
            text_encoder = text_encoder,
            tokenizer = tokenizer,
            paddle_dtype=paddle.float16,
            map_location="cpu")
        pipe.vae.enable_tiling()
        pipe.vae.enable_slicing()
        pipe.__class__.__call__ = taylorhunyuanpipeline
        pipe.transformer.__class__.forward = taylorseer_hunyuan_forward
        # pipe.enable_model_cpu_offload()
        for double_transformer_block in pipe.transformer.transformer_blocks:
            double_transformer_block.__class__.forward = taylorseer_hunyuan_double_block_forward
            
        for single_transformer_block in pipe.transformer.single_transformer_blocks:
            single_transformer_block.__class__.forward = taylorseer_hunyuan_single_block_forward
    # 加入teacache 方法的
    if args.teacache == True :
        pipe = HunyuanVideoPipeline.from_pretrained(
            model_id,
            transformer=transformer,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            paddle_dtype=paddle.float16,
            map_location="cpu",
        )

        pipe.transformer.enable_teacache = True
        pipe.transformer.cnt = 0
        pipe.transformer.num_steps = 50
        pipe.transformer.rel_l1_thresh = 0.15  # 0.1 for 1.6x speedup, 0.15 for 2.1x speedup
        pipe.transformer.accumulated_rel_l1_distance = 0
        pipe.transformer.previous_modulated_input = None
        pipe.transformer.previous_residual = None
        pipe.transformer.__class__.forward = teacache_forward

        pipe.vae.enable_tiling()
        pipe.vae.enable_slicing()

        if args.dataset == "coco10k":
            saved_path = os.path.join(args.saved_path,"teacache")
        elif args.dataset == "300Prompt":
            saved_path = os.path.join(args.saved_path,"teacache_300")
        else:
            saved_path = os.path.join(args.saved_path,"teacache0.15")
   
    if args.firstblock_predicterror_taylor == True:
        pipe = HunyuanVideoPipeline.from_pretrained(
            model_id,
            transformer = transformer,
            text_encoder = text_encoder,
            tokenizer = tokenizer,
            paddle_dtype=paddle.float16,
            map_location="cpu")
        pipe.vae.enable_tiling()
        pipe.vae.enable_slicing()
        pipe.__class__.__call__ = taylorstepfirstpredicthunyuanpipeline
        pipe.transformer.__class__.forward = taylorseer_step_firstpredict_hunyuan_forward
        pipe.transformer.cnt = 0
        pipe.transformer.num_steps = 50
        pipe.transformer.predict_loss  = None
        pipe.transformer.threshold= 0.12
        if args.dataset == "coco10k":
            saved_path = os.path.join(args.saved_path,"firstblock_predicterror_taylor")
        elif args.dataset == "300Prompt":
            saved_path = os.path.join(args.saved_path,"firstblock_predicterror_taylor_300")
        else:
            saved_path = os.path.join(args.saved_path,"firstblock_predicterror_taylor0.12BO2")
    if args.taylorseer_step == True:
        pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", paddle_dtype=paddle.bfloat16)
        #pipeline.enable_model_cpu_offload() #save some VRAM by offloading the model to CPU. Remove this if you have enough GPU power

        # TaylorSeer settings
        pipe.transformer.__class__.num_steps = args.inference_step

        pipe.transformer.__class__.forward = taylorseer_step_flux_forward
        if args.dataset == "coco10k":
            saved_path = os.path.join(args.saved_path,"taylorseer_step")
        elif args.dataset == "300Prompt":
            saved_path = os.path.join(args.saved_path,"taylorseer_step_300")
        else:
            saved_path = os.path.join(args.saved_path,"taylorseer_step_coco1k")
    os.makedirs(saved_path, exist_ok=True)
    total_time = 0
    for i, item in enumerate(tqdm(all_prompts)):
        # if i==2:
        #     break
        prompt = item.get("prompt_en", "")
        print(prompt)
        start_time = time.time()
        output = pipe(
            prompt=all_aug_prompts[i],
            height=320,
            width=512,
            num_frames=61,
            num_inference_steps=50,
            generator=paddle.Generator().manual_seed(42+i),
        ).frames[0]
        end = time.time()
        # print("Time used:" , end - start_time)
        total_time += (end - start_time)
        export_to_video_2(output, os.path.join(saved_path, f"{prompt}-{args.repeat}.mp4"), fps=24)
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
        