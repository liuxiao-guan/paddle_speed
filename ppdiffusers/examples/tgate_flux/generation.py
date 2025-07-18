import os
import argparse
import pickle
from typing_extensions import Self
from numpy import imag
import paddle
from ppdiffusers.models.transformer_flux import FluxTransformer2DModel
from tqdm import tqdm
import time 
from tgate import TgateSDXLLoader, TgateSDLoader,TgateFLUXLoader,TgatePixArtAlphaLoader
from ppdiffusers import StableDiffusionXLPipeline, PixArtAlphaPipeline, StableVideoDiffusionPipeline
from ppdiffusers import UNet2DConditionModel, LCMScheduler,FluxPipeline
from ppdiffusers import DPMSolverMultistepScheduler
from ppdiffusers.utils import load_image, export_to_video
# from ppdiffusers.models import FluxTeaCacheTransformer2DModel
# from teacache_forward import TeaCacheForward

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
        default='/root/paddlejob/workspace/env_run/gxl/output/PaddleMIX/inf_speed_bf16',
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
        else:
            saved_path = os.path.join(args.saved_path,"origin_50steps_coco1k")
        os.makedirs(saved_path, exist_ok=True)
        
    #加入tgate 方法的
    if args.tgate == True :
        pipe = FluxPipeline.from_pretrained(
                "black-forest-labs/FLUX.1-dev", paddle_dtype=paddle.bfloat16)
        pipe = TgateFLUXLoader(pipe)
        #saved_path = os.path.join(args.saved_path,"tgate_50steps")
        if args.dataset == "coco10k":
            saved_path = os.path.join(args.saved_path,"tgate_50steps")
        else:
            saved_path = os.path.join(args.saved_path,f"tgate_50steps_gs{args.gate_step}_si{args.sp_interval}_fi{args.fi_interval}coco1k")
        os.makedirs(saved_path, exist_ok=True)
        total_time =0 
        for i, prompt in enumerate(tqdm(all_prompts)):
            # if i == 3:
            #     break
            start_time = time.time()
            image = pipe.tgate(
                prompt=prompt,
                height=1024,
                width=1024,
                guidance_scale=3.5,
                max_sequence_length=512,
                gate_step=args.gate_step,
                sp_interval=args.sp_interval ,
                fi_interval=args.fi_interval,
                warm_up=args.warm_up,
                num_inference_steps=args.inference_step,
                generator=generator,
            ).images[0]
            end = time.time()
            total_time += end - start_time
            image.save(os.path.join(saved_path, f"{i}.png"))
        
        avg_time = total_time / len(all_prompts)
        # 获取最后一段作为文件名
        file_name = os.path.basename(saved_path)  # -> 'firstblock_taylorseer0.07_300'
        # 构造要写入的字符串
        content = f"{file_name}: Total {total_time:.2f}s, Avg {avg_time:.2f}s/image"
        # 写入到 file_name.txt
        txt_path = f"./output/{file_name}.txt"
        with open(txt_path, "w") as f:
            f.write(content)

        # 也可以 print 看看
        print(f"已写入: {txt_path}")
        print(f"tgate: Total {total_time:.2f}s, Avg {avg_time:.2f}s/image")
    # 加入teacache 方法的
    if args.teacache == True :

        pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", paddle_dtype=paddle.float16)
        
        FluxTransformer2DModel.forward = TeaCacheForward
        pipe.transformer.enable_teacache = True
        pipe.transformer.cnt = 0
        pipe.transformer.num_steps = 28
        pipe.transformer.rel_l1_thresh = (
            0.25  # 0.25 for 1.5x speedup, 0.4 for 1.8x speedup, 0.6 for 2.0x speedup, 0.8 for 2.25x speedup
        )
        pipe.transformer.accumulated_rel_l1_distance = 0
        pipe.transformer.previous_modulated_input = None
        pipe.transformer.previous_residual = None
        if args.dataset == "coco10k":
            saved_path = os.path.join(args.saved_path,"teacache")
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

    
    

    