import os
import argparse
import pickle
from typing_extensions import Self
from numpy import imag
import paddle
from ppdiffusers.models.transformer_flux import FluxTransformer2DModel
from tqdm import tqdm

from ppdiffusers import PyramidAttentionBroadcastConfig, apply_pyramid_attention_broadcast,DDIMScheduler, DiTPipeline
from ppdiffusers.utils import load_image, export_to_video
#from forwards import FirstBlock_taylor_predict_Forward,FirstBlock_taylor_block_predict_Forward,Taylor_predicterror_Forward,BlockDanceForward,Taylor_predicterror_base_Forward
from paddlenlp.trainer import set_seed
import time




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
        default='/root/paddlejob/workspace/env_run/gxl/output/PaddleMIX/inf_speed_dit',
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
        '--num_samples',
        type=int,
        default=50000,
        help='the number of samples',
    )
    
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    os.makedirs(args.saved_path, exist_ok=True)
    
    # Create generator if seed is provided
    set_seed(args.seed)
    generator = None
    if args.seed is not None:
        generator = paddle.Generator().manual_seed(args.seed)
    # 原始生成的
    if args.origin == True:
        dtype = paddle.float16
        pipe = DiTPipeline.from_pretrained("facebook/DiT-XL-2-256", paddle_dtype=dtype)
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        if args.dataset == "coco10k":
            saved_path = os.path.join(args.saved_path,"origin_50steps")
        elif args.dataset == "300Prompt":
            saved_path = os.path.join(args.saved_path,"origin_300")
        else:
            saved_path = os.path.join(args.saved_path,"origin_50steps_coco1k")
        os.makedirs(saved_path, exist_ok=True)
    total_time = 0
    for i in tqdm(range(0,args.num_samples)):
        # if i > 10:
        #     break
        class_ids = paddle.randint(low=0, high=1000, shape=[1])
        print(f"class_ids: {class_ids.item()}")
        start_time = time.time()
        image = pipe(class_labels=class_ids, num_inference_steps=50, generator=generator,guidance_scale=1.5).images[0]
        end = time.time()
        total_time += (end - start_time)
        image.save(os.path.join(saved_path, f"{i}.png"))
    # 平均时间计算
    avg_time = total_time / args.num_samples
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
    print(f"{file_name}: Total {total_time:.2f}s, Avg {avg_time:.2f}s/image")