from typing import Any, Dict, Optional, Tuple, Union
import time
from ppdiffusers import DiffusionPipeline
from ppdiffusers.pipelines.flux import FluxPipeline
from ppdiffusers.models import FluxTransformer2DModel
from ppdiffusers.models.modeling_outputs import Transformer2DModelOutput
from ppdiffusers.utils import USE_PEFT_BACKEND, is_paddle_version, logging, scale_lora_layers, unscale_lora_layers
import paddle
import numpy as np
from scipy.stats import linregress
from forwards import (taylorseer_flux_single_block_forward, 
                        taylorseer_flux_double_block_forward, 
                        taylorseer_flux_forward,
                        TeaCache_taylor_predict_Forward,
                        Taylor_firstblock_predicterror_Forward_eva)

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

num_inference_steps = 50
seed = 42

prompt =  "An image of a squirrel"

prompt = "An image of a squirrel in Picasso style"
prompt_list = [
    "A black colored banana.",
    "A white colored sandwich.",
    "A black colored sandwich.", 
    "An orange colored sandwich.",
    "A pink colored giraffe.",
    "A yellow colored giraffe.",
    "A brown colored giraffe.",
    "A red car and a white sheep."
]
pipe = DiffusionPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", paddle_dtype=paddle.float16)
#pipeline.enable_model_cpu_offload() #save some VRAM by offloading the model to CPU. Remove this if you have enough GPU power

with open('/root/paddlejob/workspace/env_run/gxl/paddle_speed/ppdiffusers/examples/taylorseer_flux/prompts/DrawBench.txt', 'r', encoding='utf-8') as f:
    all_prompts = [line.strip() for line in f if line.strip()]


pipe.transformer.__class__.forward = Taylor_firstblock_predicterror_Forward_eva
pipe.transformer.enable_teacache = True
pipe.transformer.cnt = 0
pipe.transformer.num_steps = num_inference_steps

pipe.transformer.pre_firstblock_hidden_states = None
pipe.transformer.previous_residual = None
pipe.transformer.pre_compute_hidden =None
pipe.transformer.predict_loss  = None
pipe.transformer.predict_hidden_states= None
pipe.transformer.threshold= 0.03
pipe.transformer.error_firstblock = []
pipe.transformer.error_output = []
# 初始化
error_firstblock_list = []
error_output_list = []
parameter_peak_memory = paddle.device.cuda.max_memory_allocated()

paddle.device.cuda.max_memory_reserved()
#start_time = time.time()
start = paddle.device.cuda.Event(enable_timing=True)
end = paddle.device.cuda.Event(enable_timing=True)

# for idx,prompt in enumerate(prompt_list):

for i in range(1):
    start_time =time.time()
    img = pipe(
        prompt, 
        num_inference_steps=num_inference_steps,
        generator=paddle.Generator().manual_seed(seed)
        ).images[0]

    elapsed1 = time.time() - start_time
    print(f"第一次运行时间: {elapsed1:.2f}s")
    peak_memory = paddle.device.cuda.max_memory_allocated()

    img.save("firstblockpredict.png")
    #img.save(f"{pkl_list[i]}.png")

    print(
        f"epoch time: {elapsed1:.2f} sec, parameter memory: {parameter_peak_memory/1e9:.2f} GB, memory: {peak_memory/1e9:.2f} GB"
    )
    error_firstblock_list.append(pipe.transformer.error_firstblock[7:])
    error_output_list.append(pipe.transformer.error_output[7:])
    pipe.transformer.error_firstblock = []
    pipe.transformer.error_output = []
    
import matplotlib.pyplot as plt
# 转 numpy 矩阵
error_firstblock_array = np.array(error_firstblock_list)  # shape: (num_prompts, N)
error_output_array = np.array(error_output_list)

# 按列平均
x_avg = np.mean(error_firstblock_array, axis=0)  # shape: (N,)
y_avg = np.mean(error_output_array, axis=0)

# 拟合
slope, intercept, r_value, p_value, std_err = linregress(x_avg, y_avg)
x_fit = np.linspace(min(x_avg), max(x_avg), 100)
y_fit = slope * x_fit + intercept
import matplotlib.font_manager as font_manager
bold_font_path = "/usr/share/fonts/truetype/msttcorefonts/timesbd.ttf"
bold_font = font_manager.FontProperties(fname=bold_font_path,size=20)
plt.rcParams["font.family"] = "Times New Roman"
    # 设置字体（可选）
# plt.rcParams["font.family"] = bold_font.get_name()  # 例如 'Times New Roman'
# plt.rcParams["font.weight"] = "bold"   
plt.rcParams["font.size"] = 20
plt.rcParams["axes.titlesize"] =18
plt.rcParams["axes.labelsize"] = 18
plt.rcParams["xtick.labelsize"] = 16
plt.rcParams["ytick.labelsize"] = 16
plt.rcParams["legend.fontsize"] = 14
plt.figure(figsize=(7.5, 4))
plt.scatter(x_avg, y_avg, s=30, alpha=0.8,c='darkorange')
    # 拟合一条直线
# x = pipe.transformer.error_firstblock[7:]
# y = pipe.transformer.error_output[7:]
# slope, intercept, r_value, p_value, std_err = linregress(x, y)
# x_fit = np.linspace(min(x), max(x), 100)
# y_fit = slope * x_fit + intercept

# # 画拟合线
plt.plot(x_fit, y_fit, color='steelblue', linestyle='-', label=f'y = {slope:.3f}x + {intercept:.3f}',linewidth=2.2)
plt.xlabel("First Block Taylor error")
plt.ylabel("Last Block Taylor error")
# plt.title("Correlation between first-block and model-output Taylor errors")
plt.grid(True,linestyle="--",linewidth=0.5, alpha=0.7)
plt.tight_layout()
plt.savefig(f"firstblock_output_correlation_avg.png",dpi=600)
plt.show()
