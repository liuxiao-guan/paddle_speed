from typing import Any, Dict, Optional, Tuple, Union
import time
from ppdiffusers import DiffusionPipeline
from ppdiffusers.pipelines.flux import FluxPipeline
from ppdiffusers.models import FluxTransformer2DModel
from ppdiffusers.models.modeling_outputs import Transformer2DModelOutput
from ppdiffusers.utils import USE_PEFT_BACKEND, is_paddle_version, logging, scale_lora_layers, unscale_lora_layers
import paddle
import numpy as np
from forwards import (taylorseer_flux_single_block_forward, 
                        taylorseer_flux_double_block_forward, 
                        taylorseer_flux_forward,
                        TeaCache_taylor_predict_Forward,
                        FirstBlock_taylor_predict_Forward)

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

num_inference_steps = 50
seed = 42

prompt = "An image of a squirrel in Picasso style"
pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", paddle_dtype=paddle.float16)
#pipeline.enable_model_cpu_offload() #save some VRAM by offloading the model to CPU. Remove this if you have enough GPU power

# TaylorSeer settings
# pipe.transformer.__class__.num_steps = num_inference_steps

pipe.transformer.__class__.forward = FirstBlock_taylor_predict_Forward

# for double_transformer_block in pipe.transformer.transformer_blocks:
#     double_transformer_block.__class__.forward = taylorseer_flux_double_block_forward
    
# for single_transformer_block in pipe.transformer.single_transformer_blocks:
#     single_transformer_block.__class__.forward = taylorseer_flux_single_block_forward

pipe.transformer.enable_teacache = True
pipe.transformer.cnt = 0
pipe.transformer.num_steps = 50

    

pipe.transformer.residual_diff_threshold = (
    0.07 #0.05  7.6s 
)
pipe.transformer.downsample_factor=(1)
pipe.transformer.accumulated_rel_l1_distance = 0
pipe.transformer.prev_first_hidden_states_residual = None
pipe.transformer.previous_residual = None


# pipe.to("cuda")

parameter_peak_memory = paddle.device.cuda.max_memory_allocated()

paddle.device.cuda.max_memory_reserved()
#start_time = time.time()
start = paddle.device.cuda.Event(enable_timing=True)
end = paddle.device.cuda.Event(enable_timing=True)

for i in range(1):
    start_time = time.time()
    img = pipe(
        prompt, 
        num_inference_steps=num_inference_steps,
        generator=paddle.Generator().manual_seed(seed)
        ).images[0]

    elapsed1 = time.time() - start_time
    print(f"第一次运行时间: {elapsed1:.2f}s")
    peak_memory = paddle.device.cuda.max_memory_allocated()

    img.save("{}.png".format('1_' + "An image of a squirrel in Picasso style"))
    start_time = time.time()
    img = pipe(
        prompt, 
        num_inference_steps=num_inference_steps,
        generator=paddle.Generator().manual_seed(seed)
        ).images[0]

    elapsed1 = time.time() - start_time
    print(f"第二次运行时间: {elapsed1:.2f}s")
    #img.save(f"{pkl_list[i]}.png")

    