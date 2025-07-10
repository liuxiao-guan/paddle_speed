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
                        Taylor_firstblock_predicterror_Forward)

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

num_inference_steps = 50
seed = 42

prompt =  "An image of a squirrel"

prompt = "An image of a squirrel in Picasso style"
pipe = DiffusionPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", paddle_dtype=paddle.float16)
#pipeline.enable_model_cpu_offload() #save some VRAM by offloading the model to CPU. Remove this if you have enough GPU power




pipe.transformer.__class__.forward = Taylor_firstblock_predicterror_Forward
pipe.transformer.enable_teacache = True
pipe.transformer.cnt = 0
pipe.transformer.num_steps = num_inference_steps

pipe.transformer.pre_firstblock_hidden_states = None
pipe.transformer.previous_residual = None
pipe.transformer.pre_compute_hidden =None
pipe.transformer.predict_loss  = None
pipe.transformer.predict_hidden_states= None
pipe.transformer.threshold= 0.13

parameter_peak_memory = paddle.device.cuda.max_memory_allocated()

paddle.device.cuda.max_memory_reserved()
#start_time = time.time()
start = paddle.device.cuda.Event(enable_timing=True)
end = paddle.device.cuda.Event(enable_timing=True)

for i in range(2):
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