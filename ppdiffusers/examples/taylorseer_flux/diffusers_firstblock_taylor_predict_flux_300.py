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

num_inference_steps = 28
generator=paddle.Generator().manual_seed(124)
seed = 124
prompts = ["The scene shows seven red begonia flowers arranged elegantly on a table, with vibrant petals that possess a layered texture. The table’s wooden texture adds a touch of natural and classical charm to the composition. The background is a simple indoor setting, emphasizing the beauty and grace of the begonia flowers.",
"A photograph captures a heartwarming family portrait, with the parents seated on a sofa, and their four children standing behind them, all smiling. The cushions and pillows on the sofa appear soft and comfortable. The background presents a cozy home environment, with sunlight shining through the window and clear skies outside.",
"The scene shows a peculiar figure walking on his head, with his legs flailing in the air. This unique character catches the eye with its unusual form. The background is a simple space, highlighting the character’s extraordinary shape. The overall style is realistic, leaving a strong impression.",
"The scene presents a slender-legged beauty, elegantly posed, with her long, delicate legs highlighted in the frame. The background is a modern, minimalist space, enhancing the visual impact of her figure. The lighting emphasizes the dreamlike atmosphere, elevating the beauty of her long legs.",
"The scene features a sports event poster. The central focus is a runner in the middle of a sprint, wearing athletic gear and running shoes, showcasing a perfect combination of speed and power. The background shows a cheering audience in the stadium, creating an energetic and passionate atmosphere, evoking the smell of the competitive field.",
]
#prompt = "An image of a squirrel in Picasso style"
pipe = DiffusionPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", paddle_dtype=paddle.float16)
#pipeline.enable_model_cpu_offload() #save some VRAM by offloading the model to CPU. Remove this if you have enough GPU power

# TaylorSeer settings
pipe.transformer.__class__.num_steps = num_inference_steps

pipe.transformer.__class__.forward = FirstBlock_taylor_predict_Forward

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
# pipeline.to("cuda")

parameter_peak_memory = paddle.device.cuda.max_memory_allocated()

paddle.device.cuda.max_memory_reserved()
#start_time = time.time()
start = paddle.device.cuda.Event(enable_timing=True)
end = paddle.device.cuda.Event(enable_timing=True)
pkl_list = [1,3,47.251,292]
for i in range(1):
    start.record()
    b=paddle.load(f'./seedpkl/{pkl_list[i]}_state.pkl')
    paddle.set_cuda_rng_state(b)
    img = pipe(
        prompts[2], 
        num_inference_steps=num_inference_steps,
        generator=paddle.Generator().manual_seed(seed)
        ).images[0]

    end.record()
    paddle.device.synchronize()
    elapsed_time = start.elapsed_time(end) * 1e-3
    peak_memory = paddle.device.cuda.max_memory_allocated()

    #img.save("{}.png".format('1_' + "flowers"))
    img.save(f"{pkl_list[i]}.png")

    print(
        f"epoch time: {elapsed_time:.2f} sec, parameter memory: {parameter_peak_memory/1e9:.2f} GB, memory: {peak_memory/1e9:.2f} GB"
    )