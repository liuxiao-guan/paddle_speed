# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle

from ppdiffusers import (
    FluxPipeline,
    PyramidAttentionBroadcastConfig,
    apply_pyramid_attention_broadcast,
)

pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", paddle_dtype=paddle.bfloat16)

config = PyramidAttentionBroadcastConfig(
    spatial_attention_block_skip_range=4,
    # temporal_attention_block_skip_range=2,
    # cross_attention_block_skip_range=4,
    spatial_attention_timestep_skip_range=(100, 800),
    current_timestep_callback=lambda: pipe._current_timestep,
)
apply_pyramid_attention_broadcast(pipe.transformer, config)

prompt = "A cat holding a sign that says hello world"
import time 
for  i in range(2):
    start =time.time()
    image = pipe(
        prompt,
        height=1024,
        width=1024,
        guidance_scale=3.5,
        num_inference_steps=50,
        max_sequence_length=512,
        generator=paddle.Generator().manual_seed(42),
    ).images[0]
    end=time.time()
    print(end-start)
    image.save("text_to_image_generation-flux-dev-result.png")