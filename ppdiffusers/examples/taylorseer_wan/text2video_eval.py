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
import inspect
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import paddle
from ppdiffusers.callbacks import MultiPipelineCallbacks,PipelineCallback
from ppdiffusers.utils import logger,USE_PEFT_BACKEND
from ppdiffusers.utils import scale_lora_layers,unscale_lora_layers
from ppdiffusers.models.modeling_outputs import Transformer2DModelOutput
from cache_functions import cal_type
from ppdiffusers import WanTransformer3DModel
import paddle

from ppdiffusers import AutoencoderKLWan, WanPipeline
from ppdiffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
from ppdiffusers.utils import export_to_video_2
import time
from forwards import wan_teacache_forward
import matplotlib.pyplot as plt 

def wan_eval_forward(
        self:WanTransformer3DModel,
        hidden_states: paddle.Tensor,
        timestep: paddle.Tensor,
        encoder_hidden_states: paddle.Tensor,
        encoder_hidden_states_image: Optional[paddle.Tensor] = None,
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Union[paddle.Tensor, Dict[str, paddle.Tensor]]:
        # if current['stream'] == 'cond_stream':
        #     cal_type(cache_dic, current)
        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)
        else:
            if attention_kwargs is not None and attention_kwargs.get("scale", None) is not None:
                logger.warning(
                    "Passing `scale` via `attention_kwargs` when not using the PEFT backend is ineffective."
                )

        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.config.patch_size
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p_h
        post_patch_width = width // p_w

        rotary_emb = self.rope(hidden_states)

        hidden_states = self.patch_embedding(hidden_states)
        hidden_states = hidden_states.flatten(2).transpose([0, 2, 1])

        temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image = self.condition_embedder(
            timestep, encoder_hidden_states, encoder_hidden_states_image
        )
        timestep_proj = timestep_proj.unflatten(1, (6, -1))

        if encoder_hidden_states_image is not None:
            encoder_hidden_states = paddle.concat([encoder_hidden_states_image, encoder_hidden_states], axis=1)

        # 4. Transformer blocks
        if paddle.is_grad_enabled() and self.gradient_checkpointing:
            for block in self.blocks:
                hidden_states = self._gradient_checkpointing_func(
                    block, hidden_states, encoder_hidden_states, timestep_proj, rotary_emb
                )
        else:
            for i ,block in enumerate(self.blocks):
                hidden_states = block(hidden_states, encoder_hidden_states, timestep_proj, rotary_emb)
                if i ==0:
                    self.hidden_states_first_block.append(hidden_states)
                #hidden_states = block(hidden_states, encoder_hidden_states, timestep_proj, rotary_emb)
        self.hidden_states_list.append(hidden_states)
        # 5. Output norm, projection & unpatchify
        shift, scale = (self.scale_shift_table + temb.unsqueeze(1)).chunk(2, axis=1)
        hidden_states = (self.norm_out(hidden_states.cast(paddle.float32)) * (1 + scale) + shift).cast(
            hidden_states.dtype
        )
        hidden_states = self.proj_out(hidden_states)

        hidden_states = hidden_states.reshape(
            [batch_size, post_patch_num_frames, post_patch_height, post_patch_width, p_t, p_h, p_w, -1]
        )
        hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
        output = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)

# Available models: Wan-AI/Wan2.1-T2V-14B-Diffusers, Wan-AI/Wan2.1-T2V-1.3B-Diffusers
model_id = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", paddle_dtype=paddle.float32)
pipe = WanPipeline.from_pretrained(model_id, vae=vae, paddle_dtype=paddle.bfloat16)

flow_shift = 5.0  # 5.0 for 720P, 3.0 for 480P
scheduler = UniPCMultistepScheduler(
    prediction_type="flow_prediction", use_flow_sigmas=True, num_train_timesteps=1000, flow_shift=flow_shift
)
pipe.scheduler = scheduler
WanTransformer3DModel.forward = wan_eval_forward
pipe.transformer.hidden_states_list = []
pipe.transformer.hidden_states_first_block = []
delta_hidden =[]
delta_hidden_firstblock =[]
prompt = "A cat and a dog baking a cake together in a kitchen. The cat is carefully measuring flour, while the dog is stirring the batter with a wooden spoon. The kitchen is cozy, with sunlight streaming through the window."
#prompt = "a still frame, a stop sign"
negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"
for i in range(2):
    start = time.time()
    output = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=480,
        width=832,
        num_frames=81,
        guidance_scale=5.0,
        generator=paddle.Generator().manual_seed(42),
    ).frames[0]
    elapsed1 = time.time() - start
    print(f"第一次运行时间: {elapsed1:.2f}s")
    export_to_video_2(output, "output.mp4", fps=16)
    for i in range(0,len(pipe.transformer.hidden_states_list)):
        if i %2 !=0:
       
            dh = ( pipe.transformer.hidden_states_list[i] - pipe.transformer.hidden_states_list[i-2]).abs().mean()/ pipe.transformer.hidden_states_list[i-2].abs().mean()
            de = (pipe.transformer.hidden_states_first_block[i] - pipe.transformer.hidden_states_first_block[i-2]).abs().mean()/pipe.transformer.hidden_states_first_block[i-2].abs().mean()
            delta_hidden.append(dh.item())
            delta_hidden_firstblock.append(de.item())

    plt.figure(figsize=(10, 6))
    # sns.heatmap(delta_hidden_array, cmap="viridis", cbar=True)
    plt.plot(delta_hidden_firstblock, label="Δ firtblock hidden")
    plt.plot(delta_hidden, label="Δ hidden")
    # plt.plot(delta_encoder, label="Δ encoder_hidden_states")
    plt.xlabel("diffusion process")
    plt.ylabel("difference between consecutive timesteps")
    plt.title("Δ Hidden States(L2 Mean Difference)")
    plt.legend()
    plt.savefig("Wan_fitstblock_relation_uncon.png")
    plt.show()
   
