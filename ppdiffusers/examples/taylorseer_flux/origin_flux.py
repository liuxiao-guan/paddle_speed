import os
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from ppdiffusers.models.modeling_outputs import  Transformer2DModelOutput
from ppdiffusers.utils import USE_PEFT_BACKEND, is_torch_version, logger, scale_lora_layers, unscale_lora_layers
import matplotlib.pyplot as plt 

import paddle.utils.flops

def TeaCacheForward(
        self,
        hidden_states: paddle.Tensor,
        encoder_hidden_states: paddle.Tensor = None,
        pooled_projections: paddle.Tensor = None,
        timestep: paddle.Tensor = None,
        img_ids: paddle.Tensor = None,
        txt_ids: paddle.Tensor = None,
        guidance: paddle.Tensor = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        controlnet_block_samples=None,
        controlnet_single_block_samples=None,
        return_dict: bool = True,
        controlnet_blocks_repeat: bool = False,
    ) -> Union[paddle.Tensor, Transformer2DModelOutput]:
        """
        The [`FluxTransformer2DModel`] forward method.

        Args:
            hidden_states (`paddle.Tensor` of shape `(batch size, channel, height, width)`):
                Input `hidden_states`.
            encoder_hidden_states (`paddle.Tensor` of shape `(batch size, sequence_len, embed_dims)`):
                Conditional embeddings (embeddings computed from the input conditions such as prompts) to use.
            pooled_projections (`paddle.Tensor` of shape `(batch_size, projection_dim)`): Embeddings projected
                from the embeddings of input conditions.
            timestep ( `paddle.Tensor`):
                Used to indicate denoising step.
            block_controlnet_hidden_states: (`list` of `paddle.Tensor`):
                A list of tensors that if specified are added to the residuals of transformer blocks.
            joint_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.transformer_2d.Transformer2DModelOutput`] instead of a plain
                tuple.

        Returns:
            If `return_dict` is True, an [`~models.transformer_2d.Transformer2DModelOutput`] is returned, otherwise a
            `tuple` where the first element is the sample tensor.
        """
        if joint_attention_kwargs is not None:
            joint_attention_kwargs = joint_attention_kwargs.copy()
            lora_scale = joint_attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)
        else:
            if joint_attention_kwargs is not None and joint_attention_kwargs.get("scale", None) is not None:
                logger.warning(
                    "Passing `scale` via `joint_attention_kwargs` when not using the PEFT backend is ineffective."
                )

        hidden_states = self.x_embedder(hidden_states)

        timestep = timestep.to(hidden_states.dtype) * 1000
        if guidance is not None:
            guidance = guidance.to(hidden_states.dtype) * 1000
        else:
            guidance = None

        temb = (
            self.time_text_embed(timestep, pooled_projections)
            if guidance is None
            else self.time_text_embed(timestep, guidance, pooled_projections)
        )
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)

        if txt_ids.ndim == 3:
            logger.warning(
                "Passing `txt_ids` 3d torch.Tensor is deprecated."
                "Please remove the batch dimension and pass it as a 2d torch Tensor"
            )
            txt_ids = txt_ids[0]
        if img_ids.ndim == 3:
            logger.warning(
                "Passing `img_ids` 3d torch.Tensor is deprecated."
                "Please remove the batch dimension and pass it as a 2d torch Tensor"
            )
            img_ids = img_ids[0]

        ids = paddle.concat((txt_ids, img_ids), axis=0)
        image_rotary_emb = self.pos_embed(ids)

        if joint_attention_kwargs is not None and "ip_adapter_image_embeds" in joint_attention_kwargs:
            ip_adapter_image_embeds = joint_attention_kwargs.pop("ip_adapter_image_embeds")
            ip_hidden_states = self.encoder_hid_proj(ip_adapter_image_embeds)
            joint_attention_kwargs.update({"ip_hidden_states": ip_hidden_states})

        
        for index_block, block in enumerate(self.transformer_blocks):
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                encoder_hidden_states, hidden_states = paddle.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    encoder_hidden_states,
                    temb,
                    image_rotary_emb,
                    **ckpt_kwargs,
                )

            else:
                encoder_hidden_states, hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                    joint_attention_kwargs=joint_attention_kwargs,
                )
                if index_block ==0:
                    self.hidden_states_first_block.append(hidden_states.detach().clone())


            # controlnet residual
            if controlnet_block_samples is not None:
                interval_control = len(self.transformer_blocks) / len(controlnet_block_samples)
                interval_control = int(np.ceil(interval_control))
                # For Xlabs ControlNet.
                if controlnet_blocks_repeat:
                    hidden_states = (
                        hidden_states + controlnet_block_samples[index_block % len(controlnet_block_samples)]
                    )
                else:
                    hidden_states = hidden_states + controlnet_block_samples[index_block // interval_control]
        hidden_states = paddle.concat([encoder_hidden_states, hidden_states], axis=1)

        for index_block, block in enumerate(self.single_transformer_blocks):
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                hidden_states = paddle.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    temb,
                    image_rotary_emb,
                    **ckpt_kwargs,
                )

            else:
                hidden_states = block(
                    hidden_states=hidden_states,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                    joint_attention_kwargs=joint_attention_kwargs,
                )

            # controlnet residual
            if controlnet_single_block_samples is not None:
                interval_control = len(self.single_transformer_blocks) / len(controlnet_single_block_samples)
                interval_control = int(np.ceil(interval_control))
                hidden_states[:, encoder_hidden_states.shape[1] :, ...] = (
                    hidden_states[:, encoder_hidden_states.shape[1] :, ...]
                    + controlnet_single_block_samples[index_block // interval_control]
                )
        self.hidden_states_list.append(hidden_states.detach().clone())
        hidden_states = hidden_states[:, encoder_hidden_states.shape[1] :, ...]

        hidden_states = self.norm_out(hidden_states, temb)
        output = self.proj_out(hidden_states)
        

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)

from ppdiffusers import UNet2DConditionModel, LCMScheduler,FluxPipeline
from ppdiffusers import DPMSolverMultistepScheduler
from ppdiffusers.utils import load_image, export_to_video
from ppdiffusers.models.transformer_flux import FluxTransformer2DModel



# import seaborn as sns
# 假设你有 prompt_list
all_delta_hidden = []
all_delta_hidden_firstblock = []

seed = 42
prompt = "An image of a squirrel"
pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", paddle_dtype=paddle.bfloat16)

FluxTransformer2DModel.forward = TeaCacheForward
pipe.transformer.enable_teacache = False
pipe.transformer.cnt = 0
pipe.transformer.num_steps = 50
pipe.transformer.rel_l1_thresh = (
    0.25  # 0.25 for 1.5x speedup, 0.4 for 1.8x speedup, 0.6 for 2.0x speedup, 0.8 for 2.25x speedup
)
pipe.transformer.accumulated_rel_l1_distance = 0
pipe.transformer.previous_modulated_input = None
pipe.transformer.previous_residual = None
# pipe.transformer.delta_hidden = []
# pipe.transformer.delta_hidden_firstblock = []
pipe.transformer.hidden_states_list = []
pipe.transformer.hidden_states_first_block = []

with open('/root/paddlejob/workspace/env_run/gxl/paddle_speed/ppdiffusers/examples/taylorseer_flux/prompts/DrawBench.txt', 'r', encoding='utf-8') as f:
    all_prompts = [line.strip() for line in f if line.strip()]


import time
for i,prompt in enumerate(all_prompts):
    # if i >4:
    #     break
    # 清空 pipe 内部记录（每个 prompt 一次）
    pipe.transformer.hidden_states_list = []
    pipe.transformer.hidden_states_first_block = []
    start_time = time.time()
    img = pipe(
        prompt, 
        num_inference_steps=50,
        generator=paddle.Generator().manual_seed(seed)
        ).images[0]
    end = time.time()
    print(f" time takes: {end - start_time:.2f} sec")
    img.save(os.path.join("/root/paddlejob/workspace/env_run/gxl/output/PaddleMIX/inf_speed_bf16/Drawbench_origin",f"{i}.png"))
    delta_hidden =[]
    delta_hidden_firstblock =[]
    for i in range(len(pipe.transformer.hidden_states_list)):
    
        dh = ( pipe.transformer.hidden_states_list[i] - pipe.transformer.hidden_states_list[i-1]).abs().mean()/ pipe.transformer.hidden_states_list[i].abs().mean()
        de = (pipe.transformer.hidden_states_first_block[i] - pipe.transformer.hidden_states_first_block[i-1]).abs().mean()/pipe.transformer.hidden_states_first_block[i].abs().mean()
        delta_hidden.append(dh)
        delta_hidden_firstblock.append(de)

    all_delta_hidden.append(delta_hidden)
    all_delta_hidden_firstblock.append(delta_hidden_firstblock)

# 转为 numpy，补齐长度（如果每步长度一样则不需要）
min_len = min(len(x) for x in all_delta_hidden)
delta_hidden_avg = np.mean([[float(v) for v in x[:min_len]] for x in all_delta_hidden], axis=0)
delta_firstblock_avg = np.mean([[float(v) for v in x[:min_len]] for x in all_delta_hidden_firstblock], axis=0)


# 绘图
plt.figure(figsize=(10, 6))
plt.plot(delta_firstblock_avg, label="Δ first block hidden (avg)", linewidth=2)
plt.plot(delta_hidden_avg, label="Δ last block hidden (avg)", linewidth=2)
plt.xlabel("Diffusion step")
plt.ylabel("Normalized Δ Hidden")
plt.title("Average Hidden State Changes Across Prompts")
plt.legend()
plt.tight_layout()
plt.savefig("Teacache_Flux_relation_multi_prompt.png")
plt.show()