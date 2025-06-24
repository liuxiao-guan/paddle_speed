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
from taylorseer_utils import step_uncond_derivative_approximation,step_cond_derivative_approximation, \
taylor_formula,firstblock_derivative_approximation,firstblock_taylor_formula


def wan_firstpredict_step_forward(
        self:WanTransformer3DModel,
        hidden_states: paddle.Tensor,
        timestep: paddle.Tensor,
        encoder_hidden_states: paddle.Tensor,
        encoder_hidden_states_image: Optional[paddle.Tensor] = None,
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        current = None,
        cache_dic = None, 
    ) -> Union[paddle.Tensor, Dict[str, paddle.Tensor]]:
        

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
        if current['stream'] == 'cond_stream' :
            #cal_type(cache_dic, current)
            pre_firstblock_hidden_states = firstblock_taylor_formula(cache_dic=cache_dic, current=current)
            hidden_states =self.blocks[0](
                hidden_states, encoder_hidden_states, timestep_proj, rotary_emb
            )
            if self.cnt > 5:
                self.predict_loss = (pre_firstblock_hidden_states - hidden_states).abs().mean()/hidden_states.abs().mean()
                can_use_cache = self.predict_loss < self.threshold
                if can_use_cache == False:
                    # 要用block得输入来预测
                    current['block_activated_steps'].append(current['step'])
                    firstblock_derivative_approximation(cache_dic=cache_dic, current=current, feature=hidden_states)
            else:
                # 要用block得输入来预测
                current['block_activated_steps'].append(current['step'])
                firstblock_derivative_approximation(cache_dic=cache_dic, current=current, feature=hidden_states)
            if self.cnt == 0 or self.cnt == self.num_steps - 1:
                self.should_calc = True
                self.predict_loss=None   
            else:
                if self.predict_loss is None:
                    can_use_cache = False
                self.should_calc = not can_use_cache
            self.cnt += 1
            if self.cnt == self.num_steps:
                self.cnt = 0
        # 4. Transformer blocks
        # self.should_calc = True
        if self.should_calc ==True:
            if paddle.is_grad_enabled() and self.gradient_checkpointing:
                for block in self.blocks:
                    hidden_states = self._gradient_checkpointing_func(
                        block, hidden_states, encoder_hidden_states, timestep_proj, rotary_emb
                    )
            else:
                if current['stream'] == 'cond_stream':
                    #current['layer'] = 0
                    current['activated_steps'].append(current['step'])
                for i ,block in enumerate(self.blocks):
                    if current['stream'] == 'cond_stream':
                        if i ==0 :
                            continue
                    # current['layer'] = i
                    #hidden_states = block(hidden_states, encoder_hidden_states, timestep_proj, rotary_emb,current,cache_dic)
                    hidden_states = block(hidden_states, encoder_hidden_states, timestep_proj, rotary_emb)
            
            if current['stream'] == "cond_stream":   
                step_cond_derivative_approximation(cache_dic=cache_dic, current=current, feature=hidden_states)
            else:
                step_uncond_derivative_approximation(cache_dic=cache_dic, current=current, feature=hidden_states)

        else:
            distance= current['step'] - current['activated_steps'][-1]
            if current['stream'] == "cond_stream":   
                hidden_states=taylor_formula(derivative_dict=cache_dic['cache']['cond_hidden'],distance=distance)
            else:
                hidden_states =taylor_formula(derivative_dict=cache_dic['cache']['uncond_hidden'],distance=distance)
            #hidden_states = step_taylor_formula(cache_dic=cache_dic, current=current)
        
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