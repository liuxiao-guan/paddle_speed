import inspect
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import paddle
from ppdiffusers import HunyuanVideoPipeline, HunyuanVideoTransformer3DModel
# from ppdiffusers import HunyuanVideoPipelineOutput
from ppdiffusers.pipelines.hunyuan_video.pipeline_output import HunyuanVideoPipelineOutput
# from .pipeline_output import HunyuanVideoPipelineOutput
# from ...callbacks import MultiPipelineCallbacks, PipelineCallback
from ppdiffusers.callbacks import MultiPipelineCallbacks,PipelineCallback
from ppdiffusers.utils import logger,USE_PEFT_BACKEND
from ppdiffusers.utils import scale_lora_layers,unscale_lora_layers
from ppdiffusers.models.modeling_outputs import Transformer2DModelOutput
from cache_functions import cal_type
from taylor_utils import firstblock_derivative_approximation,firstblock_taylor_formula,step_taylor_formula,step_derivative_approximation
def taylorseer_step_firstpredict_hunyuan_forward(
        self:HunyuanVideoTransformer3DModel,
        hidden_states: paddle.Tensor,
        timestep: paddle.Tensor,
        encoder_hidden_states: paddle.Tensor,
        encoder_attention_mask: paddle.Tensor,
        pooled_projections: paddle.Tensor,
        guidance: paddle.Tensor = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
        **kwargs, ##
    ) -> Union[paddle.Tensor, Dict[str, paddle.Tensor]]:
        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            scale_lora_layers(self, lora_scale)
        elif (
            attention_kwargs is not None
            and attention_kwargs.get("scale", None) is not None
        ):
            logger.warning(
                "Passing `scale` via `attention_kwargs` when not using the PEFT backend is ineffective."
            )

        batch_size, num_channels, num_frames, height, width = tuple(hidden_states.shape)
        p, p_t = self.config.patch_size, self.config.patch_size_t
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p
        post_patch_width = width // p

        # 1. RoPE
        image_rotary_emb = self.rope(hidden_states)

        # 2. Conditional embeddings
        temb = self.time_text_embed(timestep, guidance, pooled_projections)
        hidden_states = self.x_embedder(hidden_states)
        encoder_hidden_states = self.context_embedder(encoder_hidden_states, timestep, encoder_attention_mask)

        # 3. Attention mask preparation
        latent_sequence_length = tuple(hidden_states.shape)[1]
        condition_sequence_length = tuple(encoder_hidden_states.shape)[1]
        sequence_length = latent_sequence_length + condition_sequence_length
        attention_mask = paddle.zeros(
            shape=[batch_size, sequence_length]).to(paddle.bool) # [B, N]

        effective_condition_sequence_length = encoder_attention_mask.sum(axis=1, dtype="int32")  # [B,]
        effective_sequence_length = (latent_sequence_length + effective_condition_sequence_length)
        for i in range(batch_size):
            attention_mask[(i), : effective_sequence_length[i]] = 1
        # [B, 1, 1, N], for broadcasting across attention heads
        attention_mask = attention_mask.unsqueeze(axis=1).unsqueeze(axis=1)
        ## TaylorSeer
        cache_dic = kwargs.get("cache_dic", None)
        current = kwargs.get("current", None)

        # predict firstblock
        pre_firstblock_hidden_states = firstblock_taylor_formula(cache_dic=cache_dic, current=current)
        hidden_states, encoder_hidden_states =self.transformer_blocks[0](
            #hidden_states, encoder_hidden_states, temb, attention_mask, image_rotary_emb,cache_dic, current, 
            hidden_states, encoder_hidden_states, temb, attention_mask, image_rotary_emb
        )
        if self.cnt > 5:
            self.predict_loss = (pre_firstblock_hidden_states - hidden_states).abs().mean()/hidden_states.abs().mean()
            can_use_cache = self.predict_loss < self.threshold
            if can_use_cache == False:
                # 要用block得输入来预测
                current['block_activated_steps'].append(current['step'])
                firstblock_derivative_approximation(cache_dic=cache_dic, current=current, feature=hidden_states)
        else:
            current['block_activated_steps'].append(current['step'])
            firstblock_derivative_approximation(cache_dic=cache_dic, current=current, feature=hidden_states)

        if self.cnt == 0 or self.cnt == self.num_steps - 1:
                should_calc = True
                # self.pre_firstblock_hidden_states = None
                # self.predict_hidden_states=None
                self.predict_loss=None
                # self.pre_compute_hidden = None
                
        else:
            if self.predict_loss is None:
                can_use_cache = False
            # else:
            #     can_use_cache = self.predict_loss < self.threshold
            should_calc = not can_use_cache
        
        self.cnt += 1
        if self.cnt == self.num_steps:
            self.cnt = 0
        current['type'] = 'taylor_cache'
        # cal_type(cache_dic, current)
        current['compute'] = not (current['type'] == 'aggressive')
        if current['compute']:
            if should_calc:
                # 4. Transformer blocks
                if self.training and self.gradient_checkpointing:

                    def create_custom_forward(module, return_dict=None):
                        def custom_forward(*inputs):
                            if return_dict is not None:
                                return module(*inputs, return_dict=return_dict)
                            else:
                                return module(*inputs)

                        return custom_forward

                    ckpt_kwargs = {}

                    for block in self.transformer_blocks:
                        hidden_states, encoder_hidden_states = paddle.distributed.fleet.utils.recompute(
                            create_custom_forward(block),
                            hidden_states,
                            encoder_hidden_states,
                            temb,
                            attention_mask,
                            image_rotary_emb,
                            **ckpt_kwargs,
                        )
                        hidden_states, encoder_hidden_states = paddle.distributed.fleet.utils.recompute(
                            create_custom_forward(block),
                            hidden_states,
                            encoder_hidden_states,
                            temb,
                            attention_mask,
                            image_rotary_emb,
                            **ckpt_kwargs,
                        )

                    for block in self.single_transformer_blocks:
                        hidden_states, encoder_hidden_states = paddle.distributed.fleet.utils.recompute(
                            create_custom_forward(block),
                            hidden_states,
                            encoder_hidden_states,
                            temb,
                            attention_mask,
                            image_rotary_emb,
                            **ckpt_kwargs,
                        )

                else:
                    current['activated_steps'].append(current['step'])
                    for i,block in enumerate(self.transformer_blocks):
                        #因为上面已经算了现在就不能算了
                        if i==0:
                            continue
                        hidden_states, encoder_hidden_states = block(
                            #hidden_states, encoder_hidden_states, temb, attention_mask, image_rotary_emb,cache_dic, current, 
                            hidden_states, encoder_hidden_states, temb, attention_mask, image_rotary_emb
                        )
                    
                    for i, block in enumerate(self.single_transformer_blocks):
                        hidden_states, encoder_hidden_states = block(
                            #hidden_states, encoder_hidden_states, temb, attention_mask, image_rotary_emb,cache_dic, current, 
                            hidden_states, encoder_hidden_states, temb, attention_mask, image_rotary_emb
                        )
                    step_derivative_approximation(cache_dic=cache_dic, current=current, feature=hidden_states)
            else:
                hidden_states = step_taylor_formula(cache_dic=cache_dic, current=current)
        else:
            hidden_states = cache_dic['aggressive_feature']
        if cache_dic['duca']:
            cache_dic['aggressive_feature'] = hidden_states


        # 5. Output projection
        hidden_states = self.norm_out(hidden_states, temb)
        hidden_states = self.proj_out(hidden_states)

        hidden_states = hidden_states.reshape(
            [batch_size, post_patch_num_frames, post_patch_height, post_patch_width, -1, p_t, p, p]
        )
        hidden_states = hidden_states.transpose(perm=[0, 4, 1, 5, 2, 6, 3, 7])
        hidden_states = (hidden_states.flatten(start_axis=6, stop_axis=7).flatten(start_axis=4, stop_axis=5).flatten(start_axis=2, stop_axis=3))

        if USE_PEFT_BACKEND:
           # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (hidden_states,)

        return Transformer2DModelOutput(sample=hidden_states)