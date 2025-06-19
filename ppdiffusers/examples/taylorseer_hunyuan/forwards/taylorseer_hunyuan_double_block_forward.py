
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
from taylor_utils import taylor_cache_init,derivative_approximation,taylor_formula
from ppdiffusers.models.transformer_hunyuan_video import HunyuanVideoTransformerBlock 

def taylorseer_hunyuan_double_block_forward(
        self:HunyuanVideoTransformerBlock,
        hidden_states: paddle.Tensor,
        encoder_hidden_states: paddle.Tensor,
        temb: paddle.Tensor,
        attention_mask: Optional[paddle.Tensor] = None,
        freqs_cis: Optional[Tuple[paddle.Tensor, paddle.Tensor]] = None,
        cache_dic=None,
        current=None, 
    ) -> Tuple[paddle.Tensor, paddle.Tensor]:
        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(hidden_states, emb=temb)
        norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.norm1_context(
            encoder_hidden_states, emb=temb
        )
        #current['type'] = 'full'
        if current['type'] == 'full':
            current['module'] = 'attn'
            taylor_cache_init(cache_dic=cache_dic, current=current)
            # 2. Joint attention
            attn_output, context_attn_output = self.attn(
                hidden_states=norm_hidden_states,
                encoder_hidden_states=norm_encoder_hidden_states,
                attention_mask=attention_mask,
                image_rotary_emb=freqs_cis,
            )
            current['module'] = 'img_attn'
            taylor_cache_init(cache_dic=cache_dic, current=current)

            derivative_approximation(cache_dic=cache_dic, current=current, feature=attn_output)
            hidden_states = hidden_states + attn_output * gate_msa.unsqueeze(axis=1)
            
            current['module'] = 'txt_attn'
            taylor_cache_init(cache_dic=cache_dic, current=current)

            derivative_approximation(cache_dic=cache_dic, current=current, feature=context_attn_output)

            # 3. Modulation and residual connection
            
            encoder_hidden_states = (encoder_hidden_states + context_attn_output * c_gate_msa.unsqueeze(axis=1))

            current['module'] = 'img_mlp'
            taylor_cache_init(cache_dic=cache_dic, current=current)
            norm_hidden_states = self.norm2(hidden_states)
            norm_hidden_states = (norm_hidden_states * (1 + scale_mlp[:, (None)]) + shift_mlp[:, (None)])
            ff_output = self.ff(norm_hidden_states)
            derivative_approximation(cache_dic=cache_dic, current=current, feature=ff_output)

            current['module'] = 'txt_mlp'
            taylor_cache_init(cache_dic=cache_dic, current=current)
            norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
            norm_encoder_hidden_states = (norm_encoder_hidden_states * (1 + c_scale_mlp[:, (None)]) + c_shift_mlp[:, (None)])
            context_ff_output = self.ff_context(norm_encoder_hidden_states)
            derivative_approximation(cache_dic=cache_dic, current=current, feature=context_ff_output)

            hidden_states = hidden_states + gate_mlp.unsqueeze(axis=1) * ff_output
            encoder_hidden_states = (encoder_hidden_states + c_gate_mlp.unsqueeze(axis=1) * context_ff_output)
        elif current['type'] == 'taylor_cache':
            current['module'] = 'attn'
            # Process attention outputs for the `hidden_states`.
            current['module'] = 'img_attn'

            attn_output = taylor_formula(cache_dic=cache_dic, current=current)
            hidden_states = hidden_states + attn_output * gate_msa.unsqueeze(axis=1)
            current['module'] = 'img_mlp'
            ff_output = taylor_formula(cache_dic=cache_dic, current=current)
            hidden_states = hidden_states + gate_mlp.unsqueeze(axis=1) * ff_output
            current['module'] = 'txt_attn'
            context_attn_output = taylor_formula(cache_dic=cache_dic, current=current)
            encoder_hidden_states = (encoder_hidden_states + context_attn_output * c_gate_msa.unsqueeze(axis=1))
            current['module'] = 'txt_mlp'
            context_ff_output = taylor_formula(cache_dic=cache_dic, current=current)
            encoder_hidden_states = (encoder_hidden_states + c_gate_mlp.unsqueeze(axis=1) * context_ff_output)


        return hidden_states, encoder_hidden_states