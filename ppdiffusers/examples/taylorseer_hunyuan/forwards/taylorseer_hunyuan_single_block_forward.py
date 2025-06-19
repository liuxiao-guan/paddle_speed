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
from ppdiffusers import HunyuanVideoPipeline, HunyuanVideoTransformer3DModel
from ppdiffusers.models.transformer_hunyuan_video import HunyuanVideoSingleTransformerBlock
from taylor_utils import taylor_cache_init,derivative_approximation,taylor_formula

def taylorseer_hunyuan_single_block_forward(
        self:HunyuanVideoSingleTransformerBlock,
        hidden_states: paddle.Tensor,
        encoder_hidden_states: paddle.Tensor,
        temb: paddle.Tensor,
        attention_mask: Optional[paddle.Tensor] = None,
        image_rotary_emb: Optional[Tuple[paddle.Tensor, paddle.Tensor]] = None,
        cache_dic: Optional[Dict] = None,
        current: Optional[Dict] = None,
    ) -> paddle.Tensor:
        text_seq_length = tuple(encoder_hidden_states.shape)[1]
        hidden_states = paddle.concat(x=[hidden_states, encoder_hidden_states], axis=1)

        residual = hidden_states

        # 1. Input normalization
        norm_hidden_states, gate = self.norm(hidden_states, emb=temb)
        mlp_hidden_states = self.act_mlp(self.proj_mlp(norm_hidden_states))

        norm_hidden_states, norm_encoder_hidden_states = (
            norm_hidden_states[:, :-text_seq_length, :],
            norm_hidden_states[:, -text_seq_length:, :],
        )
        if current['type'] == 'full':
            current['module'] = 'total'
            taylor_cache_init(cache_dic=cache_dic, current=current)
            # 2. Attention
            attn_output, context_attn_output = self.attn(
                hidden_states=norm_hidden_states,
                encoder_hidden_states=norm_encoder_hidden_states,
                attention_mask=attention_mask,
                image_rotary_emb=image_rotary_emb,
            )
            attn_output = paddle.concat(x=[attn_output, context_attn_output], axis=1)

            # 3. Modulation and residual connection
            hidden_states = paddle.concat(x=[attn_output, mlp_hidden_states], axis=2)
            hidden_states = gate.unsqueeze(axis=1) * self.proj_out(hidden_states)
            derivative_approximation(cache_dic=cache_dic, current=current, feature=hidden_states)
        elif current['type'] == 'taylor_cache':
            current['module'] = 'total'
            hidden_states = taylor_formula(cache_dic=cache_dic, current=current)

        hidden_states = hidden_states + residual

        hidden_states, encoder_hidden_states = (
            hidden_states[:, :-text_seq_length, :],
            hidden_states[:, -text_seq_length:, :],
        )

        return hidden_states, encoder_hidden_states