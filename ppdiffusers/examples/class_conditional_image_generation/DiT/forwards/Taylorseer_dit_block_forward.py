
import logging
import time
import paddle
import paddle.distributed as dist
from ppdiffusers.transformers import T5EncoderModel
from xfuser import xFuserFluxPipeline, xFuserArgs
from xfuser.config import FlexibleArgumentParser
from xfuser.core.distributed import (
    get_world_group,
    get_data_parallel_rank,
    get_data_parallel_world_size,
    get_runtime_state,
    is_dp_last_group,
)

from typing import Any, Dict, Optional
from paddle import nn
import paddle


from taylor_utils import taylor_cache_init, derivative_approximation,taylor_formula


def _chunked_feed_forward(
    ff: nn.Layer, hidden_states: paddle.Tensor, chunk_dim: int, chunk_size: int, lora_scale: Optional[float] = None
):
    # "feed_forward_chunk_size" can be used to save memory
    if hidden_states.shape[chunk_dim] % chunk_size != 0:
        raise ValueError(
            f"`hidden_states` dimension to be chunked: {hidden_states.shape[chunk_dim]} has to be divisible by chunk size: {chunk_size}. Make sure to set an appropriate `chunk_size` when calling `unet.enable_forward_chunking`."
        )

    num_chunks = hidden_states.shape[chunk_dim] // chunk_size
    if lora_scale is None:
        ff_output = paddle.concat(
            [ff(hid_slice) for hid_slice in hidden_states.chunk(num_chunks, axis=chunk_dim)],
            dim=chunk_dim,
        )
    else:
        # TOOD(Patrick): LoRA scale can be removed once PEFT refactor is complete
        ff_output = paddle.concat(
            [ff(hid_slice, scale=lora_scale) for hid_slice in hidden_states.chunk(num_chunks, axis=chunk_dim)],
            axis=chunk_dim,
        )

    return ff_output

def taylorseer_dit_block_forward(
        self,
        hidden_states: paddle.Tensor,
        attention_mask: Optional[paddle.Tensor] = None,
        encoder_hidden_states: Optional[paddle.Tensor] = None,
        encoder_attention_mask: Optional[paddle.Tensor] = None,
        timestep: Optional[paddle.Tensor] = None,
        cross_attention_kwargs: Dict[str, Any] = None,
        class_labels: Optional[paddle.Tensor] = None,
        current=None, 
        cache_dic=None,
    ) -> paddle.Tensor:
        # Notice that normalization is always applied before the real computation in the following blocks.
        # 0. Self-Attention
        batch_size = hidden_states.shape[0]

        if self.use_ada_layer_norm:
            norm_hidden_states = self.norm1(hidden_states, timestep)
        elif self.use_ada_layer_norm_zero:
            norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
                hidden_states, timestep, class_labels, hidden_dtype=hidden_states.dtype
            )
        elif self.use_layer_norm:
            norm_hidden_states = self.norm1(hidden_states)
        elif self.use_ada_layer_norm_single:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                self.scale_shift_table[None] + timestep.reshape([batch_size, 6, -1])
            ).chunk(6, axis=1)
            norm_hidden_states = self.norm1(hidden_states)
            norm_hidden_states = norm_hidden_states * (1 + scale_msa) + shift_msa
            norm_hidden_states = norm_hidden_states.squeeze(1)
        else:
            raise ValueError("Incorrect norm used")

        if self.pos_embed is not None:
            norm_hidden_states = self.pos_embed(norm_hidden_states)

        # 1. Retrieve lora scale.
        lora_scale = cross_attention_kwargs.get("scale", 1.0) if cross_attention_kwargs is not None else 1.0

        # 2. Prepare GLIGEN inputs
        cross_attention_kwargs = cross_attention_kwargs.copy() if cross_attention_kwargs is not None else {}
        gligen_kwargs = cross_attention_kwargs.pop("gligen", None)
        if current['type'] == 'full': 
            
            current['module'] = 'attn'
            taylor_cache_init(cache_dic, current)
            attn_output = self.attn1(
                norm_hidden_states,
                encoder_hidden_states=encoder_hidden_states if self.only_cross_attention else None,
                attention_mask=attention_mask,
                **cross_attention_kwargs,
            )
            derivative_approximation(cache_dic, current, attn_output)
            if self.use_ada_layer_norm_zero:
                attn_output = gate_msa.unsqueeze(1) * attn_output
            elif self.use_ada_layer_norm_single:
                attn_output = gate_msa * attn_output

            hidden_states = attn_output + hidden_states
            if hidden_states.ndim == 4:
                hidden_states = hidden_states.squeeze(1)

            # 2.5 GLIGEN Control
            if gligen_kwargs is not None:
                hidden_states = self.fuser(hidden_states, gligen_kwargs["objs"])

            # 3. Cross-Attention not exist
            if self.attn2 is not None:
                if self.use_ada_layer_norm:
                    norm_hidden_states = self.norm2(hidden_states, timestep)
                elif self.use_ada_layer_norm_zero or self.use_layer_norm:
                    norm_hidden_states = self.norm2(hidden_states)
                elif self.use_ada_layer_norm_single:
                    # For PixArt norm2 isn't applied here:
                    # https://github.com/PixArt-alpha/PixArt-alpha/blob/0f55e922376d8b797edd44d25d0e7464b260dcab/diffusion/model/nets/PixArtMS.py#L70C1-L76C103
                    norm_hidden_states = hidden_states
                else:
                    raise ValueError("Incorrect norm")

                if self.pos_embed is not None and self.use_ada_layer_norm_single is False:
                    norm_hidden_states = self.pos_embed(norm_hidden_states)

                attn_output = self.attn2(
                    norm_hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=encoder_attention_mask,
                    **cross_attention_kwargs,
                )
                hidden_states = attn_output + hidden_states

            # 4. Feed-forward
            if not self.use_ada_layer_norm_single:
                norm_hidden_states = self.norm3(hidden_states)

            if self.use_ada_layer_norm_zero:
                norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

            if self.use_ada_layer_norm_single:
                norm_hidden_states = self.norm2(hidden_states)
                norm_hidden_states = norm_hidden_states * (1 + scale_mlp) + shift_mlp
            current['module'] = 'mlp'
            taylor_cache_init(cache_dic, current)
            if self._chunk_size is not None:
                # "feed_forward_chunk_size" can be used to save memory
                ff_output = _chunked_feed_forward(
                    self.ff, norm_hidden_states, self._chunk_dim, self._chunk_size, lora_scale=lora_scale
                )
            else:
                ff_output = self.ff(norm_hidden_states, scale=lora_scale)
            derivative_approximation(cache_dic, current, ff_output)

        elif current['type'] == 'Taylor':
            current['module'] = 'attn'
            attn_output = taylor_formula(cache_dic, current)
            if self.use_ada_layer_norm_zero:
                attn_output = gate_msa.unsqueeze(1) * attn_output
            elif self.use_ada_layer_norm_single:
                attn_output = gate_msa * attn_output

            hidden_states = attn_output + hidden_states
            if hidden_states.ndim == 4:
                hidden_states = hidden_states.squeeze(1)

            # 2.5 GLIGEN Control
            if gligen_kwargs is not None:
                hidden_states = self.fuser(hidden_states, gligen_kwargs["objs"])

            # 4. Feed-forward
            # if not self.use_ada_layer_norm_single:
            #     norm_hidden_states = self.norm3(hidden_states)

            # if self.use_ada_layer_norm_zero:
            #     norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

            # if self.use_ada_layer_norm_single:
            #     norm_hidden_states = self.norm2(hidden_states)
            #     norm_hidden_states = norm_hidden_states * (1 + scale_mlp) + shift_mlp
            current['module'] = 'mlp'
            ff_output = taylor_formula(cache_dic, current)


        if self.use_ada_layer_norm_zero:
            ff_output = gate_mlp.unsqueeze(1) * ff_output
        elif self.use_ada_layer_norm_single:
            ff_output = gate_mlp * ff_output

        hidden_states = ff_output + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)

        return hidden_states