

import paddle

from typing import Any, Dict, Optional
from ppdiffusers.models.modeling_outputs import Transformer2DModelOutput
from ppdiffusers.utils import USE_PEFT_BACKEND, recompute_use_reentrant,use_old_recompute
import paddle
import numpy as np

import paddle.nn.functional as F
from cache_functions import cache_init, cal_type
from paddle.distributed.fleet.utils import recompute

def taylorseer_dit_trans_forward(
        self,
        hidden_states: paddle.Tensor,
        encoder_hidden_states: Optional[paddle.Tensor] = None,
        timestep: Optional[paddle.Tensor] = None,
        added_cond_kwargs: Dict[str, paddle.Tensor] = None,
        class_labels: Optional[paddle.Tensor] = None,
        cross_attention_kwargs: Dict[str, Any] = None,
        attention_mask: Optional[paddle.Tensor] = None,
        encoder_attention_mask: Optional[paddle.Tensor] = None,
        return_dict: bool = True,
        cache_dic=None,
        current=None,
    ):
        """
        The [`Transformer2DModel`] forward method.

        Args:
            hidden_states (`paddle.Tensor` of shape `(batch size, num latent pixels)` if discrete, `paddle.Tensor` of shape `(batch size, channel, height, width)` if continuous):
                Input `hidden_states`.
            encoder_hidden_states ( `paddle.Tensor` of shape `(batch size, sequence len, embed dims)`, *optional*):
                Conditional embeddings for cross attention layer. If not given, cross-attention defaults to
                self-attention.
            timestep ( `paddle.Tensor`, *optional*):
                Used to indicate denoising step. Optional timestep to be applied as an embedding in `AdaLayerNorm`.
            class_labels ( `paddle.Tensor` of shape `(batch size, num classes)`, *optional*):
                Used to indicate class labels conditioning. Optional class labels to be applied as an embedding in
                `AdaLayerZeroNorm`.
            cross_attention_kwargs ( `Dict[str, Any]`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            attention_mask ( `paddle.Tensor`, *optional*):
                An attention mask of shape `(batch, key_tokens)` is applied to `encoder_hidden_states`. If `1` the mask
                is kept, otherwise if `0` it is discarded. Mask will be converted into a bias, which adds large
                negative values to the attention scores corresponding to "discard" tokens.
            encoder_attention_mask ( `paddle.Tensor`, *optional*):
                Cross-attention mask applied to `encoder_hidden_states`. Two formats supported:

                    * Mask `(batch, sequence_length)` True = keep, False = discard.
                    * Bias `(batch, 1, sequence_length)` 0 = keep, -10000 = discard.

                If `ndim == 2`: will be interpreted as a mask, then converted into a bias consistent with the format
                above. This bias will be added to the cross-attention scores.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.unet_2d_condition.UNet2DConditionOutput`] instead of a plain
                tuple.

        Returns:
            If `return_dict` is True, an [`~models.transformer_2d.Transformer2DModelOutput`] is returned, otherwise a
            `tuple` where the first element is the sample tensor.
        """
        # ensure attention_mask is a bias, and give it a singleton query_tokens dimension.
        #   we may have done this conversion already, e.g. if we came here via UNet2DConditionModel#forward.
        #   we can tell by counting dims; if ndim == 2: it's a mask rather than a bias.
        # expects mask of shape:
        #   [batch, key_tokens]
        # adds singleton query_tokens dimension:
        #   [batch,                    1, key_tokens]
        # this helps to broadcast it as a bias over attention scores, which will be in one of the following shapes:
        #   [batch,  query_tokens, heads, key_tokens] (e.g. paddle sdp or ppxformers attn)
        #   [batch,  heads, query_tokens, key_tokens] (e.g. classic attn)
        # pure fp16
        cal_type(cache_dic, current)
        hidden_states = hidden_states.cast(self.dtype)
        if attention_mask is not None and attention_mask.ndim == 2:
            # assume that mask is expressed as:
            #   (1 = keep,      0 = discard)
            # convert mask into a bias that can be added to attention scores:
            #       (keep = +0,     discard = -10000.0)
            attention_mask = (1 - attention_mask.cast(hidden_states.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)

        # convert encoder_attention_mask to a bias the same way we do for attention_mask
        if encoder_attention_mask is not None and encoder_attention_mask.ndim == 2:
            encoder_attention_mask = (1 - encoder_attention_mask.cast(hidden_states.dtype)) * -10000.0
            encoder_attention_mask = encoder_attention_mask.unsqueeze(1)

        # Retrieve lora scale.
        lora_scale = cross_attention_kwargs.get("scale", 1.0) if cross_attention_kwargs is not None else 1.0

        # 1. Input
        if self.is_input_continuous:
            if self.data_format == "NCHW":
                # (NOTE,zhoukangkang paddle inference ) make hit paddle inference elementwiseadd_transpose_pass.
                batch, _, height, width = hidden_states.shape
            else:
                batch, height, width, _ = hidden_states.shape
            residual = hidden_states
            shape = paddle.shape(hidden_states)
            hidden_states = self.norm(hidden_states)
            if not self.use_linear_projection:
                hidden_states = (
                    self.proj_in(hidden_states, scale=lora_scale)
                    if not USE_PEFT_BACKEND
                    else self.proj_in(hidden_states)
                )
                if self.data_format == "NCHW":
                    hidden_states = hidden_states.transpose([0, 2, 3, 1]).flatten(1, 2)
                else:
                    hidden_states = hidden_states.flatten(1, 2)
            else:
                if self.data_format == "NCHW":
                    hidden_states = hidden_states.transpose([0, 2, 3, 1]).flatten(1, 2)
                else:
                    hidden_states = hidden_states.flatten(1, 2)
                hidden_states = (
                    self.proj_in(hidden_states, scale=lora_scale)
                    if not USE_PEFT_BACKEND
                    else self.proj_in(hidden_states)
                )

        elif self.is_input_vectorized:
            hidden_states = self.latent_image_embedding(hidden_states.cast("int64"))  # NEW ADD
        elif self.is_input_patches:
            height, width = hidden_states.shape[-2] // self.patch_size, hidden_states.shape[-1] // self.patch_size
            hidden_states = self.pos_embed(hidden_states)

            if self.adaln_single is not None:
                if self.use_additional_conditions and added_cond_kwargs is None:
                    raise ValueError(
                        "`added_cond_kwargs` cannot be None when using additional conditions for `adaln_single`."
                    )
                batch_size = hidden_states.shape[0]
                timestep, embedded_timestep = self.adaln_single(
                    timestep, added_cond_kwargs, batch_size=batch_size, hidden_dtype=hidden_states.dtype
                )

        # 2. Blocks
        if self.caption_projection is not None:
            batch_size = hidden_states.shape[0]
            encoder_hidden_states = self.caption_projection(encoder_hidden_states)
            encoder_hidden_states = encoder_hidden_states.reshape([batch_size, -1, hidden_states.shape[-1]])

        if self.inference_optimize:
            hidden_states = self.simplified_facebookdit(hidden_states, timestep, class_labels)
        else:
            for idx, block in enumerate(self.transformer_blocks):
                current['layer'] = idx
                if self.gradient_checkpointing and not hidden_states.stop_gradient and not use_old_recompute():

                    def create_custom_forward(module, return_dict=None):
                        def custom_forward(*inputs):
                            if return_dict is not None:
                                return module(*inputs, return_dict=return_dict)
                            else:
                                return module(*inputs)

                        return custom_forward

                    ckpt_kwargs = {} if recompute_use_reentrant() else {"use_reentrant": False}
                    hidden_states = recompute(
                        create_custom_forward(block),
                        hidden_states,
                        attention_mask,
                        encoder_hidden_states,
                        encoder_attention_mask,
                        timestep,
                        cross_attention_kwargs,
                        class_labels,
                        **ckpt_kwargs,
                    )
                else:
                    hidden_states = block(
                        hidden_states,
                        attention_mask=attention_mask,
                        encoder_hidden_states=encoder_hidden_states,
                        encoder_attention_mask=encoder_attention_mask,
                        timestep=timestep,
                        cross_attention_kwargs=cross_attention_kwargs,
                        class_labels=class_labels,
                        cache_dic=cache_dic,
                        current= current,
                        
                    )

        # 3. Output
        if self.is_input_continuous:
            if not self.use_linear_projection:
                if self.data_format == "NCHW":
                    hidden_states = hidden_states.reshape([shape[0], shape[2], shape[3], self.inner_dim])
                else:
                    hidden_states = hidden_states.reshape([shape[0], shape[1], shape[2], self.inner_dim])
                if self.data_format == "NCHW":
                    hidden_states = hidden_states.transpose([0, 3, 1, 2])
                hidden_states = (
                    self.proj_out(hidden_states, scale=lora_scale)
                    if not USE_PEFT_BACKEND
                    else self.proj_out(hidden_states)
                )
            else:
                hidden_states = (
                    self.proj_out(hidden_states, scale=lora_scale)
                    if not USE_PEFT_BACKEND
                    else self.proj_out(hidden_states)
                )
                if self.data_format == "NCHW":
                    hidden_states = hidden_states.reshape([shape[0], shape[2], shape[3], self.inner_dim])
                else:
                    hidden_states = hidden_states.reshape([shape[0], shape[1], shape[2], self.inner_dim])
                if self.data_format == "NCHW":
                    hidden_states = hidden_states.transpose([0, 3, 1, 2])

            output = hidden_states + residual
        elif self.is_input_vectorized:
            hidden_states = self.norm_out(hidden_states)
            logits = self.out(hidden_states)
            # (batch, self.num_vector_embeds - 1, self.num_latent_pixels)
            logits = logits.transpose([0, 2, 1])

            # log(p(x_0))
            output = F.log_softmax(logits.cast("float64"), axis=1).cast("float32")

        if self.is_input_patches:
            if self.config.norm_type != "ada_norm_single":
                conditioning = self.transformer_blocks[0].norm1.emb(
                    timestep, class_labels, hidden_dtype=hidden_states.dtype
                )
                shift, scale = self.proj_out_1(F.silu(conditioning)).chunk(2, axis=1)
                hidden_states = self.norm_out(hidden_states) * (1 + scale[:, None]) + shift[:, None]
                hidden_states = self.proj_out_2(hidden_states)
            elif self.config.norm_type == "ada_norm_single":
                shift, scale = (self.scale_shift_table[None] + embedded_timestep[:, None]).chunk(2, axis=1)
                hidden_states = self.norm_out(hidden_states)
                # Modulation
                hidden_states = hidden_states * (1 + scale) + shift
                hidden_states = self.proj_out(hidden_states)
                hidden_states = hidden_states.squeeze(1)

            # unpatchify
            if self.adaln_single is None:
                height = width = int(hidden_states.shape[1] ** 0.5)
            hidden_states = hidden_states.reshape(
                shape=(-1, height, width, self.patch_size, self.patch_size, self.out_channels)
            )
            # hidden_states = paddle.einsum("nhwpqc->nchpwq", hidden_states)
            hidden_states = hidden_states.transpose([0, 5, 1, 3, 2, 4])
            output = hidden_states.reshape(
                shape=(-1, self.out_channels, height * self.patch_size, width * self.patch_size)
            )

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)