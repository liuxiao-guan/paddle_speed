# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import math
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.distributed.fleet.utils import recompute
from paddle.nn import CrossEntropyLoss
from paddlenlp.transformers.chatglm.configuration import ChatGLMConfig
from paddlenlp.transformers.chatglm.modeling import ChatGLMForCausalLM
from paddlenlp.transformers.model_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPooling,
    BaseModelOutputWithPoolingAndCrossAttentions,
    ModelOutput,
)
from paddlenlp.transformers.model_utils import (
    apply_chunking_to_forward,
    find_pruneable_heads_and_indices,
    prune_linear_layer,
)

from paddlemix.models.model_utils import MixPretrainedModel

from ...activations import ACT2FN
from ...utils.initializer import normal_, ones_, zeros_
from ...utils.log import logger
from .configuration import (
    VisualGLMConfig,
    VisualGLMQFormerConfig,
    VisualGLMVisionConfig,
)

VisualGLM_PRETRAINED_MODEL_ARCHIVE_LIST = []

__all__ = [
    "VisualGLMModel",
    "VisualGLMPretrainedModel",
    "VisualGLMQFormerModel",
    "VisualGLMVisionModel",
    "VisualGLMForConditionalGeneration",
]


def Parameter(tensor, dtype="float16"):
    tensor = paddle.cast(tensor, dtype)
    return paddle.create_parameter(
        tensor.shape,
        dtype=tensor.dtype,
        default_initializer=nn.initializer.Assign(tensor),
    )


@dataclass
class VisualGLMForConditionalGenerationModelOutput(ModelOutput):
    """
    Class defining the outputs of [`VisualGLMForConditionalGeneration`].
    Args:
        loss (`paddle.Tensor`, *optional*, returned when `labels` is provided, `paddle.Tensor` of shape `(1,)`):
            Language modeling loss from the language model.
        logits (`paddle.Tensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head of the language model.
        vision_outputs (`BaseModelOutputWithPooling`):
            Outputs of the vision encoder.
        qformer_outputs (`BaseModelOutputWithPoolingAndCrossAttentions`):
            Outputs of the Q-Former (Querying Transformer).
        language_model_outputs (`CausalLMOutputWithPast` or `Seq2SeqLMOutput`):
            Outputs of the language model.
    """

    loss: Optional[Tuple[paddle.Tensor]] = None
    logits: Optional[Tuple[paddle.Tensor]] = None
    vision_outputs: Optional[paddle.Tensor] = None
    qformer_outputs: Optional[Tuple[paddle.Tensor]] = None
    language_model_outputs: Optional[Tuple[paddle.Tensor]] = None

    def to_tuple(self) -> Tuple[Any]:
        return tuple(
            self[k]
            if k not in ["vision_outputs", "qformer_outputs", "language_model_outputs"]
            else getattr(self, k).to_tuple()
            for k in self.keys()
        )


class VisualGLMPretrainedModel(MixPretrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = VisualGLMConfig
    base_model_prefix = "visualglm"
    supports_gradient_checkpointing = True
    _keys_to_ignore_on_load_missing = [
        r"position_ids",
    ]

    def _init_weights(self, module):
        """Initialize the weights"""

        def trunc_normal_(tensor, mean=0.0, std=1.0, min=-2, max=2):
            origin_dtype = paddle.get_default_dtype()
            paddle.set_default_dtype("float32")
            with paddle.no_grad():
                normal = paddle.normal(mean=mean, std=std, shape=tensor.shape)
                trunc = paddle.clip(normal, min=min, max=max)
                if origin_dtype != "float32":
                    trunc = trunc.astype(origin_dtype)
                tensor.set_value(trunc)
            paddle.set_default_dtype(origin_dtype)
            return tensor

        factor = self.config.initializer_range
        if isinstance(module, nn.Conv2D) or isinstance(module, nn.Embedding) or isinstance(module, nn.Linear):
            normal_(module.weight, mean=0.0, std=factor)
            if hasattr(module, "bias") and module.bias is not None:
                zeros_(module.bias)

        if isinstance(module, VisualGLMVisionEmbeddings):
            if hasattr(self.config, "vision_config"):
                factor = self.config.vision_config.initializer_range

            trunc_normal_(module.position_embedding)
            trunc_normal_(
                module.class_embedding,
            )

        elif isinstance(module, nn.LayerNorm):
            zeros_(module.bias)
            ones_(module.weight)
        elif isinstance(module, nn.Linear) and module.bias is not None:
            zeros_(module.bias)

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, VisualGLMEncoder):
            module.gradient_checkpointing = value


class VisualGLMVisionEmbeddings(nn.Layer):
    def __init__(self, config: VisualGLMVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size
        self.in_channels = config.num_channels

        self.patch_embedding = nn.Conv2D(
            in_channels=self.in_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches + 1

        self.class_embedding = Parameter(
            paddle.randn([1, 1, self.embed_dim]),
            dtype=self.patch_embedding.weight.dtype,
        )
        self.position_embedding = Parameter(
            paddle.randn([1, self.num_positions, self.embed_dim]),
            dtype=self.patch_embedding.weight.dtype,
        )

    def forward(self, pixel_values: paddle.Tensor) -> paddle.Tensor:
        batch_size = pixel_values.shape[0]
        target_dtype = self.patch_embedding.weight.dtype
        patch_embeds = self.patch_embedding(pixel_values)  # shape = [*, width, grid, grid]
        patch_embeds = patch_embeds.flatten(2).transpose([0, 2, 1])

        class_embeds = self.class_embedding.expand([batch_size, 1, -1]).cast(target_dtype)
        embeddings = paddle.concat([class_embeds, patch_embeds], axis=1)
        embeddings = embeddings + self.position_embedding[:, : embeddings.shape[1], :].cast(target_dtype)
        return embeddings


class VisualGLMAttention(nn.Layer):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )
        self.scale = self.head_dim**-0.5
        self.dropout = nn.Dropout(config.attention_dropout)

        # small tweak here compared to CLIP, no bias here
        self.qkv = nn.Linear(self.embed_dim, 3 * self.embed_dim, bias_attr=False)

        if config.qkv_bias:
            q_bias = Parameter(paddle.zeros([self.embed_dim], dtype=self.qkv.weight.dtype))
            v_bias = Parameter(paddle.zeros([self.embed_dim], dtype=self.qkv.weight.dtype))
        else:
            q_bias = None
            v_bias = None

        if q_bias is not None:
            qkv_bias = paddle.concat((q_bias, paddle.zeros_like(v_bias), v_bias))
            self.qkv.bias = Parameter(qkv_bias, dtype=self.qkv.weight.dtype)

        self.projection = nn.Linear(self.embed_dim, self.embed_dim)

    def _shape(self, tensor: paddle.Tensor, seq_len: int, bsz: int):
        return tensor.reshape([bsz, seq_len, self.num_heads, self.head_dim]).transpose([0, 2, 1, 3])

    def forward(
        self,
        hidden_states: paddle.Tensor,
        head_mask: Optional[paddle.Tensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[paddle.Tensor, Optional[paddle.Tensor], Optional[Tuple[paddle.Tensor]]]:
        """Input shape: Batch x Time x Channel"""
        bsz, tgt_len, embed_dim = hidden_states.shape

        mixed_qkv = self.qkv(hidden_states)

        mixed_qkv = mixed_qkv.reshape([bsz, tgt_len, 3, self.num_heads, embed_dim // self.num_heads]).transpose(
            [2, 0, 3, 1, 4]
        )
        query_states, key_states, value_states = (
            mixed_qkv[0],
            mixed_qkv[1],
            mixed_qkv[2],
        )

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = paddle.matmul(query_states, key_states, transpose_y=True)

        attention_scores = attention_scores * self.scale

        # Normalize the attention scores to probabilities.
        attention_probs = F.softmax(attention_scores, axis=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = paddle.matmul(attention_probs, value_states).transpose([0, 2, 1, 3])

        new_context_layer_shape = context_layer.shape[:-2] + [
            self.embed_dim,
        ]
        context_layer = context_layer.reshape(new_context_layer_shape)

        output = self.projection(context_layer)

        outputs = (output, attention_probs) if output_attentions else (output, None)

        return outputs


class VisualGLMMLP(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.activation_fn = ACT2FN[config.hidden_act]
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: paddle.Tensor) -> paddle.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class VisualGLMEncoderLayer(nn.Layer):
    def __init__(self, config: VisualGLMConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = VisualGLMAttention(config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, epsilon=config.layer_norm_eps)
        self.mlp = VisualGLMMLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, epsilon=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: paddle.Tensor,
        attention_mask: paddle.Tensor,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[paddle.Tensor]:
        """
        Args:
            hidden_states (`paddle.Tensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`paddle.Tensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
                `(config.encoder_attention_heads,)`.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            head_mask=attention_mask,
            output_attentions=output_attentions,
        )
        hidden_states = hidden_states + residual
        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)

        hidden_states = hidden_states + residual

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs


class VisualGLMEncoder(nn.Layer):
    """
    Transformer encoder consisting of `config.num_hidden_layers` self attention layers. Each layer is a
    [`VisualGLMEncoderLayer`].
    Args:
        config (`VisualGLMConfig`):
            The corresponding vision configuration for the `VisualGLMEncoder`.
    """

    def __init__(self, config: VisualGLMConfig):
        super().__init__()
        self.config = config
        self.layers = nn.LayerList([VisualGLMEncoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def forward(
        self,
        inputs_embeds,
        attention_mask: Optional[paddle.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:
        r"""
        Args:
            inputs_embeds (`paddle.Tensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
                This is useful if you want more control over how to convert `input_ids` indices into associated vectors
                than the model's internal embedding lookup matrix.
            attention_mask (`paddle.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
                - 1 for tokens that are **not masked**,

                - 0 for tokens that are **masked**.
                [What are attention masks?](../glossary#attention-mask)
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        hidden_states = inputs_embeds
        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, output_attentions)

                    return custom_forward

                layer_outputs = recompute(
                    create_custom_forward(encoder_layer),
                    hidden_states,
                    attention_mask,
                )
            else:
                layer_outputs = encoder_layer(
                    hidden_states,
                    attention_mask,
                    output_attentions=output_attentions,
                )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=encoder_states,
            attentions=all_attentions,
        )


class VisualGLMVisionModel(VisualGLMPretrainedModel):
    main_input_name = "pixel_values"
    config_class = VisualGLMVisionConfig

    def __init__(self, config: VisualGLMVisionConfig):
        super().__init__(config)
        self.config = config
        embed_dim = config.hidden_size

        self.embeddings = VisualGLMVisionEmbeddings(config)
        self.encoder = VisualGLMEncoder(config)
        self.post_layernorm = nn.LayerNorm(embed_dim, epsilon=config.layer_norm_eps)

    def forward(
        self,
        pixel_values: Optional[paddle.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        Returns:
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        hidden_states = self.embeddings(pixel_values)

        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = encoder_outputs[0]
        last_hidden_state = self.post_layernorm(last_hidden_state)

        pooled_output = last_hidden_state[:, 0, :]
        pooled_output = self.post_layernorm(pooled_output)

        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

    def get_input_embeddings(self):
        return self.embeddings


class VisualGLMQFormerMultiHeadAttention(nn.Layer):
    def __init__(self, config, is_cross_attention=False):
        super().__init__()
        self.config = config
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention heads (%d)"
                % (config.hidden_size, config.num_attention_heads)
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        if is_cross_attention:
            self.key = nn.Linear(config.encoder_hidden_size, self.all_head_size)
            self.value = nn.Linear(config.encoder_hidden_size, self.all_head_size)
        else:
            self.key = nn.Linear(config.hidden_size, self.all_head_size)
            self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)
        self.save_attention = False

    def save_attn_gradients(self, attn_gradients):
        self.attn_gradients = attn_gradients

    def get_attn_gradients(self):
        return self.attn_gradients

    def save_attention_map(self, attention_map):
        self.attention_map = attention_map

    def get_attention_map(self):
        return self.attention_map

    def transpose_for_scores(self, x):
        new_x_shape = x.shape[:-1] + [
            self.num_attention_heads,
            self.attention_head_size,
        ]
        x = x.reshape(new_x_shape)
        return x.transpose([0, 2, 1, 3])

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None
        if is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = paddle.concat([past_key_value[0], key_layer], axis=2)
            value_layer = paddle.concat([past_key_value[1], value_layer], axis=2)
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))

        mixed_query_layer = self.query(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)

        past_key_value = (key_layer, value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = paddle.matmul(query_layer, key_layer, transpose_y=True)

        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            seq_length = hidden_states.shape[1]
            position_ids_l = paddle.arange(seq_length, dtype="int64").reshape([-1, 1])
            position_ids_r = paddle.arange(seq_length, dtype="int64").reshape([1, -1])
            distance = position_ids_l - position_ids_r
            positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
            positional_embedding = positional_embedding.cast(dtype=query_layer.dtype)  # fp16 compatibility

            if self.position_embedding_type == "relative_key":
                relative_position_scores = paddle.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = paddle.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                relative_position_scores_key = paddle.einsum("bhrd,lrd->bhlr", key_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = paddle.cast((attention_scores + attention_mask), attention_scores.dtype)

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(axis=-1)(attention_scores)

        if is_cross_attention and self.save_attention:
            self.save_attention_map(attention_probs)
            attention_probs.register_hook(self.save_attn_gradients)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs_dropped = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs_dropped = attention_probs_dropped * head_mask

        context_layer = paddle.matmul(attention_probs_dropped, value_layer)

        context_layer = context_layer.transpose([0, 2, 1, 3])
        new_context_layer_shape = context_layer.shape[:-2] + [
            self.all_head_size,
        ]
        context_layer = context_layer.reshape(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        outputs = outputs + (past_key_value,)
        return outputs


class VisualGLMQFormerSelfOutput(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, epsilon=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: paddle.Tensor, input_tensor: paddle.Tensor) -> paddle.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class VisualGLMQFormerAttention(nn.Layer):
    def __init__(self, config, is_cross_attention=False):
        super().__init__()
        self.attention = VisualGLMQFormerMultiHeadAttention(config, is_cross_attention)
        self.output = VisualGLMQFormerSelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads,
            self.attention.num_attention_heads,
            self.attention.attention_head_size,
            self.pruned_heads,
        )

        # Prune linear layers
        self.attention.query = prune_linear_layer(self.attention.query, index)
        self.attention.key = prune_linear_layer(self.attention.key, index)
        self.attention.value = prune_linear_layer(self.attention.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, axis=1)

        # Update hyper params and store pruned heads
        self.attention.num_attention_heads = self.attention.num_attention_heads - len(heads)
        self.attention.all_head_size = self.attention.attention_head_size * self.attention.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states: paddle.Tensor,
        attention_mask: Optional[paddle.Tensor] = None,
        head_mask: Optional[paddle.Tensor] = None,
        encoder_hidden_states: Optional[paddle.Tensor] = None,
        encoder_attention_mask: Optional[paddle.Tensor] = None,
        past_key_value: Optional[Tuple[Tuple[paddle.Tensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[paddle.Tensor]:
        self_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


class VisualGLMQFormerIntermediate(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states: paddle.Tensor) -> paddle.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class VisualGLMQFormerOutput(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        # self.LayerNorm = nn.LayerNorm(config.hidden_size, epsilon=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: paddle.Tensor, input_tensor: paddle.Tensor) -> paddle.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = hidden_states + input_tensor
        # hidden_states = self.LayerNorm()
        return hidden_states


class VisualGLMQFormerLayer(nn.Layer):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.input_layernorm = nn.LayerNorm(config.hidden_size, epsilon=config.layer_norm_eps)
        self.attention = VisualGLMQFormerAttention(config)

        self.layer_idx = layer_idx

        if layer_idx % config.cross_attention_frequency == 0:
            self.crossattention = VisualGLMQFormerAttention(config, is_cross_attention=True)
            self.has_cross_attention = True
        else:
            self.has_cross_attention = False

        self.intermediate_query = VisualGLMQFormerIntermediate(config)
        self.output_query = VisualGLMQFormerOutput(config)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
        query_length=0,
    ):
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        hidden_states = self.input_layernorm(hidden_states)
        self_attention_outputs = self.attention(
            hidden_states,  # 1, 32, 768
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:-1]

        present_key_value = self_attention_outputs[-1]

        if query_length > 0:
            query_attention_output = attention_output[:, :query_length, :]

            if self.has_cross_attention:
                if encoder_hidden_states is None:
                    raise ValueError("encoder_hidden_states must be given for cross-attention layers")
                cross_attention_outputs = self.crossattention(
                    query_attention_output,
                    attention_mask,
                    head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    output_attentions=output_attentions,
                )
                query_attention_output = cross_attention_outputs[0]
                # add cross attentions if we output attention weights
                outputs = outputs + cross_attention_outputs[1:-1]

            layer_output = apply_chunking_to_forward(
                self.feed_forward_chunk_query,
                self.chunk_size_feed_forward,
                self.seq_len_dim,
                query_attention_output,
            )

            if attention_output.shape[1] > query_length:
                layer_output_text = apply_chunking_to_forward(
                    self.feed_forward_chunk,
                    self.chunk_size_feed_forward,
                    self.seq_len_dim,
                    attention_output[:, query_length:, :],
                )
                layer_output = paddle.concat([layer_output, layer_output_text], axis=1)
        else:
            layer_output = apply_chunking_to_forward(
                self.feed_forward_chunk,
                self.chunk_size_feed_forward,
                self.seq_len_dim,
                attention_output,
            )
        outputs = (layer_output,) + outputs

        outputs = outputs + (present_key_value,)

        return outputs

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output

    def feed_forward_chunk_query(self, attention_output):
        intermediate_output = self.intermediate_query(attention_output)
        layer_output = self.output_query(intermediate_output, attention_output)
        return layer_output


class VisualGLMQFormerEncoder(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.LayerList(
            [VisualGLMQFormerLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
        query_length=0,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions else None

        next_decoder_cache = () if use_cache else None

        for i in range(self.config.num_hidden_layers):
            layer_module = self.layer[i]
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            if getattr(self.config, "gradient_checkpointing", False) and self.training:
                if use_cache:
                    logger.warn(
                        "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                    )
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, past_key_value, output_attentions, query_length)

                    return custom_forward

                layer_outputs = recompute(
                    create_custom_forward(layer_module),
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                    query_length,
                )

            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if layer_module.has_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )


class VisualGLMQFormerModel(VisualGLMPretrainedModel):
    """
    Querying Transformer (Q-Former), used in VisualGLM.
    """

    def __init__(self, config: VisualGLMQFormerConfig):
        super().__init__(config)
        self.config = config

        self.final_layernorm = nn.LayerNorm(config.hidden_size, epsilon=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.encoder = VisualGLMQFormerEncoder(config)

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def get_extended_attention_mask(
        self,
        attention_mask: paddle.Tensor,
        input_shape: Tuple[int],
        has_query: bool = False,
    ) -> paddle.Tensor:
        """
        Makes broadcastable attention and causal masks so that future and masked tokens are ignored.
        Arguments:
            attention_mask (`paddle.Tensor`):
                Mask with ones indicating tokens to attend to, zeros for tokens to ignore.
            input_shape (`Tuple[int]`):
                The shape of the input to the model.
        Returns:
            `paddle.Tensor` The extended attention mask, with a the same dtype as `attention_mask.dtype`.
        """
        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            # Provided a padding mask of dimensions [batch_size, seq_length]
            # - the model is an encoder, so make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
            extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(
                "Wrong shape for input_ids (shape {}) or attention_mask (shape {})".format(
                    input_shape, attention_mask.shape
                )
            )

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.cast(dtype=self.config.dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def invert_attention_mask(self, encoder_attention_mask: paddle.Tensor) -> paddle.Tensor:
        """
        Invert an attention mask (e.g., switches 0. and 1.).
        Args:
            encoder_attention_mask (`paddle.Tensor`): An attention mask.
        Returns:
            `paddle.Tensor`: The inverted attention mask.
        """
        if encoder_attention_mask.ndim == 3:
            encoder_extended_attention_mask = encoder_attention_mask[:, None, :, :]
        if encoder_attention_mask.ndim == 2:
            encoder_extended_attention_mask = encoder_attention_mask[:, None, None, :]
        # T5 has a mask that can compare sequence ids, we can simulate this here with this transposition
        # Cf. https://github.com/tensorflow/mesh/blob/8d2465e9bc93129b913b5ccc6a59aa97abd96ec6/mesh_tensorflow
        # /transformer/transformer_layers.py#L270
        # encoder_extended_attention_mask = (encoder_extended_attention_mask ==
        # encoder_extended_attention_mask.transpose(-1, -2))
        encoder_extended_attention_mask = encoder_extended_attention_mask.cast(
            dtype=self.config.dtype
        )  # fp16 compatibility
        encoder_extended_attention_mask = (1.0 - encoder_extended_attention_mask) * -1e4

        return encoder_extended_attention_mask

    def get_head_mask(
        self,
        head_mask: Optional[paddle.Tensor],
        num_hidden_layers: int,
        is_attention_chunked: bool = False,
    ) -> paddle.Tensor:
        """
        Prepare the head mask if needed.
        Args:
            head_mask (`paddle.Tensor` with shape `[num_heads]` or `[num_hidden_layers x num_heads]`, *optional*):
                The mask indicating if we should keep the heads or not (1.0 for keep, 0.0 for discard).
            num_hidden_layers (`int`):
                The number of hidden layers in the model.
            is_attention_chunked: (`bool`, *optional*, defaults to `False`):
                Whether or not the attentions scores are computed by chunks or not.
        Returns:
            `paddle.Tensor` with shape `[num_hidden_layers x batch x num_heads x seq_length x seq_length]` or list with
            `[None]` for each layer.
        """
        if head_mask is not None:
            head_mask = self._convert_head_mask_to_5d(head_mask, num_hidden_layers)
            if is_attention_chunked is True:
                head_mask = head_mask.unsqueeze(-1)
        else:
            head_mask = [None] * num_hidden_layers

        return head_mask

    def _convert_head_mask_to_5d(self, head_mask, num_hidden_layers):
        """-> [num_hidden_layers x batch x num_heads x seq_length x seq_length]"""
        if head_mask.ndim == 1:
            head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            head_mask = head_mask.expand([num_hidden_layers, -1, -1, -1, -1])
        elif head_mask.ndim == 2:
            head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)  # We can specify head_mask for each layer
        assert head_mask.ndim == 5, f"head_mask.dim != 5, instead {head_mask.dim()}"
        head_mask = head_mask.cast(dtype=self.config.dtype)  # switch to float if need + fp16 compatibility
        return head_mask

    def forward(
        self,
        query_embeds,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        encoder_hidden_states  (`paddle.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, `optional`):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (`paddle.Tensor` of shape `(batch_size, sequence_length)`, `optional`):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in `[0, 1]`:
            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        past_key_values (`tuple(tuple(paddle.Tensor))` of length `config.n_layers` with each tuple having 4 tensors of:
            shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`): Contains precomputed key and
            value hidden states of the attention blocks. Can be used to speed up decoding. If `past_key_values` are
            used, the user can optionally input only the last `decoder_input_ids` (those that don't have their past key
            value states given to this model) of shape `(batch_size, 1)` instead of all `decoder_input_ids` of shape
            `(batch_size, sequence_length)`.
        use_cache (`bool`, `optional`):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # past_key_values_length
        past_key_values_length = (
            past_key_values[0][0].shape[2] - self.config.query_length if past_key_values is not None else 0
        )

        query_length = query_embeds.shape[1] if query_embeds is not None else 0

        embedding_output = self.dropout(query_embeds)

        input_shape = embedding_output.shape[:-1]
        batch_size, seq_length = input_shape

        if attention_mask is None:
            attention_mask = paddle.ones(((batch_size, seq_length + past_key_values_length)))

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if encoder_hidden_states is not None:
            if type(encoder_hidden_states) == list:
                encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states[0].shape
            else:
                (
                    encoder_batch_size,
                    encoder_sequence_length,
                    _,
                ) = encoder_hidden_states.shape
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)

            if type(encoder_attention_mask) == list:
                encoder_extended_attention_mask = [self.invert_attention_mask(mask) for mask in encoder_attention_mask]
            elif encoder_attention_mask is None:
                encoder_attention_mask = paddle.ones(encoder_hidden_shape)
                encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
            else:
                encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            query_length=query_length,
        )
        sequence_output = encoder_outputs[0]
        sequence_output = self.final_layernorm(sequence_output)
        pooled_output = sequence_output[:, 0, :]

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )


class VisualGLMModel(VisualGLMPretrainedModel):
    config_class = VisualGLMConfig
    main_input_name = "pixel_values"

    def __init__(self, config: VisualGLMConfig):
        super().__init__(config)

        self.vision_model = VisualGLMVisionModel(config.vision_config)
        self.query_tokens = Parameter(
            paddle.zeros([1, config.num_query_tokens, config.qformer_config.hidden_size]),
            dtype=self.config.dtype,
        )
        self.qformer = VisualGLMQFormerModel(config.qformer_config)

        self.language_projection = nn.Linear(config.qformer_config.hidden_size, config.text_config.hidden_size)
        self.language_model = ChatGLMForCausalLM(config.text_config)

    def get_input_embeddings(self) -> nn.Layer:
        return self.vision_model.embeddings.patch_embedding

    def get_text_features(
        self,
        input_ids: Optional[paddle.Tensor] = None,
        attention_mask: Optional[paddle.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ):
        r"""
        Returns:
            text_outputs (`CausalLMOutputWithPast`, or `tuple(paddle.Tensor)` if `return_dict=False`):
                The language model outputs. If `return_dict=True`, the output is a [`CausalLMOutputWithPast`] that
                contains the language model logits, the past key values and the hidden states if
                `output_hidden_states=True`.
        Examples:
        ```python
        >>> import paddle
        >>> from paddlenlp.transformers import ChatGLMTokenizer, VisualGLMModel
        >>> tokenizer = ChatGLMTokenizer.from_pretrained("model_name")
        >>> tokenizer.pad_token = tokenizer.eos_token
        >>> model = VisualGLMModel.from_pretrained("model_name")
        >>> model.eval()
        >>> inputs = tokenizer(["a photo of a cat"], padding=True, return_tensors="pd", return_token_type_ids=False)
        >>> text_features = model.get_text_features(**inputs)
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        text_outputs = self.language_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        return text_outputs

    def get_image_features(
        self,
        pixel_values: Optional[paddle.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ):
        r"""
        Returns:
            vision_outputs (`BaseModelOutputWithPooling` or tuple of `paddle.Tensor`):
                The vision model outputs. If `return_dict=True`, the output is a [`BaseModelOutputWithPooling`] that
                contains the image features, the pooled image features and the hidden states if
                `output_hidden_states=True`.
        Examples:
        ```python
        >>> import paddle
        >>> from PIL import Image
        >>> import requests
        >>> from paddlenlp.transformers import MinitGPT4Processor, VisualGLMModel
        >>> processor = MinitGPT4Processor.from_pretrained("model_name")
        >>> model = VisualGLMModel.from_pretrained("model_name")
        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)
        >>> inputs = processor.process_images(images=image, return_tensors="pd")
        >>> image_outputs = model.get_image_features(**inputs)
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        pixel_values = paddle.cast(pixel_values, self.vision_model.embeddings.patch_embedding.weight.dtype)
        vision_outputs = self.vision_model(
            pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        return vision_outputs

    def get_qformer_features(
        self,
        pixel_values: Optional[paddle.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ):
        r"""
        Returns:
            vision_outputs (`BaseModelOutputWithPooling` or tuple of `paddle.Tensor`):
                The vision model outputs. If `return_dict=True`, the output is a [`BaseModelOutputWithPooling`] that
                contains the image features, the pooled image features and the hidden states if
                `output_hidden_states=True`.
        Examples:
        ```python
        >>> import paddle
        >>> from PIL import Image
        >>> import requests
        >>> from paddlenlp.transformers import MinitGPT4Processor, VisualGLMModel
        >>> processor = MinitGPT4Processor.from_pretrained("model_name")
        >>> model = VisualGLMModel.from_pretrained("model_name")
        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)
        >>> inputs = processor.process_images(images=image, return_tensors="pd")
        >>> qformer_outputs = model.get_qformer_features(**inputs)
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # step 1: forward the images through the vision encoder,
        # to get image embeddings of shape (batch_size, seq_len, hidden_size)
        pixel_values = paddle.cast(pixel_values, self.vision_model.embeddings.patch_embedding.weight.dtype)
        vision_outputs = self.vision_model(
            pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        image_embeds = vision_outputs[0]
        image_attention_mask = paddle.ones(image_embeds.shape[:-1], dtype="int64")

        # step 2: forward the query tokens through the QFormer, using the image embeddings for cross-attention
        query_tokens = self.query_tokens.expand([image_embeds.shape[0], -1, -1])
        query_tokens = paddle.cast(query_tokens, self.qformer.layernorm.weight.dtype)
        image_embeds = paddle.cast(image_embeds, self.qformer.layernorm.weight.dtype)
        query_outputs = self.qformer(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        return query_outputs

    def forward(
        self,
        pixel_values: paddle.Tensor,  # processed image
        first_input_ids: paddle.Tensor,
        second_input_ids: paddle.Tensor,
        first_attention_mask: Optional[paddle.Tensor] = None,
        second_attention_mask: Optional[paddle.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        labels: Optional[paddle.Tensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, VisualGLMForConditionalGenerationModelOutput]:
        r"""
        Returns:
        Examples:
        ```python
        >>> from PIL import Image
        >>> import requests
        >>> import paddle
        >>> from paddlenlp.transformers import VisualGLMProcessor, VisualGLMModel
        >>> processor = VisualGLMProcessor.from_pretrained("model_name")
        >>> model = VisualGLMModel.from_pretrained("model_name")
        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)
        >>> text = "describe this image"
        >>> prompt = "###Human: <Img><ImageHere></Img> <TextHere>###Assistant:"
        >>> inputs = processor(images=image, texts=text, prompts=prompt, return_tensors="pd")
        >>> outputs = model(**inputs)
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # step 1: forward the images through the vision encoder,
        # to get image embeddings of shape (batch_size, seq_len, hidden_size)
        vision_outputs = self.vision_model(pixel_values, return_dict=True)
        image_embeds = vision_outputs.last_hidden_state
        image_attention_mask = paddle.ones(image_embeds.shape[:-1], dtype="int64")

        # step 2: forward the query tokens through the QFormer, using the image embeddings for cross-attention
        query_tokens = self.query_tokens.expand([image_embeds.shape[0], -1, -1])
        query_tokens = paddle.cast(query_tokens, self.qformer.layernorm.weight.dtype)
        image_embeds = paddle.cast(image_embeds, self.qformer.layernorm.weight.dtype)
        query_outputs = self.qformer(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_attention_mask,
            return_dict=True,
        )
        query_output = query_outputs.last_hidden_state

        # step 3: use the language model, conditioned on the text and image
        language_model_inputs = self.language_projection(query_output)
        language_model_attention_mask = paddle.ones(language_model_inputs.shape[:-1], dtype="int64")

        first_embeds = self.language_model.chatglm.transformer.word_embeddings(first_input_ids)
        second_embeds = self.language_model.chatglm.word_embeddings(second_input_ids)
        language_model_inputs = paddle.cast(language_model_inputs, dtype=first_embeds.dtype)
        inputs_embeds = paddle.concat([first_embeds, language_model_inputs, second_embeds], axis=1)

        if first_attention_mask is None:
            first_attention_mask = paddle.ones_like(first_embeds.shape[:-1], dtype="int64")
        if second_attention_mask is None:
            second_attention_mask = paddle.ones_like(second_embeds.shape[:-1], dtype="int64")
        attention_mask = paddle.concat(
            [
                first_attention_mask,
                language_model_attention_mask,
                second_attention_mask,
            ],
            axis=1,
        )

        outputs = self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        logits = outputs.logits if return_dict else outputs[0]
        loss = None
        # we compute the loss here since we need to take into account the sequence length of the query embeds
        if labels is not None:
            logits = logits[:, -labels.shape[1] :, :]
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :]
            shift_labels = labels[..., 1:]

            # Flatten the tokens
            loss_fct = CrossEntropyLoss(reduction="mean")

            loss = loss_fct(
                shift_logits.reshape([-1, self.config.text_config.vocab_size]),
                shift_labels.reshape([-1]),
            )

        if not return_dict:
            output = (logits, vision_outputs, query_outputs, outputs)
            return ((loss,) + output) if loss is not None else output

        return VisualGLMForConditionalGenerationModelOutput(
            loss=loss,
            logits=logits,
            vision_outputs=vision_outputs,
            qformer_outputs=query_outputs,
            language_model_outputs=outputs,
        )


class ChatGLMForConditionalGenerationWithImage(ChatGLMForCausalLM):
    def __init__(self, config: ChatGLMConfig):
        super(ChatGLMForConditionalGenerationWithImage, self).__init__(config)
        self.config = config

    def generate_inputs_position_ids(self, input_ids):

        MASK, gMASK = self.config.mask_token_id, self.config.gmask_token_id
        use_gmasks = []
        mask_positions = []
        for seq in input_ids:
            mask_token = gMASK if gMASK in seq else MASK
            use_gmask = mask_token == gMASK
            use_gmasks.append(use_gmask)
            mask_positions.append(paddle.where(seq == mask_token)[0][0])

        position_ids = self.get_position_ids(input_ids, mask_positions=mask_positions, use_gmasks=use_gmasks)
        return position_ids

    def get_masks(self, input_ids):

        batch_size, seq_length = input_ids.shape
        context_lengths = []
        for seq in input_ids:
            context_lengths.append(paddle.where(seq == self.config.bos_token_id)[0][0])
        attention_mask = paddle.tril(paddle.ones([batch_size, seq_length, seq_length]))
        for i, context_length in enumerate(context_lengths):
            attention_mask[i, :, :context_length] = 1
        attention_mask = attention_mask.unsqueeze(1)
        attention_mask = (attention_mask > 0.5).astype("int64")
        return attention_mask

    def prepare_inputs_for_generation(
        self,
        input_ids,
        position_ids=None,
        attention_mask=None,
        past_key_values=None,
        cache=None,
        inputs_embeds=None,
        **kwargs
    ):

        if cache is None and inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        if cache is not None or past_key_values is not None:
            last_token = input_ids[:, -1].unsqueeze(-1)
            attention_mask = attention_mask[:, :, -1:]
            position_ids = position_ids[..., -1:]

            if cache is None:
                cache = past_key_values

            model_inputs.update(
                {
                    "input_ids": last_token,
                    "cache": cache[-1],
                    "position_ids": position_ids,
                    "use_cache": True,
                    "attention_mask": attention_mask,
                }
            )

            return model_inputs
        else:

            model_inputs.update(
                {
                    "cache": cache,
                    "position_ids": position_ids,
                    "use_cache": True,
                    "attention_mask": attention_mask,
                }
            )

            return model_inputs

    def forward(
        self,
        input_ids: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.Tensor] = None,
        attention_mask: Optional[paddle.Tensor] = None,
        cache: Optional[Tuple[paddle.Tensor]] = None,
        inputs_embeds: Optional[paddle.Tensor] = None,
        labels: Optional[paddle.Tensor] = None,
        use_cache: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = super().forward(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            cache=cache,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            return_dict=return_dict,
        )

        return outputs


class VisualGLMForConditionalGeneration(VisualGLMPretrainedModel):
    config_class = VisualGLMConfig
    main_input_name = "pixel_values"

    def __init__(self, config: VisualGLMConfig):
        super().__init__(config)
        self.config = config
        self.vision_model = VisualGLMVisionModel(config.vision_config)
        self.query_tokens = Parameter(
            paddle.zeros([1, config.num_query_tokens, config.qformer_config.hidden_size]),
            dtype=self.config.dtype,
        )
        self.qformer = VisualGLMQFormerModel(config.qformer_config)
        self.language_projection = nn.Linear(config.qformer_config.hidden_size, config.text_config.hidden_size)
        self.language_model = ChatGLMForConditionalGenerationWithImage(config.text_config)

    def get_input_embeddings(self) -> nn.Layer:
        return self.vision_model.embeddings.patch_embedding

    def encode_images(
        self,
        pixel_values: paddle.Tensor,  # processed image
    ):
        # step 1: forward the images through the vision encoder,
        # to get image embeddings of shape (batch_size, seq_len, hidden_size)
        pixel_values = paddle.cast(pixel_values, self.vision_model.embeddings.patch_embedding.weight.dtype)
        vision_outputs = self.vision_model(pixel_values, return_dict=True)
        image_embeds = vision_outputs.last_hidden_state
        image_attention_mask = paddle.ones(image_embeds.shape[:-1], dtype="int64")

        # step 2: forward the query tokens through the QFormer, using the image embeddings for cross-attention
        query_tokens = self.query_tokens.expand([image_embeds.shape[0], -1, -1])
        query_tokens = paddle.cast(query_tokens, self.qformer.final_layernorm.weight.dtype)
        image_embeds = paddle.cast(image_embeds, self.qformer.final_layernorm.weight.dtype)
        query_outputs = self.qformer(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_attention_mask,
            return_dict=True,
        )
        query_output = query_outputs.last_hidden_state

        # step 3: mapping query_output into language_model space
        language_model_inputs = self.language_projection(query_output)

        return language_model_inputs

    def forward(
        self,
        input_ids: Optional[paddle.Tensor] = None,
        labels: Optional[paddle.Tensor] = None,
        images: Optional[paddle.Tensor] = None,
        return_dict: Optional[bool] = None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        attention_mask = self.language_model.get_masks(input_ids)
        position_ids = self.language_model.generate_inputs_position_ids(input_ids)
        inputs_embeds_batch = None
        if images is not None:
            bos_pos = [
                paddle.where(input_ids[i] == self.config.vision_config["image_start_id"])[0]
                for i in range(input_ids.shape[0])
            ]

            image_features = self.encode_images(images)
            dtype = image_features.dtype
            inputs_embeds_batch = []
            image_index = 0
            for i, pos in enumerate(bos_pos):
                inputs_embeds = []

                for j in range(pos.shape[0]):
                    a = pos[j]
                    b = pos[j + 1] if j + 1 < pos.shape[0] else input_ids.shape[-1]
                    pre_ids, pad_ids, post_ids = paddle.split(input_ids[i][a:b], num_or_sections=[1, 32, -1], axis=0)
                    pre_txt_emb = self.language_model.chatglm.transformer.word_embeddings(pre_ids)
                    post_txt_emb = self.language_model.chatglm.transformer.word_embeddings(post_ids)
                    inputs_embed = paddle.concat([pre_txt_emb, image_features[image_index], post_txt_emb], axis=0)
                    inputs_embeds.append(inputs_embed)
                image_index += pos.shape[0]
                inputs_embeds = paddle.concat(inputs_embeds, axis=0)
                inputs_embeds_batch.append(inputs_embeds)

            inputs_embeds_batch = paddle.to_tensor(inputs_embeds_batch, dtype=dtype)

        outputs = self.language_model.forward(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds_batch,
            labels=labels,
            return_dict=return_dict,
        )

        return outputs

    @paddle.no_grad()
    def generate(
        self,
        pixel_values: paddle.Tensor,
        input_ids: paddle.Tensor,
        pre_image_length: int,
        attention_mask: Optional[paddle.Tensor] = None,
        **generate_kwargs,
    ) -> paddle.Tensor:
        """
        Overrides `generate` function to be able to use the model as a conditional generator.
        Args:
            pixel_values (`paddle.Tensor` of shape (batch_size, num_channels, height, width)):
                Input images to be processed.
            input_ids (`paddle.Tensor` of shape (batch_size, sequence_length), *optional*):
                The sequence used as a prompt for the generation.
            attention_mask (`paddle.Tensor` of shape (batch_size, sequence_length), *optional*):
                Mask to avoid performing attention on padding token indices
        Returns:
            captions (list): A list of strings of length batch_size * num_captions.

        Examples:
        ```python
        >>> from PIL import Image
        >>> import requests
        >>> import paddle
        >>> from paddlenlp.transformers import VisualGLMProcessor, VisualGLMForConditionalGeneration
        >>> processor = VisualGLMProcessor.from_pretrained("model_name")
        >>> model = VisualGLMForConditionalGeneration.from_pretrained("model_name")
        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)
        >>> text = "describe this image"
        >>> prompt = "###Human: <Img><ImageHere></Img> <TextHere>###Assistant:"
        >>> inputs = processor(images=image, texts=text, prompts=prompt, return_tensors="pd")
        >>> generated_ids, scores= model.generate(**inputs)
        >>> generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        """

        image_features = self.encode_images(pixel_values)
        attention_mask = self.language_model.get_masks(input_ids)
        position_ids = self.language_model.generate_inputs_position_ids(input_ids)
        if image_features is not None:
            pre_ids, pad_ids, post_ids = paddle.split(input_ids, num_or_sections=[pre_image_length, 32, -1], axis=1)
            pre_txt_emb = self.language_model.chatglm.transformer.word_embeddings(pre_ids)
            post_txt_emb = self.language_model.chatglm.transformer.word_embeddings(post_ids)
            inputs_embeds = paddle.concat([pre_txt_emb, image_features, post_txt_emb], axis=1)

        outputs = self.language_model.generate(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            attention_mask=attention_mask,
            **generate_kwargs,
        )

        return outputs
