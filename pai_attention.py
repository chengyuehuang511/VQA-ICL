import math
import types
from typing import Callable, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers.models.mistral.modeling_mistral import apply_rotary_pos_emb, repeat_kv, MistralAttention

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache, SlidingWindowCache, StaticCache
from transformers.generation import GenerationMixin
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput,
)
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.utils import (
    LossKwargs,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from transformers.utils.deprecation import deprecate_kwarg
logger = logging.get_logger(__name__)


def eager_attention_new_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    # print("attention_mask", attention_mask)
    if attention_mask is not None:
        assert attention_mask.shape[-1] == key_states.shape[-2], (
            f"Attention mask has incorrect dimensions. Expected {key_states.shape[-2]}, got {attention_mask.shape[-1]}."
        )
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    # attn_weights_raw = attn_weights.clone()

    ### PAI's modification
    if hasattr(module, "use_attn"):
        use_attn = module.use_attn
        img_start_idx = module.img_start_idx
        img_end_idx = module.img_end_idx
    else:
        use_attn = False

    if hasattr(module, "use_cfg"):
        use_cfg = module.use_cfg
    else:
        use_cfg = False

    # print("pai_use_attn", use_attn)
    # print("pai_use_cfg", use_cfg)
    if use_attn and not use_cfg:  # (bs, 1, seq_len, seq_len)
        # Compute left padding per sample
        # print("attention_mask", attention_mask[:, :, -1, :])  # (bs, 1, 1, seq_len)
        if attention_mask is None:
            left_padding_counts = torch.zeros(query.shape[0], device=query.device)
            # print("pai_left_padding_counts", left_padding_counts)
        else:
            left_padding_counts = (attention_mask[:, :, -1, :].squeeze(1).squeeze(1) != 0).sum(dim=1)  # (bs,)
            # print("pai_left_padding_counts", left_padding_counts)
        # print("pai_left_padding_counts.shape", left_padding_counts.shape)

        # Compute adjusted indices
        adjusted_start_idx = (img_start_idx + left_padding_counts).cpu().numpy()  # (bs,)
        adjusted_start_idx = adjusted_start_idx.astype(int)
        adjusted_end_idx = (img_end_idx + left_padding_counts).cpu().numpy()      # (bs,)
        adjusted_end_idx = adjusted_end_idx.astype(int)
        # print("pai_adjusted_start_idx", adjusted_start_idx)
        # print("pai_adjusted_end_idx", adjusted_end_idx)
        # print("pai_adjusted_start_idx.shape", adjusted_start_idx.shape)
        # print("pai_adjusted_end_idx.shape", adjusted_end_idx.shape)

        # for each sample, there are left padded tokens so img_start_idx should add the number of left padded tokens
        for i in range(len(attn_weights)):
            attn_weights[i, :, -1, adjusted_start_idx[i]:adjusted_end_idx[i]] = (
                attn_weights[i, :, -1, adjusted_start_idx[i]:adjusted_end_idx[i]].abs() * module.alpha
                + attn_weights[i, :, -1, adjusted_start_idx[i]:adjusted_end_idx[i]]
            )
        # attn_weights[:, :, -1, img_start_idx:img_end_idx] = (
        #     attn_weights[:, :, -1, img_start_idx:img_end_idx].abs() * module.alpha
        #     + attn_weights[:, :, -1, img_start_idx:img_end_idx]
        # )
    ### PAI's modification

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    # return attn_output, (attn_weights, attn_weights_raw)
    return attn_output, attn_weights

def mistral_new_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    past_key_value: Optional[Cache] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs: Unpack[FlashAttentionKwargs],
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_value is not None:
        # sin and cos are specific to RoPE models; cache_position needed for the static cache
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    attention_interface: Callable = eager_attention_new_forward
    # print("self.config._attn_implementation", self.config._attn_implementation)
    # if self.config._attn_implementation != "eager":
    #     if self.config._attn_implementation == "sdpa" and kwargs.get("output_attentions", False):
    #         logger.warning_once(
    #             "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
    #             'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
    #         )
    #     else:
    #         attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

    attn_output, attn_weights = attention_interface(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        sliding_window=getattr(self.config, "sliding_window", None),  # main diff with Llama
        **kwargs,
    )

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights

def mistral_modify(model, start_layer, end_layer, use_attn, alpha, use_cfg,
                 img_start_idx, img_end_idx):
    for i in range(start_layer, end_layer):
        model.layers[i].self_attn.use_attn = use_attn
        model.layers[i].self_attn.alpha = alpha
        model.layers[i].self_attn.use_cfg = use_cfg
        model.layers[i].self_attn.img_start_idx = img_start_idx
        model.layers[i].self_attn.img_end_idx = img_end_idx
        model.layers[i].self_attn.forward = types.MethodType(mistral_new_forward, model.layers[i].self_attn)

def mistral_recover(model, start_layer, end_layer):
    for i in range(start_layer, end_layer):
        model.layers[i].self_attn.forward = types.MethodType(MistralAttention.forward, model.layers[i].self_attn)
