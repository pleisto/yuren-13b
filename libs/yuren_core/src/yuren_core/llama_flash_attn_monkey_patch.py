"""
 Copyright 2023 Pleisto Inc

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      https://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.

 FlashAttention 2.0 with LLaMA 2
 """


from typing import Optional, Tuple

import torch
import transformers
from einops import rearrange
from torch import nn
from transformers.models.llama.modeling_llama import LlamaConfig, LlamaRotaryEmbedding, apply_rotary_pos_emb

flash_attn_installed = False

# pyright: reportMissingImports=false
try:
    from flash_attn.bert_padding import pad_input, unpad_input
    from flash_attn.flash_attn_interface import flash_attn_varlen_qkvpacked_func
except ImportError:
    print("flash_attn2 not installed, this is optional but recommended for faster training.")
    flash_attn_installed = False


class LlamaAttention(transformers.models.llama.modeling_llama.LlamaAttention):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        config: LlamaConfig,
    ):
        super().__init__()
        hidden_size = config.hidden_size
        num_heads = config.num_attention_heads
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = self.hidden_size // num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings

        print("FlashAttention2.0 with LLaMA2 is activated.")

        if (self.head_dim * num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {num_heads})."
            )
        self.q_proj = nn.Linear(
            hidden_size,
            num_heads * self.head_dim,
            bias=False,
        )
        self.k_proj = nn.Linear(
            hidden_size,
            num_heads * self.head_dim,
            bias=False,
        )
        self.v_proj = nn.Linear(
            hidden_size,
            num_heads * self.head_dim,
            bias=False,
        )
        self.o_proj = nn.Linear(
            num_heads * self.head_dim,
            hidden_size,
            bias=False,
        )
        self.rotary_emb = LlamaRotaryEmbedding(self.head_dim)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel

        attention_mask: [bsz, q_len]
        """
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        # [bsz, q_len, nh, hd]
        # [bsz, nh, q_len, hd]

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]

        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        # [bsz, nh, t, hd]
        assert not output_attentions, "output_attentions is not supported"
        assert not use_cache, "use_cache is not supported"
        assert past_key_value is None, "past_key_value is not supported"

        # Flash attention codes from
        # https://github.com/HazyResearch/flash-attention/blob/main/flash_attn/flash_attention.py

        # transform the data into the format required by flash attention
        qkv = torch.stack([query_states, key_states, value_states], dim=2)  # [bsz, nh, 3, q_len, hd]
        qkv = qkv.transpose(1, 3)  # [bsz, q_len, 3, nh, hd]
        # We have disabled _prepare_decoder_attention_mask in LlamaModel
        # the attention_mask should be the same as the key_padding_mask
        key_padding_mask = attention_mask

        if key_padding_mask is None:
            qkv = rearrange(qkv, "b s ... -> (b s) ...")
            max_s = q_len
            cu_q_lens = torch.arange(0, (bsz + 1) * q_len, step=q_len, dtype=torch.int32, device=qkv.device)
            output = flash_attn_varlen_qkvpacked_func(qkv, cu_q_lens, max_s, 0.0, softmax_scale=None, causal=True)
            output = rearrange(output, "(b s) ... -> b s ...", b=bsz)
        else:
            nheads = qkv.shape[-2]
            x = rearrange(qkv, "b s three h d -> b s (three h d)")
            x_unpad, indices, cu_q_lens, max_s = unpad_input(x, key_padding_mask)
            x_unpad = rearrange(x_unpad, "nnz (three h d) -> nnz three h d", three=3, h=nheads)
            output_unpad = flash_attn_varlen_qkvpacked_func(
                x_unpad, cu_q_lens, max_s, 0.0, softmax_scale=None, causal=True
            )
            output = rearrange(
                pad_input(rearrange(output_unpad, "nnz h d -> nnz (h d)"), indices, bsz, q_len),
                "b s (h d) -> b s h d",
                h=nheads,
            )
        return self.o_proj(rearrange(output, "b s h d -> b s (h d)")), None, None


# Copied from transformers.models.bart.modeling_bart.BartDecoder._prepare_decoder_attention_mask
def _prepare_decoder_attention_mask(self, attention_mask, _input_shape, _inputs_embeds, _past_key_values_length):
    # [bsz, seq_len]
    return attention_mask


def replace_llama_attn_with_flash_attn():
    if not flash_attn_installed:
        return
    transformers.models.llama.modeling_llama.LlamaModel._prepare_decoder_attention_mask = (
        _prepare_decoder_attention_mask
    )
    transformers.models.llama.modeling_llama.LlamaAttention = LlamaAttention
