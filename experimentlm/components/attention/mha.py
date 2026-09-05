"""Multi-head causal self-attention."""
from __future__ import annotations

import torch
from torch import nn
from ...parameter_policy import set_optimizer_policy
from ...registry import ATTENTIONS
from ..base import Attention, PositionEncoding
from ..norms import RMSNorm
from .common import _scaled_dot_product_attention

@ATTENTIONS.register
class MultiHeadCausalSelfAttention(Attention):
    def __init__(self, d_model: int, n_heads: int, head_dim: int | None = None,
                 bias: bool = False, dropout: float = 0.0,
                 qk_norm: bool = False, qk_norm_eps: float = 1e-6,
                 head_scale: bool = False):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim or d_model // n_heads
        if n_heads <= 0 or self.head_dim <= 0 or n_heads * self.head_dim != d_model:
            raise ValueError("attention requires n_heads * head_dim == d_model")
        if not 0 <= dropout < 1: raise ValueError("attention dropout must be in [0, 1)")
        self.dropout = dropout
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=bias)
        self.out = nn.Linear(d_model, d_model, bias=bias)
        self.out.residual_projection = True
        self.q_norm = RMSNorm(self.head_dim, qk_norm_eps) if qk_norm else None
        self.k_norm = RMSNorm(self.head_dim, qk_norm_eps) if qk_norm else None
        if head_scale:
            self.head_scale = nn.Parameter(torch.ones(self.n_heads))
            set_optimizer_policy(self.head_scale, "adamw", "attention_head_scale")

    def _normalize_qk(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.q_norm is not None:
            q, k = self.q_norm(q), self.k_norm(k)
        return q, k

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor | None,
                position_encoding: PositionEncoding) -> torch.Tensor:
        batch, length, _ = x.shape
        q, k, v = self.qkv(x).view(batch, length, 3, self.n_heads, self.head_dim).unbind(2)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        q, k = self._normalize_qk(q, k)
        q, k = position_encoding.apply(q, k)
        if hasattr(self, "head_scale"):
            q = q * self.head_scale.to(dtype=q.dtype)[None, :, None, None]
        dropout = self.dropout if self.training else 0.0
        if attention_mask is None:
            y = _scaled_dot_product_attention(q, k, v, dropout_p=dropout, is_causal=True)
        else:
            causal = torch.ones((length, length), device=x.device, dtype=torch.bool).tril()
            valid_keys = attention_mask.to(device=x.device, dtype=torch.bool)[:, None, None, :]
            y = _scaled_dot_product_attention(
                q, k, v, attn_mask=causal[None, None] & valid_keys,
                dropout_p=dropout, is_causal=False,
            )
        return self.out(y.transpose(1, 2).contiguous().view(batch, length, -1))

    def forward_cached(self, x: torch.Tensor, position_encoding: PositionEncoding,
                       past: tuple[torch.Tensor, torch.Tensor] | None = None):
        batch, length, _ = x.shape
        q, k, v = self.qkv(x).view(batch, length, 3, self.n_heads, self.head_dim).unbind(2)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        offset = 0 if past is None else past[0].size(-2)
        q, k = self._normalize_qk(q, k)
        q, k = position_encoding.apply(q, k, offset)
        if hasattr(self, "head_scale"):
            q = q * self.head_scale.to(dtype=q.dtype)[None, :, None, None]
        if past is not None:
            past_length = past[0].size(-2)
            k = torch.cat((past[0], k), dim=-2)
            v = torch.cat((past[1], v), dim=-2)
            # SDPA's is_causal flag assumes Q and K start at the same
            # position.  With a multi-token continuation, make the offset
            # explicit so query i cannot see continuation tokens > i.
            causal = torch.ones((length, k.size(-2)), device=x.device, dtype=torch.bool).tril(
                diagonal=past_length,
            )
            y = _scaled_dot_product_attention(q, k, v, attn_mask=causal, is_causal=False)
        else:
            y = _scaled_dot_product_attention(q, k, v, is_causal=True)
        output = self.out(y.transpose(1, 2).contiguous().view(batch, length, -1))
        return output, (k, v)

__all__ = ["MultiHeadCausalSelfAttention"]

