"""Grouped-query causal self-attention."""
from __future__ import annotations

import torch
from torch import nn
from ...registry import ATTENTIONS
from ..base import Attention, PositionEncoding
from ..norms import RMSNorm
from .common import _scaled_dot_product_attention

@ATTENTIONS.register
class GroupedQueryCausalSelfAttention(Attention):
    """Causal attention with independent query heads and shared KV heads."""
    def __init__(self, d_model: int, n_q_heads: int, n_kv_heads: int,
                 head_dim: int = 64, bias: bool = False, dropout: float = 0.0,
                 qk_norm: bool = False, qk_norm_eps: float = 1e-6):
        super().__init__()
        if n_q_heads <= 0 or n_kv_heads <= 0 or head_dim <= 0:
            raise ValueError("GQA head counts and head_dim must be positive")
        if n_q_heads % n_kv_heads != 0:
            raise ValueError("n_q_heads must be divisible by n_kv_heads")
        if n_q_heads * head_dim != d_model:
            raise ValueError("attention requires n_q_heads * head_dim == d_model")
        if not 0 <= dropout < 1:
            raise ValueError("attention dropout must be in [0, 1)")
        self.n_q_heads, self.n_kv_heads, self.head_dim = n_q_heads, n_kv_heads, head_dim
        self.dropout = dropout
        # PyTorch's ROCm fused SDPA kernels in this environment require equal
        # Q/KV head counts. Native enable_gqa works there, but dispatches to
        # the math kernel; use the fused path with explicit KV expansion.
        self._native_gqa = torch.version.hip is None
        # Pack Q/K/V into one GEMM.  Separate Q/K/V linears launch three
        # small kernels at this model size and erase much of GQA's benefit.
        self.qkv = nn.Linear(d_model, (n_q_heads + 2 * n_kv_heads) * head_dim, bias=bias)
        self.out = nn.Linear(d_model, d_model, bias=bias)
        self.out.residual_projection = True
        self.q_norm = RMSNorm(head_dim, qk_norm_eps) if qk_norm else None
        self.k_norm = RMSNorm(head_dim, qk_norm_eps) if qk_norm else None

    def _project(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, length, _ = x.shape
        q_size = self.n_q_heads * self.head_dim
        kv_size = self.n_kv_heads * self.head_dim
        q, k, v = self.qkv(x).split((q_size, kv_size, kv_size), dim=-1)
        q = q.view(batch, length, self.n_q_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, length, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, length, self.n_kv_heads, self.head_dim).transpose(1, 2)
        return q, k, v

    def _normalize_qk(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.q_norm is not None:
            q, k = self.q_norm(q), self.k_norm(k)
        return q, k

    def _sdpa(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
              *, is_causal: bool, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        dropout = self.dropout if self.training else 0.0
        kwargs = {"dropout_p": dropout, "is_causal": is_causal}
        if attention_mask is not None:
            kwargs["attn_mask"] = attention_mask
            kwargs["is_causal"] = False
        if self._native_gqa:
            return _scaled_dot_product_attention(q, k, v, enable_gqa=True, **kwargs)
        repeats = self.n_q_heads // self.n_kv_heads
        return _scaled_dot_product_attention(
            q, k.repeat_interleave(repeats, dim=1), v.repeat_interleave(repeats, dim=1), **kwargs,
        )

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor | None,
                position_encoding: PositionEncoding) -> torch.Tensor:
        batch, length, _ = x.shape
        q, k, v = self._project(x)
        q, k = self._normalize_qk(q, k)
        q, k = position_encoding.apply(q, k)
        if attention_mask is None:
            y = self._sdpa(q, k, v, is_causal=True)
        else:
            causal = torch.ones((length, length), device=x.device, dtype=torch.bool).tril()
            valid_keys = attention_mask.to(device=x.device, dtype=torch.bool)[:, None, None, :]
            y = self._sdpa(q, k, v, is_causal=False, attention_mask=causal[None, None] & valid_keys)
        return self.out(y.transpose(1, 2).contiguous().view(batch, length, -1))

    def forward_cached(self, x: torch.Tensor, position_encoding: PositionEncoding,
                       past: tuple[torch.Tensor, torch.Tensor] | None = None):
        batch, length, _ = x.shape
        q, k, v = self._project(x)
        offset = 0 if past is None else past[0].size(-2)
        q, k = self._normalize_qk(q, k)
        q, k = position_encoding.apply(q, k, offset)
        if past is not None:
            past_length = past[0].size(-2)
            k = torch.cat((past[0], k), dim=-2)
            v = torch.cat((past[1], v), dim=-2)
            causal = torch.ones((length, k.size(-2)), device=x.device, dtype=torch.bool).tril(
                diagonal=past_length,
            )
            y = self._sdpa(q, k, v, is_causal=False, attention_mask=causal[None, None])
        else:
            y = self._sdpa(q, k, v, is_causal=True)
        output = self.out(y.transpose(1, 2).contiguous().view(batch, length, -1))
        return output, (k, v)

__all__ = ["GroupedQueryCausalSelfAttention"]

