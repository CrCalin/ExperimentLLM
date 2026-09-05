"""Registry-built decoder language model and baseline components."""
from __future__ import annotations

from abc import ABC, abstractmethod
import math

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

try:
    from fla.ops.gated_delta_rule import (
        chunk_gated_delta_rule, fused_recurrent_gated_delta_rule,
        naive_recurrent_gated_delta_rule,
    )
    from fla.modules.convolution import causal_conv1d as fla_causal_conv1d
    from fla.modules.fused_norm_gate import rms_norm_gated as fla_rms_norm_gated
    from fla.ops.gdn2 import chunk_gdn2, fused_recurrent_gdn2, naive_recurrent_gdn2
except ImportError:
    chunk_gated_delta_rule = fused_recurrent_gated_delta_rule = None
    naive_recurrent_gated_delta_rule = None
    fla_causal_conv1d = None
    fla_rms_norm_gated = None
    chunk_gdn2 = fused_recurrent_gdn2 = naive_recurrent_gdn2 = None
    fla_parallel_attn = None
else:
    try:
        from fla.ops.attn.parallel import parallel_attn as fla_parallel_attn
    except ImportError:
        fla_parallel_attn = None


def _run_chunk_gated_delta_rule(**kwargs):
    return chunk_gated_delta_rule(**kwargs)


def _run_fused_recurrent_gated_delta_rule(**kwargs):
    return fused_recurrent_gated_delta_rule(**kwargs)

from .config import ComponentSpec, ModelConfig
from .registry import (
    ATTENTIONS, BLOCKS, EMBEDDINGS, HEADS, INITIALIZATIONS, MLPS, MODELS,
    NORMS, POSITIONS, RESIDUALS, ROUTINGS,
)
from .parameter_policy import set_optimizer_policy


def _scaled_dot_product_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, **kwargs) -> torch.Tensor:
    if q.is_cuda:
        # Do not silently accept the quadratic math implementation on an
        # accelerator. CPU/reference execution remains available for tests
        # and inference on machines without fused attention kernels.
        with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]):
            return F.scaled_dot_product_attention(q, k, v, **kwargs)
    return F.scaled_dot_product_attention(q, k, v, **kwargs)


class PositionEncoding(nn.Module, ABC):
    @abstractmethod
    def apply(self, q: torch.Tensor, k: torch.Tensor,
              position_offset: int = 0) -> tuple[torch.Tensor, torch.Tensor]: ...


class Attention(nn.Module, ABC):
    @abstractmethod
    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor | None,
                position_encoding: PositionEncoding) -> torch.Tensor: ...

    def forward_with_route(self, x: torch.Tensor, attention_mask: torch.Tensor | None,
                           position_encoding: PositionEncoding):
        return self.forward(x, attention_mask, position_encoding), None


class MLP(nn.Module, ABC):
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...


class Residual(nn.Module, ABC):
    @abstractmethod
    def forward(self, block: "TransformerBlock", x: torch.Tensor,
                attention_mask: torch.Tensor | None, position_encoding: PositionEncoding) -> torch.Tensor: ...


class Block(nn.Module, ABC):
    @abstractmethod
    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor | None,
                position_encoding: PositionEncoding) -> torch.Tensor: ...


@NORMS.register
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        if eps <= 0: raise ValueError("RMSNorm eps must be positive")
        self.weight = nn.Parameter(torch.ones(dim))
        set_optimizer_policy(self.weight, "adamw", "norm")
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps).to(x.dtype) * self.weight


@EMBEDDINGS.register
class TokenEmbedding(nn.Embedding):
    def __init__(self, vocab_size: int, d_model: int, padding_idx: int | None = None,
                 max_norm: float | None = None, norm_type: float = 2.0,
                 scale_grad_by_freq: bool = False, sparse: bool = False):
        super().__init__(vocab_size, d_model, padding_idx=padding_idx, max_norm=max_norm,
                         norm_type=norm_type, scale_grad_by_freq=scale_grad_by_freq, sparse=sparse)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


@POSITIONS.register
class RoPE(PositionEncoding):
    def __init__(self, context_length: int, base: float = 10000.0):
        super().__init__()
        if base <= 0: raise ValueError("RoPE base must be positive")
        self.context_length, self.head_dim, self.base = context_length, None, base
        self.register_buffer("inv_freq", torch.empty(0), persistent=False)
        self.register_buffer("rope_cos", torch.empty(0), persistent=False)
        self.register_buffer("rope_sin", torch.empty(0), persistent=False)

    def _build_cache(self, device: torch.device, head_dim: int) -> None:
        if head_dim % 2: raise ValueError("RoPE attention head dimension must be even")
        self.head_dim = head_dim
        inv = 1.0 / (self.base ** (torch.arange(0, self.head_dim, 2, device=device).float() / self.head_dim))
        positions = torch.arange(self.context_length, device=device, dtype=inv.dtype)
        freqs = torch.outer(positions, inv).repeat_interleave(2, dim=-1)
        self.inv_freq = inv
        self.rope_cos = freqs.cos()[None, None]
        self.rope_sin = freqs.sin()[None, None]

    def apply(self, q: torch.Tensor, k: torch.Tensor,
              position_offset: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
        length = q.size(-2)
        if position_offset < 0 or position_offset + length > self.context_length:
            raise ValueError("sequence exceeds RoPE context length")
        if self.rope_cos.numel() == 0 or self.rope_cos.device != q.device or self.head_dim != q.size(-1):
            self._build_cache(q.device, q.size(-1))
        cos = self.rope_cos[:, :, position_offset:position_offset + length].to(q.dtype)
        sin = self.rope_sin[:, :, position_offset:position_offset + length].to(q.dtype)
        return q * cos + _rotate_half(q) * sin, k * cos + _rotate_half(k) * sin


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


@ATTENTIONS.register
class SlidingWindowCausalSelfAttention(MultiHeadCausalSelfAttention):
    """Causal MHA restricted to a fixed local window.

    Attention is evaluated in query chunks against only the corresponding
    contiguous key ranges. This keeps the work and temporary attention mask
    proportional to ``sequence_length * window_size`` rather than the full
    context square, while retaining the exact MHA projection/QK-Norm path.
    """
    def __init__(self, d_model: int, n_heads: int, head_dim: int | None = None,
                 window_size: int = 512, chunk_size: int = 512,
                 bias: bool = False, dropout: float = 0.0,
                 qk_norm: bool = False, qk_norm_eps: float = 1e-6,
                 head_scale: bool = False):
        super().__init__(d_model, n_heads, head_dim, bias, dropout, qk_norm,
                         qk_norm_eps, head_scale)
        if window_size <= 0 or chunk_size <= 0:
            raise ValueError("SWA window_size and chunk_size must be positive")
        self.window_size = window_size
        self.chunk_size = chunk_size

    def _local_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                         query_start: int, key_start: int,
                         attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        batch, _, query_length, _ = q.shape
        if attention_mask is None and not torch.is_grad_enabled() \
                and query_length >= self.chunk_size \
                and query_length % self.chunk_size == 0:
            # Batch pairs of equal-sized query chunks into one SDPA dispatch.
            # Keys are padded only to the largest local range in each pair;
            # no [sequence, sequence] tensor is created.
            group_size = 2
            chunks = []
            for start in range(0, query_length, self.chunk_size):
                end = start + self.chunk_size
                first_key = max(key_start, query_start + start - self.window_size + 1)
                last_key = min(key_start + k.size(-2), query_start + end)
                key_slice = slice(first_key - key_start, last_key - key_start)
                query_positions = torch.arange(query_start + start, query_start + end, device=q.device)
                key_positions = torch.arange(first_key, last_key, device=q.device)
                mask = (
                    (key_positions[None, :] <= query_positions[:, None])
                    & (key_positions[None, :] >= query_positions[:, None] - self.window_size + 1)
                )
                chunks.append((q[..., start:end, :], k[..., key_slice, :],
                               v[..., key_slice, :], mask))
            outputs = []
            for offset in range(0, len(chunks), group_size):
                group = chunks[offset:offset + group_size]
                max_keys = max(item[1].size(-2) for item in group)
                q_group = torch.stack([item[0] for item in group], dim=1)
                k_group = torch.stack([
                    F.pad(item[1], (0, 0, 0, max_keys - item[1].size(-2))) for item in group
                ], dim=1)
                v_group = torch.stack([
                    F.pad(item[2], (0, 0, 0, max_keys - item[2].size(-2))) for item in group
                ], dim=1)
                mask_group = torch.stack([
                    F.pad(item[3], (0, max_keys - item[3].size(-1))) for item in group
                ])
                q_group = q_group.reshape(batch * len(group), self.n_heads, self.chunk_size, self.head_dim)
                k_group = k_group.reshape(batch * len(group), self.n_heads, max_keys, self.head_dim)
                v_group = v_group.reshape(batch * len(group), self.n_heads, max_keys, self.head_dim)
                mask_group = mask_group.repeat(batch, 1, 1)
                outputs.append(_scaled_dot_product_attention(
                    q_group, k_group, v_group, attn_mask=mask_group[:, None],
                    dropout_p=self.dropout if self.training else 0.0, is_causal=False,
                ).reshape(batch, len(group), self.n_heads, self.chunk_size, self.head_dim))
            return torch.cat(outputs, dim=1).permute(0, 2, 1, 3, 4).reshape(
                batch, self.n_heads, query_length, self.head_dim,
            )
        outputs = []
        for start in range(0, query_length, self.chunk_size):
            end = min(start + self.chunk_size, query_length)
            query_positions = torch.arange(
                query_start + start, query_start + end, device=q.device,
            )
            first_key = max(key_start, query_start + start - self.window_size + 1)
            last_key = min(key_start + k.size(-2), query_start + end)
            key_slice = slice(first_key - key_start, last_key - key_start)
            keys = k[..., key_slice, :]
            values = v[..., key_slice, :]
            key_positions = torch.arange(first_key, last_key, device=q.device)
            causal = (
                (key_positions[None, :] <= query_positions[:, None])
                & (key_positions[None, :] >= query_positions[:, None] - self.window_size + 1)
            )
            mask = causal[None, None]
            if attention_mask is not None:
                valid = attention_mask[:, first_key:last_key].to(device=q.device, dtype=torch.bool)
                mask = mask & valid[:, None, None, :]
            outputs.append(_scaled_dot_product_attention(
                q[..., start:end, :], keys, values, attn_mask=mask,
                dropout_p=self.dropout if self.training else 0.0, is_causal=False,
            ))
        return torch.cat(outputs, dim=-2)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor | None,
                position_encoding: PositionEncoding) -> torch.Tensor:
        batch, length, _ = x.shape
        q, k, v = self.qkv(x).view(batch, length, 3, self.n_heads, self.head_dim).unbind(2)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        q, k = self._normalize_qk(q, k)
        q, k = position_encoding.apply(q, k)
        q, k = q.to(v.dtype), k.to(v.dtype)
        if hasattr(self, "head_scale"):
            q = q * self.head_scale.to(dtype=q.dtype)[None, :, None, None]
        if x.is_cuda and attention_mask is None and fla_parallel_attn is not None:
            y = fla_parallel_attn(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                scale=self.head_dim ** -0.5, window_size=self.window_size,
            ).transpose(1, 2)
        else:
            y = self._local_attention(q, k, v, 0, 0, attention_mask)
        return self.out(y.transpose(1, 2).contiguous().view(batch, length, -1))

    def forward_cached(self, x: torch.Tensor, position_encoding: PositionEncoding,
                       past: dict | None = None):
        batch, length, _ = x.shape
        previous_length = 0 if past is None else past["length"]
        past_k = None if past is None else past["k"]
        past_v = None if past is None else past["v"]
        q, k, v = self.qkv(x).view(batch, length, 3, self.n_heads, self.head_dim).unbind(2)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        q, k = self._normalize_qk(q, k)
        q, k = position_encoding.apply(q, k, previous_length)
        if hasattr(self, "head_scale"):
            q = q * self.head_scale.to(dtype=q.dtype)[None, :, None, None]
        if past_k is not None:
            k = torch.cat((past_k, k), dim=-2)
            v = torch.cat((past_v, v), dim=-2)
        key_start = previous_length - (0 if past_k is None else past_k.size(-2))
        y = self._local_attention(q, k, v, previous_length, key_start)
        keep = max(0, self.window_size - 1)
        new_cache = {
            "length": previous_length + length,
            "k": k[..., -keep:, :] if keep else k[..., :0, :],
            "v": v[..., -keep:, :] if keep else v[..., :0, :],
        }
        output = self.out(y.transpose(1, 2).contiguous().view(batch, length, -1))
        return output, new_cache


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


@ATTENTIONS.register
class GroupedQuerySlidingWindowCausalSelfAttention(GroupedQueryCausalSelfAttention):
    """Grouped-query causal attention restricted to a fixed local window."""
    def __init__(self, d_model: int, n_q_heads: int, n_kv_heads: int,
                 head_dim: int = 64, window_size: int = 512, chunk_size: int = 512,
                 bias: bool = False, dropout: float = 0.0,
                 qk_norm: bool = False, qk_norm_eps: float = 1e-6):
        super().__init__(d_model, n_q_heads, n_kv_heads, head_dim, bias, dropout,
                         qk_norm, qk_norm_eps)
        if window_size <= 0 or chunk_size <= 0:
            raise ValueError("SWA window_size and chunk_size must be positive")
        self.n_heads = n_q_heads
        self.window_size = window_size
        self.chunk_size = chunk_size

    def _expand_kv(self, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        repeats = self.n_q_heads // self.n_kv_heads
        # Expand each KV head across its query group without allocating a
        # second KV cache.  SDPA can consume the resulting strided view.
        def expand(value: torch.Tensor) -> torch.Tensor:
            batch, _, length, head_dim = value.shape
            return value[:, :, None].expand(batch, self.n_kv_heads, repeats, length, head_dim).reshape(
                batch, self.n_q_heads, length, head_dim,
            )
        return expand(k), expand(v)

    def _local_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                         query_start: int, key_start: int,
                         attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        return SlidingWindowCausalSelfAttention._local_attention(
            self, q, k, v, query_start, key_start, attention_mask,
        )

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor | None,
                position_encoding: PositionEncoding) -> torch.Tensor:
        batch, length, _ = x.shape
        q, k, v = self._project(x)
        q, k = self._normalize_qk(q, k)
        q, k = position_encoding.apply(q, k)
        q, k = q.to(v.dtype), k.to(v.dtype)
        if x.is_cuda and attention_mask is None and fla_parallel_attn is not None:
            # FLA supports GQA directly (`HQ` divisible by `H`) and keeps the
            # two KV heads compressed inside the fused SWA kernel.
            y = fla_parallel_attn(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                scale=self.head_dim ** -0.5, window_size=self.window_size,
            ).transpose(1, 2)
        else:
            k, v = self._expand_kv(k, v)
            y = self._local_attention(q, k, v, 0, 0, attention_mask)
        return self.out(y.transpose(1, 2).contiguous().view(batch, length, -1))

    def forward_cached(self, x: torch.Tensor, position_encoding: PositionEncoding,
                       past: dict | None = None):
        batch, length, _ = x.shape
        previous_length = 0 if past is None else past["length"]
        past_k = None if past is None else past["k"]
        past_v = None if past is None else past["v"]
        q, k, v = self._project(x)
        q, k = self._normalize_qk(q, k)
        q, k = position_encoding.apply(q, k, previous_length)
        if past_k is not None:
            k = torch.cat((past_k, k), dim=-2)
            v = torch.cat((past_v, v), dim=-2)
        key_start = previous_length - (0 if past_k is None else past_k.size(-2))
        expanded_k, expanded_v = self._expand_kv(k, v)
        y = self._local_attention(q.to(expanded_v.dtype), expanded_k.to(expanded_v.dtype),
                                  expanded_v, previous_length, key_start)
        keep = max(0, self.window_size - 1)
        new_cache = {
            "length": previous_length + length,
            "k": k[..., -keep:, :] if keep else k[..., :0, :],
            "v": v[..., -keep:, :] if keep else v[..., :0, :],
        }
        output = self.out(y.transpose(1, 2).contiguous().view(batch, length, -1))
        return output, new_cache


class GatedRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5, activation: str = "silu"):
        super().__init__()
        if activation not in {"silu", "sigmoid"}:
            raise ValueError("gated RMSNorm activation must be silu or sigmoid")
        self.eps = eps
        self.activation = activation
        self.weight = nn.Parameter(torch.ones(dim))
        set_optimizer_policy(self.weight, "adamw", "norm")

    def forward(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps).to(x.dtype)
        gate = F.silu(gate) if self.activation == "silu" else torch.sigmoid(gate)
        return x * rms * self.weight * gate


def _gdn_decay(a: torch.Tensor, A_log: torch.Tensor,
               dt_bias: torch.Tensor) -> torch.Tensor:
    return -torch.exp(A_log).view(1, 1, -1) * F.softplus(
        a + dt_bias.view(1, 1, -1),
    )


def _gdn_recurrence(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    beta: torch.Tensor, g: torch.Tensor, state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference recurrent GDN scan; state is [B,H,V,K]."""
    carry = state.float()
    outputs = []
    for q_t, k_t, v_t, beta_t, g_t in zip(q.transpose(0, 1), k.transpose(0, 1),
                                             v.transpose(0, 1), beta.transpose(0, 1),
                                             g.transpose(0, 1)):
        carry = carry * g_t.float().exp()[..., None, None]
        delta = beta_t.float()[..., None] * (
            v_t.float() - torch.einsum("bhvk,bhk->bhv", carry, k_t.float())
        )
        carry = carry + torch.einsum("bhk,bhv->bhvk", k_t.float(), delta)
        outputs.append(torch.einsum("bhk,bhvk->bhv", q_t.float(), carry))
    return torch.stack(outputs, dim=1).to(v.dtype), carry.to(state.dtype)


@ATTENTIONS.register
class DeltaNetLinearAttention(Attention):
    """Canonical GatedDeltaNet-style attention using FLA's gated kernels."""
    def __init__(self, d_model: int, n_heads: int, head_dim: int = 256,
                 expand_v: float = 2.0, num_v_heads: int | None = None,
                 allow_neg_eigval: bool = False, conv_size: int = 4,
                 conv_bias: bool = False, norm_eps: float = 1e-5,
                 return_value_route: bool = False):
        super().__init__()
        if n_heads <= 0 or head_dim <= 0 or expand_v <= 0:
            raise ValueError("invalid GatedDeltaNet head configuration")
        self.hidden_size, self.n_heads, self.head_dim = d_model, n_heads, head_dim
        self.expand_v, self.num_v_heads = expand_v, num_v_heads or n_heads
        self.allow_neg_eigval, self.conv_size = allow_neg_eigval, conv_size
        self.return_value_route = return_value_route
        self.chunk_size = 16
        self.key_dim = n_heads * head_dim
        self.head_v_dim = int(head_dim * expand_v)
        self.value_dim = self.num_v_heads * self.head_v_dim
        # FLA's grouped-value path requires the query head count to be an
        # integer multiple of the value head count (e.g. 4/1, 4/2, 4/4).
        if n_heads % self.num_v_heads != 0:
            raise ValueError("n_heads must be divisible by num_v_heads")
        if not math.isclose(head_dim * expand_v, self.head_v_dim):
            raise ValueError("expand_v must produce an integer value head dimension")

        self.projection_sizes = (
            self.key_dim, self.key_dim, self.value_dim,
            self.num_v_heads, self.num_v_heads, self.value_dim,
        )
        self.in_proj_weight = nn.Parameter(torch.empty(sum(self.projection_sizes), d_model))
        set_optimizer_policy(self.in_proj_weight, "adamw", "deltanet_gating_projection")
        nn.init.normal_(self.in_proj_weight, mean=0.0, std=0.02)
        self.A_log = nn.Parameter(torch.empty(self.num_v_heads))
        self.dt_bias = nn.Parameter(torch.empty(self.num_v_heads))
        self.A_log._no_weight_decay = True
        self.dt_bias._no_weight_decay = True
        set_optimizer_policy(self.A_log, "adamw", "deltanet_decay")
        set_optimizer_policy(self.dt_bias, "adamw", "deltanet_gating")
        if conv_bias:
            raise ValueError("the optimized GatedDeltaNet path requires conv_bias=false")
        self.conv_weight = nn.Parameter(torch.empty(
            2 * self.key_dim + self.value_dim, 1, conv_size,
        ))
        nn.init.normal_(self.conv_weight, mean=0.0, std=0.02)
        self.o_norm = GatedRMSNorm(self.head_v_dim, norm_eps)
        self.o_proj = nn.Linear(self.value_dim, d_model, bias=False)
        self.o_proj.residual_projection = True
        self.reset_specialized_parameters()

    def reset_specialized_parameters(self):
        with torch.no_grad():
            nn.init.normal_(self.in_proj_weight, mean=0.0, std=0.02)
            nn.init.normal_(self.conv_weight, mean=0.0, std=0.02)
            A = torch.empty(self.num_v_heads, dtype=torch.float32).uniform_(0, 16)
            self.A_log.copy_(torch.log(A))
            dt = torch.exp(torch.empty(self.num_v_heads).uniform_(math.log(0.001), math.log(0.1)))
            self.dt_bias.copy_(dt + torch.log(-torch.expm1(-dt)))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                               missing_keys, unexpected_keys, error_msgs):
        packed_key = prefix + "in_proj_weight"
        if packed_key not in state_dict:
            old_keys = tuple(prefix + name for name in (
                "q_proj.weight", "k_proj.weight", "v_proj.weight",
                "a_proj.weight", "b_proj.weight", "g_proj.weight",
            ))
            if all(key in state_dict for key in old_keys):
                state_dict[packed_key] = torch.cat(tuple(state_dict.pop(key) for key in old_keys), dim=0)
        conv_key = prefix + "conv_weight"
        if conv_key not in state_dict:
            old_keys = tuple(prefix + name for name in (
                "q_conv1d.weight", "k_conv1d.weight", "v_conv1d.weight",
            ))
            if all(key in state_dict for key in old_keys):
                state_dict[conv_key] = torch.cat(tuple(state_dict.pop(key) for key in old_keys), dim=0)
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs,
        )

    def initial_state(self, x: torch.Tensor, dtype: torch.dtype | None = None) -> torch.Tensor:
        return x.new_zeros(x.size(0), self.num_v_heads, self.head_v_dim, self.head_dim,
                           dtype=dtype or x.dtype)

    def forward_recurrent(
        self,
        x: torch.Tensor,
        state: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        conv_state: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None]:
        output, state, conv_state, _ = self._forward_impl(
            x, state, attention_mask, output_final_state=True,
            conv_state=conv_state, use_cache=True,
        )
        if state is None: raise RuntimeError("DeltaNet recurrent state was not returned")
        return output, state, conv_state

    def _fused_short_conv(
        self,
        qkv: torch.Tensor,
        conv_state: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None,
        output_final_state: bool,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None]:
        """Apply the three depthwise short convolutions as one grouped convolution."""
        old = conv_state or (None, None, None)
        sequence = qkv.transpose(1, 2)
        if any(cache is not None for cache in old):
            if any(cache is None for cache in old):
                raise ValueError("all q/k/v convolution caches must be provided together")
            for cache, hidden_size in zip(old, (self.key_dim, self.key_dim, self.value_dim)):
                expected = (qkv.size(0), hidden_size, self.conv_size - 1)
                if cache.shape != expected:
                    raise ValueError(f"convolution cache must have shape {expected}")
            sequence = torch.cat((torch.cat(old, dim=1), sequence), dim=-1)
        y = F.conv1d(
            F.pad(sequence, (self.conv_size - 1, 0)), self.conv_weight, None,
            groups=sequence.size(1),
        )
        y = F.silu(y[..., -qkv.size(1):].transpose(1, 2))
        new_state = None
        if output_final_state:
            cache_size = self.conv_size - 1
            new_sequence = sequence[..., -cache_size:] if cache_size else sequence[..., :0]
            new_state = tuple(torch.split(
                new_sequence, (self.key_dim, self.key_dim, self.value_dim), dim=1,
            ))
        return y, new_state

    def _forward_impl(
        self,
        x: torch.Tensor,
        state: torch.Tensor | None,
        attention_mask: torch.Tensor | None,
        output_final_state: bool,
        conv_state: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None, torch.Tensor | None]:
        batch, length, _ = x.shape
        new_conv_state = None
        projected = F.linear(x, self.in_proj_weight)
        q, k, v, a, beta, gate = torch.split(projected, self.projection_sizes, dim=-1)
        qkv = projected[..., :2 * self.key_dim + self.value_dim]
        if x.is_cuda and not use_cache and fla_causal_conv1d is None:
            raise RuntimeError("CUDA DeltaNet training requires FLA causal_conv1d")
        if fla_causal_conv1d is not None and x.is_cuda and not use_cache and conv_state is None:
            qkv, _ = fla_causal_conv1d(
                qkv, weight=self.conv_weight.squeeze(1), bias=None,
                activation="silu", backend="triton",
            )
        else:
            qkv, new_conv_state = self._fused_short_conv(qkv, conv_state, use_cache)
        q, k, v = torch.split(qkv, (self.key_dim, self.key_dim, self.value_dim), dim=-1)
        q = q.view(batch, length, self.n_heads, self.head_dim)
        k = k.view(batch, length, self.n_heads, self.head_dim)
        v = v.view(batch, length, self.num_v_heads, self.head_v_dim)
        if self.num_v_heads != self.n_heads:
            # FLA's grouped-value rule has one q/k stream per value state.
            # Collapse each query-head group before dispatching; the baseline
            # (equal query/value heads) remains byte-for-byte unchanged.
            groups = self.n_heads // self.num_v_heads
            q = q.view(batch, length, self.num_v_heads, groups, self.head_dim).mean(dim=3)
            k = k.view(batch, length, self.num_v_heads, groups, self.head_dim).mean(dim=3)
        value_route = v.reshape(batch, length, self.value_dim) if self.return_value_route else None
        if attention_mask is not None:
            if attention_mask.shape != (batch, length):
                raise ValueError("attention_mask must have shape (batch, sequence)")
            mask = attention_mask[:, :, None].to(device=x.device, dtype=beta.dtype)
            beta = beta.masked_fill(mask == 0, -torch.inf)
        expected = (batch, self.num_v_heads, self.head_v_dim, self.head_dim)
        if state is not None and state.shape != expected:
            raise ValueError(f"DeltaNet state must have shape {expected}")
        if x.is_cuda and fused_recurrent_gated_delta_rule is None:
            raise RuntimeError("CUDA DeltaNet execution requires FLA fused_recurrent_gated_delta_rule")
        if x.is_cuda and (torch.is_grad_enabled() or length > 64) and chunk_gated_delta_rule is None:
            raise RuntimeError("CUDA DeltaNet training/long-context execution requires FLA chunk_gated_delta_rule")
        use_fused = x.is_cuda
        use_chunk = use_fused and (torch.is_grad_enabled() or length > 64)
        if use_chunk:
            y, state = _run_chunk_gated_delta_rule(
                q=q, k=k, v=v, g=a, beta=beta, scale=1.0,
                A_log=self.A_log, dt_bias=self.dt_bias, initial_state=state,
                output_final_state=output_final_state, use_qk_l2norm_in_kernel=True,
                use_gate_in_kernel=True, use_beta_sigmoid_in_kernel=True,
                allow_neg_eigval=self.allow_neg_eigval,
                state_v_first=True, chunk_size=self.chunk_size,
            )
        elif use_fused:
            y, state = _run_fused_recurrent_gated_delta_rule(
                q=q, k=k, v=v, g=a, beta=beta, scale=1.0,
                A_log=self.A_log, dt_bias=self.dt_bias, initial_state=state,
                output_final_state=output_final_state, use_qk_l2norm_in_kernel=True,
                use_gate_in_kernel=True, use_beta_sigmoid_in_kernel=True,
                allow_neg_eigval=self.allow_neg_eigval, state_v_first=True,
            )
        elif naive_recurrent_gated_delta_rule is not None:
            state = state if state is not None else self.initial_state(x, dtype=q.dtype)
            decay = _gdn_decay(a, self.A_log, self.dt_bias)
            q_ref, k_ref = F.normalize(q.float(), dim=-1), F.normalize(k.float(), dim=-1)
            y, naive_state = naive_recurrent_gated_delta_rule(
                q=q_ref, k=k_ref, v=v, beta=torch.sigmoid(beta), g=decay, scale=1.0,
                initial_state=state.transpose(-1, -2), output_final_state=output_final_state,
            )
            state = None if naive_state is None else naive_state.transpose(-1, -2)
        else:
            state = state if state is not None else self.initial_state(x, dtype=q.dtype)
            decay = _gdn_decay(a, self.A_log, self.dt_bias)
            y, state = _gdn_recurrence(
                F.normalize(q.float(), dim=-1), F.normalize(k.float(), dim=-1),
                v, torch.sigmoid(beta), decay, state,
            )
        if not output_final_state:
            state = None
        y = y.contiguous().view(batch, length, self.value_dim)
        gate = gate.view(batch, length, self.num_v_heads, self.head_v_dim)
        normalized = y.view(batch, length, self.num_v_heads, self.head_v_dim)
        if y.is_cuda and fla_rms_norm_gated is None:
            raise RuntimeError("CUDA DeltaNet execution requires FLA rms_norm_gated")
        if fla_rms_norm_gated is not None and y.is_cuda:
            y = fla_rms_norm_gated(
                normalized, gate, self.o_norm.weight, None,
                activation="silu", eps=self.o_norm.eps,
            )
        else:
            y = self.o_norm(normalized, gate)
        output = self.o_proj(y.reshape(batch, length, self.value_dim))
        return output, state, new_conv_state, value_route

    def forward_cached(self, x: torch.Tensor, state: torch.Tensor | None = None,
                       conv_state: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None):
        output, state, conv_state, _ = self._forward_impl(x, state, None, True, conv_state, True)
        return output, state, conv_state

    def forward_cached_with_route(self, x: torch.Tensor, state: torch.Tensor | None = None,
                                  conv_state: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None):
        return self._forward_impl(x, state, None, True, conv_state, True)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor | None,
                position_encoding: PositionEncoding) -> torch.Tensor:
        output, _, _, _ = self._forward_impl(x, None, attention_mask, False)
        return output

    def forward_with_route(self, x: torch.Tensor, attention_mask: torch.Tensor | None,
                           position_encoding: PositionEncoding):
        output, _, _, value_route = self._forward_impl(x, None, attention_mask, False)
        return output, value_route


@ATTENTIONS.register
class GatedDeltaNet2LinearAttention(Attention):
    """FLA Gated DeltaNet-2 mixer with project-compatible cache handling."""
    def __init__(self, d_model: int, n_heads: int = 16, head_dim: int = 128,
                 expand_v: float = 1.0, num_v_heads: int | None = None,
                 mode: str = "chunk", use_short_conv: bool = True,
                 allow_neg_eigval: bool = False, conv_size: int = 4,
                 conv_bias: bool = False, norm_eps: float = 1e-5,
                 lower_bound: float | None = None, safe_gate: bool = False):
        super().__init__()
        if mode not in {"chunk", "fused_recurrent"}:
            raise ValueError("GDN2 mode must be chunk or fused_recurrent")
        if n_heads <= 0 or head_dim <= 0 or expand_v <= 0:
            raise ValueError("invalid GDN2 head configuration")
        self.hidden_size, self.n_heads, self.head_dim = d_model, n_heads, head_dim
        self.expand_v, self.num_v_heads = expand_v, num_v_heads or n_heads
        self.mode, self.use_short_conv = mode, use_short_conv
        self.allow_neg_eigval, self.conv_size = allow_neg_eigval, conv_size
        self.lower_bound, self.safe_gate = lower_bound, safe_gate
        self.key_dim = n_heads * head_dim
        self.head_v_dim = int(head_dim * expand_v)
        self.value_dim = self.num_v_heads * self.head_v_dim
        if self.num_v_heads < n_heads or self.num_v_heads % n_heads:
            raise ValueError("num_v_heads must be >= and divisible by n_heads")
        if not math.isclose(head_dim * expand_v, self.head_v_dim):
            raise ValueError("expand_v must produce an integer value head dimension")

        self.projection_sizes = (self.key_dim, self.key_dim, self.value_dim)
        self.qkv_proj_weight = nn.Parameter(torch.empty(sum(self.projection_sizes), d_model))
        nn.init.normal_(self.qkv_proj_weight, mean=0.0, std=0.02)
        if use_short_conv and not conv_bias:
            self.qkv_conv_weight = nn.Parameter(torch.empty(
                2 * self.key_dim + self.value_dim, 1, conv_size,
            ))
            nn.init.normal_(self.qkv_conv_weight, mean=0.0, std=0.02)
        elif use_short_conv:
            self.q_conv1d = nn.Conv1d(self.key_dim, self.key_dim, conv_size,
                                      groups=self.key_dim, bias=conv_bias)
            self.k_conv1d = nn.Conv1d(self.key_dim, self.key_dim, conv_size,
                                      groups=self.key_dim, bias=conv_bias)
            self.v_conv1d = nn.Conv1d(self.value_dim, self.value_dim, conv_size,
                                      groups=self.value_dim, bias=conv_bias)
        self.fg_proj = nn.Linear(d_model, 2 * self.head_v_dim, bias=False)
        self.f_out_proj = nn.Linear(self.head_v_dim, self.key_dim, bias=False)
        self.bw_proj_weight = nn.Parameter(torch.empty(self.key_dim + self.value_dim, d_model))
        nn.init.normal_(self.bw_proj_weight, mean=0.0, std=0.02)
        self.g_out_proj = nn.Linear(self.head_v_dim, self.value_dim, bias=True)
        for module, reason in (
            (self.fg_proj, "gdn2_decay_projection"),
            (self.f_out_proj, "gdn2_decay_projection"),
            (self.g_out_proj, "gdn2_output_gate"),
        ):
            for parameter in module.parameters():
                set_optimizer_policy(parameter, "adamw", reason)
        set_optimizer_policy(self.bw_proj_weight, "adamw", "gdn2_erase_write_gates")
        self.A_log = nn.Parameter(torch.log(torch.empty(n_heads, dtype=torch.float32).uniform_(1, 16)))
        self.dt_bias = nn.Parameter(torch.empty(self.key_dim, dtype=torch.float32))
        self.A_log._no_weight_decay = True
        self.dt_bias._no_weight_decay = True
        set_optimizer_policy(self.A_log, "adamw", "gdn2_decay")
        set_optimizer_policy(self.dt_bias, "adamw", "gdn2_decay_bias")
        self.o_norm = GatedRMSNorm(self.head_v_dim, norm_eps, activation="sigmoid")
        self.o_proj = nn.Linear(self.value_dim, d_model, bias=False)
        self.o_proj.residual_projection = True
        self.reset_specialized_parameters()

    def reset_specialized_parameters(self):
        with torch.no_grad():
            dt = torch.exp(torch.empty(self.key_dim).uniform_(math.log(0.001), math.log(0.1))).clamp(min=1e-4)
            self.dt_bias.copy_(dt + torch.log(-torch.expm1(-dt)))

    def initial_state(self, x: torch.Tensor, dtype: torch.dtype | None = None) -> torch.Tensor:
        # FLA's state_v_first layout is [batch, value_head, value_dim, key_dim].
        return x.new_zeros(x.size(0), self.num_v_heads, self.head_v_dim, self.head_dim,
                           dtype=torch.float32)

    def _short_conv(self, x: torch.Tensor, conv: nn.Conv1d,
                    cache: torch.Tensor | None, output_final_state: bool):
        weight = conv.weight.squeeze(1)
        if x.is_cuda:
            if fla_causal_conv1d is None:
                raise RuntimeError("CUDA GDN2 execution requires FLA causal_conv1d")
            return fla_causal_conv1d(
                x, weight=weight, bias=conv.bias, initial_state=cache,
                output_final_state=output_final_state, activation="silu", backend="triton",
            )
        sequence = x.transpose(1, 2)
        if cache is not None:
            sequence = torch.cat((cache, sequence), dim=-1)
        y = F.conv1d(F.pad(sequence, (self.conv_size - 1, 0)), conv.weight, conv.bias,
                     groups=sequence.size(1))
        y = F.silu(y[..., -x.size(1):].transpose(1, 2))
        new_cache = sequence[..., -self.conv_size:]
        return y, new_cache if output_final_state else None

    def _prepare_recurrence_inputs(self, x: torch.Tensor,
                                   attention_mask: torch.Tensor | None = None,
                                   pack_projections: bool = False):
        """Build the GDN2 streams shared by writer-only and writer-reader paths."""
        batch, length, _ = x.shape
        if pack_projections:
            # The three projections consume the same input.  At the small
            # model widths used here, one larger GEMM is cheaper than three
            # independent dispatches.  ``cat`` preserves parameter identity
            # and therefore normal gradient accumulation/checkpoint names.
            qkv_size = sum(self.projection_sizes)
            projected = F.linear(x, torch.cat((
                self.qkv_proj_weight, self.fg_proj.weight, self.bw_proj_weight,
            ), dim=0))
            projected, fg, bw = torch.split(
                projected, (qkv_size, 2 * self.head_v_dim,
                            self.key_dim + self.value_dim), dim=-1,
            )
        else:
            projected = F.linear(x, self.qkv_proj_weight)
            fg = self.fg_proj(x)
            bw = F.linear(x, self.bw_proj_weight)
        q, k, v = torch.split(projected, self.projection_sizes, dim=-1)
        if self.use_short_conv:
            if not hasattr(self, "qkv_conv_weight"):
                raise RuntimeError("writer-reader GDN2 currently requires packed short-conv weights")
            # ``projected`` is already laid out as the packed QKV stream.
            # Avoid split -> cat and its full-width temporary before the
            # depthwise convolution.
            qkv = projected
            if x.is_cuda:
                if fla_causal_conv1d is None:
                    raise RuntimeError("CUDA GDN2 execution requires FLA causal_conv1d")
                qkv, _ = fla_causal_conv1d(
                    qkv, weight=self.qkv_conv_weight.squeeze(1), bias=None,
                    initial_state=None, output_final_state=False,
                    activation="silu", backend="triton",
                )
            else:
                sequence = qkv.transpose(1, 2)
                y = F.conv1d(F.pad(sequence, (self.conv_size - 1, 0)), self.qkv_conv_weight,
                             None, groups=sequence.size(1))
                qkv = F.silu(y.transpose(1, 2))
            q, k, v = torch.split(qkv, self.projection_sizes, dim=-1)
        else:
            q, k, v = F.silu(q), F.silu(k), F.silu(v)
        f_hidden, g_hidden = torch.split(fg, self.head_v_dim, dim=-1)
        f = self.f_out_proj(f_hidden).float().view(batch, length, self.n_heads, self.head_dim)
        g = -self.A_log.float().exp().view(1, 1, self.n_heads, 1) * F.softplus(
            f + self.dt_bias.view(1, 1, self.n_heads, self.head_dim),
        )
        q = q.view(batch, length, self.n_heads, self.head_dim)
        k = k.view(batch, length, self.n_heads, self.head_dim)
        v = v.view(batch, length, self.num_v_heads, self.head_v_dim)
        b, w = torch.split(bw, (self.key_dim, self.value_dim), dim=-1)
        b = b.sigmoid().view(batch, length, self.n_heads, self.head_dim)
        w = w.sigmoid().view(batch, length, self.num_v_heads, self.head_v_dim)
        if self.num_v_heads > self.n_heads:
            groups = self.num_v_heads // self.n_heads
            q, k, g, b = (t.repeat_interleave(groups, dim=2) for t in (q, k, g, b))
        if self.allow_neg_eigval:
            b = b * 2.0
        if attention_mask is not None:
            mask = attention_mask.to(device=x.device, dtype=q.dtype)[:, :, None, None]
            q, k, v, g, b, w = (t * mask for t in (q, k, v, g, b, w))
        return q, k, v, g, b, w, g_hidden

    def _forward_impl(self, x: torch.Tensor, state: torch.Tensor | None,
                      attention_mask: torch.Tensor | None, output_final_state: bool,
                      conv_state=None, use_cache: bool = False,
                      return_recurrence_inputs: bool = False):
        batch, length, _ = x.shape
        if attention_mask is not None and attention_mask.shape != (batch, length):
            raise ValueError("attention_mask must have shape (batch, sequence)")
        projected = F.linear(x, self.qkv_proj_weight)
        q, k, v = torch.split(projected, self.projection_sizes, dim=-1)
        new_conv_state = None
        if self.use_short_conv:
            old = conv_state or (None, None, None)
            if hasattr(self, "qkv_conv_weight"):
                # The projection output already has the exact packed QKV
                # layout expected by the depthwise short convolution.
                qkv = projected
                if x.is_cuda:
                    if fla_causal_conv1d is None:
                        raise RuntimeError("CUDA GDN2 execution requires FLA causal_conv1d")
                    qkv, packed_state = fla_causal_conv1d(
                        qkv, weight=self.qkv_conv_weight.squeeze(1), bias=None,
                        initial_state=None if old[0] is None else torch.cat(old, dim=1),
                        output_final_state=output_final_state, activation="silu", backend="triton",
                    )
                else:
                    sequence = qkv.transpose(1, 2)
                    if old[0] is not None:
                        sequence = torch.cat((torch.cat(old, dim=1), sequence), dim=-1)
                    y = F.conv1d(F.pad(sequence, (self.conv_size - 1, 0)), self.qkv_conv_weight, None,
                                 groups=sequence.size(1))
                    qkv = F.silu(y[..., -length:].transpose(1, 2))
                    packed_state = sequence[..., -self.conv_size:] if output_final_state else None
                q, k, v = torch.split(qkv, self.projection_sizes, dim=-1)
                new_conv_state = None if packed_state is None else tuple(
                    torch.split(packed_state, (self.key_dim, self.key_dim, self.value_dim), dim=1)
                )
            else:
                q, cq = self._short_conv(q, self.q_conv1d, old[0], output_final_state)
                k, ck = self._short_conv(k, self.k_conv1d, old[1], output_final_state)
                v, cv = self._short_conv(v, self.v_conv1d, old[2], output_final_state)
                new_conv_state = (cq, ck, cv)
        else:
            q, k, v = F.silu(q), F.silu(k), F.silu(v)
        fg = self.fg_proj(x)
        f_hidden, g_hidden = torch.split(fg, self.head_v_dim, dim=-1)
        f = self.f_out_proj(f_hidden).float().view(batch, length, self.n_heads, self.head_dim)
        g = -self.A_log.float().exp().view(1, 1, self.n_heads, 1) * F.softplus(
            f + self.dt_bias.view(1, 1, self.n_heads, self.head_dim),
        )
        q = q.view(batch, length, self.n_heads, self.head_dim)
        k = k.view(batch, length, self.n_heads, self.head_dim)
        v = v.view(batch, length, self.num_v_heads, self.head_v_dim)
        bw = F.linear(x, self.bw_proj_weight)
        b, w = torch.split(bw, (self.key_dim, self.value_dim), dim=-1)
        b = b.sigmoid().view(batch, length, self.n_heads, self.head_dim)
        w = w.sigmoid().view(batch, length, self.num_v_heads, self.head_v_dim)
        if self.num_v_heads > self.n_heads:
            groups = self.num_v_heads // self.n_heads
            q, k, g, b = (t.repeat_interleave(groups, dim=2) for t in (q, k, g, b))
        if self.allow_neg_eigval:
            b = b * 2.0
        if attention_mask is not None:
            mask = attention_mask.to(device=x.device, dtype=q.dtype)[:, :, None, None]
            q, k, v, g, b, w = (t * mask for t in (q, k, v, g, b, w))
        if state is not None:
            expected = (batch, self.num_v_heads, self.head_v_dim, self.head_dim)
            if state.shape != expected:
                raise ValueError(f"GDN2 state must have shape {expected}")
            if state.dtype != torch.float32:
                state = state.float()
        if x.is_cuda:
            if chunk_gdn2 is None or fused_recurrent_gdn2 is None:
                raise RuntimeError("CUDA GDN2 execution requires FLA GDN2 kernels")
            mode = "chunk" if torch.is_grad_enabled() else (
                "fused_recurrent" if length <= 64 else self.mode
            )
            if mode == "chunk":
                y, state = chunk_gdn2(q, k, v, g, b, w, initial_state=state,
                                      output_final_state=output_final_state,
                                      use_qk_l2norm_in_kernel=True, state_v_first=True,
                                      safe_gate=self.safe_gate, lower_bound=self.lower_bound)
            else:
                y, state = fused_recurrent_gdn2(q, k, v, g, b, w, initial_state=state,
                                                output_final_state=output_final_state,
                                                use_qk_l2norm_in_kernel=True, state_v_first=True,
                                                lower_bound=self.lower_bound)
        else:
            if naive_recurrent_gdn2 is None:
                raise RuntimeError("GDN2 reference implementation is unavailable")
            state = state if state is not None else self.initial_state(x)
            y, state = naive_recurrent_gdn2(q, k, v, g, b, w, initial_state=state,
                                            output_final_state=output_final_state)
        gate = self.g_out_proj(g_hidden).view(batch, length, self.num_v_heads, self.head_v_dim)
        y = self.o_norm(y, gate).reshape(batch, length, self.value_dim)
        result = self.o_proj(y), (state if output_final_state else None), new_conv_state
        if return_recurrence_inputs:
            result = result + ((q, k, v, g, b, w),)
        return result

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor | None,
                position_encoding: PositionEncoding) -> torch.Tensor:
        output, _, _ = self._forward_impl(x, None, attention_mask, False)
        return output

    def forward_cached_with_route(self, x: torch.Tensor, state=None, conv_state=None):
        return self._forward_impl(x, state, None, True, conv_state, True) + (None,)





@MLPS.register
class SwiGLU(MLP):
    def __init__(self, d_model: int, d_ff: int, bias: bool = False, dropout: float = 0.0):
        super().__init__()
        if d_ff <= 0: raise ValueError("SwiGLU d_ff must be positive")
        if not 0 <= dropout < 1: raise ValueError("MLP dropout must be in [0, 1)")
        self.gate = nn.Linear(d_model, d_ff, bias=bias)
        self.up = nn.Linear(d_model, d_ff, bias=bias)
        self.down = nn.Linear(d_ff, d_model, bias=bias)
        self.down.residual_projection = True
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.down(F.silu(self.gate(x)) * self.up(x)))


@RESIDUALS.register
class StandardResidual(Residual):
    """Conventional sequential attention and MLP residual branches."""
    def __init__(self): super().__init__()

    def forward(self, block: "TransformerBlock", x: torch.Tensor,
                attention_mask: torch.Tensor | None, position_encoding: PositionEncoding) -> torch.Tensor:
        if block.pre_norm:
            attn, route = block.attention_with_route(block.norm1(x), attention_mask, position_encoding)
            x = x + attn
            x = block.add_value_route(x, route, "pre_ffn")
            if block.mlp is None:
                return block.add_value_route(x, route, "post_block")
            x = x + block.mlp(block.norm2(x))
            return block.add_value_route(x, route, "post_block")
        x = block.norm1(x + block.attn(x, attention_mask, position_encoding))
        return x if block.mlp is None else block.norm2(x + block.mlp(x))


@RESIDUALS.register
class GatedResidual(Residual):
    """Sequential pre-norm residual branches with learned scalar gates."""
    def __init__(self, attention_gate_init: float = 1.0, mlp_gate_init: float | None = 1.0):
        super().__init__()
        self.attention_gate = nn.Parameter(torch.tensor(float(attention_gate_init)))
        set_optimizer_policy(self.attention_gate, "adamw", "residual_gate")
        if mlp_gate_init is not None:
            self.mlp_gate = nn.Parameter(torch.tensor(float(mlp_gate_init)))
            set_optimizer_policy(self.mlp_gate, "adamw", "residual_gate")

    def forward(self, block: "TransformerBlock", x: torch.Tensor,
                attention_mask: torch.Tensor | None, position_encoding: PositionEncoding) -> torch.Tensor:
        if not block.pre_norm: raise ValueError("GatedResidual requires pre_norm=true")
        attn, route = block.attention_with_route(block.norm1(x), attention_mask, position_encoding)
        x = x + self.attention_gate * attn
        x = block.add_value_route(x, route, "pre_ffn")
        if block.mlp is None:
            return block.add_value_route(x, route, "post_block")
        x = x + self.mlp_gate * block.mlp(block.norm2(x))
        return block.add_value_route(x, route, "post_block")


@BLOCKS.register
class TransformerBlock(Block):
    def __init__(self, d_model: int, pre_norm: bool, norm: dict, attention: dict,
                 mlp: dict | None, residual: dict, value_routing: dict | None = None,
                 d_model_override: int | None = None):
        super().__init__()
        d_model = d_model if d_model_override is None else d_model_override
        if d_model <= 0:
            raise ValueError("TransformerBlock d_model_override must be positive")
        self.model_dim = d_model
        self.pre_norm = pre_norm
        norm_spec = ComponentSpec.from_dict(norm, "TransformerBlock.args.norm")
        attention_spec = ComponentSpec.from_dict(attention, "TransformerBlock.args.attention")
        residual_spec = ComponentSpec.from_dict(residual, "TransformerBlock.args.residual")
        self.norm1 = NORMS.create(norm_spec.class_name, d_model, **norm_spec.args)
        self.attn = ATTENTIONS.create(attention_spec.class_name, d_model, **attention_spec.args)
        self.value_routing = None
        if value_routing is not None:
            route_spec = ComponentSpec.from_dict(value_routing, "TransformerBlock.args.value_routing")
            if not hasattr(self.attn, "value_dim"):
                raise ValueError("value routing requires an attention module with value_dim")
            self.value_routing = ROUTINGS.create(
                route_spec.class_name, self.attn.value_dim, d_model, **route_spec.args,
            )
        self.norm2 = None
        self.mlp = None
        if mlp is not None:
            mlp_spec = ComponentSpec.from_dict(mlp, "TransformerBlock.args.mlp")
            self.norm2 = NORMS.create(norm_spec.class_name, d_model, **norm_spec.args)
            self.mlp = MLPS.create(mlp_spec.class_name, d_model, **mlp_spec.args)
        residual_args = dict(residual_spec.args)
        if mlp is None and residual_spec.class_name == "GatedResidual":
            residual_args["mlp_gate_init"] = None
        self.residual = RESIDUALS.create(residual_spec.class_name, **residual_args)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor | None,
                position_encoding: PositionEncoding) -> torch.Tensor:
        return self.residual(self, x, attention_mask, position_encoding)

    def attention_with_route(self, x, attention_mask, position_encoding):
        attn, route = self.attn.forward_with_route(x, attention_mask, position_encoding)
        return attn, route

    def add_value_route(self, x, route, placement):
        if self.value_routing is not None and self.value_routing.placement == placement:
            return x + self.value_routing(route)
        return x


@BLOCKS.register
class FFNOnlyBlock(Block):
    """Explicit pre-norm feed-forward residual block without a mixer."""
    def __init__(self, d_model: int, norm: dict, mlp: dict):
        super().__init__()
        norm_spec = ComponentSpec.from_dict(norm, "FFNOnlyBlock.args.norm")
        mlp_spec = ComponentSpec.from_dict(mlp, "FFNOnlyBlock.args.mlp")
        self.norm = NORMS.create(norm_spec.class_name, d_model, **norm_spec.args)
        self.mlp = MLPS.create(mlp_spec.class_name, d_model, **mlp_spec.args)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor | None,
                position_encoding: PositionEncoding) -> torch.Tensor:
        return x + self.mlp(self.norm(x))


@BLOCKS.register


@HEADS.register
class LinearLMHead(nn.Linear):
    def __init__(self, d_model: int, vocab_size: int, bias: bool = False):
        super().__init__(d_model, vocab_size, bias=bias)


@INITIALIZATIONS.register
class GPTStyleInit:
    def __init__(self, std: float = 0.02, scale_residual_projections: bool = True):
        if std <= 0: raise ValueError("initialization std must be positive")
        self.std, self.scale_residual_projections = std, scale_residual_projections

    def apply(self, model: nn.Module, n_layers: int) -> None:
        for module in model.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.normal_(module.weight, mean=0.0, std=self.std)
                if getattr(module, "bias", None) is not None: nn.init.zeros_(module.bias)
                if getattr(module, "zero_init_output", False): nn.init.zeros_(module.weight)
        if self.scale_residual_projections:
            scale = 1 / math.sqrt(2 * n_layers)
            for module in model.modules():
                if getattr(module, "residual_projection", False): module.weight.data.mul_(scale)


@MODELS.register
class CausalLM(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.wte = EMBEDDINGS.create(
            config.input_embedding.class_name, config.vocab_size, config.d_model,
            **config.input_embedding.args,
        )
        self.position_encoding = POSITIONS.create(
            config.position_encoding.class_name, config.context_length,
            **config.position_encoding.args,
        )
        self.blocks = nn.ModuleList([
            BLOCKS.create(layer.class_name, config.d_model, **layer.args) for layer in config.layers
        ])
        self.norm_f = NORMS.create(config.final_norm.class_name, config.d_model, **config.final_norm.args)
        self.lm_head = HEADS.create(
            config.output_head.class_name, config.d_model, config.vocab_size,
            **config.output_head.args,
        )
        for parameter in self.wte.parameters():
            set_optimizer_policy(parameter, "adamw", "token_embedding")
        for parameter in self.lm_head.parameters():
            set_optimizer_policy(parameter, "adamw", "output_head")
        if config.tie_embeddings:
            if self.lm_head.weight.shape != self.wte.weight.shape:
                raise ValueError("input embedding and output head weights cannot be tied")
            self.lm_head.weight = self.wte.weight
            set_optimizer_policy(self.wte.weight, "adamw", "tied_embedding_output")
        initializer = INITIALIZATIONS.create(config.initialization.class_name, **config.initialization.args)
        initializer.apply(self, len(self.blocks))

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor | None = None,
                attention_mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Return logits and loss against already-next-token-aligned labels.

        ``labels[:, t]`` is the target for ``input_ids[:, t]``.  Callers
        performing causal language modeling must shift the token stream when
        constructing the inputs/labels pair; the model deliberately does not
        shift labels implicitly.  This is the contract used by ``TokenStream``
        and the trainer.
        """
        if input_ids.size(1) > self.config.context_length: raise ValueError("sequence exceeds context_length")
        x = self.wte(input_ids)
        for block in self.blocks:
            x = block(x, attention_mask, self.position_encoding)
        logits = self.lm_head(self.norm_f(x))
        loss = None if labels is None else F.cross_entropy(
            logits.float().reshape(-1, logits.size(-1)), labels.reshape(-1)
        )
        return logits, loss

    def forward_cached(self, input_ids: torch.Tensor, cache: dict | None = None):
        """Run a prefix or one decode step while retaining MHA/GDN states."""
        if input_ids.size(1) > self.config.context_length:
            raise ValueError("sequence exceeds context_length")
        cache = cache or {"length": 0, "layers": [None] * len(self.blocks)}
        if cache["length"] + input_ids.size(1) > self.config.context_length:
            raise ValueError("cached sequence exceeds context_length; rebuild the cache")
        x = self.wte(input_ids)
        new_layers = []
        for block, layer_cache in zip(self.blocks, cache["layers"]):
            if isinstance(block, FFNOnlyBlock):
                x = block(x, None, self.position_encoding)
                new_layers.append(None)
                continue
            if not isinstance(block, TransformerBlock) or block.pre_norm is not True:
                raise ValueError("cached generation requires pre-norm TransformerBlock layers")
            if isinstance(block.attn, SlidingWindowCausalSelfAttention):
                attn_out, new_cache = block.attn.forward_cached(
                    block.norm1(x), self.position_encoding, layer_cache,
                )
                gate = getattr(block.residual, "attention_gate", 1.0)
                x = x + gate * attn_out
                new_layers.append(new_cache)
            elif isinstance(block.attn, (MultiHeadCausalSelfAttention, GroupedQueryCausalSelfAttention)):
                attn_out, new_cache = block.attn.forward_cached(
                    block.norm1(x), self.position_encoding, layer_cache,
                )
                gate = getattr(block.residual, "attention_gate", 1.0)
                x = x + gate * attn_out
                new_layers.append(new_cache)
            elif isinstance(block.attn, (DeltaNetLinearAttention, GatedDeltaNet2LinearAttention)):
                state = None if layer_cache is None else layer_cache["state"]
                conv_state = None if layer_cache is None else layer_cache["conv_state"]
                attn_out, state, conv_state, value_route = block.attn.forward_cached_with_route(
                    x=block.norm1(x), state=state, conv_state=conv_state,
                )
                gate = getattr(block.residual, "attention_gate", 1.0)
                x = x + gate * attn_out
                x = block.add_value_route(x, value_route, "pre_ffn")
                new_layers.append({"state": state, "conv_state": conv_state})
            else:
                raise ValueError("cached generation requires known attention implementations")
            if block.mlp is not None:
                gate = getattr(block.residual, "mlp_gate", 1.0)
                x = x + gate * block.mlp(block.norm2(x))
            if isinstance(block.attn, (DeltaNetLinearAttention, GatedDeltaNet2LinearAttention)):
                x = block.add_value_route(x, value_route, "post_block")
        logits = self.lm_head(self.norm_f(x))
        new_cache = {"length": cache["length"] + input_ids.size(1), "layers": new_layers}
        return logits, new_cache

    @property
    def num_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    @property
    def parameter_report(self) -> dict[str, int]:
        block_counts = [sum(p.numel() for p in block.parameters()) for block in self.blocks]
        mha = sum(n for block, n in zip(self.blocks, block_counts)
                  if isinstance(getattr(block, "attn", None), MultiHeadCausalSelfAttention))
        delta = sum(n for block, n in zip(self.blocks, block_counts)
                    if isinstance(getattr(block, "attn", None), (DeltaNetLinearAttention, GatedDeltaNet2LinearAttention)))
        ffn = sum(sum(p.numel() for p in block.mlp.parameters())
                  for block in self.blocks if block.mlp is not None)
        absent = [block for block in self.blocks
                  if hasattr(block, "attn") and block.mlp is None]
        removed = len(absent) * 3 * self.config.d_model * 768
        removed += sum(self.config.d_model + (isinstance(getattr(block, "residual", None), GatedResidual))
                       for block in absent)
        return {
            "total": self.num_parameters,
            "removed_vs_d_ff_768_all_layers": removed,
            "mha_blocks": mha,
            "deltanet_blocks": delta,
            "ffn_remaining": ffn,
        }


def build_model(config: ModelConfig) -> nn.Module:
    return MODELS.create(config.class_name, config)


ExperimentLM = CausalLM
