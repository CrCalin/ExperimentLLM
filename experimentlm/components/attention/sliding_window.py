"""Sliding-window causal self-attention."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from ...registry import ATTENTIONS
from ..base import PositionEncoding
from .common import _scaled_dot_product_attention
from .gqa import GroupedQueryCausalSelfAttention
from .mha import MultiHeadCausalSelfAttention

try:
    from fla.ops.attn.parallel import parallel_attn as fla_parallel_attn
except ImportError:
    fla_parallel_attn = None

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

__all__ = ["SlidingWindowCausalSelfAttention", "GroupedQuerySlidingWindowCausalSelfAttention"]

