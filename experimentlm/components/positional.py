"""Position encoding components."""
from __future__ import annotations

import torch
from ..registry import POSITIONS
from .base import PositionEncoding

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

__all__ = ["RoPE"]

