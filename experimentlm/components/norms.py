"""Normalization components."""
from __future__ import annotations

import torch
from torch import nn
from ..parameter_policy import set_optimizer_policy
from ..registry import NORMS

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

__all__ = ["RMSNorm"]

