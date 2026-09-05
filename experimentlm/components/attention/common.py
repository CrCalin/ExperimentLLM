"""Utilities shared by attention implementations."""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from ...parameter_policy import set_optimizer_policy

def _scaled_dot_product_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, **kwargs) -> torch.Tensor:
    if q.is_cuda:
        # Do not silently accept the quadratic math implementation on an
        # accelerator. CPU/reference execution remains available for tests
        # and inference on machines without fused attention kernels.
        with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]):
            return F.scaled_dot_product_attention(q, k, v, **kwargs)
    return F.scaled_dot_product_attention(q, k, v, **kwargs)


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

__all__ = ["GatedRMSNorm", "_scaled_dot_product_attention"]

