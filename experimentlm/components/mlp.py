"""Feed-forward components."""
from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F
from ..registry import MLPS
from .base import MLP

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



def _inverse_softplus(x: float) -> float:
    return math.log(math.expm1(x))


@MLPS.register
class XIELU(MLP):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        alpha_p: float = 0.8,
        alpha_n: float = 0.8,
        bias: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()

        if alpha_p <= 0: raise ValueError("alpha_p must be > 0")
        if alpha_n <= 0.5: raise ValueError("alpha_n must be > 0.5")

        self.theta_p = nn.Parameter(torch.tensor(_inverse_softplus(alpha_p)))
        self.theta_n = nn.Parameter(torch.tensor(_inverse_softplus(alpha_n - 0.5)))
        self.theta_p._no_weight_decay = True
        self.theta_n._no_weight_decay = True

        self.up = nn.Linear(d_model, d_ff, bias=bias)
        self.down = nn.Linear(d_ff, d_model, bias=bias)
        self.down.residual_projection = True
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        alpha_p = F.softplus(self.theta_p)
        alpha_n = 0.5 + F.softplus(self.theta_n)

        x = self.up(x)

        negative_x = torch.expm1(torch.clamp_max(x, 0.0)) - x

        x = torch.where(
            x > 0,
            alpha_p * x.square() + 0.5 * x,
            alpha_n * negative_x + 0.5 * x,
        )

        return self.dropout(self.down(x))

__all__ = ["SwiGLU", "XIELU"]
