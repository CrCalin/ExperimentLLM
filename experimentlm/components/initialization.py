"""Parameter initialization components."""
from __future__ import annotations

import math
from torch import nn
from ..registry import INITIALIZATIONS

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
        for module in model.modules():
            if hasattr(module, "reset_memory_initialization"):
                module.reset_memory_initialization()

__all__ = ["GPTStyleInit"]
