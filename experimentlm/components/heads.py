"""Output head components."""
from __future__ import annotations

from torch import nn
from ..registry import HEADS

@HEADS.register
class LinearLMHead(nn.Linear):
    def __init__(self, d_model: int, vocab_size: int, bias: bool = False):
        super().__init__(d_model, vocab_size, bias=bias)

__all__ = ["LinearLMHead"]

