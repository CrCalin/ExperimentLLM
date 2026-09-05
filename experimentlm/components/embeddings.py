"""Token embedding components."""
from __future__ import annotations

from torch import nn
from ..registry import EMBEDDINGS

@EMBEDDINGS.register
class TokenEmbedding(nn.Embedding):
    def __init__(self, vocab_size: int, d_model: int, padding_idx: int | None = None,
                 max_norm: float | None = None, norm_type: float = 2.0,
                 scale_grad_by_freq: bool = False, sparse: bool = False):
        super().__init__(vocab_size, d_model, padding_idx=padding_idx, max_norm=max_norm,
                         norm_type=norm_type, scale_grad_by_freq=scale_grad_by_freq, sparse=sparse)

__all__ = ["TokenEmbedding"]

