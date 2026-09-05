"""Abstract component interfaces."""
from __future__ import annotations

from abc import ABC, abstractmethod
import torch
from torch import nn

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


class Memory(nn.Module, ABC):
    """Auxiliary recurrent memory owned by a Transformer block."""


class Residual(nn.Module, ABC):
    @abstractmethod
    def forward(self, block: "TransformerBlock", x: torch.Tensor,
                attention_mask: torch.Tensor | None, position_encoding: PositionEncoding) -> torch.Tensor: ...


class Block(nn.Module, ABC):
    @abstractmethod
    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor | None,
                position_encoding: PositionEncoding) -> torch.Tensor: ...

__all__ = ["Attention", "Block", "Memory", "MLP", "PositionEncoding", "Residual"]
