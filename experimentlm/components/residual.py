"""Residual strategies."""
from __future__ import annotations

import torch
from torch import nn
from ..parameter_policy import set_optimizer_policy
from ..registry import RESIDUALS
from .base import PositionEncoding, Residual

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

__all__ = ["GatedResidual", "StandardResidual"]

