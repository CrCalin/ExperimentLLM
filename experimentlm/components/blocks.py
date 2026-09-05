"""Transformer block components."""
from __future__ import annotations

import torch
from ..config import ComponentSpec
from ..registry import ATTENTIONS, BLOCKS, MEMORIES, MLPS, NORMS, RESIDUALS, ROUTINGS
from .base import Block, PositionEncoding

@BLOCKS.register
class TransformerBlock(Block):
    def __init__(self, d_model: int, pre_norm: bool, norm: dict, attention: dict,
                 mlp: dict | None, residual: dict, value_routing: dict | None = None,
                 d_model_override: int | None = None):
        super().__init__()
        d_model = d_model if d_model_override is None else d_model_override
        if d_model <= 0:
            raise ValueError("TransformerBlock d_model_override must be positive")
        self.model_dim = d_model
        self.pre_norm = pre_norm
        norm_spec = ComponentSpec.from_dict(norm, "TransformerBlock.args.norm")
        attention_spec = ComponentSpec.from_dict(attention, "TransformerBlock.args.attention")
        residual_spec = ComponentSpec.from_dict(residual, "TransformerBlock.args.residual")
        self.norm1 = NORMS.create(norm_spec.class_name, d_model, **norm_spec.args)
        self.attn = ATTENTIONS.create(attention_spec.class_name, d_model, **attention_spec.args)
        self.value_routing = None
        if value_routing is not None:
            route_spec = ComponentSpec.from_dict(value_routing, "TransformerBlock.args.value_routing")
            if not hasattr(self.attn, "value_dim"):
                raise ValueError("value routing requires an attention module with value_dim")
            self.value_routing = ROUTINGS.create(
                route_spec.class_name, self.attn.value_dim, d_model, **route_spec.args,
            )
        self.norm2 = None
        self.mlp = None
        if mlp is not None:
            mlp_spec = ComponentSpec.from_dict(mlp, "TransformerBlock.args.mlp")
            self.norm2 = NORMS.create(norm_spec.class_name, d_model, **norm_spec.args)
            self.mlp = MLPS.create(mlp_spec.class_name, d_model, **mlp_spec.args)
        residual_args = dict(residual_spec.args)
        if mlp is None and residual_spec.class_name == "GatedResidual":
            residual_args["mlp_gate_init"] = None
        self.residual = RESIDUALS.create(residual_spec.class_name, **residual_args)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor | None,
                position_encoding: PositionEncoding) -> torch.Tensor:
        return self.residual(self, x, attention_mask, position_encoding)

    def attention_with_route(self, x, attention_mask, position_encoding):
        attn, route = self.attn.forward_with_route(x, attention_mask, position_encoding)
        return attn, route

    def add_value_route(self, x, route, placement):
        if self.value_routing is not None and self.value_routing.placement == placement:
            return x + self.value_routing(route)
        return x


@BLOCKS.register
class MatrixMemoryTransformerBlock(TransformerBlock):
    """Pre-norm Transformer block with one independent local matrix memory."""

    def __init__(self, d_model: int, pre_norm: bool, norm: dict, attention: dict,
                 mlp: dict, residual: dict, memory: dict,
                 value_routing: dict | None = None,
                 d_model_override: int | None = None):
        residual_name = ComponentSpec.from_dict(
            residual, "MatrixMemoryTransformerBlock.args.residual"
        ).class_name
        if residual_name not in {"StandardResidual", "GatedResidual"}:
            raise ValueError("MatrixMemoryTransformerBlock requires a sequential residual")
        super().__init__(d_model, pre_norm, norm, attention, mlp, residual,
                         value_routing, d_model_override)
        if not pre_norm:
            raise ValueError("MatrixMemoryTransformerBlock requires pre_norm=true")
        if self.mlp is None:
            raise ValueError("MatrixMemoryTransformerBlock requires an MLP")
        if value_routing is not None:
            raise ValueError("MatrixMemoryTransformerBlock does not support value_routing")
        memory_spec = ComponentSpec.from_dict(memory, "MatrixMemoryTransformerBlock.args.memory")
        self.memory = MEMORIES.create(memory_spec.class_name, self.model_dim, **memory_spec.args)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor | None,
                position_encoding: PositionEncoding) -> torch.Tensor:
        output, _ = self.forward_with_memory_state(x, attention_mask, position_encoding)
        return output

    @staticmethod
    def _runtime_dtype(x: torch.Tensor) -> torch.dtype:
        if torch.is_autocast_enabled(x.device.type):
            return torch.get_autocast_dtype(x.device.type)
        return x.dtype

    def initial_memory_state(self, x: torch.Tensor) -> torch.Tensor:
        return self.memory.initial_state(
            x.size(0), device=x.device, dtype=self._runtime_dtype(x),
        )

    def memory_step(self, post_attention: torch.Tensor,
                    state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Read old state, run MLP, write raw MLP output, then add residual."""
        injection, _ = self.memory.path.read(state, post_attention)
        mlp_input = post_attention + injection
        raw_mlp = self.mlp(self.norm2(mlp_input))
        state = self.memory.path.write(state, raw_mlp)
        mlp_gate = getattr(self.residual, "mlp_gate", 1.0)
        output = mlp_input + mlp_gate * raw_mlp
        return output, state

    def forward_with_memory_state(self, x: torch.Tensor,
                                  attention_mask: torch.Tensor | None,
                                  position_encoding: PositionEncoding,
                                  initial_state: torch.Tensor | None = None,
                                  backend: str | None = None):
        attn, _ = self.attention_with_route(self.norm1(x), attention_mask, position_encoding)
        gate = getattr(self.residual, "attention_gate", 1.0)
        post_attention = x + gate * attn
        state = self.initial_memory_state(x) if initial_state is None else initial_state
        selected = backend or self.memory.backend
        if selected == "reference":
            return self._memory_reference(post_attention, state)
        if selected == "chunked":
            return self._memory_chunked(post_attention, state)
        if selected != "scan":
            raise ValueError("matrix-memory backend must be reference, scan, or chunked")
        if self.training and torch.is_grad_enabled() and self.memory.activation_checkpoint:
            from torch.utils.checkpoint import checkpoint
            return checkpoint(
                self._memory_scan, post_attention, state, use_reentrant=True,
            )
        return self._memory_scan(post_attention, state)

    @torch.compile(fullgraph=True)
    def _memory_chunk(self, tokens: torch.Tensor, state: torch.Tensor):
        outputs = []
        for token in tokens.unbind(dim=1):
            output, state = self.memory_step(token, state)
            outputs.append(output)
        return torch.stack(outputs, dim=1), state

    @torch.compiler.disable
    def _memory_chunked(self, post_attention: torch.Tensor, state: torch.Tensor):
        outputs = []
        for tokens in post_attention.split(self.memory.chunk_size, dim=1):
            if self.training and torch.is_grad_enabled() and self.memory.activation_checkpoint:
                from torch.utils.checkpoint import checkpoint
                output, state = checkpoint(
                    self._memory_chunk, tokens, state, use_reentrant=True,
                )
            else:
                output, state = self._memory_chunk(tokens, state)
            outputs.append(output)
        return torch.cat(outputs, dim=1), state

    def _memory_reference(self, post_attention: torch.Tensor, state: torch.Tensor):
        outputs = []
        for token in post_attention.unbind(dim=1):
            output, state = self.memory_step(token, state)
            outputs.append(output)
        return torch.stack(outputs, dim=1), state

    @torch.compiler.disable
    def _memory_scan(self, post_attention: torch.Tensor, state: torch.Tensor):
        # PyTorch 2.14 Inductor cannot yet lower scan inside an outer compiled
        # graph. The scan still compiles its recurrent body and contains no
        # Python token loop at runtime.
        from torch._higher_order_ops.scan import scan

        def step(carry, token):
            output, updated = self.memory_step(token, carry)
            return updated, output.clone()

        state, output = scan(step, state, post_attention.transpose(0, 1))
        return output.transpose(0, 1), state


@BLOCKS.register
class FFNOnlyBlock(Block):
    """Explicit pre-norm feed-forward residual block without a mixer."""
    def __init__(self, d_model: int, norm: dict, mlp: dict):
        super().__init__()
        norm_spec = ComponentSpec.from_dict(norm, "FFNOnlyBlock.args.norm")
        mlp_spec = ComponentSpec.from_dict(mlp, "FFNOnlyBlock.args.mlp")
        self.norm = NORMS.create(norm_spec.class_name, d_model, **norm_spec.args)
        self.mlp = MLPS.create(mlp_spec.class_name, d_model, **mlp_spec.args)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor | None,
                position_encoding: PositionEncoding) -> torch.Tensor:
        return x + self.mlp(self.norm(x))

__all__ = ["FFNOnlyBlock", "MatrixMemoryTransformerBlock", "TransformerBlock"]
