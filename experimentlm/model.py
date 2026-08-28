"""Registry-built decoder language model and baseline components."""
from __future__ import annotations

from abc import ABC, abstractmethod
import math

import torch
from torch import nn
import torch.nn.functional as F

from .config import ComponentSpec, ModelConfig
from .registry import (
    ATTENTIONS, BLOCKS, EMBEDDINGS, HEADS, INITIALIZATIONS, MLPS, MODELS,
    NORMS, POSITIONS, RESIDUALS,
)


class PositionEncoding(nn.Module, ABC):
    @abstractmethod
    def apply(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]: ...


class Attention(nn.Module, ABC):
    @abstractmethod
    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor | None,
                position_encoding: PositionEncoding) -> torch.Tensor: ...


class MLP(nn.Module, ABC):
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...


class Residual(nn.Module, ABC):
    @abstractmethod
    def forward(self, block: "TransformerBlock", x: torch.Tensor,
                attention_mask: torch.Tensor | None, position_encoding: PositionEncoding) -> torch.Tensor: ...


class Block(nn.Module, ABC):
    @abstractmethod
    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor | None,
                position_encoding: PositionEncoding) -> torch.Tensor: ...


@NORMS.register
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        if eps <= 0: raise ValueError("RMSNorm eps must be positive")
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps).to(x.dtype) * self.weight


@EMBEDDINGS.register
class TokenEmbedding(nn.Embedding):
    def __init__(self, vocab_size: int, d_model: int, padding_idx: int | None = None,
                 max_norm: float | None = None, norm_type: float = 2.0,
                 scale_grad_by_freq: bool = False, sparse: bool = False):
        super().__init__(vocab_size, d_model, padding_idx=padding_idx, max_norm=max_norm,
                         norm_type=norm_type, scale_grad_by_freq=scale_grad_by_freq, sparse=sparse)


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

    def apply(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        length = q.size(-2)
        if length > self.context_length: raise ValueError("sequence exceeds RoPE context length")
        if self.rope_cos.numel() == 0 or self.rope_cos.device != q.device or self.head_dim != q.size(-1):
            self._build_cache(q.device, q.size(-1))
        cos = self.rope_cos[:, :, :length].to(q.dtype)
        sin = self.rope_sin[:, :, :length].to(q.dtype)
        return q * cos + _rotate_half(q) * sin, k * cos + _rotate_half(k) * sin


@ATTENTIONS.register
class MultiHeadCausalSelfAttention(Attention):
    def __init__(self, d_model: int, n_heads: int, head_dim: int | None = None,
                 bias: bool = False, dropout: float = 0.0):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim or d_model // n_heads
        if n_heads <= 0 or self.head_dim <= 0 or n_heads * self.head_dim != d_model:
            raise ValueError("attention requires n_heads * head_dim == d_model")
        if not 0 <= dropout < 1: raise ValueError("attention dropout must be in [0, 1)")
        self.dropout = dropout
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=bias)
        self.out = nn.Linear(d_model, d_model, bias=bias)
        self.out.residual_projection = True

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor | None,
                position_encoding: PositionEncoding) -> torch.Tensor:
        batch, length, _ = x.shape
        q, k, v = self.qkv(x).view(batch, length, 3, self.n_heads, self.head_dim).unbind(2)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        q, k = position_encoding.apply(q, k)
        dropout = self.dropout if self.training else 0.0
        if attention_mask is None:
            y = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout, is_causal=True)
        else:
            causal = torch.ones((length, length), device=x.device, dtype=torch.bool).tril()
            valid_keys = attention_mask.to(device=x.device, dtype=torch.bool)[:, None, None, :]
            y = F.scaled_dot_product_attention(
                q, k, v, attn_mask=causal[None, None] & valid_keys,
                dropout_p=dropout, is_causal=False,
            )
        return self.out(y.transpose(1, 2).contiguous().view(batch, length, -1))


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


@RESIDUALS.register
class StandardResidual(Residual):
    """Conventional sequential attention and MLP residual branches."""
    def __init__(self): super().__init__()

    def forward(self, block: "TransformerBlock", x: torch.Tensor,
                attention_mask: torch.Tensor | None, position_encoding: PositionEncoding) -> torch.Tensor:
        if block.pre_norm:
            x = x + block.attn(block.norm1(x), attention_mask, position_encoding)
            return x + block.mlp(block.norm2(x))
        x = block.norm1(x + block.attn(x, attention_mask, position_encoding))
        return block.norm2(x + block.mlp(x))


@RESIDUALS.register
class ParallelResidual(Residual):
    """Original ExperimentLM baseline: both branches consume the same residual stream."""
    def __init__(self): super().__init__()

    def forward(self, block: "TransformerBlock", x: torch.Tensor,
                attention_mask: torch.Tensor | None, position_encoding: PositionEncoding) -> torch.Tensor:
        if not block.pre_norm: raise ValueError("ParallelResidual currently requires pre_norm=true")
        return x + block.attn(block.norm1(x), attention_mask, position_encoding) + block.mlp(block.norm2(x))


@BLOCKS.register
class TransformerBlock(Block):
    def __init__(self, d_model: int, pre_norm: bool, norm: dict, attention: dict,
                 mlp: dict, residual: dict):
        super().__init__()
        self.pre_norm = pre_norm
        norm_spec = ComponentSpec.from_dict(norm, "TransformerBlock.args.norm")
        attention_spec = ComponentSpec.from_dict(attention, "TransformerBlock.args.attention")
        mlp_spec = ComponentSpec.from_dict(mlp, "TransformerBlock.args.mlp")
        residual_spec = ComponentSpec.from_dict(residual, "TransformerBlock.args.residual")
        self.norm1 = NORMS.create(norm_spec.class_name, d_model, **norm_spec.args)
        self.attn = ATTENTIONS.create(attention_spec.class_name, d_model, **attention_spec.args)
        self.norm2 = NORMS.create(norm_spec.class_name, d_model, **norm_spec.args)
        self.mlp = MLPS.create(mlp_spec.class_name, d_model, **mlp_spec.args)
        self.residual = RESIDUALS.create(residual_spec.class_name, **residual_spec.args)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor | None,
                position_encoding: PositionEncoding) -> torch.Tensor:
        return self.residual(self, x, attention_mask, position_encoding)


@HEADS.register
class LinearLMHead(nn.Linear):
    def __init__(self, d_model: int, vocab_size: int, bias: bool = False):
        super().__init__(d_model, vocab_size, bias=bias)


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
        if self.scale_residual_projections:
            scale = 1 / math.sqrt(2 * n_layers)
            for module in model.modules():
                if getattr(module, "residual_projection", False): module.weight.data.mul_(scale)


@MODELS.register
class CausalLM(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.wte = EMBEDDINGS.create(
            config.input_embedding.class_name, config.vocab_size, config.d_model,
            **config.input_embedding.args,
        )
        self.position_encoding = POSITIONS.create(
            config.position_encoding.class_name, config.context_length,
            **config.position_encoding.args,
        )
        self.blocks = nn.ModuleList([
            BLOCKS.create(layer.class_name, config.d_model, **layer.args) for layer in config.layers
        ])
        self.norm_f = NORMS.create(config.final_norm.class_name, config.d_model, **config.final_norm.args)
        self.lm_head = HEADS.create(
            config.output_head.class_name, config.d_model, config.vocab_size,
            **config.output_head.args,
        )
        if config.tie_embeddings:
            if self.lm_head.weight.shape != self.wte.weight.shape:
                raise ValueError("input embedding and output head weights cannot be tied")
            self.lm_head.weight = self.wte.weight
        initializer = INITIALIZATIONS.create(config.initialization.class_name, **config.initialization.args)
        initializer.apply(self, len(self.blocks))

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor | None = None,
                attention_mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:
        if input_ids.size(1) > self.config.context_length: raise ValueError("sequence exceeds context_length")
        x = self.wte(input_ids)
        for block in self.blocks:
            x = block(x, attention_mask, self.position_encoding)
        logits = self.lm_head(self.norm_f(x))
        loss = None if labels is None else F.cross_entropy(
            logits.float().reshape(-1, logits.size(-1)), labels.reshape(-1)
        )
        return logits, loss

    @property
    def num_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


def build_model(config: ModelConfig) -> nn.Module:
    return MODELS.create(config.class_name, config)


ExperimentLM = CausalLM
