"""Registry-built decoder language model and baseline components."""
from __future__ import annotations

from abc import ABC, abstractmethod
import math

import torch
from torch import nn
import torch.nn.functional as F

try:
    from fla.ops.gated_delta_rule import (
        chunk_gated_delta_rule, fused_recurrent_gated_delta_rule,
        naive_recurrent_gated_delta_rule,
    )
    from fla.modules.convolution import causal_conv1d as fla_causal_conv1d
    from fla.modules.fused_norm_gate import rms_norm_gated as fla_rms_norm_gated
except ImportError:
    chunk_gated_delta_rule = fused_recurrent_gated_delta_rule = None
    naive_recurrent_gated_delta_rule = None
    fla_causal_conv1d = None
    fla_rms_norm_gated = None


def _run_chunk_gated_delta_rule(**kwargs):
    return chunk_gated_delta_rule(**kwargs)


def _run_fused_recurrent_gated_delta_rule(**kwargs):
    return fused_recurrent_gated_delta_rule(**kwargs)

from .config import ComponentSpec, ModelConfig
from .registry import (
    ATTENTIONS, BLOCKS, EMBEDDINGS, HEADS, INITIALIZATIONS, MLPS, MODELS,
    NORMS, POSITIONS, RESIDUALS,
)


class PositionEncoding(nn.Module, ABC):
    @abstractmethod
    def apply(self, q: torch.Tensor, k: torch.Tensor,
              position_offset: int = 0) -> tuple[torch.Tensor, torch.Tensor]: ...


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

    def apply(self, q: torch.Tensor, k: torch.Tensor,
              position_offset: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
        length = q.size(-2)
        if position_offset < 0 or position_offset + length > self.context_length:
            raise ValueError("sequence exceeds RoPE context length")
        if self.rope_cos.numel() == 0 or self.rope_cos.device != q.device or self.head_dim != q.size(-1):
            self._build_cache(q.device, q.size(-1))
        cos = self.rope_cos[:, :, position_offset:position_offset + length].to(q.dtype)
        sin = self.rope_sin[:, :, position_offset:position_offset + length].to(q.dtype)
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

    def forward_cached(self, x: torch.Tensor, position_encoding: PositionEncoding,
                       past: tuple[torch.Tensor, torch.Tensor] | None = None):
        batch, length, _ = x.shape
        q, k, v = self.qkv(x).view(batch, length, 3, self.n_heads, self.head_dim).unbind(2)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        offset = 0 if past is None else past[0].size(-2)
        q, k = position_encoding.apply(q, k, offset)
        if past is not None:
            k = torch.cat((past[0], k), dim=-2)
            v = torch.cat((past[1], v), dim=-2)
            y = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        else:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        output = self.out(y.transpose(1, 2).contiguous().view(batch, length, -1))
        return output, (k, v)


class GatedRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps).to(x.dtype)
        return x * rms * self.weight * F.silu(gate)


def _gdn_decay(a: torch.Tensor, A_log: torch.Tensor,
               dt_bias: torch.Tensor) -> torch.Tensor:
    return -torch.exp(A_log).view(1, 1, -1) * F.softplus(
        a + dt_bias.view(1, 1, -1),
    )


def _gdn_recurrence(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    beta: torch.Tensor, g: torch.Tensor, state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference recurrent GDN scan; state is [B,H,V,K]."""
    carry = state.float()
    outputs = []
    for q_t, k_t, v_t, beta_t, g_t in zip(q.transpose(0, 1), k.transpose(0, 1),
                                             v.transpose(0, 1), beta.transpose(0, 1),
                                             g.transpose(0, 1)):
        carry = carry * g_t.float().exp()[..., None, None]
        delta = beta_t.float()[..., None] * (
            v_t.float() - torch.einsum("bhvk,bhk->bhv", carry, k_t.float())
        )
        carry = carry + torch.einsum("bhk,bhv->bhvk", k_t.float(), delta)
        outputs.append(torch.einsum("bhk,bhvk->bhv", q_t.float(), carry))
    return torch.stack(outputs, dim=1).to(v.dtype), carry.to(state.dtype)


@ATTENTIONS.register
class DeltaNetLinearAttention(Attention):
    """Canonical GatedDeltaNet-style attention using FLA's gated kernels."""
    def __init__(self, d_model: int, n_heads: int, head_dim: int = 256,
                 expand_v: float = 2.0, num_v_heads: int | None = None,
                 allow_neg_eigval: bool = False, conv_size: int = 4,
                 conv_bias: bool = False, norm_eps: float = 1e-5):
        super().__init__()
        if n_heads <= 0 or head_dim <= 0 or expand_v <= 0:
            raise ValueError("invalid GatedDeltaNet head configuration")
        self.hidden_size, self.n_heads, self.head_dim = d_model, n_heads, head_dim
        self.expand_v, self.num_v_heads = expand_v, num_v_heads or n_heads
        self.allow_neg_eigval, self.conv_size = allow_neg_eigval, conv_size
        self.chunk_size = 16
        self.key_dim = n_heads * head_dim
        self.head_v_dim = int(head_dim * expand_v)
        self.value_dim = self.num_v_heads * self.head_v_dim
        if self.num_v_heads % n_heads != 0:
            raise ValueError("num_v_heads must be divisible by n_heads")
        if not math.isclose(head_dim * expand_v, self.head_v_dim):
            raise ValueError("expand_v must produce an integer value head dimension")

        self.projection_sizes = (
            self.key_dim, self.key_dim, self.value_dim,
            self.num_v_heads, self.num_v_heads, self.value_dim,
        )
        self.in_proj_weight = nn.Parameter(torch.empty(sum(self.projection_sizes), d_model))
        nn.init.normal_(self.in_proj_weight, mean=0.0, std=0.02)
        self.A_log = nn.Parameter(torch.empty(self.num_v_heads))
        self.dt_bias = nn.Parameter(torch.empty(self.num_v_heads))
        self.A_log._no_weight_decay = True
        self.dt_bias._no_weight_decay = True
        if conv_bias:
            raise ValueError("the optimized GatedDeltaNet path requires conv_bias=false")
        self.conv_weight = nn.Parameter(torch.empty(
            2 * self.key_dim + self.value_dim, 1, conv_size,
        ))
        nn.init.normal_(self.conv_weight, mean=0.0, std=0.02)
        self.o_norm = GatedRMSNorm(self.head_v_dim, norm_eps)
        self.o_proj = nn.Linear(self.value_dim, d_model, bias=False)
        self.o_proj.residual_projection = True
        self.reset_specialized_parameters()

    def reset_specialized_parameters(self):
        with torch.no_grad():
            nn.init.normal_(self.in_proj_weight, mean=0.0, std=0.02)
            nn.init.normal_(self.conv_weight, mean=0.0, std=0.02)
            A = torch.empty(self.num_v_heads, dtype=torch.float32).uniform_(0, 16)
            self.A_log.copy_(torch.log(A))
            dt = torch.exp(torch.empty(self.num_v_heads).uniform_(math.log(0.001), math.log(0.1)))
            self.dt_bias.copy_(dt + torch.log(-torch.expm1(-dt)))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                               missing_keys, unexpected_keys, error_msgs):
        packed_key = prefix + "in_proj_weight"
        if packed_key not in state_dict:
            old_keys = tuple(prefix + name for name in (
                "q_proj.weight", "k_proj.weight", "v_proj.weight",
                "a_proj.weight", "b_proj.weight", "g_proj.weight",
            ))
            if all(key in state_dict for key in old_keys):
                state_dict[packed_key] = torch.cat(tuple(state_dict.pop(key) for key in old_keys), dim=0)
        conv_key = prefix + "conv_weight"
        if conv_key not in state_dict:
            old_keys = tuple(prefix + name for name in (
                "q_conv1d.weight", "k_conv1d.weight", "v_conv1d.weight",
            ))
            if all(key in state_dict for key in old_keys):
                state_dict[conv_key] = torch.cat(tuple(state_dict.pop(key) for key in old_keys), dim=0)
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs,
        )

    def initial_state(self, x: torch.Tensor, dtype: torch.dtype | None = None) -> torch.Tensor:
        return x.new_zeros(x.size(0), self.num_v_heads, self.head_v_dim, self.head_dim,
                           dtype=dtype or x.dtype)

    def forward_recurrent(
        self,
        x: torch.Tensor,
        state: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        conv_state: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None]:
        output, state, conv_state = self._forward_impl(
            x, state, attention_mask, output_final_state=True,
            conv_state=conv_state, use_cache=True,
        )
        if state is None: raise RuntimeError("DeltaNet recurrent state was not returned")
        return output, state, conv_state

    def _fused_short_conv(
        self,
        qkv: torch.Tensor,
        conv_state: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None,
        output_final_state: bool,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None]:
        """Apply the three depthwise short convolutions as one grouped convolution."""
        old = conv_state or (None, None, None)
        sequence = qkv.transpose(1, 2)
        if any(cache is not None for cache in old):
            if any(cache is None for cache in old):
                raise ValueError("all q/k/v convolution caches must be provided together")
            for cache, hidden_size in zip(old, (self.key_dim, self.key_dim, self.value_dim)):
                expected = (qkv.size(0), hidden_size, self.conv_size - 1)
                if cache.shape != expected:
                    raise ValueError(f"convolution cache must have shape {expected}")
            sequence = torch.cat((torch.cat(old, dim=1), sequence), dim=-1)
        y = F.conv1d(
            F.pad(sequence, (self.conv_size - 1, 0)), self.conv_weight, None,
            groups=sequence.size(1),
        )
        y = F.silu(y[..., -qkv.size(1):].transpose(1, 2))
        new_state = None
        if output_final_state:
            cache_size = self.conv_size - 1
            new_sequence = sequence[..., -cache_size:] if cache_size else sequence[..., :0]
            new_state = tuple(torch.split(
                new_sequence, (self.key_dim, self.key_dim, self.value_dim), dim=1,
            ))
        return y, new_state

    def _forward_impl(
        self,
        x: torch.Tensor,
        state: torch.Tensor | None,
        attention_mask: torch.Tensor | None,
        output_final_state: bool,
        conv_state: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None]:
        batch, length, _ = x.shape
        new_conv_state = None
        projected = F.linear(x, self.in_proj_weight)
        q, k, v, a, beta, gate = torch.split(projected, self.projection_sizes, dim=-1)
        qkv = projected[..., :2 * self.key_dim + self.value_dim]
        if fla_causal_conv1d is not None and x.is_cuda and not use_cache and conv_state is None:
            qkv, _ = fla_causal_conv1d(
                qkv, weight=self.conv_weight.squeeze(1), bias=None,
                activation="silu", backend="triton",
            )
        else:
            qkv, new_conv_state = self._fused_short_conv(qkv, conv_state, use_cache)
        q, k, v = torch.split(qkv, (self.key_dim, self.key_dim, self.value_dim), dim=-1)
        q = q.view(batch, length, self.n_heads, self.head_dim)
        k = k.view(batch, length, self.n_heads, self.head_dim)
        v = v.view(batch, length, self.num_v_heads, self.head_v_dim)
        if attention_mask is not None:
            if attention_mask.shape != (batch, length):
                raise ValueError("attention_mask must have shape (batch, sequence)")
            mask = attention_mask[:, :, None].to(device=x.device, dtype=beta.dtype)
            beta = beta.masked_fill(mask == 0, -torch.inf)
        expected = (batch, self.num_v_heads, self.head_v_dim, self.head_dim)
        if state is not None and state.shape != expected:
            raise ValueError(f"DeltaNet state must have shape {expected}")
        use_fused = chunk_gated_delta_rule is not None and x.is_cuda
        use_chunk = use_fused and (torch.is_grad_enabled() or length > 64)
        if use_chunk:
            y, state = _run_chunk_gated_delta_rule(
                q=q, k=k, v=v, g=a, beta=beta, scale=1.0,
                A_log=self.A_log, dt_bias=self.dt_bias, initial_state=state,
                output_final_state=output_final_state, use_qk_l2norm_in_kernel=True,
                use_gate_in_kernel=True, use_beta_sigmoid_in_kernel=True,
                allow_neg_eigval=self.allow_neg_eigval,
                state_v_first=True, chunk_size=self.chunk_size,
            )
        elif use_fused:
            y, state = _run_fused_recurrent_gated_delta_rule(
                q=q, k=k, v=v, g=a, beta=beta, scale=1.0,
                A_log=self.A_log, dt_bias=self.dt_bias, initial_state=state,
                output_final_state=output_final_state, use_qk_l2norm_in_kernel=True,
                use_gate_in_kernel=True, use_beta_sigmoid_in_kernel=True,
                allow_neg_eigval=self.allow_neg_eigval, state_v_first=True,
            )
        elif naive_recurrent_gated_delta_rule is not None:
            state = state if state is not None else self.initial_state(x, dtype=q.dtype)
            decay = _gdn_decay(a, self.A_log, self.dt_bias)
            q_ref, k_ref = F.normalize(q.float(), dim=-1), F.normalize(k.float(), dim=-1)
            y, naive_state = naive_recurrent_gated_delta_rule(
                q=q_ref, k=k_ref, v=v, beta=torch.sigmoid(beta), g=decay, scale=1.0,
                initial_state=state.transpose(-1, -2), output_final_state=output_final_state,
            )
            state = None if naive_state is None else naive_state.transpose(-1, -2)
        else:
            state = state if state is not None else self.initial_state(x, dtype=q.dtype)
            decay = _gdn_decay(a, self.A_log, self.dt_bias)
            y, state = _gdn_recurrence(
                F.normalize(q.float(), dim=-1), F.normalize(k.float(), dim=-1),
                v, torch.sigmoid(beta), decay, state,
            )
        if not output_final_state:
            state = None
        y = y.contiguous().view(batch, length, self.value_dim)
        gate = gate.view(batch, length, self.num_v_heads, self.head_v_dim)
        normalized = y.view(batch, length, self.num_v_heads, self.head_v_dim)
        if fla_rms_norm_gated is not None and y.is_cuda:
            y = fla_rms_norm_gated(
                normalized, gate, self.o_norm.weight, None,
                activation="silu", eps=self.o_norm.eps,
            )
        else:
            y = self.o_norm(normalized, gate)
        output = self.o_proj(y.reshape(batch, length, self.value_dim))
        return output, state, new_conv_state

    def forward_cached(self, x: torch.Tensor, state: torch.Tensor | None = None,
                       conv_state: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None):
        return self._forward_impl(x, state, None, True, conv_state, True)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor | None,
                position_encoding: PositionEncoding) -> torch.Tensor:
        output, _, _ = self._forward_impl(x, None, attention_mask, False)
        return output


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
class GatedResidual(Residual):
    """Sequential pre-norm residual branches with learned scalar gates."""
    def __init__(self, attention_gate_init: float = 1.0, mlp_gate_init: float = 1.0):
        super().__init__()
        self.attention_gate = nn.Parameter(torch.tensor(float(attention_gate_init)))
        self.mlp_gate = nn.Parameter(torch.tensor(float(mlp_gate_init)))

    def forward(self, block: "TransformerBlock", x: torch.Tensor,
                attention_mask: torch.Tensor | None, position_encoding: PositionEncoding) -> torch.Tensor:
        if not block.pre_norm: raise ValueError("GatedResidual requires pre_norm=true")
        x = x + self.attention_gate * block.attn(block.norm1(x), attention_mask, position_encoding)
        return x + self.mlp_gate * block.mlp(block.norm2(x))


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


@BLOCKS.register
class DoubleNormTransformerBlock(Block):
    """Sequential residual block with learned pre- and post-sublayer norms."""
    def __init__(self, d_model: int, pre_attn_norm: dict, post_attn_norm: dict,
                 pre_mlp_norm: dict, post_mlp_norm: dict, attention: dict, mlp: dict):
        super().__init__()
        norm_specs = {
            "pre_attn_norm": ComponentSpec.from_dict(pre_attn_norm, "DoubleNormTransformerBlock.args.pre_attn_norm"),
            "post_attn_norm": ComponentSpec.from_dict(post_attn_norm, "DoubleNormTransformerBlock.args.post_attn_norm"),
            "pre_mlp_norm": ComponentSpec.from_dict(pre_mlp_norm, "DoubleNormTransformerBlock.args.pre_mlp_norm"),
            "post_mlp_norm": ComponentSpec.from_dict(post_mlp_norm, "DoubleNormTransformerBlock.args.post_mlp_norm"),
        }
        self.pre_attn_norm = NORMS.create(norm_specs["pre_attn_norm"].class_name, d_model, **norm_specs["pre_attn_norm"].args)
        self.post_attn_norm = NORMS.create(norm_specs["post_attn_norm"].class_name, d_model, **norm_specs["post_attn_norm"].args)
        self.pre_mlp_norm = NORMS.create(norm_specs["pre_mlp_norm"].class_name, d_model, **norm_specs["pre_mlp_norm"].args)
        self.post_mlp_norm = NORMS.create(norm_specs["post_mlp_norm"].class_name, d_model, **norm_specs["post_mlp_norm"].args)
        attention_spec = ComponentSpec.from_dict(attention, "DoubleNormTransformerBlock.args.attention")
        mlp_spec = ComponentSpec.from_dict(mlp, "DoubleNormTransformerBlock.args.mlp")
        self.attn = ATTENTIONS.create(attention_spec.class_name, d_model, **attention_spec.args)
        self.mlp = MLPS.create(mlp_spec.class_name, d_model, **mlp_spec.args)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor | None,
                position_encoding: PositionEncoding) -> torch.Tensor:
        x = x + self.post_attn_norm(self.attn(
            self.pre_attn_norm(x), attention_mask, position_encoding,
        ))
        return x + self.post_mlp_norm(self.mlp(self.pre_mlp_norm(x)))


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

    def forward_cached(self, input_ids: torch.Tensor, cache: dict | None = None):
        """Run a prefix or one decode step while retaining MHA/GDN states."""
        if input_ids.size(1) > self.config.context_length:
            raise ValueError("sequence exceeds context_length")
        cache = cache or {"length": 0, "layers": [None] * len(self.blocks)}
        if cache["length"] + input_ids.size(1) > self.config.context_length:
            raise ValueError("cached sequence exceeds context_length; rebuild the cache")
        x = self.wte(input_ids)
        new_layers = []
        for block, layer_cache in zip(self.blocks, cache["layers"]):
            if not isinstance(block, TransformerBlock) or block.pre_norm is not True:
                raise ValueError("cached generation requires pre-norm TransformerBlock layers")
            if isinstance(block.attn, MultiHeadCausalSelfAttention):
                attn_out, new_cache = block.attn.forward_cached(
                    block.norm1(x), self.position_encoding, layer_cache,
                )
                x = x + attn_out
                new_layers.append(new_cache)
            elif isinstance(block.attn, DeltaNetLinearAttention):
                state = None if layer_cache is None else layer_cache["state"]
                conv_state = None if layer_cache is None else layer_cache["conv_state"]
                attn_out, state, conv_state = block.attn.forward_cached(x=block.norm1(x), state=state,
                                                                         conv_state=conv_state)
                x = x + attn_out
                new_layers.append({"state": state, "conv_state": conv_state})
            else:
                raise ValueError("cached generation requires known attention implementations")
            x = x + block.mlp(block.norm2(x))
        logits = self.lm_head(self.norm_f(x))
        new_cache = {"length": cache["length"] + input_ids.size(1), "layers": new_layers}
        return logits, new_cache

    @property
    def num_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


def build_model(config: ModelConfig) -> nn.Module:
    return MODELS.create(config.class_name, config)


ExperimentLM = CausalLM
