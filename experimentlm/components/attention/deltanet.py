"""DeltaNet linear attention."""
from __future__ import annotations

import math
import torch
from torch import nn
import torch.nn.functional as F
from ...parameter_policy import set_optimizer_policy
from ...registry import ATTENTIONS
from ..base import Attention, PositionEncoding
from .common import GatedRMSNorm

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
                 conv_bias: bool = False, norm_eps: float = 1e-5,
                 return_value_route: bool = False):
        super().__init__()
        if n_heads <= 0 or head_dim <= 0 or expand_v <= 0:
            raise ValueError("invalid GatedDeltaNet head configuration")
        self.hidden_size, self.n_heads, self.head_dim = d_model, n_heads, head_dim
        self.expand_v, self.num_v_heads = expand_v, num_v_heads or n_heads
        self.allow_neg_eigval, self.conv_size = allow_neg_eigval, conv_size
        self.return_value_route = return_value_route
        self.chunk_size = 16
        self.key_dim = n_heads * head_dim
        self.head_v_dim = int(head_dim * expand_v)
        self.value_dim = self.num_v_heads * self.head_v_dim
        # FLA's grouped-value path requires the query head count to be an
        # integer multiple of the value head count (e.g. 4/1, 4/2, 4/4).
        if n_heads % self.num_v_heads != 0:
            raise ValueError("n_heads must be divisible by num_v_heads")
        if not math.isclose(head_dim * expand_v, self.head_v_dim):
            raise ValueError("expand_v must produce an integer value head dimension")

        self.projection_sizes = (
            self.key_dim, self.key_dim, self.value_dim,
            self.num_v_heads, self.num_v_heads, self.value_dim,
        )
        self.in_proj_weight = nn.Parameter(torch.empty(sum(self.projection_sizes), d_model))
        set_optimizer_policy(self.in_proj_weight, "adamw", "deltanet_gating_projection")
        nn.init.normal_(self.in_proj_weight, mean=0.0, std=0.02)
        self.A_log = nn.Parameter(torch.empty(self.num_v_heads))
        self.dt_bias = nn.Parameter(torch.empty(self.num_v_heads))
        self.A_log._no_weight_decay = True
        self.dt_bias._no_weight_decay = True
        set_optimizer_policy(self.A_log, "adamw", "deltanet_decay")
        set_optimizer_policy(self.dt_bias, "adamw", "deltanet_gating")
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
        output, state, conv_state, _ = self._forward_impl(
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
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None, torch.Tensor | None]:
        batch, length, _ = x.shape
        new_conv_state = None
        projected = F.linear(x, self.in_proj_weight)
        q, k, v, a, beta, gate = torch.split(projected, self.projection_sizes, dim=-1)
        qkv = projected[..., :2 * self.key_dim + self.value_dim]
        if x.is_cuda and not use_cache and fla_causal_conv1d is None:
            raise RuntimeError("CUDA DeltaNet training requires FLA causal_conv1d")
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
        if self.num_v_heads != self.n_heads:
            # FLA's grouped-value rule has one q/k stream per value state.
            # Collapse each query-head group before dispatching; the baseline
            # (equal query/value heads) remains byte-for-byte unchanged.
            groups = self.n_heads // self.num_v_heads
            q = q.view(batch, length, self.num_v_heads, groups, self.head_dim).mean(dim=3)
            k = k.view(batch, length, self.num_v_heads, groups, self.head_dim).mean(dim=3)
        value_route = v.reshape(batch, length, self.value_dim) if self.return_value_route else None
        if attention_mask is not None:
            if attention_mask.shape != (batch, length):
                raise ValueError("attention_mask must have shape (batch, sequence)")
            mask = attention_mask[:, :, None].to(device=x.device, dtype=beta.dtype)
            beta = beta.masked_fill(mask == 0, -torch.inf)
        expected = (batch, self.num_v_heads, self.head_v_dim, self.head_dim)
        if state is not None and state.shape != expected:
            raise ValueError(f"DeltaNet state must have shape {expected}")
        if x.is_cuda and fused_recurrent_gated_delta_rule is None:
            raise RuntimeError("CUDA DeltaNet execution requires FLA fused_recurrent_gated_delta_rule")
        if x.is_cuda and (torch.is_grad_enabled() or length > 64) and chunk_gated_delta_rule is None:
            raise RuntimeError("CUDA DeltaNet training/long-context execution requires FLA chunk_gated_delta_rule")
        use_fused = x.is_cuda
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
        if y.is_cuda and fla_rms_norm_gated is None:
            raise RuntimeError("CUDA DeltaNet execution requires FLA rms_norm_gated")
        if fla_rms_norm_gated is not None and y.is_cuda:
            y = fla_rms_norm_gated(
                normalized, gate, self.o_norm.weight, None,
                activation="silu", eps=self.o_norm.eps,
            )
        else:
            y = self.o_norm(normalized, gate)
        output = self.o_proj(y.reshape(batch, length, self.value_dim))
        return output, state, new_conv_state, value_route

    def forward_cached(self, x: torch.Tensor, state: torch.Tensor | None = None,
                       conv_state: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None):
        output, state, conv_state, _ = self._forward_impl(x, state, None, True, conv_state, True)
        return output, state, conv_state

    def forward_cached_with_route(self, x: torch.Tensor, state: torch.Tensor | None = None,
                                  conv_state: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None):
        return self._forward_impl(x, state, None, True, conv_state, True)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor | None,
                position_encoding: PositionEncoding) -> torch.Tensor:
        output, _, _, _ = self._forward_impl(x, None, attention_mask, False)
        return output

    def forward_with_route(self, x: torch.Tensor, attention_mask: torch.Tensor | None,
                           position_encoding: PositionEncoding):
        output, _, _, value_route = self._forward_impl(x, None, attention_mask, False)
        return output, value_route

__all__ = ["DeltaNetLinearAttention"]

