"""Gated DeltaNet v2 linear attention."""
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
    from fla.modules.convolution import causal_conv1d as fla_causal_conv1d
    from fla.ops.gdn2 import chunk_gdn2, fused_recurrent_gdn2, naive_recurrent_gdn2
except ImportError:
    fla_causal_conv1d = None
    chunk_gdn2 = fused_recurrent_gdn2 = naive_recurrent_gdn2 = None

@ATTENTIONS.register
class GatedDeltaNet2LinearAttention(Attention):
    """FLA Gated DeltaNet-2 mixer with project-compatible cache handling."""
    def __init__(self, d_model: int, n_heads: int = 16, head_dim: int = 128,
                 expand_v: float = 1.0, num_v_heads: int | None = None,
                 mode: str = "chunk", use_short_conv: bool = True,
                 allow_neg_eigval: bool = False, conv_size: int = 4,
                 conv_bias: bool = False, norm_eps: float = 1e-5,
                 lower_bound: float | None = None, safe_gate: bool = False):
        super().__init__()
        if mode not in {"chunk", "fused_recurrent"}:
            raise ValueError("GDN2 mode must be chunk or fused_recurrent")
        if n_heads <= 0 or head_dim <= 0 or expand_v <= 0:
            raise ValueError("invalid GDN2 head configuration")
        self.hidden_size, self.n_heads, self.head_dim = d_model, n_heads, head_dim
        self.expand_v, self.num_v_heads = expand_v, num_v_heads or n_heads
        self.mode, self.use_short_conv = mode, use_short_conv
        self.allow_neg_eigval, self.conv_size = allow_neg_eigval, conv_size
        self.lower_bound, self.safe_gate = lower_bound, safe_gate
        self.key_dim = n_heads * head_dim
        self.head_v_dim = int(head_dim * expand_v)
        self.value_dim = self.num_v_heads * self.head_v_dim
        if self.num_v_heads < n_heads or self.num_v_heads % n_heads:
            raise ValueError("num_v_heads must be >= and divisible by n_heads")
        if not math.isclose(head_dim * expand_v, self.head_v_dim):
            raise ValueError("expand_v must produce an integer value head dimension")

        self.projection_sizes = (self.key_dim, self.key_dim, self.value_dim)
        self.qkv_proj_weight = nn.Parameter(torch.empty(sum(self.projection_sizes), d_model))
        nn.init.normal_(self.qkv_proj_weight, mean=0.0, std=0.02)
        if use_short_conv and not conv_bias:
            self.qkv_conv_weight = nn.Parameter(torch.empty(
                2 * self.key_dim + self.value_dim, 1, conv_size,
            ))
            nn.init.normal_(self.qkv_conv_weight, mean=0.0, std=0.02)
        elif use_short_conv:
            self.q_conv1d = nn.Conv1d(self.key_dim, self.key_dim, conv_size,
                                      groups=self.key_dim, bias=conv_bias)
            self.k_conv1d = nn.Conv1d(self.key_dim, self.key_dim, conv_size,
                                      groups=self.key_dim, bias=conv_bias)
            self.v_conv1d = nn.Conv1d(self.value_dim, self.value_dim, conv_size,
                                      groups=self.value_dim, bias=conv_bias)
        self.fg_proj = nn.Linear(d_model, 2 * self.head_v_dim, bias=False)
        self.f_out_proj = nn.Linear(self.head_v_dim, self.key_dim, bias=False)
        self.bw_proj_weight = nn.Parameter(torch.empty(self.key_dim + self.value_dim, d_model))
        nn.init.normal_(self.bw_proj_weight, mean=0.0, std=0.02)
        self.g_out_proj = nn.Linear(self.head_v_dim, self.value_dim, bias=True)
        for module, reason in (
            (self.fg_proj, "gdn2_decay_projection"),
            (self.f_out_proj, "gdn2_decay_projection"),
            (self.g_out_proj, "gdn2_output_gate"),
        ):
            for parameter in module.parameters():
                set_optimizer_policy(parameter, "adamw", reason)
        set_optimizer_policy(self.bw_proj_weight, "adamw", "gdn2_erase_write_gates")
        self.A_log = nn.Parameter(torch.log(torch.empty(n_heads, dtype=torch.float32).uniform_(1, 16)))
        self.dt_bias = nn.Parameter(torch.empty(self.key_dim, dtype=torch.float32))
        self.A_log._no_weight_decay = True
        self.dt_bias._no_weight_decay = True
        set_optimizer_policy(self.A_log, "adamw", "gdn2_decay")
        set_optimizer_policy(self.dt_bias, "adamw", "gdn2_decay_bias")
        self.o_norm = GatedRMSNorm(self.head_v_dim, norm_eps, activation="sigmoid")
        self.o_proj = nn.Linear(self.value_dim, d_model, bias=False)
        self.o_proj.residual_projection = True
        self.reset_specialized_parameters()

    def reset_specialized_parameters(self):
        with torch.no_grad():
            dt = torch.exp(torch.empty(self.key_dim).uniform_(math.log(0.001), math.log(0.1))).clamp(min=1e-4)
            self.dt_bias.copy_(dt + torch.log(-torch.expm1(-dt)))

    def initial_state(self, x: torch.Tensor, dtype: torch.dtype | None = None) -> torch.Tensor:
        # FLA's state_v_first layout is [batch, value_head, value_dim, key_dim].
        return x.new_zeros(x.size(0), self.num_v_heads, self.head_v_dim, self.head_dim,
                           dtype=torch.float32)

    def _short_conv(self, x: torch.Tensor, conv: nn.Conv1d,
                    cache: torch.Tensor | None, output_final_state: bool):
        weight = conv.weight.squeeze(1)
        if x.is_cuda:
            if fla_causal_conv1d is None:
                raise RuntimeError("CUDA GDN2 execution requires FLA causal_conv1d")
            return fla_causal_conv1d(
                x, weight=weight, bias=conv.bias, initial_state=cache,
                output_final_state=output_final_state, activation="silu", backend="triton",
            )
        sequence = x.transpose(1, 2)
        if cache is not None:
            sequence = torch.cat((cache, sequence), dim=-1)
        y = F.conv1d(F.pad(sequence, (self.conv_size - 1, 0)), conv.weight, conv.bias,
                     groups=sequence.size(1))
        y = F.silu(y[..., -x.size(1):].transpose(1, 2))
        new_cache = sequence[..., -self.conv_size:]
        return y, new_cache if output_final_state else None

    def _prepare_recurrence_inputs(self, x: torch.Tensor,
                                   attention_mask: torch.Tensor | None = None,
                                   pack_projections: bool = False):
        """Build the GDN2 streams shared by writer-only and writer-reader paths."""
        batch, length, _ = x.shape
        if pack_projections:
            # The three projections consume the same input.  At the small
            # model widths used here, one larger GEMM is cheaper than three
            # independent dispatches.  ``cat`` preserves parameter identity
            # and therefore normal gradient accumulation/checkpoint names.
            qkv_size = sum(self.projection_sizes)
            projected = F.linear(x, torch.cat((
                self.qkv_proj_weight, self.fg_proj.weight, self.bw_proj_weight,
            ), dim=0))
            projected, fg, bw = torch.split(
                projected, (qkv_size, 2 * self.head_v_dim,
                            self.key_dim + self.value_dim), dim=-1,
            )
        else:
            projected = F.linear(x, self.qkv_proj_weight)
            fg = self.fg_proj(x)
            bw = F.linear(x, self.bw_proj_weight)
        q, k, v = torch.split(projected, self.projection_sizes, dim=-1)
        if self.use_short_conv:
            if not hasattr(self, "qkv_conv_weight"):
                raise RuntimeError("writer-reader GDN2 currently requires packed short-conv weights")
            # ``projected`` is already laid out as the packed QKV stream.
            # Avoid split -> cat and its full-width temporary before the
            # depthwise convolution.
            qkv = projected
            if x.is_cuda:
                if fla_causal_conv1d is None:
                    raise RuntimeError("CUDA GDN2 execution requires FLA causal_conv1d")
                qkv, _ = fla_causal_conv1d(
                    qkv, weight=self.qkv_conv_weight.squeeze(1), bias=None,
                    initial_state=None, output_final_state=False,
                    activation="silu", backend="triton",
                )
            else:
                sequence = qkv.transpose(1, 2)
                y = F.conv1d(F.pad(sequence, (self.conv_size - 1, 0)), self.qkv_conv_weight,
                             None, groups=sequence.size(1))
                qkv = F.silu(y.transpose(1, 2))
            q, k, v = torch.split(qkv, self.projection_sizes, dim=-1)
        else:
            q, k, v = F.silu(q), F.silu(k), F.silu(v)
        f_hidden, g_hidden = torch.split(fg, self.head_v_dim, dim=-1)
        f = self.f_out_proj(f_hidden).float().view(batch, length, self.n_heads, self.head_dim)
        g = -self.A_log.float().exp().view(1, 1, self.n_heads, 1) * F.softplus(
            f + self.dt_bias.view(1, 1, self.n_heads, self.head_dim),
        )
        q = q.view(batch, length, self.n_heads, self.head_dim)
        k = k.view(batch, length, self.n_heads, self.head_dim)
        v = v.view(batch, length, self.num_v_heads, self.head_v_dim)
        b, w = torch.split(bw, (self.key_dim, self.value_dim), dim=-1)
        b = b.sigmoid().view(batch, length, self.n_heads, self.head_dim)
        w = w.sigmoid().view(batch, length, self.num_v_heads, self.head_v_dim)
        if self.num_v_heads > self.n_heads:
            groups = self.num_v_heads // self.n_heads
            q, k, g, b = (t.repeat_interleave(groups, dim=2) for t in (q, k, g, b))
        if self.allow_neg_eigval:
            b = b * 2.0
        if attention_mask is not None:
            mask = attention_mask.to(device=x.device, dtype=q.dtype)[:, :, None, None]
            q, k, v, g, b, w = (t * mask for t in (q, k, v, g, b, w))
        return q, k, v, g, b, w, g_hidden

    def _forward_impl(self, x: torch.Tensor, state: torch.Tensor | None,
                      attention_mask: torch.Tensor | None, output_final_state: bool,
                      conv_state=None, use_cache: bool = False,
                      return_recurrence_inputs: bool = False):
        batch, length, _ = x.shape
        if attention_mask is not None and attention_mask.shape != (batch, length):
            raise ValueError("attention_mask must have shape (batch, sequence)")
        projected = F.linear(x, self.qkv_proj_weight)
        q, k, v = torch.split(projected, self.projection_sizes, dim=-1)
        new_conv_state = None
        if self.use_short_conv:
            old = conv_state or (None, None, None)
            if hasattr(self, "qkv_conv_weight"):
                # The projection output already has the exact packed QKV
                # layout expected by the depthwise short convolution.
                qkv = projected
                if x.is_cuda:
                    if fla_causal_conv1d is None:
                        raise RuntimeError("CUDA GDN2 execution requires FLA causal_conv1d")
                    qkv, packed_state = fla_causal_conv1d(
                        qkv, weight=self.qkv_conv_weight.squeeze(1), bias=None,
                        initial_state=None if old[0] is None else torch.cat(old, dim=1),
                        output_final_state=output_final_state, activation="silu", backend="triton",
                    )
                else:
                    sequence = qkv.transpose(1, 2)
                    if old[0] is not None:
                        sequence = torch.cat((torch.cat(old, dim=1), sequence), dim=-1)
                    y = F.conv1d(F.pad(sequence, (self.conv_size - 1, 0)), self.qkv_conv_weight, None,
                                 groups=sequence.size(1))
                    qkv = F.silu(y[..., -length:].transpose(1, 2))
                    packed_state = sequence[..., -self.conv_size:] if output_final_state else None
                q, k, v = torch.split(qkv, self.projection_sizes, dim=-1)
                new_conv_state = None if packed_state is None else tuple(
                    torch.split(packed_state, (self.key_dim, self.key_dim, self.value_dim), dim=1)
                )
            else:
                q, cq = self._short_conv(q, self.q_conv1d, old[0], output_final_state)
                k, ck = self._short_conv(k, self.k_conv1d, old[1], output_final_state)
                v, cv = self._short_conv(v, self.v_conv1d, old[2], output_final_state)
                new_conv_state = (cq, ck, cv)
        else:
            q, k, v = F.silu(q), F.silu(k), F.silu(v)
        fg = self.fg_proj(x)
        f_hidden, g_hidden = torch.split(fg, self.head_v_dim, dim=-1)
        f = self.f_out_proj(f_hidden).float().view(batch, length, self.n_heads, self.head_dim)
        g = -self.A_log.float().exp().view(1, 1, self.n_heads, 1) * F.softplus(
            f + self.dt_bias.view(1, 1, self.n_heads, self.head_dim),
        )
        q = q.view(batch, length, self.n_heads, self.head_dim)
        k = k.view(batch, length, self.n_heads, self.head_dim)
        v = v.view(batch, length, self.num_v_heads, self.head_v_dim)
        bw = F.linear(x, self.bw_proj_weight)
        b, w = torch.split(bw, (self.key_dim, self.value_dim), dim=-1)
        b = b.sigmoid().view(batch, length, self.n_heads, self.head_dim)
        w = w.sigmoid().view(batch, length, self.num_v_heads, self.head_v_dim)
        if self.num_v_heads > self.n_heads:
            groups = self.num_v_heads // self.n_heads
            q, k, g, b = (t.repeat_interleave(groups, dim=2) for t in (q, k, g, b))
        if self.allow_neg_eigval:
            b = b * 2.0
        if attention_mask is not None:
            mask = attention_mask.to(device=x.device, dtype=q.dtype)[:, :, None, None]
            q, k, v, g, b, w = (t * mask for t in (q, k, v, g, b, w))
        if state is not None:
            expected = (batch, self.num_v_heads, self.head_v_dim, self.head_dim)
            if state.shape != expected:
                raise ValueError(f"GDN2 state must have shape {expected}")
            if state.dtype != torch.float32:
                state = state.float()
        if x.is_cuda:
            if chunk_gdn2 is None or fused_recurrent_gdn2 is None:
                raise RuntimeError("CUDA GDN2 execution requires FLA GDN2 kernels")
            mode = "chunk" if torch.is_grad_enabled() else (
                "fused_recurrent" if length <= 64 else self.mode
            )
            if mode == "chunk":
                y, state = chunk_gdn2(q, k, v, g, b, w, initial_state=state,
                                      output_final_state=output_final_state,
                                      use_qk_l2norm_in_kernel=True, state_v_first=True,
                                      safe_gate=self.safe_gate, lower_bound=self.lower_bound)
            else:
                y, state = fused_recurrent_gdn2(q, k, v, g, b, w, initial_state=state,
                                                output_final_state=output_final_state,
                                                use_qk_l2norm_in_kernel=True, state_v_first=True,
                                                lower_bound=self.lower_bound)
        else:
            if naive_recurrent_gdn2 is None:
                raise RuntimeError("GDN2 reference implementation is unavailable")
            state = state if state is not None else self.initial_state(x)
            y, state = naive_recurrent_gdn2(q, k, v, g, b, w, initial_state=state,
                                            output_final_state=output_final_state)
        gate = self.g_out_proj(g_hidden).view(batch, length, self.num_v_heads, self.head_v_dim)
        y = self.o_norm(y, gate).reshape(batch, length, self.value_dim)
        result = self.o_proj(y), (state if output_final_state else None), new_conv_state
        if return_recurrence_inputs:
            result = result + ((q, k, v, g, b, w),)
        return result

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor | None,
                position_encoding: PositionEncoding) -> torch.Tensor:
        output, _, _ = self._forward_impl(x, None, attention_mask, False)
        return output

    def forward_cached_with_route(self, x: torch.Tensor, state=None, conv_state=None):
        return self._forward_impl(x, state, None, True, conv_state, True) + (None,)

__all__ = ["GatedDeltaNet2LinearAttention"]

