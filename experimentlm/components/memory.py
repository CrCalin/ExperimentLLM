"""Auxiliary recurrent square-matrix memory.

The state layout is ``[batch, value, address]``. Runtime states are ordinary
forward values, never parameters or buffers, so independent model calls reset
to zero and checkpoints contain only learned projections.
"""
from __future__ import annotations

import torch
from torch import nn

from ..parameter_policy import set_optimizer_policy
from ..registry import MEMORIES
from .base import Memory


def _unit_direction(x: torch.Tensor, eps: float) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp_min(eps)


def matrix_memory_update(state: torch.Tensor, key: torch.Tensor, erase: torch.Tensor,
                         value: torch.Tensor, alpha: torch.Tensor, beta: torch.Tensor,
                         *, erase_enabled: bool = True,
                         delta_correction: bool = True) -> torch.Tensor:
    """Canonical targeted-erase then corrective rank-one write recurrence."""
    if erase_enabled:
        recalled = torch.bmm(state, erase.unsqueeze(-1)).squeeze(-1)
        state = state - (alpha * recalled).unsqueeze(-1) * erase.unsqueeze(-2)
    predicted = torch.bmm(state, key.unsqueeze(-1)).squeeze(-1)
    delta = value - predicted if delta_correction else value
    return state + (beta * delta).unsqueeze(-1) * key.unsqueeze(-2)


class MatrixMemoryPath(nn.Module):
    """Learned read/write projections for one layer-local memory."""

    def __init__(self, d_model: int, d_memory: int, read_gate_init: float,
                 erase_bias_init: float, write_bias_init: float, eps: float,
                 erase_enabled: bool, delta_correction: bool,
                 tie_erase_to_write: bool, query_attenuation: bool,
                 key_attenuation: bool, erase_attenuation: bool,
                 erase_strength_mode: str, write_strength_mode: str,
                 fused_projections: bool = True):
        super().__init__()
        self.d_memory = d_memory
        self.eps = eps
        self.erase_enabled = erase_enabled
        self.delta_correction = delta_correction
        self.tie_erase_to_write = tie_erase_to_write
        self.query_attenuation = query_attenuation
        self.key_attenuation = key_attenuation
        self.erase_attenuation = erase_attenuation
        self.fused_projections = fused_projections
        if erase_strength_mode not in {"scalar", "vector"}:
            raise ValueError("erase_strength_mode must be scalar or vector")
        if write_strength_mode not in {"scalar", "vector"}:
            raise ValueError("write_strength_mode must be scalar or vector")

        self.alpha_dim = d_memory if erase_strength_mode == "vector" else 1
        self.beta_dim = d_memory if write_strength_mode == "vector" else 1
        if fused_projections:
            self.read_projection = nn.Linear(d_model, 2 * d_memory, bias=False)
            self.read_atten_bias = nn.Parameter(torch.zeros(d_memory))
            self.write_projection = nn.Linear(
                d_model, 5 * d_memory + self.alpha_dim + self.beta_dim, bias=False,
            )
            self.write_bias = nn.Parameter(torch.zeros(
                2 * d_memory + self.alpha_dim + self.beta_dim,
            ))
        else:
            self.q_proj = nn.Linear(d_model, d_memory, bias=False)
            self.q_atten = nn.Linear(d_model, d_memory, bias=True)
            self.k_proj = nn.Linear(d_model, d_memory, bias=False)
            self.k_atten = nn.Linear(d_model, d_memory, bias=True)
            self.e_proj = nn.Linear(d_model, d_memory, bias=False)
            self.e_atten = nn.Linear(d_model, d_memory, bias=True)
            self.v_proj = nn.Linear(d_model, d_memory, bias=False)
            self.erase_strength = nn.Linear(d_model, self.alpha_dim, bias=True)
            self.write_strength = nn.Linear(d_model, self.beta_dim, bias=True)
        self.out_proj = nn.Linear(d_memory, d_model, bias=False)
        self.out_proj.residual_projection = True
        self.read_gate_logit = nn.Parameter(torch.tensor(float(read_gate_init)))
        set_optimizer_policy(self.read_gate_logit, "adamw", "memory_read_gate")
        self._erase_bias_init = float(erase_bias_init)
        self._write_bias_init = float(write_bias_init)

    def reset_memory_initialization(self) -> None:
        """Restore nonzero configured biases after the generic initializer."""
        if self.fused_projections:
            with torch.no_grad():
                self.read_atten_bias.zero_()
                self.write_bias.zero_()
                offset = 2 * self.d_memory
                self.write_bias[offset:offset + self.alpha_dim].fill_(self._erase_bias_init)
                self.write_bias[offset + self.alpha_dim:].fill_(self._write_bias_init)
        else:
            nn.init.constant_(self.erase_strength.bias, self._erase_bias_init)
            nn.init.constant_(self.write_strength.bias, self._write_bias_init)

    def query(self, hidden: torch.Tensor) -> torch.Tensor:
        if self.fused_projections:
            q_hat, attenuation = self.read_projection(hidden).split(self.d_memory, dim=-1)
            attenuation = attenuation + self.read_atten_bias
        else:
            q_hat, attenuation = self.q_proj(hidden), self.q_atten(hidden)
        q = _unit_direction(q_hat, self.eps)
        return q * torch.sigmoid(attenuation) if self.query_attenuation else q

    def read(self, state: torch.Tensor, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        q = self.query(hidden).to(state.dtype)
        read = torch.bmm(state, q.unsqueeze(-1)).squeeze(-1)
        injection = torch.sigmoid(self.read_gate_logit) * self.out_proj(read)
        return injection, read

    def write_vectors(self, hidden: torch.Tensor):
        if self.fused_projections:
            projected = self.write_projection(hidden)
            sizes = [self.d_memory] * 5 + [self.alpha_dim, self.beta_dim]
            k_hat, k_atten, e_hat, e_atten, value, alpha, beta = projected.split(sizes, dim=-1)
            ak_bias, ae_bias, alpha_bias, beta_bias = self.write_bias.split(
                [self.d_memory, self.d_memory, self.alpha_dim, self.beta_dim]
            )
            k_atten = k_atten + ak_bias
            e_atten = e_atten + ae_bias
            alpha = alpha + alpha_bias
            beta = beta + beta_bias
        else:
            k_hat, k_atten = self.k_proj(hidden), self.k_atten(hidden)
            e_hat, e_atten = self.e_proj(hidden), self.e_atten(hidden)
            value = self.v_proj(hidden)
            alpha = self.erase_strength(hidden)
            beta = self.write_strength(hidden)
        k = _unit_direction(k_hat, self.eps)
        if self.key_attenuation:
            k = k * torch.sigmoid(k_atten)
        if self.tie_erase_to_write:
            e = k
        else:
            e = _unit_direction(e_hat, self.eps)
            if self.erase_attenuation:
                e = e * torch.sigmoid(e_atten)
        alpha = torch.sigmoid(alpha)
        beta = torch.sigmoid(beta)
        return k, e, value, alpha, beta

    def write(self, state: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        k, e, value, alpha, beta = self.write_vectors(hidden)
        k, e, value = k.to(state.dtype), e.to(state.dtype), value.to(state.dtype)
        alpha, beta = alpha.to(state.dtype), beta.to(state.dtype)
        return matrix_memory_update(
            state, k, e, value, alpha, beta,
            erase_enabled=self.erase_enabled, delta_correction=self.delta_correction,
        )

    @torch.no_grad()
    def diagnostics(self, state: torch.Tensor, read_hidden: torch.Tensor,
                    write_hidden: torch.Tensor) -> dict[str, torch.Tensor]:
        """Optional reductions; never called by the training hot path."""
        q = self.query(read_hidden)
        _, read = self.read(state, read_hidden)
        k, e, _, alpha, beta = self.write_vectors(write_hidden)
        return {
            "read_gate": torch.sigmoid(self.read_gate_logit),
            "alpha_mean": alpha.mean(), "alpha_std": alpha.std(),
            "alpha_min": alpha.min(), "alpha_max": alpha.max(),
            "beta_mean": beta.mean(), "beta_std": beta.std(),
            "beta_min": beta.min(), "beta_max": beta.max(),
            "query_norm": q.norm(dim=-1).mean(),
            "key_norm": k.norm(dim=-1).mean(),
            "erase_norm": e.norm(dim=-1).mean(),
            "memory_rms": state.float().square().mean().sqrt(),
            "memory_max_abs": state.abs().max(),
            "read_rms": read.float().square().mean().sqrt(),
        }


@MEMORIES.register
class MatrixMemory(Memory):
    """One independent recurrent memory owned by one Transformer layer."""

    def __init__(self, d_model: int, d_memory: int, backend: str = "chunked",
                 read_gate_init: float = -4.0, erase_bias_init: float = -2.0,
                 write_bias_init: float = -2.0, eps: float = 1e-6,
                 erase_enabled: bool = True, delta_correction: bool = True,
                 tie_erase_to_write: bool = False,
                 query_attenuation: bool = True, key_attenuation: bool = True,
                 erase_attenuation: bool = True,
                 erase_strength_mode: str = "vector",
                 write_strength_mode: str = "vector",
                 fused_projections: bool = True,
                 activation_checkpoint: bool = True,
                 chunk_size: int = 16):
        super().__init__()
        if d_memory <= 0: raise ValueError("d_memory must be positive")
        if backend not in {"reference", "scan", "chunked"}:
            raise ValueError("MatrixMemory backend must be reference, scan, or chunked")
        if eps <= 0: raise ValueError("MatrixMemory eps must be positive")
        if chunk_size <= 0: raise ValueError("MatrixMemory chunk_size must be positive")
        self.d_memory = d_memory
        self.backend = backend
        self.activation_checkpoint = activation_checkpoint
        self.chunk_size = chunk_size
        kwargs = dict(
            d_model=d_model, d_memory=d_memory, read_gate_init=read_gate_init,
            erase_bias_init=erase_bias_init, write_bias_init=write_bias_init,
            eps=eps, erase_enabled=erase_enabled, delta_correction=delta_correction,
            tie_erase_to_write=tie_erase_to_write,
            query_attenuation=query_attenuation, key_attenuation=key_attenuation,
            erase_attenuation=erase_attenuation,
            erase_strength_mode=erase_strength_mode,
            write_strength_mode=write_strength_mode,
            fused_projections=fused_projections,
        )
        self.path = MatrixMemoryPath(**kwargs)

    def initial_state(self, batch: int, *, device: torch.device,
                      dtype: torch.dtype) -> torch.Tensor:
        return torch.zeros(batch, self.d_memory, self.d_memory, device=device, dtype=dtype)

    def reset_memory_initialization(self) -> None:
        self.path.reset_memory_initialization()


__all__ = ["MatrixMemory", "MatrixMemoryPath", "matrix_memory_update"]
