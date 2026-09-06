"""Standalone exact recurrent matrix memory.

This module intentionally has no ``d_model`` dependency.  All inputs are
already projected low-dimensional vectors and the only recurrent value is the
``[batch, value, address]`` matrix state.

The chunked implementations are exact realizations of the canonical recurrence.
The doubled-substep backend uses a compact WY factor, while ``rank2`` uses one
two-column factor per actual token and a block-upper nilpotent inverse.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any

import torch


class MatrixMemoryError(RuntimeError):
    """Base error for standalone matrix-memory failures."""


class BF16TriangularSolveUnavailable(MatrixMemoryError):
    """The selected PyTorch backend cannot run the exact BF16 chunk solve."""


_BF16_SOLVE_SUPPORT: dict[tuple[str, int | None], bool] = {}


def _validate_inputs(
    q: torch.Tensor,
    e: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    initial_state: torch.Tensor | None,
) -> torch.Tensor:
    tensors = (q, e, k, v)
    if any(tensor.ndim != 3 for tensor in tensors):
        raise ValueError("q, e, k, and v must have shape [B, T, d_memory]")
    if not all(tensor.shape == q.shape for tensor in tensors):
        raise ValueError("q, e, k, and v must have identical shapes")
    if not all(tensor.device == q.device for tensor in tensors):
        raise ValueError("q, e, k, and v must be on the same device")
    if not all(tensor.dtype == q.dtype for tensor in tensors):
        raise ValueError("q, e, k, and v must have the same dtype")
    if not q.is_floating_point():
        raise ValueError("q, e, k, and v must use a floating-point dtype")
    if initial_state is None:
        return torch.zeros(q.size(0), q.size(-1), q.size(-1), device=q.device, dtype=q.dtype)
    if initial_state.shape != (q.size(0), q.size(-1), q.size(-1)):
        raise ValueError("initial_state must have shape [B, d_memory, d_memory]")
    if initial_state.device != q.device or initial_state.dtype != q.dtype:
        raise ValueError("initial_state must match q device and dtype")
    return initial_state


def _validate_chunk_size(chunk_size: int) -> None:
    if not isinstance(chunk_size, int) or isinstance(chunk_size, bool) or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")


def sequential_matrix_memory(
    q: torch.Tensor,
    e: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    initial_state: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the canonical read, erase, corrective-write recurrence.

    This is the correctness reference.  It vectorizes over batch but retains
    the intentional Python loop over actual tokens.
    """
    state = _validate_inputs(q, e, k, v, initial_state)
    reads: list[torch.Tensor] = []
    for t in range(q.size(1)):
        q_t, e_t, k_t, v_t = q[:, t], e[:, t], k[:, t], v[:, t]
        read = torch.bmm(state, q_t.unsqueeze(-1)).squeeze(-1)
        reads.append(read)

        erase_value = torch.bmm(state, e_t.unsqueeze(-1)).squeeze(-1)
        state = state - erase_value.unsqueeze(-1) * e_t.unsqueeze(-2)

        write_error = v_t - torch.bmm(state, k_t.unsqueeze(-1)).squeeze(-1)
        state = state + write_error.unsqueeze(-1) * k_t.unsqueeze(-2)
    if not reads:
        reads_tensor = q.new_empty(q.size(0), 0, q.size(-1))
    else:
        reads_tensor = torch.stack(reads, dim=1)
    return reads_tensor, state


def _check_bf16_triangular_solve(device: torch.device) -> None:
    key = (device.type, device.index)
    supported = _BF16_SOLVE_SUPPORT.get(key)
    if supported is None:
        matrix = torch.eye(1, device=device, dtype=torch.bfloat16).unsqueeze(0)
        rhs = torch.ones(1, 1, 1, device=device, dtype=torch.bfloat16)
        try:
            torch.linalg.solve_triangular(matrix, rhs, upper=False)
        except (NotImplementedError, RuntimeError):
            supported = False
        else:
            supported = True
        _BF16_SOLVE_SUPPORT[key] = supported
    if not supported:
        raise BF16TriangularSolveUnavailable(
            "exact chunked matrix memory requires a BF16 triangular solve, "
            f"but torch.linalg.solve_triangular is unavailable on {device}. "
            "The production BF16 chunk backend is therefore blocked; "
            "FP32 is available for diagnostic correctness/performance runs."
        )


def _record(enabled: bool, name: str):
    if not enabled:
        from contextlib import nullcontext
        return nullcontext()
    return torch.profiler.record_function(f"matrix_memory/{name}")


def _chunk(
    q: torch.Tensor,
    e: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    state: torch.Tensor,
    profile: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, tokens, d_memory = q.shape
    with _record(profile, "construct_U_Z"):
        # Each actual token becomes erase (e, 0), then write (k, v).
        u = torch.stack((e, k), dim=2).reshape(batch, 2 * tokens, d_memory)
        zero = torch.zeros_like(e)
        z = torch.stack((zero, v), dim=2).reshape(batch, 2 * tokens, d_memory)

    with _record(profile, "gram"):
        gram = u @ u.transpose(-1, -2)
        lower = torch.tril(gram, diagonal=-1)
        identity = torch.eye(2 * tokens, device=u.device, dtype=u.dtype).expand(batch, -1, -1)
        system = identity + lower

    with _record(profile, "triangular_solve"):
        rhs = z - u @ state.transpose(-1, -2)
        innovations = torch.linalg.solve_triangular(system, rhs, upper=False)

    with _record(profile, "causal_query_product"):
        query_state = q @ state.transpose(-1, -2)
        query_substeps = q @ u.transpose(-1, -2)
        causal = torch.arange(2 * tokens, device=q.device)[None, :] < (
            2 * torch.arange(tokens, device=q.device)[:, None]
        )

    with _record(profile, "masked_read_accumulation"):
        reads = query_state + query_substeps.masked_fill(~causal[None], 0) @ innovations

    with _record(profile, "final_state_update"):
        final_state = state + innovations.transpose(-1, -2) @ u
    return reads, final_state


def chunked_matrix_memory(
    q: torch.Tensor,
    e: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    *,
    chunk_size: int = 64,
    profile: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the exact chunkwise doubled-substep recurrence.

    There is no Python loop over individual tokens here: only chunk boundaries
    are sequential.  On current ROCm/PyTorch builds BF16 is rejected before
    any computation because the required triangular solve is unsupported.
    """
    _validate_chunk_size(chunk_size)
    state = _validate_inputs(q, e, k, v, initial_state)
    if q.dtype == torch.bfloat16:
        _check_bf16_triangular_solve(q.device)
    read_chunks: list[torch.Tensor] = []
    for start in range(0, q.size(1), chunk_size):
        end = min(start + chunk_size, q.size(1))
        reads, state = _chunk(q[:, start:end], e[:, start:end], k[:, start:end],
                              v[:, start:end], state, profile)
        read_chunks.append(reads)
    if not read_chunks:
        reads = q.new_empty(q.size(0), 0, q.size(-1))
    else:
        reads = torch.cat(read_chunks, dim=1)
    return reads, state


def _compact_wy_factor(lower: torch.Tensor) -> torch.Tensor:
    """Build the compact WY/UT factor ``T`` using BF16 GEMMs only.

    Let ``L = tril(U @ U.T, -1)`` and ``N = L.T``.  The product of the
    doubled rank-1 transitions is

        H = prod_j (I - u_j u_j.T) = I - U.T @ T @ U,
        T = (I + N)^-1.

    ``N`` is strictly upper triangular and therefore nilpotent.  The finite
    binary identity

        (I + N)^-1 = (I - N)(I + N^2)(I + N^4)...

    is exact in exact arithmetic.  We materialize only the small chunk factor
    and obtain every power with dense same-dtype matrix products; this avoids
    the unavailable BF16 triangular-solve primitive without changing the
    recurrence.  The loop count is logarithmic in the doubled chunk length,
    never tokenwise.
    """
    return _finite_nilpotent_inverse(lower.transpose(-1, -2))


def _finite_nilpotent_inverse(nilpotent: torch.Tensor) -> torch.Tensor:
    """Return ``(I + N)^-1`` for a strictly triangular ``N``.

    The binary identity is finite, not an iterative approximation:

        (I - N) (I + N**2) (I + N**4) ... = (I + N)^-1.

    It is used for both the doubled rank-1 factor and the rank-2 block factor.
    All products stay in the input dtype, including BF16 production execution.
    """
    length = nilpotent.size(-1)
    identity = torch.eye(length, device=nilpotent.device, dtype=nilpotent.dtype).expand(
        nilpotent.size(0), -1, -1,
    )
    power = nilpotent
    factor = identity - power
    # The factor for N**length is identity because N is strictly triangular.
    # Do not launch the otherwise-useless final ``factor @ identity`` and
    # ``power @ power`` pair.  This saves two dense launches for C=128 while
    # retaining the exact finite binary identity.
    for _ in range(max(0, (length - 1).bit_length() - 1)):
        power = power @ power
        factor = factor @ (identity + power)
    return factor


def _compact_wy_chunk_forward(
    q: torch.Tensor,
    e: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    state: torch.Tensor,
    profile: bool,
    use_triton: bool,
) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, ...]]:
    """Forward one chunk without creating a nested autograd Function.

    This is the fused sequence backend's chunk primitive.  The surrounding
    sequence Function owns the complete backward graph, so chunk boundaries do
    not create one Python autograd node per chunk.  Dense products remain
    vendor GEMMs; the fixed masks are generated as indexing metadata.
    """
    batch, tokens, d_memory = q.shape
    record = bool(profile)
    if use_triton and q.is_cuda and q.dtype == torch.bfloat16:
        with _record(record, "fused_prepare_U_Z"):
            from .matrix_memory_triton import interleave_doubled_bf16
            u, z = interleave_doubled_bf16(e, k, v)
    else:
        with _record(record, "fused_prepare_U_Z"):
            u = torch.stack((e, k), dim=2).reshape(batch, 2 * tokens, d_memory)
            z = torch.stack((torch.zeros_like(e), v), dim=2).reshape(batch, 2 * tokens, d_memory)

    with _record(record, "fused_compact_gram"):
        gram = u @ u.transpose(-1, -2)
        lower = torch.tril(gram, diagonal=-1)
    with _record(record, "fused_compact_factor"):
        factor = _compact_wy_factor(lower)
    with _record(record, "fused_innovations"):
        rhs = z - u @ state.transpose(-1, -2)
        innovations = factor.transpose(-1, -2) @ rhs
    if use_triton and q.is_cuda and q.dtype == torch.bfloat16:
        with _record(record, "fused_rocm_outputs"):
            from .matrix_memory_triton import fused_chunk_outputs_bf16
            reads, final_state = fused_chunk_outputs_bf16(q, u, innovations, state)
    else:
        with _record(record, "fused_causal_read"):
            query_state = q @ state.transpose(-1, -2)
            query_substeps = q @ u.transpose(-1, -2)
            causal = torch.arange(2 * tokens, device=q.device)[None, :] < (
                2 * torch.arange(tokens, device=q.device)[:, None]
            )
            reads = query_state + query_substeps.masked_fill(~causal[None], 0) @ innovations
        with _record(record, "fused_state_update"):
            final_state = state + innovations.transpose(-1, -2) @ u
    return reads, final_state, (q, u, state, factor, innovations)


def _compact_wy_chunk_backward(
    q: torch.Tensor,
    u: torch.Tensor,
    state: torch.Tensor,
    factor: torch.Tensor,
    innovations: torch.Tensor,
    grad_reads: torch.Tensor,
    grad_final_state: torch.Tensor,
    profile: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Analytic adjoint of one exact compact-WY chunk.

    Kept as a plain function so the sequence-level custom Function can run the
    reverse chunk loop without nested autograd bookkeeping.  The equations are
    the same adjoint used by the validated ``wy`` backend.
    """
    batch, tokens, d_memory = q.shape
    length = 2 * tokens
    causal = torch.arange(length, device=q.device)[None, :] < (
        2 * torch.arange(tokens, device=q.device)[:, None]
    )
    with _record(profile, "fused_backward_read"):
        query_substeps = q @ u.transpose(-1, -2)
        masked_queries = query_substeps.masked_fill(~causal[None], 0)
        grad_q = grad_reads @ state
        grad_state = grad_reads.transpose(-1, -2) @ q
        grad_innovations = masked_queries.transpose(-1, -2) @ grad_reads
        grad_masked_queries = (grad_reads @ innovations.transpose(-1, -2)).masked_fill(
            ~causal[None], 0,
        )
        grad_q = grad_q + grad_masked_queries @ u
        grad_u = grad_masked_queries.transpose(-1, -2) @ q

    with _record(profile, "fused_backward_state"):
        grad_state = grad_state + grad_final_state
        grad_innovations = grad_innovations + u @ grad_final_state.transpose(-1, -2)
        grad_u = grad_u + innovations @ grad_final_state

    with _record(profile, "fused_backward_factor"):
        gram = u @ u.transpose(-1, -2)
        lower = torch.tril(gram, diagonal=-1)
        rhs = innovations + lower @ innovations
        grad_rhs = factor @ grad_innovations
        grad_factor = rhs @ grad_innovations.transpose(-1, -2)
        grad_u = grad_u - grad_rhs @ state
        grad_state = grad_state - grad_rhs.transpose(-1, -2) @ u
        grad_v = grad_rhs[:, 1::2]
        grad_n = -factor.transpose(-1, -2) @ grad_factor @ factor.transpose(-1, -2)
        grad_gram = torch.tril(grad_n.transpose(-1, -2), diagonal=-1)
        grad_u = grad_u + (grad_gram + grad_gram.transpose(-1, -2)) @ u

    with _record(profile, "fused_backward_map"):
        grad_e = grad_u[:, 0::2]
        grad_k = grad_u[:, 1::2]
    return grad_q, grad_e, grad_k, grad_v, grad_state


class _FusedWYSequenceFunction(torch.autograd.Function):
    """One sequence-level exact WY op with a dedicated reverse chunk pass.

    The old implementation placed one custom autograd node around every
    chunk.  This implementation keeps the same compact factors but exposes a
    single sequence node, saves only compact chunk tensors, and performs the
    analytic adjoint in reverse chunk order.  It is the baseline for future
    ROCm kernels: no tokenwise host synchronization or d_model work is added.
    """

    @staticmethod
    def forward(ctx, q, e, k, v, state, chunk_size, profile=False, use_triton=False):
        read_chunks = []
        saved = []
        for start in range(0, q.size(1), chunk_size):
            end = min(start + chunk_size, q.size(1))
            reads, state, chunk_saved = _compact_wy_chunk_forward(
                q[:, start:end], e[:, start:end], k[:, start:end], v[:, start:end],
                state, profile, use_triton,
            )
            read_chunks.append(reads)
            saved.extend(chunk_saved)
        ctx.chunk_size = chunk_size
        ctx.chunk_count = len(read_chunks)
        ctx.profile = bool(profile)
        ctx.save_for_backward(*saved)
        if not read_chunks:
            reads = q.new_empty(q.size(0), 0, q.size(-1))
        else:
            reads = torch.cat(read_chunks, dim=1)
        return reads, state

    @staticmethod
    def backward(ctx, grad_reads, grad_final_state):
        saved = list(ctx.saved_tensors)
        chunks = [saved[i:i + 5] for i in range(0, len(saved), 5)]
        if not chunks:
            return None, None, None, None, grad_final_state, None, None, None
        q0, _, state0, _, _ = chunks[0]
        batch, d_memory = q0.shape[0], q0.shape[-1]
        total_tokens = 0
        for chunk in chunks:
            total_tokens += chunk[0].shape[1]
        if grad_reads is None:
            grad_reads = torch.zeros(batch, total_tokens, d_memory, device=q0.device, dtype=q0.dtype)
        if grad_final_state is None:
            grad_final_state = torch.zeros_like(chunks[-1][2])
        grad_q = torch.zeros(batch, total_tokens, d_memory, device=q0.device, dtype=q0.dtype)
        grad_e = torch.zeros_like(grad_q)
        grad_k = torch.zeros_like(grad_q)
        grad_v = torch.zeros_like(grad_q)
        grad_state = grad_final_state
        offset = total_tokens
        for q, u, state, factor, innovations in reversed(chunks):
            tokens = q.shape[1]
            offset -= tokens
            chunk_grads = _compact_wy_chunk_backward(
                q, u, state, factor, innovations,
                grad_reads[:, offset:offset + tokens], grad_state, ctx.profile,
            )
            grad_q[:, offset:offset + tokens] = chunk_grads[0]
            grad_e[:, offset:offset + tokens] = chunk_grads[1]
            grad_k[:, offset:offset + tokens] = chunk_grads[2]
            grad_v[:, offset:offset + tokens] = chunk_grads[3]
            grad_state = chunk_grads[4]
        return grad_q, grad_e, grad_k, grad_v, grad_state, None, None, None


class _CompactWYChunkFunction(torch.autograd.Function):
    """Exact compact-WY chunk with a compact analytic backward.

    Forward saves q, U, the incoming state, T, and P.  In particular, it does
    not retain the binary power chain used to construct T.  For
    ``T=(I+N)^-1``, its differential is ``dT=-T dN T``; this gives the exact
    adjoint of the factor without rebuilding the factor in backward.
    """

    @staticmethod
    def forward(ctx, q, e, k, v, state, profile=False, use_triton=False):
        batch, tokens, d_memory = q.shape
        record = bool(profile)
        if use_triton and q.is_cuda and q.dtype == torch.bfloat16:
            with _record(record, "triton_fused_U_Z"):
                from .matrix_memory_triton import interleave_doubled_bf16
                u, z = interleave_doubled_bf16(e, k, v)
        else:
            with _record(record, "construct_U_Z"):
                u = torch.stack((e, k), dim=2).reshape(batch, 2 * tokens, d_memory)
                z = torch.stack((torch.zeros_like(e), v), dim=2).reshape(batch, 2 * tokens, d_memory)

        with _record(record, "compact_gram"):
            gram = u @ u.transpose(-1, -2)
            lower = torch.tril(gram, diagonal=-1)

        with _record(record, "compact_wy_factor"):
            factor = _compact_wy_factor(lower)

        with _record(record, "transformed_innovations"):
            rhs = z - u @ state.transpose(-1, -2)
            innovations = factor.transpose(-1, -2) @ rhs

        with _record(record, "causal_query_product"):
            query_state = q @ state.transpose(-1, -2)
            query_substeps = q @ u.transpose(-1, -2)
            causal = torch.arange(2 * tokens, device=q.device)[None, :] < (
                2 * torch.arange(tokens, device=q.device)[:, None]
            )

        with _record(record, "masked_read_accumulation"):
            reads = query_state + query_substeps.masked_fill(~causal[None], 0) @ innovations

        with _record(record, "final_state_update"):
            final_state = state + innovations.transpose(-1, -2) @ u

        ctx.tokens = tokens
        ctx.save_for_backward(q, u, state, factor, innovations)
        return reads, final_state

    @staticmethod
    def backward(ctx, grad_reads, grad_final_state):
        q, u, state, factor, innovations = ctx.saved_tensors
        batch, tokens, d_memory = q.shape
        length = 2 * tokens
        if grad_reads is None:
            grad_reads = torch.zeros(batch, tokens, d_memory, device=q.device, dtype=q.dtype)
        if grad_final_state is None:
            grad_final_state = torch.zeros_like(state)

        # reads = q state.T + mask(q U.T) P
        query_substeps = q @ u.transpose(-1, -2)
        causal = torch.arange(length, device=q.device)[None, :] < (
            2 * torch.arange(tokens, device=q.device)[:, None]
        )
        masked_queries = query_substeps.masked_fill(~causal[None], 0)
        grad_q = grad_reads @ state
        grad_state = grad_reads.transpose(-1, -2) @ q
        grad_innovations = masked_queries.transpose(-1, -2) @ grad_reads
        grad_masked_queries = (grad_reads @ innovations.transpose(-1, -2)).masked_fill(
            ~causal[None], 0,
        )
        grad_q = grad_q + grad_masked_queries @ u
        grad_u = grad_masked_queries.transpose(-1, -2) @ q

        # final_state = state + P.T U
        grad_state = grad_state + grad_final_state
        grad_innovations = grad_innovations + u @ grad_final_state.transpose(-1, -2)
        grad_u = grad_u + innovations @ grad_final_state

        # P = T.T rhs and T^{-T}=I+L, with L=tril(UU.T,-1). Recover rhs
        # from saved P/U, avoiding storage of Z and the power-chain tensors.
        gram = u @ u.transpose(-1, -2)
        lower = torch.tril(gram, diagonal=-1)
        rhs = innovations + lower @ innovations
        grad_rhs = factor @ grad_innovations
        grad_factor = rhs @ grad_innovations.transpose(-1, -2)
        grad_u = grad_u - grad_rhs @ state
        grad_state = grad_state - grad_rhs.transpose(-1, -2) @ u
        grad_v = grad_rhs[:, 1::2]

        # T=(I+N)^-1, N=tril(UU.T,-1).T, so dT=-T dN T.
        grad_n = -factor.transpose(-1, -2) @ grad_factor @ factor.transpose(-1, -2)
        grad_gram = torch.tril(grad_n.transpose(-1, -2), diagonal=-1)
        grad_u = grad_u + (grad_gram + grad_gram.transpose(-1, -2)) @ u

        grad_e = grad_u[:, 0::2]
        grad_k = grad_u[:, 1::2]
        return grad_q, grad_e, grad_k, grad_v, grad_state, None, None


def _compact_wy_chunk(
    q: torch.Tensor,
    e: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    state: torch.Tensor,
    profile: bool,
    use_triton: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _CompactWYChunkFunction.apply(q, e, k, v, state, profile, use_triton)


def compact_wy_matrix_memory(
    q: torch.Tensor,
    e: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    *,
    chunk_size: int = 64,
    profile: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the exact compact-WY/UT backend using the input dtype throughout."""
    _validate_chunk_size(chunk_size)
    state = _validate_inputs(q, e, k, v, initial_state)
    read_chunks: list[torch.Tensor] = []
    for start in range(0, q.size(1), chunk_size):
        end = min(start + chunk_size, q.size(1))
        reads, state = _compact_wy_chunk(
            q[:, start:end], e[:, start:end], k[:, start:end], v[:, start:end],
            state, profile,
        )
        read_chunks.append(reads)
    if not read_chunks:
        reads = q.new_empty(q.size(0), 0, q.size(-1))
    else:
        reads = torch.cat(read_chunks, dim=1)
    return reads, state


def fused_wy_matrix_memory(
    q: torch.Tensor,
    e: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    *,
    chunk_size: int = 64,
    profile: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the fused exact doubled-WY sequence operator.

    Forward and backward are owned by one sequence-level custom operator. It
    keeps compact factors per chunk, uses the optional ROCm U/Z preparation
    kernel, and performs the analytic adjoint in one reverse chunk pass. The
    recurrence itself is unchanged; only orchestration and saved-graph
    overhead are removed.
    """
    _validate_chunk_size(chunk_size)
    state = _validate_inputs(q, e, k, v, initial_state)
    # Always exercise the ROCm fused kernels for the BF16 backend.  This keeps
    # forward-only benchmarks and stability tests honest; vendor output GEMMs
    # remain available through the validated ``wy`` backend for comparison.
    use_triton = q.is_cuda and q.dtype == torch.bfloat16
    return _FusedWYSequenceFunction.apply(
        q, e, k, v, state, chunk_size, profile, use_triton,
    )


def _rank2_token_factors(
    e: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build horizontal ``U``, ``V``, and ``Z`` for one-token rank-2 updates.

    For a token, ``U=[e,k]``, ``V=[e-(e.T k)k,k]``, and ``Z=[0,v]``.
    Columns are interleaved as ``[token0-col0, token0-col1, ...]`` so a
    chunk's block-causal mask is simply ``(column // 2) < token``.
    """
    c = (e * k).sum(dim=-1, keepdim=True)
    u = torch.stack((e, k), dim=-1).permute(0, 2, 1, 3).reshape(e.size(0), e.size(-1), -1)
    v_factor = torch.stack((e - c * k, k), dim=-1).permute(
        0, 2, 1, 3,
    ).reshape(e.size(0), e.size(-1), -1)
    z = torch.stack((torch.zeros_like(v), v), dim=-1).permute(
        0, 2, 1, 3,
    ).reshape(e.size(0), e.size(-1), -1)
    return u, v_factor, z


def _rank2_causal_mask(
    tokens: int,
    device: torch.device,
) -> torch.Tensor:
    """Return [C, 2C] mask allowing only complete earlier token blocks."""
    token_indices = torch.arange(tokens, device=device)
    column_tokens = torch.arange(2 * tokens, device=device) // 2
    return column_tokens[None, :] < token_indices[:, None]


def _rank2_block_upper_mask(tokens: int, device: torch.device) -> torch.Tensor:
    """Return [2C, 2C] mask for strictly earlier token blocks."""
    token_indices = torch.arange(2 * tokens, device=device) // 2
    column_tokens = torch.arange(2 * tokens, device=device) // 2
    return token_indices[:, None] < column_tokens[None, :]


class _Rank2ChunkFunction(torch.autograd.Function):
    """Exact rank-2/block-WY chunk with compact analytic backward.

    The chunk transition is ``M_out=M_in+P V.T`` with
    ``P=(Z-M_in U) F`` and ``F=(I+N)^-1``.  Here ``N`` is strictly
    block-upper-triangular: its off-diagonal 2x2 token block (i,t) is
    ``V_i.T @ U_t`` and its diagonal blocks are zero.  The finite binary
    nilpotent inverse avoids BF16 triangular solve.  Causal reads use the
    same ``P`` and mask all two columns of the current token.

    Saved tensors are q, U, V, incoming state, F, and P: one chunk factor and
    O(T*d) token factors, never tokenwise d-by-d states or the power chain.
    """

    @staticmethod
    def forward(ctx, q, e, k, v, state, profile=False):
        batch, tokens, d_memory = q.shape
        record = bool(profile)
        with _record(record, "rank2_construct_UVZ"):
            u, v_factor, z = _rank2_token_factors(e, k, v)

        with _record(record, "rank2_block_gram"):
            block_gram = v_factor.transpose(-1, -2) @ u
            read_mask = _rank2_causal_mask(tokens, q.device)
            block_mask = _rank2_block_upper_mask(tokens, q.device)
            nilpotent = block_gram.masked_fill(~block_mask[None], 0)

        with _record(record, "rank2_nilpotent_inverse"):
            factor = _finite_nilpotent_inverse(nilpotent)

        with _record(record, "rank2_innovations"):
            rhs = z - state @ u
            innovations = rhs @ factor

        with _record(record, "rank2_causal_read"):
            query_state = q @ state.transpose(-1, -2)
            scores = q @ v_factor
            reads = query_state + scores.masked_fill(~read_mask[None], 0) @ innovations.transpose(-1, -2)

        with _record(record, "rank2_final_state"):
            final_state = state + innovations @ v_factor.transpose(-1, -2)

        ctx.tokens = tokens
        ctx.profile = record
        ctx.save_for_backward(q, u, v_factor, state, factor, innovations)
        return reads, final_state

    @staticmethod
    def backward(ctx, grad_reads, grad_final_state):
        q, u, v_factor, state, factor, innovations = ctx.saved_tensors
        batch, tokens, d_memory = q.shape
        length = 2 * tokens
        if grad_reads is None:
            grad_reads = torch.zeros(batch, tokens, d_memory, device=q.device, dtype=q.dtype)
        if grad_final_state is None:
            grad_final_state = torch.zeros_like(state)

        read_mask = _rank2_causal_mask(tokens, q.device)
        block_mask = _rank2_block_upper_mask(tokens, q.device)

        with _record(ctx.profile, "rank2_backward_read"):
            # reads = q @ state.T + mask(q @ V) @ P.T
            scores = q @ v_factor
            masked_scores = scores.masked_fill(~read_mask[None], 0)
            grad_q = grad_reads @ state
            grad_state = grad_reads.transpose(-1, -2) @ q
            grad_scores = grad_reads @ innovations
            grad_innovations = (masked_scores.transpose(-1, -2) @ grad_reads).transpose(-1, -2)
            grad_masked_scores = grad_scores.masked_fill(~read_mask[None], 0)
            grad_q = grad_q + grad_masked_scores @ v_factor.transpose(-1, -2)
            grad_v_factor = q.transpose(-1, -2) @ grad_masked_scores

        with _record(ctx.profile, "rank2_backward_final_state"):
            # final_state = state + P @ V.T
            grad_state = grad_state + grad_final_state
            grad_innovations = grad_innovations + grad_final_state @ v_factor
            grad_v_factor = grad_v_factor + grad_final_state.transpose(-1, -2) @ innovations

        with _record(ctx.profile, "rank2_backward_factor"):
            # P = rhs @ F, rhs = Z - state @ U.  Recover rhs using F^-1=I+N.
            gram = v_factor.transpose(-1, -2) @ u
            nilpotent = gram.masked_fill(~block_mask[None], 0)
            rhs = innovations + innovations @ nilpotent
            grad_rhs = grad_innovations @ factor.transpose(-1, -2)
            grad_factor = rhs.transpose(-1, -2) @ grad_innovations
            grad_state = grad_state - grad_rhs @ u.transpose(-1, -2)
            grad_u = -state.transpose(-1, -2) @ grad_rhs
            grad_z = grad_rhs

            # F=(I+N)^-1, dF=-F dN F.
            grad_n = -factor.transpose(-1, -2) @ grad_factor @ factor.transpose(-1, -2)
            grad_n = grad_n.masked_fill(~block_mask[None], 0)
            grad_v_factor = grad_v_factor + u @ grad_n.transpose(-1, -2)
            grad_u = grad_u + v_factor @ grad_n

        with _record(ctx.profile, "rank2_backward_factor_map"):
            # Z=[0,v], U=[e,k], V=[e-c*k,k], c=e.T k.
            grad_v = grad_z[:, :, 1::2].transpose(-1, -2)
            grad_u_e = grad_u[:, :, 0::2].transpose(-1, -2)
            grad_u_k = grad_u[:, :, 1::2].transpose(-1, -2)
            grad_v_e = grad_v_factor[:, :, 0::2].transpose(-1, -2)
            grad_v_k = grad_v_factor[:, :, 1::2].transpose(-1, -2)
            e_input = u[:, :, 0::2].transpose(-1, -2)
            k_input = u[:, :, 1::2].transpose(-1, -2)
            c = (e_input * k_input).sum(dim=-1, keepdim=True)
            dot = (grad_v_e * k_input).sum(dim=-1, keepdim=True)
            grad_e = grad_u_e + grad_v_e - dot * k_input
            grad_k = grad_u_k + grad_v_k - dot * e_input - c * grad_v_e
        return grad_q, grad_e, grad_k, grad_v, grad_state, None


def rank2_matrix_memory(
    q: torch.Tensor,
    e: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    *,
    chunk_size: int = 64,
    profile: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the exact BF16-compatible rank-2/block-WY backend.

    Each actual token is one rank-2 transition.  The implementation receives
    only low-dimensional q/e/k/v tensors and preserves READ -> ERASE -> WRITE.
    """
    _validate_chunk_size(chunk_size)
    state = _validate_inputs(q, e, k, v, initial_state)
    read_chunks: list[torch.Tensor] = []
    for start in range(0, q.size(1), chunk_size):
        end = min(start + chunk_size, q.size(1))
        reads, state = _Rank2ChunkFunction.apply(
            q[:, start:end], e[:, start:end], k[:, start:end], v[:, start:end], state, profile,
        )
        read_chunks.append(reads)
    if not read_chunks:
        reads = q.new_empty(q.size(0), 0, q.size(-1))
    else:
        reads = torch.cat(read_chunks, dim=1)
    return reads, state


def matrix_memory(
    q: torch.Tensor,
    e: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    *,
    backend: str = "chunked",
    chunk_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select the standalone reference or exact chunked backend."""
    if backend == "reference":
        return sequential_matrix_memory(q, e, k, v, initial_state)
    if backend == "chunked":
        return chunked_matrix_memory(q, e, k, v, initial_state, chunk_size=chunk_size)
    if backend == "wy":
        return compact_wy_matrix_memory(q, e, k, v, initial_state, chunk_size=chunk_size)
    if backend == "fused_wy":
        return fused_wy_matrix_memory(q, e, k, v, initial_state, chunk_size=chunk_size)
    if backend == "rank2":
        return rank2_matrix_memory(q, e, k, v, initial_state, chunk_size=chunk_size)
    raise ValueError("backend must be 'reference', 'chunked', 'wy', 'fused_wy', or 'rank2'")


@dataclass(frozen=True)
class MatrixMemoryBenchmarkConfig:
    """Strict JSON configuration for the standalone benchmark runner."""

    batch_sizes: tuple[int, ...] = (1, 8)
    sequence_lengths: tuple[int, ...] = (512, 2048, 4096, 8192, 16384)
    memory_dimensions: tuple[int, ...] = (16, 32, 64, 128, 256)
    chunk_sizes: tuple[int, ...] = (16, 32, 64, 128)
    warmup: int = 2
    iterations: int = 5
    dtype: str = "bfloat16"
    device: str = "cuda"
    backward: bool = True
    compile: bool = False
    profile: bool = False
    backends: tuple[str, ...] = ("reference", "chunked", "wy", "fused_wy", "rank2")
    output: str = "benchmark_results/matrix_memory.json"

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "MatrixMemoryBenchmarkConfig":
        allowed = set(cls.__dataclass_fields__)
        unknown = set(value) - allowed
        if unknown:
            raise ValueError(f"unknown matrix-memory benchmark keys: {sorted(unknown)}")
        kwargs = dict(value)
        for name in ("batch_sizes", "sequence_lengths", "memory_dimensions", "chunk_sizes"):
            if name in kwargs:
                item = kwargs[name]
                if not isinstance(item, list) or not item or not all(
                    isinstance(x, int) and not isinstance(x, bool) and x > 0 for x in item
                ):
                    raise ValueError(f"{name} must be a non-empty list of positive integers")
                kwargs[name] = tuple(item)
        result = cls(**kwargs)
        allowed_backends = {"reference", "chunked", "wy", "fused_wy", "rank2"}
        if not isinstance(result.backends, (list, tuple)) or not result.backends:
            raise ValueError("backends must be a non-empty list")
        if any(backend not in allowed_backends for backend in result.backends):
            raise ValueError(f"backends must be drawn from {sorted(allowed_backends)}")
        result = cls(**{**asdict(result), "backends": tuple(result.backends)})
        if result.dtype not in {"float32", "bfloat16"}:
            raise ValueError("dtype must be float32 or bfloat16")
        for name in ("backward", "compile", "profile"):
            if not isinstance(getattr(result, name), bool):
                raise ValueError(f"{name} must be boolean")
        if not isinstance(result.device, str) or not result.device:
            raise ValueError("device must be a non-empty string")
        if not isinstance(result.output, str) or not result.output:
            raise ValueError("output must be a non-empty string")
        for name in ("warmup", "iterations"):
            if not isinstance(getattr(result, name), int) or getattr(result, name) < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if result.iterations == 0:
            raise ValueError("iterations must be positive")
        return result

    @classmethod
    def load(cls, path: str | Path) -> "MatrixMemoryBenchmarkConfig":
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("benchmark config must be an object")
        if set(value) == {"class", "args"}:
            if value["class"] != "MatrixMemoryBenchmark":
                raise ValueError("benchmark config class must be MatrixMemoryBenchmark")
            value = value["args"]
        return cls.from_dict(value)

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        for name in ("batch_sizes", "sequence_lengths", "memory_dimensions", "chunk_sizes", "backends"):
            value[name] = list(value[name])
        return {"class": "MatrixMemoryBenchmark", "args": value}


__all__ = [
    "BF16TriangularSolveUnavailable",
    "MatrixMemoryBenchmarkConfig",
    "MatrixMemoryError",
    "chunked_matrix_memory",
    "compact_wy_matrix_memory",
    "fused_wy_matrix_memory",
    "matrix_memory",
    "rank2_matrix_memory",
    "sequential_matrix_memory",
]
