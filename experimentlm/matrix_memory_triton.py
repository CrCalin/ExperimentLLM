"""ROCm Triton kernels for the standalone matrix-memory backend.

Dense chunk factor GEMMs remain vendor/PyTorch operations because they are
larger and better served by the ROCm matmul implementation.  The kernels here
fuse the output-side causal contraction and state update after those factors
are available, avoiding separate query products, masking, and accumulation
temporaries.
"""
from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - optional accelerator dependency
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def _interleave_doubled_kernel(
        e, k, value, u, z,
        B, T, D,
        stride_eb, stride_et, stride_ed,
        stride_kb, stride_kt, stride_kd,
        stride_vb, stride_vt, stride_vd,
        stride_ub, stride_ut, stride_ud,
        stride_zb, stride_zt, stride_zd,
        BD: tl.constexpr,
    ):
        pid = tl.program_id(0)
        ndb = tl.cdiv(D, BD)
        block = pid % ndb
        token = (pid // ndb) % T
        batch = pid // (ndb * T)
        d = block * BD + tl.arange(0, BD)
        mask = d < D

        e_ptr = e + batch * stride_eb + token * stride_et + d * stride_ed
        k_ptr = k + batch * stride_kb + token * stride_kt + d * stride_kd
        v_ptr = value + batch * stride_vb + token * stride_vt + d * stride_vd
        e_value = tl.load(e_ptr, mask=mask, other=0.0)
        k_value = tl.load(k_ptr, mask=mask, other=0.0)
        v_value = tl.load(v_ptr, mask=mask, other=0.0)

        u_erase = u + batch * stride_ub + (2 * token) * stride_ut + d * stride_ud
        u_write = u + batch * stride_ub + (2 * token + 1) * stride_ut + d * stride_ud
        z_erase = z + batch * stride_zb + (2 * token) * stride_zt + d * stride_zd
        z_write = z + batch * stride_zb + (2 * token + 1) * stride_zt + d * stride_zd
        tl.store(u_erase, e_value, mask=mask)
        tl.store(u_write, k_value, mask=mask)
        tl.store(z_erase, 0.0, mask=mask)
        tl.store(z_write, v_value, mask=mask)


    @triton.jit
    def _fused_read_kernel(
        q, u, innovations, state, reads,
        B, T, D,
        stride_qb, stride_qt, stride_qd,
        stride_ub, stride_ut, stride_ud,
        stride_pb, stride_pt, stride_pd,
        stride_sb, stride_sm, stride_sn,
        stride_rb, stride_rt, stride_rd,
        BD: tl.constexpr,
        BLOCK_L: tl.constexpr,
    ):
        pid = tl.program_id(0)
        ndb = tl.cdiv(D, BD)
        d_block = pid % ndb
        token = (pid // ndb) % T
        batch = pid // (ndb * T)
        d_out = d_block * BD + tl.arange(0, BD)
        d_mask = d_out < D
        q_ptr = q + batch * stride_qb + token * stride_qt

        # Match the BF16 GEMM boundaries of the validated PyTorch formula:
        # each dense contraction is rounded to BF16 before the residual add.
        query_acc = tl.zeros((BD,), dtype=tl.float32)
        for start in range(0, D, BD):
            d = start + tl.arange(0, BD)
            qv = tl.load(q_ptr + d * stride_qd, mask=d < D, other=0.0)
            sm = state + batch * stride_sb + d_out[:, None] * stride_sm + d[None, :] * stride_sn
            sv = tl.load(sm, mask=d_mask[:, None] & (d[None, :] < D), other=0.0)
            query_acc += tl.sum(sv * qv[None, :], axis=1)
        query_value = query_acc.to(tl.bfloat16)

        # Sum_{j < 2t} (q @ U_j) P_j.  The current token's two positions
        # are excluded structurally by the substep predicate.
        contribution_acc = tl.zeros((BD,), dtype=tl.float32)
        for start in range(0, 2 * T, BLOCK_L):
            sub = start + tl.arange(0, BLOCK_L)
            sub_mask = sub < 2 * T
            causal = sub < 2 * token
            score = tl.zeros((BLOCK_L,), dtype=tl.float32)
            for d_start in range(0, D, BD):
                d = d_start + tl.arange(0, BD)
                qv = tl.load(q_ptr + d * stride_qd, mask=d < D, other=0.0)
                up = u + batch * stride_ub + sub[:, None] * stride_ut + d[None, :] * stride_ud
                uv = tl.load(up, mask=sub_mask[:, None] & (d[None, :] < D), other=0.0)
                score += tl.sum(uv * qv[None, :], axis=1)
            pp = innovations + batch * stride_pb + sub[:, None] * stride_pt + d_out[None, :] * stride_pd
            pv = tl.load(pp, mask=sub_mask[:, None] & d_mask[None, :], other=0.0)
            contribution_acc += tl.sum((score * (causal & sub_mask))[:, None] * pv, axis=0)
        contribution_value = contribution_acc.to(tl.bfloat16)
        output = query_value + contribution_value
        tl.store(reads + batch * stride_rb + token * stride_rt + d_out * stride_rd, output, mask=d_mask)


    @triton.jit
    def _fused_state_update_kernel(
        u, innovations, state, final_state,
        B, T, D,
        stride_ub, stride_ut, stride_ud,
        stride_pb, stride_pt, stride_pd,
        stride_sb, stride_sm, stride_sn,
        stride_ob, stride_om, stride_on,
        BM: tl.constexpr,
        BN: tl.constexpr,
        BLOCK_L: tl.constexpr,
    ):
        pid = tl.program_id(0)
        nmb = tl.cdiv(D, BM)
        nnb = tl.cdiv(D, BN)
        col_block = pid % nnb
        row_block = (pid // nnb) % nmb
        batch = pid // (nmb * nnb)
        rows = row_block * BM + tl.arange(0, BM)
        cols = col_block * BN + tl.arange(0, BN)
        row_mask = rows < D
        col_mask = cols < D
        base = state + batch * stride_sb + rows[:, None] * stride_sm + cols[None, :] * stride_sn
        state_value = tl.load(base, mask=row_mask[:, None] & col_mask[None, :], other=0.0)
        delta_acc = tl.zeros((BM, BN), dtype=tl.float32)
        for start in range(0, 2 * T, BLOCK_L):
            sub = start + tl.arange(0, BLOCK_L)
            sub_mask = sub < 2 * T
            pp = innovations + batch * stride_pb + sub[:, None] * stride_pt + rows[None, :] * stride_pd
            up = u + batch * stride_ub + sub[:, None] * stride_ut + cols[None, :] * stride_ud
            pv = tl.load(pp, mask=sub_mask[:, None] & row_mask[None, :], other=0.0)
            uv = tl.load(up, mask=sub_mask[:, None] & col_mask[None, :], other=0.0)
            delta_acc += tl.sum(pv[:, :, None] * uv[:, None, :], axis=0)
        delta_value = delta_acc.to(tl.bfloat16)
        tl.store(
            final_state + batch * stride_ob + rows[:, None] * stride_om + cols[None, :] * stride_on,
            state_value + delta_value, mask=row_mask[:, None] & col_mask[None, :],
        )


def interleave_doubled_bf16(
    e: torch.Tensor,
    k: torch.Tensor,
    value: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build doubled U/Z in one ROCm Triton launch.

    The kernel accepts strided CUDA BF16 inputs and writes contiguous output.
    This avoids a separate chunk-slice copy in the benchmark path.
    """
    if triton is None or not e.is_cuda or e.dtype != torch.bfloat16:
        raise RuntimeError("ROCm Triton doubled interleave requires CUDA BF16")
    if e.shape != k.shape or e.shape != value.shape:
        raise ValueError("e, k, and value must have identical shapes")
    batch, tokens, d_memory = e.shape
    u = torch.empty(batch, 2 * tokens, d_memory, device=e.device, dtype=e.dtype)
    z = torch.empty_like(u)
    block_d = 128 if d_memory >= 128 else 64
    grid = (batch * tokens * triton.cdiv(d_memory, block_d),)
    _interleave_doubled_kernel[grid](
        e, k, value, u, z,
        batch, tokens, d_memory,
        e.stride(0), e.stride(1), e.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        value.stride(0), value.stride(1), value.stride(2),
        u.stride(0), u.stride(1), u.stride(2),
        z.stride(0), z.stride(1), z.stride(2),
        BD=block_d,
        num_warps=8 if d_memory >= 128 else 4,
    )
    return u, z


def fused_chunk_outputs_bf16(
    q: torch.Tensor,
    u: torch.Tensor,
    innovations: torch.Tensor,
    state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fuse causal reads and the final rank-one state accumulation.

    The inputs are the exact compact-WY intermediates.  The kernel computes
    ``reads = q @ state.T + causal(q @ U.T) @ P`` and
    ``state + P.T @ U`` directly into BF16 outputs.  It does not alter factor
    construction or recurrence ordering.
    """
    if triton is None or not q.is_cuda or q.dtype != torch.bfloat16:
        raise RuntimeError("ROCm Triton fused outputs require CUDA BF16")
    if q.ndim != 3 or u.ndim != 3 or innovations.ndim != 3 or state.ndim != 3:
        raise ValueError("invalid fused output tensor rank")
    batch, tokens, d_memory = q.shape
    if u.shape != (batch, 2 * tokens, d_memory) or innovations.shape != u.shape:
        raise ValueError("u and innovations must have shape [B, 2*T, d]")
    reads = torch.empty_like(q)
    final_state = torch.empty_like(state)
    block_d = 64 if d_memory <= 64 else 128
    grid_read = (batch * tokens * triton.cdiv(d_memory, block_d),)
    _fused_read_kernel[grid_read](
        q, u, innovations, state, reads,
        batch, tokens, d_memory,
        q.stride(0), q.stride(1), q.stride(2),
        u.stride(0), u.stride(1), u.stride(2),
        innovations.stride(0), innovations.stride(1), innovations.stride(2),
        state.stride(0), state.stride(1), state.stride(2),
        reads.stride(0), reads.stride(1), reads.stride(2),
        BD=block_d, BLOCK_L=32,
        num_warps=4 if d_memory <= 64 else 8,
    )
    block_m = 32 if d_memory <= 128 else 16
    block_n = 32 if d_memory <= 128 else 16
    grid_state = (batch * triton.cdiv(d_memory, block_m) * triton.cdiv(d_memory, block_n),)
    _fused_state_update_kernel[grid_state](
        u, innovations, state, final_state,
        batch, tokens, d_memory,
        u.stride(0), u.stride(1), u.stride(2),
        innovations.stride(0), innovations.stride(1), innovations.stride(2),
        state.stride(0), state.stride(1), state.stride(2),
        final_state.stride(0), final_state.stride(1), final_state.stride(2),
        BM=block_m, BN=block_n, BLOCK_L=32,
        num_warps=4 if d_memory <= 128 else 8,
    )
    return reads, final_state


__all__ = ["fused_chunk_outputs_bf16", "interleave_doubled_bf16"]
