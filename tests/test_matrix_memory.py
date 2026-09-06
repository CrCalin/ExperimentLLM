import os
import unittest

import torch

from experimentlm.matrix_memory import (
    BF16TriangularSolveUnavailable,
    chunked_matrix_memory,
    compact_wy_matrix_memory,
    fused_wy_matrix_memory,
    matrix_memory,
    rank2_matrix_memory,
    sequential_matrix_memory,
)
from experimentlm.matrix_memory import _rank2_token_factors


def _inputs(batch: int, length: int, d_memory: int, dtype=torch.float32):
    q, e, k, v = [torch.randn(batch, length, d_memory, dtype=dtype) for _ in range(4)]
    e = e / e.norm(dim=-1, keepdim=True).clamp_min(1e-6) * 0.5
    k = k / k.norm(dim=-1, keepdim=True).clamp_min(1e-6) * 0.5
    return [q, e, k, v]


class MatrixMemoryCorrectnessTests(unittest.TestCase):
    def test_reference_matches_direct_math(self):
        torch.manual_seed(1)
        q, e, k, v = _inputs(2, 7, 5)
        state = torch.randn(2, 5, 5)
        reads, final = sequential_matrix_memory(q, e, k, v, state)

        expected_reads = []
        expected_state = state.clone()
        for t in range(q.size(1)):
            expected_reads.append((expected_state @ q[:, t, :, None]).squeeze(-1))
            erased = (expected_state @ e[:, t, :, None]).squeeze(-1)
            expected_state = expected_state - erased[:, :, None] * e[:, t, None, :]
            error = v[:, t] - (expected_state @ k[:, t, :, None]).squeeze(-1)
            expected_state = expected_state + error[:, :, None] * k[:, t, None, :]
        torch.testing.assert_close(reads, torch.stack(expected_reads, dim=1))
        torch.testing.assert_close(final, expected_state)

    def test_chunked_matches_reference_requested_shape_coverage(self):
        torch.manual_seed(2)
        cases = [
            (1, 1, 8), (2, 2, 16), (8, 17, 32), (1, 63, 64),
            (2, 64, 128), (1, 65, 16), (2, 128, 32), (1, 257, 64),
            (2, 2048, 8),
        ]
        for batch, length, d_memory in cases:
            with self.subTest(batch=batch, length=length, d_memory=d_memory):
                q, e, k, v = _inputs(batch, length, d_memory)
                state = torch.randn(batch, d_memory, d_memory)
                ref_reads, ref_state = sequential_matrix_memory(q, e, k, v, state)
                got_reads, got_state = chunked_matrix_memory(
                    q, e, k, v, state, chunk_size=64,
                )
                torch.testing.assert_close(got_reads, ref_reads, rtol=2e-4, atol=2e-5)
                torch.testing.assert_close(got_state, ref_state, rtol=2e-4, atol=2e-5)
                wy_reads, wy_state = compact_wy_matrix_memory(
                    q, e, k, v, state, chunk_size=64,
                )
                torch.testing.assert_close(wy_reads, ref_reads, rtol=2e-4, atol=2e-5)
                torch.testing.assert_close(wy_state, ref_state, rtol=2e-4, atol=2e-5)
                rank_reads, rank_state = rank2_matrix_memory(
                    q, e, k, v, state, chunk_size=64,
                )
                torch.testing.assert_close(rank_reads, ref_reads, rtol=2e-4, atol=2e-5)
                torch.testing.assert_close(rank_state, ref_state, rtol=2e-4, atol=2e-5)
                fused_reads, fused_state = fused_wy_matrix_memory(
                    q, e, k, v, state, chunk_size=64,
                )
                torch.testing.assert_close(fused_reads, ref_reads, rtol=2e-4, atol=2e-5)
                torch.testing.assert_close(fused_state, ref_state, rtol=2e-4, atol=2e-5)

    def test_multiple_chunk_sizes_and_non_divisible_length(self):
        torch.manual_seed(3)
        q, e, k, v = _inputs(2, 257, 16)
        state = torch.randn(2, 16, 16)
        ref_reads, ref_state = sequential_matrix_memory(q, e, k, v, state)
        for chunk_size in (8, 16, 32, 64, 128):
            with self.subTest(chunk_size=chunk_size):
                reads, final = chunked_matrix_memory(q, e, k, v, state, chunk_size=chunk_size)
                torch.testing.assert_close(reads, ref_reads, rtol=3e-4, atol=3e-5)
                torch.testing.assert_close(final, ref_state, rtol=3e-4, atol=3e-5)
                reads, final = compact_wy_matrix_memory(q, e, k, v, state, chunk_size=chunk_size)
                torch.testing.assert_close(reads, ref_reads, rtol=3e-4, atol=3e-5)
                torch.testing.assert_close(final, ref_state, rtol=3e-4, atol=3e-5)
                reads, final = rank2_matrix_memory(q, e, k, v, state, chunk_size=chunk_size)
                torch.testing.assert_close(reads, ref_reads, rtol=3e-4, atol=3e-5)
                torch.testing.assert_close(final, ref_state, rtol=3e-4, atol=3e-5)

    def test_rank2_factor_algebra_special_cases(self):
        torch.manual_seed(31)
        for mode in ("random", "zero_e", "zero_k", "equal", "orthogonal", "collinear"):
            e = torch.randn(2, 3, 8)
            k = torch.randn(2, 3, 8)
            if mode == "zero_e":
                e.zero_()
            elif mode == "zero_k":
                k.zero_()
            elif mode == "equal":
                k = e.clone()
            elif mode == "orthogonal":
                k = k - (e * k).sum(-1, keepdim=True) * e / e.square().sum(-1, keepdim=True).clamp_min(1e-6)
            elif mode == "collinear":
                k = e * 0.999 + torch.randn_like(e) * 1e-4
            u, v_factor, z = _rank2_token_factors(e, k, torch.randn_like(e))
            c = (e * k).sum(-1)
            expected_u = torch.stack((e, k), dim=-1).permute(0, 2, 1, 3).reshape_as(u)
            expected_v = torch.stack((e - c[..., None] * k, k), dim=-1).permute(
                0, 2, 1, 3,
            ).reshape_as(v_factor)
            torch.testing.assert_close(u, expected_u)
            torch.testing.assert_close(v_factor, expected_v)
            identity = torch.eye(8).expand(2, 3, -1, -1)
            a = (identity - e[..., :, None] * e[..., None, :]) @ (
                identity - k[..., :, None] * k[..., None, :]
            )
            represented = torch.eye(8).expand(2, 3, -1, -1) - u.transpose(-1, -2).reshape(2, 3, 2, 8).transpose(-1, -2) @ v_factor.transpose(-1, -2).reshape(2, 3, 2, 8)
            torch.testing.assert_close(represented, a, rtol=2e-5, atol=2e-5)
            value = torch.randn_like(e)
            _, _, z = _rank2_token_factors(e, k, value)
            z4 = z.transpose(-1, -2).reshape(2, 3, 2, 8).transpose(-1, -2)
            v4 = v_factor.transpose(-1, -2).reshape(2, 3, 2, 8).transpose(-1, -2)
            torch.testing.assert_close(z4 @ v4.transpose(-1, -2), value[..., :, None] * k[..., None, :])

    def test_causality(self):
        torch.manual_seed(4)
        q, e, k, v = _inputs(2, 32, 8)
        changed = [x.clone() for x in (q, e, k, v)]
        for tensor in changed:
            tensor[:, 16:] = torch.randn_like(tensor[:, 16:]) * 100
        state = torch.randn(2, 8, 8)
        reads, _ = chunked_matrix_memory(q, e, k, v, state, chunk_size=8)
        changed_reads, _ = chunked_matrix_memory(*changed, state, chunk_size=8)
        torch.testing.assert_close(reads[:, :16], changed_reads[:, :16], rtol=3e-4, atol=3e-5)
        reads, _ = rank2_matrix_memory(q, e, k, v, state, chunk_size=8)
        changed_reads, _ = rank2_matrix_memory(*changed, state, chunk_size=8)
        torch.testing.assert_close(reads[:, :16], changed_reads[:, :16], rtol=3e-4, atol=3e-5)

    def test_read_before_erase_before_write(self):
        q = torch.tensor([[[1.0, 0.0], [1.0, 0.0]]])
        e = torch.zeros_like(q)
        k = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
        v = torch.tensor([[[7.0, 0.0], [0.0, 2.0]]])
        initial = torch.zeros(1, 2, 2)
        ref_reads, ref_state = sequential_matrix_memory(q, e, k, v, initial)
        chunk_reads, chunk_state = chunked_matrix_memory(q, e, k, v, initial, chunk_size=2)
        self.assertEqual(float(ref_reads[0, 0, 0]), 0.0)
        self.assertEqual(float(ref_reads[0, 1, 0]), 7.0)
        torch.testing.assert_close(chunk_reads, ref_reads)
        torch.testing.assert_close(chunk_state, ref_state)
        rank_reads, rank_state = rank2_matrix_memory(q, e, k, v, initial, chunk_size=2)
        torch.testing.assert_close(rank_reads, ref_reads)
        torch.testing.assert_close(rank_state, ref_state)

        # The erase clears the old association before the write replaces it.
        initial = torch.zeros(1, 2, 2)
        initial[0, :, 0] = torch.tensor([3.0, 4.0])
        q = torch.zeros(1, 1, 2)
        e = torch.tensor([[[1.0, 0.0]]])
        k = torch.tensor([[[1.0, 0.0]]])
        v = torch.tensor([[[5.0, 6.0]]])
        _, final = sequential_matrix_memory(q, e, k, v, initial)
        torch.testing.assert_close(final[0, :, 0], torch.tensor([5.0, 6.0]))

    def test_zero_addresses_delta_correction_and_independence(self):
        torch.manual_seed(5)
        state = torch.randn(2, 3, 3)
        q, e, k, v = _inputs(2, 3, 3)
        zero = torch.zeros_like(q)
        _, no_erase = sequential_matrix_memory(q, zero, k, v, state)
        _, ordinary = sequential_matrix_memory(q, e, k, v, state)
        self.assertFalse(torch.equal(no_erase, ordinary))
        _, no_write = sequential_matrix_memory(q, e, zero, v, state)
        # k=0 makes the corrective outer product exactly zero.
        expected_reads, expected_after_erase = sequential_matrix_memory(q, e, zero, zero, state)
        torch.testing.assert_close(no_write, expected_after_erase)

        address = torch.tensor([[[1.0, 0.0], [1.0, 0.0]]])
        values = torch.tensor([[[3.0, 4.0], [3.0, 4.0]]])
        _, repeated = sequential_matrix_memory(
            torch.zeros_like(address), torch.zeros_like(address), address, values,
            torch.zeros(1, 2, 2),
        )
        torch.testing.assert_close(repeated[0, :, 0], torch.tensor([3.0, 4.0]))

    def test_gradients_match(self):
        torch.manual_seed(6)
        source = _inputs(2, 11, 6)
        source.append(torch.randn(2, 6, 6))
        ref_inputs = [x.clone().requires_grad_() for x in source]
        chunk_inputs = [x.clone().requires_grad_() for x in source]
        ref_reads, ref_state = sequential_matrix_memory(*ref_inputs)
        chunk_reads, chunk_state = chunked_matrix_memory(*chunk_inputs, chunk_size=4)
        wy_inputs = [x.clone().requires_grad_() for x in source]
        wy_reads, wy_state = compact_wy_matrix_memory(*wy_inputs, chunk_size=4)
        ref_loss = (ref_reads.square().mean() + 0.7 * ref_state.square().mean())
        chunk_loss = (chunk_reads.square().mean() + 0.7 * chunk_state.square().mean())
        wy_loss = (wy_reads.square().mean() + 0.7 * wy_state.square().mean())
        ref_grad = torch.autograd.grad(ref_loss, ref_inputs)
        chunk_grad = torch.autograd.grad(chunk_loss, chunk_inputs)
        wy_grad = torch.autograd.grad(wy_loss, wy_inputs)
        rank_inputs = [x.clone().requires_grad_() for x in source]
        rank_reads, rank_state = rank2_matrix_memory(*rank_inputs, chunk_size=4)
        rank_loss = rank_reads.square().mean() + 0.7 * rank_state.square().mean()
        rank_grad = torch.autograd.grad(rank_loss, rank_inputs)
        for name, actual, expected in zip(("q", "e", "k", "v", "initial_state"), chunk_grad, ref_grad):
            with self.subTest(input=name):
                torch.testing.assert_close(actual, expected, rtol=2e-3, atol=2e-4)
        for name, actual, expected in zip(("q", "e", "k", "v", "initial_state"), rank_grad, ref_grad):
            with self.subTest(rank2_input=name):
                torch.testing.assert_close(actual, expected, rtol=2e-3, atol=2e-4)
        for name, actual, expected in zip(("q", "e", "k", "v", "initial_state"), wy_grad, ref_grad):
            with self.subTest(compact_wy_input=name):
                torch.testing.assert_close(actual, expected, rtol=2e-3, atol=2e-4)
        fused_inputs = [x.clone().requires_grad_() for x in source]
        fused_reads, fused_state = fused_wy_matrix_memory(*fused_inputs, chunk_size=4)
        fused_loss = fused_reads.square().mean() + 0.7 * fused_state.square().mean()
        fused_grad = torch.autograd.grad(fused_loss, fused_inputs)
        for name, actual, expected in zip(("q", "e", "k", "v", "initial_state"), fused_grad, ref_grad):
            with self.subTest(fused_wy_input=name):
                torch.testing.assert_close(actual, expected, rtol=2e-3, atol=2e-4)

    def test_state_isolated_per_batch_element(self):
        torch.manual_seed(7)
        q, e, k, v = _inputs(2, 19, 8)
        state = torch.randn(2, 8, 8)
        batch_reads, batch_final = chunked_matrix_memory(q, e, k, v, state, chunk_size=8)
        rank_batch_reads, rank_batch_final = rank2_matrix_memory(q, e, k, v, state, chunk_size=8)
        for index in range(2):
            reads, final = chunked_matrix_memory(
                q[index:index + 1], e[index:index + 1], k[index:index + 1],
                v[index:index + 1], state[index:index + 1], chunk_size=8,
            )
            torch.testing.assert_close(batch_reads[index:index + 1], reads)
            torch.testing.assert_close(batch_final[index:index + 1], final)
            rank_reads, rank_final = rank2_matrix_memory(
                q[index:index + 1], e[index:index + 1], k[index:index + 1],
                v[index:index + 1], state[index:index + 1], chunk_size=8,
            )
            torch.testing.assert_close(rank_batch_reads[index:index + 1], rank_reads)
            torch.testing.assert_close(rank_batch_final[index:index + 1], rank_final)

    def test_normalized_erase_and_write_transitions_are_non_expansive(self):
        torch.manual_seed(8)
        state = torch.randn(4, 32, 32)
        erase = torch.randn(4, 32)
        key = torch.randn(4, 32)
        erase = erase / erase.norm(dim=-1, keepdim=True)
        key = key / key.norm(dim=-1, keepdim=True)
        erased = state - (state @ erase[..., None]).squeeze(-1)[..., None] * erase[..., None, :]
        erased_then_write = state - (state @ key[..., None]).squeeze(-1)[..., None] * key[..., None, :]
        self.assertTrue((torch.linalg.matrix_norm(erased) <= torch.linalg.matrix_norm(state) + 1e-5).all())
        self.assertTrue((torch.linalg.matrix_norm(erased_then_write) <= torch.linalg.matrix_norm(state) + 1e-5).all())

    def test_public_backend_selector_and_dtype_preservation(self):
        q, e, k, v = _inputs(1, 9, 8)
        state = torch.zeros(1, 8, 8)
        reads, final = matrix_memory(q, e, k, v, state, backend="reference")
        self.assertEqual(reads.dtype, torch.float32)
        self.assertEqual(final.dtype, torch.float32)
        wy_reads, wy_final = matrix_memory(q, e, k, v, state, backend="wy")
        torch.testing.assert_close(wy_reads, reads)
        torch.testing.assert_close(wy_final, final)
        rank_reads, rank_final = matrix_memory(q, e, k, v, state, backend="rank2")
        torch.testing.assert_close(rank_reads, reads)
        torch.testing.assert_close(rank_final, final)
        fused_reads, fused_final = matrix_memory(q, e, k, v, state, backend="fused_wy")
        torch.testing.assert_close(fused_reads, reads)
        torch.testing.assert_close(fused_final, final)
        with self.assertRaisesRegex(ValueError, "backend"):
            matrix_memory(q, e, k, v, state, backend="not-a-backend")

    def test_torch_compile_chunked_static_shape(self):
        torch.manual_seed(9)
        q, e, k, v = _inputs(1, 17, 8)
        state = torch.randn(1, 8, 8)
        compiled = torch.compile(
            lambda q, e, k, v, state: chunked_matrix_memory(
                q, e, k, v, state, chunk_size=8,
            ),
            backend="eager",
        )
        expected = chunked_matrix_memory(q, e, k, v, state, chunk_size=8)
        actual = compiled(q, e, k, v, state)
        torch.testing.assert_close(actual[0], expected[0])
        torch.testing.assert_close(actual[1], expected[1])

    def test_bf16_compact_wy_matches_bf16_reference(self):
        q, e, k, v = _inputs(2, 17, 16, torch.bfloat16)
        state = torch.randn(2, 16, 16, dtype=torch.bfloat16)
        ref_reads, ref_state = sequential_matrix_memory(q, e, k, v, state)
        wy_reads, wy_state = compact_wy_matrix_memory(q, e, k, v, state, chunk_size=8)
        rank_reads, rank_state = rank2_matrix_memory(q, e, k, v, state, chunk_size=8)
        fused_reads, fused_state = fused_wy_matrix_memory(q, e, k, v, state, chunk_size=8)
        self.assertEqual(wy_reads.dtype, torch.bfloat16)
        self.assertEqual(wy_state.dtype, torch.bfloat16)
        torch.testing.assert_close(wy_reads, ref_reads, rtol=0.08, atol=0.08)
        torch.testing.assert_close(wy_state, ref_state, rtol=0.08, atol=0.08)
        self.assertEqual(rank_reads.dtype, torch.bfloat16)
        self.assertEqual(rank_state.dtype, torch.bfloat16)
        torch.testing.assert_close(rank_reads, ref_reads, rtol=0.08, atol=0.08)
        torch.testing.assert_close(rank_state, ref_state, rtol=0.08, atol=0.08)
        self.assertEqual(fused_reads.dtype, torch.bfloat16)
        self.assertEqual(fused_state.dtype, torch.bfloat16)
        torch.testing.assert_close(fused_reads, ref_reads, rtol=0.08, atol=0.08)
        torch.testing.assert_close(fused_state, ref_state, rtol=0.08, atol=0.08)
        ref_inputs = [x.clone().requires_grad_() for x in (q, e, k, v, state)]
        wy_inputs = [x.clone().requires_grad_() for x in (q, e, k, v, state)]
        rr, rs = sequential_matrix_memory(*ref_inputs)
        wr, ws = compact_wy_matrix_memory(*wy_inputs, chunk_size=8)
        ref_grads = torch.autograd.grad(rr.square().mean() + rs.square().mean(), ref_inputs)
        wy_grads = torch.autograd.grad(wr.square().mean() + ws.square().mean(), wy_inputs)
        rank_inputs = [x.clone().requires_grad_() for x in (q, e, k, v, state)]
        qr, qs = rank2_matrix_memory(*rank_inputs, chunk_size=8)
        rank_grads = torch.autograd.grad(qr.square().mean() + qs.square().mean(), rank_inputs)
        for actual, expected in zip(wy_grads, ref_grads):
            torch.testing.assert_close(actual, expected, rtol=0.15, atol=0.15)
        for actual, expected in zip(rank_grads, ref_grads):
            torch.testing.assert_close(actual, expected, rtol=0.15, atol=0.15)

    @unittest.skipUnless(os.environ.get("MATRIX_MEMORY_FULL_TESTS") == "1", "opt-in long stability grid")
    def test_bf16_long_stability_grid(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for length in (2048, 4096, 8192, 16384, 32768):
            for d_memory in (32, 64, 128, 256):
                with self.subTest(length=length, d_memory=d_memory):
                    q, e, k, v = _inputs(1, length, d_memory, torch.bfloat16)
                    e = e / e.norm(dim=-1, keepdim=True).clamp_min(1e-6) * 0.5
                    k = k / k.norm(dim=-1, keepdim=True).clamp_min(1e-6) * 0.5
                    q, e, k, v = [x.to(device) for x in (q, e, k, v)]
                    reads, final = sequential_matrix_memory(q, e, k, v)
                    self.assertTrue(torch.isfinite(reads).all() and torch.isfinite(final).all())
                    self.assertEqual(final.dtype, torch.bfloat16)

    def test_bf16_chunk_capability_is_explicit(self):
        q, e, k, v = _inputs(1, 4, 8, torch.bfloat16)
        try:
            reads, final = chunked_matrix_memory(q, e, k, v, chunk_size=2)
        except BF16TriangularSolveUnavailable as exc:
            self.assertIn("triangular solve", str(exc))
        else:
            self.assertEqual(reads.dtype, torch.bfloat16)
            self.assertEqual(final.dtype, torch.bfloat16)

    def test_bf16_reference_autograd_is_finite(self):
        q, e, k, v = _inputs(2, 17, 16, torch.bfloat16)
        inputs = [x.requires_grad_() for x in (q, e, k, v)]
        state = torch.zeros(2, 16, 16, dtype=torch.bfloat16, requires_grad=True)
        reads, final = sequential_matrix_memory(*inputs, state)
        (reads.square().mean() + final.square().mean()).backward()
        self.assertEqual(reads.dtype, torch.bfloat16)
        self.assertTrue(torch.isfinite(reads).all() and torch.isfinite(final).all())
        self.assertTrue(all(x.grad is not None and torch.isfinite(x.grad).all() for x in inputs))
        self.assertIsNotNone(state.grad)

    @unittest.skipUnless(torch.cuda.is_available(), "requires ROCm/CUDA")
    def test_bf16_fused_rocm_path_matches_wy(self):
        torch.manual_seed(18)
        q, e, k, v = _inputs(2, 129, 64, torch.bfloat16)
        state = torch.randn(2, 64, 64, dtype=torch.bfloat16)
        q, e, k, v, state = [x.cuda() for x in (q, e, k, v, state)]
        wy_reads, wy_state = compact_wy_matrix_memory(q, e, k, v, state, chunk_size=64)
        fused_reads, fused_state = fused_wy_matrix_memory(q, e, k, v, state, chunk_size=64)
        torch.testing.assert_close(fused_reads, wy_reads, rtol=0.15, atol=0.15)
        torch.testing.assert_close(fused_state, wy_state, rtol=0.15, atol=0.15)


if __name__ == "__main__":
    unittest.main()
