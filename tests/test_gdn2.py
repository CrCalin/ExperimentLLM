import unittest

import torch

from experimentlm.model import GatedDeltaNet2LinearAttention


class GDN2Tests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(11)
        self.attention = GatedDeltaNet2LinearAttention(
            32, n_heads=4, head_dim=8, expand_v=1.0,
        ).eval()

    def test_cpu_forward_backward_and_cache(self):
        attention = self.attention.train()
        x = torch.randn(2, 19, 32, requires_grad=True)
        output = attention(x, None, None)
        output.square().mean().backward()
        self.assertEqual(output.shape, x.shape)
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in attention.parameters()))

        attention.eval()
        with torch.no_grad():
            full = attention(x.detach(), None, None)
            pieces = []
            state = conv = None
            for token in x.detach().split(1, dim=1):
                y, state, conv, _ = attention.forward_cached_with_route(token, state, conv)
                pieces.append(y)
        torch.testing.assert_close(torch.cat(pieces, dim=1), full, rtol=1e-5, atol=1e-5)
        self.assertEqual(tuple(state.shape), (2, 4, 8, 8))

    def test_compile_eager(self):
        compiled = torch.compile(self.attention, backend="eager")
        output = compiled(torch.randn(1, 7, 32), None, None)
        self.assertEqual(output.shape, (1, 7, 32))

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA or ROCm")
    def test_accelerator_chunk_recurrent_and_reference(self):
        from fla.ops.gdn2 import chunk_gdn2, fused_recurrent_gdn2, naive_recurrent_gdn2

        device = torch.device("cuda")
        torch.manual_seed(12)
        b, t, h, k, v = 1, 65, 2, 8, 8
        q = torch.randn(b, t, h, k, device=device, dtype=torch.bfloat16)
        key = torch.randn_like(q)
        value = torch.randn(b, t, h, v, device=device, dtype=torch.bfloat16)
        g = -torch.rand(b, t, h, k, device=device, dtype=torch.bfloat16)
        erase = torch.rand_like(q)
        write = torch.rand_like(value)
        kwargs = dict(output_final_state=True, use_qk_l2norm_in_kernel=True, state_v_first=True)
        chunk, chunk_state = chunk_gdn2(q, key, value, g, erase, write, **kwargs)
        recurrent, recurrent_state = fused_recurrent_gdn2(q, key, value, g, erase, write, **kwargs)
        q_ref = torch.nn.functional.normalize(q.float(), dim=-1)
        k_ref = torch.nn.functional.normalize(key.float(), dim=-1)
        reference, reference_state = naive_recurrent_gdn2(
            q_ref, k_ref, value, g, erase, write, scale=k ** -0.5,
            output_final_state=True,
        )
        reference_state = reference_state.transpose(-1, -2)
        torch.testing.assert_close(chunk.float(), reference.float(), rtol=0.08, atol=0.08)
        torch.testing.assert_close(recurrent.float(), reference.float(), rtol=0.08, atol=0.08)
        torch.testing.assert_close(chunk_state.float(), recurrent_state.float(), rtol=0.08, atol=0.08)
        torch.testing.assert_close(chunk_state.float(), reference_state.float(), rtol=0.08, atol=0.08)


if __name__ == "__main__":
    unittest.main()
