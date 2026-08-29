import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import torch

from experimentlm import ExperimentConfig, build_model
from experimentlm.checkpoint import load_checkpoint, save_checkpoint
from experimentlm.model import DeltaNetLinearAttention, _gdn_recurrence
from experimentlm.optim import build_optimizer, build_schedule


OPTIMIZED_CONFIG = Path(__file__).parents[1] / "configs" / "hybrid_deltanet_transformer_expandv1_optimized.json"
CONFIG = OPTIMIZED_CONFIG


class DeltaNetTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.attention = DeltaNetLinearAttention(
            d_model=32, n_heads=4, head_dim=8, expand_v=2.0,
        ).eval()

    def test_optimized_config_pattern_and_parameter_count(self):
        model = build_model(ExperimentConfig.load(CONFIG).model)
        self.assertEqual(
            [type(block.attn).__name__ for block in model.blocks],
            ["MultiHeadCausalSelfAttention", "DeltaNetLinearAttention",
             "DeltaNetLinearAttention", "MultiHeadCausalSelfAttention",
             "DeltaNetLinearAttention", "DeltaNetLinearAttention",
             "MultiHeadCausalSelfAttention"],
        )
        self.assertEqual(model.num_parameters, 10_444_832)
        delta = model.blocks[1].attn
        self.assertEqual((delta.key_dim, delta.value_dim), (256, 256))
        self.assertEqual(tuple(delta.initial_state(torch.randn(2, 3, 256)).shape), (2, 4, 64, 64))

    def test_optimized_expand_v1_config_uses_best_tested_path(self):
        model = build_model(ExperimentConfig.load(OPTIMIZED_CONFIG).model)
        self.assertEqual(model.num_parameters, 10_444_832)
        delta = model.blocks[1].attn
    def test_causality_and_state_reset(self):
        x = torch.randn(2, 13, 32)
        changed = x.clone()
        changed[:, 7:] = torch.randn_like(changed[:, 7:])
        original, _, _ = self.attention.forward_recurrent(x)
        perturbed, _, _ = self.attention.forward_recurrent(changed)
        torch.testing.assert_close(original[:, :7], perturbed[:, :7], rtol=0, atol=0)
        first, state, conv = self.attention.forward_recurrent(x[:, :5])
        second, _, _ = self.attention.forward_recurrent(x[:, 5:], state=state, conv_state=conv)
        complete, _, _ = self.attention.forward_recurrent(x)
        torch.testing.assert_close(torch.cat((first, second), dim=1), complete)

    def test_matches_fla_naive_reference(self):
        from fla.ops.gated_delta_rule import naive_recurrent_gated_delta_rule

        torch.manual_seed(12)
        b, t, h, k, v = 2, 9, 3, 5, 7
        q = torch.nn.functional.normalize(torch.randn(b, t, h, k), dim=-1)
        key = torch.nn.functional.normalize(torch.randn(b, t, h, k), dim=-1)
        value = torch.randn(b, t, h, v)
        beta = torch.sigmoid(torch.randn(b, t, h))
        decay = torch.randn(b, t, h).clamp(-4, 0)
        state = torch.randn(b, h, v, k)
        expected, expected_state = naive_recurrent_gated_delta_rule(
            q, key, value, beta, decay, scale=1.0,
            initial_state=state.transpose(-1, -2), output_final_state=True,
        )
        actual, actual_state = _gdn_recurrence(q, key, value, beta, decay, state)
        torch.testing.assert_close(actual, expected.to(actual.dtype), rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(actual_state, expected_state.transpose(-1, -2), rtol=1e-5, atol=1e-5)

    def test_recurrence_gradients_match_fla_naive_reference(self):
        from fla.ops.gated_delta_rule import naive_recurrent_gated_delta_rule

        torch.manual_seed(21)
        shape = (1, 6, 2, 4)
        q1 = torch.nn.functional.normalize(torch.randn(*shape), dim=-1).requires_grad_()
        k1 = torch.nn.functional.normalize(torch.randn(*shape), dim=-1).requires_grad_()
        v1 = torch.randn(1, 6, 2, 5, requires_grad=True)
        b1 = torch.sigmoid(torch.randn(1, 6, 2)).requires_grad_()
        g1 = torch.randn(1, 6, 2).clamp(-4, 0).requires_grad_()
        s1 = torch.randn(1, 2, 5, 4, requires_grad=True)
        out1, state1 = _gdn_recurrence(q1, k1, v1, b1, g1, s1)
        loss1 = out1.square().mean() + state1.square().mean()
        loss1.backward()

        q2, k2, v2, b2, g2, s2 = [x.detach().clone().requires_grad_() for x in (q1, k1, v1, b1, g1, s1)]
        out2, state2 = naive_recurrent_gated_delta_rule(
            q2, k2, v2, b2, g2, scale=1.0,
            initial_state=s2.transpose(-1, -2), output_final_state=True,
        )
        loss2 = out2.square().mean() + state2.square().mean()
        loss2.backward()
        torch.testing.assert_close(out1, out2, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(state1, state2.transpose(-1, -2), rtol=1e-5, atol=1e-5)
        for left, right in zip((q1, k1, v1, b1, g1, s1), (q2, k2, v2, b2, g2, s2)):
            torch.testing.assert_close(left.grad, right.grad, rtol=1e-5, atol=1e-5)

    def test_no_quadratic_attention_tensor_allocation(self):
        with patch("torch.nn.functional.scaled_dot_product_attention", side_effect=AssertionError):
            output, _, _ = self.attention.forward_recurrent(torch.randn(2, 13, 32))
        self.assertEqual(output.shape, (2, 13, 32))

    def test_forward_backward(self):
        attention = self.attention.train()
        x = torch.randn(2, 13, 32, requires_grad=True)
        output, _, _ = attention.forward_recurrent(x)
        output.square().mean().backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertTrue(all(parameter.grad is not None for parameter in attention.parameters()))
        self.assertTrue(all(torch.isfinite(parameter.grad).all() for parameter in attention.parameters()))

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA or ROCm")
    def test_fla_chunk_recurrent_and_bf16_equivalence(self):
        from fla.ops.gated_delta_rule import (
            chunk_gated_delta_rule, fused_recurrent_gated_delta_rule,
        )

        device = torch.device("cuda")
        torch.manual_seed(3)
        b, t, h, k, v = 2, 128, 4, 8, 16
        q = torch.randn(b, t, h, k, device=device, dtype=torch.bfloat16)
        key = torch.randn_like(q)
        value = torch.randn(b, t, h, v, device=device, dtype=torch.bfloat16)
        a = torch.randn(b, t, h, device=device, dtype=torch.bfloat16)
        beta = torch.randn_like(a)
        A_log = torch.randn(h, device=device)
        dt_bias = torch.randn(h, device=device)
        kwargs = dict(g=a, beta=beta, scale=1.0, A_log=A_log, dt_bias=dt_bias,
                      use_qk_l2norm_in_kernel=True, use_beta_sigmoid_in_kernel=True,
                      use_gate_in_kernel=True, state_v_first=True,
                      output_final_state=True)
        chunk, chunk_state = chunk_gated_delta_rule(q=q, k=key, v=value, **kwargs)
        recurrent, recurrent_state = fused_recurrent_gated_delta_rule(q=q, k=key, v=value, **kwargs)
        torch.testing.assert_close(chunk, recurrent, rtol=0.08, atol=0.08)
        torch.testing.assert_close(chunk_state, recurrent_state, rtol=0.08, atol=0.08)

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA or ROCm")
    def test_fla_fp32_recurrent_matches_naive_reference(self):
        from fla.ops.gated_delta_rule import (
            fused_recurrent_gated_delta_rule, naive_recurrent_gated_delta_rule,
        )

        device = torch.device("cuda")
        b, t, h, k, v = 1, 17, 2, 8, 12
        q = torch.randn(b, t, h, k, device=device)
        key = torch.randn_like(q)
        value = torch.randn(b, t, h, v, device=device)
        a = torch.randn(b, t, h, device=device)
        beta = torch.randn_like(a)
        A_log = torch.randn(h, device=device)
        dt_bias = torch.randn(h, device=device)
        qn, kn = torch.nn.functional.normalize(q, dim=-1), torch.nn.functional.normalize(key, dim=-1)
        decay = -torch.exp(A_log).view(1, 1, h) * torch.nn.functional.softplus(a + dt_bias.view(1, 1, h))
        expected, expected_state = naive_recurrent_gated_delta_rule(
            qn, kn, value, beta.sigmoid(), decay, scale=1.0, output_final_state=True,
        )
        actual, actual_state = fused_recurrent_gated_delta_rule(
            q=qn, k=kn, v=value, g=a, beta=beta, scale=1.0, A_log=A_log,
            dt_bias=dt_bias, use_gate_in_kernel=True, use_beta_sigmoid_in_kernel=True,
            state_v_first=True, output_final_state=True,
        )
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(actual_state, expected_state.transpose(-1, -2), rtol=1e-4, atol=1e-4)

    def test_cached_model_matches_full_context(self):
        model = build_model(ExperimentConfig.load(CONFIG).model).eval()
        tokens = torch.randint(0, 8192, (1, 19))
        with torch.no_grad():
            _, cache = model.forward_cached(tokens[:, :11])
            cached, _ = model.forward_cached(tokens[:, 11:12], cache)
            full, _ = model(tokens[:, :12])
        torch.testing.assert_close(cached[:, -1], full[:, -1], rtol=1e-5, atol=1e-5)

    def test_optimizer_excludes_bias_norm_and_special_parameters(self):
        model = build_model(ExperimentConfig.load(CONFIG).model)
        config = ExperimentConfig.load(CONFIG)
        optimizer = build_optimizer(config.training.optimizer, model.parameters())
        no_decay = {id(p) for group in optimizer.param_groups if group["weight_decay"] == 0 for p in group["params"]}
        for name, parameter in model.named_parameters():
            if parameter.ndim == 1 or name.endswith(".bias") or getattr(parameter, "_no_weight_decay", False):
                self.assertIn(id(parameter), no_decay, name)

    def test_checkpoint_roundtrip(self):
        config = ExperimentConfig.load(CONFIG)
        model = build_model(config.model)
        optimizer = build_optimizer(config.training.optimizer, model.parameters())
        scheduler = build_schedule(config.training.schedule, optimizer, 2)
        tokens = torch.randint(0, config.model.vocab_size, (1, 8))
        expected, _ = model(tokens)
        with tempfile.TemporaryDirectory() as directory:
            save_checkpoint(directory, model, optimizer, scheduler, 1, 8, config)
            restored, blob = load_checkpoint(directory)
        actual, _ = restored(tokens)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertEqual(blob["step"], 1)


if __name__ == "__main__":
    unittest.main()
