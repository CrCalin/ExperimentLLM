import unittest
from pathlib import Path

import torch

from experimentlm import ExperimentConfig, build_model
from experimentlm.config import ComponentSpec
from experimentlm.model import MultiHeadCausalSelfAttention, RoPE
from experimentlm.optim import build_optimizer


ROOT = Path(__file__).parents[1]


class HeadScaleTests(unittest.TestCase):
    def test_initial_equivalence_and_parameter_count(self):
        base_config = ExperimentConfig.load(ROOT / "configs/gated_deltanet_qknorm.json")
        scale_config = ExperimentConfig.load(ROOT / "configs/gated_deltanet_qknorm_headscale.json")
        torch.manual_seed(7)
        baseline = build_model(base_config.model).eval()
        torch.manual_seed(7)
        scaled = build_model(scale_config.model).eval()
        tokens = torch.randint(8192, (2, 16))
        with torch.no_grad():
            expected, _ = baseline(tokens)
            actual, _ = scaled(tokens)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertEqual(scaled.num_parameters, 10_445_242)
        scales = [block.attn.head_scale for block in scaled.blocks if hasattr(block.attn, "head_scale")]
        self.assertEqual(len(scales), 3)
        self.assertTrue(all(scale.shape == (4,) for scale in scales))
        self.assertTrue(all(torch.equal(scale, torch.ones(4)) for scale in scales))

    def test_broadcasting_gradients_cache_and_compile(self):
        torch.manual_seed(2)
        attention = MultiHeadCausalSelfAttention(32, 4, 8, qk_norm=True, head_scale=True).eval()
        attention.head_scale.data.copy_(torch.tensor([0.5, 1.0, 1.5, 2.0]))
        x = torch.randn(2, 9, 32, requires_grad=True)
        rope = RoPE(32)
        full = attention(x, None, rope)
        _, cache = attention.forward_cached(x[:, :4], rope)
        cached, _ = attention.forward_cached(x[:, 4:], rope, cache)
        torch.testing.assert_close(cached, full[:, 4:], rtol=1e-5, atol=1e-5)
        full.square().mean().backward()
        self.assertEqual(attention.head_scale.grad.shape, (4,))
        self.assertTrue(torch.all(attention.head_scale.grad != 0))
        compiled = torch.compile(attention, backend="eager")
        compiled(x.detach(), None, RoPE(32))

    def test_adamw_routing_and_no_decay(self):
        attention = MultiHeadCausalSelfAttention(32, 4, 8, head_scale=True)
        optimizer = build_optimizer(
            ComponentSpec("AdamW", {"lr": 1e-3}),
            [("attention.head_scale", attention.head_scale)],
        )
        group = next(group for group in optimizer.param_groups
                     if any(parameter is attention.head_scale for parameter in group["params"]))
        self.assertEqual(group["weight_decay"], 0.0)


if __name__ == "__main__":
    unittest.main()
