from pathlib import Path
import unittest

import torch

from experimentlm import ExperimentConfig, build_model
from experimentlm.model import DeltaNetLinearAttention


ROOT = Path(__file__).parents[1]
CONFIG = ROOT / "configs/gated_deltanet_qknorm_clvr.json"
BASELINE = ROOT / "configs/gated_deltanet_qknorm.json"


class CLVRTests(unittest.TestCase):
    def test_shapes_parameters_and_zero_init(self):
        model = build_model(ExperimentConfig.load(CONFIG).model)
        routes = [block.value_routing for block in model.blocks if block.value_routing is not None]
        self.assertEqual(len(routes), 4)
        self.assertEqual(model.num_parameters, 10_510_766)
        for route in routes:
            self.assertEqual(tuple(route.down.weight.shape), (32, 256))
            self.assertEqual(tuple(route.up.weight.shape), (256, 32))
            self.assertEqual(route.up.weight.abs().sum().item(), 0.0)

    def test_value_route_gradients_and_cached_generation(self):
        model = build_model(ExperimentConfig.load(CONFIG).model).train()
        x = torch.randint(0, 8192, (2, 7))
        logits, _ = model(x)
        logits.float().square().mean().backward()
        for block in model.blocks:
            if block.value_routing is not None:
                self.assertIsNotNone(block.value_routing.down.weight.grad)
                self.assertIsNotNone(block.value_routing.up.weight.grad)
                self.assertIsNotNone(block.attn.in_proj_weight.grad)
        model.eval()
        with torch.no_grad():
            full, _ = model(x[:1])
            _, cache = model.forward_cached(x[:1, :4])
            cached, _ = model.forward_cached(x[:1, 4:5], cache)
        torch.testing.assert_close(full[:, 4], cached[:, -1], rtol=1e-5, atol=1e-5)

    def test_route_is_the_internal_value_and_baseline_compatibility(self):
        attention = DeltaNetLinearAttention(32, n_heads=4, head_dim=8, expand_v=1.0,
                                            return_value_route=True).eval()
        _, route = attention.forward_with_route(torch.randn(2, 5, 32), None, None)
        self.assertEqual(route.shape, (2, 5, 32))
        self.assertTrue(route.requires_grad)

        baseline = build_model(ExperimentConfig.load(BASELINE).model).eval()
        clvr = build_model(ExperimentConfig.load(CONFIG).model).eval()
        clvr.load_state_dict(baseline.state_dict(), strict=False)
        tokens = torch.randint(0, 8192, (1, 6))
        with torch.no_grad():
            baseline_logits, _ = baseline(tokens)
            clvr_logits, _ = clvr(tokens)
        torch.testing.assert_close(clvr_logits, baseline_logits, rtol=0, atol=0)

    def test_bf16_and_compile_eager(self):
        model = build_model(ExperimentConfig.load(CONFIG).model).eval()
        tokens = torch.randint(0, 8192, (1, 4))
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            logits, _ = model(tokens)
        self.assertEqual(logits.dtype, torch.bfloat16)
        compiled = torch.compile(model, backend="eager")
        with torch.no_grad():
            with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
                compiled(tokens)

    def test_post_block_route_runs_after_ffn(self):
        model = build_model(ExperimentConfig.load(
            ROOT / "configs/gated_deltanet_qknorm_clvr_postblock.json"
        ).model).eval()
        block = model.blocks[1]
        events = []
        block.mlp.register_forward_hook(lambda *_: events.append("mlp"))
        block.value_routing.register_forward_hook(lambda *_: events.append("route"))
        with torch.no_grad():
            block(torch.randn(1, 5, 256), None, model.position_encoding)
        self.assertEqual(block.value_routing.placement, "post_block")
        self.assertEqual(events, ["mlp", "route"])


if __name__ == "__main__":
    unittest.main()
