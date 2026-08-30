import tempfile
from pathlib import Path
import unittest

import torch

from experimentlm import ExperimentConfig, build_model
from experimentlm.checkpoint import load_checkpoint, save_checkpoint
from experimentlm.generation import generate
from experimentlm.model import MultiHeadCausalSelfAttention, RoPE
from experimentlm.optim import build_optimizer, build_schedule


ROOT = Path(__file__).parents[1]
CONFIG = ROOT / "configs" / "gated_deltanet_qknorm.json"


class QKNORMTests(unittest.TestCase):
    def test_config_count_structure_and_optimizer_policy(self):
        config = ExperimentConfig.load(CONFIG)
        model = build_model(config.model)
        self.assertEqual(model.num_parameters, 10_445_230)
        self.assertEqual([type(block.attn).__name__ for block in model.blocks], [
            "MultiHeadCausalSelfAttention", "DeltaNetLinearAttention", "DeltaNetLinearAttention",
            "MultiHeadCausalSelfAttention", "DeltaNetLinearAttention", "DeltaNetLinearAttention",
            "MultiHeadCausalSelfAttention",
        ])
        qk = [
            parameter for name, parameter in model.named_parameters()
            if ".q_norm.weight" in name or ".k_norm.weight" in name
        ]
        self.assertEqual(sum(parameter.numel() for parameter in qk), 384)
        self.assertTrue(all(torch.equal(parameter, torch.ones_like(parameter)) for parameter in qk))
        optimizer = build_optimizer(config.training.optimizer, model.named_parameters())
        no_decay = {id(parameter) for group in optimizer.param_groups if group["weight_decay"] == 0
                    for parameter in group["params"]}
        self.assertTrue(all(id(parameter) in no_decay for parameter in qk))

    def test_normalization_v_untouched_and_forward_backward(self):
        attention = MultiHeadCausalSelfAttention(32, 4, 8, qk_norm=True).train()
        x = torch.randn(2, 7, 32, requires_grad=True)
        q, k, v = attention.qkv(x).view(2, 7, 3, 4, 8).unbind(2)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        original_v = v.clone()
        q, k = attention._normalize_qk(q, k)
        torch.testing.assert_close(q.float().square().mean(-1), torch.ones_like(q[..., 0]).float(), atol=3e-5, rtol=0)
        torch.testing.assert_close(k.float().square().mean(-1), torch.ones_like(k[..., 0]).float(), atol=3e-5, rtol=0)
        torch.testing.assert_close(v, original_v)
        output = attention(x, None, RoPE(16))
        output.square().mean().backward()
        self.assertTrue(all(parameter.grad is not None for parameter in attention.parameters()))
        self.assertEqual(v.shape, (2, 4, 7, 8))

    def test_cache_generation_and_checkpoint_round_trip(self):
        config = ExperimentConfig.load(CONFIG)
        model = build_model(config.model).eval()
        tokens = torch.randint(0, 8192, (1, 10))
        with torch.no_grad():
            full, _ = model(tokens)
            _, cache = model.forward_cached(tokens[:, :6])
            cached, _ = model.forward_cached(tokens[:, 6:7], cache)
            torch.testing.assert_close(cached[:, -1], full[:, 6], rtol=1e-5, atol=1e-5)
            self.assertEqual(generate(model, tokens[:, :3], max_new_tokens=2).shape, (1, 5))
        optimizer = build_optimizer(config.training.optimizer, model.parameters())
        scheduler = build_schedule(config.training.schedule, optimizer, 2)
        with tempfile.TemporaryDirectory() as directory:
            save_checkpoint(directory, model, optimizer, scheduler, 1, 8, config)
            restored, _ = load_checkpoint(directory)
            restored_logits, _ = restored(tokens)
        torch.testing.assert_close(restored_logits, full, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
