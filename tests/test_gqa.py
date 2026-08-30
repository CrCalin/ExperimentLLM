import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from experimentlm import ExperimentConfig, build_model
from experimentlm.checkpoint import load_checkpoint, save_checkpoint
from experimentlm.generation import generate
from experimentlm.model import DeltaNetLinearAttention, GroupedQueryCausalSelfAttention, MultiHeadCausalSelfAttention
from experimentlm.optim import build_optimizer, build_schedule


ROOT = Path(__file__).parents[1]


class GQATests(unittest.TestCase):
    def _model(self, name):
        return build_model(ExperimentConfig.load(ROOT / "configs" / name).model)

    def test_both_configs_structure_forward_backward_and_counts(self):
        for name, kv_heads in (("gated_deltanet_gqa_2kv.json", 2), ("gated_deltanet_gqa_1kv.json", 1)):
            model = self._model(name)
            expected_total = 10_444_846 - 3 * ((4 - kv_heads) * 2 * 64 * 256)
            self.assertEqual(model.num_parameters, expected_total)
            self.assertEqual([type(block.attn).__name__ for block in model.blocks], [
                "GroupedQueryCausalSelfAttention", "DeltaNetLinearAttention", "DeltaNetLinearAttention",
                "GroupedQueryCausalSelfAttention", "DeltaNetLinearAttention", "DeltaNetLinearAttention",
                "GroupedQueryCausalSelfAttention",
            ])
            for block in model.blocks[::3]:
                self.assertEqual((block.attn.n_q_heads, block.attn.n_kv_heads), (4, kv_heads))
                self.assertEqual(sum(p.numel() for p in block.attn.parameters()), (4 + 2 * kv_heads) * 64 * 256 + 256 * 256)
            tokens = torch.randint(0, 8192, (2, 13))
            logits, loss = model(tokens, tokens)
            self.assertEqual(logits.shape, (2, 13, 8192))
            loss.backward()
            self.assertTrue(all(p.grad is not None for p in model.parameters()))

    def test_validation_and_rope_shapes(self):
        with self.assertRaises(ValueError):
            GroupedQueryCausalSelfAttention(32, n_q_heads=3, n_kv_heads=2, head_dim=8)
        attention = GroupedQueryCausalSelfAttention(32, n_q_heads=4, n_kv_heads=2, head_dim=8)
        from experimentlm.model import RoPE
        x = torch.randn(2, 7, 32)
        output = attention(x, None, RoPE(16))
        self.assertEqual(output.shape, x.shape)
        changed = x.clone()
        changed[:, 4:] = torch.randn_like(changed[:, 4:])
        torch.testing.assert_close(attention(x, None, RoPE(16))[:, :4], attention(changed, None, RoPE(16))[:, :4])

    def test_cache_generation_and_checkpoint_round_trip(self):
        config = ExperimentConfig.load(ROOT / "configs/gated_deltanet_gqa_2kv.json")
        model = self._model("gated_deltanet_gqa_2kv.json").eval()
        tokens = torch.randint(0, 8192, (1, 12))
        with torch.no_grad():
            _, cache = model.forward_cached(tokens[:, :8])
            cached, cache = model.forward_cached(tokens[:, 8:9], cache)
            full, _ = model(tokens[:, :9])
            self.assertEqual(cache["layers"][0][0].shape[1], 2)
            torch.testing.assert_close(cached[:, -1], full[:, -1], rtol=1e-5, atol=1e-5)
            self.assertEqual(generate(model, tokens[:, :3], max_new_tokens=2).shape, (1, 5))
        optimizer = build_optimizer(config.training.optimizer, model.parameters())
        scheduler = build_schedule(config.training.schedule, optimizer, 2)
        with tempfile.TemporaryDirectory() as directory:
            save_checkpoint(directory, model, optimizer, scheduler, 1, 8, config)
            restored, _ = load_checkpoint(directory)
            restored_logits, _ = restored(tokens[:, :9])
        torch.testing.assert_close(restored_logits, full, rtol=0, atol=0)

    def test_existing_mha_is_unchanged(self):
        model = self._model("standard_transformer_plus.json")
        self.assertTrue(all(isinstance(block.attn, MultiHeadCausalSelfAttention) for block in model.blocks))
        self.assertEqual(model.num_parameters, 10_161_920)

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA or ROCm")
    def test_cuda_deltanet_missing_fused_kernel_is_hard_error(self):
        import experimentlm.model as model_module
        attention = DeltaNetLinearAttention(32, n_heads=4, head_dim=8, expand_v=1.0).to("cuda")
        with patch.object(model_module, "fused_recurrent_gated_delta_rule", None):
            with self.assertRaisesRegex(RuntimeError, "fused_recurrent_gated_delta_rule"):
                attention(torch.randn(1, 4, 32, device="cuda"), None, None)


if __name__ == "__main__":
    unittest.main()
