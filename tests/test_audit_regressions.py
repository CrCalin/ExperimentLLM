import random
import tempfile
import unittest
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from benchmark import local_validation
from experimentlm.checkpoint import load_checkpoint, save_checkpoint
from experimentlm.config import ExperimentConfig
from experimentlm.model import (
    GroupedQueryCausalSelfAttention,
    MultiHeadCausalSelfAttention,
    RoPE,
)
from experimentlm.optim import build_optimizer, build_schedule
from experimentlm import build_model
import torch.nn.functional as F


class CachedAttentionTests(unittest.TestCase):
    def test_multi_token_cached_matches_full_for_mha_and_qk_norm(self):
        for qk_norm in (False, True):
            attention = MultiHeadCausalSelfAttention(32, 4, 8, qk_norm=qk_norm).eval()
            self._assert_cached_matches_full(attention)

    def test_multi_token_cached_matches_full_for_gqa(self):
        for kv_heads in (2, 1):
            attention = GroupedQueryCausalSelfAttention(32, 4, kv_heads, 8).eval()
            self._assert_cached_matches_full(attention)

    @staticmethod
    def _assert_cached_matches_full(attention):
        torch.manual_seed(3)
        x = torch.randn(2, 9, 32)
        rope = RoPE(32)
        with torch.no_grad():
            full = attention(x, None, rope)
            _, cache = attention.forward_cached(x[:, :4], rope)
            cached, _ = attention.forward_cached(x[:, 4:], rope, cache)
        error = (cached - full[:, 4:]).abs()
        print(f"{type(attention).__name__}: max={error.max().item():.3e} mean={error.mean().item():.3e}")
        torch.testing.assert_close(cached, full[:, 4:], rtol=1e-5, atol=1e-5)


class CheckpointRNGTests(unittest.TestCase):
    def test_checkpoint_resume_restores_all_cpu_rng_streams(self):
        config = ExperimentConfig.load(Path(__file__).parents[1] / "configs" / "standard_transformer_plus.json")
        config = ExperimentConfig.from_dict({**config.to_dict(), "model": {
            **config.model.to_dict(), "vocab_size": 32, "context_length": 8, "d_model": 32,
            "layers": [dict(config.model.to_dict()["layers"][0], args={
                **config.model.to_dict()["layers"][0]["args"],
                "attention": {"class": "MultiHeadCausalSelfAttention", "args": {"n_heads": 4, "head_dim": 8, "bias": False, "dropout": 0.0}},
                "mlp": {"class": "SwiGLU", "args": {"d_ff": 64, "bias": False, "dropout": 0.0}},
            })],
        }})
        model = build_model(config.model)
        optimizer = build_optimizer(config.training.optimizer, model.parameters())
        scheduler = build_schedule(config.training.schedule, optimizer, 2)
        random.seed(10); np.random.seed(10); torch.manual_seed(10)
        with tempfile.TemporaryDirectory() as directory:
            save_checkpoint(directory, model, optimizer, scheduler, 1, 8, config)
            inputs = torch.tensor([[1, 2, 3, 4]])
            labels = torch.tensor([[2, 3, 4, 5]])
            optimizer.zero_grad(set_to_none=True)
            _, uninterrupted_loss = model(inputs, labels)
            uninterrupted_loss.backward(); optimizer.step(); scheduler.step()
            expected_weights = {name: value.detach().clone() for name, value in model.state_dict().items()}
            expected = (random.random(), np.random.rand(), torch.rand(4))
            random.seed(99); np.random.seed(99); torch.manual_seed(99)
            resumed_model = build_model(config.model)
            resumed_optimizer = build_optimizer(config.training.optimizer, resumed_model.parameters())
            resumed_scheduler = build_schedule(config.training.schedule, resumed_optimizer, 2)
            with warnings.catch_warnings(record=True) as caught:
                resumed, _ = load_checkpoint(directory, model=resumed_model, optimizer=resumed_optimizer, scheduler=resumed_scheduler)
            self.assertFalse(caught)
            actual = (random.random(), np.random.rand(), torch.rand(4))
            resumed_optimizer.zero_grad(set_to_none=True)
            _, resumed_loss = resumed(inputs, labels)
            resumed_loss.backward(); resumed_optimizer.step(); resumed_scheduler.step()
            for name, value in resumed.state_dict().items():
                torch.testing.assert_close(value, expected_weights[name], rtol=0, atol=0)
        self.assertEqual(actual[0], expected[0])
        self.assertEqual(actual[1], expected[1])
        torch.testing.assert_close(actual[2], expected[2], rtol=0, atol=0)


class BenchmarkPrecisionTests(unittest.TestCase):
    def test_local_validation_selects_requested_autocast(self):
        class FakeModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.config = SimpleNamespace(experiment_model_config={"context_length": 4})
                self.seen = []

            def forward(self, input_ids, labels):
                self.seen.append((torch.is_autocast_enabled(input_ids.device.type), torch.get_autocast_dtype(input_ids.device.type)))
                return SimpleNamespace(loss=torch.tensor(0.0))

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "eval.bin"
            np.arange(20, dtype=np.uint16).tofile(path)
            for precision, expected in (("bfloat16", (True, torch.bfloat16)), ("float16", (True, torch.float16)), ("none", (False, torch.get_autocast_dtype("cpu")))):
                model = FakeModel()
                local_validation(directory, path, torch.device("cpu"), 1, 1, model=model, precision=precision)
                self.assertEqual(model.seen[0], expected)


class CausalLossTests(unittest.TestCase):
    def test_loss_uses_already_shifted_next_token_labels(self):
        raw = {
            "class": "CausalLM", "vocab_size": 5, "context_length": 4, "d_model": 4,
            "tie_embeddings": False,
            "position_encoding": {"class": "RoPE", "args": {}},
            "input_embedding": {"class": "TokenEmbedding", "args": {}},
            "layers": [{"class": "TransformerBlock", "args": {
                "pre_norm": True, "norm": {"class": "RMSNorm", "args": {}},
                "attention": {"class": "MultiHeadCausalSelfAttention", "args": {"n_heads": 1, "head_dim": 4}},
                "mlp": None, "residual": {"class": "GatedResidual", "args": {},},
            }}],
            "final_norm": {"class": "RMSNorm", "args": {}},
            "output_head": {"class": "LinearLMHead", "args": {}},
            "initialization": {"class": "GPTStyleInit", "args": {}},
        }
        model = build_model(ExperimentConfig.from_dict({"model": raw, "training": {
            "optimizer": {"class": "AdamW", "args": {}}, "schedule": {"class": "CosineSchedule", "args": {}},
        }, "data": {}, "evaluation": {}}).model).eval()
        inputs = torch.tensor([[0, 1]])
        labels = torch.tensor([[1, 2]])
        logits, loss = model(inputs, labels)
        expected = F.cross_entropy(logits.float().reshape(-1, 5), labels.reshape(-1))
        torch.testing.assert_close(loss, expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
