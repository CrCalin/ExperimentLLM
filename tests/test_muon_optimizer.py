import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import torch

from experimentlm import ExperimentConfig, build_model
from experimentlm.checkpoint import load_checkpoint, save_checkpoint
from experimentlm.optim import build_optimizer, build_schedule
from experimentlm.resolution import resolve_experiment


@unittest.skipUnless(hasattr(torch.optim, "Muon"), "native torch.optim.Muon unavailable")
class MuonAdamWTests(unittest.TestCase):
    def _config(self):
        return resolve_experiment(ExperimentConfig.from_dict({
            "model": {
                "class": "CausalLM", "vocab_size": 32, "context_length": 8,
                "d_model": 16, "tie_embeddings": False,
                "position_encoding": {"class": "RoPE", "args": {}},
                "input_embedding": {"class": "TokenEmbedding", "args": {}},
                "layers": [
                    {"class": "TransformerBlock", "args": {
                        "pre_norm": True,
                        "norm": {"class": "RMSNorm", "args": {}},
                        "attention": {"class": "MultiHeadCausalSelfAttention", "args": {
                            "n_heads": 2, "head_dim": 8,
                        }},
                        "mlp": {"class": "SwiGLU", "args": {"d_ff": 32}},
                        "residual": {"class": "GatedResidual", "args": {}},
                    }},
                    {"class": "TransformerBlock", "args": {
                        "pre_norm": True,
                        "norm": {"class": "RMSNorm", "args": {}},
                        "attention": {"class": "DeltaNetLinearAttention", "args": {
                            "n_heads": 2, "head_dim": 8, "expand_v": 1.0,
                        }},
                        "mlp": {"class": "SwiGLU", "args": {"d_ff": 32}},
                        "residual": {"class": "GatedResidual", "args": {}},
                    }},
                ],
                "final_norm": {"class": "RMSNorm", "args": {}},
                "output_head": {"class": "LinearLMHead", "args": {}},
                "initialization": {"class": "GPTStyleInit", "args": {}},
            },
            "training": {
                "optimizer": {"class": "MuonAdamW", "args": {
                    "muon_lr": 0.02, "adamw_lr": 0.001,
                    "momentum": 0.95, "weight_decay": 0.1,
                    "adamw_betas": [0.9, 0.95], "adamw_eps": 1e-8,
                    "adjust_lr_fn": "match_rms_adamw",
                }},
                "schedule": {"class": "CosineSchedule", "args": {
                    "warmup_fraction": 0.5, "final_lr_fraction": 0.1,
                }},
            },
        }))

    @staticmethod
    def _routes(model, optimizer):
        names = {id(parameter): name for name, parameter in model.named_parameters()}
        return {
            group["optimizer_name"]: {
                names[id(parameter)] for candidate in optimizer.param_groups
                if candidate["optimizer_name"] == group["optimizer_name"]
                for parameter in candidate["params"]
            }
            for group in optimizer.param_groups
        }

    def test_parameter_routing(self):
        config = self._config()
        model = build_model(config.model)
        optimizer = build_optimizer(config.training.optimizer, model.named_parameters())
        routes = self._routes(model, optimizer)
        muon, adamw = routes["muon"], routes["adamw"]

        self.assertIn("blocks.0.attn.qkv.weight", muon)
        self.assertIn("blocks.0.attn.out.weight", muon)
        self.assertIn("blocks.0.mlp.gate.weight", muon)
        self.assertIn("blocks.0.mlp.down.weight", muon)
        self.assertIn("blocks.1.attn.o_proj.weight", muon)
        self.assertNotIn("wte.weight", muon)
        self.assertNotIn("lm_head.weight", muon)

        self.assertIn("wte.weight", adamw)
        self.assertIn("lm_head.weight", adamw)
        self.assertIn("blocks.0.norm1.weight", adamw)
        self.assertIn("blocks.0.residual.attention_gate", adamw)
        self.assertIn("blocks.1.attn.A_log", adamw)
        self.assertIn("blocks.1.attn.dt_bias", adamw)
        self.assertIn("blocks.1.attn.in_proj_weight", adamw)
        self.assertIn("blocks.1.attn.conv_weight", adamw)

    def test_step_schedule_and_checkpoint_roundtrip(self):
        torch.manual_seed(4)
        config = self._config()
        model = build_model(config.model)
        optimizer = build_optimizer(config.training.optimizer, model.named_parameters())
        scheduler = build_schedule(config.training.schedule, optimizer, 4)
        self.assertEqual(
            [group["lr"] for group in optimizer.param_groups],
            [0.01, 0.0005, 0.0005],
        )
        initial = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
        tokens = torch.randint(0, config.model.vocab_size, (1, 8))
        _, loss = model(tokens, tokens)
        loss.backward()
        optimizer.step()
        scheduler.step()

        changed = {name for name, parameter in model.named_parameters()
                   if not torch.equal(parameter, initial[name])}
        self.assertIn("blocks.0.attn.qkv.weight", changed)
        self.assertIn("wte.weight", changed)
        self.assertEqual(
            [group["lr"] for group in optimizer.param_groups],
            [0.02, 0.001, 0.001],
        )
        with tempfile.TemporaryDirectory() as directory:
            save_checkpoint(directory, model, optimizer, scheduler, 1, 8, config)
            restored_model = build_model(config.model)
            restored_optimizer = build_optimizer(
                config.training.optimizer, restored_model.named_parameters(),
            )
            restored_scheduler = build_schedule(config.training.schedule, restored_optimizer, 4)
            load_checkpoint(
                directory, model=restored_model, optimizer=restored_optimizer,
                scheduler=restored_scheduler,
            )
        self.assertEqual(len(restored_optimizer.optimizers[0].state),
                         len(optimizer.optimizers[0].state))
        self.assertEqual(len(restored_optimizer.optimizers[1].state),
                         len(optimizer.optimizers[1].state))
        self.assertEqual(restored_scheduler.state_dict(), scheduler.state_dict())
        for expected, actual in zip(model.parameters(), restored_model.parameters()):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_native_muon_is_required(self):
        config = self._config()
        model = build_model(config.model)
        with patch.object(torch.optim, "Muon", None):
            with self.assertRaisesRegex(RuntimeError, "native torch.optim.Muon"):
                build_optimizer(config.training.optimizer, model.named_parameters())


if __name__ == "__main__":
    unittest.main()
