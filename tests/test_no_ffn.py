import tempfile
import unittest

import torch

from experimentlm import ExperimentConfig, build_model
from experimentlm.checkpoint import load_checkpoint, save_checkpoint
from experimentlm.generation import generate
from experimentlm.optim import build_optimizer, build_schedule


def make_config(optimizer="AdamW"):
    optimizer_args = {"lr": 1e-3} if optimizer == "AdamW" else {
        "muon_lr": 0.02, "adamw_lr": 1e-3, "momentum": 0.95,
    }
    return ExperimentConfig.from_dict({
        "model": {
            "class": "CausalLM", "vocab_size": 32, "context_length": 8,
            "d_model": 16, "tie_embeddings": False,
            "position_encoding": {"class": "RoPE", "args": {}},
            "input_embedding": {"class": "TokenEmbedding", "args": {}},
            "layers": [
                {"class": "TransformerBlock", "args": {
                    "pre_norm": True, "norm": {"class": "RMSNorm", "args": {}},
                    "attention": {"class": "MultiHeadCausalSelfAttention", "args": {"n_heads": 2, "head_dim": 8}},
                    "mlp": {"class": "SwiGLU", "args": {"d_ff": 32}},
                    "residual": {"class": "GatedResidual", "args": {}},
                }},
                {"class": "TransformerBlock", "args": {
                    "pre_norm": True, "norm": {"class": "RMSNorm", "args": {}},
                    "attention": {"class": "DeltaNetLinearAttention", "args": {"n_heads": 2, "head_dim": 8, "expand_v": 1.0}},
                    "mlp": None, "residual": {"class": "GatedResidual", "args": {}},
                }},
            ],
            "final_norm": {"class": "RMSNorm", "args": {}},
            "output_head": {"class": "LinearLMHead", "args": {}},
            "initialization": {"class": "GPTStyleInit", "args": {}},
        },
        "training": {
            "optimizer": {"class": optimizer, "args": optimizer_args},
            "schedule": {"class": "CosineSchedule", "args": {}},
        },
    })


class NoFFNTests(unittest.TestCase):
    def test_structure_forward_backward_generation_and_checkpoint(self):
        config = make_config()
        from experimentlm.resolution import resolve_experiment
        config = resolve_experiment(config)
        model = build_model(config.model)
        no_ffn = model.blocks[1]
        self.assertIsNone(no_ffn.mlp)
        self.assertIsNone(no_ffn.norm2)
        self.assertFalse(hasattr(no_ffn.residual, "mlp_gate"))
        self.assertEqual(config.model.to_dict()["layers"][1]["args"]["mlp"], None)
        self.assertEqual(model.parameter_report["removed_vs_d_ff_768_all_layers"], 3 * 16 * 768 + 16 + 1)

        tokens = torch.randint(0, 32, (1, 8))
        logits, loss = model(tokens[:, :-1], tokens[:, 1:])
        loss.backward()
        self.assertEqual(generate(model, tokens[:, :3], max_new_tokens=2).shape, (1, 5))

        optimizer = build_optimizer(config.training.optimizer, model.named_parameters())
        scheduler = build_schedule(config.training.schedule, optimizer, 2)
        with tempfile.TemporaryDirectory() as directory:
            save_checkpoint(directory, model, optimizer, scheduler, 1, 8, config)
            restored, _ = load_checkpoint(directory)
            restored_logits, _ = restored(tokens[:, :-1])
        torch.testing.assert_close(restored_logits, logits, rtol=0, atol=0)

    @unittest.skipUnless(hasattr(torch.optim, "Muon"), "native torch.optim.Muon unavailable")
    def test_muonadamw_routes_only_existing_parameters(self):
        from experimentlm.resolution import resolve_experiment
        config = resolve_experiment(make_config("MuonAdamW"))
        model = build_model(config.model)
        optimizer = build_optimizer(config.training.optimizer, model.named_parameters())
        routed = {id(p) for group in optimizer.param_groups for p in group["params"]}
        self.assertEqual(routed, {id(p) for p in model.parameters()})
        self.assertFalse(any(name.startswith("blocks.1.mlp") for name, _ in model.named_parameters()))


if __name__ == "__main__":
    unittest.main()
