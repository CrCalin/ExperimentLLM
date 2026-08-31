import unittest

import torch

from experimentlm import ExperimentConfig, build_model
from experimentlm.resolution import resolve_experiment


class GatedResidualTests(unittest.TestCase):
    def _config(self, attention_gate_init=1.0, mlp_gate_init=1.0):
        return resolve_experiment(ExperimentConfig.from_dict({
            "model": {
                "class": "CausalLM", "vocab_size": 32, "context_length": 8,
                "d_model": 16, "tie_embeddings": False,
                "position_encoding": {"class": "RoPE", "args": {}},
                "input_embedding": {"class": "TokenEmbedding", "args": {}},
                "layers": [{"class": "TransformerBlock", "args": {
                    "pre_norm": True,
                    "norm": {"class": "RMSNorm", "args": {}},
                    "attention": {"class": "MultiHeadCausalSelfAttention", "args": {"n_heads": 2, "head_dim": 8}},
                    "mlp": {"class": "SwiGLU", "args": {"d_ff": 32}},
                    "residual": {"class": "GatedResidual", "args": {
                        "attention_gate_init": attention_gate_init, "mlp_gate_init": mlp_gate_init,
                    }},
                }}],
                "final_norm": {"class": "RMSNorm", "args": {}},
                "output_head": {"class": "LinearLMHead", "args": {}},
                "initialization": {"class": "GPTStyleInit", "args": {}},
            },
            "training": {
                "optimizer": {"class": "AdamW", "args": {"lr": 0.001}},
                "schedule": {"class": "CosineSchedule", "args": {}},
            },
        }))

    def test_gates_are_configured_trainable_scalars_and_forward_works(self):
        model = build_model(self._config(0.25, 0.5).model)
        residual = model.blocks[0].residual
        self.assertEqual(residual.attention_gate.shape, torch.Size([]))
        self.assertEqual(residual.mlp_gate.shape, torch.Size([]))
        self.assertTrue(residual.attention_gate.requires_grad)
        self.assertAlmostEqual(residual.attention_gate.item(), 0.25)
        self.assertAlmostEqual(residual.mlp_gate.item(), 0.5)
        tokens = torch.randint(0, 32, (2, 8))
        logits, loss = model(tokens[:, :-1], tokens[:, 1:])
        self.assertEqual(logits.shape, (2, 7, 32))
        self.assertTrue(torch.isfinite(loss))

    def test_plus_sized_parameter_count_adds_fourteen_scalars(self):
        config = ExperimentConfig.load("configs/gated_residual_transformer.json")
        self.assertEqual(build_model(config.model).num_parameters, 10161934)


if __name__ == "__main__":
    unittest.main()
