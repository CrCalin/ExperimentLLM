import unittest
import copy
import json
from pathlib import Path

import torch

from experimentlm import ExperimentConfig, build_model


class DoubleNormTests(unittest.TestCase):
    def test_config_builds_separate_pre_and_post_norms(self):
        raw = json.loads((Path(__file__).parents[1] / "configs" / "standard_transformer_plus.json").read_text())
        raw["model"].update(vocab_size=128, context_length=16, d_model=32)
        layer = copy.deepcopy(raw["model"]["layers"][0])
        layer["class"] = "DoubleNormTransformerBlock"
        layer["args"] = {
            "pre_attn_norm": layer["args"]["norm"], "post_attn_norm": layer["args"]["norm"],
            "pre_mlp_norm": layer["args"]["norm"], "post_mlp_norm": layer["args"]["norm"],
            "attention": layer["args"]["attention"], "mlp": layer["args"]["mlp"],
        }
        layer["args"]["attention"]["args"].update(n_heads=4, head_dim=8)
        layer["args"]["mlp"]["args"]["d_ff"] = 64
        raw["model"]["layers"] = [layer]
        config = ExperimentConfig.from_dict(raw)
        model = build_model(config.model)
        block = model.blocks[0]

        self.assertIsNot(block.pre_attn_norm, block.post_attn_norm)
        self.assertIsNot(block.pre_mlp_norm, block.post_mlp_norm)
        self.assertTrue(all(isinstance(getattr(block, name), torch.nn.Module) for name in (
            "pre_attn_norm", "post_attn_norm", "pre_mlp_norm", "post_mlp_norm",
        )))
        inputs = torch.randint(0, 128, (2, 15))
        labels = torch.randint(0, 128, (2, 15))
        logits, loss = model(inputs, labels)
        self.assertEqual(logits.shape, (2, 15, 128))
        self.assertTrue(torch.isfinite(loss))


if __name__ == "__main__":
    unittest.main()
