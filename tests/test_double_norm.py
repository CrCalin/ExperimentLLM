import unittest

import torch

from experimentlm import ExperimentConfig, build_model


class DoubleNormTests(unittest.TestCase):
    def test_config_builds_separate_pre_and_post_norms(self):
        config = ExperimentConfig.load("configs/double_norm_transformer.json")
        model = build_model(config.model)
        block = model.blocks[0]

        self.assertEqual(model.num_parameters, 10165504)
        self.assertIsNot(block.pre_attn_norm, block.post_attn_norm)
        self.assertIsNot(block.pre_mlp_norm, block.post_mlp_norm)
        self.assertTrue(all(isinstance(getattr(block, name), torch.nn.Module) for name in (
            "pre_attn_norm", "post_attn_norm", "pre_mlp_norm", "post_mlp_norm",
        )))
        logits, loss = model(torch.randint(0, 8192, (2, 16)), torch.randint(0, 8192, (2, 16)))
        self.assertEqual(logits.shape, (2, 16, 8192))
        self.assertTrue(torch.isfinite(loss))


if __name__ == "__main__":
    unittest.main()
