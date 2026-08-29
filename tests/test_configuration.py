import copy
import json
from pathlib import Path
import unittest

import torch

from experimentlm import ExperimentConfig, build_model
from experimentlm.resolution import resolve_experiment


BASELINE = Path(__file__).parents[1] / "configs" / "standard_transformer_plus.json"


class ConfigurationTests(unittest.TestCase):
    def test_baseline_is_expanded_and_parameter_stable(self):
        config = ExperimentConfig.load(BASELINE)
        self.assertEqual(len(config.model.layers), 7)
        self.assertEqual(build_model(config.model).num_parameters, 10_161_920)
        self.assertNotIn("repeat", config.model.to_dict()["layers"][0])

    def test_heterogeneous_attention_and_residual_stack(self):
        raw = json.loads(BASELINE.read_text())
        raw["model"].update(vocab_size=128, context_length=16, d_model=32)
        first = raw["model"]["layers"][0]
        first["repeat"] = 1
        first["args"]["attention"]["args"].update(n_heads=4, head_dim=8)
        first["args"]["mlp"]["args"]["d_ff"] = 64
        second = copy.deepcopy(first)
        second["args"]["attention"]["args"].update(n_heads=8, head_dim=4)
        second["args"]["residual"]["class"] = "StandardResidual"
        raw["model"]["layers"] = [first, second]
        config = resolve_experiment(ExperimentConfig.from_dict(raw))
        logits, loss = build_model(config.model)(
            torch.randint(0, 128, (2, 16)), torch.randint(0, 128, (2, 16))
        )
        self.assertEqual(logits.shape, (2, 16, 128))
        self.assertTrue(torch.isfinite(loss))

    def test_unknown_registered_class_is_rejected(self):
        raw = json.loads(BASELINE.read_text())
        raw["model"]["layers"][0]["class"] = "NotARealBlock"
        with self.assertRaisesRegex(ValueError, "unknown block class"):
            resolve_experiment(ExperimentConfig.from_dict(raw))


if __name__ == "__main__": unittest.main()
