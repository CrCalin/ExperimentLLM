import tempfile
import unittest

import torch

from experimentlm import ExperimentConfig, build_model
from experimentlm.checkpoint import load_checkpoint, save_checkpoint
from experimentlm.optim import build_optimizer, build_schedule


ROOT = __import__("pathlib").Path(__file__).parents[1]


class ChannelGatedResidualTests(unittest.TestCase):
    def test_config_count_shapes_init_forward_backward_and_checkpoint(self):
        config = ExperimentConfig.load(ROOT / "configs/gated_deltanet_qknorm_channel_gates.json")
        baseline = ExperimentConfig.load(ROOT / "configs/gated_deltanet_qknorm.json")
        model = build_model(config.model)
        self.assertEqual(model.num_parameters, 10_448_800)
        self.assertEqual(
            model.num_parameters - build_model(baseline.model).num_parameters, 3_570,
        )
        self.assertEqual(
            [type(block.residual).__name__ for block in model.blocks],
            ["ChannelGatedResidual"] * 7,
        )
        for block in model.blocks:
            self.assertEqual(block.residual.attention_gate.shape, (256,))
            self.assertEqual(block.residual.mlp_gate.shape, (256,))
            self.assertTrue(torch.equal(block.residual.attention_gate, torch.ones(256)))
            self.assertTrue(torch.equal(block.residual.mlp_gate, torch.ones(256)))

        optimizer = build_optimizer(config.training.optimizer, model.named_parameters())
        gate_ids = {
            id(parameter)
            for name, parameter in model.named_parameters()
            if ".residual." in name
        }
        gate_groups = [
            group for group in optimizer.param_groups
            if any(id(parameter) in gate_ids for parameter in group["params"])
        ]
        self.assertEqual(len(gate_groups), 1)
        self.assertEqual(gate_groups[0]["weight_decay"], 0.0)

        tokens = torch.randint(0, 8192, (2, 16))
        _, loss = model(tokens[:, :-1], tokens[:, 1:])
        loss.backward()
        for block in model.blocks:
            self.assertIsNotNone(block.residual.attention_gate.grad)
            self.assertIsNotNone(block.residual.mlp_gate.grad)
            self.assertEqual(block.residual.attention_gate.grad.shape, (256,))
            self.assertEqual(block.residual.mlp_gate.grad.shape, (256,))

        scheduler = build_schedule(config.training.schedule, optimizer, 2)
        with tempfile.TemporaryDirectory() as directory:
            save_checkpoint(directory, model, optimizer, scheduler, 1, 32, config)
            restored = build_model(config.model)
            restored_optimizer = build_optimizer(config.training.optimizer, restored.named_parameters())
            restored_scheduler = build_schedule(config.training.schedule, restored_optimizer, 2)
            load_checkpoint(directory, model=restored, optimizer=restored_optimizer, scheduler=restored_scheduler)
            self.assertTrue(all(torch.equal(a, b) for a, b in zip(model.parameters(), restored.parameters())))

    def test_torch_compile_forward_backward(self):
        config = ExperimentConfig.load(ROOT / "configs/gated_deltanet_qknorm_channel_gates.json")
        model = build_model(config.model)
        compiled = torch.compile(model, backend="eager")
        tokens = torch.randint(0, 8192, (1, 8))
        _, loss = compiled(tokens[:, :-1], tokens[:, 1:])
        loss.backward()
        self.assertTrue(torch.isfinite(loss))


if __name__ == "__main__":
    unittest.main()
