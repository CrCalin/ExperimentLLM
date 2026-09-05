import copy
import json
from pathlib import Path
import unittest

import torch

from experimentlm import ExperimentConfig, build_model
from experimentlm.components.memory import MatrixMemoryPath, matrix_memory_update
from experimentlm.resolution import resolve_experiment


BASELINE = Path(__file__).parents[1] / "configs" / "standard_transformer_plus.json"


def memory_config(*, backend="chunked", d_memory=4, layers=2, length=8,
                  fused_projections=True):
    raw = json.loads(BASELINE.read_text())
    raw["model"].update(vocab_size=32, context_length=length, d_model=8)
    layer = raw["model"]["layers"][0]
    layer["class"] = "MatrixMemoryTransformerBlock"
    layer["repeat"] = layers
    layer["args"]["attention"]["args"].update(n_heads=2, head_dim=4)
    layer["args"]["mlp"]["args"]["d_ff"] = 16
    layer["args"]["memory"] = {
        "class": "MatrixMemory",
        "args": {"d_memory": d_memory, "backend": backend,
                 "fused_projections": fused_projections},
    }
    return resolve_experiment(ExperimentConfig.from_dict(raw))


class MatrixMemoryMathTests(unittest.TestCase):
    def test_state_and_zero_read_shapes(self):
        for batch in (1, 2, 8):
            for width in (16, 32, 64, 128):
                path = MatrixMemoryPath(8, width, -4, -2, -2, 1e-6,
                                        True, True, False, True, True, True,
                                        "vector", "vector")
                state = torch.zeros(batch, width, width)
                injection, read = path.read(state, torch.randn(batch, 8))
                self.assertEqual(read.shape, (batch, width))
                self.assertEqual(injection.shape, (batch, 8))
                self.assertEqual(read.count_nonzero(), 0)

    def test_erase_delta_and_independent_addresses(self):
        state = torch.zeros(1, 3, 3)
        state[0, :, 0] = torch.tensor([2.0, 3.0, 4.0])
        e = torch.tensor([[1.0, 0.0, 0.0]])
        k = torch.tensor([[0.0, 1.0, 0.0]])
        value = torch.tensor([[5.0, 6.0, 7.0]])
        erased = matrix_memory_update(state, k, e, value, torch.ones(1, 3),
                                      torch.zeros(1, 3))
        torch.testing.assert_close(erased[0, :, 0], torch.zeros(3))
        unchanged = matrix_memory_update(state, k, e, value, torch.zeros(1, 3),
                                         torch.zeros(1, 3))
        torch.testing.assert_close(unchanged, state)
        written = matrix_memory_update(state, k, e, value, torch.ones(1, 3),
                                       torch.ones(1, 3))
        torch.testing.assert_close(written[0, :, 0], torch.zeros(3))
        torch.testing.assert_close(written[0, :, 1], value[0])
        repeated = matrix_memory_update(written, k, e, value, torch.zeros(1, 3),
                                        torch.ones(1, 3))
        torch.testing.assert_close(repeated, written)

    def test_read_gate_controls_injection(self):
        path = MatrixMemoryPath(8, 4, -4, -2, -2, 1e-6,
                                True, True, False, True, True, True,
                                "vector", "vector")
        state, hidden = torch.randn(2, 4, 4), torch.randn(2, 8)
        with torch.no_grad(): path.read_gate_logit.fill_(-30)
        quiet, _ = path.read(state, hidden)
        with torch.no_grad(): path.read_gate_logit.fill_(5)
        loud, _ = path.read(state, hidden)
        self.assertLess(quiet.norm(), loud.norm() * 1e-5)

    def test_fused_and_separate_projections_match(self):
        fused = MatrixMemoryPath(8, 4, -4, -2, -2, 1e-6,
                                 True, True, False, True, True, True,
                                 "vector", "vector", True)
        separate = MatrixMemoryPath(8, 4, -4, -2, -2, 1e-6,
                                    True, True, False, True, True, True,
                                    "vector", "vector", False)
        with torch.no_grad():
            separate.q_proj.weight.copy_(fused.read_projection.weight[:4])
            separate.q_atten.weight.copy_(fused.read_projection.weight[4:])
            separate.q_atten.bias.copy_(fused.read_atten_bias)
            modules = [separate.k_proj, separate.k_atten, separate.e_proj,
                       separate.e_atten, separate.v_proj,
                       separate.erase_strength, separate.write_strength]
            offset = 0
            for module in modules:
                width = module.weight.size(0)
                module.weight.copy_(fused.write_projection.weight[offset:offset + width])
                offset += width
            biases = fused.write_bias.split([4, 4, 4, 4])
            separate.k_atten.bias.copy_(biases[0])
            separate.e_atten.bias.copy_(biases[1])
            separate.erase_strength.bias.copy_(biases[2])
            separate.write_strength.bias.copy_(biases[3])
            separate.out_proj.weight.copy_(fused.out_proj.weight)
            separate.read_gate_logit.copy_(fused.read_gate_logit)
        state, read_hidden, write_hidden = torch.randn(2, 4, 4), torch.randn(2, 8), torch.randn(2, 8)
        fused_read, _ = fused.read(state, read_hidden)
        separate_read, _ = separate.read(state, read_hidden)
        torch.testing.assert_close(fused_read, separate_read)
        torch.testing.assert_close(fused.write(state, write_hidden), separate.write(state, write_hidden))


class MatrixMemoryIntegrationTests(unittest.TestCase):
    def test_sequence_shape_coverage(self):
        for batch, length in ((1, 1), (2, 17), (1, 128)):
            model = build_model(memory_config(
                backend="reference", layers=1, length=length,
            ).model)
            logits, _ = model(torch.randint(0, 32, (batch, length)))
            self.assertEqual(logits.shape, (batch, length, 32))

    def test_json_roundtrip_shapes_and_initialization(self):
        for width in (4, 8, 16):
            config = memory_config(d_memory=width)
            restored = ExperimentConfig.from_dict(config.to_dict())
            model = build_model(restored.model)
            logits, loss = model(torch.randint(0, 32, (2, 7)), torch.randint(0, 32, (2, 7)))
            self.assertEqual(logits.shape, (2, 7, 32))
            self.assertTrue(torch.isfinite(loss))
            path = model.blocks[0].memory.path
            torch.testing.assert_close(path.write_bias[2 * width:], torch.full((2 * width,), -2.0))

    def test_reference_optimized_forward_and_gradient_equivalence(self):
        torch.manual_seed(7)
        reference = build_model(memory_config(backend="reference").model)
        optimized = copy.deepcopy(reference)
        ids = torch.randint(0, 32, (2, 7))
        xr, xs = reference.wte(ids), optimized.wte(ids)
        yr, sr = reference.forward_matrix_memory(xr, backend="reference")
        ys, ss = optimized.forward_matrix_memory(xs, backend="chunked")
        torch.testing.assert_close(ys, yr, rtol=1e-5, atol=1e-6)
        for actual, expected in zip(ss, sr):
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        yr.square().mean().backward(); ys.square().mean().backward()
        for (name_r, p_r), (name_s, p_s) in zip(reference.named_parameters(), optimized.named_parameters()):
            self.assertEqual(name_r, name_s)
            if p_r.grad is None or p_s.grad is None:
                self.assertIsNone(p_r.grad); self.assertIsNone(p_s.grad)
                continue
            torch.testing.assert_close(p_s.grad, p_r.grad, rtol=2e-4, atol=2e-6, msg=name_r)
        for name, parameter in reference.named_parameters():
            if ".memory." in name:
                self.assertIsNotNone(parameter.grad, name)

    def test_causality_reset_and_serialization(self):
        torch.manual_seed(8)
        model = build_model(memory_config().model).eval()
        prefix = torch.randint(0, 32, (2, 4))
        first = torch.cat((prefix, torch.randint(0, 32, (2, 4))), 1)
        second = torch.cat((prefix, torch.randint(0, 32, (2, 4))), 1)
        with torch.no_grad():
            out1, states1 = model.forward_matrix_memory(model.wte(first))
            out2, _ = model.forward_matrix_memory(model.wte(second))
            repeated, states2 = model.forward_matrix_memory(model.wte(first))
        torch.testing.assert_close(out1[:, :4], out2[:, :4])
        torch.testing.assert_close(out1, repeated)
        for actual, expected in zip(states1, states2):
            torch.testing.assert_close(actual, expected)
        clone = build_model(memory_config().model)
        clone.load_state_dict(model.state_dict())
        self.assertFalse(any("state" in key for key in model.state_dict()))

    def test_local_isolation_and_residual_propagation(self):
        model = build_model(memory_config(length=4).model)
        self.assertIsNot(model.blocks[0].memory.path, model.blocks[1].memory.path)
        hidden = model.wte(torch.randint(0, 32, (1, 4)))
        output, states = model.forward_matrix_memory(hidden, backend="reference")
        self.assertEqual(len(states), 2)
        self.assertNotEqual(states[0].data_ptr(), states[1].data_ptr())
        output[:, -1].sum().backward()
        # Layer 1 memory affects layer 2 only through layer 1's output residual.
        self.assertGreater(model.blocks[0].memory.path.out_proj.weight.grad.abs().sum(), 0)
        self.assertGreater(model.blocks[1].memory.path.out_proj.weight.grad.abs().sum(), 0)

    def test_cached_generation_matches_full(self):
        model = build_model(memory_config().model).eval()
        ids = torch.randint(0, 32, (1, 8))
        with torch.no_grad():
            full, _ = model(ids)
            cache = None; pieces = []
            for token in ids.split(1, dim=1):
                logits, cache = model.forward_cached(token, cache)
                pieces.append(logits)
        torch.testing.assert_close(torch.cat(pieces, dim=1), full, rtol=1e-5, atol=1e-5)

    def test_memory_disabled_baseline_unchanged(self):
        config = ExperimentConfig.load(BASELINE)
        self.assertEqual(build_model(config.model).num_parameters, 10_161_920)

    def test_removed_global_configuration_is_rejected(self):
        raw = memory_config().to_dict()
        raw["model"]["layers"][0]["args"]["memory"]["args"]["global_memory"] = True
        with self.assertRaisesRegex(ValueError, "unknown args for memory"):
            resolve_experiment(ExperimentConfig.from_dict(raw))

    def test_mixed_memory_and_ordinary_layers(self):
        raw = memory_config(backend="reference", layers=2).to_dict()
        ordinary = copy.deepcopy(raw["model"]["layers"][1])
        ordinary["class"] = "TransformerBlock"
        del ordinary["args"]["memory"]
        raw["model"]["layers"][1] = ordinary
        model = build_model(resolve_experiment(ExperimentConfig.from_dict(raw)).model)
        logits, _ = model(torch.randint(0, 32, (1, 8)))
        self.assertEqual(logits.shape, (1, 8, 32))

    def test_compile_capture(self):
        model = build_model(memory_config(length=4).model).eval()
        compiled = torch.compile(model, backend="eager")
        logits, _ = compiled(torch.randint(0, 32, (1, 4)))
        self.assertEqual(logits.shape, (1, 4, 32))

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA or ROCm")
    def test_bf16_long_sequence_stability(self):
        config = memory_config(d_memory=32, layers=1, length=2048)
        model = build_model(config.model).cuda().eval()
        ids = torch.randint(0, 32, (1, 2048), device="cuda")
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            hidden, states = model.forward_matrix_memory(model.wte(ids))
        self.assertTrue(torch.isfinite(hidden).all())
        self.assertTrue(all(torch.isfinite(state).all() for state in states))
        self.assertTrue(all(state.dtype == torch.bfloat16 for state in states))


if __name__ == "__main__": unittest.main()
