"""Focused BF16 forward/forward-backward benchmark for auxiliary matrix memory."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import gc
import json
from pathlib import Path
import time

import torch

from experimentlm import ExperimentConfig, build_model
from experimentlm.resolution import resolve_experiment


def make_config(length: int, d_memory: int | None, fused_projections: bool = True):
    raw = json.loads(Path("configs/standard_transformer_plus.json").read_text())
    raw["model"]["context_length"] = length
    if d_memory is not None:
        layer = raw["model"]["layers"][0]
        layer["class"] = "MatrixMemoryTransformerBlock"
        layer["args"]["memory"] = {"class": "MatrixMemory", "args": {
            "d_memory": d_memory, "backend": "chunked",
            "fused_projections": fused_projections,
        }}
    return resolve_experiment(ExperimentConfig.from_dict(raw)).model


def benchmark(config, batch: int, iterations: int, backward: bool):
    model = torch.compile(build_model(config).cuda().train(backward))
    inputs = torch.randint(0, config.vocab_size, (batch, config.context_length), device="cuda")
    labels = torch.randint_like(inputs, high=config.vocab_size)

    def iteration():
        _, loss = model(inputs, labels if backward else None)
        if backward:
            loss.backward()
            model.zero_grad(set_to_none=True)

    grad_context = nullcontext() if backward else torch.no_grad()
    with grad_context, torch.autocast("cuda", dtype=torch.bfloat16):
        iteration(); torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        samples = []
        for _ in range(iterations):
            started = time.perf_counter(); iteration(); torch.cuda.synchronize()
            samples.append(time.perf_counter() - started)
    elapsed = sorted(samples)[len(samples) // 2]
    result = {
        "seconds": elapsed,
        "tokens_per_second": batch * config.context_length / elapsed,
        "peak_vram_gib": torch.cuda.max_memory_allocated() / 2**30,
        "parameters": model.num_parameters,
        "memory_parameters": model.parameter_report.get("matrix_memory", 0),
    }
    del model, inputs, labels
    torch._dynamo.reset(); gc.collect(); torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lengths", type=int, nargs="+", default=[512, 2048])
    parser.add_argument("--dimensions", type=int, nargs="+", default=[32, 64, 128, 256])
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--backward", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError("benchmark requires CUDA or ROCm")
    results = []
    for length in args.lengths:
        baseline = make_config(length, None)
        results.append({"length": length, "variant": "baseline", **benchmark(
            baseline, args.batch, args.iterations, args.backward)})
        del baseline; torch.cuda.empty_cache()
        for dimension in args.dimensions:
            config = make_config(length, dimension)
            results.append({"length": length, "d_memory": dimension,
                            "variant": "local", **benchmark(
                                config, args.batch, args.iterations, args.backward)})
            torch.cuda.empty_cache()
    print(json.dumps(results, indent=2))


if __name__ == "__main__": main()
