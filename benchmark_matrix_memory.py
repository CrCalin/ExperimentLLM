#!/usr/bin/env python3
"""Benchmark the standalone matrix-memory reference and exact chunked paths.

The default BF16 run records the current backend capability result.  Use
``--dtype float32`` for diagnostic chunked timings when BF16 triangular solve
support is unavailable.
"""
from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
from pathlib import Path
import platform
import time

import torch
import torch._dynamo as _dynamo

from experimentlm.matrix_memory import (
    BF16TriangularSolveUnavailable,
    MatrixMemoryBenchmarkConfig,
    chunked_matrix_memory,
    compact_wy_matrix_memory,
    fused_wy_matrix_memory,
    rank2_matrix_memory,
    sequential_matrix_memory,
)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _memory_gib(device: torch.device) -> float | None:
    if device.type != "cuda":
        return None
    return torch.cuda.max_memory_allocated(device) / 2**30


def _counter_value(counter, name: str) -> int:
    try:
        return sum(counter[name].values())
    except (KeyError, AttributeError):
        return 0


def _make_inputs(batch: int, length: int, d_memory: int, dtype: torch.dtype,
                 device: torch.device, backward: bool):
    tensors = [torch.randn(batch, length, d_memory, device=device, dtype=dtype)
               for _ in range(4)]
    for index in (1, 2):
        tensors[index] = tensors[index] / tensors[index].norm(dim=-1, keepdim=True).clamp_min(1e-6) * 0.5
    state = torch.randn(batch, d_memory, d_memory, device=device, dtype=dtype)
    if backward:
        for tensor in tensors + [state]:
            tensor.requires_grad_()
    return tensors + [state]


def _run(backend: str, inputs, chunk_size: int, backward: bool, profile: bool = False):
    q, e, k, v, state = inputs
    if backend == "reference":
        reads, final_state = sequential_matrix_memory(q, e, k, v, state)
    elif backend == "chunked":
        reads, final_state = chunked_matrix_memory(
            q, e, k, v, state, chunk_size=chunk_size, profile=profile,
        )
    elif backend == "wy":
        reads, final_state = compact_wy_matrix_memory(
            q, e, k, v, state, chunk_size=chunk_size, profile=profile,
        )
    elif backend == "fused_wy":
        reads, final_state = fused_wy_matrix_memory(
            q, e, k, v, state, chunk_size=chunk_size, profile=profile,
        )
    elif backend == "rank2":
        reads, final_state = rank2_matrix_memory(
            q, e, k, v, state, chunk_size=chunk_size, profile=profile,
        )
    else:
        raise ValueError(f"unknown backend: {backend}")
    if backward:
        loss = reads.square().mean() + final_state.square().mean()
        loss.backward()
    return reads, final_state


def _profile_once(backend: str, inputs, chunk_size: int, device: torch.device,
                  backward: bool):
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    started = time.perf_counter()
    with torch.profiler.profile(activities=activities, record_shapes=False) as profiler:
        _run(backend, inputs, chunk_size, backward=backward,
             profile=backend in {"chunked", "wy", "fused_wy", "rank2"})
        _sync(device)
    wall_seconds = time.perf_counter() - started
    by_name = {}
    for event in profiler.key_averages():
        if not event.key.startswith("matrix_memory/"):
            continue
        name = event.key.removeprefix("matrix_memory/")
        row = by_name.setdefault(name, {
            "name": name, "calls": 0, "self_cpu_us": 0.0,
            "self_device_us": 0.0, "total_cpu_us": 0.0, "total_device_us": 0.0,
        })
        row["calls"] += event.count
        row["self_cpu_us"] += event.self_cpu_time_total
        row["self_device_us"] += getattr(event, "self_device_time_total", 0.0)
        row["total_cpu_us"] += event.cpu_time_total
        row["total_device_us"] += getattr(event, "device_time_total", 0.0)
    device_events = [
        event for event in profiler.key_averages()
        if not event.key.startswith("matrix_memory/")
        and getattr(event, "device_time_total", 0.0) > 0
    ]
    return list(by_name.values()), wall_seconds, {
        "device_event_count_proxy": sum(event.count for event in device_events),
        "device_event_key_count_proxy": len(device_events),
    }


def run_case(backend: str, batch: int, length: int, d_memory: int, chunk_size: int,
             dtype: torch.dtype, device: torch.device, warmup: int, iterations: int,
             backward: bool, compile: bool, profile: bool) -> dict:
    inputs = _make_inputs(batch, length, d_memory, dtype, device, backward)
    compiled = False
    compile_seconds = 0.0
    graph_breaks = None
    fn = lambda: _run(backend, inputs, chunk_size, backward)
    if compile and backend in {"chunked", "wy", "fused_wy", "rank2"}:
        _dynamo.reset()
        from torch._dynamo.utils import counters
        counters.clear()
        fn = torch.compile(fn)
        started = time.perf_counter()
        try:
            fn()
            _sync(device)
        finally:
            compile_seconds = time.perf_counter() - started
        graph_breaks = _counter_value(counters, "graph_break")
        compiled = True
        for tensor in inputs:
            tensor.grad = None
    saved_tensor_bytes = None
    if backward and not compiled:
        saved = 0

        def pack(tensor):
            nonlocal saved
            saved += tensor.numel() * tensor.element_size()
            return tensor

        with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
            fn()
        _sync(device)
        saved_tensor_bytes = saved
        for tensor in inputs:
            tensor.grad = None
    for _ in range(warmup):
        fn()
        _sync(device)
        if backward:
            for tensor in inputs:
                tensor.grad = None
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    samples = []
    for _ in range(iterations):
        started = time.perf_counter()
        fn()
        _sync(device)
        samples.append(time.perf_counter() - started)
        if backward:
            for tensor in inputs:
                tensor.grad = None
    elapsed = sorted(samples)[len(samples) // 2]
    profile_events, profile_wall_seconds, profile_summary = (
        _profile_once(backend, inputs, chunk_size, device, backward)
        if profile else ([], None, {})
    )
    result = {
        "status": "ok",
        "backend": backend,
        "batch": batch,
        "sequence_length": length,
        "d_memory": d_memory,
        "chunk_size": chunk_size,
        "dtype": str(dtype).removeprefix("torch."),
        "backward": backward,
        "compiled": compiled,
        "compile_seconds": compile_seconds,
        "graph_breaks": graph_breaks,
        "seconds": elapsed,
        "tokens_per_second": batch * length / elapsed,
        "peak_vram_gib": _memory_gib(device),
        "profile": profile_events,
        "profile_wall_seconds": profile_wall_seconds,
        "profile_summary": profile_summary,
        "autograd_saved_tensor_bytes": saved_tensor_bytes,
    }
    del fn, inputs
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    if compile:
        _dynamo.reset()
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/matrix_memory_benchmark.json")
    parser.add_argument("--dtype", choices=["float32", "bfloat16"])
    parser.add_argument("--device")
    parser.add_argument("--backward", action=argparse.BooleanOptionalAction)
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction)
    parser.add_argument("--profile", action=argparse.BooleanOptionalAction)
    parser.add_argument("--output")
    parser.add_argument("--batches", type=int, nargs="+")
    parser.add_argument("--lengths", type=int, nargs="+")
    parser.add_argument("--dimensions", type=int, nargs="+")
    parser.add_argument("--chunks", type=int, nargs="+")
    parser.add_argument("--backends", nargs="+")
    parser.add_argument("--warmup", type=int)
    parser.add_argument("--iterations", type=int)
    return parser.parse_args()


def _config_from_args(args: argparse.Namespace) -> MatrixMemoryBenchmarkConfig:
    config = MatrixMemoryBenchmarkConfig.load(args.config)
    overrides = {}
    for name in ("dtype", "device", "backward", "compile", "profile", "output"):
        value = getattr(args, name)
        if value is not None:
            overrides[name] = value
    for cli_name, config_name in (
        ("batches", "batch_sizes"), ("lengths", "sequence_lengths"),
        ("dimensions", "memory_dimensions"), ("chunks", "chunk_sizes"),
        ("backends", "backends"),
    ):
        value = getattr(args, cli_name)
        if value is not None:
            overrides[config_name] = value
    for name in ("warmup", "iterations"):
        value = getattr(args, name)
        if value is not None:
            overrides[name] = value
    if overrides:
        value = config.to_dict()["args"]
        value.update(overrides)
        config = MatrixMemoryBenchmarkConfig.from_dict(value)
    return config


def _print_summary(results: list[dict]) -> None:
    good = [r for r in results if r["status"] == "ok"]
    if not good:
        return
    print("\nFastest chunk size by d_memory (successful optimized forward cases):")
    for backend in ("chunked", "wy", "fused_wy", "rank2"):
        for d_memory in sorted({r["d_memory"] for r in good if r["backend"] == backend and not r["backward"]}):
            rows = [r for r in good if r["backend"] == backend and not r["backward"]
                and r["d_memory"] == d_memory]
            if rows:
                fastest = max(rows, key=lambda r: r["tokens_per_second"])
                print(f"  {backend:7} d_memory={d_memory}: chunk={fastest['chunk_size']} "
                      f"{fastest['tokens_per_second']:,.0f} tok/s")
    representative = [r for r in good if r["batch"] == 8 and r["sequence_length"] == 2048
                       and r["d_memory"] == 64]
    for backward in (False, True):
        rows = [r for r in representative if r["backward"] == backward]
        if rows:
            print(f"\nRepresentative B=8 T=2048 d_memory=64 "
                  f"{'forward+backward' if backward else 'forward'}:")
            for row in rows:
                print(f"  {row['backend']:10} chunk={row['chunk_size']:4} "
                      f"{row['tokens_per_second']:,.0f} tok/s")


def main() -> None:
    args = parse_args()
    config = _config_from_args(args)
    device = torch.device(config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA/ROCm device requested but unavailable")
    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[config.dtype]
    results: list[dict] = []
    print(f"device={device} dtype={config.dtype} backward={config.backward} "
          f"compile={config.compile}")
    print(f"torch={torch.__version__} platform={platform.platform()}")
    if device.type == "cuda":
        print(f"device_name={torch.cuda.get_device_name(device)}")
    try:
        triton_version = importlib.metadata.version("triton")
    except importlib.metadata.PackageNotFoundError:
        try:
            import triton
            triton_version = getattr(triton, "__version__", None)
        except ImportError:
            triton_version = None
    for batch in config.batch_sizes:
        for length in config.sequence_lengths:
            for d_memory in config.memory_dimensions:
                for chunk_size in config.chunk_sizes:
                    for backend in config.backends:
                        try:
                            result = run_case(
                                backend, batch, length, d_memory, chunk_size, dtype,
                                device, config.warmup, config.iterations,
                                config.backward, config.compile, config.profile,
                            )
                        except BF16TriangularSolveUnavailable as exc:
                            result = {
                                "status": "unsupported", "backend": backend,
                                "batch": batch, "sequence_length": length,
                                "d_memory": d_memory, "chunk_size": chunk_size,
                                "dtype": config.dtype, "backward": config.backward,
                                "error": str(exc),
                            }
                        except RuntimeError as exc:
                            result = {
                                "status": "failed", "backend": backend,
                                "batch": batch, "sequence_length": length,
                                "d_memory": d_memory, "chunk_size": chunk_size,
                                "dtype": config.dtype, "backward": config.backward,
                                "error": f"{type(exc).__name__}: {exc}",
                            }
                            gc.collect()
                            if device.type == "cuda":
                                torch.cuda.empty_cache()
                            if config.compile:
                                _dynamo.reset()
                        results.append(result)
                        print(json.dumps(result, separators=(",", ":")), flush=True)
    output = Path(config.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": config.to_dict(),
        "environment": {
            "torch": torch.__version__, "platform": platform.platform(),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "rocm": getattr(torch.version, "hip", None),
            "triton": triton_version,
        },
        "results": results,
    }
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    _print_summary(results)
    print(f"\nSaved to {output}")


if __name__ == "__main__":
    main()
