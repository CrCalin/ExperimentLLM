#!/usr/bin/env python3
"""Cost model for a future per-layer BF16 matrix-memory branch.

This file is deliberately benchmark-only.  It never changes or imports a
Transformer block implementation for integration.  H and Y are generated
fully before the branch is called, matching the proposed layer boundary.
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
from torch import nn

from experimentlm.config import ExperimentConfig
from experimentlm.matrix_memory import compact_wy_matrix_memory
from experimentlm.model import build_model


DEFAULT_DEVICE = "cuda"
DEFAULT_D_MODEL = 256
DEFAULT_CHUNK = 128
EPS = 1e-6


def triton_version() -> str | None:
    try:
        return importlib.metadata.version("triton")
    except importlib.metadata.PackageNotFoundError:
        return None


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def autocast(device: torch.device):
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                          enabled=device.type in {"cuda", "cpu"})


def reset_grads(module: nn.Module, tensors: list[torch.Tensor]) -> None:
    for parameter in module.parameters():
        parameter.grad = None
    for tensor in tensors:
        if tensor.grad is not None:
            tensor.grad = None


def _normalize(address: torch.Tensor) -> torch.Tensor:
    norm = address.float().pow(2).sum(dim=-1, keepdim=True).sqrt()
    return (address / (norm.to(address.dtype) + EPS)).to(address.dtype)


class ReadControls(nn.Module):
    def __init__(self, d_model: int, d_memory: int, fused: bool,
                 gate: bool = True, q_source: str = "projection"):
        super().__init__()
        if q_source not in {"projection", "slice"}:
            raise ValueError("q_source must be projection or slice")
        self.d_memory = d_memory
        self.fused = fused
        self.gate = gate
        self.q_source = q_source
        if q_source == "projection":
            if fused:
                self.proj = nn.Linear(d_model, (2 if gate else 1) * d_memory, bias=False)
            else:
                self.q_proj = nn.Linear(d_model, d_memory, bias=False)
                if gate:
                    self.g_proj = nn.Linear(d_model, d_memory, bias=False)
        elif gate:
            # q reuse removes only W_q; retain W_gq so this ablation isolates
            # reuse of an existing representation from removal of attenuation.
            self.g_proj = nn.Linear(d_model, d_memory, bias=False)
        if gate:
            self.gate_bias = nn.Parameter(torch.zeros(d_memory))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        if self.q_source == "slice":
            q_raw = h[..., :self.d_memory].to(torch.bfloat16)
            gate_raw = self.g_proj(h) if self.gate else None
        elif self.fused:
            projected = self.proj(h)
            if self.gate:
                q_raw, gate_raw = projected.split(self.d_memory, dim=-1)
            else:
                q_raw, gate_raw = projected, None
        else:
            q_raw = self.q_proj(h)
            gate_raw = self.g_proj(h) if self.gate else None
        q = _normalize(q_raw)
        if self.gate:
            q = q * torch.sigmoid(gate_raw + self.gate_bias).to(q.dtype)
        return q


class WriteControls(nn.Module):
    def __init__(self, d_model: int, d_memory: int, fused: bool,
                 erase_gate: bool = True, write_gate: bool = True):
        super().__init__()
        self.d_memory = d_memory
        self.fused = fused
        self.erase_gate = erase_gate
        self.write_gate = write_gate
        if fused:
            output_size = d_memory * (3 + int(erase_gate) + int(write_gate))
            self.proj = nn.Linear(d_model, output_size, bias=False)
        else:
            self.e_proj = nn.Linear(d_model, d_memory, bias=False)
            self.k_proj = nn.Linear(d_model, d_memory, bias=False)
            self.v_proj = nn.Linear(d_model, d_memory, bias=False)
            if erase_gate:
                self.ge_proj = nn.Linear(d_model, d_memory, bias=False)
            if write_gate:
                self.gk_proj = nn.Linear(d_model, d_memory, bias=False)
        if erase_gate:
            self.erase_gate_bias = nn.Parameter(torch.zeros(d_memory))
        if write_gate:
            self.write_gate_bias = nn.Parameter(torch.zeros(d_memory))

    def forward(self, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.fused:
            projected = self.proj(y)
            offset = 0
            e_raw = projected[..., offset:offset + self.d_memory]
            offset += self.d_memory
            if self.erase_gate:
                ge_raw = projected[..., offset:offset + self.d_memory]
                offset += self.d_memory
            else:
                ge_raw = None
            k_raw = projected[..., offset:offset + self.d_memory]
            offset += self.d_memory
            if self.write_gate:
                gk_raw = projected[..., offset:offset + self.d_memory]
                offset += self.d_memory
            else:
                gk_raw = None
            v = projected[..., offset:offset + self.d_memory]
        else:
            e_raw = self.e_proj(y)
            k_raw = self.k_proj(y)
            v = self.v_proj(y)
            ge_raw = self.ge_proj(y) if self.erase_gate else None
            gk_raw = self.gk_proj(y) if self.write_gate else None
        e = _normalize(e_raw)
        k = _normalize(k_raw)
        if self.erase_gate:
            e = e * torch.sigmoid(ge_raw + self.erase_gate_bias).to(e.dtype)
        if self.write_gate:
            k = k * torch.sigmoid(gk_raw + self.write_gate_bias).to(k.dtype)
        return e, k, v


class OutputProjection(nn.Module):
    def __init__(self, d_model: int, d_memory: int):
        super().__init__()
        self.proj = nn.Linear(d_memory, d_model, bias=False)
        self.read_scale = nn.Parameter(torch.ones(()))

    def forward(self, reads: torch.Tensor) -> torch.Tensor:
        return self.read_scale * self.proj(reads)


class MemoryBranch(nn.Module):
    """Synthetic future branch: H/Y -> controls -> WY -> model-width output."""
    def __init__(self, d_model: int, d_memory: int, projection: str = "fused",
                 gate_mode: str = "full", q_source: str = "projection",
                 chunk_size: int = DEFAULT_CHUNK):
        super().__init__()
        if projection not in {"fused", "separate"}:
            raise ValueError("projection must be fused or separate")
        if gate_mode not in {"full", "no_q", "no_e", "no_k", "none"}:
            raise ValueError("unsupported gate mode")
        self.d_memory, self.chunk_size = d_memory, chunk_size
        self.read = ReadControls(d_model, d_memory, projection == "fused",
                                  gate_mode not in {"no_q", "none"}, q_source)
        self.write = WriteControls(d_model, d_memory, projection == "fused",
                                   gate_mode not in {"no_e", "none"},
                                   gate_mode not in {"no_k", "none"})
        self.output = OutputProjection(d_model, d_memory)

    def forward(self, post_attention: torch.Tensor, mlp_output: torch.Tensor,
                initial_state: torch.Tensor | None = None) -> torch.Tensor:
        q = self.read(post_attention)
        e, k, v = self.write(mlp_output)
        if initial_state is None:
            initial_state = post_attention.new_zeros(
                post_attention.size(0), self.d_memory, self.d_memory,
            )
        reads, _ = compact_wy_matrix_memory(
            q, e, k, v, initial_state, chunk_size=self.chunk_size,
        )
        return self.output(reads)

    def controls(self, h: torch.Tensor, y: torch.Tensor):
        return self.read(h), self.write(y)

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


def inputs(batch: int, length: int, d_model: int, d_memory: int,
           device: torch.device, requires_grad: bool) -> tuple[torch.Tensor, ...]:
    values = [torch.randn(batch, length, d_model, device=device, dtype=torch.float32)
              for _ in range(2)]
    state = torch.zeros(batch, d_memory, d_memory, device=device, dtype=torch.bfloat16)
    if requires_grad:
        for value in values:
            value.requires_grad_()
    return values[0], values[1], state


def run_branch(module: MemoryBranch, values: tuple[torch.Tensor, ...], backward: bool):
    h, y, state = values
    with autocast(h.device):
        # The module produces g_read * W_o R.  Add it to H here to include the
        # future layer's residual add in the end-to-end branch cost; Y is
        # already present in the host layer and is not recomputed here.
        output = h + module(h, y, state)
        if backward:
            output.float().square().mean().backward()
    return output


def run_component(fn, tensors: list[torch.Tensor], device: torch.device, backward: bool):
    with autocast(device):
        output = fn()
        if backward:
            if isinstance(output, tuple):
                output = torch.cat([item.reshape(-1) for item in output])
            output.float().square().mean().backward()
    return output


def timed(fn, reset, device: torch.device, warmup: int, iterations: int) -> dict:
    for _ in range(warmup):
        fn()
        sync(device)
        reset()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    samples = []
    for _ in range(iterations):
        started = time.perf_counter()
        fn()
        sync(device)
        samples.append(time.perf_counter() - started)
        reset()
    elapsed = sorted(samples)[len(samples) // 2]
    return {
        "seconds": elapsed,
        "milliseconds": elapsed * 1000,
        "tokens_per_second": None,
        "peak_vram_gib": (torch.cuda.max_memory_allocated(device) / 2**30
                           if device.type == "cuda" else None),
    }


def benchmark_module(module: nn.Module, fn, tensors: list[torch.Tensor],
                     device: torch.device, backward: bool, batch: int,
                     length: int, warmup: int, iterations: int) -> dict:
    result = timed(fn, lambda: reset_grads(module, tensors), device, warmup, iterations)
    result["tokens_per_second"] = batch * length / result["seconds"]
    result["backward"] = backward
    result["parameters"] = sum(p.numel() for p in module.parameters())
    result["saved_tensor_bytes"] = None
    if backward:
        saved = 0

        def pack(tensor):
            nonlocal saved
            saved += tensor.numel() * tensor.element_size()
            return tensor

        with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
            fn()
        sync(device)
        reset_grads(module, tensors)
        result["saved_tensor_bytes"] = saved
    return result


def make_branch_benchmark(batch, length, d_model, d_memory, projection, gate_mode,
                          q_source, device, backward, chunk_size, warmup, iterations):
    module = MemoryBranch(d_model, d_memory, projection, gate_mode, q_source, chunk_size).to(device)
    h, y, state = inputs(batch, length, d_model, d_memory, device, backward)
    fn = lambda: run_branch(module, (h, y, state), backward)
    result = benchmark_module(module, fn, [h, y, state], device, backward,
                              batch, length, warmup, iterations)
    result.update({"kind": "branch", "projection": projection, "gate_mode": gate_mode,
                   "q_source": q_source, "batch": batch, "sequence_length": length,
                   "d_model": d_model, "d_memory": d_memory, "chunk_size": chunk_size})
    del module, h, y, state
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def benchmark_components(batch, length, d_model, d_memory, device, chunk_size,
                         warmup, iterations) -> list[dict]:
    values = inputs(batch, length, d_model, d_memory, device, True)
    h, y, state = values
    module = MemoryBranch(d_model, d_memory, "fused", "full", "projection", chunk_size).to(device)
    with torch.no_grad(), autocast(device):
        q, e, k, v = module.read(h), *module.write(y)
        reads, _ = compact_wy_matrix_memory(q, e, k, v, state, chunk_size=chunk_size)
    results = []

    def record(name, component, fn, owned, output_shape):
        for backward in (False, True):
            tensors = [tensor.detach().clone().requires_grad_() for tensor in owned]
            def call():
                return run_component(fn(tensors), tensors, device, backward)
            item = benchmark_module(component if isinstance(component, nn.Module) else nn.Identity(),
                                    call, tensors, device, backward, batch, length,
                                    warmup, iterations)
            item.update({"kind": "component", "name": name, "backward": backward,
                         "batch": batch, "sequence_length": length, "d_model": d_model,
                         "d_memory": d_memory, "chunk_size": chunk_size,
                         "output_shape": list(output_shape)})
            results.append(item)

    read = module.read
    write = module.write
    output = module.output
    # Component modules retain their own parameters; use the same BF16-autocast
    # execution as the full branch, with H/Y or projected reads as leaves.
    for name, comp, owned, fn, shape in (
        ("fused_read_controls", read, [h], lambda ts: lambda: read(ts[0]), (batch, length, d_memory)),
        ("fused_write_controls", write, [y], lambda ts: lambda: write(ts[0]), (batch, length, d_memory * 3)),
        ("output_projection", output, [reads], lambda ts: lambda: output(ts[0]), (batch, length, d_model)),
    ):
        for backward in (False, True):
            owned_tensors = [tensor.detach().clone().requires_grad_() for tensor in owned]
            def call(comp=comp, owned_tensors=owned_tensors, backward=backward, fn=fn):
                return run_component(fn(owned_tensors), owned_tensors, device, backward)
            item = benchmark_module(comp, call, owned_tensors, device, backward,
                                    batch, length, warmup, iterations)
            item.update({"kind": "component", "name": name, "backward": backward,
                         "batch": batch, "sequence_length": length, "d_model": d_model,
                         "d_memory": d_memory, "chunk_size": chunk_size,
                         "output_shape": list(shape)})
            results.append(item)

    # Split the fused control path into projection GEMM and same-dtype
    # postprocessing so sigmoid/normalization cost is visible separately.
    with torch.no_grad(), autocast(device):
        read_projected = module.read.proj(h)
        write_projected = module.write.proj(y)

    def add_function_component(name, fn, owned, output_shape):
        for backward in (False, True):
            owned_tensors = [tensor.detach().clone().requires_grad_() for tensor in owned]
            def call(owned_tensors=owned_tensors, backward=backward):
                return run_component(lambda: fn(owned_tensors), owned_tensors, device, backward)
            item = benchmark_module(nn.Identity(), call, owned_tensors, device, backward,
                                    batch, length, warmup, iterations)
            item.update({"kind": "component", "name": name, "backward": backward,
                         "batch": batch, "sequence_length": length, "d_model": d_model,
                         "d_memory": d_memory, "chunk_size": chunk_size,
                         "output_shape": list(output_shape), "parameters": 0})
            results.append(item)

    def read_post(owned):
        projected = owned[0]
        q_raw, gate_raw = projected.split(d_memory, dim=-1)
        q = _normalize(q_raw)
        return q * torch.sigmoid(gate_raw + module.read.gate_bias).to(q.dtype)

    def write_post(owned):
        projected = owned[0]
        e_raw, ge_raw, k_raw, gk_raw, v = projected.split(d_memory, dim=-1)
        e = _normalize(e_raw) * torch.sigmoid(ge_raw + module.write.erase_gate_bias).to(projected.dtype)
        k = _normalize(k_raw) * torch.sigmoid(gk_raw + module.write.write_gate_bias).to(projected.dtype)
        return torch.cat((e, k, v), dim=-1)

    add_function_component("fused_read_gemm", lambda ts: module.read.proj(ts[0]),
                          [h], (batch, length, 2 * d_memory))
    add_function_component("fused_read_normalization_and_gate", read_post,
                          [read_projected], (batch, length, d_memory))
    add_function_component("fused_write_gemm", lambda ts: module.write.proj(ts[0]),
                          [y], (batch, length, 5 * d_memory))
    add_function_component("fused_write_normalization_and_gates", write_post,
                          [write_projected], (batch, length, 3 * d_memory))
    def recurrence_fn(ts):
        return lambda: compact_wy_matrix_memory(
            ts[0], ts[1], ts[2], ts[3], state, chunk_size=chunk_size,
        )
    for backward in (False, True):
        rec_tensors = [tensor.detach().clone().requires_grad_() for tensor in (q, e, k, v)]
        def call(rec_tensors=rec_tensors, backward=backward):
            return run_component(recurrence_fn(rec_tensors), rec_tensors, device, backward)
        item = timed(call, lambda: reset_grads(nn.Identity(), rec_tensors), device,
                     warmup, iterations)
        item["tokens_per_second"] = batch * length / item["seconds"]
        item.update({"kind": "component", "name": "compact_wy_recurrence",
                     "backward": backward, "batch": batch, "sequence_length": length,
                     "d_model": d_model, "d_memory": d_memory, "chunk_size": chunk_size,
                     "parameters": 0, "output_shape": [batch, length, d_memory]})
        results.append(item)
    del module, h, y, state, q, e, k, v, reads, read_projected, write_projected
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return results


def benchmark_block(config_path: str, batch: int, length: int, device: torch.device,
                    warmup: int, iterations: int, compile_model: bool) -> list[dict]:
    config = ExperimentConfig.load(config_path)
    model = build_model(config.model).to(device).train()
    input_ids = torch.randint(0, config.model.vocab_size, (batch, length), device=device)
    labels = torch.randint(0, config.model.vocab_size, (batch, length), device=device)
    if compile_model:
        model = torch.compile(model, mode="default")
    block = model.blocks[0] if not compile_model else None
    results = []

    def full_call():
        with autocast(device):
            _, loss = model(input_ids, labels)
            loss.backward()

    result = timed(full_call, lambda: reset_grads(model, []), device, warmup, iterations)
    result.update({"kind": "baseline_model", "name": "full_model_forward_backward",
                   "batch": batch, "sequence_length": length,
                   "d_model": config.model.d_model, "compiled": compile_model,
                   "parameters": model._orig_mod.num_parameters if compile_model else model.num_parameters})
    result["tokens_per_second"] = batch * length / result["seconds"]
    results.append(result)

    if not compile_model:
        x = torch.randn(batch, length, config.model.d_model, device=device, dtype=torch.float32, requires_grad=True)
        pos = model.position_encoding
        norm1, norm2, attn, mlp = block.norm1, block.norm2, block.attn, block.mlp
        with torch.no_grad(), autocast(device):
            attn_input = norm1(x)
            attn_out = attn(attn_input, None, pos)
            after_attn = x + attn_out
            mlp_input = norm2(after_attn)
            mlp_out = mlp(mlp_input)

        def bench_named(name, fn, module, tensors):
            for backward in (False, True):
                owned = [tensor.detach().clone().requires_grad_() for tensor in tensors]
                def call(owned=owned, backward=backward):
                    with autocast(device):
                        output = fn(owned)
                        if backward:
                            output.float().square().mean().backward()
                item = timed(call, lambda: reset_grads(module, owned), device, warmup, iterations)
                item.update({"kind": "baseline_component", "name": name, "backward": backward,
                             "batch": batch, "sequence_length": length,
                             "d_model": config.model.d_model, "compiled": False,
                             "parameters": sum(p.numel() for p in module.parameters())})
                item["tokens_per_second"] = batch * length / item["seconds"]
                results.append(item)

        bench_named("transformer_block", lambda ts: block(ts[0], None, pos), block, [x])
        bench_named("attention_path_norm1_plus_attention", lambda ts: block.attn(block.norm1(ts[0]), None, pos), block, [x])
        bench_named("mlp_path_norm2_plus_mlp", lambda ts: block.mlp(block.norm2(ts[0]),), block, [after_attn])
        bench_named("norm_residual_shell", lambda ts: (ts[0] + attn_out.detach()) + mlp_out.detach(), block, [x])
        del x, attn_input, attn_out, after_attn, mlp_input, mlp_out
    del model, input_ids, labels
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return results


def profile_branch(batch, length, d_model, d_memory, device, chunk_size, backward):
    module = MemoryBranch(d_model, d_memory, "fused", "full", "projection", chunk_size).to(device)
    h, y, state = inputs(batch, length, d_model, d_memory, device, backward)
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    sync(device)
    started = time.perf_counter()
    with torch.profiler.profile(activities=activities, record_shapes=False) as profiler:
        with autocast(device):
            with torch.profiler.record_function("memory_branch/full"):
                with torch.profiler.record_function("memory_branch/read_control_projection"):
                    q = module.read(h)
                with torch.profiler.record_function("memory_branch/write_control_projection"):
                    e, k, v = module.write(y)
                with torch.profiler.record_function("memory_branch/recurrence"):
                    reads, _ = compact_wy_matrix_memory(q, e, k, v, state, chunk_size=chunk_size, profile=True)
                with torch.profiler.record_function("memory_branch/output_projection"):
                    output = module.output(reads)
                with torch.profiler.record_function("memory_branch/residual_scalar_and_add"):
                    output = h + output
                if backward:
                    with torch.profiler.record_function("memory_branch/backward"):
                        output.float().square().mean().backward()
        sync(device)
    wall = time.perf_counter() - started
    rows = []
    for event in profiler.key_averages():
        if event.key.startswith("memory_branch/") or event.key.startswith("matrix_memory/"):
            rows.append({"name": event.key, "calls": event.count,
                         "self_cpu_us": event.self_cpu_time_total,
                         "total_cpu_us": event.cpu_time_total,
                         "self_device_us": getattr(event, "self_device_time_total", 0.0),
                         "total_device_us": getattr(event, "device_time_total", 0.0)})
    result = {"kind": "profile", "batch": batch, "sequence_length": length,
              "d_model": d_model, "d_memory": d_memory, "chunk_size": chunk_size,
              "backward": backward, "wall_seconds": wall, "events": rows}
    del module, h, y, state
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def parameter_table(d_model: int, baseline_params: int) -> list[dict]:
    rows = []
    for d_memory in (32, 64, 128, 256):
        per_layer = 8 * d_model * d_memory + 3 * d_memory + 1
        total = 7 * per_layer
        rows.append({"d_memory": d_memory, "per_layer": per_layer, "seven_layers": total,
                     "baseline_parameters": baseline_params,
                     "increase_percent": 100 * total / baseline_params})
    return rows


def flops_table(d_model: int, d_memory: int, sequence_length: int, chunk_size: int) -> dict:
    # Multiply-add = 2 FLOPs.  The WY estimate includes the transient binary
    # factor GEMMs and is intentionally reported as supporting analysis only.
    read = 4 * d_model * d_memory
    write = 10 * d_model * d_memory
    output = 2 * d_model * d_memory
    length = 2 * chunk_size
    stages = max(1, (length - 1).bit_length())
    factor = 2 * stages * length**3 / chunk_size
    recurrence = (4 * d_memory**2 + 24 * chunk_size * d_memory + factor)
    baseline = 8 * d_model**2 + 4 * sequence_length * d_model + 6 * d_model * 768
    return {"read_projection": read, "write_projection": write, "output_projection": output,
            "recurrence": recurrence, "branch_total": read + write + output + recurrence,
            "baseline_transformer_estimate": baseline}


def compile_branch(batch, length, d_model, d_memory, device, chunk_size, warmup, iterations):
    module = MemoryBranch(d_model, d_memory, "fused", "full", "projection", chunk_size).to(device)
    h, y, state = inputs(batch, length, d_model, d_memory, device, False)
    _dynamo.reset()
    from torch._dynamo.utils import counters
    counters.clear()
    fn = torch.compile(lambda: run_branch(module, (h, y, state), False), mode="default")
    started = time.perf_counter()
    fn()
    sync(device)
    compile_seconds = time.perf_counter() - started
    result = timed(lambda: fn(), lambda: None, device, warmup, iterations)
    result.update({"kind": "compiled_branch", "batch": batch, "sequence_length": length,
                   "d_model": d_model, "d_memory": d_memory, "chunk_size": chunk_size,
                   "compiled": True, "compile_seconds": compile_seconds,
                   "graph_breaks": sum(counters["graph_break"].values()),
                   "backward": False})
    result["tokens_per_second"] = batch * length / result["seconds"]
    del fn, module, h, y, state
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def cost_model(results: list[dict], batch: int, length: int,
               baseline_parameters: int) -> dict:
    def first(predicate):
        return next((item for item in results if predicate(item)), None)

    baseline_eager = first(lambda x: x.get("kind") == "baseline_model" and not x.get("compiled"))
    baseline_compiled = first(lambda x: x.get("kind") == "baseline_model" and x.get("compiled"))
    branch_forward = first(lambda x: x.get("kind") == "branch" and x.get("projection") == "fused"
                           and x.get("gate_mode") == "full" and x.get("q_source") == "projection"
                           and not x.get("backward") and x.get("d_memory") == 64
                           and x.get("sequence_length") == length)
    branch_training = first(lambda x: x.get("kind") == "branch" and x.get("projection") == "fused"
                            and x.get("gate_mode") == "full" and x.get("q_source") == "projection"
                            and x.get("backward") and x.get("d_memory") == 64
                           and x.get("sequence_length") == length)
    block_training = first(lambda x: x.get("kind") == "baseline_component"
                           and x.get("name") == "transformer_block" and x.get("backward")
                           and x.get("sequence_length") == length)
    if baseline_compiled is None or branch_training is None:
        return {"status": "incomplete"}
    base = baseline_compiled["seconds"]
    branch = branch_training["seconds"]
    estimates = {}
    for name, multiplier in (("optimistic", 0.8), ("base", 1.0), ("pessimistic", 1.2)):
        predicted = base + 7 * branch * multiplier
        estimates[name] = {
            "branch_multiplier": multiplier,
            "predicted_step_seconds": predicted,
            "predicted_tokens_per_second": batch * length / predicted,
            "slowdown_percent": 100 * (predicted / base - 1),
        }
    target_limits = {}
    for slowdown in (0.30, 0.20, 0.10):
        allowed_per_branch = slowdown * base / 7
        target_limits[f"{int(slowdown * 100)}_percent_slowdown"] = {
            "allowed_ms_per_branch": allowed_per_branch * 1000,
            "required_speedup_vs_measured": branch / allowed_per_branch,
        }
    component_times = {}
    for name in ("fused_read_controls", "fused_write_controls", "compact_wy_recurrence", "output_projection"):
        item = first(lambda x, name=name: x.get("kind") == "component" and x.get("name") == name
                     and x.get("backward") and x.get("d_memory") == 64 and x.get("sequence_length") == length)
        if item is not None:
            component_times[name] = item["seconds"]
    component_total = sum(component_times.values())
    for name, seconds in component_times.items():
        component_times[name] = {"milliseconds": seconds * 1000,
                                 "percent_of_component_sum": 100 * seconds / component_total}
    stage_names = (
        "fused_read_gemm", "fused_read_normalization_and_gate",
        "fused_write_gemm", "fused_write_normalization_and_gates",
        "compact_wy_recurrence", "output_projection",
    )
    stages = {}
    for name in stage_names:
        item = first(lambda x, name=name: x.get("kind") == "component" and x.get("name") == name
                     and x.get("backward") and x.get("d_memory") == 64
                     and x.get("sequence_length") == length)
        if item is not None:
            stages[name] = item["seconds"]
    stage_sum = sum(stages.values())
    stage_breakdown = {name: {"milliseconds": seconds * 1000,
                              "percent_of_isolated_stage_sum": 100 * seconds / stage_sum}
                       for name, seconds in stages.items()}
    return {
        "baseline_eager_step_ms": baseline_eager["milliseconds"] if baseline_eager else None,
        "baseline_compiled_step_ms": baseline_compiled["milliseconds"],
        "baseline_compiled_tokens_per_second": baseline_compiled["tokens_per_second"],
        "historical_tokens_per_second": 466_000,
        "branch_forward_ms": branch_forward["milliseconds"] if branch_forward else None,
        "branch_forward_tokens_per_second": branch_forward["tokens_per_second"] if branch_forward else None,
        "branch_training_ms": branch_training["milliseconds"],
        "branch_training_tokens_per_second": branch_training["tokens_per_second"],
        "branch_saved_tensor_bytes": branch_training.get("saved_tensor_bytes"),
        "seven_layer_saved_tensor_bytes": (
            branch_training.get("saved_tensor_bytes") * 7
            if branch_training.get("saved_tensor_bytes") is not None else None
        ),
        "measured_transformer_block_training_ms": block_training["milliseconds"] if block_training else None,
        "branch_fraction_of_measured_transformer_block": (
            branch / block_training["seconds"] if block_training else None
        ),
        "branch_fraction_of_equal_share_full_model": branch / (base / 7),
        "seven_layer_estimates": estimates,
        "target_limits": target_limits,
        "component_times": component_times,
        "isolated_stage_breakdown": stage_breakdown,
        "parameter_table": parameter_table(256, baseline_parameters),
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    parser.add_argument("--config", default="configs/standard_transformer_plus.json")
    parser.add_argument("--output", default="benchmark_results/memory_integration_cost.json")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--length", type=int, default=2048)
    parser.add_argument("--d-model", type=int, default=DEFAULT_D_MODEL)
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--skip-profile", action="store_true")
    parser.add_argument("--skip-compile", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA/ROCm device requested but unavailable")
    config = ExperimentConfig.load(args.config)
    baseline_params = 10_161_920
    results = []
    print(f"device={device} torch={torch.__version__} gpu="
          f"{torch.cuda.get_device_name(device) if device.type == 'cuda' else None}")

    if not args.skip_baseline:
        print("benchmarking eager full baseline model")
        results.extend(benchmark_block(args.config, args.batch, args.length, device,
                                       args.warmup, args.iterations, False))
        print("benchmarking compiled full baseline model")
        try:
            results.extend(benchmark_block(args.config, args.batch, args.length, device,
                                           args.warmup, args.iterations, True))
        except Exception as exc:
            results.append({"kind": "baseline_model", "name": "compiled_baseline",
                            "status": "failed", "error": f"{type(exc).__name__}: {exc}"})

    print("benchmarking canonical branch variants")
    for projection in ("separate", "fused"):
        for backward in (False, True):
            result = make_branch_benchmark(
                args.batch, args.length, args.d_model, 64, projection, "full", "projection",
                device, backward, args.chunk_size, args.warmup, args.iterations,
            )
            results.append(result)
            print(json.dumps(result, separators=(",", ":")), flush=True)

    print("benchmarking control-gate ablations and q reuse")
    for gate_mode in ("no_q", "no_e", "no_k", "none"):
        for backward in (False, True):
            result = make_branch_benchmark(
                args.batch, args.length, args.d_model, 64, "fused", gate_mode, "projection",
                device, backward, args.chunk_size, args.warmup, args.iterations,
            )
            results.append(result)
    for backward in (False, True):
        results.append(make_branch_benchmark(
            args.batch, args.length, args.d_model, 64, "fused", "full", "slice",
            device, backward, args.chunk_size, args.warmup, args.iterations,
        ))

    print("benchmarking standalone branch components")
    results.extend(benchmark_components(args.batch, args.length, args.d_model, 64,
                                        device, args.chunk_size, args.warmup, args.iterations))

    if not args.skip_profile:
        print("profiling canonical full branch")
        results.append(profile_branch(args.batch, args.length, args.d_model, 64,
                                      device, args.chunk_size, True))

    if not args.skip_compile:
        print("benchmarking compiled canonical branch forward")
        try:
            results.append(compile_branch(args.batch, args.length, args.d_model, 64,
                                          device, args.chunk_size, args.warmup, args.iterations))
        except Exception as exc:
            results.append({"kind": "compiled_branch", "status": "failed",
                            "error": f"{type(exc).__name__}: {exc}"})

    print("benchmarking d_memory and sequence sweeps")
    for d_memory in (32, 64, 128, 256):
        results.append(make_branch_benchmark(
            args.batch, args.length, args.d_model, d_memory, "fused", "full", "projection",
            device, True, args.chunk_size, args.warmup, args.iterations,
        ))
    for length in (512, 2048, 4096, 8192):
        results.append(make_branch_benchmark(
            args.batch, length, args.d_model, 64, "fused", "full", "projection",
            device, True, args.chunk_size, args.warmup, args.iterations,
        ))

    payload = {
        "config": {"batch": args.batch, "sequence_length": args.length,
                   "d_model": args.d_model, "chunk_size": args.chunk_size,
                   "warmup": args.warmup, "iterations": args.iterations,
                   "precision": "bf16 autocast over fp32 parameters",
                   "optimizer_step_included": False},
        "environment": {
            "torch": torch.__version__, "platform": platform.platform(),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "rocm": getattr(torch.version, "hip", None),
            "triton": triton_version(),
        },
        "baseline_model_parameters": baseline_params,
        "parameter_table": parameter_table(args.d_model, baseline_params),
        "flops_canonical": flops_table(args.d_model, 64, args.length, args.chunk_size),
        "cost_model": cost_model(results, args.batch, args.length, baseline_params),
        "results": results,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"saved={output}")


if __name__ == "__main__":
    main()
