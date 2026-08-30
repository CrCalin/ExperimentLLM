"""Opt-in static-shape CUDA-graph training runner.

The normal trainer is intentionally untouched. This runner captures one fixed
microbatch forward/backward graph and replays it for gradient accumulation;
optimizer, scheduler, validation, and checkpoint work remain outside it.
"""
from __future__ import annotations

import math
import time
from pathlib import Path

import torch

from .checkpoint import save_checkpoint
from .config import ExperimentConfig
from .data import TokenStream
from .evaluation import evaluate_loss
from .model import build_model
from .optim import build_optimizer, build_schedule
from .resolution import resolve_experiment


def _step_count(config: ExperimentConfig) -> tuple[int, int, int]:
    training, model = config.training, config.model
    accumulation = max(1, math.ceil(
        training.tokens_per_step / (training.micro_batch_size * model.context_length)
    ))
    effective_tokens = training.micro_batch_size * accumulation * model.context_length
    micro_batches = len(TokenStream(
        config.data.train_bin, model.context_length, training.micro_batch_size, training.seed
    )) // training.micro_batch_size
    steps = micro_batches // accumulation
    if training.max_steps is not None:
        steps = min(steps, training.max_steps)
    if training.max_tokens is not None:
        steps = min(steps, math.ceil(training.max_tokens / effective_tokens))
    if steps < 1:
        raise ValueError("configuration produces zero optimizer steps")
    return steps, accumulation, effective_tokens


def _copy_batch(inputs: torch.Tensor, labels: torch.Tensor, batch) -> None:
    source_inputs, source_labels = batch
    inputs.copy_(source_inputs, non_blocking=True)
    labels.copy_(source_labels, non_blocking=True)


def _capture_microbatch(model, inputs, labels, accumulation: int) -> torch.cuda.CUDAGraph:
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        for _ in range(2):
            model.zero_grad(set_to_none=False)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                _, warm_loss = model(inputs, labels)
            warm_loss.backward()
            del warm_loss
        model.zero_grad(set_to_none=False)
    torch.cuda.synchronize()
    torch.autograd.graph.set_override_stale_capture_stream(True)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            _, captured_loss = model(inputs, labels)
        (captured_loss / accumulation).backward()
    del captured_loss
    torch.cuda.synchronize()
    model.zero_grad(set_to_none=False)
    return graph


def train_cudagraphs(config: ExperimentConfig, device_name: str = "cuda") -> Path:
    config = resolve_experiment(config)
    if device_name != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("the CUDA-graph trainer requires a CUDA/HIP device")
    device = torch.device(device_name)
    total_steps, accumulation, effective_tokens = _step_count(config)
    model = torch.compile(build_model(config.model).to(device), mode="default")
    optimizer = build_optimizer(config.training.optimizer, model.named_parameters())
    scheduler = build_schedule(config.training.schedule, optimizer, total_steps)
    inputs = torch.empty(config.training.micro_batch_size, config.model.context_length,
                         device=device, dtype=torch.long)
    labels = torch.empty_like(inputs)
    inputs.zero_()
    labels.zero_()
    graph = _capture_microbatch(model, inputs, labels, accumulation)
    train_iter = iter(TokenStream(
        config.data.train_bin, config.model.context_length,
        config.training.micro_batch_size, config.training.seed,
    ))
    output = Path("checkpoints") / config.name
    output.mkdir(parents=True, exist_ok=True)
    config.save(output / "config.json")

    for step in range(total_steps):
        optimizer.zero_grad(set_to_none=False)
        started = time.perf_counter()
        for _ in range(accumulation):
            _copy_batch(inputs, labels, next(train_iter))
            graph.replay()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.grad_clip)
        optimizer.step()
        scheduler.step()
        torch.cuda.synchronize()
        if step > 0:
            print(
                f"step={step + 1}/{total_steps} "
                f"tok/s={effective_tokens / max(time.perf_counter() - started, 1e-9):,.0f}",
                flush=True,
            )
        if (step + 1) % config.training.validation_interval_steps == 0:
            limit = config.evaluation.validation_batches or None
            validation = evaluate_loss(
                model, TokenStream(config.data.eval_bin, config.model.context_length,
                                   config.training.micro_batch_size, config.training.seed + 2),
                device, limit, config.training.precision,
            )
            print(f"validation step={step + 1} loss={validation[0]:.4f} perplexity={validation[1]:.3f}")
        if (step + 1) % config.training.checkpoint_interval_steps == 0:
            save_checkpoint(output / f"step-{step + 1}", model, optimizer, scheduler,
                            step + 1, (step + 1) * effective_tokens, config)

    limit = config.evaluation.validation_batches or None
    validation = evaluate_loss(
        model, TokenStream(config.data.eval_bin, config.model.context_length,
                           config.training.micro_batch_size, config.training.seed + 2),
        device, limit, config.training.precision,
    )
    save_checkpoint(output / "final", model, optimizer, scheduler,
                    total_steps, total_steps * effective_tokens, config)
    print(f"validation loss={validation[0]:.4f} perplexity={validation[1]:.3f}")
    return output / "final"
