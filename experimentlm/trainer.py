"""Generic experiment trainer."""
from __future__ import annotations

import json
import math
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from .checkpoint import load_checkpoint, save_checkpoint
from .config import ExperimentConfig
from .data import TokenStream
from .evaluation import evaluate_loss
from .model import build_model
from .optim import build_optimizer, build_schedule
from .resolution import resolve_experiment


class Tee:
    def __init__(self, *streams): self.streams = streams
    def write(self, text):
        for stream in self.streams: stream.write(text); stream.flush()
    def flush(self):
        for stream in self.streams: stream.flush()


def _seed_all(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def train(config: ExperimentConfig, device_name: str = "cuda", resume: str | None = None) -> Path:
    config = resolve_experiment(config)
    output = Path("checkpoints") / config.name
    output.mkdir(parents=True, exist_ok=True)
    config.save(output / "config.json")
    log_dir = Path("logs") / config.name
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"run-{datetime.now().strftime('%Y%m%d-%H%M%S-%f')}.log"
    log_file = log_path.open("w", encoding="utf-8", buffering=1)
    original_stdout = sys.stdout
    sys.stdout = Tee(original_stdout, log_file)
    try:
        training, model_config = config.training, config.model
        _seed_all(training.seed)
        device = torch.device(device_name if device_name != "cuda" or torch.cuda.is_available() else "cpu")
        accumulation = max(1, math.ceil(
            training.tokens_per_step / (training.micro_batch_size * model_config.context_length)
        ))
        effective_tokens = training.micro_batch_size * accumulation * model_config.context_length
        train_data = TokenStream(
            config.data.train_bin, model_config.context_length, training.micro_batch_size, training.seed
        )
        micro_batches = len(train_data) // training.micro_batch_size
        steps_per_epoch = micro_batches // accumulation
        total_steps = steps_per_epoch * training.epochs
        if training.max_steps is not None: total_steps = min(total_steps, training.max_steps)
        if training.max_tokens is not None:
            total_steps = min(total_steps, math.ceil(training.max_tokens / effective_tokens))
        if total_steps < 1: raise ValueError("configuration produces zero optimizer steps")

        model = build_model(model_config).to(device)
        optimizer = build_optimizer(training.optimizer, model.parameters())
        scheduler = build_schedule(training.schedule, optimizer, total_steps)
        start, tokens_seen = 0, 0
        if resume:
            model, blob = load_checkpoint(
                resume, device=device, model=model, optimizer=optimizer, scheduler=scheduler
            )
            start, tokens_seen = blob["step"], blob["tokens_seen"]
        if training.compile:
            print("compiling_model=true", flush=True)
            model = torch.compile(model)

        print(f"log_file={log_path}")
        print(
            f"device={device} parameters={model.num_parameters:,} steps={total_steps:,} "
            f"accumulation={accumulation} effective_tokens={effective_tokens:,} "
            f"optimizer={training.optimizer.class_name} schedule={training.schedule.class_name} "
            f"precision={training.precision}"
        )
        print(f"model_config={json.dumps(model_config.to_dict(), separators=(',', ':'))}")
        print(f"training_config={json.dumps(training.to_dict(), separators=(',', ':'))}")
        train_iter = iter(train_data)
        skipped_micro_batches = (start % steps_per_epoch) * accumulation
        for _ in range(skipped_micro_batches): next(train_iter)
        progress = tqdm(
            range(start, total_steps), initial=start, total=total_steps, desc="Training", unit="step",
            dynamic_ncols=True, file=original_stdout,
        )
        for step in progress:
            if step > start and step % steps_per_epoch == 0:
                train_iter = iter(train_data)
            optimizer.zero_grad(set_to_none=True)
            loss_sum, started = 0.0, time.perf_counter()
            for _ in range(accumulation):
                inputs, labels = next(train_iter)
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                    enabled=training.precision == "bf16"):
                    _, loss = model(inputs.to(device), labels.to(device))
                    (loss / accumulation).backward()
                loss_sum += float(loss.detach())
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), training.grad_clip)
            optimizer.step(); scheduler.step(); tokens_seen += effective_tokens
            if (step + 1) % training.log_interval_steps == 0 or step == start:
                throughput = effective_tokens / max(time.perf_counter() - started, 1e-9)
                peak_vram = torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else 0.0
                print(
                    f"step={step+1}/{total_steps} loss={loss_sum/accumulation:.4f} "
                    f"lr={optimizer.param_groups[0]['lr']:.3g} tokens_seen={tokens_seen:,} "
                    f"tok/s={throughput:,.0f} peak_vram={peak_vram:.2f}GiB grad_norm={float(grad_norm):.2f}", flush=True,
                )
                progress.set_postfix(loss=f"{loss_sum/accumulation:.4f}",
                                     lr=f"{optimizer.param_groups[0]['lr']:.2g}", tok_s=f"{throughput:,.0f}")
            if (step + 1) % training.validation_interval_steps == 0:
                limit = config.evaluation.validation_batches or None
                loss, perplexity, tokens = evaluate_loss(
                    model, TokenStream(config.data.eval_bin, model_config.context_length,
                                       training.micro_batch_size, training.seed + 2),
                    device, limit, training.precision,
                )
                peak_vram = torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else 0.0
                print(f"validation step={step+1} loss={loss:.4f} perplexity={perplexity:.3f} tokens={tokens:,} peak_vram={peak_vram:.2f}GiB", flush=True)
            if (step + 1) % training.checkpoint_interval_steps == 0:
                save_checkpoint(output / f"step-{step+1}", model, optimizer, scheduler,
                                step + 1, tokens_seen, config)

        limit = config.evaluation.validation_batches or None
        loss, perplexity, tokens = evaluate_loss(
            model, TokenStream(config.data.eval_bin, model_config.context_length,
                               training.micro_batch_size, training.seed + 2),
            device, limit, training.precision,
        )
        peak_vram = torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else 0.0
        print(f"validation final step={total_steps} loss={loss:.4f} perplexity={perplexity:.3f} tokens={tokens:,} peak_vram={peak_vram:.2f}GiB", flush=True)
        save_checkpoint(output / "final", model, optimizer, scheduler, total_steps, tokens_seen, config)
        results = {"step": total_steps, "tokens_seen": tokens_seen, "validation_loss": loss,
                   "validation_perplexity": perplexity, "validation_tokens": tokens,
                   "parameters": model.num_parameters, "effective_tokens_per_update": effective_tokens}
        (output / "results.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
        return output / "final"
    finally:
        sys.stdout = original_stdout
        log_file.close()
