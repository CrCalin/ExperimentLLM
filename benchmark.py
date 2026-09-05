#!/usr/bin/env python3

import argparse
import contextlib
import json
import math
import time
from pathlib import Path

import lm_eval
import torch
from lm_eval.models.huggingface import HFLM
from tokenizers import Tokenizer
from tqdm import tqdm
from transformers import AutoModelForCausalLM
from experimentlm.data import TokenStream


DEFAULT_TASKS = [
    "hellaswag",
    "piqa",
    "arc_easy",
    "lambada_openai",
]


def format_duration(seconds):
    """Format elapsed wall-clock seconds for human-readable output."""
    return f"{seconds:.2f}s"


def normalize_lambada_metrics(result, task_result, model_path):
    """Add token-normalized PPL while retaining official word-level PPL."""
    samples = result.get("samples", {}).get("lambada_openai", [])
    if not samples:
        raise RuntimeError("LAMBADA sample logging is required for token perplexity.")
    tokenizer = Tokenizer.from_file(str(model_path / "tokenizer.json"))
    total_logprob = 0.0
    total_tokens = 0
    for sample in samples:
        context, continuation = sample["arguments"][0]
        context_ids = tokenizer.encode(context, add_special_tokens=False).ids
        whole_ids = tokenizer.encode(context + continuation, add_special_tokens=False).ids
        continuation_tokens = len(whole_ids) - len(context_ids)
        if continuation_tokens <= 0:
            raise RuntimeError("LAMBADA continuation produced no tokens.")
        total_logprob += float(sample["filtered_resps"][0][0])
        total_tokens += continuation_tokens

    task_result["word_perplexity,none"] = task_result.pop("perplexity,none")
    task_result["word_perplexity_stderr,none"] = task_result.pop(
        "perplexity_stderr,none"
    )
    task_result["token_perplexity,none"] = math.exp(-total_logprob / total_tokens)
    task_result["mean_target_tokens,none"] = total_tokens / len(samples)


@torch.inference_mode()
def local_validation(
    model_path, validation_bin, device, batch_size, max_batches, model=None,
    precision="bfloat16",
):
    owns_model = model is None
    if owns_model:
        model = AutoModelForCausalLM.from_pretrained(
            str(model_path), trust_remote_code=True
        ).to(device).eval()
    else:
        model = model.to(device).eval()
    config = model.config
    model_config = getattr(config, "experiment_model_config", {}) or {}
    context_length = model_config.get("context_length", getattr(config, "context_length", 2048))
    stream = TokenStream(validation_bin, context_length, batch_size, seed=12347)
    total_loss = 0.0
    total_tokens = 0
    total_batches = len(stream.batch_starts)
    if max_batches is not None:
        total_batches = min(total_batches, max_batches)
    with tqdm(total=total_batches, desc="Validation", unit="batch") as progress:
        for index, (inputs, labels) in enumerate(stream):
            if max_batches is not None and index >= max_batches:
                break
            inputs, labels = inputs.to(device), labels.to(device)
            dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "none": None}[precision]
            autocast = (torch.autocast(device_type=device.type, dtype=dtype)
                        if dtype is not None else contextlib.nullcontext())
            with autocast:
                output = model(input_ids=inputs, labels=labels)
            tokens = labels.numel()
            total_loss += float(output.loss) * tokens
            total_tokens += tokens
            progress.update()
    if owns_model:
        del model
        if device.type == "cuda": torch.cuda.empty_cache()
    loss = total_loss / max(total_tokens, 1)
    return loss, math.exp(min(loss, 20.0)), total_tokens


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        required=True,
        help="Path to Hugging Face-compatible checkpoint",
    )

    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=DEFAULT_TASKS + ["boolq", "all"],
        default=["all"],
        help="Benchmarks to run",
    )

    parser.add_argument(
        "--batch-size",
        default=64,
        help='Batch size, e.g. "auto", 8, 16, 32',
    )

    parser.add_argument(
        "--mixed-precision",
        choices=["bfloat16", "float16", "none"],
        default="bfloat16",
        help="Autocast dtype for lm-eval model calls.",
    )

    parser.add_argument(
        "--device",
        default="cuda:0",
    )

    parser.add_argument("--trust-remote-code", action="store_true")

    parser.add_argument(
        "--output",
        default=None,
        help="Results path; defaults to benchmark_results.json inside --model.",
    )

    parser.add_argument(
        "--limit",
        type=float,
        default=None,
        help=(
            "Optional debug limit. "
            "Examples: 100 = first 100 samples, "
            "0.1 = 10%% of each task."
        ),
    )

    parser.add_argument("--validation-bin", default="eval.bin")
    parser.add_argument("--validation-batch-size", type=int, default=8)
    parser.add_argument(
        "--validation-batches", type=int, default=0,
        help="Local validation batches; 0 evaluates the complete validation stream.",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    total_start = time.perf_counter()

    tasks = DEFAULT_TASKS if "all" in args.tasks else args.tasks

    model_path = Path(args.model)
    output_path = Path(args.output) if args.output else model_path / "benchmark_results.json"

    all_results = {}

    print(f"Model: {args.model}")
    print(f"Device: {args.device}")
    print(f"Tasks: {', '.join(tasks)}")
    print()

    device = torch.device(args.device)
    mixed_precision_dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "none": None,
    }[args.mixed_precision]
    benchmark_model = HFLM(
        pretrained=args.model,
        trust_remote_code=args.trust_remote_code,
        batch_size=args.batch_size,
        device=args.device,
        mixed_precision_dtype=mixed_precision_dtype,
    )

    validation_limit = None if args.validation_batches == 0 else args.validation_batches
    validation_start = time.perf_counter()
    validation_loss, validation_ppl, validation_tokens = local_validation(
        model_path,
        args.validation_bin,
        device,
        args.validation_batch_size,
        validation_limit,
        model=benchmark_model.model,
        precision=args.mixed_precision,
    )
    validation_seconds = time.perf_counter() - validation_start
    validation_result = {
        "loss": validation_loss,
        "perplexity": validation_ppl,
        "tokens": validation_tokens,
    }
    all_results["validation"] = validation_result
    all_results["timing"] = {
        "validation_seconds": validation_seconds,
        "benchmark_seconds": {},
    }
    print("Local validation:")
    print(f"  loss                       {validation_loss:.6f}")
    print(f"  perplexity                 {validation_ppl:.6f}")
    print(f"  tokens                     {validation_tokens:,}")
    print(f"  time                       {format_duration(validation_seconds)}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f: json.dump(all_results, f, indent=2, default=str)

    task_seconds = {}
    for i, task in enumerate(tasks, 1):
        print("=" * 70)
        print(f"[{i}/{len(tasks)}] {task}")
        print("=" * 70)

        task_start = time.perf_counter()
        result = lm_eval.simple_evaluate(
            model=benchmark_model,
            tasks=[task],
            num_fewshot=0,
            batch_size=args.batch_size,
            device=args.device,
            limit=args.limit,
            log_samples=task == "lambada_openai",
        )
        task_result = result["results"][task]
        if task == "lambada_openai":
            normalize_lambada_metrics(result, task_result, model_path)
        task_seconds[task] = time.perf_counter() - task_start
        all_results[task] = task_result
        all_results["timing"]["benchmark_seconds"] = task_seconds

        print()
        print(f"{task} results:")

        for key, value in task_result.items():
            if isinstance(value, (int, float)):
                print(f"  {key:25} {value:.6f}")

        # Save incrementally in case a later benchmark fails.
        with output_path.open("w") as f:
            json.dump(
                all_results,
                f,
                indent=2,
                default=str,
            )

        print()

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    for task, metrics in all_results.items():
        if task == "timing":
            continue
        print(f"\n{task}")

        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                print(f"  {key:25} {value:.6f}")

    total_seconds = time.perf_counter() - total_start
    all_results["timing"]["total_seconds"] = total_seconds
    with output_path.open("w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print("\nTiming")
    print(f"  validation                 {format_duration(validation_seconds)}")
    for task in tasks:
        print(f"  {task:25} {format_duration(task_seconds[task])}")
    print(f"  total                      {format_duration(total_seconds)}")
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
