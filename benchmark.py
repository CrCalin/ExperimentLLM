#!/usr/bin/env python3

import argparse
import json
import math
from pathlib import Path

import lm_eval
import torch
from tokenizers import Tokenizer
from tqdm import tqdm
from transformers import AutoModelForCausalLM
from experimentlm.data import TokenStream


DEFAULT_TASKS = [
    "hellaswag",
    "piqa",
    "arc_easy",
    "boolq",
    "lambada_openai",
]


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
def local_validation(model_path, validation_bin, device, batch_size, max_batches):
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path), trust_remote_code=True
    ).to(device).eval()
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
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                output = model(input_ids=inputs, labels=labels)
            tokens = labels.numel()
            total_loss += float(output.loss) * tokens
            total_tokens += tokens
            progress.update()
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
        choices=DEFAULT_TASKS + ["all"],
        default=["all"],
        help="Benchmarks to run",
    )

    parser.add_argument(
        "--batch-size",
        default="auto",
        help='Batch size, e.g. "auto", 8, 16, 32',
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

    tasks = DEFAULT_TASKS if "all" in args.tasks else args.tasks

    model_path = Path(args.model)
    output_path = Path(args.output) if args.output else model_path / "benchmark_results.json"

    all_results = {}

    print(f"Model: {args.model}")
    print(f"Device: {args.device}")
    print(f"Tasks: {', '.join(tasks)}")
    print()

    device = torch.device(args.device)
    validation_limit = None if args.validation_batches == 0 else args.validation_batches
    validation_loss, validation_ppl, validation_tokens = local_validation(
        model_path, args.validation_bin, device, args.validation_batch_size, validation_limit
    )
    validation_result = {
        "loss": validation_loss,
        "perplexity": validation_ppl,
        "tokens": validation_tokens,
    }
    all_results["validation"] = validation_result
    print("Local validation:")
    print(f"  loss                       {validation_loss:.6f}")
    print(f"  perplexity                 {validation_ppl:.6f}")
    print(f"  tokens                     {validation_tokens:,}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f: json.dump(all_results, f, indent=2, default=str)

    for i, task in enumerate(tasks, 1):
        print("=" * 70)
        print(f"[{i}/{len(tasks)}] {task}")
        print("=" * 70)

        result = lm_eval.simple_evaluate(
            model="hf",
            model_args={
                "pretrained": args.model,
                "trust_remote_code": args.trust_remote_code,
            },
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
        all_results[task] = task_result

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
        print(f"\n{task}")

        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                print(f"  {key:25} {value:.6f}")

    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
