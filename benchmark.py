#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import lm_eval


DEFAULT_TASKS = [
    "hellaswag",
    "piqa",
    "arc_easy",
    "boolq",
    "lambada_openai",
]


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
        default="benchmark_results.json",
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

    return parser.parse_args()


def main():
    args = parse_args()

    tasks = DEFAULT_TASKS if "all" in args.tasks else args.tasks

    output_path = Path(args.output)

    all_results = {}

    print(f"Model: {args.model}")
    print(f"Device: {args.device}")
    print(f"Tasks: {', '.join(tasks)}")
    print()

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
        )

        task_result = result["results"][task]
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
