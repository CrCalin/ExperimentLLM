#!/usr/bin/env python3
"""Run sequential training experiments over a list of peak learning rates."""
import argparse
import subprocess
import sys
from pathlib import Path

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lrs", nargs="+", type=float, default=[3e-4, 6e-4, 1e-3, 2e-3],
                   help="Peak learning rates to test, in sequential order.")
    p.add_argument("--output-root", default="sweeps/lr",
                   help="Root directory for per-LR checkpoints and logs.")
    p.add_argument("--train-bin", default="train.bin")
    p.add_argument("--eval-bin", default="eval.bin")
    p.add_argument("--device", default="cuda")
    p.add_argument("--micro-batch-size", type=int, default=2)
    p.add_argument("--effective-tokens", type=int, default=131072)
    p.add_argument("--warmup-fraction", type=float, default=.015)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--eval-batches", type=int, default=100)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--max-tokens", type=int, default=50000000)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--stop-on-error", action="store_true")
    return p.parse_args()

def lr_name(lr):
    return f"lr-{lr:.0e}".replace("e-0", "e-").replace("e+0", "e+")

def main():
    args = parse_args()
    root = Path(args.output_root)
    root.mkdir(parents=True, exist_ok=True)
    common = [
        "--train-bin", args.train_bin, "--eval-bin", args.eval_bin,
        "--device", args.device, "--micro-batch-size", str(args.micro_batch_size),
        "--effective-tokens", str(args.effective_tokens),
        "--warmup-fraction", str(args.warmup_fraction), "--seed", str(args.seed),
        "--eval-batches", str(args.eval_batches), "--log-every", str(args.log_every),
    ]
    if args.max_steps is not None: common += ["--max-steps", str(args.max_steps)]
    if args.max_tokens is not None: common += ["--max-tokens", str(args.max_tokens)]

    results = []
    train_script = Path(__file__).with_name("train.py")
    for index, lr in enumerate(args.lrs, 1):
        name = lr_name(lr)
        output = root / name
        print(f"\n[{index}/{len(args.lrs)}] Starting {name}", flush=True)
        command = [
            sys.executable, str(train_script), "--lr", str(lr), "--out", str(output),
            "--log-dir", str(output / "logs"), *common,
        ]
        completed = subprocess.run(command, check=False)
        results.append((lr, completed.returncode))
        if completed.returncode == 0:
            print(f"[{index}/{len(args.lrs)}] Finished {name}", flush=True)
        else:
            print(f"[{index}/{len(args.lrs)}] Failed {name} (exit {completed.returncode})", flush=True)
            if args.stop_on_error: break

    print("\nSweep summary:")
    for lr, code in results: print(f"  lr={lr:g}: {'ok' if code == 0 else f'failed ({code})'}")
    if any(code != 0 for _, code in results): raise SystemExit(1)

if __name__ == "__main__": main()
