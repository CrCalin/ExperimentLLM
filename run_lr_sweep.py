#!/usr/bin/env python3
"""Run sequential peak-learning-rate variants of one experiment config."""
import argparse
from dataclasses import replace
import subprocess
import sys
from pathlib import Path

from experimentlm.config import ExperimentConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/standard_transformer_plus.json")
    parser.add_argument("--lrs", nargs="+", type=float, default=[3e-4, 6e-4, 1e-3, 2e-3])
    parser.add_argument("--output-root", default="sweeps/lr")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--stop-on-error", action="store_true")
    args = parser.parse_args()
    base = ExperimentConfig.load(args.config)
    root = Path(args.output_root)
    root.mkdir(parents=True, exist_ok=True)
    results = []
    for index, lr in enumerate(args.lrs, 1):
        name = f"lr-{lr:.0e}".replace("e-0", "e-").replace("e+0", "e+")
        run_dir = root / name
        optimizer = replace(base.training.optimizer, args=base.training.optimizer.args | {"lr": lr})
        config = replace(
            base, training=replace(base.training, optimizer=optimizer),
            name=name,
        )
        config_path = run_dir / f"{name}.json"
        run_dir.mkdir(parents=True, exist_ok=True)
        config.save(config_path)
        print(f"[{index}/{len(args.lrs)}] Starting {name}", flush=True)
        completed = subprocess.run([
            sys.executable, str(Path(__file__).with_name("train.py")),
            "--config", str(config_path), "--device", args.device,
        ], check=False)
        results.append((lr, completed.returncode))
        if completed.returncode and args.stop_on_error: break
    for lr, code in results: print(f"lr={lr:g}: {'ok' if code == 0 else f'failed ({code})'}")
    if any(code for _, code in results): raise SystemExit(1)


if __name__ == "__main__": main()
