#!/usr/bin/env python3
"""Measure gated residual update strength by depth on a trained checkpoint."""
from __future__ import annotations

import argparse
import itertools
import math
from pathlib import Path

import torch

from experimentlm.checkpoint import load_checkpoint
from experimentlm.data import TokenStream
from experimentlm.model import GatedResidual


def rms_accumulator():
    return [0.0, 0]


def add_rms(accumulator, value: torch.Tensor) -> None:
    value = value.float()
    accumulator[0] += float(value.square().sum())
    accumulator[1] += value.numel()


def rms(accumulator) -> float:
    return math.sqrt(accumulator[0] / max(accumulator[1], 1))


@torch.inference_mode()
def measure(model, batches, device: torch.device, precision: str) -> list[dict[str, float | int | str]]:
    stats = []
    for block in model.blocks:
        if not isinstance(block.residual, GatedResidual):
            raise TypeError("diagnostic requires GatedResidual in every block")
        stats.append({
            "residual_before_mixer": rms_accumulator(),
            "mixer_raw": rms_accumulator(),
            "mixer_gated": rms_accumulator(),
            "residual_before_ffn": rms_accumulator(),
            "ffn_raw": rms_accumulator(),
            "ffn_gated": rms_accumulator(),
        })

    autocast_dtype = torch.bfloat16 if precision == "bf16" else None
    for inputs, _ in batches:
        inputs = inputs.to(device)
        with torch.autocast(device_type=device.type, dtype=autocast_dtype,
                            enabled=autocast_dtype is not None):
            x = model.wte(inputs)
            for index, block in enumerate(model.blocks):
                layer = stats[index]
                add_rms(layer["residual_before_mixer"], x)
                attn, route = block.attention_with_route(
                    block.norm1(x), None, model.position_encoding
                )
                gated_attn = block.residual.attention_gate * attn
                add_rms(layer["mixer_raw"], attn)
                add_rms(layer["mixer_gated"], gated_attn)
                x = x + gated_attn
                x = block.add_value_route(x, route, "pre_ffn")

                add_rms(layer["residual_before_ffn"], x)
                ffn = block.mlp(block.norm2(x))
                gated_ffn = block.residual.mlp_gate * ffn
                add_rms(layer["ffn_raw"], ffn)
                add_rms(layer["ffn_gated"], gated_ffn)
                x = x + gated_ffn
                x = block.add_value_route(x, route, "post_block")

    rows = []
    for index, (block, layer) in enumerate(zip(model.blocks, stats), start=1):
        mixer_gate = float(block.residual.attention_gate.detach().float())
        ffn_gate = float(block.residual.mlp_gate.detach().float())
        mixer_raw = rms(layer["mixer_raw"])
        ffn_raw = rms(layer["ffn_raw"])
        rows.append({
            "layer": index,
            "type": block.attn.__class__.__name__,
            "mixer_gate": mixer_gate,
            "mixer_raw": mixer_raw,
            "mixer_ratio": rms(layer["mixer_gated"]) / rms(layer["residual_before_mixer"]),
            "ffn_gate": ffn_gate,
            "ffn_raw": ffn_raw,
            "ffn_ratio": rms(layer["ffn_gated"]) / rms(layer["residual_before_ffn"]),
        })
    return rows


def broadly_increasing(values: list[float]) -> bool:
    increases = sum(right >= left for left, right in zip(values, values[1:]))
    return increases >= (len(values) - 1) // 2 + 1


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("checkpoints/gated_deltanet_qknorm/final"))
    parser.add_argument("--eval-bin", type=Path, default=Path("eval.bin"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--batches", type=int, default=16,
                        help="deterministic validation batches to aggregate")
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.batch_size <= 0 or args.batches <= 0:
        raise ValueError("--batch-size and --batches must be positive")
    device = torch.device(args.device)
    model, blob = load_checkpoint(args.checkpoint, device=device)
    model.to(device).eval()
    if len(model.blocks) != 7:
        raise ValueError(f"expected 7 blocks, found {len(model.blocks)}")

    context_length = model.config.context_length
    stream = TokenStream(args.eval_bin, context_length, args.batch_size, seed=12347)
    rows = measure(model, itertools.islice(stream, args.batches),
                   device, args.precision)

    step = int(blob.get("step", -1))
    tokens = args.batches * args.batch_size * context_length
    print(f"checkpoint={args.checkpoint} step={step} validation_tokens={tokens}")
    print("layer type                              mixer_gate mixer_raw mixer_ratio  ffn_gate ffn_raw ffn_ratio")
    for row in rows:
        print(f"{row['layer']:>5} {row['type']:<34} {row['mixer_gate']:>10.4f} "
              f"{row['mixer_raw']:>9.5f} {row['mixer_ratio']:>11.5f} {row['ffn_gate']:>9.4f} "
              f"{row['ffn_raw']:>7.5f} {row['ffn_ratio']:>9.5f}")

    mixer_gates = [row["mixer_gate"] for row in rows]
    ffn_gates = [row["ffn_gate"] for row in rows]
    mixer_ratios = [row["mixer_ratio"] for row in rows]
    ffn_ratios = [row["ffn_ratio"] for row in rows]
    print("trend:")
    print(f"  mixer gates: {'monotonic' if mixer_gates == sorted(mixer_gates) else 'not monotonic'}, "
          f"broadly increasing={broadly_increasing(mixer_gates)}")
    print(f"  FFN gates:   {'monotonic' if ffn_gates == sorted(ffn_gates) else 'not monotonic'}, "
          f"broadly increasing={broadly_increasing(ffn_gates)}")
    print(f"  mixer ratios: {'monotonic' if mixer_ratios == sorted(mixer_ratios) else 'not monotonic'}, "
          f"broadly increasing={broadly_increasing(mixer_ratios)}")
    print(f"  FFN ratios:   {'monotonic' if ffn_ratios == sorted(ffn_ratios) else 'not monotonic'}, "
          f"broadly increasing={broadly_increasing(ffn_ratios)}")


if __name__ == "__main__":
    main()
