#!/usr/bin/env python3

import argparse
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    args = parser.parse_args()

    ckpt = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=False,
    )

    if "model" in ckpt:
        state = ckpt["model"]
    elif "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
    elif "state_dict" in ckpt:
        state = ckpt["state_dict"]
    else:
        state = ckpt

    gates = []

    for name, value in state.items():
        if ".residual." not in name:
            continue

        if not (
            name.endswith("attention_gate")
            or name.endswith("mlp_gate")
        ):
            continue

        gates.append(
            (name, value.detach().float().cpu().item())
        )

    if not gates:
        raise RuntimeError("No residual gates found.")

    print(f"Found {len(gates)} residual gates:\n")

    for name, value in gates:
        print(f"{name:50s} {value:+.6f}")

    values = torch.tensor([v for _, v in gates])

    print("\nSummary:")
    print(f"  mean: {values.mean().item():+.6f}")
    print(f"  min:  {values.min().item():+.6f}")
    print(f"  max:  {values.max().item():+.6f}")


if __name__ == "__main__":
    main()