#!/usr/bin/env python3
"""Run an experiment with the opt-in static-shape CUDA-graph trainer."""
import argparse

from experimentlm.config import ExperimentConfig
from experimentlm.cudagraph_trainer import train_cudagraphs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    train_cudagraphs(ExperimentConfig.load(args.config), args.device)


if __name__ == "__main__":
    main()
