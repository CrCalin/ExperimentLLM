#!/usr/bin/env python3
"""Run a JSON-configured ExperimentLM training experiment."""
import argparse

from experimentlm.config import ExperimentConfig
from experimentlm.trainer import train


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume")
    args = parser.parse_args()
    train(ExperimentConfig.load(args.config), args.device, args.resume)


if __name__ == "__main__": main()
