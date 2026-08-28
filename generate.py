#!/usr/bin/env python3
import argparse

import torch
from tokenizers import Tokenizer

from experimentlm.checkpoint import load_checkpoint
from experimentlm.generation import generate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer", default="tokenizer.json")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    tokenizer = Tokenizer.from_file(args.tokenizer)
    model, _ = load_checkpoint(args.checkpoint, device)
    model.to(device)
    input_ids = torch.tensor([tokenizer.encode(args.prompt, add_special_tokens=False).ids], device=device)
    output = generate(model, input_ids, args.max_new_tokens, args.temperature, args.top_k)
    print(tokenizer.decode(output[0].tolist(), skip_special_tokens=False))


if __name__ == "__main__": main()
