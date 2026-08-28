# ExperimentLM

Small decoder-only causal LM matching the project baseline: 7-layer, 256-wide RMSNorm/RoPE/SwiGLU model (~10M parameters), uint16 mmap data, bf16 autocast, AdamW, warmup/cosine schedule, validation and checkpoints.

```bash
python3 train.py --device cuda --micro-batch-size 2 --lr 6e-4
python3 generate.py --checkpoint checkpoints/10m/final --prompt "The future of language models is"
```

Checkpoints contain `checkpoint.pt` and `config.json`; `generate.py` loads them directly. The optimizer state is FP32 under standard AdamW. Use `--resume checkpoints/10m/step-500` to resume. The included `benchmark.py` remains the lm-eval Hugging Face entry point; use `export_hf.py` when `transformers` is installed to create a compatible model directory.
