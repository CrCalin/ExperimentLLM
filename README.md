# ExperimentLM

Small decoder-only causal LM matching the project baseline: 7-layer, 256-wide RMSNorm/RoPE/SwiGLU model (~10M parameters), uint16 mmap data, bf16 autocast, AdamW, warmup/cosine schedule, validation and checkpoints.

```bash
python3 train.py --device cuda --micro-batch-size 2 --lr 6e-4
python3 generate.py --checkpoint checkpoints/10m/final --prompt "The future of language models is"
python3 run_lr_sweep.py --max-steps 100 --micro-batch-size 2
```

Checkpoints contain `checkpoint.pt` and `config.json`; `generate.py` loads them directly. The optimizer state is FP32 under standard AdamW. Use `--resume checkpoints/10m/step-500` to resume. The included `benchmark.py` remains the lm-eval Hugging Face entry point; use `export_hf.py` when `transformers` is installed to create a compatible model directory.

`run_lr_sweep.py` runs the default peak learning rates `3e-4`, `6e-4`, `1e-3`, and `2e-3` sequentially. Override them with, for example, `--lrs 0.0003 0.0006 0.001`; each run gets its own checkpoint and timestamped log directory.
