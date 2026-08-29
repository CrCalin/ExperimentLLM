# ExperimentLM

JSON-configured decoder-only LM research framework. Architectures are explicit registered component stacks; training, optimization, data, evaluation, checkpointing, and generation are separate modules. `configs/gated_residual_transformer.json` is the control model: a 7-layer, 256-wide RMSNorm/RoPE/SwiGLU model with standard sequential residuals and learned residual gates.

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the configuration lifecycle, component contracts, extension workflow, training semantics, and compatibility notes.

```bash
python3 train.py --config configs/gated_residual_transformer.json --device cuda
python3 train.py --config configs/standard_transformer.json --device cuda
python3 train.py --config configs/standard_transformer_tied.json --device cuda
python3 train.py --config configs/standard_transformer_deep.json --device cuda
python3 train.py --config configs/standard_transformer_wide.json --device cuda
python3 train.py --config configs/standard_transformer_wide_256.json --device cuda
python3 generate.py --checkpoint checkpoints/standard_transformer/final --prompt "The future of language models is"
python3 run_lr_sweep.py --config configs/gated_residual_transformer.json
```

Checkpoints contain `checkpoint.pt` and the fully resolved `config.json`; `generate.py` loads them directly. Use `--resume checkpoints/standard_transformer/step-500` to resume. Run `export_hf.py --model checkpoints/standard_transformer/final` to create `benchmarks/standard_transformer`.

`run_lr_sweep.py` derives per-run JSON configs for `3e-4`, `6e-4`, `1e-3`, and `2e-3` by default. Repeated input layers are expanded when configs are loaded and saved, so checkpoint configs contain the exact heterogeneous stack that ran.
