**DISCLAIMER:** While the ideas are mine, the code is mostly vibecoded. I did work to make it as optimized as i could, but there might still be a fair bit of sloppy code, especially in the not-so-vital-for-performance parts of the code.

# ExperimentLM

JSON-configured decoder-only LM research framework. Architectures are explicit registered component stacks; training, optimization, data, evaluation, checkpointing, and generation are separate modules. `configs/standard_transformer_plus.json` is the canonical control model (7 layers, width 256, 4 heads × 64, SwiGLU 768, RoPE, RMSNorm, sequential `StandardResidual`, untied embeddings, no dropout); `configs/gated_residual_transformer.json` is the same stack with learned per-branch residual gates (`GatedResidual`).

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the configuration lifecycle, component contracts, extension workflow, training semantics, and compatibility notes. See [`Concise recap roughly.txt`](<Concise recap roughly.txt>) for the running experiment log in the order it was explored.

## Goal

Find small-language-model architectures and training methods that improve **final validation quality per unit of wall-clock**, ideally beating a standard full-MHA Transformer at the same or higher training speed. Parameter count is secondary; throughput and final quality are the main constraints.

## JSON-configured architecture

One experiment is one JSON file, and [`configs/standard_transformer_plus.json`](configs/standard_transformer_plus.json) is the canonical example and schema reference. It has four sections: `model` (architecture and initialization), `training` (optimizer, schedule, precision, batching, stopping, and logging/checkpoint intervals), `data` (the two token streams), and `evaluation` (validation batch count, where `0` evaluates the complete stream). The filename without `.json` becomes the experiment name and the directory used beneath both `checkpoints/` and `logs/`.

Every selectable piece is written as a registered class plus its arguments, and `class` values must be exact names from the thirteen explicit registries — models, embeddings, position encodings, blocks, norms, attention, MLPs, residuals, value routings, output heads, initializations, optimizers, schedules:

```json
{"class": "GatedDeltaNet2LinearAttention", "args": {"n_heads": 4, "head_dim": 64, "expand_v": 1.0, "conv_size": 4}}
```

Nothing is imported dynamically from configuration, so a typo fails at load listing the registered alternatives. Model dimensions are injected by the framework rather than repeated inside every component: the embedding receives `vocab_size` and `d_model`, the position encoding `context_length`, each block `d_model`, nested norms `dim`, nested attention and MLP `d_model`, and the output head `d_model` and `vocab_size`. Everything else comes from `args`, and `model.context_length` is authoritative.

Layer entries accept a `repeat` count as input shorthand. `TransformerBlock` composes a norm, attention, MLP, and residual strategy — with `"mlp": null` allowed for an attention-only stage — while `FFNOnlyBlock` is an MLP stage with no mixing at all:

```json
{"class": "TransformerBlock", "repeat": 7, "args": {
  "pre_norm": true,
  "norm": {"class": "RMSNorm", "args": {"eps": 1e-6}},
  "attention": {"class": "MultiHeadCausalSelfAttention", "args": {"n_heads": 4, "head_dim": 64}},
  "mlp": {"class": "SwiGLU", "args": {"d_ff": 768}},
  "residual": {"class": "StandardResidual", "args": {}}
}}
```

Loading a config resolves it before anything runs: `repeat` becomes one explicit entry per layer, registered constructor defaults are materialized, and unknown or missing arguments are rejected against the registered signature. A heterogeneous stack is therefore just an ordered list of entries — that is how `new_baseline` interleaves three attention blocks with four `FFNOnlyBlock` stages, and how the SWA/GDN2/global-GQA design in the results tables below is expressed. Resolution runs again inside `trainer.train`, so programmatically built configs get the same guarantees, and every checkpoint stores the fully resolved config beside `checkpoint.pt`.

There is one construction path (`build_model`) and one training loop (`trainer.train`); a new architecture normally means a class, a registry entry, and a config change — not a trainer change. `docs/ARCHITECTURE.md` carries the full injection table, component contracts, and the extension checklist.

## Environment

Ubuntu 24.04, Ryzen 9 7950X3D, 64 GB RAM, Radeon RX 7900 XTX (24 GB). Python 3.12.3 with PyTorch 2.13.0+rocm7.2 (HIP 7.2) and `fla-core[rocm]==0.5.2`.

```bash
pip install -r requirements.txt   # pin adjusted for non-ROCm hardware
```

## Dataset preparation

The corpus artifacts are gitignored; regenerate them locally before training. File paths and vocab size are module-level constants in these scripts, not CLI flags, and `download_dataset.py` refuses to overwrite an existing `smollm_500m.jsonl`.

```bash
python3 download_dataset.py     # -> smollm_500m.jsonl   (~2.3 GB, ~500M-token SmolLM mixture subset)
python3 train_tokenizer.py      # -> tokenizer.json      (byte-level BPE, 8,192 vocab, reserved chat/reasoning tokens)
python3 tokenize_dataset.py     # -> train.bin, eval.bin (uint16 streams, 90/10 document split, seed 12345)
```

The frozen streams contain 527,873,925 train tokens and 59,060,913 eval tokens under the custom tokenizer.

## Training

```bash
python3 train.py --config configs/new_baseline.json --device cuda
python3 train.py --config configs/new_baseline.json --device cuda --resume checkpoints/new_baseline/step-2000
python3 run_lr_sweep.py --config configs/new_baseline.json --lrs 3e-4 6e-4 1e-3 --device cuda
```

`train.py` takes `--config`, `--device`, `--resume`. Runs write `checkpoints/<config-stem>/{config.json,results.json,step-<N>/,final/}` and `logs/<config-stem>/run-<timestamp>.log`; `checkpoint.pt` carries model, optimizer, scheduler, step, token count, resolved config, and RNG state. `run_lr_sweep.py` derives per-run JSON configs and writes them under `sweeps/lr/`. Repeated input layers are expanded when configs are loaded and saved, so checkpoint configs contain the exact heterogeneous stack that ran.

A standard 10M run is one epoch of 4,027 steps at 131,072 tokens per optimizer step (micro-batch 8, context 2048, bf16 autocast, `torch.compile`, seed 12345).

## Generation and evaluation

```bash
python3 generate.py --checkpoint checkpoints/new_baseline/final --prompt "The future of language models is"
python3 export_hf.py --model checkpoints/new_baseline/final     # -> benchmarks/new_baseline/
python3 benchmark.py --model benchmarks/new_baseline --trust-remote-code
```

Checkpoints contain `checkpoint.pt` and the fully resolved `config.json`; `generate.py` loads them directly. `benchmark.py` first runs held-out validation over the complete `eval.bin`, then zero-shot HellaSwag, PIQA, ARC-Easy, BoolQ, and LAMBADA (`--tasks all` omits BoolQ), writing `benchmark_results.json` next to the export. Use `--limit 100` for a fast smoke pass. The exported Transformers adapter imports the local `experimentlm` package, so exports are only loadable inside this project environment.

## Experiments conducted

### How to read the tables

- **Full val** is the complete held-out stream (59,047,936 tokens) measured by `benchmark.py`. It is the only cross-architecture quality number.
- The trainer's own `validation final` line (and `results.json`) covers a 1,638,400-token slice and reads ~0.01–0.02 lower; never compare it against full val.
- **tok/s** is median steady-state throughput over the second half of the logged steps at 2048 context and 131,072 tokens per update. Every run below is a single fixed seed (12345).
- Rows marked "no export" were never run through `export_hf.py`/`benchmark.py`, so only their in-run validation exists.

### Completed runs with configs still in the repo

10M scale, one epoch of 527.8M tokens, context 2048:

| Config | Stack | Params | tok/s | Full val | PPL | Verdict |
| --- | --- | --- | --- | --- | --- | --- |
| `standard_transformer_plus` | 7× MHA(4×64)+SwiGLU, sequential residual, untied, cosine | 10,161,920 | 433.6k | 3.3873 | 29.59 | plain full-MHA control |
| `gated_residual_transformer` | 7× MHA(4×64)+SwiGLU+gates, untied, cosine | 10,161,934 | 446.2k | 3.3818 | 29.42 | residual gates ≈ free, ~+3% speed, kept |
| `xielu_test` | 7× MHA(4×64)+XIELU, untied, cosine | 10,161,934 | 459.2k | no export (in-run 3.3613) | – | xIELU MLP family, reference point for the MTP work |
| `gated_deltanet_qknorm_wsd` | MHA-Q/SwiGLU + GDN1 stages, QK-Norm, WSD | 10,445,230 | 365.5k | 3.2547 | 25.91 | −0.133 vs control but ~16% slower |
| `gated_deltanet2_qknorm_wsd` | MHA-Q/SwiGLU + GDN2 stages, QK-Norm, WSD | 10,447,262 | 298.7k | 3.2291 | 25.26 | best GDN hybrid quality, another ~18% slower than GDN1 |
| **`new_baseline`** | MHA-Q(6×64)+SwiGLU ×3 interleaved with 4 FFN-only blocks, `d_model=384`, tied, WSD | 11,112,966 | 466.1k | 3.2943 | 26.96 | **best quality-per-wall-clock**: −0.093 vs control at ~+7% speed |
| `swa_gdn2_global_mid_gqa_tied_depth_ffn_speed` | SWA→GDN2→FFN→GDN2→global GQA→GDN2→FFN→GDN2→SWA, tied, `d_ff=832`, 4Q/2KV, WSD | 10,294,046 | 262.6k | **3.2011** | 24.56 | **best 10M full val**, but ~44% slower than `new_baseline` |

100M scale, six epochs, context 2048:

| Config | Stack | Params | tok/s | Full val | PPL |
| --- | --- | --- | --- | --- | --- |
| `gated_deltanet_100m` | 10 layers, `d_model=768`, MHA/GDN1 hybrid, cosine | 98,879,780 | 75.5k | 2.6267 | 13.83 |
| `gated_deltanet_100m_qknorm_wsd` | same + QK-Norm + WSD | 98,880,292 | 75.5k | 2.5823 | 13.23 |
| `swa_gdn2_global_mid_gqa_tied_depth_ffn_speed_100m` | 11 layers, SWA/GDN2/global-GQA/FFN, tied, WSD | 98,981,210 | 60.5k | 2.5719 | 13.09 |

Zero-shot scores from the same exports (HellaSwag `acc_norm`, PIQA `acc_norm`, ARC-Easy `acc`, BoolQ `acc`, LAMBADA custom-tokenizer PPL):

| Config | HellaSwag | PIQA | ARC-E | BoolQ | LAMBADA PPL |
| --- | --- | --- | --- | --- | --- |
| `standard_transformer_plus` | 0.266 | 0.544 | 0.347 | 0.390 | 23.56 |
| `gated_deltanet2_qknorm_wsd` | 0.274 | 0.550 | 0.355 | 0.409 | 19.14 |
| `new_baseline` | 0.271 | 0.535 | 0.346 | not run | 21.55 |
| `swa_gdn2_global_mid_gqa_tied_depth_ffn_speed` | 0.269 | 0.550 | 0.348 | not run | 19.18 |
| `swa_gdn2_global_mid_gqa_tied_depth_ffn_speed_100m` | 0.308 | 0.613 | 0.480 | not run | 7.25 |

At 10M most tasks sit near chance (0.25 for four-choice tasks), so full val is the decisive metric at this scale; the zero-shot suite only becomes informative at 100M.

### What the experiments established

**Training recipe.** QK-Norm, learned residual gates, tied embeddings, and the WSD schedule are the strong baseline. QK-Norm was a clear quality win on attention layers; gates cost ~nothing and helped throughput; full-rank tied embeddings freed enough parameter budget to deepen the network and paid off strongly; WSD beat cosine by roughly −0.038 val loss on the QK-Norm baseline, later tuned to 75% stable / 15% decay / 10% minimum-LR hold (the earlier cooldown looked more useful than the min-LR hold). Rejected variants: channel-wise gates (slightly worse), per-head attention scaling on top of QK-Norm (early gain, worse later), projected embeddings at `d_embed=128` (worse and slower), and CLVR (~0.0066 loss for 3–4% slowdown, kept only as an optional branch).

**GDN1/GDN2 hybrids.** GDN1 was a substantially better direction than pure MHA and reasonably fast. GDN2 gave a real quality step (−0.0256 full val at 10M) but ~18% slower, and it stayed too expensive to win on time-to-quality against the sparse-communication design. `expand_v=2`, value-head sharing, and `num_v_heads < n_heads` were rejected or unsupported by FLA semantics. GDN2 head geometry settled on 4×64: both 8×32 and 2×128 were slower on the 7900 XTX.

**Attention geometry and context.** Head dimension 64 is the practical default (32 and 128 were worse or slower in MHA tests). GQA (4Q/2KV) cost ~+0.0057 val loss for ~3–4% speed, ~200k fewer parameters, and half the KV cache — accepted as a scalability trade. Sliding-window attention was slightly slower than global MHA at 2K context but ~9% faster at 100M/8K with essentially tied quality, so it is kept as a long-context feature. One exact global attention stage in the middle of the stack was retained to give the model a retrieval path after representations mature.

**Compression and shared-state memory ideas.** Hierarchical 4:1 compression was computationally excellent and quality-disastrous; both decompression and causal cross-attention memory variants failed, and widening the low-rate stream did not rescue it. A materialized read-only GDN2 also failed: strict causality required exposing every prefix state (~1 GiB FP32 at B8/T2048), costing ~179% of a full GDN2 per reader. A fused writer+reader GDN2 is the viable form (~131% forward, ~150% forward+backward versus ~197%/~204% for two full GDN2s, with bit-exact recurrence gradients), which turns "one writer plus a second learned read of the same state" into a usable primitive.

**Sparse communication plus FFN-only depth — the main win.** Replacing most mixers with FFN-only stages and keeping only three attention blocks, then widening the residual stream to `d_model=384`, beat the 7×MHA control by **0.0930 full val (3.3873 → 3.2943) at 466.1k vs 433.6k tok/s**. Concentrating FFN capacity into fewer, wider blocks did not help; distributed FFN width was better, and uniform `d_ff=832` (over 768) bought ~0.0113 full val plus better benchmark scores for ~2–3% throughput at 10M.

**Conditional depth / MoD-style routed FFNs.** Rejected: routing overhead made them slower than dense FFNs at this scale.

**Multi-token prediction (MTP).** All variants used the xIELU `xielu_test` reference (459.2k tok/s, 3.3613 in-run val) and were measured with `benchmark_mtp.py` for isolated head cost:

- 4-horizon equal-weight MTP: 336.7k tok/s (~27% slower) and it degraded H1.
- Corrected auxiliary weighting at λ=0.1: better H1 learning, still ~340k tok/s.
- H2-only, dense, λ=0.1: 408.8k tok/s — ~11% slowdown while retaining much of the benefit.
- λ sweep at 1,000–1,500 steps: 0.1 beat both 0.05 and 0.2 (all three ran at 406–409k tok/s).
- Low-rank xIELU H2 head (`rank=128`): 418.0k tok/s, i.e. only ~3% faster than the dense H2 head, with worse quality. Rejected.
- H2 on 50% of positions (full-width head): 434.8k tok/s and a strong early lead, but the baseline caught up and finished ahead (3.3613 → 3.3721 final).
- H2 only for the first 1,000 steps, then pure NTP: 459.4k tok/s, i.e. baseline speed, and the same story — a head start with no lasting final-quality gain (3.3643).

Conclusion: MTP changes the shape of the early training curve, not the endpoint, so it does not pay for itself under a quality-per-wall-clock objective at 10M / one epoch.

## Caveats

- Single seed for every comparison; the protocol is to re-run promising designs across multiple seeds only after a design is selected.
- Several earlier experiments (QK-Norm on/off, head-dim and GQA/SWA ablations, CLVR, hierarchical compression, matrix-memory diagnostics, and most MTP λ variants) ran from configs that have since been deleted, so their numbers are recorded here but are not reproducible from a fresh clone. `configs/` currently holds only the 10 configs listed in the tables above plus `xielu_mtp.json`.
- `checkpoints/`, `logs/`, `benchmarks/`, `sweeps/`, the corpus, and the tokenizer are all gitignored; a fresh clone ships code, configs, and this results record only.
