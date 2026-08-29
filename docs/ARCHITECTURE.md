# ExperimentLM Architecture Guide

This is the orientation document for architecture and training work. Read it before changing the framework; the source files should only be needed for the subsystem being modified.

## Project invariants

- Experiments are defined by JSON, loaded into validated frozen dataclasses, and resolved before execution.
- JSON `"class"` values are exact registered Python class names. Arbitrary imports are never accepted from configuration.
- There is one model construction path (`build_model`) and one training loop (`trainer.train`).
- New architectures should normally require a class, a registry entry, and a config change—not a trainer change.
- Checkpoints and result directories contain the fully resolved experiment config. Layer repeats and registered constructor defaults are expanded.
- The controlled baseline is `configs/standard_transformer_plus.json`: 10,161,920 parameters, 7 layers, width 256, 4 heads, head dimension 64, SwiGLU width 768, RoPE, RMSNorm, untied embeddings, and no dropout.
- The historical baseline uses `ParallelResidual`: attention and MLP both consume the same incoming residual stream. `StandardResidual` is the conventional sequential attention-then-MLP form. Do not silently switch between them.

## Repository map

| Path | Responsibility |
| --- | --- |
| `configs/standard_transformer_plus.json` | Canonical baseline experiment and schema example |
| `experimentlm/config.py` | Validated internal config dataclasses and JSON parsing |
| `experimentlm/resolution.py` | Registry validation, constructor-default expansion, repeat expansion output |
| `experimentlm/registry.py` | Explicit registries; no dynamic imports |
| `experimentlm/model.py` | Component interfaces, registered baseline components, `CausalLM`, `build_model` |
| `experimentlm/optim.py` | Optimizer and LR-schedule interfaces/factories |
| `experimentlm/data.py` | Deterministic memory-mapped token stream |
| `experimentlm/evaluation.py` | Held-out loss/perplexity evaluation |
| `experimentlm/checkpoint.py` | Checkpoint save/load and legacy model-config conversion |
| `experimentlm/generation.py` | Framework-level autoregressive generation |
| `experimentlm/trainer.py` | Architecture-agnostic training loop and run logging |
| `train.py` | Thin `--config` CLI |
| `generate.py` | Thin generation CLI |
| `run_lr_sweep.py` | Derives JSON variants and invokes the same training CLI |
| `export_hf.py`, `modeling_experimentlm.py` | Hugging Face/lm-eval adapter path |
| `benchmark.py` | Local validation and zero-shot lm-eval tasks |
| `tests/test_configuration.py` | Config expansion, parameter-count, registry, and heterogeneous-stack checks |

Dataset preparation remains separate: `download_dataset.py`, `train_tokenizer.py`, and `tokenize_dataset.py` produce the frozen JSONL corpus, tokenizer, and `uint16` streams.

## Runtime flow

```text
experiment JSON
  -> ExperimentConfig.load
  -> validated dataclasses
  -> resolve_experiment
       - validates registered class names and arguments
       - fills registered constructor defaults
       - expands every layer repeat
  -> trainer.train
       - build_model(model config)
       - build_optimizer / build_schedule
       - TokenStream train/eval iterators
       - generic update/evaluation/checkpoint loop
```

`trainer.train` resolves again so programmatically created configs receive the same guarantees as file-loaded configs.

## Configuration model

The top-level sections are:

- `model`: architecture and initialization.
- `training`: optimizer, schedule, precision, batching, stopping, logging, validation, and checkpoint intervals.
- `data`: train/eval token-stream paths.
- `evaluation`: number of validation batches; `0` means the complete stream.
- The config filename (without `.json`) is used beneath both `checkpoints/` and `logs/`.

Every selectable implementation is represented as:

```json
{"class": "RegisteredClassName", "args": {}}
```

Layer entries additionally accept `"repeat"`. It is input shorthand only: resolved and checkpoint configs contain one explicit entry per layer. A heterogeneous stack is simply an ordered list of entries with different block classes or arguments.

The model context length is authoritative. A legacy `training.context_length` is accepted only when it matches and is omitted from resolved output.

## Registries and constructor injection

Registries are grouped by role: models, embeddings, position encodings, blocks, norms, attention, MLPs, residuals, heads, initializations, optimizers, and schedules.

Framework dimensions are injected into constructors rather than repeated inside every component config:

| Component | Injected arguments |
| --- | --- |
| Model | Complete `ModelConfig` |
| Input embedding | `vocab_size`, `d_model` |
| Position encoding | `context_length` |
| Block | `d_model` |
| Norm inside `TransformerBlock` | `dim` |
| Attention/MLP inside `TransformerBlock` | `d_model` |
| Output head | `d_model`, `vocab_size` |
| Initializer | Applied with model and resolved layer count |

All other constructor arguments come from the component's JSON `args`. Resolution uses the registered constructor signature to reject unknown/missing arguments and materialize defaults.

Registration uses the class name directly:

```python
@BLOCKS.register
class MyBlock(Block):
    ...
```

If a component is placed in a new module, import that module explicitly during package initialization so registration has happened before config resolution. Do not add dynamic module paths to JSON.

## Model component contracts

- `PositionEncoding.apply(q, k) -> (q, k)` transforms attention queries and keys. The baseline `RoPE` builds device/head-dimension caches lazily and supports heterogeneous attention head dimensions.
- `Attention.forward(x, attention_mask, position_encoding) -> x` returns a projected attention branch.
- `MLP.forward(x) -> x` returns an MLP branch.
- `Residual.forward(block, x, attention_mask, position_encoding) -> x` defines branch ordering and combination.
- `Block.forward(x, attention_mask, position_encoding) -> x` is the uniform stack interface. Non-attention blocks may ignore mask/position arguments.

`CausalLM` owns the input embedding, one position-encoding instance, the explicit `ModuleList` of blocks, final norm, and output head. It enforces context length, optionally ties input/output weights, and returns `(logits, loss)`.

The baseline state-dict names (`wte`, `blocks`, `norm_f`, `lm_head`, and block `norm1`, `attn`, `norm2`, `mlp`) are intentionally retained for old checkpoint compatibility.

## Adding an architecture experiment

1. Implement the smallest appropriate interface in `experimentlm/model.py`, or in a focused new module.
2. Register the class in the matching explicit registry.
3. If it is in a new module, add an explicit package import for registration.
4. Reference the exact class name and arguments in a new JSON config.
5. Add a focused construction/forward test. For a new block family, test it in a heterogeneous stack.
6. Do not branch on the new class in `trainer.py`.

A custom SSM or recurrent block should implement `Block`; it does not need fake attention or MLP components. A new attention, norm, MLP, residual strategy, embedding, head, or position encoding can plug into `TransformerBlock` through its corresponding nested component spec.

## Optimizers and schedules

`OptimizerFactory.build(parameters)` and `ScheduleFactory.build(optimizer, total_steps)` are the extension contracts. `AdamW` and `CosineSchedule` are the baseline registrations. Add Lion, Muon, or other methods by registering another factory; the trainer only consumes the resulting PyTorch optimizer/scheduler interfaces.

The cosine baseline uses linear warmup followed by cosine decay to `final_lr_fraction` of peak LR. Optimizer parameters remain FP32 because the model parameters are FP32 and bf16 is applied through autocast.

## Data and update accounting

`TokenStream` memory-maps a flat `uint16` file, partitions it into non-overlapping context windows, shuffles window starts once using its seed, and yields already-batched `(inputs, labels)`. Document boundaries were flattened during tokenization with explicit `<eos>` tokens.

The trainer computes:

```text
accumulation = ceil(tokens_per_step / (micro_batch_size * context_length))
effective_tokens_per_update = micro_batch_size * accumulation * context_length
```

Incomplete microbatches and incomplete gradient accumulations are dropped per epoch. Each epoch replays the same deterministic shuffled ordering. Resume skips consumed microbatches within the current epoch.

The frozen streams currently contain 527,873,925 train tokens and 59,060,913 eval tokens under the custom tokenizer. The source download target was measured with the source tokenizer, which explains the difference from the nominal 500M tokens.

## Training, outputs, and resume

Run:

```bash
python train.py --config configs/standard_transformer_plus.json --device cuda
```

The trainer logs parameter count, resolved model/training settings, optimizer/schedule, effective tokens per update, loss, LR, throughput, gradient norm, and peak VRAM. It writes:

```text
checkpoints/<config-filename>/
  config.json
  results.json
  step-<N>/
    checkpoint.pt
    config.json
  final/
    checkpoint.pt
    config.json
```

`checkpoint.pt` contains model, optimizer, scheduler, step, token count, and the complete experiment config. `--resume` restores all trainable state. Old baseline checkpoints containing the pre-refactor compact model config are converted to the explicit parallel-residual architecture during loading.

## Generation and evaluation

- `generate.py` loads native checkpoints and calls `experimentlm.generation.generate`.
- `export_hf.py` writes a Transformers custom-model directory containing the resolved model config.
- `benchmark.py` runs complete/local held-out validation followed by zero-shot HellaSwag, PIQA, ARC-Easy, BoolQ, and LAMBADA.
- LAMBADA output retains official word perplexity and adds custom-tokenizer token perplexity when sample logging is available.

The exported Transformers adapter imports the local `experimentlm` package; exports are compatible inside this project environment but are not standalone distributions.

## Focused verification

Run the architecture/configuration tests after component or schema changes:

```bash
python -m unittest tests/test_configuration.py
```

For trainer changes, use a tiny temporary model and token stream before considering a full baseline run. Full baseline training processes roughly 528M train tokens and is not an appropriate smoke test.

When an intentional architecture change alters baseline parameter count, state-dict compatibility, residual semantics, data order, or update accounting, document it explicitly and use a new experiment config rather than silently changing `configs/standard_transformer_plus.json`.
