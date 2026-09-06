# Per-layer matrix-memory integration cost

This is a benchmark-only cost model. It does not modify `TransformerBlock` or
feed memory output back into attention or the MLP. The synthetic branch receives
fully materialized `H` and `Y`, generates controls, calls the validated BF16
compact-WY operator, projects reads back to `d_model`, and includes the
synthetic `H + memory_output` residual add.

## Harness

The final capture used B=8, T=2048, `d_model=256`, BF16 autocast over FP32
parameters, WY chunk size 128, two warmups, and five timed iterations. Optimizer
steps were excluded from both baseline and branch timings. GPU synchronization
was performed around every sample. The baseline was the actual seven-layer
`standard_transformer_plus` model; H and Y for the branch were independent
fully precomputed tensors.

Environment: RX 7900 XTX, ROCm 7.14.60850, PyTorch 2.14.0+rocm7.14.

## Canonical measurements

| Workload | Time | Throughput |
| --- | ---: | ---: |
| Full baseline, eager, forward+backward | 51.55 ms | 315K tok/s |
| Full baseline, compiled, forward+backward | 34.78 ms | 471K tok/s |
| One fused memory branch, forward | 6.16 ms | 2.66M tok/s |
| One fused memory branch, forward+backward | 12.77 ms | 1.28M tok/s |
| One measured baseline Transformer block, forward+backward | 6.21 ms | 2.63M tok/s |

The compiled baseline agrees with the historical approximately 466K tok/s
training figure. One memory branch is therefore 2.06x the directly measured
cost of one baseline block, or 2.57x one seventh of the compiled full-model
step time.

## Branch breakdown

The following percentages are normalized from isolated forward+backward stage
measurements; isolated calls have additional launch overhead and therefore do
not sum exactly to the end-to-end branch wall time.

| Stage | Time | Share of isolated stage sum |
| --- | ---: | ---: |
| Fused read GEMM | 0.31 ms | 2.2% |
| Read normalization + sigmoid/gate | 0.36 ms | 2.6% |
| Fused write GEMM | 0.50 ms | 3.6% |
| Write normalization + two sigmoids/gates | 0.69 ms | 4.9% |
| Compact-WY recurrence | 11.55 ms | 82.7% |
| Output projection + scalar read gate | 0.56 ms | 4.0% |

Aggregated read-control work is 4.8%, write-control work 8.5%, nonlinearities
and normalization together 7.5%, recurrence 82.7%, and output projection 4.0%.
The profiler identifies compact-WY factor construction as the dominant device
stage; backward/autograd adds approximately the difference between the 6.16 ms
forward and 12.77 ms training costs.

The measured baseline block decomposed into approximately 3.99 ms attention,
2.32 ms MLP, and 0.24 ms normalization/residual shell in forward+backward.

## Projection variants and simplifications

Separate full projections measured 13.60 ms forward+backward versus 12.77 ms
for fused read/write projections, a modest 6.1% training improvement. Forward
time was effectively similar in this capture. Both variants have 131,265
parameters; fusion reduces projection launches, not parameter count.

At the canonical workload:

| Variant | Training time | Parameters | Change vs full fused |
| --- | ---: | ---: | ---: |
| Full controls | 12.77 ms | 131,265 | — |
| No q gate | 12.82 ms | 114,817 | no reliable gain |
| No erase gate | 12.83 ms | 114,817 | no reliable gain |
| No write gate | 12.77 ms | 114,817 | neutral |
| No q/e/k gates | 12.40 ms | 81,921 | 2.9% faster |
| q reused from H slice, q gate retained | 13.09 ms | 114,881 | 2.5% slower |

The no-gate result is a small timing improvement rather than a decisive
architectural result. q-slice reuse did not help and is only a benchmark
ablation, not a proposed change.

## Seven-layer projection

Using the measured compiled baseline step and serially adding seven measured
fused branches:

| Assumption | Predicted throughput | Slowdown |
| --- | ---: | ---: |
| Optimistic: 20% branch overlap/fusion | 154K tok/s | 206% |
| Base: measured branch cost | 132K tok/s | 257% |
| Pessimistic: 20% additional integration overhead | 115K tok/s | 308% |

The base estimate is `34.78 ms + 7 × 12.77 ms = 124.19 ms` per 16,384-token
step. The model is not currently economically viable as a branch in every
layer at this implementation cost.

To meet the target slowdown bands, each branch would need to fall to:

| Target slowdown | Maximum branch time | Required speedup |
| --- | ---: | ---: |
| 30% | 1.49 ms | 8.6x |
| 20% | 0.99 ms | 12.9x |
| 10% | 0.50 ms | 25.7x |

## Scaling, parameters, FLOPs, and activations

With B=8, `d_model=256`, and C=128, fused branch forward+backward measured:

| `d_memory` | Time | Throughput |
| ---: | ---: | ---: |
| 32 | 12.86 ms | 1.27M tok/s |
| 64 | 12.77 ms | 1.28M tok/s |
| 128 | 13.27 ms | 1.23M tok/s |
| 256 | 15.90 ms | 1.03M tok/s |

At d=64, the corresponding sequence sweep was 5.29 ms / 0.77M tok/s at
T=512, 12.77 ms / 1.28M at T=2048, 25.43 ms / 1.29M at T=4096, and 49.75 ms /
1.32M at T=8192. Runtime is approximately linear in T; no quadratic sequence
dependency appeared.

For the fused full design, additional trainable parameters per layer and over
seven layers are:

| `d_memory` | Per layer | Seven layers | Increase over 10,161,920 |
| ---: | ---: | ---: | ---: |
| 32 | 65,633 | 459,431 | 4.52% |
| 64 | 131,265 | 918,855 | 9.04% |
| 128 | 262,529 | 1,837,703 | 18.08% |
| 256 | 525,057 | 3,675,399 | 36.17% |

The d=64 parameter breakdown is `d_model × (2d + 5d + d)` for the three
fused matrices, three gate biases, and one scalar read gate.

Approximate canonical FLOPs/token are:

| Component | FLOPs/token |
| --- | ---: |
| Read projection | 65,536 |
| Write projection | 163,840 |
| Compact-WY recurrence | 2,310,144 |
| Output projection | 32,768 |
| Total branch | 2,572,288 |
| Baseline Transformer estimate | 3,801,088 |

The recurrence estimate includes the transient binary factor GEMMs and is
supporting analysis only; measured wall time is authoritative.

The full fused branch retained 116,948,996 bytes of autograd tensors for the
canonical case. Seven independent branches would retain approximately
818,642,972 bytes (0.76 GiB) before allocator reuse and cross-layer scheduling.
The compact WY-specific saved tensors are compact chunk factors, U, q,
innovations, and chunk states; no tokenwise `d × d` state history is stored.

## Compilation and decision

The synthetic branch compiled successfully for forward with zero graph breaks:
8.27 s compilation and 3.85M tok/s steady-state. A compiled backward benchmark
was not used for the cost model because the standalone custom backward compile
path is materially more expensive and the current integration estimate must use
the measured eager training branch.

The next optimization target is compact-WY factor construction and its backward
bookkeeping, followed by reducing its kernel launches. Control projection
fusion is worthwhile but secondary. A specialized Triton or other ROCm kernel
should target the factor path before per-layer Transformer integration is
attempted.

All raw measurements, profiler events, environment metadata, parameter tables,
and derived cost-model thresholds are in
`benchmark_results/memory_integration_cost.json`. The benchmark implementation
is `benchmark_memory_integration.py`.
