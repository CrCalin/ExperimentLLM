# Local matrix memory

`MatrixMemoryTransformerBlock` retains ordinary full-sequence causal attention
and gives each layer one independent recurrent matrix with layout
`[batch, value, address]`. Blocks execute layer-major. Their memories neither
share state nor read or mutate another layer's state; later layers see memory
effects only through the normal residual stream.

After attention residual addition, token `t` reads `M[t-1]`, injects the gated
read, runs the normalized MLP, writes from the raw MLP output, and finally adds
the MLP residual. The update is:

```
M_minus = M - (alpha * (M @ e)) outer e
M_next  = M_minus + (beta * (v - M_minus @ k)) outer k
```

Query, key, and erase directions are L2-normalized and independently sigmoid-
attenuated. Erase and write strengths are vectors by default. There is no
passive decay. State uses the active BF16 autocast dtype, starts at zero for
every independent forward, and is never serialized.

The former shared global recurrence was removed because sharing one recurrent
state across both token and layer axes imposed `G[t,0] = G[t-1,L]`, forcing
token-major Transformer execution and causing catastrophic throughput loss.
Cross-layer communication is therefore handled by the existing residual
stream instead.

## Configuration and backends

See `configs/matrix_memory_transformer.json`. `d_memory` is unrestricted apart
from being positive. The default `chunked` backend executes compiled static
16-token cells and has no Python token loop. `reference` is the correctness
oracle; `scan` retains the PyTorch higher-order scan implementation for
comparison. `fused_projections=true` combines `[q, a_q]` and
`[k, a_k, e, a_e, v, alpha, beta]` into two GEMMs while retaining separate
bias semantics. Erase, delta correction, address tying, scalar/vector
strengths, attenuation, read-gate initialization, chunk size, and activation
checkpointing remain configurable ablations.

PyTorch 2.14 cannot lower the recurrent wrapper inside the outer compiled
model, so `_memory_chunked` is a deliberate Dynamo graph break in each memory
block. Each static chunk cell is independently compiled by Inductor. Chunk
activation recomputation bounds training memory.

## Parameter cost

With vector strengths, one layer adds
`10*d_model*d_memory + 5*d_memory + 1` parameters. At model width 256:

| d_memory | per layer | seven layers | increase over 10,161,920 baseline |
|---:|---:|---:|---:|
| 16 | 41,041 | 287,287 | 2.83% |
| 32 | 82,081 | 574,567 | 5.65% |
| 64 | 164,161 | 1,149,127 | 11.31% |
| 128 | 328,321 | 2,298,247 | 22.62% |
| 256 | 656,641 | 4,596,487 | 45.23% |

## RX 7900 XTX BF16 measurements

These use complete logits and cross-entropy backward under `torch.compile`.
The batch-8/context-2048 baseline reproduces observed training throughput.

| context/batch | variant | d_memory | forward tok/s | peak VRAM |
|---|---|---:|---:|---:|
| 512/8 | baseline | - | 1,901,738 | 0.14 GiB |
| 512/8 | local | 16 | 12,328 | 0.15 GiB |
| 512/8 | local | 32 | 12,225 | 0.15 GiB |
| 512/8 | local | 64 | 12,087 | 0.15 GiB |
| 512/8 | local | 128 | 12,130 | 0.15 GiB |
| 512/8 | local | 256 | 11,040 | 0.16 GiB |
| 2048/8 | baseline | - | 1,643,691 | 1.50 GiB |
| 2048/8 | local | 64 | 3,203 | 0.39 GiB |

| context/batch | variant | d_memory | training tok/s | peak VRAM |
|---|---|---:|---:|---:|
| 512/8 | baseline | - | 504,611 | 0.59 GiB |
| 512/8 | local | 16 | 2,784 | 0.70 GiB |
| 512/8 | local | 32 | 2,769 | 0.71 GiB |
| 512/8 | local | 64 | 2,684 | 0.72 GiB |
| 512/8 | local | 128 | 2,579 | 0.77 GiB |
| 512/8 | local | 256 | 2,778 | 0.94 GiB |
| 2048/8 | baseline | - | 466,105 | 2.03 GiB |
| 2048/8 | local | 64 | 2,787 | 2.55 GiB |
| 4096/1 | baseline | - | 303,649 | 0.60 GiB |
| 4096/1 | local | 64 | 425 | 0.72 GiB |
| 8192/1 | baseline | - | 219,634 | 1.08 GiB |
| 8192/1 | local | 64 | 414 | 1.33 GiB |

At the representative workload, memory-64 is 99.40% slower than baseline.
On the same chunked backend, fused projections give 2,684 tok/s versus 1,835
tok/s for separate projections at 512/8: a 46% throughput improvement.

A 128-token profiler sample recorded 5,391 `mm` calls and thousands of
elementwise launches. Adds/multiplies consumed about 45% of GPU time, all
`mm` operations 8.5%, vector normalization 7.5%, sigmoid 6.2%, and state
`bmm` 5.4%. Runtime is nearly flat from memory 16 to 256, confirming that the
dominant cost is sequential tokenwise MLP/control execution and launch
overhead rather than `d_memory^2` state arithmetic.

## Correctness, stability, and limitations

Reference and optimized outputs/final states and gradients match in tests.
Causality, reset, local isolation, residual propagation, overwrite/erase
semantics, configuration, serialization, cached generation, and compile
invocation are covered. A 16K-token BF16 recurrence remained finite for
memory dimensions 32, 64, 128, and 256; state RMS stayed between approximately
`1.5e-6` and `6.5e-6` without clipping or decay.

A tiny fixed-batch overfit reduced baseline loss from 3.46 to 0.075 and memory
loss from 3.47 to 0.076. Every memory parameter received a finite gradient and
the first read-gate logit moved from -4.00 to -3.43.

This implementation is semantically ready but not suitable for comparative
LLM training. Exact placement makes the write depend on an MLP whose input
depends on the prior memory read, so the MLP itself is recurrent across tokens.
No associative scan of the matrix update can parallelize that nonlinear
dependency. A practical architecture must move the write-producing MLP out of
the recurrence, lag writes, or accept approximate/chunk-delayed memory.
Learned delayed-recall and replacement gains were therefore not measured.
