# Standalone recurrent matrix memory

`experimentlm.matrix_memory` implements only the low-dimensional recurrence;
it is not connected to the Transformer, trainer, checkpoint format, or model
registry. The API consumes `q`, `e`, `k`, and `v` with shape
`[B, T, d_memory]`, plus an optional state with shape `[B, d_memory,
d_memory]`, and returns causal reads and the final state.

The reference backend performs, for every token, `read -> erase -> corrective
write`. The FP32 `chunked` backend interleaves `(e, 0)` and `(k, v)` and uses
the triangular solve as an oracle. The BF16 `wy` backend uses the same
interleaving and the following exact compact-WY/UT factorization; the BF16
`rank2` backend is described below.
The `fused_wy` path additionally uses a sequence-level analytic adjoint and
ROCm Triton kernels for the training output contractions. It remains an exact
doubled-WY implementation and does not change the recurrence.

```text
(I + tril(U @ U.T, -1)) @ P = Z - U @ initial_state.T
```

Let `L = 2C`, `G = U @ U.T`, `L_G = tril(G, -1)`, and `N = L_G.T`. The
doubled transition product is represented as

```text
H = prod_j (I - u_j u_j.T) = I - U.T @ T @ U
T = (I + N)^-1
```

Rather than calling a triangular solve, `T` is formed with the finite
nilpotent identity

```text
(I + N)^-1 = (I - N)(I + N^2)(I + N^4)...
```

where each power/factor is computed in the input dtype with GEMMs. Innovations
are `P = T.T @ (Z - U @ M_in.T)`, then `M_out = M_in + P.T @ U`. Causal reads
remain `Q @ M_in.T + tril_causal(Q @ U.T) @ P`, so each token excludes both
of its own doubled substeps. Only chunk boundaries are sequential; there is
no `d_model` projection, attention, MLP, normalization, or other
large-dimensional work.

## Exact rank-2/block-WY backend

`rank2_matrix_memory` is an additional exact realization of the same
recurrence. For each actual token, let `c=e.T @ k` and define

```text
U = [e, k]
V = [e - c*k, k]
Z = [0, v]
A = (I - e e.T)(I - k k.T) = I - U V.T
P = Z - M U
M_next = M + P V.T
```

Thus the rank-2 update is exactly erase followed by corrective write, with
independent addresses. For a C-token chunk, U/V/Z are horizontal `[B,d,2C]`
matrices, interleaved by token. The block coupling matrix has zero diagonal
2x2 blocks and off-diagonal block `N[i,j] = V_i.T @ U_j` only for `i < j`:

```text
P (I + N) = Z - M_in U
P = (Z - M_in U) F,  F = (I + N)^-1
M_out = M_in + P V.T
```

`F` uses the finite binary nilpotent identity
`(I-N)(I+N^2)(I+N^4)...`, with same-dtype GEMMs and no BF16 triangular
solve. Causal reads are `q @ M_in.T + mask(q @ V) @ P.T`; both columns of
the current token are masked. Only the final `[B,d,d]` state crosses chunk
boundaries. The custom backward saves q, U, V, the incoming state, F, and P,
not tokenwise d-by-d states or the transient power chain.

The algebraic identities, special address cases, causality, final state, and
gradients are covered by `tests/test_matrix_memory.py`. Production q/e/k/v,
state, factors, saved tensors, reads, and final state remain BF16.

## Verification

Run the focused suite with:

```bash
./.venv/bin/python -m unittest tests.test_matrix_memory -v
```

The tests cover direct semantics, all requested edge-ordering cases, causality,
batch isolation, non-divisible chunks, chunk sizes 8–128, FP32 gradients, and
dtype preservation. The long BF16 stability grid is opt-in:

```bash
MATRIX_MEMORY_FULL_TESTS=1 ./.venv/bin/python -m unittest tests.test_matrix_memory -v
```

## BF16 implementation

On the current environment (`torch 2.14.0+rocm7.14`, RX 7900 XTX),
`torch.linalg.solve_triangular` raises `NotImplementedError` for BF16 on both
CPU and ROCm. The `chunked` oracle therefore remains FP32-only and raises
`BF16TriangularSolveUnavailable` for BF16. The `wy` backend avoids that
primitive: q/e/k/v, state, compact factors, innovations, reads, and final
state remain BF16. No FP32 state or scalar recurrence reduction is used.

## Benchmark and profiler

The JSON benchmark specification is
`configs/matrix_memory_benchmark.json`. The default sweep contains the
requested batch sizes, sequence lengths, memory dimensions, and chunk sizes.
Use FP32 for diagnostic chunked timings on the current ROCm build:

```bash
./.venv/bin/python benchmark_matrix_memory.py \
  --dtype float32 --no-backward \
  --batches 1 8 --lengths 512 2048 \
  --dimensions 16 32 64 128 256 --chunks 16 32 64 128 \
  --output benchmark_results/matrix_memory_fp32.json
```

The default BF16 run records triangular-oracle unsupported cases explicitly and
benchmarks `wy` and `rank2`. `--compile`
tests `torch.compile` for both optimized backends and records compile time and
Dynamo graph-break counters separately. `--profile` records the named stages
(`U/Z`, Gram, triangular solve, causal products, masked reads, final-state
update) through `torch.profiler`; with `--backward`, its profiled wall time
includes autograd. Results are written as raw JSON, including failed or
unsupported cases, rather than being silently omitted.

The checked-in diagnostic captures were made on the RX 7900 XTX:

| workload | reference | optimized | speedup |
|---|---:|---:|---:|
| B=8, T=2048, d=64, BF16 forward | 232,154 tok/s | 3,043,380 tok/s (WY, C=128) | 13.1x |
| B=8, T=2048, d=64, BF16 forward+backward | 35,063 tok/s | 1,443,494 tok/s (WY, C=128) | 41.2x |
| B=8, T=2048, d=64, FP32 forward+backward | 32,947 tok/s | 525,106 tok/s (triangular, C=128) | 15.9x |

At B=1/T=512, WY/C=128 was fastest for every tested dimension: 358K,
339K, 372K, and 353K tok/s for d=32, 64, 128, and 256 respectively. The
representative WY profile is stored in the raw results; compact-WY factor
construction is the dominant measured device stage at about 10.4 ms of
aggregate profiler time, followed by causal-read products and masking. There
is no recurrent token loop in the optimized chunk body.

In the focused BF16 B=8/d=64/C=128 forward+backward sequence sweep, WY reached
1.19M, 1.41M, 1.43M, and 1.42M tok/s at T=512, 2048, 4096, and 8192. The
roughly flat long-sequence rate indicates that chunk-boundary serialization is
not growing quadratically with T; short sequences mainly pay fixed launch and
factor-construction overhead.

`torch.compile` on the representative static B=8/T=2048/d=64/C=128 WY case
took 8.88 s to compile, had zero graph breaks, and reached 4.257M tok/s in
steady-state forward. A BF16 compiled backward probe exceeded the practical
30 s compile window and was stopped; no compiled-backward throughput claim is
made. The BF16 stability capture completed for every combination of
T=2K/4K/8K/16K/32K and d=32/64/128/256; all states and reads were finite, with
state RMS approximately 0.51–0.55 and state max absolute value at most 2.67.
The WY path matched the BF16 reference within the documented tolerance.

The rank-2 stability capture also completed all 20 requested combinations
(T=2K/4K/8K/16K/32K and d=32/64/128/256), with finite reads and states and
near-identical RMS/max norms and BF16-reference divergence. Raw captures are
stored in `benchmark_results/matrix_memory_rank2_stability.json` and the
rank-2 scaling JSON files.

On the same RX 7900 XTX harness, the canonical rank-2 measurements were:

| workload | doubled-WY C=128 | rank-2 C=128 | rank-2 / WY |
|---|---:|---:|---:|
| B=8, T=2048, d=64, BF16 forward | 2.960M tok/s | 2.432M tok/s | 0.822x |
| B=8, T=2048, d=64, BF16 forward+backward | 1.400M tok/s | 1.049M tok/s | 0.749x |

Rank-2 was slower at all eager tested dimensions and contexts at C=128. Its
extra cost is concentrated in token-factor and block-Gram construction; the
nilpotent inverse itself was similar to doubled-WY. It saved 34.67 MB at the
canonical workload versus 30.47 MB for doubled-WY custom-backward tensors, and
peak VRAM was 0.117 GiB versus 0.114 GiB. Both paths used 16 chunk invocations
and rank-2 did not reduce the compact column count: it still has 2C columns.
Static compiled forward tied the old path at about 4.31M tok/s with zero graph
breaks, but compiled backward did not finish its 60-second probe. Rank-2 is
therefore retained as an exact semantic/fusion candidate, but should not
replace doubled-WY as the primary eager performance backend until factor
construction and backward bookkeeping are optimized.
The profiler's device-event proxy counted 7,428 rank-2 versus 4,580 doubled-WY
events for the same canonical backward run. This is an operator-event proxy,
not a hardware kernel trace: the current pure-PyTorch rank-2 implementation
saves no launches and adds orchestration events, so fusion is required to
realize its structural advantage.

## ROCm fused-WY implementation

`fused_wy_matrix_memory` is a sequence-level custom operator. Forward saves
compact factors once per chunk; backward runs the analytic adjoint in reverse
chunk order, eliminating one nested autograd node per chunk. The finite
nilpotent inverse also omits its provably redundant terminal identity-power
product.

On BF16 ROCm training, Triton writes doubled U/Z and fuses causal-read
accumulation with the final state contraction. Dense compact Gram, binary
factor, innovation, and backward factor products remain vendor BF16 GEMMs;
replacing those is the remaining major kernel opportunity. The fused backend
always exercises its ROCm kernels on CUDA BF16 inputs; the validated `wy`
backend remains the vendor-GEMM comparator.

The kernels execute natively on the RX 7900 XTX. They write BF16 outputs and
do not create persistent FP32 state. The Triton output kernel uses temporary
higher-precision contraction accumulators before BF16 output rounding; this is
not persistent state and is explicitly documented rather than silently called
pure BF16 arithmetic.

Final canonical measurement (`torch 2.14.0+rocm7.14`, HIP 7.14, RX 7900 XTX,
B=8/T=2048/d=64) was:

| path | forward | forward+backward | saved tensors | peak VRAM |
|---|---:|---:|---:|---:|
| doubled-WY comparator, C=128 | 3.091M tok/s | 1.480M tok/s | 30.47 MB | 0.114 GiB |
| fused-WY, C=128 | 2.836M tok/s | 1.636M tok/s | 30.47 MB | 0.116 GiB |

The fused training path is 1.106x the updated doubled-WY comparator and
1.134x the original 1.44M tok/s reference number. C=128 was fastest; C=256
regressed because its larger working set outweighed fewer chunk boundaries.

The final profile's aggregate device stages were approximately 8.87 ms for
compact factor construction, 6.99 ms for analytic factor backward, 4.84 ms
for backward read contractions, and 1.93 ms for fused forward outputs. The
old event-count value is only an operator-event proxy, not a hardware launch
count: it changed from 4,404 to 3,151 in the corresponding profile. Actual
custom launches are one U/Z launch plus two output launches per chunk; dense
GEMM launches remain.

At B=8/C=128, fused forward was approximately
2.64M/2.36M/2.23M/1.63M tok/s for d=32/64/128/256 at T=2048, and
2.93M/2.86M/2.28M/1.55M at T=8192. Fused forward+backward was
1.58M/1.62M/1.45M/1.06M tok/s for d=32/64/128/256 at T=2048, and
1.65M/1.63M/1.49M/1.07M at T=8192. Context scaling is linear and the
large-d forward path finally exposes the expected arithmetic scaling.

Compiled forward reached 3.19M tok/s with zero graph breaks and about 8.64 s
compile time for the static canonical case when the Triton output kernels were
forced. Eager training remains the reliable ROCm production measurement for the
custom Function/Triton path.

This is materially faster, but not every dense factor operation is fused. The
approximately 3 ms per-layer economic target still requires a specialized
factor-generation/backward kernel.

The instrumented rank-2 profile's aggregate device-time leaders were the
nilpotent inverse (10.18 ms), backward factor path (7.25 ms), token-factor
construction (3.60 ms), block Gram (3.42 ms), and backward read path (4.89 ms).
The comparable doubled-WY forward factor was 9.84 ms, while its construction
and Gram stages were 0.82 ms and 0.66 ms. This identifies rank-2 orchestration
and backward contractions—not mathematical rank—as the immediate optimization
target. The full named-stage profile is in
`benchmark_results/matrix_memory_rank2_profile_backward_instrumented.json`.

Autograd retains compact chunk intermediates rather than tokenwise `d x d`
states. The forward binary power chain is transient; the custom backward saves
`q`, interleaved `U`, incoming chunk states, compact factors, and innovations.
For B=8/T=2048/d=64/C=128 this measured 30.47 MB of saved tensors and 0.114
GiB peak VRAM, versus 419.50 MB and 0.338 GiB for the tokenwise reference
graph. The benchmark JSON records the measured footprint for every backward
case.

## Integration decision

The algebra is exactly chunkwise-parallelizable in BF16 and has no hidden
`d_model` serialization. The fused-WY path is correct and ROCm-native and is
the preferred standalone training path on this machine, but its improvement is
not enough to establish per-layer integration viability: factor construction
and factor backward still dominate. Transformer integration is intentionally
not included here. The next meaningful work is a fused factor/backward kernel.

Final fused scaling captures are in
`benchmark_results/matrix_memory_fused_final_scaling_forward.json` and
`matrix_memory_fused_final_scaling_backward.json`. Context scaling was linear;
the result remains above the per-layer economic target.
