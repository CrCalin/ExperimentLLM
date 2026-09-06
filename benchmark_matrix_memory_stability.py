#!/usr/bin/env python3
"""Run long normalized-address stability diagnostics for all BF16 paths."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import time

import torch

from experimentlm.matrix_memory import (
    BF16TriangularSolveUnavailable,
    chunked_matrix_memory,
    compact_wy_matrix_memory,
    fused_wy_matrix_memory,
    rank2_matrix_memory,
    sequential_matrix_memory,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lengths", type=int, nargs="+", default=[2048, 4096, 8192, 16384, 32768])
    parser.add_argument("--dimensions", type=int, nargs="+", default=[32, 64, 128, 256])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--output", default="benchmark_results/matrix_memory_stability.json")
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA/ROCm device requested but unavailable")
    dtype = {"bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]
    results = []
    print(f"device={device} dtype={args.dtype} torch={torch.__version__}")
    for length in args.lengths:
        for d_memory in args.dimensions:
            started = time.perf_counter()
            q, e, k, v = [torch.randn(1, length, d_memory, device=device, dtype=dtype) for _ in range(4)]
            e = e / e.norm(dim=-1, keepdim=True).clamp_min(1e-6) * 0.5
            k = k / k.norm(dim=-1, keepdim=True).clamp_min(1e-6) * 0.5
            reads, final = sequential_matrix_memory(q, e, k, v)
            wy_reads, wy_final = compact_wy_matrix_memory(q, e, k, v, chunk_size=128)
            fused_reads, fused_final = fused_wy_matrix_memory(q, e, k, v, chunk_size=128)
            rank2_reads, rank2_final = rank2_matrix_memory(q, e, k, v, chunk_size=128)
            row = {
                "length": length, "d_memory": d_memory, "dtype": args.dtype,
                "backend": "reference", "status": "ok",
                "state_nan": bool(torch.isnan(final).any()),
                "state_inf": bool(torch.isinf(final).any()),
                "read_nan": bool(torch.isnan(reads).any()),
                "read_inf": bool(torch.isinf(reads).any()),
                "nan": bool(torch.isnan(final).any() or torch.isnan(reads).any()),
                "inf": bool(torch.isinf(final).any() or torch.isinf(reads).any()),
                "state_rms": float(final.float().square().mean().sqrt()),
                "state_max_abs": float(final.float().abs().max()),
                "read_rms": float(reads.float().square().mean().sqrt()),
                "final_state_norm": float(final.float().norm()),
                "wy_state_rms": float(wy_final.float().square().mean().sqrt()),
                "wy_state_max_abs": float(wy_final.float().abs().max()),
                "wy_read_rms": float(wy_reads.float().square().mean().sqrt()),
                "wy_state_diff_rms": float((wy_final.float() - final.float()).square().mean().sqrt()),
                "wy_read_diff_rms": float((wy_reads.float() - reads.float()).square().mean().sqrt()),
                "rank2_state_rms": float(rank2_final.float().square().mean().sqrt()),
                "rank2_state_max_abs": float(rank2_final.float().abs().max()),
                "rank2_read_rms": float(rank2_reads.float().square().mean().sqrt()),
                "rank2_state_diff_rms": float((rank2_final.float() - final.float()).square().mean().sqrt()),
                "rank2_read_diff_rms": float((rank2_reads.float() - reads.float()).square().mean().sqrt()),
                "fused_state_rms": float(fused_final.float().square().mean().sqrt()),
                "fused_state_max_abs": float(fused_final.float().abs().max()),
                "fused_read_rms": float(fused_reads.float().square().mean().sqrt()),
                "fused_state_diff_rms": float((fused_final.float() - final.float()).square().mean().sqrt()),
                "fused_read_diff_rms": float((fused_reads.float() - reads.float()).square().mean().sqrt()),
                "seconds": time.perf_counter() - started,
                "chunked_status": "not_tested",
                "wy_status": "ok",
                "rank2_status": "ok",
                "fused_status": "ok",
            }
            if not torch.isfinite(rank2_reads).all() or not torch.isfinite(rank2_final).all():
                row["rank2_status"] = "nonfinite"
            if not torch.isfinite(fused_reads).all() or not torch.isfinite(fused_final).all():
                row["fused_status"] = "nonfinite"
            try:
                chunked_matrix_memory(q, e, k, v, chunk_size=64)
            except BF16TriangularSolveUnavailable as exc:
                row["chunked_status"] = "unsupported_bf16_triangular_solve"
                row["chunked_error"] = str(exc)
            else:
                row["chunked_status"] = "available"
            results.append(row)
            print(json.dumps(row, separators=(",", ":")), flush=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({
        "environment": {"torch": torch.__version__, "platform": platform.platform()},
        "config": vars(args), "results": results,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"Saved to {output}")


if __name__ == "__main__":
    main()
