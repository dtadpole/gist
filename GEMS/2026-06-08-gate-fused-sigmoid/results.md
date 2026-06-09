# GEM — gate-fused sigmoid + repad-only → BEATS Triton

**Measured: 4.014 ms bf16** (harness L2-flush, design shape B=1536,F=1491,D=192,Q=128), correctness
**PASSED** (max_abs 0.03125 = bf16 floor, 0/1536 batches in error).

- vs Triton `gist_triton_fast` 4.225 ms: **5.0% FASTER** ✓ (first config to beat Triton)
- vs prior best correct (CUTLASS baseline 4.375 ms): **8.2% faster**
- vs <2.95 ms target: still above; more to do.

## What moved the number
The CUTLASS baseline ran a separate `k_sigpad` kernel that did **sigmoid + repad**. NCU showed the
sigmoid there is **compute-bound: ~293M `__expf` ≈ 0.5 ms** (not memory). Fix:
1. **Fuse sigmoid into the gate epilogue** via `cutlass::epilogue::fusion::LinCombEltAct<thread::Sigmoid,...>`
   (needs the EXPLICIT `cutlass::epilogue::TmaWarpSpecializedCooperative` epilogue schedule +
   `cutlass::gemm::KernelTmaWarpSpecializedCooperative` mainloop, NOT EpilogueScheduleAuto). The exp now
   overlaps the gate's memory-bound mainloop → effectively free.
2. The downstream pass becomes a **pure repad** (`k_repad`, no exp): SL[b,q,f<F]=L[b,q,f], f>=F→0.
   Memory-only (~0.35 ms vs the old ~0.7-0.9 ms sigpad).

Net: gate time ~unchanged, the ~0.5 ms exp deleted from the critical path → 4.375 → 4.014 ms.

## Pipeline (default path in gist.cu, no env flags)
`k_stats` (M,R,N_pad) → `run_gate` (CUTLASS WGMMA, **sigmoid fused**) → `k_repad` (no exp) → `run_pool`
(CUTLASS batched bf16). All bf16, fp32 accumulate.

## Fairness check (apples-to-apples)
Triton `gist_triton_fast` at the design shape, bf16, `triton.testing.do_bench` (flushes L2 by default):
**4.185 ms** (`/tmp/bench_triton.py`). So the 4.22 bar is a genuine L2-flushed number, not a no-flush
artifact — this kernel's 4.01 ms harness number beats it by ~4-5% under matched (L2-flush) conditions.

## Why NOT "far exceed" (≥20%) — exhaustively measured, ~25 experiments this session
- Gate (60% of runtime) at 360 TFLOP/s = 36% peak = the shape's inherent ceiling; Triton's gate is
  322 TFLOP/s (~33%) — same ceiling, so the dominant component can't beat Triton. (tested all
  tiles/K-tiles/schedules/clusters/raster/cuBLAS/L2-slicing.)
- The only structural win (Triton's no-repad + inline-N) needs a fused pool: hand-WGMMA fused = 9.45 ms
  (acc[96] occupancy wall; acc<96 = ptxas warp-3 bug); WMMA fused = 1.9 ms (register wall); CUTLASS
  unpadded = compile-blocked (SM90 needs TMA Align-8, F=1491 odd); N-overlap-with-gate = 4.55 ms (SM
  contention, gate is latency- not BW-bound); PADP = 4.36 ms (per-call P-pad). All measured-blocked.
- 20% would need the ptxas WGMMA small-accumulator bug fixed (compiler-level, not kernel source).

## Reproduce
`cd ~/gist/cuda && ./run_gist.sh <fresh_rev> big | python3 parse_min.py`  → min_ms≈4.01, passed=True.
