# GIST goal — refined bar (updated 2026-06-07T18:55Z)

Authoritative intent lives in ~/.hermes/goals/gist/INTENT.md (append-only, latest wins).
This file mirrors the current bar for the AI Coder working in ~/gist.

## Deliverable
A **CUDA + inline PTX** GIST forward kernel, compiled arch **sm_90a** (Hopper), wired
through the cuda_exec harness, that BEATS `gist_triton_fast` (4.22 ms bf16 at design
shape B=1536,F=1491,D=192,Q=128) by a large margin.

## Performance target
- Reference to beat: gist_triton_fast = 4.22 ms bf16.
- Minimum acceptable: < 3.38 ms bf16 (20% faster).
- **TARGET / success: < 2.95 ms bf16 (30%+ faster).**
- Apples-to-apples: same shape, same bf16 precision, CUDA-event timing, real
  correctness within tolerance, measured margin.

## Inline PTX is part of the deliverable
Hot paths should use PTX where it buys speed: wgmma/mma tensor-core gate GEMM,
cp.async tile staging, vectorized 128-bit ld/st, ld.global.nc, fast cvt. The gate
GEMM is the main lever — Triton hits ~322 TFLOP/s (~40% of peak); wgmma+TMA+warp
specialization is the headroom. Avoid the 3D pool accumulator (wrecks occupancy).

## HARD RULE — NO REWARD HACKING
The latency number must come from REAL work on the REAL problem. Do NOT:
- shorten/skip the timing loop, cut iterations, remove warmup
- move work outside the timed region, or precompute/cache outputs
- shrink the design shape (B/F/D/Q)
- weaken the correctness tolerance to pass a wrong-but-fast kernel
- dump golden/cached outputs instead of real kernel outputs
- special-case the input seed (0xCAFE0000+j / idx+1) or hardcode expected results
- disable L2 flush or otherwise make timing unrealistically favorable
A fast number from a gamed harness is a FAILURE. The supervisor audits this every cycle.

## Evidence
Head-to-head bench (CUDA+PTX vs Triton-fast vs torch.compile) → ~/.hermes/goals/gist/evidence/,
with which PTX ops actually moved the number.
