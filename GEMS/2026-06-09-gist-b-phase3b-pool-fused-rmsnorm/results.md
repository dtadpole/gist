# GIST Phase 3b — RMSNorm value-path N FUSED into the hand-written WGMMA pool

**Branch B · impl slug `gist-b-phase3b-pool-fused-rmsnorm` · 2026-06-09 · H100 (CUDA_VISIBLE_DEVICES=1).**
Design shape B=1536, F=1497, D=192, Q=128 (Fp=1536). Genuine bf16. RMSNorm oracle (MAIN e33f072).
Builds on [[gist-b-phase3a-pool-handwritten-wgmma]] (hand WGMMA pool reading materialized N).

## Result — PHASE 3b MET (N computed inline; stats no longer materializes N)
| metric | value |
|---|---|
| **total pipeline** | **3.29 ms** (stable 3.283–3.294; was 3.625 with 3a materialized-N pool) |
| **correctness** | **PASS, max_abs 0.015625, mean 5.4e-4** (== CUTLASS path) vs RMSNorm `ref-pytorch` |
| stats (k_stats_mr, M+R only) | 0.487 ms (was 0.863 with N-materialization → −0.376) |
| pool (fused, inline N) | 0.785 ms (3a materialized-N pool 0.756 → near parity) |
| pool NCU | DRAM 76% / 1.86 TB/s, tensor 22% (memory-bound, as intended) |

**vs the bars:** beats min-20% (3.32), MAIN best 3.398 (cuBLAS gate, vendor-lib), Triton/RMSNorm 4.152.
The whole gate (σ-fused) AND pool (N-fused) are now hand-written CUDA+PTX.

## Design — what changed vs 3a
`O[b] = SL[b] @ N[b]`, but **N = X·rrms·W is computed inline in the pool** (never materialized), mirroring the
Triton `_pool_kernel` (`n = x * rg[:,None] * w`). stats → `k_stats_mr` writes ONLY M (mean) + R (rrms), no N.
- Producer TMA-loads A=SL (2 bricks, K-major) + X (3 bricks, **3D TMA [D,F,B] OOB-fill 0** — X is unpadded F=1497,
  so a flat 2D TMA over Fp would cross batch boundaries; 3D bounds f to [0,F) per batch) + W (3 bricks, 2D [D,F]).
- Consumers compute `N[g,d] = X[g,d]·rrms[g]·W[g,d]` IN-PLACE over the X smem brick (X,W,N share the SW128 swizzle
  → same offset; rrms[g] per-K-row), then `wgmma.m64n192k16`. S=3 ring + chunked O-store (fits 224KB).

## The optimization ladder (each NCU-measured — evidence, not assumption)
1. **Naive fused (S=2, scalar per-bf16 N-compute):** pool 1.019 ms, total 3.48. NCU: DRAM 45% (was 87% in 3a).
2. **Deeper ring (S=3) + W-from-L2:** S=3 ≈ S=2 (1.004) → ring depth is NOT the limiter. W-from-L2 was a DISASTER
   (8.35 ms — scattered, latency-bound global reads in the compute; W must be smem-staged).
3. **Pipeline N-compute one k-block ahead of the wgmma:** no help (1.035) — the compute is LONGER than the wgmma,
   so the per-iter time is gated by the compute, not the overlap.
4. **NCU root-cause (overturned the occupancy theory):** 3a and fused have the SAME occupancy (warps_active 14%,
   1 CTA/SM); the fast 3a issues only 6.9% (memory-bound thrives at low issue). So non-persistent/more-occupancy
   would NOT help. The fused was slow because the scalar inline-N compute (per-bf16, 2-byte smem ops) **extends the
   consumer's per-buffer hold time** (full→compute→wgmma→empt), throttling the producer's TMA → DRAM 87%→45%.
5. **FIX = VECTORIZE the N-compute (uint4 / bf16x2):** process 8 bf16 per 16-byte chunk (`gh_sw_chunk`, all 8 share
   rrms[k]) → 8× fewer iterations, 16-byte coalesced smem → compute < wgmma → consumer wgmma-gated → **DRAM 45→76%,
   pool 1.0→0.785, total → 3.29 ms.**

## Key recipe
- N-compute, in the SW128 brick: iterate 16B chunks `C` → `(brick,k,mn8)`, `so = brick*TILE + gh_sw_chunk(k,mn8)`,
  `uint4 x = Xs[so]; uint4 w = Ws[so]`; 4× bf16x2: `N = X·rrms[k]·W`; `Xs[so] = N` (in place). g≥F: X,W TMA-OOB-0 → N=0.
- 3D TMA over X: globalDim {D,F,B}, strides {D·2, F·D·2} (both %16), box {64,64,1}, SW128, OOB-fill 0.
- rrms (R[B,F] fp32) loaded to smem `rsm[64]` per k-block (guard g<F). W TMA-staged (NOT read from L2 in compute).

## Repro
```bash
cd ~/gist/cuda
GIST_HANDGATE=1 GIST_PHASE1RAW=1 GIST_FUSESIG=1 GIST_POOLFUSE=1 GIST_CU=<this>/gist.cu CUDA_VISIBLE_DEVICES=1 ./run_gist.sh <rev> big
#   -> gen-cuda passed=true, max_abs 0.015625, total ~3.29ms
# component differential: direct binary + /tmp/gist_env_1497.sh, +GIST_SKIP_POOL etc.
```
Kernels: `k_stats_mr` (M,R) + gate `k_gate_hand_persist` (σ via tanh.approx) + `k_repad` + `k_pool_hand_fused`
(GIST_POOLFUSE). See [[gist-pool-handwritten-kmajor]] [[gist-sigmoid-gate-fusion-verdict]].
