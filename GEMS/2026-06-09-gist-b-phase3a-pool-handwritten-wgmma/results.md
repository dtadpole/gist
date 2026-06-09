# GEM — Phase 3a: hand-written persistent warp-spec WGMMA pool (CUTLASS parity)

**Date:** 2026-06-09
**Rev:** 97020 (harness run_h8_3_gist/v1/0_gist/rev_97020)
**Live source:** `/home/zhenc/gist-opt/dir-b-warpspec/gist.cu`

## What moved the number
Replaced the CUTLASS `run_pool` batched GEMM (`O[b]=SL[b]@N[b]`, per-batch M=Q=128,
N=D=192, K=Fp=1536, batch=B=1536) with a hand-written CUDA+inline-PTX persistent
warp-specialized WGMMA pool `k_pool_hand_persist` — the same structure as the proven
gate kernel `k_gate_hand_persist`. 132 persistent CTAs grid-stride over the 1536
batch-tiles; 3 warpgroups (producer tid>=256 issues TMA, 2 consumers each own 64
Q-rows); S=4 ring; continuous gk across batches; `wgmma.m64n192k16` (acc[96]/thread),
24 k-blocks; TMA-store [128x192] O tile per batch.

## Measured (real harness, bf16, B=1536/F=1497/D=192/Q=128)
- **Correctness: passed=true, max_abs_error=0.015625** (== the CUTLASS pool path exactly),
  mean_abs 0.000537. (harness_trial_gist-big.stdout.json)
- Full pipeline (hand pool): **3.629 ms** min (harness 27-run timed loop, L2 flush on).
- Pool isolated (direct-binary differential, /tmp/gist_env_1497.sh):
  - HAND pool: full ~3.634 − skip_pool ~2.879 = **~0.755 ms**
  - CUTLASS pool (same binary, no GIST_POOLHAND): full ~3.674 − skip ~2.879 = **~0.795 ms**
  - => hand pool is at CUTLASS parity and ~5% faster on the first correct version.
  - Baseline target was CUTLASS run_pool = 0.784 ms (NCU MMA_64x192x16, tile[128,192,64]).

## The validated K-major WGMMA recipe (the one risky/new piece)
A = SL `[B,Q,Fp]` row-major is **K-major** (trans-a=0); B = N `[B,Fp,D]` row-major is
**MN-major** (trans-b=1) — opposite of the gate's MN-major A. Validated standalone in
`microtest_m64n64_kmajor.cu` and `microtest_m64n192_kmajor.cu` (both max_err 0.0000):

- **Atom:** `cute::SM90::GMMA::MMA_64x192x16_F32BF16BF16_SS<Major::K, Major::MN, One, One>`
  (wrapped as `wgmma_cute_m64n192k16`, acc[96]).
- **A smem tile:** canonical K-major SW128, i.e. a plain [64x64] contiguous bf16 brick
  (== `tile_to_shape(Layout_K_SW128_Atom<bf16>, [64,64])`). K is the contiguous dim.
- **A descriptor:** `wg_desc_sw3(&As[ks*16], LBO=16, SBO=1024)` — per-k16 advance is
  **+16 bf16 elements** (K contiguous: 16 elem = 32 B = addr>>4 +2). layout_type=1 (128B).
- **B descriptor (3 contiguous 64-bricks for D=192):** `wg_desc_sw3(&Bs[ks*1024], LBO=8192,
  SBO=1024)` — per-k16 +1024 bf16; identical to the gate's MN-major B brick recipe.
- Derived the LBO/SBO from `cute::SM90::GMMA::make_gmma_desc<Major::K>` on per-k16 slices
  (split K into (16,4) via zipped_divide); the hand `wg_desc_sw3` reproduces cute exactly.

### TMA maps (launch_pool_hand)
- A (SL, K-major): globalDim `{Fp(fast), B*Q}`, box `{64,64}`, SW128; load 2 M-halves at
  `(k0, b*Q + ms*64)`. The batch is just an M-row offset (Q-rows contiguous in flat M).
- B (N, MN-major): globalDim `{D(fast), B*Fp}`, box `{64,64}`, SW128; load 3 D-bricks at
  `(n2*64, b*Fp + k0)`.
- O store: globalDim `{D(fast), B*Q}`, box `{192,128}`, **no swizzle**; store at `(0, b*Q)`.

## Build note
cute GMMA atoms require `-gencode arch=compute_90a,code=sm_90a` (NOT `-arch=sm_90a`,
which downgrades to sm_90 and rejects wgmma). Driver API needs `-lcuda`.

## Run to reproduce
```
cd /home/zhenc/gist/cuda
GIST_HANDGATE=1 GIST_PHASE1RAW=1 GIST_FUSESIG=1 GIST_POOLHAND=1 \
  GIST_CU=/home/zhenc/gist-opt/dir-b-warpspec/gist.cu CUDA_VISIBLE_DEVICES=1 ./run_gist.sh <rev> big
```

## What's left (optimization headroom)
First-correct version uses a per-tile drained epilogue (full `bar_sync_consumers` +
`tma_store_wait<0>` per batch), so the producer's cross-tile prefetch overlap is limited
vs the gate's chunked/pipelined epilogue. Already at parity; a chunked overlapped epilogue
(3x64-col chunks, double-buffered, like the gate's HGP_NCH path) could push further if NCU
shows per-tile tensor-idle.
