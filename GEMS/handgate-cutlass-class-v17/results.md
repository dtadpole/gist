# Hand-written CUDA+PTX GIST gate — CUTLASS-class (v17, branch B)

**Date:** 2026-06-08 · **Branch:** B (warp-spec) · **Source:** `gist.cu` (this dir) ·
`GIST_HANDGATE` path. Design shape B=1536, F=1491, D=192, Q=128 (QF=190848). bf16, genuine.

## Result (measured, harness CUDA-event timing, L2-flush, fresh revs)
| metric | value | vs |
|---|---|---|
| **gate kernel (isolated)** | **~1.36 ms** | CUTLASS gate 1.26 ms → within **8%** (was 1.93 ms = 1.5×) |
| total pipeline (stats+gate+sigpad+pool) | **3.775 ms** | beats prior global best 4.014 ms by **6%**; Triton 4.225 by **11%** |
| correctness | PASS max_abs 0.03125, mean 3.2e-6 | full-O verified (raw gate → sigpad → CUTLASS pool) |
| NCU `sm__pipe_tensor_op_hmma` | **72.5%** | == CUTLASS MFU (~70% useful after 1.03× K-pad) |
| NCU Compute(SM) throughput | 72.6% | up from 47.7% pre-epilogue-fix |

Gate isolated via differential: (stats+gate, SKIP_POOL)=2.21ms − (stats-only)=0.847ms = **1.36ms**.

## What moved the number (each MEASURED, NCU-grounded)
Baseline v14 = gate 1.93 ms (TMA + B128 swizzle + 3-WG warp-spec + deep pipe, scalar epilogue).
1. **m64n256 wgmma atom** (1 issue/k16 vs 4× m64n64): NEUTRAL (1.93→1.93). Proves the gate is
   NOT wgmma-issue-bound. Recipe: B = 4 contiguous 64×64 canonical-SW128 bricks, desc LBO=8192
   (brick stride), SBO=1024; A desc LBO=128, SBO=1024 (validated /tmp/swz256.cu, max_err 0).
   acc[128] layout is bit-identical to 4 concatenated m64n64 acc[32] (g=n2*8+a) → epilogue unchanged.
2. **Staged + coalesced uint4 epilogue** (THE big win, 1.93→1.45 ms): the scalar 2-byte global
   stores were 41% of stall samples (lg_throttle), pinning SM at 47.7% (branch-C finding). Fix:
   stage acc→smem (reuse As/Bs), `bar.sync` over the 256 consumers (producer WG has exited →
   can't `__syncthreads`), then cooperative 16-byte `uint4` coalesced stores to L. → SM 72.6%.
3. **Bank-pad the staging stride 256→264** (1.45→1.36 ms): without pad the 4-byte acc→smem write
   has bank=(r0*128+c0/2)%32 with r0*128%32==0 → 8-way conflict; stride 264 spreads the 32 lanes
   to 32 distinct banks.

## Measured DEAD-ENDS (kept for the record)
- **wgmma software pipeline** (warpgroup_wait<1> + delayed empt): cut stalled_barrier 4.82→2.84
  cyc/instr but latency UNCHANGED → the kernel is tensor-pipe-throughput-bound, stalls are off the
  critical path. Reverted (simpler wait<0> is identical perf).
- **K-tail trim** (skip the all-zero k16 steps in the last tile, F=1491→24×64): REGRESSED to 1.56ms
  because the variable loop bound kills `#pragma unroll`. The 2% FLOP saved < the unroll loss.

## Why not the 2-CTA cluster (the earlier-suspected lever)
Reached CUTLASS-class tensor-pipe MFU (72.5%) with **ClusterShape<1,1,1>** (no cluster, no
multicast). Branch C independently measured that CUTLASS's own skinny-K gate also uses
ClusterShape<1,1,1>. The cluster is empirically not the lever for this shape.

## Remaining 8% to CUTLASS 1.26 ms
Pure skinny-K tensor-idle: the per-CTA epilogue + pipeline fill/drain leave the tensor pipe idle
27.5% (no wgmma during the epilogue of a non-persistent CTA). CUTLASS hides this with a
**persistent kernel** that overlaps tile N's epilogue with tile N+1's mainloop (+ swizzled tile
rasterization for L2 reuse). That is a large separate build; prior naive-persistent attempts
regressed (L2-reuse destruction) and need the rasterizer.

## Repro
```
cd ~/gist/cuda
GIST_HANDGATE=1 GIST_CU=<this>/gist.cu CUDA_VISIBLE_DEVICES=1 ./run_gist.sh <freshrev> big
# gate differential: add GIST_SKIP_POOL=1 (stats+gate) and GIST_SKIP_GATE=1 GIST_SKIP_POOL=1 (stats)
```
NCU (binary self-runs with /tmp/gist_env.sh harness env):
`ncu --kernel-name regex:k_gate_hand_b --launch-skip 2 --launch-count 1 --section SpeedOfLight <bin>`
