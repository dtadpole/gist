# GIST Phase 4b — drop the gate pad-mask (redundant; pool already zeros N[g≥F])

**Branch B · `gist-b-phase4b-nopadmask-drop` · 2026-06-09 · H100 (CUDA_VISIBLE_DEVICES=4).** B=1536, F=1497,
Fp=1536, D=192, Q=128, bf16, RMSNorm. Builds on [[gist-b-phase4-gate-padded-norepad]]. Env: `GIST_NOPADMASK=1`.
Convergent with branch C's global-best find (credit: aggregator heads-up).

## Result — NEW BRANCH-B BEST, bit-identical correctness
| variant | min latency | correctness |
|---|---|---|
| Phase 4 (pad-mask ON) | 2.654 ms | PASS, max_abs 0.015625, mean_abs 5.366e-4 |
| **Phase 4b (pad-mask OFF)** | **2.517 ms** | PASS, max_abs 0.015625, **mean_abs 5.366e-4 (identical)** |

Δ = **−0.137 ms (−5.2%)**. Same A/B build, same GPU, same harness run; only `GIST_NOPADMASK` differs.
**40% faster than Triton (4.15 ms)**, beats the 30% bar (2.91), 20% bar (3.32), MAIN cuBLAS (3.40).
(A/B on GPU4: Phase 4 measured 2.654 here vs the 2.64 originally on GPU1 — GPU/run variance; the −0.137 Δ is the signal.)

## Why it's correct (mean_abs is byte-identical, not just "close")
The padded gate writes `g_SL[B,Q,Fp]`; for pad cols `g∈[F,Fp)` the prepacked `Pp` is 0 → logit 0 → σ(0)=**0.5**.
Phase 4 masked those to 0 in the epilogue. But the **fused pool** builds `N[b,g,d]=X·rrms·W` with X **and** W loaded by
**3D/2D TMA that OOB-fills 0** for `g≥F` (gist.cu `tma_load_3d(...&tmX...) // X[D,F,B] OOB-fill 0`, and
`tma_load_2d(...&tmW...) // W[D,F] OOB-fill 0`). So `N[g≥F]=0`, and the pool accumulates
`O += SL[pad]·N[pad] = 0.5·0 = 0` **regardless of SL[pad]**. The mask was therefore pure redundant epilogue work —
removing it cannot change O, and indeed `mean_abs_error` is identical to the bit (0.0005366427358239889).

## What changed (1-line of real work removed from the hot epilogue)
`k_gate_hand_persist(..., int padfp, int padmask=1)`: the `if (padfp && padmask)` block that zeroed the `g≥F`
lanes in the chunked σ-epilogue is skipped when `padmask=0`. Plumbed via `GIST_NOPADMASK` at the GATEPAD launch.
The saving is the per-chunk branch + the 4 extra `g0>=F` compares × 24 k-free epilogue lanes per tile — small per
element but on the tensor-bound gate's critical epilogue path, so it lands as −0.14 ms.

## Pipeline now (RMSNorm/F=1497)
stats (vec int4) 0.425 + **gate+σ+pad, NO mask ~1.24** + pool (fused N) 0.79 = **2.52 ms**.

## Repro
```bash
cd ~/gist/cuda
GIST_HANDGATE=1 GIST_GATEPAD=1 GIST_FUSESIG=1 GIST_POOLFUSE=1 GIST_NOPADMASK=1 \
  GIST_CU=<this>/gist.cu CUDA_VISIBLE_DEVICES=4 ./run_gist.sh <rev> big
#   -> passed=true, max_abs 0.015625, total min ~2.52ms
# A/B: drop GIST_NOPADMASK to get the 2.65ms mask-on baseline (identical correctness).
```
See [[gist-b-phase4-gate-padded-norepad]] [[gist-pool-handwritten-kmajor]]. Next lever: epilogue-overlap (EWG) on
top of this — could stack below 2.52.
