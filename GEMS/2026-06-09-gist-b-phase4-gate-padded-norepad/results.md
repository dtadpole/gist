# GIST Phase 4 — padded gate (fuse F→Fp pad into the gate; delete the repad kernel)

**Branch B · `gist-b-phase4-gate-padded-norepad` · 2026-06-09 · H100.** B=1536, F=1497, Fp=1536, D=192, Q=128,
bf16, RMSNorm. Builds on [[gist-b-phase3c-stats-vec-int4]] (vectorized stats). Env: `GIST_GATEPAD=1`.

## Result — PHASE 4 MET (repad eliminated → beats the 30% bar)
| metric | value |
|---|---|
| **total pipeline** | **2.64 ms** (stable 2.642–2.645; was 3.23) |
| **correctness** | **PASS, max_abs 0.015625, mean 5.4e-4** vs RMSNorm `ref-pytorch` |
| gate+σ+pad | 1.374 ms (was gate 1.237 + **repad 0.776** = 2.013 → **−0.64**) |
| stats / pool | 0.425 / 0.79 ms |

**vs bars:** beats **30%-bar 2.91**, 20%-bar 3.32, MAIN best 3.398 (cuBLAS), Triton 4.152. Whole gate+pool hand-PTX.

## What changed (the padding fusion — NO remap)
Before: gate wrote flat raw `g_L[B,QF]`; a separate `k_repad` zero-padded it to `g_SL[B,Q,Fp]` (0.776 ms pure
copy — needed because the pool's SL TMA requires the F→Fp=1536 aligned stride; odd F=1497 stride*2=2994 is not %16).
Phase 4: **the gate writes `g_SL[B,Q,Fp]` directly**, σ fused (tanh.approx) in the epilogue:
- **Pre-pad P → Pp[F, Q·Fp]** (`k_ppad`, one-time — the harness holds the weight P constant, so it's cached on the
  P pointer and amortized to ~0). Pp's g≥F columns are 0.
- Gate computes `M @ Pp` → output col n' = q·Fp + g. Because **Fp=1536 is a multiple of the 256 tile and the 64-col
  store chunk, every tile lies within one q and the chunked TMA-store to g_SL is 16-byte aligned ⇒ NO remap**
  (branch C's `win_shift` byte-shuffle is unnecessary — that was only needed for the flat-P layout).
- **Pad mask:** Pp's pad cols are 0 → logit 0 → σ(0)=0.5 ≠ 0, so the epilogue writes **0** (not σ) for cols
  g = (n0 % Fp) + c·HGP_CW + lc ≥ F. (only the gtile=1280 tiles per q have any pad: g 1497..1535.)
- Cost = just **~3% extra compute** (N-dim QF 191616 → Q·Fp 196608) + the pad mask. gate 1.237→1.374; repad gone.

## Key implementation
- `k_gate_hand_persist(..., int dosig, int padfp)`: padfp=0 = flat g_L (Phase 1/2s); padfp=Fp = padded g_SL
  (Phase 4, mask g≥F→0). Same persistent warp-spec WGMMA + tanh-σ epilogue; only the B-map (Pp), store-map (g_SL),
  N-dim (Q·Fp), and the pad-mask differ — all via launcher wiring (`s_tmBp`, `s_tmSL`, `k_ppad`, no `k_repad`).
- Reuses the existing g_Ppad/s_tmBp/s_tmSL infra (originally for the abandoned fused gate); k_ppad cached on P ptr.

## Pipeline now (5 hand-written CUDA+PTX kernels, RMSNorm/F=1497)
stats (vec int4) 0.425 + **gate+σ+pad 1.374** + pool (fused N) 0.79 = **2.64 ms**. The repad kernel is deleted.

## Repro
```bash
cd ~/gist/cuda
GIST_HANDGATE=1 GIST_GATEPAD=1 GIST_FUSESIG=1 GIST_POOLFUSE=1 GIST_CU=<this>/gist.cu CUDA_VISIBLE_DEVICES=1 ./run_gist.sh <rev> big
#   -> passed=true, max_abs 0.015625, total ~2.64ms
```
Kernels: k_stats_mr (vec) + k_gate_hand_persist(padfp=Fp, σ tanh) + k_pool_hand_fused (GIST_POOLFUSE).
See [[gist-pool-handwritten-kmajor]] [[gist-sigmoid-gate-fusion-verdict]].
