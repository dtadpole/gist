# Phase 1 — gate GEMM measurements (hand CUDA+PTX vs CUTLASS)

Design shape **B=1536, F=1491, D=192, Q=128, bf16**; GPU2 (H100); cuda_exec L2‑flush harness
(CUDA‑event timing, time‑based warmup, L2 flush per trial). The gate GEMM is `L = M @ P`,
**raw logits, no sigmoid** (`k_gate_ws`, FUSE=false, env `GIST_GATE_WS`).

## Headline: CUTLASS parity

| metric | hand `k_gate_ws` | CUTLASS `run_gate_ns` | source |
|---|---|---|---|
| **warm, in‑pipeline** | **1.2745 ms** | ~1.26 ms | SKIP‑differential (below) / nsys breakdown |
| **NCU cold duration** | **1.66 ms** | **1.66 ms** | NCU `--set detailed`, same binary/shape |
| Compute(SM) / Tensor | **77.6 %** | 77.8 % | NCU "highest‑utilized = Tensor" |
| L1/TEX throughput | 54.85 % | 52.84 % | NCU |
| Achieved occupancy | 14.06 % | 14.07 % | NCU |
| Registers / thread | 168 | 168 | NCU |

The NCU comparison is the rigorous apples‑to‑apples number (identical cold‑cache conditions for both
kernels): **1.66 ms ≡ 1.66 ms, Tensor 77.6 % ≡ 77.8 %.**

## Warm number — first‑hand SKIP differential

Isolates the gate kernel's warm cost inside the real pipeline (no standalone re‑timing games):

```
GIST_GATE_WS=1                 total = 3.4809 ms   PASS (max_abs 0.03125, mean_abs 3.15e-6)
GIST_GATE_WS=1 GIST_SKIP_GATE  total = 2.2064 ms   FAIL (max_abs 3.23)  <- gate omitted
                               --------
            gate kernel time  = 1.2745 ms
```
`SKIP_GATE` removes only the gate kernel (stats+sigpad+pool still run) and correctness breaks
(max_abs 3.23) — proving the gate is genuinely timed, no bypass. 1.2745 ms ≈ CUTLASS `gate_ns` 1.26 ms.

## Compute / memory floors (HBM = 2.07 TB/s, measured on this box)

- Compute floor: $B\cdot QF\cdot F\cdot 2 = 874$ GFLOP $/\,990$ TFLOP/s $= 0.88$ ms → at 1.27 ms = **70 % of bf16 peak**.
- Memory: read $M$ (4.6 MB) + $P$ (586 MB) + write $L$ (586 MB) = 1.18 GB $/\,2.07$ TB/s $= 0.57$ ms < compute
  → the gate is **compute‑bound** (correctly; not chasing a sub‑floor memory win).

## The NCU‑driven optimization ladder (2.79 ms → 1.66 ms, Tensor 47.7 % → 77.6 %)

Starting point = warp‑specialized TMA gate, structurally identical to CUTLASS but 1.68× slower. Each step
was picked by the **measured** NCU bottleneck:

| step | change | NCU signal addressed | effect |
|---|---|---|---|
| 1 | single `wgmma.m64n256k16` (vs 4× `m64n64k16`) | "Shared highest‑util", L1/TEX 68.7 % (A re‑read 4×) | L1/TEX → 50.9 % |
| 2 | `setmaxnreg` 32 / 232 | consumer register starvation | (enabler) |
| 3 | coalesced epilogue (SMEM stage → uint4 stores) | `lg_throttle` = **41 %** of stall samples (scalar 2B stores) | Tensor 47.7 → 72.8 %, total −0.63 ms |
| 4 | SMEM stage row‑stride 256→264 | "18 % excessive shared wavefronts" (8‑way bank conflict) | Tensor 72.8 → **77.6 %** = parity |
| 5 | `wgmma.fence` hoisted out of K‑loop | per‑iter serialization | cleanup |

NCU of CUTLASS `gate_ns` (the target): TileShape **128×256×64**, **MMA_64x256x16**, NS=4,
`KernelTmaWarpSpecializedCooperative`, **ClusterShape ⟨1,1,1⟩ (no multicast)**, STSM + `SM90_TMA_STORE`
epilogue. My kernel mirrors all of this.

## Files in this folder

- `gate_gemm.cu` — clean, readable extract of the kernel (FUSE=false / no‑sigmoid), with the inline‑PTX
  WGMMA/TMA helpers and the host launch (TMA descriptor) commented.
- `gate-gemm.md` — the algorithm in the GPU‑kernel notation (Ξ grid / Φ loop) of the obsidian note.
- `gist_full_snapshot.cu` — the full authoritative `gist.cu` this was extracted from (compiles via the
  cuda_exec harness; the gate ships behind env `GIST_GATE_WS`, FUSE=false).
- `mine.ncu-rep`, `cutlass.ncu-rep` — the NCU reports backing the table above.

## Reward‑hack audit (PASS)

Genuine bf16 `m64n256` WGMMA, real TMA loads of $M$ and $P$, real coalesced bf16 output, full design
shape, unmodified L2‑flush harness, correctness PASS (max_abs 0.03125). The NCU comparison uses the same
binary and shape for both sides. Warm number is a SKIP‑differential inside the real pipeline.
