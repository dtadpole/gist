# GEMS — verified GIST wins

A "gem" = a VERIFIED improvement: a new best measured bf16 latency at the design shape
(B=1536,F=1491,D=192,Q=128), correctness within the harness tolerance, no reward hacking.

When you hit a gem, drop a timestamped subfolder here, e.g. `GEMS/2026-06-07-wgmma-gate/`:
- `kernel.cu` (or a snapshot/copy of the winning kernel source)
- `bench.json` — the harness benchmark output (real CUDA-event timing)
- `results.md` — measured bf16 latency, % vs 4.22 ms Triton bar and vs <2.95 ms target,
  which technique / inline-PTX op moved the number, and the head-to-head numbers
- `ncu/` — the NCU profile / evidence that explains the win

Only put REAL, reproducible wins here. This folder is the record of what actually worked.

## CURRENT GLOBAL BEST: 3.398 ms (MAIN, cuBLAS unpadded-K gate, RMSNorm/F=1497) — aggregator-verified 2026-06-09T08:00Z

> Aggregator independently re-ran committed cuda/gist.cu (== GEMS/2026-06-09-cublas-gate-rmsnorm/kernel.cu,
> byte-identical) on an isolated /tmp copy (GPU4, rev90201 big): min 3.3978ms, correctness PASS
> (max_abs 0.015625, mean 3.09e-6, 0/1536). Only kernel_run exported (single extern "C"). cheat-grep ZERO
> (sole "cached" hit = benign comment). 194 wgmma/bf16 sites. Measurement-bypass differential (rev90205,
> GIST_SKIP_GATE=1): latency drops 3.3978->2.2299ms AND correctness BREAKS (passed:false, max_abs 2.105)
> => the gate GEMM is genuinely timed, no bypass. Win = cuBLAS nvjet NTN 192x192 (correct unpadded-K
> col-major layout, op_B=T ldb=B) replaces CUTLASS auto-pick on the skinny-K/huge-N gate => gate
> 1.288->1.178ms. vs Triton-RMSNorm 4.1519 = -18.2%. Still ABOVE min-20% (3.32, ~2.3% over) and 30% (2.91).
> NOTE: headline rides a VENDOR lib (cuBLAS), not the hand-PTX WGMMA gate (run_gate_ns 1.268ms still
> present, GIST_CUTLASS_GATE=1 selectable, only ~2.8% slower) — a deliverable-conformance concern for owner.

## SUPERSEDED: 3.499 ms (MAIN, RMSNorm/F=1497 oracle) — aggregator-verified 2026-06-09

> SPEC NOTE: The competition reference was migrated LayerNorm->RMSNorm/F=1497 (committed to
> main e33f072, oracle cuda/gist_ref.py @22:34). The prior 3.465ms (c-gate-nosig-vecsigpad)
> was earned vs the now-RETIRED LayerNorm/F=1491 oracle and is NO LONGER the valid global best.
> New verified best is MAIN's RMSNorm kernel: aggregator independently re-ran the committed
> cuda/gist.cu on an isolated /tmp copy (GPU4, rev90001 big shape) = min 3.4992ms,
> correctness PASS (max_abs 0.015625, mean 3.09e-6, 0/1536). Only kernel_run exported
> (single extern "C"); cheat-grep ZERO; 182 wgmma/bf16 sites live; 3.499 is honestly
> ~slightly slower than the stale 3.465, no too-good signal. New Triton/RMSNorm bar 4.1519ms
> => -15.7%. New min-20% bar = 3.32ms (still above), 30% target = 2.91ms.

## Ledger (newest first)
- **2026-06-08-c-gate-nosig-vecsigpad** — **3.465 ms** bf16, correctness PASSED (max_abs 0.03125, mean 3.15e-6, 0/1536).
  **NEW GLOBAL BEST.** BEATS Triton 4.225 ms by **18.0%**, beats prior best 3.674 by 5.7%. Stacks on the
  gate-nosig win (sigmoid relocated out of the gate epilogue) PLUS a vectorized sigmoid+pad pass
  (k_sigpad_v2: read 8 contiguous g_L rows as one 16B-aligned block, sigmoid in registers) cutting the
  repad/sigpad cost 0.78→0.56 ms. Aggregator independently re-verified on isolated /tmp copy (GPU3,
  rev78021 min=3.470 PASS), incl. SKIP_GATE stage-skip differential (drops to 2.212 ms + breaks
  correctness max_abs 3.23 = gate genuinely timed, no bypass). Still ABOVE 20% min (3.38, ~2.5% over)
  and 30% target (2.95) — gate lane near floor; remaining money is in stats (0.86 vs Triton 0.41).
- **2026-06-08-c-gate-nosig** — **3.674 ms** bf16, correctness PASSED (max_abs 0.03125, mean 3.15e-6, 0/1536).
  **NEW GLOBAL BEST.** BEATS Triton 4.225 ms by **13.0%**, beats prior best 4.014 by 8.5%.
  Win = defer sigmoid OUT of the gate's CUTLASS tensor epilogue (raw logits), apply it downstream in
  k_sigpad (memory-bound pass, exp hides behind DRAM) → gate 1.66→1.26 ms. First measured win on the
  GATE lever (shared 1.65ms bottleneck). Aggregator independently re-verified on isolated /tmp copy
  (GPU3), incl. SKIP_GATE stage-skip differential (drops to 2.42ms + breaks correctness = no bypass).
  Still ABOVE 20% min (3.38) and 30% target (2.95) — gate has more headroom (53%→~75% peak).
- **2026-06-08-gate-fused-sigmoid** — **4.014 ms** bf16, correctness PASSED (max_abs 0.03125, 0/1536).
  **BEATS Triton 4.225 ms by 5.0%** (and the 4.375 CUTLASS baseline by 8.2%). First config to beat
  Triton. Win = sigmoid fused into the gate epilogue (`LinCombEltAct<Sigmoid>` + explicit cooperative
  schedule, which itself sped the gate 2.82→2.43 ms) + repad-only pass (no 293M exp, pad pre-zeroed).
- **2026-06-08-cutlass-baseline** — 4.3747 ms (CUTLASS 3-stage default). The bar = Triton 4.225 ms;
  this is +3.5% (goal-fail). Recorded as the reference point. sigpad (0.77 ms) is the removable gap.
- (milestone, not yet a speed win) hand-written inline-PTX `wgmma.mma_async.m64n64k16` fused pool —
  VERIFIED CORRECT (max_abs 0.0312, 0/1536) reading unpadded L (K=F=1491) with sigmoid fused, no
  sigpad pass. The literal CUDA+PTX deliverable, now built & numerically verified. Best correct
  config 6.90 ms (2-wg, N read once) — still ABOVE 4.375 baseline because the pool is latency-bound
  (NCU: DRAM 29%, occ 24%) and the occupancy fix (acc[32], one n-tile/block) hits a ptxas warp-3
  accumulator bug. Recipe + bug + workaround hypotheses in memory `gist-wgmma-handwritten.md`.
  Occupancy fix exhausted (6 workarounds tried: acc[64], m64n96 acc[48], scaleD=0, launch_bounds,
  per-ks commit/wait — all hit the warp-3 bug; single-group double-buffer is CORRECT but 12.09 ms
  from 2x-smem occupancy loss). The bug triggers on small GMMA accumulators + many commit/wait
  cycles. Remaining untested high-effort paths: triple-buffered pipeline with wgmma.wait_group<1>
  (never fully drains -> may dodge the trigger) at acc[32]; or TMA bulk loads. Default 4.375 stands.
