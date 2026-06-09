# GEM: gate-sigmoid-relocate + vectorized sigmoid+pad (branch C)

## Measured (design shape B=1536,F=1491,D=192,Q=128, bf16, L2-flush harness, GPU2)
- **min 3.4652 ms, p50 3.4784 ms** — correctness PASS (max_abs 0.03125, mean_abs 3.15e-6, 0/1536)
- vs Triton 4.225 ms → **18.0% faster**
- vs prior global best 4.014 ms → **13.7% faster**
- vs the just-promoted gate-nosig gem 3.674 ms → **5.7% faster** (the vectorized sigpad delta)

## What moved the number (two stacked changes, both in the DEFAULT no-env path)
1. **gate-sigmoid-relocate** (−0.34ms): run the gate with a plain LinearCombination epilogue (RAW
   logits, `gategemm_ns` / `run_gate_ns`) instead of the fused Sigmoid epilogue, and apply sigmoid
   downstream. NCU(nsys) showed the gate is COMPUTE/tensor-pipeline bound (Tensor-Active 54%, DRAM
   21%, not memory-bound as previously assumed); the fused Sigmoid epilogue's 293M `exp` competes
   with the mainloop. Moving it out drops the gate kernel **1.66 → 1.26 ms**. The exp is ~free in the
   downstream memory-bound pass (+0.05ms only).
2. **vectorized sigmoid+pad** (−0.21ms): `k_sigpad_v2` reads 8 consecutive g_L rows as ONE 16B-aligned
   contiguous block (8×F is a multiple of 8) into smem, then writes each SL row with 16B-aligned
   vectorized stores. This DODGES the odd-F (1491) per-row misalignment that forced scalar 2B access
   in the old k_sigpad/k_repad. sigpad **0.78 → 0.56 ms**.

## Per-kernel breakdown (nsys, this gem)
gate_ns 1.25ms · k_stats(M,R,N) 0.86ms · pool(CUTLASS batched) 0.80ms · k_sigpad_v2 0.56ms ≈ 3.47ms

## Correctness note
mean_abs 3.15e-6 (≪ the 4.6e-4 of the old sigmoid-in-gate path): k_sigpad_v2 does sigmoid in fp32
(`1/(1+expf(-x))`) matching the reference more precisely than CUTLASS's Sigmoid epilogue.

## Reward-hack audit (PASS)
Only `kernel_run` exported; the number is the full default path with NO env vars set. Stage-skip
differential confirms every stage is genuinely timed+validated:
- SKIP_POOLGEMM → 2.89ms FAIL (max_abs 5.84): pool GEMM real (−0.78ms, breaks correctness)
- SKIP_POOL → 2.11ms FAIL: stats+gate real
- SKIP_GATE → 2.42ms FAIL: gate real
No SKIP-guard games, no shape/tolerance weakening, full bf16, L2 flush intact.

## Fallback
The old sigmoid-in-gate + k_repad path (4.02ms) is preserved behind env `GIST_GATE_SIG`.

## Next (gate lane, diminishing): gate 1.25ms is now ~70% peak (was 53%); remaining headroom on this
## lane is small (~0.16ms via gate-epilogue EVT writing padded [B,Q,Fp] to drop the 0.56ms sigpad,
## but sigmoid must then move back into the gate +0.40ms → net ~0.16ms; MAIN compile-blocked the
## hierarchical-StrideD variant). The bigger remaining levers are the STATS kernel (0.86 vs Triton
## 0.41, not this lane) and a gate↔pool fusion that eliminates the L round-trip (large effort).
