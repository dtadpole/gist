# GEM — branch C — gate-NOSIG (defer sigmoid out of gate epilogue)

**Verified by aggregator (independent isolated re-run), 2026-06-08T21:1x Z.**

## Measured (full kernel, big shape B=1536,F=1491,D=192,Q=128, bf16)
- **min 3.674 ms** (p50 3.689, mean 3.689, 27 runs), via shared harness on GPU3, isolated /tmp copy.
- Correctness **PASS**: max_abs 0.03125, mean_abs 3.15e-6 (cleaner than prior gem's numerics).
- vs Triton 4.225 ms = **−13.0%**. vs prior global best 4.014 ms = **−8.5%**.
- Crosses the **<3.38 ms (20% min-acceptable) bar** (3.674 still ABOVE it — wait: 3.674 > 3.38, so still ABOVE min). Closes most of the gap; 30% target (2.95) and 20% min (3.38) NOT yet reached.

## What moved the number
Defer sigmoid OUT of the gate's CUTLASS tensor epilogue (`run_gate_ns` writes RAW logits),
then apply sigmoid downstream in `k_sigpad` (sigmoid+pad in ONE memory-bound pass; the exp
hides behind DRAM latency, +~0.05ms vs plain k_repad). The gate's tensor pipeline is
compute-bound; removing the per-element sigmoid from its epilogue sped the gate **1.66 → 1.26 ms**.
This is the first measured win on the GATE lever (the shared 1.65ms bottleneck), not the pool.

## Anti-reward-hack audit (PASS)
- gist.cu exports ONLY `kernel_run` (L1278). No setup/launch split, no memoize/first_call/static-done, no 0xCAFE/golden/hardcode/expected[]/tf32.
- All GIST_SKIP_*/GIST_GATE_SIG env-gated diagnostics default-OFF; default path runs full compute.
- Stage-skip differential: GIST_SKIP_GATE=1 drops latency 3.674→2.419 ms AND breaks correctness (max_abs 3.23) ⇒ gate work genuinely inside the timed region, NOT a measurement bypass.
- Sigmoid is genuinely still computed (relocated to k_sigpad), correctness preserved.

## Provenance
Source: ~/gist-opt/dir-c-multiwg/gist.cu (branch C, lane #5 reassigned to GATE). Reproduced twice by C (rev3036/3037) before aggregator re-verify.
