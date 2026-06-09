# Baseline gem — CUTLASS 3-stage pipeline (default path)

**Measured: 4.3747 ms bf16** (harness L2-flush, design shape B=1536,F=1491,D=192,Q=128).
Correctness: passed, max_abs_error 0.03125 (bf16 quantization floor), mean_abs 3.15e-6.

- vs Triton `gist_triton_fast` 4.225 ms: **+3.5% SLOWER** (goal-fail; this is the bar to beat).
- vs <2.95 ms target (30% faster): far off.

## What it is
`kernel_run` default path in `gist.cu`:
1. `k_stats` — mean M (col-major), rstd R, N=LayerNorm·γ+β padded to [B,Fp,D] (one X read).
2. `run_gate` — CUTLASS WGMMA GEMM L_raw = M@P (M ColumnMajor so K=F unpadded). 1.97 ms (45%).
3. `k_sigpad` — sigmoid + repad L→SL[B,Q,Fp]. **0.77 ms of PURE overhead** (forced by CUTLASS's
   8-alignment requirement: F=1491 is odd → can't feed unpadded to the batched pool).
4. `run_pool` — CUTLASS batched GEMM O=SL@N. 0.78 ms.

## Why it's the bar / where the win is
The 0.77 ms sigpad pass is the gap. A CUTLASS pool can't consume the unpadded F=1491 K-axis
(needs 8-aligned strides), so the sigmoid+pad must be a separate kernel. A **hand-written
inline-PTX WGMMA pool has no alignment requirement** → it reads L unpadded (K=F) directly and
fuses sigmoid on load, deleting the sigpad pass entirely. That is the wgmma-pool direction.

## Reproduce
`cd ~/gist/cuda && ./run_gist.sh <fresh_rev> big` (no env flags = default path).
