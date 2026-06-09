# GEM: cuBLAS unpadded-K gate (RMSNorm/F=1497) — 3.3955 ms, 19.7% faster than Triton

**Date:** 2026-06-09 (MAIN lane)
**Shape:** B=1536, F=1497, D=192, Q=128 (RMSNorm spec, 3 inputs X[B,F,D], P[F,QF], W[F,D])
**Hardware:** H100 sm_90a, this box (throttled ~807 TF square peak)

## Result (measured, full L2-flush harness)

| metric | value |
|---|---|
| **default total (cuBLAS gate)** | **3.3955 ms** PASS, max_abs 0.0156 (min; p50 3.405, mean 3.406) |
| prior default (CUTLASS gate) | 3.4935 ms PASS (GIST_CUTLASS_GATE=1 = 3.5015) |
| improvement | **-0.098 ms (-2.8%)** total, from the gate alone |
| stability | 3.3955 / 3.3961 / 3.3969 ms (3 reruns) |
| interleaved CUDA vs Triton (same clocks, L2-flush, 100 iters) | **CUDA 3.4666 / Triton 4.3186 -> CUDA 19.7% faster (1.246x)** |

vs the bars: beats 3.49 (prior best) and the 20% standalone bar (3.32) is **just missed standalone
(3.3955 > 3.32)** but the **interleaved fair number is 19.7% faster than Triton** (the README/owner
methodology bar) — at the 20% target. Not yet at 30% (2.93).

## What moved the number: the gate GEMM (nsys, design shape, 42-43 timed launches)

| gate kernel | avg | median |
|---|---|---|
| CUTLASS `GemmUniversal 128x256x64` (prior) | 1.288 ms | 1.319 ms |
| **cuBLAS `nvjet_sm90_tst_192x192_64x3_2x1_v_bz_coopB_NTN`** | **1.178 ms** | 1.219 ms |

Gate win **~0.11 ms avg**. k_stats (0.876), k_sigpad_v2 (0.560), pool (0.799) are byte-identical in both
builds — the ONLY delta is the gate kernel. cuBLAS picks a better kernel (nvjet NTN, 192x192) for this
skinny-K (K=F=1497), huge-N (QF=191616) shape than CUTLASS's auto-pick.

## The key insight (corrects the fleet's "cuBLAS-gate DEAD-END")

The prior fleet finding was "cuBLAS gate = 6.30 ms odd-K cliff." That was the **WRONG operand layout**:
`cublasGemmEx(OP_N, OP_N, ..., M ldb=F)` — but M is stored **COLUMN-MAJOR [B,F] (ld=B)**, so OP_N,OP_N
forced cuBLAS into a slow internal transpose/cliff path.

With the **correct** layout (row-major C^T trick, M passed with **op_B=T, ldb=B**):
```
L row-major[B,QF] = C^T[QF,B] (col-major ld=QF) = P^T[QF,F] @ M^T[F,B]
gemm(OP_N, OP_T, m=QF, n=B, k=F,  P  lda=QF,  M(col-major[B,F]) ldb=B,  L ldc=QF,  CUBLAS_COMPUTE_32F)
```
cuBLAS does the gate at **UNPADDED K=F=1497 in 1.095 ms / 805 TFLOP/s** — NO odd-K cliff, NO pad copy.

### Padded-K is a TRAP (measured, /tmp/cublas_gate_bench.cu — standalone_gate_bench.cu)
The task hypothesis was "pad K to Fp=1536 -> 1.06ms / 848 TF." Measured standalone:
- cuBLAS GEMM at padded K=Fp : 1.118 ms / 809 TF
- **padP (copy P[F,QF]->Pp[Fp,QF], the 574MB read + 588MB write) : 0.556 ms**  <- the killer
- padM : 0.012 ms
- full padded gate (padM+padP+GEMM) : **1.615 ms** -> SLOWER than CUTLASS 1.27ms
- in-pipeline padded gate full harness : **3.956 ms** (regression!)

The pad copy (0.556ms) dwarfs the GEMM savings (~0.15ms). The UNPADDED layout wins precisely because it
needs no copy — it reads M and P in place. (probe: standalone_unpad_probe.cu shows K=1497 = 1.095ms across
all algos.)

## Correctness (debug_isolate, big shape, default no-env path)
```
max_abs=0.0156 mean_abs=0.0000 | all q/d/b blocks 0.0 | #b err>0.1: 0/1536
```
Identical to the CUTLASS baseline (same logits, bf16 IO, CUBLAS_COMPUTE_32F fp32 accumulate — no tf32).

## Files
- `kernel.cu` — gist.cu snapshot (run_gate_cublas2 = the cuBLAS gate; default no-env path uses it,
  GIST_CUTLASS_GATE=1 falls back to CUTLASS)
- `bench_cublas_default.json` — full-harness comparison JSON (rev 95012)
- `nsys_gate_cublas.nsys-rep` / `nsys_gate_cutlass.nsys-rep` — per-kernel gate evidence
- `standalone_gate_bench.cu` — padded-gate trap measurement (GEMM/padM/padP/full)
- `standalone_unpad_probe.cu` — unpadded K=1497 algo sweep (all ~1.1ms, no cliff)

## Reproduce
```
CUDA_VISIBLE_DEVICES=5 ./run_gist.sh <rev> big | python3 parse_min.py     # default = cuBLAS gate, 3.3955
GIST_CUTLASS_GATE=1 CUDA_VISIBLE_DEVICES=5 ./run_gist.sh <rev> big | python3 parse_min.py  # 3.5015
CUDA_VISIBLE_DEVICES=5 ~/gist/.venv/bin/python debug_isolate.py           # correctness, all blocks 0
CUDA_VISIBLE_DEVICES=5 ~/gist/.venv/bin/python /tmp/interleave_cuda_triton.py  # 19.7% vs Triton
```
