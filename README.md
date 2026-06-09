# GIST — Gated Information Summary and Transformation

A small attention-style pooling module: it summarizes a **variable-length list of
feature embeddings** into a **fixed number of summary embeddings** (`Q`), using a
learned, content-dependent gate. Forward only.

The name is a true acronym (interior letters allowed) and also the plain English
word — the module distills the *gist* of a set of features:

> **G**ated · **I**nformation · **S**ummary · and · **T**ransformation

## What it computes

Input is a batch of `F` features, each a `D`-dim embedding; output is `Q` summary
embeddings of the same dimension `D`.

```
X = x_input                       [B, F, D]
M = mean(X, axis=-1)              [B, F]          gate input (mean over D)
L = sigmoid(M @ P)               [B, Q, F]       per-(query, feature) gate  (P: [F, Q*F])
N = RMSNorm(X, axis=-1)          [B, F, D]       normalized values (X·rsqrt(mean_D(X²)+eps)·W)
O = bmm(L, N)                    [B, Q, D]       output: Q summary embeddings
```

- **Gate** `L`: each of the `Q` output slots gets a soft, per-feature weight in
  `(0,1)` derived from the cheap per-feature mean `M` through a learned projection
  `P`. This is the "gated" + "summary" part — it decides how much each feature
  contributes to each summary slot.
- **Values** `N`: RMSNorm of the features over the channel axis `D` (no
  mean-subtraction, per-`(feature, channel)` weight `W [F, D]`, no bias) — the
  "transformation" part. This is unified with kernel_lab GAttention's `input_norm`.
- **Pool** `O = L @ N`: a weighted pool of the normalized features into `Q` slots.

Design defaults (large-scale target): `B=1536, F=1497, D=192, Q=128`.

## Files

| File | What |
|---|---|
| `gist_pytorch.py` | Readable reference forward (`class GIST(nn.Module)`) + `init_unit_scale_` (fan-in init so `O ~ O(1)`). |
| `gist_triton.py`  | v1 correctness-first fused forward (`gist_triton_forward`): single kernel, per-batch GEMV gate. Correct but ~25× slower than the fast path — kept for reference. |
| `gist_triton_fast.py` | **Fast** forward (`gist_triton_fast_forward`, `precision="bf16"`/`"tf32"`): three autotuned kernels — stats (`M,rrms`) → gate GEMM (`L=sigmoid(M@P)`, batch-tiled tensor-core, L2-grouped) → fused pool/RMSNorm. |
| `bench.py` | Benchmark vs PyTorch eager / `torch.compile` across fp32/tf32/bf16: latency, speedup, achieved TFLOP/s, rel & abs error. |
| `test_gist.py` | Hardened correctness test: 5 seeds × {bf16,tf32} × {design shape, masking edge case}, asserts rel-Frobenius vs fp32 truth **and** vs same-precision PyTorch. |

## Results (single H100, B=1536 F=1497 D=192 Q=128, ~994 GFLOP/call)

Same precision (bf16), `triton.testing.do_bench` (warmup + L2-flush + median), all in one session:

| method (bf16) | latency |
|---|---|
| PyTorch eager | 12.5 ms |
| PyTorch `compile` | 4.30 ms |
| Triton-fast (`gist_triton_fast_forward`) | 4.22 ms |
| **CUDA + inline PTX (`cuda/gist.cu`)** | **3.62 ms** |

The hand-written **CUDA + inline-PTX (Hopper WGMMA) kernel is the fastest — ~14 % faster than
Triton-fast**, ~16 % vs `torch.compile`. The lever is the gate GEMM: Triton's `tl.dot` gate reaches only
~37 % of the tensor pipe (≈2.6 ms), and `compile` burns ~1.85 ms in memory-bound glue (transpose P,
materialize N, sigmoid); both the Triton and CUDA kernels fuse that glue away, and the CUDA kernel's
**WGMMA gate runs at ≈1.25 ms**.

Per-kernel µs — Triton: gate 2.64 · pool 1.19 · stats 0.41.  CUDA: gate ≈1.25 · pool ≈0.79 · sigpad ≈0.56
· stats ≈0.86. (This box runs ~2.07 TB/s HBM and ~807 TFLOP/s bf16 tensor, both below datasheet, so the
gate at ~700 TFLOP/s is already near its compute ceiling here.)

**Correctness:** validated against an fp32 reference with `init_unit_scale_` (so `O ~ O(1)` and
tolerances are meaningful). `test_gist.py` passes all 20 checks (5 seeds × {bf16,tf32} ×
{design shape, masking edge case}): bf16 rel-Frobenius ~4.1e-3, tf32 ~8.2e-4, each vs both the
fp32 truth and the same-precision PyTorch reference; `cuda/gist.cu` passes the `cuda_exec` harness
(max_abs 0.0156, 0/1536 batches). Run `bench.py` / `test_gist.py` on a CUDA box.

## Why no online (single-pass) form

Unlike FlashAttention's softmax — whose reduction is over the *streamed* axis and
is rescalable — GIST's gate contracts the **full** per-feature mean `M`, so `M`
must be complete before any output. Clean structure is therefore **two grids**: a
cheap stats grid lands `M, rrms : [B,F]` to HBM, then the main grid fuses
gate × pool reading them back. Only the tiny `M, rrms` cross HBM between grids; the
big `L` (~0.6 GB) and `N` (~0.9 GB) never do. `X` is read twice (stats, then
RMSNorm), which is still cheaper than materializing `N`.

`gist_triton.py` (v1) is the main (Grid 2) fusion with `M, rrms` precomputed —
correctness-first, per-batch GEMV gate (no tensor cores), so slow. `gist_triton_fast.py`
is the optimized path: a separate batch-tiled tensor-core gate GEMM (so each streamed `P`
tile is reused across `BLOCK_M >= 16` rows of `M`) materializes `L` (cheap, ~0.3 ms), then
a fused pool/RMSNorm kernel consumes it. Materializing `L` is far cheaper than the gate
compute, so the two-kernel split wins over a single fused kernel (whose tensor-core gate
would force a 3D pool accumulator that wrecks occupancy).

## Status & next steps

**Status:** forward-only; **done and validated on H100**. Fastest is the hand-written **CUDA +
inline-PTX kernel** (`cuda/gist.cu`) at **3.62 ms — ~14 % faster than Triton-fast (4.22 ms)** by
replacing the `tl.dot` gate with a WGMMA gate; the Triton path (3 autotuned kernels, stats → gate
GEMM → fused pool) is ~2 % faster than `torch.compile`. See Results above; v1 (`gist_triton.py`)
kept for reference.

Environment: `.venv` (uv-managed CPython, not Meta python) with torch 2.12 + triton 3.7.
Run `.venv/bin/python bench.py` (pin an idle GPU with `CUDA_VISIBLE_DEVICES`).

Possible further work:
1. **bf16 gate** dominates the cost; the whole kernel runs at ~227 TFLOP/s total on this
   small-M/huge-N shape — a plain batch-tiled GEMM is near its floor here. A Hopper TMA +
   warp-specialized / persistent gate could push higher.
2. **tf32 variant** loses to `compile`-tf32: cuBLAS's tf32 GEMM is strong and inductor
   overlaps the pool. Beating it needs a single fully-fused kernel (hard: the batch-tiled
   tensor-core gate forces a 3D pool accumulator) — not done.
3. Backward pass; smaller-batch / variable-F regimes.

## Design note

Full notation and the grid/loop derivation live in the personal Obsidian note
*"A Notation for GPU Kernel and Layout Algebra"*, §4 (GIST) — kept as the single
source of truth, not duplicated here. (`obsidian-vault` repo; pull it alongside
this one.)
