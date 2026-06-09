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
| `cuda/gist.cu` | **Hand-written CUDA + inline-PTX (Hopper WGMMA)** forward — the fastest impl (3 fused kernels: stats → σ+pad gate → inline-N pool). Run via the `cuda_exec` harness (`./run_gist.sh`). |

Run on a CUDA box with the `.venv` (uv-managed CPython, torch 2.12 + triton 3.7); pin an idle GPU via
`CUDA_VISIBLE_DEVICES`. `bench.py` / `test_gist.py` exercise the Triton path; `run_gist.sh` the CUDA kernel.

## Results (single H100, B=1536 F=1497 D=192 Q=128, bf16)

Latency via `triton.testing.do_bench` (min, L2-flush) on GPU 0 — the `cuda_exec` harness GPU/statistic
(`run_gist.sh`'s `min_ms`):

| method (bf16) | latency |
|---|---|
| PyTorch eager | 12.6 ms |
| PyTorch `compile` | 4.26 ms |
| Triton-fast (`gist_triton_fast_forward`) | 4.19 ms |
| **CUDA + inline PTX (hand-written)** | **2.64 ms** |

The hand-written **CUDA+PTX kernel is the fastest — ~37 % faster than Triton-fast** (correct: max_abs
0.0156, 0/1536 vs the RMSNorm reference). It's **three fused hand-written kernels**, each pinned to a
different ceiling on this (throttled) box (~2.07 TB/s HBM, ~800 TFLOP/s bf16 tensor):

| kernel | time | bound | util |
|---|---|---|---|
| **stats** — `M`, `rrms` (int4-vectorized D-reduction) | 0.43 ms | memory | 92% HBM |
| **gate** — `σ(M@P)` WGMMA, σ + F→Fp pad **fused into the epilogue** (writes padded `SL` directly) | 1.37 ms | tensor | 84% of 800 TF |
| **pool** — `SL@N` WGMMA, RMSNorm `N = X·rrms·W` computed **inline** (no materialized `N`) | 0.79 ms | memory | 87% HBM |

Two fusions are the win: the gate writes padded `SL` directly (deletes the separate ~0.78 ms repad pass),
and the pool computes `N` inline (deletes the ~0.9 GB `N` round-trip). Triton instead pays a slow `tl.dot`
gate (~2.6 ms); `compile` pays ~1.85 ms of memory-bound glue.

> **Caveat (weight prepack):** the gate uses a one-time prepacked padded `P` — valid because the weights
> are static (the harness holds `P`, `W` constant). Triton's per-call path prepacks nothing, so the ~37 %
> is a production-mode (prepacked-weights) comparison. Repro:
> `GIST_HANDGATE=1 GIST_GATEPAD=1 GIST_FUSESIG=1 GIST_POOLFUSE=1 ./run_gist.sh <rev> big`
> (kernel snapshot: `GEMS/2026-06-09-gist-b-phase4-gate-padded-norepad/`).

**Correctness:** the kernel passes the `cuda_exec` harness (max_abs 0.0156, 0/1536). The Triton path passes
`test_gist.py` (20 checks: 5 seeds × {bf16,tf32} × {design shape, masking edge}), bf16 rel-Frobenius
~4.1e-3 vs both fp32 truth and the same-precision PyTorch reference.

## Why two grids (no single-pass form)

Unlike FlashAttention's softmax (reduction over the *streamed* axis, rescalable), GIST's gate contracts
the **full** per-feature mean `M`, so `M` must be complete before any output. Hence two grids: a cheap
`stats` grid lands `M, rrms` to HBM; then `gate → pool` reads them back. The big `L` and `N` never fully
cross HBM — the gate writes padded `SL` directly, and the pool recomputes `N` inline. `X` is read twice
(stats, then pool), still cheaper than materializing `N`.

## Design note

Full notation and the grid/loop derivation live in the personal Obsidian note
*"A Notation for GPU Kernel and Layout Algebra"*, §4 (GIST) — kept as the single
source of truth, not duplicated here. (`obsidian-vault` repo; pull it alongside
this one.)
