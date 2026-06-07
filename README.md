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
M = mean(X, axis=-1)              [B, F]          gate input (== LayerNorm mean)
L = sigmoid(M @ P)               [B, Q, F]       per-(query, feature) gate  (P: [F, Q*F])
N = LayerNorm(X, axis=-1)        [B, F, D]       normalized values
O = bmm(L, N)                    [B, Q, D]       output: Q summary embeddings
```

- **Gate** `L`: each of the `Q` output slots gets a soft, per-feature weight in
  `(0,1)` derived from the cheap per-feature mean `M` through a learned projection
  `P`. This is the "gated" + "summary" part — it decides how much each feature
  contributes to each summary slot.
- **Values** `N`: LayerNorm of the features over the channel axis — the
  "transformation" part. Note `M` is reused: it *is* the LayerNorm mean.
- **Pool** `O = L @ N`: a weighted pool of the normalized features into `Q` slots.

Design defaults (large-scale target): `B=1536, F=1491, D=192, Q=128`.

## Files

| File | What |
|---|---|
| `gist_pytorch.py` | Readable reference forward (`class GIST(nn.Module)`). Materializes the big intermediates `L [B,Q,F]` and `N [B,F,D]`. |
| `gist_triton.py`  | Fused Triton forward (`gist_triton_forward`). Computes the gate tile-by-tile on-chip and recomputes LayerNorm on the fly, so `L` and `N` never touch HBM — only `X`, `P` are read and `O` is written. |

The two agree to `< 1e-3` max abs error (see each file's `__main__` self-test; the
Triton test needs a CUDA box).

## Why no online (single-pass) form

Unlike FlashAttention's softmax — whose reduction is over the *streamed* axis and
is rescalable — GIST's gate contracts the **full** per-feature mean `M`, so `M`
must be complete before any output. Clean structure is therefore **two grids**: a
cheap stats grid lands `M, rstd : [B,F]` to HBM, then the main grid fuses
gate × pool reading them back. Only the tiny `M, rstd` cross HBM between grids; the
big `L` (~0.6 GB) and `N` (~0.9 GB) never do. `X` is read twice (stats, then
LayerNorm), which is still cheaper than materializing `N`.

The Triton kernel here is the main (Grid 2) fusion with `M, rstd` precomputed; it
is **correctness-first, not speed-optimal** (no tensor cores on the gate yet). The
fast path tiles a batch block so the gate becomes a tensor-core GEMM that reuses
each streamed `P` tile — see the design note.

## Status & next steps

**Status:** forward-only; correctness-first. Written on a machine with no GPU and
no `torch`, so **nothing here has been run yet** — the self-tests are unexecuted.

To continue on a GPU box (in order):

1. **CPU reference** — `python gist_pytorch.py` (needs `torch`). Confirms shapes
   and the reference math.
2. **Triton vs reference** — `python gist_triton.py` on CUDA. Confirms the fused
   kernel matches to `< 1e-3`. The self-test uses `F=80` (not a multiple of
   `BLOCK_G`) on purpose, to exercise the padded-`g` masking path.
3. **Optimize the gate (the real work).** The current gate is a per-batch GEMV via
   `tl.sum` with the `P` sub-tile register-resident — correct but slow (no tensor
   cores). Make both stages `tl.dot` GEMMs and **tile a batch block** (`BLOCK_B >=
   16` rows of `M`) so the gate becomes a tensor-core GEMM `M[BLOCK_B,F] @ P` that
   reuses each streamed `P` tile across the batch. `P` (~0.6 GB, shared across all
   `B`) is the dominant HBM traffic, so batch reuse is the main win. Optionally
   split into the explicit two-grid form (cheap stats grid lands `M, rstd`, main
   grid fuses gate × pool) — see Obsidian §4.

## Design note

Full notation and the grid/loop derivation live in the personal Obsidian note
*"A Notation for GPU Kernel and Layout Algebra"*, §4 (GIST) — kept as the single
source of truth, not duplicated here. (`obsidian-vault` repo; pull it alongside
this one.)
