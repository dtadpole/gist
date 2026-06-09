# GIST gate GEMM — hand-written CUDA+PTX (Phase 1, no sigmoid)

Hand-written replacement for **Grid 2 — `run_gate_ns`** of [[A Notation for GPU Kernel and Layout Algebra]]
(§4 GIST). Same math, same notation; CUTLASS swapped out for raw CUDA + inline PTX (Hopper WGMMA + TMA).
**Phase 1 = the bare GEMM only**: it emits the raw gate **logits** $G$ (the note's $L$ before $\sigma$).
Sigmoid + repad are *not* fused here (that is Phase 2).

## Tensors (notation as in the obsidian note)

$$X:\,[B,F,D],\quad M:\,[B,F],\quad P:\,[F,QF],\quad L:\,[B,QF]\equiv[B,Q,F],\qquad \sigma=\text{sigmoid}.$$

$M$ = `x_input_mean` (per-feature mean of $X$ over $D$), $P$ = `proj_params`, $L$ = `linear_output`.
The gate is the matmul $L = M\cdot P$ — the note's Grid 2 line: $M:[B,F]_{\text{col}}\times P:[F,QF]\to L:[B,QF]$,
**epilogue = logits, no $\sigma$**. Design shape $B{=}1536,\ F{=}1491,\ F_p{=}1536,\ D{=}192,\ Q{=}128$, bf16.

As a GEMM this is the **skinny‑K** problem:
$$\underbrace{M_{\text{dim}}=B=1536}_{\text{rows}}\ ,\quad \underbrace{N_{\text{dim}}=QF=190848}_{\text{cols, } n=qF+g}\ ,\quad \underbrace{K_{\text{dim}}=F\ (K_{\text{pad}}=F_p=1536)}_{\text{contraction }f'} .$$

## Partition (tags)

| tag | axis | block size | #blocks |
|---|---|---|---|
| $\beta$ | batch / $M_{\text{dim}}$ rows of $L$ | $\bar\beta = 128$ | $\lvert\beta\rvert = \lceil B/128\rceil = 12$ |
| $\nu$ | $N_{\text{dim}}=QF$ cols ($n=qF+g$) | $\bar\nu = 256$ | $\lvert\nu\rvert = \lceil QF/256\rceil = 746$ |
| $\tau$ | gate reduction $f'$ (the $F$/$K$ axis) | $\bar\tau = 64$ | $\lvert\tau\rvert = F_p/64 = 24$ |

Tiles (superscript = which tile, subscript = local entry):
$$A^{(\beta,\tau)}=M^{(\beta,\tau)}:[\bar\beta\times\bar\tau]=[128\times64],\quad
P^{(\tau,\nu)}:[\bar\tau\times\bar\nu]=[64\times256],\quad
L^{(\beta,\nu)}:[\bar\beta\times\bar\nu]=[128\times256].$$
Accumulator $G:[128\times256]$ in fp32 (bf16 inputs, fp32 acc — the WGMMA shape `m64n256k16` ×2 row‑halves).

## Kernel — warp‑specialized cooperative WGMMA (one grid)

One grid cell computes one output tile $L^{(\beta,\nu)}$. Inside the cell, **3 warpgroups** specialize:
a **producer** $\Phi^{\text{prod}}$ (TMA loads, no math) and **2 cooperative consumers** $\Phi^{\text{cons}}_{0},\Phi^{\text{cons}}_{1}$
(pure WGMMA; consumer $w$ owns the 64‑row half $\beta_w$ of the 128‑row tile). They communicate through an
**$N_s=4$‑deep async ring buffer** of staged tiles $(\tilde A_s,\tilde P_s)$ in SMEM, gated by `full`/`empty` mbarriers
— i.e. $\Phi^{\text{prod}}$ runs ahead of $\Phi^{\text{cons}}$ by up to $N_s$ K‑tiles.

$$
\begin{aligned}
&\Xi[\beta,\nu] &&\text{grid cell: one output tile }L^{(\beta,\nu)}:[128\times256] \\[2pt]
&\;\textbf{producer } \Phi^{\text{prod}}[\tau],\ \tau=1,\dots,\lvert\tau\rvert &&\text{1 warpgroup; }\texttt{setmaxnreg.dec}\ (\text{32 regs}) \\
&\quad \textbf{wait } \text{empty}[s],\ \ s=\tau \bmod N_s &&\text{ring slot free (consumers done with it)} \\
&\quad \mathbf{TMA}\ \tilde A_s \gets A^{(\beta,\tau)},\ \ \tilde P_s \gets P^{(\tau,\nu)} &&\texttt{cp.async.bulk.tensor.2d}\to\text{B128‑swizzled SMEM} \\
&\quad \textbf{arrive } \text{full}[s] &&\text{(TMA tx‑count completes it; K/N tails auto‑zeroed)} \\[4pt]
&\;\textbf{consumer}_w\ \Phi^{\text{cons}}[\tau],\ w\in\{0,1\} &&\text{2 warpgroups; }\texttt{setmaxnreg.inc}\ (\text{232 regs}) \\
&\quad G \gets 0 &&G:[64\times256]\ \text{fp32 acc (this consumer's }\beta_w\text{ half)} \\
&\quad \textbf{wait } \text{full}[s],\ \ s=\tau\bmod N_s &&\text{staged }(\tilde A_s,\tilde P_s)\text{ ready} \\
&\quad G \mathrel{+}= \tilde A_s^{(\beta_w)}\,\tilde P_s &&\texttt{wgmma.m64n256k16}\times(\bar\tau/16)\ \text{(one async group)} \\
&\quad \textbf{wait\_group}\langle1\rangle;\ \ \textbf{arrive } \text{empty}[s\!-\!1] &&\text{1 group in flight (pipelined); free prior slot} \\[4pt]
&\quad \textbf{wait\_group}\langle0\rangle &&\text{drain last WGMMA} \\
&\quad \tilde G \gets G &&\text{stage }G\to\text{SMEM, row stride }264\ (\text{conflict‑free})\\
&\quad \mathbf{store}\ L^{(\beta_w,\nu)} \gets \tilde G &&\text{16‑byte (uint4) coalesced} \to \text{HBM (raw logits, } \textbf{no }\sigma)
\end{aligned}
$$

$A^{(\beta,\tau)}$ is read from $M$ stored **column‑major** ($B$‑contiguous) and $P^{(\tau,\nu)}$ **row‑major**
($QF$‑contiguous); both load MN‑major (WGMMA `trans-a = trans-b = 1`) with **no transpose**. There is **no
cluster / TMA multicast** ($\text{ClusterShape}=\langle1,1,1\rangle$, matching CUTLASS) — the skinny‑K win is the
schedule + epilogue, not multicast.

## Why each line is there (NCU‑driven; full ladder in `results.md`)

- **`wgmma.m64n256k16`** (one instruction, not $4\times$ `m64n64k16`) → reads $A^{(\beta,\tau)}$ **once** instead of 4×;
  the descriptor auto‑walks $N{=}256$ across the 4 contiguous B128 bricks via `LBO = 8192`, `SBO = 1024`. (L1/TEX $68.7\!\to\!50.9\%$.)
- **`setmaxnreg`** producer 32 / consumer 232 → the consumers get the register budget the WGMMA pipeline needs.
- **coalesced epilogue** ($\tilde G\to$SMEM $\to$ uint4 stores): scalar 2‑byte stores were **41 % of all stall samples**
  (`lg_throttle`) and pinned Tensor util to 47.7 %. This is the traffic shape of CUTLASS's STSM + TMA‑store epilogue.
  (Tensor $47.7\!\to\!72.8\%$.)
- **SMEM stage stride 264** (= 256 + 8 pad): the fragment store hit only 4 banks (8‑way conflict, 18 % excessive
  shared wavefronts); the pad makes $\text{bank}=\text{const}+\text{lane}$ (all 32 distinct) while keeping uint4 16‑byte
  alignment. (Tensor $72.8\!\to\!77.6\%$ = CUTLASS parity.)

## Result (full numbers in `results.md`)

| | hand `k_gate_ws` | CUTLASS `run_gate_ns` |
|---|---|---|
| warm (SKIP‑differential) | **1.27 ms** | ~1.26 ms |
| NCU cold duration | **1.66 ms** | 1.66 ms |
| Tensor pipe util | **77.6 %** | 77.8 % |

$\Rightarrow$ **parity reached.** This kernel is the drop‑in hand‑written body of the note's Grid 2; Phase 2 will fuse
$\sigma$ + repad ($L\to SL:[B,Q,F_p]$) into the $\mathbf{store}$ step to delete the note's Grid 3 (`k_sigpad_v2`).
