# GIST gate GEMM **with fused sigmoid** — hand-written CUDA+PTX (Phase 2.s)

Hand-written body of **Grid 2 — `run_gate`** *with its activation* of
[[A Notation for GPU Kernel and Layout Algebra]] (§4 GIST). CUTLASS is replaced by raw CUDA + inline PTX
(Hopper WGMMA + TMA + mbarriers). **Phase 2.s fuses $\sigma$ into the GEMM epilogue**, so this single kernel emits
the **sigmoid-gated** output $L=\sigma(M\cdot P)$ directly — folding the note's Grid 2 ($M\cdot P$) and the
activation of Grid 3 (`k_sigpad`'s $\sigma$) into one. (Only the *pad* $L\to SL:[B,Q,F_p]$ remains downstream =
Phase 2.t.) Builds on Phase 1 [[gist-b-phase1-wgmma-persist-tmastore]] (raw-logit persistent gate, 1.255 ms).

## Tensors

$$X:[B,F,D],\quad M:[B,F],\quad P:[F,QF],\quad L:[B,QF]\equiv[B,Q,F],\qquad \sigma=\text{sigmoid}.$$

$M$ = `x_input_mean` (per-feature mean of $X$ over $D$; **norm-invariant** — same under RMSNorm/LayerNorm),
$P$ = `proj_params`, $L$ = `sigmoid(linear_output)`. Design shape $B{=}1536,\ F{=}1497,\ F_p{=}1536,\ D{=}192,\ Q{=}128$, bf16.

The activation is a **hardware-approx tanh** identity (1 SFU op, `tanh.approx.f32`):
$$\boxed{\ \sigma(x)\;=\;\tfrac12\,\tanh\!\big(\tfrac{x}{2}\big)+\tfrac12\ }\qquad(\text{vs }1/(1{+}e^{-x})=\text{2-SFU chain EX2}\to\text{RCP}).$$

As a GEMM this is the **skinny-K** problem:
$$\underbrace{M_{\text{dim}}=B=1536}_{\text{rows }\beta}\ ,\quad
\underbrace{N_{\text{dim}}=QF=191616}_{\text{cols }\nu,\ n=qF+g}\ ,\quad
\underbrace{K_{\text{dim}}=F=1497\ (K_{\text{pad}}=F_p=1536)}_{\text{contraction }\tau\ (f')} .$$

## Partition (tags)

| tag | axis | block size | #blocks |
|---|---|---|---|
| $\beta$ | batch / $M_{\text{dim}}$ rows of $L$ | $\bar\beta = 128$ | $\lvert\beta\rvert=\lceil B/128\rceil=12$ |
| $\nu$ | $N_{\text{dim}}=QF$ cols ($n=qF+g$) | $\bar\nu = 256$ | $\lvert\nu\rvert=\lceil QF/256\rceil=749$ |
| $\tau$ | gate reduction $f'$ ($F$/$K$ axis) | $\bar\tau = 64$ | $\lvert\tau\rvert=\lceil F/64\rceil=24$ |

Output tiles $\lvert\beta\rvert\,\lvert\nu\rvert = 12\times749 = 8988$. Each $\bar\tau{=}64$ k-block = 4 WGMMA `m64n256k16` steps.

Tiles (superscript = which tile, subscript = local):
$$A^{(\beta,\tau)}=M^{(\beta,\tau)}:[128\times64],\quad P^{(\tau,\nu)}:[64\times256],\quad L^{(\beta,\nu)}:[128\times256].$$
Per-consumer accumulator $G_w:[64\times256]$ fp32 = `acc[128]` regs/thread (the `m64n256` acc; bit-identical to 4
concatenated `m64n64` acc[32], local index $g=n_2\!\cdot\!8+a$).

## Kernel — PERSISTENT, warp-specialized cooperative WGMMA, σ fused in a chunked TMA-store epilogue

The grid is **NOT** the output tiles: it is **one CTA per SM**, $\lvert\Xi\rvert=S_M=132$. Each CTA $\Phi$-loops a
grid-strided set of tiles, and **one continuous TMA→WGMMA pipeline runs across tile boundaries** (ring index $g$
never resets ⇒ no per-tile fill bubble). 3 warpgroups (384 threads, no `setmaxnreg` — base $65536/384=170\ge$ the
WGMMA's 154-reg need): producer $\mathcal P$ (tid≥256), consumers $\mathcal C_0,\mathcal C_1$ (tid 0–255; $\mathcal C_w$
owns the 64-row half $\beta_w$). Shared per CTA: ring $\tilde A_s,\tilde P_s$ with $N_s{=}4$ stages; mbarriers
$\text{full}[N_s],\text{empt}[N_s]$ (phase-parity); double-buffered chunk staging $C_s:[2][128][64]$.

$$
\begin{aligned}
&\Xi[c],\ c=1\dots S_M:\quad \mathcal P \ \parallel\ \mathcal C_0 \ \parallel\ \mathcal C_1 &&\text{persistent grid (1 CTA/SM)}\\[4pt]
% ---------- producer ----------
&\;\textbf{producer } \mathcal P:\ \ \Phi[t=c,\,c+\lvert\Xi\rvert,\dots]\ \ \ (\beta,\nu)\!\leftarrow\! t &&\beta=t\bmod\lvert\beta\rvert,\ \nu=t\,\mathrm{div}\,\lvert\beta\rvert\\
&\quad \Phi[\tau=1\dots\lvert\tau\rvert]\ \ (g:\text{continuous}) && \\
&\quad\quad \textbf{wait } \text{empt}[g\bmod N_s] &&\text{ring slot free (skip first }N_s)\\
&\quad\quad \textbf{arrive.expect\_tx } \text{full}[g\bmod N_s] &&\text{tx} = (2{+}4)\!\cdot\!64\!\cdot\!64\!\cdot\!2\ \text{bytes}\\
&\quad\quad \mathbf{TMA}\ \tilde A_s \gets A^{(\beta,\tau)}\ (2\ \text{boxes}),\ \ \tilde P_s \gets P^{(\tau,\nu)}\ (4\ \text{boxes}) &&\texttt{cp.async.bulk.tensor.2d}\ \to\ \text{B128-swizzled SMEM}\\[4pt]
% ---------- consumers ----------
&\;\textbf{consumer}_w\ \mathcal C_w\ (w\!\in\!\{0,1\}):\ \ \Phi[t=c,\,c+\lvert\Xi\rvert,\dots] && \\
&\quad\quad G_w \gets 0 &&G_w:[64\times256]\ \text{fp32 acc[128]}\\
&\quad\quad \Phi[\tau=1\dots\lvert\tau\rvert]: && \\
&\quad\quad\quad \textbf{wait } \text{full}[g\bmod N_s] && \\
&\quad\quad\quad G_w \mathrel{+}= \tilde A_s^{(\beta_w)}\cdot\tilde P_s &&4\times\texttt{wgmma.mma\_async.m64n256k16}\ (\text{MN-major, B128 desc})\\
&\quad\quad\quad \textbf{commit};\ \textbf{wait\_group}\langle0\rangle;\ \textbf{arrive } \text{empt}[g\bmod N_s] && \\[3pt]
&\quad\quad \boxed{G_w \gets \sigma(G_w)}\ \ \text{(in registers, } \texttt{tanh.approx.f32}) &&\textbf{Phase 2.s: fused activation}\\
&\quad\quad \Phi[\chi=0\dots3]\ \ (\text{chunked TMA-store, }cb=\chi\bmod2): && \bar\nu/4=64\text{-col chunks}\\
&\quad\quad\quad \text{if }\chi\!\ge\!2:\ \textbf{store\_wait}\langle1\rangle &&\text{chunk }\chi{-}2\text{ drained}\to C_s[cb]\text{ free}\\
&\quad\quad\quad \tilde C[cb]\gets \text{bf16}(G_w[:,64\chi{:}64\chi{+}64]) &&\text{coalesced }\texttt{bf16x2}\text{ stage (σ already applied)}\\
&\quad\quad\quad \textbf{fence.proxy.async};\ \ \mathbf{TMA\text{-}store}\ L^{(\beta_w,\nu,\chi)}\gets \tilde C[cb] &&\texttt{cp.async.bulk.tensor.2d.global.shared};\ \text{overlaps tile }t{+}1
\end{aligned}
$$

**Layouts / descriptors.** $A^{(\beta,\tau)}$ reads $M$ **column-major** ($B$-contiguous); $P^{(\tau,\nu)}$ **row-major**
($QF$-contiguous). Both load MN-major (WGMMA `trans-a=trans-b=1`, no transpose). WGMMA descriptors: $A$ `LBO=128,
SBO=1024`; $B$ `LBO=8192, SBO=1024` (the `m64n256` descriptor auto-walks $N{=}256$ across 4 contiguous $64\times64$
canonical-B128 bricks). TMA loads use `cuTensorMapEncodeTiled` SW128, box $64\times64$, OOB-fill 0 (handles the
$K$-pad $F\!\to\!F_p$ and the $N$-tail of $QF$ for free). TMA store: SWIZZLE_NONE, box $64\times128$, OOB-clips the
$QF$ tail. **No cluster / multicast** ($\text{ClusterShape}=\langle1,1,1\rangle$) — the skinny-K win is schedule+epilogue.

**The only Phase-2.s change** vs Phase 1 is the boxed $G_w\gets\sigma(G_w)$ — applied to the fp32 accumulator **in
registers** (no SMEM round-trip), interleaved into the chunked store loop so its SFU work overlaps the in-flight
async TMA stores of the previous chunks.

## Why each line (NCU-driven; full ladder in `results.md`)

- **persistent + continuous pipeline + chunked TMA-store** (Phase 1): producer prefetches tile $t{+}1$ while
  consumers run tile $t$'s epilogue ⇒ no per-tile fill bubble; deep $N_s{=}4$ ring (hides TMA, `long_scoreboard` 2.0)
  + 32 KB double-buffered chunk staging all fit in 224 KB. Tensor pipe 72.5→**78.7%**, gate 1.34→**1.255 ms**.
- **`σ` via `tanh.approx.f32`, in registers, interleaved** (Phase 2.s): the gate is tensor-bound (78%), so $\sigma$
  can only sit in the consumer epilogue (tensor otherwise idle there). With `__expf` (2-SFU chain `EX2→RCP`, ~48 cyc)
  $\sigma$ lay on the **serial** epilogue critical path (σ→store) and extended the tensor-idle window → tensor
  78→57%, **+0.35 ms** (NCU: XU only 9.6% — *latency*, not throughput; `long_scoreboard` 0.99 = not memory).
  `tanh.approx` (**1 SFU + 1 FMA**, ~28 cyc) fits inside the epilogue's existing store/barrier latency → tensor
  recovers to **78.6%** → **$\sigma$ adds ≈ 0 ms.** Accuracy: 1 bf16 ULP vs exact σ (verified end-to-end, max_abs
  0.0156 = exact-σ CUTLASS-gate path) — well inside the 1e-2 harness tolerance.

## Result (full numbers in `results.md`)

| metric | hand `k_gate_hand_persist` (dosig=1) |
|---|---|
| **gate + σ (isolated, SKIP-differential)** | **1.24 ms** (raw gate 1.277 → σ ≈ free; target 1.40 beaten) |
| Tensor pipe util (NCU) | **78.6 %** (= raw gate's 78%) |
| full pipeline correctness (RMSNorm `ref-pytorch`) | **PASS, max_abs 0.015625** (= exact-σ CUTLASS-gate path) |
| total pipeline | 3.658 ms |

$\Rightarrow$ **Phase 2.s met**: $L=\sigma(M\cdot P)$ in one hand-written WGMMA kernel, σ fused for free, below the
1.26+0.14 = 1.40 ms floor, verified correct. Next = Phase 2.t (fold the pad $L\to SL:[B,Q,F_p]$ into the store).
