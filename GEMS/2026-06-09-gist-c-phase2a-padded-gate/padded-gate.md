# GIST padded gate — hand-written CUDA+PTX (Phase 2.a, pre-padded weight)

Hand-written fusion of **Grid 2 (`run_gate_ns`) + Grid 3 (`k_sigpad_v2` repad)** of
[[A Notation for GPU Kernel and Layout Algebra]] (§4 GIST). The gate writes the pool's **padded**
layout $SL:[B,Q,F_p]$ **directly**, deleting the separate repad grid — not by remapping inside the
epilogue (the rejected `win_shift`, +0.41 ms) but by **pre-padding the weight once**: $P\to P_p$. The
gate then becomes a plain GEMM over the padded $N$-axis whose store is naturally aligned.

Oracle: **RMSNorm**, design shape $B{=}1536,\ F{=}1497,\ F_p{=}1536,\ D{=}192,\ Q{=}128$, bf16,
$\sigma=\text{sigmoid}$. (Note $F_p=\lceil F/64\rceil\cdot64=1536$.)

## Tensors (notation as in the obsidian note)

$$
X:\,[B,F,D],\quad M:\,[B,F],\quad R:\,[B,F],\quad W:\,[F,D],\quad N:\,[B,F_p,D],
$$
$$
P:\,[F,QF],\quad P_p:\,[F,QF_p],\quad L,\,SL:\,[B,QF_p]\equiv[B,Q,F_p],\quad O:\,[B,Q,D].
$$

$M$ = `x_input_mean` $=\operatorname{mean}_d X$, $R$ = `rrms` (RMSNorm reciprocal-RMS), $W$ = RMSNorm
weight, $P$ = `proj_params` (gate weight), $O$ = pooled output. $n=qF_p+f$ indexes the padded $N$-axis.

## GIST forward in kernel algebra (4 grids, single `kernel_run`)

$$
\begin{aligned}
\textbf{(0) reformat (once, cached)}\quad
&P_p[f',\,qF_p+f] \;=\; \begin{cases} P[f',\,qF+f] & f<F\\[2pt] 0 & F\le f<F_p\end{cases}
&&\texttt{k\_ppad}\;:\;P:[F,QF]\to P_p:[F,QF_p]\\[6pt]
\textbf{(1) stats}\quad
&M[b,f] = \tfrac1D\!\sum_{d} X[b,f,d],\qquad
R[b,f] = \big(\tfrac1D\!\sum_{d} X[b,f,d]^2+\varepsilon\big)^{-\frac12} \\
&N[b,f,d] = X[b,f,d]\cdot R[b,f]\cdot W[f,d],\qquad N[b,f\!\ge\!F,d]=0
&&\texttt{k\_stats}\\[6pt]
\textbf{(2) gate}\quad
&L[b,q,f] \;=\; \sum_{f'} M[b,f']\,P_p[f',\,qF_p+f]
&&\texttt{k\_gate\_ws}\ \ (L=M\!\cdot\!P_p)\\
&\Rightarrow L[b,q,f]=0\ \ \text{for}\ F\le f<F_p\ \ (\text{multiplies the 0-padded }P_p\text{ cols})\\[6pt]
\textbf{(3) sigmoid}\quad
&SL[b,q,f] = \begin{cases}\sigma\!\big(L[b,q,f]\big) & f<F\\[2pt] 0 & F\le f<F_p\end{cases}
&&\texttt{k\_sig\_valid}\ (\text{valid cols only; pad stays raw }0)\\[6pt]
\textbf{(4) pool}\quad
&O[b,q,d] = \sum_{f=0}^{F_p-1} SL[b,q,f]\,N[b,f,d] \;=\; \sum_{f<F} SL\,N
&&\texttt{run\_pool}\ (SL\!\cdot\!N)
\end{aligned}
$$

The pad cols never corrupt the pool: $SL[\,F\!:\!F_p\,]=0$ (we $\sigma$ only $f<F$, **not** $\sigma(0)=\tfrac12$),
and $N[\,F\!:\!F_p\,]=0$, so $\sum_{f\ge F}SL\cdot N=0$. (0) runs once — $P$ is a **constant weight**
(harness `HARNESS_WEIGHT_INPUTS`), so $P_p$ is cached and is **not** in the timed loop.

## Gate as a GEMM — the **padded** skinny-K problem

$$
\underbrace{M_{\text{dim}}=B=1536}_{\text{rows }\beta}\,,\quad
\underbrace{N_{\text{dim}}=QF_p=196608}_{\text{cols, }n=qF_p+f}\,,\quad
\underbrace{K_{\text{dim}}=F\ (K_{\text{pad}}=F_p=1536)}_{\text{contraction }f'} .
$$

The win vs Phase 1: $QF_p=196608$ is an **exact multiple of $\bar\nu=256$** ($|\nu|=768$) and of $64$,
so every output tile / store is 16-byte aligned with **no odd-$F$ remap** — the padding is paid as
$\sim$3 % extra compute ($QF\!\to\!QF_p$), not as an epilogue scatter.

## Partition (tags)

| tag | axis | block size | #blocks |
|---|---|---|---|
| $\beta$ | batch / $M_{\text{dim}}$ rows of $L$ | $\bar\beta=128$ | $\lvert\beta\rvert=\lceil B/128\rceil=12$ |
| $\nu$ | padded $N_{\text{dim}}=QF_p$ ($n=qF_p+f$) | $\bar\nu=256$ | $\lvert\nu\rvert=QF_p/256=768$ |
| $\tau$ | gate reduction $f'$ ($F$/$K$ axis) | $\bar\tau=64$ | $\lvert\tau\rvert=F_p/64=24$ |

$$A^{(\beta,\tau)}=M^{(\beta,\tau)}:[128\times64],\quad P_p^{(\tau,\nu)}:[64\times256],\quad
L^{(\beta,\nu)}:[128\times256],\quad G:[128\times256]\ \text{fp32 acc}.$$

## Kernel (2) — warp-specialized cooperative WGMMA, store to padded $SL$

One grid cell computes one tile $L^{(\beta,\nu)}$. 3 warpgroups: producer $\Phi^{\text{prod}}$ (TMA),
2 cooperative consumers $\Phi^{\text{cons}}_{0,1}$ (WGMMA, each owns a 64-row half $\beta_w$), an
$N_s{=}4$-deep async ring $(\tilde A_s,\tilde P_s)$ gated by `full`/`empty` mbarriers.

$$
\begin{aligned}
&\Xi[\beta,\nu] &&\text{grid cell: one output tile }L^{(\beta,\nu)}:[128\times256]\\[2pt]
&\;\textbf{producer } \Phi^{\text{prod}}[\tau],\ \tau=1,\dots,24 &&\texttt{setmaxnreg.dec}\ (32\ \text{regs})\\
&\quad \textbf{wait } \text{empty}[s],\ s=\tau\bmod N_s &&\text{ring slot free}\\
&\quad \mathbf{TMA}\ \tilde A_s\gets A^{(\beta,\tau)},\ \ \tilde P_s\gets P_p^{(\tau,\nu)} &&\texttt{cp.async.bulk.tensor.2d}\to\text{B128-swizzled SMEM}\\
&\quad \textbf{arrive } \text{full}[s] &&\text{(TMA tx-count; }K/N\text{ tails auto-zeroed)}\\[4pt]
&\;\textbf{consumer}_w\ \Phi^{\text{cons}}[\tau],\ w\in\{0,1\} &&\texttt{setmaxnreg.inc}\ (232\ \text{regs})\\
&\quad G\gets 0 &&G:[64\times256]\ \text{fp32 (this }\beta_w\text{ half)}\\
&\quad \textbf{wait } \text{full}[s] &&\text{staged }(\tilde A_s,\tilde P_s)\ \text{ready}\\
&\quad G\mathrel{+}=\tilde A_s^{(\beta_w)}\,\tilde P_s &&\texttt{wgmma.m64n256k16}\times(\bar\tau/16)\ \text{(one async group)}\\
&\quad \textbf{wait\_group}\langle1\rangle;\ \textbf{arrive } \text{empty}[s\!-\!1] &&\text{1 group in flight; free prior slot}\\[4pt]
&\quad \textbf{wait\_group}\langle0\rangle;\ \ \tilde G\gets G &&\text{drain; stage }G\to\text{SMEM (stride 264, conflict-free)}\\
&\quad \mathbf{store}\ SL^{(\beta_w,\nu)}\gets \tilde G &&\textbf{16-byte uint4 coalesced},\ \ SL[\,b,\,qF_p+f\,]\ \text{(raw }L\text{, no }\sigma)
\end{aligned}
$$

Because $\nu$ tiles the **padded** $QF_p$ (multiple of 256), the store address $b\cdot QF_p+n$ is always
uint4-aligned — the same coalesced epilogue as Phase 1, now landing $SL:[B,Q,F_p]$ with **no win_shift,
no scatter**. $A$ from $M$ column-major, $P_p$ row-major; both MN-major (`trans-a=trans-b=1`),
$\text{ClusterShape}=\langle1,1,1\rangle$.

## Why pre-padded weight, not the inline remap

| approach | gate | how the padded cols are produced |
|---|---|---|
| **PPAD (this)** | **1.28 ms** | weight $P_p$ pre-padded once; GEMM over $QF_p$, contiguous aligned store |
| win_shift (removed) | 1.69 ms | flat-$N$ tile remapped to $q$-segments via register byte-shuffle (`funnelshift`) — +0.41 ms |

$1.28 \approx 1.26\ (\text{flat gate, Phase 1}) + {\sim}0.02$ for the $+3\%$ ($QF\!\to\!QF_p$) extra MACs:
**padding is nearly free.** The remap had no value and was deleted.

## Result (full numbers in `results.md`)

| kernel | ms | | total | 3.54 ms |
|---|---|---|---|---|
| stats | 0.87 | | vs default (CUTLASS) | 3.50 ms |
| **gate (padded $SL$)** | **1.28** | | vs Triton | 4.22 ms |
| sigmoid | 0.57 | | correctness (big) | PASS max_abs 0.0156 |
| pool | 0.80 | | | |

$\Rightarrow$ Grid 2+3 fused into one padded GEMM. **Phase 2.b** folds Grid 3 ($\sigma$) into the
$\mathbf{store}$ step (the `SIG` template hook + branch-B `tanh.approx`), dropping `k_sig_valid` →
$\;0.87+1.28+0.80\approx\mathbf{2.95}$ ms (the 30 % bar).
