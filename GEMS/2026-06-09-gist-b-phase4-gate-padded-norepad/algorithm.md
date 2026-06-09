# GIST Phase 4 — pipeline in GPU Kernel & Layout Algebra notation

## Notation
Per [[A Notation for GPU Kernel and Layout Algebra]] §1:

| Form | Meaning |
|---|---|
| `A` — uppercase | tensor |
| `a` — lowercase | scalar / element / vector |
| `A^(…)` — superscript | tile partition (*which tile*) |
| `a_(…)` — subscript | element index (*which entry*) |
| `Ξ[…]` — Xi, uppercase Greek | **grid**: axes launched in parallel; cells are independent |
| `Φ[…]` — Phi, uppercase Greek | **stateful loop**: sequential, carries state across iterations |

Partition tags (lowercase Greek): **β** batch, **τ** gate-reduction (f′), **ρ** pool axis (g), **ν** output cols,
**χ** epilogue chunk. Axis *sizes* stay uppercase Latin: **B, F, Fₚ, D, Q**.

---

## Summary
**(a) What this is.** GIST (*Gated Information Summary & Transformation*) forward. For each batch $b$ it summarizes the
$F$ features into $Q$ query slots: a **gate** $SL=\sigma(\bar X\,P)$ scores every (query, feature) pair from the
per-feature means $\bar X$ through a learned low-rank projection $P$; a **pool** mixes the RMS-normalized features
$N=\hat X\odot W$ into each slot, $O=SL\,N$. Three stages: $\Xi_1$ `stats` (per-feature mean $M$ and inverse-RMS $r$),
$\Xi_2$ `gate` ($SL=\sigma(M P)$), $\Xi_3$ `pool` ($O=SL\,N,\ N=X r W$). Shape **B=1536, F=1497, Fₚ=1536, D=192,
Q=128**, bf16, RMSNorm.

**(b) The problem we solve.** Beat the reference by ≥30% with a **hand-written CUDA + inline-PTX** kernel (no cuBLAS,
no Triton, no CUTLASS in the hot path). Reference `gist_triton_fast` = **4.15 ms** bf16; the cuBLAS baseline = 3.40 ms.
This pipeline: **2.64 ms** (PASS, max_abs 0.0156) — **36% faster than Triton**, clearing the 30% bar (2.91) and the
20% bar (3.32). The win came from owning every stage in PTX: WGMMA `wgmma.mma_async` GEMMs, TMA `cp.async.bulk.tensor`
loads, an mbarrier producer/consumer ring, plus three fusions that delete HBM traffic — σ fused into the gate epilogue
(`tanh.approx`, free), the F→Fₚ pad fused into the gate (deletes the 0.78 ms repad copy), and $N=X r W$ fused into the
pool (so $N$, 0.9 GB each way, never touches HBM).

**How it runs.** All three kernels are **persistent**: the grid is one CTA per SM ($|Ξ|=S_M=132$); each cell runs an
outer $Φ[t]$ over grid-strided tiles with a continuous ring index $g$. Warp-specialized: producer $\mathcal P$ (TMA) ∥
consumers $\mathcal C_0,\mathcal C_1$ (WGMMA). Only $M,r$ (tiny) + $SL$ cross HBM; **$N$ never materialized**; $X$ read twice.

$$X \xrightarrow[\text{read }X;\ \text{write }M,r]{\ \Xi_1\ \texttt{stats}\ } (M,r) \xrightarrow[M\cdot P_p,\ \sigma,\ \text{pad}]{\ \Xi_2\ \texttt{gate}\ } \boxed{SL} \xrightarrow[N{=}X r W\ \text{fused};\ SL\cdot N]{\ \Xi_3\ \texttt{pool}\ } O$$

---

## Roofline / performance
This H100: bf16 ceiling **800 TFLOP/s**, HBM read **2,290 GB/s**, fp32 67 TFLOP/s; ridge = 800e12/2290e9 = **349 FLOP/byte**.

| kernel | time | FLOP | bandwidth | **BW util** | TFLOP/s | **MFU (÷800)** | AI (F/B) | bound |
|---|---|---|---|---|---|---|---|---|
| **Ξ₁ stats** | 0.425 ms | 1.32 GF | 2,113 GB/s | **92%** | 3.1 | 0.4% | 1.5 | **memory** |
| **Ξ₂ gate** | 1.380 ms | 928 GF | 869 GB/s | 38% | 672 | **84%** | 773 | **tensor** |
| **Ξ₃ pool** | 0.791 ms | 117 GF | 1,998 GB/s | **87%** | 148 | 18% | 74 | **memory** |
| total | 2.64 ms | | | | | | | |

- **FLOP**: stats 3/elem·B·F·D; gate 2·B·(Q·Fₚ)·Fₚ (executed, incl. K/N-pad); pool 2·B·Q·D·Fₚ + N-compute.
- **MFU vs the 800 TF ceiling** (not the 989 datasheet): gate **84%** (near-saturated; ~99% of peak *when* the tensor
  pipe is active — the 31% idle is the σ+pad+store epilogue, dominant stall = `barrier` 3.7/inst).
- **BW util** vs the measured **2,290 GB/s** read ceiling: stats **92%**, pool **87%** — both at their memory floors.
- Each kernel hugs a *different* ceiling: stats/pool memory, gate tensor. AI confirms (1.5 & 74 ≪ ; 773 ≫ ridge 349).

---

## Ξ₁ `stats` (M, rrms).  Memory-bound; SG-lane sub-group reduction over D.
$$M[b,f]=\tfrac1D\!\sum_d X_{b,f,d},\qquad r[b,f]=\Big(\tfrac1D\!\sum_d X_{b,f,d}^2+\epsilon\Big)^{-1/2}.$$

$$
\begin{aligned}
&\Xi[\,w\,] &&\text{grid: one warp }w;\ \text{owns }32/\!\mathrm{SG}\text{ rows }(\mathrm{SG}{=}8\Rightarrow 4) \\
&\quad \text{lane}=(\text{sub},sl),\ \ \rho = w\!\cdot\!(32/\mathrm{SG}) + \text{sub} &&\rho=(\beta,f)\ \text{the row this sub-group owns} \\
&\quad s\gets 0,\ \ s_2\gets 0 &&s,s_2:[1]\ \text{(fp32, per row)} \\
&\quad \Phi[\,i = sl,\,sl{+}\mathrm{SG},\,\dots,\,D/8\,] &&\text{loop the }D\text{ axis in int4 (8 bf16) strides} \\
&\quad\quad \mathbf{load}\ x_4 \gets X^{(\rho,i)} &&x_4:[\text{8 bf16}]\ \text{= one 16-byte int4} \\
&\quad\quad s \mathrel{+}= \textstyle\sum x_4,\ \ s_2 \mathrel{+}= \sum x_4^2 &&\text{accumulate }\Sigma x,\ \Sigma x^2 \\
&\quad \mathbf{reduce}_{\mathrm{SG}}\ s,\,s_2 &&\texttt{shfl.down}\times\log_2\mathrm{SG}\ \text{within the sub-group} \\
&\quad sl{=}0:\ \ \mathbf{store}\ M^{(\rho)}{=}s/D,\ \ r^{(\rho)}{=}(s_2/D{+}\epsilon)^{-1/2} &&\to\text{HBM},\ M\,(\text{col-major}),r:[B{\times}F]\ \text{(18 MB)}
\end{aligned}
$$

---

## Ξ₂ `gate` (padded, σ-fused).  Tensor-bound skinny-K GEMM; fuses σ AND the F→Fₚ pad.
$$SL=\sigma(M\cdot P_p),\quad M:[B,F]_{\text{col}},\ \ P_p:[F,\,Q\!\cdot\!F_p]\ (\text{per-q padded}),\ \ SL:[B,Q,F_p].\quad \sigma(x)=\tfrac12\tanh\tfrac x2+\tfrac12.$$
Tiles: $\bar\beta{=}128,\ \bar\nu{=}256,\ \bar\tau{=}64,\ |τ|{=}24$; ring depth $N_s{=}4$.

$$
\begin{aligned}
&\Xi[c],\ c=1\dots S_M &&\text{persistent: one CTA/SM};\ \ \mathcal P\,\|\,\mathcal C_0\,\|\,\mathcal C_1 \\[2pt]
&\;\mathcal P:\ \Phi[t = c,\,c{+}|Ξ|,\dots]\ \ t{=}(\beta,\nu) &&\text{grid-strided output tiles} \\
&\quad \Phi[\tau=1\dots|τ|]\ \ (g\ \text{continuous}) && \\
&\quad\quad \mathbf{wait}\ \text{empt}[g\bmod N_s] &&\text{ring slot free} \\
&\quad\quad \mathbf{TMA}\ \tilde A_g\gets M^{(\beta,\tau)},\ \ \tilde B_g\gets P_p^{(\tau,\nu)} &&\text{B128-swizzled ring}\to\mathbf{arrive}\ \text{full}[g\bmod N_s] \\[3pt]
&\;\mathcal C_w\ (w\!\in\!\{0,1\},\ \text{rows }\beta{+}64w):\ \Phi[t=c,\dots] && \\
&\quad\quad O\gets 0 &&O:[64\times\bar\nu]\ \text{fp32 acc} \\
&\quad\quad \Phi[\tau=1\dots|τ|]:\ \ \mathbf{wait}\ \text{full}[g];\ \ O \mathrel{+}= \tilde A_g^{(w)}\!\cdot\tilde B_g;\ \ \mathbf{arrive}\ \text{empt}[g] &&\texttt{wgmma.m64n256k16} \\
&\quad\quad \Phi[\chi=0\dots3]: && \text{chunked epilogue, }\bar\nu/4\text{ cols} \\
&\quad\quad\quad O_\chi \gets \sigma(O_\chi) &&\texttt{tanh.approx}\ (\text{1 SFU}) \\
&\quad\quad\quad O_\chi[\,g\!\ge\!F\,] \gets 0,\ \ g=(\nu\!\bmod\!F_p)+\chi\bar\nu/4+\cdot &&\textbf{pad-mask}\ (P_p{=}0\Rightarrow\sigma(0){=}0.5\ne0) \\
&\quad\quad\quad \mathbf{TMA\text{-}store}\ SL^{(\beta,\nu,\chi)}\gets O_\chi &&\text{16-byte aligned }(F_p{=}1536\,|\,256)\Rightarrow\textbf{no remap}
\end{aligned}
$$
$P_p$ is the one-time prepack $P[F,QF]\to P_p[F,Q\!\cdot\!F_p]$ (grid $Ξ[F\!\cdot\!Q]$, cached — harness holds $P$ constant).
Cost over the flat gate = ~3% extra compute ($\nu$: $QF\to Q F_p$); the repad grid $Ξ_3$ is **deleted**.

---

## Ξ₃ `pool` (N fused).  Memory-bound batched GEMM with inline RMSNorm value.
$$O[b,q,d]=\sum_g SL[b,q,g]\cdot N[b,g,d],\qquad N[b,g,d]=X[b,g,d]\cdot r[b,g]\cdot W[g,d].$$
Per batch a GEMM $[Q,D]=[Q,F_p]\!\cdot\![F_p,D]$; $A{=}SL$ K-major, $B{=}N$ built inline. $\bar\rho{=}64,\ |ρ|{=}24$.

$$
\begin{aligned}
&\Xi[c],\ c=1\dots S_M &&\text{persistent};\ \ \mathcal P\,\|\,\mathcal C_0\,\|\,\mathcal C_1;\ \ \Phi[t{=}\beta\ \text{(one batch/tile)}] \\[2pt]
&\;\mathcal P:\ \Phi[\rho=1\dots|ρ|]\ (g\ \text{cont.}):\ \ \mathbf{wait}\ \text{empt};\ \mathbf{TMA}\ \tilde A_g\!\gets\!SL^{(\beta,\rho)},\ \tilde X_g\!\gets\!X^{(\beta,\rho)} &&X\ \text{via 3D }[D,F,B]\ \text{OOB-0} \\
&\;\mathcal C:\ O\gets 0;\ \Phi[\rho=1\dots|ρ|]: && O:[Q\times D]\ \text{fp32 acc} \\
&\quad\quad \mathbf{wait}\ \text{full}[g];\ \ \tilde r \gets r^{(\beta,\rho)} &&r\ \text{slice (per-K-row scale)} \\
&\quad\quad N_g \gets \tilde X_g \cdot \tilde r \odot W^{(\rho)} &&\textbf{inline N, in-place, uint4-vectorized} \\
&\quad\quad O \mathrel{+}= \tilde A_g \cdot N_g;\ \ \mathbf{arrive}\ \text{empt}[g] &&\texttt{wgmma.m64n192k16} \\
&\quad \Phi[\chi]:\ \mathbf{TMA\text{-}store}\ O^{(\beta,\chi)} &&O:[B,Q,D]\to\text{HBM}
\end{aligned}
$$
$N$ never lands in HBM (vs the note's 4-grid which writes+reads $N$, 0.9 GB each way); $X$ is re-read instead.
