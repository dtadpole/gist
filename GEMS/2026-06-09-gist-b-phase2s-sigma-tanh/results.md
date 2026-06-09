# GIST Phase 2.s — sigmoid FUSED into the hand-written CUDA+PTX gate (tanh.approx), RMSNorm parity

**Branch B · impl slug `gist-b-phase2s-sigma-tanh` · 2026-06-09 · H100 (CUDA_VISIBLE_DEVICES=1).**
Design shape $B{=}1536,\ F{=}1497,\ D{=}192,\ Q{=}128$ ($QF{=}Q\!\cdot\!F{=}191616,\ F_p{=}1536$). Genuine bf16.
**RMSNorm oracle** (MAIN `e33f072`). Builds on [[gist-b-phase1-wgmma-persist-tmastore]] (raw-logit gate, 1.255ms).

## Result — PHASE 2.s MET (σ fused in the gate, below the 1.40ms floor, end-to-end correct)
| metric | value |
|---|---|
| **gate + σ (isolated)** | **1.24 ms** (raw gate 1.277 → σ ≈ **FREE**); target 1.40 (=1.26+0.14) **beaten by ~0.16** |
| **full-pipeline correctness** | **PASS — max_abs 0.015625, mean 5.4e-4** (== exact-σ CUTLASS-gate path) vs RMSNorm `ref-pytorch` |
| total pipeline | 3.658 ms |
| NCU `sm__pipe_tensor_op_hmma` | **78.6%** (= raw gate's 78% → σ fully overlapped, zero added tensor idle) |

---

## Design — in GPU Kernel Algebra Notation

**Operation.** Phase 2.s adds the elementwise activation $\sigma$ to the Phase-1 skinny-$K$ gate GEMM, *fused in the
epilogue* (the `Grid 2` GEMM of [[A Notation for GPU Kernel and Layout Algebra]] §4, now WITH its $\sigma$):
$$L \;=\; \sigma\!\big(M\cdot P\big),\qquad M:[B,F]\ \text{(col-major)},\quad P:[F,QF]\ \text{(row-major)},\quad L:[B,QF].$$
Contraction axis $K=F$ (tag $\tau$); output rows $B$ (tag $\beta$), cols $QF$ (tag $\nu$). $M$ = per-feature mean
(norm-invariant: identical for RMSNorm/LayerNorm). The activation uses a hardware-approx tanh identity:
$$\sigma(x)\;=\;\tfrac12\,\mathrm{tanh}\!\big(\tfrac{x}{2}\big)+\tfrac12 \quad(\text{1 SFU op: }\texttt{tanh.approx.f32}),$$
chosen over $1/(1{+}e^{-x})$ because the latter is a **2-SFU dependency chain** (`EX2`$\to$`RCP`) — see "why" below.

**Tiling & schedule — identical to Phase 1.** Output tile $\bar\beta\times\bar\nu=128\times256$; $\bar\tau=64$
($|\tau|{=}\lceil F/64\rceil{=}24$ k-blocks). PERSISTENT: grid $|\Xi|=S_M=132$ (one CTA/SM), each CTA $\Phi$-loops
grid-strided output tiles with one continuous TMA$\to$WGMMA pipeline ($g$ never resets). 3 warpgroups: producer
$\mathcal P$ (TMA), consumers $\mathcal C_0,\mathcal C_1$ (each owns 64 rows). $S{=}4$ ring; double-buffered chunk
staging $C_s:[2][\bar\beta][\bar\nu/4]$; chunked async TMA-store (`SM90_TMA_STORE`).

$$
\begin{aligned}
&\Xi[c],\ c{=}1\dots S_M:\quad \mathcal P \ \parallel\ \mathcal C_0 \ \parallel\ \mathcal C_1 &&\text{persistent (1 CTA/SM), warp-specialized}\\[4pt]
% ---- PRODUCER ----
&\textbf{producer } \mathcal P\ (\text{tid}\ge256):\ \Phi[t{=}c,c{+}|\Xi|,\dots],\ \ \beta{=}t\bmod|\beta|,\ \nu{=}t\,\mathrm{div}\,|\beta| && \text{TMA engine}\\
&\quad \Phi[\tau{=}1\dots|\tau|]\ \ (g:\ \text{continuous across tiles}) && \\
&\quad\quad \mathbf{wait}\,\text{empt}[g\bmod N_s] && \text{ring slot free (skip first }N_s)\\
&\quad\quad \mathbf{arrive.expect\_tx}\,\text{full}[g\bmod N_s],\ \ \text{tx}{=}(2{+}4)\!\cdot\!64\!\cdot\!64\!\cdot\!2\,\text{B} && \\
&\quad\quad \mathbf{TMA}\ \tilde A_s\!\gets\!A^{(\beta,\tau)}(2\,\text{box}),\ \tilde P_s\!\gets\!P^{(\tau,\nu)}(4\,\text{box}) && \texttt{cp.async.bulk.tensor.2d}\!\to\!\text{B128 SMEM}\\[4pt]
% ---- CONSUMERS ----
&\textbf{consumer}_w\ \mathcal C_w\ (w\!\in\!\{0,1\}):\ \Phi[t{=}c,c{+}|\Xi|,\dots]\ \ (\beta,\nu) && \mathcal C_w\text{ owns 64-row half }\beta_w\\
&\quad\quad O\gets 0;\ \ \Phi[\tau{=}1\dots|\tau|]:\ \mathbf{wait}\,\text{full}[g\bmod N_s];\ O \mathrel{+}= \tilde A_s^{(w)}\!\cdot \tilde P_s;\ \mathbf{arrive}\,\text{empt}[g\bmod N_s] &&\text{4×wgmma.m64n256k16}\\
&\quad\quad \boxed{O \gets \sigma(O)\ \text{via}\ \texttt{tanh.approx.f32}} &&\text{\textbf{Phase 2.s: fused, in registers}}\\
&\quad\quad \Phi[\chi{=}0\dots3]:\ \text{stage}\,O[:,64\chi{:}]\!\to\!C_s[\chi\bmod2];\ \mathbf{TMA\text{-}store}\,L^{(\beta_w,\nu,\chi)} &&\text{overlaps tile }t{+}1\text{ mainloop}
\end{aligned}
$$

The **producer** $\mathcal P$ and **consumers** $\mathcal C_0,\mathcal C_1$ are all hand-written (inline PTX): $\mathcal P$
streams $A{=}M$ (2 boxes) + $P$ (4 boxes) into the $N_s{=}4$ ring via `cp.async.bulk.tensor.2d`, handing off through
the `full`/`empt` mbarriers; the consumers run the WGMMA mainloop + the σ-fused chunked TMA-store. Only the boxed
line is new vs Phase 1: $\sigma$ on the fp32 accumulator $O$ (in registers, no smem round-trip), interleaved into the
chunked store loop so its SFU work overlaps the in-flight async TMA stores. Full algorithm: `gate-sigma-gemm.md`.

**Why `tanh.approx`, not `__expf` (NCU-grounded).** The gate is tensor-pipe-bound (78%); $\sigma$ can only live in
the consumer epilogue, where the tensor pipe is otherwise idle. With `__expf` the $\sigma$ is a long 2-SFU chain
(`FMA·log2e`$\to$`EX2`$\to$`FADD`$\to$`RCP`, ~48 cyc) on the **serial** epilogue critical path (σ$\to$store); it
extends the tensor-idle window → tensor 78%$\to$57%, **+0.35 ms** (XU only 9.6% — it was *latency*, not SFU
throughput; long_scoreboard 0.99 = not memory). `tanh.approx.f32` is **1 SFU + 1 FMA** (~28 cyc); the short chain
fits *inside* the epilogue's existing store/barrier latency → tensor recovers to **78.6%** → $\sigma$ free.

## Correctness — genuine, two independent checks
- **Standalone:** $\sigma_{\tanh}$ vs exact $\sigma$ → max $|$fp32$|$ 3.9e-6, max $|$bf16$|$ **3.9e-3 = 1 bf16 ULP** ($<$ 1e-2 tol).
- **End-to-end (harness `ref-pytorch`, RMSNorm):** hand gate+σ → pool → **max_abs 0.015625**, *identical* to the
  exact-σ CUTLASS-gate path → the approximation adds zero error beyond bf16 rounding.

## Also in this snapshot — LayerNorm→RMSNorm/F=1497 port (required to run end-to-end on the live oracle)
$M{=}\mathrm{mean}_D X$ (unchanged), $R{=}\mathrm{rrms}{=}\mathrm{rsqrt}(\mathrm{mean}_D(X^2){+}\epsilon)$ (no centering),
$N{=}X\cdot\mathrm{rrms}\cdot W$ with $W{:}[F,D]$ per-(feature,channel) (was $\gamma[D]/\beta[D]$). 3 inputs $(X,P,W)$.
Port bug fixed: $W$ must be indexed per-feature $W[f,\cdot]$ (offset $f\!\cdot\!D$), not $W[0,\cdot]$.

## Measured dead-ends (evidence, not assumption)
- Dedicated epilogue-WG for σ: 2.60 ms even optimized (smem round-trip + mbar-spin handshake; long_scoreboard 9.25).
- Upfront σ-all-acc (128-deep ILP): 1.76 ms — kills σ/store overlap (store waits for all 128 σ).
- S=3 ring: ÷3 magic-multiply + dyn-indexed parity arrays → LOCAL memory; power-of-2 S=4 keeps them in registers.

## Repro
```bash
cd ~/gist/cuda
# end-to-end correctness (RMSNorm ref-pytorch golden):
GIST_HANDGATE=1 GIST_PHASE1RAW=1 GIST_FUSESIG=1 GIST_CU=<this>/gist.cu CUDA_VISIBLE_DEVICES=1 ./run_gist.sh <rev> big
#   -> gen-cuda correctness.passed=true, max_abs 0.015625
# gate+σ isolated: direct binary + /tmp/gist_env_1497.sh: (stats+gate+σ,SKIP_POOL) − (stats,SKIP_GATE+SKIP_POOL).
```
Kernel: `k_gate_hand_persist` (dosig=1 ⇒ σ fused via `tanh.approx` in the epilogue). `sigmoidf` = tanh identity.
NCU: `--kernel-name "regex:k_gate_hand_persist$"`.
