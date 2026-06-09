# GIST Phase 1 — hand-written CUDA+PTX gate GEMM at CUTLASS parity

**Branch B · impl slug `wgmma-persist-tmastore` · 2026-06-08 · GPU H100 (CUDA_VISIBLE_DEVICES=1).**
Design shape $B{=}1536,\ F{=}1491,\ D{=}192,\ Q{=}128$ ($QF{=}Q\!\cdot\!F{=}190848$, $F_p{=}1536$). Genuine bf16.

## Result — PHASE 1 MET
| metric | value |
|---|---|
| **gate kernel (isolated)** | **1.255 ms** (= stats+gate 2.104 − stats 0.849; stable 1.255–1.259 across runs/grids) |
| CUTLASS gate reference | 1.26 ms → **PARITY** |
| correctness | PASS, max_abs 0.03125, mean 3.2e-6 (full-O verified: hand gate → sigpad → CUTLASS pool) |
| NCU `sm__pipe_tensor_op_hmma` | **78.7%** (≥ CUTLASS MFU; non-persistent v17 was 72.5%) |
| NCU `stalled_long_scoreboard` | 2.0 cyc/instr (TMA hidden) |
| total pipeline | 3.670 ms |

Raw logits only (no $\sigma$) — Phase 2 (sigmoid + padded-SL fusion in the epilogue) is deferred by design.
NO CUTLASS in the gate: hand-written WGMMA + TMA + inline PTX. `GIST_HANDGATE` path; this is the default
(`GIST_HGNONPERSIST` falls back to the non-persistent variant, gate 1.34 ms).

---

## Design — in GPU Kernel Algebra Notation

**Operation.** The gate is a single skinny-$K$ GEMM producing raw logits (the `Grid 2` GEMM of
[[A Notation for GPU Kernel and Layout Algebra]] §4, with $\sigma$ *removed* — Phase 1):
$$L \;=\; M\cdot P,\qquad M:[B,F]\ \text{(col-major)},\quad P:[F,QF]\ \text{(row-major)},\quad L:[B,QF]\ \text{(raw logits)}.$$
Contraction axis $K=F$ (the gate reduction $f'$, tag $\tau$); output rows $B$ (tag $\beta$), output cols
$QF$ (tag $\nu$). $M$ is the per-feature mean from the stats grid; $P$ = `proj_params`.

**Tiling.** Output tile $\bar\beta\times\bar\nu = 128\times256$; reduction block $\bar\tau=64$, so
$|\tau|=\lceil F/64\rceil=24$ k-blocks. $|\beta|=B/\bar\beta=12$, $|\nu|=\lceil QF/\bar\nu\rceil=746$;
total output tiles $|\beta|\,|\nu| = 8952$. Each $\bar\tau{=}64$ k-block = 4 WGMMA `m64n256k16` steps.

**The kernel is PERSISTENT** — the grid is **not** the output tiles; it is one CTA per SM
($|\Xi|=S_M=132$). Each CTA $\Phi$-loops over a grid-strided set of output tiles, and **one continuous
TMA→WGMMA pipeline runs across tile boundaries** (ring index $g$ never resets). Three warp-groups per
CTA: one producer $\mathcal P$, two consumers $\mathcal C_0,\mathcal C_1$ (each owns a 64-row sub-tile).

Shared per CTA: ring buffers $A_s,B_s$ with $S{=}4$ stages; mbarriers $\text{full}[S],\text{empt}[S]$
(phase-parity handoff); double-buffered chunk staging $C_s:[2][\bar\beta][\bar\nu/4]$.

$$
\begin{aligned}
&\Xi[c],\ \ c=1\dots S_M &&\text{persistent grid: one CTA per SM ($S_M{=}132$)}\\[2pt]
&\quad \mathcal P\ \parallel\ \mathcal C_0\ \parallel\ \mathcal C_1 &&\text{warp-specialize: 1 producer WG, 2 consumer WGs}\\[4pt]
% ---- producer ----
&\quad \mathcal P:\ \ \Phi[t=c,\,c+|\Xi|,\dots] &&\text{grid-strided tiles } t{=}(\beta,\nu),\ \beta{=}t\bmod|\beta|,\ \nu{=}t\,\mathrm{div}\,|\beta|\\
&\quad\quad \Phi[\kappa=1\dots|\tau|]\ \ (g\!:\!=\!\text{continuous}) && \\
&\quad\quad\quad \mathbf{wait}\ \text{empt}[g\bmod S] &&\text{ring slot free}\\
&\quad\quad\quad \mathbf{TMA}\ A_s[g\bmod S]\gets M^{(\beta,\kappa)} &&[\bar\beta\times\bar\tau]\ \text{(2 boxes, col-major / trans-a)}\\
&\quad\quad\quad \mathbf{TMA}\ B_s[g\bmod S]\gets P^{(\kappa,\nu)} &&[\bar\tau\times\bar\nu]\ \text{(4 boxes)}\ \to\ \mathbf{arrive}\ \text{full}[g\bmod S]\\[4pt]
% ---- consumers ----
&\quad \mathcal C_w\ (w\!\in\!\{0,1\},\ \text{owns rows } \beta{+}64w):\ \ \Phi[t=c,\,c+|\Xi|,\dots] && \\
&\quad\quad O\gets 0 &&O:[64\times\bar\nu]\ \text{(fp32 acc[128] in regs)}\\
&\quad\quad \Phi[\kappa=1\dots|\tau|] && \\
&\quad\quad\quad \mathbf{wait}\ \text{full}[g\bmod S] && \\
&\quad\quad\quad O \gets O + A_s^{(w)}\!\cdot B_s &&\text{4}\times\text{wgmma.m64n256k16 (MN-major, B128 swizzle desc)}\\
&\quad\quad\quad \mathbf{arrive}\ \text{empt}[g\bmod S] && \\
&\quad\quad \Phi[\chi=0\dots 3]\ \ \text{(chunked TMA-store epilogue)} &&\text{4 chunks of } \bar\nu/4{=}64\text{ cols}\\
&\quad\quad\quad \text{stage } O[:,\,64\chi:64\chi{+}64]\to C_s[\chi\bmod 2] &&\text{regs}\to\text{smem}\\
&\quad\quad\quad \mathbf{TMA\text{-}store}\ L^{(\beta,\nu,\chi)}\gets C_s[\chi\bmod 2] &&\text{async bulk; overlaps tile } t{+}1\text{'s mainloop}
\end{aligned}
$$

**Why persistent + chunked-store is the parity unlock (all NCU-measured).**
- The op is **tensor-pipe-bound**. The non-persistent version (one CTA per output tile) idles the tensor
  pipe during each CTA's epilogue + pipeline fill — ceiling 72.5% → 1.34–1.36 ms.
- **Persistent + continuous pipeline** lets $\mathcal P$ prefetch tile $t{+}1$'s k-blocks while
  $\mathcal C$ runs tile $t$'s epilogue ⇒ no per-tile fill bubble; tile order ($\beta$ fastest) keeps the
  same L2 reuse of $P$ as the hardware wave scheduler.
- The catch: epilogue overlap needs the staging buffer **separate** from the live ring, but a full
  $128\times256$ stage (66 KB) + a deep $S{=}4$ ring (196 KB) blows the 227 KB smem ceiling. Measured
  dead-ends: shrinking the ring (S=3 → 1.48 ms) or making it asymmetric A2/B4 starves the pipe
  (`long_scoreboard` 5.8–6.8, 1.7–1.9 ms).
- **CHUNKED TMA-STORE** (the CUTLASS `SM90_TMA_STORE` recipe) resolves it: store the tile in 4 column
  chunks via async `cp.async.bulk.tensor` from a small double-buffered $2\times128\times64$ (32 KB) buffer.
  Deep $S{=}4$ ring (hides TMA, `long_scoreboard` 2.0) **and** overlap **and** 32 KB staging all fit in
  224 KB. Result: tensor pipe 72.5% → **78.7%**, gate 1.34 → **1.255 ms**.

**Key PTX / layout recipes (all validated):**
- WGMMA `m64n256k16` (one instr per k16 vs 4× m64n64): B = 4 contiguous $64\times64$ canonical-SW128
  bricks; descriptor `LBO=8192` (brick stride), `SBO=1024`; A descriptor `LBO=128,SBO=1024`. acc[128]
  layout is bit-identical to 4 concatenated m64n64 acc[32] ($g{=}n_2\!\cdot\!8{+}a$).
- TMA load: `cuTensorMapEncodeTiled` SW128, box $64\times64$, OOB-fill 0 (handles $K$-pad + $N$-tail free).
- MN-major both operands (trans-a=1, trans-b=1) → no transpose; cooperative shared-$B$ reuse ($\bar\beta{=}128$ = 2 m64 sub-tiles share the streamed $P$ tile).
- TMA store: `cuTensorMapEncodeTiled` SWIZZLE_NONE, box $64\times128$; `cp.async.bulk.tensor.2d.global.shared`, OOB-clips the $QF$ tail.
- NO `fence.proxy.async` per k-tile in the consumer — TMA writes via the async proxy and the mbarrier already orders visibility (the fence is only needed for generic-store producers).

## Journey (each step measured, NCU-grounded)
gate 1.93 (v14 scalar epilogue) → m64n256 (neutral; not issue-bound) → **staged uint4 coalesced epilogue
1.45** (the scalar 2B stores were 41% lg_throttle, SM 47.7→72.6%) → bank-pad 256→264 **1.36** → drop
per-k-tile proxy fence **1.355** → TMA-store epilogue **1.342** → **persistent + deep-ring + chunked
TMA-store 1.255 (PARITY)**. Dead-ends measured: 2-CTA cluster (moot — hit CUTLASS MFU at
ClusterShape⟨1,1,1⟩), wgmma wait<1> pipeline (tensor-bound), K-tail trim (kills `#pragma unroll`),
sym-S3 / asym-A2B4 persistent (smem-forced shallow pipe).

## Repro
```bash
cd ~/gist/cuda
GIST_HANDGATE=1 GIST_CU=<this>/gist.cu CUDA_VISIBLE_DEVICES=1 ./run_gist.sh <freshrev> big
# gate isolation: + GIST_SKIP_POOL=1 (stats+gate) and + GIST_SKIP_GATE=1 GIST_SKIP_POOL=1 (stats)
# fallback non-persistent: + GIST_HGNONPERSIST=1
```
Kernel: `k_gate_hand_persist` (gist.cu). NCU: binary self-runs with the harness env; profile
`--kernel-name regex:k_gate_hand_persist`.
