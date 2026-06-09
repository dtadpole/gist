# GIST Phase 3c — vectorized `stats` (int4 loads + SG=8 sub-group reduction)

**Branch B · `gist-b-phase3c-stats-vec-int4` · 2026-06-09 · H100 (CUDA_VISIBLE_DEVICES=1).**
B=1536, F=1497, D=192, Q=128, bf16, RMSNorm. Builds on [[gist-b-phase3b-pool-fused-rmsnorm]] (fused-N pool).
The full pipeline snapshot in this dir = phase1 gate + phase2s σ + phase3b fused pool + this phase3c stats.

## Result — PHASE 3c MET (stats at the BW floor)
| metric | value |
|---|---|
| **total pipeline** | **3.23 ms** (was 3.29; correctness PASS max_abs 0.015625) |
| **stats (k_stats_mr)** | **0.425 ms** (was 0.487 → −0.062) |
| stats memory BW | **2,117 GB/s = 92% of this GPU's ~2,290 GB/s read ceiling** (was ~1,850 GB/s) |
| stats CUDA-core FLOP | ~3.1 TFLOP/s = 4.6% of the 67 TFLOP/s fp32 peak (NOT flop-bound) |

Component breakdown: stats 0.425 + gate+σ 1.237 + repad 0.776 + pool(fused) 0.785 = 3.23 ms.

## Why / what changed
`stats` (k_stats_mr): M[b,f]=mean_D X, R[b,f]=rrms=rsqrt(mean_D(X²)+eps). One streaming pass over X[B,F,D].
- **Fundamentally memory-bound:** arithmetic intensity = 1.32 GFLOP ÷ 0.9 GB = **1.47 FLOP/byte** vs the roofline
  ridge ~29 FLOP/byte (67 TF ÷ 2.29 TB/s). Total IO = 900 MB (X read 0.883 GB = 98%; M 4.6 MB + R 9.2 MB write).
- The OLD RPW=1 / 4-byte (`__nv_bfloat162`) loads were **SM-issue-bound** (NCU: sm__throughput 85.6%, DRAM only
  54%, inst_executed 86%) — too many small load + convert instructions saturated issue *above* the BW floor.
- **FIX (ported from GEMS/2026-06-08-stats-mr-vec128, math swapped LayerNorm→RMSNorm):** int4 (16-byte) X loads +
  an **SG=8-lane sub-group reduction** over D (each warp does 32/SG = 4 rows). Fewer, wider load transactions →
  frees SM issue → BW-bound at 92% of the real read ceiling. (Swept earlier: SG=8 best; SG=16 0.433, SG=4 0.458.)

## Kernel (stats_mr_body<SG>, SG=8)
- Each lane: `sub = lane/SG` (row in warp), `sl = lane%SG` (lane in sub-group). Loads `int4 x4[i], i=sl..D8 step SG`
  (D8 = D/8 = 24 int4); each int4 = 4 bf16x2 → accumulate s (Σx) and s2 (Σx²) in fp32.
- Reduce over the SG-lane sub-group: `__shfl_down_sync(sgmask, ., o, SG)` for o=SG/2..1.
- `sl==0`: `m=s/D; rrms=rsqrt(s2/D+eps)` (RMSNorm, no centering); write M (col-major [B,F]), R (fp32 [B,F]).
- Launch: `mr_blocks = ceil(B*Fp/(32/SG) / STATS_WPB)`, block = STATS_WPB(8)*32 = 256.
- Requires D%8==0 (192 ✓) and 16-byte-aligned rows (row*D*2 = 384 B ✓).

## Note on the GPU's real bandwidth ceiling
This box's HBM read tops out ~**2.29 TB/s** (int4-copy microbench), NOT the 3.35 TB/s theoretical — the mem clock
caps it. So a memory-bound kernel's floor is set by ~2.3 TB/s. stats at 2.12 TB/s = 92% → effectively done
(only ~0.02 ms left to the 0.40 ms pure-read floor).

## Repro
```bash
cd ~/gist/cuda
GIST_HANDGATE=1 GIST_PHASE1RAW=1 GIST_FUSESIG=1 GIST_POOLFUSE=1 GIST_CU=<this>/gist.cu CUDA_VISIBLE_DEVICES=1 ./run_gist.sh <rev> big
#   -> passed=true, max_abs 0.015625, total ~3.23ms
# stats isolated: + GIST_SKIP_GATE=1 GIST_SKIP_POOL=1 (no PHASE1RAW) = k_stats_mr only.
```
Kernel: `k_stats_mr` -> `stats_mr_body<STATS_SG=8>`. See [[gist-pool-handwritten-kmajor]].
