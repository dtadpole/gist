# GEM: stats (M,R-only) vectorized 128-bit sub-group reduction — Triton-class

## What this is
A BW-efficient rewrite of `k_stats_mr` (the M,R-only stats kernel used by the GIST_MPF / fused-pool
path). Mirrors Triton's `_stats_kernel` efficiency: 128-bit (int4 = 8 bf16) vectorized X loads + a
sub-group reduction over D, instead of the old RPW=1 / one-warp-per-row / 4-byte-load structure.

## Measured (design shape B=1536,F=1491,D=192,Q=128, bf16, L2-flush harness, this H100)
Isolated `k_stats_mr` (M,R only — no N write), via `GIST_MPF=1 GIST_SKIP_GATE=1 GIST_SKIP_POOL=1`:
- **before (RPW=1, 4-byte loads): 0.491 ms**
- **after  (SG=8, int4 loads):   0.4226 ms  → 14% faster on this kernel**
- Triton's `_stats_kernel` reference is ~0.41 ms → **we are Triton-class** (within ~3%).

Correctness: the full GIST_MPF path (which uses this `k_stats_mr` + inline-N pool) PASSES at
**max_abs 0.03125** (identical to baseline) — confirms the M,R math is exact. See
`mpf_full_correctness_bench.json`.

## Why it works (measured BW, not theory)
A raw int4-copy microbench on THIS GPU gives the achievable HBM ceiling:
- **pure read 879 MB: 0.4024 ms = 2.29 TB/s** (NOT the 3.35 TB/s theoretical — this box's mem clock
  caps ~2.3 TB/s read).
- k_stats_mr moves 879 MB read + 14 MB write = 893 MB. At 0.4226 ms = **2.11 TB/s = ~92% of the
  measured read ceiling.** The old 0.491 ms was 1.82 TB/s (~79%). So we went 79% → 92% of the REAL
  ceiling by switching 4-byte bf16x2 loads → 16-byte int4 loads.

Root cause of the old slowness: RPW=1 issued 4-byte loads (one __nv_bfloat162 per lane per iter) →
24 transactions of 4 B per row. The int4 path issues 16-byte coalesced transactions and packs 4
rows per warp (SG=8 lanes/row), so the memory subsystem runs near peak.

## Winning structure
- **SG = 8** lanes per (b,f) row → 4 rows/warp; each lane loads 3 int4 (48 B contiguous), reduces D
  over its 8-lane sub-group (3 shfl). Swept SG∈{2,4,8,16,32}: SG=8 best (0.422), SG=16 0.433,
  SG=4 0.458, SG=2 0.497, SG=32 0.564.
- **MR_WPB = 8** warps/block (flat 0.421–0.426 across 4..16; 8 best).
- Plain cached loads beat `__ldcs` streaming (0.422 vs 0.438) — L2 helps the coalesced read.

## Key finding for the owner (k_stats with N — the default M,R,N kernel)
The SAME int4 structure was applied to `k_stats` (writes M,R AND the 906 MB N_pad) and did **NOT**
help: 0.845 → 0.855 ms. Reason: k_stats is **write-bound on the 906 MB N**, not read-bound. The
mixed read+write workload (879R+906W ≈ 922+922 MB copy) microbenches at **0.89 ms = 2.07 TB/s** —
the original k_stats already runs at 0.845 ms (AT that mixed floor). So the original RPW=1 k_stats is
kept for the safe default (byte-identical to baseline; default total stays 4.013–4.02 ms PASS).
The "0.60–0.65 ms floor" assumed 3.35 TB/s; this GPU does not reach that.

## Trade-off summary
- **Safe default (keeps CUTLASS pool, guaranteed correct):** k_stats-with-N 0.845 ms isolated,
  default total ~4.013 ms — unchanged (at the memory floor, no further win available here).
- **Triton-class standalone:** k_stats_mr 0.4226 ms (M,R only). Using it in the default would require
  the pool to compute N inline (fused-pool path) — a separate decision, NOT implemented here.

## Reward-hack audit (PASS)
All numbers from real harness runs I executed (revs 50006–50503). bf16 operands + timed run, fp32
accumulators only, no tf32/fp32 substitution, no shape/tolerance/seed/flush/warmup changes. The
`GIST_NVCC_APPEND` knob added to run_gist.sh is additive (empty by default) and only fed `-DSTATS_SG`
/`-DMR_WPB_DEF` for sweeps; the committed kernel hardcodes SG=8/WPB=8, default compile uses no append.

## File:line of edits
- `cuda/gist.cu` ~241–300: `stats_mr_body<SG>` + `k_stats_mr` (int4 sub-group reduction)
- `cuda/gist.cu` ~1608–1612: `MR_WPB`/`mr_blocks` launch grid; k_stats_mr launches use it
- `cuda/gist.cu` k_stats body restored byte-identical to baseline (default path unchanged)
