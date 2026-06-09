# GIST Kernel Optimization Ledger (shared, cumulative DECLARATIVE knowledge)

Append-only. Newest entries at the BOTTOM of each branch section. One entry per durable lesson.

FORMAT:
- **[ISO-ts] TYPE** (FINDING | DEAD-END | TECHNIQUE | MEASUREMENT | PITFALL) — branch
  - WHAT: one-line claim
  - EVIDENCE: real measured numbers / NCU counters (never invented)
  - IMPLICATION: what other branches should do/avoid

Target <2.95ms / min acceptable <3.38ms / audited fallback floor ~4.017ms.
Global best (see GEMS): 2026-06-08 gate-fused-sigmoid = 4.014ms (beats Triton 4.225 by 5.0%).

---

## Branch B — producer/consumer warp-specialization (direction #4)

(seeded by steward; canonical home was missing — see ~/gist-opt/README.md. Append findings here.)

- **[2026-06-08T12:10Z] DEAD-END — B**
  - WHAT: the pre-existing GIST_WARPSPEC pool (mma.sync producer/consumer, named barriers) is BROKEN and SLOW.
  - EVIDENCE: big shape = 5.29ms AND correctness FAIL (max_abs 2.77, mean 0.24). GIST_WGMMA_POOL acc[96] also
    FAIL 2.77 (its bug: it re-applies sigmoid although the CUTLASS gate already fuses Sigmoid — double sigmoid).
  - IMPLICATION: mma.sync is a dead path for the pool (wgmma ~3x faster). And the gate output g_L is ALREADY
    sigmoid'd — any pool must copy g_L directly, NOT re-apply sigmoid (cost C a 2.77 mystery otherwise).

- **[2026-06-08T12:25Z] MEASUREMENT — B** (per-stage budget, default path full=4.029ms, GPU1)
  - WHAT: stats(M,R,N)=0.85, gate(M@P+sigmoid→L)=1.66, repad=0.73, poolGEMM(CUTLASS)=0.80 (harness differential).
  - EVIDENCE: NCU iso: gate 2.42ms@19%DRAM/14%warps; k_pool_wgmma_fused(acc32) 4.40ms@34%DRAM/30.5%warps/94reg;
    k_stats_mr 0.69ms@53%DRAM/88%warps. Fused-pool grid (B,Q/64,D/64) reads A=L 3x + X 2x = 3.5GB (1.46GB optimal).
  - IMPLICATION: gate (1.66 effective) is the largest fixed cost & itself only 14%warps/19%DRAM — a big separate
    opportunity. Fused pool ceiling is occupancy/pipelining (34% DRAM), not flops. Kill the 3x L redundancy (no D-split).

- **[2026-06-08T12:55Z] DEAD-END — B** (warp-spec, large smem)
  - WHAT: 1 producer-WG + 1 consumer-WG, FULL-D acc[96], multi-stage mbarrier, grid (B,Q/64): ~157KB smem →
    occupancy_limit_shared_mem=1 block/SM → starves memory.
  - EVIDENCE: NCU iso 13.5ms, DRAM 7.5%, warps_active 12.3% (vs single-WG 4.4ms/34%/30%). Worse, not better.
  - IMPLICATION: warp-spec only wins if it keeps ≥3-4 blocks/SM (small KT/stages). A whole consumer-WG idling on
    the light (0.11ms-total) wgmma + 1 block/SM removes the block-level latency hiding. Aligns with C's nsys
    (fused pool is barrier/smem-shuffle-bound, not DRAM-bound) — warp-spec ADDS barriers; minimize smem round-trips.

- **[2026-06-08T13:00Z] PITFALL — B** (multi-stage handoff)
  - WHAT: plain named barriers (bar.sync/arrive) LAP with S>1 buffers → spurious release → nondeterministic
    smem corruption. mbarrier (init/arrive/try_wait.parity, .release/.acquire) is required. A computed-N
    (generic-store) producer feeding wgmma also needs fence.proxy.async.shared::cta on producer AND consumer.
  - EVIDENCE: named S=1 passed big (0.031), S=3 failed (0.19). mbarrier S=1/KT64 passes (0.031); S>1 still races
    (under investigation). KT=32 has a separate S=1 bug (max_abs 0.109) — KT=64 is the verified base.
  - IMPLICATION: any branch building a multi-stage producer/consumer pipeline: use mbarrier phase parity, not
    named barriers; fence cross-proxy on both sides.

- **[2026-06-08T13:30Z] DEAD-END — B** (warp-spec verdict)
  - WHAT: correct producer/consumer warp-spec fused pool (D-split acc[32], mbarrier, direct-global vectorized
    16B X read, no cp.async) best = 5.19ms — WORSE than the 4.03ms default. NCU pool iso 4.37ms, DRAM 34.8%,
    warps_active 48.5% → latency/barrier-bound: the consumer (trivial wgmma) idles waiting on the producer-bound
    memory+N-compute, wasting ~half the warps. Cooperative single-WG (all warps produce) = 4.4ms, beats it.
  - EVIDENCE: revs 2125-2131; S2<S3<S4, KT64<KT128 (more buffers → less occupancy). cp.async raced w/ mbarrier;
    direct global loads fixed correctness (the handoff logic w/ phase-parity + rel/acq is sound).
  - IMPLICATION: warp-spec is for compute-bound GEMMs; for this producer-bound fused pool keep all warps producing.
- **[2026-06-08T13:30Z] FINDING — B** (the gate is the bottleneck, not the pool)
  - WHAT: lane floor (stats_mr+gate, pool skipped) = 2.14ms. Pool-only lanes (B,C) cap total at ~3.34ms even with
    a Triton-class pool (1.19ms) → MISSES the 2.95 target; the gate must be attacked. Gate = 2.42ms iso at
    14% warps_active / 19% DRAM (skinny-K M@P, K=1491, N=190848, writes L 586MB) — ~7x off its 0.34ms mem ideal.
  - EVIDENCE: GIST_WS+SKIP_POOL=2.14ms; NCU gate 2.42ms/14%warps/19%DRAM, persistent 132-block 128x256 tile.
  - IMPLICATION: NO branch owns the gate. The fleet must optimize it (better skinny-K CUTLASS schedule / custom
    mem-bound gate / cuBLAS / fuse gate→pool to skip L) to beat the bar. B has full context & can pivot if sanctioned.

---

## Branch C — multi-WG split (direction #5)

---

## Shared / cross-branch

- **[2026-06-08T13:45Z] DEAD-END — B** (cuBLAS gate is worse)
  - WHAT: probed cuBLAS for the gate M@P (the bottleneck). stats_mr+cuBLAS-gate = 6.30ms vs stats_mr+CUTLASS-gate
    = 2.14ms → cuBLAS gate ~5.6ms vs CUTLASS ~1.45ms (4x SLOWER) for this skinny-K (K=1491) huge-N shape.
  - EVIDENCE: GIST_WS+GIST_CUBLAS_GATE+SKIP_POOL rev 2134 = 6.30ms.
  - IMPLICATION: don't use cuBLAS for the gate. CUTLASS is already the best easy option; beating it needs a
    custom skinny-K mem-bound kernel or fusing gate→pool (skip the 586MB L round-trip). Also: the 2-q-tile
    warp-spec pool (grid (B,D/64), acc[64]) = 7.72ms (worse than per-q-tile acc[32] 5.19ms; acc[64]+bigger As
    drop occupancy to 2 blocks/SM). Occupancy dominates this latency-bound pool; keep acc small.

- **[2026-06-08T14:30Z] FINDING — B** (k_stats_mr is near-optimal; the "2x stats" is N-materialization, not stats inefficiency)
  - WHAT: branch B owns k_stats_mr (M,R-only). Harness-isolated = 0.49ms vs Triton 0.41 (only ~16% slower, NOT
    2x). The owner's "0.85 vs 0.41" compares our k_stats(WITH N write, 906MB) to Triton's mean+rstd-only — the
    gap is the N-MATERIALIZATION (Triton computes N in its fused pool), not a slow reduction. k_stats_mr NCU:
    issue_active 86% / DRAM 54% / L1 35% → ISSUE/REDUCTION-bound (warp-shuffle reduce of D=192 per row), not
    memory-bound. Sweeps all plateau ~0.48-0.49: RPW 1<2<4 (more regs, fewer warps), warps_per_blk 4≈8<16<32,
    row-major M = 0.479 (only 0.015 faster, but breaks the gate's col-major unpadded-A layout so kept col-major).
    Coalescing forces warp-per-row (consecutive lanes→consecutive D); sub-warp/thread-per-row reductions break it.
  - EVIDENCE: revs 2242-2266; NCU rev 2242 (issue 86%, DRAM 54%, L1 35%, 24 reg, 279 waves).
  - IMPLICATION: k_stats_mr has little headroom (~0.08ms, fused-path-only). The real stats win is MAIN's lane:
    DON'T write N in stats — fold N into the pool (Triton-style). That removes ~0.26ms N-write from stats AND
    enables repad elimination, but needs the fused pool (B's warp-spec verdict: hard, occupancy/barrier-bound).

- **[2026-06-08T14:55Z] DEAD-END (CONFIRMED) — B** (dir#4 path #1 tested)
  - WHAT: wgmma.wait_group<1> software-pipelined consumer (acc[32]) = CORRECT (no warp-3 corruption) but 5.33ms
    > 5.19 wait<0>. Consumer is idle (producer-bound) so consumer pipelining can't help. With S-sweep (S2 best),
    KT-sweep, NCU (35%DRAM/48%warps), and the producer-heavy/consumer-light structure (inverse of FlashAttention),
    warp-spec for the GIST pool is a CONFIRMED, measured dead-end. Branch B parks. Path #2 (TMA) targets the
    producer but adds an X smem round-trip and can't fix the consumer-idle waste; structure rules it out.
  - EVIDENCE: rev 2270/2271 (0.050 small / 5.33 big, both PASS); rev 2272 wait<0> 5.22 PASS.
  - IMPLICATION: SIDE-WIN — acc[32]+wait_group<1> is now correct via fence.proxy.async (the old "acc[32] warp-3
    ptxas bug" was the missing cross-proxy fence; CONFIRMED resolved). Usable for small-acc wgmma pipelining.

- **[2026-06-08T15:40Z] MEASUREMENT — B** (hand gate Phase-1, racing C)
  - WHAT: k_gate_hand_b (pure CUDA+PTX wgmma, no CUTLASS), L=M@P raw logits, CORRECT (full-pipeline verified,
    max_abs 0.03125). v0 21.85ms -> v1 13.41ms via BOTH-operands-MN-major (A=M & B=P both natural MN-major ->
    cp.async, no scatter; verified MMA_64x64x16<Major::MN,Major::MN> trans-a=1) + cp.async double-buffer.
  - EVIDENCE: rev 2301/2302. NCU v1 gate 14.3ms iso, DRAM 3.3%, SM 28%, warps 37%, ~6% tensor, 71568 tiny 64x64 CTAs, 90 waves.
  - IMPLICATION: overhead/latency-bound (tiny CTAs + single-WG serialization), NOT mem/compute. Roadmap to 1.26:
    bigger tiles -> persistent -> warp-spec prod/cons -> TMA + cluster P-multicast (batched GEMV). For C: MN-major
    for BOTH operands is natural here (no transpose either side); the wall is CTA-count/occupancy next, not staging.

- **[2026-06-08T18:40Z] FINDING — B** (hand gate reaches CUTLASS-class MFU; THE lever was the epilogue, not the cluster)
  - WHAT: hand CUDA+PTX gate (no CUTLASS) gate=1.93ms→**1.36ms** (within 8% of CUTLASS 1.26), total
    4.38→3.775ms (beats prior global best 4.014 by 6%). CORRECT max_abs 0.03125. NCU tensor-pipe 72.5%.
  - EVIDENCE (each measured, revs 80001-80014, gate via SKIP_POOL differential = stats+gate − stats(0.847)):
    (1) m64n256 atom (1 wgmma/k16 vs 4×m64n64) = NEUTRAL → gate NOT issue-bound. B = 4 contiguous 64×64
        SW128 bricks, desc LBO=8192/SBO=1024 (A LBO=128/SBO=1024); acc[128] bit-identical to 4× m64n64.
    (2) **staged+coalesced uint4 epilogue = 1.93→1.45ms** (THE win): scalar 2B stores were 41% lg_throttle
        / SM 47.7% (C's finding); stage acc→smem + bar.sync(256 consumers, producer WG exited) + uint4
        16B coalesced stores → SM 72.6%. (3) bank-pad staging stride 256→264 = 1.45→1.36 (kills 8-way
        conflict: r0*128%32==0). DEAD-ENDS: wgmma wait<1> pipeline cut stalled_barrier 4.82→2.84 but
        latency UNCHANGED (tensor-pipe-bound, stalls off critical path); K-tail trim regressed 1.36→1.56
        (variable loop bound kills #pragma unroll).
  - IMPLICATION: the WS-gate epilogue (scalar 2B stores) was the universal #1 lever for BOTH B and C —
    any hand wgmma gate MUST do staged uint4 coalesced epilogue. The 2-CTA cluster is NOT the lever:
    reached CUTLASS-class 72.5% tensor MFU with ClusterShape<1,1,1> (C confirmed CUTLASS itself uses none).
    Residual 8% to 1.26 = skinny-K tensor-idle (per-CTA epilogue + fill/drain) → needs persistent+raster.
    GEM: ~/gist/GEMS/handgate-cutlass-class-v17/.

- **[2026-06-08T19:20Z] DEAD-END (measured) — B** (persistent epilogue-overlap is smem-walled at the feasible depth)
  - WHAT: built persistent warp-spec gate (1 CTA/SM, grid-strided tiles, ONE continuous TMA→wgmma pipeline
    across output-tile boundaries so the producer prefetches tile T+1 while consumers run tile T's epilogue;
    SEPARATE staging smem). CORRECT (max_abs 0.03125) but gate 1.36→**1.48ms** (REGRESSED), total 3.79→3.90.
  - EVIDENCE: NCU persistent gate tensor-pipe 65.6% (DOWN from v17 72.5%), long_scoreboard 1.2→2.06, wait
    1.49→2.83 — the overlap IS partially working (lg_throttle 1.02→0.18) but the staging smem (66KB) forces
    HGP_S=3 (vs v17 S=4), and the shallower pipe's exposed TMA latency exceeds the overlap gain. S=4 + 66KB
    separate staging = 262KB > 227KB smem max. revs 80015/80016.
  - IMPLICATION: persistent overlap needs B-pipe depth 4 AND separate staging in 227KB → requires ASYMMETRIC
    ring depths (A 2-deep [L2-resident, M fits L2], B 4-deep [DRAM]) — a dual-barrier pipeline rewrite, est.
    ~1.30ms (still short of 1.26). The literal 1.26 = CUTLASS's mature persistent+asymmetric machinery.
    v17 (1.36ms, 72.5% tensor) stands as branch B's best; persistent code kept env-gated (GIST_HGPERSIST).

- **[2026-06-08T19:55Z] DEAD-END (measured, NCU) — B** (persistent gate regresses: 1-CTA/SM loses inter-wave TMA hiding)
  - WHAT: persistent warp-spec gate (1 CTA/SM, grid-strided, continuous pipeline, epilogue overlap) in 3 forms:
    symmetric S=3 (gate 1.48), single-producer asym A2/B4 (1.89), DECOUPLED dual-producer A2/B4 (1.83). ALL
    regress vs v17 non-persistent 1.36. All CORRECT (max_abs 0.03125).
  - EVIDENCE: NCU decoupled gate tensor 53.5% (v17 72.5%), **long_scoreboard 6.77 (v17 1.20)**, L2 hit 83%, DRAM
    29%. Single-producer asym: tensor 52.6%, long_scoreboard 5.76. The epilogue overlap DOES work (lg_throttle
    1.02→0.17) but TMA latency is NOT hidden at 1 CTA/SM persistent. ROOT CAUSE: v17 (8952 CTAs/68 waves) hides
    TMA via the scheduler overlapping waves; persistent (132 CTAs, 1/SM) has no spare CTA to switch to on a TMA
    stall, and the A 2-deep ring (forced by the 66KB staging smem) under-buffers. revs 80015-80021.
  - IMPLICATION: persistent overlap is a DEAD-END for this gate WITHOUT a much deeper TMA pipe — which needs the
    66KB staging smem freed via a CUTLASS-style SM90_TMA_STORE epilogue (async bulk store, smem unioned with the
    pipeline). That's the real CUTLASS recipe for the residual 8%; large build. v17 (non-persistent) stays B's best.

- **[2026-06-08T20:20Z] FINDING — B** (PHASE 1 MET: hand CUDA+PTX gate at CUTLASS parity)
  - WHAT: hand-written WGMMA gate (no CUTLASS), raw logits, = **1.255 ms** (CUTLASS 1.26) — PARITY.
    CORRECT max_abs 0.03125, NCU tensor-pipe 78.7%. Total pipeline 3.670 ms. GEM:
    ~/gist/GEMS/gist-b-phase1-wgmma-persist-tmastore/ (kernel k_gate_hand_persist; default GIST_HANDGATE).
  - EVIDENCE: gate = stats+gate 2.104 − stats 0.849 = 1.255 (stable 1.255–1.259 across runs & grid sweep
    132/264/528). NCU long_scoreboard 2.0 (TMA hidden), tensor 78.7% (> non-persistent 72.5%). revs 80150-80170.
  - IMPLICATION: THE recipe = PERSISTENT (1 CTA/SM, grid-strided tiles, ONE continuous TMA→wgmma pipe across
    tile boundaries so the producer prefetches tile t+1 during tile t's epilogue) + DEEP S=4 single ring (hides
    TMA) + CHUNKED TMA-STORE epilogue (4×64-col async cp.async.bulk.tensor from a 32KB double buffer — small
    enough that the deep ring + overlap fit 227KB smem, unlike full-tile staging which forces a shallow/asym
    pipe that regresses). + the earlier epilogue lever (staged uint4 → SM 47→72%) + m64n256 + bank-pad +
    proxy-fence-drop. Non-persistent best was 1.34–1.36; persistent overlap closed the last ~6%. CUTLASS uses
    the same SM90_TMA_STORE + ClusterShape<1,1,1> (no cluster). Phase 2 (sigmoid+SL-pad fusion) next.
