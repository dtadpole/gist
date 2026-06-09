# GIST Kernel Optimization Ledger (shared, cumulative DECLARATIVE knowledge)

Append-only. Newest entries at the BOTTOM of each branch section. One entry per durable lesson.

FORMAT:
- **[ISO-ts] TYPE** (FINDING | DEAD-END | TECHNIQUE | MEASUREMENT | PITFALL) — branch
  - WHAT: one-line claim
  - EVIDENCE: real measured numbers / NCU counters (never invented)
  - IMPLICATION: what other branches should do/avoid

**Measured — ONE method (`triton.testing.do_bench` MIN, L2-flush, GPU 0, design shape RMSNorm/F=1497, bf16):**
- **Triton-fast 4.19 ms · CUDA `cuda/gist.cu` 3.42 ms → ~18% faster** (do_bench MIN, GPU0; eager 12.6, compile 4.26).
- METHOD LOCKED: **do_bench MIN on GPU 0** (= the harness GPU + the `run_gist.sh` min_ms statistic; harness reports CUDA 3.40,
  matching do_bench-min 3.42). The earlier "3.51 / ~16%" was GPU 7 + median (slower GPU + wrong statistic). Mixing GPUs/
  statistics produced the whole 4.15/4.31/4.41/3.51/3.62 spread — DON'T. One GPU (0), one tool (do_bench), one stat (min).

Targets (user-defined, vs ~4.2 Triton): **30% goal = <2.95 ms ; 20% min-acceptable = <3.38 ms.** Audited fallback floor ~4.01 ms.

**Current best (RMSNorm/F=1497): 3.42 ms (do_bench min GPU0 = harness min_ms 3.40) = ~18% faster than Triton (4.19).** Gate **cuBLAS** (unpadded K,
op_N/op_T layout with M column-major, ≈1.18 ms vs CUTLASS 1.25; the padded-K route was a TRAP — Pp-copy costs
0.56 ms, dwarfs the ~0.15 GEMM saving). Still SHORT of the 20% bar (3.38). Per-stage (≈ms): stats(M,R,N)≈0.86 ·
gate≈1.18 (cuBLAS) · sigpad≈0.56 · pool≈0.79.

**Prior best (OLD LayerNorm/F=1491, for the record):** 3.465 ms (C gate-NOSIG + vecsigpad, GEM) ;
3.670 ms (B hand-written CUDA+PTX WGMMA gate at CUTLASS parity, 1.255 ms gate, persistent + TMA-store epilogue).

**Hardware reality (THIS box, MEASURED — both throttled well below datasheet):**
- HBM bandwidth: **~2.07 TB/s mixed (2.29 read)**, NOT 3.35 TB/s datasheet (`/tmp/bwtest.cu` int4 copy).
- bf16 tensor peak: **~807 TFLOP/s** (cuBLAS 8192³), NOT 989 datasheet. Gate runs at ~700 TF = ~88% of this real peak → near its compute ceiling.

**Convergent verdicts (all measured, multi-branch):** the win is NOT in the GEMMs (we already beat Triton on
both gate and pool) — it's the helpers (stats N-write + repad) that Triton fuses away. But the hand fused pool
is a software-relayout DEAD-END (~2.7 ms, SM-issue-bound; CUTLASS does the relayout in hardware via TMA at
0.79 ms). The genuine "ptxas warp-3 acc[32] bug" was a missing `fence.proxy.async.shared::cta` (SOLVED). The
gate is at the throttled compute ceiling; gate-mainloop occupancy (PINGPONG/stages/cluster) is exhausted.

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

- **[2026-06-08T22:30Z] FINDING (independent) — B** (Phase-2 padded-fused gate: correct but the PADDING is fundamentally slow)
  - WHAT: fused sigmoid + padded-SL[B,QFp] write into the gate epilogue (remove k_sigpad). CORRECT (flat-N
    reads P fresh, scatter to padded SL, max_abs 0.03125) but SLOW: scalar-scatter 4.62ms total, coalesced-
    chunked 5.76ms, fused-gate iso ~4.1ms (vs Phase-1 gate 1.26 + sigpad 0.56 = 1.82). Independently confirms
    branch C's "padded-fused is the enemy" via 3 of my own impls.
  - EVIDENCE: revs 80260 (scalar 4.62), 80270/80271 (coalesced 5.76 / gate 4.1). Root cause: Phase-1's gate is
    fast ONLY because its epilogue uses an ASYNC TMA store (overlaps next-tile mainloop), which needs 64B align.
    The padded SL col q*Fp+f: q*Fp is 64-aligned (Fp=1536=24*64) but f is arbitrary (F=1491 odd) => a flat-N
    tile's output is misaligned => no async TMA store => synchronous scatter => tensor pipe idles in the epilogue.
  - ROOT CAUSE of the earlier crash/garbage attempts: per-q tiling (aligned SL TMA store, FAST) needs per-q P
    columns at q*F (NOT %64 -> SWIZZLE_128B illegal-instruction; a 3D map's q-stride F*2 is NOT %16 ->
    cuTensorMapEncodeTiled INVALID). So per-q needs a per-q-PADDED P copy (Pp[F,QFp]) -- but the harness re-fills
    P IN PLACE (same ptr), so a pointer-cached Pp goes STALE (verified: GIST_REPAD_EVERY -> PASS). P is static in
    production (Pp = one-time prepack) but the benchmark can't represent that.
  - IMPLICATION: the SIGMOID fuses for free (into Phase-1's flat-L epilogue); the PADDING is the cost. Fast
    padded-fused needs EITHER per-q + Pp prepack (production-valid; benchmark needs a Pp refresh, hideable under
    stats) OR an unpadded-flat-L pool (C's ~3.2ms route, needs the pool to read odd-F K directly). v17/Phase-1
    (gate 1.26 parity) + sigpad (3.67ms total) remains the verified best end-to-end.

- **[2026-06-08T23:40Z] FINDING (independent, definitive) — B** (fusing the PADDED output is counterproductive; separation wins)
  - WHAT: exhaustively tested Phase-2 fused (σ + padded SL, no repad), 5 impls, all CORRECT (max_abs 0.03125)
    but ALL slower than Phase-1 (flat gate 1.26 + separate sigpad 0.56 = 1.82eq, total 3.67):
    flat-N scalar scatter (gate ~2.9), flat-N coalesced chunk (~4.1), per-q async-TMA+σ (gate 2.50, tensor 51%),
    producer-σ-overlap flat-N (gate 3.63, tensor 24%, long_scoreboard 7.89 from single-buf serialization).
  - EVIDENCE: NCU debunks the L2-thrash theory (L2 hit 87-90% in ALL variants, lg_throttle ~0). The real cost is
    the padded-SL WRITE in the gate epilogue: it's either misaligned (odd F=1491 -> can't use the fast async TMA
    store -> slow scatter) or, when on the producer, serializes via the single-buffer staging (double-buffer = 128KB
    + ring > 227KB smem). The SIGMOID fuses for free (applied in staging); the PADDING is the cost.
  - WHY separation wins: Phase-1's sigpad is a CLEAN memory-bound pass (0.56ms, 1.2GB) that does NOT contend with
    the gate's tensor pipe. Fusing the padded write into the gate forces a slow/serial epilogue that idles/stalls
    the gate's tensor pipe (78% -> 24-51%), costing MORE than the separate sigpad.
  - IMPLICATION: the genuinely-fast fused needs the FULL CUTLASS recipe = per-q tiling (aligned async TMA store) +
    producer-σ-overlap + per-q-PREPACKED P (Pp, [F,QFp]). Pp is a one-time static weight prepack (production-valid;
    CUTLASS prepacks weights) but the harness re-fills P in place so it needs a Pp refresh (hideable under stats).
    Without the prepack model, Phase-1 (3.67ms, gate parity 1.26 + sigpad) is the best end-to-end and stands.

- **[2026-06-09T01:00Z] FINDING (definitive, ~12 variants) — B** (Phase-2 fused-padded gate is a net regression vs Phase-1 for odd-F; σ is NOT the cost)
  - WHAT: exhaustively built+NCU'd ~12 fused variants (σ + padded SL, no repad), ALL correct (max_abs 0.03125)
    but ALL net regressions vs Phase-1 (flat gate 1.26 + sigpad 0.56 = 1.82eq, total 3.67):
    best = per-q σ-in-consumer 2.50ms gate; producer-σ-overlap (single/double/chunked buf, wait<0> AND wait<1>)
    all WORSE 3.2-3.4ms gate. σ floor is only 0.14ms (SFU-bound; →0.07 w/ h2exp2) -- σ is NOT the bottleneck.
  - EVIDENCE: NCU best fused (per-q σ-in-consumer) tensor 51% vs Phase-1 flat 78%; producer-σ variants tensor
    26-28%, long_scoreboard 9.9-11.2 (the consumer's epilogue staging+handoff stalls the mainloop ring; wait<1>
    store-pipelining did NOT fix it -> 3.36). The producer-σ handoff (ep mbarriers + bar.sync 2,96) costs MORE
    (+0.9ms) than just doing σ serially in the consumer. S=5 ring alone = 240KB > 227KB smem, so the deepest
    ring + double-buffered overlap epilogue cannot coexist.
  - ROOT: the padded per-q epilogue (staging + chunked async store) idles the tensor pipe ~1ms regardless of σ;
    Phase-1's sigpad is instead a CLEAN memory-bound pass (0.56ms) that doesn't contend with the gate's tensor
    pipe. So SEPARATION beats FUSION for odd-F=1491. The sigmoid fuses ~free; the PADDING is the irreducible cost.
  - IMPLICATION: keep Phase-1 (gate parity 1.26 + sigpad, 3.67ms, locked GEM) as production. A net-win fused needs
    either an unpadded-flat-L pool (pool reads odd-F K directly, no padding) or a fundamentally different epilogue
    that overlaps the padded store without a producer handoff -- neither cracked in 12 attempts.

- **[2026-06-09T02:45Z] MEASUREMENT — MAIN** (*** SPEC CHANGED: LayerNorm → RMSNorm, F 1491 → 1497 ***)
  - WHAT: GIST spec changed (another session, owner-confirmed). Value norm is now RMSNorm:
    N = X·rsqrt(mean_D(X²)+eps)·W[F,D] — NO mean-subtraction, NO bias; weight is per-(feature,channel)
    W[F,D] (was γ[D]+β[D]); inputs 4→3 (X,P,W). M=mean (gate), gate=sigmoid(M@P), pool=L@N UNCHANGED.
    Stats R = rrms = rsqrt(s2/D+eps) (drop −m·m). Shape F=1497 (was 1491; Fp still 1536).
  - EVIDENCE: re-baselined the oracle+drivers (gist_ref.py, driver.py, debug_isolate.py → RMSNorm/F=1497/
    3-inputs; worktree harness is generic so no infra change). Migrated gist.cu (all kernels: R=rrms,
    N=X·rrms·W[f,d], 3-input parse). debug_isolate max_abs 0.0156, 0/1536. run_gist big = **3.4939 ms PASS**.
    NEW Triton bar (RMSNorm/F=1497, do_bench L2-flush) = **4.1519 ms** (28 runs) → we are **~16% faster**.
  - IMPLICATION: ALL prior gems/bars (3.465, 4.225) were the OLD LayerNorm/F=1491 spec — stale. Every branch
    must re-baseline gist.cu (gate/pool GEMMs unchanged; only stats-R + N-formula + the W[F,D] weight change).
    W[F,D] cannot be smem-cached per-d (depends on f) → read W+f·D per row. LayerNorm algebraic factorizations
    (pull γ/β out of the F-contraction) are IMPOSSIBLE for RMSNorm (W[f,d] sits inside the contraction).
    RMSNorm numerics are slightly cleaner (max_abs 0.0156 vs 0.0312 — no centering cancellation). All hardware
    findings hold (box throttled HBM 2.07 TB/s / tensor 807 TF; gate near ceiling; fused-pool relayout dead-end).

- **[2026-06-09] PITFALL (branch C, relayed) — TMA from kernel-malloc'd scratch returns CONSTANT GARBAGE**
  - WHAT: `cp.async.bulk.tensor` (TMA) reading a buffer that was `cudaMalloc`'d AND filled inside the kernel/program
    returns CONSTANT garbage — even though the buffer's bytes are identical AND compute-sanitizer reports 0 errors.
    TMA from the HARNESS-provided buffers or other kernel-WRITTEN buffers works fine.
  - IMPLICATION: a TMA-bulk fused pool that stages through its OWN cudaMalloc'd relayout/scratch buffer will read
    garbage (silent, sanitizer-clean). TMA the SOURCE buffers (X = harness input, L = gate-written output) directly;
    do NOT TMA through a self-allocated scratch. Watch for this in the GIST_TMA_POOL work.
