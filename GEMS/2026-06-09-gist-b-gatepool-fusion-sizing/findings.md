# DIRECTION: gate→pool fusion (eliminate the SL HBM round-trip)

## 5-field hypothesis (registered 2026-06-09, branch B)

- **name**: `gatepool-l2blk` — capture the SL HBM round-trip via L2-resident batch-blocking
  (the on-chip data-fusion variant is INFEASIBLE, see below).

- **description**: The pipeline writes `SL[B,Q,Fp]` (0.6 GB) from the gate to HBM and the pool
  re-reads it (0.6 GB). Eliminate the *read* by keeping each batch-block's SL resident in L2
  between its gate-write and pool-read: loop the existing `k_gate_hand_persist` + `k_pool_hand_fused`
  over batch-blocks (block ≤ ~96 batches so SL-block ≤ ~37 MB pinnable L2), with
  `cudaLimitPersistingL2CacheSize` + a per-block stream `accessPolicyWindow(persisting)` on the SL
  block. Reuse the verified kernels via a `b_off` batch-offset (shift the TMA tile coordinate; no
  per-block tensor-map re-encode). NO smem fusion (won't fit), NO algorithm change.

- **opportunity (MEASURED)**: pool DRAM read 1.50 GB, of which SL = 0.604 GB = **40%**; pool L2 hit
  only **30.8%** (SL is HBM-missed, not already cached). BW-bound pool 0.79 ms → eliminating the SL
  read scales it to ~0.47 ms = **0.22–0.32 ms prize**. Launch gaps ~0. Partial L2 capture (block fits
  ~74% of SL into the 37 MB persist region at block=128, ~100% at block≤96) ⇒ realistic **0.15–0.25 ms**
  ⇒ total 2.52 → ~2.30 ms. Clears the 0.1 ms go/no-go bar.

- **evidence (NCU rev88032, pool k_pool_hand_fused, GPU4)**: dram__bytes_read 1.50 GB, dram_write
  75 MB, dram__throughput 77.4%, gpu__time 834µs(prof). lts__t_sector_hit_rate 30.8%,
  l1tex hit 88.6%. SL=B·Q·Fp·2=0.604 GB by construction. Gate DRAM only 38% (spare BW). Gate SL
  write is hidden under tensor (lg_throttle 0).

- **ideas**:
  1. PRIMARY: L2-persistence + batch-blocked relaunch of the existing 2 kernels (`b_off` param,
     `cudaStreamSetAttribute` accessPolicyWindow on SL-block). Lowest code risk; reuses verified kernels.
  2. ALT: concurrent gate∥pool stream-pipelining across blocks (pool[N] ‖ gate[N+1]) to hide the pool's
     memory under the gate's tensor — but SM-contention-limited (both want 132 SMs; splitting costs the
     gate tensor throughput). Uncertain.
  3. REJECTED: on-chip smem/register fusion — INFEASIBLE. Gate M=batch (≥64 batched for tensor eff.) ⟂
     pool M=Q (per-batch reduction over Fp). Full SL[b]=393 KB > 227 KB smem; SL[128-block]=50 MB; and
     L2-residency of SL is evicted by the gate's 0.59 GB Pp weight stream.

## GO/NO-GO gate (cheap, before sustained build)
Build the b_off-blocked relaunch + L2 window, measure the pool's lts hit-rate on SL and pool time for
ONE block. If SL L2-hit rises (>~70%) and pool block-time drops proportionally → continue + tune block
size. If L2 reuse doesn't materialize (Pp/X still evict) → STOP, hold at 2.52, document.

## GO/NO-GO PROBE RESULT (2026-06-09): NO-GO — L2-persist does NOT capture the SL round-trip
Built the b_off-blocked relaunch (BLK=128) + `cudaLimitPersistingL2CacheSize=max` + per-block
`accessPolicyWindow(base=SL[block], hitProp=persisting, missProp=streaming)`. Reused the verified gate+pool
kernels via a `b_off` batch-offset (isolated in /tmp/gist_probe.cu; production source untouched). PASS,
max_abs 0.015625.

NCU pool, per 128-batch block: **dram_read = 125 MB = full SL(50 MB) + X(74 MB)** → the pool read SL
**entirely from DRAM**; lts hit **25.4%** (below the 30.8% baseline). **SL capture ≈ 0%, far below the ~70%
go/no-go bar → NO-GO.** Root cause (as predicted): the gate streams its 0.59 GB `Pp` weight while producing
each SL block, evicting the 50 MB SL persist region (block can't drop below 128 — the gate M-tile — so SL-block
≥ 50 MB exceeds the persist region anyway). End-to-end probe time 7.16 ms (relaunch/setattr overhead; irrelevant
to the L2-capture verdict).

**Conclusion: gate→pool fusion is not capturable** — on-chip data-fusion is infeasible (orthogonal gate
M=B / pool M=Q tilings) AND the L2-residency workaround is disproven by measurement. **HOLD at 2.52 ms.**
All three kernels (stats 92% BW, gate 78% tensor = CUTLASS parity, pool 87% BW) are at their ceilings; no
remaining structural lever identified for this kernel architecture/shape.
