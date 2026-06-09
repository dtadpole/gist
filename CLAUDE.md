# GIST CUDA+PTX — How to work on this goal

Read this fully. It is the operating contract for this project. It persists across
/compact, so re-read it after compaction.

## The goal (non-negotiable)
Write a **CUDA + inline PTX** GIST forward kernel that BEATS `gist_triton_fast` by
**30%+**: reference 4.22 ms bf16 at the design shape (B=1536,F=1491,D=192,Q=128).
- Target / success: **< 2.95 ms bf16** (30% faster).
- Minimum acceptable: < 3.38 ms bf16 (20% faster).
- Measured apples-to-apples: same shape, bf16, CUDA-event timing, correctness within
  the harness tolerance, with a real margin.

## BF16 is mandatory
All compute operands and the benchmarked/timed run must be genuine **bf16**. fp32
*accumulators* inside a bf16 tensor-core kernel are fine and expected. Do NOT substitute
tf32/fp32 for the math to dodge a correctness failure or to make timing easier — fix the
numerics (accumulation order, layout) instead. Earn correctness at bf16.

## NO REWARD HACKING (this is a hard failure if violated)
The latency number must come from REAL work on the REAL problem. Never:
- shorten/skip the timing loop, cut iterations, remove warmup, or move work outside the
  timed region; never disable the L2 flush
- shrink the design shape; weaken the correctness tolerance
- dump golden/cached outputs instead of real kernel outputs; special-case the input seed
  (0xCAFE0000+j timing / idx+1 correctness); hardcode expected results
- **invent evidence** — never claim a bottleneck or a speedup you did not measure. Made-up
  evidence is reward hacking too.
A fast number from a gamed harness or a gamed dtype is a FAILURE, not a success. The
supervisor audits the harness diff and the kernel every few minutes.

## HOW TO WORK — the method (this is how I want you to act)

### 1. EXPLORE, EXPLORE, EXPLORE before you build
Do NOT guess. Before committing to an approach, do real research and write down what you
learned:
- Read NVIDIA docs: PTX ISA (wgmma/mma/cp.async/TMA), CUDA C++ Programming Guide, Hopper
  tuning guide. Understand the actual instruction semantics, not your priors.
- Read reference implementations: the Triton `gist_triton_fast.py` (what does it actually
  emit?), CUTLASS WGMMA/Hopper GEMM examples, FlashAttention's pipelining.
- Profile with NCU and COMPARE your kernel vs the reference side by side. Numbers, not
  intuition. (You already have a good example: gate mma.sync-bound, DRAM 6%/SM 32%,
  Triton 14x faster via wgmma.mma_async + cp.async + L2 swizzle.)
- Search the web for how others solved the same bottleneck.
The more external perspective you gather, the stronger the hypothesis. Brainstorming
purely from your own knowledge without searching docs/reference is not allowed.

### 2. Hold real BRAINSTORM sessions
When you reach a decision point, brainstorm multiple concrete approaches, compare their
trade-offs with reference to data, and pick the one most supported by your profiling.
Don't spread thin across 4 ideas — go deep on the best one, keep the alternatives noted.

### 3. FORM A HYPOTHESIS = register a DIRECTION (this is paramount)
Before sustained building, state your direction explicitly with these 5 fields:
- **name** — a specific architectural approach (e.g. `wgmma-gate`), not "optimize".
- **description** — WHAT architectural change (e.g. "1 producer warpgroup for TMA loads,
  2 consumer warpgroups for wgmma.mma_async, N-stage async pipeline, matrix descriptors").
- **opportunity** — expected gain grounded in data (e.g. "gate 38ms → ~3ms; Triton does
  this op in 2.78ms; closes most of the 14x gap → total < 2.95ms").
- **evidence** — from ACTUAL profiling/NCU, not general knowledge. Cite your numbers.
- **ideas** — primary route + alternatives, each concrete.
Write the direction down (e.g. in a DIRECTION.md or your notes) so it's checkable. If you
have no direction, STOP and form one — don't thrash edits.

### 4. Build, measure, iterate (metric-driven)
- Correctness BEFORE performance, always. If a config fails correctness, fix that first.
- After an architecture change, EXPECT initial regression. Judge the direction AFTER you
  optimize it (pipeline depth, prefetch overlap, epilogue), not on the first untuned try.
- Every claim of "faster/slower/doesn't work" must be backed by a real benchmark/profile.
- If stuck on correctness: DECOMPOSE — shrink to the smallest reproducing config, verify
  building blocks in isolation, integrate back one at a time. Don't thrash at full-kernel
  level.
- If in an edit-compile-fail loop: step back, profile the failure, search docs.

### 5. The current direction (adopted)
Hand-written **WGMMA gate** using inline PTX (`wgmma.mma_async`, cp.async pipeline, L2
swizzle) to close the ~14x gate gap vs Triton. This IS the CUDA+PTX deliverable the goal
demands. Don't pause or call it exhausted — register it as a proper hypothesis (above)
and commit. This is a large, CUTLASS-level effort; that's expected.

## SUCCESS → write it to GEMS/
When you achieve a result that beats the bar (or any meaningful milestone gem = a new
best measured bf16 latency), record it under `~/gist/GEMS/`:
- the kernel source (or a copy/snapshot), the harness benchmark JSON, a short results.md
  with the measured latency + % vs 4.22ms + which technique/PTX op moved the number, and
  the NCU evidence. A "gem" is a verified improvement, captured so it's not lost.

## Context hygiene
This is a long run. When context gets high (>~80%), run `/compact` and then re-read this
CLAUDE.md. Persist key learnings to your memory and to notes so they survive compaction.
