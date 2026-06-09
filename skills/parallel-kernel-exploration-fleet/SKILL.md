---
name: parallel-kernel-exploration-fleet
description: Run multiple AI Coders (Claude Code in tmux) in parallel to explore orthogonal optimization directions for one kernel/problem, harvesting findings between branches and capturing verified winners (GEMS) so the whole exploration compounds instead of each branch re-learning the same lessons. Use when one Coder has enumerated several independent candidate levers and GPUs/compute are free.
version: 1.2.0
author: gist project
license: MIT
metadata:
  hermes:
    tags: [parallel, fleet, ai-coder, claude-code, tmux, kernel-optimization, knowledge-accumulation, gems, gpu]
    related_skills: [goal-navigator]
---

# Parallel Kernel-Exploration Fleet

## When to use
- One AI Coder has enumerated several INDEPENDENT candidate directions (e.g. CC listed levers
  #1-#5), and exploring them serially would be slow.
- Compute is available (e.g. 8× H100 mostly idle) so GPU cost is ~0; the real cost is N× token
  spend, so only launch ORTHOGONAL directions (no overlap = no wasted tokens).
- The deliverable is KNOWLEDGE + GEMS (verified fast kernels), not any single attempt.

## Core principle
The fleet is worth more than N solos ONLY if knowledge accumulates. A branch that hits a wall
must turn that into a lesson so no other branch repeats it. Negative results (measured
dead-ends) are first-class deliverables.

## Knowledge model (do NOT build a duplicate ledger)
Each Claude Code already keeps its OWN cumulative notes and persists learnings across /compact
(its CLAUDE.md instructs it to). Do NOT stand up a parallel Hermes-maintained wiki ledger that
restates what CC already records — that is duplicate machinery that drifts out of sync. Instead:
- The Coder owns its findings (its notes + observations.log).
- The AGGREGATOR harvests findings from each branch's `observations.log` + tmux pane and
  cross-pollinates them as one-line heads-ups (below).
- `GEMS/` = verified winning kernels (kernel.cu + bench.json + results.md per gem). This is the
  durable cross-run artifact.
- One reusable PROCEDURAL artifact only: this SKILL.md (how to run the fleet). That is the
  reusable knowledge — not a per-run findings ledger.

## Isolation model (owner-specified)
- SEPARATE FOLDER PER DIRECTION holding ONLY the per-direction source file (e.g. `gist.cu`).
  NOT git worktrees — plain folders are cleaner and easier to navigate.
- ALL other code (driver, harness, reference, toolchain) stays SHARED in one folder.
- The shared driver reads the per-direction source via an env override, e.g. add to driver.py:
  `CU = Path(os.environ.get("GIST_CU", str(HERE / "gist.cu"))).read_text()`
  then build with: `GIST_CU=<dir>/gist.cu CUDA_VISIBLE_DEVICES=<gpu> ./run_harness.sh <rev> big`
- Each direction gets: its own folder, its own tmux session, a pinned GPU, and a disjoint
  revision-number range (so harness run dirs never collide).

## Setup procedure
1. Pick orthogonal directions from the lead Coder's OWN lever enumeration (guarantees no overlap).
2. `mkdir -p <root>/dir-<x>/` per direction; copy the current verified source into each.
3. Add the env override to the shared driver (backward-compatible default = local source).
4. Smoke-test: compile one direction's source through the shared driver on its GPU.
5. Launch a Claude Code per direction:
   `tmux new-session -d -s <name> -x 200 -y 50`
   `tmux send-keys -t <name> 'cd <shared-dir> && export CUDA_VISIBLE_DEVICES=<gpu> && claude --dangerously-skip-permissions' Enter`
   handle trust/permission dialogs if they appear, then send a kickoff pointing to a per-branch
   TASK.md (multi-line tasks go in a file; send-keys just says "read TASK.md and follow it").
6. The kickoff/TASK.md must tell the branch: its direction (and to NOT drift into others'
   lanes), to keep its own notes durable across /compact, the bf16 + no-reward-hacking rules,
   and how to submit a GEM.

## Supervision (reuse goal-navigator discipline)
- One Navigator cron PER branch (deliver=local/quiet): reward-hack audit incl. measurement-
  bypass + direction check + methodology REDIRECTs only. NEVER /compact. Silence when ON_TRACK.
  Do NOT make the branch maintain a duplicate wiki — just confirm its own notes survive compaction.
- One AGGREGATOR cron (deliver=origin, in the owner's language): the keystone of accumulation —
  it (a) cross-pollinates: harvests findings from each branch's observations.log + pane and
  queues one-line heads-ups into other branches' tmux when one branch unblocks another (NEVER
  code); (b) arbitrates GEMS: it is the SOLE writer of "global best" — independently re-verifies
  each candidate (re-run bench on an isolated copy + reward-hack/measurement-bypass audit)
  before promoting, so parallel branches can't corrupt the registry or self-promote a gamed
  number; (c) delivers ONE consolidated report.

## Injecting into a Coder = NOT done until VERIFIED (owner-corrected, critical)

Every cross-branch heads-up, REDIRECT, or any message a Navigator/aggregator sends into a Coder's
tmux is NOT delivered just because `tmux send-keys` returned exit 0. The single most common silent
failure (owner-reported, real GIST session): a long/multi-line message typed as text+Enter in ONE
`send-keys` call leaves the text STUCK in the Coder's input box, unsubmitted — the Coder flies blind
for a whole cycle and nobody notices. Sending ≠ delivered. The discipline, every time:

1. Send the message TEXT first, WITHOUT Enter:  `tmux send-keys -t <session> "<text>"`
2. `sleep 1`  (let the paste render/settle)
3. Send Enter as its OWN separate call:  `tmux send-keys -t <session> Enter`
4. `sleep 1`, then VERIFY non-destructively:  `tmux capture-pane -t <session> -p -S -6`
   - PASS = input box empty, OR shows "Press up to edit queued messages", OR a new ●/spinner appeared.
   - FAIL = your text still sits above the `❯` border → send ONE more bare Enter and re-capture.

The LAST action of any cycle in which you injected anything MUST be a capture confirming it landed
(one per branch you touched). If you did not SEE it land, assume it did NOT and retry. This is a
specific instance of the general rule: verify the EFFECT of every Coder interaction (inject, Enter,
/compact, Esc, relaunch) — never assume the action achieved its effect.

## GEMS arbitration protocol (prevents races)
- A branch writes only its OWN candidate folder `GEMS/<date>-<branch>-<slug>/`. It does NOT edit
  "global best".
- The aggregator re-verifies independently, then promotes (records the new global best in GEMS/
  + queues a one-line "new global best <ms>" heads-up to all branches) or rejects (notes the
  reason in its own _AGG.log + one-line heads-up to the owning branch). Single-writer on global
  best = no corruption.

## Pitfalls
1. Overlapping directions waste tokens — only launch orthogonal levers (from the Coder's own enumeration).
2. Building a duplicate Hermes wiki ledger when CC already keeps its own notes = drift + wasted
   machinery. Harvest from the Coder's notes/observations; don't re-implement them.
3. Letting branches self-promote GEMS corrupts the registry — the aggregator must be sole verifier/writer.
4. Per-branch Navigators quiet (deliver=local); only the aggregator reports, or the owner drowns in N streams.
5. Don't /compact any branch; don't collide with a branch mid-work — cross-links are one-line, non-interrupting.
6. Treating `send-keys` exit 0 as "delivered" — the #1 silent failure. Text+Enter in ONE call leaves
   long messages stuck in the input box, undelivered. ALWAYS send text and Enter as SEPARATE calls and
   capture-pane to confirm it landed (see "Injecting into a Coder = NOT done until VERIFIED").
