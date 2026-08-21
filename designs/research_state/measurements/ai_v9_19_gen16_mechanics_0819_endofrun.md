# END-OF-RUN — gen-16 `ai_v9_19_gen16_mechanics_0819` (the mechanics generation)

25,067,520 steps · 13.9 h · 0 real crashes · final aggregate WR 91.7%.
Reference: gen-15 `ai_v9_18_gen15_v8rewards_0818`. Registered in `gen16_endofrun_runbook.md`.

## §1 — INFERIOR. Number of record: DIRECT PLAY.

| instrument | Δ | SE | CI95 |
|---|---|---|---|
| **DIRECT arena (16 pairs × 400 = 6,400 games)** | **−41.57** | **4.37** | **[−50.15, −33.00]** |
| bot-mediated paired refit (`c'Σc`) | −52.38 | 9.82 | [−71.63, −33.13] |
| bootstrap cross-check (B=300, refit per draw) | −51.03 | 9.79 | [−69.82, −32.95] |
| endofrun orientation (sparse `main.elo`) | −84.9 | — | [−124.0, −45.9] |

Every gen-16 node sits below every gen-15 node; there is no ordering in which gen-16 leads.

**Frame bias = |−41.57 − (−52.38)| = 10.81 < 15 ⇒ BOT-MEDIATION VALIDATED as a fallback.** Past
cross-wall §1 deltas gain no standing caveat and direct pairs do not become mandatory at future
architecture walls. Honest rider: the offset leans the direction the PFSP account predicts, but
sits under the pre-registered bar.

### §1a Hodge — clears, on the MERGED graph

24 players, 148 edges, **488 triangles**, 1 component, 300 bootstrap reps: spine 330.5 ELO, width
raw 29.05 / null 28.39 / **excess 6.19**, **p = 0.392**, **0 cross-lineage significant cycles**.
Δ is a clean scalar contrast; no mElo-class refit required; the BT gate is recorded as VALIDATED
against cycle contamination rather than assumed.

> ⚠️ **STRUCTURAL LESSON — never evaluate curl on the cross edges alone.** The cross-only graph is
> **K(4,4), complete bipartite, and therefore contains exactly 0 triangles.** HodgeRank's curl is
> defined on triangles, so a cross-only read returns a confident zero that means nothing. A
> bipartite graph's cyclic content lives in 4-cycles, which this instrument does not measure.

## §2 — the mechanism moved substantially, and still misses

Matched scope: 92 `sentinel_*` battles at step 24M, both runs.

**Cell liveness @5M: PASS.** All four zero-init cells came off exactly 0.000 — weight norms
0.32–0.77, grad norms 2.0e-4–3.3e-4, still taking gradient. The gating precondition holds.

| bar | gen-15 | gen-16 | passes if | |
|---|---|---|---|---|
| **B1** re-click rate | 0.3736 | 0.2025 | < 0.16 | ❌ FAIL (drop is significant: z=2.44, p=0.015) |
| **B2** loop-battle rate | 0.1522 | 0.0652 | < 0.07 | ✅ PASS (n=6/92; exposure-normalised companion agrees: 0.0143→0.0078) |
| **B3** chosen-prob, loop steps | 0.985 | 0.97 | < 0.85 | ❌ FAIL |
| **B4** α/β guard | — | mixed | flat or up | ⚠️ MIXED |

**B4 is not the failure it was written to catch.** α switch top-1 on loop steps 0.694→**1.000**,
on all pivots 0.476→**0.817**, β species-correct 0.085→**0.355**; β *slot* accuracy fell
(0.947→0.70 on loop steps). The belief **reallocated**, it was not lost.

### THE KEY FINDING
`whiff_rate_per_pivot` **rose** (0.158→0.1767) and `whiff_rate_per_decision` is flat to three
decimals (0.0265→0.0268). **Gen-16 walks into the bait exactly as often as gen-15; what it stopped
doing is REPEATING it. The substrate bought repetition-suppression, not bait-avoidance.**

## §4 (order 4) — B1 is a MEMORY effect

Model-free, from the raw-protocol detector: the GAP in turns between consecutive same-pair whiffs.

| stratum | gen-15 | gen-16 | ratio |
|---|---|---|---|
| **gap ≤ 4 (in-window)** | 0.253 /whiff | **0.051** | **0.20×** |
| **gap > 4 (out-of-window)** | 0.121 | 0.152 | 1.26× |

Fisher: cut-4 OR 6.27 p=**0.0067**; cut-6 OR 7.15 p=**0.0042**; cut-8 OR 3.00 p=0.105.

**IN-WINDOW-CONCENTRATED ⇒ the eff fix (memory) owns B1.** Short-gap re-clicks fell 5×; long-gap
re-clicks did not fall at all. The substrate's behavioural contribution therefore rests on the
injection probe. *Caveat: the boundary is in TURNS; the 32-event window was not converted to turn
units. The short/long asymmetry does not depend on that — the claim "the boundary IS the window
edge" does.*

## Order 1 — α/β INJECTION: **LEARNED BUT INSUFFICIENT**, quantified

Bait-conditioned (the ledger's own population), identical code both generations:

| arm | gen-15 | gen-16 | amplification |
|---|---|---|---|
| **β kl_mean, LOOP steps** | **4.104e-07** (max\|Δp\| 0.0002) | **6.262e-04** (max\|Δp\| 0.0488) | **1,526×** |
| β kl_mean, whiff steps | 2.260e-07 | 2.166e-04 | 959× |
| **β / α on loop steps** | **0.0074** | **0.862** | — |
| argmax flips | **0** | **0** | — |

The ledger's gen-15 "bit-exactly zero" **reproduces exactly at its own scope** (4.1e-07, max|Δp|
0.0002 = the trace rounding floor). The global 3,000-state read gave only 4.3× because most states
are not baits — **conditioning is what made the effect visible**, exactly the dilution the
consequence-family amendment warns about.

**The channel arrived and carries 86% of α's influence — and still flips zero decisions**, buying
at most 0.049 of probability mass against a policy committed to the whiff at p≈0.97. That is a
credit-assignment **magnitude** problem, not a representation one.

**`switch_branch` is a THIN channel:** zeroing it leaves β's KL unchanged (1.916e-4 → 1.920e-4)
while removing **all 7** marginal argmax flips. It carries none of the mass and all of the margin.

**Critic-route deletion receipt:** `dv_mean` under injection **0.1579 → 0.0000 exactly** — the
first *causal* confirmation that the `--intent-value-reduce` hygiene deletion severed α from the
value head, rather than merely reading sub-bar on dV.

## ADDITION 1 — TD-aux is NOT the bait lever (pre-registered direction FALSIFIED)

Pre-registered: *λ>0 shows B3 falling.* It rose, monotonically. 90 `sentinel_*` battles/arm @28M.

| | λ=0.0 | λ=1.0 | λ=3.0 |
|---|---|---|---|
| **B3 loop steps** | 0.967 | 0.986 | **0.992** |
| B3 all baits | 0.867 | 0.923 | 0.928 |
| **B1 re-click** | 0.143 | 0.146 | **0.198** |
| α switch top-1 loop | 0.850 | 0.950 | 0.964 |
| β slot acc loop | 0.750 | 0.933 | 0.955 |

**The beliefs got better and the policy got more committed to the whiff.** Same dissociation the
injection probe found; TD-aux widened it. *This says nothing about the rung-2 GATES (critic
dispersion / explained variance / PIT) — a separate question on its own line.* n is small (11/13/17
re-click events); the monotonicity across three arms plus its agreement with the α/β direction is
the signal, not any single gap.

## Order 3 — the PFSP account splits in two

| @24M | gen-15 | gen-16 |
|---|---|---|
| `win_rate_vs_bots` | 0.914 | 0.882 |
| `win_rate_vs_pool` | 0.760 | 0.714 |
| `eval/ladder_elo` (live) | 2079.1 | 2007.1 |

Per-bot Δ: **heuristic −0.110**, aggressive_v2 −0.080, heuristic2 −0.060, staller −0.040,
staller_v2 −0.020, random 0.000, aggressive/setup_sweep/setup_sweep_v2 **+0.020**.

- **FRAME-BIAS variant: DEAD.** The sag is *concentrated* with four bots flat-or-better; a frame
  artifact predicts a uniform depression.
- **CAPABILITY variant: PRIME SUSPECT.** PFSP shifts experience toward hardest peers; skills vs
  under-pool-represented styles atrophy — which predicts exactly this concentration.
- `win_rate_vs_pool` **cannot adjudicate across runs** — each run plays its own snapshots, so a
  lower number is equally consistent with "I am weaker" and "my pool is stronger".

> **CORRECTION (owner, checked at metadata): gen-16 is NOT the first PFSP production run.**
> `ai_v8_14_distill3` ran `--pfsp-scale 2.5` + `--team-pfsp onesided` + `--stable-opponent-pfsp` —
> the +69 flywheel PoC. "First PFSP generation" is true of the **ai_v9 lineage only**;
> `pfsp_hardest_win_rate`'s absence before gen-16 dates the **metric**, not the feature
> (instrument-vs-subject). The load-bearing contrast: v8_14's PFSP ran over a **diverse, anchored**
> pool (stable opponents + teachers) under a distill KL anchor — every pathology mechanism had an
> opposing force. Gen-16 ran it over a **homogeneous fresh-lineage self-pool with none.** Any
> conviction must be written narrowly as **"pure PFSP over a homogeneous self-pool"**, and PFSP's
> return vehicle is the **flywheel** (teachers in pool + a uniform-sampling floor), not a tuning patch.

## §6 — the disposition was NOT pre-registered

§6 names (a) *non-inferior + cells live + bars flat* → exploiter gates, and (b) *cells never off
zero* → fix the arrival channel. **Gen-16 is neither**: cells demonstrably live, two of four bars
moved significantly, §1 INFERIOR by −41.6. Owner pre-commitment executed: direct games confirmed
Δ ≤ −40 ⇒ gen-16 INFERIOR *simpliciter*, gen-17 is a single-variable PFSP-off re-run.

**The eff fix and the substrate revert on NOTHING** — cells-live + B1/B2 movement is not a revert
case under any branch of §6.

## Reward-composition note
Startup reads `1 TERMINAL + 7 PBRS + 1 BIAS (no_progress_tax)`. **7 is correct** for this
composition: the per-term trace census (200 battles/side) measured exactly 7 distinct nonzero PBRS
terms on gen-15 — material, belief, hazard, status, boost, opp_boosts, roar. The "8" in older prose
counts `pbrs_progress`, which only exists under `--stall-pbrs` (off in gen-15/16/17 alike).

## The cross-generation arena (new instrument)
`tmp/arena/{arena_side.py,run_arena.sh,preflight.sh}`. Two processes over a 9XXX-port Showdown
server — gen-15 from a worktree pinned to its own `git_hash`, gen-16 from HEAD — because a
pre-generation checkpoint (cfg v95) cannot be loaded by current code and the two ladders could
otherwise meet only through the pinned bot anchors. Latin-square concurrency, 4 rounds × 4 pairs.
Raw rows: `gen16_gen15_direct_arena_games.jsonl`.

**Three environment faults it surfaced, each cheap to assert and expensive to discover — now all
asserted by `preflight.sh`:** a self-referential `deps/pokemon-showdown/dist/dist` symlink (plus a
twin in `node_modules`) that had broken `node build` with `ELOOP` since 2026-07-23 and made every
websocket-server path unusable; a relative log redirect after `cd` into the pinned tree that killed
the acceptor silently; and a pinned worktree with no `deps/pokemon-showdown` submodule at all.
A fourth — usernames derived as `step % 100000`, which collides for 18000000 and 24000000 — is
fixed by index-derived names and an explicit invariant assert.

> ⚠️ **`CLAUDE.md`'s worktree `ln -s` commands are only safe from a FRESH worktree.** Run from the
> main checkout, where `deps/pokemon-showdown/dist` already exists, `ln -s` creates the link
> *inside* it as `dist/dist` pointing at its parent. Guard with `[ -e "$L" ] || ln -s "$T" "$L"`.
