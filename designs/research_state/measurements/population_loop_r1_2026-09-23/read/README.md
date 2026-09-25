# Population loop round 1 — THE REGISTERED READ (2026-09-24) · VERDICT: NOT DETECTED → branch N+

The registration is [`../../../population_loop_round1_2026-09-23.md`](../../../population_loop_round1_2026-09-23.md)
(§4 the read, §5 the branches, §6 the findings). This directory is the read, carried out as
registered except where a DEVIATION is declared below with its reason.

**The codes, each time:** **B** = `ai_v13_22_popr1_loop`, the LOOP generalist (G0 +8M against the two
offense exploiters of G0 at share 0.40). **C** = `ai_v13_23_popr1_ctrl`, the NO-EXPLOITER CONTROL (B's
argv, share 0.0). **G0** = `ai_v13_12_plateau`, the plateau parent. **A** =
`ai_v13_18_teach5_offense_hidose`, the round-0 offense exploiter of G0 (1.78× dose). **A2** =
`ai_v13_26_popr0_exploit5_offense_s1002`, A's seed-1002 replicate (the reader floor). **RB** =
`ai_v13_24_popr1_read_loop`, the fresh offense exploiter that READS B; **RC** =
`ai_v13_25_popr1_read_ctrl`, the one that reads C. `ai_v13_13` = `ai_v13_13_exploit5_offense`, the
second (0.39×) offense exploiter of G0.

## Status

| piece | state |
|---|---|
| §4.1 manipulation check | **DONE — ABSORBED**, M = +8.50 pp [+1.76, +15.13] |
| §4.2 `--check` gate | **DONE — 0 mismatches**, 3 runs read (not VOID) |
| §4.2 primary Δ = gap(RB) − gap(RC) | **−10.00 pp [−16.60, −3.27]; F = 2.00 pp (A2 +12.00 vs A +14.00), bar = 5.0 → NOT DETECTED** |
| §4.3 G-U (untaught 8) | **DONE — does NOT fire**; reproduction check PASSED EXACTLY (`plateau_b1` 963/1600 = 60.19 pp) |
| §4.3 G-A (SmallRL anchors) | **DONE — does NOT fire**, B − C = −1.75 pp [−5.62, +2.13] at n = 1,200 each, all 36 units `regime_verified_decisions` true |
| §5 branch | **N+** — Δ NOT DETECTED, manipulation ABSORBED, guards clean. Counts 1 of 3 toward the stopping rule; round 2 = the loop one round deeper |

---

## 0. Trees, and every deviation from the registration

**G-U ran on `6eb9c776`** — the arms' OWN pin (every arm's `pin_history` reads `6eb9c776…`, first/last
step as registered) — from its own worktree (`gen3ai-wt/pin-6eb9c776-popr1`, cwd = that tree,
`PYTHONPATH` = its `src/`) with its OWN release `sim_bridge` (built in that worktree; `strings` shows
only that worktree's dex path). The registered reproduction check is what licenses the tree, and it
passed EXACTLY (§3). `data/` is byte-identical between `6eb9c776`, `7c511161` and main (`git diff
--stat` empty), and the Showdown submodule is `e0551883` in all three.

**G-A ran on `7c511161`** (orchestrator's ruling, 2026-09-24, relayed from the Training Run session and
re-verified here): `6eb9c776`'s `main.anchors` predates `6e25c380` (`regime_verified_decisions`, which
§4.3 requires on every row), `33338688` (`--server rust`) and `558586d0` (serial challenges).
`7c511161` is the parent of `65f22334` (the training-input boundary after it, and before `c97358e8`),
and contains `6eb9c776` and all three (`git merge-base --is-ancestor`, checked). **Its differences from
`6eb9c776` that touch play, declared:** a move-LOCKED request names the locked move (`db2e6515`); the
Rust port TIES at Showdown's 1000-turn limit instead of panicking (`1ffa13bc` — unreachable here, the
anchors forfeit at 250); the vendored teambuilder skips a trailing blank line instead of packing an
empty 7th mon (`9a189de8` — hazard H4, which on `6eb9c776` would HANG on every Metamon `competitive`
file); the Rust emission path moved to typed log calls (`6f3eda49`, byte-gated); a model-version
migration entry for v120 (`a5a649a5`); the rest is search / Rust-core scaffolding training does not
read. **B, C and G0 are all played on this one tree**, so any residual tree effect is common to all
three and cancels in B − C. Every checkpoint loaded `bare` (`model_loader` on every row).

| # | deviation | reason | what licenses it |
|---|---|---|---|
| D1 | **G-U as 4,800 per-battle rows from 192 units** (ref × team × 25 battles), not ONE `main.untaught_meter` invocation | owner rule 2026-09-24 (ORCHESTRATOR_SOP §2): long measurements are INCREMENTAL | the meter's own contract: a cell is a pure function of (ref, team index, battle index) — sim seed, both policy seeds and the pool draw are reset per battle. The driver runs `play_cells`' loop body verbatim. **Validated by executing**: one unit killed at battle 22 with a torn half-line appended, resumed → 25 rows identical to the TOOL's own shard output for the same (ref, team, battles 0–24): 16/25 wins, identical opponent-team sequence (`validation/gu_resume_test/`). And the registered reproduction check came back EXACT |
| D2 | **G-U at 4 workers**, not 8 | box cap (≤ 6 own workers; A2 + a 3-worker battery + two agents on 16 cores) | workers split teams, never change a number at concurrency 1 |
| D3 | **G-A as 36 × 100-game units at `--team-seed` ∈ {0,10,20,30,40,50}** (× away/home × B/C/G0), not one 600-game cell per teamset at `--team-seed 0` | owner rule (incremental); a 100-game unit needs its own draw | the banked `anchor_ab_continuation_2026-09-20` convention (N × 100-game sub-cells at distinct seeds, SHARED across arms). Stride 10 because the tool seeds THEIR draw at seed+1. `--seed-base` = the team seed, the same across B, C, G0: identical team draws and dice. Resume: a unit without an OK `summary.json` is moved aside (never deleted) and re-run — **exercised**: killed at game 34 of the first unit, resumed from game 1 (the partial is archived at `~/gen3ai_archive/popr1_read_2026-09-24/ga_rows_full/B_loop_away_s0.partial.*`) |
| D4 | **G-A tree `7c511161`**, not the arms' pin | §0 above | one tree for all three models |
| D5 | **The §4.2 secondary (`--play 400 --greedy` on RB and RC) was NOT run** | its registered use is only "the sign must agree … for a DETECTION to stand", and a detection is arithmetically impossible here for any F (§4) | nothing it could return changes the branch |

---

## 1. §4.1 MANIPULATION CHECK — did B absorb what it was shown? **ABSORBED**

From each arm's own `eval_results.jsonl` `externals` — 100 greedy-vs-greedy games per specialist per
cycle, `sentinel_regime {greedy: true, symmetric_teams: true}` (`scripts/manipulation_check.py` →
`manipulation_check.json`).

| cycle step | B vs A | B vs `ai_v13_13` | C vs A | C vs `ai_v13_13` |
|---:|---:|---:|---:|---:|
| 96,000,000 | 34 | 35 | 36 | 33 |
| 98,000,016 | 43 | 40 | 37 | 35 |
| 100,000,032 | 44 | 44 | 36 | 40 |
| 102,000,000 | 41 | 41 | 34 | 26 |

**M = B's last two cycles (170/400 = 0.425) − C's (136/400 = 0.340) = +8.50 pp, Newcombe 95 %
[+1.76, +15.13] → lower bound > 0 → ABSORBED.** G0's own rates were 0.360 (vs A) and 0.3425 (vs
`ai_v13_13`); C sits at them (0.350 / 0.330), B moved to 0.425 against both. Per-specialist
descriptors: vs A +7.50 [−2.03, +16.85], vs `ai_v13_13` +9.50 [+0.00, +18.76]. B's curve: 0.345 →
0.415 → 0.440 → 0.410 (flat after the second cycle). **The loop engaged** — but partially: B still
loses ~58 % to the specialists it trained against.

STOP-list facts re-read from the runs (the launch banners are not in the child logs): TB
`train/stable_fraction` **0.36** on B after the fork, **0.00** on C; `train/selfplay_fraction` 0.54 (B)
/ 0.90 (C); every eval row's `matchup_hash` **`ef5242cffd`** on both B and C; readers `9a8ed39136`
(RB) and `159aedc3b8` (RC), A `80dea7c93b` (all different, as finding F12 predicts). All runs landed on
the registered steps (B, C 103,219,200; RB, RC 111,280,128).

## 2. §4.2 PRIMARY — Δ = gap(RB) − gap(RC)

```
python -m main.best_response_gap ai_v13_18_teach5_offense_hidose ai_v13_24_popr1_read_loop ai_v13_25_popr1_read_ctrl --check
  → 0 mismatch(es); 3 run(s) read.   budget 8060928 · dose 3.8147e-08 · archetype offense on all three   (brgap_check.txt)
python -m main.best_response_gap ai_v13_18_teach5_offense_hidose ai_v13_25_popr1_read_ctrl ai_v13_24_popr1_read_loop \
    --rounds ai_v13_25_popr1_read_ctrl=2 ai_v13_24_popr1_read_loop=3 --json brgap_r1.json --md brgap_r1.md
```

(Main's meter code; it only reads run files. Finding F2 reproduced LIVE: without `--rounds` the
`--check` listing put RB in round 2 and RC in round 3 — the reverse of the registered mapping.)

| reader | target | pooled | gap | 95 % CI | per-cycle vs-target |
|---|---|---:|---:|---|---|
| A (round 0) | G0 | 256/400 | **+14.00** | [+9.18, +18.55] | 0.65 / 0.58 / 0.67 / 0.66 |
| RC | C (control) | 268/400 | **+17.00** | [+12.25, +21.43] | 0.67 / 0.68 / 0.64 / 0.69 |
| RB | B (loop) | 228/400 | **+7.00** | [+2.10, +11.76] | 0.52 / 0.59 / 0.57 / 0.60 |

**Δ = gap(RB) − gap(RC) = −10.00 pp, Newcombe 95 % [−16.60, −3.27]** (the `round 3 − round 2` row;
the tool's own VERDICT line reads "UNREADABLE — fewer than two archetypes", finding F1, and is not the
verdict). Descriptors (never folded): gap(RC) − gap(A) = +3.00 [−3.58, +9.54] (plain continuation did
not move the gap — inside the bar, as predicted); gap(RB) − gap(A) = −7.00 [−13.68, −0.23].

**A2's floor** (`ai_v13_26_popr0_exploit5_offense_s1002`, A's seed-1002 replicate; pin `6eb9c776`,
budget 8,060,928, dose 3.815e-08 = A's), read ALONE (finding F3, `brgap_A2.{json,md,txt}`):
**pooled 248/400 = 0.620, gap +12.00 pp [+7.15, +16.62]**, per cycle 0.60 / 0.67 / 0.61 / 0.60.

**F = |gap(A) − gap(A2)| = |14.00 − 12.00| = 2.00 pp → bar = max(2.00, 5.0) = 5.0 pp.**

**THE RULE (`primary_rule.py` → `primary_verdict.json`):** |Δ| = 10.00 > 5.0 (clause 1 holds), but the
CI's upper end −3.27 does NOT exclude −5.0 (clause 2 fails) → not OUTSIDE; the CI is not inside
[−5.0, +5.0] → not EQUIVALENT. **VERDICT: NOT DETECTED.**

## 3. §4.3 G-U — the untaught 8. **Does NOT fire. Reproduction check PASSED EXACTLY.**

Tree `6eb9c776` (§0). `untaught_meter_opponent` (= `ai_v9_29_rev1_0823@24,000,000`), the untaught-8
manifest in canonical order, `--seed 0`, concurrency 1, 200 games/team, **4,800 battles, 0 timeouts**.
Paired bootstrap VERBATIM from `hidose_delta.py` (20,000 draws, ONE shared team index set, seed
20260915, the TEAM is the unit). Files: `untaught_popr1.{json,md}` (the tool's own `aggregate` and
renderer), `gu_guard.json`, rows in `gu_rows/`.

| ref | untaught-8 | wins/finished |
|---|---:|---:|
| `popr1_loop` (B, the loop) | **58.81 pp** | 941/1600 |
| `popr1_ctrl` (C, the control) | **61.25 pp** | 980/1600 |
| `plateau_b1` (G0, the plateau parent) | **60.19 pp** ✅ | **963/1600 — the registered 60.19 EXACTLY** |

| contrast | Δ pp | 95 % CI | floor 3.69 | reading |
|---|---:|---|---|---|
| **B − C (the guard)** | **−2.44** | **[−6.31, +0.88]** | \|Δ\| < floor; CI contains −3.69 | **WITHIN — KILL does NOT fire** |
| C − G0 (descriptor) | +1.06 | [−2.06, +3.94] | within | predicted WITHIN — **came in WITHIN** (C is the plateau's block 2) |
| B − G0 (descriptor) | −1.37 | [−3.75, +1.13] | within | — |

Per team, B − C: +0.015 / 0.000 / **−0.110** / **−0.110** / +0.015 / 0.000 / +0.030 / −0.035 — the
negative point estimate is carried by two teams (`U_ce35b736`, `U_9909f2e9`), where C rose above G0
(+0.075, +0.045) rather than B falling far below it (−0.035, −0.065). NOT DETECTED at the registered
standard; "within floor" is not "no cost".

## 4. §4.3 G-A — SmallRL anchors (greedy vs greedy)

Tree `7c511161` (§0). `python -m main.anchors --model <final_model.zip> --opponent metamon:SmallRL
--regime greedy --teamset {away,home} --games 100 --team-seed S --seed-base S --device cpu --nice 15`,
`OMP_NUM_THREADS=1`, `--server rust` (the default: the in-repo front end, NO Node server), one cell at
a time, S ∈ {0,10,20,30,40,50}, the same S for all three models (D3). **3,600 games; all 36 units
status OK, n = 100, `regime_verified_decisions` true, `argmax_match_rate` 1.0000 on every unit, no
`team_source_asymmetry`, every model loaded `bare`.** 16 games (0.44 %) hit the 250-turn forfeit, 5
ties. Regime: **greedy vs greedy**. Files: `ga_guard.json` (`scripts/ga_aggregate.py`), units in
`ga_rows/`.

| model | pooled 1,200 | rate | Wilson 95 % | away 600 | home 600 |
|---|---:|---:|---|---:|---:|
| B (the loop) | 733 | **0.611** | [0.583, 0.638] | 348 | 385 |
| C (the control) | 754 | **0.628** | [0.601, 0.655] | 352 | 402 |
| G0 (the plateau parent) | 744 | **0.620** | [0.592, 0.647] | 341 | 403 |

| contrast | Δ | Newcombe 95 % | floor 0.110 | reading |
|---|---:|---|---|---|
| **B − C (the guard)** | **−0.0175** | **[−0.0562, +0.0213]** | \|Δ\| < 0.110; CI does not exclude −0.110 | **WITHIN — KILL does NOT fire** |
| B − C, away (descriptor) | −0.0067 | [−0.0623, +0.0490] | — | — |
| B − C, home (descriptor) | −0.0283 | [−0.0818, +0.0254] | — | — |
| C − G0 (descriptor) | +0.0083 | [−0.0304, +0.0470] | — | — |
| B − G0 (descriptor) | −0.0092 | [−0.0480, +0.0297] | — | — |

The registered honest limit applies: the 0.110 floor was measured at 100 games a cell, so this guard
fires only on a collapse. Beyond the guard, B − C's own CI ([−5.6, +2.1] pp) bounds any anchor cost of
the loop at ≲ 6 pp; NOT DETECTED, not "equivalent" (no equivalence rule is registered for G-A).

⚠️ **6 of 36 units have `peer_clean` false** — every one a COMPLETE half (50/50 games,
`argmax_match_rate` 1.0, `regime_verified_decisions` true) whose Metamon peer died of a post-game
`RecursionError` ("Battle is already finished, call reset", ~2,000 recursive resets), each unit
containing ≥ 1 of our 250-turn forfeits. This is hazard H17's defect — it cost no games — **but 4 of
the 6 are in the `ours_challenge` half, where Metamon is the ACCEPTOR** (`C_ctrl_home_s10`,
`G0_plateau_away_s0`, `…_s10`, `…_s40`); only 2 are in `peer_challenge`. See finding H-3 (§6).

## 5. The §5 branch

Read in the registered order (the first that applies governs):

| branch | condition | status |
|---|---|---|
| **V** (void) | `--check` refuses / `plateau_b1` ≠ 60.19 / a STOP-list fails | **ruled out** — 0 mismatches; 963/1600 = 60.19 exactly; stable fractions 0.36 / 0.00, hashes `ef5242cffd` on both, steps as registered |
| **K** (kill) | G-U or G-A: B − C OUTSIDE BELOW | **ruled out** — G-U −2.44 [−6.31, +0.88] pp vs 3.69; G-A −1.75 [−5.62, +2.13] pp vs 11.0 |
| **R** (rose) | Δ OUTSIDE, ABOVE | **impossible** — Δ = −10.00 < 0 |
| **T** (turns) | Δ OUTSIDE, BELOW (+ secondary, guards) | **impossible for every F** — the CI's upper end −3.27 never clears −bar ≤ −5.0 |
| **N+** | Δ NOT DETECTED, ABSORBED | **✅ GOVERNS** — NOT DETECTED at bar 5.0 (F = 2.00); ABSORBED (M lower bound +1.76) |
| M | Δ NOT DETECTED, NOT ABSORBED | ruled out (ABSORBED) |

**The registered consequence of N+ (§5, verbatim in substance):** *"B beat the specialists it saw, and a
fresh best responder is not measurably weaker: absorption did not generalize at this power"* — it
**counts 1 of 3 toward the stopping rule**, and **round 2 = the loop one round deeper**: B continued
with **RB added to its stable set {A, `ai_v13_13`, RB} at share 0.40**, against C continued at share
0.0 with the same set loaded, both +8M, each read by a fresh offense reader at A's exact recipe.

(The EQUIVALENT case, which §5 has no row for, needed F ≥ 16.60; F = 2.00, so it did not arise.)

**What the number says, beside the rule (a descriptor, not a verdict).** The point estimate is in the
registered direction and large (−10.0 pp; gap(RB) +7.0 vs gap(RC) +17.0 and gap(A) +14.0), and its
CI excludes zero — but the registered bar is 5.0 and the test was
powered for ≈ −11.7 pp. Rule 22 and the bar exist because the per-reader Newcombe interval holds draw
noise only (F4): one B/C pair at one seed. The reader curves add a caveat (H-1 below). The honest
reading is a CANDIDATE-SIZED point estimate the registered test does not detect. The reader floor
itself is small (F = 2.00 pp between two seeds on one target), so the bar was set by the registered
5.0 minimum, not by instrument noise — what failed was the CI's reach (−3.27), i.e. POWER at n = 400.

## 6. Hazards and findings from this read

- **H-1 — RB's vs-target curve is still RISING; RC's and A's are flat.** RB 0.52 / 0.59 / 0.57 / 0.60;
  RC 0.67 / 0.68 / 0.64 / 0.69; A 0.65 / 0.58 / 0.67 / 0.66. §2.1 chose D_e = 1.78× so that "a
  best-response gap is a property of the generalist, not of how fast the reader learns"; for RB that
  premise is not established — part of Δ may be RB learning more slowly against B, not converging
  lower. The ENDPOINT contrast (0.60 − 0.69 = −9 pp at +6.8M) is close to the pooled one, so the
  sign does not hinge on it; the size might. Round 2's readers (or a longer RB) would show it.
- **H-2 — finding F2 reproduced LIVE.** The unassisted `--check` listing put RB in round 2 and RC in
  round 3 — the reverse of §4.2's mapping; `--rounds` fixed it as registered. F1 (UNREADABLE verdict
  line) also reproduced. F3 is honoured by reading A2 alone.
- **H-3 — the anchors SOP's H17 is WRONG about the ROLE.** H17 says every Metamon post-game
  `RecursionError` is "in the half where Metamon CHALLENGES". Here 4 of 6 were in the ACCEPTOR half
  (`ours_challenge`), with Metamon's own "Battle is already finished, call reset" recursion. Every one
  was a complete half at `argmax_match_rate` 1.0 and cost no games, but `main.anchors` stamps
  `peer_recursion_upstream` only on a complete `peer_challenge` half, so these 4 rows carry
  `peer_clean` false with no named cause. A doc + tool finding for the anchors owner (not fixed here).
- **H-4 — the launch banners are not in the runs' child logs** (`launcher_child*.log` start at
  "child attached"), so the STOP-list lines (ForkLR, STABLE, MATCHUP) could only be re-verified from
  TB scalars and eval-row `matchup_hash`, not from the banners themselves.
- **H-5 — B absorbed PARTIALLY.** B's rate against the specialists rose 0.345 → 0.41–0.44 and flattened
  after the second cycle; it still loses ≈ 58 % to the two policies it trained against at 0.36 of its
  episodes. ABSORBED (by the registered rule) is not "neutralised".
- **H-6 — the G-U driver re-runs the meter's loop body outside the tool.** It is licensed by the
  meter's pure-function contract, the exact reproduction (963/1600) and a direct row-for-row match to
  the tool's own shard output on a killed-and-resumed unit; a future change to `play_cells` would not
  reach this driver (it runs the pinned tree's engine, by design).
- **H-7 — G-A's durable grain is the 100-game unit, not the game**: `main.anchors` writes `games.jsonl`
  only when a cell ends, so a kill loses up to one unit (~8 min here). Resume moves the partial aside
  (archived: `~/gen3ai_archive/popr1_read_2026-09-24/ga_rows_full/B_loop_away_s0.partial.*`, the kill test) and replays it from game 1.
- **H-8 — the §3 queue's CPU estimates were optimistic under this box's load (25–38 on 16 cores):**
  G-A ran 8.5 min per 100 games (the SOP's 3.0 s/game → ~5 min), and A2 trained at 378 fps, putting
  the verdict after the registered ≈ 17:00 PT ETA.
- **H-9 — `git submodule update --init` SUCCEEDED in both pin worktrees here** (the registration's F11
  failed); the untaught teambuilder validates teams through Node, so a pin worktree without the
  submodule + `dist`/`node_modules` links cannot run G-U at all (`Cannot find module
  …/deps/pokemon-showdown`) — an easy trap for the next pinned read.
- No poke-env READING defect was hit. No timeout anywhere (0 / 4,800 untaught battles; 36 / 36
  anchor units OK).

## Files

| file | what |
|---|---|
| `manipulation_check.json` · `scripts/manipulation_check.py` | §4.1 |
| `brgap_check.txt` · `brgap_r1.{json,md,txt}` | §4.2 gate + the registered read |
| `brgap_A2.{json,md,txt}` | A2's own invocation (finding F3) |
| `primary_verdict.json` · `scripts/primary_rule.py` | the §4.2 rule applied |
| `gu_rows/` · `scripts/gu_driver.py` · `scripts/gu_env.sh` | G-U: 4,800 per-battle rows, the resumable driver, its environment |
| `untaught_popr1.{json,md}` · `gu_guard.json` · `gu_aggregate.txt` | G-U aggregate + guard rule + reproduction check |
| `ga_rows/` · `scripts/ga_driver.sh` · `scripts/ga_aggregate.py` | G-A: 36 unit directories (each `games.jsonl` + `summary.json` + logs), driver, aggregate |
| `ga_guard.json` | G-A aggregate + guard rule |
| `validation/gu_resume_test/` | the G-U kill + torn-line + resume test against the tool's own shard output |

Committed per G-A unit (the `anchor_ab_continuation_2026-09-20` convention): `summary.json`,
`games.jsonl.gz` and the two peer reports. The COMPLETE unit trees — server and peer logs, Metamon's
battle CSVs, the killed partial — are in `~/gen3ai_archive/popr1_read_2026-09-24/ga_rows_full/` (and
the G-U rows again in `…/gu_rows_full/`).
