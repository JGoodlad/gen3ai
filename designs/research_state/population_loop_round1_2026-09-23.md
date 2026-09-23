# Population loop, round 1 — PRE-REGISTRATION (2026-09-23)

**Status: REGISTERED, NOTHING LAUNCHED.** Written before any round-1 arm exists. The launch scripts,
the argvs, and the evidence that they launch are in
[`measurements/population_loop_r1_2026-09-23/`](measurements/population_loop_r1_2026-09-23/README.md).
This session trained nothing, touched no GPU, and wrote nothing under `models/`.

**The question.** When the plateau parent **G0** trains against its own best responders (fixed
exploiters that each pilot their own five offense teams, used as OPPONENTS in its training mix, with
no distillation), does a **fresh** offense exploiter trained against the result find a **smaller**
exploit than a fresh exploiter trained against a matched continuation that never faced them? The
meter is the best-response gap, `gap = P(a fresh exploiter beats the generalist it trained on) − 0.5`,
read with `python -m main.best_response_gap`.

---

## 1. The arms, one line each (every code carries its description)

| code | run name | what it is | depends on | GPU-h |
|---|---|---|---|---:|
| **A** (round-0 exploiter, dose-clean) | `ai_v13_18_teach5_offense_hidose` | **ALREADY EXISTS.** The fresh 5-team offense exploiter of G0 at a NAMED, FROZEN dose. It is `ai_v13_13`'s recipe with only `--fork-lr 2.5e-4 --fork-lr-freeze` added (ledger 2026-09-21 *THE HIGH-DOSE TEACHER IS PREPARED*), which is exactly arm A's definition | — | **0** |
| **A2** (A's seed replicate) | `ai_v13_26_popr0_exploit5_offense_s1002` | A with `--seed 1002`. Its only job is the reader instrument's RUN-level floor | — | ~4.0 |
| **B** (the loop) | `ai_v13_22_popr1_loop` | G0 +8M at G0's own frozen dose, the two offense exploiters of G0 as stable opponents at share 0.40, loss-weighted, never retired, no distillation | A exists ✓ | ~4.5 |
| **C** (the control) | `ai_v13_23_popr1_ctrl` | B's argv with ONE value changed: `--stable-opponent-selfplay-share 0.4 → 0.0` (the specialists are loaded and evaluated, never trained against) | — | ~4.5 |
| **RB** (the reader of the loop) | `ai_v13_24_popr1_read_loop` | A's recipe token-exact, re-pointed at B's final | B finished | ~4.0 |
| **RC** (the reader of the control) | `ai_v13_25_popr1_read_ctrl` | A's recipe token-exact, re-pointed at C's final | C finished | ~4.0 |

**The primary statistic is Δ = gap(RB) − gap(RC)**: the loop's best-response gap minus the control's,
both readers at A's exact recipe, dose and budget. **gap(A) = +14.00 pp** is the round-0 reference.

### Why A needs no new run (a finding, not a shortcut)

The design asked for "a fresh offense 5-team exploiter vs G0, the same five teams, recipe and budget
as `ai_v13_13` except a NAMED, FROZEN dose". `ai_v13_18_teach5_offense_hidose` is exactly that
object: its argv is `ai_v13_13`'s recorded command plus `--fork-lr 2.5e-4 --fork-lr-freeze` and a new
`--run-name` (a four-token diff, ledger 2026-09-21), `main.dose` reads it **`[FROZEN; pinned
2.50e-04]` 3.815e-8 = 1.78×**, and `main.best_response_gap` already reads it: **pooled 256/400 =
0.640, gap +14.00 pp [+9.18, +18.55]**, endpoint 0.660 (ledger 2026-09-22 *THE DOSE ACCOUNT IS DEAD*).
It was built as a teacher and refused as one (−7.15 pp on its own teams against a third party), but a
reader is not a teacher: a best-response gap asks only how hard the exploiter beats its target, and
it beat G0 exactly as hard as the 0.39× arm did. **Choosing D_e = 1.78× is what makes A free.**

---

## 2. The numbers, each with its provenance

### 2.1 The exploiter dose D_e — **1.78× (`--fork-lr 2.5e-4 --fork-lr-freeze`, 3.815e-8)**

Every run in this lineage shares `--batch-size 2048 --grad-accum-steps 32 --n-epochs 10`, so
`updates/step = 10 / 65,536 = 1.52588e-4` and `dose = lr × 1.52588e-4`; the v8 reference is
`ai_v8_14_distill3_0725` at 2.145e-8. `python -m main.dose`, run this session:

| run | role | lr (median) | dose | × v8 |
|---|---|---:|---:|---:|
| `ai_v13_12_plateau` (G0) | the parent | 2.8e-5 `[FROZEN]` | 4.272e-9 | 0.20× |
| `ai_v13_13_exploit5_offense` | round-0 offense exploiter, **unnamed** lr | 5.5e-5 (drifted 2.8e-5 → 8.36e-5) | 8.392e-9 | 0.39× |
| `ai_v13_16_teach5_offense_dist` | self-play teacher | 5.5e-5 `[FROZEN]` | 8.392e-9 | 0.39× |
| `ai_v13_18_teach5_offense_hidose` (**A**) | offense exploiter, **named** | 2.5e-4 `[FROZEN]` | 3.815e-8 | **1.78×** |
| `ai_v13_11_split_lossoff` | loss-off fold ecology (era 1) | 2.8e-5 `[FROZEN]` | 4.272e-9 | 0.20× |
| `ai_v13_05_exploit_big5starmie` | era-1 exploiter | 2.5e-4 (annealed) | 3.815e-8 | 1.78× |

- **Why 1.78×.** (1) A exists at it (zero GPU-h, and B no longer waits on a round-0 arm). (2) **A
  best-response gap is only a property of the generalist if the exploiter has CONVERGED within its
  budget**; otherwise Δ measures how fast each reader learns. A's vs-target series is flat from its
  first cycle — **0.65 / 0.58 / 0.67 / 0.66** at +0.84M / +2.84M / +4.84M / +6.84M — while the 0.39×
  arm was still climbing at its endpoint (**0.59 / 0.67 / 0.67 / 0.70**, ledger `7afa2b34`), and all
  three era-1 exploiters at 1.78× had flattened at 0.740 by +3M. (3) The two doses bought the SAME
  gap (+14.00 vs +15.75, CIs nearly coincident), so 1.78× costs nothing in exploitation. (4) The
  hidose harm (−7.15 pp on its own teams, −14.69 pp off-slice) is harm to a POLICY that is then used
  elsewhere; a reader is an instrument whose policy is discarded after its series is read.
- **Runner-up: 0.39× (`--fork-lr 5.5e-5 --fork-lr-freeze`, 8.392e-9)** — the realized dose of
  `ai_v13_13`, so the meter would accept `ai_v13_13` as a (dose-matched, lr-trajectory-unmatched)
  replicate. Rejected because its curve had not converged in 8M and it needs a new round-0 arm first.
- `ai_v13_13` cannot sit beside A in any comparison: `best_response_gap --check` on the pair REFUSES,
  `UnmatchedDoseError`, **3.815e-08 vs 8.392e-09, 4.55× apart** (rc 2; `validation/brgap_check_A_vs_A13.txt`).

### 2.2 The generalist dose D_g — **0.20× (`--fork-lr 2.8e-5 --fork-lr-freeze`, 4.272e-9), G0's own**

- **For it.** (1) **C becomes the plateau parent's own next block**: the argv is G0's recorded
  command, so the control has a banked prior — G0's last +8.06M at this dose moved the untaught 8 by
  **−1.50 pp [−3.75, +0.62], WITHIN the 3.69 floor** (ledger 2026-09-20 *THE PLATEAU IS REACHED*). No
  learning-rate step at the fork. (2) **The one loss-off precedent absorbed at this dose**:
  `ai_v13_11_split_lossoff` (0.20×, era-1 exploiters as stable opponents at 0.2 share, loss OFF) moved
  the parent's greedy win rate against them from ~0.26 (the era-1 exploiters' 0.740) to **0.54 / 0.50
  at +11M** (per-cycle `eval_results.jsonl`: vs big5starmie 0.42 0.39 0.66 0.57 0.58 0.54; vs ddtar
  0.43 0.49 0.68 0.49 0.46 0.50). Its twin with the distill loss ON (`ai_v13_07_fold1` /
  `ai_v13_08_fold1_cont`) stayed at 0.12–0.45, and the three era-2 K-ladder folds (loss ON, the
  offense exploiter as a stable opponent at 0.18 of episodes, 0.20×) stayed at **0.19–0.37 for 12M**.
  **What blocks absorption in the banked record is the distill loss, not the dose.** (3) 1.78× is
  convicted on this parent (the hidose arm, registered branch (d)); 0.39× has no banked
  self-play-GENERALIST read on this parent at all.
- 🚨 **The precedent is confounded, stated up front.** The split arm also carried
  `--distill-team-bias 0.4` and `--team-block-episodes 64` (the fold ecology): its trainee PILOTED
  the exploiters' own teams 40 % of the time. B carries neither. The ai_v7 capstone (no team bias,
  flat 0.35 share over three exploiters) rose only 0.20 → 0.30 and flattened; relaunched at 0.5 with
  PFSP it reached at best 0.46–0.47, one exploiter stuck at ~0.29 (memory archive,
  `project_tss_specialist_poc.md`). **So absorption at 0.20×
  without team bias is NOT established — which is why §4.1's manipulation check exists and why its
  failure has its own branch.**
- **Runner-up: 0.39× (`--fork-lr 5.5e-5 --fork-lr-freeze`).** It is round 2's first escalation if
  the manipulation check fails at share 0.60 too (§5, branch M). It is common-mode to B and C, so it
  would not confound Δ; what it costs is C's banked prior.

### 2.3 The stable set and share — **{A, `ai_v13_13`} at 0.40, PFSP on, retirement off**

**The opponent mix, from the code** (`wrappers.py` `_select_episode_opponent` /
`_pick_challenge_opponent` / `_pick_stable`, identical at pin `6eb9c776`; `selfplay_callback.py`
`_opponent_mix_fractions`). The per-episode coin enters the CHALLENGE bucket with probability `sf`;
inside it an un-mastered stable opponent is picked with probability `s` (the share), else the pool.
`sf = 1 − heuristic_fraction(win_rate_vs_bots)` = **0.90** for G0 (win rate vs bots ≥ 0.80 ⇒ the
0.10 `HEURISTIC_FLOOR`; `snapshot_pool.py`). **Verified on a banked arm**: `ai_v13_17_fold_k1`
(share 0.20, one stable) logs `train/selfplay_fraction 0.72`, `train/stable_fraction 0.18`,
`train/nonbot_fraction 0.90` (`main.ops.tb_read`) = 0.9 × 0.8, 0.9 × 0.2, 0.9.

| arm | bots | self-play pool | stable slice | per specialist (uniform) | per specialist (PFSP, at G0's rates) |
|---|---:|---:|---:|---:|---:|
| **B** (loop), s = 0.40 | 0.10 | **0.54** | **0.36** | 0.18 each | A ≈ 0.178, `ai_v13_13` ≈ 0.182 |
| **C** (control), s = 0.00 | 0.10 | **0.90** | 0.00 | 0 | 0 |
| G0's own block, s = 0.20, no stable | 0.10 | 0.90 | 0 | — | — |

PFSP column: `_pick_stable` weights each by `max(0.05, 1 − win_rate)` using the trainee's EMA'd greedy
eval rate against it; at G0's rates (1 − 0.640 = 0.360 vs A; 1 − 0.6575 = 0.3425 vs `ai_v13_13`) the
weights are 0.640 : 0.6575. Until the first eval pushes rates the pick is uniform. **The share is
FIXED for the whole run**: with retirement off, a specialist B has mastered keeps its slice; PFSP only
moves mass between the two.

- **Why these two and not {A, balance, stall}.** Balance and stall carry **no detected gap**: pooled
  0.525 [0.476, 0.573] and 0.455 [0.407, 0.504] against G0 (ledger `fd06fd33`). They are not best
  responses — there is nothing in them for B to absorb — and in the design's original set {A, balance, stall} under PFSP they
  would take ≈ 60 % of the slice (weights 0.525 / 0.455 against A's 0.640), cutting the offense
  exposure from 0.36 to ≈ 0.14 of episodes. The only two exploiters of G0 with a detected gap are the two offense ones (+14.00
  [+9.18, +18.55] and +15.75 [+10.97, +20.23]); they are two independent best responses on the same
  five teams, which is the population the reader will draw from. Each pilots its own five pinned
  teams (`resolve_stable_opponents`, run this session — `validation/resolve_opponents.txt`).
  **Runner-up: {A, `ai_v13_13`, balance, stall} at 0.50** (the design's original set) — the round-2
  form if round 1 turns and the question becomes cross-archetype.
- **Why 0.40.** At 0.20 over two opponents each gets 0.09 of episodes — the split arm's exposure,
  whose absorption leaned on team bias B does not have; the ai_v7 capstone's ≈ 0.12 per opponent
  flattened at 0.30. 0.40 gives **0.18 per specialist, 2× the split arm and 1.5× ai_v7's**, while
  the self-play pool keeps 0.54 of episodes (ai_v7's capstone held its bot rate, ~0.91 against a
  0.923 baseline, with the stable slice at 0.35 and then 0.5 of self-play). Higher shares are round 2's lever (branch M).
- **Retirement off: `--stable-opponent-mastered-wr 1.01`, VERIFIED in code.** The only readers of the
  value are `SelfPlayCallback._push_stable_mastered` (`if wr >= self._stable_opponent_mastered_wr`,
  where `wr` is a win fraction ≤ 1.0, so 1.01 never fires and the streak resets every cycle) and the
  startup banner (prints "≥ 101%"). There is no range check on it (`config.py` checks the share and
  the bot fraction only) and no `combination_checks` rule; `set_stable_mastered([])` is pushed each
  cycle, `_pick_floor_opponent` sees no mastered stable, and the PFSP EMA map keeps both. 1.0 would
  NOT be safe: a 100/100 eval cycle is possible and two in a row would retire the opponent.
- **Why C keeps the specialists at share 0.0 instead of dropping `--stable-opponents`.** The argv diff
  is then ONE value. C still loads both specialists in every worker and plays them in every eval
  cycle, so (a) C's own series measures its rate against them — the comparator the manipulation check
  needs, which a C without them could not produce; (b) eval load and per-worker memory match B's; (c)
  the per-episode stable coin (`rng.random() < share`) is drawn in both arms. `rng.random() < 0.0` is
  never true, so C never trains against them. The externals are kept OUT of promotion, the ELO fit and
  `win_rate_vs_pool` (`selfplay_callback.py`, "kept OUT of win_rate_vs_bots / win_rate_vs_pool / the
  ELO fit / promotion"). One residual path, stated: `_pick_challenge_opponent` returns a stable
  opponent UNCAPPED if the pool is not ready; the fork auto-seeds 20 snapshots, so this requires a
  pool-load failure — the launch script's STOP-list requires `train/stable_fraction` = 0.00 exactly.
  **Runner-up (the design's letter): C with `--stable-opponents` removed** — a two-flag diff, no
  comparator for the manipulation check, and different eval load.

### 2.4 Budgets, and the step arithmetic

A rollout is 48 envs × 2,048 = 98,304 steps; `--steps` on a fork is a TOTAL and lands on the next
rollout boundary.

| arm | fork step | `--steps` | lands | post-fork budget |
|---|---:|---:|---:|---:|
| A (exists) | 95,158,272 | 103,158,272 | 103,219,200 | **8,060,928** |
| A2 | 95,158,272 | 103,158,272 | 103,219,200 | **8,060,928** |
| B, C | 95,158,272 | 103,158,272 | **103,219,200** | 8,060,928 |
| RB, RC | 103,219,200 | 111,219,200 | 111,280,128 | **8,060,928** |

**8M for B and C**: the plateau block's own length, so C is block 2 of the plateau at identical
length; and the split arm's absorption was already visible at its third cycle (+5M). **8M for every
reader**: A's budget, which the meter matches at 2 % tolerance. The reader launch scripts REFUSE a
target whose `num_timesteps` is not 103,219,200 (the budget would move).

### 2.5 Seeds

B and C both run **seed 1001** — G0's own seed and every arm's in this lineage — at pin `6eb9c776`,
from the same parent file, with the same auto-seeded pool (20 snapshots from `ai_v13_12_plateau`).
That matches initial conditions, not trajectories: the streams part at the first episode whose coin
lands in B's stable slice (B then draws `_pick_stable`), and GPU kernels are not deterministic. RB and
RC both run **seed 1001** (A's). A2 runs **seed 1002**. **Rule 22 applies to any detection:** a
single B/C pair is a CANDIDATE until its seed-1002 replicate pair agrees (§5 branch T).

### 2.6 Pin and template

Every arm pins **`6eb9c776`** (G0's, A's, and every arm's in this lineage; the checkargs/dry-run
parser is the pinned one). B and C are built token-exactly from `ai_v13_12_plateau`'s recorded
`original_command`; A2, RB and RC from `ai_v13_18_teach5_offense_hidose`'s
(`scripts/build_argvs.py`, which prints each diff). The fork-LR inherit guard (`35258dcc`) and the
untaught-teacher guard (`1393d192`) are NOT in the pinned code; every argv names its dose anyway, and
the five offense teams are clean of the untaught 8 (ledger 2026-09-20).

---

## 3. The GPU queue (single GPU; the live `ai_v13_21_wcont_b` holds it first)

Measured wall clock per +8.06M on this box: self-play arms **4 h 05 m** (`ai_v13_16`) to **4 h 39 m**
(`ai_v13_12`); the split arm with two stable opponents ran **~2.0M/h**; exploiters **3 h 29 m**
(`ai_v13_13`) to **4 h 28 m** (A, under a concurrent CPU read). `ai_v13_21_wcont_b` attached at 12:16 PT
for +12M ⇒ free at ≈ **19:30 PT 09-23** at the plateau's 1.73M/h.

| order | arm | start ≈ | end ≈ | GPU-h |
|---:|---|---|---|---:|
| 1 | **B** (loop) | 19:30 PT 09-23 | 00:00 09-24 | 4.5 |
| 2 | **C** (control) | 00:00 | 04:30 | 4.5 |
| 3 | **RB** (reader of the loop) | 04:30 | 08:30 | 4.0 |
| 4 | **RC** (reader of the control) | 08:30 | 12:30 | 4.0 |
| 5 | **A2** (A's seed replicate) | 12:30 | 16:30 | 4.0 |
| | **total** | | | **≈ 21 (range 19–23)** |

B before C is arbitrary (they do not depend on each other); keep them ADJACENT so the box's
contention is similar. A2 depends on nothing and may run anywhere in the queue; it is REQUIRED before
the verdict (§4.2's floor), optional for a reading. **CPU reads** (§4.3), ≈ 5–6 CPU-h, run after C
finishes, overlapping RB/RC: untaught cell ≈ 2–3 h, anchors ≈ 1 h per model (3.0 s/game). The meter
itself takes seconds. **Verdict ETA ≈ 17:00 PT 09-24.**

---

## 4. The read

### 4.1 Manipulation check — DID B ABSORB WHAT IT WAS SHOWN? (read FIRST, from B's and C's own series)

Both B and C evaluate greedy-vs-greedy against A and `ai_v13_13` every cycle (100 games each,
`eval_results.jsonl` `externals`). **M = [B's pooled rate over its LAST TWO cycles against both
specialists] − [C's same], n = 400 per arm, Newcombe 95 %.** G0's own rate is 0.360 vs A and 0.3425
vs `ai_v13_13`. **ABSORBED iff M's CI lower bound > 0.** Descriptor beside it: B's per-cycle curve.
Prior **P(ABSORBED) ≈ 0.60** (the split arm absorbed, but with team bias; ai_v7 absorbed partially).

### 4.2 PRIMARY — Δ = gap(RB) − gap(RC)

```bash
export PYTHONPATH=$PYTHONPATH:src
# the matched gate: must print 0 mismatches (budget / dose / regime) — a refusal VOIDS the read
python -m main.best_response_gap ai_v13_18_teach5_offense_hidose \
    ai_v13_24_popr1_read_loop ai_v13_25_popr1_read_ctrl --check
# the read. --rounds is REQUIRED: B and C land on the SAME step, so without it their order is the
# alphabetical order of their paths (see §6, finding F2)
python -m main.best_response_gap ai_v13_18_teach5_offense_hidose \
    ai_v13_25_popr1_read_ctrl ai_v13_24_popr1_read_loop \
    --rounds ai_v13_25_popr1_read_ctrl=2 ai_v13_24_popr1_read_loop=3 \
    --json <out>/brgap_r1.json --md <out>/brgap_r1.md
# the floor, in its OWN invocation (A and A2 share a target; together they COLLAPSE — finding F3)
python -m main.best_response_gap ai_v13_26_popr0_exploit5_offense_s1002 --json <out>/brgap_A2.json
```

- **Statistic:** the `round 3 − round 2` per-archetype row (offense) = **gap(RB) − gap(RC)**, stat
  `pooled` (4 post-fork cycles × 100 = 400 games per reader), its **Newcombe 95 %** interval. The
  tool's own `VERDICT` line will read "UNREADABLE — fewer than two archetypes" and is NOT the verdict
  (finding F1); the rule below is.
- **The bar.** `F = |gap(A) − gap(A2)|` (pooled), the run-level floor of the reader at a fixed target
  (rule 3: max pairwise |Δ| over replicates in hand). **bar = max(F, 5.0 pp)**; 5.0 pp is the smallest
  effect worth a loop — one third of A's +14.00 pp gap (4.67, rounded up).
- **Rule** (the campaign's): OUTSIDE iff |Δ| > bar AND the Newcombe CI excludes the bar point on that
  side. EQUIVALENT iff the whole CI sits inside [−bar, +bar] (rule 6). Otherwise NOT DETECTED.
- 🚨 **Power, stated before the number.** At n = 400 per reader and rates near 0.6 the Newcombe
  half-width is **≈ 6.7 pp** (the stand-in read below: −8.25 [−14.93, −1.46]). With bar 5.0 a
  detection needs **Δ ≲ −11.7 pp**, i.e. gap(RB) ≲ +2.5 pp if gap(RC) sits near A's +14 — the loop must
  remove ~80 % of the gap. **EQUIVALENCE is unreachable** at this n (it needs a half-width < bar).
  Round 1 is therefore a test for a LARGE effect; a partial one reads NOT DETECTED. (Orchestrator
  option, NOT the default: raise both readers' `--eval-battles` to 200 — the two readers stay matched
  to each other, half-width ≈ 4.8 pp, but `cycle_games` then differs from A and the round-0 row needs
  `--allow-unmatched`.)
- **Secondary (confirmatory, never folded):** `python -m main.best_response_gap <reader> --play 400
  --greedy` on RB and RC — 400 FRESH endpoint-policy games each in the SAME greedy regime, seed 0,
  concurrency 1. Registered use: the sign of its gap(RB) − gap(RC) must agree with the primary for a
  detection to stand.
- **References beside it, descriptors only:** gap(RC) − gap(A) (what plain continuation did to the
  gap; prior: inside the bar) and gap(RB) − gap(A).
- **Prior:** P(OUTSIDE, below) ≈ **0.20**; P(NOT DETECTED) ≈ **0.72**; P(OUTSIDE, above) ≈ **0.08**.
  The low detection prior is mostly POWER (above), and partly the campaign's standing finding that
  exploiter gain is target-specific counterplay — a fresh best responder may find a different line.

### 4.3 KILL guards — generality must not be what pays for it (both read B against C, CRN)

- **G-U, the untaught 8** — `main.untaught_meter` with THREE refs in ONE invocation (identical teams
  and dice): `popr1_loop=models/ai_v13_22_popr1_loop/final_model.zip
  popr1_ctrl=models/ai_v13_23_popr1_ctrl/final_model.zip plateau_b1=models/ai_v13_12_plateau/final_model.zip`,
  `--opponent untaught_meter_opponent --workers 8 --seed 0`, concurrency 1, the harness of
  `split_lossoff_read_2026-09-20/scripts/harness/run_untaught.sh` verbatim but for the refs; paired
  bootstrap over the 8 TEAMS (rule 10). **Reproduction check: `plateau_b1` must read 60.19 pp**, or
  the cell is void. **Floor 3.69 pp** (the continuation read's, imported unchanged since 2026-09-20).
  **KILL iff B − C is OUTSIDE and BELOW** (|Δ| > 3.69 and the CI excludes −3.69). Descriptor: C − G0
  (predicted WITHIN, as the plateau block was).
- **G-A, the external anchor** — `main.anchors --opponent metamon:SmallRL --regime greedy` per
  `designs/ops/EXTERNAL_ANCHORS_SOP.md`, **1,200 games per model = 600 `--teamset away` + 600
  `--teamset home`**, `--team-seed 0`, `--device cpu`, `OMP_NUM_THREADS=1`, one cell at a time, for B,
  C and G0; every row's `regime_verified_decisions` must be true. **KILL iff B − C (pooled 1,200,
  Newcombe) is OUTSIDE and BELOW the 0.110 floor** — the SOP's three-seed RUN-level floor for this
  opponent (§4, measured 2026-09-16). Honest limit: that floor was measured at 100 games a cell and is
  inflated by draw noise, so this guard fires only on a collapse (≳ 11 pp).

---

## 5. Pre-registered branches — what each outcome means and what round 2 does

Read in this order; the first that applies governs.

| | condition | reading | round 2 |
|---|---|---|---|
| **V** | `best_response_gap --check` refuses, or `plateau_b1` fails to reproduce 60.19, or any arm's STOP-list fails | the read is VOID | fix and re-run the failed piece; nothing is interpreted |
| **K** | a KILL guard fires (G-U or G-A, B − C OUTSIDE BELOW) | **the loop bought exploitability with generality** — the on-slice-for-off-slice trade that convicted the era-1 fold. Δ is reported, never promoted | B is NOT a round-2 parent. Back to the orchestrator; no launch |
| **R** | Δ OUTSIDE, ABOVE | **the loop made G0 MORE exploitable** by a fresh offense best response | stop the loop at this configuration; report; counts toward the stopping rule |
| **T** | Δ OUTSIDE, BELOW, secondary agrees in sign, guards clean | **THE LOOP TURNS — a CANDIDATE** (rule 22, one pair, one seed) | FIRST the seed-1002 replicate pair (B′, C′ + their readers, ≈ 17 GPU-h). If it agrees: round 2 continues B with RB added to the stable set {A, `ai_v13_13`, RB} at 0.40 against C continued at share 0.0 with the same set, both +8M, read by fresh readers at A's recipe; and the balance/stall question opens (§2.3 runner-up set) |
| **N+** | Δ NOT DETECTED, manipulation ABSORBED | B beat the specialists it saw, and a fresh best responder is not measurably weaker: **absorption did not generalize at this power** | counts 1 of 3 toward the stopping rule. Round 2 = the loop one round deeper (RB joins B's set, as in T), since a single absorbed pair may be too narrow a population |
| **M** | Δ NOT DETECTED, manipulation NOT ABSORBED | **the loop did not engage** — Δ tested nothing about generalization | counts toward the stopping rule. Round 2 changes ONE lever on B and C together: share 0.40 → 0.60 first; if that also fails to absorb, D_g 0.20× → 0.39× (§2.2 runner-up) |

**STOPPING RULE (registered):** three rounds in which Δ is not OUTSIDE BELOW (branches R, N+, M) ⇒
**the loop does not turn at this parent**; report, and the next question is a different parent (the
same account left standing for the teacher line, ledger 2026-09-22). Round 1 counts.

---

## 6. Hazards and findings (each is a finding, not a footnote)

- **F1 — the meter cannot issue a verdict on a one-archetype round.** `build_report`'s paired mean
  needs ≥ 2 archetypes present in both rounds; with offense alone it prints "UNREADABLE" as the
  VERDICT and puts the number only in the per-archetype Newcombe row. This registration reads that
  row by its own rule (§4.2).
- **F2 — B and C are SIBLINGS, the meter only knows ROUNDS.** `assign_rounds` orders targets by STEP;
  B's and C's finals share 103,219,200, so their order falls to the path-string tie-break, and the
  delta is only computed between CONSECUTIVE rounds. `--rounds RC=2 RB=3` forces the 2 → 3 delta to be
  exactly gap(RB) − gap(RC) — demonstrated on stand-in run dirs (`validation/brgap_standin_rounds_override.txt`).
  Mapping RB and RC to the SAME round would merge them silently (F3).
- **F3 — SILENT COLLAPSE: two exploiters of one archetype on one target.** The delta keys each
  round's rows by ARCHETYPE, so A and A2 in one invocation land in round 1 together and only the
  later-sorted row reaches the delta — demonstrated: with a stand-in A2 at 0.70 the round-1 → 2 delta
  used A2's +20.00 and silently dropped A's +14.00 (`validation/brgap_standin_same_round_collapse.txt`).
  Read A2 alone. **A meter defect worth a backlog row** (refuse, or pool, same-(round, archetype) rows).
- **F4 — the Newcombe interval contains draw noise only.** It treats a reader's 400 pooled games as
  iid from one policy; the exploiter-run variance (A2) and the generalist-run variance (the seed
  replicate) are outside it. Hence the floor and rule 22.
- **F5 — `launcher --dry-run` said "would launch" where `checkargs` said "WOULD FAIL IN
  resolve_config".** On an extra stand-in (a reader forked from `ai_v13_17_fold_k1`, whose config
  records `distill_target: action`), checkargs flagged `--distill-target action requires
  --distill-coef > 0` (INHERITED) while the dry-run printed ✓ (`validation/*_STANDIN2.txt`). Not live
  for this round — B and C inherit G0's `distill_target: kl`, and B's, C's and the stand-ins' checkargs
  list NO refused combination — but the dry-run is not the executing complement of the combination
  checks that the SOP takes it to be.
- **F6 — the stand-in target is not B.** RB/RC were validated against `ai_v13_16_teach5_offense_dist`
  (a frozen-lr self-play fork of G0 landing at exactly 103,219,200). What cannot be validated before B
  exists: that B's `model_config.json` inherits nothing a reader would trip on. Stable-opponent flags
  are NOT in `model_config.json` (checked on `ai_v13_17_fold_k1`), so they cannot be inherited; the
  real reader scripts re-run checkargs + dry-run before launching.
- **F7 — the readers pilot the SAME five teams the specialists piloted.** A falling gap therefore
  measures "G no longer loses this matchup to a best responder", inside the offense subgame — the
  claim the loop makes. Generality is what the guards are for; they read B against C, not G0.
- **F8 — the specialists are not neutral opponents.** A is a WORSE pilot of its own teams than G0
  against a third party (−7.15 pp); B learns to beat two particular policies. That is the loop's
  premise; `ai_v13_13` is in the set partly so B is not shaped by one policy.
- **F9 — the plateau parent is not a registry baseline.** Everything here names
  `models/ai_v13_12_plateau/final_model.zip` by path; `main.baselines check` passes but has no entry
  for it. Suggest `python -m main.baselines set plateau_parent ai_v13_12_plateau@95158272 --reason …`
  before the read (orchestrator's call).
- **F10 — not done here, for the Training Run session:** the SOP §1.3 60-second `--debug --steps 8000`
  CPU smoke of each argv (this session was barred from training), especially **C** — share 0.0 with
  stable opponents loaded is a configuration no banked run has used. The first two minutes of the real
  launch are the only test of the preload layer (STOP-lists in each script).
- **F12 — the matchup hash cannot confirm the readers are matched.** `MatchupSpec.to_dict` hashes the
  EXPLOITER-TARGET PATH, so RB, RC and A print three different hashes by construction; only B and C
  (no exploiter target, stable opponents and share not hashed) must both print G0's `ef5242cffd`.
  Reader matching is checked by `best_response_gap --check` instead.
- **F11 — `git submodule update --init` FAILED in this worktree** (the clone could not be fetched); the
  `dist` / `node_modules` links were made and every gate and dry-run ran. Irrelevant to the argvs (the
  launch path pins `6eb9c776` in its own worktree), recorded so nobody assumes the worktree is complete.
