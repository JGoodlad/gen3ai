# PRE-REGISTRATION — THE CONTINUATION CONTROL, THE ERA's G5 CELL (`ai_v13_09_wcont`)

**Committed BEFORE any registered number in this directory exists.** The GO is the completion entry
of 2026-09-19 (`e77963c4`, `ai_v13_09_wcont` COMPLETE at 87,097,344), whose own closing words hand
this read its question: *"if the control gains the same off-slice amount as fold-1's +5.4/+5.1 pp,
that lean is CONTINUATION, not teaching."*

Its two predecessors are [`fold1_read_2026-09-19/`](../fold1_read_2026-09-19/) (the fold's +6M
registered read, branch **(d) NOT DETECTED**) and
[`fold1_cont_read_2026-09-19/`](../fold1_cont_read_2026-09-19/) (the converged endpoint, **no
registered branch met**, DDTar `+0.1538` the first row this programme ever cleared). **Every
instrument, opponent, manifest, manifest ORDER, seed, concurrency, cell size and decision rule
below is REUSED VERBATIM from them**, so that the three reads are ONE series and the path × depth
table is a single measurement rather than three stitched together.

🚨 **EVERY NUMBER OF THE FOLD PATH IS BANKED AND IS IMPORTED, NOT RE-DERIVED** (§1.2), under the
same reproduction-check warrant those reads used. This job measures the CONTROL and arm W only.

---

## 0. The arm, and what is already known about it

`ai_v13_09_wcont` — **the era's G5 cell**: a **FORK of `ai_v13_02_flywheel_winprob` (arm W, the 75M
win-prob flywheel arm) at 75,005,952 with NO TEACHERS**, `--distill-coef 0.0`, the same
`--fork-lr 2.8e-5 --fork-lr-freeze` (realized dose **4.272e-9 = 0.20× the v8 reference**,
`[FROZEN; pinned 2.80e-05]` — **identical to the fold path's**), pool auto-seeded from arm W's 20
snapshots, `--checkpoint-every-steps 500000`, seed 1001, pin `6eb9c776` in **ONE** row, config v119
/ `gen3_critic_route_wave_v1`, **+12,000,000 steps exactly to 87,097,344** — **the fold path's own
endpoint step, to the step**. 07:07 → 16:50 PT, 9 h 43 m, 3 restarts, zero crashes.

**Post-fork step counting is CUMULATIVE from 75,005,952**, the convention both fold reads use, so
the control spans +0 → +12.09M on the same axis as the fold path.

### Banked and NOT re-derived here (from `e77963c4`)

* `grep -c DISTILL` on the launcher log is **0** across all 9 h 43 m — no teachers, no distill loss,
  no anchor monitor, for the whole run.
* `[MATCHUP ef5242cffd]` — **arm W's own, unchanged**; the control stayed in arm W's ladder era while
  the fold path moved to `0a7b730a4d`.
* **Bots and G7 are indistinguishable from the fold path** at every matched post-fork point (control
  0.891–0.933 vs fold path 0.925–0.934; G7 worst 1.083 vs 1.090, both under bar).
* 🚨 **ENTROPY: `H_end` 0.9447 (control) vs 0.7465 (fold path) — a 0.198-nat gap, ~10× the 75M
  run-level floor's 0.019**, in the direction an action-target top-1 distillation predicts.
* Live-frame critic rows differ (control ECE 0.0060 / skill 0.3004 / Brier 0.1438 vs the fold
  path's 0.0206 / 0.2766 / 0.1301) — ⚠️ **rule 20: live-frame, single-update, NOT evidence about
  the offline population.** They are not a row here.

### 🚨 Five facts LOOKED AT while scoping this job, and disclosed here rather than hidden

Pre-registration is worthless if the registrant has already seen the answer. Five facts about the
control's **tree and its own sidecars** were read while scoping, and all five are declared:

1. **The control's checkpoint grid**, listed in §1.1. Its `checkpoint_78006048_steps.zip` sits at
   **exactly the same step number** as the fold's +3M checkpoint, and its `final_model.zip` at
   **exactly the same step** as the fold path's endpoint (87,097,344). Only the +6M point differs,
   by 42,480 steps (§1.1's ⚠️).
2. **`metadata.json`**: `lineage.role` = `fork`, `fork_parent` = arm W `final_model.zip`
   @75,005,952 resolved `[rung=explicit_zip rule=explicit_zip]`, **`teachers: []`**,
   `exploiter_target: null`, `pin_history` **ONE row** `6eb9c776`, 75,005,952 → 87,097,344,
   `pool_seeded_from` arm W's `snapshots/`, 20 files.
3. 🚨 **THE CONTROL PROMOTED SNAPSHOTS AND HAS A `snapshot_ladder/`; THE FOLD PATH PROMOTED ZERO
   AND HAS NONE.** The control's `snapshots/` holds 20 zips of which **five are its OWN post-fork
   promotions** (78.0M, 80.0M, 82.0M, 84.0M, 86.0M), the five oldest arm-W seeds (26–34M) having
   been evicted. ⇒ **A STRUCTURAL FINDING, registered here as a DESCRIPTOR with no bar** (§3.5):
   over the same 12.09M steps against the same auto-seeded arm-W pool at the same 0.55 gate, the
   plain continuation cleared the gate five times and the fold path never once.
4. **`model_config.json`** carries `distill_target: "kl"`, `distill_topk: 1`, `distill_beta: 1.0`,
   `distill_gate: "none"` — ⚠️ **INHERITED from arm W's checkpoint config and INERT at
   `--distill-coef 0.0` with no teacher**, the standing instance of "any flag the argv does not NAME
   re-resolves". Declared as hazard **W-C** in advance.
5. The ledger's own banked descriptor table (bots / G7 / entropy / live-frame critic), quoted above,
   **not re-derived as a finding**.

**Nothing about the control's piloting COMPETENCE has been measured.** The untaught meter, the
per-slice piloting row and the anchor cell have **no numbers of any kind** at the time of this
commit. No `win_rate_vs_pool` series, no `ladder.json`, no TensorBoard scalar of this run has been
opened.

---

## 1. The comparators, by name and by RESOLVED file

| role | run | resolved file | cumulative post-fork |
|---|---|---|---|
| **CONTROL, +3M** | `ai_v13_09_wcont` | `checkpoints/checkpoint_78006048_steps.zip` | **+3.0001M** |
| **CONTROL, +6M** | `ai_v13_09_wcont` | `checkpoints/checkpoint_81143280_steps.zip` | **+6.1373M** |
| **CONTROL, +12M** | `ai_v13_09_wcont` | `final_model.zip` @ 87,097,344 | **+12.0914M** |
| **PARENT / BASELINE — arm W** | `ai_v13_02_flywheel_winprob` | `final_model.zip` @ 75,005,952 | +0 |
| the SEED FLOOR arm — W_b | `ai_v13_04_flywheel_winprob_b` | `final_model.zip` @ 75,005,952 | +0 (IMPORTED) |
| **fold path, +3M** | `ai_v13_07_fold1` | `checkpoints/checkpoint_78006048_steps.zip` | +3.0001M (IMPORTED) |
| **fold path, +6M** | `ai_v13_07_fold1` | `final_model.zip` @ 81,100,800 | +6.0948M (IMPORTED) |
| **fold path, +12M** | `ai_v13_08_fold1_cont` | `final_model.zip` @ 87,097,344 | +12.0914M (IMPORTED) |
| teacher t1 (Big-5, balance) | `ai_v13_05_exploit_big5starmie` | `final_model.zip` @ 83,066,880 | (IMPORTED) |
| teacher t2 (DDTar, offense) | `ai_v13_06_exploit_ddtar_spikes` | `final_model.zip` @ 83,066,880 | (IMPORTED) |

### 1.1 ⚠️ THE DEPTH MATCH IS EXACT AT TWO OF THREE POINTS AND 42,480 STEPS OFF AT THE THIRD

Checkpoints land every 500k and the two runs' restart boundaries differ, so the grids are not
identical. Named in advance so the choice cannot be fitted to a result:

| grid point | control's file | fold path's file | offset |
|---|---|---|---|
| **+3M** | `78,006,048` | `78,006,048` | **0 steps — the same step number** |
| **+6M** | `81,143,280` (+6.1373M) | `81,100,800` (+6.0948M) | **control is 42,480 steps DEEPER (+0.042M)** |
| **+12M** | `87,097,344` (final) | `87,097,344` (final) | **0 steps — the same step number** |

The +6M offset is **0.35 % of the fold path's own post-fork budget** and the control is the DEEPER
of the two, i.e. if seniority is what drives the untaught row the offset flatters the CONTROL by a
hair. Recorded as hazard **W-D**, not corrected.

### 1.2 🚨 THE DECLARED IMPORTS, AND THE WARRANT FOR THEM — registered before the first battle

**The whole fold path, W_b and both teachers are NOT re-run.** Their cells are taken verbatim from
the two banked reads:

| imported ref | untaught source | slice source |
|---|---|---|
| `fold_p3M`, `fold_p6M` | `fold1_cont_read_2026-09-19/out/untaught/untaught_registry.json` | `fold1_read_2026-09-19/out/fold_slice_read.json` |
| `cont_p12M` (the fold path's endpoint) | as above | `fold1_cont_read_2026-09-19/out/cont_slice_read.json` |
| `armWb`, `t1_big5`, `t2_ddtar` | as above | `fold1_read_2026-09-19/out/fold_slice_read.json` |

🚨 **THE WARRANT IS THE ARM-W REPRODUCTION CHECK, AND IT IS REGISTERED AS PASS/FAIL HERE.** `armW`
IS re-run, in both invocations, and it must return:

* **untaught: 46.19 pp = 739 / 1600**, and **all eight PER-TEAM rows identical** to the banked ones;
* **slice: 394 / 800 on Big-5 and 376 / 800 on DDTar.**

The meter is deterministic at `--seed 0` / concurrency 1 — demonstrated on 2026-09-16, 2026-09-18,
in the +6M read and five more times inside the convergence read, across three DIFFERENT ref lists —
and a cell is a pure function of (ref, team index, battle index), which is the property that
licenses both `--workers` and a changed ref list. **If arm W returns anything else, every imported
cell is VOID and the affected row is reported as INVALID rather than patched.** Registering the
import here is the alternative to spending ~9,600 battles re-deriving rows that are the same games.

### 1.3 The floors, and where each one comes from

| row | floor | provenance |
|---|---:|---|
| untaught meter | **3.69 pp** | `\|arm W − W_b\|`, IMPORTED from `flywheel_wb_floor_read_2026-09-18/`, a **75M FRESH-ARM seed pair**; re-measured on the identical games inside this job's own invocation |
| per-slice, Big-5 cell | **0.0475** | `\|arm W − W_b\|` **MEASURED on this very cell**, same opponent, same 800 games (`fold1_read_2026-09-19/`); arm W's half re-measured here |
| per-slice, DDTar cell | **0.0850** | as above |
| `SmallRL` greedy away | **0.090** | `\|arm W − W_b\|`, IMPORTED; itself inside the anchors SOP's three-seed 0.110 |

🚨 **ONE PAIR BOUNDS A FLOOR; IT DOES NOT ESTIMATE ONE** (rules 19/22). No CI attaches to a floor,
and WITHIN FLOOR is never "equivalent" (rule 6). 🚨 **A floor is a property of the DEPTH and the
REGIME** (rule 3): 3.69 pp and 0.090 are fresh-arm imports onto a CONTINUATION and a twice-forked
FOLD, labelled as IMPORTS in every row they bar. The meter's own **fold/continuation** floors for
reference: 1.19 (END-depth replicate), 1.66 (frozen-dose), 2.46 (K=6), 4.27 (controller-live) and
**1.00 pp (the gen-era G5 three-continuation-arm cell — the nearest in KIND to this arm, and the
one this read would most like to have at 75M and does not)**.

---

## 2. THE DECISION RULE — fixed here, copied verbatim from both predecessors

> A contrast is **OUTSIDE THE FLOOR** iff **(a)** `|Δ| > floor` **AND** **(b)** the Δ's own 95 % CI
> **excludes the floor POINT**. Otherwise **WITHIN FLOOR at n = 2** — never "equivalent".

**Two sign conventions, both used, both labelled at every use:**

* `Δ_path = <path> − arm W` — POSITIVE means ahead of the frozen parent. This is the convention of
  both earlier reads and of every trajectory row.
* 🚨 `Δ_extract = fold path − control` — **the new quantity this read exists to produce.** POSITIVE
  means the fold's teaching bought something the plain continuation did not.

🚨 **CLAUSE (b) IS KNOWN IN ADVANCE TO BE NEAR-UNSATISFIABLE ON ROW 1, AND THE RULE STILL STANDS.**
The two earlier reads measured the untaught contrast's team-clustered CI half-width at **≈3.8–4.0 pp
on the fold's own points and ≈4.9–6.2 pp on the continuation's**, against a 3.69 pp floor — so
clause (b) can pass only at **|Δ| ≳ 7.5 pp**, and more on the wider points. This is registered here,
before any number, as a property of the bar and not as an escape from it. A CI clear of ZERO is
reported separately and **is not a verdict**. The honest lever is a second ARM (rules 19/22), never
more games.

🚨 **AND THE POOLING TRAP IS PRE-DECLARED AGAIN.** At the fold's +6M, pooled over both taught teams
the fold read **+0.003** over its parent while the per-team rows were **−0.061** and **+0.068**,
each CI clear of zero. **The per-team rows are the result at every depth here; any pooled number is
a labelled footnote and no verdict is taken from one** (rule 10).

---

## 3. The registered rows, their bars, and the prediction attached to each

### 3.1 Row 1 — THE UNTAUGHT METER (off-slice; this IS the collateral read, and THE row of this job)

**Instrument, REUSED VERBATIM.** `python -m main.untaught_meter`, registry opponent
`untaught_meter_opponent` (= `ai_v9_29_rev1_0823@24,000,000`), the untaught-8 manifest in its
canonical order, **200 games/team, `--seed 0`, concurrency 1**, sharded over teams, **all four
measured refs (`wcont_p3M`, `wcont_p6M`, `wcont_p12M`, `armW`) in ONE invocation**, 6,400 battles.
The fold path's three depths and W_b are IMPORTED under §1.2's warrant.

**Bars, both of them.**

1. `|Δ_path(control − armW)| > 3.69 pp` AND CI excludes the floor point — the control's own G5
   reading, at each of the three depths.
2. 🚨 `|Δ_extract(fold path − control)| > 3.69 pp` AND CI excludes the floor point — **the paired
   extraction row**, computed on the meter's own per-team rows over ONE shared bootstrap index set
   (rule 10, team the unit, 20,000 draws, seed 20260915 — verbatim from `floor_untaught_delta.py`),
   at each of the three matched depths.

**Prediction (mine, before the games), and the reasoning stated so it can be scored.**

*Against the control gaining:* §2.2 is three draws on two gen-era parents at two depths and **no
gen-era parent has been observed to gain from ordinary continued training** (G5 −1.92 pp against a
1.00 floor; M9 −0.37 and −4.56).

*For the control gaining:* the parent here is not a gen-era parent. It is a 75M `--critic winprob`
arm, and 🚨 **the control PROMOTED FIVE SNAPSHOTS past a 0.55 gate against its own frozen
auto-seeded pool** (§0.5, disclosed) — i.e. by its own run's evidence it was still improving, which
is exactly the property §2.2's parents lacked. The fold path's off-slice row moved in its first 3M
and was then **flat to +12M (+0.62 pp over the last 50 % of the budget)** — the shape of extra
training rather than of accumulating teaching. And the ledger records the two paths as
**indistinguishable on bots and G7 at every matched point**.

| | P |
|---|---|
| Δ_path(control) is POSITIVE at +12M | **0.80** |
| Δ_path(control) exceeds **+3.69 pp as a POINT estimate** at ≥ 2 of the 3 depths | **0.55** |
| Δ_path(control) clears the floor by BOTH clauses at ≥ 1 depth | **0.12** |
| **branch (a)'s clause** — `\|Δ_extract\|` ≤ 3.69 pp (point) at ≥ 2 of 3 depths | **0.55** |
| **branch (b)'s clause** — `\|Δ_path(control)\|` < 3.69 pp at ≥ 2 of 3 AND Δ_extract clears BOTH clauses at ≥ 1 depth | **0.08** |
| Δ_extract is POSITIVE at +12M (the fold path ahead of the control off-slice) | **0.45** |
| the control's untaught level at +12M is within ±2 pp of the fold path's 51.94 | **0.45** |

### 3.2 Row 2 — PER-SLICE PILOTING: 🚨 THE TEAM-DIFFERENTIAL CONTINUATION READ

**Instrument, REUSED VERBATIM.** The same `main.untaught_meter` engine pointed at the **2-team
taught-slice manifest in TEACHER ORDER** (t1 Big-5 first, t2 DDTar second — **the order is the seed
offset; re-ordering it changes every game**), against **the untaught meter's own fixed opponent**
(`untaught_meter_opponent`) and **not** arm W's pool sentinels, because a sentinel is the trainee's
OWN snapshot and arm W's differ from W_b's (floor-read hazard F-G), which would make the seed floor
and the treatment contrast face different opponents. **800 games per (ref, team), `--seed 0`**, four
ref-shards × 2 workers. Measured refs: `wcont_p3M`, `wcont_p6M`, `wcont_p12M` and **`armW` — the
reproduction check that warrants every import**. 6,400 battles.

⚠️ **The control never trained on these two teams with any bias.** It has no `--distill-team-bias`
and no teacher piloting them in its stable pool; it played the ordinary 719-team pool for 12.09M
steps. **That is exactly what makes it the control for the split** — and it also means this row
cannot distinguish "a continuation gets better at DDTar-like teams" from "a continuation gets
better at everything and DDTar is where it shows".

**Bar.** Per team: `Δ_path(control − armW)` against that cell's own **measured** seed floor (0.0475
Big-5, 0.0850 DDTar), clause (a) then clause (b) on the Newcombe interval. ⚠️ **Newcombe is
CONSERVATIVE here** — the games are paired under CRN and the tool retains no per-battle outcome
vector, so the pairing cannot be exploited. A cluster bootstrap over TWO teams is not a CI and none
is printed. **Also registered: `Δ_extract` per team**, fold path − control, same intervals.

🚨 **THE REGISTERED SPLIT STATISTIC.** The quantity that decides branches (c)/(d) is the control's
own **`Big-5 minus DDTar` difference-of-deltas at +12M**, against the fold path's
**−0.1287 at +6M → −0.2063 at +12M**. No interval is printed for a difference of two independent
differences on two cells — that is a DESCRIPTOR by construction, as the convergence read recorded,
and a bar on it would invite a verdict this design cannot carry.

| | P |
|---|---|
| the control's DDTar delta at +12M is POSITIVE | **0.55** |
| the control's Big-5 delta at +12M is NEGATIVE | **0.45** |
| the control's DDTar delta clears its 0.0850 floor by BOTH clauses | **0.15** |
| **branch (c)'s clause** — the control's +12M signs match the fold path's (DDTar +, Big-5 −) AND its Big-5 − DDTar ≤ −0.103 (half the fold path's −0.2063) | **0.20** |
| **branch (d)'s clause** — the control's Big-5 − DDTar ≥ 0, or the sign pattern is absent | **0.60** |
| the control's DDTar delta exceeds the fold path's +0.1538 at +12M | **0.08** |

### 3.3 Row 3 — EXTERNAL ANCHOR: Metamon `SmallRL`, greedy, AWAY team set, at +12M

**Instrument.** `python -m main.anchors --opponent metamon:SmallRL --regime greedy --teamset away
--games 100 --device cpu`, the tool of record, the convergence read's own `run_away_cell.sh` with
the ref swapped and the port inside this job's 9400–9499 band. 🚨 `--out` under the job tmp, never
the cwd.

**Comparators, all banked:** arm W **0.500** [0.404, 0.596]; W_b **0.590**; fold @ +6.09M **0.540**;
fold path @ +12.09M **0.430** [0.337, 0.528]; floor `|W − W_b|` = **0.090**.

**Bar.** `|Δ| > 0.090` AND the CI excludes 0.090. **Registered as DESCRIPTIVE**: at 100 games the
cell resolves ~±0.10 against a 0.090 run-level floor, so it cannot separate the arms whatever they
did. Saying so now rather than afterwards is the point of registering it.

| | P |
|---|---|
| the away cell clears the 0.090 floor in either direction | **0.15** |
| the control is ABOVE the fold path's 0.430 | **0.65** |

### 3.4 Row 4 — ENTROPY: a DESCRIPTOR, already banked, related to row 1 only if the data allow

`H_end` **0.9447** (control) vs **0.7465** (fold path), a **0.198-nat** gap against the 75M
run-level entropy floor of **0.019** — ~10×. 🚨 **This is banked in `e77963c4` and is NOT re-derived
as a finding here**; it is reproduced from the events as an instrument check only, and it carries
**no bar**. It is related to row 1 **only if row 1 says something**: registered in advance, the only
licensed statement is a conditional one of the form *"the higher-entropy arm is the one that
does/does not pilot better off-slice"*, with no mechanism claimed and no direction predicted.
⚠️ **Two arms cannot support an entropy-to-piloting claim**, and none is made.

### 3.5 Row 5 — the PROMOTION asymmetry, a STRUCTURAL DESCRIPTOR, disclosed pre-look

🚨 **The control promoted FIVE snapshots and has a `snapshot_ladder/`; the fold path promoted ZERO
across 12,091,392 steps and has none** (§0.3). Reported as a structural finding with no bar:
a Bradley-Terry fit over five own nodes is far below the n ≥ 12 report floor, **no Elo is quoted for
either arm from it in either direction**, and the two arms' pools are not the same object by the end
(the control evicted arm W's five oldest seeds; the fold path never added to its 20). The
`win_rate_vs_pool` series of each is the descriptor beside it. Registered in advance so that neither
the presence nor the absence can later be dressed up as a strength comparison.

### 3.6 What is deliberately NOT run

* 🚨 **`main.exploitability` is NOT APPLICABLE**, for the third read running: its `--help` says
  *"Generation exploitability curve from fleet_admission-schema artifacts. Bookkeeping only — no
  battles, no models."* `ai_v13_09_wcont` has produced no such artifact.
* **No `--config auto` re-resolution.** The floor read measured three 75M arms of this exact config
  under BOTH resolutions and got levels identical to 0.01 pp on every one; `config_path` drives only
  the observation-family CHECK in `load_foreign_opponent`. NOT RUN, on budget, and declared.
* **No ladder refit** (§3.5), **no Foul Play cell, no Metamon HOME cell, no critic frame.**
* **No second seed of the control.** One arm; rules 19/22 make every verdict here a CANDIDATE.

---

## 4. THE BRANCHES, as the GO wrote them — with the uncovered middle registered IN ADVANCE

| branch | condition (operationalised) | reading |
|---|---|---|
| **(a) SENIORITY / CONTINUATION** | control untaught ≈ +5, i.e. `\|Δ_extract\|` ≤ 3.69 pp at ≥ 2 of 3 depths, with Δ_path(control) POSITIVE there | the fold's off-slice lean is CONTINUATION — **"our parents do not gain from continuation" is REFUTED for this era's 75M win-prob parent**, and the fold added nothing off-slice |
| **(b) EXTRACTION** | control untaught ≈ 0 (`\|Δ_path(control)\|` < 3.69 pp at ≥ 2 of 3) **AND** Δ_extract clears 3.69 by BOTH clauses at ≥ 1 depth | the era's first fold paid off-slice |
| **(c) TEAM-DIFFERENTIAL CONTINUATION** | the control ALSO splits DDTar up / Big-5 down at +12M, with Big-5 − DDTar ≤ −0.103 | the slice split is continuation, and **the DDTar "first cleared row" is not teaching** |
| **(d) THE SPLIT IS THE FOLD's** | the control's Big-5 − DDTar ≥ 0, or the sign pattern is absent | the split belongs to the fold |

🚨 **(a)/(b) and (c)/(d) are on DIFFERENT ROWS and are independently determined.** All four
combinations are possible and each is a legitimate outcome; the read reports one branch per pair.

🚨 **THE UNCOVERED MIDDLE IS REGISTERED NOW, because the convergence read met NO branch and had to
report that as an uncovered outcome.** Neither pair is exhaustive:

* **Row 1** — if the control lands in the middle (e.g. Δ_path(control) ≈ +2 to +3.5 pp, so it
  neither "≈ +5" by (a)'s clause nor "≈ 0" by (b)'s), **the outcome is reported as UNCOVERED**, with
  every clause mechanically checked and printed, the nearest description given in the
  registration's own vocabulary, and **no branch declared met and no branch rewritten**.
* **Row 2** — likewise if the control's signs match but its split is smaller than half the fold
  path's, or if one cell is null.
* **Tie-break, registered:** where both a branch and "uncovered" could be argued, the read reports
  **UNCOVERED** and names which clause failed. A trajectory is more informative than a forced label.

**My prior over the pairs, before any game:** (a) **0.55** · (b) **0.08** · uncovered-middle on row
1 **0.37**; (c) **0.20** · (d) **0.60** · uncovered-middle on row 2 **0.20**.

**The question this read exists to answer, in one sentence:** *is the era-1 fold's off-slice lean —
and its DDTar slice gain — teaching, or is it what arm W would have done anyway?*

---

## 5. Execution constraints (recorded so the numbers can be audited)

* **CPU ONLY** — `CUDA_VISIBLE_DEVICES=""`, `nice 15`, `OMP_NUM_THREADS=1` for the anchor peer. The
  GPU may host a new arm; **it is never touched and never read.**
* `POKESIM_SIM_BRIDGE_BIN=/home/goodlad/dev/gen3ai/src/rust_sim/target/release/sim_bridge`.
* Ports **9400–9499** only. **:8000 and :8001 are never touched**; `main.anchors` starts and stops
  its own server. Other agents own `gatecurve/` (9700–9799) and `matflip/` (9600–9699) — untouched.
  No process this job did not start is signalled; kills by explicit PID only.
* Every battle runs **from the MAIN checkout**; `--json` / `--out` under
  `/home/goodlad/.claude/jobs/9ab51de6/tmp/wcont_read/`. **Nothing is written under `models/`**, and
  no file under `src/` is changed.
* A run whose timeouts exceed 25 % of attempted battles is **INCONCLUSIVE** (rule 12). Timeouts are
  reported for every cell. The box carries another agent's CPU battery; load ran ~12 at launch.

## 6. What this read CANNOT do, whatever it returns

* It cannot make any row a **family verdict** (rules 19/22). **One control arm, one fold path.** A
  control-replicate floor bounds the CONTROL's variance, not the fold's, and the replicate that
  promotes a detection is a replicate OF THE ARM.
* It cannot separate a **team-differential continuation effect** from *"the control got better at
  everything and DDTar is where a 0.47 baseline had room"*.
* It cannot say whether the +6M offset (§1.1), the fold path's extra FORK CROSSING, or the
  **one-continuous-run vs two-forks** difference accounts for any gap it finds.
* It cannot attribute anything to **ENTROPY** (§3.4) — two arms, one axis, fully confounded with the
  distill block itself.
* It cannot say anything about the fold recipe **in general**, at another dose, with other
  teachers, or at 277M.
* It cannot read a **TREND** off three points on one arm (the standing short-window refusal, now
  7-for-7 on this campaign).
