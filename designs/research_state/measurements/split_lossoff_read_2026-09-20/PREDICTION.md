# PRE-REGISTRATION — THE SPLIT ARM, `ai_v13_11_split_lossoff`: which LEVER carried the era-1 fold's cost?

**Committed BEFORE any registered number in this directory exists.** The GO is the completion entry
of 2026-09-20 (`f389fce4`, `ai_v13_11_split_lossoff` COMPLETE at 87,097,344), whose own closing
words hand this read its question and whose headline hands it a NEW hazard: 🚨 **the era-1 fold
moved FOUR levers, not three** — `--team-block-episodes 1 → 64` is in fold-1's argv and in neither
the continuation nor arm W, and no registration before that entry had ever named it.

This is **THE THIRD ARM OF ONE SERIES**. Its two predecessors are
[`fold1_read_2026-09-19/`](../fold1_read_2026-09-19/) +
[`fold1_cont_read_2026-09-19/`](../fold1_cont_read_2026-09-19/) (the FOLD path) and
[`wcont_control_read_2026-09-20/`](../wcont_control_read_2026-09-20/) (the CONTINUATION CONTROL).
**Every instrument, opponent, manifest, manifest ORDER, seed, concurrency, cell size, floor and
decision rule below is REUSED VERBATIM from them**, so that the three paths × three depths × three
rows are ONE measurement rather than three stitched together.

🚨 **THE CONTROL's AND THE FOLD PATH's ROWS ARE IMPORTED, NOT RE-DERIVED** (§1.2), under the
arm-W reproduction warrant that those reads established and that this job re-derives. This job
measures **the SPLIT arm and arm W only**.

---

## 0. The arm, and what is already known about it

`ai_v13_11_split_lossoff` — **fold-1's exact argv with the DISTILLATION LOSS OFF.** A FORK of
`ai_v13_02_flywheel_winprob` (**arm W**) at 75,005,952, built token-exactly from
`ai_v13_07_fold1`'s own `metadata.json original_command`, `--distill-coef 0.1761 → 0.0`,
`--fork-lr 2.8e-5 --fork-lr-freeze` (realized dose **4.272e-9 = 0.20× the v8 reference**,
`[FROZEN; pinned 2.80e-05]` — **identical to the control's and the fold path's**), seed 1001, pin
`6eb9c776` in **ONE** row, config v119 / `gen3_critic_route_wave_v1`,
**+12,091,392 steps exactly to 87,097,344 — the control's and the fold path's own endpoint step, to
the step.** 3 restarts, 1 crash (the teardown-SIGTERM shape, #8, at the full target).

**Post-fork step counting is CUMULATIVE from 75,005,952**, the convention all three earlier reads
use, so all three paths span +0 → +12.09M on the same axis.

### 🚨 THE LEVER LEDGER — verified by token-set diff of the three `original_command` strings

| lever | arm W | CONTROL `ai_v13_09_wcont` | **SPLIT `ai_v13_11_split_lossoff`** | FOLD `ai_v13_07_fold1` |
|---|---|---|---|---|
| `--distill-coef` | — | **0.0** | **0.0** | **0.1761** |
| `--distill-team-bias` | — | absent | **0.4** | **0.4** |
| `--stable-opponents` (the two specialists) | — | absent | **both** | **both** |
| `--team-block-episodes` | **1** | **1** | **64** | **64** |
| `--distill-target` | — | absent (⇒ `kl`, INERT) | absent (⇒ `kl`, INERT) | `action` |

* **`split` − `fold` = ONE difference: the LOSS.** (`--distill-target action` is DROPPED rather
  than kept because `combination_checks.distill_target_needs_coef` makes `action` + coef 0 a fatal
  refusal; it re-resolves to the INHERITED `kl`, inert at coef 0, as is `--distill-topk 1`. The
  drop is FORCED, not a second lever — `f389fce4` documents this.)
* **`split` − `control` = THREE differences: the 40 % team bias, the two specialists in the stable
  pool, and the 64× team-block.** 🚨 **"The ecology" from here means bias + pool + team-block, a
  THREE-way unidentified bundle**, and `--team-block-episodes 64` is a named candidate carrier that
  nobody has tested: 64 consecutive episodes on one team against a 40 % teacher-team bias means
  long uninterrupted stretches piloting one of two teams, a plausible mechanism for an off-slice
  loss in its own right.

### Banked and NOT re-derived here (from `f389fce4`)

* 🚨 **THE MANIPULATION IS VERIFIED TWO WAYS**: the startup banner's `coef=0.0 (LOSS OFF — team
  bias only, no teacher loaded, no distill_mask) | trainee biased 40 % across all 2 teacher
  team(s)`, **and** the run's TensorBoard contains **ZERO `distill/*` scalars** — the term was
  ABSENT, not merely zero-weighted. `grep -c DISTILL` on both launcher logs is **0**.
* **Ecology present and confirmed**: matchup `[MATCHUP 0a7b730a4d]` — **fold-1's own era hash**,
  not arm W's `ef5242cffd`; both specialists resolved to `final_model.zip @83,066,880`
  `[rung=latest_txt rule=last_snapshot]`, the same files fold-1 used.
* 🚨 **ENTROPY at the identical endpoint**: split **0.9589 ± 0.0155** · control **0.9407 ± 0.0146**
  · fold **0.7373 ± 0.0185**. The split lands **WITH the control** (+0.0182 ≈ one sd) and **0.2216
  ABOVE the fold**, ≈12× the 75M run-level floor of 0.019.
* Bots at 86,000,016 **0.9513**, `win_rate_vs_pool` **0.6700**; G7 below bar on BOTH halves.
  ⚠️ **The bot row and G7 are BLIND at this depth** (the control read's hazard W-I).

### 🚨 Seven facts LOOKED AT while scoping this job, and disclosed here rather than hidden

Pre-registration is worthless if the registrant has already seen the answer. Seven facts about the
split arm's **tree, its argv and its sidecars** were read while scoping. All seven are declared:

1. **The split's checkpoint grid**, listed in §1.1, including that its `checkpoint_78006048_steps.zip`
   sits at **exactly the same step number** as BOTH earlier paths' +3M point and its
   `final_model.zip` at **exactly the same step** as both endpoints. Only +6M differs (§1.1's ⚠️).
2. **`metadata.json`**: `lineage.role` = `fold`, `fork_parent` = arm W `final_model.zip`
   @75,005,952 resolved `[rung=explicit_zip rule=explicit_zip]`, **`teachers` = the two
   specialists** (both `latest_txt`/`last_snapshot` → `final_model.zip` @83,066,880),
   `exploiter_target: null`, `pin_history` **ONE row** `6eb9c776` 75,005,952 → 87,097,344,
   `pool_seeded_from` arm W's `snapshots/`, 20 files, `latest.txt` = `final_model.zip`.
3. 🚨 **THE SPLIT PROMOTED SIX SNAPSHOTS POST-FORK AND HAS A `snapshot_ladder/`** — 76.0M, 78.0M,
   80.0M, 82.0M, 84.0M, 86.0M — **like the CONTROL (five) and unlike the FOLD PATH (zero)**.
   ⇒ **A STRUCTURAL FINDING, registered here as a DESCRIPTOR with NO BAR** (§3.4). No Elo is
   quoted from it in any direction (n = 6 ≪ the 12-node report floor), and the three arms' pools
   are not the same object by the end.
4. **`metadata.json` / `model_config.json`** carry `distill_target: "kl"`, `distill_topk: 1`,
   `distill_beta: 1.0`, `distill_gate: "none"` — ⚠️ **INHERITED from arm W's checkpoint config and
   INERT at `--distill-coef 0.0`**; the same standing instance the control read declared as its
   hazard W-C. Declared here in advance as hazard **S-C**.
5. **The three-way argv token diff above**, computed from the three `original_command` strings.
6. The ledger's own banked descriptors (entropy / bots / pool / G7 / the two verifications),
   quoted above, **not re-derived as findings**.
7. The 2026-09-20 anchor read (`068534aa`) — 8,400 `SmallRL` battles put the CONTROL at
   **+0.039 [−0.000, +0.079]** over arm W away and **+0.020 [−0.019, +0.058]** home, NOT DETECTED
   on either, while the fold path reads ~0.08 BELOW its parent on both team sets: **the untaught
   meter RANKS but does not SCALE.** ⚠️ Registered here as the standing interpretation limit on
   every pp in this file (hazard **S-J**). **Anchors are NOT part of this read** (§3.6).

**Nothing about the split arm's piloting COMPETENCE has been measured.** The untaught meter and the
per-slice piloting row have **no numbers of any kind** at the time of this commit. No
`win_rate_vs_pool` series, no `ladder.json`, no TensorBoard scalar of this run has been opened
beyond the banked entropy/bots figures quoted above.

---

## 1. The comparators, by name and by RESOLVED file

| role | run | resolved file | step | cumulative post-fork |
|---|---|---|---:|---|
| **SPLIT, +3M** | `ai_v13_11_split_lossoff` | `checkpoints/checkpoint_78006048_steps.zip` | 78,006,048 | **+3.0001M** |
| **SPLIT, +6M** | `ai_v13_11_split_lossoff` | `checkpoints/checkpoint_81404208_steps.zip` | 81,404,208 | **+6.3983M** |
| **SPLIT, +12M** | `ai_v13_11_split_lossoff` | `final_model.zip` | 87,097,344 | **+12.0914M** |
| **PARENT / BASELINE — arm W** | `ai_v13_02_flywheel_winprob` | `final_model.zip` | 75,005,952 | +0 (RE-MEASURED — the warrant) |
| CONTROL, +3M | `ai_v13_09_wcont` | `checkpoints/checkpoint_78006048_steps.zip` | 78,006,048 | +3.0001M (IMPORTED) |
| CONTROL, +6M | `ai_v13_09_wcont` | `checkpoints/checkpoint_81143280_steps.zip` | 81,143,280 | +6.1373M (IMPORTED) |
| CONTROL, +12M | `ai_v13_09_wcont` | `final_model.zip` | 87,097,344 | +12.0914M (IMPORTED) |
| fold path, +3M | `ai_v13_07_fold1` | `checkpoints/checkpoint_78006048_steps.zip` | 78,006,048 | +3.0001M (IMPORTED) |
| fold path, +6M | `ai_v13_07_fold1` | `final_model.zip` | 81,100,800 | +6.0948M (IMPORTED) |
| fold path, +12M | `ai_v13_08_fold1_cont` | `final_model.zip` | 87,097,344 | +12.0914M (IMPORTED) |
| the SEED FLOOR arm — W_b | `ai_v13_04_flywheel_winprob_b` | `final_model.zip` | 75,005,952 | +0 (IMPORTED) |
| teacher t1 (Big-5, balance) | `ai_v13_05_exploit_big5starmie` | `final_model.zip` | 83,066,880 | (IMPORTED, slice only) |
| teacher t2 (DDTar, offense) | `ai_v13_06_exploit_ddtar_spikes` | `final_model.zip` | 83,066,880 | (IMPORTED, slice only) |

### 1.1 ⚠️ THE DEPTH MATCH IS EXACT AT TWO OF THREE POINTS AND OFF AT THE THIRD — the SPLIT is the DEEPEST there

Checkpoints land every 500 k and the three runs' restart boundaries differ, so the grids are not
identical. **The convention is §1.1 of the control read's registration, applied unchanged: take the
checkpoint at the same step number if it exists, else the NEAREST, and record the gap.** Named in
advance so the choice cannot be fitted to a result:

| grid point | SPLIT's file | CONTROL's | FOLD's | offset (split − control) | offset (split − fold) |
|---|---|---|---|---|---|
| **+3M** | `78,006,048` | `78,006,048` | `78,006,048` | **0 — the same step number** | **0 — the same step number** |
| **+6M** | `81,404,208` | `81,143,280` | `81,100,800` | **+260,928 (split DEEPER)** | **+303,408 (split DEEPER)** |
| **+12M** | `87,097,344` (final) | `87,097,344` (final) | `87,097,344` (final) | **0 — the same step number** | **0 — the same step number** |

The split's two nearest candidates to the control's +6M are `81,404,208` (+260,928) and
`80,506,128` (−637,152); **81,404,208 is the nearer to the control's AND to the fold's +6M**, so
the choice is unambiguous under the convention. The offset is **2.2 % of the post-fork budget** and
the split is the DEEPER of the three, i.e. if seniority drives the untaught row the offset flatters
the SPLIT on that row. Recorded as hazard **S-D**, not corrected. 🚨 **No verdict below rests on
the +6M row**: the registered branch clauses are evaluated at **+12M**, where all three paths sit
on the identical step, with **+3M** (also identical) as the registered robustness check.

### 1.2 🚨 THE DECLARED IMPORTS, AND THE WARRANT FOR THEM — registered before the first battle

**The whole control path, the whole fold path, W_b and both teachers are NOT re-run.** Their cells
are taken verbatim from the three banked reads:

| imported ref | untaught source | slice source |
|---|---|---|
| `wcont_p3M`, `wcont_p6M`, `wcont_p12M` | `wcont_control_read_2026-09-20/out/untaught/untaught_registry.json` | `wcont_control_read_2026-09-20/out/wcont_slice_read.json` |
| `fold_p3M`, `fold_p6M`, `cont_p12M`, `armWb` | `fold1_cont_read_2026-09-19/out/untaught/untaught_registry.json` | `wcont_control_read_2026-09-20/out/wcont_slice_read.json` |
| `t1_big5`, `t2_ddtar` | — | `wcont_control_read_2026-09-20/out/wcont_slice_read.json` |

🚨 **THE WARRANT IS THE ARM-W REPRODUCTION CHECK, AND IT IS REGISTERED AS PASS/FAIL HERE.** `armW`
IS re-run, **in BOTH invocations — this is the ONE re-derived verification cell the GO names**, and
it must return:

* **untaught: 46.19 pp = 739 / 1600**, and **all eight PER-TEAM rows identical** to the banked ones;
* **slice: 394 / 800 on Big-5 and 376 / 800 on DDTar.**

The meter is deterministic at `--seed 0` / concurrency 1 — demonstrated on 2026-09-16, 2026-09-18,
in the fold's +6M read, five more times inside the convergence read and a sixth time in the control
read, across four DIFFERENT ref lists — and a cell is a pure function of (ref, team index, battle
index), which is the property that licenses both `--workers` and a changed ref list.
🚨 **If arm W returns anything else, every imported cell is VOID, the affected row is reported as
INVALID rather than patched, and the job STOPS and reports.** Registering the import here is the
alternative to spending ~19,200 battles re-deriving rows that are the same games.

### 1.3 The floors, and where each one comes from

| row | floor | provenance |
|---|---:|---|
| untaught meter | **3.69 pp** | `\|arm W − W_b\|`, IMPORTED from `flywheel_wb_floor_read_2026-09-18/`, a **75M FRESH-ARM seed pair**; re-measured on the identical games inside this job's own invocation |
| per-slice, Big-5 cell | **0.0475** | `\|arm W − W_b\|` **MEASURED on this very cell**, same opponent, same 800 games; arm W's half re-measured here |
| per-slice, DDTar cell | **0.0850** | as above |

🚨 **ONE PAIR BOUNDS A FLOOR; IT DOES NOT ESTIMATE ONE** (rules 19/22). No CI attaches to a floor,
and WITHIN FLOOR is never "equivalent" (rule 6). 🚨 **A floor is a property of the DEPTH and the
REGIME** (rule 3): 3.69 pp is a fresh-arm import onto three CONTINUATIONS, labelled as an IMPORT in
every row it bars. The meter's own fold/continuation floors for reference: 1.19 (END-depth
replicate), 1.66 (frozen-dose), 2.46 (K = 6), 4.27 (controller-live) and **1.00 pp (the gen-era G5
three-continuation-arm cell)**.

---

## 2. THE DECISION RULE — fixed here, copied VERBATIM from all three predecessors

> A contrast is **OUTSIDE THE FLOOR** iff **(a)** `|Δ| > floor` **AND** **(b)** the Δ's own 95 % CI
> **excludes the floor POINT**. Otherwise **WITHIN FLOOR at n = 2** — never "equivalent".

**Three sign conventions, all used, all labelled at every use:**

* `Δ_path = <path> − arm W` — POSITIVE means ahead of the frozen parent. The convention of all
  three earlier reads and of every trajectory row.
* 🚨 `Δ_ecology = split − control` — **THREE levers** (bias + pool + team-block). POSITIVE means the
  ecology bundle bought something the plain continuation did not.
* 🚨 `Δ_loss = split − fold` — **ONE lever, the distillation LOSS.** POSITIVE means turning the loss
  OFF bought something, i.e. the loss COST that much.

🚨 **CLAUSE (b) IS KNOWN IN ADVANCE TO BE NEAR-UNSATISFIABLE ON ROW 1 AT SMALL EFFECT SIZES, AND
THE RULE STILL STANDS.** The three earlier reads measured the untaught contrast's team-clustered CI
half-width at ≈3.8–6.2 pp against a 3.69 pp floor — so clause (b) can pass only at `|Δ| ≳ 7.5 pp`.
The control read's +12–15 pp effects passed it without difficulty; a 5 pp effect cannot. This is
registered here, before any number, as a property of the bar and not as an escape from it. A CI
clear of ZERO is reported separately and **is not a verdict**. The honest lever is a second ARM
(rules 19/22), never more games.

🚨 **AND THE POOLING TRAP IS PRE-DECLARED AGAIN.** At the fold's +6M, pooled over both taught teams
the fold read **+0.003** over its parent while the per-team rows were **−0.061** and **+0.068**,
each CI clear of zero. **The per-team rows are the result at every depth here; any pooled number is
a labelled footnote and no verdict is taken from one** (rule 10).

---

## 3. The registered rows and their bars

### 3.1 Row 1 — THE UNTAUGHT METER (off-slice; this IS the collateral read)

**Instrument, REUSED VERBATIM.** `python -m main.untaught_meter`, registry opponent
`untaught_meter_opponent` (= `ai_v9_29_rev1_0823@24,000,000`), the untaught-8 manifest in its
canonical order, **200 games/team, `--seed 0`, concurrency 1**, sharded over teams with
`--workers 8`, **all four measured refs (`split_p3M`, `split_p6M`, `split_p12M`, `armW`) in ONE
invocation**, 6,400 battles. The control path, the fold path and W_b are IMPORTED under §1.2.

**Bars, all three.** At each of the three depths, on the meter's own per-team rows over ONE shared
bootstrap index set (rule 10, team the unit, 20,000 draws, seed 20260915 — verbatim from
`floor_untaught_delta.py`):

1. `|Δ_path(split − armW)| > 3.69 pp` AND CI excludes the floor point.
2. 🚨 `|Δ_ecology(split − control)| > 3.69 pp` AND CI excludes the floor point.
3. 🚨 `|Δ_loss(split − fold)| > 3.69 pp` AND CI excludes the floor point.

### 3.2 Row 2 — PER-SLICE PILOTING, the two TAUGHT teams

**Instrument, REUSED VERBATIM.** The same `main.untaught_meter` engine pointed at the **2-team
taught-slice manifest in TEACHER ORDER** (t1 Big-5 first, t2 DDTar second — **the order is the seed
offset; re-ordering it changes every game**), against **the untaught meter's own fixed opponent**
(`untaught_meter_opponent`) and **not** any pool sentinel (a sentinel is the trainee's OWN
snapshot, floor-read hazard F-G). **800 games per (ref, team), `--seed 0`**, four ref-shards × 2
workers. Measured refs: `split_p3M`, `split_p6M`, `split_p12M` and **`armW` — the reproduction
check**. 6,400 battles. The manifest file is **byte-identical** to the three earlier reads'.

⚠️ 🚨 **UNLIKE THE CONTROL, THE SPLIT ARM DID TRAIN ON THESE TWO TEAMS WITH A 40 % BIAS, IN
64-EPISODE BLOCKS, AGAINST TWO OPPONENTS THAT PILOT THEM** — with **no teacher loss**. That is
precisely the cell this read exists to produce: it separates *"being fed the teacher's teams and
opponents"* from *"being fed the teacher's ACTIONS."*

**Bar.** Per team, per depth: `Δ_path`, `Δ_ecology` and `Δ_loss` against that cell's own
**measured** seed floor (0.0475 Big-5, 0.0850 DDTar), clause (a) then clause (b) on the Newcombe
interval. ⚠️ **Newcombe is CONSERVATIVE here** — the games are paired under CRN and the tool
retains no per-battle outcome vector, so the pairing cannot be exploited. A cluster bootstrap over
TWO teams is not a CI and none is printed.

**Also registered as a DESCRIPTOR with no bar:** the `Big-5 minus DDTar` difference-of-deltas at
+12M for the split, against the control's **−0.0650** and the fold path's **−0.2063**.

### 3.3 Row 3 — ENTROPY: a DESCRIPTOR, already banked, carrying NO bar

`H_end` split **0.9589** · control **0.9407** · fold **0.7373**, at the identical endpoint step,
against a 75M run-level entropy floor of **0.019**. 🚨 **Banked in `f389fce4` and NOT re-derived as
a finding here.** It is the **stated prior** for this read's branches (§4) and it is **not this
read's endpoint**: entropy is a property of the policy's action distribution, piloting competence
is not, and the two have never been shown to move together. ⚠️ **Three arms, one axis, entropy
perfectly confounded with the distill block itself.** The only licensed statement is a conditional
one of the form *"the arm that lands with the control on entropy does / does not land with it on
piloting"*, with no mechanism claimed.

### 3.4 Row 4 — the PROMOTION record, a STRUCTURAL DESCRIPTOR, disclosed pre-look

🚨 **The split promoted SIX snapshots post-fork and has a `snapshot_ladder/`; the control promoted
FIVE and has one; the fold path promoted ZERO and has none.** Reported with **no bar**: a
Bradley-Terry fit over six own nodes is far below the n ≥ 12 report floor, **no Elo is quoted for
any arm in either direction**, and the three arms' pools are not the same object by the end.
The `win_rate_vs_pool` series of each is the descriptor beside it. ⚠️ **The three series are not
comparable in kind** — the fold path's pool was FROZEN (zero promotions), the control's and the
split's GREW, so their later cycles face a strengthening opponent set (hazard **S-G**).
Registered in advance so neither the presence nor the absence can later be dressed up as a strength
comparison.

### 3.5 The ONE re-derived verification cell, registered explicitly

**`armW`, re-run in BOTH invocations** — 1,600 untaught battles and 1,600 slice battles. Expected:
untaught **739 / 1600 = 46.19 pp with all eight per-team rows identical**; slice **394 / 800** and
**376 / 800**. 🚨 **If it does not match EXACTLY, the job STOPS and reports rather than patching.**

### 3.6 What is deliberately NOT run

* 🚨 **NO ANCHOR CELL.** The GO excludes `SmallRL` from this read. The standing external reading is
  `068534aa`'s 8,400-battle result (§0.7) and nothing here adds to it.
* 🚨 **`main.exploitability` is NOT APPLICABLE**, for the fourth read running — it is bookkeeping
  over `fleet_admission`-schema artifacts and this arm has produced none.
* **No `--config auto` re-resolution.** The floor read measured three 75M arms of this exact config
  under BOTH resolutions and got levels identical to 0.01 pp; `config_path` drives only the
  observation-family CHECK in `load_foreign_opponent`. NOT RUN, on budget, and declared.
* **No ladder refit** (§3.4), **no Foul Play cell, no critic frame, no TB re-derivation beyond the
  banked descriptors.**
* **No second seed of the split arm.** One arm; rules 19/22 make every verdict here a CANDIDATE.
* 🚨 **NOTHING is written under `models/`; no training, no launcher, no :8000/:8001.** The GPU is
  carrying `ai_v13_12_plateau` block 1 and has first call on the box.

---

## 4. THE BRANCHES — operationalised, with the uncovered case registered IN ADVANCE

**THE THREE ROWS** are: the untaught 8 (floor 3.69 pp), the Big-5 slice cell (floor 0.0475) and the
DDTar slice cell (floor 0.0850).

🚨 **THE BRANCH CLAUSES ARE EVALUATED AT +12M**, the depth at which all three paths sit on the
IDENTICAL step (87,097,344). **+3M (also identical, 78,006,048) is the registered ROBUSTNESS
CHECK** and is reported beside it; **+6M carries the 260,928-step offset (§1.1) and carries no
branch clause.** If +12M and +3M disagree on the branch, **both are printed and the outcome is
reported as SPLIT BY DEPTH**, with no branch declared met.

| branch | condition at +12M (operationalised) | reading |
|---|---|---|
| **(a) THE LOSS IS THE CARRIER** | `\|Δ_ecology\|` WITHIN FLOOR on **all three** rows **AND** `\|Δ_loss\|` OUTSIDE THE FLOOR on **≥ 2** rows | the split tracks the CONTROL ⇒ the distillation LOSS carried the era-1 fold's cost; **the ecology bundle is exonerated at this dose** |
| **(b) THE ECOLOGY IS THE CARRIER** | `\|Δ_loss\|` WITHIN FLOOR on **all three** rows **AND** `\|Δ_ecology\|` OUTSIDE THE FLOOR on **≥ 2** rows | the split tracks the FOLD ⇒ the **bias + pool + team-block-64** bundle carried it; **the loss is exonerated** |
| **(c) BOTH CARRY** | `\|Δ_ecology\|` OUTSIDE on ≥ 2 rows **AND** `\|Δ_loss\|` OUTSIDE on ≥ 2 rows, **with the split BETWEEN** (Δ_ecology NEGATIVE, i.e. split below the control, on ≥ 2 of those rows) | both levers carry; the two sizes are stated per row |
| **(d) THE ECOLOGY HELPED** | `Δ_ecology` **POSITIVE** and OUTSIDE THE FLOOR on ≥ 2 rows | the split is ABOVE the control ⇒ the ecology helped and the loss cost MORE than the whole control-to-fold gap |
| **(e) UNCOVERED** | anything else | reported as UNCOVERED, every clause mechanically printed, the nearest description given in this registration's own vocabulary, **and no branch rewritten** |

🚨 **TIE-BREAK, REGISTERED:** where both (c) and (d) could be argued, the **SIGN of Δ_ecology
decides** — positive ⇒ (d), negative ⇒ (c). Where a branch and (e) could both be argued, the read
reports **(e) UNCOVERED** and names which clause failed. **A trajectory is more informative than a
forced label.** The control read met NO branch on its row 1 and had to report that; this
registration's (e) exists so the same outcome has a home.

### 4.1 🚨 THE PRIOR, STATED FROM THE ENTROPY DESCRIPTOR — and why it is only a prior

`f389fce4` banked, at the identical endpoint, `H_end` **split 0.9589 ± 0.0155 · control 0.9407 ±
0.0146 · fold 0.7373 ± 0.0185`. **On that channel the split lands WITH the control (≈ one sd) and
0.2216 ABOVE the fold (≈12× the run-level floor), which predicts branch (a): the LOSS is the
carrier and the ecology is not.**

🚨 **AND ENTROPY IS A DESCRIPTOR, NOT THIS READ's ENDPOINT.** A top-1 action-target distillation
concentrates the policy **by construction**, so the loss moving entropy is nearly a tautology and
says nothing about whether the loss is what moved PILOTING. The ecology bundle has its own
untested mechanism for an off-slice loss that would leave entropy untouched: **64-episode blocks
against a 40 % two-team bias is a large reduction in the diversity of teams the trainee pilots**,
and an off-slice meter is exactly the instrument that would see it. **The two channels can
disagree, and if they do, this row is the one that counts.**

| | P |
|---|---|
| **branch (a)** — the LOSS is the carrier | **0.35** |
| **branch (b)** — the ECOLOGY is the carrier | **0.20** |
| **branch (c)** — BOTH carry (the split between) | **0.30** |
| **branch (d)** — the ecology HELPED (split above the control) | **0.05** |
| **branch (e)** — UNCOVERED | **0.10** |
| `Δ_path(split − armW)` is POSITIVE on the untaught row at +12M | **0.85** |
| `Δ_loss(split − fold)` is POSITIVE on the untaught row at +12M | **0.80** |
| `Δ_ecology(split − control)` is NEGATIVE on the untaught row at +12M | **0.70** |
| the split's untaught level at +12M is within ±3.69 pp of the CONTROL's 61.69 | **0.40** |
| the split's untaught level at +12M is within ±3.69 pp of the FOLD's 51.94 | **0.20** |
| 🚨 the split CLEARS the DDTar cell against arm W by BOTH clauses at +12M | **0.55** |
| 🚨 the split's Big-5 delta at +12M is POSITIVE (the fold's was −0.0525, the control's +0.1125) | **0.60** |
| the +12M and +3M branches AGREE | **0.65** |

**The question this read exists to answer, in one sentence:** *of the FOUR levers the era-1 fold
moved off plain continuation, did the DISTILLATION LOSS carry its cost, or did the ECOLOGY BUNDLE —
the 40 % team bias, the two specialists in the pool, and the 64-episode team blocks?*

---

## 5. Execution constraints (recorded so the numbers can be audited)

* 🚨 **CPU ONLY** — `CUDA_VISIBLE_DEVICES=""`, `nice 15`, `OMP_NUM_THREADS=1`. **`ai_v13_12_plateau`
  block 1 is LIVE on the GPU and has first call on the box.** The GPU is never touched and never
  read. **At most 8 worker processes at a time**, and the two invocations run SEQUENTIALLY rather
  than side by side for that reason. **The load average is recorded with every cell.**
* `POKESIM_SIM_BRIDGE_BIN=/home/goodlad/dev/gen3ai/src/rust_sim/target/release/sim_bridge`.
* Every battle runs **from the MAIN checkout**; `--json` under
  `/home/goodlad/.claude/jobs/split_read_20260920/tmp/`. **Nothing is written under `models/`**, and
  no file under `src/` is changed. **No server is started at all** (the meter is serverless;
  no anchor cell is run). **:8000 and :8001 are never touched.** Kills by explicit PID only.
* A run whose timeouts exceed 25 % of attempted battles is **INCONCLUSIVE** (rule 12). Timeouts are
  reported for every cell.

## 6. What this read CANNOT do, whatever it returns

* It cannot make any row a **family verdict** (rules 19/22). **One split arm, one control arm, one
  fold path.** Every verdict is a CANDIDATE.
* 🚨 **It cannot separate the THREE levers inside the ecology bundle** — the 40 % bias, the two
  specialists in the pool, and `--team-block-episodes 64`. A (b) or (c) outcome names the BUNDLE
  and nothing finer. **`--team-block-episodes` has never been registered in any read.**
* It cannot separate the **FORK-BOUNDARY asymmetry**: the split and the control are each ONE
  continuous run; the fold path is two runs with a FORK between. Any `Δ_loss` row carries that
  difference too. (The control read BOUNDED this: at +3M no path has crossed a fork.)
* It cannot attribute anything to **ENTROPY** (§3.3).
* It cannot read a **TREND** off three points on one arm (the standing short-window refusal).
* 🚨 **It cannot convert any pp into ladder STRENGTH.** `068534aa` measured the untaught meter's
  +15.50 pp control effect at **+0.039 externally, NOT DETECTED** — the meter RANKS but does not
  SCALE, and no rating is quoted for any arm here.
* It cannot say anything about the fold recipe **in general**, at another dose, with other
  teachers, or at 277M.
