# PRE-REGISTRATION — THE CONVERGED-ENDPOINT READ OF THE ERA-1 FOLD (`ai_v13_08_fold1_cont`)

**Committed BEFORE any registered number in this directory exists.** The GO is the completion entry
of 2026-09-19 (`ai_v13_08_fold1_cont` COMPLETE at 87,097,344, ledger `L19618`), whose own closing
words hand this read its question: *"whether the remaining 25 % changes any row is the
continuation's own read."*

Its predecessor is [`fold1_read_2026-09-19/`](../fold1_read_2026-09-19/) — the +6M registered read
of `ai_v13_07_fold1`, branch **(d) NOT DETECTED**. **Every instrument, opponent, manifest, seed,
concurrency, cell size and decision rule below is REUSED VERBATIM from it**, so that the two reads
are one series and the trajectory table is a single measurement rather than two stitched together.

---

## 0. The arm, and what the fold's own +6M read already established

`ai_v13_08_fold1_cont` — a **FORK of `ai_v13_07_fold1`** at 81,100,800 carrying the **identical
distill block** (argv token-set diff exactly `--model`, `--steps`, run name), pin `6eb9c776`,
config v119 / `gen3_critic_route_wave_v1`, `--fork-lr 2.8e-5 --fork-lr-freeze` (realized dose
**4.272e-9 = 0.20× the v8 reference**, `[FROZEN; pinned 2.80e-05]`), `--distill-coef 0.1761`,
`--distill-team-bias 0.4`, `--checkpoint-every-steps 500000`, both teachers in the stable pool each
piloting its own pinned team. It ran to **87,097,344**.

**Post-fork step counting is CUMULATIVE from the ORIGINAL fold point, 75,005,952** — the ledger's
own convention for this path, and the one the matchup hash `0a7b730a4d` keeps unchanged across the
fork. So the continuation spans **+6.10M → +12.09M** and the whole fold path spans +0 → +12.09M.

### The two taught teams (THE SLICE) — unchanged from the +6M read

| teacher | run | its pinned team | archetype (human description) |
|---|---|---|---|
| **t1 — Big-5** | `ai_v13_05_exploit_big5starmie` | `data/teams/sample/f6229d2c867e21d6.txt` (pin `450fb83c20`) | **balance** — climbed 0.640 → 0.740 vs arm W, ~45 turns/game |
| **t2 — DDTar** | `ai_v13_06_exploit_ddtar_spikes` | `data/teams/sample/9eb3abdc52876a63.txt` (pin `6212de2e8c`) | **offense** — opened 0.700, closed 0.740, ~32 turns/game |

### Banked and NOT re-derived here

From the completion entry (`L19618`): 🚨 **the distill STOP RULE FIRED at step 82,673,664 = +7.667M
post-fork** — *"`teacher_agreement_on_slice` EMA 0.8178 has PLATEAUED (improvement over 8 rollouts
< 0.005) while `collateral_kl_vs_parent` EMA 0.38416 is still RISING"* — held for 42 of 57
post-resume points, and `--distill-stop warn` is log-only so nothing annealed and nothing stopped.
**Teacher agreement saturates at ≈0.82–0.83 EMA (0.8233 / 0.8209 / 0.8329 at +7.5M / +9M / +12M).**
The teacher divergence did NOT continue (t1/t2 gated 0.2635 / 0.2135 at the end, ~1.2× apart
against 2.5× at the fold's end — the campaign's fifth vindicated short-window refusal).

From the +6M read (`fold1_read_2026-09-19/`), the rows this one extends:

| row | +1M | +3M | +6M |
|---|---:|---:|---:|
| untaught meter, Δ vs arm W (pp) | **+1.44** [−0.44, +3.19] | **+5.38** [+1.63, +9.19] | **+5.13** [+1.06, +9.00] |
| per-slice Big-5, Δ vs arm W | — | −0.0250 | 🚨 **−0.0612** |
| per-slice DDTar, Δ vs arm W | — | **+0.1238** | **+0.0675** |
| ladder nodes | 0 | 0 | **0** |

Verdict at +6M: **branch (d), NOT DETECTED on every registered row**, with the gen-era fold's shape
INVERTED — no early off-slice hole, off-slice the best row, on-slice split in SIGN.

### 🚨 Facts LOOKED AT while scoping this job, and disclosed here rather than hidden

Pre-registration is worthless if the registrant has already seen the answer. Four facts about the
continuation's **tree and its own logged meters** were read while scoping, and all four are declared:

1. **`ai_v13_08_fold1_cont` has NO `snapshot_ladder/` and promoted ZERO snapshots.** Its
   `snapshots/` holds exactly the 20 files seeded from `ai_v13_07_fold1` (26.0M → 72.0M, all with
   arm W's original 2026-09-15/16 mtimes), plus `pool_seed.json` naming the fold as the parent
   pool. ⇒ **Registered row 4 (ladder) is again NOT RUNNABLE at n = 0** — the same structural
   finding as the +6M read, now at double the budget.
2. The run's own `latest_eval.pool` at 84,000,000 reads `win_rate` **0.40**, `snapshot_count`
   **20** — i.e. the frozen pool never grew and the promotion gate (0.55) was never approached.
3. The distill-meter table above (agreement, gated fractions, `stop_signal`) is **quoted from the
   ledger, not re-derived as a finding**; this read reproduces it from the events as an instrument
   check only.
4. The interrupted-checkpoint sidecar at 84,443,136 records `distill_stop_state.detector.fired:
   true, rollouts_since_fire: 17`, confirming the fire independently of the launcher log.

**Nothing about the continuation's COMPETENCE has been measured.** The untaught meter, the
per-slice piloting row and the anchor cell have **no numbers of any kind** at the time of this
commit.

---

## 1. The comparators, by name and by RESOLVED file

🚨 **The +7.5M / +9M / +12M grid is CUMULATIVE post-fork, and the checkpoints nearest each grid
point are named here in advance** so the choice cannot be fitted to a result:

| role | run | resolved file | cumulative post-fork |
|---|---|---|---|
| **the fold, +1M** | `ai_v13_07_fold1` | `checkpoints/checkpoint_76005984_steps.zip` | +1.000M |
| **the fold, +3M** | `ai_v13_07_fold1` | `checkpoints/checkpoint_78006048_steps.zip` | +3.000M |
| **the fold, +6M** | `ai_v13_07_fold1` | `final_model.zip` @ 81,100,800 | +6.095M |
| **the cont, "+7.5M"** | `ai_v13_08_fold1_cont` | `checkpoints/checkpoint_82600848_steps.zip` | **+7.595M** |
| **the cont, "+9M"** | `ai_v13_08_fold1_cont` | `checkpoints/checkpoint_84100896_steps.zip` | **+9.095M** |
| **the cont, "+12M"** | `ai_v13_08_fold1_cont` | `final_model.zip` @ 87,097,344 | **+12.091M** |
| **PARENT / CONTROL — arm W** | `ai_v13_02_flywheel_winprob` | `final_model.zip` @ 75,005,952 | +0 |
| **the SEED FLOOR arm — W_b** | `ai_v13_04_flywheel_winprob_b` | `final_model.zip` @ 75,005,952 | +0 |
| teacher t1 (Big-5, balance) | `ai_v13_05_exploit_big5starmie` | `final_model.zip` @ 83,066,880 | (+8.06M off its own fork) |
| teacher t2 (DDTar, offense) | `ai_v13_06_exploit_ddtar_spikes` | `final_model.zip` @ 83,066,880 | (+8.06M) |

⚠️ **THE "+7.5M" POINT IS THE STOP POINT, AND IT IS 72,816 STEPS EARLY.** Checkpoints land every
500k; the stop rule fired at 82,673,664. `checkpoint_82600848` (+7.595M) is the LAST checkpoint
before the fire and the nearest to both the +7.5M grid point and the fire itself (the next one,
+8.095M, is 427,200 steps after it). **It is therefore a lower bound on "the recipe's own
endpoint" by ≈0.07M steps, and is labelled +7.59M everywhere below, never "+7.5M" bare.**

⚠️ **THE +6M → +7.5M LEG CROSSES A FORK BOUNDARY.** `ai_v13_07_fold1` ends at 81,100,800 and
`ai_v13_08_fold1_cont` starts there. The fork re-seeds the optimizer path, re-pins the LR (to the
same frozen 2.8e-5), re-seeds the pool (from the fold, i.e. the same 20 arm-W files) and starts a
new event file. The matchup hash is unchanged, so this is not a regime boundary in the rule-15
sense — but it is a discontinuity in the run, and every trajectory row below prints it.

### 1.1 The floors, and where each one comes from

| row | floor | provenance |
|---|---:|---|
| untaught meter | **3.69 pp** | `\|arm W − W_b\|`, IMPORTED from `flywheel_wb_floor_read_2026-09-18/`, a **75M FRESH-ARM seed pair**; re-measured on the identical games in this invocation |
| per-slice, Big-5 cell | **0.0475** | `\|arm W − W_b\|` **MEASURED on this very cell**, same opponent, same 800 games (`fold1_read_2026-09-19/`) |
| per-slice, DDTar cell | **0.0850** | as above |
| `SmallRL` greedy away | **0.090** | `\|arm W − W_b\|`, IMPORTED; itself inside the anchors SOP's three-seed 0.110 |
| ladder | 25.2 Elo | IMPORTED — **not runnable, n = 0** |

🚨 **ONE PAIR BOUNDS A FLOOR; IT DOES NOT ESTIMATE ONE** (rules 19/22). No CI attaches to a floor,
and WITHIN FLOOR is never "equivalent" (rule 6). 🚨 **A floor is a property of the DEPTH and the
REGIME** (§3.3): every 3.69/0.090 bar here is a fresh-arm import onto a FOLD, labelled as one in
every row it bars. The meter's own fold floors are 1.19 (END-depth replicate), 1.66 (frozen-dose),
2.46 (K=6), 4.27 (controller-live) and 1.00 pp (G5 continuation).

---

## 2. THE DECISION RULE — fixed here, copied verbatim from the +6M read

> A contrast is **OUTSIDE THE FLOOR** iff **(a)** `|Δ| > floor` **AND** **(b)** the Δ's own 95 % CI
> **excludes the floor POINT**. Otherwise **WITHIN FLOOR at n = 2** — never "equivalent".

**Sign convention throughout: `Δ = fold path − arm W`.** POSITIVE means ahead of the parent.

🚨 **CLAUSE (b) IS KNOWN IN ADVANCE TO BE NEAR-UNSATISFIABLE ON ROW 1, AND THE RULE STILL STANDS.**
The +6M read measured the untaught contrast's team-clustered CI half-width at **≈3.8–4.0 pp**
against a 3.69 pp floor, so clause (b) can pass only at **|Δ| ≳ 7.5 pp** (its hazard D-A). This is
registered here, before any number, as a property of the bar and not as an escape from it: the
verdict word will be WITHIN FLOOR at any |Δ| below ~7.5 pp however clear of zero it is, and the
honest lever is a second fold ARM (rules 19/22), never more games. **The same arithmetic on the
per-slice row:** Newcombe half-widths ran ≈0.049 at 800 games against cell floors 0.0475 and
0.0850, so clause (b) needs roughly |Δ| ≳ 0.097 (Big-5) and ≳ 0.134 (DDTar).

🚨 **AND THE POOLING TRAP IS PRE-DECLARED.** At +6M, pooled over both taught teams the fold read
**+0.003** over its parent while the two per-team rows were **−0.061 and +0.068**, each CI clear of
zero — an exact null manufactured out of a real split (rule 10; hazard D-G). **The per-team rows
are the result at every depth here; any pooled number is printed as a labelled footnote and no
verdict is taken from one.**

---

## 3. The registered rows, their bars, and the prediction attached to each

### 3.1 Row 1 — THE UNTAUGHT METER (off-slice; this IS the collateral read)

**Instrument, REUSED VERBATIM.** `python -m main.untaught_meter`, registry opponent
`untaught_meter_opponent` (= `ai_v9_29_rev1_0823@24,000,000`), the untaught-8 manifest in its
canonical order, **200 games/team, `--seed 0`, concurrency 1**, sharded over teams.

🚨 **ALL EIGHT REFS IN ONE INVOCATION** — the fold's +1M/+3M/+6M, the continuation's
+7.59M/+9.09M/+12.09M, arm W and W_b — so every ref sees the identical 8 teams and the identical
200 games each (CRN), **the whole six-point trajectory is paired on ONE index set**, and the
bootstrap (rule 10, team the unit, 20,000 draws, one shared index set, seed 20260915 — verbatim
from `floor_untaught_delta.py`) is legitimate across depths as well as against the parent. 12,800
battles.

**FIVE reproduction checks ride in it and are registered as PASS/FAIL in advance:** arm W must
return **46.19 pp (739/1600)**, W_b **49.88 pp (798/1600)**, and the fold's three depths **47.62 /
51.56 / 51.31 pp** — all to the last win and on the PER-TEAM rows. A failure on any one of them
makes every level in this directory suspect and will be reported as the headline.

**Bar.** `|Δ(cont − W)| > 3.69 pp` AND the Δ's cluster-bootstrap CI excludes the 3.69 point.
**Reported at all three new checkpoints**, and the registered branch clause is **≥ 2 of the 3**.

**Prediction (mine, before the games).**

| | P |
|---|---|
| Δ at +12.09M is POSITIVE (the fold path still above its frozen parent off-slice) | **0.80** |
| Δ clears the 3.69 pp floor **UPWARD** at ≥ 2 of the 3 new points (branch a's first clause) | **0.12** |
| Δ clears the floor **DOWNWARD** at ≥ 1 new point (branch c) | **0.08** |
| every new checkpoint WITHIN FLOOR | **0.80** |
| the untaught level at +12.09M is within ±2 pp of the +6M level (51.31) — i.e. the off-slice row is FLAT across the last 50 % of the budget | **0.60** |

*Reasoning, stated so it can be scored.* The off-slice row went 47.62 → 51.56 → 51.31: it moved in
the first 3M and then stopped. Teacher agreement saturated at +7.7M. If the off-slice lean is
mostly the continuation/seniority effect the untaught meter has always been sensitive to, it should
drift slowly upward; if it is teaching, it should have stopped with the teaching. Neither is
separable here — that is what `ai_v13_09_wcont` exists for, and it is NOT READ in this job.

### 3.2 Row 2 — PER-SLICE PILOTING (the matched-extraction row)

**Instrument, REUSED VERBATIM.** The same `main.untaught_meter` engine pointed at the **2-team
taught-slice manifest in teacher order** (the order is the seed offset; re-ordering changes every
game), against **the untaught meter's own fixed opponent** — NOT arm W's pool sentinels, because a
sentinel is the trainee's OWN snapshot and arm W's differ from W_b's (floor-read hazard F-G), which
would make the seed floor and the treatment contrast face different opponents. **800 games per
(ref, team)**, `--seed 0`, 3 shards × 2 workers.

**REGISTERED refs measured here:** `cont_p7.5M`, `cont_p12M` (the two the GO names), `cont_p9M` and
`fold_p1M` (trajectory points, registered as such), and 🚨 **`armW`, re-measured as the
REPRODUCTION CHECK** — it must return **394/800** on Big-5 and **376/800** on DDTar.

**DECLARED IMPORTS, and the warrant for them.** `fold_p3M`, `fold_p6M` and **`W_b` (the floor)** are
**NOT re-run**; their cells are taken from `fold1_read_2026-09-19/out/fold_slice_read.json`
(Big-5: fold+3M 374/800, fold+6M 345/800, W_b 432/800; DDTar: 475/800, 430/800, 444/800). The
meter is deterministic at seed 0 / concurrency 1 — demonstrated three times before this read and
re-demonstrated five more times inside it (§3.1) — and the same opponent, manifest, order, seed and
cell size are used. **The arm-W reproduction check above is the warrant**: if arm W returns anything
other than 394/800 and 376/800, every imported cell is void and this row is reported as INVALID
rather than patched. Registering the import here, in advance, is the alternative to spending 4,800
battles re-deriving rows that are the same games.

**Matched-extraction discipline** (the era standard): one harness; a MATCHED, ZERO-HEAD-START
baseline (arm W itself, same role, same opponent, same pin); draws PAIRED by construction (CRN on
battle index); **per-cell first with Wilson intervals**, pooled second and labelled. Newcombe on
every difference — **CONSERVATIVE under CRN**, because the tool retains no per-battle outcome
vector so the pairing cannot be exploited.

🚨 **SENIORITY IS A SEPARATE TERM AND IT IS NOW LARGER, NOT SMALLER.** `Δ = seniority + extraction`
and at +12.09M the fold path carries **twelve million** steps arm W does not. This read cannot split
them; the nearest bound in hand is row 1 (the same steps read off-slice), and the arm that actually
splits them is `ai_v13_09_wcont`, **live on the GPU and NOT READ here**.

**Bar.** Per team: `Δ = cont − arm W` against that cell's own measured seed floor (0.0475 Big-5,
0.0850 DDTar); clause (a) then clause (b) on the Newcombe interval.

| | P |
|---|---|
| Big-5 at +12.09M is still BELOW the parent (negative) | **0.65** |
| Big-5 at +12.09M is ABOVE its own +6M value (−0.0612), i.e. the decline stopped or reversed | **0.50** |
| DDTar at +12.09M is POSITIVE | **0.70** |
| either slice clears its own cell's floor at +12.09M | **0.25** |
| **the +3M peak stands** — no later point exceeds fold+3M on EITHER slice (Big-5 0.4675, DDTar 0.5938) | **0.60** |
| a teacher still beats the fold path on the Big-5 team (t1 = 0.5763) | **0.85** |

### 3.3 Row 3 — EXTERNAL ANCHOR: Metamon `SmallRL`, greedy, AWAY team set, at +12.09M

**Instrument.** `python -m main.anchors --opponent metamon:SmallRL --regime greedy --teamset away
--games 100 --device cpu`, the tool of record, the +6M read's own `run_away_cell.sh` with the ref
swapped and the port inside this job's band. 🚨 `--out` under the job tmp, never the cwd.

**Comparators.** arm W **0.500** [0.404, 0.596]; W_b **0.590**; the fold @ +6M **0.540**
[0.443, 0.634]; floor `|W − W_b|` = **0.090**.

**Bar.** `|Δ(cont − W)| > 0.090` AND the CI excludes 0.090. **Registered as DESCRIPTIVE**: at 100
games the cell resolves ~±0.10 against a 0.090 run-level floor, so it cannot separate the arms
whatever they did. Saying so now rather than afterwards is the point of registering it.

| | P |
|---|---|
| the away cell clears the 0.090 floor in either direction | **0.15** |

### 3.4 Row 4 — LADDER at matched snapshot COUNT

**Registered as almost certainly NOT RUNNABLE, with the check named:** `eval/pool_snapshot_count`
(and the `snapshots/` directory, and the absence of `snapshot_ladder/`). **If any promotion
occurred, the ladder is refit at matched snapshot COUNT against arm W and W_b and reported with its
recipe stamp (`eval_sentinel_edges_dropped`, rule 24); if none did, the row is a STRUCTURAL FINDING
and the frozen-pool `win_rate_vs_pool` series is its descriptor, with no bar attached.** n = 0 ⇒ no
rating quoted for this arm in either direction.

### 3.5 Row 5 — COLLATERAL, and the instruments deliberately NOT run

**The untaught meter IS the off-slice/collateral read** — row 1 — and no second instrument is added.

🚨 **`python -m main.exploitability` is NOT APPLICABLE and is declared so in advance**, for the same
reason as at +6M: its `--help` says *"Generation exploitability curve from fleet_admission-schema
artifacts. Bookkeeping only — no battles, no models."* `ai_v13_08_fold1_cont` has produced no such
artifact, and manufacturing one is a separate 800-games-per-arm battery carrying a fixed cross-era
REFERENCE that row 2 does not have.

🚨 **`ai_v13_09_wcont` IS NOT READ.** It is live on the GPU as of 07:07 PT and is the era's G5
continuation control. Reading a live arm through an offline meter is a different operation with its
own preconditions, and the GO reserves it for a separate read. **No checkpoint, event file or log
of that run is opened by this job.**

---

## 4. THE BRANCHES, as the GO wrote them

| branch | condition | reading |
|---|---|---|
| **(a) THE FOLD PAID AT CONVERGENCE** | any row clears its bar at ≥ 2 of the three new points | the teaching needed the budget, and it arrived |
| **(b) EXTRACTION SATURATED BEFORE AGREEMENT DID** | the +3M peak stands and the later points are flat/down | agreement is not the meter — the stop rule's own conjunct measures the wrong quantity |
| **(c) NOT DETECTED AT CONVERGENCE** | nothing clears and there is no trajectory | and then `ai_v13_09_wcont` (banking ~18:00 PT, read separately) decides seniority vs extraction |

**My prior over the branches, before any game:** (b) **0.55** · (c) **0.30** · (a) **0.15**.
⚠️ (b) and (c) are not exclusive as written — a read can show the +3M peak standing AND nothing
clearing. **Registered tie-break: if both hold, the branch is reported as (b), because a trajectory
is more informative than a null**, and (c) is recorded as the verdict-word underneath it.

**The question this read exists to answer, in one sentence:** *did the last 25 % of the teaching —
from +6M to the stop point at +7.67M and on to +12M — move ANY row?*

---

## 5. Execution constraints (recorded so the numbers can be audited)

* **CPU ONLY** — `CUDA_VISIBLE_DEVICES=""`, `nice 15`, `OMP_NUM_THREADS=1` for peers. **The GPU
  hosts the live control `ai_v13_09_wcont`, which is never touched and never read.**
* `POKESIM_SIM_BRIDGE_BIN=/home/goodlad/dev/gen3ai/src/rust_sim/target/release/sim_bridge`.
* Ports **9400–9499** only; **:8000 and :8001 never touched**; `main.anchors` starts and stops its
  own server. No process this job did not start is signalled; kills by explicit PID only.
* Every battle runs **from the MAIN checkout**; `--json`/`--out` under
  `/home/goodlad/.claude/jobs/9ab51de6/tmp/fold1c_read/`. **Nothing written under `models/`**; no
  file under `src/` changed. Other agents own `leaf_control/` and `viewemit/` — untouched.
* A run whose timeouts exceed 25 % of attempted battles is **INCONCLUSIVE** (rule 12). Timeouts are
  reported for every cell. The box carries the live GPU arm; load ran 23–29 at launch.

## 6. What this read CANNOT do, whatever it returns

* It cannot separate **EXTRACTION from SENIORITY**, and the confound is now twice the size it was
  at +6M. `ai_v13_09_wcont` is that arm; it is live and is not read here.
* It cannot supply a **continuation control** at matched depth for the same reason, so every delta
  is against a **FROZEN** parent (ledger 2026-09-06 cell 2: +3.45 pp on v8's parent; §2.2 says the
  correction does not bite on OUR parents, but that was measured at ~28M parent depth and ~1M fold
  depth — an IMPORT, printed beside every delta).
* It cannot make a row that clears an imported floor a **family verdict** (rules 19/22). One arm,
  and the continuation is the SAME arm with more steps — **more steps are not a replicate.**
* It cannot say anything about the fold recipe **in general**, at another dose, with other
  teachers, or at 277M.
* It cannot read a **trend** off a short window (the standing refusal, now 5-for-5 on this
  campaign). A six-point trajectory on ONE arm is a description of that arm.
