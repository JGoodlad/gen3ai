# PRE-REGISTRATION — THE ERA-1 FOLD's REGISTERED READ (`ai_v13_07_fold1`)

**Committed BEFORE any registered number in this directory exists.** The GO is the ledger entry of
2026-09-18 (`deddfcd5`, the fold's launch) and the completion entry of 2026-09-19 (`0158dff4`),
whose closing paragraph names exactly the four rows read here and says in its own words that
*"nothing here says the fold paid"*.

---

## 0. The arm, and what is already known about it

`ai_v13_07_fold1` — **the era-1 fold**: a FORK of `ai_v13_02_flywheel_winprob` (**arm W**, the 75M
win-prob flywheel arm) at 75,005,952, distilling from the era's **two win-prob exploiters** as
teachers, `--fork-lr 2.8e-5 --fork-lr-freeze`, `--distill-team-bias 0.4`, both teachers in the
stable pool each piloting its own pinned team, `--checkpoint-every-steps 500000`, +6,000,000 steps
to **81,100,800**. Pin `6eb9c776` throughout, config v119 / `gen3_critic_route_wave_v1`.

🚨 **CORRECTION (2026-09-21, ledger `f389fce4`) — THE ARM DESCRIPTION ABOVE UNDER-COUNTS ITS OWN
LEVERS BY ONE.** `ai_v13_07_fold1` also passes **`--team-block-episodes 64`**, where arm W, the
continuation control `ai_v13_09_wcont` and *every* era exploiter pass **1**. It is named in the
fold's argv and in no registration — this one, the launch entry or the control read — so every
statement anywhere that the fold moved "three levers" off plain continuation (the distill LOSS, the
0.4 team bias, the two specialists in the pool) is wrong by one: **it moved FOUR**, the fourth being
a 64× increase in how many consecutive episodes are played on one team, which interacts directly
with the team bias this arm also carries.

**What the correction does and does not change.** It does NOT touch any number in this file, nor the
fold-vs-split contrast (`ai_v13_11_split_lossoff` copies this argv exactly and differs only in the
distill coefficient, so THAT comparison remains one lever). It DOES change what "the ecology" names
— from bias + pool to **bias + pool + team-block**, a three-way bundle — and therefore what a
fold-vs-CONTINUATION difference can be attributed to. The split read (`7c26076e`) subsequently found
the whole bundle inert at this dose, team-block 64 included, so nothing downstream depends on
separating them; but a reader should not learn the lever count from a registration that was written
before anyone had compared the argvs side by side. **This correction is appended, never edited in:
the body above is what was believed when it was committed.**

**The two taught teams (the SLICE):**

| teacher | run | its pinned team | archetype (human description) |
|---|---|---|---|
| **t1 — Big-5** | `ai_v13_05_exploit_big5starmie` | `data/teams/sample/f6229d2c867e21d6.txt` | **balance** — climbed 0.640 → 0.740 vs arm W, ~45 turns/game |
| **t2 — DDTar** | `ai_v13_06_exploit_ddtar_spikes` | `data/teams/sample/9eb3abdc52876a63.txt` | **offense** — opened at 0.700, closed at 0.740, ~32 turns/game |

**Banked and NOT re-derived here** (from `0158dff4`): the stop signal never fired;
`teacher_agreement_on_slice` was STILL RISING at the end (0.758 → 0.796 → **0.815**), so the budget
stopped the fold, not its stop rule; the two teachers diverged **2.5×** (`t1_gated_frac`
0.226 → 0.305, `t2_gated_frac` 0.210 → **0.122**); `on_slice_kl` DOUBLED (0.327 → 0.653) while
overall `kl` FELL (0.662 → 0.490); dose 4.272e-9, **0.20× the v8 reference**.

### 🚨 Two things were LOOKED AT before this registration was written, and are disclosed here

Pre-registration is worthless if the registrant has already seen the answer. Two facts about the
run's own tree were read while scoping this job, and both are declared:

1. **The fold has NO `snapshot_ladder/` directory and ZERO post-fork snapshots.** `snapshots/`
   contains exactly the 20 files the pool-seeding copied from arm W (26.0M → 72.0M, all with arm
   W's own mtimes). ⇒ **Registered row 3 (ladder at matched snapshot COUNT) is NOT RUNNABLE at
   n = 0**, which is not merely "below the n ≥ 12 report floor" — there is no node to fit. It is
   reported as a STRUCTURAL FINDING, not as a measurement.
2. **The reason is visible in the run's own eval rows: `win_rate_vs_pool` reads 0.488 → 0.446 →
   0.412 across the three post-fork eval cycles**, never near the 0.55 promotion gate. Because the
   fold promoted nothing, those three points are against an **identical, frozen** opponent set
   (arm W's own 20 late snapshots) — an unusually clean within-run series. It is a DESCRIPTOR here
   and is **not** one of the registered rows; no bar attaches to it.

Nothing else about the fold's competence has been measured. The untaught meter, the per-slice
piloting row and the anchor cell have **no numbers of any kind** at the time of this commit.

---

## 1. The comparators, by name

| role | run | resolved file |
|---|---|---|
| **the fold, +1M** | `ai_v13_07_fold1` | `checkpoints/checkpoint_76005984_steps.zip` (+1,000,032) |
| **the fold, +3M** | `ai_v13_07_fold1` | `checkpoints/checkpoint_78006048_steps.zip` (+3,000,096) |
| **the fold, +6M** | `ai_v13_07_fold1` | `final_model.zip` @ 81,100,800 (+6,094,848) |
| **PARENT / CONTROL — arm W** | `ai_v13_02_flywheel_winprob` | `final_model.zip` @ 75,005,952 |
| **the SEED FLOOR arm — W_b** | `ai_v13_04_flywheel_winprob_b` | `final_model.zip` @ 75,005,952 |
| **teacher t1 (Big-5, balance)** | `ai_v13_05_exploit_big5starmie` | `final_model.zip` @ 83,066,880 |
| **teacher t2 (DDTar, offense)** | `ai_v13_06_exploit_ddtar_spikes` | `final_model.zip` @ 83,066,880 |

**The run-level floors, imported VERBATIM from
[`flywheel_wb_floor_read_2026-09-18/`](../flywheel_wb_floor_read_2026-09-18/):** untaught
**3.69 pp**, ladder **25.2 Elo** at 20 matched-count nodes (53.1 on the common-step frame), Metamon
`SmallRL` greedy **away 0.090**. Every one is `|arm W − W_b|` — **one pair BOUNDS a floor and does
not estimate one** (rules 19/22); there is no CI on a floor; WITHIN FLOOR is never "equivalent"
(rule 6).

⚠️ **A FLOOR IS A PROPERTY OF THE DEPTH AND THE REGIME** (§3.3). The 3.69 pp untaught floor is a
**75M fresh-arm seed** floor; this arm is a **FOLD**, and the meter's fold floors are 1.19 pp
(END-depth replicate), 1.66 pp (frozen-dose), 4.27 pp (controller-live) and 1.00 pp (G5
continuation). 3.69 pp sits between the frozen-dose and controller-live fold floors, which is why
the GO chose it; it is still an IMPORT and is labelled as one everywhere below.

---

## 2. THE DECISION RULE — fixed here, in advance

Copied from the floor read's `PREDICTION.md` §2 so the two reads are commensurable:

> A contrast is **OUTSIDE THE FLOOR** iff **(a)** `|Δ| > floor` **AND** **(b)** the Δ's own 95 % CI
> **excludes the floor POINT**. Otherwise **WITHIN FLOOR at n = 2** — never "equivalent".

Sign convention throughout: **Δ = fold − arm W**, so POSITIVE means the fold is ahead of its parent.

---

## 3. The registered rows, their bars, and the prediction attached to each

### 3.1 Row 1 — THE UNTAUGHT METER (off-slice; this IS the collateral read)

**Instrument.** `python -m main.untaught_meter`, the same tool / opponent / config / seed /
concurrency as the floor read: registry opponent `untaught_meter_opponent`
(= `ai_v9_29_rev1_0823@24,000,000`), the untaught-8 manifest in its canonical order,
**200 games/team, `--seed 0`, concurrency 1**, shard over teams, and **all five refs in ONE
invocation** so every ref sees the identical 8 teams and the identical games (CRN). Both config
resolutions (registry and `--config auto`) — §8.5's rule that a level existing under only one
resolution is not a level.

**Bar.** `|Δ(fold − W)| > 3.69 pp` AND the Δ's cluster-bootstrap CI (team is the unit, 20,000 draws,
ONE shared index set — rule 10) excludes the 3.69 point. **Reported at all three checkpoints; the
endpoint alone is not a result** (§2.3: a fold's off-slice track is a HOLE-then-recovery shape and
reading only the end reports the recovery as the effect, or the hole as the effect, depending on
where the budget happened to stop).

**Prediction (mine, before the games).** The gen-era law is that **every fold digs an early
off-slice hole, ~3–4 pp by +1M, regardless of teacher content** (§2.3, three independent cells), and
that near-parent teachers let the student climb back out while funded/distant ones do not. These
teachers are **8M-step specialists forked off the parent itself** — near in weights, far in
behaviour on their own team. Against that, the dose is **0.20× the v8 reference and FROZEN**, a
quarter of the usual fold dose, which should make any hole shallower.

| | P |
|---|---|
| Δ at +1M is NEGATIVE (a hole, any depth) | **0.70** |
| Δ clears the 3.69 pp floor **UPWARD** at ≥ 2 of the 3 checkpoints (branch a's first clause) | **0.10** |
| Δ clears the floor **DOWNWARD** at ≥ 1 checkpoint (branch c) | **0.35** |
| every checkpoint WITHIN FLOOR | **0.55** |

### 3.2 Row 2 — PER-SLICE PILOTING (the matched-extraction row)

**Instrument, and the choice the GO asked me to declare.** The same
`python -m main.untaught_meter` engine, pointed at a **2-team manifest containing only the two
taught teams**, with **the untaught meter's own fixed opponent** (`untaught_meter_opponent`
= `ai_v9_29_rev1_0823@24,000,000`) — **NOT** arm W's pool sentinels.

*Why that opponent.* It is ONE frozen, era-external checkpoint, identical for every ref and already
the instrument of record for this programme's piloting reads; arm W's pool sentinels are the
**trainee's own snapshots**, which differ between arm W and W_b (floor-read hazard F-G) and would
make the seed floor and the treatment contrast face different opponents. Every ref pilots the
**same pinned team** against the **same opponent drawing the same 800-long pool-team sequence under
the same dice**, so this is a PILOTING read and not a head-to-head.

**Matched-extraction discipline** (the era standard): both arms on ONE harness; the baseline is
MATCHED and ZERO-HEAD-START (arm W itself, same role, same opponent); **the draws are PAIRED** by
construction (CRN on battle index); per-cell first with Wilson intervals, pooled second and labelled
as such. ⚠️ **Seniority is a SEPARATE term and it is NOT zero here** — the fold has 6M steps arm W
does not, and the teachers have 8M. The fold-vs-W delta therefore decomposes as
`seniority + extraction` and this read CANNOT separate them; the nearest available bound on
seniority is the untaught row (row 1), which is the same 6M of ordinary progress read off-slice.
Recorded as a standing limitation, not fixed.

**Refs, 800 games per (ref, team):** fold@+6M, fold@+3M, **arm W**, **W_b**, **t1**, **t2** — the
teachers on their own team give the TEACHER'S CEILING, and on the other team a descriptor of
cross-team transfer.

**Bar.** Per team: `Δ = fold − arm W` against the **seed floor measured on that same cell**,
`|arm W − W_b|` on that team, run here at the same 800 games. Clause (a) `|Δ| > floor`; clause (b)
the Δ's CI excludes the floor point. ⚠️ **The interval is the Newcombe two-proportion interval,
which is CONSERVATIVE under CRN** — the games are paired and the tool does not retain per-battle
outcomes, so the pairing cannot be exploited. A cluster bootstrap over 2 teams is not a CI and is
not reported as one.

**Also registered:** whether the **Big-5 slice gains more than the DDTar slice** — the divergence
the completion entry recorded (t1's gated share 0.226 → 0.305, t2's 0.210 → 0.122) predicts it.

| | P |
|---|---|
| fold@+6M − arm W is POSITIVE on ≥ 1 slice | **0.75** |
| fold@+6M − arm W clears its own cell's seed floor on ≥ 1 slice | **0.40** |
| …on BOTH slices | **0.20** |
| the **Big-5** slice gain exceeds the **DDTar** slice gain | **0.60** |
| a teacher beats the fold on its own team (teacher ceiling not reached) | **0.85** |

### 3.3 Row 3 — LADDER at matched snapshot COUNT

**NOT RUNNABLE, and the reason is the finding** (§0.1): the fold has **zero** post-fork snapshots
and no `snapshot_ladder/`, so there is no fit to make. Reported as a structural finding with the
`win_rate_vs_pool` series beside it as a descriptor. **n = 0 ⇒ no verdict, no Elo quoted for this
arm, in either direction.** Registered in advance so that the absence cannot later be dressed up as
"the ladder did not separate them".

### 3.4 Row 4 — EXTERNAL ANCHOR: Metamon `SmallRL`, greedy, AWAY team set

**Instrument.** `python -m main.anchors --opponent metamon:SmallRL --regime greedy --teamset away
--games 100 --device cpu --out <job tmp>`, the tool of record, exactly the floor read's
`run_away_cell.sh` with the ref swapped. 🚨 `--out` under the job tmp, never the cwd.

**Comparators.** arm W **0.500** [0.404, 0.596], W_b **0.590**; **floor `|W − W_b|` = 0.090**, itself
inside the anchors SOP's imported three-seed 0.110.

**Bar.** `|Δ(fold − W)| > 0.090` AND the Δ's CI excludes 0.090. At 100 games this cell resolves
~±0.10, so it is **registered as descriptive** — I expect it to be WITHIN FLOOR whatever the fold
did, and say so now rather than after.

| | P |
|---|---|
| the away cell clears the 0.090 floor in either direction | **0.15** |

### 3.5 Row 5 — COLLATERAL, and the exploitability meter

**The untaught meter IS the off-slice/collateral read** — that is what row 1 measures and no second
instrument is added for it.

🚨 **`python -m main.exploitability` is NOT APPLICABLE to this arm, and that is declared in advance
rather than discovered later.** Its `--help` is unambiguous: *"Generation exploitability curve from
fleet_admission-schema artifacts. Bookkeeping only — no battles, no models."* It consumes admission
JSONs in generation order; `ai_v13_07_fold1` has produced none, and manufacturing one is a separate
job. **The tool is read, found to be bookkeeping over an artifact this arm does not have, and not
run.**

---

## 4. The branches, as the GO wrote them

| branch | condition | reading |
|---|---|---|
| **(a) THE FOLD PAID** | untaught clears the floor **UPWARD** at ≥ 2 checkpoints **AND** the per-slice gains clear | the first gen-era-style fold to clear its parent off-slice |
| **(b) LOCAL TRANSFER** | per-slice gains clear, untaught WITHIN floor | what every gen-era fold has done (§2.3) — it TEACHES on-slice and does not carry off it |
| **(c) THE LEAK** | untaught DOWN past the floor | taught content arriving off-slice; the classic fold hole, now at 75M depth and a 0.20× dose |
| **(d) NOT DETECTED** | nothing clears | at +6M with the fold demonstrably unconverged (agreement still rising) — and then the question is what the already-live continuation `ai_v13_08_fold1_cont` (+6M more, launched 2026-09-19 01:28 PT) would decide |

**My prior over the branches, before any game:** (b) **0.40** · (c) **0.25** · (d) **0.25** ·
(a) **0.10**.

**What the continuation would decide, registered now so the answer is not fitted to the result.**
`ai_v13_08_fold1_cont` doubles the budget on the SAME teachers at the SAME frozen dose. It is
decisive for exactly one question — *did the budget, or the recipe, stop the fold?* — and only in
one direction: **if the per-slice gain grows and the untaught row does not fall further, the +6M
budget was the binding constraint**; if the per-slice gain is flat at +12M while
`teacher_agreement_on_slice` keeps rising, then agreement is not the quantity that predicts
piloting and the stop rule is measuring the wrong thing. It cannot rescue branch (c): a leak that
deepens over 6M will not close over the next 6M at a frozen dose.

---

## 5. Execution constraints (recorded so the numbers can be audited)

* **CPU ONLY** — `CUDA_VISIBLE_DEVICES=""`, `nice 15`, `OMP_NUM_THREADS=1` for peers. The GPU hosts
  `ai_v13_08_fold1_cont`, which is **never touched**.
* `POKESIM_SIM_BRIDGE_BIN=/home/goodlad/dev/gen3ai/src/rust_sim/target/release/sim_bridge`.
* Ports **9400–9499** only; **:8000 and :8001 are never touched**; `main.anchors` starts and stops
  its own server.
* Every battle runs **from the MAIN checkout**; `--json`/`--out` under
  `/home/goodlad/.claude/jobs/9ab51de6/tmp/fold1_read/`. **Nothing is written under `models/`**, and
  no file under `src/` is changed.
* A run whose timeouts exceed 25 % of attempted battles is **INCONCLUSIVE** (rule 12). Timeouts are
  reported for every cell.

## 6. What this read cannot do, whatever it returns

* It cannot separate **extraction from seniority** (§3.2). The fold has 6M steps its parent lacks.
* It cannot supply a **continuation control** at matched depth — `--control` needs a plain +6M
  continuation of arm W with no teacher, and none exists. Every delta here is against a **FROZEN**
  parent and therefore credits the fold with whatever ordinary progress arm W would have made
  (ledger 2026-09-06, cell 2: +3.45 pp on v8's parent). ⚠️ On OUR parents that correction is not
  significant (§2.2, three draws, two gen-era parents) — but that was measured at ~28M and ~1M fold
  depth, not at 75M, so it is an IMPORT and the missing control is a standing limitation of this
  read, printed beside every delta.
* It cannot make a row that clears an imported floor a **family verdict** (rules 19/22). One arm.
* It cannot say anything about the fold recipe **in general**, at another dose, with other teachers,
  or at 277M.
