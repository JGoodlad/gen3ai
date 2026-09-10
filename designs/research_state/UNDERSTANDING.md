# UNDERSTANDING.md — what we believe about the research, NOW

## 0. What this file is, and how to read a claim

`ARCHITECTURE.md` states what the MODEL is now; `ledger.md` records what was believed at each moment
and is never edited; the `learning/` notes explain concepts rather than holding a position. **What we
currently believe about the research** had no home. This file is it — the research twin of
`ARCHITECTURE.md`, **always-current, present tense only.** It carries no narrative of how a belief
changed; that is the ledger's job, and when the two disagree **the later ledger entry wins and this
file is a bug — fix it in the same pass.**

Every claim carries two things:

**(a) An evidence tag**, in the project's own vocabulary. The floor referred to is always an
**operational replicate floor**: what two same-recipe arms differ by, measured as the MAX pairwise
|Δ| over the replicates in hand.

| tag | means |
|---|---|
| **SIGNIFICANT** | magnitude above the replicate floor AND the CI excludes zero |
| **WITHIN FLOOR** | the CI excludes zero but the magnitude is under the floor — the games are consistent, the instrument cannot separate the arm from a re-run of the same recipe |
| **NOT DETECTED** | magnitude above the floor but the CI spans zero |
| **EQUIVALENCE SUPPORTED** | the DELTA's own CI lies inside the floor band — the strong form, never "the intervals overlap" |
| **INDETERMINATE** | outside the floor AND straddling zero — the instrument answered nothing |
| **REFUTED** | a pre-registered prediction contradicted, often with the sign reversed |
| **UNVERIFIED** | asserted, not checked against code or a measurement |

"No effect" is retired as a label for any of these.

**(b) A pointer** — a ledger entry by DATE + TITLE, or a path under `measurements/`. A number with
no pointer is not a measurement.

---

## TL;DR — fifteen bullets

1. **The goal is a gen3 OU generalist that keeps improving.** The intended engine is the
   **flywheel**: train narrow exploiters (best responses on a few pinned teams), distil them back
   into the generalist (the *fold*), repeat. It has delivered once — v8_14, **+69 anchored ELO** —
   and has not gifted since. [SIGNIFICANT · `learning/distillation_flywheel_lessons.md`]
2. **v8's celebrated gift REPLICATES**: three fresh arms of its exact recipe give **+4.56pp
   [+1.14, +7.81]** untaught at +1.09M against v8's own +4.64. [SIGNIFICANT · ledger 2026-09-05 ·
   *P1 — v8's gift REPLICATES*]
3. **But a plain continuation of the same parent gains the same** — no teacher, no distillation
   term, no stable opponents: **+3.45pp [+0.46, +6.48]**, and the full recipe minus the
   continuation is **−1.11pp [−3.12, +0.91]**, inside the ±3.22 floor. **v8's recipe ≡ training its
   parent on.** The gift was the parent still learning, measured against a frozen copy of itself.
   [EQUIVALENCE SUPPORTED · ledger 2026-09-06 · *CELL 2*]
4. **Our parents gain NOTHING from a plain continuation ON THE UNTAUGHT SLICE** — G5 **−1.92pp
   [−3.98, +0.46]**, three draws across two gen-era parents, none gains. **The frozen-parent
   baseline stands on our side for the untaught meter**, and the cell-2 alarm ("every untaught delta
   is against the wrong baseline") does NOT generalise there.
   [NOT DETECTED · ledger 2026-09-06 · *G5 RESULT*]
4b. **But the SAME continuation DOES gain on the TAUGHT slice: +2.09pp [+0.47, +3.79], clearing
   zero.** The slices answer differently, so "the frozen parent is a fair baseline" is a claim about
   the untaught meter only. Re-based on the continuation, the folds' on-slice gift is **+2.6 to
   +3.4pp** (four of six arms clear zero) against ~+4.2 to +5.5pp read against the frozen parent —
   **the gift survives the correction**. ⚠️ That re-based column is an **UPPER BOUND**: G5's taught
   reading is at fork+1.18M against the fold readings' ~fork+4.45M, so it subtracts too small a
   continuation, and a fuller correction pushes the numbers down, possibly below zero. **The
   asymmetry is the carrying finding** — ordinary continued training improves taught teams while
   doing nothing measurable for the untaught 8, so part of what was credited to a fold on-slice was
   available from training alone. [SIGNIFICANT, with the depth caveat · ledger 2026-09-06 ·
   *G5 slice (iii)*]
5. **⇒ The cross-era difference is the PARENT, not the fold and not the meter.** *Maturity* is the
   SHAPE of that result, one CONFOUNDED candidate cause — the two parents differ in step count
   (277M vs 28M) **and** in architecture, obs dim, reward composition and ecology. [UNVERIFIED as a
   cause · ledger 2026-09-06 · *G5 RESULT*]
6. **What our folds do**: teach **~+5pp on the taught slice** (SIGNIFICANT), dig a **~3–4pp
   off-slice hole by +1M** regardless of teacher content, and recover only when the teachers are the
   parent's own near exploiters. Funded (further-travelled) teachers ROB **−2.41pp [−4.37, −0.63]**
   and buy nothing on-slice. **DOSE is null** across a 4× range and at two frozen doses.
   [ledger 2026-09-05 · *RETRACTION + full 2×2 series* and *K=6 CELL*]
7. **On the gen side the distillation LOSS and its team-sampling bias carry both the gift and the
   leak** — with the loss off (C1) the fold moves the untaught meter by nothing, with it on it robs
   6–9pp. On v8's side the SAME cell gifts +4.92pp. Same cell, opposite verdicts.
   [ledger 2026-09-03 · *C1 vs B2 COMPLETE*; ledger 2026-09-06 · *CELL 1*]
8. **A fold's Δθ is a dose-scaled RANDOM WALK of the trunk** — `|Δθ| ∝ t^0.48`, replicate cosine
   0.56, the same magnitude with the loss on or off and the same for robbing and neutral teachers.
   The off-slice KL is carried by encoders + transformer (**51–74%**); the pointer head carries
   **0.5–5.6%**. **Robbery is a DIRECTION, not a size**, and no scalar we own measures it.
   [meas: `arch_transfer_2026-09-05/fold_displacement/`]
9. **Teacher facts that survive.** v8's teachers are **LOCAL** (sibling-control R **1.83
   [1.53, 2.17]**) and DESCEND from their fold parent; ours are **GLOBAL-from-origin SIBLINGS** of
   theirs (0.53 nats from the parent before any exploiter training), and the fork origin explains
   only **~24% / 12%** of the gap. [SIGNIFICANT cross-era · meas:
   `arch_transfer_2026-09-05/content_locality_v2/`]
10. **Exploiter drift is FORGETTING, not specialisation.** Every exploiter set pilots the untaught 8
    WORSE than its origin; the funded teachers by **−7.81pp [−15.23, −0.39]**, at roughly **3pp of
    on-slice edge per 1pp of untaught win rate lost**. [SIGNIFICANT · meas:
    `arch_transfer_2026-09-05/exploiter_competence/`]
11. **The ARCHITECTURE account of the era gap is CLOSED on every leg** — head gradient kernel, trunk
    sharing kernel, v8's FiLM team code (3.8% of on-slice KL), and search depth (re-ranks +0.19% of
    changed decisions) are each NOT DETECTED or REFUTED. [meas: `arch_transfer_2026-09-05/`]
12. **v8_14 is genuinely the stronger policy**: it beats v9_59 **63.3% [59.3, 67.2]** over 559
    decisive games, and the anchored-ELO extrapolation is corroborated (+91.7 predicted, +94.9
    measured). 560 direct games resolve the gap to ±30 where 720 bot games leave ±86.
    [SIGNIFICANT · meas: `arch_transfer_2026-09-05/cross_era_head_to_head/`]
13. **The eval-trace quota is LOSS-ENRICHED and raw calibration reads are artifacts** — the same
    traces read ECE 0.237 raw and **0.025 reweighted**, skill −0.080 raw and **+0.265** reweighted.
    [SIGNIFICANT · meas: `winprob_critic_baseline_2026-09-06/`]
14. **The critic's disease is RESOLUTION (blur), not level** — population-mean gaps are small and
    sign-flip by ecology; ~39% of conviction-region blur is the IRREDUCIBLE hidden-information
    floor. The win-prob gate's primary endpoint is therefore a resolution gate.
    [`critic_calibration_plan.md` §0]
15. **Live now: `ai_v12_02_winprob_critic`** (arm 2, the full production surface at 2048×32;
    arm 1 `ai_v12_01` ran a stripped architecture and is DEAD, not evidence) — `V(s) = σ(win logit)
    ∈ [0,1]`, the value loss is that head's BCE against the terminal WIN INDICATOR, γ=1, no PopArt,
    no shaping. Read by `python -m main.critic_gate`; famine pre-test floor **38 Elo** against
    `ai_v9_29_rev1_0823`: **10M read NO KILL (−30 at n = 4), n = 12 read WITHIN FLOOR (−34 at 26M)**;
    the standing kill is G7 (stall rate + episode length on the EVAL traces), at ~0%. [§4.2; ledger
    2026-09-07 · *10M DECIDING READ* and *THE n = 12 STRENGTH READ*]

---

## 1. The mission and the era map

**The goal.** A gen3 OU generalist that keeps improving — not a specialist, not a search agent. The
owner's permanent side constraint is that it must be able to play the public ladder; the external
milestone is Metamon's published gen3ou result, ~Elo 1511 / GXE 64 (`ladder_readiness.md`).

**The intended engine is the flywheel.** Self-play stalls because the generalist plays every team
adequately and none expertly (the amortization gap). So: fork narrow **exploiters** against a few
pinned teams, then **fold** them back into the generalist with a distillation loss on top of PPO,
then repeat against the improved generalist. The value of the wheel is entirely in the **untaught**
teams — taught teams are a rounding error of the 719-team pool.
[`learning/distillation_flywheel_lessons.md`, `learning/population_game_theory.md`]

**The 2026-09-06 pivot.** With the flywheel's gift dissolved into "the parent was still learning"
(§2), the owner-ordered next move is a **cleaner value function** as the starting point for long
(multi-day, 75M-step) runs and for search: the win-probability head promoted to BE the critic.
That is the `ai_v12` chapter and it is what is running now (§4).

**The goal for the coming week, in the owner's words (2026-09-06):** *"understand the value network
as win prob — do we need PBRS, do we need to bootstrap with a frozen win-prob value head. Right now
we are working on running a model further and validating the win-prob head."* **The long-term goal
(same statement):** *"a mature model, free of tech debt, that is understood to both benefit from
search and teacher distillation, to see if we can repro v8's gift."* Consequences that are now
standing (owner rulings, same day): **the recipe of record is SPARSE** — the owner's strong bias, because
it has fewer moving parts and an A/B needs no parent. The FROZEN-φ bootstrap (actor-only potential,
`--win-prob-pbrs-frozen`) is a CHECK on that bias, and its potential is **the SPARSE arm's OWN mature
win-prob head**, not the gen-era fold parent the design §5.4 command names — the registered gen-era
source survives only on the famine-KILL branch, where no mature self head exists. Its read is
**wall time to roughly equivalent strength** at matched snapshot count, read early: unless the
bootstrap is *massively* faster, SPARSE stays the recipe and the bootstrap arm is stopped. Under this
critic the SELF-φ rung does not exist (design §3.7), so "do we need PBRS" IS that comparison. After it,
the mature checkpoint is read for a SEARCH dividend and for a teacher FOLD (untaught meter with
continuation control) before any v8-gift replication attempt. The era-boundary flip (win-prob critic as
the default, `MIGRATION_FLOOR` 109) is licensed once the SPARSE arm proves out; retiring HEAD-loadability
of the gen-era archive is accepted. Decisions are taken arm by arm from the data, not from a fixed batch.
[owner rulings · ledger 2026-09-06 · *OWNER RULINGS — the win-prob ladder's shape*]

### The eras

| era | what it is | status |
|---|---|---|
| **ai_v5 – ai_v7** | self-play / league, then specialisation + ladder. The chapters that ESTABLISHED the pattern — snapshot pool, win-rate gating, exploiters, the fold | closed; their mechanisms are live |
| **ai_v8** | the conditioning epoch on the flat-positional action head with a two-round physics `damage_refine_rounds` loop. **The one line that ever gifted** | closed; its code still runs from a pinned era checkout at `b13b30b2` |
| **ai_v9** | the entity-graph FRESH generation: pointer-NATIVE action head, no refine loop, physics as attention edge biases. Every generation gen-1…gen-17 | the incumbent line |
| **ai_v10 / ai_v11** | exploiter-SCALING and human-ladder-replay. **Nothing built**; ai_v11 is owner-PUNTED | open, unrun |
| **ai_v12** | clean-world / win-prob. `design_winprob_only_critic.md` is the design of record | **LIVE** |

The live ARM is **`ai_v12_02_winprob_critic`** (relaunched 2026-09-06, pinned `f971caf2`);
`ai_v12_01_winprob_critic` ran ~7 h on a STRIPPED architecture and is dead. `ARCHITECTURE.md` and
`designs/CLAUDE.md`'s state table both name it (the 2026-09-07 ⚠️ that they did not is discharged),
and gen-17 `ai_v9_21_gen17_pfspoff_0820` is what the §4/§5 measurements were taken on — the last
full GENERATION, not the live arm. Where a doc and the ledger disagree about what is running, the
ledger is later and wins.

### The runs that matter

🚨 **Every run below is a NAMED BASELINE** — `designs/baselines.json`
(`gen3_baselines_registry_v1`, 2026-09-06). Cite it by NAME and resolve it through
`agents.training.baselines`; never copy a path, and never re-point one by editing a module
(`python -m main.baselines set <name> <run>/<file>.zip --reason "<ledger title>"`, which
prints the ledger line to append). Each entry pins an EXPLICIT checkpoint, so the
last-snapshot rule cannot silently move what a name means.

| run | baseline NAME | what it is |
|---|---|---|
| `ai_v9_21_gen17_pfspoff_0820` | `production` | **the production ARCHITECTURE SURFACE** — `designs/production_config.json` is CONSTRUCTED from it (migrated v97 → v109 with a 13-key critic override block), and `ARCHITECTURE.md` + the delivery graph are generated from that mirror |
| `ai_v9_29_rev1_0823` (**rev-1**) | `v9_long_baseline` · `famine_comparator` · `untaught_meter_opponent` | the gen-era fresh from-zero run, final @25,067,760. The ORIGIN all eight R5F exploiters fork from, and the famine pre-test's comparator |
| `ai_v9_59_R2ACTION_0827` (**R2ACTION**) | `v9_fold_parent` | **the gen-era fold parent** — itself a `role=fold` fork of rev-1 at the same step, ending 28,115,184. Every rev-4 / 2×2 / K=6 / G1 / G5 arm forks from it |
| `ai_v8_04_distill_4teacher_0722` (**v8_04**) | `v8_parent` | v8's fold parent, 277,583,267 steps |
| `ai_v8_14` | `v8_line` | **v8's fold** — the one that gifted, +69 anchored ELO, ended `final_model_INTERRUPTED.zip` because it was stopped from outside |
| `ai_v12_01_winprob_critic` | *(the live arm — not a baseline)* | the live arm (§4) |

---

## 2. What we know about the flywheel

The week of 2026-09-01 → 09-06 is the densest evidence in the programme. What follows is the
current position, not its history.

### 2.1 v8's gift was the parent, not the fold

| cell (v8's line, era code + era meter, 3 arms each, ~1.08M steps) | untaught vs FROZEN parent |
|---|---|
| phase 1 — the full recipe (loss ON, 40% team bias, 3 stable opponents) | **+4.56pp [+1.14, +7.81]**, floor 3.22 |
| cell 1 — loss OFF, bias off, stable opponents ON | **+4.92pp [+1.63, +8.04]**, floor 3.71 |
| cell 2 — loss OFF, bias off, **no stable opponents** (plain continuation) | **+3.45pp [+0.46, +6.48]**, floor 2.54 |

- cell 2 − phase 1 = **−1.11pp [−3.12, +0.91]** ⇒ **EQUIVALENCE SUPPORTED** against the ±3.22 floor.
  Adding the whole fold machinery to a plain continuation is not detectably better than the
  continuation. [ledger 2026-09-06 · *CELL 2*]
- cell 2 − cell 1 = −1.46pp [−4.10, +1.01] ⇒ **INDETERMINATE**.
- Re-based on the continuation, v8's celebrated **+4.64pp becomes ≈ +1.2pp and is not significant**.
- ⚠️ At the era pin, `--distill-coef 0` disables the teacher-team BIAS as well as the loss, so cell
  1's reading is *"the loss AND its sampling bias, together, are not the carrier"* — those arms do
  not isolate the loss channel on v8's side.

**One arm would have been wrong two times in three.** Only one of phase 1's three arms clears zero
alone — the exact position v8's original +5.42 occupied for months. [ledger 2026-09-05 · *P1*]

### 2.2 Our parents do not gain from a continuation

**G5** (`ai_v9_195/196/197_G5PLAIN{A,B,C}_0906`) is the same cell on R2ACTION: **−1.92pp
[−3.98, +0.46]**, own replicate floor **1.00pp** ⇒ **NOT DETECTED**, point estimate NEGATIVE.
Beside it sit M9's two draws on rev-1 (−0.37 and −4.56). Three draws, two gen-era parents, two
depths — **no gen-era parent has been observed to gain from ordinary continued training.**
[ledger 2026-09-06 · *G5 RESULT*]

Two consequences:

- **The frozen-parent baseline STANDS on our side.** The meter-level alarm cell 2 raised is a true
  statement about `ai_v8_04` and a false one about `ai_v9_59`. On the strength of cell 2 alone the
  re-basing would have been propagated across the ledger and been wrong for our era.
- The re-based TC column moves folds the OPPOSITE way (every fold looks *better*), but the
  correction is not significant, **so the re-based column is NOT the reported number** — it is a
  sensitivity check that happens to be favourable.

**⇒ The cross-era difference is the PARENT.** *Maturity* is the shape of that result and one
**CONFOUNDED** candidate: 277M vs 28M steps, but also different architecture, obs dim, reward
composition and opponent ecology. Step counts are not commensurable across architectures with
different sample efficiency. [UNVERIFIED as a cause]

### 2.3 What a gen-era fold actually does

Measured over the 2×2 teacher-content batch (4 arms) and the K=6 dose cell (2 arms), all frozen-dose
(`--fork-lr 2.8e-5 --fork-lr-freeze`), parent R2ACTION, 8 untaught teams, cluster-bootstrapped:

| | taught (on-slice, endpoint) | untaught (off-slice, endpoint) |
|---|---|---|
| FUNDED teachers | **+5.11 [+2.66, +7.42]** | **−2.41 [−4.37, −0.63]** ROBS |
| UNFUNDED teachers | **+4.86 [+3.16, +6.64]** | +1.97 [−0.13, +4.09] NOT DETECTED |
| funded − unfunded | +0.25 [−2.03, +2.67] WITHIN FLOOR | **−4.37 [−5.78, −2.78]** SIGNIFICANT |

- **The fold TEACHES.** ~+5pp on-slice against the FROZEN parent, both halves equally, all four arms
  clearing zero individually. Re-based on a plain CONTINUATION — which itself gains +2.09pp
  [+0.47, +3.79] on this slice — it is **+2.6 to +3.4pp**, four of six arms clearing zero, and that
  column is an UPPER BOUND (the continuation is measured at fork+1.18M against the folds' ~+4.45M).
  The fold teaches; part of the headline number did not need it. [SIGNIFICANT, with the depth caveat
  · ledger 2026-09-06 · *G5 slice (iii)*]
- **Every fold digs an early off-slice HOLE**, ~3–4pp by +1M, regardless of teacher content (funded
  −3.12, unfunded −3.28, K=6 −4.19 — indistinguishable). Near-parent teachers let the student climb
  out (UNFUNDED − FUNDED recovery **+6.28 [+3.16, +9.81]** at p1M→mid); funded teachers show **no
  consistent recovery** (their two arms disagree with each other). [SIGNIFICANT]
- **Teacher funding is a pure LOSS**: the extra 1.0M/team buys nothing on-slice and costs 4.4pp
  off-slice.
- **DOSE is NULL.** Flat across a 4× range in the dose cell (K=12/6/3) and again between two frozen
  doses at fixed teachers (K=6 − K=3 spans zero or WITHIN FLOOR at all three depths). At v8's own
  dose (K=6, 1.06×) the fold is parent-neutral at the end, **−0.22 [−2.03, +1.81]** — **no gift.**
  [ledger 2026-09-05 · *K=6 CELL*]
- **The hole-then-recovery shape is a property of FOLDING**, replicated in two independent dose
  cells — the sign-flipped mirror of v8's transient hump.
- **Teacher budget/homogeneity is INVISIBLE at p1M and REAL at the end.** G1 (eight teachers pinned
  at exactly fork+1.20M, spread 0) − TCUNF pooled reads **+0.22 [−2.47, +2.34]** at p1M against a
  4.00 floor (EQUIVALENCE) and **−2.16 [−4.44, −0.25]** at END against a 1.19 floor (SIGNIFICANT).
  ⚠️ Confounded: G1's partner is a MIXTURE (six teachers at fork+2.93M, two at fork+0.93M), so this
  is *uniform-1.2M vs that mixture*, not "budget is irrelevant". [ledger 2026-09-06 · *G1 QUALIFIED*]

### 2.4 The channel: the loss, on our side

**C1** (the fold with `--distill-coef 0` and the 40% team bias still ON — the sampling axis isolated
for the first time) never robs and never gifts: +1.25 / +0.00 / +2.50 vs the parent at three depths,
every interval covering zero. B2, the same argv with the loss ON, robs −8.88 / −5.87 / −2.75.
**C1 − B2 = +10.12pp [+6.00, +14.31] at +1M** — decisive at any bar in the floor's interval;
supported but bar-uncertain at mid and end. ⇒ **the robbery travels through the distillation LOSS,
not through the data distribution.** [SIGNIFICANT at +1M · ledger 2026-09-03 ·
*C1 vs B2 COMPLETE*]

**And C1 shows NO GIFT either** — the loss carries BOTH directions: under gen-era conditions leak
only, under v8's parent gift − leak. ⚠️ B2 and C1 are **not dose-matched** (live KL controllers,
1.20× apart); every B2-vs-C1 claim carries that.

The **offline displacement** meter separates the dose axis (+0.1203 [+0.0663, +0.1795]) but **does
NOT** separate C1 from B2 (−0.0245 [−0.0841, +0.0267]) — an instrument limit, never a contradiction
of the win-rate result. [NOT DETECTED · meas: `reuse_batch_2026-09-03/offline_collateral_kl/`]

### 2.5 What a fold does to the weights

- **Displacement magnitude is set by DOSE, not by the distillation term** — C1 (loss OFF) has the
  second-largest |Δθ| of all 17 models scored.
- **`|Δθ| ∝ t^0.48`, replicate cosine 0.56** ⇒ a fold's Δθ is substantially a **RANDOM WALK**.
- **The off-slice KL is carried by the TRUNK**: encoders + team_transformer **51–74%** (≈90% with
  `projection_mlp`); pointer head **0.5–5.6%**; critic exactly 0.
- **Funded − unfunded off-slice KL: +0.0382 [−0.0067, +0.0928] NOT DETECTED** — the robbing half is
  not measurably farther from the parent off-slice than the neutral half.

⇒ **the robbery is not "more movement"; it is WHERE the movement points**, and a scalar KL is a norm,
not a direction. [meas: `arch_transfer_2026-09-05/fold_displacement/`]

### 2.6 The teachers

- **v8's teachers are LOCAL and DESCEND from their parent** (forked FROM v8_04). Sibling-control
  locality **R = 1.83 [1.53, 2.17]** on the files the fold actually loads. [SIGNIFICANT]
- **Ours are GLOBAL-from-origin SIBLINGS**: all eight R5F exploiters fork from rev-1's final, and
  R2ACTION is their `--exploiter` TARGET, not their origin — two 3.047M-step walks from one θ₀, with
  **KL(θ₀ ‖ parent) = 0.534 before any exploiter training**. Against the TRUE origin our R rises only
  to 1.25 / 1.20, so **the fork origin explains ~24% / 12% of the locality gap** — a quarter of the
  story, not the story. [meas: `content_locality_v2/`, `exploiter_drift/`]
- **Within the gen era, locality does NOT separate robbing from neutral teachers** (+0.0345
  [−0.0795, +0.1636]). What separates them is **MAGNITUDE** — funded teachers are farther from the
  parent everywhere. [NOT DETECTED / SIGNIFICANT respectively]
- **Teacher DISTANCE orders teacher SETS** (ρ −0.90, CI excludes zero) but is **NOT ESTABLISHED as a
  slope** (point-level slope [−35.5, +0.4] pp/nat spans zero) and is **CONFOUNDED with teacher
  budget** (ρ(budget, Δ) = −0.949 ≥ ρ(D_off, Δ)). "Distance causes the leak" and "longer-trained
  teachers cause the leak" are the same claim on this evidence. [meas: `teacher_distance/`]
- **The drift is FORGETTING.** Every exploiter set pilots the untaught 8 worse than its origin; the
  funded set by **−7.81pp [−15.23, −0.39]** (1.67× the floor), at an exchange rate of **−0.343pp
  untaught per pp of on-slice edge** ≈ 3pp of on-slice edge per 1pp of untaught win rate.
  The exploiter is **global from its FIRST checkpoint**, and its distance to the PARENT is flat for
  1.2M steps (0.534 → 0.522) before rising into the robbing bracket (0.660 unfunded → 0.755 funded).
  [SIGNIFICANT · meas: `exploiter_competence/`, `exploiter_drift/`]
- **A homogeneous short-budget teacher set folds NEUTRAL** where the long mixture ends ~+2pp above
  the parent (§2.3, G1 at END).

### 2.7 Ecology

On our side, **teachers-as-opponents is safe but is not a gift**: the loss × ecology 2×2 already
existed as the `fd` factorial — three gen-era folds ran in v8's exact shape (teachers == stable
opponents at 0.35), and `fdC` (coef 0 with teachers in the pool) returned **−1.2pp [−3.4, +1.0]**,
an EQUIVALENCE within the taught floor. On v8's parent the same cell returns **+4.92pp**.
[ledger 2026-09-06 · *CELL 1*, retraction paragraph]

`--stable-opponent-selfplay-share` is a share **of the self-play slice**, not of all episodes:
`p_stable = sf·s`. v8 ran 0.35 of self-play, the gen era runs 0.20 — neither is an absolute until
`sf` is named, and `sf` ramps. A mastered stable opponent silently leaves the challenge bucket.

### 2.8 The architecture account is CLOSED

Every leg, all NOT DETECTED or REFUTED [meas: `arch_transfer_2026-09-05/`]:

| leg | result |
|---|---|
| **H2 sharing kernel** — "the pointer head shares more" | **NOT DETECTED**, and the registered norm-share check fails for the story: the pointer head carries **0.66%** of the policy-gradient norm vs the flat head's 6.2%, 9.4× the wrong way |
| **H2b displacement** — where off-slice movement lives | the TRUNK, in both eras; the head 0.5–5.6% |
| **H1/H1b content locality** — "closed-form physics leaves teachers only team-specific content" | **REFUTED with the sign REVERSED**: the teachers that GIFTED are the LOCAL ones |
| **H7 FiLM z-swap** — v8's per-team code carries the specialisation | **REFUTED**: swapping the code removes **3.8%** of a v8 teacher's on-slice KL (rail 20%) and leaves locality untouched ⇒ "add a per-team FiLM code to the gen era" is not to be funded on this |
| **H4 search depth** | depth fired on 76% of searched decisions and re-ranked **+0.19%** of them (rail 20%) ⇒ **NOT DETECTED**; the binding constraint is the LEAF |
| **H9 continuation drift** — "young updates are noise, mature updates are drift" | **DEAD**: both parents walk as t^≈0.48 (perm p 0.70). Only the CRITIC group separates (0.29 vs 0.42 after removing PopArt's gradient-free rescale) ⇒ *the mature critic has largely stopped moving*, never "directed" |

---

## 3. Strength and meters

### 3.1 The head-to-head

**v8_14 beats v9_59 63.3% [59.3, 67.2]** over 559 decisive games (paired team draws from the
sha256-verified 32-team pool intersection, both orientations, both sides greedy, 0 timeouts) ⇒
implied ELO **+94.9 [+65.1, +124.7]** against a pre-registered +91.7. **SIGNIFICANT**, and the
anchored Bradley-Terry extrapolation is **CORROBORATED** — one successful out-of-sample test of the
ladder, not a validation of it as an instrument.

**Power, not disagreement**: the same-day bot calibration gives +57 [−29, +143] — an interval
containing the ladder's +91.7, the head-to-head's +94.9 **and zero**. 560 direct games pin the gap
to ±30; 720 bot games cannot establish which model is better.
[meas: `cross_era_head_to_head/`]

### 3.2 Reading an ELO — five rules

1. The headline is `<run>/snapshot_ladder/ladder.json` (dense, ±10), never `eval/elo` (±29).
2. A rating is only final once the run is — BT re-solves every node on every add and the newest is
   **systematically inflated** (gen-10's 12M fell 2089 → 2021 over 12 refits).
3. A cross-run comparison must be at **matched snapshot COUNT**, not matched step.
4. **New clause**: a bot-anchored gap between two models far ABOVE the anchors needs a direct match
   before it is quoted as a difference.
5. 🚨 **`win_rate_vs_pool` and `eval/elo` carry an OPPONENT-REGIME BOUNDARY at 2026-09-07** and are
   not comparable across it. From that date eval pool sentinels play GREEDY and draw the trainee's
   own teams by default (`gen3_eval_sentinel_greedy_default_v1`); before it the trainee played a
   temperature-1.0 sentinel drawing from a different team distribution, worth **+8.9 pp
   [+7.0, +10.7]** to the trainee on the SAME frozen pair the dense ladder plays symmetrically
   (60 paired pairs on `ai_v12_02_winprob_critic`). Equal skill therefore reads ~9 pp LOWER on the
   new regime, and the promotion gate moves 0.65 → 0.55 with it. The same flag was ON for 49 runs
   (v5.5–v8) and dropped UNRECORDED at the v9 launch, which is the boundary nobody marked the first
   time. **The regime is now RECORDED per run (`model_config.json` config v112) and per eval row
   (`sentinel_regime`), and is INHERITED on a flagless resume** — read it, never assume it.
   **`ladder.json` and every bot edge are UNAFFECTED**: the ladder always played greedy-vs-greedy
   with symmetric builders, and since `3e6875a5` it drops the eval sentinel edges from its fit
   entirely. Evidence: VERIFIED (ledger 2026-09-07 · *dense_reuse*, and the RECIPE CHANGE entry of
   the same date). Under the new regime the ladder also REUSES the pairs a cycle already measured
   (≈500 battles/promotion) — provenance rides in `games.jsonl` as `source: "eval_cycle"`.

### 3.3 The untaught meter

`python -m main.untaught_meter` — the win rate of a checkpoint **piloting** a fixed team slice
against ONE fixed opponent, cluster-bootstrapped over TEAMS. It is the number every fold verdict
rests on, and it lives in the tree rather than in per-batch copies.

- **A continuation control is REQUIRED**, not optional: `--control <arms…>` supplies it at matched
  depth, pooled with its own max-pairwise floor, and a run without it says in print what it is
  leaving out. (On our side the correction is currently NOT significant — §2.2 — so the
  frozen-parent column is what we report; the control is what proves that.)
- **Refs resolve as a launch resolves them** (§5, `gen3_last_snapshot_resolution_v1`) and the
  resolved file + rung are stamped in the JSON.
- **Reproducibility takes BOTH halves**: all five `$GEN3AI_*_SEED` seams AND `concurrency = 1`
  (seeded at concurrency 3, two runs of the same measurement differed by up to +0.043 in level).
  Concurrency > 1 is REFUSED.
- **Floors are OPERATIONAL, not pure-draw floors.** Every arm self-plays, so from its first
  promotion two "identical" arms have diverging opponent pools; the floor is the right bar for two
  arms as this programme actually runs them, and "pure draw floor" is withdrawn.
- **A floor is a property of the DEPTH, not of the meter.** End-depth replicate pairs agree to
  0.06/0.19/1.19pp; p1M pairs scatter by 2.56/1.13/4.00pp. Quoting one floor across depths makes a
  result either invisible or spurious.

Floors currently in evidence — always quoted with the regime they were measured in:

| floor | regime | source |
|---|---|---|
| 4.19pp | no-fold, one pair | the original bar |
| 4.27pp [+1.23, +6.92] | **controller-live** folds, pooled over 3 depths of ONE pair | N1/N2 |
| 1.66pp | **frozen-dose** folds, six draws | 2×2 |
| 2.46pp | K=6 cell's own three draws | K=6 |
| 4.00 (p1M) / 1.19 (END) | the TC replicate pairs, by depth | G1's bars |
| 3.22 / 3.71 / 2.54 | v8-era phase 1 / cell 1 / cell 2, max pairwise of 3 arms | the v8 line |
| 1.00pp | G5's three continuation arms | G5 |

### 3.4 The eval-trace quota is loss-enriched

The recorder PREFERS losses. The captured slice's outcome rate is 0.456/0.463 where the same
cycles' own `eval_results.jsonl` records 0.901 vs bots and 0.702 vs the pool. Read raw and you
conclude the win-prob head is grossly optimistic and, at 28M, worse than a base-rate coin — **both
readings are artifacts of the quota**:

| | raw quota | selection-reweighted |
|---|---|---|
| ECE (26M / 28M) | 0.237 / 0.281 | **0.025 / 0.035** |
| skill (26M / 28M) | 0.071 / −0.080 | **+0.336 / +0.265** |

Since `gen3_trace_selection_manifest_v1` each cycle's `eval_manifest.json` RECORDS the selection, and
`main.scaffolding_gauge --reliability-reweight`, `calibration` and `falsify-scan` read it off the
tree; a tree recording none is labelled SELECTION UNKNOWN rather than read as uniform.
[meas: `winprob_critic_baseline_2026-09-06/`]

### 3.5 The critic case file

- **RESOLUTION, not level.** Population-mean gaps are small and **sign-flip by ecology** (−0.065 vs
  bots, +0.058 vs pool); the disease is BLUR — within-decile true spread 0.11–0.36. Meter:
  `sd_true_excess`, floor-subtracted.
- **~39% of conviction-region blur is the IRREDUCIBLE hidden-information floor**, concentrated in
  ~10–20% of states. No value head removes it; quote effects on the EXCESS.
- **Bias, not variance** — 32× averaging is flat, so search amplifies a SYSTEMATIC leaf error.
- **Room exists**: critic rank re-expands under richer targets; residuals are sub-Gaussian, so the
  distributional lever is dead and the parameterization is already adequate.
- The head's **ecology split reproduces**: the pool class carries 2.4×/8.4× the bot class's
  reliability error. A pooled number describes neither.
- ⚠️ **On the raw slice an affine map of the SHAPED critic out-predicts the win-prob head**
  (Brier 0.1835 vs 0.2304 at 26M). **UNVERIFIED whether this survives reweighting** — the affine
  gauge has no weighted form.
  [`critic_calibration_plan.md` §0; `winprob_critic_baseline_2026-09-06/`]

### 3.6 The truncation defect

**This env never truncates in SB3's sense** — both `terminated` and `truncated` require
`battle.finished`, so "truncated" here has always meant *finished, not by a wipe*: the 250-turn cap,
a forfeit, or a tie. `gen3_env` forfeits at the cap, Showdown answers `|win|<opp>`, poke-env's
`calc_term_trunc` tests "was exactly one side wiped", a forfeit leaves six alive a side, so
`TimeLimit.truncated` reaches `ppo_mask` and `r += γ·V(s_last)`.

⇒ **under the shaped critic a cap forfeit and a tie bootstrap `0.9999·V(s_last)` on top of a
terminal reward that already paid the penalty — on every run to date.** That behaviour is kept
byte-identical and is now STATED rather than silent; only `--critic winprob` relabels them terminal
(`gen3_winprob_truncation_v1`, `e798c13a`; measured live with the cap lowered to 6).

**This is a CANDIDATE mechanism for the optimistic-V-at-the-cap pathology (a positive V on the final
decision in 13 of 14 timeout losses, the defect the deadline clock was built for), NOT a proof.**
No arm has isolated it. [UNVERIFIED as the cause]

---

## 4. The win-prob critic era — what is true now

### 4.1 The design's end state

`--critic winprob` (default is `shaped`, and a flagless run is byte-identical). Promoted, the mode is:

- **`V(s) = σ(win logit) ∈ [0,1]`**, and the value loss IS that head's BCE against the terminal
  outcome, weighted by `--vf-coef` (one critic, one coefficient).
- **The reward stream is the terminal WIN INDICATOR alone**: `+victory_value` on a win, `0.0` on a
  loss, a tie AND a 250-turn timeout alike. At `--victory-value 1.0` and `--gamma 1.0` the
  undiscounted return is exactly `1{win}`, so **`V(s) = P(win | s)` with no approximation term** —
  the identity the mode rests on.
- **PopArt is REFUSED** (a bounded stationary Bernoulli payoff has no scale to track), the
  distributional head leaves, and the shaped-return currency is deleted. The BCE joins the `value`
  noise-scale group.
- **The critic's target stops being bootstrapped** — an MC outcome label, not `returns = advantages
  + values`. GAE still bootstraps for the ADVANTAGE. **This is the single biggest open risk in the
  design and it is argued, not measured.** [UNVERIFIED]

Corrections that landed with it, and are what is true now rather than what the design first said:

1. **The label is the WIN INDICATOR, not ±1.** A draw is a not-win (`y = 0`) by explicit decision.
2. **Timeouts, forfeits and ties are TERMINAL** under this mode (§3.6) — the fix that lets the
   critic SEE a timeout at all. Registered expectation: timeouts FALL.
3. **Frozen-φ is declared but HELD.** `--win-prob-pbrs-frozen` exists in the mode's shape (a path,
   no coefficient) and is refused-as-deferred; it is the SELF-φ / FROZEN-φ ladder's later rung.

🚨 **A critic bounded in [0,1] cannot represent "a timeout is worse than a loss."** The −35 < −30
ordering is *unrepresentable*, so `--draw-penalty` is REFUSED. Anti-stall pressure comes from the obs
deadline clock plus `--arm-no-progress-tax`. **Stall rate and mean episode length are PRIMARY
endpoints, not monitored ones.**

### 4.2 The first arm and its pre-registered read

`ai_v12_02_winprob_critic`, fresh, pinned `f971caf2`, `--steps 75000000` (~33 h: **601 fps** once the
self-play pool is non-empty, 919 fps pool-empty before the first promotion at 4M — a 1.53× step, the
cost of a neural opponent, not a per-promotion tax; ~312 s per 3 h restart; 75M ETA Tue 08 Sep
08:00–09:00 [ledger 2026-09-06 · *6M read* and *RESTART READ RULINGS*]), relaunched 2026-09-06 20:27 on the PRODUCTION architecture
surface — 49 derived toggles diffed against `production_config.json`, 0 differing. Its predecessor
`ai_v12_01_winprob_critic` ran ~7 GPU-hours with 31 architecture flags at their OFF defaults and is
DEAD and not evidence [ledger 2026-09-06 · *INCIDENT — ai_v12_01 ran ~7h on a stripped
architecture*]; the first `ai_v12_02` launch at `--batch-size 4096 --grad-accum-steps 16` OOMed at
iteration 1 inside the gradient-balance probe [ledger 2026-09-06 · *OOM — the win-prob batch
correction is 2048x32*]. The shape that fits the full surface on the 12 GB card is
**`--batch-size 2048 --grad-accum-steps 32`** — the same 65,536 effective batch and the exactly
identical gradient; the micro-batch sets the activation peak and the peak is a function of the
architecture surface, so a fit measured on the stripped arm was evidence about a different model.
In flight: ~7.3 GB at 87% utilisation, 0 OOM.

**The read is one command**: `python -m main.critic_gate <run> --parent models/ai_v9_59_R2ACTION_0827
--control <G5 arms>` — the anchored ladder at **matched SNAPSHOT COUNT** against the parent
CONTINUED, the §4.3 calibration gate with **RESOLUTION primary** and `bot`/`pool` never pooled, the
G7 stall kill condition, and `main.untaught_meter` with its continuation control. Every bar is READ
from `measurements/winprob_critic_baseline_2026-09-06/`, never hardcoded.

**The gate, as it now stands** (owner ruling 2026-09-06): **G1 resolution** is the primary endpoint
and must strictly beat the matched-stratum baseline. **G2 reliability / G3 ECE are PER-STRATUM
NON-INFERIORITY bars against the baseline's same-stratum value at the MATCHED checkpoint** — the
absolute numbers (≤0.005, ≤0.05) stay printed as ASPIRATIONAL and gate nothing, because **the
baseline itself breaches them on the `pool` stratum** (reliability 0.0064/0.0103, ECE 0.0667/0.0875),
so as first written the arm had to clear a bar its own predecessor never cleared. **G4 skill > 0**.
**G7 stall rate + episode length** is the KILL condition. **G5 / G6 / G8 are gaps and print as NOT
RUNNABLE on every report** — a gate with unrunnable criteria quietly becomes the runnable ones.

**The FAMINE PRE-TEST** (registered before any read): this arm is the SPARSE rung of the
SPARSE / SELF-φ / FROZEN-φ ladder and went out at generation scale without the runbook's 5M
starvation pre-test, so the pre-test runs INSIDE it. Comparator **`ai_v9_29_rev1_0823`**;
**floor 38 Elo** — the max |Δ| between two same-class fresh runs at matched steps, **NOT** the
adjacent-node spread (172/186 Elo, which is steep early LEARNING and would pass an arm that learned
nothing). **At ~10M (amended 2026-09-06 — at 5M the full-surface arm has ONE snapshot, and a one-node
ladder is not a rating): trailing rev-1 by more than 38 Elo at matched snapshot count AND
`win_rate_vs_bots` not rising over ≥3 cycles ⇒ kill and launch FROZEN-φ. ~5M is a SMOKE of the read
(inputs resolve, comparator prints, composition guard 0, draw-rate bar ≥ 0.03 with rising episode length).**
[ledger 2026-09-06 · *READ AMENDMENT — the famine read moves to ~10M*] Pre-registered confound: the incumbent
had PBRS *and* PopArt *and* the shaped critic, so this is a rate comparison ACROSS RECIPES — a trail
inside 38 Elo is **not** evidence of equivalence, only that starvation was not demonstrated.

**THE n = 12 STRENGTH READ (2026-09-07 ~09:05, the strength half of D2):** at matched snapshot
count AND matched fit size (both 12-node fits, no refit) the arm **TRAILS rev-1 by 34 Elo against the
38 floor — WITHIN FLOOR**, margin 4, with 2M extra steps in the arm's favour; sparse is not
demonstrated inferior at this depth and not demonstrated equivalent [ledger 2026-09-07 · *THE n = 12
STRENGTH READ*; `measurements/winprob_critic_26M_n12_read_2026-09-07/`]. 26M is a new ladder peak
(2064.1; rev-1's 12th node 2098.4). Calibration G1 fails on every stratum and the BOT stratum's
resolution has decayed 0.030 → 0.014 since 14M while pool holds ~0.05; G2/G3 pass at 26M; G4 fails
on bots. Two dissociations between the fixed-bot axis and the self-play-lineage axis in five
snapshots, opposite signs, no account. **Reframed at 26M: the axis is FIXED-BOT vs POOL, not
strength vs calibration** — three instruments (eval episode length, G1 resolution, G4 skill) show
the arm's relationship to the fixed bots degrading or flat while its relationship to its own pool
holds or improves; the bot win-rate "decline" was noise (per-bot rows rebounded at 26M); the bot-side
calibration decay survives base-rate normalisation at ÷2.2 (skill) against the pool's ÷1.2 [ledger
2026-09-07 · *REFRAMING — FIXED-BOT vs POOL*]. UNVERIFIED account; the per-bot calibration rows at
28M are the registered discriminator.

**THE 10M READ (2026-09-07 ~00:40): NO KILL — the arm continues to 75M.** [ledger 2026-09-07 ·
*10M DECIDING READ*; artifacts `measurements/winprob_critic_10M_read_2026-09-07/`] At matched
snapshot count AND matched fit size the arm **TRAILS rev-1 by 30 Elo against the 38 floor** (WITHIN
FLOOR; the arm holds 2M more steps at its 4th node), and `win_rate_vs_bots` rose over all five
cycles 0.556 → 0.894 (SIGNIFICANT, safe direction). Neither half of the famine AND is met.
🚨 As shipped, `critic_gate` printed **"+64 lead"** — it compared the arm's 4-node fit against rev-1's
final 12-node fit, and rev-1's 8M node deflates 94 Elo between the two (rule 3 of *Reading an ELO*
in `src/agents/training/CLAUDE.md`); fixed in `462e9cdb`, which refits the longer ladder on its
first n. The Δ vs the fold parent (+159) is at UNMATCHED fit size (the fork's early nodes are rated
only through edges to its late selves) and is not a lead claim. **G1 resolution is BELOW its bar at
10M on every stratum** (`all` 0.0375 [0.0267, 0.0544] vs 0.0618; G2 also fails, G3 passes; the
falsification clause is not triggered because G2 fails too) — the bar is a 28M model's and the arm
is at 10M, a maturity asymmetry STATED and NOT used to re-base the bar; G1's registered read is the
75M gate. **The open finding, cause UNVERIFIED: strength and critic calibration are moving in
OPPOSITE directions on this arm** — it is winning at the shaped incumbent's pace without the
calibration the design predicted would carry it. **Candidate mechanism, from the run's own gradient
scalars [ledger 2026-09-07 · *RESTART 2 RULING*]: the POLICY gradient on the shared trunk tripled
after self-play began (0.56 → ~1.05) while the value gradient stayed flat (0.36 → 0.21–0.28), so the
value head's share of the trunk update fell 0.44 → 0.13, and the train value loss and Brier have been
FLAT since ~6M — the critic is riding a trunk the policy increasingly owns.** The vf_coef flag
(`grad/value_policy_logratio` −0.298 → −0.5006 → −0.638 across the three restarts) is CONFIRMED and
closed as a loop; arm B's `--vf-coef` (keep 0.5, or raise to restore the value share) is decided at
D2 on this evidence. G7 clean (stall 0.0000 except 4M 0.0062). The
registered ep_len clause 2 read MET (+4.12 turns) on a competence SAWTOOTH locked to the 2M pool
promotions with draw rate at a third of its bar — the AND with clause 1 is what kept a met clause
from reading as famine, and the +3.0 bar is banked as too tight for the next registration.

### 4.2b The 75M read (2026-09-08) — VERDICT [ledger 2026-09-08 · *D2 VERDICT*]

**The win-prob-only critic FAILED its primary bar.** [MEASURED, `23164e03` + `cbad491d`] At run end
(75,005,952 steps; last trace cycle 74.0M) the head's resolution sits BELOW the matched-stratum
gen-era baseline on 60 of 60 rows (all 0.0452 [0.0331, 0.0615] vs 0.0618), reliability and ECE
fail on the unhandicapped bot stratum, skill stays positive (+0.240 [+0.175, +0.294]). The identity
test explains the failure: V over-values positions against its own Monte-Carlo continuation by
+0.0965 [+0.0671, +0.1268], and the offset is late-game (late − early +0.167 [+0.086, +0.251]);
per-state error 0.255 is 2.6× the offset. The design's own falsification clause (§5.5) is exceeded.
**Sparse is WITHIN FLOOR on strength** on the one clean read (26M, −34 vs floor 38). The run-end
strength read is NOT COMPARABLE: pool grooming deleted the early snapshots and the tool slices the
survivors — a P1 tool defect (refit from `games.jsonl` over all rated steps). **Every committed A
ladder number is ~+73 high** (pre-fix updater folds sentinel edges; fixed-fit final node 1984.2,
peak 2000.2). **Open:** is the late optimism the γ = 1 terminal-label target or a moving-policy
target? Test 3 (bootstrap consistency) separates them. **Next arm per D2:** the frozen GEN-ERA head
as an actor-only potential on a fresh run (the D3 upper-bound arm), sequenced after arm C's first
read. Arm C (`ai_v12_04_pfsp_fork25M`, PFSP 2.5 one-sided, greedy sentinels) is LIVE on the GPU
since 2026-09-08 10:12, pinned at `ef981a89` [MEASURED, pin_history].

**The resolution failure has a named component: the head does not condition on the opponent.**
[MEASURED, `winprob_mixture_diagnostic_2026-09-09`] Over the last four trace cycles the
between-opponent spread of `V` on turn-1–3 states is 0.334× [0.318, 0.451] the between-opponent
spread of the outcome, where a calibrated critic's must be 1.0; over all states, 0.573
[0.509, 0.689]. Bias `V` − true win rate rises with opponent Elo at +0.0171 [+0.0131, +0.0209]
per 100 Elo (14 opponents; +0.0094, +0.0300 with opponents resampled), running −0.092 against
`random` to +0.106 against the strongest sentinel. The loss-preferring capture rate and the
greedy-sentinel handicap both work AGAINST the finding, and the effect is largest on early
states and reverses late. **A second conditioning failure, larger, was found in passing: the
trainee's OWN team explains ~9× more of the critic's residual than the opponent does**
(+0.0246 [+0.0202, +0.0385] vs +0.0027 [+0.0013, +0.0074], both in excess of a permutation
null). **Open:** whether feeding the value path an opponent-strength scalar closes any of it —
the D-ladder's conditioning arm, sketched in the measurement dir §12. **Caveat:** the bot-only
slope is NOT DETECTED once the nine pinned bots are themselves resampled; the detection rests
on the 14-opponent set and, more strongly, on the spread identity, which needs no strength axis.

**The conditioning failure is a HEAD failure, not a representation failure.**
[MEASURED, `winprob_probe_read_2026-09-09`] On turn-1 states of `ai_v12_02_winprob_critic` @74M,
`stash.value_pooled` — the tensor the win head reads, with `V = sigmoid(head(value_pooled))` —
linearly decodes the opponent's class at AUC 0.846 [0.811, 0.877], the trainee's team identity at
macro AUC 0.974 [0.966, 0.981], the team's leave-one-out win rate at R² 0.671 [0.527, 0.790] and the
opponent's Elo at R² 0.177 [0.112, 0.234], each clear of a permutation null run through the identical
pipeline. `V` itself reads 0.532 (inside its null), 0.631, 0.010 and 0.006 on the same four; every
paired V−pooled delta CI excludes zero. The raw-obs → value-path loss exists but is 4.3–8.7× smaller
than the value-path → V loss at that decision point. The ladder's own control substrate
(`ai_v12_11_ladder_ctrl10M` @10M) agrees, with `V` inside its null on opponent class at every bucket
past turn 1. A 64-unit MLP probe recovers MORE from `value_pooled` than the linear probe (own-team
win rate R² 0.746 vs 0.585), so non-linear coding is not the escape. **Consequence:** the
value-side opponent-CONDITIONING arm sketched in the mixture diagnostic's §12 targets a gap that is
not binding; the binding one is between `value_pooled` and the head's output, i.e. the TARGET the head
is trained on. Where `V` does clear its own null (opponent class and Elo from turn 3 on, team identity
throughout) it still sits far below the tensor it reads, every V−pooled delta CI excluding zero. **Open:** whether refitting the win head alone on a frozen `value_pooled` against a
conditional target (per-(opponent, team) empirical win rate, or an MC continuation) recovers the
between-opponent spread — an offline CPU test, not a GPU arm. **Caveat:** a proper scoring rule with
one 0/1 label per episode makes shrinking toward the marginal the variance-minimising response, so
"has it and does not use it" is a statement about what the head is PAID to do, not about what it can
represent.

---

**The head's failure is a TARGET failure, and the target's defect is the SHARE of the objective
the opponent holds.** [MEASURED, `winprob_head_refit_2026-09-09`] Refitting the win head alone on
a frozen `value_pooled`, out of fold under battle-grouped CV and HT-reweighted, against the
TERMINAL 0/1 outcome the online head actually trains on reproduces the online failure exactly on
both `ai_v12_02_winprob_critic` @74M and the ladder control `ai_v12_11_ladder_ctrl10M` @10M: the
turn-1–3 between-opponent spread ratio moves 0.149 → 0.000 (Δ [−0.055, +0.234]) and 0.066 → 0.068
(Δ [−0.044, +0.042]), and the prediction's turn-1 opponent-class decode moves 0.532 → 0.546
(Δ [−0.049, +0.081]). Swapping ONLY the target for the per-(cycle, opponent) × own-team
leave-one-battle-out win rate recovers a DETECTED part: 0.149 → 0.323 (Δ [+0.063, +0.296]) and
0.066 → 0.259 (Δ [+0.051, +0.192]), class AUC 0.532 → 0.646 (Δ [+0.059, +0.170]), own-team
win-rate R² 0.010 → 0.569 and 0.028 → 0.467 against `value_pooled`'s 0.671 / 0.628 — while
forecasting the real outcome BETTER on turns 1–3 (Brier 0.1608 → 0.1190, Δ [−0.0495, −0.0344]).
The mechanism is that only 10.2 % / 14.4 % of the terminal label's variance lies between (cycle,
opponent) cells, so a head minimising a proper scoring rule buys its resolution from the board and
its own team instead; the conditional target raises that share to 24.0 % / 58.8 %. **A head
initialised from the online weights lands where a scratch head lands, under both targets and on
both substrates**, so the online head is NOT in a basin and the head-side-optimisation treatment
class (value replay, periodic head refit, head-specific lr) is RULED OUT. Capacity is a real
second-order term: the MLP beats a linear head only under the conditional target. **The online
win-prob critic's turn-1–3 Brier skill against the base rate is −0.129 on arm A** — in the window
where the mixture defect lives it forecasts worse than a constant. **Consequence:** the treatment
is target-side (counterfactual / MC labels, search leaves, and above all opponent-stratified
weighting of the value loss, which raises the between-cell share with no new labels). **Open:**
whether an online arm with a lower-variance target moves the same two meters, and whether a head
that conditions plays better — the identity test against an MC continuation remains the arbiter.
**Caveat:** the recovery is PARTIAL (46 % / 29 % of the conditional target's own ceiling; 31 % / 23 % of the gap to it), and
the offline head saw ~2,500× fewer distinct episodes than the online one, so the size of the
recovery does not transfer even though the variance-share mechanism, being scale-free, does.

**Ladder arm 1 read (2026-09-08, `--vf-coef 1.5` vs the fresh 10M control):** [MEASURED · NOT DETECTED,
`measurements/critic_ladder_reads/vf15_vs_ctrl10M_2026-09-08/`] resolution Δ bot +0.0140 [−0.0080,
+0.0389], late identity bias Δ +0.0486 [−0.0305, +0.1311], turn-contrast Δ +0.1634 [−0.0807, +0.3733] —
none detected; the registered identity bias on ALL states reads +0.048 [+0.014, +0.082] MORE optimistic
than the control (no replicate floor yet). The fresh control reproduces A's 10M G1 failure on every
stratum (bot 0.0187 vs 0.0337). Tripling the critic's gradient share is not the large effect the
starvation reading predicts; the separate-value-trunk build is HELD pending the replicate floor.
Arm 3 (cf labels) refused itself at launch — duty cycle 6.2 % vs a 25 % floor at 48 envs — and is
amended to `--checkpoint-every-steps 500000` [ledger 2026-09-08 · *READ · critic ladder arm 1*].

**Ladder arm `tdaux` read (2026-09-09, `--td-aux-coef 1.0` vs the control, no span):** [MEASURED · NOT
DETECTED, `measurements/critic_ladder_reads/tdaux_vs_ctrl10M_2026-09-09/`] resolution Δ bot +0.0149
[−0.0045, +0.0343], late bias Δ −0.0151 [−0.0977, +0.0654], turn-contrast Δ +0.0092 [−0.2117, +0.2030];
calibration on points worse (G2/G3 fail on `all` and `bot`, n.d.). **The clock inversion (L4) is ABSENT at
10M** — control −0.094, arm −0.085 against arm A's +0.309 at 74M — so it is a late-training phenomenon
(formed by 50M) and no 10M read can see a lever's effect on it; the ladder's 10M reads stay valid for
resolution and for the head/mixture defect, both present at 10M. Two 10M lever arms now lean +0.014 /
+0.015 on bot resolution with CIs over zero; the replicate floor (`ctrl10M_b`, next) says whether that is
the control's shortfall.

**Ladder arm `cflabels` read (2026-09-09, cf continuation labels vs the control, pinned 10M):** [MEASURED · NOT
DETECTED, `measurements/critic_ladder_reads/cflabels_vs_ctrl10M_2026-09-09/` + `matched_quota/`] the own-team
decode delta +0.084 [+0.032, +0.183] first read as DETECTED was a DECODER-POWER ARTEFACT of the arm's 4× trace
quota (arm R² −0.025 on the battle-matched frame; a monotone frame-size curve) and is WITHDRAWN; every
conditioning row is NOT DETECTED, like `vf15` and `tdaux`; resolution nothing. The head reads unbiased against
its own continuation where the control is pessimistic (Δ +0.040 [+0.007, +0.071], a mean, not a fit — stands
vs zero, floor pending). The label is a state-conditional CONTINUATION label, so a null here is evidence about
the class only at 10M and at this power. Instrument consequence: the decoder-based rows must be read on
quota-matched frames — fix in flight before the `strata` and `truevalue` reads.

**THE REPLICATE FLOOR (2026-09-09, `ctrl10M_b` vs `ctrl10M`, pinned 10M, quota-matched):** [MEASURED,
`measurements/critic_ladder_reads/ctrl10M_b_vs_ctrl10M_2026-09-09_FLOOR/`, `replicate_floor_10M.json`] two
identical configurations differ by 0.070 [+0.036, +0.101] on the registered identity bias on the FIRST draw and
by 0.015 [−0.020, +0.049] on the second — a single draw's CI position is not a property of the quantity (the SIGN
of the 10M head's bias vs its own continuation is a draw; the 0.070 is one outlier head), 0.010 on bot resolution, 0.028 on the turn-1–3 spread
ratio, 0.39 on the all-states spread ratio, and **45 Elo** at matched count; the floor narrows with step
(bots 0.12 apart at 2M, 0.02 at 10M). Relabelled: every DETECTED-vs-zero on `vf15` and `cflabels` is WITHIN
FLOOR. **Standing: five runs, three levers, ZERO detected registered rows.** The floor is one draw — the
meters' CIs remain the inference; it can only demote.

**THE FLOOR AT 400 GAMES (2026-09-09, offline-generated cycles, 4,800 battles a side):** [MEASURED · MAJOR]
the replicate difference on bot resolution collapses to **+0.0004 [−0.007, +0.007]** (its 100-game floor was
battle noise — the row is now instrument-grade), while identity bias (+0.074 [+0.051, +0.096] on the `ctrl10M_b` draw; the `ctrl10M_c` draw reads +0.015 live —
the first draw is an outlier head) and the turn-1–3 spread ratio (−0.066 [−0.107, −0.031]) do not collapse with
more battles — run noise that more eval cannot remove, sized by the wider draw and typified by the narrower. Consequence: levers are read on bot resolution / own-team R² / class AUC from an
offline 400-game cycle (no retraining), and on identity bias / spread ratio only with replicates per condition.

**Ladder arm 7 `strata` read (2026-09-09, class-stratified win-prob BCE, realized bot share 0.473 vs 0.5):**
[MEASURED · NOT DETECTED, `measurements/critic_ladder_reads/strata_vs_ctrl10M_2026-09-09/`] nothing on any
registered row at 10M (bot resolution +0.005 [−0.012, +0.016], spread ratio t1–3 −0.049 n.d., own-team R²
−0.054 n.d.); six runs, four levers, zero detected registered rows; the 400-game read decides the slot.

**THE ARMS AT 400 GAMES (2026-09-09):** [MEASURED · NOT DETECTED at ±0.01] on offline 4,800-battle cycles both
sides, `vf15` / `tdaux` / `cflabels` move bot resolution by +0.003 / −0.002 / +0.001 with intervals inside about
±0.010 against a replicate pair agreeing to ±0.007, and the turn-1–3 spread ratio by less than the replicate
difference. The four 10M levers are dead at this length and power, not unresolved. No replicate is spent; the
next arm is a target-side lever with literature behind it — λ-return targets (`--win-prob-lambda 0.9`, arm 8,
in build), then KataGo-style dense auxiliary targets.

**THE FLOOR, SECOND DRAW (2026-09-09, `ctrl10M_c` vs `ctrl10M`):** [MEASURED] the replicate floor is a RANGE per row —
identity bias 0.015–0.070 (the first draw clear of zero, the second not; the three heads read −0.042 / +0.028 /
−0.027 against their own continuation, so the sign at 10M is a draw), bot resolution 0.001–0.010, spread ratio
t1–3 0.026–0.028, own-team R² 0.012–0.039; the bar is the wider draw, demote-only. Three draws of one
configuration span 45 Elo, 0.12 in early bot win rate, 0.21 in the first self-play jump and 0.93–1.07 on the G7
worst-ratio itself — none of those is a lever signal. Arm 8 (λ = 0.9) is live with `lambda_target_shift` 0.59 as
its dose meter.

### 4.3 What is UNVERIFIED in this era

- **`--vf-coef` was NOT retuned.** 0.5 multiplied an MSE/CE over 51 atoms; it now multiplies a BCE.
  The first-rollout scale banner is a **DEFECTIVE INSTRUMENT** — at epoch 1 of rollout 1 the clipped
  surrogate has ratio ≡ 1 and sits at its stationary point, so `|policy loss| ≈ 0` BY CONSTRUCTION
  and any ratio against it is inflated. **The decisive instrument is `grad/value_policy_logratio` at
  the first restart boundary.**
- **The SELF-φ deletion is by ARGUMENT, not by measurement.** Deleting PBRS costs speed, not
  correctness (invariance cuts both ways: if PBRS cannot change the optimum it cannot be why a run
  succeeds) — but **nothing in this tree has measured the dynamics cost**, and a 75M-step run is an
  expensive place to find out.
- **Maturity** as the cause of the era gap (§2.2) — the shape of a result, not a demonstrated cause.
- **Whether `win_head`'s architecture is right for a critic** — it was sized as a side readout
  (`LayerNorm → 128 → ReLU → 1`) and has never carried a gradient into `pi`.
- **Whether the MC-only target loses more to variance than it gains in correctness.**
- **The label is still self-referential** (outcomes under the current policy); promotion makes the
  loop tighter, not looser.
- ⚠️ **A provenance hazard, live**: `model_config.json` records `all_shaping_pbrs=True` while the
  child announces `1 TERMINAL + 0 PBRS + 0 BIAS`. Those flags are INERT under
  `--terminal-indicator`. **The announcer is the authority.**

---

## 5. Retired hypotheses

Each is retired for the flywheel's gift question. None is retired as a general claim about RL.

| hypothesis | retired by |
|---|---|
| **DOSE** — a gentler/heavier fold explains the sign | flat across K=12/6/3 (dose cell), and again K=6 vs K=3 at fixed teachers; **no gift at v8's own dose** [ledger 2026-09-05 · *K=6 CELL*] |
| **LENGTH** — v8's fold was 3.26× longer | interpolated at OUR fold length v8 was ~+8.5pp and it was +4.64 at +1.09M where every fold of ours digs a hole. Whatever differs, differs EARLY [ledger 2026-09-05 · *THE V8 LINE, pulled up*] |
| **TEAM COUNT / breadth-for-differentiation** | slope +0.0003 ± 0.0013 (z=0.23) across a 2/3/4/9-team ladder (probe A) |
| **BUDGET (per-team steps)** | +0.0019 z=0.16 between 1.5M and 2.5M per team; the budget law is dissolved, not merely unconfirmed (rev-3 admission) |
| **TEACHER QUALITY** | teachers converge to a set-mean **~0.6881 [0.672, 0.704]** ceiling, invariant to budget and to target strength; extraction is HEADROOM, not a teacher property |
| **ARCHITECTURE** (head kernel, trunk sharing, FiLM code, depth) | every leg NOT DETECTED or REFUTED — §2.8 [meas: `arch_transfer_2026-09-05/`] |
| **"Young updates are noise, mature updates are drift"** | both parents walk as t^≈0.48 under plain continuation, perm p 0.70 [meas: `continuation_drift/`] |
| **"Ecology carries a gift on our side"** | the `fd` factorial already answered it: `fdC` (coef 0, teachers as opponents) = −1.2pp, an EQUIVALENCE within the taught floor. The cell was promoted on a false census before the census corrected it [ledger 2026-09-06 · *CELL 1*, retraction] |
| **"Every untaught delta needs re-basing on a continuation"** | true of `ai_v8_04`, FALSE of `ai_v9_59`: G5 −1.92pp [−3.98, +0.46] [ledger 2026-09-06 · *G5 RESULT*] |
| **"Best-against-target" teacher selection** | `best_model/best_model.zip` is BOT-win-rate-selected and is not always the run's last checkpoint (2 of 8 R5F teachers were ~0.93M, not ~2.93M). **Owner ruling 2026-09-06: a bare run dir means the run's LAST SNAPSHOT** (`gen3_last_snapshot_resolution_v1`); `<run>@<step>` and an explicit `.zip` are used verbatim [ledger 2026-09-06 · *H8* and its G1 correction] |

**A retired hypothesis stays retired at the power it was retired with.** "F6-CURR curriculum: NULL
z=−1.40" rules out >4.5pp, not >0.

---

## 6. Open questions

| question | the test that would settle it | cost |
|---|---|---|
| **Does the win-prob critic RESOLVE better than the shaped one?** | `python -m main.critic_gate ai_v12_02_winprob_critic --parent ai_v9_59_R2ACTION_0827 --control <G5 arms>` — G1 is the primary endpoint | free once the arm reaches its eval cycles; the arm is ~14 h at the measured 5.2M steps/h (a FLOOR: `ep_len_mean` lengthens as play improves) |
| **Does terminal-only reward STARVE?** | the famine pre-test at ~5M against rev-1 at matched snapshot count, floor 38 Elo, AND-gated with `win_rate_vs_bots` rising | already inside the live arm |
| **Is MATURITY the cause of the era gap, or is it era?** | a gen-era parent trained to a comparable step count, then the same continuation cell. There is no cheap version — step counts are not commensurable across architectures | a multi-week generation; **not scheduled** |
| **Does a FOLD ON v8's PARENT with OUR teachers gift?** | the origin factorial's unrun half — 8 exploiters forked FROM R2ACTION in v8's recipe (`TC_ORIGIN`), then the fold ×3. A gift here makes the fork origin the whole story | 8 exploiter runs + 3 folds, ~30+ GPU-h |
| **Does a SHORT-budget teacher set (stopped at ~1.2M, before the drift rises) fold neutral or better?** | `TC_SHORT` ×3 — the TC_UNF recipe with the eight R5F exploiters at their existing ~1.2M checkpoints. No exploiter training needed. G1 is a partial, confounded answer (§2.3) | ~18 GPU-h for 3 arms |
| **Does the anti-stall pressure survive without the −35 ordering?** | G7 on the live arm: stall rate + `ep_len_mean` against the era. It is the KILL condition, not a monitor | free, in-flight |
| **Does the MC-only critic target cost more variance than it buys correctness?** | offline: compare the MC label against the bootstrapped return on existing traces and measure each one's variance | ~hours of CPU; **not done** |
| **What does `--no-hand-shaping` cost in SPEED?** | a short paired A/B at ~2M steps | ~2 GPU-h; would have priced the 75M run |
| **Does the affine-shaped-critic-beats-the-head result survive reweighting?** | give the affine gauge a weighted form and re-run on the committed baseline | ~a day of build; first item on the design's gap list |
| **Does an ANCHORED exploiter (KL-to-parent trust region on the teacher) hand the fold a local teacher?** | an anchored arm's untaught win rate should return toward the origin's 0.578 while its on-slice edge holds (~500 battles/teacher set). It is currently indistinguishable from "train exploiters less", which is FREE — so `TC_SHORT` is the honest first test | the meter is cheap; the arm is a build |

---

## 7. Standing rules of evidence

1. **PRE-REGISTER, before any number exists.** Both branches, the bar, and the comparator. The rule
   that catches fitting: state the PRIOR the result reads against, in the same entry.
2. **Three arms per cell.** One of P1's three arms cleared zero alone — **a single-arm study of that
   real effect had a two-in-three chance of reporting "no gift"**. Attach this to every one-arm
   reading.
3. **A floor is the MAX pairwise |Δ| over the replicates in hand, never the mean and never the
   smaller single-pair bar**, and it belongs to the REGIME (frozen vs controller-live) and to the
   DEPTH. A floor from few draws at one depth is close to uninformative — three separate readings in
   this programme were retracted for it.
4. **A continuation control at matched depth**, wherever a delta is taken against a frozen parent.
   Whether it bites is a measurement, not an assumption (it bites on v8's parent, not on ours).
5. **Matched SNAPSHOT COUNT** for any cross-run rating comparison, never matched step; and a
   bot-anchored gap between two models far above the anchors needs a direct match.
6. **Equivalence needs the DELTA's own CI inside the bar.** A bar against a POINT estimate is
   vacuous. Same for "the intervals overlap".
7. **VALIDATE BY EXECUTING, not by clause-checking.** `checkargs` has now been wrong in both
   directions; the launch path is `resolve_config`, and `python -m main.launcher --dry-run` is its
   executing complement (and the one that is safe on a same-run restart).
8. **Check the banked factorials before promoting a cell.** The `fd` 2×2 had already answered
   ecology on the gen side; a cell was promoted on a general claim made after checking ONE run.
9. **Quote a teacher, a parent or a target from the RESOLVED file, never from the argv** — a bare run
   dir is a directory, and a directory is not a file (§5).
10. **The team is the unit.** Cluster-bootstrap over TEAMS; state-weighted and team-weighted
    statistics have disagreed in SIGN on the same data (the Simpson class).
11. **Human-readable arm names.** Every code — `G1`, `C1`, `cell 2`, `H8`, `TC_UNF_A` — gets a
    description in the same sentence, every time it is used.
12. **A timeout is never a semantic outcome**, and neither is a draw. Separate buckets; a run above
    25% timeouts is INCONCLUSIVE.
13. **A result is not a result until every registered depth has landed.**
14. **Write the kills.** A finding that is not written here or in the ledger evaporates and gets
    re-believed.
15. **A windowed statistic never crosses an OPPONENT-REGIME boundary.** A regime is a constant
    `train/selfplay_fraction` (and by extension any step change in the opponent distribution). The
    tooling restricts the window to the current regime or reads NOT EVALUABLE, and prints the regime
    marker (current value, the step it changed at) beside every windowed number; a regime marker is
    reported as its current value plus change history, never as a central tendency. Four instruments
    on `ai_v12_02` were corrupted by the one 4.00M boundary in a single evening — the ep_len kill
    clause (amendments 3→4), the `selfplay_fraction` median (0.45, a value never occupied), the
    throughput diagnosis (a compile pause blamed for a regime change), and the vf_coef window (a
    "drift" that was the window sliding regimes). Ledger 2026-09-06 · STANDING RULE 15.
16. **An instrument that reads an artifact ANOTHER PROCESS IS STILL WRITING asks first whether it
    is complete, and refuses naming what is missing.** Five instances on the night of 2026-09-06/07:
    the ring-buffered child log, the events file mid-restart ("+2 s startup" = no restart in the
    data), the ladder mid-update (a correctly formatted verdict about the previous node), the
    compile lines mid-promotion, and the anchor pass under a policy that did not play the trace.
    Ledger 2026-09-07 · *THE n = 12 STRENGTH READ*.

---

17. **A statistic on the eval-trace tree is reweighted by each cycle's recorded capture rate, or it is not a measurement.** The tree is loss-enriched by design; selecting on the outcome breaks every outcome-conditional property (the martingale test, bias by V-level, overdispersion). Read raw, the 75M bootstrap-consistency test returned the opposite answer with a confident interval; reweighted by `eval_manifest.json`'s per-opponent `capture_rate_win/loss` it inverted and the calibration CLI's raw bias of +0.31 collapsed to ~0. A tree without a manifest is SELECTION UNKNOWN. [MEASURED, 73c929e1; ledger 2026-09-08 *test 3*]

## 8. Pointers

| what | where |
|---|---|
| the append-only record | [`ledger.md`](ledger.md) — cite an entry by its DATE + TITLE, or by its landing sha |
| FINDING an entry in it | [`ledger_index.md`](ledger_index.md) — generated, one line per heading (date · line · title). `python -m main.ledger_index --write` after any append; never hand-edit it |
| every number behind a claim | [`measurements/`](measurements/) — each artifact carries its checkpoint, step, state count and date |
| the week's campaign | [`measurements/arch_transfer_2026-09-05/`](measurements/arch_transfer_2026-09-05/) (H1–H9, the head-to-head), [`measurements/teacher_content_2x2_2026-09-04/`](measurements/teacher_content_2x2_2026-09-04/), [`measurements/reuse_batch_2026-09-03/`](measurements/reuse_batch_2026-09-03/) |
| the win-prob baseline the gate reads | [`measurements/winprob_critic_baseline_2026-09-06/`](measurements/winprob_critic_baseline_2026-09-06/) |
| the orientation, the frontier, the defect genres | [`README.md`](README.md) |
| the critic plan this era deviates from, on purpose · the ladder gap list | [`critic_calibration_plan.md`](critic_calibration_plan.md) · [`ladder_readiness.md`](ladder_readiness.md) |
| the design of record for the live era | [`../ai_v12/design_winprob_only_critic.md`](../ai_v12/design_winprob_only_critic.md) |
| why a fold on eight teams moves the other 711 · what the v8 campaign taught | [`../learning/negative_transfer_and_shared_functions.md`](../learning/negative_transfer_and_shared_functions.md) · [`../learning/distillation_flywheel_lessons.md`](../learning/distillation_flywheel_lessons.md) |
| the five-axis "the critic was wrong" taxonomy · bootstrap error propagation and the four critic-failure causes | [`../learning/win_prob_decomposition.md`](../learning/win_prob_decomposition.md) · [`../learning/credit_assignment_and_value_errors.md`](../learning/credit_assignment_and_value_errors.md) |
| population game theory, exploitability, PSRO | [`../learning/population_game_theory.md`](../learning/population_game_theory.md) |
| the tests-that-pass-without-asserting taxonomy | [`../learning/vacuous_tests_and_guards.md`](../learning/vacuous_tests_and_guards.md) |
| what the MODEL is now · how it got here | [`../ARCHITECTURE.md`](../ARCHITECTURE.md) · [`../CHANGELOG.md`](../CHANGELOG.md) |

**Instruments** (all offline, none needs a GPU): `python -m main.untaught_meter` ·
`python -m main.critic_gate` · `python -m main.elo` · `python -m main.exploitability` ·
`python -m main.scaffolding_gauge` · `python -m main.dose` · `python -m main.lineage` ·
`python -m main.capacity` · `python -m main.prober.query` · `python -m main.ledger_index`.
