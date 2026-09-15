# ARM S — the run-end reads, 2026-09-14

**Arm S** (`ai_v13_01_flywheel_shaped`) is the flywheel-era pair's **SHAPED** arm: the era's reward
and critic composition (`--critic shaped`, PopArt ON, a 51-atom distributional `E[Z]`, the win-prob
head kept as an AUXILIARY at `--win-prob-coef 0.05`) transplanted onto the current pin and the
current ladder ecology. It is **COMPLETE at 75,005,952 steps**, `final_model.zip`, 20 ladder nodes.
Its partner **arm W** (`ai_v13_02_flywheel_winprob`, the win-prob critic) was live on the GPU while
these reads ran and **is not touched or compared to anywhere in this note** — it has no ladder nodes
yet.

This note produces every run-end read the registration
([`designs/research_state/flywheel_era_pair_2026-09-12.md`](../../flywheel_era_pair_2026-09-12.md))
fixed **before either arm started**, so that the pair read is one diff away when W lands. Nothing
here is the pair's verdict.

**Everything ran CPU-only** (`CUDA_VISIBLE_DEVICES=""`), `nice`, from the main checkout, with
nothing written under `models/`. Two Showdown servers were started and stopped by their own
recorded PIDs on **:9417** and **:9450**; **:8000 and :8001 were never touched**.

---

## 0. The headline rows, in one table

| row | arm S | comparator, in its own regime | verdict under the registered rule |
|---|---|---|---|
| **ladder Elo**, 20 nodes, current-recipe refit | **2036.6** (se 8.9) at 72.0M; 2042.3 at 68.0M | the 0.02 leg refit the same way: **1984.2** | Δ **+52.4** [+28.3, +76.5] — **NOT DETECTED** (smallest claimable ≈ 69) |
| **H_end** (median-20, nats) | **1.0292** | `ent05` @10M **1.0861**; v8 plateau 1.07–1.11 | Δ −0.057 — **below the 0.074-nat floor, NOT READ** |
| **H over 4M→75M** | slope −3e-5 nats/M (t = −0.57) | the 0.02 leg: −6.9e-4 (t = −9.6) | arm S holds entropy FLAT for 71M steps; the 0.02 leg decays |
| **`cond.opp_class_auc.t4_10`** | **S-V 0.7683 / 0.7579**, **S-WP 0.7671 / 0.7494** (two draws) | levels only — the one 75M control tree is at the OTHER regime (0.7378 / 0.7444) | the delta is **REFUSED by the tool**, §4.1; the two columns agree to 0.001–0.009 |
| **untaught meter** | **54.50 pp** [52.25, 56.62], identical under both config resolutions | the 75M win-prob run **58.25 pp** | paired Δ **−3.75 pp** [−6.37, −1.06] — a run-level CANDIDATE, **descriptor not endpoint** |
| **vs Metamon `SmallRL`**, greedy-v-greedy | **0.630** [0.532, 0.718] | 75M win-prob run **0.520** [0.423, 0.615] | Δ +0.110 [−0.027, +0.241] — **NOT DETECTED** |
| **vs Metamon `SmallRL`**, ours greedy / theirs t1.0 | **0.840** [0.756, 0.899] | 75M win-prob run **0.742** [0.657, 0.812] | Δ +0.098 [−0.011, +0.202] — **NOT DETECTED** |
| **vs Foul Play** @ 1000 ms | **0.450** [0.346, 0.559] at **1.249 M** visits/decision | 75M win-prob run **0.388** [0.288, 0.497] at **1.40 M** | Δ +0.063 [−0.089, +0.210] — **NOT DETECTED**, and the widths are **unmatched in arm S's favour** |

🚨 **NOT DETECTED is never "equivalent"** (rule 6): equivalence needs the DELTA's own CI *inside* a
bar, and every bar in reach here is imported from another depth.

---

## 1. Provenance

| thing | value |
|---|---|
| arm | `models/ai_v13_01_flywheel_shaped`, **COMPLETE at 75,005,952**, `latest.txt` → `final_model.zip` |
| arm's pin | `6eb9c776940ed6040ddb90acfbc28b038457fc42` (the registration's pin; NOT bumped) |
| arm's `model_config.json` | `config_version` 119, `arch_signature` `gen3_critic_route_wave_v1`, `critic: shaped`, `use_popart: true`, `win_prob_mode: shaping` @ 0.05, `value_dist_mode: shaping`, `terminal_indicator: false`, `victory_value: 30.0`, `draw_penalty: −35.0`, `hand_shaping: true`, **`eval_sentinel_greedy: true`** |
| reads ran at | `dc02f847` + the `--v-column` patch in this branch |
| the 0.02 leg | `models/ai_v12_02_winprob_critic`, 75,005,952 steps, pin `f971caf2`, `config_version` 110 |
| the v8-line control | `models/ai_v8_03_zarch_control_0718`, 10 ladder nodes over 150.9M–248.0M |
| Showdown pin | `deps/pokemon-showdown` @ `e0551883f`; servers on **:9417** (Foul Play) and **:9450** (Metamon), each started and stopped by its recorded PID |
| box | 16 cores carrying the **live arm W** throughout; load average 9–37 across the session |

**Arm W was verified unharmed** mid-session: at 14:38 PT it was at 4,030,464 steps with
`train/selfplay_fraction` already at 0.84 since 4,000,032 — i.e. it had crossed into self-play at
the same step arm S did. Nothing in this note touched it.

---

## 2. STRENGTH — the ladder at 20 nodes

**Instrument:** `<run>/snapshot_ladder/ladder.json` (dense, ±10), never `eval/elo` (±29)
[§3.2 rule 1]. Every refit goes through `fit_ladder(run_dir, first_n=N, write=False)`; `first_n`
forces `write=False`, so nothing under `models/` was written.

### 2.1 🚨 THE HAZARD THAT WOULD HAVE FLIPPED THE SIGN — a committed `ladder.json` can be at a STALE RECIPE

The registration says to read the headline from `ladder.json`. On arm S that is exactly right. On
the 0.02 leg it is **wrong by +73.1 Elo**, and the direction is the one that matters:

| run | `ladder.json` computed | pairs in the committed fit | `eval_sentinel_edges_dropped` | committed newest | current-code refit, same nodes | committed − refit |
|---|---|---|---|---|---|---|
| **arm S** | 2026-09-14 | 190 of 190 possible | **51** | 2036.6 | **2036.6** | **0.0** |
| the 0.02 leg | 2026-09-08 | **494** of 190 possible | `null` | 2057.3 | **1984.2** | **+73.1** |
| the v8 control | 2026-07-22 | 45 of 45 | `null` | 2026.7 | 2022.3 | +4.4 |

The 0.02 leg's file predates `3e6875a5`, which **dropped the eval-sentinel edges from the fit**.
Those edges are a greedy trainee against a *stochastic* sentinel on an asymmetric teambuilder —
worth **+8.9 pp to the newer snapshot and +21..+29 Elo on the newest nodes** [§3.2 rule 5]. Its
committed file carries 494 pairs where only 190 frozen pairs exist, i.e. it is dominated by exactly
those edges.

**Quoting the two committed files against each other gives arm S − the 0.02 leg = −20.7 Elo.
Refitting both under the current code gives +52.4.** The sign flips. **Rule: when arm W lands, its
ladder will be written by current code, so any comparison to `ai_v12_02_winprob_critic` must use a
REFIT of that run, never its committed `ladder.json`.** The arm-S-vs-arm-W comparison is unaffected
(both are current-recipe) — this bites only the free third leg.

### 2.2 Arm S's nodes

20 nodes, **22,000,032 → 72,000,000**, per-node se **8.7–8.9**.

| step (M) | 22.0 | 24.0 | 26.0 | 28.0 | 30.0 | 32.0 | 34.0 | 36.0 | 38.0 | 40.0 |
|---|---|---|---|---|---|---|---|---|---|---|
| Elo | 1982.7 | 1967.8 | 2005.6 | 1991.3 | 1988.9 | 2016.2 | 2025.2 | 2020.9 | 2038.5 | 2020.2 |

| step (M) | 42.0 | 44.0 | 46.0 | 48.0 | 50.0 | 54.0 | 56.0 | 62.0 | **68.0** | **72.0** |
|---|---|---|---|---|---|---|---|---|---|---|
| Elo | 2040.0 | 2017.7 | 2025.6 | 2022.4 | 2019.6 | 2039.1 | 2041.9 | 2043.5 | **2042.3** | **2036.6** |

**Headline: 2036.6 at 72.0M (se 8.9).** The registered inflation cross-check — the **second-newest**
node [§3.2 rule 2] — reads **2042.3**, i.e. the newest node is *below* it here; there is no visible
newest-node inflation in this fit.

### 2.3 The node count and the resolution the registration computed

The registration derived `n = 20` by refitting the 0.02 leg at increasing `first_n`. **That table
reproduces here exactly**, which is the instrument validating itself:

| nodes | 0.02 leg: se / se(Δ) / CI95 (registration) | 0.02 leg, refit here | **arm S, refit here** |
|---|---|---|---|
| 4 | 15.70 / 22.20 / ±43.5 | 15.70 / 22.20 / ±43.5 | 16.4 / 23.19 / ±45.5 |
| 6 | 13.60 / 19.23 / ±37.7 | 13.60 / 19.23 / ±37.7 | 14.7 / 20.79 / ±40.7 |
| 8 | 12.50 / 17.68 / ±34.6 | 12.50 / 17.68 / ±34.6 | 13.1 / 18.53 / ±36.3 |
| 10 | 11.60 / 16.40 / ±32.2 | 11.60 / 16.40 / ±32.2 | 11.9 / 16.83 / ±33.0 |
| 12 | 10.70 / 15.13 / ±29.7 | 10.70 / 15.13 / ±29.7 | 11.0 / 15.56 / ±30.5 |
| 16 | 9.30 / 13.15 / ±25.8 | 9.30 / 13.15 / ±25.8 | 9.8 / 13.86 / ±27.2 |
| **20** | **8.50 / 12.02 / ±23.6** | **8.50 / 12.02 / ±23.6** | **8.9 / 12.59 / ±24.7** |

**arm S has n = 20 — the registered read's node count, and well clear of the n ≥ 12 floor below
which the registration says not to report at all.** On arm S's own se the pair's resolution is
se(Δ) **12.59**, CI95 **±24.7**, so the smallest claimable |Δ| against the **imported 45.0 Elo
floor** is **≈ 69.7 Elo**. (The floor is the MAX pairwise |Δ| over the three same-argv 10M controls
— 45.0 / 35.3 / 9.7 — and it belongs to a 10M four-node DEPTH, not to 75M. §9.1: there is no
run-to-run floor at 75M and none is affordable.)

### 2.4 The within-run late slope (§8.2 — registered, and NO bar attaches)

Fit on each run's own nodes with the **newest node dropped** [§3.2 rule 2]; se from the residual
spread. "Late" is the last third of the surviving nodes, "middle" the middle third.

| run | nodes used | middle third | late third | late − middle |
|---|---|---|---|---|
| **arm S** | 19 | **+0.02 ± 1.30** Elo/M (34–44M) | **+1.14 ± 0.44** Elo/M (48–68M) | +1.12 ± 1.37 — CI covers zero |
| the 0.02 leg | 19 | +1.33 ± 1.52 (48–58M) | +0.15 ± 1.23 (62–72M) | −1.18 ± 1.95 — CI covers zero |
| the v8 control | 9 | +0.91 ± 0.64 (192–208M) | +0.32 ± 0.11 (210–242M) | −0.59 ± 0.65 — CI covers zero |

**Arm S's late slope is +1.14 ± 0.44 Elo/M, ~2.6 se clear of zero: at 75M arm S is still gaining.**
The 0.02 leg's late slope is not distinguishable from flat (+0.15 ± 1.23), and the v8 line's is
small but clearly positive at a *quarter-billion* steps (+0.32 ± 0.11). Reported as a description —
the registration attaches no bar to this row, and the three runs are at wholly different depths.

🚨 **The v8 comparison is SHAPE ONLY** [§3.2 rule 4]: different architecture, obs dim, action-space
era, reward composition, clip range (0.10 vs 0.15) and ecology; 20 nodes over 22–72M against 10
nodes over 150.9–248.0M — a different depth *and* a different count; both far above the bot anchors,
so **a direct match is required before any gap is quoted as a difference**, and none was played.
The Elo row (arm S − v8 control = +14.3 at newest, +35.3 at second-newest) is in `strength_read.json`
for completeness and **must not be quoted as a difference**.

### 2.5 The within-run floor proxy (§9.1)

| run | adjacent-node \|Δ\| max | median | mean |
|---|---|---|---|
| arm S | 37.8 | 9.0 | 12.2 |
| the 0.02 leg | 29.3 | 11.6 | 12.6 |

This bounds **within-run wobble only**. It is dominated by early LEARNING and is **not** a
run-to-run floor — the registration rejects it as one explicitly, citing `famine_comparator`.

### 2.6 The cross-run row, with every confound named

| | arm S | the 0.02 leg | Δ | se(Δ) | CI95 | verdict |
|---|---|---|---|---|---|---|
| newest node | 2036.6 @72.0M | 1984.2 @74.0M | **+52.4** | 12.31 | [+28.3, +76.5] | **NOT DETECTED** (< 69.1) |
| second-newest | 2042.3 @68.0M | 1980.3 @72.0M | **+62.0** | 12.31 | [+37.9, +86.1] | **NOT DETECTED** (< 69.1) |

Matched snapshot **COUNT** (20 vs 20), which is the rule [§3.2 rule 3] — but the step spans differ
(22–72M vs 36–74M). **Confounds: different pin (`6eb9c776` vs `f971caf2`), different `--ent-coef`
(0.05 vs 0.02), different critic objective, and the stale-recipe trap of §2.1.** `ladder.json` is
UNAFFECTED by the 2026-09-07 opponent-regime boundary (the ladder always played greedy-vs-greedy
with symmetric builders); `eval/elo` and `win_rate_vs_pool` are NOT and are not read here.
**This is not the pair's read.** Both deltas point the same way and both sit under the registered
bar; under rule 22 a single-arm direction is a CANDIDATE at best, and this one does not even reach
that.

---

## 3. ENTROPY — `train/entropy_loss`, own span only (§8.3)

**Instrument:** the TENSORBOARD EVENTS via `main.ops.tb_read`, never the child log's table (which is
a rendering and undersamples the rollouts). 🚨 **SB3 logs `train/entropy_loss` as the NEGATIVE mean
policy entropy** — it is the loss term. Every H below is its negation, in nats, positive.

**The span is arm S's own, and was verified so.** `tb_read` reports 741 points over
**196,608 → 75,005,952** — one fresh run, no inherited prefix. The self-play crossing is at
**4,128,768** (first `*_pool` scalar; promotion logged at 4,000,032) — the same step as every
healthy arm in the family, and the boundary that rule 15 forbids a windowed statistic from crossing.

| run | H_start (median-20) | H_peak | **H_end (median-20)** | slope 4M→75M | late-third slope |
|---|---|---|---|---|---|
| **arm S** (`--ent-coef 0.05`) | 1.4679 | 1.6724 | **1.0292** | **−0.00003 ± 0.00005** (t = −0.57) | −0.00211 ± 0.00019 (t = −11.3), 51.7–75.0M |
| the 0.02 leg (`--ent-coef 0.02`) | 1.4950 | 1.6725 | **0.7132** | −0.00069 ± 0.00007 (t = −9.6) | −0.00034 ± 0.00024 (t = −1.4), 51.6–75.0M |
| `ent05` @10M (the registered comparator) | 1.5487 | 1.6762 | **1.0861** | −0.01415 ± 0.00274 | +0.03911 ± 0.00815, 8.4–10.0M |

`ent05`'s H_end reproduces its banked **1.0861** exactly — the instrument validating itself.

Arm S's five-million-step trajectory, median H per bucket:

| step (M) | 0 | 5 | 10 | 15 | 20 | 25 | 30 | 35 | 40 | 45 | 50 | 55 | 60 | 65 | 70 | 75 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **arm S** | 1.179 | 1.029 | 1.051 | 1.070 | 1.055 | 1.082 | 1.055 | 1.047 | 1.066 | 1.075 | 1.063 | 1.074 | 1.047 | 1.038 | 1.029 | 1.038 |
| the 0.02 leg | 1.270 | 0.800 | 0.741 | 0.729 | 0.700 | 0.733 | 0.734 | 0.751 | 0.747 | 0.730 | 0.723 | 0.737 | 0.733 | 0.716 | 0.718 | 0.690 |

### What this says, against the registered comparators

* **Against `ent05`'s 1.086:** arm S reads **1.0292**, a difference of **−0.057 nats — BELOW the
  0.074-nat three-seed replicate floor. NOT READ.** The prediction registered before launch (both
  arms sit near v8's plateau because `ent05` PASSED at this coefficient) is not contradicted.
* **Against v8's 1.07–1.11 plateau:** arm S's H_end sits **0.041 nats below the bottom of it** —
  also inside the 0.074 floor, so "outside the plateau" is **not a readable statement** either. Its
  5M-bucket medians are *inside* the plateau from 15M to 60M and drift to its lower edge after.
* 🚨 **The row that is not noise: arm S HOLDS entropy flat for 71M steps.** Over the whole
  post-crossing span its slope is **−3e-5 ± 5e-5 nats/M (t = −0.57)** — indistinguishable from zero
  across 71M steps — while the 0.02 leg falls at −6.9e-4 (t = −9.6) and sits 0.32 nats lower
  throughout (4.3× the floor). This is the first current-architecture arm long enough to ask the
  question, and 0.05 is the reason: `--ent-coef` is LIVE on a resume (`model_build.py:543`), unlike
  `--lr`, so 0.05 held across all 13 restarts.
* Arm S's **late third does decay** (−0.0021 nats/M, t = −11.3, over 51.7–75.0M) — a total of
  ~0.049 nats over 23M steps, itself below the floor as a level change. `ent05`'s late slope was
  *positive* at 10M; the two are at incomparable depths and the registration attaches no bar here.

**No PASS/FAIL attaches to H** (§8.3). And the arm-S-vs-arm-W entropy comparison, which the
registration says would be a FINDING rather than a bar, cannot be made yet — W has 4M steps.

---

## 4. THE CRITIC ROWS — read BOTH ways, and both LABELLED

### 4.1 🚨 The delta the registration asked for is REFUSED, by the tool, for two reasons

The registered control for a descriptive delta was `ai_v12_02_winprob_critic`'s own two 800-game
trees at 73M, and they **do exist** at
`/home/goodlad/.claude/jobs/9ab51de6/tmp/stepcurve/73121280/{draw1,draw2}`. They are **not usable as
a delta partner**, and `main.ops.critic_read` says so itself rather than returning a number:

```
REFUSING: both cycles are offline-generated, but to DIFFERENT specs — they are not two draws
from one population.
  eval_sentinel_greedy: arm=True  control=False
  n_games: arm=800  control=400
```

* **Step and regime, stated as asked.** The control trees are at step **73,121,280**, pin
  `f971caf2`, `config_version` 110 — a run that records **no** `eval_sentinel_greedy`, so the regime
  was **DECLARED on the command line** as the pre-2026-09-07 ASYMMETRIC one (stochastic sentinels,
  flat pool builder, worth ~+8.9 pp to the trainee). Arm S records `eval_sentinel_greedy: true` and
  its cycles are GREEDY + symmetric (source: **recorded**, not declared).
* **And the frames differ in size**: 400 games per opponent there, **800** here, which the
  registration itself specifies. A fitted conditioning row's expectation moves with frame size.

**So arm S's critic rows are reported as LEVELS ONLY**, exactly as the task's fallback specifies.
The 73M levels are printed beside them as *context*, never as a delta, in `out/control_73M_levels.json`.
The full transcript of the refusal is `out/critic_read_REFUSAL.txt`.

*Cost note:* a complete `critic_read` pair costs ~9,700 s (the banked 73M read's own
`elapsed_sec`), nearly all of it the `cf_audit` identity half, which is not one of this task's
registered rows. The levels below therefore come from **the same library functions `critic_read`
calls** — `conditioning_meters.conditioning_block` and `critic_read.gate_block` — so the numbers are
the tool's, computed without paying for a block nothing here reads.

### 4.2 The frames

Two independent offline full-capture draws of arm S at its **last evaluated step**:

| | draw1 | draw2 |
|---|---|---|
| seed | 20260910 | 20260911 |
| games per opponent | 800 | 800 |
| opponents | 9 scripted bots + 3 pool sentinels | same |
| sentinel steps | 22,000,032 / 42,000,000 / 72,000,000 | same |
| capture | **ALL** (no quota) | same |
| battles played / expected | 9,600 / 9,600, `complete: true`, shortfall 0 | (see `out/critic_levels.json`) |
| regime | `eval_sentinel_greedy: true`, source **recorded** | same |
| checkpoint | `eval_traces/step_74000016/snapshot.zip`, sha `1f08931fb04ad151` | same |
| reproducible | yes (seeded, concurrency 1) | yes |

🚨 **A DEVIATION FROM THE TASK, DECLARED.** The task asked for the draws "at its final step". Arm S's
final weights are at **75,005,952**, but the run has **no `eval_traces` cycle there** — its last is
`step_74000016` — and `eval_trace_gen` deliberately prefers the bit-exact
`eval_traces/step_<N>/snapshot.zip` that PLAYED the live cycle over a checkpoint that merely shares a
step. `@75005952` would have been refused ("no weights for … at step 75,005,952"). **The critic rows
are therefore at 74,000,016 — 1.3 % of the run short of the end — and the strength, entropy, untaught
and external-anchor rows are all at the true final model.** When arm W lands, read its critic rows at
*its* last evaluated step the same way, and check the two are at comparable distances from their ends.

### 4.3 The rows, BOTH WAYS, LABELLED

| | what it is on arm S |
|---|---|
| **S-V** — the npz's `values` | **the ACTUAL critic**: the distributional `E[Z]` in raw shaped-return units, PopArt ON, γ = 0.9999. **The treatment's own row.** |
| **S-WP** — the npz's `win_probs` | **the AUXILIARY win-prob head** at `--win-prob-coef 0.05` — a diagnostic, not the value function. **This is the column the v6 tool reads by DEFAULT.** |

🚨 **`max_abs_values_minus_winprobs` reads 75.12 (draw1) and 119.06 (draw2).** On the 75M win-prob
control the same scalar reads **0.0** — the two columns are one tensor there. **On arm S the large
value is not a defect; it is the treatment**, exactly as hazard H5 registered.

🚨 **AND ONLY THE RANK-BASED ROWS CAN BE READ BOTH WAYS.** `cond.opp_class_auc.*` is an AUC, invariant
to any monotone rescaling of V, so it is genuinely readable on either column. The calibration family
regresses on `logit(V)`, and every `gate.*` row is a Murphy decomposition of a PROBABILITY forecast —
neither is defined on a raw shaped-return column, and the gauge computes `gate.*` from `win_probs`
unconditionally anyway. **Those rows are S-WP only**, and they are marked so below rather than left
for a reader to infer. (This constraint is the one thing the registration did not anticipate; it is
now written into the new `--v-column` flag's own help.)

#### The decision row and its siblings

| row | **S-V** draw1 / draw2 | **S-WP** draw1 / draw2 | 73M win-prob trees *(context ONLY — other regime + frame)* |
|---|---|---|---|
| **`cond.opp_class_auc.t4_10`** | **0.7683** [0.7588, 0.7775] / **0.7579** [0.7478, 0.7679] | **0.7671** [0.7569, 0.7763] / **0.7494** [0.7391, 0.7595] | 0.7378 / 0.7444 |
| `cond.opp_class_auc.t1_3` | 0.5894 / 0.5667 | 0.5757 / 0.5539 | 0.5882 / 0.5872 |
| `cond.opp_class_auc.t1` | 0.4959 / 0.5085 | 0.4908 / 0.5128 | 0.5094 / 0.4924 |

**Turn 1 reads nothing (≈0.50) on both columns and in both draws** — the established shape; turns
4–10 are where the opponent is observable, which is why that is the registered decision row.

🚨 **The finding this pair of columns was registered to produce: on arm S the AUXILIARY head ranks
opponent class as well as the actual critic does.** S-V and S-WP differ by **+0.0012** on draw1 and
**+0.0085** on draw2 on `t4_10` — both inside each draw's own interval and far inside the imported
floors. A 0.05-coefficient diagnostic readout and a fully-weighted distributional critic carry the
same between-opponent information at the turns where it exists. **That is a level statement about
one arm, not a delta**, and it says nothing yet about arm W.

#### The calibration and gate rows — **S-WP ONLY** (see above)

| row | arm S draw1 / draw2 | 73M win-prob trees *(context ONLY)* |
|---|---|---|
| `gate.resolution.bot` | **0.01891** [0.0154, 0.0224] / **0.02129** [0.0173, 0.0262] | 0.01852 / 0.01711 |
| `gate.resolution.all` | **0.05505** [0.0514, 0.0594] / **0.05812** [0.0540, 0.0635] | 0.04170 / 0.03710 |
| `gate.resolution.pool` | 0.05877 / 0.05965 | 0.05190 / 0.04485 |
| `gate.ece.all` | **0.01777** [0.0131, 0.0232] / **0.01537** [0.0119, 0.0213] | 0.01698 / 0.01921 |
| `gate.ece.bot` | 0.07237 / 0.07386 | 0.04422 / 0.03485 |
| `gate.skill.bot` | 0.1219 / 0.14183 | 0.15327 / 0.14244 |
| `gate.reliability.all` | 0.00053 / 0.00048 | 0.00059 / 0.00053 |
| **calibration slope, all** | **1.3097** [1.246, 1.372] / **1.2743** [1.218, 1.336] | 1.0695 / 1.0632 |
| calibration intercept, all | 0.0448 [−0.033, 0.125] / 0.0697 [−0.006, 0.148] | 0.1194 / 0.1556 |
| calibration slope, t1_3 | 1.0080 / 0.9155 | 0.8384 / 0.8071 |

Descriptively — and **never as a delta**, since the control frame is at the other sentinel regime and
half the games (§4.1) — arm S's auxiliary head has a **calibration slope of ~1.29 against the
win-prob run's ~1.07**: an interval well clear of 1.0, i.e. the outcome moves *more* than
`logit(V)` does, so this head is **under-confident**. That is the expected shape for a readout
carrying 5 % of the value weight rather than being the value function, and it is the concrete sense
in which S-WP "is not arm S's critic".

#### The eval-DRAW spread — the floor proxy this read does have

Two independent draws of the same checkpoint (rule 19: the eval-draw component sits above sampling
noise, so **two draws BOUND a floor; they never give it a CI**). The imported v6 floors are
**0.0220** (first draw) and **0.0245** (second).

| row | S-V spread | S-WP spread | vs the imported floors |
|---|---|---|---|
| `cond.opp_class_auc.t4_10` | **0.0104** | **0.0177** | both **inside** |
| `cond.opp_class_auc.t1_3` | 0.0227 | 0.0218 | both **at** the 0.0220 floor — this sibling is the noisier row and should not carry a claim |
| `cond.opp_class_auc.t1` | 0.0126 | 0.0221 | at/inside |
| `gate.resolution.bot` | — | 0.0024 | far inside |
| `gate.resolution.all` | — | 0.0031 | far inside |
| `gate.ece.all` | — | 0.0024 | far inside |
| calibration slope, all | — | 0.0354 | — |

**The decision row is stable across draws on both columns**, which is what makes it usable as the
pair's decision row when arm W lands. The `t1_3` sibling is not, and the registration already flags
it as "the weaker sibling, ~⅓ the effect size".


---

## 5. THE UNTAUGHT METER at run end (§8.5)

`python -m main.untaught_meter` — the win rate of a checkpoint **piloting** a fixed 8-team slice
against ONE fixed opponent, cluster-bootstrapped over TEAMS. Offline, CPU, no server; **nothing was
written under `models/`** (`--json` / `--md` under the job tmp).

The opponent is the **registry name** `untaught_meter_opponent`, which the tool resolved and stamped:
`ai_v9_29_rev1_0823/snapshots/snapshot_000024000000.zip` @ 24,000,000 steps. 200 games/team, seed 0,
concurrency 1 (above 1 is REFUSED — seeded at concurrency 3 two runs of the same measurement have
differed by +0.043 in level). **3,200 battles, 0 timeouts** (rule 12's INCONCLUSIVE threshold is
25 %).

Both arms are read under **both** config resolutions, as the registration requires — neither is
architecture-identical to the meter's registry config, and *a level that only exists under one
resolution is not a level*.

### Levels — identical under BOTH config resolutions

| ref | config resolution | win rate | CI95 | wins / finished | timeouts |
|---|---|---:|---|---:|---:|
| **arm S** | registry `untaught_meter_config` (`ai_v9_29_rev1_0823`'s, v101) | **54.50 pp** | [52.25, 56.62] | 872 / 1600 | 0 |
| **arm S** | `--config auto` (its own v119 `model_config.json`) | **54.50 pp** | [52.25, 56.62] | 872 / 1600 | 0 |
| the 75M win-prob run | registry config | **58.25 pp** | [56.44, 60.44] | 932 / 1600 | 0 |
| the 75M win-prob run | `--config auto` (its own v110) | **58.25 pp** | [56.44, 60.44] | 932 / 1600 | 0 |

**The two resolutions agree to the last win.** The JSONs confirm the `auto` pass really did swap the
config (`config_path` per ref is each run's own file, and the registry pass's is `ai_v9_29`'s for
both), so this is a genuine robustness check and it passes: **this level is not an artefact of one
config resolution** — which is exactly what §8.5 asked for, and the condition it set for the level
counting as a level.

### The paired contrast (rule 10: the team is the unit)

`main.untaught_meter` publishes a delta only against a `--baseline`, and neither of these is the
other's parent, so the contrast is taken here from the meter's own per-team rows on ONE resampled
team index set (20,000 draws, paired — which removes the team-difficulty component the two refs
share):

**arm S − the 75M win-prob run = −3.75 pp, CI95 [−6.37, −1.06]**, identical under both config
resolutions, **7 of 8 teams favouring the win-prob run**.

| team | `U_61590463` | `U_92832108` | `U_ce35b736` | `U_9909f2e9` | `U_9d5f8458` | `U_f7ba5702` | `U_90b94599` | `U_dbf81d8e` |
|---|---|---|---|---|---|---|---|---|
| arm S | 56.50 | 49.50 | 56.00 | 54.50 | 55.50 | 55.00 | 49.50 | **59.50** |
| 0.02 leg | 61.00 | 59.00 | 64.50 | 55.50 | 57.50 | 57.00 | 55.00 | 56.50 |

**Reading, with the floors quoted with their regimes** (UNDERSTANDING §3.3 — *a floor is a property
of the DEPTH, not of the meter*): the magnitude clears the **1.19 pp END-depth replicate-pair
floor** and the CI excludes zero, but it sits well inside the **4.27 pp controller-live** floor and
is of the same order as the **1.66 pp frozen-dose** one. **Reported as a DESCRIPTOR, not an
endpoint**, exactly as the registration says — the meter's axes and floors are established at ~1M
fold depths, not at 75M fresh, and *neither of these arms is a fold*. Under rule 22 this is a
**run-level CANDIDATE at n = 1 per arm**, and it carries three confounds at once (pin, `--ent-coef`
0.05 vs 0.02, critic objective) — it is not attributable to the critic.

⚠️ **Direction worth noting because it cuts against the strength row:** on the ladder arm S is
+52.4 Elo over the 0.02 leg; on the untaught meter it is 3.75 pp *below* it. Both are single arms
and neither clears its registered bar, so the pair of signs is a reason to hold the strength row
loosely, not a finding.


---

## 6. EXTERNAL ANCHOR 1 — Metamon `SmallRL`

**Harness:** the matched-regime 2×2's scripts
(`designs/research_state/measurements/metamon_matched_regime_2026-09-14/`), with ONE addition — a
`mixed` regime, because the de-risk's banked `0.742` is a mixed-regime number and **a comparator must
be quoted in the regime it was measured in**. Metamon `@0a00a759`, conda env `metamon`, torch CPU
only, `VanillaAttention`, `SmallRL` ckpt 40 (13.9M). Our side is
`models/ai_v13_01_flywheel_shaped/final_model.zip` @ **75,005,952** (the true final model), CPU,
`main.play`'s own builders. Our own pinned Showdown on **:9450**, both sides drawing from OUR
719-team pool (the "home" set). 100 games per regime, split into two 50-game half-cells that differ
only in **who sends the challenge** (Showdown makes the challenger p1, and p1/p2 is not a priori
neutral).

| regime | what it is | n | W–L–T | **our win rate** | Wilson 95% | mean turns |
|---|---|---|---|---|---|---|
| **greedy** | BOTH sides greedy | 100 | 63–37–0 | **0.630** | [0.532, 0.718] | 31.9 |
| **mixed** | ours greedy, Metamon at ITS OWN eval default (t = 1.0) | 100 | 84–16–0 | **0.840** | [0.756, 0.899] | 36.2 |

Per half-cell (role balance): greedy 0.66 (Metamon challenges) / 0.60 (we challenge); mixed 0.90 /
0.78.

### Against the 75M win-prob run, each comparison inside its own regime

| regime | arm S | 75M win-prob run, SAME harness and regime | Δ (Newcombe 95%) | verdict |
|---|---|---|---|---|
| greedy | 0.630 [0.532, 0.718] | **0.520** [0.423, 0.615] (n=100, matched-regime 2×2) | **+0.110** [−0.027, +0.241] | **NOT DETECTED** |
| mixed | 0.840 [0.756, 0.899] | **0.742** [0.657, 0.812] (n=120, the de-risk) | **+0.098** [−0.011, +0.202] | **NOT DETECTED** |

Both point the same way — arm S ~10 pp above the win-prob run against the same external opponent —
and **neither difference interval excludes zero**. 🚨 *Never* "equal": rule 6 needs the delta's own
CI inside a bar, and no bar exists for this row.

🚨 **The two rows are NOT comparable to each other.** The 21-point gap between arm S's greedy row and
its mixed row is the *opponent's* regime moving, not ours (our side is greedy in both). The
matched-regime 2×2 already convicted "temperature 1.0" as **not a comparable setting across models**:
at t = 1.0 `SmallRL` plays its own argmax only **64.7 %** of the time, so playing it at its default
is playing a materially perturbed policy. This read reproduces that: our own instrument measured
`SmallRL`'s `argmax_match_rate` = **0.6586** in the mixed half-cell it recorded, inside the 2×2's
0.623–0.680 band.

### The regime was VERIFIED, not assumed

| half-cell | our `stochastic` kwarg | our temperature | Metamon `sample` kwarg | Metamon `argmax_match_rate` |
|---|---|---|---|---|
| greedy · Metamon challenges | `[false]` | 0.0 | `[false]` | **1.0000** |
| greedy · we challenge | `[false]` | 0.0 | `[false]` | **1.0000** |
| mixed · we challenge | `[false]` | 0.0 | `[true]` | **0.6586** |
| mixed · Metamon challenges | `[false]` | 0.0 | *n/a* — see hazard M1 | *n/a* |

The instrument has power: it reads 1.0000 **exactly** in greedy and materially below 1.0 in
sampling. Ties count in the denominator and not the numerator; **0 ties in 200 games**.

### Hazards (each a finding)

* **M1 — the `mixed · Metamon challenges` half-cell lost its Metamon-side record.** The Metamon
  process died *after its last game* with a `RecursionError` out of its own
  `metamon_to_amago.py::step` (988 frames), the de-risk's H-B shape. **Our side's 50 results are
  complete** (50 `GAME` lines, 0 connection errors) and the win rate above is ours, so the row
  stands; what is lost is that half-cell's per-decision regime check and its timing record.
* **M2 — and the positional join in that half-cell is therefore UNRELIABLE.** Metamon's per-battle
  CSV carries a random `Battle ID` with no join key to the Showdown room, so the two sides are
  matched **positionally**; in that half-cell `sides_disagree` = 7 and `metamon_row_missing` = 2,
  which is the signature of a drifted alignment, not of seven contested results. **Every other
  half-cell reads `sides_disagree` = 0.** Only OUR side's record is used for the win rate.
* **M3 — one 250-turn forfeit fired**, in that same half-cell (1 of 200 games, 0.5 %) — far under
  rule 12's 25 % INCONCLUSIVE threshold, and it is bucketed as a loss for us on our side's record.
* **M4 — two of our own tracebacks** appeared in that half-cell
  (`Decision context is missing at turn 1. get_mask() / embed_battle() must run before
  action_to_order()`), coincident with Metamon's crash storm. No game was lost (50/50 recorded).
  Worth a look if it recurs on a clean opponent.
* **M5 — the `production` registry name still cannot be used**, as both de-risks recorded: it
  resolves to `ai_v9_21_gen17_pfspoff_0820` @ `config_version` 97 and no longer loads at HEAD. The
  comparator here is an explicit `.zip`, never the registry name.
---

## 7. EXTERNAL ANCHOR 2 — Foul Play (search) at `--search-time-ms 1000`

**Harness:** the de-risk's scripts (`designs/research_state/measurements/foul_play_derisk_2026-09-14/scripts/`)
verbatim but for the port, the model and the usernames. Foul Play `@6c467c08` + poke-engine `0.0.48`
built `--features poke-engine/gen3 --no-default-features`, its own conda env, `--search-time-ms 1000
--search-parallelism 1 --search-threads 1`. Our side is arm S's `final_model.zip` @ **75,005,952**,
CPU, through `main.play`'s own client path, on our pinned Showdown at **:9417**. 8 sessions × 10
games, our side pinning one pool team per session (the same 8 the de-risk used, drawn nickname-free
and HP-IV-explicit), Foul Play drawing a random pool team every battle from the same export.

**Our win rate 0.450 (36/80), Wilson 95% [0.346, 0.559].** Mean 32.5 turns, median 29, max 92;
**0/80 reached the 250-turn forfeit**, 0 error lines on Foul Play's side, 0 games without a matched
Foul Play log.

| session | our team | games | our wins | win rate | mean turns | our think ms | FP think s | **FP visits/decision** |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | `009e3d0244` | 10 | 2 | 0.20 | 28.9 | 627 | 2.17 | 1.25 M |
| 1 | `4239fc5ba2` | 10 | 4 | 0.40 | 32.1 | 740 | 2.14 | 1.22 M |
| 2 | `64a691c473` | 10 | 5 | 0.50 | 35.6 | 513 | 2.13 | 1.65 M |
| 3 | `7d0337af97` | 10 | 7 | 0.70 | 31.6 | 528 | 2.03 | 1.19 M |
| 4 | `9b454d9ea7` | 10 | 1 | 0.10 | 40.4 | 425 | 2.19 | 1.05 M |
| 5 | `a185b2d193` | 10 | 6 | 0.60 | 27.1 | 111 | 1.94 | 1.36 M |
| 6 | `b904dbe059` | 10 | 6 | 0.60 | 31.3 | 471 | 2.10 | 1.15 M |
| 7 | `e11829f0561ef5a9` | 10 | 5 | 0.50 | 33.0 | 433 | 2.08 | 1.12 M |
| **all** | 8 pool teams | **80** | **36** | **0.450** | **32.5** | **481** | **2.10** | **1.25 M** |

### Against the 75M win-prob run, and why the point estimate must not be read alone

| | arm S | the 75M win-prob run (the de-risk, same harness) |
|---|---|---|
| our win rate | **0.450** [0.346, 0.559], n=80 | **0.388** [0.288, 0.497], n=80 |
| realized MCTS visits / decision | **1.249 M** | **1.40 M** |
| session-mean visit range | **1.05 M – 1.65 M** | 1.21 M – 1.53 M |
| our think time (mean) | 481 ms | ~37 ms |

Δ = **+0.063**, Newcombe 95% **[−0.089, +0.210]** — **NOT DETECTED**, and never "equal" (rule 6).

🚨 **AND THE WIDTHS ARE NOT MATCHED, IN THE DIRECTION THAT FLATTERS ARM S.** Foul Play has **NO
iteration budget** — `--search-time-ms` is wall clock only (`fp/search/main.py:53`), which makes it a
**WIDTH METER** under standing rule 23, and every read must carry its realized visit count. Arm S's
opponent searched **1.249 M** visits/decision against the de-risk's **1.40 M**, i.e. **~11 % narrower**,
because this campaign ran beside a live 75M training arm *and* two other CPU jobs of mine (load
average 26–37 throughout, against the de-risk's 22.6). Our own decision time moved 13× for the same
reason, which is the cleanest evidence that the box, not the model, set the clock. **So the +6.3 pp
is an upper bound on any real difference, and the honest statement is: at a 1000 ms nominal budget
arm S and the 75M win-prob arm are both BELOW Foul Play, by a margin this experiment cannot
separate.** The two external anchors continue to bracket our policies — Metamon's search-free
`SmallRL` below, Foul Play's search above.

---

## 8. HAZARDS — every one a finding

| # | hazard | why it matters, and what was done |
|---|---|---|
| **H-A** | 🚨 **A committed `ladder.json` can be at a STALE RECIPE, and the error is +73 Elo in the direction that matters.** `ai_v12_02_winprob_critic`'s file was computed 2026-09-08 with **494 pairs against 190 possible** and `eval_sentinel_edges_dropped: null` — before `3e6875a5` dropped the eval-sentinel edges from the fit. Its newest node reads 2057.3 committed against **1984.2** on a current-code refit of the same 20 nodes. | Quoting the two committed files gives arm S − the 0.02 leg = **−20.7**; refitting both gives **+52.4**. **The sign flips.** Every cross-run Elo in this note is a current-code refit, and **the pair read must refit `ai_v12_02` when arm W lands** — arm W's own ladder will be current-recipe, so a raw file-vs-file comparison would hand the 0.02 leg 73 free Elo. §2.1 |
| **H-B** | **The critic rows are at 74,000,016, not at the final 75,005,952.** Arm S has no `eval_traces` cycle at its final step, and `eval_trace_gen` prefers the bit-exact snapshot that PLAYED a live cycle over a checkpoint that merely shares a step; `@75005952` is refused outright. | Declared in §4.2. Strength, entropy, untaught and both external anchors are all at the true final model. When W lands, read its critic rows at ITS last evaluated step and check the two are comparably far from their ends. |
| **H-C** | **The registered critic CONTROL is unusable and the tool says so.** `main.ops.critic_read` REFUSES the 73M trees as a delta partner on two counts at once — `eval_sentinel_greedy: arm=True control=False` and `n_games: arm=800 control=400`. | Arm S's critic rows are LEVELS ONLY. The 73M levels are printed as context, never as a delta. §4.1, transcript in `out/critic_read_REFUSAL.txt`. |
| **H-D** | 🚨 **The registration's "read arm S BOTH ways" is only partly possible, and the limit is structural.** `cond.opp_class_auc.*` is an AUC and is invariant to any monotone rescaling of V, so it reads on `values` and `win_probs` alike. **The calibration family and every `gate.*` row are not.** They regress on `logit(V)` / Murphy-decompose a PROBABILITY forecast, which a raw shaped-return column is not — and `gate.*` is computed by the scaffolding gauge on `win_probs` unconditionally. | Reported: AUCs both ways and labelled; calibration and gate rows **S-WP only**, with the reason printed beside them. The new `--v-column` flag documents the constraint in its own help. §4.3 |
| **H-E** | **Foul Play is a WIDTH meter and this campaign was narrower than the de-risk's** — 1.249 M realized visits/decision against 1.40 M (−11 %), because the box carried the live arm W plus two other jobs of mine (load 26–37 vs the de-risk's 22.6). Our own decision time moved 13× (37 → 481 ms) for the same reason. | Rule 23. The +6.3 pp is an **upper bound**; the honest statement is that both arms are below Foul Play by a margin this experiment cannot separate. §7 |
| **M1–M5** | The Metamon half-cell crash (`RecursionError`, Metamon-side record lost), its consequently unreliable positional join (`sides_disagree` = 7 in that half-cell alone), one 250-turn forfeit, two of our own `Decision context is missing at turn 1` tracebacks, and the still-stale `production` registry name. | §6. Only OUR side's record feeds the win rate; every other half-cell reads `sides_disagree` = 0. |
| **(banked)** | The arm's crash-at-teardown shape — SIGTERM during the final eval's shutdown, after `Training complete` and after the full evaluation, the 4th occurrence. | **Already banked in the ledger of 2026-09-14; not re-investigated here.** |

### What was NOT done, and why

* **No `main.ops.critic_read` pair report was produced.** The one registered control refuses (H-C),
  and a complete pair costs **~9,700 s**, nearly all of it the `cf_audit` identity half — which is
  not among this task's registered rows. The levels in §4 come from **the same library functions
  `critic_read` calls**, so the numbers are the tool's.
* **No comparison to arm W anywhere.** It had ~4 M steps and zero ladder nodes while these ran.
* **No seed replicate of arm S.** There is none and none is affordable (~35 GPU-h). Rule 22 binds:
  every direction below is a CANDIDATE at best.

---

## 9. Ledger paragraph — ready to append (nothing in `ledger.md`, `UNDERSTANDING.md` or any design note was edited from here)

> ### 2026-09-14 · MEASUREMENT · ARM S's RUN-END READS — every row the flywheel-pair registration fixed before launch, taken on the SHAPED arm alone: 20 ladder nodes at **2036.6**, entropy **FLAT for 71M steps** at `--ent-coef 0.05`, the decision row read BOTH WAYS with the auxiliary head matching the actual critic, and 🚨 a **+73 Elo STALE-RECIPE trap** in `ai_v12_02_winprob_critic`'s committed `ladder.json` that would have flipped the sign of the free third leg
>
> `designs/research_state/measurements/flywheel_armS_reads_2026-09-14/`. `ai_v13_01_flywheel_shaped`
> (**arm S — the flywheel pair's SHAPED arm**, the era's reward/critic composition transplanted onto
> the current pin) COMPLETE at 75,005,952, pin `6eb9c776`, config v119, `eval_sentinel_greedy: true`.
> **Nothing here is the pair's read** — `ai_v13_02_flywheel_winprob` (**arm W — the win-prob critic**)
> was live at ~4M steps throughout and is not compared to anywhere; it was verified unharmed
> mid-session (crossed into self-play at 4,000,032, the same step as arm S's 4,128,768). Everything
> ran CPU-only, `nice`, from the main checkout, nothing written under `models/`; two Showdown
> servers on :9417 and :9450, started and stopped by their own PIDs, :8000/:8001 untouched.
> 🚨 **THE TRAP, and it is the most transferable thing here: a committed `ladder.json` can be at a
> STALE RECIPE.** `ai_v12_02_winprob_critic`'s file was computed 2026-09-08 with **494 pairs against
> 190 possible** and `eval_sentinel_edges_dropped: null` — before `3e6875a5` dropped the
> eval-sentinel edges (a greedy trainee vs a *stochastic* sentinel on an asymmetric builder, worth
> +8.9 pp to the newer snapshot and +21..+29 Elo on the newest nodes). Its newest node reads
> **2057.3** committed and **1984.2** on a current-code refit of the same 20 nodes: **+73.1 Elo of
> stale-recipe inflation.** Quoting the two committed files gives arm S − the 0.02 leg = **−20.7**;
> refitting both gives **+52.4**. *The sign flips.* Arm S's own committed file and its refit agree to
> **0.0** (computed 2026-09-14, 190 of 190 pairs, 51 sentinel edges dropped), and the v8 control's
> differ by 4.4. **When arm W lands its ladder will be current-recipe, so any comparison to
> `ai_v12_02` must REFIT that run.** **STRENGTH.** 20 nodes, 22.0M → 72.0M, per-node se 8.7–8.9;
> headline **2036.6** at 72.0M with the registered second-newest cross-check at **2042.3** (no
> visible newest-node inflation in this fit). The registration's node-count table **reproduces
> exactly** on a refit of the 0.02 leg (n=4 → se 15.70 / se(Δ) 22.20 / CI95 ±43.5; n=20 → 8.50 /
> 12.02 / ±23.6); on arm S's own se, n=20 gives se(Δ) **12.59**, CI95 **±24.7**, so the smallest
> claimable |Δ| against the imported 45.0 Elo floor is **≈ 70**. **n = 20 clears the registered
> n ≥ 12 floor.** Against the 0.02 leg (refit): **+52.4 [+28.3, +76.5]** at the newest node and
> **+62.0 [+37.9, +86.1]** at the second-newest — same sign, both under the bar, **NOT DETECTED**,
> never "equivalent" (rule 6), and confounded by pin, `--ent-coef` (0.05 vs 0.02) and the critic
> objective. **The within-run late slope (no bar attaches): arm S is +1.14 ± 0.44 Elo/M over
> 48–68M with the newest node dropped — ~2.6 se clear of zero, i.e. still gaining at 75M** — against
> the 0.02 leg's +0.15 ± 1.23 and the v8 control's +0.32 ± 0.11 at 210–242M; the v8 row is SHAPE
> ONLY (rule 4) and no Elo gap to it is quoted. **ENTROPY.** H = −`train/entropy_loss`, TB events,
> own span verified (741 points, 196,608 → 75,005,952, crossing 4,128,768). H_start (median-20)
> 1.4679, peak 1.6724, **H_end 1.0292** — **0.057 below `ent05`'s 1.0861 and 0.041 below v8's
> 1.07–1.11 plateau, BOTH inside the 0.074-nat replicate floor, so NOT READ in either direction**;
> the pre-registered prediction is not contradicted. 🚨 **The row that is not noise: arm S holds
> entropy FLAT across the whole post-crossing span — slope −3e-5 ± 5e-5 nats/M (t = −0.57) over
> 4M→75M** — while the 0.02 leg falls at −6.9e-4 (t = −9.6) and sits 0.32 nats lower throughout
> (4.3× the floor). `--ent-coef` is live on a resume, so 0.05 held across all 13 restarts. Arm S's
> last third does decay gently (−0.0021 nats/M, t = −11.3, 51.7–75.0M, ~0.049 nats total — itself
> under the floor as a level change). **CRITIC ROWS, BOTH WAYS, BOTH LABELLED.** Two independent
> offline full-capture draws, seeds 20260910 / 20260911, 800 games × 12 opponents = **9,600 battles
> each, complete, shortfall 0**, sentinels GREEDY (regime **recorded**, not declared), replayed from
> the bit-exact `eval_traces/step_74000016/snapshot.zip` — 🚨 **the critic rows are at 74,000,016,
> the last EVALUATED step, because the run has no eval cycle at its final 75,005,952 and
> `eval_trace_gen` refuses a step with no weights of its own**; every other row here is at the true
> final model. 🚨 **THE REGISTERED CONTROL IS UNUSABLE AND THE TOOL SAYS SO:**
> `main.ops.critic_read` REFUSES the existing 73M win-prob trees on two counts at once —
> `eval_sentinel_greedy: arm=True control=False` (those trees were generated at the **DECLARED
> pre-2026-09-07 asymmetric** regime, pin `f971caf2`, config v110, which records none) and
> `n_games: arm=800 control=400`. **So arm S's critic rows are LEVELS ONLY**, with the 73M levels
> printed beside them as context and never as a delta. `max_abs_values_minus_winprobs` reads
> **75.12 / 119.06** against **0.0** on the win-prob control — **large BY CONSTRUCTION, the
> treatment and not a defect** (hazard H5). **The decision row `cond.opp_class_auc.t4_10`: S-V (the
> ACTUAL critic, the distributional E[Z]) 0.7683 / 0.7579; S-WP (the AUXILIARY win-prob head at
> coef 0.05) 0.7671 / 0.7494** — the two columns differ by 0.0012 and 0.0085, inside every interval
> and far inside the imported 0.0220 / 0.0245 floors: **on arm S the 0.05-coefficient diagnostic
> ranks opponent class as well as the fully-weighted critic does.** Turn 1 reads ≈0.50 on both
> columns in both draws. Supporting rows (S-WP, see below): `gate.resolution.bot` 0.0189 / 0.0213,
> `gate.resolution.all` 0.0551 / 0.0581, `gate.ece.all` 0.0178 / 0.0154, calibration slope
> **1.310 / 1.274** against the win-prob trees' ~1.07 — i.e. the auxiliary head is
> **under-confident**, the expected shape for a 5 %-weight readout. Eval-draw spread on the decision
> row is **0.0104 (S-V) / 0.0177 (S-WP)**, both inside the floors, so the row is stable enough to
> carry the pair read; its `t1_3` sibling spreads 0.0227 / 0.0218, *at* the floor, and must not carry
> a claim. 🚨 **A CONSTRAINT THE REGISTRATION DID NOT ANTICIPATE:** only the RANK-based rows can be
> read both ways. The calibration family regresses on `logit(V)` and every `gate.*` row is a Murphy
> decomposition of a PROBABILITY forecast, neither of which is defined on a raw shaped-return
> column — and the scaffolding gauge computes `gate.*` from `win_probs` unconditionally. **Those rows
> are S-WP only and are labelled so.** The engineering item the registration named is built:
> **`--v-column {win_probs,values}`** on `main.ops.critic_read`, threaded into
> `conditioning_meters.conditioning_block`/`extract_cycle`, defaulting to `win_probs` so every banked
> read is byte-identical; 134 tests pass. **UNTAUGHT METER at run end**, registry opponent
> `untaught_meter_opponent` (= `ai_v9_29_rev1_0823@24,000,000`), 200 games/team over the untaught 8,
> concurrency 1, **3,200 battles, 0 timeouts**: arm S **54.50 pp [52.25, 56.62]** against the 75M
> win-prob run's **58.25 pp [56.44, 60.44]** — **IDENTICAL to the last win under BOTH config
> resolutions** (the registry `untaught_meter_config` and `--config auto`, whose JSONs confirm each
> ref really loaded against its own file), which is the condition §8.5 set for the level counting as
> a level. Paired team-clustered contrast **−3.75 pp [−6.37, −1.06]**, 7 of 8 teams — past the
> 1.19 pp END-depth replicate floor, well inside the 4.27 pp controller-live one, **reported as a
> DESCRIPTOR, not an endpoint**, and a run-level CANDIDATE at n=1 per arm carrying three confounds.
> ⚠️ **It points the opposite way to the strength row**, which is a reason to hold that row loosely.
> **EXTERNAL ANCHORS, each quoted inside its own regime and never across one.** vs Metamon `SmallRL`
> (ckpt 40, 13.9M, VanillaAttention, CPU) on our pinned Showdown with both sides drawing our 719-team
> pool, 100 games per regime as two role-balanced 50-game half-cells: **greedy-vs-greedy 0.630
> [0.532, 0.718]** against the matched-regime 2×2's **0.520** for the 75M win-prob run — Δ +0.110,
> Newcombe **[−0.027, +0.241], NOT DETECTED**; **ours greedy vs Metamon's own t = 1.0 default 0.840
> [0.756, 0.899]** against the de-risk's **0.742** — Δ +0.098 **[−0.011, +0.202], NOT DETECTED**.
> Regime VERIFIED per decision, not assumed: `argmax_match_rate` **1.0000** in both greedy half-cells
> and **0.6586** in the sampling one (inside the 2×2's 0.623–0.680 band for `SmallRL`), our
> `stochastic` kwarg `[false]` everywhere; 0 ties in 200 games. The 21-point gap between arm S's two
> rows is the OPPONENT's regime moving, not ours — the 2×2's finding that "temperature 1.0" is not a
> comparable setting across models, reproduced. vs **Foul Play** (`6c467c08` + poke-engine 0.0.48
> `--features gen3`) at `--search-time-ms 1000 --search-parallelism 1`, 80 games over the same 8
> pinned pool teams: **our win rate 0.450 [0.346, 0.559]** against the de-risk's **0.388** for the
> 75M win-prob run — Δ +0.063 **[−0.089, +0.210], NOT DETECTED**; 0/80 hit the 250-turn forfeit, 0
> error lines. 🚨 **AND THE WIDTHS ARE NOT MATCHED, in the direction that flatters arm S:** realized
> search was **1.249 M MCTS visits/decision (sessions 1.05–1.65 M)** against the de-risk's **1.40 M**
> — ~11 % narrower, because this campaign ran beside the live arm W plus two other CPU jobs (load
> 26–37 vs 22.6), and our own decision time moved 37 → 481 ms for the same reason. Under rule 23
> Foul Play is a WIDTH meter and every read carries its visit count, so **+6.3 pp is an upper
> bound**; the honest statement is that **both 75M arms sit BELOW Foul Play by a margin this
> experiment cannot separate**, and the two external anchors continue to bracket the policy.
> **Hazards, each a finding:** the stale-recipe ladder trap above; the 74,000,016-vs-75,005,952 step
> deviation, declared; the tool's double refusal of the registered critic control; the rank-vs-scale
> constraint on reading a shaped arm both ways; Foul Play's width; and on the Metamon side one
> half-cell lost its Metamon-side record to a `RecursionError` *after* its last game (the de-risk's
> H-B shape) making that half-cell's POSITIONAL join unreliable (`sides_disagree` 7 there, **0 in
> every other half-cell**) — only our own side's complete 50-game record feeds the win rate — plus
> one 250-turn forfeit in 200 games (0.5 %, far under rule 12's 25 %), two of our own
> `Decision context is missing at turn 1` tracebacks in that same half-cell, and the still-stale
> `production` registry name. The arm's crash-at-teardown shape (4th occurrence) is **already banked
> and was not re-investigated**. **No `critic_read` pair report was produced** — the one registered
> control refuses, and a complete pair costs ~9,700 s of which nearly all is the `cf_audit` identity
> half, which is not among these registered rows; the levels come from the same library functions
> `critic_read` itself calls. **Nothing is claimed about the critic objective:** every row above is
> arm S alone or arm S against a confounded third leg, n = 1 per arm, rule 22 binding. Tag:
> **MEASURED · arm S run-end reads COMPLETE · n=20 nodes · stale-recipe ladder trap (+73 Elo) ·
> entropy FLAT 71M · decision row BOTH WAYS, aux ≈ critic · anchors NOT DETECTED, widths unmatched ·
> the pair read is one diff away**.

---

## 10. What is in this directory

| path | what |
|---|---|
| `README.md` | this note |
| `scripts/strength_read.py` | the ladder read: committed vs current-code refit, the resolution curve, the slopes, the adjacent-node spread, the cross-run deltas |
| `scripts/entropy_read.py` | `train/entropy_loss` from the TB events, sign-corrected to H, median-20, the crossing, the slopes |
| `scripts/critic_levels.py` | the BOTH-WAYS critic read — `conditioning_block(v_column=…)` + `gate_block`, two draws, with the eval-draw spread |
| `scripts/untaught_delta.py` | the paired team-clustered contrast off the meter's own per-team rows |
| `scripts/metamon_cells.py` | the Metamon regime rows, Wilson + Newcombe, against the banked comparators |
| `scripts/harness/` | the run scripts as executed: the draw generator, the untaught driver, the Metamon cell runner (with the `mixed` regime added) and its arm-S plan, and the Foul Play session/campaign scripts |
| `out/strength_read.json` | every ladder number, all three runs |
| `out/entropy_read.json` | every entropy number, all three runs |
| `out/critic_levels.json` | S-V and S-WP levels on both draws, the frames, the eval-draw spread |
| `out/control_73M_levels.json` | the 73M win-prob trees' levels — **context only, a different regime and frame** |
| `out/critic_read_REFUSAL.txt` | `main.ops.critic_read`'s verbatim refusal of that pair |
| `out/untaught_delta.json`, `out/untaught/` | the meter under both config resolutions, and the paired contrast |
| `out/metamon/` | `armS_cells.json`, the merged `games.jsonl` (200 rows), the harness `summary.json` with the per-decision regime instruments |
| `out/foul_play/` | the merged `games.jsonl` (80 rows), `summary.json`, `session_table.md` |

### The one code change

`--v-column {win_probs,values}` on `main.ops.critic_read`, threaded into
`main.ops.conditioning_meters.conditioning_block` / `extract_cycle`. **The default is `win_probs`,
so every banked read is byte-identical**; the flag exists because a shaped-critic arm's two readouts
are different tensors, and the registration named building it as the engineering item before this
read. The docstrings carry the constraint of H-D. `conditioning_meters_test.py` and
`critic_read_test.py` pass (134 tests), as do the ruff / mypy / file-size / `CLAUDE.md`-freshness
gates.

