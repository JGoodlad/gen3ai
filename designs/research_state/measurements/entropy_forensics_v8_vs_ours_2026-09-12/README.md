# Entropy forensics — v8's line vs ours (2026-09-12)

**Question (owner, 2026-09-12).** v8's parent was "strong and STILL LEARNING" at 277M steps
(`UNDERSTANDING` §2.1: a plain 1M continuation gained **+3.45pp** untaught), while our gen-era
parents gain nothing from a continuation (§2.2: G5 **−1.92pp**, three draws). v8 ran
`--ent-coef 0.05 --lr 7e-5 --n-epochs 7`; everything since runs `--ent-coef 0.02 --lr 3e-4
--n-epochs 10`. **Owner's hypothesis: our policy's entropy collapses early, self-play goes stale,
and that is why our parents stop learning while v8's did not.**

**Zero GPU.** Everything here is a read of TensorBoard events, `snapshot_ladder/ladder.json`,
`metadata.json` and `snapshots/` mtimes, on CPU under `nice`. Nothing was written under `models/`.

---

## TL;DR — the verdict in five lines

1. **The entropy half of the hypothesis is SUPPORTED, and the separation is complete.** Every v8-line
   arm ends at **H = 1.068–1.110 nats** (0.44–0.46 × ln 11); every fresh arm of ours ends at
   **0.713–0.825**; the gen-era fold ends at **0.529**. The smallest cross-era gap is **0.243 nats**
   against a **0.0737-nat** replicate floor measured on three same-recipe seeds ⇒ **3.3× the floor,
   no overlap.** [SIGNIFICANT in magnitude; the CI is a within-era spread, not a paired delta]
2. **Our entropy is still falling where it is well powered; v8's was still RISING.** `ai_v12_02` over
   its last 19M steps: **−0.0011 ± 0.00034 nats/M (FALLING)**. `ai_v8_03` over its last 30M:
   **+0.0010 ± 0.00017 nats/M (RISING)**. Same statistic, n = 741 and n = 1178 rollouts, opposite sign.
3. **The STALENESS half is NOT SUPPORTED — it is the wrong way round.** Our runs promote a new
   self-play snapshot every **~2.1M** steps and end **1–4%** of the run behind their last promotion;
   `ai_v8_03` promoted every **~10.3M** steps and ended **20.4M steps / 17%** behind its last one.
   Whatever our parents' problem is, it is not a staler pool by this meter.
4. **The premise's dose arithmetic is WRONG by the `grad_accum_steps` factor, and the correction
   removes the claimed 6× gap for the fresh arms.** `python -m main.dose`: v8's line ran
   **2.14e-8 – 7.44e-8**; our two fresh win-prob arms run **3.18e-8 and 6.04e-8** — *inside* the v8
   line's own range. Only `ai_v9_59` (grad-accum 2) is the 6.6× outlier. **The dose is not the
   cross-era difference; the entropy coefficient still might be.**
5. **The ent-coef has NEVER been isolated, and it CANNOT be isolated from these logs**: across 212
   runs the archive holds exactly two values, and **no architecture carries both** — 0.05 appears
   only on v6–v8 signatures, 0.02 only on v9+. Perfect collinearity with the era.
   ⇒ **the causal claim is NOT TESTABLE offline. The cheapest test that isolates it costs ~4.3 GPU-h**
   (one fresh 10M arm; the 3-seed control is already banked).

**And one finding the question did not ask for, which cuts against its framing:** on the *dense
ladder* our era is not the plateaued one. `ai_v12_02` rises **+1.50 ± 0.21 Elo/M** over 36→72M;
`ai_v8_03` rises **+0.68 ± 0.22 Elo/M** over 151→242M. Neither shows a detectable late deceleration.
**The "our parents stop learning" result lives on the untaught meter and does not reproduce on the
ladder** — which means "our parents stop learning" and "our entropy is 0.35 nats lower" are two
true statements that this evidence does not join into one mechanism.

---

## 1. Method, and why the numbers are commensurable

**The tag, and its sign.** The project logs **`train/entropy_loss` = `-mean(entropy_per_decision)`
in NATS** (`src/agents/training/instrumented_ppo/ppo.py:415` — the standard *unweighted* metric,
computed before and independently of the defensive/bait entropy weights, both of which are `1.0` in
every run here). So throughout this note

> **H = −`train/entropy_loss`, in nats, over the 11-way MASKED categorical.**

Companions read from the same events: `train/approx_kl`, `train/clip_fraction`,
`train/explained_variance`, `train/learning_rate`, `train/grad_norm`, `train/clip_range`,
`train/selfplay_promoted_steps`, `eval/pool_snapshot_count`, `eval/win_rate_vs_pool`,
`eval/win_rate_vs_bots`, `hparams/ent_coef`, `train/n_epochs`.

**The reader.** `main.ops.tb_read.load()` and `main.ops.tb_read.selfplay_crossing_step()` (the
project's own readers over the event files — no hand-rolled event parsing), `main.ops.run_ref` for
run resolution, `main.lineage` and `main.tb_inherit --show` for the fork provenance, `main.dose` for
§4. `utils.paths.main_models_dir()` locates the archive from the worktree.

**Commensurability — is entropy comparable across the eras? YES in SCALE, with one caveat.**
`ACTION_SPACE_SIZE = 11` and the `[switch ×6, move ×4, struggle]` layout
(`src/agents/action/constants.py`) have been unchanged since `f02e633a` (2026-05-22), i.e. before
the earliest run here. `H_max = ln(11) = 2.3979` when all 11 are legal. What v51
(`gen3_pointer_native_v1`, `designs/CHANGELOG.md`) changed is the head's *parameterisation* — the
flat `Linear(latent, 11)` was deleted and the **pointer head** became the action head — not the
action space. Two consequences, both stated rather than assumed:

- the v9+ pointer scorers are **zero-init after SB3's ortho pass**, so a fresh run starts at exactly
  uniform-over-legal; the v8-era flat head was ortho-init and started slightly below uniform. This
  affects the very first rollouts only, and every v8 observation here is at ≥148M steps anyway.
- **the mask's mean SUPPORT is NOT controlled.** The number of legal actions is a function of the
  board and the ecology (faints, trapping, forced switches, team composition), which differ by era.
  So a **LEVEL** difference between eras carries an ecology confound — it is reported, and the
  well-powered readings that carry the verdict are **WITHIN-run slopes**, which that confound moves
  only through drift in the ecology itself.

**Two provenance hazards the script encodes** (both are findings in their own right, §7).

---

## 2. Part 1 — the entropy trajectories

`H_start` / `H_end` are medians over the first / last 2% of the run's **own** rollouts (a per-rollout
entropy is noisy; a median over a 2% window is not). `late` is an OLS slope over the last quarter of
the own segment; `mid` over the middle half. Verdicts are `slope ∓ 2·se` against zero.

| run | era | ent-coef *(TB)* | own rollouts | own step span (M) | **H_start** | **H_end** | H_end/H_start | **H_end/ln 11** | H_min (step) | late slope nats/M | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `ai_v8_01_zarch_film_0717` | v8 | 0.05 | 227 | 148.4 → 171.3 | 1.112 | **1.110** | 0.998 | 0.463 | 0.976 (148.4M) | −0.0020 ± 0.0011 | NOT DETECTED |
| `ai_v8_03_zarch_control_0718` | v8 | 0.05 | 1178 | 148.6 → 268.4 | 1.041 | **1.068** | 1.026 | 0.445 | 0.952 (258.7M) | **+0.0010 ± 0.00017** | **RISING** |
| `ai_v8_04_distill_4teacher_0722` | v8 | 0.05 | 90 | 268.7 → 277.6 | 1.180 | **1.100** | 0.932 | 0.459 | 1.084 (277.1M) | −0.0083 ± 0.0041 | FALLING |
| `ai_v8_14_distill3_0725` | v8 | 0.05 | 148 | 277.8 → 292.6 | 1.204 | **1.094** | 0.909 | 0.456 | 1.081 (287.3M) | −0.0022 ± 0.0016 | NOT DETECTED |
| `ai_v9_29_rev1_0823` | gen | 0.02 | 249 | 0.2 → 25.1 | 1.649 | **0.748** | 0.453 | 0.312 | 0.708 (24.5M) | −0.0083 ± 0.0010 | FALLING |
| `ai_v9_59_R2ACTION_0827` | gen | 0.02 | 30 | 25.3 → 28.0 | 0.675 | **0.529** | 0.783 | **0.221** | 0.515 (27.9M) | −0.0598 ± 0.0294 | FALLING |
| `ai_v12_02_winprob_critic` | v12 | 0.02 | 741 | 0.2 → 75.0 | 1.549 | **0.713** | 0.460 | 0.297 | 0.636 (20.3M) | **−0.0011 ± 0.00034** | **FALLING** |
| `ai_v12_11_ladder_ctrl10M` | v12 | 0.02 | 100 | 0.2 → 10.0 | 1.672 | **0.751** | 0.449 | 0.313 | 0.697 (7.8M) | +0.0027 ± 0.0081 | NOT DETECTED |
| `ai_v12_15_ladder_ctrl10M_b` | v12 | 0.02 | 100 | 0.2 → 10.0 | 1.665 | **0.825** | 0.495 | 0.344 | 0.761 (9.6M) | +0.0151 ± 0.0072 | RISING |
| `ai_v12_16_ladder_ctrl10M_c` | v12 | 0.02 | 100 | 0.2 → 10.0 | 1.671 | **0.772** | 0.462 | 0.322 | 0.728 (8.8M) | +0.0051 ± 0.0053 | NOT DETECTED |

`entropy_trajectories.png` (vs absolute step, log-x, and vs fraction-of-run) · `entropy_spines.png`
(the chained spines) · `entropy.json` (every field, plus the thinned curves).

### 2.1 The replicate floor — what makes this a reading and not a number

`ai_v12_11` / `_15_b` / `_16_c` are **seed replicates of one recipe**. Their endpoints give the
floor for free (`entropy_replicate_floor.json`):

| quantity | seed A | seed B | seed C | **floor = max pairwise \|Δ\|** |
|---|---|---|---|---|
| H_start | 1.672 | 1.665 | 1.671 | **0.0074 nats** |
| H_min | 0.697 | 0.761 | 0.728 | **0.0645 nats** |
| **H_end** | 0.751 | 0.825 | 0.772 | **0.0737 nats** |
| late slope (nats/M) | +0.0027 | +0.0151 | +0.0051 | 0.0124 — *the three seeds disagree in verdict* |

**Read against that floor:**

- v8-line `H_end` spans **1.068 – 1.110** (max internal spread **0.042**, *inside* the floor — the
  four v8 arms are one population); ours span **0.713 – 0.825**.
  **Smallest cross-era gap = 1.068 − 0.825 = 0.243 nats = 3.3× the floor. Largest = 0.397 = 5.4×.**
  Complete separation, and the two eras' internal spreads are each at or inside the floor.
- **`H_start` is a control that behaves**: the three seeds agree to 0.007 nats, so the instrument is
  not noisy — the era gap is not instrument scatter.
- 🚨 **the late SLOPE at 10M is NOT resolvable**: the three seeds read RISING / NOT DETECTED /
  NOT DETECTED. Any "still declining at the end?" claim at 10M is below this floor. The two slope
  readings that *are* well powered are `ai_v12_02` (n = 741, −0.0011 ± 0.00034) and `ai_v8_03`
  (n = 1178, +0.0010 ± 0.00017) — and those are the ones quoted in the TL;DR.

### 2.2 Shape, in words

Every fresh arm of ours follows the same curve with almost no seed-to-seed variation: start at
**~1.67** (uniform-over-legal, as the zero-init pointer head predicts), cross **0.9 × H₀ by
~1.1–1.6M**, **0.75 × by 1.8–3.1M**, **0.5 × by 4.0–8.6M**, and settle in a **0.71–0.83** band that it
then holds. `ai_v12_02` shows the band is not a floor: it dips to **0.636 at 20M**, recovers to ~0.73
mid-run (mid slope **+0.0006 ± 0.00014, RISING**), and then resumes falling to **0.713 at 75M**.

The v8 line, over the 144M steps we can observe (148.4M → 292.6M, four arms, two folds), never
leaves **0.95–1.22** and its per-arm endpoint band is **1.068–1.110**. It is a plateau, not a decline.
`entropy_spines.png` is the whole finding in one frame: five of our spines lie on top of each other
through the 1.67 → 0.75 decay and then run flat out to 75M, and the v8 spine sits as a separate,
higher band 0.3 nats above them at 150–293M with no visible trend.

⚠️ **The windows do not overlap.** v8 is observed only at **148–293M**; ours only at **0.2–75M**.
There is no matched-step comparison available, and v8's own early curve is unrecoverable (§7,
hazard 3). The honest form of the level claim is therefore: *our arms' entropy at 10M and at 75M is
0.24–0.40 nats below where v8's line sat throughout 148–293M, and v8's line was flat-to-rising over
that whole window while ours is still falling at 75M.*

---

## 3. Part 1b — the fork discontinuity (not asked for; it is the sharpest cross-era contrast here)

A fold resumes its parent's weights **and** optimizer state, so a step change in H at the fork is
attributable to the fold's objective and rollout distribution, not to re-initialisation
(`fork_entropy_discontinuity.json`):

| fold | era | parent H_end | fold H_start | **Δ at the fork** | fold H_end | fold end − parent end |
|---|---|---|---|---|---|---|
| `ai_v8_04_distill_4teacher_0722` | v8 | 1.068 | 1.180 | **+0.112** | 1.100 | **+0.032** |
| `ai_v8_14_distill3_0725` | v8 | 1.100 | 1.204 | **+0.104** | 1.094 | −0.005 |
| `ai_v9_59_R2ACTION_0827` | gen | 0.748 | 0.675 | **−0.072** | 0.529 | **−0.218** |

**Both v8 folds RAISE entropy by ~+0.11 nats and finish at or above their parent's level. Ours LOWERS
it by −0.07 and finishes 0.218 below.** Both Δs clear the 0.074 floor (the floor is on `H_end`, not on
a fork Δ — treat this as suggestive, n = 1 per fold).

Two candidate accounts, both already named elsewhere in the research state, and this note does not
choose between them:

- **TARGET FORM** (memory: *"TARGET FORM is the cause"*, and ledger 2026-08-28 *Probe D*). `ai_v9_59`
  ran `--distill-target action --distill-topk 1` — an **argmax cross-entropy**, a mode-sharpening
  objective that mechanically removes entropy. v8's folds at their pin ran the full-distribution KL,
  which copies the teacher's whole distribution — and v8's teachers were themselves sitting at
  ent-coef 0.05, i.e. at H ≈ 1.1–1.2. **A full-KL fold onto a high-entropy teacher pulls entropy UP;
  a top-1 CE fold pulls it DOWN.** This is a *mechanism* for the sign flip, and it is independent of
  the ent-coef.
- **Ecology**: `--distill-team-bias 0.4` puts 40% of episodes on unfamiliar teacher teams in *both*
  eras, so team bias is common to both and cannot be the sign flip.

---

## 4. Part 2 — late-slope of STRENGTH, within run

`<run>/snapshot_ladder/ladder.json` (dense, the §3.2 rule 1 headline), **own nodes only** (a fork's
`snapshots/` carries its parent's — §7 hazard 2), WLS on step with weights `1/se²`, and the
**newest node DROPPED** for the headline per §3.2 rule 2 (the newest BT node is systematically
inflated). `se` is the **wider** of the analytic and residual SEs; both understate, because BT
re-solves every rating on every add so the nodes are not independent. `ladder_slopes.json` carries
both variants and every node.

| run | own nodes | span (M) | total ΔElo | full slope Elo/M | **late slope** | late window | middle slope | **late − middle** |
|---|---|---|---|---|---|---|---|---|
| `ai_v8_03_zarch_control_0718` | 10 (→9) | 150.9 → 242.0 | +40.9 | **+0.68 ± 0.22 RISING** | **+0.32 ± 0.26** NOT DETECTED | 210→242M | +0.93 ± 0.65 ND | −0.60 ± 0.70 **ND** |
| `ai_v8_14_distill3_0725` | 2 | 278 → 292 | +33.5 | **REFUSED** — 2 nodes | — | — | — | — |
| `ai_v9_29_rev1_0823` | 12 (→11) | 2.0 → 22.0 | +390.7 | **+15.89 ± 3.01 RISING** | **+2.97 ± 2.27** NOT DETECTED | 16→22M | +14.04 ± 3.92 RISING | −11.07 ± 4.53 **FALLING** |
| `ai_v9_59_R2ACTION_0827` | 2 (12 inherited dropped) | 26 → 28 | +18.2 | **REFUSED** — 2 nodes | — | — | — | — |
| `ai_v12_02_winprob_critic` | 20 (→19) | 36.0 → 72.0 | +50.5 | **+1.50 ± 0.21 RISING** | **+0.89 ± 0.96** NOT DETECTED | 60→72M | +0.81 ± 1.11 ND | +0.08 ± 1.47 **ND** |
| `ai_v12_11_ladder_ctrl10M` | 4 (→3) | 4.0 → 8.0 | +262.3 | **+66.7 ± 6.4 RISING** | +66.7 ± 6.4 (same 3 nodes) | 4→8M | — | 0.00 ± 9.08 ND |

**What this says, and what it refuses to say.**

- **v8's line WAS still rising at its end, weakly**: `ai_v8_03` **+0.68 ± 0.22 Elo/M** over 151→242M,
  with **no detectable deceleration** (late − middle −0.60 ± 0.70). With the newest node kept, its
  late window alone reads **+0.44 ± 0.20 RISING**. Directionally consistent with §2.1's "strong and
  still learning", from a completely different instrument.
- **`ai_v12_02` is NOT plateaued**: **+1.50 ± 0.21 Elo/M** over 36→74M, **twice v8's rate**, with
  no detectable deceleration either (late − middle +0.08 ± 1.47).
- **`ai_v9_29` IS the decelerating one**: late − middle **−11.07 ± 4.53 FALLING** — but that is the
  ordinary saturation shape of the first 25M of a fresh run (its full slope is +15.9 Elo/M), not a
  late-life plateau, and `ai_v12_02` at the same steps is not comparable to it (different recipe).
- 🚨 **The two runs that §2.1/§2.2 are actually ABOUT — `ai_v8_04` (277M, no `snapshot_ladder` at
  all) and `ai_v9_59` (2 own nodes) — cannot be read here.** The v8-side evidence is its parent
  `ai_v8_03`; the gen-side evidence is `ai_v9_29`. **No cross-run level comparison is made** (per the
  brief): the pins differ, and while the 2026-09-07 eval-regime boundary does not touch
  `ladder.json`, only `ai_v12_11` records `eval_sentinel_greedy` at all (config v113; every other run
  here predates v112 and its regime is UNRECORDED — read the launch line, never assume).
- 🚨 **`ai_v12_02`'s ladder starts at 36M** because its earlier snapshots were groomed. Its 0–36M
  strength curve is not in `ladder.json` and is not reported.

---

## 5. Part 3 — the dose column, and a correction to the premise

`python -m main.dose` (`dose.json`; `lr_median` is over the run's own per-checkpoint sidecars, i.e.
the rate the run **realised**, not the one it was launched with):

| run | era | **ent-coef** | clip-range | launched `--lr` | **realised lr_median** (min–max) | n_epochs | batch × grad-accum = eff. batch | updates/env-step | **dose_rate** | vs `ai_v8_14` | own-segment length |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `ai_v8_01` | v8 | 0.05 | 0.10 | *(surgery script — no argv)* | 1.742e-4 (flat) | 7 | 2048 × 8 = 16,384 | 4.272e-4 | **7.44e-8** | 3.47× | 22.9M |
| `ai_v8_03` | v8 | 0.05 | 0.10 | 7e-5 | 1.695e-4 (1.45–1.74e-4) | 7 | 2048 × 16 = 32,768 ⚠ | 2.136e-4 | **3.62e-8** | 1.69× | 119.8M |
| `ai_v8_04` | v8 | 0.05 | 0.10 | 7e-5 | 1.205e-4 (flat) | 7 | 2048 × 16 = 32,768 | 2.136e-4 | **2.57e-8** | 1.20× | 8.8M |
| `ai_v8_14` | v8 | 0.05 | 0.10 | 7e-5 | 1.004e-4 (1.00–1.20e-4) | 7 | 2048 × 16 = 32,768 | 2.136e-4 | **2.14e-8** | 1.00× | 14.8M |
| `ai_v9_29` | gen | 0.02 | 0.15 | 3e-4 | 1.206e-4 (0.70–3.60e-4) | 10 | 2048 × 8 = 16,384 | 6.104e-4 | **7.36e-8** | 3.43× | 25.1M |
| `ai_v9_59` | gen | 0.02 | 0.15 | 3e-4 *(INERT — resume)* | 5.814e-5 (0.40–0.70e-4) | 10 | 2048 × 2 = 4,096 | 2.441e-3 | **1.42e-7** | **6.62×** | 2.8M |
| `ai_v12_02` | v12 | 0.02 | 0.15 | 3e-4 | 2.083e-4 (1.74–4.32e-4) | 10 | 2048 × 32 = 65,536 | 1.526e-4 | **3.18e-8** | 1.48× | 75.0M |
| `ai_v12_11` | v12 | 0.02 | 0.15 | 3e-4 | 3.960e-4 (3.60–4.32e-4) | 10 | 2048 × 32 = 65,536 | 1.526e-4 | **6.04e-8** | 2.82× | 10.0M |

⚠ `ai_v8_03` is flagged `[SHAPE MOVED]` by `main.dose` — its grad-accum changed mid-flight (launched
at 8, ran mostly at 16). The row uses the sidecar-median shape.

🚨 **The premise's dose figures are wrong, and correcting them changes the conclusion.** The question
states v8 ≈ 2.4e-7 and ours ≈ 1.5e-6 (~6×). Both omit **`grad_accum_steps`**, which the project's own
definition includes: `dose = lr_median × n_epochs / (batch_size × grad_accum_steps)`. With it:

- the **v8 line itself spans 2.14e-8 → 7.44e-8, a 3.5× internal range**;
- our two **fresh** win-prob arms sit at **3.18e-8 and 6.04e-8 — inside that range**;
- only `ai_v9_59`, whose `--grad-accum-steps 2` is 16× smaller than the v12 arms', is the 6.6×
  outlier — and it is a **fold**, not a parent.

⇒ **"our runs train at ~6× v8's dose" is true of one fold and false of the fresh arms.** The dose
cannot carry the cross-era entropy difference. (It is also already CLOSED as a fold lever:
`UNDERSTANDING` §2.3, *DOSE is NULL* across a 4× range.)

**The KL controller also inverts the naive `--lr` reading.** v8 launched at 7e-5 and the controller
drove it *up* to 1.0–1.7e-4; we launch at 3e-4 and it drives *down* to 0.6–2.1e-4 (`ai_v12_11`, only
10M long, is the exception at 3.96e-4). **The realised rates overlap.** Read `lr_median`, never
`--lr` — and on a resume `--lr` is INERT while **`--ent-coef` is NOT** (`model_build.py:543` sets
`model.ent_coef = args.ent_coef` on the resume path; §8 relies on this).

### The companion PPO health scalars — start / median / end of the own segment

| run | `approx_kl` | `clip_fraction` | `explained_variance` | `grad_norm` | `clip_range` |
|---|---|---|---|---|---|
| `ai_v8_01` | .0151 / .0157 / .0163 | .292 / .286 / .282 | .811 / .819 / .821 | 0.83 / 0.72 / 0.71 | 0.10 |
| `ai_v8_03` | .0153 / .0114 / **.0102** | .269 / .205 / .198 | .718 / .798 / .790 | 0.70 / 0.74 / 0.67 | 0.10 |
| `ai_v8_04` | .0298 / .0077 / .0071 | .375 / .180 / .165 | .745 / .800 / .778 | 0.75 / 0.47 / 0.44 | 0.10 |
| `ai_v8_14` | .0224 / .0057 / .0062 | .319 / .136 / .144 | .781 / .879 / .900 | 0.51 / 0.47 / 0.48 | 0.10 |
| `ai_v9_29` | .0040 / .0184 / .0159 | .042 / .187 / .155 | .504 / .733 / .748 | 1.22 / 1.18 / 1.02 | 0.15 |
| `ai_v9_59` | .0673 / .0392 / .0276 | .335 / .235 / .191 | .705 / .722 / .721 | 1.56 / 1.61 / 1.61 | 0.15 |
| `ai_v12_02` | .0024 / .0165 / **.0174** | .048 / .148 / .143 | .718 / .807 / .811 | 0.20 / 0.70 / 0.75 | 0.15 |
| `ai_v12_11` | .0033 / .0146 / .0162 | .084 / .166 / .160 | .658 / .792 / .801 | 0.17 / 0.79 / 0.73 | 0.15 |

Nothing here reads as a sick optimiser on either side: `explained_variance` converges to 0.78–0.82
in both eras, and `clip_fraction` to 0.14–0.20. The one systematic difference is that v8's end-state
`approx_kl` is **~40% smaller** (0.0102 vs 0.0174) at a **tighter clip range** (0.10 vs 0.15) — a
second, independent lever that points the same way as the entropy coefficient and that this evidence
cannot separate from it (§6).

---

## 6. Part 4 — the STALENESS leg, and why it fails

`train/selfplay_promoted_steps` is recorded **only at a promotion** and its value is the promoted
step (`selfplay_callback.py:800`), so the series **is** the promotion ladder. A run that stops
promoting keeps training against a frozen pool — the operational meaning of a stale pool. The
**trailing lag** (last own rollout − last own promotion) is the direct read.

| run | own promotions | mean gap | max gap | **trailing lag** | lag as share of own segment | pool size (first→last) | `win_rate_vs_pool` (median / last) | `eval_sentinel_greedy` |
|---|---|---|---|---|---|---|---|---|
| `ai_v8_01` | *no scalar* (pool inherited, `selfplay_fraction` 0.40 flat) | — | — | — | — | 1 → 1 | 0.44 / 0.43 | UNRECORDED (cfg 44) |
| `ai_v8_03` | **8** | **10.29M** | **18.00M** | **20.42M** | **17.0%** | 1 → 10 | **0.52 / 0.52** | UNRECORDED (cfg 45) |
| `ai_v8_04` | **0** | — | — | (no own promotion in 8.8M) | — | 1 → 1 | 0.38 / 0.50 | UNRECORDED (cfg 45) |
| `ai_v8_14` | 1 | — | — | 0.62M | 4.2% | 1 → 1 | 0.50 / 0.56 | UNRECORDED (cfg 45) |
| `ai_v9_29` | **10** | **2.22M** | 4.00M | 1.07M | 4.3% | 1 → 11 | 0.78 / 0.78 | UNRECORDED (cfg 101) |
| `ai_v9_59` | 2 | 2.00M | 2.00M | 0.02M | 0.6% | 12 → 13 | 0.68 / 0.70 | UNRECORDED (cfg 103) |
| `ai_v12_02` | **33** | **2.13M** | 4.00M | 1.01M | **1.3%** | 0 → 20 | 0.73 / 0.69 | UNRECORDED (cfg 110) |
| `ai_v12_11` | 3 | 2.00M | 2.00M | 0.03M | 0.3% | 0 → 3 | 0.78 / 0.74 | **True** (cfg 113) |

**NOT SUPPORTED, with the sign reversed.** Our runs promote ~5× more often (every ~2.1M vs ~10.3M)
and end essentially *on* their newest snapshot (1–4% of the run behind) where `ai_v8_03` ended 17% of
its run — **20.4M steps** — behind its last promotion, and `ai_v8_04`, the 277M parent itself, made
**zero** promotions in its 8.8M-step life. By the promotion-lag definition of staleness, **v8's pool
was the stale one.**

The one number that *does* point the owner's way is `win_rate_vs_pool`: `ai_v8_03` sat at **0.52**
against its own pool while ours sit at **0.69–0.78**, i.e. ours beat their pool much more easily —
consistent with "our self-play opponents are too weak to be a challenge". But this cannot be read as
a cross-era fact:

- 🚨 **`win_rate_vs_pool` carries the 2026-09-07 opponent-regime boundary** (§3.2 rule 5, worth
  **+8.9pp** to the trainee). `ai_v8_03` / `ai_v8_04` / `ai_v8_14` each pass
  `--eval-sentinel-greedy` explicitly in their recorded argv, and `ai_v12_11` records
  `eval_sentinel_greedy: true` in `model_config.json` (cfg v113);
  `ai_v9_29` / `ai_v9_59` / `ai_v12_02` ran in the v9-launch window when it was dropped **unrecorded**,
  and none of them records a regime at all (config < v112). Only the `ai_v8_03` (0.52) vs
  `ai_v12_11` (0.74) pair is plausibly same-regime — and those two differ in **pool size (10 vs 3)**
  and **maturity (268M vs 10M)**, so the comparison is confounded twice over.
- The promotion threshold itself moved with the regime (0.65 → 0.55), which mechanically changes
  the equilibrium `win_rate_vs_pool` a promoting run sits at.

⇒ **"our pool is weak relative to the trainee" is UNDER-DETERMINED by these logs.** It is a live
candidate and it needs a same-regime, pool-size-matched read, which does not exist offline.

---

## 7. Hazards found — each one a finding

1. 🚨 **A fork's TensorBoard directory carries its PARENT's entire history, and `tb_read.load()`
   reads all of it.** `gen3_tb_inherit_v1` writes an `events.out.tfevents.0000000000.inherited.0.0`
   prefix truncated at `fork_step` (backfilled 2026-09-08 for the runs that predate the feature):
   `ai_v8_14` merges **1,268 inherited** rollouts with **148** of its own, `ai_v9_59` **249** with
   **30**. **A naive entropy read of `ai_v9_59` returns the 0→28M chain and would have reported its
   H_start as 1.65 instead of 0.675 — inverting the fork-discontinuity sign in §3.** Every series
   here is split at `fork_step`, read from the recorded lineage and never re-derived. The same
   hazard makes `tb_read.selfplay_crossing_step()` return the **PARENT's** crossing for a fork (the
   script now stamps a `HAZARD` note on those rows) — a fork inherits a seeded pool and has no
   crossing of its own.
2. 🚨 **The same hazard reaches `ladder.json`.** A genuine fork auto-seeds its parent's self-play
   pool, so the parent's snapshots land in the fork's `snapshots/` and its ladder fit.
   **`ai_v9_59`'s 14-node ladder is 12 inherited nodes + 2 of its own** — all twelve share the
   mtime `Aug 27 10:25`, the fork-seeding moment. A "within-run late slope" over that file is a
   slope over the *parent's* ladder. Filtering to `step > fork_step` leaves 2 nodes and the correct
   answer is REFUSE.
3. 🚨 **The v8 line has no fresh-from-zero ancestor in the archive.** `main.lineage` reports
   `ai_v8_01_zarch_film_0717` as `role=fresh` — but that only means *the run records no parent*. Its
   `original_command` is **`tmp/fork_zarch_v8.py`** (an architecture-surgery script) and its own
   first rollout is at **step 148,401,357**. So **v8's entropy trajectory before 148M is
   unrecoverable**, and `role=fresh` must not be read as "trained from initialisation".
4. 🚨 **`ai_v8_01`'s `--ent-coef` is unrecoverable from its argv and recoverable from TB.** Its
   `original_command` is a script path, so an argv survey reports it as the parser **default
   (0.02)** while `hparams/ent_coef` in its events reads **0.05**. The archive survey script labels
   that class `unstated` rather than defaulting it — a fabricated default here would have put a
   0.02 run on the v8 arch and destroyed §8's collinearity finding.
5. ⚠️ **`ai_v8_04` — the 277M parent the whole question is about — has no `snapshot_ladder` at all**,
   and `ai_v12_02`'s ladder begins at 36M because its earlier snapshots were groomed. The strength
   leg is therefore answered by neighbours (`ai_v8_03`, `ai_v9_29`), never by the two runs named.
6. ⚠️ **The per-node `se` in `ladder.json` is a marginal BT standard error and the nodes are not
   independent** (BT re-solves every rating on every add). Every slope SE here is the *wider* of the
   analytic and residual estimates and is still optimistic. §3.2 rule 2 is handled by dropping the
   newest node for the headline; both variants are in `ladder_slopes.json`, and for `ai_v8_03` and
   `ai_v12_11` the drop changes the *verdict*, not just the point.
7. ⚠️ **Entropy under a mask is not scale-free across ecologies.** The action space is 11 in both
   eras (verified back to `f02e633a`), but the mean number of *legal* actions is not controlled and
   is not logged. The level claim in §2 carries that confound; the within-run slope claims do not.

---

## 8. Verdict

| leg of the hypothesis | verdict | on what |
|---|---|---|
| "our policy's entropy collapses early" | **SUPPORTED** | H falls to 0.5 × H₀ by 4.0–8.6M in every fresh arm; ends at 0.713–0.825 vs v8's 1.068–1.110; gap 3.3–5.4× the 0.0737-nat replicate floor; complete separation |
| "…and is still collapsing" | **SUPPORTED where powered** | `ai_v12_02` −0.0011 ± 0.00034 nats/M FALLING at 75M; v8's `ai_v8_03` **+0.0010 ± 0.00017 RISING**. At 10M the question is below the floor (3 seeds, 3 different verdicts) |
| "self-play goes stale" | **NOT SUPPORTED** (promotion-lag definition); **UNDER-DETERMINED** (opponent-strength definition) | we promote every 2.1M and trail 1–4%; `ai_v8_03` promoted every 10.3M and trailed 17%; `ai_v8_04` never promoted. `win_rate_vs_pool` points the owner's way but is cut by the 09-07 regime boundary and by pool size |
| "…and that is why our parents stop learning" | **NOT TESTABLE from these logs** | `--ent-coef` is perfectly collinear with the era: 0.05 on 58 runs, all v6–v8 signatures; 0.02 on 147 runs, all v9+; **no architecture carries both** (`ent_coef_archive_survey.json`). The only non-0.02 gen-era run, `ai_v9_50_fdF_p1c_0826` (`--ent-coef 0.0`), also carries `--policy-grad-coef 0.0` and `--distill-coef 1.0` — a pure-KL arm with no policy gradient, isolating nothing about entropy in the RL loop (and voided: ledger 2026-08-26, *ARM F PHASE 1 CONFOUNDED*) |
| the premise's own framing ("our parents plateau") | **DOES NOT REPRODUCE on the dense ladder** | `ai_v12_02` +1.50 ± 0.21 Elo/M over 36→72M vs `ai_v8_03` +0.68 ± 0.22 over 151→242M; neither shows a detectable late deceleration. The §2.1/§2.2 asymmetry is an **untaught-meter** fact |

**Has the entropy coefficient ever been isolated? NO — confirmed by grep, with citations.**
`designs/research_state/ledger.md` mentions it five times and never as an experiment:

- **L05251** (2026-08-28, *PROBES C + D DISPATCHED*) lists "hyperparams (**ent-coef**, lr)" among
  Probe C's candidate era-diff explanations — an **observational** archaeology brief, not a run.
- **L07039** (2026-08-30, R3/ai_v12 readiness) records, verbatim: *"G3/G4 = vf-coef
  (resume-immutable!) and **ent-coef sizing open**"* — an explicitly OPEN gap at the v12 launch.
- **L04882** (2026-08-26, *ARM F PHASE 1 CONFOUNDED AND VOIDED*) is the only time an ent-coef value
  was deliberately changed: `--ent-coef 0.02` left *unopposed* beside `--policy-grad-coef 0`
  drove entropy 0.892 → 1.354 (+52%) and dissolved two teams to 3.7%/10.0%; the fix was to rerun
  with `--ent-coef 0`. **That is an entropy-bonus artefact inside a policy-gradient-zero cell, not
  an entropy-coefficient experiment.**
- **L02446** (`--bait-entropy-boost`, BUILT/OFF) pins the identity
  `(ent_coef=c, boost=B) ≡ (ent_coef=B·c, boost=1)` — the machinery that *could* dose entropy per
  decision, never run; both boosts are `1.0` in every run in this note.
- **L06390 / L06602** note only that "ent-coef is NOT rescaled (advantages minibatch-normalized)" —
  a sizing remark. No ledger entry anywhere reports a **policy-entropy trajectory**.

### Confounds, named

The v8→v9 boundary changed all of the following in one wave, and the archive contains no run that
crosses any of them at fixed ent-coef:

| axis | v8 line | ours |
|---|---|---|
| **ent-coef** | 0.05 | 0.02 |
| **clip-range** | 0.10 | 0.15 |
| architecture | `gen3_opp_hp_typed_candidates_v1` — flat `Linear(latent,11)` action head + ZArch FiLM (`--zarch-film heads --zarch-dim 32`) | `gen3_critic_route_wave_v1` — **pointer-native** head (v51), zero-init scorers, edge-bias trunk, op block |
| obs space | pre-`unified-obs` era layout | 2501-dim entity/unified layout |
| reward composition | `--all-shaping-pbrs`, `--draw-penalty -35`, PopArt ON, `--value-dist-mode shaping`, `--win-prob-mode shaping` | `ai_v12_*`: `--critic winprob --no-hand-shaping --terminal-indicator`, γ=1, **no PopArt**, no shaping |
| critic | shaped, PopArt, `value-from-dist` | win-prob BCE on the terminal win indicator |
| ecology | `--team-pfsp onesided`, `--pfsp-scale 2.5`, `--pool-spread`, `--n-sentinels 10`, `--bot-weights aggressive_v2=3,heuristic2=3`, `--heuristic-floor 0.2`, `selfplay_fraction` 0.40–0.80 | `--team-pfsp off`, `--n-sentinels 5`, default bot weights, `selfplay_fraction` 0.90 |
| eval sentinel regime | greedy ON (recorded nowhere; the flag was dropped **unrecorded** at the v9 launch) | OFF for `ai_v9_*`/`ai_v12_02`; ON again from 2026-09-07 (`ai_v12_11`, cfg v113) |
| maturity of the observed window | 148M – 293M | 0.2M – 75M — **no overlap** |
| distillation target form | full-distribution KL | `--distill-target action --distill-topk 1` (argmax CE) on `ai_v9_59` |
| dose | 2.14e-8 – 7.44e-8 | 3.18e-8 – 6.04e-8 (fresh); 1.42e-7 (`ai_v9_59`) — **overlapping, not a confound** |

**Two of these are ranked above the rest as candidate carriers of the entropy gap**, because both act
directly on how far a policy may move per update and both point the same way: **`--ent-coef`
(0.05 → 0.02, a 2.5× smaller bonus)** and **`--clip-range` (0.10 → 0.15, a 50% wider trust region,
with a measured 40% larger end-state `approx_kl`)**. A design that varies only ent-coef leaves the
clip-range leg alive; it is the cheaper first cut, not the whole answer.

---

## 9. The cheapest experiment that isolates the entropy coefficient

**Measured cost on this box** (own-segment TB wall-clock ÷ own steps, the three ctrl replicates and
the 75M arm): **4.31 / 4.22 / 5.03 h per 10M** and **4.97 h per 10M** — the brief's 4.25 h/10M is
confirmed; use **~4.3 h/10M** and quote the 4.2–5.0 spread.

**Rung 1 — the necessary condition. ONE fresh 10M arm. ~4.3 GPU-h.**
`ai_v12_11_ladder_ctrl10M`'s argv verbatim with **only `--ent-coef 0.05`** changed (validate with
`python -m main.checkargs --argv "…"`; `--arch production` is already implied by that argv). The
control side is **already banked at n = 3** (`ai_v12_11` / `_15_b` / `_16_c`), so this buys the whole
cell for one arm.

- **Primary endpoint: `H_end` at 10M**, floor **0.0737 nats** (§2.1). The effect it must clear to
  support the mechanism is the cross-era gap, **0.243–0.397 nats = 3.3–5.4× the floor** — so a single
  arm is comfortably powered on this endpoint. Pre-register the reading: **H_end ≥ 1.00 nats ⇒ the
  coefficient reproduces v8's entropy regime; H_end inside 0.71–0.83 ⇒ the mechanism is REFUTED at
  2.5× the dose and the clip-range / architecture legs inherit.**
- **Secondary, free**: `approx_kl`, `clip_fraction`, `explained_variance` and the promotion cadence
  from the same events; `train/selfplay_promoted_steps` for the staleness leg.
- 🚨 **NOT an endpoint: strength at 10M.** The ledger banked (2026-09-11/12) that no 10M arm sits
  outside the 45-Elo floor and that seed replicates differ by 58–83 Elo against the control pair's
  10. A 4-node ladder cannot resolve this arm's strength and must not be quoted for it.

**Rung 2 — give the arm its own floor. +1 seed, ~4.3 GPU-h (cumulative ~8.6).** Only worth paying if
rung 1 lands *between* the two bands.

**Rung 3 — the causal claim the owner actually asked about.** "Does a higher ent-coef make the parent
keep learning?" needs a mature parent and the G5 continuation cell, and it is an order of magnitude
more expensive: a 28M parent + a ~1.2M plain continuation + the frozen comparator ≈ 29.2M steps
≈ **12.6 GPU-h per arm**, and G5's own design used **3 replicates per cell** across **2 cells**
(ent-coef 0.02 vs 0.05) ⇒ **≈ 75 GPU-h**. **Do not fund rung 3 before rung 1 answers**: if 0.05 does
not move H into v8's band at 10M, rung 3 is measuring nothing.

⚠️ **A design note that matters.** Unlike `--lr`, **`--ent-coef` is NOT inert on a resume** —
`src/main/train/model_build.py:543` sets `model.ent_coef = args.ent_coef` on the resume path. So a
*continuation* arm can legally change it, which makes an even cheaper (but confounded, and therefore
not recommended as rung 1) probe possible: resume an existing gen-era parent at `--ent-coef 0.05` for
1M steps and watch H. That answers "can the bonus lift a collapsed policy?", not "does the bonus
prevent the collapse?" — a different question.

---

## 10. Files

| file | what it holds |
|---|---|
| `entropy_forensics.py` | the whole read: entropy trajectories, the fork discontinuity, the replicate floor, the ladder slopes, the plots. `--out DIR` |
| `ent_coef_archive_survey.py` | every `--ent-coef` the archive has run, crossed with `arch_signature` — the collinearity table |
| `entropy.json` | per run: the own/inherited split, H_start/H_end/H_min, the crossing shares, both slopes, the companion scalars, the staleness block, the thinned curve |
| `entropy_replicate_floor.json` | the 3-seed floor on H_start / H_min / H_end |
| `fork_entropy_discontinuity.json` | §3's table |
| `ladder_slopes.json` | every own node with its se, and full/late/middle slopes both with and without the newest node |
| `dose.json` | `python -m main.dose --json` for all eight runs |
| `ent_coef_archive_survey.json` | the cross-tab + examples |
| `entropy_trajectories.png` | H vs absolute step (log-x) and vs fraction-of-run, all ten arms |
| `entropy_spines.png` | the chained optimisation spines |
| `ladder_late_slopes.png` | the dense ladders, own nodes, newest node ringed |

Reproduce:

```bash
export PYTHONPATH=$PYTHONPATH:src
D=designs/research_state/measurements/entropy_forensics_v8_vs_ours_2026-09-12
nice -n 15 python3 $D/entropy_forensics.py --out $D
nice -n 15 python3 $D/ent_coef_archive_survey.py --json $D/ent_coef_archive_survey.json
nice -n 15 python3 -m main.dose ai_v8_01_zarch_film_0717 ai_v8_03_zarch_control_0718 \
  ai_v8_04_distill_4teacher_0722 ai_v8_14_distill3_0725 ai_v9_29_rev1_0823 \
  ai_v9_59_R2ACTION_0827 ai_v12_02_winprob_critic ai_v12_11_ladder_ctrl10M --json
```

---

## 11. Ready-to-append ledger paragraph

> This file does NOT edit `ledger.md`, `UNDERSTANDING.md` or any design note. The paragraph below is
> offered for a separate, deliberate append; if it lands, `ledger_index.md` must be regenerated with
> `python -m main.ledger_index` (`src/ledger_index_gate_test.py` fails otherwise).

```markdown
### 🔬 ENTROPY FORENSICS — our policies end 0.24–0.40 nats BELOW v8's plateau (3.3–5.4× the floor); the STALENESS leg is REVERSED; the dose premise was WRONG; ent-coef is PERFECTLY COLLINEAR with the era (2026-09-12, zero GPU)

Owner's hypothesis — our entropy collapses, self-play goes stale, and that is why our parents stop
learning while v8's did not — read off the TensorBoard events of ten arms (`main.ops.tb_read`, the
project reader; `H = -train/entropy_loss` in nats over the unchanged 11-way masked categorical,
`ACTION_SPACE_SIZE = 11` verified back to `f02e633a`). **ENTROPY LEG SUPPORTED, and the separation is
complete**: every v8-line arm ends at **1.068–1.110 nats** (internal spread 0.042) and every fresh arm
of ours at **0.713–0.825**, the gen-era fold at **0.529** — smallest gap **0.243 nats = 3.3×** the
**0.0737-nat** replicate floor measured free on three same-recipe seeds (`ai_v12_11`/`_15_b`/`_16_c`;
the `H_start` control agrees to 0.007 nats, so this is not scatter). Where the slope is powered the
signs are OPPOSITE: `ai_v12_02` **−0.0011 ± 0.00034 nats/M FALLING** at 75M against `ai_v8_03`
**+0.0010 ± 0.00017 RISING** at 268M (n = 741 / 1178 rollouts). At 10M the slope is BELOW the floor —
three seeds, three verdicts — so "still declining" is only claimable on the 75M arm.
**STALENESS LEG NOT SUPPORTED, SIGN REVERSED**: `train/selfplay_promoted_steps` shows our runs
promoting every **~2.1M** steps and ending **1–4%** of the run behind their newest snapshot, while
`ai_v8_03` promoted every **10.3M** and ended **20.4M steps / 17%** behind, and **`ai_v8_04` — the
277M parent — made ZERO promotions in its 8.8M-step life**. `win_rate_vs_pool` (v8 0.52 vs ours
0.69–0.78) points the owner's way but is cut twice: the 2026-09-07 sentinel-regime boundary (+8.9pp,
and only `ai_v12_11` records a regime at all) and pool size 10 vs 3 — UNDER-DETERMINED, not refuted.
**THE PREMISE'S DOSE ARITHMETIC WAS WRONG** — it omitted `grad_accum_steps`. `main.dose`: the v8 line
spans **2.14e-8 – 7.44e-8** and our two FRESH win-prob arms sit at **3.18e-8 / 6.04e-8, inside it**;
only `ai_v9_59` (grad-accum 2, a fold) is the 6.6× outlier. The realised `lr_median` also overlaps
(v8's controller drove 7e-5 UP to 1.0–1.7e-4; ours drives 3e-4 DOWN to 0.6–2.1e-4). **Dose is not the
cross-era difference.** **THE CAUSAL CLAIM IS NOT TESTABLE OFFLINE**: across 212 runs the archive
holds `--ent-coef 0.05` on 58 runs (all v6–v8 signatures) and 0.02 on 147 (all v9+), and **NO
architecture carries both** — the coefficient is perfectly collinear with architecture, obs space,
reward composition, clip-range (0.10 → 0.15, with a measured 40% larger end-state `approx_kl`),
ecology and maturity. The only non-0.02 gen-era arm, `ai_v9_50_fdF_p1c_0826` (`--ent-coef 0.0`), runs
`--policy-grad-coef 0` + `--distill-coef 1.0` and isolates nothing (ARM F, voided 2026-08-26). Grep
confirms no ledger entry has ever isolated the coefficient or reported a policy-entropy trajectory;
it appears only as a Probe C candidate (2026-08-28) and as the still-open gap **"G3/G4 = vf-coef and
ent-coef sizing open"** (2026-08-30). **⚠️ AND THE PREMISE'S OWN FRAMING DOES NOT REPRODUCE ON THE
DENSE LADDER**: within-run, `ai_v12_02` rises **+1.50 ± 0.21 Elo/M** over 36→74M against
`ai_v8_03`'s **+0.68 ± 0.22** over 151→242M, and NEITHER shows a detectable late deceleration
(late−middle +0.08 ± 1.47 and −0.60 ± 0.70). Our era is not the plateaued one on strength — the
§2.1/§2.2 asymmetry is an **untaught-meter** fact, and "our parents stop learning" and "our entropy
is 0.35 nats lower" are two true statements this evidence does not join. **Bonus cross-era
contrast**: both v8 folds RAISE entropy at the fork (**+0.112 / +0.104 nats**) and finish at or above
their parent, while `ai_v9_59` LOWERS it (**−0.072**) and finishes **−0.218** below — mechanistically
consistent with TARGET FORM (full-distribution KL onto a high-entropy teacher pulls H up; the
`--distill-target action --distill-topk 1` argmax CE pulls it down), n = 1 per fold, suggestive only.
**FOUR HAZARDS BANKED**: (1) a fork's TB dir carries its PARENT's history and `tb_read.load()` reads
all of it — a naive read gives `ai_v9_59` `H_start` 1.65 instead of 0.675 and INVERTS the fork
discontinuity; split at `fork_step`, always; (2) the same hazard reaches `ladder.json` — `ai_v9_59`'s
14 nodes are 12 inherited + 2 own, all twelve sharing the fork-seeding mtime; (3) `main.lineage`
`role=fresh` means *records no parent*, NOT *trained from init* — `ai_v8_01`'s own first rollout is at
step 148,401,357 and its `original_command` is `tmp/fork_zarch_v8.py`, so **v8's entropy before 148M
is unrecoverable**; (4) that same run's `--ent-coef` is UNRECOVERABLE from its argv and reads 0.05 in
`hparams/ent_coef` — an argv survey that defaulted it to 0.02 would have destroyed the collinearity
finding. **NEXT, AND CHEAP: ONE fresh 10M arm, `ai_v12_11`'s argv with only `--ent-coef 0.05`, ~4.3
GPU-h** (measured 4.22–5.03 h/10M over four arms; the n=3 control is already banked). Primary
endpoint **H_end at 10M** against the 0.0737-nat floor, pre-registered: **≥1.00 ⇒ the coefficient
reproduces v8's regime; inside 0.71–0.83 ⇒ REFUTED at 2.5× the bonus and the clip-range leg
inherits.** Strength at 10M is NOT an endpoint (45-Elo floor, 58–83-Elo seed spread). The causal
"does it keep the parent learning" version needs the G5 continuation cell at ~**75 GPU-h** and must
not be funded before rung 1 answers. Note `--ent-coef` is NOT inert on a resume
(`model_build.py:543`), unlike `--lr`. Measurement:
`designs/research_state/measurements/entropy_forensics_v8_vs_ours_2026-09-12/`.
```
