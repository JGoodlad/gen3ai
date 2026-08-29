# PROBE H — forced-vs-contested decision triage for ladder search

**Date** 2026-08-28 · **Registered** ledger `f5b7da5` · **Data** `search_triage_policy_2026-08-28.json`
· CPU-only, `nice -n 15`, ≤2 cores, `models/` read-only.

**The question.** At 150 s/game a ladder player cannot search every decision. Can a cheap
(<1 ms) classifier split decisions into FORCED (play the policy immediately) and CONTESTED
(spend the budget), so that search concentrates where it matters?

---

## 1. Verdict

**The registered primary reading is REFUTED. The registered fallback reading is SELECTED on its
antecedent, and its proposed remedy is half-overturned.**

| registered | outcome |
|---|---|
| "≥60% of decisions classify as FORCED with an in-class search-flip rate **<2%**" | **REFUTED.** The *marginal* flip rate is **69.4%** [68.1, 70.7], n=4986. A 60%-coverage class with a 2% flip rate is arithmetically impossible. The in-class flip rate is **flat at ~0.69 across the entire frontier** and *rises* to 0.78 as the forced class shrinks — no cheap feature separates flips at all. |
| "yielding ~5–10× budget concentration" | **ACHIEVABLE** (5.7–9.0× at the recommended thresholds) — but not at a 2% flip rate. |
| "If the flip rate in the forced class is high, cheap triage is unsafe and the time manager needs the V-gap feature (more compute per decision)" | **SELECTED on the antecedent.** Remedy split: cheap triage is *not* unsafe when safety is scored in **win-probability** rather than raw action flips, and the one feature that works is **free** (it falls out of the same forward pass). But an expensive oracle triage retains **91%** of the dividend at 70% coverage against the free feature's **53%**, so there *is* real headroom for a costlier signal. |

**The one-line finding: raw action-flip rate is uninformative and unseparable; flip *cost* is
highly separable, and the only cheap feature that separates it is the critic's own P(win)
extremity — policy confidence (logit gap, entropy, top-1 probability) is at or **below** the
random null everywhere.** The policy does not know when search will overrule it; the critic does
know when being overruled would not matter.

---

## 2. Method

**Population** — 51,731 recorded decisions from 1,937 bridge-eval battles across 7 current-arch
R3 runs (`models/ai_v9_6[3-9]_R3*`) at steps 26M/28M/30M, 10 opponents. 86.6% are
`move_selection`; 13.4% are forced-switch phases.

**Triage features** (all available at play time from the forward pass already being run, <1 ms):
`n_legal` · masked top1−top2 logit gap · masked policy entropy · masked top-1 probability ·
`|P(win) − 0.5|` (win-prob extremity) · turn number.

> ⚠️ **The pre-mask logits trap (project finding #30) was handled.** Recorded `logits` are
> PRE-MASK; every gap/entropy/top-1 feature here masks with the recorded `action_mask` and
> renormalizes over the legal actions before any statistic is taken.

**Acid test on the recorded action.** `argmax(masked logits) == recorded action` for **51,731 of
51,731** decisions — eval traces record deterministic argmax play, so "the policy's action" is
unambiguous.

**Ground truth** — a reusable corpus does not exist (search-dividend outputs are game-level
aggregates; only 4 per-decision search-vs-policy pairs exist anywhere on disk), so it was
computed. **n = 4,986** `move_selection` decisions, simple random sample of the population,
one-ply CRN lookahead via `main.prober.lookahead` (`ProbeSession(impl="rust")`): re-roll the turn
under every legal action with the opponent playing its **recorded** move on the **realized dice**,
materialize each successor through the real encoder, read the loaded checkpoint's V(s′) and
P(win)(s′). 14 errors (0.28%, replay desync). 0.45 s/decision.

- **flip** = `argmax_a V(s′_a) ≠ policy action`.
- **Δwp** = `P(win)(s′_best) − P(win)(s′_chosen)`, floored at 0 — the win-probability the search
  *claims* is forgone by not searching this decision. Terminal arms (no successor state) are
  imputed 1.0/0.0; 246 of 4,986 (4.9%).

**Acid test on the load path.** The chosen action's *counterfactual* V(s′) must reproduce the
trace's recorded next V (the CRN anchor). Over 2,289 non-terminal decisions:
**corr = 1.000000, mean |diff| = 3e-5, max |diff| = 1.3e-4.** The re-roll → materialize → model
path is verified end to end.

---

## 3. Is the 69% flip rate real?

Three independent checks say yes.

**(a) Dice-noise control** (n = 683, 4 fresh CRN seeds each, `n_seeds=4`). If the flip were
argmax-over-noise, re-rolling the dice would move the argmax:

| | |
|---|---|
| argmax unstable under re-roll | **18.4%** [15.7, 21.5] — i.e. 82% dice-stable |
| median top1−top2 V margin, in dice-σ | **2.85 σ** |
| fraction of decisions with margin < 1 dice-σ | 25.6% |
| flip rate judged on 1 CRN sample | 0.693 |
| flip rate judged on the 5-sample dice mean | **0.679** |

Averaging out the dice moves the flip rate by 1.4 points. The disagreement is a stable property
of the model, not sampling noise. *(This bounds one noise source. It does not bound critic
approximation error on off-distribution successors — see caveats.)*

**(b) External corroboration.** The search-dividend battery's own logs
(`tmp/search_dividend/*.jsonl`, independent machinery and runs) record `n_changed/n_searched`
for a real multi-world, α-pruned, iterative-deepening search over **~40,000 searched decisions**:

| arm | 1 s | 3 s | 10 s |
|---|---|---|---|
| honest (belief-determinized) | 0.611 | 0.638 | — |
| oracle (true hidden state) | 0.639 | 0.602 | 0.632–0.655 |
| **playoff** (top-2 settled by *paired rollouts to a terminal*, 20 s) | — | — | **0.074** |

A proper search changes the action 60–67% of the time. The one-ply proxy's 69.4% is in family.

**(c) The playoff exception is the tell.** The only arm that settles the choice by *actual
rollouts* rather than by averaging leaf values flips 7.4%, not 64% — the optimizer's-curse
signature. It is also the only search arm that does not lose (§6).

---

## 4. The frontier

Rule shape: route to FORCED if `n_legal ≤ 1 OR |P(win) − 0.5| ≥ Y`. Coverage is the forced
fraction; concentration = 1/(1−coverage); seconds/contested decision from the budget model in §6.

*Coverage in this table is measured on the **labelled sample** (n=4,986 `move_selection`
decisions). §6 re-states the chosen row against the **full 51,731-decision population** including
forced-switch phases, which shifts coverage by ≈1 point (83.5% → 82.5% at Y=0.15).*

| Y | forced | in-class flip rate | in-class material (≥5 pp) | dividend **retained** | missed Δwp / decision | concentration | s / contested decision |
|---|---|---|---|---|---|---|---|
| 0.02 | 98.0% | 0.694 [0.681,0.707] | 0.223 | 3.8% | 3.54 pp | 50.9× | 144.4 |
| 0.06 | 94.0% | 0.693 [0.679,0.706] | 0.215 | 12.3% | 3.23 pp | 16.8× | 47.7 |
| 0.10 | 89.5% | 0.693 [0.680,0.707] | 0.204 | 20.1% | 2.94 pp | 9.5× | 26.9 |
| 0.12 | 87.0% | 0.692 [0.679,0.706] | 0.197 | 24.8% | 2.77 pp | 7.7× | 21.8 |
| **0.15** | **83.5%** | **0.693 [0.679,0.707]** | **0.188** | **31.0%** | **2.54 pp** | **6.0×** | **17.2** |
| 0.20 | 76.4% | 0.694 [0.679,0.709] | 0.169 | 43.8% | 2.07 pp | 4.2× | 12.0 |
| 0.25 | 68.3% | 0.700 [0.684,0.715] | 0.149 | 55.6% | 1.64 pp | 3.2× | 8.9 |
| **0.30** | **57.8%** | **0.713 [0.696,0.729]** | **0.123** | **67.6%** | **1.19 pp** | **2.4×** | **6.7** |
| 0.35 | 47.0% | 0.731 [0.712,0.748] | 0.094 | 78.7% | 0.78 pp | 1.9× | 5.4 |
| 0.40 | 34.7% | 0.753 [0.732,0.773] | 0.057 | 88.6% | 0.42 pp | 1.5× | 4.3 |
| 0.45 | 21.7% | 0.778 [0.753,0.802] | 0.024 | 95.8% | 0.16 pp | 1.3× | 3.6 |
| 0.49 | 8.0% | 0.773 [0.730,0.812] | 0.000 | 99.8% | 0.01 pp | 1.1× | 3.1 |

**Read the flip-rate column and the retained column against each other.** The flip rate is flat —
and *worse* in the small forced classes. The dividend column moves by 26× across the same sweep.
The two orderings are opposite: `|P(win)−0.5|` selects a class the search flips *more* often and
where those flips are worth *less*. That is the whole result.

**Dividend concentration.** Median Δwp is **0.48 pp**; **83.0%** of the total Δwp sits in the
**22.7%** of decisions worth ≥5 pp. Most decisions genuinely are not worth searching — the ORACLE
floor below routes 40% of decisions to FORCED at *zero* cost.

---

## 5. Per-feature ablation — which feature carries the classification

Missed-Δwp share / in-class flip rate, at **matched coverage**. Lower missed share is better;
the RANDOM null misses exactly its coverage by construction.

| feature | cov 0.2 | cov 0.3 | cov 0.4 | cov 0.5 | cov 0.6 | cov 0.7 |
|---|---|---|---|---|---|---|
| **`win_prob_extremity`** | **0.032** / 0.77 | **0.082** / 0.76 | **0.155** / 0.74 | **0.238** / 0.72 | **0.352** / 0.71 | **0.470** / 0.70 |
| `masked_top1_top2_gap` | 0.201 / 0.55 | 0.297 / 0.58 | 0.401 / 0.61 | 0.509 / 0.64 | 0.601 / 0.65 | 0.711 / 0.66 |
| `masked_top1_prob` | 0.203 / 0.54 | 0.310 / 0.58 | 0.415 / 0.60 | 0.511 / 0.63 | 0.627 / 0.64 | 0.719 / 0.65 |
| `masked_policy_entropy` | 0.208 / 0.54 | 0.316 / 0.58 | 0.426 / 0.60 | 0.550 / 0.63 | 0.656 / 0.64 | 0.756 / 0.65 |
| `turn` | 0.204 / 0.61 | 0.249 / 0.62 | 0.370 / 0.64 | 0.505 / 0.66 | 0.625 / 0.67 | 0.724 / 0.68 |
| `n_legal` | 0.241 / 0.65 | 0.384 / 0.67 | — | 0.549 / 0.69 | — | 0.766 / 0.69 |
| *RANDOM null* | *0.189* | *0.308* | *0.425* | *0.523* | *0.616* | *0.708* |
| `LEARNED` (all features, 5-fold CV) | 0.031 | 0.083 | 0.145 | 0.231 | 0.328 | 0.467 |
| **ORACLE floor** | 0.000 | 0.000 | 0.000 | 0.005 | 0.031 | 0.090 |

Two readings, both sharp:

1. **Every policy-confidence feature is a null.** Logit gap, entropy and top-1 probability sit
   *at or above* the random null from coverage 0.3 upward. Turn number and `n_legal` likewise.
   The mission's suggested rule shape (`gap ≥ X`) contributes **nothing**.
2. **The learned all-feature model does not beat `win_prob_extremity` alone** (0.328 vs 0.352 at
   coverage 0.6 — and 0.145 vs 0.155 at 0.4). One feature is the whole classifier.

**Drop-one ablation** (5-fold CV gradient boosting, coverage 0.6; full model = 0.328):

| feature removed | missed Δwp share |
|---|---|
| **`wp_extremity`** | **0.497** ← collapses to the random null |
| `turn` | 0.353 |
| `n_legal` | 0.352 |
| `entropy` | 0.337 |
| `top1_prob` | 0.333 |
| `gap` | **0.328** ← removing it *improves* the model; it is pure noise |

**Confound check — is `wp_extremity` just a proxy for `n_legal`?** Δwp is a max over candidates,
so more candidates means more max-of-noise inflation, and a feature correlated with `n_legal`
could win for the wrong reason. Re-run *inside* each `n_legal` stratum (50% coverage, null = 0.500):

| stratum | n | `wp_extremity` | `gap` | `top1_prob` |
|---|---|---|---|---|
| n_legal=5 | 425 | **0.295** | 0.451 | 0.450 |
| n_legal=6 | 597 | **0.286** | 0.527 | 0.541 |
| n_legal=7 | 855 | **0.225** | 0.499 | 0.526 |
| n_legal=8 | 1149 | **0.231** | 0.546 | 0.574 |
| n_legal=9 | 1539 | **0.254** | 0.481 | 0.477 |

The advantage holds in every stratum. Not a confound.

**Headroom for an expensive signal.** The ORACLE retains 91.0% of the dividend at 70% coverage;
the free feature retains 53.0%. Those 38 points are what a costlier triage signal (the registered
fallback's "V-gap feature") could in principle buy — a sized, unbuilt lever.

---

## 6. The operating rule and its budget

**Budget model.** 150 s/game timer, 20% safety margin, **42 decisions/game** (self-play mirror
`n_decisions` 2534/60 — the strong-opponent proxy; vs-bots games run 24.7), 18 ms/decision base
inference latency (the `ladder_drift_scan` websocket measurement). Usable think time
**119.2 s/game**; searching *uniformly* affords **2.84 s/decision**.

### Recommended: `if n_legal <= 1 or abs(P_win - 0.5) >= 0.15: play immediately`

Measured over the full 51,731-decision population (forced-switch phases included):

| | |
|---|---|
| coverage — all decisions | **82.5%** forced (move_selection 83.1%, forced-switch 78.7%) |
| **budget concentration** | **5.7×** |
| seconds per contested decision | **16.2 s** (vs 2.84 s uniform) |
| in-class flip rate | **0.693** [0.679, 0.707] |
| in-class **material** (≥5 pp) flip rate | **0.188** |
| missed Δwp | **2.54 pp/decision claimed**, retaining **31.0%** of the total dividend |
| vs a random triage at the same coverage | random retains 16.5% ⇒ **1.88× better** |

**Why this point.** It is the setting at which the 150 s budget affords ~16–20 s per searched
decision on ~15% of decisions — which is *exactly* the regime of the `playoff` arm (14.5% of
decisions searched, 20 s budget), the only search configuration in the battery that does not
lose (§6.1). The free feature reproduces that searched-fraction from the forward pass alone.

### Alternative (if a cheap, *trustworthy* search estimator lands): `>= 0.30`

57.1% forced · **2.3×** concentration · 6.6 s/contested decision · retains **67.6%** of the
dividend · in-class material-flip rate 0.123 · missed 1.19 pp/decision.

### 6.1 The context that governs the choice

The search-dividend battery's **mirror** arms (same network both sides, search on one side only,
so the null is 0.50 by construction) measure whether acting on these flips actually wins:

| arm | win rate (null 0.50) | fraction of decisions searched |
|---|---|---|
| `mirror_base` (no search) | 0.500 | 0.000 |
| `mirror_honest_1s` | **0.292** ±0.058 | 0.865 |
| `mirror_honest_3s` | **0.266** ±0.097 | 0.853 |
| `mirror_oracle_1s` | 0.317 ±0.083 | 0.874 |
| `mirror_oracle_3s` | 0.436 ±0.110 | 0.877 |
| `rladder` oracle 10 s (R=1…32) | 0.125–0.263 | 0.844 |
| **`playoff_10s`** (paired rollouts, 20 s) | **0.450** ±0.109 | **0.145** |

**Search as currently implemented loses, badly, and the arm that loses least is the one that
triages hardest.** This does not invalidate the frontier — the frontier ranks *where the search's
verdict differs materially*, which is the necessary input to any time manager regardless of
estimator — but it does mean the Δwp column is the critic's **claim**, not a realized gain. Two
of this probe's results point at the same cause: a 69% flip rate that survives dice-averaging but
collapses to 7.4% under paired rollouts is an argmax-over-noisy-leaf-values problem, i.e. the
estimator, not the budget.

**Practical implication.** Triage here is *damage control* before it is *budget allocation*.
`|P(win) − 0.5| ≥ 0.15` is the safe half of it on its own merits — never search a decided
position — and it costs nothing to ship.

---

## 7. Caveats

1. **Δwp is an upper bound, inflated by the optimizer's curse.** Mean claimed Δwp is 3.68 pp ×
   42 decisions = 155 pp/game, which is impossible. It is a max over noisy leaf estimates. The
   frontier is therefore a **relative** ranking of where the search's verdict differs, not a
   forecast of realized win rate. §6.1 is the direct evidence that acting on it currently loses.
2. **One search, not all searches.** Ground truth is one-ply CRN lookahead with the opponent
   playing its recorded move — no belief determinization, no depth, no opponent modelling. The
   flip *rate* is corroborated by the real battery (60–67% vs 69%); the Δwp *distribution* is not.
3. **`move_selection` only.** Re-rolls anchor at start-of-turn move rounds, so the 13.4% of
   decisions that are forced-switch phases carry no flip label. Population coverage of the rule
   is reported for them (78.7%) but their in-class safety is **unmeasured**.
4. **Eval-trace distribution.** The 9 fixed bots plus one external sentinel, not ladder humans.
   `|P(win)−0.5|` is large more often against weak bots than it would be on ladder, so the
   *coverage* of any threshold is optimistic; the *ordering* of features should transfer.
5. **One generation.** 7 R3 arms at 26M/28M/30M steps of gen-15-era current-arch runs.
6. **P(win) calibration is inherited.** The rule is only as good as the win-prob head's
   calibration; `python -m main.prober.query calibration` is the existing instrument for that and
   was not re-run here.
7. **Terminal imputation** — 246/4,986 (4.9%) decisions had a terminal candidate whose win-prob
   was imputed 1.0/0.0.
8. **The 42 decisions/game budget input** comes from self-play mirror games; a ladder game against
   a human may differ, which scales the seconds-per-contested-decision column linearly.

---

## 8. Reproduction

Scratch scripts (untracked, `tmp/`): `probeH_harvest.py` (feature harvest, 51,731 decisions) →
`probeH_lookahead.py` (ground truth; `<n> <seed> <budget_s> <n_seeds> <out>`) →
`probeH_analyze.py` (frontier + ablation, writes the `.json`) → `probeH_tables.py` (renders
these tables from the `.json`). `probeH_acid.py` is the CRN-anchor acid test;
`probeH_dividend.py` aggregates the external battery; `probeH_popcov.py` the deployment coverage.
Total compute: ~45 min single-threaded on 2 nice-15 cores.
