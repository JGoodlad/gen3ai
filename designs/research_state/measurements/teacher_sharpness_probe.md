# Teacher-sharpness probe — are the tick-1 tock teachers degenerately sharp?

**Date** 2026-08-25 · **Producer** `designs/research_state/measurements/teacher_sharpness_probe.py`
· **Data** `teacher_sharpness_probe.json` · **n** 5056 decisions over 80 bridge battles.

## What was pre-registered

The era's live working theory (ledger, three corrections deep) holds that the tick-1 tock teachers
are **distributionally immature**: 3M steps of specialization against one frozen opponent improved
their OUTCOMES (+9pp extraction, verified) while collapsing their action distributions into
over-sharp scripts, so KL-matching them injects overconfident narrowness through the shared trunk.

Its one cheap falsifiable prediction: **on the states the distillation KL fires on, a teacher's
policy entropy should sit FAR below the base model's.**

| reading | verdict |
|---|---|
| teacher entropy ≪ base, on-pin | theory CONSISTENT |
| teacher entropy ≈ base | theory IN TROUBLE |

## Method

Three models — base `ai_v9_29_rev1_0823/final_model.zip` (25M), teachers
`ai_v9_31_tock1_k4_0824` (tock-1a, 4 pinned teams) and `ai_v9_36_tock1c_q6_0824` (tock-1c, 2 pinned
teams), each base+3M. All three carry the same `model_config.json`
(`gen3_critic_route_wave_v1`, v101), so the loader's `check_compatible` was left live.

One pilot generates the states; **every model is then forwarded on the STORED obs**, so the
teacher-vs-base contrast is paired on identical states and the bootstrap CI resamples BATTLES, never
decisions. The distribution measured is byte-for-byte the one the loss uses —
`DistillTerms._distill_loss` masks illegal actions to −∞ and normalises both sides over the legal
set, so the probe computes `softmax(logits + (mask−1)·1e9)`. States with one legal action are
excluded from every distributional statistic (a forced decision carries no signal); they are 0.5–4%
of rows and counted separately.

Four conditions per teacher, 10 battles each vs the frozen rev-1 24M snapshot — the exact opponent
the tocks trained as exploiters against:

* **pilot = base** — the distill-time state distribution (the KL fires on states the TRAINEE visits,
  and the trainee is a base fork). **PRIMARY.**
* **pilot = teacher** — the teacher's own trajectory, i.e. the distribution it actually plays.
* **on-pin** — the teacher's own `--trainee-teams`, read verbatim from its `original_command`.
* **off-pin** — 6 pool teams in no teacher's slice, same opponent.

## 1. Absolute distributions (mean over legal actions, `n_legal ≥ 2`)

Mean legal actions is ~6.4–7.0, so a uniform policy would sit at ~1.86–1.94 nats.

| cell | model | H (nats) | eff. #actions | H / H_uniform | top-1 median | top-1 p90 | P(top-1 > 0.9) |
|---|---|---|---|---|---|---|---|
| **1a on-pin, pilot base** | teacher | **0.841** | 2.57 | 0.467 | 0.659 | 0.974 | 20.1% |
| | base | 0.780 | 2.39 | 0.430 | 0.701 | 0.980 | 22.1% |
| **1a on-pin, pilot teacher** | teacher | **0.788** | 2.42 | 0.441 | 0.703 | 0.974 | 22.8% |
| | base | 0.712 | 2.23 | 0.398 | 0.739 | 0.989 | 26.8% |
| 1a off-pin, pilot base | teacher | 0.811 | 2.49 | 0.446 | 0.689 | 0.973 | 22.3% |
| | base | 0.789 | 2.45 | 0.436 | 0.708 | 0.981 | 24.3% |
| 1a off-pin, pilot teacher | teacher | 0.802 | 2.49 | 0.437 | 0.691 | 0.977 | 25.5% |
| | base | 0.758 | 2.41 | 0.409 | 0.728 | 0.984 | 30.3% |
| **1c on-pin, pilot base** | teacher | **0.664** | 2.13 | 0.354 | 0.757 | 0.990 | 31.7% |
| | base | 0.625 | 2.09 | 0.337 | 0.806 | 0.996 | 40.7% |
| **1c on-pin, pilot teacher** | teacher | **0.735** | 2.30 | 0.390 | 0.724 | 0.980 | 27.8% |
| | base | 0.667 | 2.14 | 0.356 | 0.788 | 0.987 | 31.6% |
| 1c off-pin, pilot base | teacher | 0.785 | 2.42 | 0.432 | 0.710 | 0.972 | 24.4% |
| | base | 0.743 | 2.31 | 0.413 | 0.722 | 0.980 | 25.2% |
| 1c off-pin, pilot teacher | teacher | 0.880 | 2.67 | 0.493 | 0.648 | 0.974 | 20.0% |
| | base | 0.780 | 2.43 | 0.440 | 0.715 | 0.988 | 27.5% |

## 2. Paired contrasts on identical states (teacher − base; 95% CI, battle-clustered bootstrap)

| cell | n states | Δ entropy (nats) | Δ top-1 | KL(teacher ‖ base) |
|---|---|---|---|---|
| **1a on-pin, pilot base** | 453 | **+0.0611** [+0.0035, +0.1367] | −0.042 [−0.082, −0.010] | 0.453 [0.383, 0.537] |
| **1a on-pin, pilot teacher** | 527 | **+0.0759** [+0.0422, +0.1099] | −0.039 [−0.052, −0.026] | 0.476 [0.398, 0.551] |
| 1a off-pin, pilot base | 725 | +0.0214 [−0.0376, +0.0873] | −0.009 [−0.037, +0.016] | 0.409 [0.351, 0.476] |
| 1a off-pin, pilot teacher | 881 | +0.0443 [−0.0241, +0.1189] | −0.021 [−0.054, +0.009] | 0.360 [0.292, 0.420] |
| **1c on-pin, pilot base** | 467 | **+0.0384** [+0.0001, +0.0787] | −0.018 [−0.041, +0.001] | 0.431 [0.365, 0.507] |
| **1c on-pin, pilot teacher** | 522 | **+0.0684** [+0.0266, +0.0979] | −0.036 [−0.053, −0.014] | 0.459 [0.364, 0.558] |
| 1c off-pin, pilot base | 660 | +0.0415 [−0.0164, +0.1152] | −0.011 [−0.041, +0.014] | 0.357 [0.294, 0.416] |
| 1c off-pin, pilot teacher | 735 | +0.1003 [+0.0042, +0.2139] | −0.043 [−0.094, −0.001] | 0.420 [0.343, 0.510] |

**Every one of the 8 cells has Δentropy > 0 and Δtop-1 < 0** — the teacher is *flatter* than the
base on identical states, never sharper. All four on-pin CIs exclude zero on the positive side.
Sign consistency across 8 cells is itself p ≈ 0.008 under a sign test, though the cells are not
fully independent (2 teachers × 4 conditions).

## 3. The one thing that IS large: divergence without narrowness

`KL(teacher ‖ base)` is **0.43–0.48 nats on-pin** — a big number next to a base entropy of only
0.63–0.78 nats. The teacher is a substantially different policy. But it is not a *narrower* one, and
**the divergence is only ~10–20% larger on-pin than off-pin** (1a: 0.45 vs 0.41; 1c: 0.43 vs 0.36,
pilot=base). The teacher's 3M steps moved its policy roughly as much on teams it never piloted as on
the teams it specialized in, while the distillation KL only ever fires on the on-pin subset.
*(Comparison across populations is suggestive, not paired — on-pin and off-pin are different state
distributions, so the two KLs are not measured on the same states.)*

## Verdict

**The theory's prediction is falsified, and the sign is reversed.** Teacher entropy is not ≪ base on
the on-pin states the distill KL fires on; it is consistently and significantly *higher*
(+0.038 to +0.076 nats on-pin, all four CIs excluding zero), the teachers are less often
near-deterministic than the base (P(top-1 > 0.9) 20–32% vs 22–41%), and neither policy is anywhere
near degenerate — both sit at ~0.34–0.47 of the uniform entropy over ~6.6 legal actions. The
"over-sharp script" mechanism as stated does not exist in these two teachers, so the ledger's
narrowness hypothesis (a) should be marked REFUTED-AS-STATED rather than carried into a 9M-step
"better tocks" prescription; what survives is a weaker and different claim — the teachers are
*divergent* (KL ~0.45 nats) rather than *narrow*, and their divergence is barely slice-specific,
which points the KL-injection story toward "the trunk is pulled toward a globally different policy"
rather than "the trunk is pulled toward overconfidence".

## Limitations

* **n and clustering.** 10 battles per cell (453–881 scored decisions). The bootstrap resamples 10
  clusters, so the CIs are honest but coarse; the two on-pin `pilot=base` intervals only barely
  exclude zero. The *direction* is what 8/8 cells agree on, not the magnitude.
* **Two of the tocks.** tock-1a and tock-1c only; tock-1b was not measured.
* **Opponent regime is simplified.** Training was ~50% scripted bots / 50% the frozen snapshot
  (`exploiter_bot_fraction 0.5`); this probe plays 100% frozen snapshot. The opponent also pilots
  **one fixed pool team** throughout, chosen by a fixed seed, so the matchup is held constant for
  pairing — training sampled the whole pool.
* **What is measured.** The teachers' *current* distributions, not what the KL gradient does to the
  student's trunk. This falsifies the stated prediction; it does not exhaust every narrowness story
  (e.g. a low-rank-but-flat target could still be a poor thing to regress onto).
* **Checkpoint choice.** `final_model.zip` for each tock. If the distill run consumed a different
  checkpoint of the same tock, the numbers would need re-taking against that file.
* Base and teacher pilots produce *different* state distributions; each contrast is paired within its
  own population, and the two pilots are reported separately rather than pooled.
