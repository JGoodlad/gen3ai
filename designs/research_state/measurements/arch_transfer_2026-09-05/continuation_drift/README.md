# Continuation drift — is a young model's progress NOISE or DRIFT?

**Status: PRE-REGISTERED 2026-09-06, before any displacement, exponent, cosine or KL was
computed.** Everything below the `## PRE-REGISTRATION` heading was written first; results are
appended under `## RESULTS` and the pre-registration is not edited afterwards.

Predecessors: [`../fold_displacement/README.md`](../fold_displacement/README.md) (|Δθ| ∝ t^0.48 on
our folds; replicate cosine 0.56 at fold depth; the per-parameter-group decomposition and the
`load_foreign_opponent` loader this probe reuses) and
[`../exploiter_drift/README.md`](../exploiter_drift/README.md) (exponent 0.548 on the exploiter
chains, and the KL-vs-t output twin).

---

## THE QUESTION

Ledger `2026-09-06 · CELL 2`: three plain continuations of **v8's 277M-step parent** — no teacher,
no distillation loss, no teacher-team bias, no stable opponents — gained **+3.45pp [+0.46, +6.48]**
on the era's 16-team untaught meter over ~1.08M steps. Ledger `2026-09-06 · G5`: the matched
control on **our 28M-step parent** (`ai_v9_195/196/197_G5PLAIN{A,B,C}_0906`, +1.18M steps).

The learning note [`designs/learning/negative_transfer_and_shared_functions.md`](../../../../learning/negative_transfer_and_shared_functions.md)
§4 offers one candidate mechanism for why a young parent might not show such a gain:

> a young critic gives noisy advantages, so the update is a random walk; a mature critic gives
> directed drift.

A random walk in parameter space accumulates displacement as `|Δθ| ∝ t^0.5`; a directed drift
accumulates it as `|Δθ| ∝ t^1`. Our folds measured **0.48** and our exploiters **0.548** — both
indistinguishable from pure diffusion. Nobody has ever measured the exponent on a *mature* parent.

The two plain-continuation cells are the cleanest possible contrast for this: neither has a
teacher, a distillation term, or stable opponents, so the only things that differ are the parent's
maturity, the era's code/architecture, and the optimiser hyperparameters (registered as confounds
below).

## THE DATA

| | v8 cell 2 | G5 (ours) |
|---|---|---|
| parent | `ai_v8_04_distill_4teacher_0722/final_model_interrupted.zip` @ **277,583,267** | `ai_v9_59_R2ACTION_0827/final_model.zip` @ **28,115,184** |
| arms | `v8rep_p2self_{A,B,C}_0905` | `ai_v9_19{5,6,7}_G5PLAIN{A,B,C}_0906` |
| code | `b13b30b2` (era, obs 2992, config_version 45) | `407b27c0` (current, obs 2501) |
| depths | **2** per arm | **3** per arm |

**⚠️ The depth grids are what exists on disk and cannot be extended without training.** The v8
arms carry exactly one own self-play snapshot (~fork+417k, the identical file also written to
`best_model/` and `eval_traces/step_*/snapshot.zip`) plus the end; their `checkpoints/*.zip` sits
30–5,019 steps from `final_model_interrupted.zip` and is therefore the SAME depth, measured as an
agreement check and never entered as a second point in a fit. The G5 arms carry two checkpoints
(+500,016, +1,000,032) plus the final (+1,179,648).

Consequence, registered in advance: **a per-arm exponent on the v8 side is a two-point slope with
no residual and therefore no within-arm error bar.** All uncertainty on that side comes from the
three arms.

## PRE-REGISTRATION

### P1 — the displacement exponent `b`

For each cell, each arm, each fit depth: `Δθ = θ_arm − θ_parent` over the policy's **named
parameters** (buffers — PopArt statistics and the constant data tables — excluded from every group
and reported separately). Fit

```
log |Δθ_g| = a_g + b_g · log t
```

by ordinary least squares, for `g` = ALL and for each of the six parameter groups
(`action_head` / `encoders` / `team_transformer` / `projection_mlp` / `belief_op` / `critic`),
grouping rules **imported** from `../sharing_kernel/kernel.py` — which already maps both eras'
module names onto the same six ROLES (v8's `action_net` and the gen `pointer_head` both →
`action_head`), which is the brief's "match by role, not by name".

Two estimators, both reported:

* **per-arm** `b` (v8: the exact two-point slope; gen: OLS over 3 points), then the **mean over
  the cell's three arms** with a bootstrap over ARMS;
* **pooled** `b` — one OLS over all of a cell's (arm, depth) points, with a cluster bootstrap
  resampling ARMS.

**Matched-lever-arm variant.** The v8 slope spans t ∈ [0.417M, 1.081M] (log-range 0.95); the gen
slope over all three points spans [0.500M, 1.180M] (log-range 0.86). To remove the (small)
difference in fitting window, the gen two-point slope over its OUTER depths only (d1, d3) is
reported beside the three-point fit, and the **matched pair is the headline**.

**Prediction P1.** v8's continuation is more DIRECTED: `b_ALL(v8)` closer to 1, `b_ALL(gen)` ≈ 0.5,
matching our folds (0.48) and exploiters (0.548).

**Null / verdict rule.** `SIGNIFICANT` requires (a) the arm-bootstrap CIs of the two cells' mean
`b_ALL` to be DISJOINT, and (b) every one of v8's three arm slopes to exceed every one of gen's
(a complete separation — the only 3-vs-3 arrangement a permutation test can call extreme).
`WITHIN FLOOR` if the cells' CIs overlap and the point difference is smaller than the within-cell
arm spread. `NOT DETECTED` otherwise.

**⚠️ Registered before looking: with 3 arms per cell the smallest attainable two-sided permutation
p-value is 2/20 = 0.10.** A p < 0.05 is unreachable by design, so the verdict rule above is stated
as complete separation rather than as a p-value, and no result here can be called significant at
the 5% level. Three arms per cell is thin and the honest ceiling is "consistent / not consistent".

### P2 — the replicate cosine

`cos(Δθ_X, Δθ_Y)` for the three replicate pairs (A·B, A·C, B·C) at each matched depth, per group
and for ALL. Two arms differing only in seed are two draws of the same update process; their
cosine is a direct estimate of how much of the displacement is a SHARED direction rather than
noise. A pure random walk in P dimensions gives cosine ≈ 0 (±P^-1/2, i.e. ±0.0006 here); a purely
directed drift gives 1.

**Prediction P2.** v8's replicates agree MORE than ours. Our fold-depth reference is **0.56**
(`../fold_displacement`, the TCFUND/TCUNF replicate pairs).

**Null.** The two cells' three-pair means differ by less than the larger cell's own pair spread ⇒
`NOT DETECTED`.

**Registered as a caveat before computing:** cosine has a floor set by the parameters both arms
share by construction (e.g. every arm inherits the same Adam-scale asymmetries across layers), so
a non-zero cosine is not by itself evidence of directed *learning*; only the CONTRAST between the
two cells is interpretable, and only if the exponent agrees with it.

### P3 — the output-side twin

`KL(parent ‖ arm)` over LEGAL actions on that era's frozen 456-state batch
(`../sharing_kernel/states_gen.npz` for ours, `states_v8.npz` for v8 — **both already exist; no
states are regenerated and no battles are played**), fitted the same way: `log KL ∝ c · log t`.
For a locally-quadratic KL, `c ≈ 2b`, so this is the exponent's output-side twin and a consistency
check on P1: a parameter-space result that the output does not corroborate is a result about a
reparameterisation, not about behaviour. Reported all / taught / untaught, with a cluster
bootstrap over the 24 teams for the level and over arms for the exponent.

**Prediction P3.** `c(v8) > c(gen)`, and each cell's `c ≈ 2b` of its own P1.

### Confounds registered in advance

1. **Different code, different observation space, different architecture.** v8 is `b13b30b2` /
   obs 2992 / SB3 flat `action_net`; ours is `407b27c0` / obs 2501 / pointer head. The two `P`s
   are different numbers of parameters. A cross-era `|Δθ|` LEVEL is meaningless; only the
   EXPONENT and the COSINE — both scale-free — are compared.
2. **Different dose.** `python -m main.dose` (executed, table in RESULTS): ours runs at
   `4.557e-08` per env step (frozen lr 2.8e-5, effective batch 6,144, 10 epochs), v8's at
   `2.574e-08` (lr 1.205e-4, effective batch 32,768, 7 epochs) — **ours is 1.77× v8's**. A
   constant dose multiplier scales `|Δθ|` by a constant and therefore moves the log-log
   INTERCEPT, not the SLOPE; that is the reason `b` rather than `|Δθ|` is the statistic. This
   argument requires the dose to be constant IN t within each arm, which is checked and reported.
3. **Three arms per cell** — see the permutation floor above.
4. **Different self-play ecology.** Recorded before computing: G5's fork auto-seeded the parent's
   14-snapshot pool (`snapshots/pool_seed.json`), while the v8 arms started POOLLESS under era
   code and promoted their first snapshot only at ~fork+417k — so v8's first depth was trained
   largely against the BOT pool. This is an ecology difference between the cells that no
   parameter-space statistic can remove.
5. **A displacement exponent is a statement about the size of the update, not its usefulness.**
   Directed drift toward a worse policy would look identical to directed drift toward a better
   one. P3 is on the output, not on the meter; neither speaks to WIN RATE. The link to cell 2's
   +3.45pp is an inference, not a measurement.

### What each outcome would mean, stated in advance

* **v8 directed, ours diffusive** ⇒ SUPPORTS "the critic is the lever" — a mature critic yields a
  directed update while a young one yields a random walk — but does **not prove** it, because the
  hyperparameter and code/ecology confounds (1, 2, 4) are not controlled and would produce the
  same signature.
* **Both diffusive, or no detectable difference** ⇒ the noise-vs-drift account is **dead for this
  contrast**: whatever makes v8's continuation gain on the untaught meter, it is not that its
  updates accumulate more directionally than ours.

---

## RESULTS

Executed 2026-09-06 on this worktree (`nice -n 10`, BLAS pinned to 1 via each script's own
`os.environ.setdefault`, `torch.set_num_threads(4)` for the tensor ops — declared as the brief
requires). No battles, no training, no model was written. Wall clock: gen half **13 s**, v8 half
**7 s**, analysis <1 s. Everything below is re-derived by `python verify_readme.py`.

```
gen : PYTHONPATH=<worktree>/src nice -n 10 python drift.py --era gen --out drift_gen.json
v8  : cd /tmp/v8rep_era && PYTHONPATH=/tmp/v8rep_era/src PYTHONDONTWRITEBYTECODE=1 \
        nice -n 10 python <this dir>/drift.py --era v8 --out <this dir>/drift_v8.json
      (same pattern for popart_split.py)
      nice -n 10 python analyze.py ; nice -n 10 python tables.py > results_table.txt
```

### The dose, executed

`python -m main.dose` over all six arms (reference `ai_v8_14_distill3_0725`, `dose_rate=2.145e-08`):

```
run                                                 steps       eff.batch  epochs  lr_median  updates/step  dose_rate  vs ref
ai_v9_195_G5PLAINA_0906  [FROZEN; pinned 2.80e-05]  29,294,832  6,144      10      2.8e-05    0.001628      4.557e-08  2.12x
ai_v9_196_G5PLAINB_0906  [FROZEN; pinned 2.80e-05]  29,294,832  6,144      10      2.8e-05    0.001628      4.557e-08  2.12x
ai_v9_197_G5PLAINC_0906  [FROZEN; pinned 2.80e-05]  29,294,832  6,144      10      2.8e-05    0.001628      4.557e-08  2.12x
v8rep_p2self_A_0905                                 —           32,768     7       0.0001205  0.0002136     2.574e-08  1.20x
v8rep_p2self_B_0905                                 —           32,768     7       0.0001205  0.0002136     2.574e-08  1.20x
v8rep_p2self_C_0905                                 —           32,768     7       0.0001205  0.0002136     2.574e-08  1.20x
```

**Ours runs at 1.77× the v8 cell's dose rate.** The **dose is constant in t within each cell**,
which is what the exponent argument needs: G5 is `--fork-lr 2.8e-05 --fork-lr-freeze`
(`lr_frozen: true`, `kl_controller.phase: "frozen"`), and the v8 arms record
`current_lr = 1.204763827879769e-04` in the PARENT's metadata, in the checkpoint sidecar AND in
`final_model_interrupted`'s — the era's KL controller was parked at one value across the whole
measured window. So the dose difference moves the log-log intercept, not the slope.

### Depth grid actually measured

| cell | key | t (steps since parent) | absolute step | \|Δθ\| | rel | KL(parent‖arm) |
|---|---|---|---|---|---|---|
| v8 | A@d1 | 416,750 | 278,000,017 | 1.7659 | 0.00717 | 0.0337 |
| v8 | A@d3 | 1,080,663 | 278,663,930 | 2.7901 | 0.01132 | 0.0530 |
| v8 | B@d1 | 416,747 | 278,000,014 | 1.7699 | 0.00718 | 0.0342 |
| v8 | B@d3 | 1,081,344 | 278,664,611 | 2.7496 | 0.01116 | 0.0546 |
| v8 | C@d1 | 416,747 | 278,000,014 | 1.7623 | 0.00715 | 0.0311 |
| v8 | C@d3 | 1,086,364 | 278,669,631 | 2.9342 | 0.01191 | 0.0626 |
| gen | A@d1 / d2 / d3 | 500,016 / 1,000,032 / 1,179,648 | 28,615,200 / 29,115,216 / 29,294,832 | 1.9054 / 2.6385 / 2.8949 | 0.00822 / 0.01138 / 0.01249 | 0.0710 / 0.1090 / 0.1215 |
| gen | B@d1 / d2 / d3 | same | same | 1.9024 / 2.6151 / 2.8493 | 0.00821 / 0.01128 / 0.01229 | 0.0852 / 0.1246 / 0.1410 |
| gen | C@d1 / d2 / d3 | same | same | 1.9116 / 2.5998 / 2.8499 | 0.00825 / 0.01121 / 0.01229 | 0.0763 / 0.1141 / 0.1433 |

Parameter counts: v8 **3,512,397** policy parameters, gen **3,147,887** — so `|Δθ|` LEVELS are not
cross-era comparable and are never compared here.

Same-depth agreement on the v8 side (`final_model_interrupted.zip` vs that arm's one
`checkpoints/*.zip`), which is why the checkpoint is not entered as a second fit point: A and C are
**bit-identical** to the final despite recorded step gaps of 29 and 5,019, and B differs by a
relative **2.36e-03**.

### P1 — the displacement exponent (the headline)

Matched two-point windows (v8 d1→d3, gen d1→d3), per arm:

| group | v8 A / B / C | v8 mean [CI over arms] | gen A / B / C | gen mean [CI over arms] | v8−gen | verdict |
|---|---|---|---|---|---|---|
| **ALL** | 0.4800 / 0.4620 / 0.5321 | **0.4914** [0.4620, 0.5321] | 0.4873 / 0.4706 / 0.4652 | **0.4744** [0.4652, 0.4873] | **+0.0170** | **WITHIN FLOOR** |
| action_head | 0.4961 / 0.4746 / 0.5486 | 0.5064 | 0.5841 / 0.5811 / 0.5667 | 0.5773 | −0.0709 | separated — see caveat |
| encoders | 0.5322 / 0.4903 / 0.5729 | 0.5318 | 0.4896 / 0.4755 / 0.4702 | 0.4784 | +0.0534 | separated — see caveat |
| team_transformer | 0.4799 / 0.4620 / 0.5174 | 0.4865 | 0.4772 / 0.4634 / 0.4523 | 0.4643 | +0.0221 | WITHIN FLOOR |
| projection_mlp | 0.4804 / 0.4654 / 0.5389 | 0.4949 | 0.5408 / 0.5279 / 0.5271 | 0.5319 | −0.0370 | WITHIN FLOOR |
| belief_op | 0.5873 / 0.5353 / 0.6364 | 0.5863 | 0.4301 / 0.4233 / 0.4169 | 0.4235 | +0.1629 | separated — see caveat |
| critic | 0.1597 / 0.1766 / 0.1639 | 0.1667 | 0.4862 / 0.3939 / 0.3748 | 0.4183 | −0.2515 | separated → **see PopArt** |

The gen three-point OLS gives 0.4694 on ALL against 0.4744 on the outer two, so the fitting window
is not doing the work.

**🔴 P1 IS NOT DETECTED, and the direction of the point estimate is not the interesting part.**
`b_ALL` is **0.49 on v8 and 0.47 on ours** — both at the diffusive value, and both matching this
program's two prior readings on entirely different processes (our folds **0.48**, our exploiters
**0.548**). The pre-registered prediction was `b_ALL(v8) → 1`; the measured 0.4914 is 0.51 away
from that and 0.0086 away from a pure random walk. The arm-bootstrap intervals overlap heavily, the
permutation p is 0.700 (against a possible minimum of 0.100), and the +0.0170 difference is a
quarter of the within-cell arm spread (0.0701).

**⚠️ Read the per-group "separated" rows as EXPLORATORY, not as findings.** The two eras' groups
share a ROLE label but neither a composition nor a size: v8's `action_head` is a flat
`Linear(latent, 11)` of **5,643** parameters while ours is a pointer head of **55,683**; v8's
`belief_op` is 204,065 to our 512,267; v8's `critic` is 1,261,891 to our 742,650. A per-group
cross-era exponent difference is confounded by construction, and the permutation floor means each
of them reads p = 0.100 at best.

### P1 follow-up (NOT pre-registered) — the critic row is partly PopArt bookkeeping

`PopArtNormalizer.update` rescales exactly one module — `policy.value_net`, **513** parameters —
by `W *= σ_old/σ_new`, `b = (σ_old·b + μ_old − μ_new)/σ_new`, to preserve the de-normalized output.
No gradient is behind it, so it is the one mechanism that can move critic weights without learning,
and the critic row is exactly the row it can fake. The buffers show it moving: v8's `popart.mu`
swings −1.6077 → +1.36 at d1 and back to **+0.04 … +0.41** at d3 (non-monotone), while ours drifts
3.6440 → 3.23–3.43 with `popart.sigma` rising 13.80 → 14.38–14.93.

| era | PopArt share of critic squared displacement (d1 → d3) | `b` critic RAW | `b` critic **PopArt layer EXCLUDED** |
|---|---|---|---|
| v8 | **25.2–27.6% → 5.8–8.6%** | 0.1667 | **0.2883** (0.2654 / 0.3150 / 0.2846) |
| gen | 0.10–0.82% → 0.12–0.35% | 0.4183 | **0.4196** (0.4889 / 0.3938 / 0.3763) |

The PopArt layer ALONE fits `b = −0.53` on v8 (it moves out and comes back) and ~0.16 on ours, so
roughly **half of the raw critic gap was bookkeeping**. Corrected, the gap survives — 0.288 vs
0.420, complete separation (min gen arm 0.376 > max v8 arm 0.315) — and reads: **v8's mature critic
accumulates displacement SUB-diffusively while our young one is near-diffusive**, i.e. the mature
critic is closer to done moving. That is the only row here that points at the critic at all, but it
is not the pre-registered claim (which was that the whole update is DIRECTED), it inherits the
group-composition confound above, and its p-value floor is 0.100.

### P2 — the replicate cosine: the prediction is not merely unmet, it is REVERSED

Random-walk floor `1/√P` is **5.34e-04** (v8) / **5.64e-04** (gen); the fold-depth reference is
**0.56**.

| group | v8 @ d3 (A·B / A·C / B·C) | v8 mean | gen @ d3 (A·B / A·C / B·C) | gen mean | v8 − gen |
|---|---|---|---|---|---|
| **ALL** | 0.2219 / 0.2221 / 0.2153 | **0.2198** | 0.4102 / 0.3995 / 0.4020 | **0.4039** | **−0.1841** |
| action_head | 0.1647 / 0.2126 / 0.1878 | 0.1884 | 0.1687 / 0.1510 / 0.1556 | 0.1584 | +0.0300 |
| encoders | 0.3656 / 0.3619 / 0.3693 | 0.3656 | 0.4370 / 0.4361 / 0.4323 | 0.4351 | −0.0695 |
| team_transformer | 0.4161 / 0.4120 / 0.4020 | 0.4100 | 0.5154 / 0.5103 / 0.5101 | 0.5119 | −0.1019 |
| projection_mlp | 0.1025 / 0.1065 / 0.0981 | 0.1024 | 0.0717 / 0.0606 / 0.0641 | 0.0655 | +0.0369 |
| belief_op | 0.8058 / 0.7881 / 0.7782 | 0.7907 | 0.7299 / 0.7239 / 0.7253 | 0.7263 | +0.0644 |
| critic | 0.5138 / 0.5388 / 0.5642 | 0.5389 | 0.5823 / 0.5599 / 0.5734 | 0.5719 | −0.0330 |

**🔴 OUR replicates agree MORE than v8's** — 0.404 vs 0.220 on ALL, with complete separation (all
three of our pairs above all three of theirs) at the permutation floor p = 0.100. The
pre-registration predicted `v8 > gen`. Two things hold this to "the prediction is not supported"
rather than "the reverse is established": the permutation floor makes 0.100 the best attainable p
at 3 pairs vs 3 pairs (and three pairs from three arms are not independent), and **registered
confound 4 acts directly here** — our arms all seeded the SAME 14-snapshot parent pool while the v8
arms each started poolless and built their own opponent distribution, so their three trajectories
face three different ecologies by construction and would be expected to agree less for a reason
that has nothing to do with critic maturity.

The one thing both cells agree on: at t ≈ 1M, three seeds of the SAME recipe on the SAME parent
share only **22–40%** of their displacement direction. Most of a continuation's movement is
seed-specific.

### P3 — the output-side twin

`KL(parent‖arm) ∝ t^c` over the same two depths, on each era's own frozen 456-state batch (both
batches already existed in `../sharing_kernel/`; nothing was regenerated):

| slice | v8 A / B / C | v8 mean [CI] | gen A / B / C | gen mean [CI] | v8−gen | verdict |
|---|---|---|---|---|---|---|
| all | 0.4762 / 0.4917 / 0.7311 | 0.5663 [0.4762, 0.7311] | 0.6259 / 0.5872 / 0.7347 | 0.6493 [0.5872, 0.7347] | −0.0829 | **WITHIN FLOOR** |
| taught | 0.4994 / 0.3950 / 0.6892 | 0.5278 | 0.7396 / 0.5239 / 0.7192 | 0.6609 | −0.1331 | WITHIN FLOOR |
| untaught | 0.4318 / 0.6963 / 0.8137 | 0.6473 | 0.3884 / 0.7512 / 0.7821 | 0.6406 | +0.0067 | WITHIN FLOOR |

KL levels with a cluster bootstrap over the 24 teams, at d3 — reported for provenance only, since
levels are cross-era incomparable (different obs, different heads): v8 A **0.0530** [0.0457,
0.0605] · B 0.0546 [0.0464, 0.0634] · C 0.0626 [0.0536, 0.0719]; gen A **0.1215** [0.1057, 0.1383]
· B 0.1410 [0.1235, 0.1587] · C 0.1433 [0.1197, 0.1688].

**The output twin CORROBORATES P1's null and adds nothing against it**: both cells sit at
`c ≈ 0.57–0.65` with the arm intervals overlapping almost completely, and if anything OUR output
moves slightly faster per unit t. `c/b` is **1.15 (v8)** and **1.37 (gen)**, both below the 2.0 a
locally-quadratic KL would give — in BOTH eras a growing share of the parameter displacement lands
in directions the output is insensitive to. That is a shared property of continuation, not a
discriminator.

### E1 — EXPLORATORY (not pre-registered): within-arm direction persistence

`cos(Δθ_{t1}, Δθ_{t2})` inside ONE arm. A pure random walk predicts exactly `√(t1/t2)`; a pure
directed drift predicts 1. Independent of both the exponent fit and the replicate cosine, and the
sharpest single reading here:

| era | arm | √(t1/t2) | ALL | action_head | encoders | team_tf | proj_mlp | belief_op | critic |
|---|---|---|---|---|---|---|---|---|---|
| v8 | A | **0.6210** | **0.6472** | 0.6783 | 0.6671 | 0.6564 | 0.6440 | 0.6716 | 0.6292 |
| v8 | B | **0.6208** | **0.6555** | 0.6587 | 0.6751 | 0.6669 | 0.6546 | 0.7024 | 0.5156 |
| v8 | C | **0.6194** | **0.6043** | 0.6200 | 0.6263 | 0.6246 | 0.6003 | 0.6407 | 0.5336 |
| gen | A | **0.6511** | **0.6533** | 0.6599 | 0.6582 | 0.6864 | 0.6352 | 0.6497 | 0.7081 |
| gen | B | **0.6511** | **0.6511** | 0.6499 | 0.6660 | 0.6976 | 0.6410 | 0.6445 | 0.6543 |
| gen | C | **0.6511** | **0.6502** | 0.6541 | 0.6583 | 0.6870 | 0.6394 | 0.6444 | 0.6777 |

**Both eras land on the random-walk prediction on the ALL column**, and ours lands on it almost
exactly: our three arms are 0.6533 / 0.6511 / 0.6502 against a prediction of 0.6511 — within
**0.0022**, agreement to three decimal places. v8's are 0.6472 / 0.6555 / 0.6043 against 0.6210 —
within **0.035**, slightly ABOVE the diffusive prediction on two arms and below it on the third.

Per GROUP the agreement is looser and worth stating honestly rather than rounding away: deviations
from `√(t1/t2)` run from **−0.105** (v8 arm B's critic, i.e. *less* persistent than a random walk)
to **+0.082** (v8 arm B's `belief_op`) on the v8 side, and from −0.016 to **+0.057** on ours. But
the comparison that matters is against the two hypotheses, and it is not close: the largest
observed cosine anywhere in the table is 0.7081, so **every group in both eras is at least 2.7×
closer to the diffusive prediction than to the directed-drift prediction of 1.0** (worst distance
to `√(t1/t2)` = 0.105; smallest distance to 1.0 = 0.292). Nothing here separates a mature
continuation from a young one; to the resolution available, both are random walks.

---

## VERDICT

| prediction | result | verdict |
|---|---|---|
| **P1** `b_ALL(v8)` closer to 1 than ours | 0.4914 [0.4620, 0.5321] vs 0.4744 [0.4652, 0.4873]; Δ +0.0170 against a 0.0701 arm-spread floor; perm p 0.700 | **NOT DETECTED** (WITHIN FLOOR by the registered rule) |
| **P2** v8 replicates agree more | 0.2198 vs 0.4039 — **the opposite direction**; complete separation at the p = 0.100 floor; confounded by the pool-seeding difference | **NOT SUPPORTED** (direction reversed) |
| **P3** `c(v8) > c(gen)` | 0.5663 [0.4762, 0.7311] vs 0.6493 [0.5872, 0.7347]; Δ −0.0829 against a 0.2942 floor | **WITHIN FLOOR** |
| E1 (exploratory) | both eras match `√(t1/t2)` on ALL (gen to 0.0022, v8 to 0.035); every group is ≥2.7× closer to that prediction than to 1.0 | consistent with a random walk on BOTH sides |

**🔴 THE NOISE-VS-DRIFT ACCOUNT IS DEAD FOR THIS CONTRAST.** A plain 1.08M-step continuation of
v8's 277M-step parent accumulates parameter displacement as `t^0.49`, and one of ours off a 28M-step
parent as `t^0.47`. Three independent estimators — the displacement exponent, the output-side KL
exponent, and the within-arm direction persistence — agree; the one statistic where the cells
separate cleanly, the replicate cosine, separates in the direction OPPOSITE to the prediction.
Whatever makes v8's continuation gain +3.45pp on the untaught meter while ours has never been shown
to, **it is not that its updates accumulate more directionally than ours.** The learning note's §4
candidate mechanism should be marked refuted for this pair.

Because P1 failed, the conditional the brief attached to a positive result does not apply: this
does **not** support "the critic is the lever", and it was never in a position to prove it — the
hyperparameter, code and ecology confounds below would have produced the same signature.

**The one row that still names the critic** is a by-product and is offered as a hypothesis, not a
result: after removing PopArt's non-learning rescale, **v8's critic accumulates sub-diffusively
(b = 0.288; arms 0.265 / 0.315 / 0.285) where ours is near-diffusive (b = 0.420; arms 0.489 / 0.394
/ 0.376)**, with complete separation. Read it as "the mature critic has largely stopped moving",
never as "the mature update is directed" — the whole-network exponent says the update is a random
walk in BOTH cells. It carries the group-composition confound (v8's critic group holds 1,261,891
parameters to our 742,650, with different heads in it) and the 0.100 permutation floor.

### Caveats, restated against what was actually measured

1. **Different code, obs space, architecture.** `b13b30b2` / obs 2992 / flat `action_net` versus
   `407b27c0` / obs 2501 / pointer head; 3,512,397 vs 3,147,887 policy parameters. Only the
   scale-free statistics (`b`, `c`, cosines) were compared; every per-GROUP cross-era row is
   additionally confounded by group composition and must not be quoted as a finding.
2. **Different dose, 1.77× in our favour.** Constant in t within each cell (verified from the
   metadata), so it shifts the intercept and not the slope — but a nonlinearity in the
   dose→displacement relation would break that argument, and this cell cannot test it.
3. **Three arms per cell, and only TWO depths on the v8 side.** The exact permutation floor is
   p = 2/20 = 0.100; nothing here can be significant at 5%. The v8 per-arm slopes have no residual
   (two points determine a line), so all v8 uncertainty comes from the three arms.
4. **Different self-play ecology.** G5 auto-seeded the parent's 14 snapshots
   (`snapshots/pool_seed.json`); the v8 arms started poolless under era code and promoted their
   first snapshot only at ~fork+417k, so their first depth was trained largely against the BOT
   pool. This bears directly on P2 and cannot be removed post hoc.
5. **This measures the SIZE and DIRECTION of the update, never its usefulness.** Directed drift
   toward a worse policy would look identical to directed drift toward a better one, and no
   statistic here touches win rate. The link to cell 2's +3.45pp is an inference.
6. **The v8 arms' argv still carries `--distill-teacher` and `--distill-team-bias 0.4`** beside
   `--distill-coef 0`. At the era pin `args._distill_pairs` is populated only when
   `distill_coef > 0`, so both the loss and the team bias were off — the ledger's cell-2 entry
   states this and the arms printed the ordinary `full pool + 10% sample-team bias`. Recorded here
   because the argv alone reads otherwise.

### Instrument findings (reported as findings, per the brief)

* **A v8-era `final_model_interrupted.zip` can carry a LARGER `num_timesteps` than the checkpoint
  written beside it while holding bit-identical weights** — arm C's differ by 5,019 recorded steps
  with `|Δθ|` equal to within 1e-16 relative, arm A's by 29. The interrupted save stamps the
  env-step counter at save time, not the last optimiser step. Anything using a v8-era interrupted
  final's step as an x-axis inherits up to ~0.5% of slop at this depth.
* **`metadata.json` on the v8-era runs has no top-level `num_timesteps`** (the key postdates them),
  so every step here was read from the SB3 zip's own `data` blob and cross-checked against the
  declared parent step, with a hard exit on disagreement.
* **PopArt is a live confounder for any parameter-space statistic on the critic**, and it is
  concentrated: 513 parameters carrying up to **27.6%** of the critic group's squared displacement
  on the v8 side. Any future `|Δθ|` work touching the critic should split `value_net.*` out first;
  `popart_split.py` is reusable for exactly that.
* **The v8 cell has no third depth and cannot get one** without training. The arms' only
  intermediate artefact is the single self-play snapshot they promoted at ~+417k (the same file also
  written to `best_model/` and `eval_traces/step_*/snapshot.zip`), so a two-point slope is the most
  the disk supports on that side.

### Files

| file | what |
|---|---|
| `drift.py` | the measurement — per-group `\|Δθ\|`, replicate cosines, `KL(parent‖arm)`; era-parameterised, run once per era |
| `popart_split.py` | the not-pre-registered follow-up: `value_net.*` vs the rest of the critic |
| `analyze.py` | pure-numpy statistics: exponents, arm bootstraps, exact permutations, cluster bootstrap, E1 |
| `tables.py` | renders `analysis.json` + the two `popart_split_*.json` into `results_table.txt`; reads, never recomputes |
| `verify_readme.py` | re-derives every number quoted above from the JSON and fails on any mismatch |
| `drift_gen.json` · `drift_v8.json` · `popart_split_gen.json` · `popart_split_v8.json` · `analysis.json` · `results_table.txt` | the artefacts |
