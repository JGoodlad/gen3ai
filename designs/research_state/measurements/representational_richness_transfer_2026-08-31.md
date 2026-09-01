# PROBE M2 — representational richness as the transfer vehicle (2026-08-31)

**Question.** The plasticity audit recorded, and nobody chased, two numbers: the gen era runs at
**half v8's participation ratio at equal width (20.6 vs 50.2)**, and v8's teacher deltas were
**TRUNK-heavy (0.47) vs the gen era's (0.28)**. The hypothesis under test: a teacher's content must
land somewhere; a rich representation can express it as a combination of directions the student
already has (composes ⇒ radiates off-slice), an impoverished one can only express it by overwriting
(local by construction ⇒ "robbery").

**Companion data:** `representational_richness_transfer_2026-08-31.json`.
**Scripts:** `representational_richness_transfer_{locus,forward,analyze}.py` (this directory).

---

## 0. Headline, before the detail

**Prediction 1 PASSES on the two taps it was written from and FAILS on six of eight. Prediction 2
passes on ORDERING and is uninformative on inspection. And two things were measured that
substantially DEMOTE the hypothesis as a causal account:**

1. **The ~2× richness gap is already present in the INPUT.** The raw observation matrix's own
   participation ratio is **36.19 (v8) vs 17.06 (rev-1)** — the same factor, before any network.
   The network's *amplification over its own input* is similar in both eras (1.39 vs 1.20). Under a
   scale-invariant reading the two eras' inputs are **equally rich to three decimal places**
   (PR/live-dim 0.0463 both). The gen era is not running an impoverished network; it is running a
   comparable network on a state description whose **variance** is concentrated in fewer directions.
2. **v8's own gifting fold compressed the representation MORE than any robbing gen fold does**
   (`pi_features` PR −4.60 [−5.45,−2.20] vs the gen folds' −1.00 [−1.26,−0.55]). The corollary
   "gifting folds preserve the representation" is refuted outright.

What survives is narrower and more actionable than what was proposed: **within the gen lineage the
input's participation ratio collapsed on a datable schedule** — obs PR 37.76 at gen-12 → 22.88 at
gen-13 → 16.20 at gen-14 — bracketing the H-B event-window addition and the frame deletion, and
landing the era at less than half the value it held one generation before v8-parity was lost.

---

## 1. Method

**Read-only.** No battles, no training, nothing under `models/` written. Every number is a forward
pass or a `state_dict` arithmetic over archived checkpoints, CPU, `nice -n 15`, ≤2 threads.

**Estimator.** `agents.training.rank_metrics.effective_rank` — the project's canonical participation
ratio `(Σσ²)²/Σσ⁴`, i.e. the *same function* that produced the audit's 50.24 / 20.59. A fast
covariance-eigenvalue PR is used inside the bootstrap and is asserted equal to the canonical one
before use (agreement **1.07e-14**).

**State sets.** The 2026-08-28 audit's own shared sets, reused byte-for-byte: n = 3000 decisions
pooled from each PARENT's eval_traces at the step nearest its fork point (`ai_v8_04/step_276000000`,
201 files, 95 teams; `ai_v9_29/step_24000000`, 229 files, 81 teams). The audit's acid test — forward
the snapshot the trace itself shipped, reproduce the recorded logits — passed at max |Δ| 1.07e-05
(v8) / 1.91e-05 (gen), so the reconstructed Dict obs is faithful. Hook repeats were re-verified
byte-identical here as a **value** (max repeat diff **0.0** over all 112 tap dumps), not merely as a
silent branch.

**Era-pinned loads.** Current code cannot load a 2992-dim v8 obs. All v8-side forwards ran under
`/tmp/probeP_v8era` pinned to `b13b30b2` with `PYTHONPATH` pointed at the era `src/` so it beats the
editable install — the path `v8_redistribution_pfsp_2026-08-30` validated at logit r = 0.982.

**Uncertainty.** Cluster bootstrap over the state set's `src_file` (one trace file ≈ one battle),
400 resamples. Differences between two models forwarded on the SAME states are bootstrapped
**paired**, because resampling battles moves both arms together and the unpaired intervals are
consequently much wider than the contrast deserves. Cross-ERA differences are deliberately never
bootstrapped: the two eras have different observations, different parents and different state sets,
and cannot be paired at all.

---

## 2. The richness table — participation ratio per checkpoint, per tap

All taps are the analogous module in both eras; the width is printed because "matched width" is
half the prediction. Bold = the two taps the audit's 50.2/20.6 headline was measured at.

| model | enc 1536 | trunk 768 | **proj 512** | pool 128 | **pi 512** | vf 512 | pnet 512 | vnet 512 |
|---|---|---|---|---|---|---|---|---|
| **v8 PARENT** `ai_v8_04` @277.2M | 36.24 | 42.62 | **30.08** | 22.03 | **50.24** | 3.85 | 26.88 | 2.50 |
| **v8 FOLD** `ai_v8_14` @292.1M | 36.99 | 43.15 | **30.15** | 20.80 | **45.64** | 4.05 | 29.41 | 2.98 |
| v8 fork semistall3 @matched | 36.73 | 43.83 | 30.50 | 21.55 | 49.76 | 4.13 | 29.39 | 2.87 |
| v8 fork pool10 @matched | 36.96 | 42.98 | 29.91 | 21.36 | 49.43 | 4.05 | 29.79 | 2.59 |
| v8 fork defensive10 @matched | 36.31 | 43.37 | 30.18 | 21.26 | 49.39 | 4.09 | 28.57 | 2.60 |
| **gen PARENT** rev-1 @25.1M | 38.86 | 30.26 | **13.52** | 19.07 | **20.59** | 3.24 | 29.70 | 3.72 |
| **gen FOLD** rev-2 R2ACTION @28.1M | 38.41 | 30.31 | **13.70** | 17.61 | **19.59** | 3.69 | 29.23 | 4.14 |
| **gen FOLD** rev-3 R3ACTION @32.6M | 38.39 | 31.15 | 13.07 | 16.91 | 19.05 | 3.76 | 28.07 | 4.25 |
| **gen FOLD** rev-4 R4ACTION | 38.60 | 30.81 | 13.49 | 16.79 | 18.90 | 3.43 | 28.19 | 3.87 |
| **gen FOLD** COMPFOLD | 38.24 | 30.30 | 13.48 | 17.06 | 19.04 | 3.82 | 28.28 | 4.31 |
| gen CTRL R2CTRL (no fork) | 38.31 | 30.66 | 14.85 | 19.45 | 22.89 | 3.85 | 32.33 | 4.38 |
| gen CTRL R2PLAIN (no fork) | 38.54 | 30.90 | 14.00 | 19.88 | 19.77 | 3.72 | 29.29 | 4.30 |
| gen fork F5a | 38.81 | 29.17 | 13.37 | 18.72 | 19.84 | 3.59 | 26.95 | 4.07 |
| gen fork F5c | 39.06 | 30.43 | 13.73 | 18.80 | 20.61 | 3.71 | 28.12 | 4.16 |

**Reproduction check.** The parent rows reproduce the audit's published `projection` 30.08 / 13.52
and `pi_features` 50.24 / 20.59 to the decimal, on the same state sets through a re-derived path.
The scale is therefore the audit's.

**The era ratio, tap by tap (v8 parent ÷ gen parent):**

| tap | v8 | gen | ratio | ≥2×? |
|---|---|---|---|---|
| encoder (per-mon, 1536) | 36.24 | 38.86 | **0.93** | INVERTED |
| trunk tokens (768) | 42.62 | 30.26 | 1.41 | no |
| **trunk projection (512)** | 30.08 | 13.52 | **2.22** | **YES** |
| pooled cls (128) | 22.03 | 19.07 | 1.16 | no |
| **policy head input (512)** | 50.24 | 20.59 | **2.44** | **YES** |
| value head input (512) | 3.85 | 3.24 | 1.19 | no |
| policy head mlp out (512) | 26.88 | 29.70 | **0.91** | INVERTED |
| value head mlp out (512) | 2.50 | 3.72 | **0.67** | INVERTED |

**The richness gap is LOCALIZED to the policy-side trunk** (`projection` → `pi_features`), and is
absent or reversed at the input encoder, at both mlp heads and on the whole value path. The
"20.6 vs 50.2" headline is a true statement about two taps, not a property of the representation.

Interval estimates (cluster bootstrap over trace files, 400 reps; unpaired, so wide by
construction — the eras are nonetheless disjoint by a wide margin):

| model | `projection` | `pi_features` | `cls_pool` |
|---|---|---|---|
| v8 parent | 30.08 [24.15, 31.05] | 50.24 [41.14, 51.00] | 22.03 [16.78, 23.18] |
| v8 fold | 30.15 [24.45, 30.92] | 45.64 [39.00, 47.00] | 20.80 [16.84, 22.01] |
| gen parent (rev-1) | 13.52 [12.53, 13.80] | 20.59 [18.95, 20.98] | 19.07 [15.86, 19.61] |
| gen fold rev-2 | 13.70 [12.64, 14.02] | 19.59 [18.08, 20.01] | 17.61 [14.84, 18.03] |
| gen fold rev-3 | 13.07 [12.01, 13.68] | 19.05 [17.33, 19.58] | 16.91 [15.29, 17.22] |

---

## 3. 🔴 THE GAP IS INHERITED FROM THE INPUT

> ⚠️ **SUPERSEDED 2026-08-31 by `obs_conditioning_2026-08-31.md` (probe M8), which ran §12's own
> recommended next test.** The numbers below reproduce (M8 measures 36.22 / 38.39 / 23.39 / 16.11
> against 36.19 / 37.76 / 22.88 / 16.20) — but **433 of the observation's 2023 live columns are raw
> dex numbers the extractor casts with `.long()` into `nn.Embedding`, and they carry 99.993% of the
> raw variance**, so `PR(full obs) ≈ PR(ID columns only)` to two decimals. On the 1,590 columns the
> network reads as magnitudes the PR is **flat across the whole ladder** (45.14 gen-12 → 44.62
> gen-14 → 45.70 rev-1) and *rises* per live dimension, with a total dynamic range of 5.6×.
> **There was no input richness collapse.** The datable schedule in §3.1 is real and M8 §3
> attributes it completely — the event window added 704 dims of repeated species/move dex numbers,
> the frames removed 1113, and the six surviving blocks' PR never moved — but it is a fact about
> Showdown's numbering, not about the state description. Read §3/§3.1 as the measurement that led
> to the right question, not as a finding.

The participation ratio of a representation is a property of *(model, state distribution)*. The two
eras cannot share a state set, so the null that has to be ruled out is: **is the input itself half
as rich?** It is. The observation matrix, with no model involved at all:

| era / state set | obs dim | live dims | **PR (covariance)** | PR (signed-log) | PR (correlation) | PR_corr / live |
|---|---|---|---|---|---|---|
| v8 (`ai_v8_04` traces) | 2992 | 2249 | **36.19** | 23.40 | 104.06 | **0.0463** |
| gen (`ai_v9_29` traces) | 2501 | 2004–2039 | **17.06–17.41** | 11.95–12.19 | 92.76 | **0.0463** |

- On the **variance-weighted** readings — raw covariance (2.11×) and signed-log (1.92×) — the input
  gap is the same size as the representation gap.
- On the **scale-invariant** reading (per-column z-score, i.e. the correlation matrix) the two eras
  are **equal**: 0.0463 effective directions per live dimension, both. The information is not
  missing; its **variance is concentrated differently.**
- Model amplification over its own input, `pi_features` PR ÷ obs PR: **v8 1.39, gen 1.20.** At
  `projection`: 0.83 vs 0.79 — indistinguishable.

**A network's output covariance is dominated by its high-variance inputs, so the variance-weighted
reading is the apt null for a downstream PR** — and against that null the gen-era trunk is doing
about as well with what it is given as v8's did. This is the single most consequential thing this
probe measured, and it points the lever away from "make the trunk richer" and at the observation's
own variance structure.

### 3.1 Where it went — model-free, and datable

Same statistic, run over each generation's own `eval_traces` (no checkpoint loaded, so architecture
drift is irrelevant):

| run | obs dim | **PR (cov)** | PR (signed-log) | PR (corr) | PR_corr / live |
|---|---|---|---|---|---|
| v8_04 (v8 era) | 2992 | 36.19 | 23.40 | 104.06 | 0.0463 |
| **gen-12** (frames LIVE) | 2921 | **37.76** | 26.01 | 131.13 | 0.0601 |
| **gen-13** (frames LIVE, +event window) | 3529 | **22.88** | 17.80 | 130.84 | 0.0470 |
| **gen-14** (frames DELETED) | 2501 | **16.20** | 12.50 | 94.46 | 0.0467 |
| gen-15 | 2501 | 15.59 | — | — | — |
| gen-17 | 2501 | 18.35–19.04 | 12.89 | 98.95 | 0.0490 |
| rev-1 | 2501 | 17.06–17.41 | 11.95 | 92.76 | 0.0463 |

**gen-12 MATCHED v8** (37.76 vs 36.19) at a comparable obs dim. The collapse is internal to the gen
lineage and brackets two known changes: the H-B event-window addition (gen-12→gen-13: PR_corr/live
0.0601 → 0.0470, total held while 608 sparse dims were added) and the frame deletion (gen-13→gen-14:
per-dim spread held, covariance PR fell with the dims).

⚠️ **Six different runs means six different policies generating the states**, so this row set
confounds architecture with state distribution and cannot on its own attribute the drop to either
change. It is a dated bracket, not an attribution. The v8_04-vs-rev-1 coincidence at
PR_corr/live = 0.0463 across entirely different eras is either an invariant or a coincidence, and
this probe cannot tell which.

⚠️ It also connects to a rule this tree already wrote down. `gen3_frame_deletion_v1` was licensed on
a dV ablation, and `design_frame_deletion_coverage_gaps.md` records that **a dV ablation says
whether the model LEANS on a block and cannot say whether each FACT in it has a home elsewhere**.
Here is a third reading that neither test performs: **what share of the observation's effective
dimensionality the block was carrying.** Whether that mattered is not established by this probe.

---

## 4. The locus table — where a FOLD's parameter delta lands

Estimator verbatim from the audit's Phase A (same group map, same SB3 triple-copy dedupe, same
share-of-total-‖ΔW‖²). Trunk = G2 (`projection` + `team_transformer` + norms); shared = G2+G5
(+`mlp_extractor`); head = G4 (aux/belief heads) + G6 (action/value head).

**The audit's 0.47/0.28 pair is a FORK statistic. Nobody had measured it on the FOLDS** — which is
the object the untaught-externality outcome is a property of. The fork rows below reproduce the
published means (v8 0.465, gen 0.278) and fix the scale.

| pair | kind | **trunk** | shared | head | ‖Δ‖/‖W‖ | untaught externality |
|---|---|---|---|---|---|---|
| **v8_14 − v8_04** | **FOLD** | **0.381** | 0.658 | 0.149 | 0.0593 | **+5.42pp** [+3.44,+7.42] |
| **R3ACTION − R2ACTION** | **FOLD** | **0.336** | 0.511 | 0.355 | 0.0322 | **−0.75pp** [−4.56,+3.00] |
| **R2ACTION − rev-1** | **FOLD** | **0.330** | 0.493 | 0.372 | 0.0394 | **−7.06pp** [−10.56,−3.50] |
| R4ACTION − R2ACTION | FOLD | 0.338 | 0.511 | 0.352 | 0.0331 | not measured |
| COMPFOLD − R2ACTION | FOLD | 0.339 | 0.515 | 0.346 | 0.0320 | not measured |
| *R2CTRL − rev-1* | *no-fold CTRL* | *0.327* | *0.490* | *0.372* | *0.0370* | — |
| *R2PLAIN − rev-1* | *no-fold CTRL* | *0.333* | *0.495* | *0.371* | *0.0371* | — |
| v8 fork semistall3 | fork anchor | 0.450 | 0.734 | 0.128 | 0.0206 | — |
| v8 fork pool10 | fork anchor | 0.471 | 0.760 | 0.105 | 0.0210 | — |
| v8 fork defensive10 | fork anchor | 0.473 | 0.768 | 0.102 | 0.0209 | — |
| gen forks F5a–e | fork anchor | 0.263–0.291 | 0.370–0.413 | 0.458–0.508 | 0.0253–0.0268 | — |

Every pair shares 100% of its keys with its parent (`n_parent_only = n_child_only = 0`), so no delta
is a shape artifact.

⚠️ **The HEAD column is era-asymmetric and should not be read across eras.** v8's G6 is a
5.6k-parameter flat `action_net`; the gen era's is a 55k-parameter structured `pointer_head`, and
G4 holds 6.9% of v8's parameters against 19.7% of the gen era's — with the gen aux heads still
actively learning (the no-fold control's G4 share is 0.345, i.e. most of that mass is ordinary
training). The audit issued this warning about its own table and it applies unchanged here. The
**trunk** column is the era-comparable one.

---

## 5. The ordering against untaught externality

Three folds have a measured untaught-team externality. **No coefficient is fitted to three points**;
what is asked is whether the ordering is consistent.

| fold | parent's `pi` PR | fold trunk share | fold head share | **untaught externality** |
|---|---|---|---|---|
| v8_14 | **50.24** | **0.381** | 0.149 | **+5.42pp** [+3.44,+7.42] |
| R3ACTION | 19.59 | 0.336 | 0.355 | **−0.75pp** [−4.56,+3.00] |
| R2ACTION | 20.59 | 0.330 | 0.372 | **−7.06pp** [−10.56,−3.50] |

**Prediction 2's ordering HOLDS: 0.381 > 0.336 > 0.330 against +5.42 > −0.75 > −7.06.**

**And it is uninformative, for a reason visible in the table above it.** The rev-2/rev-3 separation
is 0.006 in trunk share — smaller than the gap between the two no-fold CONTROLS (0.327 vs 0.333),
which by construction carry no teacher content at all and no externality claim. Every gen fold sits
inside the control band. So the ordering has **one real degree of freedom, not three**: it is the
era gap (0.381 vs ~0.335) restated, and the within-era half of it is noise. Parent richness behaves
the same way — 19.59 vs 20.59 orders the *wrong* way against the two gen folds' externalities and is
rescued only by rounding to "both ≈20".

The honest statement is: **nothing measured here distinguishes a robbing fold from a null fold**,
and the ledger's rev-4 scorecard has since recorded that rev-3's −0.75 was probably floor exhaustion
rather than innocence — which, if right, means the middle row is not a distinct outcome and the
ordering reduces to two points.

---

## 6. The representation-level test — does the fold's content fit in directions the parent already has?

The parameter-level locus cannot be read across eras at the head. These two statistics are computed
on **features over the same states**, so they compare behaviour, not two differently-shaped modules.

- `cka_distance` — 1 − linear CKA(parent tap, fold tap).
- `energy_in_parent_top-k` — of the feature delta's squared norm, the share lying inside the
  **parent's own** top-k principal directions. This is the hypothesis stated as a number. Reported at
  **fixed k (21 and 51)** because the natural choice k = ⌈parent PR⌉ differs across eras (51 vs 21)
  and confounds the comparison outright.
- `ΔPR / parent PR` — k-free: how many more effective directions the change uses than the
  representation it is written into. 1.0 = exactly commensurate.

### `pi_features` (the policy head's input — the object the hypothesis is about)

| model | ckaD | in parent top-21 | in parent top-51 | ΔPR/parentPR |
|---|---|---|---|---|
| **v8 FOLD** | 0.1021 | **0.318** | **0.479** | **1.35** |
| gen FOLD rev-2 | 0.0668 | 0.239 | 0.364 | 2.23 |
| gen FOLD rev-3 | 0.0546 | 0.239 | 0.355 | 2.30 |
| gen FOLD rev-4 | 0.0543 | 0.238 | 0.364 | 2.30 |
| gen FOLD COMPFOLD | 0.0528 | 0.229 | 0.345 | 2.12 |
| *gen CTRL R2CTRL* | *0.0626* | *0.210* | *0.332* | *2.30* |
| *gen CTRL R2PLAIN* | *0.0546* | *0.208* | *0.328* | *2.27* |

At matched k, v8's fold delta is the most concentrated in its parent's leading directions, and its
change is dimensionally commensurate with the representation (1.35) where every gen change spreads
over ~2.2–2.3× as many directions as its parent actually uses. **This is the one place the
hypothesis's mechanism reads cleanly.**

### `projection` (the trunk) — and it goes the other way

| model | in parent top-21 | in parent top-51 | ΔPR/parentPR |
|---|---|---|---|
| v8 FOLD | **0.275** | **0.478** | 1.31 |
| gen FOLD rev-2 | 0.397 | 0.559 | 1.64 |
| gen FOLD rev-3 | 0.387 | 0.542 | 1.51 |
| *gen CTRL R2PLAIN* | *0.350* | *0.515* | *1.69* |

At the trunk the **gen** deltas are more concentrated in the parent's leading directions than v8's.
At `mlp_extractor.policy_net` the k-free statistic inverts again (v8 4.96 vs gen 3.33–3.70).
**The statistic does not order consistently across taps, so it does not support the hypothesis
cleanly** — and in every gen row the FOLDS are indistinguishable from the no-fold CONTROLS, so
nothing on this axis is a property of distillation in the gen era.

---

## 7. Maturity — does richness rise with training, and where does the fleet's base sit?

One FIXED state set per era, so the only thing varying down each column is the checkpoint.

| rev-1 (fresh generation, gen era) | 2.40M | 4.80M | 8.05M | 12.10M | 16.00M | 20.07M | 24.99M |
|---|---|---|---|---|---|---|---|
| `projection` | 6.83 | 9.57 | 11.55 | **13.61** | 13.67 | 13.22 | 13.55 |
| `pi_features` | 11.17 | 16.20 | 18.83 | **20.82** | 20.90 | 19.25 | 20.69 |
| trunk tokens | 19.27 | 24.03 | 26.07 | 28.75 | 29.97 | 28.88 | 30.26 |

| v8 lineage (era code) | v8_03 @149.6M | v8_03 @200.4M | v8_03 @267.6M | v8_04 @269.7M | v8_04 @277.2M |
|---|---|---|---|---|---|
| `projection` | 30.65 | 29.60 | 29.97 | 30.63 | 30.08 |
| `pi_features` | 51.34 | 51.45 | 44.10 | 48.11 | 50.24 |
| trunk tokens | 42.61 | 41.73 | 44.37 | 42.65 | 42.62 |

**Richness rises steeply and then PLATEAUS, in both lineages, well before each one's end.** rev-1
nearly doubles its `pi_features` PR by 12.1M and then does nothing for its last 13M steps. The v8 lineage
is already at 51.34 at 149.6M and has no trend through 277.2M (44–51, non-monotone).

**Where the current fleet's base sits:** the R5 fleet forks from **rev-1** (`--model` of
`ai_v9_92/93`) with `R2ACTION` as its exploiter target, so the fleet's base is the 13.55 / 20.69 row
— i.e. the plateau, not a point on the way up.

⚠️ **The 25M→149M window is unobserved in both lineages** (the archive keeps nothing below ~149M in
the v8 family). So "a one-week 138-GPU-h run would reach v8-class richness" is neither supported nor
refuted by this table. What the table *does* say is that **no observable trend supports it**: the
gen era stopped climbing 13M steps before its end, and §3 says the ceiling it stopped at is set by
its input's variance structure, which more steps do not change.

---

## 8. A NEW fold-specific effect: the POOLED representation compresses, in every fold and in neither control

Paired cluster bootstrap of the PR difference (same states, same resampled clusters, 400 reps):

| contrast | `projection` | **`cls_pool`** | `pi_features` |
|---|---|---|---|
| **v8 fold − v8 parent** | +0.07 [−0.51,+0.65] | −1.24 [−2.00,+0.06] | **−4.60 [−5.45,−2.20]** |
| rev-2 fold − rev-1 parent | +0.18 [−0.05,+0.43] | **−1.46 [−2.06,−0.46]** | −1.00 [−1.26,−0.55] |
| rev-2 fold − R2CTRL | −1.15 [−1.45,−0.72] | **−1.84 [−2.17,−0.90]** | −3.30 [−3.61,−2.54] |
| rev-2 fold − R2PLAIN | −0.30 [−0.56,+0.06] | **−2.27 [−2.54,−1.39]** | −0.18 [−0.57,+0.33] |
| rev-3 fold − R2PLAIN | −0.93 [−1.17,−0.49] | **−2.97 [−3.31,−1.68]** | −0.72 [−1.09,−0.15] |
| COMPFOLD − R2PLAIN | −0.52 [−0.81,−0.11] | **−2.82 [−3.20,−1.65]** | −0.73 [−1.14,−0.18] |
| *R2CTRL − rev-1 parent* | *+1.33 [+0.92,+1.63]* | *+0.38 [−0.19,+0.78]* | *+2.29 [+1.64,+2.67]* |

**`cls_pool` is the clean one:** every fold loses 7–15% of the pooled representation's participation
ratio against BOTH no-fold controls with CIs excluding zero, while neither control moves against the
parent. Distillation specifically compresses the pooled readout.

**And it does not distinguish gifting from robbing.** v8's fold compressed `pi_features` by **−4.60**
— four and a half times the gen folds' −1.00 — and gifted +5.42pp anyway. Any account of the form
"the gen fold robs because it damages the representation" has to explain why the fold that gifted
damaged it most.

⚠️ **The two gen controls disagree on `pi_features`** — R2CTRL *gains* 2.29 over the parent while
R2PLAIN sits 0.82 *below* it (22.89 and 19.77 against 20.59), so "the fold costs policy-side richness relative to no fold" is **not
established** — it depends entirely on which control is used. Only the `cls_pool` column survives
both. This is exactly why two controls were run.

---

## 9. Predictions, scored

| # | prediction | verdict |
|---|---|---|
| **1** | v8's participation ratio ≥2× the gen era's at matched width and matched tap (reproducing 50.2 vs 20.6) | **PASS at the two named taps, FAIL at six of eight.** `projection` 2.22×, `pi_features` 2.44×, intervals disjoint, the audit's figures reproduced exactly. But the input encoder (0.93×), both mlp heads (0.91×, 0.67×) and the value path (1.19×) are equal or INVERTED. **Materially DEMOTED by §3:** the ratio is 2.11× in the raw observation before any network, and vanishes under a scale-invariant reading. |
| **2** | trunk-located delta fraction ORDERS WITH untaught externality (v8 highest and positive; rev-2 lowest and negative) | **PASS on ordering, UNINFORMATIVE on inspection.** 0.381 > 0.336 > 0.330 against +5.42 > −0.75 > −7.06. But every gen fold lies inside the no-fold control band (0.327–0.333), so the ordering carries one degree of freedom — the era gap — and the within-era half is noise. |

---

## 10. What would REFUTE the hypothesis, and what did

The hypothesis: **a richer representation lets a teacher's content be written as a combination of
directions the student already has, so it composes with existing competence and radiates off-slice;
an impoverished one forces overwriting, which is local by construction.**

| # | refuter | fired? |
|---|---|---|
| R1 | the richness gap is absent or reversed at matched width and matched tap | **partially.** Present at 2 taps, absent or inverted at 6. |
| R2 | trunk locus fails to order with externality | no — it orders. |
| R3 | a high-trunk-share gen fold still robs, or a low-trunk-share fold gifts | **untestable here.** All five gen folds sit at 0.330–0.339, and per the ledger's rev-4 scorecard all of them rob. There is no variation to read. |
| R4 | the richness gap turns out to be inherited from the INPUT rather than produced by the network | **🔴 FIRED.** obs PR 36.19 vs 17.06 (2.11×) with no model; equal under scale-invariant PR; amplification 1.39 vs 1.20. |
| R5 | the GIFTING fold damages the representation as much as or more than the robbing ones | **🔴 FIRED.** v8's fold: `pi_features` −4.60 [−5.45,−2.20]. Gen folds: −1.00 [−1.26,−0.55]. |

**Two of five fired, and they are the two that attack the mechanism rather than the correlation.**
The hypothesis as stated — richness of the *network's representation* is the transfer vehicle — is
**NOT SUPPORTED as a causal account.** The correlation it predicted is real and reproduces, but it
is (a) confined to two taps, (b) fully accounted for by the input's variance structure, and (c)
accompanied by a compression signature that the gifting fold exhibits *more strongly* than the
robbing ones.

**What survives, and is testable:** the gen era's *observation* carries roughly half the effective
dimensionality it carried at gen-12, on a datable schedule, while carrying the same amount per live
dimension under a scale-invariant reading. That is a statement about the obs vector's variance
structure, not about capacity or maturity, and it has a cheap in-era test (below) rather than a
138-GPU-hour one.

---

## 11. Limits, stated plainly

- **n = 3 folds on the outcome axis, and arguably 2.** The mission forbade fitting a coefficient to
  three points; §5 argues the effective count is lower still, because rev-3's −0.75 is now suspected
  (ledger, rev-4 scorecard) to be floor exhaustion rather than a distinct outcome.
- **Era asymmetry is irreducible.** obs 2992 vs 2501, different module sets, parent maturity 277M vs
  25M, fork counts 3 vs 5. Every cross-era statistic reported is dimensionless for that reason and
  none of them makes the differences vanish.
- **PR is a property of (model, state set) and the eras cannot share a state set.** §3 is the
  attempt to bound that confound rather than to wish it away; it did not survive the attempt intact.
- **§3.1's generation row set confounds architecture with state distribution** — six runs, six
  policies. It brackets a drop; it does not attribute one.
- **Raw-covariance PR is variance-weighted.** Reported alongside signed-log and correlation-matrix
  readings precisely because the three disagree, and the disagreement is the finding.
- **The head column of §4 is era-asymmetric** (5.6k flat `action_net` vs 55k `pointer_head`; G4 at
  6.9% vs 19.7% of parameters, still learning in the gen era).
- **No v8 no-fold control exists** — the audit's largest gap, unchanged. Every v8 fold/fork delta is
  read against a parent, never against ordinary continued training, so v8's rows cannot be
  control-corrected the way the gen rows are.
- **The `cls_pool` compression (§8) is measured, not explained**, and it has no outcome attached: it
  fires identically in a fold that gifted and in three that robbed.
- **Era-pinned loads** inherit their validation from `v8_redistribution_pfsp_2026-08-30` (logit
  r = 0.982) and the audit's acid test (max |Δ| 1.07e-05); they were not re-validated here.

---

## 12. What this changes, and the cheap next test

**It removes "make the trunk richer" from the menu** and replaces it with a narrower question that
costs no GPU: *is the gen-era observation's variance concentration depressing what the encoder can
express, and does normalizing it recover the participation ratio?*

The pre-registerable version, all of it offline on existing checkpoints:

1. Take rev-1 and the gen-12 checkpoint. Compute per-block contribution to the obs covariance
   spectrum. If a small number of blocks dominate in gen-14+ and did not in gen-12, the concentration
   has a named source.
2. Forward rev-1 with the obs **z-scored per column** (running stats from its own traces) and
   re-read `projection` / `pi_features` PR. If PR rises toward v8's, richness is a normalization
   question, not a capacity or maturity one — and the intervention is an obs-preprocessing change,
   not a 138-GPU-hour run.
3. Only if (2) is null does the maturity branch of the ledger's fallback tree keep its priority.

This is stated as a proposal, not a result. Nothing in §12 was measured.

---

## Provenance

| claim | source |
|---|---|
| participation ratio, all taps | `agents.training.rank_metrics.effective_rank` over forward dumps; `representational_richness_transfer_analyze.py` |
| the 50.24 / 20.59 scale | `plasticity_forensics_v8_vs_gen_2026-08-28.{md,json}` Phase B, reproduced here |
| fork trunk shares 0.465 / 0.278 | same, Phase A — reproduced here as the locus table's anchor rows |
| untaught externality, v8 +5.42pp | `v8_redistribution_pfsp_2026-08-30.md` §3 (352 cells, 7,680 untaught battles) |
| untaught externality, rev-2 −7.06 / rev-3 −0.75 | `rev3_untaught_pulldown_2026-08-30.md` (4,800 paired battles) |
| fold→parent lineages | each run's `metadata.json` → `original_command` `--model` / `--distill-teachers` |
| era-pinned load path | `/tmp/probeP_v8era` @ `b13b30b289c5eaba136a930a4ab63451e209fbe5` |
| obs PR by generation | each run's own `eval_traces/step_*/**/*_states.npz`, model-free |
| every raw number | the sibling `.json` |
