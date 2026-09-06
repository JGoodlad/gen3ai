# The sharing kernel — does the pointer-native head share more across teams than the v8 flat head?

**Status: PRE-REGISTERED 2026-09-05, before any state was generated or any gradient taken.**
Everything below the `## PRE-REGISTRATION` heading was written first and is not edited after data;
results are appended under `## RESULTS`.

---

## PRE-REGISTRATION

### The hypothesis (given, verbatim)

> **"Sharing kernel"**: how much a parameter update driven by states from TAUGHT teams moves the
> policy on UNTAUGHT teams is governed by the gradient kernel between the two state sets. Claim:
> the pointer-native entity head (current architecture, v51+) has a LARGER cross-team kernel (more
> sharing at the action-scoring layer) than the v8-era flat positional action head, so gifts AND
> leaks are amplified; the sign is not set by architecture.

### The prediction

1. **Primary.** The cross/within cosine ratio on ALL parameters is **higher on the gen-era parent
   (`ai_v9_59_R2ACTION_0827`, pointer-native) than on the v8-era parent
   (`ai_v8_04_distill_4teacher_0722`, SB3 flat `action_net`)**.
2. **Localisation.** The **ACTION HEAD** parameter group is where that difference concentrates —
   the gen era's `pointer_head` shows a higher cross/within ratio than the v8 era's `action_net`,
   by more than the gap seen in the shared trunk groups (encoders, team_transformer,
   projection/mlp).
3. **Falsifier, stated in advance.** If the ratio is equal within the permutation/bootstrap
   resolution, or reversed, this README says so plainly in the verdict line. A ratio near 1.0 in
   BOTH eras would mean the taught/untaught split is not a direction the kernel distinguishes at
   all, which refutes the framing rather than the sign.
4. **Norm-share sanity, registered as a way to be wrong.** "Sharing lives in the head" is only
   meaningful if the head carries appreciable gradient norm. The norm-share column is reported for
   every group in both eras and the verdict must be read against it: a head holding <2% of the
   gradient norm cannot be the mechanism no matter what its cosine says.

### What is measured

Per state `s`, the score-function direction a distillation or policy-gradient update pushes along:

```
g_s = ∇_θ log π_θ(a*|s),    a* = argmax over LEGAL actions of the masked policy
```

taken at ONE parameter point (the parent's own weights), through the policy's own funnel
(`policy.get_distribution(obs)` → `apply_masking(mask)` → `log_prob(a*)`, i.e. the pointer head via
`_get_action_dist_from_latent` in the gen era and SB3's `action_net` in the v8 era). Only the
policy path is differentiated; the critic path receives no gradient from `log π` and is reported
separately (it is identically zero and that is a check, not a result).

The kernel is the cosine `K(s,s') = <g_s,g_s'> / (|g_s||g_s'|)`, computed for ALL parameters and
per parameter group.

### Primary statistic, and why same-team pairs are excluded

`ratio = cross / within`, where

* `within_taught` = mean `K` over pairs of states from **two DIFFERENT taught teams**
* `within_untaught` = mean `K` over pairs from two different untaught teams
* `within` = the pooled mean of those two
* `cross` = mean `K` over (taught state, untaught state) pairs

Same-team pairs are **excluded from the primary**: two states from one battle on one team are
near-duplicates and would inflate `within` for reasons that have nothing to do with the
taught/untaught split. They are reported separately as `within_same_team` (a scale reference).

### Inference

* **Permutation null (2000 draws).** Relabel the 24 teams — 16 "taught", 8 "untaught" — uniformly
  at random, keeping every team's whole block of states together, and recompute `ratio`. The
  p-value is the two-sided fraction of permuted ratios at least as extreme as observed. Every team
  contributes an EQUAL number of states (19) precisely so this relabelling is exchangeable.
* **Cluster bootstrap over teams (2000 draws).** Resample the 16 taught and the 8 untaught teams
  with replacement within their own strata, recompute, take the 2.5/97.5 percentiles.
* **Vocabulary.** SIGNIFICANT / WITHIN FLOOR / NOT DETECTED. "WITHIN FLOOR" is reserved for a
  difference whose CI lies inside the resolution the permutation null implies.

### Recipe and seeds

* **States.** 24 probe teams — the 16 **taught** by the teacher-content 2×2 fleet (copied verbatim
  from `teacher_content_2x2_2026-09-04/taught_probe.py::SLICES`, which resolved them at run time
  from each arm's own recorded `--distill-teacher`) and the 8 **untaught** teams of
  `rev3_untaught_pulldown_selection.json` (the same off-slice set
  `reuse_batch_2026-09-03/offline_collateral_kl.py` uses). Each team is piloted by the era's parent
  against a FIXED opponent model on 3 FIXED opponent teams (the first three of probe P's
  pre-registered opponent set), 1 battle per cell = 72 battles per era. **19 evenly-spaced states
  per team**, equal across all 24 teams.
* **Teams resolved by CONTENT SHA, never filename.** The 16 taught teams were promoted into
  `data/teams/sample/` with sha10 filenames after `b13b30b2`; at the era they sit in
  `data/teams/others/giraffe/`. Both trees hold 770 team files and all 24 team strings —
  **0 substitutions**.
* **Determinism, one protocol in both eras.** `stochastic=False` on both sides, a pinned single
  team per side, and an explicit 4-int sim seed per battle drawn from `random.Random(f"{team}:{opp}")`
  (probe P's CRN construction, verbatim). This is a **deliberate deviation** from
  `offline_collateral_kl.py`'s stochastic piloting: the five `GEN3AI_*_SEED` variables do not exist
  at `b13b30b2`, so running the two eras on two protocols would confound the comparison. Greedy
  piloting is also the coherent behaviour policy for a statistic defined at the argmax action.
  The five seed vars are exported by the caller regardless and are inert at the era commit.
* **Bridge.** `rust` in the gen era, `node` in the v8 era (the era predates the seedless-seed fix
  `bc00d4d`, so node is mandatory there). CPU only.
* **Permutation / bootstrap seeds**: 20260905.

### Also registered: a within-era control with no architecture confound

The gen-era parent's kernel on **FUNDED-teacher taught teams vs UNFUNDED**. Both halves of the
teacher-content 2×2 resolve to the **same 16 teams**, so the two labels name one identical state
set. The measured ratio must therefore come out at exactly 1.0 with a degenerate null. This is a
check that the kernel is a property of the model and the states, not of the label.

### Caveats fixed in advance

1. **One parameter point, not a training trajectory.** This is the kernel at the parent's weights.
   A fold moves the weights, and the kernel moves with them. It bounds nothing about the integrated
   effect of an actual fold.
2. **Argmax action only.** `∇ log π(a*|s)` is one direction in a distribution over 11 actions. A
   full-distribution kernel (e.g. the Fisher) could order the eras differently.
3. **THE ERA CONFOUND, and whether the decomposition can break it.** The two parents differ in
   architecture AND in maturity (≈28.1M steps vs ≈277.6M) AND in observation space (2501 vs 2992)
   AND in everything else 84 commits apart changed. The per-group decomposition **cannot fully
   separate head structure from maturity** — maturity acts on every group at once, so a
   head-specific effect is only distinguishable from maturity if the head's ratio moves and the
   trunk groups' ratios do not. That is a weak, one-directional discriminator: it can FAIL to
   support head-structure (if all groups move together), but if the head alone moves it still
   cannot exclude a maturity effect that happens to be head-localised. Whatever this measurement
   returns, it is not a clean architecture experiment, and it is written down here as such before
   the numbers exist.

---

## RESULTS

Run 2026-09-05 on an idle box (load average 2.1–2.7 on 16 cores). CPU only; the gradient pass and
the Gram matmul used **4 torch/BLAS threads**, everything else 1. Wall cost: state generation
159 s (gen) + 135 s (v8); kernel 32.5 s (gen) + 23.4 s (v8); comparison ~60 s. **456 states per
era** — 24 teams × 19, equal per team, and 0 states with a zero gradient after the
≥2-legal-actions filter.

### Artifacts

| file | what |
|---|---|
| `gen_states.py` | the one era-agnostic state recipe |
| `kernel.py` | gradients, groups, per-era kernel + permutation + cluster bootstrap |
| `compare.py` | the paired between-era test |
| `control_funded_vs_unfunded.py` | the label-identity control |
| `states_{gen,v8}.npz` + `_meta.json` | the frozen state batches and their provenance |
| `kernel_{gen,v8}.json` | per-era results incl. the full 24×24 team-block cosine matrices |
| `compare.json`, `control_gen.json`, `results_table.txt` | comparison, control, printed tables |

### Sanity checks, all passed

* **The critic is exactly zero in both eras.** `value_net`, `mlp_extractor.value_net`, and every
  `value_*` / `win_head` / `cf_*` / `film_vf` parameter receives **no gradient at all** from
  `log π`. Reported as its own group and never pooled.
* **The control is exact.** All four teacher-content 2×2 arms were re-resolved from their own
  `metadata.json` → `--distill-teacher` → each teacher's `--trainee-teams`: 8 teachers × 2 teams →
  the **same 16 teams** for all four arms, matching `gen_states.py::TAUGHT_SHA` exactly. Labelling
  those 16 "funded" on one side and "unfunded" on the other gives `ratio = 1.000000000` in every
  group, `max |ratio − 1| = 2.2e-16`. The meter reads the model and the states, not the label.
* **Zero team substitutions between eras.** Both trees hold 770 team files and all 24 probe team
  strings. The 16 taught teams live at `data/teams/sample/<sha10>.txt` today and at
  `data/teams/others/giraffe/<basename>.txt` at `b13b30b2`; resolving by content sha1 made the
  rename invisible.
* **Pilot ≠ opponent** (parameter L2 16.13 gen / 53.33 v8).
* **Reproducible.** `kernel.py --era gen` re-run from the committed `states_gen.npz` reproduced
  every reported statistic — ratio, cross, within, permutation p, bootstrap CI — with
  `max |diff| = 0` exactly. (The *state generation* is likewise pinned: greedy play, pinned teams,
  explicit per-battle sim seeds, `concurrency=1`.)

### Per-era table (cosines; same-team pairs excluded from within/cross)

`within` is the pair-count-weighted pooled mean over same-label distinct-team pairs (120 taught +
28 untaught team pairs); `cross` is over the 128 taught×untaught team pairs. `permP` is the
two-sided team-label permutation p-value on `cross − within` (2000 draws); the bracket is the
cluster bootstrap 95% CI on the ratio (2000 draws).

| group | era | within_taught | within_untaught | within | cross | **ratio** | permP | boot CI (ratio) | **norm share** |
|---|---|---|---|---|---|---|---|---|---|
| ALL | gen | +0.00407 | +0.00701 | +0.00462 | +0.00568 | **1.228** | 0.184 | [0.842, 1.772] | 100% |
| ALL | v8 | +0.00339 | +0.00402 | +0.00351 | +0.00334 | **0.951** | 0.760 | [0.592, 1.364] | 100% |
| action_head | gen | +0.00926 | +0.01080 | +0.00956 | +0.01064 | **1.114** | 0.167 | [0.904, 1.309] | **0.66%** |
| action_head | v8 | +0.00974 | +0.01567 | +0.01086 | +0.01296 | **1.193** | 0.473 | [0.662, 1.941] | **6.23%** |
| encoders | gen | +0.00319 | +0.00721 | +0.00395 | +0.00508 | 1.286 | 0.306 | [0.611, 2.403] | 51.77% |
| encoders | v8 | +0.00349 | +0.00243 | +0.00329 | +0.00210 | 0.638 | 0.085 | [0.063, 1.279] | 31.52% |
| team_transformer | gen | +0.00353 | +0.00611 | +0.00402 | +0.00476 | 1.185 | 0.289 | [0.772, 1.754] | 33.64% |
| team_transformer | v8 | +0.00379 | +0.00363 | +0.00376 | +0.00348 | 0.925 | 0.631 | [0.570, 1.311] | 34.59% |
| projection_mlp | gen | +0.00826 | +0.00938 | +0.00847 | +0.00996 | 1.176 | 0.191 | [0.822, 1.401] | 6.20% |
| projection_mlp | v8 | +0.00143 | +0.00196 | +0.00153 | +0.00173 | 1.125 | 0.376 | [0.836, 1.473] | 26.35% |
| belief_op | gen | +0.00645 | +0.00823 | +0.00679 | +0.00770 | 1.135 | 0.381 | [0.678, 1.551] | 7.73% |
| belief_op | v8 | +0.00464 | +0.00581 | +0.00486 | +0.00466 | 0.958 | 0.792 | [0.556, 1.456] | 1.31% |
| **critic** | both | — | — | — | — | — | — | — | **0.0000 (zero gradient)** |

Permutation-null SD of the ratio (the **resolution**): gen 0.078–0.253, v8 0.133–0.253. Every
observed within-era deviation from 1.0 sits inside ~1.6 null SDs.

### Between-era, paired on the same 24 teams

| group | ratio gen | ratio v8 | **Δ (gen−v8)** | boot CI95 | perm p | Δ ratio_halves | perm p | norm share g/v8 |
|---|---|---|---|---|---|---|---|---|
| ALL | 1.228 | 0.951 | **+0.277** | [−0.268, +0.954] | 0.193 | +0.124 | 0.468 | 100 / 100 |
| **action_head** | 1.114 | 1.193 | **−0.079** | [−0.829, +0.461] | 0.761 | +0.041 | 0.817 | **0.66 / 6.23** |
| encoders | 1.286 | 0.638 | +0.648 | [−0.280, +2.072] | 0.055 | +0.268 | 0.362 | 51.8 / 31.5 |
| team_transformer | 1.185 | 0.925 | +0.260 | [−0.300, +0.996] | 0.232 | +0.050 | 0.748 | 33.6 / 34.6 |
| projection_mlp | 1.176 | 1.125 | +0.051 | [−0.504, +0.478] | 0.784 | +0.113 | 0.288 | 6.2 / 26.4 |
| belief_op | 1.135 | 0.958 | +0.178 | [−0.560, +0.772] | 0.416 | +0.159 | 0.293 | 7.7 / 1.3 |

`ratio_halves` divides `cross` by the **unweighted** mean of the two within-halves instead of the
pair-count-weighted pool. It matters: the 120 taught-taught pairs outvote the 28 untaught-untaught
ones ~4:1, and the two halves are not equally homogeneous (in the gen era `within_untaught`
+0.00701 is **1.7× `within_taught`** +0.00407 — the untaught 8 are balance/stall-heavy by their own
selection note's `known_gap`). Under that normalization every gen-era ratio collapses toward 1.0
(ALL 1.228 → **1.025**), so most of the gen era's apparent cross > within is the taught set being
internally more heterogeneous, not extra cross-group sharing.

---

## VERDICT

### Prediction 1 — gen cross/within ratio higher than v8: **NOT DETECTED**

The direction matches (+0.277 on ALL parameters, gen 1.228 vs v8 0.951) but the paired cluster
bootstrap CI is **[−0.268, +0.954]** and the paired permutation p is **0.193**. Under the
composition-robust `ratio_halves` the difference shrinks to +0.124 (p = 0.468). No group's era
difference clears zero under either normalization; the closest is `encoders` at p = 0.055, which is
**not** the group the hypothesis named.

### Prediction 2 — the difference concentrates in the ACTION HEAD: **NOT DETECTED, and what signal there is points the other way**

`action_head` is the **only** group whose primary-ratio era difference is **negative** (−0.079: the
v8 flat head's ratio 1.193 is *higher* than the pointer head's 1.114), and under `ratio_halves` it
is the flattest group of all (+0.041, p = 0.817). Every trunk group moves in the same direction
(+0.05 … +0.65) and the head does not join them.

The **registered norm-share sanity check fails for the gen era outright**: the pointer head carries
**0.66%** of the gradient norm against the v8 flat head's **6.23%** — a factor of 9.4 the *wrong*
way for a story in which the pointer head is the amplifier. In the gen era 85% of the gradient norm
is in the encoders (51.8%) and the team transformer (33.6%). Whatever transmits a fold's
displacement across teams in the current architecture, it is not the action-scoring layer, because
the action-scoring layer is where almost none of the update lives.

*(One descriptive note, offered as scale rather than inference: the v8 head has only 5,643
parameters, so the isotropic-noise cosine scale there is 1/√P = 0.0133 — its cross-team cosine of
0.0130 sits **at** that scale, while the gen head's 0.0106 sits ~2.5× above its own 0.00424. A
single pair's cosine at the v8 head is therefore uninformative; the reported means over 128 team
pairs × 361 state pairs each are still resolved. This is a dimensional normalization, not a test,
and it does not rescue prediction 2 — it only says the two heads' raw cosines are not on one scale.)*

### Within-era contrasts — **WITHIN FLOOR**

No era shows a cross-vs-within contrast that clears its own permutation null: the smallest p in
either era is 0.085 (v8 encoders), every other group is 0.17–0.79, and every observed ratio sits
inside ~1.6 permutation-null SDs of 1.0.

### The framing, not the sign

Falsifier #3 was registered in advance and **it is the one that fired**: at the parent's parameter
point, in *both* architectures, the taught/untaught split is not a direction the gradient kernel
distinguishes. Cross-team cosines are ~0.003–0.011 everywhere — the gradient at one state is very
nearly orthogonal to the gradient at a state from another team, whichever teams they are and
whichever head scores them. The right reading is not "the pointer head shares more" or "less"; it
is that **the taught/untaught partition is not a feature of the kernel at this parameter point at
all**, so a sharing-kernel account of why folds gift or rob gets no support here and needs a
different mechanism — or a different place to look (below).

### The control

**PASS, exactly.** Funded vs unfunded → `ratio = 1.000000000` in all six groups,
`max |ratio − 1| = 2.2e-16`.

---

## CAVEATS — what this measurement cannot say

1. **One parameter point, not a training trajectory.** This is the kernel at the parent's weights
   before any fold step. A real fold moves the weights and the kernel with them; the integrated
   effect over ~4.45M steps is not bounded by this.
2. **Argmax action only.** `∇ log π(a*|s)` is one direction out of a distribution over 11 actions.
   A full-distribution kernel (Fisher / all-action Jacobian) could order the eras differently, and
   a *distillation* loss is a KL over the whole distribution, not a score function at the argmax.
   That is the single most likely reason this instrument could be looking at the wrong object.
3. **THE ERA CONFOUND — and the honest answer to "can the decomposition separate it?": NO.** The
   two parents differ in head structure, in maturity (28.1M vs 277.6M steps), in observation space
   (2501 vs 2992 dims), in parameter count (3.15M vs 3.51M) and in everything else 84 commits apart
   changed. The per-group decomposition was registered as a weak one-directional discriminator and
   it behaved as one: the five trunk groups all move the same way and the head does not, which
   **argues against** head structure being the amplifier — but it **cannot** say what the trunk
   movement is. A whole-model maturity effect, an obs-space change, and a distributed architectural
   difference all produce exactly the pattern observed. Nothing here is a clean architecture
   experiment and no sentence above should be read as one.
4. **Composition of the two team sets is not matched.** The untaught 8 are balance/stall-heavy and
   internally more homogeneous than the taught 16 (their own selection note records that
   hyper_offense is structurally unrepresentable in the untaught set). `ratio_halves` is the defence
   against that and is reported throughout, but no reweighting fixes a set that cannot contain an
   archetype.
5. **Greedy piloting.** Both eras ran at `stochastic=False` for era parity, so the state
   distribution is the parent's greedy trajectory, not its training-time rollout distribution.
6. **The `permP` column tests cross-vs-within, not gen-vs-v8.** The between-era question has its own
   paired test in `compare.py`; do not read a within-era p-value as an era comparison.

## Where a follow-up should look

The two findings most likely to survive a rerun are *norm* facts, not cosine facts: **the critic
path receives exactly zero score-function gradient**, and **the current architecture puts 85% of its
policy-gradient norm in the encoders + team transformer and 0.66% in the pointer head**. If a fold's
collateral is to be localised, that norm distribution says the place to instrument is the
encoder/transformer trunk, not the action head. Testing the kernel account properly would mean
(a) the KL / full-distribution kernel rather than the argmax score function, and (b) the kernel
measured *along* a fold rather than at its start.
