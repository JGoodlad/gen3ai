# Architecture diagnosis of P\*: does a critic or tower change belong in BOUNDARY-1's base? (2026-11-09)

**Status: REPORTED · MEASURED · DESCRIPTIVE (no pre-registration, no verdict; one checkpoint, CPU only, no
training, no games, no GPU).** A decision memo for
[`../boundary1_scratch_2026-11-09/DRAFT_DESIGN.md`](../boundary1_scratch_2026-11-09/DRAFT_DESIGN.md): the fresh
2 × 2 (width × team breadth) launches about 2026-11-13, and an independent review claimed three architecture faults.
Each claim is read here on P\* (`pstar_plateau`: `models/rb_x26_s2601/checkpoints/checkpoint_280000170_steps.zip`,
sha256 `814823c5…91ed`, config v158) by its own code (pin `9c0334890`), on banked rows.

## 0. One screen

| the review's claim | finding | numbers |
|---|---|---|
| (a1) 15 % of the parameters are in the attention trunk, 43 % in flat MLPs after pooling | **CONFIRMED** | 401,536 of 2,609,183 (15.39 %) and 1,117,952 (42.85 %); matches `designs/ARCHITECTURE.md` to the parameter |
| (a2) those MLPs' tanh units are 84–92 % saturated | **PARTLY** | true of ONE layer: actor tanh-1 0.897 of activations at \|a\| > 0.99. Tanh-2 is 0.235. The 22.6 % projection is ReLU with no dead unit. The saturated layer's weights are 10 % of the model |
| (a3) so the tower is close to a lookup table and passes almost no gradient | **PARTLY, and not a dead gradient** | the gradient through the tower is cut 15–18× against an identity backward pass, and 71 of 512 tanh-1 units (13.9 %) are constants; but the tower path still carries the larger share of the trunk's policy gradient (6.98 against 2.37 without it), the tower's weights move MORE than the trunk's (relative displacement 270M → 280M: 0.067 against 0.040), and the refit probe read NO-LOSS (`8bebdfe31`) |
| (a4) and that explains play on unfamiliar teams | **REFUTED as a differential** | saturation is the same on and off the pool: 0.897 / 0.897 (tanh-1), per-unit rates correlate 0.982; every ablation reads the same on both |
| (b1) the value is a LINEAR head on a 128-d pooled vector | **REFUTED as built, true as trained** | the head is LayerNorm → Linear(128, 128) → ReLU → Linear(128, 1). The trained function is 98.9 % linear in its normalised input (held-out R²), and 36 % of its hidden units never fire |
| (b2) it is miscalibrated on unfamiliar teams | **PARTLY: a level shift against one opponent, not a ranking failure** | against Kakuna, U − home bias +0.054 [+0.003, +0.108] over all decisions, already +0.077 at turn 1; against PastaMittens +0.002 [−0.059, +0.057]. Calibration error on U is no worse than at home (0.028 against 0.033). Ordering on U is equal or better: AUC 0.813 against 0.781, slope 0.94 against 0.77 |
| (c) the belief heads memorise the pool and are worse than the Smogon prior off it | **CONFIRMED on calibration (cited, not re-measured); its game cost NOT DETECTED** | ledger 2026-10-26 H3 (`94156c0c5`): unfamiliar species log-presence −6.24 / −4.63 / −10.04 against the prior's −2.36 / −1.98 / −3.24. Mixing the prior in moved U by +1.50 pp [−2.74, +5.74] and −1.50 [−5.47, +2.47] (`4ee655648`) |

**Recommendation for BOUNDARY-1's shared base: leave all three changes OUT.**

| candidate change | evidence it binds | build by Friday | launch risk | where it goes |
|---|---|---|---|---|
| a critic with a hidden layer | **against**: the hidden layer exists and training does not use it | nothing to build | none | **nowhere**; fix the "linear on 128" wording |
| a non-saturating policy tower | **none for strength**: a real 15–18× gradient attenuation, but the tower still learns, and nothing differs off the pool | 0.5 agent-day + an architecture-signature bump; **no time to screen it** | **high**: an unscreened actor change in all eight runs; a fault there hurts both arms and the 2 × 2 cannot see it | **a separate later single-lever arm** on the winning width |
| rebalancing parameters toward the trunk | **none measured** | the width arm already is this lever; removing the tower is built (`--policy-readout trunk`, never screened) | high for the same reason | **the width arm, then a later arm** |

The designer's line stands, now with numbers under it: no measurement here shows that the critic's depth, the
activation or the parameter split binds.

## 1. What was read, and on which rows

- **The model:** P\* loaded once by the pinned checkout `/home/goodlad/.cache/gen3ai/x26/pin_9c033489` (its own code,
  read-only). Learner parameters 2,609,183; ride-along 3,014,996.
- **On-pool and off-pool states with observations:** the banked self-play rows of
  [`../arch_generalisation_2026-10-26/`](../arch_generalisation_2026-10-26/): P\* against itself, 48 games a panel.
  "home72" is 3,349 decisions on pool teams. "Piloting U" is 5,549 decisions where P\* holds a team from an
  unfamiliar panel (H, F or P72, `../../heldout_panels.json`); "facing U" is 5,440 where it holds a pool team
  against one.
- **A second on-pool population:** 16,785 states from `models/rb_x60t_s3502/` (a P\* fork: its eval traces, rollout
  scans and the compile-parity fixture), the tie-rate reconstruction's population. No outcome labels.
- **Outcomes at scale:** H-MID's stored value traces
  ([`../h_mid_value_test_2026-10-26/`](../h_mid_value_test_2026-10-26/), prereg `2dd522785`, read `f200258f6`):
  1,600 games against Kakuna and 1,600 against PastaMittens, a quarter on home72, the rest on H, F and P72. Each row
  is P\*'s V at a decision plus the game's result; no observations.
- The self-play rows and H-MID's traces are read from the run archive (`/home/goodlad/gen3ai_archive/`,
  `arch_generalisation_2026-10-26/` and `h_mid_value_test_2026-10-26/`), not from the repository.
- **Weights over time:** `policy.pth` of ten X26 checkpoints, 20M to 380M.
- Every interval is 95 %, a bootstrap over games (B = 1,000, seed 20261109). "Saturated" is \|a\| > 0.99, which is
  \|pre-activation\| > 2.647: the plasticity probe's threshold.

## 2. Claim (a): where the parameters are, and whether the tower is dead

**The census** (`model_read.json` → `census`; it equals `designs/ARCHITECTURE.md`'s table):

| module | parameters | share | activation |
|---|---|---|---|
| attention trunk (`team_transformer`) | 401,536 | 15.39 % | |
| projection 1152 → 512 | 590,336 | 22.63 % | ReLU |
| actor MLP 512 → 512 → 512 | 525,312 | 20.13 % | tanh, tanh |
| (flat policy tower in all, with its input norm) | 1,117,952 | 42.85 % | |
| belief and intent stack + `hidden_opp_belief` | 556,582 | 21.33 % | |
| three attention pools (`cls_pool`) + the critic's pool | 241,856 | 9.27 % | |
| per-mon encoder, embeddings, other extractor modules | 219,317 | 8.41 % | |
| pointer head (the policy's output) | 55,043 | 2.11 % | tanh |
| `win_head` (the critic) | 16,897 | 0.65 % | ReLU |

**Saturation, as a fraction of activations** (`model_read.json` → `saturation_pooled`, `saturation`):

| layer | home72 | piloting U | facing U | the fork's eval traces |
|---|---|---|---|---|
| actor tanh-1 | 0.897 | 0.897 | 0.900 | 0.903 |
| actor tanh-2 | 0.235 | 0.229 | 0.247 | 0.246 |
| pointer tanh, per legal action | 0.695 | 0.693–0.704 | 0.705–0.715 | 0.712 |
| projection ReLU, share at zero | 0.552 | 0.555 | 0.557 | 0.565 |
| `win_head` ReLU, share at zero | 0.782 | 0.779 | 0.784 | 0.780 |

- **Tanh-1 is deeply saturated, everywhere.** Median \|pre-activation\| 13.7; mean tanh′ 0.039; every unit is saturated
  on at least 75 % of rows; 71 units (13.9 %) on at least 99 % of rows, nearly all stuck on one sign. The other units
  still switch sign: 66 % take their minority sign on at least 5 % of rows.
- **It is a property of the weights, not of the team.** On-pool and off-pool rates agree to the third decimal, and
  the per-unit rates correlate 0.982. The mechanism is weight growth with nothing normalising the tower's input: the
  projection's weight norm goes 33.5 (20M) → 51.3 (P\*), the first actor layer's 32.8 → 42.4, under weight decay 1e-5.
- **How much of a "lookup table" is it?** (`tower_ablation`; share of decisions keeping P\*'s top action, on-pool /
  off-pool) Replacing tanh-1 by its sign keeps 0.913 / 0.912 (mean total variation 0.060 / 0.058); tanh-2 by its sign
  0.876 / 0.883; both 0.857 / 0.857. So layer 1 is mostly a 512-bit sign code, and about one decision in eleven
  needs more than the sign. Replacing the tower's whole output by its dataset mean keeps 0.713 / 0.714: the trunk
  tokens reach the logits past the tower, and the tower's 43 % of the parameters decide about 29 % of top actions.

**Gradients** (`model_read.json` → `gradients`; 256 rows, a policy pseudo-loss −mean(A · log π(a|s)) with A drawn
N(0, 1), on-pool / off-pool):

| quantity | on-pool | off-pool |
|---|---|---|
| backward gain through tanh-2 (norm out / norm in) | 0.525 | 0.525 |
| backward gain through tanh-1 | 0.167 | 0.170 |
| gradient norm at the projection's output, real | 0.0203 | 0.0294 |
| the same with an identity backward pass through both tanh layers | 0.370 | 0.442 |
| **attenuation** | **18.2×** | **15.0×** |
| trunk gradient norm, real | 6.98 | 9.09 |
| trunk gradient norm with the tower path cut (pointer tokens only) | 2.37 | 2.45 |
| trunk gradient norm with an identity backward pass | 120.7 | 145.7 |

- The attenuation is real and large. It is not a dead path: with the tower cut, the trunk's policy gradient falls
  to a third, so the saturated tower is still the trunk's main policy teacher. The projection has no other source of
  gradient at all.
- Per parameter the tower is not starved against its neighbours: gradient RMS 2.9e-3 in the first actor layer,
  2.4e-3 in the pointer head, 1.0e-3 in the projection, 1.1e-2 in the trunk. The optimiser is AdamW, which
  rescales each parameter's step by its own gradient history.
- **Dead units:** the 71 always-saturated units carry 0.26 % of the first layer's gradient energy on-pool and
  0.006 % off it. About 12 % of units have a gradient under 1 % of the median unit's.
- The value loss never touches the tower or the pointer head; it reaches the critic's pool, the trunk and the
  encoders.

**Do the weights still move?** (`drift_read.json`; relative displacement ‖Δw‖ / ‖w‖)

| span | whole model | trunk | projection | actor MLP | critic head |
|---|---|---|---|---|---|
| 20M → 30M | 0.079 | 0.116 | 0.235 | 0.150 | 0.051 |
| 270M → 280M (P\*) | 0.030 | 0.040 | 0.095 | 0.067 | 0.015 |
| 280M → 290M | 0.028 | 0.037 | 0.085 | 0.061 | 0.014 |
| 280M → 380M | 0.080 | 0.100 | 0.241 | 0.172 | 0.041 |

At the plateau the tower is the fastest-moving part of the network, not a frozen one. Within tanh-1 the cost is
visible: from 270M to 280M the 71 always-saturated units' input rows moved 0.027 against 0.069 for all units (0.39×),
and row movement correlates −0.48 with a unit's saturation rate. From 20M to 30M the same 71 units moved like the
rest (0.167 against 0.169).

**What this leaves open.** Whether the attenuation or the 14 % of dead units costs STRENGTH is **UNVERIFIED**: no A/B
of the activation exists at any dose. The record's one causal read near it is the plasticity check (NO-LOSS, 0.986×).

## 3. Claim (b): the value head

**Structure.** `WinProbHead.net` is LayerNorm(128) → Linear(128, 128) → ReLU → Linear(128, 1); V is its sigmoid,
trained by cross-entropy on the terminal win indicator. It is not a bare linear head, and the code says so
(`src/agents/model/aux_value_heads.py`). But on 14,338 self-play states, split by game, the best linear map from the
normalised input to the trained logit has held-out R² 0.989 (mean \|ΔV\| 0.014, 95th percentile 0.041), and 36 % of the
hidden units never fire on-pool (58 % fire on under 1 % of rows). Training had a hidden layer available and kept a
near-linear read of the pooled vector. Adding a hidden layer adds what is already unused.

**Calibration and discrimination against Kakuna, all decisions** (`value_read.json` → `hmid:kakuna`):

| | home72 (16,857 rows) | U (62,794 rows) | U − home |
|---|---|---|---|
| win rate | 0.579 | 0.475 | |
| mean V | 0.553 | 0.503 | |
| bias, V − win | −0.026 [−0.070, +0.012] | +0.028 [+0.001, +0.055] | **+0.054 [+0.003, +0.108]** |
| expected calibration error (ECE) | 0.033 | 0.028 | −0.005 [−0.049, +0.020] |
| ECE after one uniform logit shift | 0.033 | 0.017 | |
| ECE after a shift and a slope | 0.015 | 0.014 | |
| discrimination, AUC | 0.781 [0.738, 0.820] | 0.813 [0.794, 0.831] | +0.032 [−0.011, +0.079] |
| recalibration intercept at slope 1 (logit) | +0.153 [−0.069, +0.416] | −0.165 [−0.322, −0.006] | **−0.318 [−0.635, −0.023]** |
| recalibration slope | 0.774 [0.589, 1.001] | 0.942 [0.855, 1.046] | +0.168 [−0.077, +0.386] |

- **The error on U is a level, and it is there before a move is played.** One row a game: turn 1 Δbias +0.077
  [+0.020, +0.136] with AUC 0.576 against 0.577; turn 10 +0.081 [+0.029, +0.135] with AUC 0.712 against 0.714;
  turn 20 +0.057 [+0.017, +0.103] with AUC 0.810 against 0.822. V over-rates an unfamiliar team's prospects by a
  constant that no later evidence removes; it orders states as well as at home at every depth.
- **Reliability on U** (stated → realised): 0.6–0.7 → 0.604, 0.7–0.8 → 0.673 (the largest gap), 0.8–0.9 → 0.801,
  0.9–1.0 → 0.939. The home curve is the less calibrated of the two in its low bins (0.1–0.2 → 0.250).
- **By panel:** H bias +0.033, AUC 0.819, slope 1.02; F +0.004, 0.829, 1.01; P72 +0.048, 0.789, 0.82.
- **It does not replicate against PastaMittens** (`hmid:pm`): Δbias +0.002 [−0.059, +0.057], AUC 0.802 against 0.813,
  slope 0.917 against 0.906.
- **In self-play** (48 games a panel, descriptive) the shift is larger and symmetric: piloting U, bias +0.126
  [+0.071, +0.182] with AUC 0.862; the pool team facing U, −0.084 [−0.134, −0.025] with AUC 0.849; home72 +0.030
  with AUC 0.862. V under-rates the pool team's edge over an unfamiliar one, and ranks no worse.

**Verdict on its shape: a uniform bias, not a ranking failure, on every across-state measure here.** A uniform
level cancels when one-ply search compares sibling actions. Two limits:
- Ranking WITHIN a decision's siblings on U is still **UNMEASURED** (V-RANK is an unregistered sizing probe,
  [`../v_rank_on_u_2026-10-27/`](../v_rank_on_u_2026-10-27/)).
- On pool teams, refitting the head alone did not improve sibling ranking (ledger 2026-10-06, the label refit:
  −0.014 [−0.032, +0.002]): where ranking fails, the record puts it in the representation, not the head.

## 4. Claim (c): the belief heads (cited, not re-measured)

Already MEASURED, and already in the view (`UNDERSTANDING.md`, "The belief stack memorises the pool, but that is not
where games are lost"):
- **Memorisation, CONFIRMED** (ledger 2026-10-26 H3, `94156c0c5`,
  [`../arch_generalisation_2026-10-26/`](../arch_generalisation_2026-10-26/)): on unfamiliar species the heads'
  log-presence is −6.24 / −4.63 / −10.04 (F / H / P72) against the Smogon prior's −2.36 / −1.98 / −3.24; hit@k 0.258 /
  0.367 / 0.152 against 0.296 / 0.392 / 0.158. On familiar species they beat the prior (hit@k 0.71–0.76 against
  0.34–0.49).
- **Game cost, NOT DETECTED at ± 5 pp** (`4ee655648`, [`../belief_prior_mix_2026-10-26/`](../belief_prior_mix_2026-10-26/)):
  mixing the prior into the beliefs at inference moved U by +1.50 pp [−2.74, +5.74] at weight 0.25 and −1.50
  [−5.47, +2.47] at weight 1.

A re-measure would repeat H3 on the same rows, so none was run. The belief stack is 21 % of the parameters; a
change to it is outside this memo's three candidates.

## 5. The decision, per candidate

1. **A critic with a hidden layer: leave it out; there is nothing to add.** The head has one. Training left 36 % of
   it dead and the rest near-linear, so depth after the pool is not the constraint. The U error is a constant offset
   that a deeper head has no particular reason to remove and that one-ply search does not feel. The width arm already
   doubles the head's input. Build cost avoided: 1–2 agent-days (the draft's §8).
2. **A non-saturating policy tower: leave it out of the base; run it later as one lever.**
   - *For:* a measured 15–18× attenuation of the policy gradient into the projection and the trunk; 14 % of first-layer
     units are constants whose weights move 0.39× as far as the rest; saturation still rising at 480M (0.92, the
     plasticity probe).
   - *Against:* the tower path still carries the larger share of the trunk's policy gradient and is the
     fastest-moving module at the plateau; the refit probe read NO-LOSS; saturation is identical off the pool, so it
     explains nothing about unfamiliar teams; no strength A/B exists.
   - *Build:* a flag for the tower's activation, 0.5 agent-day (the draft's §8), plus an architecture-signature bump
     and the golden files. Buildable by Friday; **not screenable by Friday** (a fresh 15M cross is 35–41 GPU-hours,
     the draft's §9.2, and none is registered).
   - *Risk:* high. It would put an unscreened actor change, under an entropy coefficient chosen on the tanh tower,
     into all eight runs. If it hurts, it hurts both arms equally and the 2 × 2 cannot detect it; it also ends the
     control arm's comparability with X26's matched-step snapshot (the draft's §1).
3. **Rebalancing parameters toward the trunk: the width arm is that experiment.** The 15 % / 43 % split is a fact;
   that it binds is not measured. `--d-model 256` is the registered lever that moves it. The split at the wide
   arm's shape was not computed here (**UNVERIFIED**). Removing the tower outright is already built
   (`--policy-readout trunk`, config v138, 1,547,039 learner parameters) and has never been screened; the
   mean-ablation above says the tower decides about 29 % of P\*'s top actions, so it is not free to delete.

**A later single-lever arm, on the width BOUNDARY-1 names.** The two candidates, cheapest first: `--policy-readout
trunk` (0 build days), then a non-saturating tower (0.5 agent-day). Either is one lever against that width's
control, read by the same head-to-head cross.

**Free observability for BOUNDARY-1.** The saturation read is a CPU job of about four minutes a checkpoint. A copy
of `scripts/model_read.py` pointed at both arms' 25M and 100M checkpoints, each at its own pin, would show whether
the wide arm saturates the same way from scratch, with no launch change.

## 6. FINDINGs

1. **The draft's lever table and the review both call the critic "linear on 128".** It has a hidden layer of 128
   ReLU units. The draft now carries a pointer here; its row was not rewritten.
2. **The banked rows were encoded by later encoders** than P\*'s pin (the self-play rows by main of about
   2026-10-25, the fork's rows by pin `f4b25238`) and read here by P\*'s `9c0334890` code. The observation width
   matches (2,845); the semantic equality of every field across those pins was not verified.
3. **The tie-rate reconstruction's `full.npz` holds no observations.** Its population was re-loaded from
   `models/rb_x60t_s3502/`: on-pool states of a P\* descendant, with no home / U split and no outcomes.
4. **The only on / off-pool rows with observations AND outcomes are 48 self-play games a panel.** They are
   descriptive. The powered calibration read uses H-MID's stored V, which has no observations, so the head's
   internals cannot be read on those games.
5. **The gradient read is a pseudo-loss** (random unit advantages, 256 rows), not a PPO minibatch with its clipping,
   entropy and value terms. Ratios between paths are the claim; absolute norms are not.
6. **This read's turn-10 Δbias is +0.081 [+0.029, +0.135]; H-MID's registered one is +0.076 [+0.016, +0.139].** The
   row selection and bootstrap differ slightly. The registered number stays the record.
7. **Within-decision ranking of V on U is unmeasured**, here as before.
8. `model_read.json` groups `hidden_opp_belief` under "pooling"; it is a belief module, and §2's table counts it so.
9. The pinned checkout carries one unrelated modified file (`designs/ops/slow_tier_status.json`); no `src/` file
   differs from `9c0334890`.
10. `value_read.py` and `drift_read.py` were lint-cleaned after the run (context managers, dead variables, a
    default-bound closure); the committed results are the as-run scripts' output. A function-level check found the
    cleaned `auc`, `linearity_read` (equal to the committed JSON) and `hmid_read` (both opponents, B = 20) identical
    to the as-run ones. A full re-run had not left the heavy-slot queue at ship time. `model_read.py`'s two lint
    edits (an unused `noqa`, a redundant `int()`) were not re-run.

## 7. Files and how to reproduce

- [`scripts/common.py`](scripts/common.py): loaders. [`scripts/model_read.py`](scripts/model_read.py) →
  [`model_read.json`](model_read.json): census, saturation, ablations, gradients (228 s, peak 2.7 GB).
- [`scripts/value_read.py`](scripts/value_read.py) → [`value_read.json`](value_read.json): calibration,
  discrimination, the head's linearity. It reads `acts.npz` (47 MB, not committed), which `model_read.py` writes.
- [`scripts/drift_read.py`](scripts/drift_read.py) → [`drift_read.json`](drift_read.json): weight movement.

```bash
cd /home/goodlad/dev/gen3ai && PIN=/home/goodlad/.cache/gen3ai/x26/pin_9c033489 OUT=<a scratch dir> \
  D=designs/research_state/measurements/arch_diag_2026-11-09/scripts PY=/home/goodlad/miniconda3/envs/gen3ai_torch28/bin/python3
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PIN/src CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 \
  scripts/ops/mem_cap.sh --max-s 600 10 nice -n 19 taskset -c 5-7,13-15 "$PY" $D/model_read.py $OUT
# then, same environment, 6 GB: "$PY" $D/drift_read.py $OUT && "$PY" $D/value_read.py $OUT
```

These scripts are a frozen record of this read, never a tool.
