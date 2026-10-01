# X26 — the ride-along baseline: pre-registration (2026-09-30)

> **AMENDED 2026-09-30, BEFORE LAUNCH** (the run has not started): **Amendment 1**, at the end, adds
> the RND VARIANT ENSEMBLE (`--ridealong-rnd-variants all`), its pre-registered comparisons (a)–(e)
> with a Holm correction across variants, and the startup-acquisition lifecycle gate. Everything above
> it stands as registered.

**Owner, 2026-09-30:** *"Once we are ready to train on the GPU, establish a baseline just with the
ensemble of value heads to test uncertainty, the trick where you predict a random output to see how
often you've seen the state, and the Q, A and B values. Start getting baselines before we add more
complicated arms."*

**What this run is.** The FIRST GPU training run on the M5 Rust core after the switch is the
production recipe plus the four DETACHED ride-along heads (`gen3_ridealong_heads_v1`, config v126;
`designs/ARCHITECTURE.md` §3.4, `designs/model/readouts_and_value_routes.md`). The heads observe and
never steer. `ridealong_update_test` pins one real update with the heads ON vs OFF as BIT-IDENTICAL
in the policy, trunk and V parameters, the PPO optimizer state, the PPO scalars and the RNG. So this
run IS the production-recipe run, with meters attached. **Every later arm compares against THIS run:
X23 (policy sharpness) first, then X5, X13, and any learner or recipe arm.** They compare on strength
(the standard reads) AND on the ride-along meters below. An arm carries the same four flags so its
meters exist; they cost it nothing in learning.

## The argv

```
python -m main.launcher --restart-interval-hours 3 --device cuda --arch production \
  <the K10(a) RECIPE block, applied by --arch production once it lands> \
  --critic winprob --terminal-indicator --victory-value 1.0 --draw-penalty 0 \
  --env-core rust \
  --ridealong-ensemble 5 --ridealong-rnd --ridealong-adv 5 --ridealong-opp 5 \
  --steps <the production run's registered length> --run-name <assigned at launch>
```

- **The recipe.** K10(a)'s recipe block in `designs/production_config.json` is not on main as of
  this registration. If it has not landed at launch, the recipe is N0's (`ai_v14_01_base`)
  RECORDED training flags (`metadata.json:original_command`), with every value that differs from
  the parser default typed out, per the K10(a) note on the five silent defaults. `python -m
  main.checkargs --argv "…"` must print the ARCH SURFACE as matching and no refused combination.
  The four ride-along flags are `family=CRITIC`, so they are off the ARCH surface by declaration.
- **`--opp-intent-coef` > 0 is required** (production's 0.05, written by `--arch production`). It
  is what aligns the one-ahead opponent labels that B trains on.
- **Launch gates, in addition to the SOP's:** `ridealong_update_test` and `ridealong_heads_test`
  green at the launch commit. `ridealong/*` present in TensorBoard from the first update.
  `ridealong/disabled` = 0 at every read. If it reads 1, the heads hit a non-finite step and
  stopped: that read point is VOID for the heads, and the run itself is unaffected.

## Overhead (GPU learner benchmark, `learner_benchmark run --device cuda --ridealong`)

MEASURED, 2026-09-30, arm C's real 98,304-row buffer, `--compile-trainer` on, one RTX 3080 Ti, box
load 30–50 on 16 cores (contended: warned, not stretched). The ride-along config is the baseline spec
(ensemble 5, RND, A 5, B 5), attached to the loaded policy exactly as a v126 build makes them.
- **Heads trained on all 10 PPO epochs:** update 75.97 s vs 67.05 s baseline (median of 3), **+8.9 s =
  +13 %**. The bracketed `ridealong` phase is 6.44 s per update. That is far past the ~2 % instrument
  budget, so the heads now train on PPO's FIRST epoch only (`RIDEALONG_EPOCHS` = 1) with fused Adam.
- **Heads on epoch 0 only (the shipped config): +0.59 s per update = 0.88 % of arm C's 67.05 s
  update.** MEASURED, 2026-09-30, `src/agents/training/ridealong_step_benchmark.py` on the GPU (the
  heads are separable by construction, so what they add is one step × the minibatches of one epoch).
  One ride-along step at batch 2048 is 12.3 ms (median of 30; 11.6–14.5) × 48 minibatches. Load was 8.5 at
  start and end ("box looks idle"). It cross-checks with the end-to-end run's bracketed `ridealong`
  phase above: 6.44 s / 480 steps = 13.4 ms a step. Row: `overhead_step_2026-09-30.json`.
- **Why the end-to-end tool could not re-measure it.** Two 35-min GPU-lock units at load 16–80 never
  reached a timed call: the trainer's compile startup alone filled them.

## The read

**Instrument:** the offline reader `python -m main.ridealong_read` (CPU forward passes, no games;
it lands with its plumbing smoke in `smoke_2026-09-30/`). It reads the run's
own checkpoints, which carry the TRAINED heads (`policy.ridealong`), on two sets:
- the fixed M5 Lane S bank (`m5_laneS/bank_v1`: 20,712 decisions, 580 battles, outcome-labelled);
- the X4 pre-read's 1,600 ground-truth turns (`m5_laneS/truth_v2`, variant M per F-X4-1).

**Floors.** Every verdict is read against the SAME reader on the SAME checkpoint with FRESH heads
(randomized priors only). The heads must beat their own untrained floor, not only chance.
`smoke_2026-09-30/` holds those floors for the three pre-read checkpoints.

**Read points:** 10M and 25M (trajectory only; no verdict), and the run's FINAL checkpoint (the
verdict). **Intervals:** 95 % battle-clustered percentile bootstrap, 1,000 draws, fixed seed (the
pre-read's method). The TB `ridealong/*` series are MONITORING: they are on-policy rows the heads
just trained on, and are never a verdict.

| # | question | meter | decision rule (FINAL checkpoint) |
|---|---|---|---|
| R1 | Does ENSEMBLE disagreement predict V's actual error BEYOND V's own uncertainty? | the members' spread in LOGIT space (`ens_logit_std`). PRIMARY: its AUROC for \|V − z\| > 0.5 on the bank read WITHIN quintiles of V's own binary entropy (`auroc_within_ref_quintiles`). A spread of member PROBABILITIES is mechanically largest where V is near 0.5, so the raw AUROC partly re-reads V's own uncertainty: the smoke's UNTRAINED priors score raw 0.63–0.66 but within-quintile 0.54–0.56. Secondary: the raw AUROC, and the top/bottom-decile error ratio | **USABLE** for allocation iff the within-quintile AUROC's lower bound > 0.56, i.e. above the untrained-prior floor of 0.54–0.56, **and** the top/bottom-decile ratio's lower bound > 1.25. **NOT DETECTED** iff the within-quintile interval contains the untrained floor, re-read by the same reader on the SAME checkpoint with fresh heads. Anything else is **WEAK** |
| R2 | Does RND novelty predict V's error, and flag unfamiliar STATES? (State-level floors and method: `rnd_states_2026-09-30/`) | the same AUROC for `rnd_z`. Coverage: the AUROC of `rnd_z` for held-out rows whose TEAM is absent from the reader's train split vs present | error: the R1 rule. **COVERAGE METER** iff the unseen-team AUROC's lower bound > 0.55 |
| R1×R2 | Which is the better allocator? | the paired AUROC difference (ensemble − RND), same resamples | the winner is the one whose paired interval excludes 0. Otherwise **TIED** (use both) |
| R3 | Does A rank actions consistently with the ground truth? | within-turn Spearman of A (the member mean) vs truth, over legal actions, on the 1,600 turns. Reference rows: the policy's own logits on the same turns, and the pre-read's one-ply Q̂ (ρ 0.226–0.243) | **CONSISTENT** iff ρ_A's lower bound > 0. **ADDS BEYOND THE POLICY** iff the paired ρ_A − ρ_logit has lower bound > 0. **A IS A POLICY ECHO** iff that paired interval contains 0 while `corr(policy logit, A)` > 0.5. Argmax-A regret on decisive turns is reported beside the policy argmax's |
| R4 | Are the starved near-best moves flagged as uncertain? | AUROC of the per-action A member spread (`adv_std`: each member centred on its OWN legal-action mean, uniform weights, NOT under π; π-centring pulls the members together on the actions π plays, so a π-centred spread flags starved moves even untrained, AUROC 0.73–0.75 at init). It compares STARVED near-best actions (π < 1 %, truth within 0.1 of the best) with (a) every other legal action and (b) the starved actions that are NOT near-best, on the truth turns | **FLAGGED** iff (b)'s lower bound > 0.55, i.e. the spread singles out the GOOD starved moves, not starvation as such, **and** (a) also clears the fresh-heads floor on the same checkpoint. This decides whether A's spread can steer where counterfactual labels or search effort go (X25 USE), when those come back in scope |
| R5 | Does B carry outcome information? | the Brier score of Q_played = V + A(a) + B(b) vs Brier(V) on bank rows where the opponent's action is in α's support. Paired difference, same resamples | **B INFORMATIVE** iff Brier(V) − Brier(Q_played) has lower bound > 0. Report the α support's label rate (1 − the miss rate) beside it |

**Not claimed by this run.** Strength. The heads cannot move it, by construction and by test, so
the run's strength reads (ladder, anchors, untaught meter) are the PRODUCTION RECIPE's own numbers.
They are recorded as the strength baseline for later arms, not as a result about the heads.

**Declared limits.**
- RND is a STATE- and TRAJECTORY-level signal, MEASURED offline (`rnd_states_2026-09-30/`). It
  falls with visitation (ρ −0.34) and flags off-distribution classes (0.75). It does NOT separate the
  successors of starved near-best moves from other unplayed moves. So R2 is read against V's error
  and state coverage, never as an action-level starvation flag; R4's A-spread is the action-level
  read. The offline obs-RND memorised whole battles (0.85 held-out-battle vs train). R2's coverage
  AUROC therefore carries the same check on the run's own checkpoints: held-out rows from
  the same distribution vs the predictor's training rows, reported beside it.
- A and B learn from GAE advantages (λ = the run's `policy_gae_lambda`, V-bootstrapped). Their
  ceiling is therefore bounded by V's within-turn blindness, which the pre-read measured. R3 against
  Q̂ is a like-for-like floor, not a target.
- B is the simple pre-X5 parameterisation: α's support (seats by move id + SWITCH), conditional on
  the opponent choosing a listed option. It is to be re-based onto X5's flat pointer.
- The heads train on PPO's FIRST epoch only (each rollout row once; `RIDEALONG_EPOCHS` = 1, set by
  the GPU overhead measurement). V trains on all epochs, so the ensemble members see each row once
  where V sees it ten times. R1 asks whether their DISAGREEMENT predicts V's error, not whether
  they match V.
- The heads' Adam state resets at every launcher restart (it is not checkpointed).
- On a pre-v126 checkpoint the reader can only attach FRESH heads. Its numbers there are a plumbing
  smoke, never a read (`smoke_2026-09-30/`).


---

## Amendment 1 — the RND VARIANT ENSEMBLE (2026-09-30, registered BEFORE the run launched)

**Owner, 2026-09-30:** *"Can we just ensemble RND, toss one a different learning rate or something, so
we knock them out all at once? If they're fully detached it's a free win at small cost, and we can
compare RND strategies."* The coordinator added the identification meter (e) the same day: does a
predictor slowly IDENTIFY the frozen target everywhere (the distillation, "dark knowledge" effect),
rather than learning it only on the states it visits?

**Why variants.** Plain RND measures cumulative visitation, so it can saturate. "Recently seen" needs
FORGETTING, and the lever for forgetting is the PREDICTOR's memory, not the target. The offline reads
found that obs-RND memorises whole battles (0.85, `rnd_states_2026-09-30/`), and that feature-RND
drifts 2.5–8× as the representation moves.

### What the run adds

- **The argv adds `--ridealong-rnd-variants all`** (`gen3_ridealong_rnd_variants_v1`, config v127).
  It requires `--ridealong-rnd`, which stays exactly as registered: it is `base`, the reference.
- **Checkpoint cadence (RECOMMENDED, the launcher's call):** `--checkpoint-every-steps 2000000`, so
  that a checkpoint exists at every eval cycle. The default is 50,000 vec-calls, which is 2.4M env
  steps at 48 envs, plus one per restart. The snapshot pool (`snapshots/`) is capped and EVICTS, so
  the series below reads `checkpoints/`, which keeps every save. Whether the flag is learning-neutral
  is **UNVERIFIED** (it is expected to be: saving draws no RNG).
- **The variants**, each its own detached predictor with its own Adam, gradient clip, error z-score,
  fail-closed switch and `ridealong/rndv_<name>_*` series. Declarations: `RND_VARIANT_DECLS` in
  `agents/model/ridealong_heads.py`.

| key | variant | input | forgetting lever (declared) | predictor vs target |
|---|---|---|---|---|
| `rnd` | **base** (`--ridealong-rnd`, unchanged) | the observation, running-normalised, ±5 | none: one pass per row at 3e-4 | obs→256→256→64 ReLU MLP vs an obs→256→64 target: the SAME family, predictor one layer deeper |
| `rndv_fast` | **fast** | base's normalised observation | **10×** base's rate (3e-3). Adam's per-step move is ~lr, so the tracking timescale scales ~1/lr; 10× is the top of the 5–10× range, for a detectable contrast | base's predictor, started from base's EXACT weights (same family) |
| `rndv_decay` | **decay** | base's normalised observation | pulled toward its OWN init once per PPO update, θ ← θ0 + γ(θ − θ0), **half-life 10 updates** (γ = 0.933). One eval cycle is 2M steps ≈ 20 updates, so unrenewed learning keeps 25 % after one cycle and 6 % after two. Shrink-toward-init (Ash & Adams 2020; Kumar et al. 2023) | as fast |
| `rndv_small` | **small** | base's normalised observation | cannot memorise: obs→**32**→64, 90,496 parameters (11.5 % of base's 789,312) | **DELIBERATELY LESS EXPRESSIVE** than the 256-wide target, so it cannot identify the target exactly |
| `rndv_feat` | **feat** | the detached `value_pooled` (128-dim), its OWN running normalisation | none (it exists to measure drift LIVE) | base's shapes over the features (128→256→256→64 vs 128→256→64), its OWN frozen target (same family) |

- **Paired by construction.** The three observation variants share base's frozen target and base's
  observation normalisation: one target output per row, several predictors. The same stream would
  give identical statistics, so sharing them is exact. `fast` and `decay` ARE base at step 0, so every
  later difference is the strategy.
- **Window replay: NOT BUILT.** Memory would fit: one update's worth of 10 % fp16 subsamples is
  9.8k rows × 2761 × 2 B ≈ 54 MB of GPU. The design does not. Each row is trained on once, so "train
  only on a recent window" differs from base only if the predictor also FORGETS outside the window. A
  hard window needs a periodic reset or re-fit, which makes its novelty a saw-tooth in the reset
  phase (so (c) would read the phase, not saturation) and its cost bursty. `decay` already
  implements the soft, exponential window at zero memory, with a declared half-life.
- **Detachment, extended and proven** (both proofs fail on revert, mutation-checked 2026-09-30):
  `ridealong_heads_test`: no variant loss reaches a trunk, policy or V parameter, and each variant's
  loss trains EXACTLY its own predictor (not base, not another variant). `feat` also detaches its own
  input, and the TEETH test pins that second stop-grad. `ridealong_update_test`: one real update with
  every variant ON is bit-identical to OFF (params, PPO optimizer state, PPO scalars, RNG). A third arm,
  the four heads WITHOUT the variants, shows every other ride-along tensor, **base included**, is
  bit-identical with them: the reference is untouched.
- **Lifecycle (K8):** every ride-along optimizer is acquired at `_setup_model` with its Adam state
  pre-allocated, and the update creates no optimizer, state or buffer. There is no lazy fallback:
  a missing optimizer raises `RideAlongLifecycleViolation` at the first update. **Launch gate,
  added:** every `ridealong/rndv_<name>_disabled` = 0 at every read. A disabled variant's read points after its disable are VOID for that variant only.
- **Overhead, MEASURED** 2026-09-30 (`ridealong_step_benchmark.py`, RTX 3080 Ti, load ~10 on 16 cpus,
  "box looks idle"; one epoch-0 pass = 48 steps of 2048 + the per-update `decay` pull + the per-update
  CPU meters, median of 5; `overhead_variants_2026-09-30.json`):

| configuration | added per update | % of arm C's 67.05 s update |
|---|---|---|
| the four heads (the registered config) | 0.653 s | 0.97 % |
| + all four variants (this amendment) | **0.974 s** | **1.45 %** |
| each variant's MARGINAL cost inside the full set (all − all-but-it) | fast 0.140 s · decay 0.161 s · small 0.108 s · feat 0.143 s | 0.21 · 0.24 · 0.16 · 0.21 % |
| each variant ALONE on top of the four heads | fast 0.124 s · decay 0.122 s · small 0.108 s · feat 0.084 s | 0.19 · 0.18 · 0.16 · 0.13 % |

  The all-heads total stays inside the ~2 % instrument budget. The marginal and alone costs differ
  because the variants share fixed costs (the identification probe's chimera target forward, one
  backward). Pass-to-pass spread is ±0.05 s, so the per-variant figures are ±~0.07 % of the update.

### The read, amended

**Instrument:** `python -m main.ridealong_read` reads every key: `rnd` (base) and `rndv_<name>`.
`--run-checkpoints <run>` reads every retained `checkpoints/checkpoint_*_steps.zip` as a series
(`rnd_variants_series.json`). Bank, truth turns, battle-clustered bootstrap (B = 1000, seed
20260930) and read points (10M, 25M, FINAL) are as registered.

**Correction for multiple comparisons.** Each comparison below is ONE Holm family at α = 0.05 over
the variants. Bootstrap p-values come from the SAME battle resamples for every key: two-sided p =
min(1, 2·min(#{d ≤ 0} + 1, #{d ≥ 0} + 1) / (B + 1)), and one-sided where stated. An untestable member
enters its family at p = 1, so a family keeps its registered size.

**R2 per variant.** R2's rules (error: the R1 rule on the key's z; coverage) are read for EVERY key,
each against ITS OWN fresh-heads floor: the same reader with `--fresh-heads` on the same checkpoint.
Note that `fast` and `decay` are base at init, so their floors equal base's.

| # | question | meter | decision rule (FINAL checkpoint unless stated) |
|---|---|---|---|
| (a) | Which variant best predicts V's error BEYOND V's own uncertainty? | per key: the within-V-entropy-quintile AUROC of its z for \|V − z\| > 0.5 (R1's primary meter, draws excluded); per variant the PAIRED difference (variant − base) | Family: the 4 variants vs base. **BEATS_BASE** iff Holm rejects with diff > 0 **AND** the variant's own within-quintile AUROC lower bound exceeds its fresh-heads floor's point estimate on the same checkpoint. (The floor gate exists because an UNTRAINED `feat` already "beat base" in the plumbing smoke: it reads V's own trunk features.) **WORSE_THAN_BASE** iff Holm rejects with diff < 0. Else NOT_DETECTED. **Best** = the BEATS_BASE variant with the largest point difference; if there is none, **BASE STANDS**. The ranking of all five is reported, descriptively |
| (b) | Which variant best flags off-distribution and RARE states? | two coverage AUROCs of each key's novelty on all bank rows: (b1) `opp_class` = exploiter vs every other row (the run's opponents are its bots and its own self-play pool; **verify at launch that no exploiter is in its pool**); (b2) late game, the bank's `turn` > the bank's own 95th-percentile turn vs the rest (the threshold is recorded). Paired difference vs base per contrast | Family: 4 variants × 2 contrasts = 8. Per variant: **COVERS_BETTER** iff ≥ 1 contrast rejects positive and none rejects negative; **COVERS_WORSE** the mirror; **MIXED** if both; else NOT_DETECTED. Unseen-TEAM coverage cannot be read: every bank team is a pool team (`rnd_states_2026-09-30/`, FINDING there) |
| (c) | SATURATION: does a variant's novelty stop discriminating as it trains? | per key, the RAW error's median, IQR and rel_spread = IQR ÷ median on every bank row, with CIs. Ratio r = rel_spread(FINAL) ÷ rel_spread(10M), paired resamples | Family: all 5 keys (base included), two-sided on log r. **SATURATED** iff Holm rejects with r < 1 **and** point r ≤ 0.5 (the spread at least halved). Guard: if base, fast AND decay are all SATURATED, every SATURATED reads **STREAM_HOMOGENISED**: the stream lost its variety, which no predictor can be blamed for. Trajectory (no verdict): the reader's series over every retained checkpoint, and the in-run `ridealong/<key>_err_median` / `_err_rel_spread`, aggregated per eval cycle (the median over the cycle's updates). Those are fresh rows each scored BEFORE the predictor trains on them |
| (d) | feat's DRIFT, live | the 10M checkpoint's feat head (its predictor, target and normaliser) scored on the same bank rows through the `value_pooled` of the 25M and FINAL checkpoints: inflation = median error via B ÷ median error via the 10M features; `frac_above_a_p90`; the renormalised control (B's features re-standardised) beside it | Family: the 2 pairs, one-sided H0 inflation ≤ 1. **DRIFT_CONFIRMED** iff Holm rejects. **MATCHES THE OFFLINE PREDICTION** iff the interval overlaps [2.5, 8]. The obs keys' input does not depend on the checkpoint: their drift is 0 by construction. (Plumbing smoke, fresh heads, N0 @ 11.9M → 75M: 7.38× [6.19, 8.58], renormalised 0.94×. That is a smoke, not a read) |
| (e) | IDENTIFICATION: does a predictor learn the target EVERYWHERE, rather than on the states it visits? | two FIXED probe sets: (i) 4,096 bank rows (seed 20260930, seeded shuffled order; on-distribution, never trained on); (ii) their block chimeras, generator `chimera_v1` (frozen): row j's block k is block k of row (j + k·⌊n/7⌋) mod n; the 7 blocks come from the encoder's layout (our team, their team, the actives' context, global, reactive, pair history, event window); block 0 stays row j's own; every block is a real block, but the combination is one no battle produced. For `feat` the chimera observations go through the checkpoint's trunk (action mask: the context donor's). Per key: m_i, m_ii = median raw error on (i), (ii); R = m_ii ÷ m_i. Floor at the SAME checkpoint: every predictor reset to its declared init, with the checkpoint's own normalisers and targets: m_i0, m_ii0, R0. f_i = m_i ÷ m_i0, f_ii = m_ii ÷ m_ii0, ρ = R ÷ R0 = f_ii ÷ f_i, and the TRANSFER INDEX S = log f_ii ÷ log f_i | **INDETERMINATE** iff f_i > 0.7 (the error on visited-like states fell by under 30 %: too little learning to judge). Otherwise, family: all 5 keys, one-sided in each direction. **IDENTIFIES** iff Holm rejects S ≤ 0.5; **SELECTIVE** iff Holm rejects S ≥ 0.5; else UNDECIDED. As a ratio threshold: a key IDENTIFIES iff ρ < f_i^(−1/2). **Justification for 0.5:** S = 0 is "learns only what it visits" (the off-manifold error stays at init while (i) falls, ρ = 1/f_i). S = 1 is "learns the target everywhere" (both fall together, ρ = 1). 0.5 is the geometric midpoint in log-error space: more than half of the on-distribution learning has transferred to states no battle produces. It is scale-free and adapts to how much a key learned. A key that IDENTIFIES is losing its novelty signal by identification, which is saturation by another route. The healthy pattern is (i) falling while (ii) stays high. **Over time:** R, ρ, f_i, f_ii and S per key at every retained checkpoint (`e_identification_series`), the ratio the coordinator asked for. In-run monitor (descriptive): `ridealong/<key>_ident_ratio` (chimera ÷ real-row error on each update's fresh rows; the observation keys only, since `feat`'s chimeras would need a trunk forward) |

**What the architecture column predicts, stated before the run so the read can falsify it.** `small` is
the only key that cannot represent its target, so it is the least able to IDENTIFY. `fast` and `decay`
share base's family and init, so if base identifies they can too, unless forgetting holds them back.
`feat` reads a 128-dim feature space in which a deeper net can fit a fixed random MLP well, so it is
the likeliest to identify. These are expectations, not rules: nothing above depends on them.

### Declared limits of the amendment

- The observation variants share ONE normaliser. A defect in it (a non-finite batch) disables all
  three together, and the four heads too (pinned by `instrumented_ppo_ridealong_terms_test`). `feat`
  keeps running.
- base is clipped JOINTLY with the other three heads (its registered behaviour, unchanged), while each
  variant is clipped alone. Adam is invariant to a constant gradient scale, so the coupling acts only
  as a per-step re-weighting. It is a confound of `fast` / `decay` vs base, declared, not measured.
- The heads' and the variants' Adam state resets at every launcher restart. For `fast` (10× the rate)
  the first steps after a restart are the largest.
- `feat`'s normaliser is CUMULATIVE over the run while its features drift. Part of (d)'s inflation is
  the normaliser lagging the features; the renormalised control separates the two.
- (b) has no unseen-team contrast: the bank has no off-pool states.
- (e)'s probes are bank rows from earlier runs of the same lineage, so they are "on-distribution" only
  as far as this run's states resemble them. Its floor is per checkpoint, so that drift cancels in ρ.
- The in-run TB series are MONITORING. Every verdict above is the reader's, on fixed probes.
