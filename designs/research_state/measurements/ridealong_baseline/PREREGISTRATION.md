# X26 — the ride-along baseline: pre-registration (2026-09-30)

> **AMENDED 2026-09-30, BEFORE LAUNCH** (the run has not started): **Amendment 1**, at the end, adds
> the RND VARIANT ENSEMBLE (`--ridealong-rnd-variants all`), its pre-registered comparisons (a)–(e)
> with a Holm correction across variants, and the startup-acquisition lifecycle gate. Everything above
> it stands as registered.
>
> **AMENDED AGAIN 2026-10-15, BEFORE LAUNCH** (X26 has not started; no X26 data exists): **Amendment 2**,
> at the very end, re-bases this registration onto X26 as it will actually run — the END-STATE production
> network with the bf16 trunk, a fresh seed `rb_x26_s2601`, a 500M cap and the plateau stop rule. It
> REPLACES the argv, the overhead basis, the checkpoint cadence and the read points above, restates R1–R5
> and Amendment 1's (a)–(e) against the end-state graph, declares the K9(b) tie-share handling and records
> the orchestrator's defaults for the owner's open questions. **Where Amendment 2 and an earlier clause
> disagree, Amendment 2 wins**; every earlier clause it does not touch stands.
>
> **AMENDED A THIRD TIME 2026-10-17, AFTER LAUNCH, BEFORE THE RESUME** (X26 is HELD at 136,974,910 steps
> by a K9(b) stop under §A2.7 item 3, diagnosed NUMERICAL at `177f61c0`): **Amendment 3**, at the very end,
> changes ONE thing — how X26 handles K9(b) from the resume on (`--behaviour-check warn` under fixed
> ceilings). No read, comparison, arm, argv flag other than that one, or stop rule of the research changes.

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
- **Launch checklist — the self-play pool load: FIXED 2026-10-01** (`gen3_opponent_inference_load_v1`).
  Before the fix, the first pool load ran the ride-along acquisition on the snapshot after the
  learner froze, and K6 FATALed the run. In this run that would have happened at the self-play
  seeding win rate, hours in. A pool snapshot that differed from the trainee in heads alone was also
  refused (F-MEM). Opponent loads are now inference-only, and they may differ in the ride-along keys
  alone. `opponent_inference_load_test` must be green at the launch commit.

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


---

## Amendment 2 — X26 on the END STATE (2026-10-15, registered BEFORE the run launched; no X26 data exists)

**Who and why.** Written by unit T32 (`designs/ops/TASK_BACKLOG.md` row X26-3) for the X26 exit check, item E9 of
[`design_x26_readiness.md`](../../../endstate/design_x26_readiness.md) §4. Between 2026-09-30 and today the
production network changed under this registration: the end state was ADOPTED (2026-10-12; static per-mon tokens,
trunk depth 3, X5 hypothesis tokens, move resolution, the recovery and fact levers, no critic threat injection), the
bf16 trunk joined the recipe (2026-10-14), the shaped-reward era's constants became namespace constants, the
legacy deletion's U0 / U1 / U3 landed (2026-10-15; U3 changed the fresh-init random stream), and the restart
interval became 6 h. The literal argv above no longer parses, and nine of its clauses describe a run that will not
happen (readiness I11). This amendment states the run that WILL happen and re-bases every clause on it. It is
written before X26 exists, so nothing in it is chosen with knowledge of X26's results.

**What X26 is now.** A FRESH seed of the end-state production recipe at the exit-check commit P_X26, with the four
ride-along heads and the four RND variants attached, run to its own plateau (cap 500M). It is still "the production
recipe with meters attached": the heads observe and never steer, now proven at the bf16 trunk too (§A2.5). It remains
the baseline every later arm compares against, on strength and on these meters.

### A2.1 What changed and why — every stale clause

| # | clause (where above) | was | now | why |
|---|---|---|---|---|
| 1 | the argv | `--restart-interval-hours 3 … --critic winprob --terminal-indicator --victory-value 1.0 --draw-penalty 0 --env-core rust …` | §A2.2's argv, validated at HEAD by `checkargs` (exit 0) and the launcher `--dry-run` ("would launch"), outputs in §A2.3 | the five reward / critic / core flags are DELETED (namespace constants since config v122 / v144; `designs/deleted_flags.md`), so `checkargs` exits 1 on them |
| 2 | the restart interval | 3 h | the launcher default, **6 h** (not typed) | owner 2026-10-04 (`run.DEFAULT_RESTART_INTERVAL_HOURS`) |
| 3 | "the K10(a) RECIPE block … once it lands", N0's recorded flags as the fallback | a block not yet on main | **`--arch production`** applies `designs/production_config.json`'s ARCH surface (39 keys), `recipe.fresh` (29 knobs, including `trunk_precision` **bf16** and `opp_intent_coef` 0.05) and `recipe.sizing` (N 256, n_steps max 384, 98,304 rows per update); `checkargs` reports both surfaces matching | the recipe block landed (K10(a)), and since 2026-10-14 it carries the bf16 trunk |
| 4 | "The FIRST GPU training run on the M5 Rust core" | X26 was to be the first run carrying the heads | X26 is the first DEEP run carrying them; the heads have already ridden 32 runs of the closing test and the bf16 A/B (`rb_es_*`, `rb_bf_*`, 15M each; readiness I10) | history |
| 5 | read points | 10M and 25M (trajectory), FINAL (verdict) | **10M** (the reference point of Amendment 1's (c) and (d); trajectory only), **every 25M** (25M, 50M, 75M, …; trajectory only), **the plateau candidate** (the checkpoint at X26's first `TIER1_PLATEAU`; trajectory only) and **FINAL** (the VERDICT: the checkpoint X26 stops at under §A2.6's rule, or 500M at the cap) | X26 runs to a plateau, not to a fixed 15M–75M length; a curve every 25M shows WHEN a meter becomes usable, and only one verdict is drawn |
| 6 | the overhead basis | +0.974 s per update = 1.45 % of arm C's 67.05 s update (fp32 legacy network) | **declared, not re-measured.** The only measured basis for the end state is the throughput of the 32 runs above, every one of which carried the heads: production bf16 `train_ms` 42.1 s, ≈ 1,600–1,630 env steps/s (`rb_bf_eb`, readiness §2.3), so X26's cost estimate already contains them. If the heads' absolute cost were unchanged (the same 98,304-row buffer, one epoch-0 pass over it), it would be ≈ 2.3 % of the bf16 update and ≈ 1.6 % of the 61.8 s update cycle (**UNVERIFIED** on the end-state network). A heads-OFF production bf16 run does not exist to subtract | the 67.05 s update no longer exists; X26's budget (readiness §2.3) is measured with the heads on |
| 7 | the checkpoint cadence | RECOMMENDED `--checkpoint-every-steps 2000000`; "the default is 50,000 vec-calls, 2.4M env steps at 48 envs" | **`--checkpoint-every-steps 1000000`** (typed) | Tier 1 of the plateau meter needs a checkpoint in [grid, grid + 500k) at its 10M grid (readiness I18), and the read points above fall on the 1M grid. Saving draws no RNG; every closing-test and bf16 A/B run saved at 1M |
| 8 | B "on α's support … to be re-based onto X5's flat pointer" | B's columns were α's seats + SWITCH, conditional on the opponent choosing a listed option | B is **re-based** (`350fb83b`; `FlatOppEffectEnsemble`): its columns are the flat opponent pointer's K seats, OTHER_move, the six switch targets and OTHER_species; its labels are `flat_intent_targets`', so a belief miss is a ROW (an OTHER class), not a masked label. `ridealong_update_test::test_fixed_mass_B_on_the_FLAT_pointer_is_BIT_IDENTICAL_to_learning` pins it | X5 is the only belief representation; α's blob support is deleted (v144) |
| 9 | Amendment 1's window-replay memory figure | "9.8k rows × 2761 × 2 B ≈ 54 MB" | 9.8k × **2845** × 2 B ≈ **56 MB**; window replay stays NOT BUILT, for the reason given there | the observation is 2845-dim; RND reads `obs_dim` at build (readiness I10) |
| 10 | Amendment 1 (d): "MATCHES THE OFFLINE PREDICTION iff the interval overlaps [2.5, 8]" | a decision label tied to the offline fp32 measurement | **DESCRIPTIVE only** (reported, never a verdict); DRIFT_CONFIRMED's rule is unchanged | [2.5, 8] was measured on the pre-era fp32 legacy trunk's `value_pooled` (N0 / K2). The end state's `value_pooled` comes from a different trunk whose rounds run in bf16 in training, so the old interval is no prediction for it |
| 11 | R1's USABLE threshold "lower bound > 0.56, i.e. above the untrained-prior floor of 0.54–0.56" | a floor measured on the three pre-era checkpoints | USABLE iff the within-quintile AUROC's lower bound > **max(0.56, the point estimate of the fresh-heads floor read by the same reader on the SAME checkpoint)**, and the decile-ratio clause unchanged | the registered floor was the untrained priors on pre-era checkpoints; the same-checkpoint floor is what NOT DETECTED already reads. Never looser than registered |
| 12 | the launch gates | `ridealong_update_test`, `ridealong_heads_test`, `opponent_inference_load_test` green at the launch commit | **unchanged, plus** the new bf16 test in `ridealong_update_test` (§A2.5) and `diagnostics_cadence_test::test_the_K9b_excluded_share_reaches_TENSORBOARD_on_EVERY_update_not_on_the_diagnostics_cadence` (§A2.7) | they are what makes "the heads cannot steer" and "a K9(b) spike is visible" true at P_X26 |

Clauses NOT changed: the four heads' spec (ensemble 5, RND, A 5, B 5) and `--ridealong-rnd-variants all`; the
instrument (`python -m main.ridealong_read`), its bank (`m5_laneS/bank_v1`, re-encoded at the current layout) and
truth set (`truth_v2`); the bootstrap (battle-clustered, B = 1000, seed 20260930); every decision rule except 10 and
11; "Not claimed by this run"; Amendment 1's Holm families, (a)–(c) and (e); the declared limits, except as §A2.5
extends them.

### A2.2 The X26 argv: every non-default flag, and why

**The training session builds, validates and launches the actual argv** (rule: a design-doc block is not a launch
command). It launches from the MAIN checkout at P_X26, which records P_X26 as the run's pin. The argv validated for
this amendment at HEAD `264a35e3` (§A2.3):

```
python -m main.launcher --arch production \
  --ridealong-ensemble 5 --ridealong-rnd --ridealong-adv 5 --ridealong-opp 5 --ridealong-rnd-variants all \
  --checkpoint-every-steps 1000000 --steps 500000000 --seed 2601 --device cuda --run-name rb_x26_s2601
```

| flag | value | why it is typed |
|---|---|---|
| `--arch production` | — | the production ARCH surface + `recipe.fresh` + `recipe.sizing` (bf16 trunk included); X26 IS production's recipe at depth |
| `--ridealong-ensemble 5 --ridealong-rnd --ridealong-adv 5 --ridealong-opp 5` | the registered spec | this registration's instruments; critic READOUTS, outside the ARCH surface by declaration, so `--allow-nonproduction-arch` is not needed |
| `--ridealong-rnd-variants all` | fast, decay, small, feat | Amendment 1 |
| `--checkpoint-every-steps 1000000` | 1M | clause 7 (Tier 1's grid and the read points) |
| `--steps 500000000` | 500M | a CAP, not a target; the stop is §A2.6's rule |
| `--seed 2601` | — | a fresh seed AT OR AFTER U3 (`882a3f86`): no pre-U3 seed is "the same seed" at HEAD |
| `--device cuda` | — | the GPU run |
| `--run-name rb_x26_s2601` | — | the Rustboro era's name; a second seed is `rb_x26_s2602` (§A2.8) |

**Deliberately NOT typed (each at its default, each verified in §A2.3 or named here):**
- `--restart-interval-hours`: the default 6 h.
- `--snapshot-ladder-games`: the default 100, **not 0**. Every reference run typed 0 for its speed rule; X26 has
  none, and the interim stop rule's Hodge read needs `snapshot_ladder/` (readiness F8).
- `--behaviour-check`: the default `fatal` (§A2.7 declares the tie-share handling).
- `--trunk-precision`: bf16, from `recipe.fresh` (the fp32 trunk is the declared FALLBACK only if I30's GPU read
  contradicts the canary diagnosis — that would be an Amendment 3, never a silent change).
- `--promotion-sprt`, `--eval-mirrored-pairs`: OFF (§A2.8, Q4).
- `--diagnostics-every`: the fresh default 10 (the K9(b) share is NOT on that cadence: §A2.7).
- `--compile-trainer`: automatic for cuda. KL early stop: OFF (decided 2026-10-03; `target_kl` in the recipe is the
  KL→LR controller's target, not an early stop). `--exploiter`: none, so Amendment 1's (b1) precondition "no
  exploiter in the pool" holds by construction — still verify it from `metadata.json` at the first read.
- The interpreter: `gen3ai_torch28` (torch 2.8.0+cu126; readiness I4). The launcher refuses a cross-build resume.

**The `data/` freeze** (standing rule 6) holds for X26's whole run and is announced at launch (readiness I24).

### A2.3 Validation at HEAD (`264a35e3`, 2026-10-15; `GEN3AI_MODELS_DIR` a scratch dir, rule 5b cores)

The log is committed as `smoke_endstate_2026-10-15/x26_argv_checkargs_dryrun_264a35e3.log`. Re-run both at P_X26
from the MAIN checkout before the launch (exit-check E10); a difference from the output below is a finding.

`python -m main.checkargs --argv "<the argv's flags>"` → **exit 0**:

```
checked 11 flags from --argv
  parser                         : the CURRENT tree's (no --model and no --pin-commit — this argv would run on HEAD)
  accepted by the trainer parser : 11
  launcher-owned (not forwarded) : 0
  unrecognized                   : 0
  no --model: the argv IS the config (nothing to inherit) — the checks below read it with the launch path's own desugars applied
  oracle reveal                  : off (the production observation)
  [ForkLR] no --model: a fresh run has no parent LR to inherit
  shaped-reward parent             : none (no --model, or its reward was the terminal alone)
  retired-lever parent             : none (no --model, or no deleted lever was ON)
  pre-break parent                 : none (no --model, or its config is this generation's)

  ARCH SURFACE vs designs/production_config.json  [production_config@7edfd2bace08]
    --arch production applied the production surface (explicit flags still win)
    ⚠️  NOT applied — by neither the ARCH surface nor the RECIPE surface (the RECIPE block below applies the critic mode and the supervision doses). These critic READOUTS are implied by the win-prob critic or sit at their defaults; type them yourself if you want them pinned:
        --win-prob-mode 'shaping'  --belief-grad-mode 'shaping'  --ridealong-ensemble 5 (typed)  --ridealong-rnd True (typed)  --ridealong-adv 5 (typed)  --ridealong-opp 5 (typed)  --ridealong-rnd-variants 'fast,decay,small,feat' (typed)
    ✓ every ARCH-surface key matches the production mirror (39 of 47 registry toggles are the ARCH surface; 6 critic readouts + 2 non-structural rows are excluded by their own declaration)

  RECIPE SURFACE vs designs/production_config.json recipe.fresh  [production_config@7edfd2bace08]
    --arch production applied recipe.fresh + recipe.sizing (every knob this argv did not type; explicit flags still win)
    sizing      : N 256, n_steps max 384 — verdict SIZING 2026-10-02 (program_rust_core.md Decision record): N* = 256, Part L guard NO LOSS DETECTED (n = 1 flagged), fits at steady state on 07eebe13; recipe.fresh n_epochs stays 10
    trunk       : bf16 — the learner's trunk rounds run under bf16 autocast in R1 (T2 / eval / the K9(b) probe stay fp32); `--trunk-precision fp32` is the explicit opt-out
    ✓ every RECIPE knob matches recipe.fresh (29 knobs)
  ✓ this command still launches
checkargs exit 0
```

`python -m main.launcher --dry-run <the argv flags>` → **exit 0, "would launch"**:

```

── launcher --dry-run — resolving only, nothing will be created ────────────
  role        : FRESH
  run dir     : /tmp/t32_models/rb_x26_s2601   [would be created]
  on restart  : RESUME role — drops FRESH-only --arch (inherited from model_config.json); --model → the run's latest checkpoint
  pin         : 264a35e3a671d74e65b0b7ce354f01e309fce763  (source: head)
                ↳ docs(legacy-deletion): U3 shipped -- the post-U3 fresh-init record (D-LD7), the MAJOR ledger entry, CHANGELOG, manifest section 8 and the next steps for U2 / U4 / U5 / U6
                (a real launch would create an isolated worktree at this commit — the dry run does not)
  steps       : --steps 500,000,000 (fresh run, from 0)
  interpreter : /home/goodlad/miniconda3/envs/gen3ai_torch28/bin/python3  (torch 2.8.0+cu126) — FRESH launch; this torch is recorded at the run's first save
  transport   : in-process bridge [rust] (no Showdown server)
  restarts    : every 6.0h, grace 20.0 min, max 3 crash restart(s), nice 10
  effective   : the argv IS the config (no --model, nothing to inherit)
      --grad-accum-steps     32           (from the argv)
      --fork-lr              None         (from the argv)
      --fork-lr-freeze       False        (from the argv)
  env core    : rust  [the only core]
  (child-only: architecture compatibility (check_compatible against the checkpoint's model_config.json))
  (child-only: the ModelVersion round-trip smoke test)
  (child-only: the resolved --compile-trainer / --compile-opponents decisions (they read the device))
  (child-only: self-play pool SEEDING from a fork parent (agents.training.pool_seed runs in the child))
  (child-only: the observation dimension and arch signature)
  arch        : --arch production — 39 ARCH-surface key(s) applied from designs/production_config.json for every flag this argv leaves unset
  ARCH SURFACE vs designs/production_config.json  [production_config@7edfd2bace08]
    --arch production applied the production surface (explicit flags still win)
    ⚠️  NOT applied — by neither the ARCH surface nor the RECIPE surface (the RECIPE block below applies the critic mode and the supervision doses). These critic READOUTS are implied by the win-prob critic or sit at their defaults; type them yourself if you want them pinned:
        --win-prob-mode 'shaping'  --belief-grad-mode 'shaping'  --ridealong-ensemble 5 (typed)  --ridealong-rnd True (typed)  --ridealong-adv 5 (typed)  --ridealong-opp 5 (typed)  --ridealong-rnd-variants 'fast,decay,small,feat' (typed)
    ✓ every ARCH-surface key matches the production mirror (39 of 47 registry toggles are the ARCH surface; 6 critic readouts + 2 non-structural rows are excluded by their own declaration)
  RECIPE SURFACE vs designs/production_config.json recipe.fresh  [production_config@7edfd2bace08]
    --arch production applied recipe.fresh + recipe.sizing (every knob this argv did not type; explicit flags still win)
    sizing      : N 256, n_steps max 384 — verdict SIZING 2026-10-02 (program_rust_core.md Decision record): N* = 256, Part L guard NO LOSS DETECTED (n = 1 flagged), fits at steady state on 07eebe13; recipe.fresh n_epochs stays 10
    trunk       : bf16 — the learner's trunk rounds run under bf16 autocast in R1 (T2 / eval / the K9(b) probe stay fp32); `--trunk-precision fp32` is the explicit opt-out
    ✓ every RECIPE knob matches recipe.fresh (29 knobs)
  desktop GPU : ✓ no display process (gnome-shell / Xorg / Xwayland / …) holds the GPU
  disk space  : ✓ 42.74 GiB free on /tmp/t32_models vs 32.65 GiB REQUIRED (checkpoint no checkpoint of this run, no fork source and no sized run in the archive)
                ⚠️  checkpoint size UNKNOWN (no checkpoint, fork source or sized run to read) — the checkpoint terms are 0; the in-run watch uses the real size from the first save
                shared compile cache: ✓ 331.89 GiB free on /home/goodlad/.cache/gen3ai/compile_shared (another filesystem) vs its growth 1.44 GiB (cap 20.00 GiB - 18.56 GiB in use) + reserve 4.00 GiB
                the arithmetic:
          periodic checkpoints       0.0 MiB   = 500 x 0.0 MiB  (steps 500,000,000 // 1,000,000 - resume 0 // 1,000,000)
          final/best/aborted         0.0 MiB   = 3 x 0.0 MiB
          eval traces              17.58 GiB   = 250 cycles x 72.0 MiB  (ceil(500,000,000 / 2,000,000))
          pool snapshots             0.0 MiB   = 20 x 0.0 MiB  (cap 20)
          compile cache             2.39 GiB   = 2.39 GiB - 0.0 MiB already on disk
          shared compile cache       0.0 MiB   = growth 1.44 GiB at /home/goodlad/.cache/gen3ai/compile_shared: ANOTHER filesystem, checked below
          sidecar + tb + logs       2.93 GiB   = 500.0M steps x 6.0 MiB/M
          fixed                     16.0 MiB   = metadata, ledgers, sidecar json
          subtotal                 22.92 GiB
          + margin 25%              5.73 GiB
          + reserve left free       4.00 GiB   (the 2026-10-09 near-miss had 2.2 GB free at 100 %)
          = REQUIRED               32.65 GiB
  showdown deps : ✓ /home/goodlad/dev/gen3ai/.claude/worktrees/agent-a512b7af5f71a4fe2/deps/pokemon-showdown serves team validation (13 required files present, node /usr/bin/node)
  ✓ DRY RUN — this command would launch. Nothing was created or modified.
dry-run exit 0
```

The `⚠️ NOT applied` line is informational: the ride-along flags are typed (they are this registration's
instruments), and `--win-prob-mode` / `--belief-grad-mode` resolve to the win-prob critic's implied `shaping`.

### A2.4 The instrument still runs on the end-state graph (plumbing smoke, no verdict)

`python -m main.ridealong_read` read `rb_bf_eb_s2001`'s 9.0M checkpoint (config v155, bf16 trunk, TRAINED heads) at
HEAD on 40 of the bank's 580 battles: the bank re-encoded at the 2845-dim layout (1,172 decisions), the heads read
as `"trained"`, and meters (i)–(iii) and Amendment 1's (a), (b) and (e) produced output in 50 s (peak 1.23 GB).
Evidence and caveats: `smoke_endstate_2026-10-15/README.md`. **Two registered reads have NO reader code** and must
be built, offline, before the VERDICT read (FINDING A2-F2): R1×R2's paired AUROC difference (ensemble − RND, the same
resamples) and R5's Brier(V) − Brier(Q_played). Neither blocks the launch: every read is offline over retained
checkpoints, and the first one with a verdict is FINAL.

### A2.5 R1–R5 and Amendment 1's (a)–(e), restated against the end-state graph

**The guarantee under the bf16 trunk (readiness I12).** Production's learner runs its trunk rounds under bf16
autocast in R1; T2, the eval core and the K9(b) probe stay fp32 (`trunk_precision`). The heads ON vs OFF guarantee
was proven at fp32 only. `ridealong_update_test::test_BF16_one_update_is_BIT_IDENTICAL_with_the_heads_on_and_off`
now proves it at bf16: one real update of the production network (rollout at fp32, learner at bf16), heads OFF and
ON with every variant, leaves the policy / trunk / V parameters, the PPO optimizer state, every PPO scalar and the
torch / numpy / python RNG BIT-IDENTICAL. It is not vacuous: both arms' rounds are the bf16 classes, and the bf16
update differs from the fp32 update on the same rows. Mutation-checked 2026-10-15: dropping the `.detach()` in
`RideAlongBatch.detached` fails it (the heads move `species_embedding`); a ride-along step that leaves the trunk at
fp32 (a no-op at fp32) fails it too.

**What each read means on the end state** (the rules are §"The read" and Amendment 1, with clauses 10–11 above):

| read | what it reads on the end state | change |
|---|---|---|
| R1 ensemble vs V's error | the members over the DETACHED `value_pooled` of the end state's value side (the value CLS pool + the entity pool; no threat injection since U3) | USABLE threshold per clause 11 |
| R2 RND novelty | base RND over the 2845-dim observation (running-normalised); coverage as registered | none |
| R1×R2 | the paired AUROC difference | **reader code NOT BUILT** (A2-F2) |
| R3 A vs ground truth | A's member mean vs `truth_v2`'s per-action values on the 1,600 turns. The truth's continuations are PRE-ERA greedy policies (K2, N0, C_fix), so ρ_A is consistency with THEIR action values; the policy-logit row is the like-for-like reference, and the pre-read's Q̂ (ρ 0.226–0.243) is DESCRIPTIVE | the Q̂ row is descriptive (its policies are pre-era) |
| R4 A's spread on starved near-best moves | as registered (`adv_std`, own-legal-mean centring) | none |
| R5 B carries outcome information | Q_played = V + A(a) + B(b), b = the opponent's FLAT-pointer class (a seat, OTHER_move, a switch target or OTHER_species); every bank row with an aligned one-ahead opponent label, not "in α's support" (clause 8); report B's label rate beside it | **reader code NOT BUILT** (A2-F2); also UNVERIFIED: that `bank_v1`'s rows carry the one-ahead opponent label in flat-pointer terms — the R5 build checks this first |
| (a)–(c), (e) | as registered; (e)'s seven blocks are re-derived from the current encoder layout by `block_chimera`'s edges (`chimera_v1`, code frozen) | none |
| (d) | as registered | its "MATCHES THE OFFLINE PREDICTION" is DESCRIPTIVE (clause 10) |

**Declared limits, added.**
- The bank (`bank_v1`) and the truth turns are states from PRE-ERA battles (`ai_v14_*`), re-encoded at the current
  layout. X26 never visits them, so every bank AUROC is a read on a shifted distribution, and RND's novelty there is
  "novel to X26" by construction. Amendment 1 already declared this for (e). Each read's own fresh-heads floor on the
  same checkpoint is the control.
- The heads are trained at bf16 trunk features but READ by an fp32 reader forward (the reader builds a fresh fp32
  policy from the checkpoint, as T2 does). The difference is the trunk's bf16 rounding of `value_pooled` and the
  pointer inputs (|Δ log π| p50 3e-5 to 5e-3 by state, `k9_bf16_2026-10-10/`). It is declared, not measured, for the
  heads' outputs.
- A 500M run takes ~5,100 updates and ~25 launcher restarts at 6 h (≈ 3–4 days, readiness §2.3). The heads' and the
  variants' Adam state resets at every restart (registered); for `fast` the first steps after each restart are the
  largest.

### A2.6 Read points and the stop rule

**Read points** (clause 5): 10M, every 25M, the plateau candidate, FINAL. Each read runs OFFLINE under rule 5b,
over `checkpoints/checkpoint_<N>_steps.zip` (the nearest retained checkpoint at or after the point; `--run-checkpoints`
reads the whole series for the trajectories), and for every read point ALSO with `--fresh-heads` (the floor). A read
point's output dir is new per point; nothing is read from `models/` but the checkpoints.

**The stop rule (readiness §2.4; the interim rule until the cycle monitor U5 and the two-tier decision U9 land).**
X26 stops only when ALL of the following hold:
- Tier 1 reads `TIER1_PLATEAU` (`python -m main.plateau tick` then `status`, registered Δ 10M / W 50M, so 70M at the
  earliest);
- the Hodge cyclic width (`python -m main.elo <run> --hodge-bootstrap 200`) is within its bootstrap noise floor;
- the bot rates in the in-loop `eval_results.jsonl` and a Metamon SmallRL greedy anchor read are flat over the same
  50M window.

Any disagreement means CONTINUE. Once U5 and U9 land, the two-tier rule replaces the last two bullets for every later
check. The cap is 500M. The rule can only stop LATE, never on a false plateau. The checkpoint the run stops at is
FINAL; the read at it is the verdict.

### A2.7 The K9(b) tie-share handling, declared up front (readiness I19)

K9(b) excludes a row whose forward sits within a relative 2e-4 of a selection cutoff, and stops the run under the
default `--behaviour-check fatal` if ONE update's excluded share exceeds `FP32_EXCLUDED_CEILING` 0.15
(`agents/training/rust_rollout/consistency.py`). Production's mean share is ≈ 0.03 at 15M with a highest single
read of 0.077, and the mean is unlikely to reach 0.15 before ≈ 200M (readiness §1.1); the risk at X26's length is a
single-update SPIKE. X26 adopts the closing test's handling (`design_endstate_closing_test.md` §3.7), adapted to a
single deep run:

1. **A tie-only stop is a known false alarm.** A K9(b) stop whose ONLY violated condition is the excluded share,
   with `max |d log π|` over the judged rows < 1e-4 (both in the printed message and in
   `behaviour_violations.jsonl`), resumes from its own latest checkpoint: the launcher's crash restart, and once its
   3-crash budget is spent, a `--model` resume of its own latest checkpoint at the same pin and argv. There is no limit
   on the number of such resumes; each stop is banked in the run's notes. Resuming from the latest checkpoint in
   place is safe for the plateau ledger: the stop falls after that checkpoint, so no decided Tier-1 node is
   overwritten (unlike readiness F17's rollback).
2. **Once the crash budget is spent on tie-only stops,** the resume runs with `--behaviour-check warn`, provided
   EVERY probe after the switch keeps `behaviour/max_abs_dlogp_judged` < 1e-4. The training session verifies that
   from the TensorBoard events at each read point and before each restart (`python -m main.ops.tb_read <run> --tag
   behaviour/max_abs_dlogp_judged --tag behaviour/excluded_frac`), and verifies with the launcher's `--dry-run`
   that a restart carries `warn`. Banked as a deviation.
3. **Any K9(b) stop with `max |d log π|` ≥ 1e-4, or any other failure, is NOT a false alarm.** It is a GIGO stop:
   the run stays stopped until its cause is diagnosed (the violation dump holds the rows and weights). X26 is one
   deep run, not a seed of an A/B, so there is no "replace the seed": a real mismatch is a finding that may void the
   checkpoints after the last clean update.
4. **A move-reveal or moves-like belief change to production** would sit at the ceiling from step 1 (the knock-in
   forks' moves arm read 0.16 medians, readiness §1.1). That is a re-measure-the-ceiling trigger for a FUTURE arm,
   not an X26 risk (X26 has no reveal).

**The share is in the record for EVERY update, so a spike is read, not inferred.** The probe records
`behaviour/excluded_frac`, `rows_excluded`, `rows_judged` and `max_abs_dlogp_judged` on every update, independent of
`--diagnostics-every` — verified on `rb_bf_eb_s2005`'s raw event file (153 values for its 153 updates; an update is
98,304 rows, so a 15M run has ≈ 153 updates, not ≈ 1,490), and now pinned by
`diagnostics_cadence_test::test_the_K9b_excluded_share_reaches_TENSORBOARD_on_EVERY_update_not_on_the_diagnostics_cadence`
(three real updates at the production cadence, two of them non-diagnostic: three events per tag in a real
TensorBoard file; mutation-checked by gating the probe on the diagnostics plan). **Read it with
`python -m main.ops.tb_read`** (it loads every event). The TensorBoard WEB UI keeps at most 1,000 points per scalar tag
(`samples_per_plugin` default), so on X26's ≈ 5,100 updates its plot shows ≈ 1 in 5: never read a per-update maximum
off the UI.

**Dated note, 2026-10-15 (the X26 exit check). This is an operational note, not a change to any registered rule.
It covers the bf16 CANARY beside K9(b).** X26 runs the compiled bf16 learner, so the R1 canary (every 100 updates)
and the startup gate (every restart) judge it on the K9 golden's rows, relocated off the PPO clip boundary (T36,
confirmed on CUDA at `87e493c3`).
- The rule refuses only an arm whose taken-action |Δ log π| from the fp32 reference exceeds the margin
  M = 0.72, or an arm that switches a row's clip branch.
- On the toy's checkpoints, the compiled arm's shift grew from 0.050 (15M) to 0.110 (72.66M), with eager bf16
  growing alike.
- The X26 Training session watches `canary_verdicts.jsonl` `.watch.scheduled.clip_margin.max_abs_dlogp.compiled` at
  every read point above and before every planned restart. The bands and actions are in
  [`design_x26_readiness.md`](../../../endstate/design_x26_readiness.md) §4.2: GREEN ≤ 0.144; AMBER ≤ 0.18 means
  stage a raised constant; RED ≤ 0.72 means deliver it at the next planned restart, never stopping the run for it;
  REFUSED means the documented FATAL procedure.
- As with item 3 above, a canary FATAL is NOT a tie-share false alarm and is never resumed blind. A move to the fp32
  trunk would be an Amendment 3. A constant raised under §4.2 changes only the check's fixture, never training, so
  it is banked as a deviation in the run's notes, like the `warn` switch of item 2.

### A2.8 The owner's open questions: the orchestrator's defaults, recorded

The owner's Q1, Q2 and Q4 (readiness §6.2) are unanswered at registration. Under the 15-minute standing rule the
orchestrator adopts the recommendations; an owner answer before the launch supersedes them by an Amendment 3, and
none after the launch changes this run's argv.
- **Q1 (seeds): ONE seed, `rb_x26_s2601`, plus a SECOND, `rb_x26_s2602`, started at seed 1's FIRST Tier-1 FLAT
  check** (not at its stop), so the replicate is running when the claims fall due. Seed 2 is the same argv with
  `--seed 2602 --run-name rb_x26_s2602` and `--pin-commit P_X26` (HEAD will have moved; a replicate is the same pin
  + a new seed). How the GPU is shared between the two is the orchestrator's scheduling call. This registration's
  verdict reads seed 1; seed 2's reads are the replication, read by the same rules at its own FINAL.
- **Q4 (promotion): `--promotion-sprt` and `--eval-mirrored-pairs` stay OFF**, so X26's plateau is production's
  recipe at depth. SPRT promotion becomes a registered challenger that must beat that plateau.
- **Q2 (the plateau meter): the stop rule plus a VALIDATED Tier 1** (T33, 2026-10-15), with the interim rule of §A2.6;
  U5 and U9 land during X26's first ~100M. Tier 1 ticks at the registered Δ 10M / W 50M on the offline CPU lane.

### A2.9 Findings of this amendment (standing rule 7)

- **A2-F1. The readiness doc's F5 premise was wrong.** TensorBoard does NOT keep "~1 in 10" K9(b) shares: the
  probe records the share on every update, and a 15M production run has ≈ 153 updates (98,304 rows each), all 153 in
  the raw events. The "~1,490 updates" figure was off by 10×. What IS true: TensorBoard's web UI downsamples to 1,000
  points per tag, which bites at X26's ≈ 5,100 updates. So T32(c)'s build is a regression TEST of the every-update
  guarantee plus the declared reader (`main.ops.tb_read`), not a new jsonl writer (a second record of the same
  quantity would only add a second producer to keep in sync).
- **A2-F2. Two registered reads have no reader code:** R1×R2's paired AUROC difference and R5's Brier(V) −
  Brier(Q_played) on the flat pointer (plus the UNVERIFIED bank-label precondition for R5). Both are offline builds
  (~0.5 agent-day together, est.) that must land before the FINAL read; neither blocks the launch.
- **A2-F3. The heads' overhead on the end-state network is UNVERIFIED** (clause 6): ≈ 2.3 % of the bf16 update if the
  2026-09-30 absolute cost carried over, above the ~2 % instrument budget the 2026-09-30 note set. It is inside every
  measured end-state throughput, so X26's GPU-h estimate holds either way. A GPU re-measure
  (`ridealong_step_benchmark.py` on a production bf16 checkpoint) can ride any leased window; it is not a launch gate.
- **A2-F4. The launcher dry-run's disk check under a scratch `GEN3AI_MODELS_DIR` cannot size checkpoints** (no sized
  run there): it required 32.65 GiB with the checkpoint terms at 0. At X26's 1M cadence the checkpoints alone are
  ≈ 500 × 55 MB ≈ 27 GB more (readiness §2.3: ≈ 36 GB at 500M in total). The real launch runs from the main checkout,
  whose archive holds sized runs.

---

## Amendment 3 (2026-10-17, after launch, BEFORE the resume): K9(b) under `warn` with fixed ceilings

**Registered before X26 resumes.** X26 stopped at update 13,900 (136,974,910 steps, 2026-10-17 06:49:40) on a
K9(b) max-violation: one judged row at fp32 `max |d log π|` 1.01e-4 against the 1e-4 bar (p99 9.2e-6). Under §A2.7
item 3 that was a GIGO stop, held until diagnosed. The diagnosis (`177f61c0`,
[`x26_k9b_stop_2026-10-17/README.md`](../x26_k9b_stop_2026-10-17/README.md)) found it **NUMERICAL**. The violating
row is the most fp32-ill-conditioned row in the scan. Every fp32 path misses the exact float64 value by 0.6–1.65e-4,
and the stored behaviour value is the closest of them. Stale weights, row misalignment, a mask fault and a near-tie
are each ruled out, and no checkpoint is void. The per-update judged max grows linearly with training. Under `fatal`
the forecast is ≈ 110 more stops by 500M (one per ≈ 26M steps now, one per ≈ 1.5M by the cap), each one a
predicted rounding event that rule 8 forbids treating as a fault.

### A3.1 What changes (X26 only, from the resume on)

1. **X26 resumes with `--behaviour-check warn`.** K9(b) still runs and records on EVERY update. A violation of the
   1e-4 bar is logged and dumped (under `warn` its scan is a seeded 4,096-row sample) and no longer stops the run.
2. **The run is STOPPED as a GIGO stop under §A2.7 item 3** if ANY ONE update shows:
   1. `behaviour/max_abs_dlogp_judged` ≥ **1e-3**: 10× the bar, and ≈ 16× the forecast median judged max at the
      500M cap (≈ 6e-5). The fitted rounding tail puts the chance at ≈ 5e-9 per update even there, so a fault that
      moves one row by 1e-3 is caught;
   2. `behaviour/p99_abs_dlogp_current` ≥ **1e-4**: 10× the 9.3e-6 at the stop, and ≈ 4× the cap forecast
      (≈ 2.4e-5). Stale weights or a transport fault move EVERY row by ≥ 1e-2, so the p99 is the many-row detector;
   3. any non-finite |Δ|.

   The ceilings are fixed thresholds on the logged per-update statistic (rule 8), each ≥ 4× from its forecast at
   the cap. There is no averaging and no tolerance.
3. **The cost, declared:** a LOCALIZED fault between 1e-4 and 1e-3 on ONE row now passes. That is exactly the band
   the diagnosis shows fp32 rounding reaches on this policy.
4. **Item 2's `warn` clause of §A2.7 (judged max < 1e-4) is superseded for X26 by these ceilings.** Items 1, 3 and 4
   of §A2.7 stand: a tie-only stop cannot occur under `warn`, but any stop for another cause is still a GIGO stop.

### A3.2 How the ceilings are read (until a watcher enforces them)

The Training session reads both statistics from the TensorBoard events (never the web UI, which downsamples):

    python -m main.ops.tb_read models/rb_x26_s2601 --tag behaviour/max_abs_dlogp_judged --tag behaviour/p99_abs_dlogp_current

It reads them at every read point (every 10M plateau tick), before and after every restart, and at least once a
day. The read covers every update since the previous read, judged update by update.

**The CLI prints four decimals**, too coarse to judge a 1e-4 ceiling deterministically (+0.0001 covers
0.5e-4 … 1.5e-4). The verdict therefore comes from the SAME events at full precision, through the reader's own
loader (`main.ops.tb_read.load`). The Training session's ceiling read does this, one point at a time, and prints
each tag's window max, its step, and OK / CEILING / ABSENT.

The read of X26's whole history (1,390 points, 201,984 → 136,974,910, every rollout) at registration:

| tag | window max | at step |
|---|---|---|
| judged max | 9.73e-5 | 130,094,227 |
| p99 | 1.35e-5 | 111,802,321 |

There are 0 non-finite values and both are below their ceilings. The u13900 violation (1.01e-4) is not among the
logged points, because the probe raised before it recorded that update. A ceiling crossing is a stop
handled by the item-3 procedure. A watcher that applies the ceilings automatically is worth building (README §4
(C)), but it is new ops code and not a precondition of this amendment.

### A3.3 The resume

- X26 resumes from **`final_model_exception.zip`**: the state after update 13,899, bit-identical to the weights that
  played the fatal rollout, with its optimizer state.
- It does NOT resume from `checkpoint_136000153`. With a NUMERICAL verdict nothing after 136.0M is void, so falling
  back would discard ≈ 975k clean steps.
- It resumes at the SAME pin `9c033489`, with the run's own argv plus `--behaviour-check warn`. checkargs and the
  launcher's `--dry-run` run first; the `--dry-run` shows that `warn` is carried. Every later restart is verified the
  same way.
- **Restart interval, in the same relaunch (an operational change, not a research rule):**
  `--restart-interval-hours 24`, up from 6. The orchestrator decided this on 2026-10-17 under the owner's standing
  15-minute order, and the owner may revert it at any restart. The reasons:
  - the restart research found no throughput benefit from restarts, and a real cost per restart;
  - every restart re-draws the startup gate's per-restart lottery (`9ab23c93`).

  The compensating guard: the Training session samples the trainer's VmRSS at every heartbeat. Growth > 2 GB over its
  post-start level ⇒ a manual restart, on the orchestrator's call.

### A3.4 Findings (standing rule 7)

- **A3-F1.** The ceilings are read by hand until a watcher exists. An over-ceiling update is caught at the next read,
  not at once. Up to one read interval (≤ 10M steps, ≤ 1 day) of training can follow it before the stop. That
  training is then void back to the last update below the ceilings.
- **A3-F2.** The linear-growth forecast is an extrapolation (README F5). If the judged max grows faster than linearly,
  the p99 ceiling is reached sooner than forecast. That would be a stop and a re-diagnosis, not a silent pass.

## Amendment 4 (2026-10-17, X26 at ≈ 212M) — a WATCH TRIGGER on the K9(b) rounding tail

Registered after the tail refresh `d29f781c` (`measurements/x26_k9b_tail_refresh_2026-10-17/`) and before any trigger
below has fired. The orchestrator registers it under the owner's full-discretion grant (2026-10-17). The Amendment-3
ceilings are UNCHANGED: a stop still needs judged max ≥ 1e-3, p99 ≥ 1e-4 or a non-finite value.

### A4.1 Why

- Every new dump with its weights (u19300, u20490, u20630) is NUMERICAL in 177f61c0's class. Each violator ranks 1–2
  of 33 on fp32 error and dither spread, with a gap 3.2–3.7 dither σ. There is no fault.
- The SCALE grows faster than linearly. The best-AIC exponential form gives P(p99 ≥ 1e-4 before 500M) = 0.66, with
  5 % / 50 % first crossings at 439M / 492M. The linear and power forms give ≤ 0.04. The data cannot tell the forms
  apart yet (ΔAIC 3.0); it should by ≈ 300M.
- Amendment 3's ≥ 4× margin at the cap no longer holds for p99: it is 1.4× (exponential) or 3.2× (linear). That is
  A3-F2's case, so it gets an early warning rather than a surprise stop.

### A4.2 The triggers (read with the A3 full-precision read, at every tick, every restart and daily)

- **W1:** any CLOSED 10M window whose median per-update p99 is ≥ 3.0e-5.
- **W2:** any single probe with p99 ≥ 5e-5 or judged max ≥ 5e-4.
- **Rule 8:** a value within 1e-9 (relative) of a trigger is recorded and does not fire.

**Either trigger ⇒ the A3-F2 RE-DIAGNOSIS.**
- What runs: the tail refresh's `replay_dumps.py` and `forecast.py` on the newest dumps, with X26's pinned code, CPU
  only.
- The verdict:
  - **NUMERICAL** ⇒ continue under the unchanged ceilings, and record the new forecast.
  - **NOT NUMERICAL** ⇒ HOLD at the next checkpoint per §A2.7(3).
- A trigger is NOT a stop and voids nothing.

### A4.3 Standing items

- **The forecast:** `forecast.py` re-runs at every 25M read point (≈ 1 CPU-min) and its table is logged with the tick.
- **A4-F1 (from the refresh's F1):** only 3 violations per process dump their weights (`_DUMP_WEIGHTS_MAX = 3`).
  - u21290 and u21440 cannot be replayed, and no later event can be until the next restart, which resets the count.
  - A re-diagnosis after the cap is hit reads the newest restart's dumps. If none exists, it reads the JSON-record
    checks: same legal set and argmax, per-action shift.
  - Raising the cap is a code change at the pin, so it is NOT done mid-run.
- **A4-F2 (correction to §A3.3):** the live child resumed from `checkpoint_136000153`, not `final_model_exception.zip`.
  That is the startup-gate refusal recorded in readiness §4.2. TensorBoard therefore holds two lineages over
  136.2–137.0M, and u13900 is on the abandoned one.

## Amendment 5 (2026-10-18, X26 at ≈ 272M) — the OWNER's stop rule replaces §A2.6's interim rule

**Owner, 2026-10-18 ≈ 12:05 PDT:** "I would like 100M extra steps after a plateau or 500M, whichever occurs first."
**Disclosed timing:** this was decided AFTER the 270M tick read FLAT_ONCE (49.4 %, 720 pairs), and after the orchestrator
reported that §A2.6's Hodge clause (b) had read CONTINUE at every tick (p 0.005), so the interim rule would run to the
cap. It is the owner's decision, recorded as made with that data in view, not a pre-registered rule.

### A5.1 The rule

- **The plateau** = the FIRST Tier-1 `TIER1_PLATEAU` call, i.e. two consecutive FLATs under the registered Δ 10M / W 50M
  GSPRT. Its step P is the tick's node step, e.g. 280,000,000 if the 280M tick is the second FLAT.
- **X26 stops at the first periodic checkpoint at or after P + 100,000,000 steps, or at 500M, whichever comes first.**
  The run is stopped by PID at that checkpoint; the launcher's `--steps` cap is unchanged.
- **The clock is not reset.** A later GAIN at a Tier-1 tick neither moves P nor the stop.
- **Superseded:** §A2.6's clauses (b) Hodge and (c) bots + SmallRL no longer gate the stop. They are still read at every
  tick and reported, descriptive only.
- **Unchanged:** the Amendment-3 GIGO ceilings, the Amendment-4 watch, and every registered read.
- **The stop checkpoint is FINAL; the read at it is the verdict.** The checkpoint at P is also retained and named
  `x26_plateau` for the plateau-drawn bank and the X35 forks (P* = P, unless the orchestrator records otherwise before
  any fork).
- Tier-1 ticks continue after P, under the 6,000-pair ops cap, as a trajectory.
