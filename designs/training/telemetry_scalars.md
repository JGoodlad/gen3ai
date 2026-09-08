# Training — telemetry scalars

Moved verbatim out of `src/agents/training/CLAUDE.md` on **2026-09-07**, when that leaf was
split by topic (it was 8,219 lines / 676 KB, loaded in full for any session touching the
training package). Each section below is unchanged, including its dated measurements.

`src/agents/training/CLAUDE.md` keeps the heading and the opening paragraph of each, and
points here. **This file is the owner of the detail.** The census detail, the gradient-balance
probe and the per-edge/per-cell liveness families were added in the **2026-09-08** second pass.

---

> **Note:** the census's *currency table*, its `gen3_tb_relevance_v1` headline and the 28-tag
> WIN-PROB dashboard stay in `src/agents/training/CLAUDE.md` — `tb_relevance_test.py::
> test_the_census_table_carries_an_era_column` pins both anchors there. Everything under them —
> the site/tag counts, the group table, the NOISE/REDUNDANT/CONDITIONAL classification and the
> five per-group tag tables — is in this file, together with the capacity telemetry, the
> gradient-balance probe, the `signal/` group and the scaffolding gauge.

## Live capacity telemetry (`--capacity-telemetry`, `capacity_telemetry.py`)

**Three continuous saturation early-warnings that ride the train loop.** They exist because every
previous answer to *"is the network out of capacity?"* here has been an expensive one-shot probe —
a rank sweep, an ablation, an offline battery — each returning a NUMBER at a MOMENT. Saturation is
not a moment; it is a trend, and a trend measured twice is a line through two points. Everything
here is cheap enough to run on every `train()`, so the reading that matters (the SHAPE of the curve
over tens of millions of steps) exists at all.

**Read every one of these as a TREND. None has a meaningful absolute value.**

| scalar | what it answers | alarm shape |
|---|---|---|
| `capacity/canary_loss` | how well a small detached head fits K=4 synthetic obs targets (EMA) | rising |
| `capacity/canary_recovery` | post-reset ÷ pre-reset loss of the target that was last re-seeded | **degrading from reset to reset, at a MATCHED `canary_age`** |
| `capacity/canary_age` | env steps since the last reset | — (the x-axis `recovery` must be read against) |
| `capacity/canary_loss_reset` · `canary_resets` · `canary_steps` | the reset target's own EMA · reset count · updates this `train()` | `canary_steps` = **0** with the flag ON ⇒ the probe is measuring nothing |
| `capacity/halfbatch_cosine` | do two halves of one minibatch agree on the shared trunk? | falling to 0 / **negative** |
| `capacity/halfbatch_grad_norm_ratio` | are the two halves' gradients comparable in size? | ≪1 ⇒ one half dominates; read it before believing a low cosine |
| `capacity/feature_velocity` · `_cos` · `_rel` | how far a FROZEN 256-row probe batch's `value_pooled` moved since the last measurement | falling **while `train/grad_norm` holds** ⇒ weights move, functions do not |

### 1. The plasticity canary — the centerpiece, and the only SUPPLY-side probe

A small head (`LayerNorm → Linear → ReLU → Linear`, K outputs) regresses the trunk's **detached**
`value_pooled` onto K=4 synthetic targets that are pure functions of the observation. Every
`--canary-reset-steps` env steps (default 1,000,000) **ONE target is re-seeded, round-robin** — and
that reset is the whole instrument. Re-fitting a *brand-new* random function of the obs, from the
same representation, with the **same head weights** (they are deliberately NOT re-initialised),
measures how much usable structure the representation still SUPPLIES. A trunk that has collapsed
onto the policy's current answers re-fits slower and plateaus higher.

**THE TARGET FAMILY — quoted here because a deferred OFFLINE probe must use the SAME one or the two
instruments do not cross-validate.** `capacity_telemetry_test.py` pins the arithmetic literally:

```
seed(k, e) = 20260823 + k + 1_000_000 * e         # k = target index, e = its reseed count
P[:, k]    = torch.randn(obs_dim, generator=torch.Generator("cpu").manual_seed(seed(k, e)))
target_k   = tanh( obs @ P[:, k] / sqrt(obs_dim) )
```

The generator is **CPU-seeded always**, so the same `(k, e)` gives the same column on any device and
in any process. `tanh` bounds the target into (-1, 1) so `canary_recovery` is a ratio of two
comparable scales; an unbounded target would make the two sides of that ratio incomparable.

⚠️ **It measures the REPRESENTATION's richness, not the policy's headroom, and those are different
claims.** A rising `canary_loss` says the trunk carries less recoverable obs structure than it did.
It does not say the policy would be stronger if it carried more. Treat it as a signal that
something is narrowing, then go find out what — the same discipline `dV`-ablation results get here.

**The gradient CANNOT reach the trunk, and that is structural rather than careful.** The head is
owned by the **PPO object**, not the extractor — no `state_dict` key, no policy-optimizer position
(the ai_v6_13 "128 vs 5" class is unreachable) — it trains through its own Adam over its own
parameters, and `PlasticityCanary.step` detaches its input unconditionally. Three assertions on a
real `MaskablePPO` measure that on the actual parameter update and on `.grad`, with a LIVE graph
handed to it on purpose.

### 2. Half-batch trunk-gradient cosine — INTERFERENCE

Every `--capacity-cosine-every` minibatches (default 50), the current minibatch is split in half,
each half's gradient on the **shared trunk** is taken, and the cosine between them is logged. Two
halves of one on-policy batch are i.i.d. draws from the same distribution, so a healthy batch has
them broadly agreeing; a cosine trending to zero or negative means the batch is increasingly
fighting itself — capacity going into trading one part of the state space against another instead
of improving on both.

* **TRUNK is `grad_balance.shared_trunk_parameters`** — the existing allow-list (`embeddings`,
  `pokemon_encoder`, `team_transformer`, `assembler`), reused rather than re-defined so this cosine
  and the `grad/*_share` family are talking about the same weights.
* **The surrogate is the PLAIN PPO objective** (clipped policy loss + `vf_coef`·MSE), not the run's
  full fold. The question is whether the two halves agree about the RL objective; folding in a dozen
  auxiliaries would make the answer a statement about the auxiliaries instead. PopArt and
  tail-weighting are skipped for the same reason — both are monotone rescalings of the same
  per-sample residual, so they move the gradient's LENGTH, not this angle.
* **Advantages are sliced from the caller's already-normalized tensor.** Re-normalizing each half
  against its own mean/std would inject a difference the batch does not have and bias the cosine down.
* **Orientation, not a threshold**: on a fresh `--debug` run at ~6k steps the cosine reads **0.99**
  with `halfbatch_grad_norm_ratio` 0.86 (measured 2026-08-23) — an untrained policy's two halves
  agree almost perfectly, which is what "healthy" looks like at the start. There is no calibrated
  alarm LEVEL and this is not one; what a saturating run is expected to show is that number
  *descending* over tens of millions of steps.
* **It corrupts nothing**: `th.autograd.grad`, never `backward()`, so `.grad` is never written and
  an in-flight grad-accumulation group survives bit-for-bit. That is a test, not a comment — the
  gate populates a real accumulated gradient and real Adam state first, because an all-`None`
  `.grad` would hide exactly the failure worth catching.

### 3. Feature velocity — do the functions still move?

One **fixed** probe batch of 256 obs rows, captured from the first rollout after launch and never
changed, is forwarded every `--capacity-velocity-every` `train()` calls (default 50) under
`no_grad`; the mean L2 displacement of `value_pooled` since the previous measurement is logged.
Read it **beside `train/grad_norm`**: weights moving while functions do not is the fingerprint of a
network burning gradient on a representation that has stopped changing. `feature_velocity_rel`
divides by the representation's own norm, because a falling raw velocity can also mean the features
merely shrank. The forward is the EAGER `type(fe).forward` with an observation-key-only dict, for
the reasons `cf_terms` gives (the compile flags patch the BOUND `fe.forward`, and a second obs
shape through the compiled entry point would add a graph for a diagnostic); the
ObservationDebugger is suppressed, since these are replayed rows.

### Where it sits in `train()`, and the flag's class

**It is NOT a fold step.** All three probes run at the END of the minibatch body — after the loss
fold, after `loss.backward()`, after the optimizer step — so nothing they do can reach `loss` or
`.grad`, and the placement is the proof. The one thing taken from inside the fold is a SNAPSHOT of
this minibatch's `value_pooled`, grabbed right after `evaluate_actions` for the same reason steps
2-4 of the fold sit where they do: the TD-aux / counterfactual folds run their own forward and
REPLACE the stash. A stale or missing snapshot is a **skip** (row-count checked), counted in
`capacity/canary_steps` rather than silently mis-pairing features with observations.

The flag is the **`training_coef` class** (`td_aux_coef` / `cf_records`): an argparse entry
defaulting to `None`, a `_resolve` line, and a recorded `ModelVersion` field (config **v101**,
`gen3_capacity_telemetry_v1`) — never in `check_compatible`. It is NOT `structural` (no module in
the policy tree, no `state_dict` key, forward bit-identical), NOT `resume_immutable` (changing it
mid-run changes nothing about training, so there is nothing to protect), and NOT `runtime` — a
runtime knob's value is unrecoverable after the fact, and a DIAGNOSTIC whose provenance is
unrecoverable is a number nobody can interpret later. It is not a `flag_registry` row either: that
registry declares **extractor** toggles, and nothing here reaches the extractor.

**OFF is byte- AND cost-identical**: no head, no optimizer, no projection matrix, no probe batch,
no extra forward or backward — one boolean per minibatch. Gated both ways (an OFF run's update
equals a no-telemetry run's; an ON run's update equals the OFF run's, exactly).

**Overhead, measured** (2026-08-23, CPU, real 2,047,958-param policy over the live 2501-dim obs, 80
minibatches per `train()` = the production ratio, 10 reps/arm INTERLEAVED, `OMP_NUM_THREADS=4`, quiet
box). Two independent clean runs:

| run | OFF median | ON median | overhead (median / minima) |
|---|---|---|---|
| 1 | 5797.2 ms | 5943.4 ms | **+2.52% / +2.42%** |
| 2 | 5800.6 ms | 5938.6 ms | **+2.38% / +2.32%** |

Inside the <3% budget, with the two arms' 10-sample ranges **disjoint** in both runs. That figure is
a CONSERVATIVE bound for production: it runs the BASELINE extractor chain (no damage op, no belief
heads), whose forward is far cheaper than the production one, and a cheaper denominator makes the
probes' share LARGER — smaller again on CUDA under `--compile-trainer`, where the canary's tiny eager
MLP is noise against a compiled forward+backward. Reproduce:
`src/agents/training/capacity_overhead_benchmark.py`.

⚠️ **A third run, taken while the 4-worker test suite shared the box, read +4.28%** — and its OFF
arm alone spread 5840→6618 ms (13%) where a clean arm spreads 1.7%. `warn_if_contended` did NOT
fire (the one-minute load average lags a job that just started). The interleaving saves the SIGN of
the effect under contention but not its size, so: read the within-arm spread before believing the
delta, and take the number on a quiet box.

🚨 **THE CANARY'S STATE IS NOT CHECKPOINTED, and this is the honest limitation.** `_capacity_state`
is in `_excluded_save_params`, so the head, its Adam state, the projection matrix and the frozen
probe batch are all re-created on every resume — and the launcher restarts every 3 hours. The
canary's loss therefore JUMPS at each restart and `canary_recovery` restarts its curve. The trade
was taken deliberately (persisting a diagnostic's optimizer into every checkpoint is worse), and the
usable reading is **compare recoveries WITHIN a restart window**: at production throughput a 3-hour
window is ~16M env steps, so a 1M-step reset interval still fires ~16 times inside one. The startup
banner says so out loud, because a silent ON here is a misreading waiting to happen.
## The `signal/` group — advantage density × outcome entropy (`gen3_signal_rate_metrics_v1`)

**How much action-attributable learning signal is PPO actually receiving?** Two always-on, flagless
scalar families answer it live. Pure observability: no gradient path, no extra battle, no env call —
a handful of numpy means per rollout.

**THE PAIR IS THE INSTRUMENT. Neither number is readable alone.**

| scalar | recorded by | is |
|---|---|---|
| `signal/adv_raw_std` | `instrumented_ppo/ppo.py::train()` | population std of the rollout's RAW GAE advantages |
| `signal/adv_raw_abs_mean` | same | `E|Â|` — the outlier-robust companion to std |
| `signal/adv_kurtosis` | same | EXCESS kurtosis (Fisher; normal = 0), **scale-free** |
| `signal/outcome_entropy` | `signal_callback.py::SignalMetricsCallback` | `p(1−p)` over a rolling 200-episode window, POOLED |
| `signal/outcome_entropy_{bots,pool,stable,target}` | same | the same, split by `MaskableAgentWrapper.OPP_CLASS_*` |
| `signal/outcome_win_rate`, `signal/outcome_n[_<kind>]` | same | the window's `p` and its depth — so a thin split is visible as thin |
| `signal/outcome_entropy_rung` | `exploiter_ladder.py::ExploiterLadderCallback._record` | `p(1−p)` of the LIVE `--exploiter-ladder` rung's gate window |

### Why the pair — the MIRROR PARADOX

Outcome entropy is **maximal (0.25) against a near-twin**, which is exactly the regime where a
single action's effect on the outcome is *smallest* and the games are closest to coin flips. So a
high `outcome_entropy` is **not** "lots of signal" — it is "lots of outcome VARIANCE", which only
becomes signal to the extent the critic localizes it onto actions. That localization is what
`adv_raw_std` / `adv_kurtosis` measure. Read the 2×2:

| outcome entropy | advantage density | reading |
|---|---|---|
| high | high | decisive moments exist and the critic finds them — healthy |
| high | **LOW** | coin-flip games nothing can be attributed to — the mirror paradox, or a stale critic |
| **LOW** | high | a lopsided matchup, but its few live decisions are sharp — a curriculum problem |
| low | low | the opponent is a wall or a pushover — no gradient to be had |

`adv_kurtosis` is the third axis and it is the one that distinguishes *shape* from *scale*: exploit
signal is sparse — a few decisive turns inside a long stretch of forced or irrelevant ones — so a
healthy rollout is HEAVY-TAILED (positive). Near 0 or negative means the advantage mass is smeared
evenly across decisions, i.e. nothing is being localized even though the std may look fine.

### ⚠️ UNITS — within a run freely, across runs only cautiously

The advantages ride the run's own returns — **PopArt-normalized when `--use-popart` is on** (it is
OFF by default and REFUSED under `--critic winprob`, where they are raw `[0,1]` probability returns),
in which case σ moves over training. `adv_raw_std` / `adv_raw_abs_mean` are therefore in *this run's current
normalized-return units*, not a fixed scale: their TREND is meaningful, their absolute level is not
portable. Across two runs with different reward composition, `gamma`/`gae_lambda`, or PopArt state,
only **`adv_kurtosis`** — scale-free by construction — compares directly.

### ⚠️ This is a TRIPWIRE, not the attributable-share measurement

`signal/` tells you *when* to go and measure; it does not do the measurement. The gold standard for
how much of an outcome was actually action-reducible remains the **offline counterfactual
decomposition** — `python -m main.prober.query falsify-scan` (the luck / unattributed /
proven-`policy_reducible` crater bracket) and `cf_audit.py`. Those re-roll the real dice and sweep
alternative actions; `signal/` only reports the critic's own opinion of its rollout.

### Where each half is measured, and why there

**Advantage density is read ONCE per `train()`, off `self.rollout_buffer.advantages`, BEFORE the
epoch loop** — because that is the last point at which the raw GAE advantages still exist. The
minibatch loop applies `normalize_advantage`, which forces std→1 per minibatch and so *destroys the
quantity being measured*. Composes with `--grad-accum-steps` for free (the read is per-rollout, not
per-optimizer-step) and is untouched by `--compile-trainer` (it is numpy over the buffer, outside
any traced graph). Degenerate rollouts are NaN-safe: an empty buffer publishes nothing, a constant
rollout reports a real std/abs-mean with `adv_kurtosis` **NaN** — TensorBoard drops NaN, so the
curve gaps rather than reporting a fabricated 0.0 that would read as "evenly smeared".

**Outcome entropy rides the info dicts the loop already sees.** `MaskableAgentWrapper.step` publishes
`info["win_outcome"]` (which the win-prob head already used) and, new here, `info["opponent_class"]`
— purely additive keys, so nothing downstream changes. `SignalMetricsCallback` pushes each `done`
into rolling per-kind deques in `_on_step` and records in `_on_rollout_end`.

**`--async-rollout` IS covered.** The stock collector publishes `infos`/`dones` in the callback
locals; `collect_rollouts_async` publishes `wave_infos`/`wave_dones` (a wave = a macro-step over
whichever envs came ready). The callback reads whichever pair is present. Unlike
`WinProbLabelCallback` — which needs the `(step, env)` BUFFER ROW and therefore cannot use the wave
batching at all — outcome entropy is a per-episode aggregate with no row alignment, so the wave form
carries everything it needs. The advantage half is transport-agnostic (both paths call
`compute_returns_and_advantage` into the same buffer). Works under `--debug` — the callback is in
`build_callbacks`' unconditional base list.

**Which opponent splits are REAL.** The wrapper's four `OPP_CLASS_*` values are the whole of what
survives the env-worker boundary — only the integer crosses the pipe. So `bots` / `pool` / `stable`
/ `target` are real, and finer identity is **not**: which *heuristic* bot (the class collapses
random and every heuristic into one), and which *pool snapshot* (its provenance — the step it was
frozen at — is held by `SnapshotPool` in the parent and never reaches the wrapper). `_rung` is the
one finer split that exists, and it exists only because the ladder callback keeps its own per-rung
window in the parent process.

State is process-local and NOT checkpointed — a launcher restart re-warms the windows in a few
hundred episodes, the same contract the noise-scale EMAs take.

Tests: `signal_metrics_test.py` — hand-computed moments, an independently-written closed form, the
sparse-vs-spread kurtosis discrimination at MATCHED std, scale-freeness, every degenerate input,
the rolling-window eviction, the kind routing, both rollout paths' locals, the
`OPP_CLASS_SUFFIX` ↔ wrapper-constant pin, and the byte-identity of `train()` with the read
monkeypatched out.
## The SCAFFOLDING GAUGE — `train/scaffolding_gauge` + `python -m main.scaffolding_gauge`

🚨 **THIS GAUGE IS A `--critic shaped` INSTRUMENT AND IS DEGENERATE ON THE PRODUCTION RUN.** It
measures the divergence between TWO readouts; under `--critic winprob` there is one — the win-prob
head IS the critic, so the gauge compares a head with itself and its rank correlation is 1 by
construction. Read it on a shaped run, or on an archived one; do not read it as a scaffolding
measurement of a terminal-only run, which has no scaffolding to measure.

**How far apart are the two value readouts, and is the gap closing?** Under `shaped`, the critic
estimates the **shaped** return (PopArt units, `gamma`-discounted) while the win-prob head estimates
the **game** (outcome units, no discount distortion, no PopArt drift). Neither is a repair of the
other — the two-head structure is the automatic consequence of choosing shaped rewards. What their DIVERGENCE
measures is the reward scaffolding still doing work, and its trajectory is the registered signal
for when shaping coefficients can begin annealing toward the pure game.

Pure math in **`agents/training/scaffolding.py`** (numpy only, no torch, no filesystem), shared by
the live scalar and the offline CLI so the two can never drift apart.

| scalar | recorded by | is |
|---|---|---|
| `train/scaffolding_gauge` | `instrumented_ppo/ppo.py::train()` | `(1 − Spearman ρ(V, P(win))) / 2` over epoch 0's paired reads. 0 = identical ordering, 0.5 = independent, 1 = inverted |
| `train/scaffolding_rho` | same | the raw ρ, so nothing is hidden by the transform |
| `train/scaffolding_n` | same | rows the ρ was computed from |

**ALWAYS ON when the win-prob head exists** (`--win-prob-mode != none`), flagless, gated on the
head's EXISTENCE and not on `win_prob_coef` — a `read_only` head at coefficient 0 still says
something worth curving. A run with no head publishes **no key at all**, so the curve is absent
rather than flat at zero.

**Where it is read, and why there.** Inside the minibatch loop, right after the win-prob block and
BEFORE the cf-twin fold clobbers the extractor stashes: that is the one place both readouts exist
for the SAME states from the SAME forward (`evaluate_actions` produced `values` and stashed
`last_win_prob_logits`). **Epoch 0 only** — by epoch 3 the policy that produced a pair is not the
policy the pair would be attributed to. The logit is NOT sigmoided: the sigmoid is monotone, so ρ
is identical and float32 ranks never saturate.

### ⚠️ UNITS — the rank form is the ONLY one that is live-legal

Under `--critic shaped` `V` is a PopArt-normalized SHAPED return and there is no general unit
conversion to a probability. (Under `--critic winprob` the question dissolves: `V` IS the
probability, which is the same fact that makes this gauge degenerate there.) The
live scalar is therefore **rank-based and claims ORDERING only** — nothing about magnitude or
calibration. It also goes **AMBIGUOUS at the PBRS constancy endpoint**: under a good frozen
potential, all evaluative content migrates into the reward stream and `V_shaped` is driven toward a
CONSTANT (ledger db9bb5c), at which point ρ degenerates into noise and a falling curve cannot be
told from V running out of variance to rank with. Read it beside the value-scale meters, and beside
the offline constancy row.

The magnitude question is answered OFFLINE, by `python -m main.scaffolding_gauge <run>`, which
walks the run's own `eval_traces` (model-FREE — recorded `values` + `win_probs`, so it works on a
run whose checkpoints no longer load) and ships **both** gauges per checkpoint step:

* **rank gauge** — the same statistic as the live scalar, unit-free.
* **calibrated-affine gauge** — fit `q = clip(a·V+b, 0, 1)` against the REALIZED per-battle
  outcomes on that slice, then report `rms|q − P(win)|` in probability units. The map is a
  **per-checkpoint FIT, not a conversion**, and it does not transport. Part of every residual is
  the affine family being a worse outcome predictor rather than the heads disagreeing, and that
  part ships as `readout_penalty` = Brier(readout) − Brier(head): a large `rms` with a large
  penalty is a readout finding, not a divergence finding.
* **the constancy sanity row** — `v_std` / `v_iqr` / `dispersion` plus the within-vs-between-battle
  split, i.e. the db9bb5c prediction as a one-line check a frozen-φ arm's battery can quote
  (`--constancy` prints only this block). Low `v_std` with `within_frac ≈ 0` is the look-alike
  FAILURE: V has become a per-battle matchup lookup, not a flattened potential.

Every offline CI is a **CLUSTER bootstrap over BATTLES**, because outcome labels are per-battle and
broadcast to every state; an i.i.d. interval over states would be fabricated tightness of roughly
`sqrt(states-per-battle)`. And the step-to-step curve is **not** a controlled comparison — each
point carries whatever the eval quota sampled at that checkpoint, so a verdict needs arm-vs-control
at matched step.

```bash
python -m main.scaffolding_gauge models/<run>               # table + <run>/scaffolding_gauge.json
python -m main.scaffolding_gauge models/<run> --plot        # + a 3-panel PNG
python -m main.scaffolding_gauge models/<run> --constancy   # only the db9bb5c row
python -m main.scaffolding_gauge models/<run> --reliability --reliability-reweight   # section (4)
```

### The RELIABILITY block (`--reliability`) — the head against the TRUTH, not against V

Both gauges above compare the two READOUTS to each other. `--reliability` adds the third question,
which is the one a calibration gate needs: **how far is the win-prob head from the realized
outcome?** Opt-in, so the default JSON and render are byte-stable (pinned by a test). It emits per
checkpoint, stratified `all` / `bot` / `pool` / per-opponent, with cluster-bootstrap CIs over
battles: `brier`, the Brier **`skill`** score against the slice's own base rate, `ece` / `mce`, the
Murphy **`reliability` − `resolution` + `uncertainty`** split with its binning residual reported,
and the per-bin reliability curve. Math: `scaffolding.reliability_table`.

**Read `resolution`, not `reliability`, as the meter.** A base-rate forecaster scores a perfect 0
reliability and a useless 0 resolution — `designs/learning/win_prob_decomposition.md` axis 2 is the
statement that the blur, not the level, is this project's critic disease, and the 2026-09-06
baseline measured exactly that shape on `ai_v9_59_R2ACTION_0827` (reliability 0.0013–0.0020 against
a resolution of 0.062 / 0.045 out of an available 0.182 / 0.165).

**Read `bot` and `pool` separately; a pooled row describes neither.** Axis 3's ecology split
measured the head's mean bias FLIPPING SIGN between the two populations, so the split is the
default rendering and `opponent_class` is where the `sentinel_*` ⇒ pool rule is declared.

🚨 **`--reliability-reweight` is not optional on this tree's traces.** The eval recorder's quota is
loss-enriched — measured on `ai_v9_59_R2ACTION_0827`, the captured outcome rate is **0.46** against
the same cycles' recorded **0.901 vs bots / 0.702 vs pool** — so an unweighted table scores the head
against a population it was never deployed against. The flag importance-weights each opponent's rows
back to the win/loss mix the cycle itself recorded (weights constant within a
battle, so the clustering survives), and reports Kish `ess` beside `n` so the cost is visible. It
**REFUSES** when the true rates cannot be resolved rather than falling back — an unweighted table
looks identical and answers a different question. The size of the correction is the finding: raw, the
same traces read ECE 0.237 / 0.281 and skill +0.071 / **−0.080**; reweighted, ECE 0.025 / 0.035 and
skill **+0.336 / +0.265**. Reading raw-first inverts the verdict.

**TWO SOURCES for the true rates, in preference order** (`gen3_trace_selection_manifest_v1`). Each
cycle's own **`eval_manifest.json` selection block** wins where it exists — the recorder writes the
per-opponent played/won counts into the same directory as the traces, so there is no cross-file
join and, in particular, no positional sentinel inference. Everything it does not cover falls back
to **`eval_results.jsonl`** with the existing behaviour, unchanged; a run with neither still
REFUSES. On a legacy tree the manifest half is empty, so the numbers this tool prints there do not
move (pinned as a byte-identity test on a synthetic tree). `--reliability` additionally emits a
`trace_selection` block and prints, per step, the capture rates and **which source** the
reweighting used — a step that records no selection is labelled **SELECTION UNKNOWN**, never read
as uniform.

Record: `designs/research_state/measurements/winprob_critic_baseline_2026-09-06/`. The design that
consumes it as a gate: `designs/ai_v12/design_winprob_only_critic.md`.

A run whose `win_probs` column is all NaN (`--win-prob-mode none`) **REFUSES** with that diagnosis
rather than curving zeros — "the two readouts agree perfectly" and "there is no second readout"
must not render the same.

Tests: `scaffolding_test.py` — the three known regimes (monotone ⇒ exactly 0, inverted ⇒ exactly 1,
independent ⇒ ~0.5), affine-rescale invariance, the constant-axis NaN, the affine gauge's
`readout_penalty` convicting the FAMILY on a constructed step function while a linear control
collapses it, the cluster bootstrap beating an i.i.d. one by ~`sqrt(50)`, and the live scalar's
byte-identity + NaN-safety + epoch-0-only read. For the reliability block: a calibrated forecaster
reading REL→0 with RES>0, a base-rate one reading exactly 0 skill (the meter's whole point), the
Murphy identity holding to its own reported residual, `p == 1.0` landing in the last bin rather than
a phantom one, and uniform weights reproducing the unweighted table bit-for-bit.
`main/scaffolding_gauge_test.py` folds a constructed three-regime trace tree end to end, plus a
two-opponent-CLASS fixture where the pooled row INVERTS the bot verdict (the ecology confound as a
test) and a loss-enriched-quota fixture where the reweighting moves the base rate onto the cycle's
and takes reliability from 0.32 to 0.

## TensorBoard export census — the counts, the group table and the ERA-RELEVANCE classification

> The census's currency table, its `gen3_tb_relevance_v1` headline and the 28-tag WIN-PROB dashboard
> stay in `src/agents/training/CLAUDE.md` (pinned there by
> `tb_relevance_test.py::test_the_census_table_carries_an_era_column`). What follows is the detail
> underneath them.

**Counts, measured 2026-09-06** — 153 static `logger.record(` sites across `src/agents/training`,
`src/main/train`, `src/main/eval_worker.py` and `src/main/elo.py`; 185 distinct tags observed
across three CPU smokes (a plain run, a `--win-prob-mode read_only` run, and a `--use-popart` run).
The two do not match and should not: one site can emit a whole dict (`f"reward/{k}"`), and many
sites are flag-gated off in any one run. **Recount before quoting** — `tmp_census.py`'s recipe is
`grep -rn "logger.record(" src/agents/training src/main/train src/main/eval_worker.py` for the
sites and an `EventAccumulator` walk of a run's `tb/` for the tags.

| group | sites | tags seen | cadence | currency | **era** (`--critic winprob`) | computed in |
|---|---:|---:|---|---|---|---|
| `reward/` | 1 | 46 | **per rollout** | RAW REWARD | 14 emitted, of which **6 NOISE (gated)** + 5 REDUNDANT — a 1-term composition has nothing to apportion | `reward_term_callback` ← `reward_term_stats` |
| `train/` | 53 | 23 | per rollout (`train()`) | MIXED — see per-tag below | LIVE, but `return_*` / `value_loss` / `explained_variance` **change currency to P(win)**; `scaffolding_*` **NOISE (gated)** | `instrumented_ppo/ppo.py`, `grad_balance`, `run_io` |
| `win_prob/` | 5 | 42 (+10 under `--critic winprob`) | per rollout | PROBABILITY | **the era's core group.** The `critic_*` ten are LIVE and primary; the 19 `contested`/`material` tags are **NOISE** on a spread-free margin and are ALL gated (see below) | `ppo.py` ← `value_terms`, `calibration`, `scaffolding.reliability_table` |
| `eval/` | 35 | — | **per EVAL CYCLE** | win rate / ELO / reward | LIVE — except **13 `mean_reward_*` REDUNDANT** (byte-identical to their `win_rate_*` twin under the indicator) | `eval_callback`, `selfplay_callback` |
| `eval_final/` | 2 | 10 | once, at run end | win rate | LIVE | `main/train/final_eval.py` |
| `signal/` | 4 | 12 | per rollout | NORMALIZED (adv) / probability (outcome) / rate (draw) | LIVE — `draw_rate` is **promoted to a PRIMARY endpoint** (the G7 kill condition) | `signal_metrics`, `signal_callback` |
| `popart/` | 5 | 5 | per rollout | raw (μ,σ) + unitless (norm) | CONDITIONAL — **structurally silent**: PopArt is REFUSED under `winprob` | `ppo.py` |
| `grad/` | (dynamic) | 16 | per rollout | unitless shares | LIVE — `win_prob_*` **NOISE (gated)**: it IS the value term, and counting it twice deflated every share | `grad_balance` |
| `rank/` | 6 | 18 | per rollout | unitless | CONDITIONAL (needs the rank probe); `tripwire_no_reading` correctly reports its own blindness | `rank_tripwire`, `rank_metrics` |
| `belief/` | 1 | 8 | per rollout | accuracy / CE | LIVE (unchanged by the critic mode) | `belief_bank` |
| `distill/` | 7 | — | per rollout | KL / MSE / rate | CONDITIONAL — silent, no teacher | `distill_terms`, `distill_anchor*`, `distill_stop_callback` |
| `cf/` | 5 | — | per rollout | probability + counts | CONDITIONAL — silent, no `--cf-records` | `cf_terms`, `cf_label_buffer` |
| `teacher/` · `opd/` | 11 | — | per cycle / rollout | CE / KL | CONDITIONAL — silent | `teacher/callback`, `ppo.py` |
| `team_pfsp/` · `hparams/` · `capacity/` · `defent/` · `baitent/` · `value_dist/` · `td_aux/` · `q_winprob/` | 20 | — | per rollout | see each section | CONDITIONAL — all silent; `value_dist/` is **REFUSED** by the mode, the rest are flag-off | their own callbacks |

### ERA RELEVANCE — a tag whose SOURCE is absent is not emitted (`gen3_tb_relevance_v1`)

**Measured 2026-09-06 over `models/ai_v12_01_winprob_critic`'s first hours: 216 tags emitted.** The
`--critic winprob` era changes no tag NAME but removes the SOURCE behind several, and the recorders
kept publishing — as flat constants and byte-identical duplicates that a reader cannot tell from a
measurement. Every classification below is from that arm's own tfevents, not from reading code.

| class | on the arm | means |
|---|---:|---|
| **LIVE** | 166 | meaningful as published |
| **NOISE** | **31** | emitted, content-free *here* — a constant, a tautology, or a copy created by the era. All 31 are now GATED |
| **REDUNDANT** | 19 | an exact duplicate of another tag by identical formula or affine invariance |
| **CONDITIONAL** | (48 more observed on other configs, + ~14 whole families) | correctly SILENT — their flag is off, or the mode refuses them |
| **DEAD** | **0** | nothing in the recorder set is unreachable; the v75 latent-belief and v88 `V_pub` purges left no orphans |

**The NOISE tags are GATED** — the gate is on the SOURCE, never on the value, so a shaped run is
byte-identical (verified: 172 tags, empty before/after diff):

| gated | why it was content-free | gate lives in |
|---|---|---|
| `train/scaffolding_{gauge,rho,n}` | `V = sigmoid(win_prob_logit)`, so ρ ≡ 1.0 and the gauge ≡ 5.5e-13. A rank gauge between a quantity and **itself** | `scaffolding._same_ordering` — identical rank vectors ⇒ no keys |
| `grad/win_prob_{share,norm_shared,policy_cosine}` | the critic loss IS the win-prob BCE — the SAME tensor object, passed as `value_term` *and* as `aux_terms["win_prob"]` | `grad_balance_metrics` skips an aux term that `is value_term` |
| `win_prob/{brier,acc}_contested` · `contested_{frac,label_mean}` · `brier_material` · `skill_vs_material` | **NO LONGER NOISE — the SOURCE was repaired 2026-09-06.** `win_margin` was identically 0.0 on the whole win-prob era (a MATERIAL-potential by-product whose compute the composition gated away), so `contested_frac` ≡ 1.0, every `*_contested` ≡ its pooled twin, `brier_material` ≡ 0.25 and `skill_vs_material` collapsed to `1 − 4·brier`. `gen3_obs_margin_unconditional_v1` computes Φ_mat unconditionally, so the margin now spreads and the six tags are LIVE — see *WHAT THE SHORT CIRCUIT MUST NEVER SKIP* below. ⚠️ A run started before that commit carries the degenerate series | `value_terms._win_prob_loss` treats a **spread-free** margin as absent — kept, as the consumer-side guard against any future flat margin |
| `reward/{bias_refund,class_refund}_{mean,abs_mean,abs_share}` | the refund is the BIAS class's accumulate-and-refund MECHANISM; with no bias term it is structurally 0.0 | `reward_term_stats._has_bias` |

Plus `eval/{win_rate,mean_reward,mean_ep_len}_vs_pool` and `eval/sentinel_monotonicity`, which fell
back to a confident **0.0** / a perfect **1.0** when no sentinel had been measured yet — the first
two cycles of the live arm published exactly that beside a `pool_snapshot_count` of 1.
`selfplay_callback` now leaves a GAP until a sentinel actually reports (the in-process values keep
their 0.0 default, because `_check_promotion` and the ELO fit consume them; only the EXPORT is gated).

🚨 **The `grad/` one was not merely cosmetic — it was a GIGO defect.** The duplicated norm sat in the
shared denominator too, so on that arm **every `grad/*_share` was deflated by the critic's own pull**:
the published norms are policy 0.4555, value 0.1380, win_prob 0.1380 (the same number), hp_type
0.0723, move_latent 0.0039, and the published `policy_share` 0.5639 is `0.4555 / 0.8077` — a
denominator carrying 0.1380 twice. Over a de-duplicated denominator the true shares are **policy
0.680 · value 0.206 · aux 0.114**. The shares summed to 1.0 throughout, which is why nothing caught it.

**The last 13 are GATED too** — `win_prob/contested_{ece,mce,rel_gap_b0…b9,rel_n}`, which come
through `calibration.contested_mask`. It used to return `None` only when the margin **key** was
missing, not when the margin had no **spread**, so a flat column made `|const| < tau` all-ones and
the family emitted as byte-identical copies of its pooled twins. It now returns `None` on a
spread-free (or empty) margin, MIRRORING `value_terms._win_prob_loss`'s own `max − min > 0`
predicate — the two consumers stratify on the same column, so a run in which one publishes the
split and the other does not would be worse than either answer alone (a source-read test pins that
the two predicates cannot drift apart). `_CalibrationAccumulator.metrics()` already returned `{}`
for an unobserved accumulator, so the whole family now disappears rather than duplicating.

**REDUNDANT (19) — kept, deliberately, and named so nobody measures them twice:**

* **`eval/mean_reward_*` ≡ `eval/win_rate_*`, all 13 of them.** At `--victory-value 1.0
  --draw-penalty 0 --terminal-indicator` the episode reward IS the win indicator, so the two families
  are byte-identical on every opponent (verified to the last bit on the live arm). They are NOT
  redundant on a `shaped` run and the TUI reads both, so neither is dropped — but quoting both as
  "two agreeing signals" would be quoting one number twice.
* `reward/class_terminal_*` ≡ `reward/win_loss_*` and `reward/total_{mean,abs_mean}` ≡ the same —
  a class rollup over ONE term, and a total over one term. `*_abs_share` is then a constant 1.0: a
  share partition of a single term.
* `train/nonbot_fraction` ≡ `train/selfplay_fraction` whenever `--stable-opponents` is unused
  (`nonbot = pool + stable`). Kept because it separates the moment a stable opponent is added, and a
  curve that appears mid-run is worse than a duplicate.

**Three tags are the same quantity in three WINDOWS and are not duplicates**: `train/return_mean`
(this rollout's returns), `rollout/ep_rew_mean` (SB3's 100-episode deque) and
`signal/outcome_win_rate` (a 200-episode window). Under the indicator all three read a win rate;
disagreement between them is a window effect, never a defect.


### The five groups the diagnostic contract names

#### `reward/` — WHAT THE REWARD IS MADE OF (`gen3_reward_term_export_v1`, 2026-09-06)

Per-rollout, RAW REWARD units, one triple per ACTIVE term of this run's composition plus four class
rollups and four totals. Full rationale — including why the share is `|·|`-weighted and why the
residual is a GIGO guard rather than a rounding term — is in
`agents/training/reward_term_stats.py`'s module docstring.

| tag | is |
|---|---|
| `reward/<term>_mean` | Σterm ÷ decisions. **A PBRS term should read ≈0** over an episode-complete window — that is the telescoping, visible |
| `reward/<term>_abs_share` | `Σ\|term\| / Σ_terms Σ\|term\|` — this term's share of the reward stream's MOVEMENT. **The shares partition to 1** |
| `reward/<term>_abs_mean` | `Σ\|term\|` ÷ decisions, the un-normalized magnitude |
| `reward/class_{terminal,pbrs,bias,refund}_{mean,abs_mean,abs_share}` | the same, rolled up by `RewardClass` |
| `reward/total_{mean,abs_mean}` · `reward/n_decisions` | the stream itself and the window size |
| `reward/untracked_abs_mean` | **THE GIGO GUARD** — `mean\|bd.total − Σ tracked\|`. Reads exactly 0.0 when the startup composition census and the folds agree; anything else means they do not |

The tracked set is derived from `reward_class_composition(config)` — the SAME `_pbrs_term_active` /
`_bias_term_active` predicates the folds are gated on — so the exported terms cannot disagree with
the startup line. Under the `shaped`-critic default composition that is 10 terms (1 terminal +
7 PBRS + 1 bias + the refund mechanism) → 46 tags; under `--no-all-shaping-pbrs` it is 28 terms →
~100 tags. ⚠️ **The production run is neither**: `--no-hand-shaping --terminal-indicator` is ONE
terminal term, so it emits the smallest tag set of the three. Bounded
by the REGISTRY, never per-team.

**Transport: an `env_method` PULL, not an info-dict thread.** The reward is computed in the env
WORKER, and under `--async-rollout` the callback's step locals arrive wave-batched with no way to
recover which buffer row a step landed on — the same reason `TeamWinRateCallback` uses this seam.
`AsyncSubprocVecEnv.env_method` is drain-safe, so one seam covers both collectors, and
`RewardTermAccumulator.drain()` zeroes the window so a double pull cannot double-count. ALWAYS ON,
no flag: the accumulator folds only the ACTIVE terms — 9 of 35 under the `shaped` default, and
**1 of 35 under the production composition** (`--no-hand-shaping --terminal-indicator`: 1 TERMINAL,
0 PBRS, 0 BIAS). Read `metadata.json`'s `reward_composition`, never a remembered count.

#### `win_prob/` — the head's PREDICTION, its CALIBRATION, and the paired episode-start read

Per rollout, PROBABILITY units, epoch 0 only (by epoch 3 the policy that produced a pair is not the
policy it is attributed to).

| tag | is | added |
|---|---|---|
| `loss` · `acc` · `brier` · `pred_mean` · `label_mean` · `coverage` | the pre-existing fit meters | |
| `brier_contested` · `acc_contested` · `contested_frac` · `contested_label_mean` · `brier_material` · `skill_vs_material` | the information-value half, restricted to material-EVEN decisions | |
| **`ece`** | 10-bin count-weighted Expected Calibration Error. **Brier is a PROPER score and decomposes as reliability − resolution + uncertainty, so it can stay flat while calibration drifts**; this isolates the reliability term | ✅ 2026-09-06 |
| **`mce`** | the WORST readable bin's gap — ECE is an average, so a head badly wrong only on the confident tail holds a small ECE | ✅ |
| **`rel_gap_b0` … `rel_gap_b9`** | the reliability HISTOGRAM, one scalar per bin, so the SHAPE of the miscalibration is readable. **A bin under 100 samples publishes NaN**, which TensorBoard renders as a hole — a 3-sample bin's "error" is sampling noise | ✅ |
| **`rel_n`** | rows the diagram was built from | ✅ |
| **`contested_*`** | every one of the above, restricted to `\|win_margin\| < 0.25` — a blowout's P(win) is trivially recoverable from material, so the pooled ECE is flattered by exactly the states nobody needs the head for | ✅ |
| **`start_pred_mean` · `start_realized_mean` · `start_gap` · `start_n`** | **THE PAIRED EPISODE-START READ** — what the head says at the LEAST-informed state against what those very episodes went on to do | ✅ |
| **`start_*_{bots,pool,stable,target}`** | the same, split by opponent class — **`start_*_pool` IS "the self-play win probability at episode start vs the realized self-play win rate"** | ✅ |

🚨 **THE EPISODE-START READ IS PAIRED, AND THAT IS THE WHOLE POINT.** `win_target` is back-filled by
`WinProbLabelCallback` from the episode's own outcome to every step of that episode, so at an
episode-START row it IS the realized outcome of the game that starts there. Prediction and
realization therefore come from ONE set of episodes and `start_gap` is a paired difference — not
the difference of two independently-windowed averages, which would carry the two windows'
disagreement as well as the head's error. **POSITIVE = optimistic at the opening board.** Cost: one
EAGER forward over the episode-start rows (capped at `_WINPROB_START_MAX_ROWS` = 1024, a
deterministic prefix, never sampled) once per `train()` — eager `type(fe).forward` for the
capacity-probe's reason (both compile flags patch the BOUND attribute, and a second obs shape
through the compiled entry point would add a dynamo graph for a diagnostic).

⚠️ **The per-class split is OPPORTUNISTIC**: it needs the `opp_class` obs key, which the env emits
only alongside the opponent-intent labels (`--opp-intent-coef > 0`). Without it the POOLED read
still ships, and `signal/outcome_win_rate_<kind>` carries the realized per-class rate
unconditionally — so the self-play realized rate is never missing, only its paired partner is.

**`win_prob/vs_critic_divergence` does not exist under that name; the scalar is
`train/scaffolding_gauge`** — `(1 − Spearman ρ(V, P(win)))/2` over epoch 0's paired reads, with
`train/scaffolding_rho` and `train/scaffolding_n` beside it. It is NOT renamed: the name is
non-obvious but not misleading, it is the subject of a documented section and an offline CLI
(`python -m main.scaffolding_gauge`), and dashboards read it. See *The SCAFFOLDING GAUGE* below.

#### `signal/` — advantage density and the REALIZED per-class win rate

| tag | is | currency |
|---|---|---|
| **`adv_raw_mean`** | mean RAW GAE advantage — the NO-HARM watch. A mean far from 0 relative to `adv_raw_std` is a systematically MIS-CENTRED critic, and `normalize_advantage` erases it per minibatch so nothing else can report it. **Read as the ratio to `adv_raw_std`** | NORMALIZED ✅ 2026-09-06 |
| `adv_raw_std` · `adv_raw_abs_mean` · `adv_kurtosis` | the density and its shape | NORMALIZED (kurtosis scale-free) |
| `outcome_entropy[_<kind>]` · `outcome_n[_<kind>]` | `p(1−p)` over a rolling 200-episode window | probability |
| **`outcome_win_rate_<kind>`** | **the REALIZED per-class win rate.** `p(1−p)` is SYMMETRIC about 0.5, so `outcome_entropy_pool = 0.16` means p = 0.2 **or** 0.8 and nothing in the export said which — for two generations only the entropy shipped per kind. Free: the same deque, one more mean | probability ✅ 2026-09-06 |

#### `popart/` — and whether the currency conversion is CURRENT

| tag | is |
|---|---|
| `mu` · `sigma` | what the normalizer BELIEVES the return mean and scale are. **Should TRACK `train/return_mean` / `train/return_std`** |
| `value_weight_norm` | the POP rescale staying bounded |
| **`norm_return_mean`** | `mean((returns − μ)/σ)` — **≈0 when the conversion is current.** Far from 0 is an offset the value head has to carry itself | ✅ 2026-09-06 |
| **`norm_return_std`** | `std((returns − μ)/σ)` — **≈1 when the conversion is current.** Drifting from 1 is PopArt LAGGING the return scale, and the value gradient is then mis-scaled against the shared trunk by exactly that factor | ✅ 2026-09-06 |

μ and σ alone say what the normalizer believes; these two apply the conversion to THIS rollout's own
returns and say whether the belief is current. Free — a mean and a std over an array
`value_scale_metrics` has already read. Emitted only under `--use-popart`.

#### `train/` — value-function health, and the EXPLAINED-VARIANCE currency question

| tag | is | currency |
|---|---|---|
| `explained_variance` | `1 − Var(returns − values)/Var(returns)`, over the whole rollout pooled | **see the note below** |
| `return_mean` · `return_std` · `return_abs_max` | the value TARGETS' scale. **`return_std` IS the value-target std** | RAW SHAPED RETURN |
| `value_pred_std` | the critic's own output spread | RAW SHAPED RETURN |
| `value_loss` | the fitted loss | NORMALIZED under PopArt, raw otherwise |
| `policy_gradient_loss` · `entropy_loss` · `loss` · `approx_kl` · `clip_fraction` · `clip_range[_vf]` · `grad_norm` · `n_updates` | the stock PPO step | unitless / loss units |
| `scaffolding_gauge` · `scaffolding_rho` · `scaffolding_n` | the shaped critic vs the win-prob head — **DEGENERATE under `--critic winprob`**, where the two readouts are one head | unitless (rank) |
| `noise_scale[_ratio][_<term>]` · `dose_rate` · `effective_batch` · `grad_accum_steps` · `train_ms` | the step-size controllers | see their sections |

🚨 **`train/explained_variance` IS THE SAME NUMBER IN BOTH CURRENCIES, and a second "normalized"
tag would be a duplicate curve rather than a second measurement.** EV is
`1 − Var(y − ŷ)/Var(y)`, and PopArt applies the SAME affine map `(·−μ)/σ` to both `y` and `ŷ`
(`policy._critic_value` de-normalizes, so `rollout_buffer.values` and `returns` are both RAW). A
shared affine map cancels: `Var(a(y−ŷ))/Var(a·y)` is unchanged for any `a ≠ 0`, and the `−μ` cancels
inside both variances. **So SB3's default is computed on the RAW shaped-return arrays, and the
PopArt-normalized EV is numerically identical to it.** That is worth stating rather than shipping,
because "which currency is this EV in?" is a question a reader will otherwise ask on every run.

⚠️ **`train/value_target_std` does not exist and is not needed: it is `train/return_std`.** The
value targets ARE `rollout_buffer.returns`. Not renamed — `return_std` is accurate, sits beside its
own `return_mean`/`return_abs_max` family, and dashboards read it.
Likewise **`train/advantage_mean` / `train/advantage_std` are `signal/adv_raw_mean` /
`signal/adv_raw_std`** — the `signal/` group is where the raw pre-normalization advantages are read
(the ONE place they still exist), and duplicating them under `train/` would give two names to one
number.

### What is NOT exported, deliberately

* **A per-team reward or win-rate SERIES.** Owner rule (design_flywheel_tick_tock.md §6b): per-team
  curves are noisy spam. The per-team win-rate table rides `metadata.json`'s `team_win_rates` block
  instead, and `TeamWinRateCallback` has a test that FAILS if anything is emitted to TensorBoard.
* **A reliability histogram bin's COUNT.** `rel_gap_b<k>` publishes NaN below 100 samples, so an
  under-populated bin renders as a hole; adding 10 count tags to say the same thing would double the
  group for no reading.
* **The opponent's identity beyond its CLASS.** Only the `OPP_CLASS_*` integer crosses the env-worker
  pipe, so `_bots` / `_pool` / `_stable` / `_target` are real and finer identity (which heuristic,
  which pool snapshot) is not. `signal/outcome_entropy_rung` is the one finer split, and it exists
  only because `ExploiterLadderCallback` keeps its own per-rung window in the parent process.


## Gradient-balance + value-scale diagnostics (`grad_balance.py`)

The dual-head extractor shares ONE transformer trunk between the policy and value heads
(`src/agents/model/CLAUDE.md`); both losses' gradients compete there. When the value loss
dominates (large / unclipped, big-return scale) it **swamps the trunk** and the policy barely
updates — visible before only *indirectly* as suppressed `train/approx_kl` + `train/clip_fraction`
while `train/explained_variance` races ahead. `InstrumentedMaskablePPO.train()` now measures it
**directly** via the pure helpers in `grad_balance.py` (no SB3 / logging coupling → unit-tested in
`grad_balance_test.py`), recorded once per `train()` call through the standard logger → TensorBoard
**and** the launcher TUI (the new scalars ride the generic `MetricsExporterCallback` →
`ipc.send_metrics` path with zero extra wiring; ordering/labels live in `launcher/format.py`).

- **Gradient balance — every head's *pull* on the shared trunk, on ONE common denominator.** Sampled
  on the first minibatch (graph alive) by **read-only** `autograd.grad` probes (`retain_graph=True`, so
  the real `loss.backward()` is unaffected) against the shared-trunk params. "Shared" =
  `SHARED_TRUNK_PHASES = {embeddings, pokemon_encoder, team_transformer, assembler}` (the allow-list
  is the single source of truth), which **excludes** `cls_pool` (head-private `our_cls`/`their_cls`/
  `value_cls` queries) and both projection heads — only *truly contested* params count. With the
  belief / move / latent / move-latent / win-prob / value-dist auxiliaries there are now **many**
  competitors, not just value-vs-policy, so **every `grad/*_share` is on the SAME total**
  `T = ‖g_pi‖ + ‖g_vf‖ + Σ‖g_aux‖` — the shares are mutually comparable, **sum to ~1**, and any one
  term crowding out the rest is read off directly. (L1-of-norms — an upper-bound proxy, not a variance
  decomposition, since `‖a+b‖ ≠ ‖a‖+‖b‖` — but the same convention for every term.)
  - `grad/policy_share` + `grad/value_share` — the two RL heads' slices of the **whole** pie (ALWAYS
    present). Each is weighted by the live `ent_coef` / `vf_coef`, so `value_share` is a `vf_coef`
    tuning read — but it now *moves with the aux count* (it is value's slice of the full pie), so prefer
    the aux-independent `value_policy_logratio` below for the pure value/policy balance.
  - `grad/aux_share` (only when ≥1 aux is on) — Σ of all the aux shares, the **total non-RL draw** on
    the trunk: one curve for "are the scaffolds collectively crowding out policy/value".
  - `grad/value_policy_logratio` = `log10(‖g_value‖/‖g_policy‖)` — the **AUX-INDEPENDENT** value-vs-policy
    imbalance (a pure ratio of the two RL norms, unchanged by how many auxiliaries are on), *linear &
    non-saturating* (0 = balanced, >0 = value dominates, <0 = policy dominates, e.g. ≈+1.8 at a 66:1
    swamp). The legible gauge for **watching a PopArt / `vf_coef` fix land** — it moves linearly toward 0
    where `value_share` would crawl. `vf_coef` is **fixed per run** (recorded in `model_config.json`,
    FATAL to change on resume — it rescales this very gradient; tune on a fresh run; see
    `src/agents/model/CLAUDE.md` → resume-immutable training hparams).
  - `grad/policy_value_cosine` — scale-invariant (hence `vf_coef`-independent) structural-conflict
    signal: <0 ⟹ the two RL heads pull the trunk in opposing directions.
  - `grad/policy_norm_shared` / `grad/value_norm_shared` — the weighted norms, for absolute context.
  - **Per-aux breakout** (each present only when ITS head is active this minibatch — passed as the
    `aux_terms` dict): `grad/{species_belief, move_belief, move_latent, win_prob, value_dist}_*`,
    each with `_share` (on the common `T`), `_norm_shared`, and `_policy_cosine` (<0 = that aux fights
    the policy). So the species CE, move BCE, SimSiam latent, move-latent grading, win-prob and value-dist
    pulls are **attributable individually** (the old combined `belief_share` lump is gone) — watch each
    sit small (~a few %); a spike with a degrading policy → lower THAT term's coef. `win_prob`/`value_dist`
    are ≈0 under `read_only` (stop-grad), real under `shaping`.
  - **`grad/distill_share`** (`gen3_grad_distill_share_v1`): the exploiter-distillation **policy KL**'s
    own entry in the same dict — the dose meter `design_advantage_gated_distillation.md` §6.2
    dose-matches the G1/G2 arms on (gradient share, not coefficient). Policy KL ONLY, deliberately:
    the value-side distill coefficients are held fixed across those arms, so folding them in would
    compress the very differences the meter reads. When distill is on, the once-per-`train()` sample
    waits for a minibatch with a live distill term (unless the whole rollout holds no teacher-team
    rows, in which case it samples immediately rather than suppressing the probe); distill off → not
    logged, zero cost.
- **Per-edge-family LIVENESS — `edge/<fam>_weight_norm` + `edge/<fam>_grad_norm`**
  (`edge_family_metrics`, sampled once per `train()` right after the backward so `.grad` is still
  populated; parameters only, so the forward — and therefore the CPU opponent path — pays nothing).
  Every family enters as a ZERO-INIT `Linear(cell → 2·n_heads)`, which means **a family that never
  learns anything is bit-identical in the logs to one that works**: both write zero into the
  attention bias and neither says a word. The v79 `h` (pair-history) family shipped into a
  production run with exactly that blindness, and the only recourse would have been a post-hoc
  ablation at run end.
  - `weight_norm` — how far the map moved off its zero init. **Has it learned anything?**
  - `grad_norm` — how hard the loss is pushing it right now. **Does anything want it to?**
  - Read as a PAIR: both ~0 = genuinely dead (the cell carries nothing the loss can use); weight ~0
    with grad > 0 = still climbing off init (the expected early reading); weight > 0 with grad ~0 =
    converged and contributing. Weight norm alone cannot separate the first two.
  - ⚠️ **Neither is an EFFECT SIZE.** Both scale with the cell's input magnitude, so a family with
    larger-magnitude inputs shows a bigger gradient regardless of usefulness — measured at init on
    the gen-12 config, `h` reads the largest `grad_norm` of all 16 families (0.0100 vs d3's 0.0032),
    which says it is alive and being pushed, **not** that it is the most useful. The per-family
    ABLATION audit remains the only thing that measures importance, and these must never be quoted
    in its place.
- **Per-CELL LIVENESS — `cell/<name>_weight_norm` + `cell/<name>_grad_norm`**
  (`cell_family_metrics`, same window, same backward, same parameters-only cost). The identical gap
  one layer over: `SwitchBranchMoveCell`, `PairOutcomeMoveCell`, `PairOutcomeSwitchCell` and
  `ConditionalThreatCell` each enter through a **ZERO-INIT `proj` Linear** — deliberately, so that
  ON-at-init is byte-identical to OFF and any measured effect is something the run LEARNED — which
  means an enabled cell that never learns contributes exactly zero to every action logit and looks
  exactly like one that works. gen-16 turns four of them on at once, in the run meant to decide
  whether the switch-branch channel kills the bait-loop pathology, where **"the behaviour did not
  change" and "the cell never came off zero" must not be the same observation**
  (`designs/research_state/bait_loop_hunt.md` §6 makes this the launch-window check).
  Read the pair exactly as the edge families' — and under the same ⚠️: a parameter magnitude is not
  an effect size. `CELL_FAMILIES` is DECLARED, not duck-typed, so a renamed cell breaks the test
  rather than going quietly unmonitored. Nothing is emitted for a cell that is off — it is absent
  from the extractor, and a zero would read as "enabled but dead", a different claim.
- **Value scale — PopArt prep.** From the full rollout buffer: `train/return_mean` / `train/return_std`
  / `train/return_abs_max` (exactly the `(μ, σ)` + tail an adaptive return normalizer / PopArt's ART
  half tracks) and `train/value_pred_std` (the value head's actual output spread). Watch these to SEE
  the non-stationary value-scale drift (reward annealing / policy improvement) that a static `vf_coef`
  cannot follow. Plus `train/grad_norm` (pre-clip total grad norm, mean over minibatches → grad-clip
  activity).

Cost: **2 partial backward passes on ONE minibatch per `train()` call**, plus **one more per ACTIVE
auxiliary** (species/move/latent/move-latent belief + win-prob + value-dist → up to ~8 total when every
head is on; each is the `aux_terms` dict's per-term `autograd.grad`) — all on the single sampled
minibatch, negligible vs the `n_epochs × n_minibatches` the loop already runs + trivial NumPy stats. The
probe is a **no-op**
(records nothing) when `shared_trunk_parameters` finds no matching modules (a non-Gen3 policy). **Why
it exists:** to prepare for **reducing `vf_coef`** and **adding return normalization (PopArt)** — both
target the value→trunk pressure, which can now be tuned to a number instead of inferred. (The
`+INSTRUMENTATION` markers in `instrumented_ppo.py` flag the added lines; the upstream-drift hash check
is unaffected since it hashes only `sb3_contrib.MaskablePPO.train`.)


## What to watch on a WIN-PROB run — the 28-tag dashboard, annotated

> The leaf carries this heading (it is pinned there) with a one-line-per-tag form. The
> annotated version, with each tag's currency and reading rule, is below.

PROBABILITY units, which is the era's whole point: the critic, the returns and the head finally share
one scale. Read the run's `🎯 [CRITIC]` startup line first — a `shaped` run's `train/*` value tags are
not comparable with these.

Grouped by the question each answers, **with its currency**. Everything in the first two blocks is in

**Is it getting stronger?** *(win rate / ELO)*
1. `eval/elo` **+ `eval/elo_ci`** — anchored Bradley-Terry, the only cross-run number. Quote the run-END `snapshot_ladder/ladder.json`, never a mid-run node (the newest BT node is systematically inflated).
2. `eval/win_rate_vs_bots` — saturates, so read it as an ALARM: a fall is a regression.
3. `eval/win_rate_vs_pool` — the promotion gate pins it near 0.50; leaving 0.50 is the news.
4. `signal/outcome_win_rate_bots` · `_pool` — the same, REALIZED, per rollout instead of per cycle.
5. `rollout/ep_rew_mean` — under the indicator this IS the training win rate (see the window note above).

**Is the critic HONEST?** *(probability — the era's central claim, and the §4.3 gate's inputs)*
6. **`win_prob/critic_resolution`** — Murphy RES, **HIGHER is better**. The G1 PRIMARY meter; `main.critic_gate` reads its bar from the committed baseline artifact.
7. `win_prob/critic_reliability` — Murphy REL, LOWER is better. Miscalibration only.
8. `win_prob/critic_brier` · `critic_skill` — the DEPLOYED value's proper score, and its skill over the base rate.
9. `win_prob/critic_uncertainty` · `critic_base_rate` — the decomposition's context; a skill score without them is unreadable.
10. `win_prob/critic_decomp_residual` — must sit at ≈0, or REL−RES+UNC is not a decomposition of that Brier.
11. `win_prob/ece` · `win_prob/mce` — the HEAD's own calibration: the count-weighted average, and the worst READABLE bin.
12. `win_prob/rel_gap_b0`…`b9` — the SHAPE of the miscalibration. A NaN renders as a hole and means an under-populated bin, never a zero error.
13. `win_prob/start_gap` — the PAIRED episode-start read. **Positive = optimistic at the opening board.**
14. `train/explained_variance` — now EV in P(win) units (the same number in either currency — see below).
15. `win_prob/coverage` — the fraction of rows carrying a label. A fall here invalidates every line above it.

**Is the OPTIMIZATION healthy?** *(unitless)*
16. `train/approx_kl` — the KL controller's input.
17. `train/learning_rate` · **`train/dose_rate`** — the step size, and the quantity that predicts collateral.
18. `train/clip_fraction` — how much of the surrogate is at the clip bound.
19. `train/grad_norm`.
20. **`grad/value_policy_logratio`** — value-vs-policy pull on the shared trunk, aux-independent. Now that the BCE IS the critic, this is the `--vf-coef` gauge.
21. `grad/policy_share` · `value_share` · `aux_share` — the trunk partition (corrected this pass; see the GIGO note).
22. `train/noise_scale_ratio_policy` — read BESIDE `train/noise_scale_ratio`, never alone.
23. `signal/adv_raw_mean` **as a ratio to** `signal/adv_raw_std` — a systematically mis-centred critic; the no-harm watch.

**The G7 KILL condition, and the GIGO guards** *(the two that must read an exact number)*
24. **`rollout/ep_len_mean`** — a PRIMARY endpoint, not a monitored one. A `[0,1]` critic cannot represent "a timeout is worse than a loss", so stalling is this era's registered failure mode.
25. **`signal/draw_rate`** — the same failure as a rate.
26. **`reward/untracked_abs_mean` must read exactly 0.0.** Anything else means the startup composition census and the reward folds disagree.
27. **`train/return_abs_max` must read exactly 1.0.** A departure means the reward stream is not the win indicator the mode's `V = P(win)` identity rests on.
28. `time/fps` — the run is still moving.

**What to read when the contested split is ABSENT:** on a run whose margin has no spread the whole
`contested_*` family is now gated off rather than duplicated, so the contested-vs-blowout question
is answered offline by `python -m main.scaffolding_gauge --reliability --reliability-reweight`
(which stratifies bot vs pool sentinel, where the head's bias FLIPS SIGN) and by
`main.critic_gate`. ⚠️ A run started before `gen3_obs_margin_unconditional_v1` carries the
degenerate margin AND, if it also predates the gate, the duplicate tags — read them as absent.
