# Training — telemetry scalars

Moved verbatim out of `src/agents/training/CLAUDE.md` on **2026-09-07**, when that leaf was
split by topic (it was 8,219 lines / 676 KB, loaded in full for any session touching the
training package). Each section below is unchanged, including its dated measurements.

`src/agents/training/CLAUDE.md` keeps the heading and the opening paragraph of each, and
points here. **This file is the owner of the detail.**

---

> **Note:** the *TensorBoard export census* is deliberately NOT here — it stays in
> `src/agents/training/CLAUDE.md`, pinned by `tb_relevance_test.py`. This file holds the
> capacity telemetry, the `signal/` group and the scaffolding gauge.

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
