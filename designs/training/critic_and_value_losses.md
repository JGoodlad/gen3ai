# Training — the critic mode and the value losses

Moved verbatim out of `src/agents/training/CLAUDE.md` on **2026-09-08**, in the second pass of the
topic split (that leaf was 3,183 lines / 264 KB, loaded in full for any session touching the
training package). Each section below is unchanged, including its dated measurements.

`src/agents/training/CLAUDE.md` keeps the heading and a short summary of each, and points here.
**This file is the owner of the detail.**

---

## THE VALUE LOSS has a MODE — `--critic {shaped,winprob}` (`gen3_winprob_critic_mode_v1`)

**Default `shaped`; a flagless run is byte-identical.** Design of record:
`designs/ai_v12/design_winprob_only_critic.md`. The model-side half (the route, the version gate)
is `src/agents/model/CLAUDE.md` → *The CRITIC MODE*; this is what `train()` does about it.

| | `shaped` | `winprob` |
|---|---|---|
| the value TERM | `vf_coef · _value_loss_from_se(...)`, or the HL-Gauss CE at `vf_coef` under `value_from_dist` | `vf_coef · _win_prob_loss(...)` — the head's **BCE against the terminal outcome** |
| noise-scale group | `value` | **`value`** |
| the scalar `value_loss` | the loss | a DIAGNOSTIC only (its term is dropped, `_vf_term = 0.0`), and computed UNCLIPPED |
| `--win-prob-coef` | weights the auxiliary BCE, tagged `aux` | refused — the BCE is the value loss now |
| gate | `win_prob_coef != 0` | `win_prob_coef != 0` **or** the critic — the head's own loss cannot be switched off by a coefficient |

**The `"value"` tag is the point of `gen3_value_diagnostics_v1`'s sibling finding, applied one
critic over.** §1.4 of the design records that under `--value-from-dist` the REAL critic loss was
folded as `_ntg.add("aux", …)` while `_vf_term` was 0.0 — so `train/noise_scale_value` spent that
entire era describing a term with weight zero, and the grad-balance probe had to compensate
separately. The promoted BCE joins `value`, and `grad/value_share`'s term follows the critic
(`win_prob_term if critic_winprob …`) for the same reason: a `grad/value_share` measuring the
FROZEN scalar head's pull is the 2026-07-22 catch (`grad/value_dist_share` stuck at ~0.05).

**`win_prob/critic_*` — the P(win)-currency reliability read, once per rollout.** Under `winprob`
only, from `agents.training.scaffolding.reliability_table` — **imported, never re-implemented**, so
the live number and `python -m main.scaffolding_gauge --reliability` are the same statistic and a
run's series is comparable with the committed 2026-09-06 baseline. It reads the rollout BUFFER's
`values` (which under this critic ARE the P(win) GAE bootstrapped from) against
`win_target`/`win_mask`, i.e. the DEPLOYED quantity — where the `win_prob/ece` family above reads
the head's logits per minibatch. Keys: `critic_{brier,skill,ece,mce,reliability,resolution,uncertainty,decomp_residual,base_rate,n}`.

🚨 **`critic_resolution` IS the meter; `critic_reliability` is not.** A base-rate forecaster scores
a perfect 0 reliability and a useless 0 resolution. The committed baseline measured this head at
reliability ~0.002 against a resolution of 0.062 out of an available 0.182 — already calibrated in
the MEAN, starved of SEPARATION — so a promotion that improves ECE and leaves `critic_resolution`
flat has moved the meter that was never the disease. ⚠️ It is computed on the TRAINING population
rather than the loss-enriched eval quota, so it needs no selection reweighting and its LEVEL is
**not** comparable with the offline gate's — only its trend. An unmeasurable rollout publishes
**`{}`, never zeros**: a calibration of nothing and a perfect calibration must not render the same.

**What `winprob` does to the REWARD, and the one thing it gives up.** The stream becomes the
TERMINAL **win indicator** alone (`--no-hand-shaping` + `--terminal-indicator` +
`--victory-value 1.0` + `--draw-penalty 0`, all four REQUIRED and each named by its own
`combination_checks` refusal), so the undiscounted return is exactly `1{win}` and, at `--gamma 1.0`,
`V(s) = P(win | s)` with no approximation term. The cost is stated rather than buried: **a critic
bounded in [0,1] cannot represent "a timeout is worse than a loss."** `--draw-penalty`'s
`−35 < −30` ordering is unrepresentable there, so the anti-stall pressure is the obs deadline clock
plus **`--arm-no-progress-tax`** (design gap B4) — which re-arms `no_progress_tax` alone under
`--no-hand-shaping`, without reviving the other 24 BIAS terms. **Stall rate and mean episode length
are PRIMARY, kill-condition-bearing endpoints on a `winprob` arm.**

**THE DRAW BRANCH IS EXPLICIT, and `signal/draw_rate` states its frequency** (design §3.2 / gap
B9). `battle.won` is a TRI-STATE — True / False / **None**, the last being a draw or the 250-turn
timeout — and `MaskableAgentWrapper.step` used to reach `0.0` for the third case through a boolean
test, i.e. by accident. It is now a named branch: **a draw is scored as a NOT-WIN by decision**
(`y = 0`), because that makes "P(win)" literally P(win); 0.5 would make the critic systematically
wrong exactly where stalling tempts; and masking the episode out would leave its ~250 decisions
with no learning signal at all. It is SCORED, never dropped — `info["win_draw"]` rides beside
`win_outcome`, `SignalMetricsCallback` counts it on the terminal scan it already runs (both rollout
paths), and `signal/draw_rate` + `signal/n_terminals` publish the rate per rollout.

⚠️ **It is `signal/draw_rate`, not the `train/draw_rate` the design proposes**, for two reasons:
that callback carries a PINNED prefix contract (every row it emits must start with `signal/`), and
on the merits the draw rate is an OUTCOME statistic whose literal siblings — `signal/outcome_win_rate`,
`signal/outcome_entropy` — are computed from the same terminal `info` in the same loop.

⚠️ **The label and the objective DISAGREE about draws under the shaped terminal, and that is a real
property of that composition rather than a defect here.** The label scores a timeout as a not-win
(`y = 0`, the same as a loss) while the reward pays `--draw-penalty` (−35, i.e. WORSE than the −30
loss). Under `--terminal-indicator` they agree: a timeout pays 0.0, exactly like a loss.

**`--gamma` is now a flag** (design gap B6; it was hardcoded at `model_build.py`'s
`InstrumentedMaskablePPO(...)` call). Its `shaped` default is `reward_weights.PBRS_GAMMA` itself
rather than a retyped `0.9999`, and the `PBRS_GAMMA == reward_config.gamma == model.gamma` assert
is now **GATED on a hand potential actually being folded** on both build paths — under
`--no-hand-shaping` every `_fold_*_pbrs` early-returns, so there is nothing for the invariance
claim to be about and an ungated assert would refuse a coherent run. It is **INERT ON A RESUME**
like `--lr`: SB3 restores the checkpoint's own γ, the resume path SAYS so, and it re-points
`reward_config.gamma` at the value actually in force so the two cannot silently disagree.

### 🚨 THE 250-TURN CAP IS A TERMINAL, NOT A TRUNCATION — and it was the other way round (B6)

**THIS ENV NEVER TRUNCATES IN THE SB3 SENSE, AND THAT IS THE WHOLE FINDING.**
`PokeEnv.calc_term_trunc` (`poke_env/environment/env.py:951`) sets EITHER flag only when
`battle.finished`; it then splits a finished battle by *how* it finished — exactly one side wiped ⇒
`terminated`, anything else ⇒ `truncated`. "Anything else" is the **250-turn cap forfeit** and a
genuine **tie**. Both are OUTCOMES. Nothing here is a time limit that interrupted an episode
mid-flight, which is the one thing `TimeLimit.truncated` is supposed to mean.

MEASURED 2026-09-06 on a real bridge battle with `StallConfig.threshold` lowered to 6:
`gen3_env.action_to_order` returns a `ForfeitBattleOrder` at `turn >= threshold` (`== MAX_TURNS` in
production), Showdown answers `|win|<opponent>`, and the boundary reported `finished=True`,
`won=False`, **six mons alive a side**, `terminated=False, truncated=True`.

That flag then becomes `info["TimeLimit.truncated"]` in `DummyVecEnv`/`SubprocVecEnv`, and
`MaskablePPO.collect_rollouts` (`sb3_contrib/ppo_mask/ppo_mask.py:251-260`, mirrored by our
`async_vec_env.collect_rollouts_async:216-219`) does `rewards[idx] += gamma * V(s_last)`. Under
`--critic winprob` the terminal reward is the win indicator — **0** for a cap loss — and γ is
**1**, so the last step's target becomes `0 + 1.0·V(s_last) = V(s_last)`: **a TD error of
identically zero.** The timeout leaves the loss entirely, and a policy that stalls to the cap is
taught nothing about it — while G7's stall rate is a KILL CONDITION on that arm, i.e. the gate
would have been reading a signal the critic never received. Verified by revert: the row reads
exactly `V`.

**`agents/training/wrappers.resolve_episode_end` is the fix** — under `winprob` only, a
`trunc and not term` end is re-labelled TERMINAL. It sits at the END of
`MaskableAgentWrapper.step`, so every outcome consumer above it still sees the flags the sim
produced, and it is the ONE seam upstream of BOTH rollout loops (which is why it is here and not in
`Gen3Env.calc_term_trunc`). The mode arrives as `MaskableAgentWrapper(critic=…)` from
`env_factory`. Pinned by `winprob_truncation_test.py`, including the SB3 composition through a real
`collect_rollouts`.

**`shaped` is UNCHANGED and byte-identical, and what it does is worth stating rather than leaving
implicit:** a cap forfeit and a tie both bootstrap `0.9999·V(s_last)` on top of a terminal reward
that already paid `--draw-penalty`. Two wrongs that partly cancel — the shaped terminal
double-counts the ending while the bootstrap removes it — and re-deriving that composition is a
`shaped`-era question this change does not reopen.

**Two adjacent paths are CLEAN, checked rather than assumed.** A launcher SIGTERM
(`lifecycle.abort_training`) saves a checkpoint and `os._exit`s, so the partially-collected rollout
buffer is discarded and never reaches an update — SB3 does not persist it. The eval
reset-mid-battle forfeit lives in `PokeEnv.reset`, which returns only an observation and fills no
rollout buffer. Neither can leak a truncation into the loss.

⚠️ **CORRECTION to a comment that shipped with the mode: the cap forfeit is a LOSS, not a draw.**
`won_by` (`poke_env/battle/abstract_battle.py:1619-1626`) sets `_won = False` on `|win|<opponent>`;
`won is None` is `|tie|` ALONE. The label is 0.0 either way so nothing downstream moved, but
`signal/draw_rate` counts **ties and not timeouts** — the series that watches the cap is
`signal/stall_rate` / mean episode length, which is why those and not `draw_rate` are G7's kill
condition.

### `--vf-coef` multiplies a BCE now, and the first NON-DEGENERATE update SAYS what that is worth

`--vf-coef` means a different quantity under each critic while keeping its name: under `shaped` an
MSE on a PopArt-normalised shaped return (O(100) unnormalised, on a ±30 scale); under `winprob` the
win-prob head's **BCE against a Bernoulli outcome**, which is `ln 2 ≈ 0.693` per sample at
initialisation and falls. The 0.5 default was tuned against the first and carries no information
about the second, and the normalisation that made the two comparable is refused here.

So `calibration.announce_vf_coef_scale` prints, ONCE, two numbers that answer different questions:
the **raw BCE** (a statement about the HEAD — `ln 2` is chance, well below it means the head
already calls lopsided games) and the **ratio of the value term's shared-trunk gradient norm to
the policy term's** (the statement about the COEFFICIENT — the value norm scales linearly in
`vf_coef`, so the ratio IS the factor to divide it by, `≫1` cut / `≪1` raise).

🚨 **IT USED TO DIVIDE THE VALUE TERM BY `|policy loss|`, AND THAT DENOMINATOR IS DEGENERATE BY
CONSTRUCTION.** On epoch 1 the clipped surrogate has `ratio ≡ 1` and sits at its stationary point,
so `|policy loss| ≈ 0`. The live arm `ai_v12_01_winprob_critic` printed `value term = 0.5 × BCE
0.1330 = 0.0665 against |policy loss| 0.0004 → **165×**` on rollout 1 — and its own gradient series
reads UNREADABLE at rollout 1 (`grad/policy_norm_shared` exactly 0.0 against a value norm of 7.53),
**91×** at rollout 2 and **4.6×** by rollout 17. 165× was not a noisy estimate of 4.6×; it was a
number about the epoch, and a ratio of LOSS MAGNITUDES was the wrong quantity anyway — what
competes on a shared trunk is the gradient.

**The norms are READ from `grad_balance_metrics`, never recomputed.** That read-only
`autograd.grad(retain_graph=True)` probe already runs per-term on every `train()`, so the printed
ratio is exactly `10 ** grad/value_policy_logratio` and no second backward pass is run for a
banner. **The threshold is on the NORM, not the epoch** (`MIN_POLICY_GRAD_NORM`, 1e-6): the probe
samples ONE minibatch per `train()` and it is in epoch 1 by construction, so there is no later
epoch to wait for inside a reading — and epoch 1 is not degenerate in general, since the policy
GRADIENT at `ratio ≡ 1` is `A·∇log π ≠ 0`. It does NOT latch on an update it could not read (no
scorable label, no grad probe, or a policy norm under the floor — the next rollout tries again),
and `_vf_scale_announced` is in `_excluded_save_params`, so a launcher restart re-prints it beside
the startup `[CRITIC]` banner. It is a PRINT, not a scalar: past the first reading the right
instrument is `grad/value_policy_logratio` (0 = balanced). Reading rule: design §5.4.

## PopArt value-target normalization (`--use-popart`)

The fix for the swamping the diagnostics above reveal. `train()` reads `self.popart =
getattr(self.policy, "popart", None)` (built by the policy when `--use-popart`; see
`src/agents/model/CLAUDE.md` → PopArt for the math + version-checking). When present: once per
`train()` (before the epochs) `popart.update(self.rollout_buffer.returns, self.policy.value_net)`
advances the running `(mu, sigma)` **and** POP-rescales `value_net`; the value loss then becomes
`MSE(popart.normalize(returns), popart.normalize(values))` — the **normalized**-space loss, so the
value gradient into the shared trunk drops by ≈`sigma²` and stops swamping the policy. The policy's
value sites de-normalize, so `rollout_buffer.values` / GAE / advantages stay real-unit — the policy
path is untouched. **`--use-popart` requires an explicit `--clip-range-vf none`** (errors otherwise —
self-documenting config; clipping is unnecessary with value normalization, and would clip in
un-normalized units). New diagnostics ride the same generic metrics path:
`popart/mu`, `popart/sigma` (watch them track `train/return_mean`/`return_std`),
`popart/value_weight_norm` (POP keeps it bounded). Under PopArt `train/value_loss` is the normalized
loss (≈O(1)) and `grad/value_policy_logratio` should fall from a large positive value toward ~0 (the
aux-independent value/policy balance — `grad/value_share` also drops but moves with the aux count, so the
log-ratio is the cleaner confirmation it worked).

## Tail-weighted value loss (`--value-tail-weight`)

A probe-driven critic-tail lever (off by default). A representation probe found the critic's TD-residual
tail is fat and barely anticipated (the V-tail crater the `eval/td_resid_tail` CVaR@5% already tracks),
so `InstrumentedMaskablePPO._value_loss_from_se` replaces the plain `F.mse_loss` at all **three** value
sites (PopArt-normalized / unclipped / clipped) with a **CVaR blend**:
`value_loss = (1−β)·MSE + β·mean(worst _VALUE_TAIL_FRAC=10% squared errors)`, computed in whichever
space the branch uses (NORMALIZED under PopArt, so the tail selection matches the loss scale). At **β=0
it is `se.mean()`, byte-identical to `F.mse_loss`** (the default no-op). β>0 makes the critic prioritise
the big over-claim misses it under-prices; it is **symmetric in error sign**, so V stays an unbiased
mean estimate and the GAE advantages the policy reads are unaffected — a weighting change, not a new
target. The hparam is set on the model after construction (like `_async_rollout`), **resume-immutable**
(recorded in `model_config.json`, FATAL to change on resume via `ModelVersion.check_value_tail_weight`,
`MODEL_CONFIG_VERSION` v11; excluded from `check_compatible` since a frozen opponent never runs the value
loss), and **not weight-shape** (no `ARCH_SIGNATURE` bump). The v10
`value_active_readout` value-head fix that used to pair with it is **deleted** (v88
`gen3_dead_flag_purge_v1` — it was never enabled in a gen-8+ run and the multi-seed readout /
`--value-threat-inject` superseded it; a checkpoint recording it ON is refused by the migration). Validate by watching
`eval/td_resid_tail` fall.
Tests: `instrumented_ppo_test.py` (β=0 == MSE, β>0 == the exact blend).

## TD-consistency auxiliary (`--td-aux-coef`, `td_aux.py`)

**What it fixes.** The critic's only signal is a PER-STATE regression, `MSE(V(s_t), G_t)`. That
constrains each state's LEVEL and says nothing about the DIFFERENCE between two adjacent states — so
independent per-state noise ε in V arrives in `ΔV` at `2·Var(ε)`, exactly where the truth is nearly
constant. Since ΔV is what GAE reads, that is injected advantage noise on **every** transition, not
just the dramatic ones. `--td-aux-coef λ` adds the Bellman identity the critic already owes, as an
explicit loss:

```
loss += λ · mean_pairs[ ( V(s_t) − r_t − γ·V(s_{t+1}) )² ]
```

Both residual ends carry gradient (the residual-gradient / Baird form — see the *Cons* in the
pre-registration). `λ = 0.0` is the default and the whole block is skipped, so an OFF run is
byte-identical. **Pre-registered band: 1.0–3.0, 3.0 the favourite; `λ ≤ 0.1` measured significantly
WORSE than control offline, so the small-coef regime is to be avoided, not treated as "a bit of the
effect".** Full pre-registration (rung-1 evidence, the honest ceiling, the rung-2 gates):
`designs/research_state/levers/td_consistency_aux.md` (ledger C5). Do not edit that file — it is the
pre-registration.

**Where the pairs come from — this is the whole engineering problem.** `RolloutBuffer.get()` yields
a RANDOM PERMUTATION, so a PPO minibatch contains **no adjacent pairs at all**; the pairs have to be
drawn from the buffer's surviving `[n_steps, n_envs]` structure. `td_aux.sample_contiguous_pairs`
draws `TD_AUX_STATES` (512) rows as contiguous per-env runs of `TD_AUX_SEG_LEN` (16) and pairs their
adjacent rows. Four facts make it correct:

- **Row convention.** After the first `get()`, `observations` are `swap_and_flatten`ed to ENV-MAJOR
  (`row = env·n_steps + t`), so temporal adjacency survives; the sampler returns rows in exactly
  that convention and `_td_aux_term` **raises** if `generator_ready` is False rather than indexing
  an un-flattened array (which would silently mis-pair states with rewards at any `n_envs > 1`).
- **`rewards` / `episode_starts` are NOT in `get()`'s flatten list**, so they stay `[n_steps,
  n_envs]` and are read in their native shape (rewards are swapped to env-major at use).
- **Episode boundaries DROP the pair, never zero it.** `episode_starts[t+1] == 1` means the
  successor begins a new episode, so (t, t+1) is not a transition; zeroing would train
  `V(s_t) → r_t` at every battle end. This also disposes of SB3's time-limit bootstrap (which folds
  `γ·V(s_term)` into the stored reward at the done step): that row's successor always starts an
  episode, so the pair never forms.
- **Segments, not random pairs.** L contiguous states serve L−1 pairs off L forwards — the
  "K+1 forwards serve K pairs" economy the pre-registration calls for, ~2× cheaper per pair than
  independent pair sampling. Rung 1 also found whole-battle batching beat a random-permutation
  control by 12%, so the within-segment correlation is a feature.

**It runs per MINIBATCH, with its own sample and its own critic forward** — modelled on the
search-teacher / OPD folds, not on the once-per-`train()` diagnostic probes. Those are read-only;
this one carries gradient, and a once-per-`train()` fold would give it ONE contribution against the
value loss's `n_epochs × n_minibatches` (~240 in production), so λ would have to be ~240× rung-1's
band to mean the same thing. Cost is bounded by `TD_AUX_STATES`, not by `batch_size`: one extra
512-state critic forward per minibatch, ≈10% of the train step at production shapes.

**The value path is `policy.predict_values`, never a hand-rolled one.** That method is what routes
to the DISTRIBUTIONAL head's mean under `--value-from-dist` (where the scalar `value_net` is FROZEN)
and applies PopArt's de-normalization — reading `value_net` directly would train a critic the run
does not use.

**Units.** `predict_values` returns REAL-unit values and the buffer's rewards are real-unit, so the
raw residual is real-unit. But under PopArt the value loss trains in NORMALIZED space, so the
residual is divided by σ — which *is* the normalized-space residual, since
`normalize(V) − normalize(r + γV′) = (V − r − γV′)/σ` (the μ cancels). λ therefore keeps the meaning
rung 1 calibrated in both regimes; σ = 1.0 with PopArt off.

**Metrics (`td_aux/` prefix).** `resid_rms` is the headline — the quantity being minimised, the live
counterpart of the offline ΔV-dispersion instrument, and it should FALL. `resid_mean` (SIGNED) is
the no-harm watch: rung 1's decomposition says this is dispersion suppression, so a bias drifting
away from ~0 means the residual-gradient term is shifting the LEVEL rather than tightening it — read
it beside `train/explained_variance`. Also `loss`, `n_pairs`, `scale` (the σ the residual is
expressed in) and `pair_drop_frac` (share of candidate pairs lost to episode boundaries). The
shared-trunk pull rides `grad/td_aux_share` + `grad/td_aux_policy_cosine`; the term reaches the trunk
through the CRITIC path only, so `td_aux_share` against `value_share` is the read for "is the
consistency term crowding out the level regression it is meant to complement".

**Class: `training_coef`.** Scales a loss, touches no forward pass ⇒ NO `ARCH_SIGNATURE` bump, NOT in
`check_compatible` and no `check_*` of its own; recorded on `ModelVersion` (`MODEL_CONFIG_VERSION`
v90) purely for provenance and so a **flagless resume inherits it** via `_resolve`, exactly like
`--opp-belief-aux-coef`. It is deliberately NOT in `agents/model/flag_registry.py` — that registry's
scope is extractor architecture toggles, and this reaches the extractor not at all.

Tests: `td_aux_test.py` — the sampler (env-major row convention, (t, t+1) adjacency, boundary pairs
DROPPED not zeroed, the all-boundary degenerate → `None` not 0.0, the segment economy, fail-loud on a
flattened `episode_starts`), the residual math on a hand-built case, the PopArt scale identity, both
ends carrying gradient, and on a REAL `train()`: coef-0 byte-identity (asserted twice — identical
parameters AND the sampler monkeypatched to raise, so a future sampler change cannot perturb an off
run), coef>0 moving the update and logging every metric, gradient landing on `value_net`, and the
un-flattened-buffer refusal.

## Distributional value head (`--value-dist-mode` / `--value-dist-coef`)

The training half of the v29 interpretability side head (model side: `src/agents/model/CLAUDE.md` →
distributional value head). A categorical readout off `value_pooled` whose softmax is the critic's
predicted **return DISTRIBUTION** — the shape the scalar V collapses (sharp = confident, wide =
uncertain, bimodal = coinflip). **Phase A** (interpretability-only): it does NOT replace the scalar
critic, so the GAE/advantage/value-loss path is untouched — this loss is an ADD-ON, like the win-prob
aux. Design + the K1 honesty frame: `designs/ai_v6/design_distributional_value_critic.md`.

- **Loss (`instrumented_ppo._value_dist_loss`).** **HL-Gauss** (Farebrother et al. 2024): build a
  Gaussian-smoothed soft target by integrating `N(target, σ_g²)` (σ_g = 0.75·Δ) over each atom's bin,
  with the two EDGE bins absorbing the outer tails (graceful out-of-support handling), then cross-entropy
  against the head's `log_softmax`. `train()` reads the stashed `last_value_dist_logits` + the rollout
  return as the target, **PopArt-normalized when the scalar critic is** (so the target lands in the head's
  support space — set `--value-dist-vmin/vmax` to a normalized range like ±5 under `--use-popart`). Folded
  at `value_dist_coef`. Pure + static → unit-tested in `value_dist_loss_test.py`.
- **Metrics (`value_dist/*`).** Aggregate interpretability health under its own TB prefix (the
  `grad/`/`popart/`/`win_prob/` group convention): `ce`, `entropy` + `std` (fall as the critic sharpens),
  `pit_mean` (≈ 0.5 ⟺ **calibrated** — the PIT anchor), `mean_abs_err` (`|E[Z] − return|` in support
  units). Ride the generic logger → TensorBoard + launcher TUI (`value_dist/*` labels in `format.py`).
- **Versioning.** `value_dist_mode` (str) + `value_dist_bins` (int) are version-checked structural toggles
  (fresh-only); the support `vmin`/`vmax` is resume-immutable (`check_value_dist`); `value_dist_coef` is
  **training-only**, read back on a flagless resume (like `win_prob_coef`). Threaded into
  `current_model_version` / `arch_toggles_from_model` / `_run_arch_toggles`.
- **Forensic trace + prober.** `RLPlayer._value_dist` reads the stashed logits at capture (softmax ⇒ the
  per-atom distribution) → `BattleRecorder.states_arrays` writes a `value_dist [T, bins]` npz array (key
  OMITTED when the head is off → the prober's KeyError "unavailable" path; NaN rows = uncaptured). The
  prober renders the per-decision **histogram** + mean/std/P10–P90/entropy/bimodality
  (`engine.build_value_dist` → `ValueDistView`, model-free; in the Summary panel + the `analyze` CLI). See
  `src/main/prober/CLAUDE.md`.
- **Honesty gate.** Ledger **K1 already killed the distributional critic as a WIN-RATE lever** (sub-Gaussian
  residuals — no tail). This is justified on INTERPRETABILITY only; its strongest use is upgrading the
  prober calibration/`falsify-scan` luck-vs-mistake split (predicted spread vs realized return = a
  within-model PIT). "Learns ≠ helps" — validate calibration (PIT ≈ uniform), not win-rate.
- **Tests.** Unit: `value_dist_loss_test.py` (HL-Gauss math + diagnostics), `agents/model/
  value_dist_head_test.py` (module build, off byte-identical, grad gating, the v29 version gate),
  `main/prober/engine_test.py` (`build_value_dist`). End-to-end `--debug --debug-eval --use-bridge=node
  --value-dist-mode read_only` smoke captures a trace whose npz carries `value_dist`.

