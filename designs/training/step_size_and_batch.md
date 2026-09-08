# Training — step size, batch size and the dose

Moved verbatim out of `src/agents/training/CLAUDE.md` on **2026-09-08**, in the second pass of the
topic split (that leaf was 3,183 lines / 264 KB, loaded in full for any session touching the
training package). Each section below is unchanged, including its dated measurements.

`src/agents/training/CLAUDE.md` keeps the heading and a short summary of each, and points here.
**This file is the owner of the detail.**

---

## Gradient accumulation (`--grad-accum-steps`)

A **GPU-memory lever** for keeping a large effective batch when the full minibatch OOMs. Stock
`MaskablePPO.train()` does one `forward → backward → optimizer.step()` **per minibatch**, so
`batch_size` couples the effective-batch size to the activation-memory peak — there is no
`accumulation_steps` knob upstream. `InstrumentedMaskablePPO.train()` adds one: with
`--grad-accum-steps K` it runs K `batch_size`-sized **micro-batches**, summing their gradients, and
calls `optimizer.step()` only **once per group of K**. Because gradients are additive and each
micro-loss is scaled by `1/K`, the accumulated gradient is the **exact** gradient of one
`(batch_size·K)` batch — but the backward graph only ever holds **one micro-batch's** activations.
So `--batch-size 4096 --grad-accum-steps 4` trains with the dynamics of `--batch-size 16384` at ~¼
the activation peak (the `DamageOperator`'s `[B,6,~416]` tensors + the grad-balance probe's retained
graph scale with the micro-batch, not the effective batch). `K=1` (default) is **byte-identical to
upstream** (one step per minibatch).

- **The step is gated on a full group** (`micro_in_group == accum`); a **trailing partial group**
  (#minibatches not divisible by accum) is flushed at epoch end with its accumulated grad rescaled
  `accum/micro_in_group` so the short group's step has the right magnitude. Grad-norm clipping
  (`max_grad_norm`) is applied **once per optimizer step** (per group) — i.e. to the full
  effective-batch gradient, exactly as the big batch would clip it.
- **Bit-exact when the rollout divides cleanly.** The accumulation math reproduces a literal
  `batch_size·K` batch to the float32 noise floor (~3e-8, empirically) **when `batch_size` divides the
  rollout (`n_steps·n_envs`) AND `K` divides the minibatch count** — then every group is `K` equal-size
  micro-batches. Production power-of-2 configs satisfy this (e.g. rollout 131072, `--batch-size 4096
  --grad-accum-steps 4` → 32 micro-batches, 8 groups, exact). For a NON-divisible rollout the single
  smaller remainder minibatch in the **final group of each epoch** is weighted as if full-size — a
  bounded mis-weighting of one remainder per epoch (≈8e-5 on params in a toy probe; negligible vs a
  100k-sample rollout, and no worse than stock SB3, which gives that remainder minibatch its own
  full-weight optimizer step).
- **KL early-stop** (`target_kl`, `None` by default so this path is dormant) discards the partial
  group (`zero_grad`, no step) on a trip — a true `(batch_size·K)` batch checks KL over the whole
  effective batch and would discard it as one unit.
- **The other (always-present) non-identity is per-micro-batch advantage normalization**
  (`normalize_advantage`, default on): stock SB3 already normalizes advantages *per-minibatch*, so
  here the normalization sample is the micro-batch (e.g. 4096) rather than the effective batch
  (16384). The difference is the normalization sample size — statistically negligible for batches of
  thousands (and it is this term, not the accumulation math, that the bit-exact check above isolates
  by running with `normalize_advantage=False`). (The grad-balance probe also samples on the first
  **micro**-batch instead of the first minibatch — a smaller, cheaper, still-representative sample; its
  `retain_graph` memory shrinks with the micro-batch.)
- **Not version-locked / not in `model_config.json`.** It is a pure train-loop knob (no forward
  change, no weight-shape effect, no `ARCH_SIGNATURE`/`MODEL_CONFIG_VERSION` bump) — like `batch_size`
  / `n_epochs`, **forwarded as a CLI flag on every launcher resume** (set on the model in both the
  fresh-build and resume paths of `train_rl_agent.py`; surfaced in `_model_hparams` for the sidecar).
  Change it freely on resume; only the *effective* batch (`batch_size·K`) matters for dynamics, so
  `--batch-size 16384` (K=1) and `--batch-size 4096 --grad-accum-steps 4` continue a run identically.
- **The upstream-drift hash check is unaffected** (it hashes only `sb3_contrib.MaskablePPO.train`).

Tests: `instrumented_ppo_test.py` — `test_grad_accum_matches_full_batch` runs the REAL `train()` on a
minimal `MaskablePPO` and asserts `K=accum` over `batch/K` micro-batches reproduces the parameter
update of `K=1` over the full batch to `rtol=1e-4` (parametrized over a divisible 16=4×4 **and** a
non-divisible 15=5×3 case that exercises the partial-group rescale), plus default-is-1 + source-marker
guards.

### Gradient noise scale (`train/noise_scale`) — "is the batch big enough?"

A **free byproduct of accumulation** (only emitted under `--grad-accum-steps >= 2`) that answers *how
big a batch is enough* with a number instead of intuition: the McCandlish et al. 2018 **simple
gradient noise scale** `B_simple = tr(Σ)/|G|²` — the critical batch size where gradient noise stops
dominating. Below it, a bigger batch buys ~linear per-step progress; above it, diminishing returns
(you're averaging out noise that was already small, and could shrink the batch for more update steps).

The estimator needs the squared gradient norm at **two batch sizes** — and accumulation produces
exactly that for free each `train()`: ‖g‖² of one micro-batch (`B=batch_size`, read from `.grad`
right after the first micro-batch's backward, un-scaled by `accum²`) and of the accumulated first
group (`B=batch_size·accum`, the pre-clip norm `clip_grad_norm_` already returns). From the model
`E‖Ĝ_B‖² = |G|² + tr(Σ)/B`, two `(B, ‖Ĝ_B‖²)` points pin both `|G|²` and `tr(Σ)` (`_noise_scale_estimate`,
pure/unit-tested). Both single-call estimates are noisy (either can go negative), so the **numerator and
denominator are EMA'd separately** (`_NOISE_SCALE_EMA_DECAY`=0.99 ≈ a few-hundred-call window) and only
then divided — and the scalar is emitted only once both EMAs are positive (so a warmup transient never
logs a garbage value). Cost: one extra global grad-norm read per `train()` (the group norm is reused
from clipping); no extra backward. EMA state is **process-local** (resets on a launcher restart →
re-converges in a few hundred calls; not saved).

Two scalars ride the standard logger → TensorBoard + launcher TUI (`format.py` labels `noise scale` /
`noise/batch`, in the train column by `train/grad_norm`):
- **`train/noise_scale`** = `B_simple` (compare directly to your effective batch `batch_size·accum`).
- **`train/noise_scale_ratio`** = `B_simple / (batch_size·accum)` — the actionable read: **≫1 ⇒
  noise-limited** (enlarge the effective batch), **≪1 ⇒ diminishing returns** (you have more than
  enough; could shrink for more/cheaper update steps), **~1 ⇒ the sweet spot**.
- **The NSR advisor (`_noise_scale_advice` / `_emit_noise_scale_warnings`)** — when the SMOOTHED
  ratio leaves the band, a `⚠️ [NOISE]` warning goes to the launcher **Events panel** (via
  `main.launcher.ipc.emit`; plain print standalone) naming the concrete fix: ratio > 2 → "raise
  `--grad-accum-steps` ~ratio× (free — no VRAM/FPS cost, same rollout)"; < 0.5 → "over-batched,
  lower it for more steps per sample". **Rate-limited to one warning per key per 30 min** and
  suppressed for the first ~20 EMA folds (warm-up false-alarm guard). Pure decision logic
  unit-tested (`instrumented_ppo_test.test_noise_scale_advice_bands_and_fixes`). A FiLM-group half
  (`film/noise_scale*`, `--film-grad-accum-steps` and `_GroupGradAccumulator`) measured the same
  thing for the conditioning params until v78 and was deleted with the zarch family.

Tests: `instrumented_ppo_test.py` — `test_noise_scale_estimate_recovers_known_values` (the two-point
math recovers a planted `|G|²`/`tr(Σ)` exactly), `_smaller_batch_is_noisier_sign`, `_global_grad_sq`
matches a manual sum, and `_logged_only_when_accumulating` (real `train()`: skipped at accum=1, EMA
updated + scalar emitted at accum=2).

### 🚨 The total is NOT the policy gradient — the PER-TERM noise scale (`noise_scale_terms.py`)

**`train/noise_scale` is measured on the TOTAL gradient, and on this tree the total gradient is
mostly not PPO.** The loss is the clipped surrogate + the value term + the entropy bonus + a dozen
DENSE supervised auxiliaries (belief heads, win-prob, spread/nature/HP-type, value-dist, TD-aux,
the counterfactual family) + a distillation KL on a fold. A supervised head's per-example gradients
**agree** — its target is a label, not an advantage — so its `tr(Σ)` is small and its `|G|²` is not.
Mixing it into the total therefore **DEFLATES** `B_simple = tr(Σ)/|G|²`, and the run reads
"over-batched" while the term you are actually trying to train may be starved. Acting on the total
in that state shrinks the batch the policy gradient needed.

That confound is not hypothetical here: the live runs read `train/noise_scale_ratio` **0.001 early
and 0.05 late** on the generalists and **1.1** on the v8 fold, i.e. "over-batched 16-1000x" — a
conclusion no batch-size decision should rest on until the policy term has been read on its own.

**Five groups, the SAME estimator.** `PerTermNoiseSampler` accumulates each group's gradient over
the same two batch sizes the total already uses for free (one micro-batch, and the accumulated
first group of epoch 0) and feeds them through the SAME `_noise_scale_estimate` two-point solve and
the SAME separately-EMA'd numerator/denominator. The math is not forked — the whole point of the
comparison is that a disagreement can only be the *gradient*, never the estimator.

| group | is |
|---|---|
| `policy` | the clipped surrogate AS FOLDED (`_policy_grad_term`; at the 1.0 default that is `policy_loss` itself) |
| `value` | `vf_coef · value_loss` (0.0 and therefore absent under `value_from_dist`) |
| `entropy` | `ent_coef · ent_loss_used` — **degenerate at `--ent-coef 0`** (a 0.0-scaled tensor still folds, so the group is present but its norms are 0 and both EMAs stay non-positive ⇒ nothing is emitted, which is the right answer, not a gap) |
| `aux` | every belief / win-prob / value-dist / TD-aux / search-teacher / OPD / counterfactual term, as ONE bucket (`grad/<term>_share` already breaks the heads out individually) |
| `distill` | the `--distill-coef` family — separated because it comes and goes with a fold and its dose is the thing being tuned |

Three scalars per group, beside the existing pair:
- **`train/noise_scale_<g>`** — that group's own `B_simple`.
- **`train/noise_scale_ratio_<g>`** — over the effective batch. **`_ratio_policy` is the headline.**
- **`train/noise_scale_share_<g>`** — `|G_g|² / |G_total|²`, i.e. who owns the true gradient's
  squared length. ⚠️ **The shares do NOT sum to 1 and must not be read as a partition**:
  `|G_total|² = ‖Σ_g G_g‖²` carries the cross terms, so groups pulling together sum above 1 and
  groups fighting sum below it.

**The advisor now reads BOTH.** `_noise_scale_advice` takes the policy-term ratio, quotes it inside
the OVER-BATCHED / NOISE-LIMITED bands, and — when the two land in different bands **or** differ by
≥3x inside one — emits its own `total_vs_policy_disagree` warning naming the aux deflation and
pointing at `train/noise_scale_share_*`. **That disagreement is the finding this exists for**, so it
is a warning of its own rather than a footnote on the total's. The policy ratio is read from the EMA
state (`_per_term_ratio`), not from the last fold, so a call the cadence did not sample still quotes
it.

**It cannot change training, structurally.** The tagger is threaded through the fold as
`loss = loss + _ntg.add("aux", term)` and **`add` returns its argument unchanged**, so the loss
expression is tensor-for-tensor the one that was there (`_ent_term` merely names a sub-expression
whose operations and order are unchanged). Gradients come from `torch.autograd.grad(…,
retain_graph=True)`, which never writes `.grad` — the same read-only mechanism
`grad_balance_metrics` has used per-term on every `train()` for generations, **which is also why
`--compile-trainer` is not a new risk**: the compiled backward is already called repeatedly with
`retain_graph` by that probe. Any exception retires the probe for the call with one printed line and
leaves the step untouched.

**COST — measured, and the default follows the measurement.** The probe costs `n_groups` extra
backward traversals on `accum` micro-batches of a sampled `train()`, against
`n_epochs × n_minibatches` fwd+bwd for the call — so **the overhead is governed by minibatches per
`train()`, not by batch size**. It self-reports (`train/noise_per_term_ms`) against a new
`train/train_ms` (the whole call's wall clock, recorded as `train()`'s last line — the honest
denominator for this and every future probe's cost claim):

| shape (epochs × minibatches per `train()`, accum 2) | `noise_per_term_ms / train_ms` |
|---|---|
| `--debug --n-steps 1024 --batch-size 512` (5 × 2 = 10 units, 4 live groups) | **24.3%** (5 calls) |
| `--debug --n-steps 1024 --batch-size 128` (5 × 8 = 40 units, 4 live groups) | **7.9% / 8.0%** (two runs, 11 calls each) |
| production `--n-steps 2048 --n-envs 64 --batch-size 16384 --n-epochs 10` (10 × 8 = 80 units, 5 groups) | **≈5.0% — EXTRAPOLATED** (2× the units, 1.25× the groups), not measured on GPU |

*(Both measured rows are CPU `--debug` runs on a box carrying a live fleet. That does not
invalidate them: the numerator and denominator are wall clocks from the SAME `train()` call, so
contention stretches both and the RATIO is what survives — which is exactly why `train/train_ms`
was added rather than an external stopwatch.)*

**Default ON** (`PpoHyperparameters.noise_scale_per_term`), because the production shape is well
under the 10% bar. Peak extra memory is one gradient accumulator per live group
(`n_groups × Σ|params|` ≈ 5 × ~16 MB), freed at the end of the call.

**⚠️ WARM-UP: `_policy` is the LAST tag to appear, and that is the signal, not a gap.** A group is
emitted only once both its EMAs are positive. For a strongly noise-limited term `|G|²` is genuinely
near zero at these batch sizes — with `accum=2` the estimate is `2·g_big − g_small ≈ 0` — so its
single-sample estimate SIGN-FLIPS and only the average resolves it. So EVERY reading here —
per-group AND the total — folds through the ONE `noise_scale.debiased_ema`: effective decay
`min(decay, 1 − 1/(n+1))`, i.e. a plain running MEAN until the `1/(1−decay)` window fills and the
exponential decay takes over, which is Adam's `ema / (1 − beta^t)` spelled as a decay. One negative
first sample therefore cannot suppress a tag for hundreds of calls. Measured effect on a 12k-step
debug smoke: without the debiasing `_policy` never emitted in 11 calls; with it, it emits by call
~10.

🚨 **THE TOTAL USED TO BE THE EXCEPTION, AND IS NOT ANY MORE (`gen3_noise_scale_warmup_v1`,
2026-09-03).** `train/noise_scale`'s EMA anchored on its FIRST sample at a fixed decay 0.99, so
after two samples it read `0.99·x₁ + 0.01·x₂` — the first sample, essentially, for its first few
hundred calls. That is why the first production reading on R5F15 had to be published as
"provisional, n=2", and it is the mechanism behind the smoke below in which the total never emitted
at all across 11 calls while every per-term tag did. Both halves now warm up identically, so a
young run's `train/noise_scale` and `train/noise_scale_ratio_policy` are comparable to each other
from the first reading. ⚠️ **A run's `train/noise_scale` series is NOT comparable across this
change** during its first ~100 folds — the fix moves early values by construction; the steady state
past the warm-up window is unchanged. ⚠️ And an EMA is now `(value, COUNT)`: priming
`_noise_ema_s`/`_noise_ema_g2` by hand without also setting `_noise_ema_n` leaves the fold on
sample 1, which takes the next sample whole (the one live edge, pinned in the test file).

**FIRST READING (2026-09-01, two `--debug --steps 12000 --n-steps 1024 --batch-size 128
--grad-accum-steps 2` runs, CPU, default flags so `aux` is small and `distill` absent).** It
reproduces the confound in miniature. Run A: `train/noise_scale_share_value` = **1.00001** and
`train/noise_scale` == `train/noise_scale_value` to five figures — **the "total" IS the value
term**, contributing ~100% of |G|² — with `noise_scale_ratio` = **0.081** ("over-batched 12×").
Run B (post-debiasing) put the policy tag on the board: `noise_scale_ratio_policy` = **6.2 then
2.7** ("noise-limited") on a run whose total ratio read **0.074–0.090**. Same run, same call, ~30–80×
apart, in the direction the total hides. Do NOT read those numbers as the production runs' answer
(different device, batch, flag set, and eleven calls) — read them as the instrument working: the
term PPO actually optimizes says *noise-limited* where the total says *over-batched*.

**One thing that smoke exposed, and that is now FIXED (2026-09-03).** The TOTAL's own EMA had the
same anchor-on-first-sample fragility the per-term half was debiased for — in run B the total's
`tr(Σ)` EMA started negative and `train/noise_scale{,_ratio}` therefore never emitted at all across
11 calls, while every per-term tag did. It was deferred at the time because the byte-identity of
the total series was a requirement of that work; it has since cost a reading (R5F15's
"provisional, n=2"), so the deferral was paid off with the shared `debiased_ema` above. The
revert-catcher is
`instrumented_ppo_noise_scale_terms_test.py::test_a_negative_first_sample_no_longer_suppresses_the_total_for_hundreds_of_calls`,
which reproduces exactly this failure. ⚠️ **A CONSTANT synthetic stream cannot detect this class**
— an anchored fold reads a constant correctly too — so the constant-stream test is the analytic
anchor and the outlier test is the guard.

Two levers, both ENV/constant rather than CLI flags — the probe changes no training math, so it
never belongs in `model_config.json` and should not have to survive a resume's argv:
- **`$GEN3AI_NOISE_SCALE_PER_TERM=0`** turns it off for a process (wins over the class default).
- **`_NOISE_PER_TERM_EVERY`** (`constants.py`, currently `1`) samples one `train()` call in N,
  dividing the cost directly. It slows the per-group EMA's convergence in wall-clock, never its
  value — the EMA is per SAMPLE. **Raise it on a config with few minibatches per `train()`**, which
  is the only regime where this probe is expensive.

Tests: `instrumented_ppo_noise_scale_terms_test.py` — the per-group fold recovers a planted
`B_simple` per group (with `aux` planted 4000x below `policy`, the confound itself); `share` is
pinned as NOT a partition; the sampler's `small_sq`/`big_sq` are checked against independently
computed gradients; a partial group and a group that first appears on a later micro-batch both yield
nothing; `.grad` is never written; `add` returns the identical object; a raising probe self-disables;
**the byte-identity gate** runs two identically-seeded fresh models (a `train()` on the toy is *not*
reproducible from a restored `state_dict` — three consecutive restores drift ~5e-4 with the probe
absent — so the arms are fresh, and a third OFF arm is the control); and the advisor's disagreement
family. A source scan asserts the tags in `train()` and `NOISE_TERM_GROUPS` are the same set.

## THE DOSE, and pinning a fork's step size (`--fork-lr` / `--fork-lr-freeze`, `dose.py`)

**`--lr` is INERT on a resume.** `main/train/model_build.py`'s resume path restores the checkpoint's
optimizer LR and prints `(arg --lr=… ignored on resume)` — correct for a launcher RESTART (the KL
controller should keep the rate it settled on) and wrong for a FORK, which then inherits whatever
the PARENT had annealed to. `--batch-size` and `--n-steps` are inherited the same way.

**The quantity that predicts a distillation fold's collateral is the DOSE, not the LR** (ledger M7):

```
updates_per_env_step = n_epochs / (batch_size * grad_accum_steps)
dose_rate            = lr * updates_per_env_step
```

`grad_accum_steps` is in the DENOMINATOR because K micro-batches are summed into ONE optimizer step
(see *Gradient accumulation* above), so two runs at the same `--lr` differ 8× in dose when one
accumulates 16 micro-batches and the other 2. Measured over the archive's own sidecars:

| run | eff. batch | epochs | lr median | dose_rate | vs v8 |
|---|---:|---:|---|---|---|
| `ai_v8_14_distill3_0725` | 32,768 | 7 | 1.004e-4 | **2.145e-8** | 1.00× |
| `ai_v9_59_R2ACTION_0827` (rev-2) | 4,096 | 10 | 5.814e-5 | 1.419e-7 | **6.62×** |
| `ai_v9_70_R3ACTION_0828` (rev-3) | 4,096 | 10 | 2.804e-5 | 6.845e-8 | 3.19× |
| `ai_v9_92_R5F00_0831` | 16,384 | 10 | 6.977e-5 | 4.258e-8 | 1.99× |

Three folds launched with the same `--lr` ran at three different rates, and nothing in any of them
said so — the controller's inherited state was a hidden confound in every fold comparison. Two
flags and one recorded block close that.

### `--fork-lr FLOAT` (resume-only) — and the fork-vs-restart rule

Sets the resumed model's **optimizer LR**, its **`model.lr_schedule`** and the **KL controller's
`_current_lr`** at load. All three, because each is a separate no-op risk: SB3 re-installs the
schedule's value at the top of every `train()` (so the optimizer alone would be overwritten on the
first update), and the controller's multiplicative ladder starts from wherever it thinks it is (so
seeding it from the checkpoint would walk straight back there). The pin is still clamped into
`[--min-lr, --max-lr]` — a bound the user set is a bound.

🚨 **It applies ONLY on a genuine FORK.** The launcher re-invokes the same argv every
`--restart-interval-hours` into the same run dir, so a flag that fires "on resume" fires every few
hours forever and would reset the adapted rate each time. `main/train/fork_lr.py` keys on WHERE the
resumed checkpoint lives — outside the run dir ⇒ FORK; `<run>/checkpoints/*.zip` or `<run>/*.zip`
(the legacy root layout) ⇒ RESTART. That is the predicate `run_io._resolve_fresh_model_dir` already
uses for its clobber guard and `launcher/checkpoint.resolve_fork_resume_model` uses to decide
whether a restart re-inits from the source; the launcher SWAPS `--model` to the fork's own
checkpoint once the fork has progress, so restart #2 of a fork reads RESTART for the same reason a
plain resume does. `<run>/warmstart/…` is deliberately a FORK — the consensus warm-start is an INIT
built from foreign teachers, not this run's own progress. A fresh run is REFUSED (use `--lr`).

### `--fork-lr-freeze` — a constant, recordable step size

Disables the KL adaptation **and** the two-phase cosine (`frozen` on both callbacks, plus
`freeze_at`), so the LR stays at `--fork-lr` exactly. A fold experiment wants a constant dose; an
adapting LR makes it a per-rollout variable nothing records. Unlike the pin it is a **property of
the RUN** and DOES persist across every periodic restart — re-read from the pin recorded in
`metadata.json`, or from the argv a launcher restart reproduces verbatim.

### The recorded `dose` block, and `python -m main.dose`

Every metadata write (and every checkpoint sidecar, through the one `_model_hparams` dict) carries
`dose`: `lr_now` · `lr_flag` (what `--lr` said, so the inertness is VISIBLE) · `fork_lr` ·
`lr_frozen` · `batch_size` · `grad_accum_steps` · `effective_batch` · `n_epochs` ·
`updates_per_env_step` · `dose_rate_now` · `kl_controller` {target_kl, kl_factor, lr_factor,
min_lr, max_lr, phase} · `fork_lr_pin` when one was applied. **metadata.json ONLY** — never
`model_config.json`, which is the weight-shape record `check_compatible` reads (root CLAUDE.md's
provenance rule). Live: `train/dose_rate` + `train/effective_batch` every rollout, because a groomed
run keeps no sidecars and the rate alone is ambiguous (a falling `dose_rate` is the KL controller
annealing OR an operator having raised `--grad-accum-steps`, and only the second moves the batch).

⚠️ **The `kl_controller` field is a PLAIN-DATA SNAPSHOT, never the callback.** `model.save()`
cloudpickles the model's `__dict__`, and an LR callback back-references the model and SB3's
`Logger`, which carries a `_contextvars.Context` and cannot be pickled — stashing the live object
breaks EVERY save in the run at the pre-train round-trip smoke (observed while building this, the
`_correction_buffer` hazard again). The snapshot is taken AFTER the pin so a freeze is captured.

`python -m main.dose <run>…` answers the same question for runs already on disk, from what they
already wrote down: median LR over the **checkpoint sidecars** (preferred over `snapshot_history`,
which is CAPPED at ~15 rows while sidecars keep every un-groomed checkpoint; then the run-level
`current_lr` as a single point), the shape from the SAME rows, and a ratio against a `--reference`
run (default `ai_v8_14_distill3_0725`). A run whose shape MOVED mid-flight is flagged rather than
averaged. Torch-free and model-free, so it reads a run whose architecture drifted past current code.

**Flag class: training-runtime.** Neither flag reaches the extractor, scales a loss or changes a
weight shape ⇒ no `ARCH_SIGNATURE` bump, not in `model_config.json`/`ModelVersion`, not in
`check_compatible`, and deliberately **not** in `agents/model/flag_registry.py` (whose scope is
extractor architecture toggles). They land in `metadata.json`'s `cli_args` like every train-loop
knob, and the launcher forwards them verbatim.

Tests: `src/main/fork_lr_test.py` (the discrimination rule incl. the warm-start case, the four
decisions, the freeze surviving a restart from the record AND from the argv, the three-site pin, the
clamp, the freeze holding across a KL excursion a control arm demonstrably moves on, and the three
config refusals), `src/agents/training/dose_test.py` (the arithmetic against v8's own recorded row,
the block, the pickle-safety of the snapshot), `src/main/dose_test.py` (source precedence, step
ordering, the shape-moved flag, the CLI, and that importing it pulls in no torch).

### `--adaptive-batch` — CLOSING the loop on the noise scale (`gen3_adaptive_batch_v1`)

Everything above is a **reading**. The NSR advisor printed *"raise `--grad-accum-steps` ~N×"* into
the Events panel and a human typed it on the next relaunch. `--adaptive-batch {off,total,policy}`
turns that into a controller — the second one this trainer runs, beside the KL-driven lr loop.
**OFF by default; an `off` run registers no callback at all and is byte-identical** (pinned by
`test_the_callback_cannot_change_the_ppo_update` + `test_the_flag_defaults_to_off_and_registers_no_callback`).

**THE RULE, in one paragraph.** Every rollout the controller reads the smoothed noise-scale ratio
of the chosen term and the number of EMA folds behind it. It does nothing until the EMA is warm
(20 folds — the NSR advisor's own warm-up, because a single-sample `B_simple` can SIGN-FLIP) and at
least `--adaptive-batch-every` rollouts (default 4) have passed since the last move. Then, if the
ratio has left `[target/band, target·band]` (defaults 1.0 / 2.0), K is **DOUBLED** when it is ABOVE
(noise-limited: each update is mostly sideways) and **HALVED** when BELOW (over-batched: buy update
steps instead of averaging), clamped into `[max(2, --adaptive-batch-min-accum), --adaptive-batch-max-accum]`.
An unreadable ratio, a cold EMA, a within-band reading or a clamp is a **named no-op**, reported
ONCE (a silently idle loop is indistinguishable from a broken one; a loop that says so every
rollout is noise).

**Why K and never `--batch-size`** — three independent reasons and all three matter: (1) SHAPE —
`--compile-trainer` keys graphs on shape against a `cache_size_limit` of 8, so a moving batch size
is the unbounded shape set `check_shape_stability` exists to refuse, and dropping to eager is
invisible (~1.75×); moving K leaves every forward shape byte-identical. (2) MEMORY — the activation
peak is one micro-batch, so K is the one batch lever with no VRAM cost. (3) EXACTNESS — K
micro-batches summed **is** the gradient of a `batch_size·K` batch. `check_shape_stability` takes
`n_steps`/`n_envs`/`batch_size`/`async_rollout` and *not* K, which is the proof rather than the
claim (`test_shape_stability_does_not_depend_on_k`), and a source scan fails any assignment to
`batch_size` in the controller module.

**🚨 THE FLOOR IS 2, NOT `--adaptive-batch-min-accum`.** The noise-scale estimator needs gradient
norms at TWO batch sizes and gets the second from the accumulation group, so at K=1 it emits
nothing — a loop allowed to reach K=1 would blind the signal it steers by and could never climb
back out. The requested floor is raised to 2 and the raise is ANNOUNCED at startup.

**`policy` is the mode to use.** It steers by `train/noise_scale_ratio_policy`; `total` steers by
the legacy scalar. The section above is the whole argument: the total is ~100% the value term plus
a dozen dense supervised aux heads and reads "over-batched" on runs whose policy term reads
"noise-limited", so sizing on the total shrinks the batch the policy gradient needed. `policy`
REQUIRES the per-term probe, and `$GEN3AI_NOISE_SCALE_PER_TERM=0` alongside it is a
`parser.error` rather than a loop that silently never reads anything.

**⚠️ WHY THE STEP IS 2× AND THE BAND MUST BE ≥ √2 — measured, and it corrected the design.** K is
the ratio's denominator, so a move changes the reading INSTANTLY and exactly: doubling K halves the
ratio. A correction therefore crosses to the *other* side of the band only when `target·band < ratio`
and `ratio/2 < target/band` can both hold, i.e. iff **`band² < 2`**. The first draft of this
documented the boundary as 2.0; running the test found `band=1.5` settling cleanly and the algebra
put the real boundary at **√2 ≈ 1.4142** — pinned by a parametrized test straddling it (1.30 and
1.41 chatter forever; 1.50 and 2.0 settle). The overshoot window `(target·band, 2·target/band)` is
only 0.6% wide at 1.41, so the test starts *just* outside the band on purpose: a start further out
takes several one-directional moves and lands in band, which is progress, not chatter, and would
have passed for the wrong reason. Default 2.0 sits comfortably above; a narrower band is for a
smoke that WANTS movement in a handful of rollouts.

**THE TWO-CONTROLLER INTERACTION — read this before tuning either.** The KL lr controller and this
one are COUPLED through the update: at a fixed `target_kl`, a larger K means each optimizer step
consumes more data, so per-step KL falls, so the lr controller RAISES lr. **That is intended** —
the batch loop fixes an update's signal-to-NOISE, the lr loop fixes its STEP SIZE — but it means
the effective **dose is a product of two controllers**, and the scalar to watch is `train/dose_rate`,
never either loop's own series. Two controllers chasing each other on one timescale is the classic
oscillation, and they are separated by their SIGNALS rather than merely their cadences: the lr loop
reads a KL EMA at `α=0.20` (half-life ~3 rollouts) and this one reads the noise-scale EMA at decay
`0.99` (a several-hundred-call window) — ~30–100× slower-moving by construction. `--adaptive-batch-every`
(4) is the second-order guard on top of that, and the lr loop's own 7-rollout post-move cooldown
means the fast loop has re-settled before the slow one looks again.

**PERSISTENCE is free and deliberately so.** `_model_hparams` already writes `grad_accum_steps`
into every checkpoint sidecar, straight off the model attribute the callback owns — so a moved K is
persisted by the EXISTING checkpointer with **no new key and no edit to the checkpoint path**.
`build_callbacks` reads it back with `read_checkpoint_metadata` (the same sidecar `handoff_lr`
rides in) and hands it to the callback as `resume_accum`, which installs it in
`_on_training_start` — i.e. AFTER `model_build` applied the CLI `--grad-accum-steps`, so the
controller's own history wins over the launch argv. A restart that TIGHTENS `--adaptive-batch-max-accum`
re-clamps rather than reinstating the old K. **The noise-scale EMA itself is process-local and does
NOT persist**, so after a restart the loop re-warms (20 folds) before it may move again — the right
behaviour, not a gap: a cold EMA is not a reading.

**The three series to read**: `train/grad_accum_steps` (K in force for the `train()` that follows),
`train/effective_batch` (`batch_size·K`), `train/adaptive_batch_ratio_used` (the exact number each
decision was made on). Every move also emits one Events line naming the ratio, the direction, the
old and new effective batch, and that the dose moved.

**⚠️ Like the two `--compile-*` flags, it is NOT inherited on a flagless resume** — a bare
`--model … --steps …` gets `off`. The launcher forwards its own recorded argv verbatim, so a
launcher-managed run keeps it across every periodic and crash restart; a hand-typed resume must
re-type it.

**SMOKE (2026-09-01, CPU `--debug --steps 34000 --n-steps 1024 --batch-size 128 --grad-accum-steps 2
--adaptive-batch policy --adaptive-batch-every 1 --adaptive-batch-band 1.5`).** Read back from
`tb/`: `train/grad_accum_steps` held at **2 for the first 20 rollouts** (the warm-up refusing to
act — and `noise_scale_ratio_policy` read 0.09–0.99 through that window, so the guard is what kept
a cold EMA from driving K DOWN), then **2 → 4 → 8 → 16 → 32** on readings of **432 → 24.1 → 6.76 →
3.38**, every one above the band's 1.5, before parking at the `max_accum` clamp with the
report-once line. Do not read those magnitudes as a production answer — a 128-row micro-batch on a
toy is not the production shape — read them as the loop tracking the series it is supposed to.

Tests: `adaptive_batch_callback_test.py` — the pure controller (planted ratio sequences → K
trajectory, the walked-feedback hysteresis, the √2 boundary, cadence, both clamps, the floor,
every unreadable-ratio form, constructor validation); the read seam returning exactly the recorded
value under the same emit gate; the callback mutating `grad_accum_steps` and nothing else
(behavioural + source scan + the `check_shape_stability` signature); report-once; the sidecar
round-trip through real `record_checkpoint`/`read_checkpoint_metadata`; and the byte-identity gate
(two identically-seeded fresh models, one with the callback attached and not moving K).

