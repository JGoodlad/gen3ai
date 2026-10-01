# The learner's GIGO gates (K9) — the learner golden, behaviour-policy consistency, fail-closed non-finite

M5 Lane K item K9 (`designs/endstate/program_rust_core.md`, the Lane K row and its Decision record).
Three gates on the PPO learner, each aimed at a failure that was INVISIBLE before it: a refactor that
silently changes what an update computes (a), a rollout played by different weights or a different
function than the learner recomputes (b), and a NaN / Inf that silently poisons the weights (c).
Code: `src/agents/training/learner_golden.py`, `src/agents/training/instrumented_ppo/learner_gates.py`.

## (a) The LEARNER GOLDEN (`learner_golden.py`, `learner_golden_test.py`)

**What is pinned.** A production-surface learner REBUILT from a fixed seed (`testkit.fresh_model` —
the production policy kwargs, seed 0, a seeded perturbation so the pointer head is not exactly uniform;
then `main.train.model_build.apply_training_hparams` with `production_args()`, exactly what a launch
applies) runs ONE eager fp32 `InstrumentedMaskablePPO.train()` on a committed buffer:

| input | pinned as |
|---|---|
| the "checkpoint" | NOT committed (3.07M params = 12 MB) — rebuilt from the seeds; its bytes are hashed (`init_params_sha256`) so a construction/seeding change is told apart from an update change |
| the rollout buffer | `learner_golden_buffer.npz` (65 KB): 16 steps x 4 envs = 64 REAL rows of a complete-game rollout on the Rust collector at the production surface — every observation key and label (win targets, belief / item / spread / HP-type labels, intent labels), actions, masks, rewards, episode starts, values, the behaviour log-probs, advantages, returns. Its sha256 is in the golden |
| the recipe | `learner_golden.RECIPE`: the SB3 knobs the production mirror does not carry yet, from `design_learner_recipe.md` §2's live column (clip 0.15, entropy 0.05, lr 2.8e-5, grad clip 0.5, per-micro-batch advantage normalisation), at a scaled-down shape that keeps the live shape's structure: micro 16 x K 3 over 64 rows = one full accumulation group + a RAGGED short group per epoch, 2 epochs |

**What is hashed, and why exact bytes.** `post_params_sha256` = sha256 over every parameter's float32
bytes in `named_parameters()` order (names included); `group_sha256` = the same per parameter group
(first two name components) so a failure says WHERE; `losses` = every logged loss / objective scalar
plus `train/approx_kl`, `train/clip_fraction`, `train/explained_variance`, stored as exact float64 and
compared exactly, so a failure says WHICH term. Diagnostics-only tags (grad-balance shares, ranks, noise
scale, calibration) are NOT compared — a telemetry-only change must not move the golden; any change
that reaches the weights moves the parameter hash anyway. Rounded values were rejected: with 3M values
some always sit on a rounding boundary, so rounding buys no portability and loses teeth.

**Determinism and the torch key.** CPU, eager, fp32, `torch.set_num_threads(1)` for the update (a CPU
reduction order can depend on the thread count), numpy + torch seeded before `train()`. Two processes
with different `PYTHONHASHSEED` reproduce it bit for bit. **Across torch builds the INIT is identical but
the update is not** (measured 2026-09-30: `gen3ai_torch28` vs `gen3ai_stable` — every pinned loss within
1.2e-7, every parameter group's bytes different), so the golden is KEYED BY `torch.__version__`
(`2.5.1+cu121`, `2.8.0+cu126` today). A build with no entry FAILS; it never skips and never records.

**Cost:** ~3 s (build ~1 s + one update), unmarked (routine tier).

**Teeth (measured 2026-09-30):** changing the advantage-normalisation epsilon `1e-8 → 1e-6`, or the
win-prob critic's coefficient by x1.001, each fails it naming 13+ losses and every parameter group;
reverting passes. `learner_golden_test` keeps one such perturbation (advantage normalisation off) as a
standing test. Every other K9 change below was verified byte-neutral by it under both torch builds.

**Re-recording is EXPLICIT** — the test never writes:

    python -m agents.training.learner_golden record --reason "why the update is MEANT to change"
    python -m agents.training.learner_golden check           # what the test does; prints the diff
    python -m agents.training.learner_golden rebuild-buffer --reason "..."   # obs layout / label schema changed

`record` writes this torch build's entry (keeping the others) with the reason, date, commit and
versions, and appends an append-only `history` row carrying the diff it replaced. Run it under EVERY
interpreter that has an entry. `rebuild-buffer` (needs the `rust_env` self-check build) records a new
buffer and invalidates every entry (they were computed on different input).

**Scope limits.** Not pinned: the rollout and GAE (the buffer's advantages / returns are inputs), the
KL→LR controller and every other callback (outside `train()`), CUDA / compiled / TF32 numerics (the
compile parity gate and K6's canary own those). When K10(a)'s recipe block lands, `RECIPE` should read
it and the golden be re-recorded.

## (b) Behaviour-policy consistency on BOTH env cores (`--behaviour-check`, default `fatal`)

Before any optimizer step of every update, the learner's recomputed log π(a|s) must equal the stored
behaviour log-prob: max |Δ| < 1e-4, else `BehaviourMismatch` (`rust_rollout/consistency.py` — one type,
one bar). WHICH implementation runs is decided once per `train()` by `learner_gates.behaviour_gate_mode`:

* **the buffer carries per-row policy versions** (`--env-core rust`): Lane G's pre-loop probe — its own
  forward on one micro-batch while the buffer is still `[n_steps, n_envs]`, so current rows are held to
  the bar and older rows are age-bucketed into `staleness/*` (`rust_collector.md`);
* **it does not** (`--env-core python`: every row was played by the weights the learner holds): the
  IN-LOOP gate — the FIRST micro-batch of epoch 0's own `evaluate_actions` output against
  `rollout_data.old_log_prob`, before `values.flatten()`. No second forward; one host read per update.
  Never both.

It logs `behaviour/max_abs_dlogp_current`, `behaviour/p99_abs_dlogp_current`, `rows_current`,
`rows_probed` and one `behaviour/bar_<statistic>` per condition either way.

**The gate is KEYED BY the float32 matmul precision the run uses** (`consistency.BEHAVIOUR_GATES` — ONE
table, read by Lane G's probe and by the in-loop gate through `judge_behaviour`; an undeclared precision
is a typed `UndeclaredPrecision`). Every condition must hold on the micro-batch's current rows:

| precision | condition | catches | bar | FATAL when |
|---|---|---|---|---|
| `highest` (fp32, the default) | max \|Δ\| | every fault class | 1e-4 | the first violation |
| `high` (TF32) | p99 \|Δ\| | GLOBAL faults (every row moves: stale weights, a mode / sampling / temperature mismatch) | 3.6e-3 = 3 x the healthy per-row p99.9, rounded up | the first violation |
| `high` (TF32) | max \|Δ\| | LOCALIZED gross faults (< 1 % of rows: misaligned rows, a wrong action index, a mask mismatch) | 0.071 = 1.75 x the healthy per-row max, rounded up | **4 CONSECUTIVE updates** (`TF32_MAX_PERSISTENCE`); a single violation is a loud warning |

**Every violation** prints `🚨` (or `🛑` when fatal) with the condition, its value, its bar and the
streak, logs the offending rows (index, |Δ|, action, mask — the largest 10; the p99 for context), and
appends a JSON line with the largest 200 to `<run_dir>/behaviour_violations.jsonl`
(`consistency.VIOLATION_DUMP`). The streaks live on the model (`_behaviour_streaks`, per precision and
statistic); a clean update resets a condition's streak; an update with no current rows (Rust path)
leaves every streak standing; a NaN |Δ| is FATAL at once (never rounding); `--behaviour-check warn`
only ever warns. `behaviour/streak_<statistic>` is on the dashboard every update.

**Why the TF32 max is PERSISTENT, and why k = 4 (orchestrator 2026-09-30; k from data).** Its 1.8x
headroom over the first pass's healthy max (0.040) came from a small sample, and the max of a noise
statistic grows with the number of draws. A real localized fault is systematic and recurs every update
(every corrupted micro-batch was caught — the table above); a rounding outlier is a fresh draw from
fresh rows. The EXTREME-VALUE read (`tail_result.json`, from `tail.py` + `tail_derive.py`: 96 fresh seeds,
589,824 healthy TF32 rows, 288 independent 2,048-row micro-batches — the per-update statistic):

| healthy TF32 max per update (one 2,048-row micro-batch) | value |
|---|---|
| empirical: n / median / q90 / q99 / largest | 288 / 7.2e-3 / 0.021 / 0.043 / **0.063** (none over 0.071) |
| per-row: p99.9 / p99.99 / max (n = 589,824) | 1.2e-3 / 0.017 / 0.063 |
| peaks over threshold (rows above their p99.9; generalized Pareto) | shape ξ = **0.40 — a HEAVY tail** |
| P(an update's max > 0.071) — POT [95 % seed bootstrap] / GEV on the 288 maxima | **1.1 %** [0.28 %, 2.2 %] / 3.3 % |
| expected max over 10,000 updates (the 1/10k return level) — POT / GEV | 0.51 / 7.8 (the GEV extrapolates a shape fitted on 288 blocks) |
| P(≥ 1 crossing in 10,000 updates) | ~1 (every model) |
| **P(false FATAL in 10,000 updates) at k = 2 / 3 / 4 consecutive** — POT | 0.72 / 0.014 / **1.6e-4** |
| … at the POT bootstrap's upper end | 0.99 / 0.11 / **2.5e-3** |
| … GEV | ~1 / 0.31 / 0.012 |

The DECLARED criterion: the smallest k whose 10k-update false-FATAL rate is < 1 % at the POT bootstrap's
upper end ⇒ **k = 4**. The default 2 would kill most 10k-update runs for nothing. The cost: a real
localized fault is FATAL on its 4th update instead of its 2nd (it warns loudly from its 1st). The
single-shot p99 is safe on the same rows: its largest of 288 is 7.8e-4 (4.6x under 3.6e-3) and a GEV fit
gives ~0 over 10k updates. ASSUMED: rows i.i.d. within a micro-batch, updates independent (each scores
fresh rollout rows), a fresh perturbed learner standing in for a trained one — **UNVERIFIED** on a
trained checkpoint and on a live run, where recurring states could correlate consecutive crossings.


**The measurement** — `designs/research_state/measurements/k9_behaviour_bar_2026-09-30/`: `result.json`
(`measure.py`, `derive.py`: the healthy distribution and the stale faults) and `corrupt_result.json`
(`corrupt.py`: the localized faults) and `tail_result.json` (`tail.py`, `tail_derive.py`: the healthy TF32
max's extreme-value read, below). RTX 3080 Ti, torch 2.5.1. Rows: per seed a real complete-game
Rust-collector rollout at the production surface (48 envs x 128 steps = 6,144 rows); the learner is the
learner golden's production learner with a per-seed perturbation; 4 seeds x 3 REAL updates each, eager
and `--compile-trainer`. The rollout side is the python core's (eval mode, no grad, 48-row batches), the
learner side a 2,048-row micro-batch (train mode, grad). Per precision: n = 147,456 rows, 72
micro-batches (stale faults 72 / 48; localized faults: 4 seeds x 2 states x 3 micro-batches, eager).

**THE TABLE** — the statistic the gate reads over a 2,048-row micro-batch (per-row figures where noted),
healthy and under each fault class; margin = the fault's SMALLEST micro-batch value / the bar,
headroom = the bar / the healthy LARGEST. TF32 is CUDA `high`; fp32 is CUDA `highest` (healthy, stale)
or CPU fp32 (localized: the fault SIGNAL — the CUDA fp32 localized units were not run).

| | fp32: max (bar 1e-4) | TF32: p99 (bar 3.6e-3) | TF32: max (bar 0.071) |
|---|---|---|---|
| healthy per-row p99.9 / max (n = 147,456 rows) | 9.5e-7 / 1.9e-6 | 1.17e-3 / 4.0e-2 | 1.17e-3 / 4.0e-2 |
| **healthy micro-batch, largest of 72** | 1.9e-6 — **headroom 52x** | 8.5e-4 — **headroom 4.2x** | 4.0e-2 — **headroom 1.8x** |
| healthy, eval/train MODE alone (per-row max) | 1.8e-6 | 4.4e-3 compiled / 8.1e-4 eager (per row) | — |
| GLOBAL: rollout ONE optimizer step stale, lr 2.8e-5 (72) | 0.020 — **200x** | 0.0112 — **3.1x** | 0.022 — 0.31x (caught 31 %) |
| GLOBAL: one optimizer step stale, lr 3e-4 (72) | 0.16 — 1,600x | 0.106 — 30x | 0.166 — 2.3x |
| GLOBAL: one full update stale (48) | 0.108 — 1,085x | 0.074 — 21x | 0.109 — 1.5x |
| GLOBAL: eval/train-mode mismatch | VACUOUS (below) | VACUOUS | VACUOUS |
| LOCALIZED, 0.5 % of rows: wrong action index (24) | 0.244 — 2,440x | 9.9e-4 — **NOT caught (by design)** | 0.239 — **3.4x** |
| LOCALIZED, 0.5 %: obs swap / rows misaligned (24) | 0.549 — 5,490x | 1.05e-3 — NOT caught | 0.549 — **7.7x** (82 of 256 rows illegal: 1e8) |
| LOCALIZED, 0.5 %: mask mismatch (24) | 0.457 — 4,570x | 2.5e-3 — NOT caught | 0.457 — **6.4x** |

Every GLOBAL and every LOCALIZED fault micro-batch is caught by the gate at its precision (TF32: the
global ones by the p99 — FATAL at once — the localized ones by the max — FATAL on the 4th consecutive
update); no healthy micro-batch trips any condition. The
localized rows' OWN |Δ| (TF32): wrong action index median 0.094, **42 % of them below 0.071**, some
exactly 0 (a tie, or a row with one legal action — nothing to swap to); obs swap median 0.51, 13 % below;
mask mismatch median 0.42, none below. A micro-batch is caught because its LARGEST corrupted row clears
the bar — with 10 corrupted rows, all ten below 0.071 is ~0.42^10 ≈ 2e-4 for a wrong action index.
These are a FRESH (near-uniform) learner's signals; a trained, sharper policy should separate wrong
actions more (**UNVERIFIED** on a trained checkpoint).

**Why TWO statistics under TF32 — the OVERLAP, reported.** Under TF32 the healthy per-row tail (max
0.040, a few numerically sensitive rows, batch-shape driven) sits ABOVE the smallest one-step-stale
fault (micro-batch max 0.022), so no max bar separates a GLOBAL fault from TF32 noise: the max alone at
0.071 catches only **31 %** of one-step-stale micro-batches at the recipe lr. A stale or mismatched
policy moves EVERY row; TF32 noise moves a few — the micro-batch's p99 separates them (4.2x healthy
headroom, 3.1x fault margin). The p99 is blind BY CONSTRUCTION to a fault on < 1 % of the rows (≤ 20 of
2,048), which the max catches when it is gross. At fp32 the max alone does both (52x headroom).

⚠️ **The residual blind spot under TF32:** a SUBTLE fault — smaller than the TF32 noise floor's tail
(|Δ| < 0.071) — confined to FEWER than 1 % of a micro-batch's rows passes both conditions; a gross one
is FATAL on its 4th consecutive update, not its 1st. At fp32 the
max (bar 1e-4) still sees it. ⚠️ An earlier pass (1,024-row micro-batches, the same design) saw one fp32
healthy row at 2.6e-5 — 3.9x under 1e-4, the thinnest fp32 headroom observed.

**The eval/train-MODE fault is VACUOUS for this policy.** It has no dropout and no batch norm (checked
on the built module tree), so eval and train mode compute the same FUNCTION; the only mode-dependent
code is kernel choice (`nn.TransformerEncoderLayer`'s eval fast path, gradient checkpointing, the
`is_grad_enabled` branches that stash training targets). That component is measured above ("mode
ALONE") and is inside the healthy distribution the bar is set from. No real mode fault can be
constructed; a future dropout / batch-norm layer would be one, and this gate is what would see it.

**Measured on the python path at fp32 (CPU):** the `--debug` smoke (2,048-row micro-batches): |Δ| = 0
on update 1 (a fresh pointer head is uniform) and 2.38e-7 after; the learner golden's buffer (Rust T2
eager log-probs vs the learner): 3.6e-7. Lane G's GPU read on the Rust path (compiled learner vs T2
graph): 1.45e-5.

The four paused learner-battery argvs (L95, T32_STANDIN, Cfix, T32b_sentinel — two of them TF32, none
typing `--behaviour-check`) parse unchanged and resolve to `fatal` at their precision's bar
(`learner_gates_test`).

**The other legitimate reasons it could differ, checked:** batch composition — the rollout forwards
`n_envs` rows, the learner a micro-batch (the healthy rows above); `--compile-trainer` — measured in
both arms above; grad-checkpointing — the recompute is bit-exact
(dropout 0, `use_reentrant=False`); fork-arm branch rows — their log-probs come from the parent's live
policy (`fork_buffer.build_branch_rows`), i.e. the behaviour policy; distill rows — the trainee's own
log-probs; `_align_opp_intent_labels` — rewrites label keys only, never a policy input.
**UNVERIFIED:** a real CUDA python-core launch (no GPU training during the M5 halt) — the CUDA numbers
are the production learner's forwards on real rollout rows, not a live rollout loop.

## (c) Every non-finite loss / gradient path is FAIL-CLOSED (`NonFiniteLearnerError`)

Before K9(c) nothing in `train()` looked: a NaN loss back-propagated NaN gradients, `clip_grad_norm_`
(`error_if_nonfinite=False`) scaled every gradient by `max_norm / NaN` in place, and `optimizer.step()`
wrote NaN into every parameter and into Adam's moments — silently. The gates, all BEFORE the optimizer
can move anything (so a crash-save holds the last finite weights):

| gate | where | catches |
|---|---|---|
| `check_buffer_finite` | once per update, right after the intent-label alignment — BEFORE PopArt's advance (which rewrites `value_net` outside the optimizer) and before any forward | a NaN/Inf reward, value, behaviour log-prob, advantage, return, or FLOAT label key (the flat `observation` is not scanned: an input whose NaN reaches the loss, and a full scan is ~0.3 s at production size) |
| `check_loss_finite` | once per micro-batch, on the assembled loss, before the grad-balance / noise probes and the backward. Since K8 the region R1's `isfinite(loss)` rides the micro-batch's ONE host read (`micro_step.pack`); the full check (its own read) runs when the eager tail folded a term onto R1's loss, or to NAME the term(s) on a failure — same timing, same message | any term; NAMES the non-finite term(s) (policy, entropy, value, and every `aux_probe_terms` entry) |
| `check_kl_finite` | per micro-batch, on sb3's own approx-KL host read | an Inf KL under a FINITE loss (an overflowed ratio on a positive-advantage row takes the clipped branch) |
| `clip_grad_norm_checked` | every optimizer step (the in-loop step and the accumulation flush) | a NaN/Inf gradient from a finite loss; `error_if_nonfinite=True` raises BEFORE the in-place scaling, so the named parameters are the ones the backward poisoned |

### The audit (2026-09-30) — every site, its verdict, and what was done

**Absorbed BEFORE the total (a total-loss check could not see these) — FIXED:**

| site | was | fix |
|---|---|---|
| `ppo.py` β reachability `_reach = isfinite(target logit)` | dropped a NaN target logit's row like a deliberate −inf one (weighted path: the row left the graph) | `~isneginf(...)`: only −inf is unreachable; a NaN stays supervised and reaches the loss check. Byte-identical otherwise (golden) |
| `opp_intent.set_valued_switch_loss` `avail = believed & isfinite` | a row whose only believed slots were NaN was dropped | `~isneginf`, same reasoning |
| `ppo.py` advantage normalisation | a ONE-row final micro-batch (python core, n_steps·n_envs ≡ 1 mod the micro-batch; the Rust collector's target is a multiple of it) had std() = NaN → NaN into every weight, silently | stock SB3 PPO's `numel() > 1` guard (sb3_contrib's MaskablePPO lacks it) |
| `model/popart.py` update (in `_train_probe_setup`) | NaN returns poisoned μ/σ and rewrote `value_net` before any loss ran | `check_buffer_finite` runs before it |
| `rust_rollout/collector.py` `won = reward > 0` | a NaN reward became a finite LOSS label | the NaN reward is refused at the buffer check |
| `distill_grad_project.py` (`grad_project` mode) | a NaN constraint gradient made `removed_sq > 0` False: the projection was SKIPPED silently, loss and `.grad` finite | typed FATAL on a non-finite `g_sq` / `removed_sq`, re-raised past the projector's broad `except` |
| `adaptive_lr_callback.py` (both controllers) | a NaN KL froze `_kl_ema` (and the LR) for the rest of the run; an Inf walked the LR to `min_lr` | typed FATAL (the train-side `check_kl_finite` is the first line) |
| `win_prob_rollout.py` rollout labels | a non-finite continuation label was skipped (the row kept its terminal label) | `None` still skips; a non-finite NUMBER is a typed FATAL |

**Reach the total — covered by the loss / gradient checks:** the clipped surrogate, the value MSE (tail /
PopArt / clipped), entropy; the win-prob BCE and dense aux (multiplicative masks: NaN x 0 = NaN); the
distill family, the anchor, TD-aux, the cf binomial / beta-binomial / shadow losses; the belief bank
(index selection: a selected NaN propagates); the switch-branch / intent-conditional weights
(`clamp` keeps NaN); SB3's and the Rust collector's GAE (no checks of their own — the buffer check
covers their output); the noise-scale EMA (reads the same `.grad`); the distill stop / dual-ascent
controllers (their meters come from loss terms); a NaN INPUT row (torch's `Categorical` argument
validation raises before any loss — untyped, pinned by `learner_gates_test`).

**LEGIT masks:** `damage_op_blocks.py` `where(isfinite(cheapest))` (every candidate is a constant or
the +inf "no cure path" sentinel, no learned value); sb3_contrib's `where(mask, logits, -1e8)` (illegal
slots, zero gradient); `fork_arm.py` gap `isfinite` (selection of which states fork); the NaN
"not terminal" sentinels of `win_prob_callback` / `dense_aux_callback`; integer label keys (int64 —
a NaN cannot be stored).

**DETACHED METERS that may fail open by design — the ride-along heads** (`instrumented_ppo/ridealong_terms.py`,
`gen3_ridealong_heads_v1`, OFF in production): a non-finite ride-along loss or gradient is never stepped
and DISABLES the heads for the rest of the process — LOUDLY (a `🛑 [RIDE-ALONG]` line with the grad norm
and every loss) and on the dashboard (`ridealong/disabled` = 1, recorded every update) — never the run.
Legitimate, verified: the heads' inputs are `.detach()`ed stashes, they have their OWN optimizer (no
PPO param group) and their gradients are set back to `None` before PPO's loss exists, so they never
reach a PPO gradient, the policy, V or the LR (`ridealong_heads_test`: no ride-along loss reaches a
trunk / policy / V parameter; `ridealong_update_test`: one update ON vs OFF is bit-identical in params,
PPO optimizer state, PPO scalars and RNG; no callback or controller reads a `ridealong/*` tag). K9(c)'s
own checks are unaffected: the heads' cleared gradients are invisible to `clip_grad_norm_checked`, and
a NaN in the stashes they read is also in PPO's loss, where `check_loss_finite` fires.

**TELEMETRY only (never a gradient / weight / LR):** `calibration.py`, `signal_metrics.py`,
`scaffolding.py`, `value_terms._masked_auc`, the NaN-omitting metric folds in `ppo.py`, the fork-arm
rate functions, `win_prob_rollout`'s rate metrics, the `try/except` telemetry blocks in
`noise_scale*`, `rollout_probes`, `capacity_terms`, `cf_terms`; `consistency.py`'s own bar (`not worst
< BAR` also fails on NaN).

**OPEN — off in production, recorded rather than changed:** `cf_label_buffer.py` rejects non-finite
external labels at ingest and COUNTS them (`cf/labels_skipped_total`) — whether an external producer's
bad row is fatal is a policy call; `teacher/produce.py` + `teacher/callback.py` encode a NaN π′ target
as the "no target" sentinel; `teacher/winprob_oneply.py` treats a NaN win-prob read as "not contested";
`distill_terms.py` search-teacher AWR weight `exp(adv/β).clamp(max)` clamps a +Inf confirmed advantage;
`value_terms._value_dist_loss` absorbs a ±Inf return into its edge bin (the buffer check now refuses
one); `keyed_draw.py` reports a NaN logit row as "no legal action" (fail-closed, misleading message);
`--target-kl nan` parses (no finiteness validation in the parser).

**The exit side (cutover-prep, `743008c1`).** Every K9(c) raise is `main.exit_codes.NonFiniteLearnerError`
(a `FloatingPointError`; `learner_gates.nonfinite(msg)` builds it) with its message tagged
`[Learner] FATAL`. The trainer's fail-fast handlers map it through `exit_code_for` to
`TrainExitCode.FATAL_NONFINITE` (4) and the launcher STOPS on 4 instead of resuming the checkpoint that
produced it — a restart would replay the same update. K9 defines no class of its own.
