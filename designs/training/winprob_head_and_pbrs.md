# Training — the win-probability head and its PBRS routes

Moved verbatim out of `src/agents/training/CLAUDE.md` on **2026-09-08**, in the second pass of the
topic split (that leaf was 3,183 lines / 264 KB, loaded in full for any session touching the
training package). Each section below is unchanged, including its dated measurements.

`src/agents/training/CLAUDE.md` keeps the heading and a short summary of each, and points here.
**This file is the owner of the detail.**

---

## Win-probability head (`--win-prob-mode` / `--win-prob-coef`)

The training half of the tri-state win-probability head (model side: `src/agents/model/CLAUDE.md` →
win-probability head, v22). A calibrated **P(win|state)** the shaped critic can't give — supervised by the
Monte-Carlo episode OUTCOME. Off by default (`--win-prob-mode none`). Three pieces live here:

- **The label is a FUTURE quantity** — the outcome is only known when the battle ends, so (unlike the
  per-step belief labels, which are privileged info known *each* step) it CANNOT ride as a real per-step
  obs key. The plumbing reuses the obs-dict-label STORAGE path with post-hoc population:
  - **`gen3_env.py`** declares two TRAINING-ONLY obs keys when `emit_win_target` (`--win-prob-mode != none`):
    `win_target` [1] + `win_mask` [1] (float32), and emits PLACEHOLDER zeros each step (`_merge_training_keys`).
    The rollout buffer therefore stores + shuffles them automatically (the belief-label path). Read ONLY by
    the loss; the model forward reads only `obs["observation"]`, so the OUTCOME can't leak.
  - **`MaskableAgentWrapper.step` (`wrappers.py`)** sets `info["win_outcome"]` (1.0 win / 0.0 loss-or-tie,
    from `battle1.won`) at the done step (before the VecEnv auto-resets).
  - **`WinProbLabelCallback` (`win_prob_callback.py`)** captures each terminal outcome during collection
    (SYNC: in `_on_step` at `rollout_buffer.pos`; ASYNC: the `collect_rollouts_async` collector records it
    inline at the env's just-written `(t, i)` buffer row — it owns the row, the wave-batched `on_step`
    can't recover it), into a shared `model._win_terminal_scratch` [n_steps, n_envs]. At `_on_rollout_end`
    (before `train()`) it propagates each episode's outcome BACKWARD to all its steps (γ_win = 1, undiscounted
    → P(win|s) = "probability this state leads to a win") and OVERWRITES the buffer's `win_target`/`win_mask`
    placeholders. The trailing IN-PROGRESS episode (no terminal yet in-buffer) gets `win_mask=0` and is
    excluded — never trained toward a fabricated label. Only added to the callback list when the head is on.
- **Loss (`instrumented_ppo.py` `_win_prob_loss`).** `train()` reads `last_win_prob_logits` (stashed by the
  `evaluate_actions` forward) + `rollout_data.observations["win_target"]`/`["win_mask"]`, folds
  `win_prob_coef · masked-BCE`. read_only vs shaping differ ONLY in whether the extractor stop-grads the
  head's input (the trunk gradient) — the loss term itself is identical. Folded whenever the extractor's
  `win_prob_mode != none` AND `win_prob_coef != 0`.
- **Metrics (`win_prob/*` — its OWN TB prefix, not `train/`, matching the `grad/`/`popart/`/`eval/`
  groups).** Calibration: `acc` (top-1 win/loss) + `brier` (lower = predicted P(win) tracks the win
  rate); `pred_mean` vs `label_mean` (base-rate-collapse watch); `coverage` (fraction with a known label);
  `loss`. **Information value (the aggregate Brier hides it — a blowout's P(win) is trivially recoverable
  from material):** `brier_contested`/`acc_contested` restrict to CLOSE games (`|win_margin| <
  _WIN_CONTESTED_TAU`=0.25, the normalized material margin from `_compute_phi_mat`, emitted as the
  `win_margin` obs key) — judge `brier_contested` vs a 50/50 game's ~0.25 no-skill floor;
  `contested_frac`/`contested_label_mean` (≈0.5 confirms even); and **`skill_vs_material`** = the Brier
  skill score vs a material-only baseline (`P_mat = clip(0.5+0.5·margin)`) — **>0 ⇒ the head adds info
  beyond counting mons** (the headline value number; `brier_material` is the baseline for context). The
  shared-trunk pull rides `grad/win_prob_share` (the `grad_balance_metrics(aux_terms=…)` `"win_prob"` entry) — **≈0 under
  read_only** (stop-grad, the live confirmation the diagnostic isn't perturbing the policy), real under
  shaping (watch it sit small; a spike with a degrading policy → lower `--win-prob-coef`).
- **Versioning.** `win_prob_mode` (str) is the structural + resume-IMMUTABLE toggle (any change FATALs;
  threaded into `current_model_version` / `arch_toggles_from_model` so a win-prob-ON self-play run doesn't
  FATAL on its own sentinels); `win_prob_coef` is training-only, **read back on a flagless resume**.
- **Forensic trace + prober.** `RLPlayer._win_prob` (`inference/player.py`) reads the stashed
  `last_win_prob_logits` at trace-capture time (sigmoid ⇒ P(win)) into the per-decision `state`, which
  `BattleRecorder.states_arrays` writes as a `win_probs` npz array (NaN = no head / not captured, parallel
  to `values`). The prober renders **P(win) + ΔP(win)** in the Summary + Outcome panels beside CRITIC's
  V/ΔV — "how a move moved the win odds" — model-free from that array (`engine.WinProbView`); `None`/absent
  on a non-win-prob run. See `src/main/prober/CLAUDE.md`.
- **Tests.** Unit: `agents/training/win_prob_test.py` (loss masking + None guards + the callback MC-fill
  backward-propagation + in-progress masking + sync-capture-at-pos + async-skip), `agents/model/
  win_prob_head_test.py` (module build, off byte-identical projection dims, the read_only-stop-grad /
  shaping-flows gradient gating, the v22 version gate). End-to-end `--debug --use-bridge=node
  --win-prob-mode read_only` smoke confirms the roundtrip + `train/win_prob_*` metrics + `win_prob_share`=0.

🚨 **`--win-prob-mode shaping` carries NO behavioral force, and the word has misled readers.** It is
**REPRESENTATION** shaping: the BCE gradient reaches the shared trunk, so outcome-predictive features
get a subsidy there. There is no gradient path anywhere from *predicting wins* to *choosing winning
actions* — the logit is a SIDE readout, never concatenated into pi/vf (leak-safety, since its label is
the privileged future outcome), so the policy is free to ignore the subsidised features and V compresses
to its own target regardless. **The head is a BAROMETER, not a coach.** It is also self-referential: its
labels are outcomes under the CURRENT policy, so a habitual whiff that still wins 55% teaches it "55%",
never "the whiff was the mistake". Action-level badness needs a counterfactual contrast the state label
structurally lacks. That is why "shaping has been live for generations and the bait loops persist" was
never a dose mystery — the live mode was never pointed at behavior. The routes that ARE pointed at it
are below and in `designs/ai_v12/design_winprob_behavior_coupling.md`.

## Win-prob PBRS reward shaping (`--win-prob-pbrs-coef`, `winprob_pbrs.py`, ai_v12 route 1)

**OFF by default (`0.0`) and byte-identical when off** — the module is not even imported. Design:
[`designs/ai_v12/design_winprob_behavior_coupling.md`](../../../designs/ai_v12/design_winprob_behavior_coupling.md).
**Nothing has run this; no arm is registered.**

The reward-level route that gives the barometer force. With `φ(s) = σ(win-prob logit)`, DETACHED:

```
r'(s, a, s')  =  r(s, a, s')  +  coef · ( γ·φ(s') − φ(s) )
```

A move that drops the model's own win probability now costs literal reward, and the drop flows through
GAE → advantage → policy gradient. It **SUPPRESSES without knowing the alternative** (softmax
renormalization redistributes the suppressed mass), which is the complement of what a distillation
target does — see the design doc's §2.1.

- **THE SHIELD.** Potential-based shaping (Ng, Harada & Russell 1999) leaves the **optimal policy set
  unchanged** for any *fixed* φ: the shaping telescopes to `γ^T·φ(s_T) − φ(s_0)`, a constant per start
  state. A miscalibrated φ therefore costs learning SPEED, not correctness.
- ⚠️ **THE CAVEAT THE SHIELD DOES NOT COVER: our φ is LEARNED and DRIFTING.** Exact invariance holds
  **per rollout** (PPO freezes the policy during collection and φ is read once, after it, with the
  collection-time weights) and degrades to **approximate** invariance across rollouts, bounded by φ's
  drift over one credit-assignment window. Operationally: **prefer a MATURE base**; a fresh run tests
  the shield's worst case. The one reassuring fact is the G0 bias map's diagnosis — the head's defect is
  **RESOLUTION, not offset** — and a blurry potential is a WEAK one, not a wrong one (a φ constant over
  a set of states contributes nothing over it and cannot mislead within it).
- **WHERE IT RUNS, and why there.** Env workers hold no model, so the reward cannot be shaped where it
  is produced. `InstrumentedMaskablePPO.collect_rollouts` applies it **after collection, before
  `train()`**: read φ for the whole buffer in one batched `no_grad` forward, add the term to
  `rollout_buffer.rewards` **in place**, then RE-RUN `compute_returns_and_advantage`. That window is the
  only one that works — both collectors compute GAE as their last act, and PopArt reads
  `rollout_buffer.returns` at the top of `train()`, so the shaping lands in **RAW reward space** and
  PopArt — when it is on — normalizes the shaped returns (the only order that keeps the value loss in
  the units of the stream being optimized). ⚠️ This whole `--win-prob-pbrs-*` family is REFUSED under
  `--critic winprob`, so the path described here is a `shaped`-critic path throughout.
- **Both collectors are COVERED, not documented around.** The φ read is a batched re-forward rather than
  a per-step callback capture *because* `--async-rollout` forwards a wave of envs at a time and its
  callback locals cannot recover the env→row mapping (the same reason `WinProbLabelCallback`'s terminal
  capture had to be inlined into `collect_rollouts_async`). One re-forward gives both paths the
  identical quantity, at ≈ one forward pass over the rollout — roughly `1/n_epochs` of one epoch.
- **The two conventions, which are NOT the same case.** **TERMINAL** (`episode_starts[t+1] == 1`, the
  identical test SB3's own GAE uses for `next_non_terminal`, so the two notions of "terminal" cannot
  drift apart): **φ(s′) := 0**, which is what makes the per-episode discounted sum telescope to exactly
  `−coef·φ(s_0)`. **BUFFER-BOUNDARY TRUNCATION** (the episode is still running when the rollout ends):
  φ(s′) is the **bootstrap** φ(s_T) from `model._last_obs`, *not* 0 — forcing 0 there is the classic
  PBRS bug, a phantom penalty for the rollout ending. `TimeLimit.truncated` (the 250-turn deadline)
  arrives as `done=True` and takes the terminal branch, which here is arguably *correct* rather than an
  approximation: that cap IS the forfeit deadline and the reward manager scores it as a real outcome.
- **φ carries no gradient, structurally.** The forward is `no_grad` and the result is numpy before it
  touches the buffer, whose `rewards` is a numpy array — no tensor, no graph, no path back.
- **Config gates (the ONLY gates — nothing version-checks a training-only coefficient).** Negative is
  refused (it inverts the potential; the theorem still holds for `−φ`, so it would train, converge and
  be wrong). `> 0` with `--win-prob-mode none` is refused at config time: the potential IS the head, and
  under `none` no head is built, so the shaping would be a silent no-op. A missing head at runtime is a
  `WinProbPbrsError`, never a skip.
- **Metrics: `train/pbrs_shaping_mean`, `train/pbrs_shaping_absmean`, `train/pbrs_phi_mean`,
  `train/pbrs_reward_share`.** Under `train/` deliberately — this is a property of the reward stream PPO
  is fitting, not of the head. **`pbrs_reward_share` is the one to watch**: mean |shaping| over mean
  |UNSHAPED reward|, i.e. how much of the return signal the coefficient has replaced. Quoted against the
  unshaped stream on purpose, so the ratio does not flatter itself as the coefficient rises.
  ⚠️ **It reads `NaN`, never `0.0`, when the unshaped stream is empty** (R1 adversarial review). Under
  `--no-hand-shaping` the unshaped stream is TERMINAL-ONLY, so any rollout that ends no episode has
  `mean|r| == 0` exactly — and the shaping is then 100% of the reward. The old `0.0` sentinel was the
  reading an operator scans past ("negligible") for the one case where it is everything, in precisely
  the arm the metric exists to watch. Same ABSENT-never-zero rule as `train/q_winprob_loss`.
  **`pbrs_reward_share` is still the WRONG meter for sizing on that stream, and NaN only fixes the
  worst reading.** Where it IS defined its denominator is "±V ÷ episode length", so it moves with the
  EPISODE LENGTH rather than with the coefficient — measured at 2.1-3.1x the true dose across the
  clean-world launch smokes. Hence three companions whose denominator is a CONSTANT — the run's own
  terminal magnitude: **`train/pbrs_episode_dose`, `train/pbrs_episode_dose_n`,
  `train/pbrs_terminal_share`.**
  **`pbrs_episode_dose` is the meter the coefficient ladder is sized in**: the mean |discounted
  shaping sum| of a COMPLETE episode ÷ the terminal magnitude. By the telescoping identity that is
  `coef·E[φ(s_0)]/V` — the shaping's entire per-episode budget priced against one win, i.e. *"this
  run's shaping is worth X% of a win"*. It also checks the telescoping in production rather than only
  in the test: a value that drifts from `coef·phi_mean` means the terminal/truncation convention is
  not doing what it claims on real episodes — and it separates a FROZEN φ from a live one at a glance
  (measured over three launch-smoke iterations: frozen `0.231/0.234/0.228`, live `0.187/0.104/0.087`).
  `pbrs_episode_dose_n` reports the episodes it averaged, so "no complete episode this rollout" never
  reads as "the dose is small". `pbrs_terminal_share` is the per-step companion, always defined.
  The denominator is `model.win_prob_pbrs_terminal_scale`, DERIVED from `--victory-value` in
  `apply_training_hparams` (both build paths) — not a knob, never in the loss. The class default is
  `0.0`, and at `0.0` the two companions are **omitted** rather than divided by a fictitious 30, so a
  smoke/unit test/frozen opponent that never sets it invents no denominator.
- **Versioning.** Training-only, the `td_aux_coef` class exactly: config **v104**, recorded on
  `ModelVersion` for provenance + flagless-resume read-back, never in `check_compatible`, no
  `ARCH_SIGNATURE` bump. Forwarded on both build paths by the one `_TRAINING_HPARAMS` row.
- 🚨 **THE OTHER CONSTANT `--victory-value` SILENTLY INVALIDATES: the distributional critic's
  SUPPORT** — guarded by `_terminal_scale_guards` (R1's F1), which prints
  `[Reward] ⚠️ VALUE-DIST SUPPORT vs TERMINAL SCALE` when the dist head is on, PopArt is OFF and the
  raw-return support either fails to bracket `max(victory, |draw|)` or quantizes it into too few
  atoms. Same genre as the coefficient re-sizing — a constant calibrated against a scale, carried
  across a change of scale. The LAUNCH RULE it implies is carried by
  `designs/ai_v12/launch_runbook.md` §6.3: the guard warns, nothing stops the run.
- **Tests.** `agents/training/winprob_pbrs_test.py` (22): the telescoping identity on a hand case and
  over 40 random episode layouts; the truncation-vs-terminal split; an off-by-one revert-catcher on the
  `episode_starts[t+1]` test; grad-disabled + detached-to-numpy (fails if the `no_grad` is deleted);
  coef-0 buffer identity + the source contract that the import is local to the non-zero branch; the
  raw-reward/GAE-recompute order; chunk-boundary coverage; the loud-refusal path; both config gates; the
  v104 migration; and the frozen-φ group below. Five revert-catchers verified failing on a
  deliberate revert.

### FROZEN φ (`--win-prob-pbrs-source <ckpt>`, `gen3_winprob_pbrs_source_v1`, config v105)

**The caveat above, removed.** The invariance theorem assumes φ is a **fixed** function of state;
ours is a head inside the network being trained. `--win-prob-pbrs-source` points the potential at a
**frozen foreign checkpoint** instead, so the shield holds exactly rather than approximately.
Absent (the default) ⇒ the live head, byte-identical to what v104 shipped.

- **One seam, one loader.** Only `winprob_pbrs.phi_model(model)` changes: it returns
  `model._winprob_phi_source or model`, and `buffer_potentials` / the bootstrap read it. The
  loading is `--distill-teacher`'s path verbatim — `fixed_opponent_pool._resolve_zip_and_config`
  → `snapshot.load_foreign_opponent` → `set_training_mode(False)`, in `main/train/model_build.py`.
  A bad path is `os._exit(FATAL_CONFIG)`, never a crash-restart loop.
- ⚠️ **A FULL frozen extractor forward is required; there is no head-only shortcut.**
  `WinProbHead.forward` consumes `value_pooled` — the whole-board value pool produced by *that*
  network's own trunk with its own weights. Running the frozen head over the LIVE trunk's pooled
  features computes a function of a representation the head never saw, AND it would drift with the
  live trunk, destroying the exact property the frozen source buys.
- **Cost.** The frozen forward **REPLACES** the live-φ one rather than adding to it, so the compute
  is unchanged (~1/`n_epochs` of one epoch). New: one frozen extractor of memory (the
  `--distill-teacher` class, which the tree already runs at N ≥ 3) and one load at startup.
- ⚠️ **Two forwards on the post-rollout obs now, and the split is load-bearing.** `last_values` is
  the **GAE bootstrap** and must stay the LIVE critic's; φ(s_T) must come from the φ network. With
  no source the two coincide and it stays ONE forward exactly as before. Frozen φ on the buffer
  rows with a LIVE φ on the last row would break the telescoping at every truncation boundary.
- **A prior-generation φ is viable and is the point** (that is where a MATURE potential lives).
  `load_foreign_opponent` validates the obs FAMILY (`arch_signature`), and `_phi_obs` passes only
  the keys the source's own space declares — the same filter the distillation teachers use.
- **`--win-prob-mode` governs the LIVE head only** here, i.e. whether it trains as a diagnostic.
  `read_only` is the right choice on this arm: risk-free, and it keeps a live φ trajectory to
  compare against the frozen one — a free measurement of how far the potential has drifted from
  the run's own beliefs.
- **`--compile-trainer` interaction: the source is left EAGER, deliberately.** The compile patches
  the bound `forward` of the LIVE policy's extractor for the per-minibatch train step; the frozen
  source runs once per **rollout**, so a second Inductor graph would buy a warm-up and nothing else.
  ⚠️ **UNEXERCISED:** a real CUDA `torch.compile` with a frozen source attached has not been run —
  `compile_trainer_extractor` refuses a non-cuda device, so the CPU test tier cannot reach it. What
  IS tested is the seam that makes it safe (the compile module never names `_winprob_phi_source`;
  replacing the live extractor's bound `forward` with a poisoned callable leaves φ unchanged).
- **The coefficient carries the [−1,+1] mapping, NOT a `2p−1` spelling of φ.** They are equivalent
  up to `coef ← 2·coef` plus a per-step constant `coef·b·(γ−1)`, and at `b = −1` that constant is
  `+1e-4·coef` per step — small, but a wrongly-signed **stall incentive** in an arm that has deleted
  every anti-stall term. It also breaks `successor_potential`'s `φ(terminal) := 0` convention, which
  is correct for a [0,1] potential and is the MIDDLE of a [−1,+1] one. Write
  `--win-prob-pbrs-coef 2c`; keep φ = σ(logit).
- **Provenance.** Recorded on `ModelVersion` (`win_prob_pbrs_source`) and **inherited on a flagless
  resume** (`_resolve`), because a resume that silently reverted to live-φ would swap exact
  invariance for approximate with nothing saying so. Listed in `_excluded_save_params` — a frozen
  foreign model is never pickled into our checkpoint. Startup prints the resolved zip, its
  `arch_signature` and its `config_version`: a clean-world run is uninterpretable if the identity
  of its frozen potential is not pinned.
- **Config gate.** A source with no positive coefficient is refused — it would load a whole extra
  network, forward it once per rollout, and multiply the result by zero.
- **THE correctness test** (`winprob_pbrs_test.py`): point the frozen source at the run's **own
  current checkpoint**, through the real `load_foreign_opponent`, on a real `Gen3FeaturesExtractor`
  — every φ must come back **bit-identical** to the live path. A head-only shortcut fails it, and so
  does any obs-key or eval-mode discrepancy. Its anti-vacuity twin drifts the live weights and
  requires the frozen φ not to move while the live φ does.

### THE ACTOR-ONLY FROZEN POTENTIAL — `--win-prob-pbrs-frozen` (`gen3_frozen_phi_actor_only_v1`)

**The value loss is the ONE thing this flag does not touch, and that is the whole construction.**
`agents/training/frozen_phi.py` reads φ = σ(logit) from a FROZEN checkpoint's win-prob head and adds
`γφ(s′) − φ(s)` to the stream that feeds the POLICY's advantages — writing **only**
`rollout_buffer.advantages`. `rewards` and `returns` are restored to what the collector produced (by
ASSIGNMENT from a snapshot; `(a+b)−b` is not `a` in float32), so the critic keeps regressing the
unshaped terminal indicator and every consumer of the value target — the scalar-MSE diagnostic
`train/value_loss`, `train/explained_variance`, `value_scale_metrics` — reads the UNSHAPED return.
Under this critic the real value loss is the head's BCE against `win_target`, which never reads
`rewards` at all, so `V ≡ P(win|s)` is preserved **bit-for-bit** with or without the flag.

**A potential added to the REWARD could not be**, and that is why the rung was held rather than
because of any doubt about the invariance: the shaped return telescopes to `1{win} − φ(s)`, negative
wherever the frozen head was optimistic about a lost game, and a sigmoid cannot output a negative
number — so the critic would be fitted to a target outside its own range and the identity the search
leaf, the calibration gate and `--vf-coef`'s meaning all rest on would be false by a known function.
Restricting the term to the advantage keeps the Ng shield (φ is FIXED, so the theorem holds
**exactly**) and, at λ = 1, makes the shaped advantage the unshaped one minus `−coef·φ(s)` — a
state-dependent BASELINE, i.e. zero-bias for a policy gradient. `φ(terminal) := 0` both makes the sum
telescope and stops the potential leaking the outcome the terminal state just revealed; the
conventions come from `winprob_pbrs.successor_potential`, IMPORTED, so the two shaping paths cannot
drift on the one convention the theorem rests on. **The coefficient is exactly 1.0, DERIVED** (φ is
already one unit of V per unit of V) and PRINTED at startup, never a knob.

**BOTH SEAMS LIVE IN `frozen_phi.py`**, in the `distill_anchor.py` shape: `shape_after_rollout` at
`collect_rollouts` — the ONE point both rollout loops pass through, so the async collector is covered
by construction — and `record_metrics` in `train()`. `ppo.py` carries one call each, because it sits
AT the file-size ratchet's 2,000-line hard bound. The frozen network rides `_winprob_phi_source`, the
attribute the shaped ladder's `--win-prob-pbrs-source` already uses (the two are mutually exclusive
by refusal), so `phi_model` and `_excluded_save_params` need no second name and the foreign weights
are never pickled into our checkpoint.

⚠️ **Read `pbrs/frozen_phi_mean` FIRST, and it must be FLAT** — φ is a fixed function of state, so a
mean that wanders like a live head's means the frozen source is not the thing being read (the
runbook §4.2 check: frozen `0.403 → 0.391 → 0.391` against live-φ's `0.680 → 0.347 → 0.216`).
`pbrs/frozen_phi_episode_dose` prices the shaping against one win, and
`signal/adv_shaped_minus_unshaped_mean` is the telescoping term — at λ = 1 on complete episodes
exactly `−coef ×` the φ mean, so the pair is a one-glance audit that the terminal convention holds on
real episodes. **The cost is stated with the benefit**: the frozen head's biases are now inside every
advantage, and §4.1's baseline measured this class of head starved of RESOLUTION — so a FROZEN-φ arm
that beats SPARSE has measured *this head's* separation, not dense credit in general.

Training-only (config **v110**, recorded and `_resolve`-inherited) — and the inheritance is
load-bearing here in a way it is not for a coefficient: the flag is **boolean by PRESENCE**, so
"not typed" and "off" are the same argv, and a launcher restart re-invoking the original command
would otherwise turn a FROZEN-φ arm into the SPARSE arm mid-run under the same run name. Refused
under `--critic shaped` (there φ and V are in different units and the dose is a real question — use
`--win-prob-pbrs-coef` / `--win-prob-pbrs-source`). Gate: `frozen_phi_test.py`.

