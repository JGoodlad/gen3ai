# Training — the win-probability head (its PBRS routes were DELETED, deletion pass L1)

Moved verbatim out of `src/agents/training/CLAUDE.md` on **2026-09-08**, in the second pass of the
topic split. **The two win-prob PBRS routes this file also held — self-φ (`--win-prob-pbrs-coef` /
`--win-prob-pbrs-source`, `winprob_pbrs.py`) and the actor-only frozen potential
(`--win-prob-pbrs-frozen`, `frozen_phi.py`) — were DELETED with the shaped critic's levers (deletion pass
L1, config v131); their sections are gone and `designs/deleted_flags.md` carries the citations.** The
filename stays so existing links resolve.

`src/agents/training/CLAUDE.md` keeps the heading and a short summary of each, and points here.
**This file is the owner of the detail.**

---

## Win-probability head (`--win-prob-mode`)

The training half of the tri-state win-probability head (model side: `src/agents/model/CLAUDE.md` →
win-probability head, v22). A calibrated **P(win|state)** the shaped critic can't give — supervised by the
Monte-Carlo episode OUTCOME. Off by default (`--win-prob-mode none`); under the win-prob critic (the only critic) it is THE CRITIC.
Three pieces live here:

- **The label is a FUTURE quantity** — the outcome is only known when the battle ends, so (unlike the
  per-step belief labels, which are privileged info known *each* step) it CANNOT ride as a real per-step
  obs key. The plumbing reuses the obs-dict-label STORAGE path with post-hoc population:
  - **The trainee's observation space** (`agents.training.trainee_spaces`; the deleted Python `gen3_env.py` used to declare it) carries two TRAINING-ONLY obs keys when `emit_win_target` (`--win-prob-mode != none`):
    `win_target` [1] + `win_mask` [1] (float32). The rollout buffer therefore stores + shuffles them automatically (the belief-label path). Read ONLY by
    the loss; the model forward reads only `obs["observation"]`, so the OUTCOME can't leak.
  - **The Rust collector** (`rust_rollout/collector.py`) records `info["win_outcome"]` (1.0 win / 0.0 loss-or-tie)
    at each done step, and fills `win_target` / `win_mask` itself: complete-game mode gives every row its own game's outcome (every row `win_mask` 1); the window fill calls `win_prob_callback.backfill_terminal_labels`
    (γ_win = 1, undiscounted → P(win|s) = "probability this state leads to a win"). The trailing IN-PROGRESS episode of a window (no terminal yet in-buffer) gets `win_mask=0` and is
    excluded — never trained toward a fabricated label. (The Python `MaskableAgentWrapper.step` and `WinProbLabelCallback` that did this, sync and async, were deleted in U3.)
- **Loss (`instrumented_ppo.py` `_win_prob_loss`).** `train()` reads `last_win_prob_logits` (stashed by the
  `evaluate_actions` forward) + `rollout_data.observations["win_target"]`/`["win_mask"]`, folds
  the masked BCE (at `vf_coef` under the win-prob critic (the only critic), a fixed weight 1.0 as an aux otherwise — the
  `--win-prob-coef` flag that once set it was deleted). read_only vs shaping differ ONLY in whether the
  extractor stop-grads the head's input (the trunk gradient) — the loss term itself is identical. Folded
  whenever the extractor's `win_prob_mode != none`.
- **Metrics (`win_prob/*` — its OWN TB prefix, not `train/`, matching the `grad/`/`eval/`
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
  shaping (watch it sit small).
- **Versioning.** `win_prob_mode` (str) is the structural + resume-IMMUTABLE toggle (any change FATALs;
  threaded into `current_model_version` / `arch_toggles_from_model` so a win-prob-ON self-play run doesn't
  FATAL on its own sentinels).
- **Forensic trace + prober.** `RLPlayer._win_prob` (`inference/player.py`) reads the stashed
  `last_win_prob_logits` at trace-capture time (sigmoid ⇒ P(win)) into the per-decision `state`, which
  `BattleRecorder.states_arrays` writes as a `win_probs` npz array (NaN = no head / not captured, parallel
  to `values`). The prober renders **P(win) + ΔP(win)** in the Summary + Outcome panels beside CRITIC's
  V/ΔV — "how a move moved the win odds" — model-free from that array (`engine.WinProbView`); `None`/absent
  on a non-win-prob run. See `src/main/prober/CLAUDE.md`.
- **Tests.** Unit: `agents/training/win_prob_test.py` (loss masking + None guards + the callback MC-fill
  backward-propagation + in-progress masking + sync-capture-at-pos + async-skip), `agents/model/
  win_prob_head_test.py` (module build, off byte-identical projection dims, the read_only-stop-grad /
  shaping-flows gradient gating, the v22 version gate). End-to-end `--debug
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
never a dose mystery — the live mode was never pointed at behavior. The routes that WERE pointed at it
(the win-prob PBRS family) are deleted; their design is `designs/ai_v12/design_winprob_behavior_coupling.md`.
