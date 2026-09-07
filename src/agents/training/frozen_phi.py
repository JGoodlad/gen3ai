"""`--win-prob-pbrs-frozen` — ACTOR-ONLY potential shaping from a FROZEN win-prob head
(`gen3_frozen_phi_actor_only_v1`; the FROZEN-φ rung of the ladder registered in
[`designs/ai_v12/launch_runbook.md`](../../../designs/ai_v12/launch_runbook.md), owner amendment in
[`design_winprob_only_critic.md`](../../../designs/ai_v12/design_winprob_only_critic.md) §3.7).

WHY THIS IS NOT `winprob_pbrs.py` WITH A DIFFERENT COEFFICIENT
==============================================================
Route 1 (`winprob_pbrs.py`) adds ``coef·(γφ(s′) − φ(s))`` to ``rollout_buffer.rewards`` and re-runs
GAE, so the shaping reaches the ADVANTAGE **and** the critic's regression target — which is exactly
right when the critic regresses a return. Under ``--critic winprob`` it is exactly wrong, and the
reason is the identity the whole mode rests on:

    V(s) = σ(z) ∈ [0, 1] ≡ P(win | s)

A potential added to the REWARD stream makes the critic's target the SHAPED return. With
Φ(terminal) = 0 and γ = 1 that telescopes to ``1{win} − φ(s)``, a quantity that is **negative
wherever ``φ(s) > 1{win}``** — i.e. on every state of every lost game the frozen head was optimistic
about. A sigmoid cannot represent a negative number at all, so the critic would be fitted to a
target outside its own range, and ``V ≡ P(win)`` — the search leaf's contract
(`search_dividend`), the calibration gate's contract (`win_prob/critic_resolution`,
`main.scaffolding_gauge --reliability`) and the reason `--vf-coef` multiplies a BCE — would be
false by a known, state-dependent function. That is why §3.7 held this rung DEFERRED rather than
shipping route 1 with a new coefficient.

THE CONSTRUCTION: shape the ACTOR, leave the CRITIC alone
=========================================================
The critic's target and the advantage estimator are **separate choices** — §3.4 already leans on
that (an MC critic target beside a bootstrapped advantage). This module takes the same split one
step further:

* the CRITIC is trained on the UNSHAPED terminal indicator, exactly as it is today. Under
  ``--critic winprob`` its loss is the win-prob head's BCE against ``win_target`` — a
  Monte-Carlo outcome label carried in the obs dict, which never reads ``rewards`` at all. So
  ``V`` stays ``P(win|s)``, bit-for-bit, with or without this flag.
* the POLICY's advantages are computed from the SHAPED stream: GAE over
  ``r + γφ(s′) − φ(s)`` with the UNSHAPED ``V`` as the baseline.

Both facts are made structural rather than promised: this module writes **only**
``rollout_buffer.advantages``. ``rewards`` and ``returns`` are restored to the values the
collector produced, so every consumer of the value target — the scalar-MSE diagnostic
``train/value_loss``, ``train/explained_variance``, ``value_scale_metrics`` — reads the UNSHAPED
return and says so.

IS IT STILL POTENTIAL-BASED SHAPING? YES — and the guarantee is the STRONGER one
================================================================================
Ng, Harada & Russell (1999) says: for a FIXED potential φ, adding ``F(s,a,s′) = γφ(s′) − φ(s)`` to
the reward leaves the optimal policy set unchanged, because over any trajectory the added term
telescopes::

    Σ_{t=0}^{T-1} γ^t · (γφ(s_{t+1}) − φ(s_t))  =  γ^T φ(s_T) − φ(s_0)  =  −φ(s_0)

with the convention ``φ(terminal) := 0``. The sum depends only on the endpoints, so it adds the
same constant to every policy's return from a given start state, and no policy ordering can move.

Our φ is a checkpoint's win-prob head, loaded once and never trained, so it IS a fixed function of
state — the theorem's hypothesis holds **exactly**, which is the whole reason the frozen rung
exists and route 1's live-head form does not get this (its docstring's own caveat: exact per
rollout, approximate across).

Applying the telescoping term to the ADVANTAGE only is a strict *restriction* of the transformation,
not a departure from it. At λ = 1 the identity is exact and per-row::

    A′_t − A_t  =  Σ_{k≥0} γ^k · (γφ(s_{t+k+1}) − φ(s_{t+k}))  =  γ^{T−t} φ(s_T) − φ(s_t)  =  −φ(s_t)

so the shaped advantage is the unshaped advantage minus a **function of the state alone** — a
state-dependent baseline, which is the textbook zero-bias modification of a policy gradient
(``E_a[∇log π(a|s) · b(s)] = 0`` for any ``b`` that does not depend on ``a``). At λ < 1 the sum is
truncated geometrically and the same argument applies to each partial sum. So the actor-only form
inherits the invariance **and** avoids the one thing the reward-stream form would have cost: the
critic's target is not touched, so ``V ≡ P(win)`` is preserved exactly rather than approximately.

``φ(terminal) := 0`` is doing double duty and both jobs matter here:

1. it is what makes the sum telescope to a constant (above), and
2. it is what stops the potential leaking OUTCOME information into the advantage. A frozen head
   evaluated at a terminal state could be read as a prediction of the result that state just
   revealed; forcing it to 0 means the last transition's shaping is ``−φ(s_{T−1})``, a function of
   the state the agent ACTED in, never of what happened. (The buffer-boundary TRUNCATION case is
   different and is NOT forced to 0 — see `winprob_pbrs.successor_potential`, whose conventions
   this module reuses rather than re-deriving.)

THE COEFFICIENT IS 1.0, DERIVED — NOT A KNOB
============================================
Under ``--critic winprob`` the terminal is the WIN INDICATOR (``+victory_value`` on a win, ``0.0``
on a loss, a tie and a 250-turn timeout alike) at ``--victory-value 1.0``, so the undiscounted
return is ``1{win}`` and ``V(s) = P(win|s) ∈ [0,1]``. The potential ``φ = σ(win-prob logit)`` is
already in **exactly that currency** — one unit of φ is one unit of V — so the currency-matched
coefficient is ``1.0``. It is fixed here, PRINTED at startup, and the flag is boolean by presence
(owner, 2026-09-06: *"it seems like it would be a boolean, not a scalar"*). The alternative
currency gives the same answer scaled: with a ±1 terminal and ``V = 2p − 1`` the matched
coefficient is 2 — and that currency would also break ``φ(terminal) := 0``, which is the correct
zero for a [0,1] potential and the MIDDLE of a [−1,+1] one.

WHAT IT BUYS, AND WHAT IT COSTS — both stated, because only one of them is the pitch
===================================================================================
BUYS: **dense credit from a calibrated head.** The clean-world composition is 1 TERMINAL + 0 PBRS +
0 BIAS, i.e. ~1 bit per ~40 decisions, and the SPARSE arm is the famine test of whether that is
learnable at all. A frozen mature φ turns every transition into a scored one without changing the
optimum — the accelerant the design deleted, restored in the one form whose invariance is exact.

COSTS: **the frozen head's own biases become part of every advantage.** The committed baseline
(`designs/research_state/measurements/winprob_critic_baseline_2026-09-06/`) measured this class of
head at reliability ~0.002 against a resolution of 0.062 out of an available 0.182 — already
calibrated in the MEAN and **starved of SEPARATION**. A blurry potential is a WEAK potential rather
than a wrong one (a φ constant across a set of states contributes nothing over that set and cannot
mislead within it), so the failure mode is "the shaping does less than hoped", not "the shaping
teaches the wrong thing" — the invariance covers the second. But the resolution starvation the
baseline measured is now inside the policy gradient, and a FROZEN-φ arm that beats SPARSE has
measured the value of *this specific head's* separation, not of dense credit in general.

φ CARRIES NO GRADIENT, structurally: the forward runs under ``torch.no_grad()`` and the result is
numpy before it touches the buffer, whose ``advantages`` is a numpy array.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import torch as th
from stable_baselines3.common.utils import obs_as_tensor

from agents.training.winprob_pbrs import (
    WinProbPbrsError,
    _forward_phi,
    _phi_obs,
    buffer_potentials,
    episode_dose,
    phi_model,
    pbrs_shaping,
    successor_potential,
)

#: THE CURRENCY-MATCHED COEFFICIENT, derived in the module docstring: under `--critic winprob` the
#: return is the win indicator and V(s) = P(win|s), so φ = σ(logit) ∈ [0,1] is already one unit of V
#: per unit of V. Not a flag, not a knob — printed at startup so the operator reads it rather than
#: chooses it. The dose ladder belonged to the shaped critic, where φ and V were in different units.
FROZEN_PHI_COEF = 1.0


def advantage_shaping_delta(adv_shaped: np.ndarray, adv_unshaped: np.ndarray) -> np.ndarray:
    """``A′ − A`` — the telescoping term the policy gradient gained, per buffer row.

    At λ = 1 this is exactly ``−coef·φ(s_t)`` on every row of a COMPLETE episode (the module
    docstring's derivation), which is what `frozen_phi_test` pins on a synthetic buffer. It is
    published as ``signal/adv_shaped_minus_unshaped_mean`` so the identity is checkable in
    production and not only in the test: a mean that drifts away from ``−coef·φ_mean`` means the
    terminal convention is not doing what it claims on real episodes.
    """
    return np.asarray(adv_shaped, dtype=np.float64) - np.asarray(adv_unshaped, dtype=np.float64)


def apply_frozen_phi_shaping(model, rollout_buffer) -> Dict[str, float]:
    """Shape the POLICY's advantages with the frozen potential; leave the critic's target alone.

    Called from `InstrumentedMaskablePPO.collect_rollouts` — **the one seam both rollout loops pass
    through**. SB3's stock collector and `collect_rollouts_async` each compute GAE as their last
    act and then return to `learn()`, which calls `train()` next; this wrapper sits between them,
    so the async path is covered by construction rather than by a parallel implementation. (That is
    the same seam `apply_winprob_pbrs` uses, and for the same reason: env workers hold no model, so
    the potential cannot be read where the rewards are produced.)

    The order, and every step of it is load-bearing:

    1. read φ for the whole buffer from the FROZEN source, in one chunked `no_grad` forward;
    2. one forward on ``model._last_obs`` gives the GAE bootstrap ``last_values`` from the LIVE
       critic, and a second (frozen) forward gives the bootstrap potential φ(s_T). Two forwards,
       not one, because a frozen φ on the buffer rows beside a LIVE φ on the last row would break
       the telescoping at every truncation boundary;
    3. build φ(s′) with the TERMINAL (``:= 0``) and BUFFER-TRUNCATION (``:= φ(s_T)``) conventions —
       `winprob_pbrs.successor_potential`, IMPORTED rather than re-derived, so the two shaping
       paths cannot drift apart on the one convention the invariance theorem rests on;
    4. recompute GAE on the UNSHAPED rewards with this bootstrap → the reference advantages. This
       is deliberately a recomputation rather than a copy of what the collector produced: both arms
       then share one ``last_values``/``dones`` pair, so their difference is the shaping and
       nothing else;
    5. add the shaping to ``rewards``, recompute GAE → the shaped advantages;
    6. **RESTORE** ``rewards`` and ``returns`` to their unshaped values and keep only the shaped
       ``advantages``. This is what makes the construction actor-only, and it is an assignment
       rather than a subtraction: ``(a + b) − b`` is not ``a`` in float32.

    Returns the TB metrics, already prefixed.
    """
    if not bool(getattr(model, "_frozen_phi_on", False)):
        return {}                        # defensive: the caller gates on this too
    phi_net = phi_model(model)
    if phi_net is model:
        raise WinProbPbrsError(
            "--win-prob-pbrs-frozen is on but no FROZEN φ source is attached "
            "(`model._winprob_phi_source` is None), so φ would be read from the LIVE win-prob head "
            "— which under --critic winprob IS the critic, making the shaping the TD residual GAE "
            "already computes. That is the SELF-φ double counting this flag exists to avoid. "
            "`main.train.model_build` owns the attachment and FATALs on a bad path; reaching here "
            "means the gate and the loader disagree.")

    coef = float(getattr(model, "frozen_phi_coef", FROZEN_PHI_COEF))
    gamma = float(model.gamma)

    phi = buffer_potentials(model, rollout_buffer)

    with th.no_grad():
        last_values = model.policy.predict_values(obs_as_tensor(model._last_obs, model.device))
        phi_boot = _forward_phi(phi_net, _phi_obs(phi_net, model._last_obs))
    dones = np.asarray(model._last_episode_starts, dtype=np.float64).reshape(-1)

    phi_next = successor_potential(phi, rollout_buffer.episode_starts, phi_boot, dones)
    shaping = pbrs_shaping(phi, phi_next, gamma, coef)

    rewards_unshaped = np.array(rollout_buffer.rewards, copy=True)

    # ARM A — the UNSHAPED reference. Same bootstrap as arm B, so the difference below is purely
    # the potential.
    rollout_buffer.compute_returns_and_advantage(last_values=last_values, dones=dones)
    adv_unshaped = np.array(rollout_buffer.advantages, copy=True)
    returns_unshaped = np.array(rollout_buffer.returns, copy=True)

    # ARM B — the SHAPED stream. Only its `advantages` survive.
    rollout_buffer.rewards += shaping.astype(rollout_buffer.rewards.dtype)
    rollout_buffer.compute_returns_and_advantage(last_values=last_values, dones=dones)
    adv_shaped = np.array(rollout_buffer.advantages, copy=True)

    # THE ACTOR-ONLY RESTORE. `rewards` and `returns` go back to the collector's stream so the
    # value target, `train/value_loss`, `train/explained_variance` and `value_scale_metrics` all
    # read the UNSHAPED return; `advantages` keeps the shaped one, which only the policy reads.
    rollout_buffer.rewards[:] = rewards_unshaped
    rollout_buffer.returns[:] = returns_unshaped
    rollout_buffer.advantages[:] = adv_shaped

    delta = advantage_shaping_delta(adv_shaped, adv_unshaped)
    dose_abs, dose_n = episode_dose(shaping, rollout_buffer.episode_starts, gamma)
    out = {
        "pbrs/frozen_phi_coef": coef,
        "pbrs/frozen_phi_mean": float(phi.mean()),
        "pbrs/frozen_phi_shaping_mean": float(shaping.mean()),
        "pbrs/frozen_phi_shaping_absmean": float(np.abs(shaping).mean()),
        "pbrs/frozen_phi_episode_dose_n": float(dose_n),
        # THE TELESCOPING TERM, read directly off the two advantage arrays. At λ=1 on complete
        # episodes this is exactly `−coef·mean(φ)`; the per-EPISODE discounted shaping sum is
        # exactly `−coef·φ(s_0)`, which `pbrs/frozen_phi_episode_dose` reports below. Both are
        # pinned in `frozen_phi_test`.
        "signal/adv_shaped_minus_unshaped_mean": float(delta.mean()),
        "signal/adv_shaped_minus_unshaped_absmean": float(np.abs(delta).mean()),
    }
    # THE SIZING METER — the per-episode budget in "fraction of a win". Its denominator is the run's
    # TERMINAL magnitude (a constant), never the reward stream's own mean, which under the
    # clean-world composition is terminal-only and therefore zero on a rollout with no episode end
    # (`winprob_pbrs.episode_dose`'s docstring, probe N §7.5).
    scale = abs(float(getattr(model, "win_prob_pbrs_terminal_scale", 0.0) or 0.0))
    if scale > 0.0 and dose_n:
        out["pbrs/frozen_phi_episode_dose"] = dose_abs / scale
    return out


# ──────────────────────────────────────────────────────────────────────────────────────────────
# THE TWO SEAMS — both live HERE, and `ppo.py` carries one call each
# ──────────────────────────────────────────────────────────────────────────────────────────────
#
# `instrumented_ppo/ppo.py` sits at the file-size ratchet's 2,000-line hard bound, and the tree's
# answer to that is the `distill_anchor.py` shape: the mechanism and its prose live in the module
# that owns them, and the train loop carries ONE line per seam. Nothing here is a policy decision
# `ppo.py` should be making — the first is "when does the shaping run", which is answered by the
# flag, and the second is a metrics drain.

def shape_after_rollout(model, rollout_buffer, collected: bool) -> None:
    """THE `collect_rollouts` SEAM — the ONE point both rollout loops pass through.

    `InstrumentedMaskablePPO.collect_rollouts` wraps `collect_rollouts_async` AND
    `super().collect_rollouts`, and `learn()` calls `train()` next, so this window is after every
    collector's GAE and before every update — which is why the async path is covered by
    construction rather than by a parallel implementation, and why env workers (which hold no
    model, and so cannot read a potential where the rewards are produced) need no changes.

    Stashes the metrics on the model for `record_metrics` to drain, mirroring `_pbrs_metrics`.
    `collected` is the collector's own return value: a rollout that was interrupted has no buffer
    worth shaping. OFF (the flag absent) is a single boolean test.
    """
    if collected and bool(getattr(model, "_frozen_phi_on", False)):
        model._frozen_phi_metrics = apply_frozen_phi_shaping(model, rollout_buffer)


def record_metrics(model, logger) -> None:
    """THE `train()` SEAM — drain this rollout's metrics, which arrive ALREADY PREFIXED.

    They are prefixed because they land in TWO TensorBoard groups, and the split is the reading
    order:

    * ``pbrs/frozen_phi_*`` — the POTENTIAL. ``_mean`` must be **FLAT** across rollouts, because φ
      is a fixed function of state; a wandering mean means the frozen source is not the thing being
      read. ``_episode_dose`` is the sizing meter — "the shaping is worth X% of a win".
    * ``signal/adv_shaped_minus_unshaped_mean`` — the TELESCOPING TERM the policy gradient gained.
      At λ = 1 on complete episodes it is exactly ``−coef · mean(φ)``, so reading it BESIDE
      ``pbrs/frozen_phi_mean`` audits the terminal convention on real episodes rather than only in
      `frozen_phi_test`.

    It is a property of the ADVANTAGE and is computed in `collect_rollouts`: the term edits the
    buffer, not the loss, so it has no per-minibatch existence.
    """
    for key, value in (getattr(model, "_frozen_phi_metrics", None) or {}).items():
        logger.record(key, float(value))
