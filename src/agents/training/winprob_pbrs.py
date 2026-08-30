"""`--win-prob-pbrs-coef` — POTENTIAL-BASED REWARD SHAPING from the win-probability head (ai_v12 route 1).

Design: `designs/ai_v12/design_winprob_behavior_coupling.md`.

WHAT IT DOES, and why the head cannot do it on its own. `WinProbHead` emits a calibrated
P(win | state); `--win-prob-mode shaping` lets that BCE reach the shared trunk. But that is
*representation* shaping — a feature subsidy — and it exerts **zero force on behavior**: there is no
gradient path anywhere from "predict wins" to "choose winning actions", because the head's logit is a
SIDE readout that is never concatenated into pi/vf (leak-safety: its label is the privileged future
outcome). The head is a BAROMETER, not a coach.

This module is the reward-level route that converts it into force. With φ(s) = σ(win-prob logit), every
transition's reward becomes::

    r'(s, a, s') = r(s, a, s') + coef · ( γ·φ(s') − φ(s) )

which the RL machinery then answers for: the shaping enters GAE, the advantage, and the policy gradient.
A move that drops the model's own win probability now costs literal reward.

**THE SHIELD (Ng, Harada & Russell 1999).** For any *fixed* potential φ, this transformation leaves the
optimal policy set UNCHANGED: over any trajectory the shaping telescopes to γ^T·φ(s_T) − φ(s_0), which
depends only on the endpoints, so it adds the same constant to every policy's return from a given start
state. A miscalibrated φ therefore costs learning SPEED, not CORRECTNESS. That is an unusually strong
guarantee for a research lever.

⚠️ **THE CAVEAT THE SHIELD DOES NOT COVER: our φ is LEARNED and DRIFTING.** The theorem assumes φ is a
fixed function of state; ours is a head inside the network being trained. Consequences (design §2.4):
  * Within a rollout the shaping is EXACTLY telescoping — PPO freezes the policy during collection and
    this module reads φ once, after collection, with the collection-time weights. So each rollout's
    contribution obeys the theorem exactly.
  * ACROSS rollouts φ drifts, and the per-start-state constant moves with it. Exact invariance degrades
    to approximate invariance, bounded by φ's drift over one credit-assignment window.
  * Therefore the lever wants a MATURE φ. Enabling it on a fresh run tests the shield's worst case.
The one reassuring fact: the G0 bias map found the scalar head's defect is RESOLUTION, not offset — and
a blurry potential is a WEAK potential, not a wrong one (a φ constant across a set of states contributes
nothing over that set and cannot mislead within it).

WHERE IT RUNS, and why there. Env workers hold no model, so φ cannot be computed where rewards are
produced. This is TRAINER-SIDE buffer augmentation, applied by `InstrumentedMaskablePPO.collect_rollouts`
AFTER collection and BEFORE `train()`: read φ for the whole buffer in one batched `no_grad` forward, add
the shaping to `rollout_buffer.rewards` in place, then RE-RUN `compute_returns_and_advantage` so returns
and advantages see the shaped stream. PopArt reads `rollout_buffer.returns` at the top of `train()`, i.e.
after this — so the shaping lands in RAW reward space and PopArt normalizes the shaped returns, which is
the only order that keeps the value loss in the units of the stream being optimized.

**The batched re-forward is deliberate, not laziness.** The per-step `last_win_prob_logits` stash IS
available to a callback's `_on_step` on the stock path (that is how `WinProbLabelCallback` captures
terminals) — but the ASYNC collector forwards a wave of envs at a time and its callback locals cannot
recover the env→row mapping (the same reason the win-target capture had to be inlined there). One
batched re-forward gives BOTH paths the identical, obviously-correct quantity, so `--async-rollout` is
covered rather than documented around. Cost ≈ one forward pass over the rollout, i.e. roughly
1/`n_epochs` of one training epoch.

**φ CARRIES NO GRADIENT, structurally.** The forward runs under `torch.no_grad()` and the result is
converted to numpy before it touches the buffer, whose `rewards` is a numpy array. There is no tensor,
no graph, and no possible path from the policy loss back through the potential.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch as th
from stable_baselines3.common.utils import obs_as_tensor

#: How many buffer rows to forward at once when reading φ. Sized so the peak activation memory of the
#: extra forward stays well under one PPO minibatch's (which is `--batch-size`, typically 4096-16384),
#: and small enough that a 250k-row buffer never allocates one giant tensor.
PHI_FORWARD_CHUNK = 2048


class WinProbPbrsError(RuntimeError):
    """The shaping was requested but φ is unreadable. FAIL-LOUD by design.

    Silently skipping a reward-shaping term is the invisible-regression class: the run trains
    correctly, just with a lever the operator believes is on, and no metric says otherwise.
    """


# ──────────────────────────────────────────────────────────────────────────────────────────────
# PURE — the shaping arithmetic. No torch, no model, no buffer. This is what the tests pin.
# ──────────────────────────────────────────────────────────────────────────────────────────────

def successor_potential(
    phi: np.ndarray,
    episode_starts: np.ndarray,
    phi_bootstrap: np.ndarray,
    last_episode_starts: np.ndarray,
) -> np.ndarray:
    """φ(s′) per buffer row, with the TERMINAL and TRUNCATION conventions applied.

    ``phi`` / ``episode_starts`` are ``[n_steps, n_envs]``; ``phi_bootstrap`` / ``last_episode_starts``
    are ``[n_envs]`` (read from ``model._last_obs`` / ``model._last_episode_starts`` — the same pair
    SB3's own GAE uses for its bootstrap, so the two agree by construction).

    The two conventions, which are NOT the same case:

    * **TERMINAL** (row *t* ended its episode ⇒ ``episode_starts[t+1] == 1``): **φ(s′) := 0**. This is
      what makes the per-episode discounted shaping sum telescope to exactly ``−coef·φ(s_0)`` — a
      constant per start state, which is the invariance theorem's whole content.
    * **BUFFER-BOUNDARY TRUNCATION** (the episode is still running when the rollout ends): φ(s′) is the
      BOOTSTRAP φ(s_T), *not* 0. Forcing 0 here would hand the policy a large spurious negative reward
      for the crime of the rollout ending — the classic PBRS bug. The episode's shaping simply
      continues into the next rollout.

    ``TimeLimit.truncated`` (the env's 250-turn deadline) arrives as ``done=True`` and therefore takes
    the TERMINAL branch. In this project that is arguably correct rather than an approximation: the
    250-turn cap IS the forfeit deadline (``StallConfig.threshold`` imports ``MAX_TURNS``) and the reward
    manager scores it as a real outcome. The behaviour is pinned by a test either way, so a future change
    to the timeout's semantics fails loudly instead of silently changing the shaping.
    """
    phi = np.asarray(phi, dtype=np.float64)
    episode_starts = np.asarray(episode_starts, dtype=np.float64)
    n_steps, n_envs = phi.shape
    if episode_starts.shape != (n_steps, n_envs):
        raise ValueError(f"episode_starts {episode_starts.shape} must match phi {phi.shape}")
    phi_bootstrap = np.asarray(phi_bootstrap, dtype=np.float64).reshape(-1)
    last_episode_starts = np.asarray(last_episode_starts, dtype=np.float64).reshape(-1)
    if phi_bootstrap.shape != (n_envs,) or last_episode_starts.shape != (n_envs,):
        raise ValueError("phi_bootstrap and last_episode_starts must both be [n_envs]")

    phi_next = np.empty_like(phi)
    # Rows 0..T-2: the successor is the next row's state, unless that row STARTS a new episode — the
    # identical `1.0 - episode_starts[step + 1]` test SB3's own `compute_returns_and_advantage` uses for
    # `next_non_terminal`, so the shaping's notion of "terminal" and GAE's cannot drift apart.
    if n_steps > 1:
        cont = 1.0 - (episode_starts[1:] >= 0.5).astype(np.float64)
        phi_next[:-1] = phi[1:] * cont
    # The final row: bootstrap where the episode continued, 0 where it ended.
    phi_next[-1] = phi_bootstrap * (1.0 - (last_episode_starts >= 0.5).astype(np.float64))
    return phi_next


def pbrs_shaping(phi: np.ndarray, phi_next: np.ndarray, gamma: float, coef: float) -> np.ndarray:
    """The shaping term ``coef · (γ·φ(s′) − φ(s))``, elementwise. γ is the RUN's own discount.

    There is deliberately no separate PBRS discount: a shaping γ that differs from the return γ breaks
    the telescoping identity, which is the entire shield.
    """
    return float(coef) * (float(gamma) * np.asarray(phi_next, dtype=np.float64)
                          - np.asarray(phi, dtype=np.float64))


def episode_shaping_sum(
    shaping: np.ndarray, episode_starts: np.ndarray, gamma: float, env_index: int = 0
) -> list:
    """DIAGNOSTIC / TEST helper: the γ-discounted shaping sum of each COMPLETE episode in one env column.

    Returns ``[(start_row, end_row, discounted_sum), ...]`` for every episode that both starts and ends
    inside the buffer. The invariance identity says each sum must equal ``−coef·φ(s_start)``; the
    telescoping unit test asserts exactly that, which is the cheapest possible proof that the terminal
    convention above is implemented rather than merely described.
    """
    shaping = np.asarray(shaping, dtype=np.float64)
    es = np.asarray(episode_starts, dtype=np.float64)[:, env_index] >= 0.5
    col = shaping[:, env_index]
    n = col.shape[0]
    starts = [t for t in range(n) if es[t]]
    out = []
    for i, s in enumerate(starts):
        # The episode ends the row before the NEXT start; an episode with no next start is still
        # in progress at the buffer boundary and is not complete.
        if i + 1 >= len(starts):
            continue
        e = starts[i + 1] - 1
        disc = np.array([gamma ** k for k in range(e - s + 1)], dtype=np.float64)
        out.append((s, e, float((col[s:e + 1] * disc).sum())))
    return out


# ──────────────────────────────────────────────────────────────────────────────────────────────
# THE φ READ — one batched no_grad forward over the buffer, transport-agnostic.
# ──────────────────────────────────────────────────────────────────────────────────────────────

def _phi_from_logits(logits: Optional[th.Tensor], where: str) -> np.ndarray:
    if logits is None:
        raise WinProbPbrsError(
            f"--win-prob-pbrs-coef is non-zero but the extractor left no `last_win_prob_logits` "
            f"({where}). The PBRS potential IS the win-prob head, so this needs "
            f"--win-prob-mode read_only|shaping. Refusing to train with a shaping term that "
            f"silently does nothing.")
    return th.sigmoid(logits.detach()).reshape(-1).to(th.float64).cpu().numpy()


def phi_model(model):
    """WHICH network supplies φ — the FROZEN source if one was attached, else the live model.

    `--win-prob-pbrs-source <ckpt>` (`gen3_winprob_pbrs_source_v1`) attaches a frozen foreign model
    at `model._winprob_phi_source`; `main.train.model_build` owns the loading. Absent ⇒ None ⇒ the
    live model, i.e. the shipped v104 behaviour, byte-identical.

    WHY IT MATTERS: the invariance theorem above assumes φ is a FIXED function of state. Our live
    head is a module inside the network being trained, so exact invariance holds only WITHIN a
    rollout and degrades across them. A frozen source removes that caveat entirely — the whole
    reason the flag exists.

    ⚠️ A FULL frozen forward is REQUIRED; there is no head-only shortcut. `WinProbHead.forward`
    consumes `value_pooled`, the whole-board value pool produced by that network's OWN trunk with
    its OWN weights. Running the frozen HEAD over the LIVE trunk's pooled features would compute a
    function of a representation the head never saw AND would drift with the live trunk, destroying
    the exact property the frozen source buys. The forward REPLACES the live-φ one rather than
    adding to it, so the cost is unchanged from the live-φ path (plus one frozen extractor of
    memory, the `--distill-teacher` class).
    """
    return getattr(model, "_winprob_phi_source", None) or model


def _forward_phi(model, obs_batch) -> np.ndarray:
    """One `no_grad` extractor forward over ``obs_batch`` (a dict of numpy arrays) → φ [B] float64.

    ``model`` here is the φ NETWORK (`phi_model(...)`'s answer), not necessarily the trainee.

    Reads the SIDE stash `features_extractor.last_win_prob_logits` — the same seam
    `search_dividend.search.batch_scores` reads, and the same one the aux BCE reads in `train()`.
    """
    with th.no_grad():
        obs_t = obs_as_tensor(obs_batch, model.device)
        model.policy.predict_values(obs_t)   # runs the extractor; the win-prob logit is a side stash
        fe = getattr(model.policy, "features_extractor", None)
        return _phi_from_logits(getattr(fe, "last_win_prob_logits", None), "buffer forward")


def _phi_obs(phi_net, obs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Restrict ``obs`` to the keys the φ network's observation space declares.

    A frozen source may be a PRIOR-GENERATION checkpoint with an older Dict obs space (that is what
    makes a mature φ available at all — `load_foreign_opponent` validates the obs FAMILY, not the
    exact key set). SB3's `preprocess_obs` iterates the passed keys against the space, so an extra
    key is a KeyError. The identical filter the exploiter-distillation teachers use in `train()`.
    A live-model φ has its own space by construction, so this is the identity there.
    """
    space = getattr(getattr(phi_net, "observation_space", None), "spaces", None)
    if not space:
        return obs
    return {k: v for k, v in obs.items() if k in space}


def buffer_potentials(model, rollout_buffer, chunk: int = PHI_FORWARD_CHUNK) -> np.ndarray:
    """φ for every row of the buffer, ``[n_steps, n_envs]`` float64. `no_grad`, chunked, eval-mode."""
    obs = rollout_buffer.observations
    if not isinstance(obs, dict):
        raise WinProbPbrsError(
            "--win-prob-pbrs-coef requires the Dict observation buffer (Gen3Env's "
            "{observation, action_mask, ...}); this buffer stores a plain array.")
    keys = list(obs.keys())
    n_steps, n_envs = rollout_buffer.buffer_size, rollout_buffer.n_envs
    flat = {k: np.asarray(obs[k]).reshape((n_steps * n_envs,) + np.asarray(obs[k]).shape[2:])
            for k in keys}
    out = np.empty(n_steps * n_envs, dtype=np.float64)
    total = n_steps * n_envs
    step = max(1, int(chunk))
    phi_net = phi_model(model)
    for lo in range(0, total, step):
        hi = min(lo + step, total)
        out[lo:hi] = _forward_phi(phi_net, _phi_obs(phi_net, {k: flat[k][lo:hi] for k in keys}))
    return out.reshape(n_steps, n_envs)


# ──────────────────────────────────────────────────────────────────────────────────────────────
# THE INTEGRATION — called from InstrumentedMaskablePPO.collect_rollouts when the coef is non-zero.
# ──────────────────────────────────────────────────────────────────────────────────────────────

def apply_winprob_pbrs(model, rollout_buffer) -> Dict[str, float]:
    """Add the PBRS term to this rollout's rewards and RE-RUN GAE. Returns TB metrics.

    Order, and every step of it is load-bearing:

    1. read φ for the whole buffer (collection-time weights — `train()` has not run yet);
    2. one forward on ``model._last_obs`` gives BOTH the GAE bootstrap ``last_values`` and the
       bootstrap potential φ(s_T) — the same call and the same tensor the collector itself used, so
       the recomputed advantages are the shaped-stream counterpart of the ones it produced;
    3. build φ(s′) with the terminal / truncation conventions;
    4. add ``coef·(γ·φ(s′) − φ(s))`` to ``rollout_buffer.rewards`` IN PLACE (raw reward space —
       PopArt has not run; it reads `returns` at the top of `train()`);
    5. recompute returns and advantages from the shaped rewards.

    The metrics are quoted against the reward stream they perturb: `reward_share` is the mean absolute
    shaping over the mean absolute unshaped reward, which is the number that says whether a coefficient
    is sane — a raw magnitude alone does not.
    """
    coef = float(getattr(model, "win_prob_pbrs_coef", 0.0) or 0.0)
    if coef == 0.0:                       # defensive: the caller gates on this too
        return {}
    gamma = float(model.gamma)

    phi = buffer_potentials(model, rollout_buffer)

    # The post-rollout observation gives TWO different things, and WHICH network produces each is
    # load-bearing: `last_values` is the GAE bootstrap and must come from the LIVE critic (it is the
    # same call and the same tensor the collector itself used, so the recomputed advantages are the
    # shaped-stream counterpart of the ones it produced), while φ(s_T) must come from whatever
    # network supplies φ everywhere else. With no frozen source the two are the same model and this
    # stays ONE forward, exactly as shipped. With one, the frozen bootstrap gets its own forward —
    # a frozen φ on the buffer rows and a LIVE φ on the last row would break the telescoping at
    # every truncation boundary.
    phi_net = phi_model(model)
    with th.no_grad():
        last_obs_t = obs_as_tensor(model._last_obs, model.device)
        last_values = model.policy.predict_values(last_obs_t)
        if phi_net is model:
            fe = getattr(model.policy, "features_extractor", None)
            phi_boot = _phi_from_logits(getattr(fe, "last_win_prob_logits", None),
                                        "bootstrap forward")
        else:
            phi_boot = _forward_phi(phi_net, _phi_obs(phi_net, model._last_obs))
    dones = np.asarray(model._last_episode_starts, dtype=np.float64).reshape(-1)

    phi_next = successor_potential(phi, rollout_buffer.episode_starts, phi_boot, dones)
    shaping = pbrs_shaping(phi, phi_next, gamma, coef)

    raw_absmean = float(np.abs(rollout_buffer.rewards).mean())
    rollout_buffer.rewards += shaping.astype(rollout_buffer.rewards.dtype)
    rollout_buffer.compute_returns_and_advantage(last_values=last_values, dones=dones)

    shaping_absmean = float(np.abs(shaping).mean())
    return {
        "shaping_mean": float(shaping.mean()),
        "shaping_absmean": shaping_absmean,
        "phi_mean": float(phi.mean()),
        # Against the UNSHAPED stream, so the ratio does not flatter itself as the coefficient rises.
        "reward_share": float(shaping_absmean / raw_absmean) if raw_absmean > 0.0 else 0.0,
    }
