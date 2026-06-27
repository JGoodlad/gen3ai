"""Async, non-barrier rollout collection for MaskablePPO (`--async-rollout`).

The throughput problem this solves: stock ``SubprocVecEnv.step()`` is a **barrier** — it sends
actions to all N workers, then blocks until *every* one returns (`step_wait` recv's all remotes).
A slow env (a long battle turn, a heavy self-play opponent forward, oversubscription jitter)
stalls the whole batch each step, and the policy forward + env stepping never overlap. On this box
(16 cores, n_envs up to 64) the rollout is latency-bound and the trainer sits idle most of the
wall-clock (py-spy: ~86% rollout, GPU ~86% idle).

This module replaces the per-step barrier with a **ready-subset** collector:

  * Every worker is kept continuously in-flight (exactly one outstanding step).
  * The policy forwards whichever envs are READY (a dynamic-size batch) and dispatches their next
    action immediately — it never waits for the slowest env.
  * Each env fills its OWN column of the rollout buffer; collection ends when every column has
    ``n_steps`` transitions.

**It stays exactly ON-POLICY.** PPO freezes the policy during ``collect_rollouts`` (no gradient
step until the buffer is full), so every action in a rollout still comes from the *same* frozen
policy — this is purely a scheduling change (overlap GPU forward with CPU stepping, drop the
max-latency-per-step barrier), NOT an algorithm change like APPO/IMPALA (which introduce policy-lag
and need V-trace). Per-env trajectories stay contiguous, so per-column GAE is unchanged.

Two integration details make it correct + cheap:
  1. **Masks ride in ``info``.** Stock collection does a separate ``get_action_masks(env)`` barrier
     each step; async would need a per-env mask as each env becomes ready. Instead the env wrapper
     (``MaskableAgentWrapper``, when ``emit_action_mask=True``) bundles each obs's action mask into
     ``info["action_mask"]`` / the reset ``info``, so the collector gets it with the step result —
     no extra round-trip. (One ``get_action_masks`` barrier still happens once per rollout, for the
     carried-over ``_last_obs``.)
  2. **``env_method`` is drain-safe.** The eval callback calls ``env.env_method(...)``
     (set_self_play_target / opponent_default_stats) from inside ``on_step``,
     which fires mid-collection while some pipes hold un-recv'd step results. Sending an
     ``env_method`` command then would interleave with those results and desync. ``AsyncSubprocVecEnv``
     overrides ``env_method`` / ``get_attr`` / ``set_attr`` to first **stash** every in-flight step
     result, so the barrier RPC sees clean pipes; the collector reads stashed results transparently.

Flag-guarded: only used when ``--async-rollout`` is set (and not ``--debug``); the default path is
the unchanged stock ``SubprocVecEnv`` + ``MaskablePPO.collect_rollouts``.
"""

from __future__ import annotations

from multiprocessing.connection import wait as mp_wait

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.utils import obs_as_tensor
from stable_baselines3.common.vec_env import SubprocVecEnv
from sb3_contrib.common.maskable.utils import is_masking_supported


class AsyncSubprocVecEnv(SubprocVecEnv):
    """``SubprocVecEnv`` that supports per-env async stepping + drain-safe RPC.

    Adds ``send_step`` / ``poll_ready`` / ``recv_step`` (used by ``collect_rollouts_async``) and
    makes ``env_method`` / ``get_attr`` / ``set_attr`` stash any in-flight step results first so a
    barrier RPC can't interleave with an outstanding ``step`` on the same pipe. The stock
    ``step_async`` / ``step_wait`` (barrier) still work unchanged (e.g. for ``reset`` / non-async
    fallbacks).
    """

    def __init__(self, env_fns, start_method=None):
        super().__init__(env_fns, start_method=start_method)
        self._conn_to_idx = {remote: i for i, remote in enumerate(self.remotes)}
        self._inflight: set[int] = set()       # env idxs with an un-recv'd step result
        self._stash: dict[int, tuple] = {}      # env_idx -> drained step result

    # ── per-env async stepping (the collector's interface) ─────────────────────

    def send_step(self, env_idx: int, action) -> None:
        """Dispatch a single env's step (non-blocking); mark it in-flight."""
        self.remotes[env_idx].send(("step", action))
        self._inflight.add(env_idx)

    def poll_ready(self, idxs, timeout: float | None = None) -> list[int]:
        """Return the subset of ``idxs`` whose step result is ready to read (stashed results are
        served immediately, without blocking). Blocks up to ``timeout`` (None = until ≥1 ready)."""
        stashed = [i for i in idxs if i in self._stash]
        if stashed:
            return stashed
        conns = [self.remotes[i] for i in idxs]
        ready = mp_wait(conns, timeout)
        return [self._conn_to_idx[c] for c in ready]

    def recv_step(self, env_idx: int) -> tuple:
        """Read one env's step result (from the stash if it was drained early, else the pipe)."""
        self._inflight.discard(env_idx)
        if env_idx in self._stash:
            return self._stash.pop(env_idx)
        return self.remotes[env_idx].recv()

    def _drain_inflight(self) -> None:
        """Recv every in-flight step result into the stash so the pipes are clean for a barrier RPC.
        Blocking (waits for the slowest in-flight env), but only called on the rare env_method path —
        never in the per-step hot loop."""
        for i in list(self._inflight):
            self._stash[i] = self.remotes[i].recv()
            self._inflight.discard(i)

    # ── drain-safe barrier RPC (called from on_step mid-collection) ────────────

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        self._drain_inflight()
        return super().env_method(method_name, *method_args, indices=indices, **method_kwargs)

    def get_attr(self, attr_name, indices=None):
        self._drain_inflight()
        return super().get_attr(attr_name, indices=indices)

    def set_attr(self, attr_name, value, indices=None):
        self._drain_inflight()
        return super().set_attr(attr_name, value, indices=indices)

    def env_is_wrapped(self, wrapper_class, indices=None):
        self._drain_inflight()
        return super().env_is_wrapped(wrapper_class, indices=indices)


# Phase markers for the collector's per-env state machine.
_READY = 0      # env has a fresh obs to act on
_STEPPING = 1   # env has an outstanding action (result pending)


def collect_rollouts_async(
    model,
    env: AsyncSubprocVecEnv,
    callback,
    rollout_buffer,
    n_rollout_steps: int,
    use_masking: bool = True,
) -> bool:
    """Non-barrier, on-policy rollout collection — drop-in for ``MaskablePPO.collect_rollouts``.

    Fills ``rollout_buffer`` per-env-column (each env contributes ``n_rollout_steps`` contiguous
    transitions) by keeping all workers continuously in-flight and acting on whichever are ready.
    Mirrors the bookkeeping of the stock loop (``num_timesteps``, timeout bootstrap,
    ``_update_info_buffer``, ``_last_obs`` / ``_last_episode_starts`` carry-over, GAE) exactly.

    Targets this project's **Dict observation** — ``{"observation": Box, "action_mask": Box}``
    (``Gen3Env``) — where the per-decision action mask rides natively in the obs. The collector
    reads each env's mask from ``obs["action_mask"]`` (same value the stock path gets from
    ``get_action_masks(env)``: both are ``last_ctx.mask`` for the obs being acted on), so it needs
    no per-env ``env_method`` round-trip and no wrapper change.
    """
    assert model._last_obs is not None, "No previous observation was provided"
    assert isinstance(model.observation_space, spaces.Dict) and "action_mask" in model.observation_space.spaces, (
        "async rollout targets a Dict obs space with an 'action_mask' key (Gen3Env)")
    assert isinstance(rollout_buffer.observations, dict) and hasattr(rollout_buffer, "action_masks"), (
        "async rollout requires a MaskableDictRolloutBuffer")
    if use_masking and not is_masking_supported(env):
        raise ValueError("Environment does not support action masking.")

    n_envs = env.num_envs
    obs_keys = list(model._last_obs.keys())
    action_dim = rollout_buffer.action_dim
    mask_dims = rollout_buffer.mask_dims
    model.policy.set_training_mode(False)
    rollout_buffer.reset()
    callback.on_rollout_start()

    # Per-env current state: the obs to act on (a dict of [n_envs, *shape] arrays) and the
    # episode-start flag for the NEXT transition (carried from _last_*; for the first rollout these
    # are the post-reset values). The mask lives in cur_obs["action_mask"].
    cur_obs = {k: np.array(model._last_obs[k], copy=True) for k in obs_keys}
    cur_estart = np.array(model._last_episode_starts, copy=True).astype(np.float32)  # [n_envs]

    phase = [_READY] * n_envs
    filled = [0] * n_envs                       # transitions written into each env's column
    pend: dict[int, tuple] = {}                 # env_idx -> (obs_dict, action, value, log_prob, mask, estart)
    total = n_envs * n_rollout_steps
    collected = 0

    def _abort_drain():
        """On a callback-requested abort, recv any in-flight results so the pipes are clean for the
        next rollout (no desync)."""
        for i in range(n_envs):
            if phase[i] == _STEPPING:
                env.recv_step(i)
                phase[i] = _READY

    while collected < total:
        # ── READY phase: batch-forward + dispatch every env that still needs transitions ──
        ready = [i for i in range(n_envs) if phase[i] == _READY and filled[i] < n_rollout_steps]
        if ready:
            obs_batch = {k: np.stack([cur_obs[k][i] for i in ready]) for k in obs_keys}
            masks_batch = np.stack([cur_obs["action_mask"][i] for i in ready]) if use_masking else None
            with th.no_grad():
                obs_t = obs_as_tensor(obs_batch, model.device)
                actions_t, values_t, log_probs_t = model.policy(obs_t, action_masks=masks_batch)
            actions_np = actions_t.cpu().numpy()
            values_np = values_t.detach().cpu().numpy().flatten()
            log_probs_np = log_probs_t.detach().cpu().numpy().flatten()
            for k, i in enumerate(ready):
                pend[i] = (
                    {key: np.array(cur_obs[key][i], copy=True) for key in obs_keys},
                    actions_np[k], float(values_np[k]), float(log_probs_np[k]),
                    (np.array(cur_obs["action_mask"][i], copy=True) if use_masking else None),
                    float(cur_estart[i]),
                )
                env.send_step(i, actions_np[k])
                phase[i] = _STEPPING

        # ── WAIT phase: block for ANY in-flight env (non-barrier — whoever's ready) ──
        stepping = [i for i in range(n_envs) if phase[i] == _STEPPING]
        if not stepping:
            break  # nothing in flight and nothing ready (all columns full) → done
        wave_infos, wave_dones = [], []
        for i in env.poll_ready(stepping):
            new_obs, reward, done, info, reset_info = env.recv_step(i)
            phase[i] = _READY
            po, pa, pv, plp, pm, pes = pend.pop(i)

            # Timeout bootstrap (GH #633), identical to the stock loop — only on a truncated done
            # that carried a terminal observation.
            r = float(reward)
            if done and info.get("terminal_observation") is not None and info.get("TimeLimit.truncated", False):
                with th.no_grad():
                    term_t = model.policy.obs_to_tensor(info["terminal_observation"])[0]
                    r += model.gamma * float(model.policy.predict_values(term_t)[0].item())

            t = filled[i]
            for key in obs_keys:
                rollout_buffer.observations[key][t, i] = po[key]
            rollout_buffer.actions[t, i] = np.asarray(pa).reshape(action_dim)
            rollout_buffer.rewards[t, i] = r
            rollout_buffer.episode_starts[t, i] = pes
            rollout_buffer.values[t, i] = pv
            rollout_buffer.log_probs[t, i] = plp
            if use_masking:
                rollout_buffer.action_masks[t, i] = np.asarray(pm).reshape(mask_dims)
            filled[i] += 1
            collected += 1

            # +WIN-PROB: capture the terminal OUTCOME at this env's just-written buffer row (t, i). The
            # async collector owns the per-env row, so it records it inline; the WinProbLabelCallback's
            # wave-batched on_step can't recover (env→row). No-op unless the win-prob head is on (the
            # scratch is allocated only then by the callback's on_rollout_start, run just above).
            if done:
                _win_scr = getattr(model, "_win_terminal_scratch", None)
                if _win_scr is not None and "win_outcome" in info:
                    _win_scr[t, i] = float(info["win_outcome"])

            # Advance this env's current obs/episode-start for its NEXT action. On a done step the
            # worker auto-reset, so new_obs is already the fresh episode's first obs (mask included).
            for key in obs_keys:
                cur_obs[key][i] = new_obs[key]
            cur_estart[i] = 1.0 if done else 0.0
            wave_infos.append(info)
            wave_dones.append(done)

        # Per-wave bookkeeping (mirrors the stock per-step block; a "wave" ≈ a macro-step).
        model.num_timesteps += len(wave_dones)
        model._update_info_buffer(wave_infos, np.array(wave_dones))
        callback.update_locals(locals())
        if not callback.on_step():
            _abort_drain()
            return False

    # ── Finalize: buffer is full; carry env state forward; compute GAE per column ──
    rollout_buffer.pos = n_rollout_steps
    rollout_buffer.full = True
    model._last_obs = cur_obs
    model._last_episode_starts = cur_estart

    with th.no_grad():
        last_values = model.policy.predict_values(obs_as_tensor(cur_obs, model.device))
    rollout_buffer.compute_returns_and_advantage(last_values=last_values, dones=cur_estart)

    callback.on_rollout_end()
    return True
