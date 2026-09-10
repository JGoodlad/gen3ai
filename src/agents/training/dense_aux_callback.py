"""DENSE-AUXILIARY label plumbing (`gen3_dense_aux_v1`, `--win-prob-dense-aux`).

`WinProbLabelCallback`'s twin, and deliberately its exact shape — the same three-step bridge, for
the same reason (the labels are FUTURE quantities, so they cannot ride as per-step obs keys the way
the belief labels do):

1. `Gen3Env` declares three TRAINING-ONLY obs keys (`agents.training.dense_aux`) — `aux_target`
   [25] and `aux_mask` [25], plus `aux_turn` [1]. `aux_target` is a PLACEHOLDER of zeros;
   `aux_mask` is a REAL present-state value (which of the 25 outputs names an entity this state's
   observation actually carries); `aux_turn` is this state's turn number.
2. The END-OF-BATTLE facts are captured at the done step — `info["aux_terminal"]` /
   `["aux_terminal_mask"]` / `["aux_terminal_turn"]`, published by `MaskableAgentWrapper` from the
   trainee's own finished battle — into a shared `model._dense_aux_scratch`, by this callback on
   the SYNC path and inline by the async collector (which owns the per-env buffer row).
3. `_on_rollout_end`, before `train()`, propagates each episode's terminal facts backward to every
   step of that episode, computes the per-state turns-left, and ANDs the terminal-availability mask
   with the env's per-state visibility and the episode-known bit.

Only added to the callback list when the dense-aux head is on, so a default run pays nothing.

🚨 **IT RUNS AFTER `WinProbLabelCallback` AND TOUCHES NOTHING IT OWNS.** The λ recursion overwrites
`win_target` / `win_mask` in place; these keys are disjoint from both, and there is no λ blend of an
end-of-battle fact — see `agents.training.dense_aux`'s module docstring for why that is a design
constraint and not an omission.
"""

from __future__ import annotations

import math

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from agents.model.dense_aux_head import (
    DENSE_AUX_DIM_OUT, DENSE_AUX_MAX_TURNS, DENSE_AUX_TURNS,
)
from agents.training.dense_aux import AUX_MASK_KEY, AUX_TARGET_KEY, AUX_TURN_KEY

#: The keys `MaskableAgentWrapper` publishes at the done step.
INFO_TARGET = "aux_terminal"
INFO_MASK = "aux_terminal_mask"
INFO_TURN = "aux_terminal_turn"


class DenseAuxScratch:
    """The per-rollout terminal capture: `[n_steps, n_envs, 25]` targets + masks, `[n_steps,
    n_envs]` terminal turns. `NaN` in `turn` is the "no terminal captured at this (step, env)"
    sentinel — the same convention `_win_terminal_scratch` uses, and for the same reason a separate
    boolean array is not used: one array cannot get out of step with itself."""

    def __init__(self, n_steps: int, n_envs: int):
        self.shape = (n_steps, n_envs)
        self.target = np.zeros((n_steps, n_envs, DENSE_AUX_DIM_OUT), dtype=np.float32)
        self.mask = np.zeros((n_steps, n_envs, DENSE_AUX_DIM_OUT), dtype=np.float32)
        self.turn = np.full((n_steps, n_envs), np.nan, dtype=np.float32)

    def clear(self) -> None:
        self.target.fill(0.0)
        self.mask.fill(0.0)
        self.turn.fill(np.nan)

    def record(self, t: int, env_i: int, info: dict) -> bool:
        """Capture one env's terminal facts at buffer row `(t, env_i)`. Returns whether it landed."""
        tgt = info.get(INFO_TARGET)
        msk = info.get(INFO_MASK)
        trn = info.get(INFO_TURN)
        if tgt is None or msk is None or trn is None:
            return False
        self.target[t, env_i] = np.asarray(tgt, dtype=np.float32).reshape(DENSE_AUX_DIM_OUT)
        self.mask[t, env_i] = np.asarray(msk, dtype=np.float32).reshape(DENSE_AUX_DIM_OUT)
        self.turn[t, env_i] = float(trn)
        return True


def _scale_turns_left_array(dt: np.ndarray) -> np.ndarray:
    """The vectorised `dense_aux.scale_turns_left`. Kept as one expression rather than a python
    loop over ~130k rows; `dense_aux_targets_test` asserts the two agree elementwise, because two
    spellings of one scale is exactly how a target and its documentation drift apart."""
    return np.minimum(1.0, np.log1p(np.maximum(0.0, dt)) / math.log1p(DENSE_AUX_MAX_TURNS))


class DenseAuxLabelCallback(BaseCallback):
    """Captures per-episode END-OF-BATTLE facts during collection and back-fills the rollout
    buffer's `aux_target` / `aux_mask` obs keys before `train()`."""

    def _scratch(self) -> DenseAuxScratch:
        n_steps, n_envs = self.model.n_steps, self.model.n_envs
        scr = getattr(self.model, "_dense_aux_scratch", None)
        if scr is None or scr.shape != (n_steps, n_envs):
            scr = DenseAuxScratch(n_steps, n_envs)
            self.model._dense_aux_scratch = scr
        return scr

    def _on_rollout_start(self) -> None:
        self._scratch().clear()
        # A STALE `win_prob/aux_*` family would read as a live measurement of the rollout that did
        # not produce it. Cleared here, published only by a back-fill that actually ran.
        self.model._dense_aux_metrics = None

    def _on_step(self) -> bool:
        # SYNC capture only — the async collector records terminals inline (it owns the per-env
        # buffer row). `on_step` fires BEFORE `rollout_buffer.add()` in the stock loop, so
        # `buffer.pos` is the row about to be written for THIS transition.
        if getattr(self.model, "_async_rollout", False):
            return True
        buf = self.locals.get("rollout_buffer")
        infos = self.locals.get("infos")
        dones = self.locals.get("dones")
        if buf is None or infos is None or dones is None:
            return True
        t = int(buf.pos)
        if t >= self.model.n_steps:
            return True                       # defensive: never index past the buffer
        scratch = self._scratch()
        for env_i, done in enumerate(dones):
            if done and isinstance(infos[env_i], dict):
                scratch.record(t, env_i, infos[env_i])
        return True

    def _on_rollout_end(self) -> None:
        buf = self.model.rollout_buffer
        obs = buf.observations
        if not isinstance(obs, dict) or AUX_TARGET_KEY not in obs or AUX_MASK_KEY not in obs:
            return          # head on but env not emitting the keys (config mismatch) — skip
        scratch = getattr(self.model, "_dense_aux_scratch", None)
        if scratch is None:
            return
        at = obs[AUX_TARGET_KEY]                       # [n_steps, n_envs, 25] — placeholder zeros
        am = obs[AUX_MASK_KEY]                         # [n_steps, n_envs, 25] — per-state VISIBILITY
        turn_now = np.asarray(obs.get(AUX_TURN_KEY), dtype=np.float32)
        es = buf.episode_starts                        # [n_steps, n_envs]
        n_steps, n_envs = scratch.shape
        # The env's per-state visibility, taken BEFORE the mask column is overwritten. A copy, not
        # a view: the loop below writes into `am` row by row.
        vis = np.array(am, dtype=np.float32, copy=True)

        known = np.zeros(n_envs, dtype=bool)
        val = np.zeros((n_envs, DENSE_AUX_DIM_OUT), dtype=np.float32)
        tmask = np.zeros((n_envs, DENSE_AUX_DIM_OUT), dtype=np.float32)
        tturn = np.zeros(n_envs, dtype=np.float32)
        for t in range(n_steps - 1, -1, -1):
            has_terminal = ~np.isnan(scratch.turn[t])              # [n_envs]
            known = known | has_terminal
            sel = has_terminal[:, None]
            val = np.where(sel, scratch.target[t], val).astype(np.float32)
            tmask = np.where(sel, scratch.mask[t], tmask).astype(np.float32)
            tturn = np.where(has_terminal, scratch.turn[t], tturn).astype(np.float32)
            at[t] = val
            # TURNS LEFT is the one PER-STATE target: (this episode's terminal turn) − (this
            # state's turn), on the log scale the head is trained in. `turn_now` is a real
            # present-state value the env emitted; an unknown row's entry is masked out below, so
            # the arithmetic there is harmless.
            if turn_now.size:
                at[t, :, DENSE_AUX_TURNS[0]] = _scale_turns_left_array(
                    tturn - turn_now[t, :, 0])
            # THE THREE-WAY AND: the episode finished inside this buffer (`known`), the slot has an
            # end-of-battle fact (`tmask` — an unrevealed opponent slot has none), and the slot
            # names an entity THIS state's observation carries (`vis`).
            am[t] = known[:, None].astype(np.float32) * tmask * vis[t]
            starts = es[t] >= 0.5           # an episode start at t ⇒ t−1 belongs to a prior episode
            known = known & ~starts
            reset = starts[:, None]
            val = np.where(reset, 0.0, val).astype(np.float32)
            tmask = np.where(reset, 0.0, tmask).astype(np.float32)
            tturn = np.where(starts, 0.0, tturn).astype(np.float32)

        # THE PLUMBING METERS, published even when nothing was labelled: an absent `win_prob/aux_*`
        # family must mean "the flag is off", and nothing else. `aux_coverage` is the fraction of
        # rows whose episode finished inside the buffer; `aux_masked_frac` is the fraction of the
        # 25 outputs that are NOT scored on those rows — the honest size of the opponent's
        # unrevealed party plus any short own team, and the number to read before any aux loss.
        n_rows = float(n_steps * n_envs)
        row_known = (am.reshape(n_steps * n_envs, DENSE_AUX_DIM_OUT).max(axis=1) > 0.5)
        n_known = float(row_known.sum())
        scored = float(am.reshape(-1, DENSE_AUX_DIM_OUT)[row_known].sum()) if n_known else 0.0
        self.model._dense_aux_metrics = {
            "aux_active": 1.0,
            "aux_coverage": (n_known / n_rows) if n_rows else 0.0,
            "aux_masked_frac": (1.0 - scored / (n_known * DENSE_AUX_DIM_OUT)) if n_known else 1.0,
        }
