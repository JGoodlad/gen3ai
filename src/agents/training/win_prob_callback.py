"""Win-probability label plumbing (`--win-prob-mode read_only|shaping`).

The auxiliary win-probability head (`WinProbHead`, `features_extractor.py`) is supervised by the
Monte-Carlo episode OUTCOME (win=1 / loss=0) — but unlike the belief labels (privileged info known
each step), the outcome is a FUTURE quantity, only known when the battle ends. So it cannot ride as a
per-step obs key the way belief labels do. This callback bridges that gap with **zero new buffer
internals** by reusing the proven obs-dict-label storage path:

1. `Gen3Env` declares two TRAINING-ONLY obs keys — `win_target` and `win_mask` ([1] float32 each) —
   and emits PLACEHOLDER zeros every step (`emit_win_target`). The rollout buffer therefore stores +
   shuffles them automatically, exactly like the belief labels.
2. The terminal OUTCOME of each episode is captured during collection (the env puts it in
   ``info["win_outcome"]`` at the done step; `MaskableAgentWrapper`): this callback records it for the
   SYNC path (in `_on_step`, using ``rollout_buffer.pos``), and the async collector records it inline
   (it owns the per-env buffer row) — both into a shared ``model._win_terminal_scratch`` [n_steps, n_envs].
3. After collection (`_on_rollout_end`, before `train()`), the per-episode outcome is propagated
   backward to every step of its episode and the buffer's ``observations["win_target"]`` /
   ``["win_mask"]`` placeholders are OVERWRITTEN with the MC label + a known-mask. `train()` then reads
   ``rollout_data.observations["win_target"]`` / ``["win_mask"]`` (shuffle-aligned for free) and folds
   the BCE at ``win_prob_coef``.

Transitions in the trailing IN-PROGRESS episode (no terminal yet within the buffer) get ``win_mask=0``
and are excluded from the loss — never trained toward a fabricated label. The win label is **undiscounted**
(γ_win = 1): every step of an episode carries that episode's actual outcome, so P(win|s) is directly
"the probability this state leads to a win" — the interpretable quantity the head exists to give.

Only added to the callback list when the win-prob head is on, so a default run pays nothing.
"""

from __future__ import annotations

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class WinProbLabelCallback(BaseCallback):
    """Captures per-episode win/loss outcomes during rollout collection and back-fills the rollout
    buffer's ``win_target`` / ``win_mask`` obs keys with the Monte-Carlo label before ``train()``."""

    def _scratch(self) -> np.ndarray:
        n_steps = self.model.n_steps
        n_envs = self.model.n_envs
        scr = getattr(self.model, "_win_terminal_scratch", None)
        if scr is None or scr.shape != (n_steps, n_envs):
            scr = np.full((n_steps, n_envs), np.nan, dtype=np.float32)
            self.model._win_terminal_scratch = scr
        return scr

    def _on_rollout_start(self) -> None:
        # Fresh scratch each rollout: NaN = "no terminal captured at this (step, env)". The async
        # collector writes into this same array inline (it runs after on_rollout_start).
        self._scratch().fill(np.nan)

    def _on_step(self) -> bool:
        # SYNC capture only — the async collector records terminals inline (it owns the per-env buffer
        # row; the wave-batched on_step here can't recover each env's row). on_step fires BEFORE
        # rollout_buffer.add() in the stock loop, so buffer.pos is the row about to be written for THIS
        # transition, and infos/dones in locals describe it.
        if getattr(self.model, "_async_rollout", False):
            return True
        buf = self.locals.get("rollout_buffer")
        infos = self.locals.get("infos")
        dones = self.locals.get("dones")
        if buf is None or infos is None or dones is None:
            return True
        t = int(buf.pos)
        if t >= self.model.n_steps:
            return True  # defensive: never index past the buffer
        scratch = self._scratch()
        for env_i, done in enumerate(dones):
            if done and infos[env_i] is not None and "win_outcome" in infos[env_i]:
                scratch[t, env_i] = float(infos[env_i]["win_outcome"])
        return True

    def _on_rollout_end(self) -> None:
        buf = self.model.rollout_buffer
        obs = buf.observations
        if not isinstance(obs, dict) or "win_target" not in obs or "win_mask" not in obs:
            return  # win-prob head on but env not emitting the keys (config mismatch) — skip, don't crash
        scratch = getattr(self.model, "_win_terminal_scratch", None)
        if scratch is None:
            return
        es = buf.episode_starts                       # [n_steps, n_envs] (1.0 = start of a new episode)
        wt = obs["win_target"]                        # [n_steps, n_envs, 1]
        wm = obs["win_mask"]                          # [n_steps, n_envs, 1]
        n_steps, n_envs = scratch.shape
        # Backward scan per timestep (vectorised over envs): carry each episode's terminal outcome back
        # to its earlier steps, resetting at an episode boundary. `known` flags steps whose episode
        # finished within the buffer (so the label is real); the trailing in-progress episode stays 0.
        known = np.zeros(n_envs, dtype=bool)
        val = np.zeros(n_envs, dtype=np.float32)
        for t in range(n_steps - 1, -1, -1):
            s = scratch[t]                            # [n_envs]
            has_terminal = ~np.isnan(s)
            known = known | has_terminal
            val = np.where(has_terminal, s, val).astype(np.float32)
            wt[t, :, 0] = val
            wm[t, :, 0] = known.astype(np.float32)
            # An episode start at step t means the EARLIER (t-1) step belongs to a prior episode.
            starts = es[t] >= 0.5
            known = known & ~starts
            val = np.where(starts, 0.0, val).astype(np.float32)
