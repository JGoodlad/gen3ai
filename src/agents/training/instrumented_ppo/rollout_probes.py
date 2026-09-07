"""`RolloutProbes` — rollout collection, the entropy-boost schedule, and the episode-start read.

Everything `InstrumentedMaskablePPO` does per ROLLOUT rather than per minibatch. None of it is
part of `train()`'s fold sequence: `collect_rollouts` runs before `train()` is called at all, the
two entropy-boost accessors are a pure schedule the loop reads, and `_winprob_start_metrics` is a
read-only probe the metrics export publishes. They live here so `ppo.py` holds the fold and its
contract and nothing else.
"""
import contextlib

import numpy as np
import torch as th

from agents.training.async_vec_env import AsyncSubprocVecEnv, collect_rollouts_async
from agents.training import frozen_phi          # gen3_frozen_phi_actor_only_v1 (both seams live there)
from agents.training.instrumented_ppo.calibration import (   # the MODULE path, never the hub:
    as_numpy as _calib_as_numpy,                              # a submodule importing the package
    episode_start_rows as _calib_episode_start_rows,          # __init__ back closes the import
    sigmoid as _calib_sigmoid,                                # cycle `ppo` sits at the end of
    start_metrics as _calib_start_metrics,                    # (pinned by the hub-contract test).
)
from agents.training.instrumented_ppo.constants import _WINPROB_START_MAX_ROWS
from agents.training.instrumented_ppo.signal_metrics import (
    OPP_CLASS_SUFFIX as _OPP_CLASS_SUFFIX,
)


class RolloutProbes:
    """Mixin: `collect_rollouts` + the per-rollout probes. Mixed in BEFORE `MaskablePPO`, so the
    `super().collect_rollouts(...)` below reaches upstream."""

    def collect_rollouts(self, env, callback, rollout_buffer, n_rollout_steps, use_masking=True):
        if self._async_rollout and isinstance(env, AsyncSubprocVecEnv):
            ok = collect_rollouts_async(
                self, env, callback, rollout_buffer, n_rollout_steps, use_masking)
        else:
            ok = super().collect_rollouts(env, callback, rollout_buffer, n_rollout_steps, use_masking)
        # +WIN-PROB PBRS (ai_v12 route 1, gen3_winprob_pbrs_v1): coef·(γ·φ(s′) − φ(s)) onto this
        # rollout's rewards, then RE-RUN GAE, φ = the DETACHED win-prob head. HERE — after collection,
        # before train() — is the one window between GAE and PopArt's read of `returns` (both
        # collectors; see winprob_pbrs.py). At coef 0 (default) not even the import runs.
        if ok and float(getattr(self, "win_prob_pbrs_coef", 0.0) or 0.0) != 0.0:
            from agents.training.winprob_pbrs import apply_winprob_pbrs
            self._pbrs_metrics = apply_winprob_pbrs(self, rollout_buffer)
        frozen_phi.shape_after_rollout(self, rollout_buffer, ok)   # --win-prob-pbrs-frozen
        return ok

    def _annealed_entropy_boost(self, B: float, af: float) -> float:
        """The state-conditioned entropy-boost multiplier at the CURRENT step. Constant `B` if the anneal
        fraction is 0; else linearly annealed toward 1.0, reaching 1.0 once `af` of training has elapsed
        (uses SB3's `_current_progress_remaining`, which runs 1.0 at the start → 0.0 at the end). Shared by
        the defensive (`gen3_defensive_entropy_v1`) and bait (`gen3_bait_entropy_v1`) boosts — ONE schedule,
        so the two flags can never drift apart. Pure → unit-testable."""
        B, af = float(B), float(af)
        if af <= 0.0 or B == 1.0:
            return B
        done = 1.0 - float(getattr(self, "_current_progress_remaining", 1.0))   # 0 → 1 over training
        return 1.0 + (B - 1.0) * max(0.0, 1.0 - done / af)

    def _defensive_entropy_boost_eff(self) -> float:
        """gen3_defensive_entropy_v1: this run's defensive boost at the current step."""
        return RolloutProbes._annealed_entropy_boost(
            self, self.defensive_entropy_boost, self.defensive_entropy_anneal_frac)

    def _bait_entropy_boost_eff(self) -> float:
        """gen3_bait_entropy_v1: this run's bait boost at the current step (same schedule)."""
        return RolloutProbes._annealed_entropy_boost(
            self, self.bait_entropy_boost, self.bait_entropy_anneal_frac)

    def _winprob_start_metrics(self, head_on: bool) -> dict:
        """`win_prob/start_*` — the head's P(win) at each EPISODE-START row of this rollout, paired
        with that episode's own realized outcome (`gen3_winprob_calibration_export_v1`).

        The pairing is the point. `win_target` is back-filled by `WinProbLabelCallback` from the
        episode's outcome to EVERY step of that episode, so at an episode-start row it IS what that
        game went on to do — the prediction and the realization come from one set of episodes, and
        `start_gap` is a paired difference rather than the difference of two independent windows.
        At the opening board a miscalibration cannot be excused by a lost position, which is what
        makes this the readable calibration point for "does the head's 0.5 mean 0.5".

        The per-opponent-class split is OPPORTUNISTIC: it needs the `opp_class` obs key, which the
        env emits only alongside the opponent-intent labels. Without it the pooled read still
        ships, and `signal/outcome_win_rate_<kind>` carries the realized per-class rate
        unconditionally.

        Read-only and best-effort: any failure returns `{}` rather than taking down a diagnostic's
        host. Returns `{}` when the head is off, when the buffer holds no complete episode, or when
        the win-prob label keys are absent.
        """
        if not head_on:
            return {}
        try:
            buf = self.rollout_buffer
            obs = getattr(buf, "observations", None)
            if not isinstance(obs, dict) or "win_target" not in obs or "win_mask" not in obs:
                return {}
            rows = _calib_episode_start_rows(
                buf.episode_starts, int(buf.buffer_size), int(buf.n_envs))
            if rows.size == 0:
                return {}
            # A rollout can hold thousands of episode starts at production n_envs; the read is a
            # mean, so a bounded prefix is the same measurement at a fixed cost. Deterministic
            # (the first rows in env-major order), never sampled — a diagnostic that moves because
            # of its own RNG is one nobody can compare across arms.
            if rows.size > _WINPROB_START_MAX_ROWS:
                rows = rows[:_WINPROB_START_MAX_ROWS]
            y = np.asarray(obs["win_target"], dtype=np.float64).reshape(-1)[rows]
            m = np.asarray(obs["win_mask"], dtype=np.float64).reshape(-1)[rows]
            if not (m > 0.5).any():                  # only in-progress episodes — nothing realized
                return {}
            fe = self.policy.features_extractor
            ob = th.as_tensor(obs["observation"][rows]).to(self.device)
            dbg_ctx = getattr(fe, "suppress_observation_debugger", contextlib.nullcontext)()
            with dbg_ctx, th.no_grad():
                type(fe).forward(fe, {"observation": ob})
                z = getattr(fe, "last_win_prob_logits", None)
            if z is None:
                return {}
            p = _calib_sigmoid(_calib_as_numpy(z).reshape(-1))
            if p.size != y.size:                     # pragma: no cover - defensive
                return {}
            cls = None
            if "opp_class" in obs:
                cls = np.asarray(obs["opp_class"]).reshape(-1)[rows]
            return _calib_start_metrics(p, y, m, opp_class=cls, class_names=_OPP_CLASS_SUFFIX)
        except Exception:                            # pragma: no cover - a probe never kills a run
            return {}
