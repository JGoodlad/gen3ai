"""`RolloutProbes` — rollout collection and the episode-start read.

Everything `InstrumentedMaskablePPO` does per ROLLOUT rather than per minibatch. None of it is
part of `train()`'s fold sequence: `collect_rollouts` runs before `train()` is called at all, and
`_winprob_start_metrics` is a read-only probe the metrics export publishes. They live here so `ppo.py` holds the fold and its
contract and nothing else.
"""
import time

import numpy as np
import torch as th

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
    """Mixin: `collect_rollouts` (the Rust collector's entry) + the per-rollout probes. Mixed in BEFORE
    `OwnedLoop`."""

    def collect_rollouts(self, env, callback, rollout_buffer, n_rollout_steps, use_masking=True):
        # `rollout/collect_ms` + `rollout/collect_decisions` (M5 Lane G): the wall clock of this
        # collection and the trainee decisions it played. Recorded, never read by anything that trains.
        _t0, _n0 = time.perf_counter(), int(self.num_timesteps)
        # The Rust env core is the ONLY env core (the Python core and its `_collect_python` were
        # deleted — deletion pass U3): the rollout is the Rust collector's (`_collect_rust`), attached
        # by `RustVecEnv.startup(model)` before `learn()`. A learner without one has nothing to collect
        # from, and says so rather than falling into an upstream loop that would step a VecEnv that
        # does not step.
        rc = getattr(self, "_rust_collector", None)
        if rc is None:
            raise RuntimeError(
                "collect_rollouts: this learner has no Rust collector (`_rust_collector`) — the Rust env "
                "core is the only env core, and `RustVecEnv.startup(model)` attaches it before learn()")
        ok = self._collect_rust(rc, callback, rollout_buffer)
        self._record_collect(_t0, _n0)
        return ok

    def _collect_rust(self, rc, callback, rollout_buffer):
        """M5 Lane G: the learner's weights are LOADED into the inference service after every update
        (`after_update`, detected by the update counter moving), then the complete-game (or window)
        fill replaces the model's buffer contents. `rust_rollout/collector.py`."""
        if getattr(rc, "seen_updates", None) is None:
            rc.seen_updates = self._n_updates
        elif self._n_updates != rc.seen_updates:
            rc.after_update(self)
            rc.seen_updates = self._n_updates
        return rc.collect(self, callback, rollout_buffer)

    def _record_collect(self, t0: float, n0: int) -> None:
        logger = getattr(self, "_logger", None)
        if logger is not None:
            logger.record("rollout/collect_ms", 1000.0 * (time.perf_counter() - t0))
            logger.record("rollout/collect_decisions", float(int(self.num_timesteps) - n0))

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
            with th.no_grad():
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
