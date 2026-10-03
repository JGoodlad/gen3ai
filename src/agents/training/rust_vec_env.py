"""``RustVecEnv`` — the trainer's env (M5 Lane G; the Rust env core is the only env core).

The PPO model holds an env for three things: its SPACES (the model's observation / action space,
checked on a load), ``num_envs`` (the buffer's columns), and the ENV SURFACE its callbacks call
(``env_method`` pushes and pulls). It does NOT step it: the rollout is ``RustCollector.collect``
(``InstrumentedMaskablePPO.collect_rollouts`` routes there), so ``step_async`` / ``step_wait`` are a typed
refusal. It is a `trainer_env.TrainerVecEnv` — the learner's declared env surface — and no longer an sb3
``VecEnv`` (deletion pass U4): nothing here is sb3's.

STARTUP is two-phase, both before the first rollout (the DECLARED LIFECYCLE): the constructor fixes
the spaces and N (what the model needs to be built or loaded); ``startup(model)`` then builds the
env core, the inference service and the arena from the model's OWN policy (T2's slot templates must
be real weights — and must be copied BEFORE ``--compile-trainer`` patches the extractor's forward),
stages and RESETs every env, and attaches the collector to the model.

THE ENV SURFACE — every ``env_method`` the production callbacks call, mapped (the lane table's "71
env_method / get_attr sites"; ``SURFACE`` is the table of record, ``rust_vec_env_test.py`` pins it
against the callers):

    set_self_play_target / set_opponent_win_rates / set_stable_mastered / set_stable_win_rates
        → Lane E's ``RustEnvOpponents`` (a new generation LOADS new snapshots into free T2 slots)
    opponent_default_stats      → per env (p2 policy decisions served, 0, 0): T2 never defaults
    drain_reward_terms          → the collector's ``reward/`` accumulator (env 0; None elsewhere)
    drain_team_wr_counts        → the per-env seeded teambuilders' tables (``TeamStager``)

Anything else is a typed ``RustEnvSurfaceError`` naming the method — a callback that reaches for an
unmapped method fails at the call, never silently gets ``None``. (Every ``env_method`` a production
callback calls is in ``SURFACE``: the methods only a Python-core-only flag called — the exploiter
ladder's rung push, team-PFSP's pulls — were deleted with those flags, deletion pass L4; the exploiter
temperature curriculum's push and its win-rate pull, deletion pass P11c.)
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

import numpy as np

from agents.training.trainer_env import TrainerVecEnv


class RustEnvSurfaceError(RuntimeError):
    """A VecEnv call the Rust env does not serve (named)."""


#: method -> what serves it (the table of record; see the module docstring).
SURFACE: Dict[str, str] = {
    "set_self_play_target": "RustEnvOpponents.set_self_play_target (a generation LOADS into free T2 slots)",
    "set_opponent_win_rates": "RustEnvOpponents.set_opponent_win_rates (the pool's PFSP weights)",
    "set_stable_mastered": "RustEnvOpponents.set_stable_mastered",
    "set_stable_win_rates": "RustEnvOpponents.set_stable_win_rates",
    "opponent_default_stats": "per env (policy p2 decisions served, 0 defaults, 0 re-decides)",
    "drain_reward_terms": "the collector's reward/ accumulator (the terminal is the one term)",
    "drain_team_wr_counts": "TeamStager.drain_team_wr_counts (per-env seeded teambuilders)",
}


class RustVecEnv(TrainerVecEnv):
    """See the module docstring. ``build(model) -> RustCollector`` does the startup's heavy half."""

    def __init__(self, *, n_envs: int, observation_space: Any, action_space: Any,
                 build: Callable[[Any], Any], describe: str = ""):
        self._build = build
        self.collector: Optional[Any] = None
        self.describe = describe
        super().__init__(int(n_envs), observation_space, action_space)

    # ---- startup
    def startup(self, model: Any) -> Any:
        if self.collector is not None:
            raise RustEnvSurfaceError("RustVecEnv.startup ran twice (a declared-lifecycle startup runs once)")
        col = self._build(model)
        col.start()
        self.collector = col
        model._rust_collector = col
        return col

    # ---- the TrainerVecEnv contract
    def reset(self) -> Dict[str, np.ndarray]:
        """``_setup_learn`` asks for a first observation it never uses here (the collector owns every
        row); the core's RESET is part of ``startup``. Zeros of the declared spaces."""
        return {k: np.zeros((self.num_envs, *sp.shape), dtype=sp.dtype)
                for k, sp in self.observation_space.spaces.items()}

    def step_async(self, actions: np.ndarray) -> None:
        raise RustEnvSurfaceError("RustVecEnv is not stepped: the rollout is RustCollector.collect "
                                  "(InstrumentedMaskablePPO.collect_rollouts routes there)")

    def step_wait(self) -> Any:
        raise RustEnvSurfaceError("RustVecEnv.step_wait: see step_async")

    def close(self) -> None:
        if self.collector is not None:
            self.collector.close()

    # ---- the env surface
    def env_method(self, method_name: str, *method_args: Any, indices: Any = None, **method_kwargs: Any) -> List[Any]:
        if method_name not in SURFACE:
            raise RustEnvSurfaceError(f"RustVecEnv.env_method({method_name!r}) is not served "
                                      f"(served: {sorted(SURFACE)}); map it in rust_vec_env.SURFACE first")
        col = self.collector
        if col is None:
            raise RustEnvSurfaceError(f"env_method({method_name!r}) before startup")
        idx = list(self._indices(indices))
        fn = getattr(self, f"_m_{method_name}")
        out = fn(col, *method_args, **method_kwargs)
        return [out[i] for i in idx]

    def _all(self, value: Any) -> List[Any]:
        return [value] * self.num_envs

    def _m_set_self_play_target(self, col: Any, fraction: float, generation: int) -> List[Any]:
        col.opponents.set_self_play_target(float(fraction), int(generation))
        return self._all(None)

    def _m_set_opponent_win_rates(self, col: Any, rates: Any) -> List[Any]:
        col.opponents.set_opponent_win_rates(rates)
        return self._all(None)

    def _m_set_stable_mastered(self, col: Any, labels: Any) -> List[Any]:
        col.opponents.set_stable_mastered(labels)
        return self._all(None)

    def _m_set_stable_win_rates(self, col: Any, rates: Any) -> List[Any]:
        col.opponents.set_stable_win_rates(rates)
        return self._all(None)

    def _m_opponent_default_stats(self, col: Any) -> List[Any]:
        return [(int(n), 0, 0) for n in col.p2_policy_by_env]

    def _m_drain_reward_terms(self, col: Any) -> List[Any]:
        out: List[Any] = self._all(None)
        out[0] = col.reward_terms.drain()
        return out

    def _m_drain_team_wr_counts(self, col: Any) -> List[Any]:
        return col.stager.drain_team_wr_counts()
