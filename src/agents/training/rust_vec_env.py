"""``RustVecEnv`` — the trainer's VecEnv when ``--env-core rust`` (M5 Lane G).

The PPO model holds a VecEnv for three things: its SPACES (the model's observation / action space,
checked on a load), ``num_envs`` (the buffer's columns), and the ENV SURFACE its callbacks call
(``env_method`` pushes and pulls). It does NOT step it: under ``--env-core rust`` the rollout is
``RustCollector.collect`` (``InstrumentedMaskablePPO.collect_rollouts`` routes there), so
``step_async`` / ``step_wait`` are a typed refusal.

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
    set_exploiter_temperature   → the opponent server's temperature (sampling is the host's)
    opponent_default_stats      → per env (p2 policy decisions served, 0, 0): T2 never defaults
    exploiter_winrate_totals    → per env (games, wins) against the exploiter class
    drain_reward_terms          → the collector's ``reward/`` accumulator (env 0; None elsewhere)
    drain_team_wr_counts        → the per-env seeded teambuilders' tables (``TeamStager``)

Anything else is a typed ``RustEnvSurfaceError`` naming the method — a callback that reaches for an
unmapped method fails at the call, never silently gets ``None``.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
from stable_baselines3.common.vec_env.base_vec_env import VecEnv


class RustEnvSurfaceError(RuntimeError):
    """A VecEnv call the Rust env does not serve (named)."""


class RustEnvAttributeError(RustEnvSurfaceError, AttributeError):
    """``get_attr`` of an attribute the Rust env does not serve — an ``AttributeError`` too, so SB3's
    ``has_attr`` (``is_masking_supported``) answers False instead of crashing."""


#: method -> what serves it (the table of record; see the module docstring).
SURFACE: Dict[str, str] = {
    "set_self_play_target": "RustEnvOpponents.set_self_play_target (a generation LOADS into free T2 slots)",
    "set_opponent_win_rates": "RustEnvOpponents.set_opponent_win_rates (the pool's PFSP weights)",
    "set_stable_mastered": "RustEnvOpponents.set_stable_mastered",
    "set_stable_win_rates": "RustEnvOpponents.set_stable_win_rates",
    "set_exploiter_temperature": "PolicyOpponentServer.set_temperature('exploiter', T)",
    "opponent_default_stats": "per env (policy p2 decisions served, 0 defaults, 0 re-decides)",
    "exploiter_winrate_totals": "per env (games, wins) vs the exploiter class",
    "drain_reward_terms": "the collector's reward/ accumulator (the terminal is the one term)",
    "drain_team_wr_counts": "TeamStager.drain_team_wr_counts (per-env seeded teambuilders)",
}

#: env_method names a callback calls ONLY under a flag ``--env-core rust`` refuses at startup
#: (``combination_checks``' ``env_core_rust_unported_paths``) — method -> that flag.
REFUSED_WITH_FLAG: Dict[str, str] = {
    "set_exploiter_rung": "--exploiter-ladder",
    "exploiter_rung_totals": "--exploiter-ladder",
    "get_team_pfsp_keys": "--team-pfsp",
    "drain_team_pfsp_counts": "--team-pfsp",
    "set_team_pfsp_weights": "--team-pfsp",
}


class RustVecEnv(VecEnv):
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

    # ---- the VecEnv contract
    def reset(self) -> Dict[str, np.ndarray]:
        """SB3's ``_setup_learn`` asks for a first observation it never uses here (the collector owns
        every row); the core's RESET is part of ``startup``. Zeros of the declared spaces."""
        return {k: np.zeros((self.num_envs, *sp.shape), dtype=sp.dtype)
                for k, sp in self.observation_space.spaces.items()}

    def step_async(self, actions: np.ndarray) -> None:
        raise RustEnvSurfaceError("RustVecEnv is not stepped by SB3: the rollout is RustCollector.collect "
                                  "(InstrumentedMaskablePPO.collect_rollouts routes there under --env-core rust)")

    def step_wait(self) -> Any:
        raise RustEnvSurfaceError("RustVecEnv.step_wait: see step_async")

    def close(self) -> None:
        if self.collector is not None:
            self.collector.close()

    def get_attr(self, attr_name: str, indices: Any = None) -> List[Any]:
        if attr_name == "render_mode":
            return [None] * len(self._indices(indices))
        raise RustEnvAttributeError(f"RustVecEnv.get_attr({attr_name!r}) is not served (only render_mode)")

    def set_attr(self, attr_name: str, value: Any, indices: Any = None) -> None:
        raise RustEnvSurfaceError(f"RustVecEnv.set_attr({attr_name!r}) is not served")

    def env_is_wrapped(self, wrapper_class: Any, indices: Any = None) -> List[bool]:
        return [False] * len(self._indices(indices))

    def _indices(self, indices: Any) -> Sequence[int]:
        if indices is None:
            return range(self.num_envs)
        if isinstance(indices, int):
            return [indices]
        return list(indices)

    # ---- the env surface
    def env_method(self, method_name: str, *method_args: Any, indices: Any = None, **method_kwargs: Any) -> List[Any]:
        if method_name not in SURFACE:
            raise RustEnvSurfaceError(f"RustVecEnv.env_method({method_name!r}) is not served under --env-core rust "
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

    def _m_set_exploiter_temperature(self, col: Any, temperature: float) -> List[Any]:
        if col.server is not None and "exploiter" in col.server.temperature:
            col.server.set_temperature("exploiter", float(temperature))
        return self._all(None)

    def _m_opponent_default_stats(self, col: Any) -> List[Any]:
        return [(int(n), 0, 0) for n in col.p2_policy_by_env]

    def _m_exploiter_winrate_totals(self, col: Any) -> List[Any]:
        return [(int(g), float(w)) for g, w in zip(col.exploiter_games, col.exploiter_wins)]

    def _m_drain_reward_terms(self, col: Any) -> List[Any]:
        out: List[Any] = self._all(None)
        out[0] = col.reward_terms.drain()
        return out

    def _m_drain_team_wr_counts(self, col: Any) -> List[Any]:
        return col.stager.drain_team_wr_counts()
