"""Pins for ``RustVecEnv``'s env surface (M5 Lane G): every ``env_method`` a training callback calls is
SERVED (``SURFACE``); the routing reaches the collector's pieces; an unmapped call is a typed
refusal, and it carries no sb3 ``VecEnv`` surface (deletion pass U4: it is a ``TrainerVecEnv``)."""
from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from gymnasium import spaces

from agents.training import rust_vec_env as R
from utils.paths import src_path


def _callers():
    """Every string literal passed as the first argument of an ``env_method(...)`` call in the training
    sources (tests excluded)."""
    names = {}
    for root in (src_path("agents", "training"), src_path("main", "train")):
        for f in Path(root).rglob("*.py"):
            if f.name.endswith("_test.py"):
                continue
            tree = ast.parse(f.read_text())
            for node in ast.walk(tree):
                if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                        and node.func.attr == "env_method" and node.args
                        and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str)):
                    names.setdefault(node.args[0].value, []).append(f"{f.name}:{node.lineno}")
    return names


def test_every_env_method_a_callback_calls_is_served():
    names = _callers()
    assert "set_self_play_target" in names and "drain_team_wr_counts" in names  # the scan works
    unmapped = {n: w for n, w in names.items() if n not in R.SURFACE}
    assert not unmapped, f"env_method(s) a callback calls that --env-core rust does not serve: {unmapped}"


class _Opp:
    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        return lambda *a: self.calls.append((name, a))


def _env():
    sp = spaces.Dict({"observation": spaces.Box(0, 1, (3,), np.float32)})
    col = SimpleNamespace(
        opponents=_Opp(), p2_policy_by_env=np.array([3, 0]),
        reward_terms=SimpleNamespace(drain=lambda: {"n": 5}),
        stager=SimpleNamespace(drain_team_wr_counts=lambda: [("c0", "k"), ("c1", "k")]),
        start=lambda: None, close=lambda: None)
    env = R.RustVecEnv(n_envs=2, observation_space=sp, action_space=spaces.Discrete(11), build=lambda m: col)
    env.startup(SimpleNamespace())
    return env, col


def test_the_surface_routes_to_the_collector():
    env, col = _env()
    assert env.env_method("set_self_play_target", 0.5, 3) == [None, None]
    assert col.opponents.calls[-1] == ("set_self_play_target", (0.5, 3))
    env.env_method("set_stable_mastered", ["a"])
    assert col.opponents.calls[-1] == ("set_stable_mastered", (["a"],))
    assert env.env_method("opponent_default_stats") == [(3, 0, 0), (0, 0, 0)]
    assert env.env_method("drain_reward_terms") == [{"n": 5}, None]
    assert env.env_method("drain_team_wr_counts", indices=[1]) == [("c1", "k")]


def test_unmapped_calls_are_typed_refusals():
    env, _col = _env()
    with pytest.raises(R.RustEnvSurfaceError, match="set_team_pfsp_weights"):
        env.env_method("set_team_pfsp_weights", [1.0])      # a DELETED lever's method: still a typed refusal
    with pytest.raises(R.RustEnvSurfaceError, match="not stepped"):
        env.step_async(np.zeros(2))
    assert not hasattr(env, "action_masks") and not hasattr(env, "get_attr")   # no sb3 VecEnv surface
    with pytest.raises(R.RustEnvSurfaceError, match="twice"):
        env.startup(SimpleNamespace())
