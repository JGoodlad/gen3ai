"""THE M5 LANE E GATE — slice N with a POLICY opponent (``rust_env_opponents_parity`` is the harness;
its docstring states the method and the bars).

COMMIT tier (routine, CPU): T2's EAGER backend vs the per-env EAGER ``RLPlayer`` path, on a pool of
perturbed fresh snapshots — greedy at the production stall threshold, and SAMPLED at a low threshold
(every episode ends in the stall forfeit) with a mid-run POOL REFRESH that must be a LOAD (the
service's ``*_after_freeze`` counters 0) and whose new snapshot must actually be played. No
allowlist: zero divergences of any kind, zero phantom polls. Teeth: one corrupted recorded action
fails; the wrapper polling the opponent on a step whose order is never sent fails
(``gen3_no_phantom_opponent_poll_v1``).

``slow``: the COMPILED per-env path (``--compile-opponents``). MILESTONE (``slow`` + an idle GPU,
``GEN3AI_TEST_ALLOW_GPU=1``, under ``flock /home/goodlad/.claude/jobs/gpu.lock``): T2's ``graph``
backend on CUDA vs the compiled CPU path on a real production pool.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from unittest.mock import MagicMock

import pytest

from utils.paths import main_models_dir, src_path

pytestmark = [pytest.mark.sim, pytest.mark.integration]

FEATURES = ("--profile", "selfcheck", "--features", "emission-selfcheck")


@pytest.fixture(scope="module")
def built():
    cargo = shutil.which("cargo") or os.path.expanduser("~/.cargo/bin/cargo")
    if not os.path.exists(cargo):
        pytest.fail("cargo is not installed — the Lane E gate cannot run (install rustup)")
    crate = src_path("rust_env")
    env = dict(os.environ, CARGO_TARGET_DIR=str(crate / "target"))
    r = subprocess.run([cargo, "build", "--lib", *FEATURES, "--manifest-path", str(crate / "Cargo.toml")],
                       env=env, capture_output=True, text=True, timeout=1800)
    assert r.returncode == 0, f"building the cdylib failed:\n{r.stderr[-4000:]}"
    return True


def _cfg(**kw):
    base = {"pool": "fresh:2", "n_envs": 4, "threads": 2, "episodes": 10, "mode": "greedy", "device": "cpu",
            "backend": "eager", "buckets": [2, 8], "lanes": 1, "turn_limit": 0, "compile": False,
            "refresh_at": 0, "key_base": 71_000, "teams": "pool"}
    base.update(kw)
    return base


def _assert_clean(summary, rep, *, dlogp_bar):
    assert rep["div"] == {}, (rep["div"], {k: rep.get(k) for k in ("row_examples", "outcome_examples", "p2_count_examples")})
    assert rep["phantom_polls"] == 0, rep["phantom_polls"]
    assert rep["episodes"] == summary["n_episodes"] and rep["decisions"] == summary["p2_decisions"] > 0, (rep, summary)
    assert rep["max_dlogp"] < dlogp_bar, rep["max_dlogp"]
    assert all(v == 0 for v in summary["svc_counters_delta"].values()), summary["svc_counters_delta"]
    assert all(v == 0 for v in summary["core_after_freeze"].values()), summary["core_after_freeze"]


def test_commit_greedy_t2_equals_the_per_env_path(built, tmp_path):
    from agents.training.rust_env_opponents_parity import run

    summary, rep = run(_cfg(episodes=6), str(tmp_path))
    _assert_clean(summary, rep, dlogp_bar=1e-5)
    assert rep["decisions"] >= 120, rep["decisions"]


def test_commit_sampled_with_stall_forfeits_and_a_refresh(built, tmp_path):
    from agents.training.rust_env_opponents_parity import run

    summary, rep = run(_cfg(n_envs=8, episodes=40, mode="sampled", turn_limit=6, refresh_at=20, key_base=71_900),
                       str(tmp_path))
    _assert_clean(summary, rep, dlogp_bar=1e-5)
    assert rep["p1_forfeits"] == rep["episodes"], "every episode must end in the stall forfeit at this threshold"
    ref = summary["refresh"]
    assert ref is not None and len(ref["routes_loaded"]) == 1, ref
    assert summary["route_counts"][ref["routes_loaded"][0]] > 0, ("the refreshed snapshot was never played", summary["route_counts"])
    assert summary["svc_loads"] == 3, summary["svc_loads"]           # 2 at startup + 1 refresh: LOADS, never compiles


def test_the_gate_has_teeth(built, tmp_path):
    """One recorded opponent action moved to another legal action must be caught."""
    import numpy as np

    from agents.training.rust_env_opponents_parity import run

    def mutate(rec):
        for e in rec["episodes"]:
            for d in e.p2:
                legal = np.flatnonzero(d.mask)
                if len(legal) > 1:
                    d.greedy = int(legal[legal != d.greedy][0])
                    return
        raise AssertionError("no decision with two legal actions")

    _summary, rep = run(_cfg(n_envs=2, episodes=2, key_base=71_500), str(tmp_path), mutate=mutate)
    assert sum(rep["div"].get(k, 0) for k in ("greedy", "greedy_neartie")) == 1, rep["div"]


def test_the_wrapper_never_polls_the_opponent_on_a_step_it_does_not_send():
    """gen3_no_phantom_opponent_poll_v1: with ``agent2_to_move`` False the order would be dropped, so
    ``choose_move`` (an embed that records a decision, a sample, a bot's RNG draw) must not run.
    FAILS on revert: the wrapper called it whenever ``battle2`` was not a ``wait``."""
    from poke_env.environment.single_agent_wrapper import SingleAgentWrapper

    env = MagicMock()
    env.agent1.username, env.agent2.username = "a1", "a2"
    env.observation_spaces = {"a1": MagicMock()}
    env.action_spaces = {"a1": MagicMock()}
    env.battle2.wait = False
    env.battle2.teampreview = False
    env.agent2_to_move = False
    env.step.return_value = ({"a1": 0}, {"a1": 0.0}, {"a1": False}, {"a1": False}, {"a1": {}})
    opp = MagicMock()
    w = SingleAgentWrapper(env, opp)
    w._settle_opponent_battle = lambda: None
    w.step(0)
    opp.choose_move.assert_not_called()
    env.agent2_to_move = True
    w.step(0)
    opp.choose_move.assert_called_once()


@pytest.mark.slow
def test_slow_compiled_per_env_path(built, tmp_path):
    from agents.training.rust_env_opponents_parity import run

    summary, rep = run(_cfg(n_envs=8, episodes=24, mode="sampled", compile=True, key_base=72_000), str(tmp_path))
    _assert_clean(summary, rep, dlogp_bar=1e-4)
    assert rep["compiled"]


@pytest.mark.slow
@pytest.mark.parametrize("mode", ["sampled", "greedy"])
def test_milestone_gpu_graph_backend_vs_compiled_cpu_on_a_real_pool(built, tmp_path, mode):
    """Run under ``flock /home/goodlad/.claude/jobs/gpu.lock`` with ``GEN3AI_TEST_ALLOW_GPU=1``."""
    import torch

    if os.environ.get("GEN3AI_TEST_ALLOW_GPU") != "1" or not torch.cuda.is_available():
        pytest.skip("GPU tier: set GEN3AI_TEST_ALLOW_GPU=1 on an idle GPU (under the GPU lock)")
    models = main_models_dir()
    snaps = None if models is None else models / "ai_v14_06_lbat_ctrl_fix" / "snapshots"
    if snaps is None or not snaps.is_dir():
        pytest.skip("no run archive with ai_v14_06_lbat_ctrl_fix/snapshots")
    from agents.training.rust_env_opponents_parity import run

    summary, rep = run(_cfg(pool=f"run:{snaps}:6", device="cuda", backend="graph", buckets=[2, 8, 16], n_envs=16,
                            threads=4, episodes=64, mode=mode, compile=True, refresh_at=120,
                            key_base=73_000 if mode == "sampled" else 74_000), str(tmp_path))
    near = {"greedy_neartie", "sample_neartie", "action_neartie"}
    assert not {k: v for k, v in rep["div"].items() if k not in near}, rep["div"]
    assert rep["phantom_polls"] == 0 and rep["decisions"] == summary["p2_decisions"] > 0
    assert rep["max_dlogp"] < 1e-3, rep["max_dlogp"]
    assert all(v == 0 for v in summary["svc_counters_delta"].values()), summary["svc_counters_delta"]
