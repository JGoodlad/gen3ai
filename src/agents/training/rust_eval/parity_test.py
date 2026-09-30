"""THE M5 LANE H GATE — the same seed set on both eval paths (``rust_eval.parity`` is the harness; its
docstring states the method, the tie rule and the bars).

COMMIT (routine, CPU): the full nine-bot roster + two perturbed-fresh sentinels, shard units of one
and two games (several units per env, filler games at the tail), T2 EAGER vs the Python worker's eager
CPU forward — every game equal, metrics equal, the kept traces the same files and the same decisions.
Teeth: a Python path on another cycle seed is caught as FATAL game differences; one tampered shard
record is caught as a metric difference; the tie rule excuses only a flip under its margin.

MILESTONE (``slow``): the production eval shape on CPU — nine bots + five sentinels x 25 games, shard 25,
64 envs. GPU MILESTONE (``slow`` + ``GEN3AI_TEST_ALLOW_GPU=1`` under ``scripts/ops/gpu_lock.sh``):
a REAL trainee and five of its pool snapshots (``ai_v14_06_lbat_ctrl_fix``), T2 ``graph`` on CUDA vs the
Python worker's COMPILED CPU forward (``--compile-opponents``' path), bar 1e-3.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from agents.training.rust_eval import parity as PAR

pytestmark = [pytest.mark.sim, pytest.mark.integration]


@pytest.fixture(scope="module")
def built():
    from agents.training.rust_rollout.testkit import build_selfcheck

    build_selfcheck()
    return True


def _assert_pass(rep):
    g = rep["games"]
    assert not g["missing"], g["missing"]
    assert not g["fatal"], g["fatal"][:3]
    assert g["max_dlogp"] <= rep["cfg"]["bar"], g["max_dlogp"]
    if not g["ties"]:
        assert rep["metrics_equal"], rep["metrics_diffs"][:5]
    t = rep["traces"]
    assert "error" not in t, t.get("error")
    assert not t["only_rust"] and not t["only_python"], (t["only_rust"][:3], t["only_python"][:3])
    assert t["checked"] > 0 and not t["decision_diffs"], t["decision_diffs"][:3]
    assert rep["pass"]


@pytest.fixture(scope="module")
def commit_run(built, tmp_path_factory):
    wd = tmp_path_factory.mktemp("laneH_commit")
    return PAR.run({"games": 2, "shard_games": 1, "sentinels": 2, "n_envs": 8, "device": "cpu", "backend": "eager",
                    "bar": PAR.BAR_CPU, "seed": 20260930}, str(wd))


def test_commit_the_same_seed_set_plays_the_same_games_on_both_paths(commit_run):
    _assert_pass(commit_run)
    g = commit_run["games"]
    assert g["equal"] == g["games"] == 22
    assert commit_run["rust_stats"]["lifecycle"] and not any(commit_run["rust_stats"]["lifecycle"].values())


def test_teeth_another_seed_set_on_the_python_side_is_fatal(built, tmp_path):
    """The gate compares GAMES: a Python path on a different seed set must fail it."""
    rep = PAR.run({"games": 2, "shard_games": 2, "sentinels": 0, "n_envs": 4, "bots": ["heuristic", "staller"],
                   "device": "cpu", "backend": "eager", "bar": PAR.BAR_CPU, "seed": 7, "python_seed": 8}, str(tmp_path))
    assert rep["games"]["fatal"] and not rep["pass"]


def test_teeth_the_judge_excuses_only_a_flip_under_the_tie_margin():
    base = {"item": "heuristic", "game": 0, "winner": 1, "end_turn": 30, "actions": [6, 7, 8], "margins": [1.0, 1.0, 1.0],
            "logp": [-0.1, -0.2, -0.3]}
    same = dict(base)
    tie = dict(base, actions=[6, 9, 8], margins=[1.0, 1e-6, 1.0], winner=2)
    fatal = dict(base, actions=[6, 9, 8], winner=2)
    outcome = dict(base, winner=2)
    assert PAR.compare_games([base], [same], 1e-5)["equal"] == 1
    r = PAR.compare_games([dict(base, margins=[1.0, 1e-6, 1.0])], [tie], 1e-5)
    assert len(r["ties"]) == 1 and not r["fatal"]
    assert len(PAR.compare_games([base], [fatal], 1e-5)["fatal"]) == 1
    assert len(PAR.compare_games([base], [outcome], 1e-5)["fatal"]) == 1, "equal actions, different end = FATAL"


def test_teeth_a_tampered_shard_record_is_a_metric_difference(commit_run, tmp_path):
    import shutil

    wd = Path(commit_run["workdir"])
    src_r = wd / "run_rust" / ".eval_runs" / "step_1000"
    src_p = wd / "run_python" / ".eval_runs" / "step_1000"
    r, p = tmp_path / "r", tmp_path / "p"
    shutil.copytree(src_r, r)
    shutil.copytree(src_p, p)
    assert not PAR.compare_metrics(r, p)["diffs"]
    shard = sorted(p.glob("shard__*.json"))[0]
    d = json.loads(shard.read_text())
    d["sum_ep_len"] += 1.0
    shard.write_text(json.dumps(d))
    assert PAR.compare_metrics(r, p)["diffs"]


@pytest.mark.slow
def test_milestone_the_production_eval_shape_on_cpu(built, tmp_path):
    rep = PAR.run({"games": 25, "shard_games": 25, "sentinels": 5, "n_envs": 64, "device": "cpu",
                   "backend": "eager", "bar": PAR.BAR_CPU, "seed": 20261001, "trace_limit": 40}, str(tmp_path))
    _assert_pass(rep)
    assert rep["games"]["games"] == 14 * 25


@pytest.mark.slow
def test_milestone_gpu_graph_backend_vs_the_compiled_python_worker_on_a_real_pool(built, tmp_path):
    if os.environ.get("GEN3AI_TEST_ALLOW_GPU") != "1":
        pytest.skip("GPU milestone: set GEN3AI_TEST_ALLOW_GPU=1 and run under the GPU lock")
    from utils.paths import main_models_dir

    root = main_models_dir()
    run = (root / "ai_v14_06_lbat_ctrl_fix") if root is not None else None
    if run is None or not run.exists():
        pytest.skip("the models archive (ai_v14_06_lbat_ctrl_fix) is not reachable")
    snaps = sorted((run / "snapshots").glob("snapshot_*.zip"))
    trainee = run / "final_model.zip"
    assert len(snaps) >= 5 and trainee.exists()
    picks = [str(snaps[int(i * (len(snaps) - 1) / 4)]) for i in range(5)]
    rep = PAR.run({"games": 20, "shard_games": 25, "sentinels": 5, "n_envs": 64, "device": "cuda",
                   "backend": "graph", "bar": PAR.BAR_GPU, "seed": 20261002, "trainee": str(trainee),
                   "sentinel_paths": picks, "compile_python": True, "trace_limit": 40}, str(tmp_path))
    _assert_pass(rep)
