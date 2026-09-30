"""THE M5 LANE H GATE — the same seed set on both eval paths (``rust_eval.parity`` is the harness; its
docstring states the method, the tie rule and the bars).

COMMIT (routine, CPU): the full nine-bot roster + two perturbed-fresh sentinels, shard units of one
and two games (several units per env, filler games at the tail), T2 EAGER vs the Python worker's eager
CPU forward — every game equal, metrics equal, the kept traces the same files and the same decisions.
Teeth: a Python path on another cycle seed is caught as FATAL game differences; one tampered shard
record is caught as a metric difference; the tie rule excuses only a flip under its margin.

COMMIT, F-LH-10's two paths (routine, CPU):
* FIXED opponents — two perturbed-fresh ``ext_`` opponents, one REUSING the training plan's stable slot
  and pinned to two sample teams, one on its own eval slot (which starts with ANOTHER policy's weights
  until ``build_eval_core`` loads it) on the pool; every game equal on both paths.
* the SAMPLED sentinel regime (``--no-eval-sentinel-greedy``) — the Rust keyed draw vs the Python worker's
  keyed sampler: every game equal AND every sentinel draw ``[decision, action]`` equal, with most draws
  off the argmax (the regime samples); the greedy commit row's sentinels read 0 off-argmax.
Teeth (the Python path re-run on one item of each row, one input mutated): a fixed opponent playing the
TRAINEE's weights (the shape of the bug the fixed row found) and a sentinel sampling at another
temperature are each FATAL.

MILESTONE (``slow``): the fixed + sampled rows at a larger shape (three sampled sentinels at
``--self-play-temp 0.8`` and two fixed opponents, one pinned to three teams, 6 games each, shard 3,
12 envs). The production eval
shape on CPU — nine bots + five sentinels x 25 games, shard 25, 64 envs. GPU MILESTONE (``slow`` + ``GEN3AI_TEST_ALLOW_GPU=1`` under ``scripts/ops/gpu_lock.sh``):
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


def test_commit_greedy_sentinels_never_draw_off_the_argmax(commit_run):
    s = commit_run["sampled"]
    assert s["sentinel_decisions"] > 0 and s["drawn_not_argmax"] == 0, s
    assert commit_run["games"]["opp_games"] == 4, "each greedy sentinel game's decisions compared on both paths"


@pytest.fixture(scope="module")
def fixed_run(built, tmp_path_factory):
    wd = tmp_path_factory.mktemp("laneH_fixed")
    return PAR.run({"games": 3, "shard_games": 2, "sentinels": 0, "bots": [], "n_envs": 4, "device": "cpu",
                    "backend": "eager", "bar": PAR.BAR_CPU, "seed": 20261003,
                    "fixed": [{"pins": 2, "reused": True}, {"pins": 0}]}, str(wd))


def test_commit_fixed_opponents_play_the_same_games_on_both_paths(fixed_run):
    """F-LH-10 (1). The declared layout as `rust_env_setup` builds it: ext_fixed0 REUSES slot 0 (the
    training plan's stable slot), ext_fixed1 has its own eval slot, LOADED at startup — without that
    load it plays the group template's weights and its games are FATAL (found by this row)."""
    _assert_pass(fixed_run)
    g = fixed_run["games"]
    assert g["equal"] == g["games"] == g["opp_games"] == 6, "and every fixed opponent decision compared"
    assert fixed_run["rust_fixed_slots"] == [["ext_fixed0", 0], ["ext_fixed1", 2]]
    ids = fixed_run["rust_slot_model_ids"]
    assert ids[0] == "stable:ext_fixed0" and ids[2] == "eval:fixed:ext_fixed1", ids
    assert fixed_run["rust_opp_teams"]["ext_fixed0"] == 2, "the 2-team pin must be SAMPLED, not its first team"
    assert fixed_run["rust_stats"]["p2_policy_decisions"] > 0
    assert not any(fixed_run["rust_stats"]["lifecycle"].values())


@pytest.fixture(scope="module")
def sampled_run(built, tmp_path_factory):
    wd = tmp_path_factory.mktemp("laneH_sampled")
    return PAR.run({"games": 2, "shard_games": 1, "sentinels": 2, "bots": [], "n_envs": 4, "device": "cpu",
                    "backend": "eager", "bar": PAR.BAR_CPU, "seed": 20261004, "sentinel_greedy": False}, str(wd))


def test_commit_the_sampled_sentinel_regime_plays_the_same_draws_on_both_paths(sampled_run):
    """F-LH-10 (2). Both paths draw the KEYED draw keyed by the game, so the sampled regime is compared
    game for game: every trainee action and every sentinel draw equal."""
    _assert_pass(sampled_run)
    g, s = sampled_run["games"], sampled_run["sampled"]
    assert g["equal"] == g["games"] == 4 and not g["ties"]
    assert g["opp_games"] == 4, "every sentinel game's draw stream must be compared"
    assert g["opp_decisions"] == s["sentinel_decisions"] > 0
    assert s["drawn_not_argmax"] >= 0.1 * s["sentinel_decisions"], s   # it SAMPLES (greedy reads 0)


def test_teeth_a_fixed_opponent_playing_the_trainees_weights_is_fatal(fixed_run):
    """The Python path's ext_fixed1 plays the TRAINEE's zip — what the Rust slot played before
    ``build_eval_core`` loaded it: its games must be FATAL (the Rust games are the fixed row's own)."""
    g = PAR.rerun_python(fixed_run["workdir"], keep=["ext_fixed1"], python_fixed_paths={"ext_fixed1": "trainee"},
                         shards=[1], tag="trainee_swap")
    assert g["games"] == 1 and g["fatal"] and not g["missing"], g


def test_teeth_a_sentinel_sampling_at_another_temperature_is_fatal(sampled_run):
    """The Python sentinel draws its keyed draw at T = 0.5 against the Rust path's 1.0: a DRAW differs."""
    g = PAR.rerun_python(sampled_run["workdir"], keep=["sentinel_0"], python_self_play_temp=0.5, shards=[0],
                         tag="temp")
    assert g["games"] == 1 and g["fatal"] and not g["missing"], g
    assert any("opp_first_diff" in r for r in g["fatal"]), g["fatal"]


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
    # the sentinel's keyed draws ([dec, action, argmax, CDF margin]), compared only where both logged them
    ob = dict(base, opp=[[0, 3, 3, 0.2], [1, 4, 2, 0.3]])
    assert PAR.compare_games([ob], [dict(ob)], 1e-5)["equal"] == 1
    assert PAR.compare_games([ob], [dict(base, opp=None)], 1e-5)["equal"] == 1, "one side unlogged = not compared"
    near = dict(ob, opp=[[0, 3, 3, 0.2], [1, 5, 2, 1e-6]], winner=2)
    r = PAR.compare_games([dict(ob, opp=[[0, 3, 3, 0.2], [1, 4, 2, 1e-6]])], [near], 1e-5)
    assert len(r["ties"]) == 1 and r["ties"][0]["opp_first_diff"] == 1
    far = dict(ob, opp=[[0, 3, 3, 0.2], [1, 5, 2, 0.3]])
    assert len(PAR.compare_games([ob], [far], 1e-5)["fatal"]) == 1, "a draw flip above the margin = FATAL"
    short = dict(ob, opp=[[0, 3, 3, 0.2]])
    assert len(PAR.compare_games([ob], [short], 1e-5)["fatal"]) == 1, "a missing draw = FATAL"
    both = dict(near, actions=[6, 9, 8])
    r = PAR.compare_games([dict(ob, opp=[[0, 3, 3, 0.2], [1, 4, 2, 1e-6]])], [both], 1e-5)
    assert len(r["fatal"]) == 1, "with no clock, a near-tie draw never excuses a trainee flip above the margin"
    # on the Rust path's clock (turn, index within the turn): the EARLIEST first difference is judged
    def clocked(trainee_turns, opp_turn):
        rr = dict(ob, turns=trainee_turns, opp=[[0, 3, 3, 0.2, 1], [1, 4, 2, 1e-6, opp_turn]])
        return PAR.compare_games([rr], [dict(both, opp=[[0, 3, 3, 0.2], [1, 5, 2, 1e-6]])], 1e-5)
    assert len(clocked([1, 6, 7], 5)["ties"]) == 1, "an opponent tie at turn 5 excuses the trainee's turn-6 flip"
    assert len(clocked([1, 5, 5], 5)["fatal"]) == 1, "a flip SIMULTANEOUS with the tie is judged on its own"
    assert len(clocked([1, 4, 7], 5)["fatal"]) == 1, "a flip BEFORE the tie is judged on its own"
    assert len(clocked([5, 5, 7], 5)["ties"]) == 1, "a forced switch after the tied move (same turn) is later"
    # |Δ log p(chosen)| stops at the first divergence on EITHER stream: after the opponent's tied flip the
    # trainee sees another state, so a later log-prob difference is no measurement of the two forwards
    ra = dict(ob, turns=[1, 6, 7], opp=[[0, 3, 3, 0.2, 1], [1, 4, 2, 1e-6, 5]])
    pb = dict(ob, logp=[-0.1, -0.9, -0.3], opp=[[0, 3, 3, 0.2], [1, 5, 2, 1e-6]])
    r = PAR.compare_games([ra], [pb], 1e-5)
    assert len(r["ties"]) == 1 and r["max_dlogp"] == 0.0
    r = PAR.compare_games([ra], [dict(pb, logp=[-0.3, -0.9, -0.3])], 1e-5)
    assert abs(r["max_dlogp"] - 0.2) < 1e-12, "a difference BEFORE the flip is measured"


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
def test_milestone_fixed_opponents_and_the_sampled_regime_on_cpu(built, tmp_path):
    rep = PAR.run({"games": 6, "shard_games": 3, "sentinels": 3, "bots": [], "n_envs": 12, "device": "cpu",
                   "backend": "eager", "bar": PAR.BAR_CPU, "seed": 20261006, "sentinel_greedy": False,
                   "self_play_temp": 0.8, "fixed": [{"pins": 3, "reused": True}, {"pins": 0}]}, str(tmp_path))
    _assert_pass(rep)
    g = rep["games"]
    assert g["games"] == 30 and g["equal"] + len(g["ties"]) == 30
    assert g["opp_games"] + len(g["ties"]) == 30, "every policy opponent's decisions compared, both regimes"
    assert rep["sampled"]["drawn_not_argmax"] > 0


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
