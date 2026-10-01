"""THE M5 LANE H GATE, F-LH-10's SAMPLED-sentinel row (split out of ``parity_test.py`` 2026-09-30 so the routine
gate's file-unit schedule can run it on its own worker; ``parity_test.py``'s docstring states the gate,
``rust_eval.parity`` the method, the tie rule and the bars)."""
from __future__ import annotations

import pytest

from agents.training.rust_eval import parity as PAR
from agents.training.rust_eval.parity_gate_kit import assert_pass as _assert_pass

pytestmark = [pytest.mark.sim, pytest.mark.integration]


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


def test_teeth_a_sentinel_sampling_at_another_temperature_is_fatal(sampled_run):
    """The Python sentinel draws its keyed draw at T = 0.5 against the Rust path's 1.0: a DRAW differs."""
    g = PAR.rerun_python(sampled_run["workdir"], keep=["sentinel_0"], python_self_play_temp=0.5, shards=[0],
                         tag="temp")
    assert g["games"] == 1 and g["fatal"] and not g["missing"], g
    assert any("opp_first_diff" in r for r in g["fatal"]), g["fatal"]
