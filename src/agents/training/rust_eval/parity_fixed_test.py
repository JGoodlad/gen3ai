"""THE M5 LANE H GATE, F-LH-10's FIXED-opponents row (split out of ``parity_test.py`` 2026-09-30 so the routine
gate's file-unit schedule can run it on its own worker; ``parity_test.py``'s docstring states the gate,
``rust_eval.parity`` the method, the tie rule and the bars)."""
from __future__ import annotations

import pytest

from agents.training.rust_eval import parity as PAR
from agents.training.rust_eval.parity_gate_kit import assert_pass as _assert_pass

pytestmark = [pytest.mark.sim, pytest.mark.integration]


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


@pytest.mark.slow  # 40 s on a quiet box (2026-10-08): the tier budget enforces 30 s
def test_teeth_a_fixed_opponent_playing_the_trainees_weights_is_fatal(fixed_run):
    """The Python path's ext_fixed1 plays the TRAINEE's zip — what the Rust slot played before
    ``build_eval_core`` loaded it: its games must be FATAL (the Rust games are the fixed row's own)."""
    g = PAR.rerun_python(fixed_run["workdir"], keep=["ext_fixed1"], python_fixed_paths={"ext_fixed1": "trainee"},
                         shards=[1], tag="trainee_swap")
    assert g["games"] == 1 and g["fatal"] and not g["missing"], g


def test_teeth_another_seed_set_on_the_python_side_is_fatal(fixed_run):
    """The gate compares GAMES: a Python path on a different seed set must fail it. Replays one shard of
    the module's already-played fixed row on another cycle seed (the Rust games are the row's own), so no
    second gate run is paid for (a full `PAR.run` of this took 35-43 s, over the unmarked tier's 30 s)."""
    g = PAR.rerun_python(fixed_run["workdir"], keep=["ext_fixed1"], shards=[1], python_seed=fixed_run["cfg"]["seed"] + 1,
                         tag="seed_swap")
    assert g["games"] == 1 and g["fatal"] and not g["missing"], g
