"""The ⚠️ [MATCHUP DRIFT] headline names what the launch IS.

2026-09-29: an exploiter FORK of `ai_v14_07_g0p_k2` printed "this resume declares matchup …", which
read as a mid-run curriculum change on a run that had just been created. A fork's matchup is compared
against its PARENT's recorded one, and a fork is expected to differ (every exploiter fork of a
self-play parent does). The header now says FORK and names the parent. A same-run restart keeps the
mid-run-change wording.
"""
import argparse
import os

from main.train.matchup_setup import _planned_run_dir, matchup_drift_header


def test_fork_names_the_parent_run_and_says_fork():
    h = matchup_drift_header("models/ai_v14_07_g0p_k2/final_model.zip",
                             "models/ai_v14_09_r0_offense_a", "3bc2dbf787", "ef5242cffd")
    assert "FORK of ai_v14_07_g0p_k2" in h
    assert "resume" not in h.lower() and "RESTART" not in h
    assert "3bc2dbf787" in h and "ef5242cffd" in h


def test_fork_from_a_periodic_checkpoint_names_the_run_not_the_checkpoints_dir():
    h = matchup_drift_header("models/parent_run/checkpoints/checkpoint_500000_steps.zip",
                             "models/child_run", "aaa", "bbb")
    assert "FORK of parent_run" in h


def test_same_run_restart_keeps_the_mid_run_change_wording():
    for model in ("models/r/final_model_interrupted.zip", "models/r/checkpoints/c_1_steps.zip"):
        h = matchup_drift_header(model, "models/r", "new", "old")
        assert "RESTART" in h and "mid-run" in h and "FORK" not in h


def test_no_known_run_dir_is_a_fork():
    # An unnamed launch writes a fresh timestamped dir, which no --model can already live inside.
    h = matchup_drift_header("models/p/final_model.zip", None, "n", "o")
    assert "FORK of p" in h


def test_planned_run_dir_precedence():
    ns = argparse.Namespace
    assert _planned_run_dir(ns(run_dir="models/x", run_name="y")) == "models/x"
    assert _planned_run_dir(ns(run_dir=None, run_name="y")) == os.path.join("models", "y")
    assert _planned_run_dir(ns(run_dir=None, run_name=None)) is None
