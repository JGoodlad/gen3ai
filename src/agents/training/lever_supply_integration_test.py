"""`gen3_supply_guard_v2` with REAL processes — the trainer's own exit code.

`lever_supply_test.py` pins each guard; this pins what a launch actually DOES:

* (slow, sim) a `--self-play` run whose pool never seeds exits ``FATAL_SUPPLY`` (5);
* (slow, sim) a `--search-teacher` run with no eval loss traces to select from exits 5.

The trainer is started through `runpy` with its module name split, so no argv on this box carries
the trainer's literal name (watchers key on it). Every run dir is under `tmp_path`.
"""
from __future__ import annotations


import pytest

from main.exit_codes import TrainExitCode
from main.train.fatal_config_exits_integration_test import _trainer

pytestmark = [pytest.mark.integration, pytest.mark.slow, pytest.mark.sim]


def test_a_self_play_run_whose_pool_never_seeds_exits_FATAL_SUPPLY(tmp_path):
    """The `ai_v12_27` shape end to end: `--self-play` with a seeding gate the run cannot reach
    (`--self-play-start-wr 1.01`), so every eval cycle leaves the pool EMPTY."""
    rc, text = _trainer(tmp_path, "--debug-eval", "--self-play", "--self-play-start-wr", "1.01",
                        "--steps", "60000", "--supply-starve-cycles", "self_play_pool=2",
                        "--run-dir", str(tmp_path / "run"), timeout=1200)
    assert rc == int(TrainExitCode.FATAL_SUPPLY), (rc, text[-4000:])
    assert "[SUPPLY] FATAL: --self-play is live" in text
    assert "Training complete" not in text


def test_a_search_teacher_with_no_loss_traces_exits_FATAL_SUPPLY(tmp_path):
    """`--debug` without `--debug-eval` writes no eval traces, so crater selection finds nothing,
    every cycle — the composition test's docstring names this as the way it could go green on a run
    that did nothing."""
    rc, text = _trainer(tmp_path, "--steps", "60000", "--search-teacher",
                        "--teacher-search-freq", "1000", "--n-steps", "512",
                        "--supply-starve-cycles", "search_teacher=2",
                        "--run-dir", str(tmp_path / "run"), timeout=1200)
    assert rc == int(TrainExitCode.FATAL_SUPPLY), (rc, text[-4000:])
    assert "[SUPPLY] FATAL: --search-teacher is live" in text
    assert "eval_traces/ is EMPTY" in text
