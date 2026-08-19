"""A fork whose snapshot pool is EMPTY at startup must WARN — it will train on bots, not selves.

Regression guard for the 2026-08-18 defect: three 3M-step forks off a 25M base each started
with an empty pool (the pool is directory-derived), so --self-play silently fell back to the
bot pool and ~9M fork-steps were bot games, voiding a three-arm A/B's external validity.
"""
import os
from agents.training.selfplay_callback import SelfPlayCallback


class _Probe(SelfPlayCallback):
    """Only the guard is under test — bypass __init__'s heavy wiring."""
    def __init__(self, model_dir):
        self._model_dir = model_dir


def test_empty_pool_warns(tmp_path, capsys):
    os.makedirs(tmp_path / "snapshots")
    _Probe(str(tmp_path))._warn_if_fork_pool_empty()
    out = capsys.readouterr().out
    assert "pool is EMPTY" in out
    assert "BOT pool" in out, "the warning must name the actual consequence"
    assert "cp " in out, "the warning must name the fix, not just the problem"


def test_populated_pool_is_silent(tmp_path, capsys):
    d = tmp_path / "snapshots"; os.makedirs(d)
    (d / "snapshot_000002000016.zip").write_bytes(b"x")
    _Probe(str(tmp_path))._warn_if_fork_pool_empty()
    assert capsys.readouterr().out == ""


def test_missing_snapshots_dir_warns_too(tmp_path, capsys):
    _Probe(str(tmp_path))._warn_if_fork_pool_empty()
    assert "pool is EMPTY" in capsys.readouterr().out


def test_no_model_dir_is_a_noop(capsys):
    _Probe(None)._warn_if_fork_pool_empty()
    assert capsys.readouterr().out == ""
