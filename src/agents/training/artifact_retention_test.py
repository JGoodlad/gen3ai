"""Unit tests for the per-run debug-artifact retention (stalls/ + crashes/)."""

import os

from agents.training.artifact_retention import (
    prune_run_artifacts, discover_run_dirs, KEEP_STALLS_DEFAULT, KEEP_CRASHES_DEFAULT,
)
from agents.training.eval_callback import PerOpponentEvalCallback


def _seed(directory, prefix, suffix, n, size=10):
    """Create n files prefix{i}suffix with strictly increasing mtime (i = newest).

    Returns the paths oldest→newest. Explicit mtimes make "most-recent" deterministic
    even though the files are written in the same second.
    """
    os.makedirs(directory, exist_ok=True)
    paths = []
    for i in range(n):
        p = os.path.join(directory, f"{prefix}{i:04d}{suffix}")
        with open(p, "w") as f:
            f.write("x" * size)
        os.utime(p, (1_000_000 + i, 1_000_000 + i))   # i larger => newer
        paths.append(p)
    return paths


def test_keeps_n_most_recent_stalls(tmp_path):
    run = tmp_path / "run"
    paths = _seed(run / "stalls", "stall_", ".html", 5)
    rep = prune_run_artifacts(str(run), keep_stalls=2, keep_crashes=0)
    kept = sorted(os.listdir(run / "stalls"))
    assert kept == ["stall_0003.html", "stall_0004.html"]   # 2 newest by mtime
    assert rep["stalls"]["removed"] == 3
    assert not os.path.exists(paths[0])                       # oldest gone


def test_keeps_n_most_recent_crashes(tmp_path):
    run = tmp_path / "run"
    _seed(run / "crashes", "restart_err_", ".txt", 4)
    prune_run_artifacts(str(run), keep_stalls=0, keep_crashes=1)
    assert sorted(os.listdir(run / "crashes")) == ["restart_err_0003.txt"]


def test_zero_keeps_all(tmp_path):
    run = tmp_path / "run"
    _seed(run / "stalls", "stall_", ".html", 3)
    prune_run_artifacts(str(run), keep_stalls=0, keep_crashes=0)   # 0 = keep all
    assert len(os.listdir(run / "stalls")) == 3


def test_only_matches_own_prefix(tmp_path):
    """A stray file in the dir (wrong prefix/suffix) is never deleted — we only prune
    what the producer wrote."""
    run = tmp_path / "run"
    _seed(run / "stalls", "stall_", ".html", 3)
    keep_me = run / "stalls" / "README.txt"
    keep_me.write_text("notes")
    prune_run_artifacts(str(run), keep_stalls=1, keep_crashes=0)
    survivors = set(os.listdir(run / "stalls"))
    assert "README.txt" in survivors
    assert "stall_0002.html" in survivors and "stall_0000.html" not in survivors


def test_missing_dirs_are_noop(tmp_path):
    run = tmp_path / "bare"
    run.mkdir()
    rep = prune_run_artifacts(str(run))                # no stalls/ or crashes/
    assert rep["bytes_reclaimed"] == 0
    assert rep["stalls"]["removed"] == 0 and rep["crashes"]["removed"] == 0


def test_none_model_dir_is_noop(tmp_path):
    rep = prune_run_artifacts(None)
    assert rep["run_dir"] is None and rep["bytes_reclaimed"] == 0


def test_dry_run_deletes_nothing_but_reports(tmp_path):
    run = tmp_path / "run"
    _seed(run / "stalls", "stall_", ".html", 5, size=100)
    rep = prune_run_artifacts(str(run), keep_stalls=2, keep_crashes=0, apply=False)
    assert len(os.listdir(run / "stalls")) == 5         # nothing deleted
    assert rep["applied"] is False
    assert rep["stalls"]["removed"] == 3 and rep["bytes_reclaimed"] == 300


def test_discover_run_dirs_sweeps_nested(tmp_path):
    models = tmp_path / "models"
    (models / "run_A" / "stalls").mkdir(parents=True)
    (models / "run_B" / "crashes").mkdir(parents=True)
    (models / "_goldens" / "run_C" / "stalls").mkdir(parents=True)
    (models / "no_artifacts").mkdir()
    found = {os.path.basename(d) for d in discover_run_dirs(str(models))}
    assert found == {"run_A", "run_B", "run_C"}         # not "no_artifacts"


def test_discover_single_run_dir_returns_itself(tmp_path):
    run = tmp_path / "run"
    (run / "stalls").mkdir(parents=True)
    assert discover_run_dirs(str(run)) == [str(run)]


def test_defaults_are_50_and_10():
    assert KEEP_STALLS_DEFAULT == 50 and KEEP_CRASHES_DEFAULT == 10


def test_callback_prunes_via_periodic_hook(tmp_path):
    """The wiring the trainer actually calls every eval cycle."""
    run = tmp_path / "run"
    _seed(run / "stalls", "stall_", ".html", 6)
    _seed(run / "crashes", "restart_err_", ".txt", 5)
    cb = PerOpponentEvalCallback(model_dir=str(run), keep_stalls=2, keep_crashes=1)
    cb._prune_run_artifacts()
    assert len(os.listdir(run / "stalls")) == 2
    assert len(os.listdir(run / "crashes")) == 1
