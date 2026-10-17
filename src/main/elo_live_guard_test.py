"""`python -m main.elo <run>` REFUSES to write its default `<run>/elo/` into a LIVE run.

2026-10-17 15:01: a bare `main.elo models/rb_x26_s2601` overwrote the live run's
`elo/elo_ratings.json` and `elo_curve.png` (standing rule 6: models/ is read-only). The liveness
signal is the repo's one live-run rule, `utils.procfs.live_training_runs` (a `train_rl_agent.py`
process naming `--run-dir`); here it is read from a PLANTED /proc so the real parse is exercised.
"""
from __future__ import annotations

import json
import os
import sys

import pytest

from main import elo as elo_cli
from main.exit_codes import TrainExitCode
from utils import procfs
from utils.paths import LiveRunDirError, refuse_live_run_dir


def _run_with_results(tmp_path):
    run = tmp_path / "models" / "rb_fixture"
    run.mkdir(parents=True)
    rows = [{"step": 1_000_000 * i, "n_games": 100,
             "bots": {"random": 0.9, "heuristic": 0.5 + 0.05 * i}, "sentinels": []}
            for i in range(1, 4)]
    (run / "eval_results.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return run


def _plant_live_trainer(tmp_path, monkeypatch, run):
    proc = tmp_path / "proc"
    d = proc / "4242"
    d.mkdir(parents=True)
    argv = ["python3", "src/main/train_rl_agent.py", "--run-dir", str(run)]
    (d / "cmdline").write_bytes(b"\0".join(a.encode() for a in argv) + b"\0")
    (d / "cwd").symlink_to(tmp_path)
    real = procfs.live_training_runs
    monkeypatch.setattr(procfs, "live_training_runs", lambda proc_root=str(proc): real(proc_root))
    assert real(str(proc)) == {str(run): 4242}, "the planted trainer is what the repo calls LIVE"


def _elo(monkeypatch, capsys, *argv):
    monkeypatch.setattr(sys, "argv", ["elo", *argv, "--no-hodge", "--no-plot"])
    rc = elo_cli.main()
    return rc, capsys.readouterr()


def test_elo_REFUSES_its_default_output_in_a_live_run_and_out_or_the_override_works(
        tmp_path, monkeypatch, capsys):
    run = _run_with_results(tmp_path)
    _plant_live_trainer(tmp_path, monkeypatch, run)

    rc, cap = _elo(monkeypatch, capsys, str(run))
    assert rc == int(TrainExitCode.FATAL_CONFIG)
    assert "LIVE run" in cap.err and "--out <dir>" in cap.err and "pid 4242" in cap.err
    assert not (run / "elo").exists(), "nothing was written into the live run"

    out = tmp_path / "elsewhere"
    rc, cap = _elo(monkeypatch, capsys, str(run), "--out", str(out))
    assert rc == 0, cap.err
    assert (out / "elo_ratings.json").is_file() and not (run / "elo").exists()

    rc, cap = _elo(monkeypatch, capsys, str(run), "--allow-live-run-dir")
    assert rc == 0, cap.err
    assert (run / "elo" / "elo_ratings.json").is_file()


def test_elo_default_output_into_a_NON_live_run_is_unchanged(tmp_path, monkeypatch, capsys):
    run = _run_with_results(tmp_path)
    other = tmp_path / "models" / "rb_other"
    other.mkdir()
    _plant_live_trainer(tmp_path, monkeypatch, other)        # a DIFFERENT run is live
    rc, cap = _elo(monkeypatch, capsys, str(run))
    assert rc == 0, cap.err
    assert (run / "elo" / "elo_ratings.json").is_file()


def test_refuse_live_run_dir_is_a_typed_error_and_compares_real_paths(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    link = tmp_path / "link"
    link.symlink_to(run)
    with pytest.raises(LiveRunDirError, match="--out"):
        refuse_live_run_dir(link, meter="elo", live={str(run): 7})
    refuse_live_run_dir(link, meter="elo", live={str(run): 7}, allow=True)
    refuse_live_run_dir(link, meter="elo", live={})
    assert os.path.realpath(link) == str(run.resolve())
