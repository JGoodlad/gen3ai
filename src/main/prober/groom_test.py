"""Tests for the eval-data groomer (pure filesystem)."""

import os

from main.prober.groom import groom_run


def _make_run(tmp_path, steps=(1, 2, 3, 4, 5), snapshot_steps=(4, 5)):
    run = tmp_path / "run"
    for s in steps:
        d = run / "eval_traces" / f"step_{s}000000" / "Random"
        os.makedirs(d, exist_ok=True)
        (d / "win_001_summary.json").write_text("{}")
        (d / "win_001_states.npz").write_text("x" * 1000)
        if s in snapshot_steps:
            (run / "eval_traces" / f"step_{s}000000" / "snapshot.zip").write_text("S" * 5000)
    return str(run)


def test_dry_run_plans_without_deleting(tmp_path):
    run = _make_run(tmp_path)
    rep = groom_run(run, keep_trace_steps=3, keep_snapshots=1, apply=False)
    assert rep["applied"] is False
    # 5 steps, keep top 3 (5M,4M,3M) → remove 2M,1M
    assert sorted(rep["removed_steps"]) == [1000000, 2000000]
    # snapshots exist at 4M,5M; keep_snapshots=1 → keep 5M, drop 4M (4M is within kept traces)
    assert rep["dropped_snapshots"] == [4000000]
    assert rep["bytes_reclaimed"] > 0
    # nothing actually deleted
    assert os.path.exists(os.path.join(run, "eval_traces", "step_1000000"))
    assert os.path.exists(os.path.join(run, "eval_traces", "step_4000000", "snapshot.zip"))


def test_apply_deletes_to_retention(tmp_path):
    run = _make_run(tmp_path)
    rep = groom_run(run, keep_trace_steps=3, keep_snapshots=1, apply=True)
    assert rep["applied"] is True
    # old step dirs gone
    assert not os.path.exists(os.path.join(run, "eval_traces", "step_1000000"))
    assert not os.path.exists(os.path.join(run, "eval_traces", "step_2000000"))
    # recent steps kept
    assert os.path.exists(os.path.join(run, "eval_traces", "step_5000000", "Random"))
    # 4M snapshot dropped, its traces kept; 5M snapshot kept
    assert not os.path.exists(os.path.join(run, "eval_traces", "step_4000000", "snapshot.zip"))
    assert os.path.exists(os.path.join(run, "eval_traces", "step_4000000", "Random"))
    assert os.path.exists(os.path.join(run, "eval_traces", "step_5000000", "snapshot.zip"))


def test_under_retention_is_noop(tmp_path):
    run = _make_run(tmp_path, steps=(1, 2), snapshot_steps=(1, 2))
    rep = groom_run(run, keep_trace_steps=10, keep_snapshots=10, apply=True)
    assert rep["removed_steps"] == [] and rep["dropped_snapshots"] == []
    assert rep["bytes_reclaimed"] == 0


def test_accepts_eval_traces_dir_directly(tmp_path):
    run = _make_run(tmp_path)
    rep = groom_run(os.path.join(run, "eval_traces"), keep_trace_steps=2, keep_snapshots=5)
    assert sorted(rep["removed_steps"]) == [1000000, 2000000, 3000000]
