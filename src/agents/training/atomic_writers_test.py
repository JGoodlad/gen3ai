"""Every file a resume or a reader loads from a run / pool / eval cycle is written through `utils.atomic_io` (T46
review F8): temp file, fsync, `os.replace`. A plain `open(path, "w")` or a temp + replace WITHOUT the fsync fails
here, so this file fails on a revert of any one writer to the old form.

(The checkpoint zips, the pointer's order and the metadata/sidecar JSON have their end-to-end tests in
`main/train/checkpoint_commit_test.py`; the helper's own contract is `utils/atomic_io_test.py`.)
"""
from __future__ import annotations

import json
import os
from unittest.mock import MagicMock, patch

import pytest

from utils import atomic_io


@pytest.fixture
def fsyncs(monkeypatch):
    """The paths `atomic_io` fsync'd, in order; the real fsync still runs."""
    seen = []
    real = atomic_io.fsync_path

    def spy(path):
        seen.append(os.path.basename(str(path)))
        real(path)

    monkeypatch.setattr(atomic_io, "fsync_path", spy)
    return seen


def _fsynced(seen, final: str) -> bool:
    """Was a temp of ``final`` fsync'd (the temp is ``.<final>.<pid>.<token>.tmp``)?"""
    return any(n.startswith("." + final + ".") and atomic_io.is_temp_name(n) for n in seen)


def test_latest_txt_is_fsynced_and_replaced(tmp_path, fsyncs):
    from main.train.run_io import _write_latest_txt

    _write_latest_txt(str(tmp_path), "checkpoints/checkpoint_1_steps.zip")
    assert _fsynced(fsyncs, "latest.txt") and os.listdir(tmp_path) == ["latest.txt"]


def test_the_eval_shard_result_and_plan_are_fsynced(tmp_path, fsyncs):
    from agents.training.eval_sharding import EvalItem, ShardedEvalPool
    from agents.training.eval_sharding.results import ShardResult, _shard_path, read_shard_result, write_shard_result

    res = ShardResult(unit_id="u0", item_key="bot", worker_id=0, n_won=1, n_finished=2, sum_reward=1.0,
                      n_episodes=2, sum_ep_len=40.0, duration_sec=1.0)
    write_shard_result(str(tmp_path), res)
    assert read_shard_result(str(tmp_path), "u0") == res
    pool = ShardedEvalPool([EvalItem("bot", "bot", 10)], 5, step=100)
    pool.write_plan(str(tmp_path))
    assert _fsynced(fsyncs, os.path.basename(_shard_path(str(tmp_path), "u0"))), fsyncs
    assert _fsynced(fsyncs, "plan.json")


def test_the_ladder_the_eval_manifest_and_the_pool_files_are_fsynced(tmp_path, fsyncs):
    from agents.training import snapshot_ladder
    from agents.training.eval_launch import EVAL_MANIFEST_NAME, write_eval_manifest
    from agents.training.pool_seed import SEED_RECORD_FILE, write_seed_record
    from agents.training.snapshot_pool import SnapshotPool

    snapshot_ladder.atomic_write_json(str(tmp_path / "ladder.json"), {"a": 1})
    assert _fsynced(fsyncs, "ladder.json") and json.loads((tmp_path / "ladder.json").read_text()) == {"a": 1}
    write_eval_manifest(str(tmp_path), 100, opponents=["bot"], n_games=2)
    assert _fsynced(fsyncs, EVAL_MANIFEST_NAME)
    pool_dir = tmp_path / "snapshots"
    pool_dir.mkdir()
    write_seed_record(str(pool_dir), {"schema": "x"})
    assert _fsynced(fsyncs, SEED_RECORD_FILE)
    version = MagicMock()
    version.to_json.return_value = "{}"
    with patch("agents.training.snapshot_pool.load_opponent_snapshot"):
        pool = SnapshotPool(pool_dir=pool_dir, current_version=version)
    pool.persist_summary(win_rate_vs_bots=0.5)
    assert _fsynced(fsyncs, "summary.json") and _fsynced(fsyncs, "win_rate_vs_bots.txt")


def test_a_copied_or_linked_weights_file_is_fsynced_before_it_takes_its_name(tmp_path, fsyncs):
    from agents.training.weights_link import link_or_copy

    src, twin = tmp_path / "snapshot.zip", tmp_path / "checkpoint_1_steps.zip"
    src.write_bytes(b"PK-weights")
    twin.write_bytes(b"PK-weights")
    assert link_or_copy(str(src), str(tmp_path / "best_model.zip"), None)[0] == "copy"
    assert link_or_copy(str(src), str(tmp_path / "pool.zip"), str(twin))[0] == "hardlink"
    assert _fsynced(fsyncs, "best_model.zip") and _fsynced(fsyncs, "pool.zip")
    assert [n for n in os.listdir(tmp_path) if atomic_io.is_temp_name(n)] == []


def test_a_seeded_pool_copy_that_dies_leaves_no_torn_snapshot(tmp_path, monkeypatch):
    """`seed_pool` copies the parent's snapshots into a fork's pool: a copy killed half-way used to leave a torn
    `snapshot_*.zip` in the child's pool that its workers then loaded (and re-seeding overwrote in place)."""
    import shutil

    from agents.training import pool_seed

    parent = tmp_path / "parent" / "snapshots"
    parent.mkdir(parents=True)
    (parent / "snapshot_000000000100.zip").write_bytes(b"PK-parent-weights")
    pool = tmp_path / "fork" / "snapshots"
    pool.mkdir(parents=True)
    child = pool / "snapshot_000000000100.zip"
    child.write_bytes(b"PK-previous-seed")

    def dying_copy(src, dst, *a, **k):
        with open(dst, "wb") as f:
            f.write(b"PK-half")
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(shutil, "copy2", dying_copy)
    with pytest.raises(OSError):
        pool_seed.seed_pool(str(tmp_path / "parent"), str(pool))
    assert child.read_bytes() == b"PK-previous-seed", "the seeding wrote a half snapshot into the pool"
    assert os.listdir(pool) == [child.name]
    monkeypatch.undo()
    pool_seed.seed_pool(str(tmp_path / "parent"), str(pool))
    assert child.read_bytes() == b"PK-parent-weights"
    assert sorted(os.listdir(pool)) == sorted([child.name, pool_seed.SEED_RECORD_FILE])
