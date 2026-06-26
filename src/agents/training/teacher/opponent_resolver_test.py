"""Unit tests for the exact-opponent resolver (bot / sentinel / unresolved) — no models, tmp metadata."""

import json
import os

from agents.training.teacher.opponent_resolver import _latest_eval_block, resolve_opponent


def _make_run(tmp_path, eval_step, sentinels, on_disk):
    run = tmp_path / "run"
    (run / "snapshots").mkdir(parents=True)
    for snap in on_disk:
        (run / "snapshots" / snap).write_text("x")          # a stand-in snapshot file
    meta = {"latest_eval": {"step": eval_step, "pool": {"sentinels": sentinels}}}
    (run / "metadata.json").write_text(json.dumps(meta))
    _latest_eval_block.cache_clear()                         # the resolver caches per run_dir
    return str(run)


def test_bot_resolves_to_bot_source(tmp_path):
    run = _make_run(tmp_path, 100, [], [])
    assert resolve_opponent(run, "heuristic", 100) == (None, "bot")
    assert resolve_opponent(run, "aggressive_v2", 100) == (None, "bot")


def test_sentinel_resolves_to_its_snapshot(tmp_path):
    sents = [{"snapshot": "snapshot_000000074.zip"}, {"snapshot": "snapshot_000000088.zip"}]
    run = _make_run(tmp_path, 100, sents, ["snapshot_000000074.zip", "snapshot_000000088.zip"])
    path, src = resolve_opponent(run, "sentinel_1", 100)
    assert src == "ckpt" and path.endswith("snapshot_000000088.zip") and os.path.exists(path)


def test_sentinel_unresolved_when_snapshot_aged_out(tmp_path):
    # metadata names a snapshot that no longer exists on disk (pool window slid) → honest skip.
    run = _make_run(tmp_path, 100, [{"snapshot": "snapshot_gone.zip"}], on_disk=[])
    assert resolve_opponent(run, "sentinel_0", 100) == (None, "unresolved")


def test_sentinel_unresolved_when_step_mismatch(tmp_path):
    # the trace's step != the latest eval cycle → the positional mapping is stale → skip.
    run = _make_run(tmp_path, 100, [{"snapshot": "snapshot_x.zip"}], ["snapshot_x.zip"])
    assert resolve_opponent(run, "sentinel_0", 999) == (None, "unresolved")


def test_ext_and_garbage_are_unresolved(tmp_path):
    run = _make_run(tmp_path, 100, [], [])
    assert resolve_opponent(run, "ext_some_run", 100) == (None, "unresolved")
    assert resolve_opponent(run, "sentinel_notanint", 100) == (None, "unresolved")


def test_missing_metadata_is_unresolved(tmp_path):
    run = tmp_path / "empty"; run.mkdir()
    _latest_eval_block.cache_clear()
    assert resolve_opponent(str(run), "sentinel_0", 100) == (None, "unresolved")
