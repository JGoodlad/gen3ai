"""`gen3_eval_wall_sec_v1` — `eval/wall_sec` is the eval cycle's WALL time; `eval/duration_sec` is the
SUMMED UNIT TIME (every shard's own duration added up), which concurrent units inflate far past the
wall. With fake units that each report 1 s and play "concurrently" (instantly), the wall must come out
far BELOW the summed time, on both env cores. FAILS on revert (no `eval/wall_sec` recorded).
"""
from __future__ import annotations

import inspect
from typing import Any, Dict

import pytest

from agents.training import eval_callback as ec
from agents.training.eval_dump_isolation_test import _cb, _logger, _publish_all
from agents.training.eval_sharding.pool import ShardedEvalPool


def _eval_dump(rec: Any) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for step, d in rec.dumps:
        if step == 2_000_000:
            out.update(d)
    return out


def _check(d: Dict[str, Any]) -> None:
    assert "eval/wall_sec" in d, "no eval/wall_sec recorded for the cycle"
    assert d["eval/duration_sec"] >= 5.0, d["eval/duration_sec"]      # >= 5 fake units x 1 s each
    assert 0.0 <= d["eval/wall_sec"] < d["eval/duration_sec"], (d["eval/wall_sec"], d["eval/duration_sec"])


def test_rust_core_wall_is_below_the_summed_unit_time(tmp_path: Any, monkeypatch: Any) -> None:
    monkeypatch.setattr(ec, "spawn_eval_workers", lambda *_a, **_k: pytest.fail("rust spawned workers"))
    logger, rec = _logger()
    cb = _cb(tmp_path, "rust", logger)
    cb._on_step()
    _check(_eval_dump(rec))


def test_python_core_wall_is_below_the_summed_unit_time(tmp_path: Any, monkeypatch: Any) -> None:
    def spawn(run_dir: str, base_cfg: Any, n_workers: int) -> list:
        _publish_all(ShardedEvalPool.from_plan(run_dir), run_dir)
        return []

    monkeypatch.setattr(ec, "spawn_eval_workers", spawn)
    logger, rec = _logger()
    cb = _cb(tmp_path, "python", logger)
    cb._on_step()
    cb.num_timesteps += 98_304
    cb._on_step()
    _check(_eval_dump(rec))


def test_the_self_play_callback_records_the_wall_too() -> None:
    from agents.training.selfplay_callback import SelfPlayCallback

    src = inspect.getsource(SelfPlayCallback._collect_pending)
    assert "record_cycle_wall(self, pending, merged" in src
    assert '"t_launch": t_launch' in inspect.getsource(SelfPlayCallback._launch_eval)
