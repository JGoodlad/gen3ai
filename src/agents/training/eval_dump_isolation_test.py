"""`gen3_eval_dump_isolation_v1` — an eval cycle's logger dump no longer takes the last update's
`train/*` with it (`designs/endstate/design_own_ppo_loop.md` §2.1, stage 2).

Each test below FAILS on revert (drop `@isolated_dump` from `_collect_pending`): the eval dump would
carry `train/approx_kl` at the eval step and leave the bus empty for the KL controller.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, List, Tuple
from unittest.mock import MagicMock

import pytest

from agents.training import logger_scope as LS
from agents.training.eval_callback import PerOpponentEvalCallback
from agents.training.eval_sharding import ShardResult
from agents.training.train_logger import Logger

KL = 0.0123


class _Rec:
    def __init__(self) -> None:
        self.dumps: List[Tuple[int, Dict[str, Any]]] = []

    def write(self, key_values: Dict[str, Any], key_excluded: Dict[str, Any], step: int = 0) -> None:
        self.dumps.append((int(step), dict(key_values)))

    def write_sequence(self, _seq: Any) -> None:
        pass

    def close(self) -> None:
        pass


def _logger() -> Tuple[Logger, _Rec]:
    rec = _Rec()
    return Logger(folder=None, output_formats=[rec]), rec


def _publish_all(pool: Any, run_dir: str) -> None:
    for u in pool.units:
        pool.publish(run_dir, ShardResult(
            unit_id=u.unit_id, item_key=u.item.key, worker_id=0, n_won=3 * u.n_games // 4,
            n_finished=u.n_games, sum_reward=0.0, n_episodes=u.n_games, sum_ep_len=40.0 * u.n_games,
            duration_sec=1.0, td_residuals=[-0.1, 0.2], traces_written=0, traces_won=0))


class _Stats:
    def __init__(self, n: int) -> None:
        self.n = n

    def as_dict(self) -> Dict[str, Any]:
        return {"games": self.n, "units": self.n, "host_steps": 1, "trainee_decisions": 10 * self.n,
                "p2_policy_decisions": 0, "traces": 0, "near_ties": 0,
                "seconds": {k: 0.0 for k in ("load", "stage", "submit", "drain", "act", "core", "finish", "trace",
                                             "total")}, "lifecycle": {}}


class _FakeEvaluator:
    def run_cycle(self, pool: Any, run_dir: str, **kw: Any) -> Any:
        _publish_all(pool, run_dir)
        return _Stats(len(pool.units))


def _cb(tmp_path: Any, logger: Logger) -> PerOpponentEvalCallback:
    cb = PerOpponentEvalCallback(model_dir=str(tmp_path),
                                 best_model_save_path=str(tmp_path / "best"), eval_games=8)
    cb.model = MagicMock()
    cb.model.logger = logger
    cb.model.save = lambda base: open(base + ".zip", "w").close()
    cb.model.gamma = 0.99
    cb.model._rust_collector = SimpleNamespace(evaluator=_FakeEvaluator(), cfg=SimpleNamespace(run_seed=5))
    cb.num_timesteps = 2_000_000
    cb._init_callback()
    return cb


def _assert_isolated(rec: _Rec, logger: Logger) -> None:
    # the cycle's dumps: its scalars, then its wall clock (gen3_eval_wall_sec_v1) — both at the eval step
    eval_dumps = [(s, d) for s, d in rec.dumps if any(k.startswith("eval/") for k in d)]
    assert eval_dumps and all(s == 2_000_000 for s, _ in eval_dumps), [s for s, _ in rec.dumps]
    for _s, d in eval_dumps:
        assert "train/approx_kl" not in d, "the eval dump took the last update's train/* (the §2.1 defect)"
    assert logger.name_to_value.get("train/approx_kl") == KL, \
        "the last update's approx_kl did not survive the eval cycle for the KL controller"


def test_the_blocking_in_process_cycle_keeps_the_update_scalars(tmp_path: Any) -> None:
    logger, rec = _logger()
    cb = _cb(tmp_path, logger)
    logger.record("train/approx_kl", KL)          # the previous update's scalar, pending on the bus
    cb._on_step()                                  # launches AND collects (blocking) in this step
    _assert_isolated(rec, logger)


def test_both_eval_callbacks_collect_through_the_isolation() -> None:
    from agents.training.selfplay_callback import SelfPlayCallback

    for cls in (PerOpponentEvalCallback, SelfPlayCallback):
        assert getattr(cls._collect_pending, LS.ISOLATED_ATTR, False), cls.__name__


def test_hold_restore_round_trip_and_the_scope_dumps_only_its_own() -> None:
    logger, rec = _logger()

    class _Cb:
        def __init__(self, lg: Logger) -> None:
            self.logger = lg

        @LS.isolated_dump
        def publish(self) -> None:
            self.logger.record("eval/x", 1.0)
            self.logger.dump(7)

    logger.record("train/approx_kl", KL)
    logger.record("train/n_updates", 3, exclude="tensorboard")
    logger.record_mean("train/m", 2.0)
    _Cb(logger).publish()
    assert rec.dumps == [(7, {"eval/x": 1.0})]
    assert logger.name_to_value["train/approx_kl"] == KL
    assert logger.name_to_excluded["train/n_updates"] == ("tensorboard",)
    assert logger.name_to_count["train/m"] == 1
    logger.dump(9)
    assert rec.dumps[-1][0] == 9 and set(rec.dumps[-1][1]) == {"train/approx_kl", "train/n_updates", "train/m"}


def test_the_held_values_come_back_even_when_the_cycle_raises() -> None:
    logger, _rec = _logger()

    class _Cb:
        def __init__(self, lg: Logger) -> None:
            self.logger = lg

        @LS.isolated_dump
        def publish(self) -> None:
            raise RuntimeError("eval cycle failed")

    logger.record("train/approx_kl", KL)
    with pytest.raises(RuntimeError):
        _Cb(logger).publish()
    assert logger.name_to_value["train/approx_kl"] == KL
