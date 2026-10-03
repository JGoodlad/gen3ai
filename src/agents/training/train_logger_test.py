"""`gen3_owned_logger_v1` — the owned logger writes what sb3's did (`agents/training/train_logger.py`).

While stable-baselines3 is installed (until stage 4 drops it) its ``Logger`` is the ORACLE: the same records
dumped through both give the same stdout table, byte for byte, and the same TensorBoard calls (tag, value,
step, kind) in the same order; the pending-value bus (``name_to_value`` / ``name_to_count`` /
``name_to_excluded``) behaves the same between dumps. The readers of the event files (killbar, tb_read,
the launcher, plot_tb) and of the bus (the metrics pipe, the K9 golden, `logger_scope`) rest on both.
"""
from __future__ import annotations

import io
from typing import Any, List, Tuple

import numpy as np
import pytest
import torch as th

from agents.training import train_logger as TL


def _records(log: Any) -> None:
    log.record("train/approx_kl", 0.0123456)
    log.record("train/n_updates", 7, exclude="tensorboard")
    log.record("time/fps", 812)
    log.record("time/total_timesteps", 4096, exclude="tensorboard")
    log.record("rollout/ep_rew_mean", np.float32(0.25))
    log.record("rust_env/version", 3.0)
    log.record("note/text", "hello")
    log.record("train/a_very_long_key_name_that_will_be_truncated_by_the_table", 1.5)
    log.record("loose_key", np.int64(4))
    log.record("hist/param", th.arange(4.0))
    log.record_mean("train/m", 2.0)
    log.record_mean("train/m", 4.0)


def test_the_stdout_table_is_sb3s_byte_for_byte():
    from stable_baselines3.common.logger import HumanOutputFormat, Logger

    mine, ref = io.StringIO(), io.StringIO()
    a = TL.Logger(None, [TL.HumanOutputFormat(mine)])
    b = Logger(None, [HumanOutputFormat(ref)])
    for log in (a, b):
        _records(log)
    assert dict(a.name_to_value).keys() == dict(b.name_to_value).keys()
    assert a.name_to_value["train/m"] == b.name_to_value["train/m"] == 3.0
    assert a.name_to_excluded == b.name_to_excluded and a.name_to_count == b.name_to_count
    a.dump(step=100)
    b.dump(step=100)
    assert mine.getvalue() == ref.getvalue() and "approx_kl" in mine.getvalue()
    assert not a.name_to_value and not a.name_to_count and not a.name_to_excluded


class _FakeWriter:
    def __init__(self) -> None:
        self.calls: List[Tuple[Any, ...]] = []

    def add_scalar(self, k: str, v: Any, s: int) -> None:
        self.calls.append(("scalar", k, float(v), s))

    def add_text(self, k: str, v: str, s: int) -> None:
        self.calls.append(("text", k, v, s))

    def add_histogram(self, k: str, v: Any, s: int) -> None:
        self.calls.append(("hist", k, tuple(th.as_tensor(v).tolist()), s))

    def flush(self) -> None:
        self.calls.append(("flush",))


def test_the_tensorboard_calls_are_sb3s_call_for_call():
    from stable_baselines3.common.logger import Logger, TensorBoardOutputFormat

    ours = TL.TensorBoardOutputFormat.__new__(TL.TensorBoardOutputFormat)
    theirs = TensorBoardOutputFormat.__new__(TensorBoardOutputFormat)
    for f in (ours, theirs):
        f.writer, f._is_closed = _FakeWriter(), False
    a, b = TL.Logger(None, [ours]), Logger(None, [theirs])
    for log in (a, b):
        _records(log)
        log.dump(step=4096)
    assert ours.writer.calls == theirs.writer.calls
    assert ("scalar", "train/approx_kl", 0.0123456, 4096) in ours.writer.calls
    assert not any(c[1] == "train/n_updates" for c in ours.writer.calls if len(c) > 1)   # excluded


def test_a_read_inserted_key_fails_the_dump_loudly_as_sb3s_did():
    """``name_to_value`` is a defaultdict: an indexing READ of a missing key inserts 0.0 without an exclusion
    entry. sb3's writers pair the two sorted dicts STRICTLY — so did and does this one: a loud error, never
    a phantom 0 written to TensorBoard."""
    log = TL.Logger(None, [TL.HumanOutputFormat(io.StringIO())])
    log.record("train/x", 1.0)
    _ = log.name_to_value["train/never_recorded"]
    with pytest.raises(ValueError):
        log.dump(step=1)


def test_a_pair_of_EQUAL_LENGTH_dicts_with_different_keys_fails_the_dump_loudly(tmp_path):
    """P10 follow-up F1 (item 5): `zip(sorted(a.items()), sorted(b.items()), strict=True)` checks LENGTHS.
    A phantom value key (an indexing read) together with a stranded exclusion key (the values dict edited
    on its own, as `name_to_value.clear()` does) is the SAME length, so the old writer paired the rows
    wrongly and wrote on — here it is the same loud error naming both keys, on BOTH writers."""
    for fmt in (TL.HumanOutputFormat(io.StringIO()), TL.TensorBoardOutputFormat(str(tmp_path / "tb"))):
        log = TL.Logger(None, [fmt])
        log.record("train/x", 1.0)
        log.record("train/y", 2.0, exclude="tensorboard")
        _ = log.name_to_value["train/phantom"]                  # a value with no `record`
        del log.name_to_value["train/y"]                        # ... and an exclusion with no value
        assert len(log.name_to_value) == len(log.name_to_excluded)       # the case strict-zip cannot see
        with pytest.raises(ValueError, match=r"train/phantom.*train/y"):
            log.dump(step=1)
        log.name_to_value.clear()
        log.name_to_excluded.clear()
        log.record("train/ok", 3.0)
        log.dump(step=2)                                         # a consistent bus still dumps


def test_configure_serves_stdout_and_tensorboard_only(tmp_path, capsys):
    null = TL.configure(None, [])
    null.record("a/b", 1.0)
    null.dump(0)                                            # writes nothing, creates nothing
    assert capsys.readouterr().out == ""
    with pytest.raises(ValueError, match="not served"):
        TL.configure(str(tmp_path), ["csv"])
    with pytest.raises(ValueError, match="folder"):
        TL.configure(None, ["tensorboard"])
    tb = TL.configure(str(tmp_path / "tb"), ["stdout", "tensorboard"])
    assert f"Logging to {tmp_path / 'tb'}" in capsys.readouterr().out
    tb.record("train/loss", 0.5)
    tb.dump(10)
    tb.close()
    assert any(p.name.startswith("events.out.tfevents") for p in (tmp_path / "tb").iterdir())
