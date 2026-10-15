"""`gen3_owned_logger_v1` — the owned logger writes what sb3's did (`agents/training/train_logger.py`).

While stable-baselines3 is installed (until stage 4 drops it) its ``Logger`` is the ORACLE: the same records
dumped through both give the same stdout table, byte for byte, and the same TensorBoard calls (tag, value,
step, kind) in the same order; the pending-value bus (``name_to_value`` / ``name_to_count`` /
``name_to_excluded``) behaves the same between dumps. The readers of the event files (killbar, tb_read,
the launcher, plot_tb) and of the bus (the metrics pipe, the K9 golden, `logger_scope`) rest on both.
"""
from __future__ import annotations

import io
import re
import types
from typing import Any, List, Tuple

import numpy as np
import pytest
import torch as th

from agents.training import train_logger as TL


LONG_KEY = "train/a_very_long_key_name_that_will_be_truncated_by_the_table"


def _records(log: Any, long_key: bool = True) -> None:
    log.record("train/approx_kl", 0.0123456)
    log.record("train/n_updates", 7, exclude="tensorboard")
    log.record("time/fps", 812)
    log.record("time/total_timesteps", 4096, exclude="tensorboard")
    log.record("rollout/ep_rew_mean", np.float32(0.25))
    log.record("rust_env/version", 3.0)
    log.record("note/text", "hello")
    if long_key:
        log.record(LONG_KEY, 1.5)
    log.record("loose_key", np.int64(4))
    log.record("hist/param", th.arange(4.0))
    log.record_mean("train/m", 2.0)
    log.record_mean("train/m", 4.0)


def test_the_stdout_table_is_sb3s_byte_for_byte_for_every_key_that_fits():
    """sb3's table, text for text — for keys within the 36-column key width. A LONGER key is the one
    deliberate departure (`gen3_metric_key_collision_v1`; the next test)."""
    from stable_baselines3.common.logger import HumanOutputFormat, Logger

    mine, ref = io.StringIO(), io.StringIO()
    a = TL.Logger(None, [TL.HumanOutputFormat(mine)])
    b = Logger(None, [HumanOutputFormat(ref)])
    for log in (a, b):
        _records(log, long_key=False)
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


# ── gen3_metric_key_collision_v1: no metric key collides in ANY output format ─────────────────────
#: The pair that killed every production run at a commit >= 65b71f6a at its first dump (2026-10-15):
#: under the `cell/` tag both rows were cut to '   move_resolution_restored_move_...' and sb3's table RAISED.
CRASH_PAIR = ("cell/move_resolution_restored_move_weight_norm", "cell/move_resolution_restored_move_grad_norm")
#: F-XC-5's pair (fixed then by renaming the key): it killed every fixed_mass self-play run the same way.
FXC5_PAIR = ("opp_intent/flat_switch_target_recall_top1_bot", "opp_intent/flat_switch_target_recall_top1_pool")
#: The LATENT case the survey found (2026-10-15): a stable opponent's eval keys carry its run label, so two seeds of
#: one arm share a head — rows past even the 48-character column, shortened by `truncate_label`.
STABLE_PAIR = ("eval/mean_ep_len_vs_ext_rb_hib_species_moves_s3101@15000000",
               "eval/mean_ep_len_vs_ext_rb_hib_species_moves_s3102@15000000")


@pytest.mark.parametrize("pair", [CRASH_PAIR, FXC5_PAIR, STABLE_PAIR])
def test_two_long_keys_with_one_head_render_as_two_distinct_rows_and_never_raise(pair):
    """THE CRASH, at the unit: sb3's `key[:33] + "..."` gave both keys one label and the dump raised
    ``ValueError`` mid-run. Fails on revert (the dump raises). The first two now print in full (48
    columns); the third is shortened, distinctly."""
    out = io.StringIO()
    log = TL.Logger(None, [TL.HumanOutputFormat(out)])
    for i, k in enumerate(pair):
        log.record(k, float(i))
    log.dump(step=1)
    text = out.getvalue()
    tag = pair[0][: pair[0].index("/") + 1]
    labels = [TL.truncate_label("   " + k[len(tag):]) for k in pair]
    assert labels[0] != labels[1]
    for lab in labels:
        assert re.search(re.escape(f"| {lab}") + r" *\|", text), (lab, text)
    assert TL.metric_key_collisions(pair) == []
    assert all(len("   " + k[len(tag):]) > 36 for k in pair)              # each one sb3 would have cut


def test_a_shortened_label_is_stable_exact_width_and_keeps_head_and_tail():
    k = "   mean_ep_len_vs_ext_rb_hib_species_moves_s3101@15000000"
    lab = TL.truncate_label(k)
    assert lab == TL.truncate_label(k) and len(lab) == TL.STDOUT_MAX_LENGTH
    assert lab.startswith(k[:10]) and lab.endswith("s3101@15000000") and TL.key_digest(k) in lab
    every_production_cell_row = "   move_resolution_restored_switch_weight_norm"        # the longest, 46
    assert TL.truncate_label(every_production_cell_row) == every_production_cell_row
    assert TL.key_digest(k) == TL.key_digest(str(k))                       # blake2s: process-independent
    assert TL.truncate_label("short") == "short"
    assert TL.truncate_label("x" * TL.STDOUT_MAX_LENGTH) == "x" * TL.STDOUT_MAX_LENGTH


def test_a_residual_clash_prints_the_full_key_and_the_startup_check_refuses_it(monkeypatch):
    """Should two labels still meet (here a forced digest clash), the table prints the later key IN FULL —
    a display never stops a run — and `check_metric_keys` refuses the key set (typed, FATAL_CONFIG)."""
    monkeypatch.setattr(TL, "key_digest", lambda key, n=TL.LABEL_DIGEST_HEX: "0" * n)
    keys = ("train/" + "a" * 24 + "_middle_one_" + "z" * 32, "train/" + "a" * 24 + "_middle_two_" + "z" * 32)
    assert TL.truncate_label("   " + keys[0][6:]) == TL.truncate_label("   " + keys[1][6:])
    out = io.StringIO()
    log = TL.Logger(None, [TL.HumanOutputFormat(out)])
    for k in keys:
        log.record(k, 1.0)
    log.dump(step=1)
    assert f"| {keys[1]} |" in out.getvalue()
    with pytest.raises(TL.MetricKeyCollisionError, match="stdout"):
        TL.check_metric_keys(keys, where="test")


def test_the_check_refuses_two_producers_of_one_key_and_a_tensorboard_text_clash():
    with pytest.raises(TL.MetricKeyCollisionError, match="identity"):
        TL.check_metric_keys(["cell/a_weight_norm", "cell/a_weight_norm"], where="test")
    with pytest.raises(TL.MetricKeyCollisionError, match="tensorboard"):
        TL.check_metric_keys(["note/text", "note/text/text_summary"], where="test")
    TL.check_metric_keys([*CRASH_PAIR, *FXC5_PAIR, "train/loss", LONG_KEY], where="test")


def test_a_collision_is_FATAL_CONFIG_and_still_a_ValueError():
    from main.exit_codes import TrainExitCode, exit_code_for
    err = TL.MetricKeyCollisionError("x")
    assert isinstance(err, ValueError)
    assert exit_code_for(err) == int(TrainExitCode.FATAL_CONFIG)


def test_tensorboard_tags_are_the_keys_verbatim_however_long(tmp_path):
    """Dashboards read the TB tags BY NAME: the fix changes only the stdout ROW, never a tag."""
    ours = TL.TensorBoardOutputFormat.__new__(TL.TensorBoardOutputFormat)
    ours.writer, ours._is_closed = _FakeWriter(), False
    log = TL.Logger(None, [ours])
    for k in CRASH_PAIR:
        log.record(k, 1.0)
    log.dump(step=3)
    assert {c[1] for c in ours.writer.calls if c[0] == "scalar"} == set(CRASH_PAIR)


# ── gen3_metric_producer_guard_v1: ONE producer per key per dump interval ────────────────────────
def _site_a(log: Any, key: str = "train/x", value: float = 1.0) -> None:
    log.record(key, value)


def _site_b(log: Any, key: str = "train/x", value: float = 2.0) -> None:
    log.record(key, value)


def test_a_strict_logger_raises_when_a_second_call_site_writes_a_pending_key():
    log = TL.Logger(None, [], strict_producers=True)
    _site_a(log)
    with pytest.raises(TL.DuplicateProducerError) as ei:
        _site_b(log)
    msg = str(ei.value)
    assert "'train/x'" in msg and "train_logger_test.py" in msg and "gen3_metric_producer_guard_v1" in msg
    assert log.name_to_value["train/x"] == 1.0, "a refused write must not replace the pending value"
    assert isinstance(ei.value, TL.MetricKeyCollisionError)


def test_the_same_site_may_write_a_key_again_and_a_dump_frees_it_for_another_site():
    log = TL.Logger(None, [], strict_producers=True)
    for v in (1.0, 2.0, 3.0):          # a loop / a per-minibatch record: ONE producer
        _site_a(log, value=v)
    assert log.name_to_value["train/x"] == 3.0
    log.dump(step=1)
    _site_b(log)                       # a new interval: the key is free
    assert log.name_to_value["train/x"] == 2.0 and set(log.name_to_site) == {"train/x"}


def test_record_mean_from_many_sites_is_the_running_mean_by_contract_but_mixing_with_record_is_a_duplicate():
    log = TL.Logger(None, [], strict_producers=True)
    log.record_mean("train/m", 2.0)
    log.record_mean("train/m", 4.0)    # a different line: still the running mean, not a duplicate
    assert log.name_to_value["train/m"] == 3.0
    with pytest.raises(TL.DuplicateProducerError):
        log.record("train/m", 9.0)


def test_a_lenient_logger_counts_reports_once_and_never_stops(capsys):
    """Production: the later write wins as before, the key is counted, reported ONCE, and the dump carries
    `telemetry/duplicate_producer_keys` (absent on a clean interval)."""
    seen: List[dict] = []
    sink = types.SimpleNamespace(write=lambda kv, ex, step=0: seen.append(dict(kv)),
                                 write_sequence=lambda s: None, close=lambda: None)
    log = TL.Logger(None, [sink], strict_producers=False)
    for key in ("train/x", "train/y", "train/z"):
        _site_a(log, key)
        _site_b(log, key)
    out = capsys.readouterr().out
    assert out.count("[metric-producers] WARNING") == 3 and "'train/x'" in out
    assert log.name_to_value["train/x"] == 2.0                 # the later write won (unchanged behaviour)
    log.dump(step=1)
    assert seen[0][TL.DUPLICATE_PRODUCERS_KEY] == 3.0
    assert set(log.duplicate_producers) == {"train/x", "train/y", "train/z"}
    _site_a(log)
    _site_b(log)                       # the same key again: counted in the next dump, NOT reported again
    assert "[metric-producers]" not in capsys.readouterr().out
    log.dump(step=2)
    assert seen[1][TL.DUPLICATE_PRODUCERS_KEY] == 1.0
    _site_a(log)
    log.dump(step=3)                   # a clean interval carries no such key
    assert TL.DUPLICATE_PRODUCERS_KEY not in seen[2]


def test_the_strictness_comes_from_the_env_then_pytest(monkeypatch):
    assert TL.producer_guard_strict() is True                  # under pytest
    monkeypatch.setenv(TL.STRICT_PRODUCERS_ENV, "0")
    assert TL.producer_guard_strict() is False and TL.Logger(None, []).strict_producers is False
    monkeypatch.setenv(TL.STRICT_PRODUCERS_ENV, "1")
    assert TL.producer_guard_strict() is True
    monkeypatch.delenv(TL.STRICT_PRODUCERS_ENV)
    monkeypatch.delenv("PYTEST_CURRENT_TEST")
    assert TL.producer_guard_strict() is False                 # a production process: lenient


def test_a_debug_trainer_arms_the_strict_guard():
    """`--debug` is a smoke: `train_rl_agent` sets the env var (a typed `=0` still wins via setdefault)."""
    import inspect
    from main import train_rl_agent
    src = inspect.getsource(train_rl_agent)
    assert 'setdefault("GEN3AI_STRICT_METRIC_PRODUCERS", "1")' in src and "if args.debug:" in src
    assert TL.STRICT_PRODUCERS_ENV == "GEN3AI_STRICT_METRIC_PRODUCERS"


def test_a_held_key_keeps_its_producer_through_an_isolated_dump():
    """`logger_scope.hold` / `restore` carry the producer site with the value: an eval cycle's own dump must not
    make a held key forget (or invent) its producer."""
    from agents.training import logger_scope as LS
    log = TL.Logger(None, [], strict_producers=True)
    _site_a(log)
    held = LS.hold(log)
    _site_b(log)                       # the eval's own scope writes the key: nothing is pending, no clash
    log.dump(step=1)
    LS.restore(log, held)
    _site_a(log)                       # the held producer again: fine
    with pytest.raises(TL.DuplicateProducerError):
        _site_b(log)
