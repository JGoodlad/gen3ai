"""``main.ops.tb_read`` on a SYNTHETIC events file — the three properties the tool is trusted for.

The reader was promoted into the repo because it had already caught two log-rendering misreads on
a live arm, and it is the tool `designs/ops/TRAINING_RUN_SOP.md` §5's "validate at the SOURCE"
rule leans on. That makes its three contracts worth a test that FAILS when they break:

1. **A MISSING TAG WARNS AND EXITS NON-ZERO.** It never prints 0 and never silently omits the
   row. This is the whole reason the tool exists: a default of zero and a real zero are
   indistinguishable in a report, and one of them is not a measurement.
2. **`--last N` takes the LAST N BY STEP**, across every event directory, not the last N written.
3. **The median is the median of that window**, not of the whole series and not the last value.

The events are written with the real ``SummaryWriter``, so the test exercises the real
``EventAccumulator`` read path rather than a stub of it — a stub here would pass while the tag
namespacing that caused one of the two original misreads was wrong.
"""
from __future__ import annotations

import statistics as st

import pytest

from main.ops import tb_read


def _write_events(tb_dir, tag_values, subdir="run_0"):
    """Write ``{tag: [(step, value), ...]}`` into ``<tb_dir>/<subdir>`` as real TB events."""
    from torch.utils.tensorboard import SummaryWriter

    d = tb_dir / subdir
    d.mkdir(parents=True, exist_ok=True)
    w = SummaryWriter(log_dir=str(d))
    for tag, pairs in tag_values.items():
        for step, value in pairs:
            w.add_scalar(tag, value, global_step=step)
    w.flush()
    w.close()
    return d


@pytest.fixture
def run_dir(tmp_path):
    """A run directory holding one event subdirectory with two tags of known shape."""
    run = tmp_path / "synthetic_run"
    _write_events(
        run / "tb",
        {
            # 30 rollouts; the last 20 are steps 11..30 with values 11.0..30.0 -> median 20.5
            "grad/value_policy_logratio": [(s, float(s)) for s in range(1, 31)],
            # a step function, deliberately: the regime-marker path must not take a median
            "train/selfplay_fraction": [(1, 0.0), (2, 0.0), (10, 0.9), (20, 0.9)],
        },
    )
    return run


def test_a_present_tag_reads_its_series_and_the_last_n_window(run_dir):
    series = tb_read.load(run_dir, ["grad/value_policy_logratio"])
    got = series["grad/value_policy_logratio"]
    assert [s for s, _ in got] == list(range(1, 31)), "steps must come back in step order"
    last20 = [v for _, v in got[-20:]]
    assert last20 == [float(s) for s in range(11, 31)]
    assert st.median(last20) == pytest.approx(20.5)
    # ...and NOT the whole series' median, which is the value a windowless read would print
    assert st.median([v for _, v in got]) == pytest.approx(15.5)


def test_last_n_is_by_step_across_every_event_directory(tmp_path):
    """Two event dirs, written out of step order. The window is the last N BY STEP."""
    run = tmp_path / "two_dirs"
    _write_events(run / "tb", {"m/x": [(s, float(s)) for s in range(50, 60)]}, subdir="b_later")
    _write_events(run / "tb", {"m/x": [(s, float(s)) for s in range(1, 11)]}, subdir="a_earlier")
    got = tb_read.load(run, ["m/x"])["m/x"]
    assert [s for s, _ in got] == list(range(1, 11)) + list(range(50, 60))
    assert [v for _, v in got[-3:]] == [57.0, 58.0, 59.0]


def test_a_missing_tag_WARNS_and_never_defaults_to_zero(run_dir, capsys):
    missing = tb_read.report(
        run_dir,
        tb_read.load(run_dir, ["m/never_emitted"]),
        ["m/never_emitted"],
        n=20,
    )
    out = capsys.readouterr().out
    assert missing == 1
    assert "NO SCALAR FOUND" in out
    assert "absence is not a zero" in out
    # The failure mode this guards: a row of zeros that reads like a measurement.
    assert "median +0.0000" not in out


def test_a_missing_tag_makes_the_WHOLE_READ_exit_non_zero(run_dir):
    """The exit status is what a chain reads. A missing instrument must not look clean."""
    rc = tb_read.main([str(run_dir), "--tag", "m/never_emitted"])
    assert rc == 1
    rc_ok = tb_read.main([str(run_dir), "--tag", "grad/value_policy_logratio", "--last", "20"])
    assert rc_ok == 0


def test_the_regime_marker_is_never_reported_as_a_central_tendency(run_dir, capsys):
    """`train/selfplay_fraction` is a STEP FUNCTION. A median across regimes (here 0.45) is a
    value the run has never been at — the exact misread that put this branch in the tool."""
    tb_read.report(
        run_dir,
        tb_read.load(run_dir, ["train/selfplay_fraction"]),
        ["train/selfplay_fraction"],
        n=20,
    )
    out = capsys.readouterr().out
    assert "regime marker" in out
    assert "CURRENT 0.90 since step 10" in out
    assert "0.00@1 -> 0.90@10" in out
    assert "median" not in out.split("=== vf_coef")[0].split("train/selfplay_fraction")[-1]


def test_the_calibration_cost_bar_flags_only_above_its_share(run_dir, capsys):
    """The registered bar is reliability > 10% of resolution — computed at every read, not
    remembered. Both sides of the bar are exercised so a flipped comparison fails."""
    run = run_dir.parent / "calib"
    _write_events(run / "tb", {
        "win_prob/critic_reliability": [(1, 0.02), (2, 0.02)],
        "win_prob/critic_resolution": [(1, 0.10), (2, 0.10)],
    })
    tb_read.report(run, tb_read.load(run, ["win_prob/critic_reliability",
                                           "win_prob/critic_resolution"]),
                   ["win_prob/critic_reliability", "win_prob/critic_resolution"], n=20)
    assert "FLAG: 20.0% > 10%" in capsys.readouterr().out

    run2 = run_dir.parent / "calib_ok"
    _write_events(run2 / "tb", {
        "win_prob/critic_reliability": [(1, 0.005), (2, 0.005)],
        "win_prob/critic_resolution": [(1, 0.10), (2, 0.10)],
    })
    tb_read.report(run2, tb_read.load(run2, ["win_prob/critic_reliability",
                                             "win_prob/critic_resolution"]),
                   ["win_prob/critic_reliability", "win_prob/critic_resolution"], n=20)
    out2 = capsys.readouterr().out
    assert "ok (5.0% <= 10%)" in out2
    assert "FLAG" not in out2


def test_a_run_with_no_tb_directory_is_a_refusal_not_an_empty_read(tmp_path, capsys):
    empty = tmp_path / "no_tb"
    empty.mkdir()
    rc = tb_read.main([str(empty), "--tag", "m/x"])
    assert rc == 1
    assert "NO SCALAR FOUND" in capsys.readouterr().out
