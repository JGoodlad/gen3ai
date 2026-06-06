"""Unit tests for the TD-residual tail fold (#4), the eval-cycle aggregator over per-decision
critic surprise δ. CVaR@frac isolates the value-cliffs; below min_samples it reports the single
worst δ; empty → None."""

import pytest

from agents.training.eval_callback import td_tail, TD_TAIL_FRAC, TD_TAIL_MIN_SAMPLES


def test_td_tail_empty_is_none():
    assert td_tail([]) is None


def test_td_tail_below_min_samples_returns_single_min():
    rs = [-3.0, 1.0, -10.0, 2.0]   # < TD_TAIL_MIN_SAMPLES
    assert len(rs) < TD_TAIL_MIN_SAMPLES
    assert td_tail(rs) == -10.0


def test_td_tail_cvar_over_worst_fraction():
    # 100 residuals 0..-99 → worst 5% are the 5 most negative (-99..-95) → mean -97.
    rs = [-float(i) for i in range(100)]
    assert td_tail(rs, frac=0.05) == pytest.approx(-97.0)


def test_td_tail_more_negative_when_the_tail_worsens():
    base = [-float(i) for i in range(100)]
    worse = base[:-1] + [-500.0]   # one catastrophic extra cliff
    assert td_tail(worse) < td_tail(base)


def test_td_tail_default_frac_is_five_percent():
    assert TD_TAIL_FRAC == pytest.approx(0.05)
