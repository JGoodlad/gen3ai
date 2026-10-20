"""K9(b)'s excluded-share CEILING is judged only on enough rows (`gen3_k9_excluded_min_rows_v1`, 2026-10-19).

THE BUG (backlog row "K9(b)'s excluded-share CEILING (0.15) FATALs tiny rollouts"): `FP32_EXCLUDED_CEILING` applied at
any row count, though it was derived on 1,024-row blocks (`designs/training/learner_gates.md`). A 57-row `--debug`
probe read 0.193 and a 108-row one 0.194 — healthy ties, clustered in a few states — and a hard FATAL followed in
a harness that has nothing to do with the check. The ceiling now needs `FP32_EXCLUDED_MIN_ROWS` current rows; below
it the share is REPORTED (``behaviour/excluded_frac``, ``behaviour/excluded_ceiling_judged`` = 0, one note per
process) and cannot FATAL. The ``max`` bar judges at every size.

Each test fails on revert of what it guards (the table's ``min_rows``, `judge_behaviour`'s skip). The production
guarantee is a PIN, not a hope: the production probe judges `batch_size // 2` current rows at the least
(`choose_rows`), and `rb_x26_s2601`'s 3,872 updates read ``behaviour/rows_current`` min 1,024.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from agents.training.rust_rollout import consistency as K
BehaviourMismatch = K.BehaviourMismatch

MIN = K.FP32_EXCLUDED_MIN_ROWS


@pytest.fixture(autouse=True)
def _fresh_note(monkeypatch):
    monkeypatch.setattr(K, "_CEILING_SKIP_NOTED", [])        # the once-per-process note, reset per test


def _rows(n: int, tied: int, *, bad: float = 0.0):
    """``n`` current rows at fp32's healthy noise (1e-6), the first ``tied`` at an EXACT tie (margin 0, so excluded
    whatever epsilon is); a non-tied row with ``|d log pi|`` = ``bad`` when asked. Returns ``(abs_d, margins)``."""
    d = np.full(n, 1e-6)
    margins = np.ones(n)
    margins[:tied] = 0.0
    if bad:
        d[n - 1] = bad
    return d, margins


def _enforce(n: int, tied: int, *, bad: float = 0.0, check: str = "fatal"):
    d, m = _rows(n, tied, bad=bad)
    model = SimpleNamespace(behaviour_check=check)
    return K.enforce_behaviour(model, d, where="the test's rows", margins=m, sites=["test"] * n)


def test_the_gate_table_declares_the_minimum_on_the_ceiling_only():
    assert [(c.statistic, c.min_rows) for c in K.BEHAVIOUR_GATE] == [("max", 0), ("excluded_frac", MIN)]
    assert MIN == 1024, "the block size the ceiling was derived on (learner_gates.md)"


def test_a_tiny_rollout_over_the_ceiling_is_REPORTED_not_FATAL(capsys):
    """The measured case: 11 of 57 rows at a tie = 0.193 > 0.15, every judged row clean."""
    out = _enforce(57, 11)                                                  # no raise
    assert out["behaviour/excluded_frac"] == pytest.approx(11 / 57) and out["behaviour/excluded_frac"] > K.FP32_EXCLUDED_CEILING
    assert out["behaviour/excluded_ceiling_judged"] == 0.0
    assert out["behaviour/violations_total_excluded_frac"] == 0.0
    note = capsys.readouterr().out
    assert "NOT judged on 57 current rows" in note and "0.193" in note, note
    _enforce(57, 11)
    assert capsys.readouterr().out == "", "the note is once per process, not per update"


def test_the_judgement_flag_is_per_condition_and_the_value_is_still_computed():
    d, m = _rows(57, 11)
    j = {x.condition.statistic: x for x in K.judge_behaviour(d, m)}
    assert j["max"].judged and j["max"].ok
    assert not j["excluded_frac"].judged and j["excluded_frac"].ok and j["excluded_frac"].value == pytest.approx(11 / 57)


@pytest.mark.parametrize("n", [MIN, MIN + 1, 2 * MIN])
def test_at_and_above_the_minimum_the_ceiling_still_FATALs(n):
    """A production-size probe (1,024 current rows is its floor, 2,048 its cap) over the ceiling: FATAL, as before."""
    tied = int(0.2 * n)
    with pytest.raises(BehaviourMismatch, match=r"TOO MANY rows sit at a tie.*ceiling 0\.15"):
        _enforce(n, tied)
    # and warn mode only warns
    assert _enforce(n, tied, check="warn")["behaviour/excluded_ceiling_judged"] == 1.0


def test_one_row_under_the_minimum_the_same_share_is_not_judged():
    """The decision is a COUNT: MIN rows at a 0.2 share FATAL, MIN - 1 rows at the same share do not."""
    with pytest.raises(BehaviourMismatch):
        _enforce(MIN, int(0.2 * MIN))
    out = _enforce(MIN - 1, int(0.2 * (MIN - 1)))
    assert out["behaviour/excluded_ceiling_judged"] == 0.0 and out["behaviour/excluded_frac"] > K.FP32_EXCLUDED_CEILING


def test_a_healthy_share_passes_at_every_size():
    for n in (1, 57, MIN - 1, MIN, 2 * MIN):
        assert _enforce(n, int(0.04 * n))["behaviour/violations_total_excluded_frac"] == 0.0


def test_a_tiny_rollout_still_judges_the_max_bar_and_the_message_says_what_was_not_judged():
    with pytest.raises(BehaviourMismatch, match=r"max 0\.001 NOT < 0\.0001.*excluded_frac 0\.193 \(not judged: under 1024 rows\)"):
        _enforce(57, 11, bad=1e-3)


def test_the_production_probe_never_judges_fewer_current_rows_than_the_minimum():
    """THE PRODUCTION GUARANTEE. The probe picks `batch_size` rows and takes at most half of them from the current
    version when older rows exist (`choose_rows`), so `behaviour/rows_current` >= batch_size // 2 whenever the buffer
    holds that many current rows (production: `staleness/current_share` min 0.91 of 98,304 rows). With the recipe's
    batch_size 2048 that is 1,024 == the minimum: the ceiling is judged on every production update, as before.
    Lower the production batch_size under 2 * MIN and this fails — the ceiling would otherwise go silently dark."""
    from utils.paths import repo_path

    batch = int(json.loads(repo_path("designs", "production_config.json").read_text())["recipe"]["fresh"]["batch_size"])
    assert batch // 2 >= MIN, (batch, MIN)
    rng = np.random.default_rng(0)
    n_buf = 98_304
    for older_share in (0.0, 0.09, 0.5, 0.9):                      # 0.09 = production's measured worst case
        ages = np.zeros(n_buf, dtype=np.int64)
        ages[rng.permutation(n_buf)[:int(older_share * n_buf)]] = 1
        picked = K.choose_rows(ages, batch, np.random.default_rng(1))
        assert picked.size == batch
        assert int((ages[picked] == 0).sum()) >= MIN, older_share
