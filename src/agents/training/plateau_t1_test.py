"""The plateau's TIER-1 rule (``plateau_t1``): the registered constants, the GSPRT's boundaries on synthetic rows,
the rounding band, the cap's UNDECIDED, the abort rule, the decision vocabulary, the check plan and the run-level
status. Every case is a fixed input with one right answer (standing rule 8)."""
from __future__ import annotations

import random

import pytest

from agents.training import plateau_t1 as T
from agents.training import sprt as SP
from agents.training.eval_ledger import testkit as K

REQ = {"request_id": "plateau_t1:run:60000000:w50000000", "kind": "plateau_t1", "family": None,
       "ts": "2026-10-06T12:00:00+00:00"}
EVEN = (0, 4, 32, 4, 0)          # score 0.500, low variance: H0's side
WIN = (0, 2, 26, 10, 2)          # score 0.575: H1's side
LOSE = (2, 10, 26, 2, 0)         # score 0.425


def rows_of(batches, *, req=REQ, voided=0, aborted=None, p=K.SHA_A, o=K.SHA_B):
    return [K.make_row(f"w:{b}", request=req, batch=b, pc=pc, purpose="plateau", voided=voided, aborted=aborted, p=p, o=o,
                       cycle_seed=100 + b) for b, pc in enumerate(batches)]


def test_the_registered_rule_is_section_8_1():
    r = T.REGISTERED
    assert (r.p0, r.p1, r.alpha, r.beta, r.batch_pairs, r.min_pairs, r.max_pairs, r.abort_limit) == \
        (0.50, 0.52, 0.05, 0.05, 40, 40, 6000, 0.25)
    assert r.registered and r.bounds == SP.bounds(0.05, 0.05)
    assert T.Tier1Rule.parse(r.rule_string()) == r
    other = T.Tier1Rule(max_pairs=80)
    assert not other.registered and T.Tier1Rule.parse(other.rule_string()) == other
    assert other.rule_string() != r.rule_string()
    for bad in ("plateau_t1_gsprt(p0=0.5)", "sprt(p0=0.5)", r.rule_string().replace("wald", "siegmund")):
        with pytest.raises(ValueError):
            T.Tier1Rule.parse(bad)
    with pytest.raises(ValueError):
        T.Tier1Rule(max_pairs=100)       # not a whole number of 40-pair batches


def test_the_verdict_vocabulary_is_closed():
    assert T.VERDICTS == ("GAIN", "FLAT", "UNDECIDED", "INCONCLUSIVE")
    assert T.CONTINUE not in T.VERDICTS
    assert T.STATUSES == ("NOT_YET", "CLIMBING", "FLAT_ONCE", "TIER1_PLATEAU", "CONTINUE")


def test_even_batches_accept_h0_at_the_first_lower_crossing():
    res = T.evaluate(rows_of([EVEN] * 8))
    # the LLR trail crosses the lower Wald bound first at the 5th batch (200 pairs) and the test STOPS there
    assert (res.verdict, res.reason, res.stop_batch, res.n_pairs) == (T.FLAT, T.BY_LOWER, 4, 200)
    assert res.llr < SP.bounds()[0] < SP.llr([0, 16, 128, 16, 0], 0.5, 0.52)
    assert res.surplus_batches == [5, 6, 7]


def test_winning_batches_accept_h1():
    res = T.evaluate(rows_of([WIN] * 3))
    assert (res.verdict, res.reason, res.stop_batch, res.n_pairs) == (T.GAIN, T.BY_UPPER, 1, 80)
    assert res.llr > SP.bounds()[1]
    assert T.evaluate(rows_of([WIN])).verdict == T.CONTINUE   # one batch is not enough


def test_losing_batches_are_flat_never_a_separate_verdict():
    assert T.evaluate(rows_of([LOSE] * 2)).verdict == T.FLAT


def test_the_fold_is_in_batch_order_whatever_the_ledger_order():
    rows = rows_of([WIN, EVEN, LOSE, EVEN, EVEN, WIN, EVEN, EVEN])
    want = T.evaluate(rows).to_json()
    for seed in range(5):
        shuffled = list(rows)
        random.Random(seed).shuffle(shuffled)
        assert T.evaluate(shuffled).to_json() == want


def test_the_cap_is_undecided_not_h0():
    rule = T.Tier1Rule(max_pairs=80)
    # alternating halves keep the LLR inside both bounds for two batches
    res = T.evaluate(rows_of([(0, 4, 31, 5, 0), (0, 5, 31, 4, 0), EVEN]), rule)
    lo, hi = rule.bounds
    assert lo < res.llr < hi
    assert (res.verdict, res.reason, res.stop_batch, res.surplus_batches) == (T.UNDECIDED, T.BY_CAP, 1, [2])


def test_an_llr_within_the_rounding_band_is_not_a_crossing():
    lo, hi = T.REGISTERED.bounds
    assert T._crossing(hi + 0.5 * T.ROUNDING_BAND, lo, hi) is None
    assert T._crossing(hi, lo, hi) is None
    assert T._crossing(hi + 2 * T.ROUNDING_BAND, lo, hi) == T.BY_UPPER
    assert T._crossing(lo - 0.5 * T.ROUNDING_BAND, lo, hi) is None
    assert T._crossing(lo - 2 * T.ROUNDING_BAND, lo, hi) == T.BY_LOWER


def test_min_pairs_defers_a_crossing():
    rule = T.Tier1Rule(min_pairs=120)
    res = T.evaluate(rows_of([WIN] * 3), rule)
    assert (res.verdict, res.stop_batch) == (T.GAIN, 2)       # crossed at batch 1, decided at the minimum


def test_too_many_aborted_games_is_inconclusive():
    # 28 finished pairs + 12 voided per batch: 24 aborted of 80 attempted games (30 % > 25 %); the LLR crosses the
    # upper bound at the third batch, where the abort rule turns GAIN into INCONCLUSIVE
    rows = rows_of([(0, 0, 0, 8, 20)] * 4, voided=12, aborted=24)
    res = T.evaluate(rows)
    assert res.n_pairs == 84 and res.stop_batch == 2 and (res.verdict, res.reason) == (T.INCONCLUSIVE, T.BY_ABORTS)
    assert res.aborted_games == 72 and res.attempted_games == 240
    # the same rows under a 50 % abort limit are a GAIN at the same batch
    lenient = T.evaluate(rows, T.Tier1Rule(abort_limit=0.5))
    assert (lenient.verdict, lenient.stop_batch) == (T.GAIN, 2)


@pytest.mark.parametrize("mutate,match", [
    (lambda rs: rs[:1] + rs[2:], "gap"),
    (lambda rs: rs + [dict(rs[0], row_id="dup")], "twice"),
    (lambda rs: rs + rows_of([EVEN], o=K.SHA_C)[:1], "combinations"),
])
def test_rows_that_are_not_one_test_are_refused(mutate, match):
    with pytest.raises(T.Tier1RowsError, match=match):
        T.evaluate(mutate(rows_of([EVEN, EVEN, EVEN])))


def test_a_batch_of_the_wrong_length_is_refused():
    with pytest.raises(T.Tier1RowsError, match="40"):
        T.evaluate(rows_of([(0, 4, 30, 4, 0)]))


def test_rederive_owns_only_its_rule():
    rows = rows_of([WIN] * 2)
    d = {"rule": T.REGISTERED.rule_string()}
    assert T.rederive(d, rows) == T.GAIN
    assert T.rederive({"rule": "plateau_two_tier(x=1)"}, rows) is None
    got = T.rederive(d, rows[1:])
    assert got.startswith("REFUSED(") and got not in T.VERDICTS


def test_the_ledger_audit_registers_the_rule():
    from agents.training.eval_ledger import audit as A

    assert A.RULES["plateau"]({"rule": T.REGISTERED.rule_string()}, rows_of([LOSE] * 2)) == T.FLAT


# ------------------------------------------------------------------------------------------------ the plan
def test_plan_checks_nodes_and_missing_nodes():
    steps = [10_000_180, 20_000_100, 30_000_050, 40_600_000, 50_000_200, 60_000_300, 70_000_000, 71_000_000]
    checks = T.plan_checks(steps)
    assert [c.grid for c in checks] == [50_000_000, 60_000_000, 70_000_000]
    assert checks[0] == T.Check(50_000_000, 50_000_200, None)            # no checkpoint at step 0
    assert checks[1] == T.Check(60_000_000, 60_000_300, 10_000_180) and checks[1].playable
    assert checks[2] == T.Check(70_000_000, 70_000_000, 20_000_100)
    # 40.6M is past the 500k slack: the 40M node is missing, so the 90M check's W-back would be missing
    assert T.node_at(40_000_000, steps, T.DEFAULT_PLAN.slack_steps) is None
    assert T.plan_checks([]) == []
    assert T.request_id("rb_x26", 60_000_000) == "plateau_t1:rb_x26:60000000:w50000000"


def test_the_plan_refuses_an_ambiguous_slack():
    with pytest.raises(ValueError):
        T.CheckPlan(delta_steps=10, lag=1, slack_steps=10)


# ------------------------------------------------------------------------------------------------ the status
D = 10_000_000


@pytest.mark.parametrize("checks,status,flat", [
    ([], T.NOT_YET, []),
    ([(50 * D // 10, "NOT_STARTED"), (60 * D // 10, T.CONTINUE)], T.NOT_YET, []),
    ([(60 * D // 10, T.GAIN)], T.CLIMBING, []),
    ([(60 * D // 10, T.FLAT)], T.FLAT_ONCE, [60 * D // 10]),
    ([(60 * D // 10, T.FLAT), (70 * D // 10, T.FLAT)], T.TIER1_PLATEAU, [60 * D // 10, 70 * D // 10]),
    ([(60 * D // 10, T.FLAT), (70 * D // 10, T.UNDECIDED), (80 * D // 10, T.FLAT)], T.FLAT_ONCE, [80 * D // 10]),
    ([(60 * D // 10, "MISSING_NODE"), (70 * D // 10, T.FLAT)], T.FLAT_ONCE, [70 * D // 10]),
    ([(60 * D // 10, T.FLAT), (70 * D // 10, T.GAIN)], T.CLIMBING, []),
    ([(60 * D // 10, T.FLAT), (70 * D // 10, T.INCONCLUSIVE)], T.CONTINUE_STATUS, []),
])
def test_tier1_status(checks, status, flat):
    got = T.tier1_status(checks, D)
    assert got["status"] == status and got["flat_checks"] == flat


def test_a_pending_check_does_not_erase_a_called_plateau():
    got = T.tier1_status([(6 * D, T.FLAT), (7 * D, T.FLAT), (8 * D, T.CONTINUE)], D)
    assert got["status"] == T.TIER1_PLATEAU and got["as_of_check"] == 7 * D and got["pending_after"] == [8 * D]
