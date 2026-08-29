"""The offline A/B's arithmetic and the one equivalence its validity rests on.

The A/B's whole claim is that the two allocators see the SAME samples, so the difference between
them is allocation. That claim has two halves, and both are tested here rather than argued: the
bank is drawn by the same round loop the live racer uses (so the samples are the ones a live race
would have got), and the replay charges each arm for exactly what it consumed.
"""

from __future__ import annotations

import json
import math

import pytest

from main.search_dividend.ab_racing import (AGREEMENT_TARGETS, BankedDecision, Price,
                                            _budget_ratios, _grid_under_seconds,
                                            _race_under_seconds, bank_decision, replay,
                                            separation_profile)
from main.search_dividend.budget import WidthCaps
from main.search_dividend.racing import RacingConfig
from main.search_dividend.racing_test import _Session, _bank, _choose, _engine, _script
from main.search_dividend.search_test import OBSERVED, RECORD

CFG = RacingConfig(rule="z", min_samples=3)


def _decision(rows, actions=(0, 1, 2, 3), policy_action=1, battle="b", inv=0, turn=5):
    return BankedDecision(battle=battle, inv=inv, turn=turn, actions=list(actions),
                          policy_action=policy_action,
                          rounds=[[r[a] for a in actions] for r in rows])


# -- the bank is the live loop ------------------------------------------------


def test_the_bank_draws_the_SAME_first_round_the_live_racer_would():
    """The equivalence the A/B rests on. `bank_decision` is `_run_racing`'s loop with the
    elimination removed; if the two ever drew different worlds or different dice, the banked
    samples would not be the ones a live race sees and every curve below would be about a
    different search."""
    def first_round(strategy):
        eng = _engine(strategy, caps=WidthCaps(m_opp=2, k_worlds=4, r_dice=4))
        eng._session = _Session(_Root_ok())
        seen = _script(eng, [{0: 0.4, 1: 0.3, 2: 0.2, 3: 0.1}] * 12)
        if strategy == "racing":
            _choose(eng)
        else:
            bank_decision(eng, record=RECORD, side="p1", turn=1, our_history=[],
                          our_tokens={0: "move surf", 1: "move ice", 2: "switch 2",
                                      3: "switch 3"},
                          observed_our_lines=OBSERVED, pub=None, rounds=4, m_opp=2,
                          opp_true_packed="T2")
        return seen[0]

    assert first_round("racing") == first_round("grid")


def _Root_ok():
    from main.search_dividend.racing_test import REQ
    from main.search_dividend.search_test import _Root
    return _Root(OBSERVED, requests=REQ)


def test_the_bank_keeps_only_rounds_where_EVERY_action_scored():
    """A partial round would give the two allocators different samples on the same index, which is
    the one thing the paired design must not allow."""
    eng = _engine("grid", caps=WidthCaps(m_opp=2, k_worlds=4, r_dice=4))
    eng._session = _Session(_Root_ok())
    _script(eng, [{0: 0.4, 1: 0.3, 2: 0.2}] * 2 + [{0: 0.4, 1: 0.3, 2: 0.2, 3: 0.1}] * 2)
    bank, meta = bank_decision(
        eng, record=RECORD, side="p1", turn=1, our_history=[],
        our_tokens={0: "move surf", 1: "move ice", 2: "switch 2", 3: "switch 3"},
        observed_our_lines=OBSERVED, pub=None, rounds=4, m_opp=2, opp_true_packed="T2")
    assert len(bank) == 2 and meta["dropped"] == 2
    assert all(set(row) == {0, 1, 2, 3} for row in bank)


def test_the_bank_prices_a_round_and_an_arm_from_its_OWN_measurement():
    eng = _engine("grid", caps=WidthCaps(m_opp=2, k_worlds=4, r_dice=4))
    eng._session = _Session(_Root_ok())
    _script(eng, [{0: 0.4, 1: 0.3, 2: 0.2, 3: 0.1}] * 4)
    _bankrows, meta = bank_decision(
        eng, record=RECORD, side="p1", turn=1, our_history=[],
        our_tokens={0: "move surf", 1: "move ice", 2: "switch 2", 3: "switch 3"},
        observed_our_lines=OBSERVED, pub=None, rounds=4, m_opp=2, opp_true_packed="T2")
    assert meta["opens"] == 4 and meta["open_s"] >= 0.0


# -- the replay's accounting --------------------------------------------------


def test_a_wall_clock_budget_buys_the_grid_only_WHOLE_rounds():
    """A partial round is not a cheaper round — it is a round scored on some actions and not
    others, which is not a sample of anything."""
    bank = _bank([1.0, 0.0, 0.0, 0.0], sd=0.05, n=20)
    price = Price(open_s=0.1, arm_s=0.1)          # a round = 0.1 + 4*0.1 = 0.5 s
    _act, rounds, arms = _grid_under_seconds(bank, [0, 1, 2, 3], 1.7, price, 1)
    assert rounds == 3 and arms == 12


def test_racing_buys_MORE_rounds_than_the_grid_from_the_same_seconds():
    """The mechanism, priced. A narrowed field makes each later round cheaper, so the same clock
    reaches further into the bank — which is the only way an adaptive allocator can pay for the
    samples it spent proving the field was narrow."""
    # TIED leaders: the field narrows to two and then stays there, which is the configuration in
    # which extra budget has somewhere to go. A field that RESOLVES stops instead — correctly, but
    # it is a different saving (unspent clock, not more samples) and mixing them would hide both.
    bank = _bank([1.0, 1.0, 0.0, 0.0], sd=0.03, n=40, seed=1)
    price = Price(open_s=0.01, arm_s=0.1)
    _g, g_rounds, _ga = _grid_under_seconds(bank, [0, 1, 2, 3], 2.0, price, 1)
    _r, r_rounds, _ra = _race_under_seconds(bank, [0, 1, 2, 3], 2.0, price, CFG, 1)
    assert r_rounds > g_rounds


def test_a_budget_too_small_for_one_round_scores_NOTHING_rather_than_guessing():
    bank = _bank([1.0, 0.0], sd=0.05, n=10)
    price = Price(open_s=1.0, arm_s=1.0)
    assert _grid_under_seconds(bank, [0, 1], 0.5, price, 0)[0] is None
    assert _race_under_seconds(bank, [0, 1], 0.5, price, CFG, 0)[0] is None


def test_the_open_overhead_is_charged_PER_ROUND_so_racing_pays_more_of_it():
    """The honest half of the cost model. Racing runs more, cheaper rounds, so it pays the
    per-round `open_root` more often — an arm-evaluation-only ledger would hide that entirely and
    report a saving the clock does not see."""
    bank = _bank([1.0, 1.0, 0.0, 0.0], sd=0.02, n=40)
    dear = Price(open_s=10.0, arm_s=0.001)        # opens dominate
    _g, g_rounds, _ = _grid_under_seconds(bank, [0, 1, 2, 3], 50.0, dear, 1)
    _r, r_rounds, _ = _race_under_seconds(bank, [0, 1, 2, 3], 50.0, dear, CFG, 1)
    assert r_rounds == g_rounds                   # elimination buys nothing when opens dominate


# -- the report ---------------------------------------------------------------


def _report(**kw):
    decisions = [_decision(_bank([1.0, 0.5, 0.0, 0.0], sd=0.05, n=16, seed=s))
                 for s in range(8)]
    return replay(decisions, price=Price(open_s=0.05, arm_s=0.01), cfg=CFG,
                  budgets_s=kw.get("budgets", [0.09, 0.2, 0.5, 1.0]))


def test_the_report_publishes_the_GOLD_STABILITY_check_rather_than_assuming_it():
    """A gold reference that moves when you double its budget is not a reference, and the number
    has to be in the artifact — an unstable gold makes every agreement figure below it a
    measurement of the gold's own noise."""
    rep = _report()
    assert "doubling_check_frac" in rep["gold"]
    assert 0.0 <= rep["gold"]["doubling_check_frac"] <= 1.0


def test_both_arms_are_reported_at_EVERY_swept_budget():
    rep = _report()
    assert len(rep["curves"]["grid"]) == len(rep["curves"]["racing"]) == 4
    assert [r["budget_s"] for r in rep["curves"]["grid"]] == \
           [r["budget_s"] for r in rep["curves"]["racing"]]


def test_a_decision_with_fewer_than_three_legal_actions_is_EXCLUDED():
    """The registration's own scope: with two actions there is nothing for an allocator to
    allocate, and including them would dilute both arms toward 100% agreement."""
    rep = replay([_decision(_bank([1.0, 0.0], sd=0.05, n=16), actions=(0, 1)),
                  _decision(_bank([1.0, 0.5, 0.0], sd=0.05, n=16), actions=(0, 1, 2))],
                 price=Price(), cfg=CFG, budgets_s=[1.0])
    assert rep["n_decisions"] == 1


def test_the_budget_ratio_is_reported_at_every_target_and_is_None_when_unreached():
    rows = {"grid": [{"budget_s": 1.0, "agreement": 0.5}, {"budget_s": 2.0, "agreement": 1.0}],
            "racing": [{"budget_s": 1.0, "agreement": 0.5}]}
    out = _budget_ratios(rows)
    assert set(out) == {f"{int(t*100)}%" for t in AGREEMENT_TARGETS}
    # grid hits 80% by interpolation between (1.0, 0.5) and (2.0, 1.0); racing never does.
    assert math.isclose(out["80%"]["grid_s"], 1.6, abs_tol=1e-6)
    assert out["80%"]["racing_s"] is None and out["80%"]["reduction_x"] is None


def test_the_budget_ratio_INTERPOLATES_rather_than_quantizing_to_a_swept_point():
    rows = {"grid": [{"budget_s": 1.0, "agreement": 0.80}],
            "racing": [{"budget_s": 0.2, "agreement": 0.60},
                       {"budget_s": 0.4, "agreement": 1.00}]}
    out = _budget_ratios(rows)
    assert math.isclose(out["80%"]["racing_s"], 0.3, abs_tol=1e-6)
    assert math.isclose(out["80%"]["reduction_x"], 1.0 / 0.3, rel_tol=1e-3)


def test_the_separation_profile_reports_the_NEVER_bucket_explicitly():
    """Where racing saves nothing. A mean over the decisions that DID separate would hide this
    mass completely, and it is the half a time manager has to act on."""
    prof = separation_profile(
        [_decision(_bank([1.0, 0.0, 0.0, 0.0], sd=0.02, n=20, seed=1)),
         _decision(_bank([0.0, 0.0, 0.0, 0.0], sd=0.30, n=20, seed=2))], RacingConfig(rule="seq"))
    assert prof["n"] == 2
    assert prof["never_separated"] == 1
    assert prof["histogram"]["never"] == 1
    assert prof["median_rounds_to_separate"] is not None


def test_a_banked_decision_round_trips_through_JSON():
    """The bank is written once and replayed many times, so a lossy round trip would silently
    change every curve cut after the first."""
    d = _decision(_bank([1.0, 0.5, 0.0, 0.0], sd=0.05, n=6))
    back = BankedDecision(**json.loads(json.dumps(d.as_dict())))
    assert back.bank() == d.bank()


def test_the_replay_is_deterministic():
    assert _report() == _report()


@pytest.mark.parametrize("budget", [0.09, 0.2, 1.0])
def test_neither_arm_can_score_a_decision_it_never_sampled(budget):
    """`n` counts decisions the arm actually produced an action for. An arm that could not afford a
    single round must shrink its own denominator rather than borrow the other's."""
    rep = _report(budgets=[budget])
    for arm in ("grid", "racing"):
        row = rep["curves"][arm][0]
        assert row["n"] <= rep["n_decisions"]
        if row["n"]:
            assert row["mean_rounds"] >= 1
