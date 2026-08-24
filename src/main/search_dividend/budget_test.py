"""The width allocator and the deadline — the two halves of "budget buys width, in THIS order"."""

from __future__ import annotations

import pytest

from main.search_dividend.budget import (WIDTH_ORDER, CostModel, Deadline, RealizedWidths,
                                         WidthCaps, WidthPlan, allocate)


class _Clock:
    """An injectable monotonic clock, so a deadline test never sleeps."""

    def __init__(self):
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


def test_the_registered_order_is_a_single_source():
    assert WIDTH_ORDER == ("m_opp", "k_worlds", "r_dice")


def test_budget_is_spent_on_opponent_actions_FIRST():
    """The registered order is the claim, so the test is that a budget which can afford exactly
    one axis spends it on the opponent — not on worlds, not on dice."""
    cost = CostModel(world_open_s=0.0, arm_s=0.01)
    # Every axis is MULTIPLICATIVE in the arm count: with 4 our-actions, m_opp=1 is 4 arms
    # (0.04 s) and m_opp=2 is 8 (0.08 s). A budget of 0.085 affords exactly one bump, and the
    # registered order says it must go to the opponent axis.
    plan = allocate(0.085, 4, cost, WidthCaps(m_opp=4, k_worlds=4, r_dice=4))
    assert plan.m_opp == 2, plan
    assert (plan.k_worlds, plan.r_dice) == (1, 1)


def test_worlds_come_before_dice_once_the_opponent_axis_is_capped():
    cost = CostModel(world_open_s=0.0, arm_s=0.01)
    caps = WidthCaps(m_opp=1, k_worlds=4, r_dice=4)
    plan = allocate(0.025, 1, cost, caps)      # 1 our-action, 1 candidate -> 0.01 s per world
    assert plan.m_opp == 1
    assert plan.k_worlds == 2
    assert plan.r_dice == 1


def test_dice_are_last_and_only_get_what_is_left():
    cost = CostModel(world_open_s=0.0, arm_s=0.01)
    caps = WidthCaps(m_opp=1, k_worlds=1, r_dice=8)
    plan = allocate(0.035, 1, cost, caps)
    assert (plan.m_opp, plan.k_worlds) == (1, 1)
    assert plan.r_dice == 3


def test_a_bigger_budget_moves_the_plan_LEXICOGRAPHICALLY_up():
    """The monotonicity a priority-ordered allocator actually has — and the one it does NOT.

    Per-axis monotonicity is FALSE here by design, and the falsifying case is worth recording:
    at a 1 s budget the plan is (m=6, k=1, r=2) and at 3 s it is (m=6, k=3, r=1) — the DICE axis
    went DOWN. That is not a bug, it is what "spend on worlds before dice" means: the extra
    second bought a world, and a world multiplies the arm count, so the dice resample it had been
    affording no longer fits. A test asserting per-axis monotonicity would therefore be asserting
    that the registered priority order is not honoured.

    The true invariant is lexicographic: the tuple (m_opp, k_worlds, r_dice) never decreases."""
    cost = CostModel(world_open_s=0.02, arm_s=0.01)
    caps = WidthCaps(m_opp=6, k_worlds=8, r_dice=8)
    prev = (0, 0, 0)
    seen_axis_decrease = False
    for budget in (0.05, 0.1, 0.5, 1.0, 3.0, 8.0):
        plan = allocate(budget, 4, cost, caps)
        cur = (plan.m_opp, plan.k_worlds, plan.r_dice)
        assert cur >= prev, f"lexicographic regression at budget {budget}: {prev} -> {cur}"
        seen_axis_decrease |= any(c < p for c, p in zip(cur, prev))
        prev = cur
    assert seen_axis_decrease, (
        "this sweep is supposed to CONTAIN a per-axis decrease — if it stops doing so the test "
        "has quietly become vacuous and no longer documents the trade-off")


def test_the_world_open_cost_is_charged_per_world_not_per_arm():
    """`open_root` replays the battle prefix, so it is charged K times. A model that charged it
    once would over-plan the world axis by exactly the factor that matters late-game."""
    cost = CostModel(world_open_s=1.0, arm_s=0.0)
    plan = allocate(2.5, 4, cost, WidthCaps(m_opp=1, k_worlds=8, r_dice=1))
    assert plan.k_worlds == 2


def test_a_floor_plan_is_returned_even_when_it_does_not_fit():
    """A decision must always have a plan; the CLOCK is what stops an over-budget one, because a
    zero-width plan would silently turn a search arm into the control."""
    cost = CostModel(world_open_s=100.0, arm_s=100.0)
    plan = allocate(0.001, 4, cost, WidthCaps())
    assert (plan.m_opp, plan.k_worlds, plan.r_dice) == (1, 1, 1)


def test_caps_below_the_floor_still_win():
    """The base arm's `k_worlds=0` cap must survive the floor — a base arm that searched one
    world would not be a control."""
    plan = allocate(10.0, 4, CostModel(), WidthCaps(m_opp=0, k_worlds=0, r_dice=0))
    assert (plan.m_opp, plan.k_worlds, plan.r_dice) == (0, 0, 0)


def test_n_arms_is_the_product_of_all_four_axes():
    assert WidthPlan(3, 2, 4).n_arms(5) == 3 * 2 * 4 * 5


@pytest.mark.parametrize("axis", WIDTH_ORDER)
def test_bumped_touches_exactly_one_axis(axis):
    base = WidthPlan(1, 1, 1)
    got = base.bumped(axis)
    assert getattr(got, axis) == 2
    for other in WIDTH_ORDER:
        if other != axis:
            assert getattr(got, other) == 1


# -- the clock ----------------------------------------------------------------


def test_deadline_expires_on_the_clock_not_on_the_plan():
    clk = _Clock()
    d = Deadline(1.0, clock=clk)
    assert not d.expired()
    assert d.fits(0.9)
    clk.t = 0.5
    assert d.remaining() == pytest.approx(0.5)
    assert not d.fits(0.6)     # declining an over-run BEFORE starting it is the point
    assert d.fits(0.4)
    clk.t = 1.0
    assert d.expired()


def test_realized_widths_round_trip_as_a_dict():
    """The row on disk is what the report reads, so the dataclass must serialize whole."""
    w = RealizedWidths(planned=WidthPlan(2, 3, 4).as_dict(), n_our_actions=5,
                       opp_candidates=2, worlds_gated_ok=3, worlds_gate_failed=1,
                       arms_scored=17, deadline_truncated=True, elapsed_s=0.42)
    d = w.as_dict()
    assert d["planned"] == {"m_opp": 2, "k_worlds": 3, "r_dice": 4}
    assert d["worlds_gate_failed"] == 1
    assert d["deadline_truncated"] is True
    assert d["elapsed_s"] == 0.42
