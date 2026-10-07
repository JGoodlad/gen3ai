"""gen3_speed_physics_v1 (v143, architecture audit F7b) — THE gen-3 move-order rule (`move_order.py`), each rule on
constructed values. Every test fails on a revert of the rule it names (the Showdown citation is in the module).
"""
from __future__ import annotations

import itertools

import pytest
import torch

from agents.gen3_data import priors
from agents.model.move_order import (QUICK_CLAW_P, belief_speed_scale, gen3_final_speed, gen3_para_speed,
                                     gen3_speed_stat, gen3_stage_speed, p_first_quick_claw,
                                     p_first_same_priority, p_outspeed_belief, p_seat_first, quick_claw_live)

T = torch.tensor


# ------------------------------------------------------------------------------------------ the stat
def test_speed_stat_is_showdowns_integer_formula() -> None:
    """``statModify``: tr(2·base + iv + tr(ev/4)) + 5, then the nature tr(stat·110/100) — against the facade's
    exact integer `priors.gen3_stat` over a grid, fed as the observation encodes them (iv/31, ev/252, the
    nature multiplier in float32)."""
    for base, ev, mult in itertools.product((5, 45, 80, 100, 130, 160), range(0, 253, 4), (0.9, 1.0, 1.1)):
        iv_obs = T([31.0 / 31.0]) * 31.0
        ev_obs = T([ev / 252.0]) * 252.0
        got = gen3_speed_stat(T([float(base)]), iv_obs, ev_obs, T([mult], dtype=torch.float32))
        assert int(got.item()) == priors.gen3_stat(base, ev, mult), (base, ev, mult)
    # Jolteon, Timid 252: (260 + 31 + 63 + 5) · 1.1 = 394.9 → 394 (a FLOOR, not a round)
    assert gen3_speed_stat(T([130.0]), T([31.0]), T([252.0]), T([1.1])).item() == 394.0
    # an odd EV floors before the sum: 255/4 = 63.75 → 63
    assert gen3_speed_stat(T([130.0]), T([31.0]), T([255.0]), T([1.0])).item() == 359.0


# ---------------------------------------------------------------------------------------- the stages
@pytest.mark.parametrize("stage,want", [(0, 301), (1, 451), (2, 602), (6, 1204), (-1, 200), (-2, 150), (-6, 75),
                                        (7, 1204), (-9, 75)])
def test_stage_is_getstats_floor_table(stage: int, want: int) -> None:
    """``getStat``: floor(stat·(2+s)/2) up, floor(stat·2/(2−s)) down, clamped to ±6."""
    assert gen3_stage_speed(T([301.0]), T([float(stage)])).item() == want


def test_stage_reads_the_observations_float_stage_as_its_integer() -> None:
    """(pos − neg)·6 from the observation can land a hair off the integer; the stage is ROUNDED."""
    assert gen3_stage_speed(T([301.0]), T([1.9999998])).item() == 602
    assert gen3_stage_speed(T([301.0]), T([-1.0000002])).item() == 200


# ---------------------------------------------------------------------------------------- paralysis
@pytest.mark.parametrize("stat,want", [(300, 75), (301, 75), (302, 75), (303, 76), (8, 2), (394, 98)])
def test_paralysis_is_modify_quarter_rounding_half_down(stat: int, want: int) -> None:
    """``modify(spe, 0.25)`` = tr((tr(spe·1024) + 2047)/4096): 302/4 = 75.5 → 75 (half DOWN), 303/4 → 76 (not a
    floor: floor(75.75) = 75)."""
    assert gen3_para_speed(T([float(stat)]), T([1.0])).item() == want
    assert gen3_para_speed(T([float(stat)]), T([0.0])).item() == stat


def test_stage_applies_before_paralysis() -> None:
    """``getStat`` boosts first, then ``ModifySpe``: 303 at +1 is floor(454.5) = 454 → modify → 113; the reverse
    order would give 76 → 114."""
    assert gen3_final_speed(T([303.0]), T([1.0]), T([1.0])).item() == 113
    assert gen3_stage_speed(gen3_para_speed(T([303.0]), T([1.0])), T([1.0])).item() == 114


def test_paralysis_quarters_a_believed_speed_and_flips_the_order() -> None:
    """Their believed 300 (spread 10) vs our 200: we are slower; paralysed, their mean AND spread quarter."""
    mu, sd = T([300.0]), T([10.0])
    assert p_outspeed_belief(T([200.0]), mu, sd).item() < 1e-6
    m = belief_speed_scale(T([0.0]), T([1.0]))
    assert m.item() == 0.25
    assert p_outspeed_belief(T([200.0]), mu * m, sd * m).item() > 1 - 1e-6
    # a +2 stage doubles the belief; a −1 makes it 2/3
    assert belief_speed_scale(T([2.0]), T([0.0])).item() == 2.0
    assert belief_speed_scale(T([-1.0]), T([0.0])).item() == pytest.approx(2.0 / 3.0)


# --------------------------------------------------------------------------------------- the speed tie
def test_an_exact_speed_tie_is_a_coin_flip() -> None:
    """``speedSort`` shuffles equal speeds: a point belief AT our speed reads ½, and so does a spread belief
    centred on it (the lattice integral is symmetric)."""
    assert p_outspeed_belief(T([394.0]), T([394.0]), T([0.0])).item() == 0.5
    assert p_outspeed_belief(T([394.0]), T([394.0]), T([7.5])).item() == pytest.approx(0.5, abs=1e-6)


def test_a_point_belief_is_the_exact_step() -> None:
    assert p_outspeed_belief(T([395.0]), T([394.0]), T([0.0])).item() == 1.0
    assert p_outspeed_belief(T([393.0]), T([394.0]), T([0.0])).item() == 0.0


def test_the_integral_runs_over_the_integer_lattice() -> None:
    """A belief one stat point below us, nearly certain: we are first (≈ 1) — half of the tie mass is a
    coin flip, so a belief centred ON us reads ½, not the 1 a strict ``>`` would give."""
    assert p_outspeed_belief(T([300.0]), T([299.0]), T([0.05])).item() > 1 - 1e-6
    assert p_outspeed_belief(T([300.0]), T([300.0]), T([0.05])).item() == pytest.approx(0.5, abs=1e-6)


def test_a_wider_belief_pulls_the_probability_toward_half() -> None:
    """The uncertainty-aware form: a fixed gap reads less certain as the believed spread widens."""
    sds = [2.0, 5.0, 10.0, 20.0, 40.0, 80.0]
    above = [p_outspeed_belief(T([300.0]), T([280.0]), T([s])).item() for s in sds]
    below = [p_outspeed_belief(T([260.0]), T([280.0]), T([s])).item() for s in sds]
    assert all(a > b for a, b in zip(above, above[1:])) and above[-1] > 0.5
    assert all(a < b for a, b in zip(below, below[1:])) and below[-1] < 0.5
    assert above[0] > 0.99 and below[0] < 0.01


def test_the_belief_mean_carries_a_gradient() -> None:
    mu = T([290.0], requires_grad=True)
    p_outspeed_belief(T([300.0]), mu, T([10.0])).sum().backward()
    assert mu.grad is not None and mu.grad.item() < 0     # a faster belief lowers P(we are first)


# ------------------------------------------------------------------------------------ priority bracket
def test_the_priority_bracket_decides_before_speed() -> None:
    p_out = T([0.9])
    assert p_seat_first(T([0.0]), T([1.0]), p_out).item() == 1.0     # their +1 beats our faster 0
    assert p_seat_first(T([1.0]), T([0.0]), p_out).item() == 0.0     # our +1 always first
    assert p_seat_first(T([-3.0]), T([0.0]), T([1.0])).item() == 1.0  # Focus Punch (−3) moves last
    assert p_seat_first(T([0.0]), T([0.0]), p_out).item() == pytest.approx(0.1)   # equal: speed


# --------------------------------------------------------------------------------------- Quick Claw
def test_quick_claw_is_one_shared_roll_per_turn() -> None:
    """Gen 3: ``quickClawRoll = randomChance(1, 5)`` once per turn; on it a holder's speed is 65535, two holders
    tie (a coin flip)."""
    p = T([0.3])
    q = QUICK_CLAW_P
    assert q == 0.2
    assert p_first_quick_claw(p, T([0.0]), T([0.0])).item() == pytest.approx(0.3)
    assert p_first_quick_claw(p, T([1.0]), T([0.0])).item() == pytest.approx(0.8 * 0.3 + 0.2)
    assert p_first_quick_claw(p, T([0.0]), T([1.0])).item() == pytest.approx(0.8 * 0.3)
    assert p_first_quick_claw(p, T([1.0]), T([1.0])).item() == pytest.approx(0.8 * 0.3 + 0.2 * 0.5)
    # a belief: P(they hold it) = 0.5 is the average of the two worlds
    assert p_first_quick_claw(p, T([0.0]), T([0.5])).item() == pytest.approx(0.5 * 0.3 + 0.5 * 0.8 * 0.3)
    assert p_first_same_priority(T([300.0]), T([300.0]), T([0.0]), T([1.0]), T([0.0])).item() \
        == pytest.approx(0.8 * 0.5 + 0.2)


def test_quick_claw_is_banned_in_gen3ou() -> None:
    """Owner 2026-10-07 + Showdown master `config/formats.ts` ``[Gen 3] OU`` banlist: the op's term is OFF — read
    from the FORMAT SPEC (one source), so a planted spec without the ban turns it on."""
    assert quick_claw_live() is False


def test_quick_claw_live_reads_the_format_spec(monkeypatch) -> None:
    import dataclasses

    from agents.gen3_data import format_spec as fs
    rules = tuple(dataclasses.replace(r, bans=tuple(b for b in r.bans if b.ids != ("quickclaw",)))
                  for r in fs.GEN3OU.rules)
    monkeypatch.setattr(fs, "_ACTIVE", fs.FormatSpec(fs.FORMAT_ID, rules))
    assert quick_claw_live() is True
