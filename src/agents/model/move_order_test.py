"""gen3_speed_physics_v1 (v143, architecture audit F7b) — THE gen-3 move-order rule (`move_order.py`), each rule on
constructed values. Every test fails on a revert of the rule it names (the Showdown citation is in the module).
"""
from __future__ import annotations

import itertools

import pytest
import torch

from agents.gen3_data import priors
from agents.model.move_order import (QUICK_CLAW_P, gen3_final_speed, gen3_para_speed, gen3_speed_stat,
                                     gen3_stage_speed, p_first_quick_claw, p_first_same_priority,
                                     p_outspeed_mixture, p_seat_first, quick_claw_live)

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


# ------------------------------------------------------------------------- the DISCRETE speed mixture
def _mix(points: dict, stage: float = 0.0, para: float = 0.0, V: int = 512):
    """A mixture ``{speed: weight}`` on the lattice, through the exact stage / paralysis: ``(final [1,V], cum
    [1,V+1])`` as `damage_op_speed._opp_speed_mix` builds them."""
    w = torch.zeros(1, V, dtype=torch.float64)
    for v, p in points.items():
        w[0, v] = p
    final = gen3_final_speed(torch.arange(V, dtype=torch.float32).view(1, -1), T([[stage]]), T([[para]]))
    cum = torch.cat([torch.zeros(1, 1, dtype=torch.float64), w.cumsum(-1)], dim=-1)
    return final, cum


def _p(ours: float, points: dict, **kw) -> float:
    final, cum = _mix(points, **kw)
    return p_outspeed_mixture(T([[ours]]), final, cum).item()


def test_a_lumpy_spread_keeps_its_max_speed_mass() -> None:
    """THE case the Gaussian missed (2026-10-07): a Blissey whose sets are 90 % uninvested (148) and 10 % Timid
    252 (229), against our 214 — P(we first) is 0.9, never the 1.0 a Gaussian at 148 ± 5 gave."""
    assert _p(214.0, {148: 0.9, 229: 0.1}) == pytest.approx(0.9)
    assert _p(230.0, {148: 0.9, 229: 0.1}) == pytest.approx(1.0)
    assert _p(100.0, {148: 0.9, 229: 0.1}) == 0.0


def test_an_exact_speed_tie_is_a_coin_flip() -> None:
    """``speedSort`` shuffles equal speeds: mass AT our speed counts ½."""
    assert _p(394.0, {394: 1.0}) == 0.5
    assert _p(300.0, {299: 0.5, 300: 0.5}) == pytest.approx(0.75)
    assert _p(395.0, {394: 1.0}) == 1.0 and _p(393.0, {394: 1.0}) == 0.0


def test_their_stage_and_paralysis_take_the_exact_arithmetic() -> None:
    """Each support point runs `gen3_final_speed` (stage floor, then paralysis rounding half DOWN): 303 at +1 is
    floor(454.5) = 454 → modify → 113, so our 113 TIES it (½) where a continuous ×1.5×¼ scale (113.6) would not."""
    assert _p(113.0, {303: 1.0}, stage=1.0, para=1.0) == 0.5
    assert _p(200.0, {300: 1.0}) == 0.0
    assert _p(200.0, {300: 1.0}, para=1.0) == 1.0                          # paralysed: 75
    assert _p(400.0, {300: 1.0}, stage=2.0) == 0.0                         # +2: 600
    assert _p(199.0, {300: 1.0}, stage=-1.0) == 0.0 and _p(201.0, {300: 1.0}, stage=-1.0) == 1.0   # −1: 200


def test_q_values_per_row_and_saturation_is_exact() -> None:
    """Several of our speeds against one mixture at once (the op's [B,n,Q] shape), and a certain answer is EXACTLY
    0 / 1 in float32 (the cumulative mass runs in float64)."""
    final, cum = _mix({100: 0.3, 200: 0.3, 300: 0.4})
    got = p_outspeed_mixture(T([[50.0, 100.0, 150.0, 250.0, 350.0]]), final, cum)
    assert got.dtype == torch.float32
    assert got.tolist()[0] == pytest.approx([0.0, 0.15, 0.3, 0.6, 1.0])
    assert got[0, 0].item() == 0.0 and got[0, -1].item() == 1.0


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
    final, cum = _mix({300: 1.0})
    assert p_first_same_priority(T([[300.0]]), final, cum, T([1.0]), T([0.0])).item() \
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
