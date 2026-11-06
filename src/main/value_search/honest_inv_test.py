"""THE LEAK TEST (INV, ``designs/endstate/design_sprime_honest.md`` §3; :mod:`main.value_search.honest_inv`).

On the committed fixture — one constructed pair (D, D′) per channel of the leak inventory (L1-L6), each D′ shipping
the searched side exactly D's information set — the live searcher's decision record under ``r0s4_honest`` must be
byte-identical on D and D′ (pointwise: root rule 8), and under ``r0s4_rolls`` it must DIFFER on every case (the test's
teeth: it fails on the leaky builder, channel by channel). The control: the honest record at another I differs.

Slow (each case runs four searches with a fixed stub network; no checkpoint).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from main.value_search import honest_inv as INV

FIXTURE = Path(__file__).with_name("honest_inv_fixtures") / "cases.json.gz"

pytestmark = [pytest.mark.slow, pytest.mark.sim]


@pytest.fixture(scope="module")
def fx() -> dict:
    return INV.load_fixture(FIXTURE)


def test_the_fixture_covers_every_channel(fx):
    assert sorted({c["channel"] for c in fx["cases"]}) == sorted(INV.CHANNELS)
    assert fx["control"] is not None
    for c in fx["cases"]:
        assert INV.same_info(INV.Decision(**c["d"]), INV.Decision(**c["d_alt"])), f"{c['name']}: D′ must ship I"


@pytest.mark.parametrize("ch", INV.CHANNELS)
def test_the_honest_record_does_not_move(fx, ch):
    for c in [c for c in fx["cases"] if c["channel"] == ch]:
        r = INV.check_case(c, "r0s4_honest@10")
        assert r["identical"], f"{c['name']}: r0s4_honest's record moved with hidden state: {r['d']} vs {r['d_alt']}"


@pytest.mark.parametrize("ch", INV.CHANNELS)
def test_the_leaky_builder_fails_it(fx, ch):
    """The TEETH: run on today's ``r0s4_rolls`` (its key pinned to I except for L5), every channel's case moves."""
    for c in [c for c in fx["cases"] if c["channel"] == ch]:
        assert not INV.check_case(c, "r0s4_rolls@10")["identical"], f"{c['name']}: r0s4_rolls did not move"


def test_the_control_moves_with_I(fx):
    c = dict(fx["control"], channel="control")
    a, b = INV.Decision(**c["d"]), INV.Decision(**c["d_alt"])
    assert not INV.same_info(a, b)
    s = INV.searcher("r0s4_honest@10")
    try:
        assert INV.record(s, a) != INV.record(s, b)
    finally:
        s.close()
