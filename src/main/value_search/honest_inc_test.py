"""THE INCREMENTAL LEAK-HONEST BUILDER (``r0s4_honest_inc``; :mod:`main.value_search.honest_inc`,
``designs/endstate/design_sprime_honest.md`` §9).

On the leak test's committed fixture (``honest_inv_fixtures/cases.json.gz``: one pair (D, D′) per channel L1-L6, D′
shipping the searched side exactly D's I), in SEQUENCE mode — every decision the live searcher searches, from turn 10
up to the case's, each truth on its own searcher (its own carry cache):

1. ``r0s4_honest_inc``'s records are byte-identical on D and D′ at EVERY decision, on every channel;
2. ``r0s4_rolls``'s move on every channel (the teeth: the sequence check fails on the leaky builder);
3. DETERMINISM: a sequence re-run from scratch reproduces every record, and the case's decision run ALONE (a fresh
   searcher, so its carry is REBUILT from the earlier decisions of its I) reproduces the sequence's record;
4. the first searched decision of a battle is ``r0s4_honest``'s own (its build is the cold one);
5. the correlation bound (soft): >= FRESH_MIN fresh worlds a searched warm step unless its chain failed or waits, or
   its fresh replays failed, and the record's overage counts exactly the worlds older than ⌈J / FRESH_MIN⌉ − 1 decisions.

Slow (each runs a few dozen searches with the fixed stub network; no checkpoint).
"""
from __future__ import annotations

import math
from pathlib import Path

import pytest

from main.value_search import honest_inc as HI
from main.value_search import honest_inv as INV

FIXTURE = Path(__file__).with_name("honest_inv_fixtures") / "cases.json.gz"
INC = "r0s4_honest_inc@10"

pytestmark = [pytest.mark.slow, pytest.mark.sim]


@pytest.fixture(scope="module")
def fx() -> dict:
    return INV.load_fixture(FIXTURE)


def _case(fx: dict, ch: str) -> dict:
    return next(c for c in fx["cases"] if c["channel"] == ch)


@pytest.mark.parametrize("ch", INV.CHANNELS)
def test_the_incremental_record_does_not_move_at_any_decision(fx, ch):
    r = INV.check_sequence(_case(fx, ch), INC)
    assert r["decisions"] >= 2, f"non-vacuity: {r['decisions']} decisions"
    assert r["identical"], f"{r['name']}: r0s4_honest_inc moved with hidden state at {r['moved']}"


@pytest.mark.parametrize("ch", INV.CHANNELS)
def test_the_leaky_builder_fails_the_sequence(fx, ch):
    assert not INV.check_sequence(_case(fx, ch), "r0s4_rolls@10")["identical"], f"{ch}: r0s4_rolls did not move"


@pytest.fixture(scope="module")
def l6_run(fx):
    """L6's D (turn 29): its whole sequence on one searcher, then again on another, then its last decision alone."""
    d = INV.Decision(**_case(fx, "L6")["d"])
    runs = []
    carries = {}
    for _ in range(2):
        s = INV.searcher(INC)
        try:
            seq = INV.sequence(d, 10, s.engine)
            runs.append([INV.record(s, x) for x in seq])
            carries = dict(s.sampler.inner.store)
        finally:
            s.close()
    s = INV.searcher(INC)
    try:
        alone = INV.record(s, d)
        rebuilt = s.sampler.inner.last_rebuilt
    finally:
        s.close()
    return seq, runs, alone, rebuilt, carries


def test_a_sequence_re_run_from_scratch_reproduces_every_record(l6_run):
    seq, (a, b), _alone, _r, _c = l6_run
    assert len(a) == len(seq) > 10, "non-vacuity: a long sequence"
    assert a == b
    assert sum(1 for r in a if r.get("searched")) > 10


def test_a_decision_alone_rebuilds_its_carry_and_reproduces_its_record(l6_run):
    seq, (a, _b), alone, rebuilt, _c = l6_run
    assert rebuilt == len(seq) - 1, f"a fresh searcher rebuilds every earlier checkpoint ({rebuilt} of {len(seq) - 1})"
    assert alone == a[-1]


def test_the_carries_chain(l6_run):
    """Each warm step's carried digest is the previous step's returned one (the record's audit trail)."""
    _seq, (a, _b), _alone, _r, _c = l6_run
    incs = [(r.get("honest") or {}).get("inc") for r in a]
    assert all(incs), "every decision carries its inc block (searched or excluded)"
    assert incs[0]["carry_in"] is None and incs[0]["mode"] == "cold"
    for x, y in zip(incs, incs[1:]):
        assert y["carry_in"] == x["carry_out"] and y["step"] == x["step"] + 1


def test_the_correlation_bound(l6_run):
    """The SOFT bound (design §9, C17): every searched warm decision draws >= FRESH_MIN fresh worlds unless its warm chain
    or its fresh worlds' own replays failed, and ``overage`` counts exactly the worlds older than the bound (a decision
    with too few survivors uses every one)."""
    _seq, (a, _b), _alone, _r, _c = l6_run
    life = math.ceil(8 / HI.FRESH_MIN)
    for r in a:
        inc = r["honest"]["inc"]
        if inc.get("ages") is not None:
            assert inc.get("overage", 0) == sum(1 for g in inc["ages"] if g >= life), inc
        if inc.get("mode") == "warm" and r.get("searched"):
            short = inc["chain"] in ("failed", "deferred") or any(k in (r["honest"].get("fails") or {})
                                                     for k in ("world_replay", "illegal_set", "illegal_team",
                                                               "never_seen", "playout_refused"))
            assert inc["new"] >= HI.FRESH_MIN or short, inc


def test_the_first_decision_is_r0s4_honests(fx):
    d = INV.Decision(**_case(fx, "L1")["d"])
    si, sh = INV.searcher(INC), INV.searcher("r0s4_honest@10")
    try:
        first = INV.sequence(d, 10, si.engine)[0]
        ri, rh = INV.record(si, first), INV.record(sh, first)
    finally:
        si.close()
        sh.close()
    assert ri["honest"].pop("inc")["mode"] == "cold"
    assert ri == rh


def test_a_decision_re_scored_from_its_recorded_carry_needs_no_rebuild(l6_run):
    """The re-score aid: a fresh searcher SEEDED with the carry the decision started from (what a re-score-sampled
    record keeps) reproduces its record without rebuilding, and the carry's digest is the record's ``carry_in``."""
    seq, (a, _b), _alone, _r, carries = l6_run
    j = len(seq) - 2
    # the carry decision j started from: the run's cache holds it under decision j-1's I, found by its digest
    carry = next(c for c in carries.values() if HI.digest(c) == a[j]["honest"]["inc"]["carry_in"])
    t = INV.searcher(INC)
    try:
        t.sampler.inner.seed(HI.unpack(HI.pack(carry)))
        got = INV.record(t, seq[j])
        assert t.sampler.inner.last_rebuilt == 0
    finally:
        t.close()
    assert got == a[j]
