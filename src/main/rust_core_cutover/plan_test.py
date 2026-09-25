"""The cutover stress's PLAN: every registered battle exactly once, keys unique, the coverage the
targets claim actually produced by the unit recipes."""
import collections

import pytest

from main.rust_core_cutover import plan as PL

SIZES = dict(pool_n=719, ladder_full_n=22813, ladder_policy_n=800)


def _streams():
    return {s.name: s for s in PL.streams(**SIZES)}


def _n_teams(s):
    return {"pool": SIZES["pool_n"], "ladder": SIZES["ladder_full_n"] if s.params.get("tier") == "full"
            else SIZES["ladder_policy_n"]}.get(s.params.get("source"))


def test_every_unit_range_tiles_its_stream_exactly_once():
    for s in PL.streams(**SIZES):
        seen = [b for i in range(s.units) for b in s.unit_range(i)]
        assert seen == list(range(s.battles)), s.name
        with pytest.raises(IndexError):
            s.unit_range(s.units)


def test_parity_keys_are_unique_across_every_stream():
    """The key seeds the players' RNGs and the sim; a key reused across streams would replay one
    battle twice and count it twice."""
    keys = collections.Counter()
    for s in PL.streams(**SIZES):
        if s.kind != "parity":
            continue
        n = _n_teams(s) or 2 * s.per_unit
        for i in range(s.units):
            keys.update(b.key for b in PL.parity_battles(s, i, n))
    dup = [k for k, c in keys.items() if c > 1]
    assert not dup, dup[:10]
    assert max(keys) <= PL.MAX_KEY and min(keys) >= 0


def test_ladder_a_is_the_milestone_recipe_and_b_swaps_the_slots():
    from agents.battle.rust_core_parity import LADDER_KNOWN_DIVERGENCES

    s = _streams()
    a = [b for i in range(s["ladder_full_a"].units) for b in PL.parity_battles(s["ladder_full_a"], i, SIZES["ladder_full_n"])]
    b = [x for i in range(s["ladder_full_b"].units) for x in PL.parity_battles(s["ladder_full_b"], i, SIZES["ladder_full_n"])]
    # the MILESTONE recipe: key k plays teams 2k / 2k+1 — so the named known divergences replay
    for k in LADDER_KNOWN_DIVERGENCES:
        assert (a[k].key, a[k].t1, a[k].t2) == (k, 2 * k, 2 * k + 1)
    assert [(x.t1, x.t2) for x in b] == [(x.t2, x.t1) for x in a]
    n = SIZES["ladder_full_n"]
    for side in ("t1", "t2"):
        assert {getattr(x, side) for x in a} | {getattr(x, side) for x in b} == set(range(n))
    assert {x.t1 for x in a} | {x.t2 for x in a} == set(range(n))   # every team once per pass
    assert {x.t1 for x in b} == {x.t2 for x in a}


def test_pool_streams_play_every_team_from_both_slots():
    s = _streams()
    for name, offs in (("pool_random", PL.POOL_RANDOM_OFFSETS), ("pool_policy", PL.POOL_POLICY_OFFSETS)):
        st = s[name]
        bs = [b for i in range(st.units) for b in PL.parity_battles(st, i, SIZES["pool_n"])]
        as_p1 = collections.Counter(b.t1 for b in bs)
        as_p2 = collections.Counter(b.t2 for b in bs)
        assert set(as_p1) == set(as_p2) == set(range(SIZES["pool_n"]))
        assert set(as_p1.values()) == set(as_p2.values()) == {len(offs)}
        assert all(b.t1 != b.t2 for b in bs)


def test_procedural_units_index_their_own_team_draw():
    st = _streams()["procedural_random"]
    bs = PL.parity_battles(st, 3, 2 * st.per_unit)
    assert [(b.t1, b.t2) for b in bs] == [(2 * m, 2 * m + 1) for m in range(st.per_unit)]
    assert PL.procedural_seed(st, 3) != PL.procedural_seed(st, 4)
    assert PL.procedural_seed(st, 0) != PL.procedural_seed(_streams()["procedural_policy"], 0)


def test_the_ladder_policy_stream_plays_every_milestone_team_from_both_slots():
    st = _streams()["ladder_policy"]
    bs = [b for i in range(st.units) for b in PL.parity_battles(st, i, SIZES["ladder_policy_n"])]
    assert {b.t1 for b in bs} == {b.t2 for b in bs} == set(range(SIZES["ladder_policy_n"]))


def test_fuzz_streams_use_distinct_master_seeds_and_the_full_ladder_tier():
    seeds = collections.Counter()
    for s in PL.streams(**SIZES):
        if s.kind != "fuzz":
            continue
        seeds.update((s.name, s.params["seed_base"] + i) for i in range(s.units))
        if "ladder" in s.params["argv"]:
            assert s.params["argv"][-2:] == ["--ladder-tier", "full"], s.name
    assert max(seeds.values()) == 1


def test_unit_ids_round_trip():
    assert PL.parse_unit_id(PL.unit_id("fz_state_ladder", 7)) == ("fz_state_ladder", 7)
