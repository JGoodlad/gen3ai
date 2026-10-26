"""S-S's panel rule (``ss_rule.py``, REVIEW E B1) on planted teams and a planted matcher: the simulated reveal obeys
its shape, the redraw keeps every near team out, and the draw is a pure function of its generators."""
from __future__ import annotations

import gzip
import json

import numpy as np
import pytest

from main.human_data import ss_rule as SR
from main.human_data import textpass as T


class _MS:
    def __init__(self, sp, moves, item="", ability=""):
        self.species, self.moves, self.item, self.ability = sp, tuple(moves), item, ability


class _Team:
    def __init__(self, mons):
        self.mons = tuple(mons)


def _matcher(panel_team):
    m = T.PanelMatcher.__new__(T.PanelMatcher)
    m.panels = ["px"]
    m._canon = {}
    m._sid = lambda s: s
    m.index = {ms.species: [("px", 0, ms)] for ms in panel_team}
    return m


PANEL = [_MS("snorlax", ["bodyslam", "curse", "rest", "sleeptalk"], "leftovers", "immunity"),
         _MS("zapdos", ["thunderbolt", "hiddenpowerice", "toxic", "rest"], "leftovers", "pressure"),
         _MS("gengar", ["thunderbolt", "icepunch", "firepunch", "explosion"], "leftovers", "levitate"),
         _MS("skarmory", ["spikes", "roar", "drillpeck", "rest"], "leftovers", "keeneye")]
NEAR_TEAM = _Team(PANEL + [_MS("tyranitar", ["rockslide", "earthquake", "dragondance", "hiddenpowerbug"],
                               "leftovers", "sandstream"),
                           _MS("swampert", ["earthquake", "surf", "icebeam", "protect"], "leftovers", "torrent")])
FAR_TEAM = _Team([_MS(f"mon{i}", ["tackle", "growl", "tailwhip", "scratch"]) for i in range(6)])
FULL = ((4, 1, 1),) * 6


def test_side_shape_counts_moves_items_abilities_and_ignores_struggle():
    mons = {"snorlax": (["bodyslam", "struggle"], "leftovers", None), "zapdos": ([], None, "pressure")}
    assert SR.side_shape(mons) == ((1, 1, 0), (0, 0, 1))


def test_a_full_reveal_is_the_whole_set_with_a_bare_hidden_power():
    rule = SR.PairRule([FULL], np.array([1.0]), [NEAR_TEAM, FAR_TEAM], matcher=_matcher(PANEL))
    side = rule.simulated_side(0, np.random.default_rng(0))
    assert side["zapdos"] == (sorted(["thunderbolt", "hiddenpower", "toxic", "rest"]), "leftovers", "pressure")
    assert len(side) == 6 and rule.near(0, np.random.default_rng(0)) and not rule.near(1, np.random.default_rng(0))


def test_a_shape_reveals_exactly_its_counts():
    shape = ((2, 0, 1), (1, 1, 0), (0, 0, 0))
    rule = SR.PairRule([shape], np.array([1.0]), [NEAR_TEAM], matcher=_matcher(PANEL))
    for s in range(50):
        side = rule.simulated_side(0, np.random.default_rng(s))
        got = sorted(((len(mv), int(it is not None), int(ab is not None)) for mv, it, ab in side.values()),
                     reverse=True)
        assert tuple(got) == shape
        for sp, (mv, _, _) in side.items():
            full = {m if not m.startswith("hiddenpower") else "hiddenpower"
                    for m in next(ms for ms in NEAR_TEAM.mons if ms.species == sp).moves}
            assert set(mv) <= full


def test_draw_pair_never_returns_a_near_team_and_is_pure():
    teams = [NEAR_TEAM, FAR_TEAM, FAR_TEAM, FAR_TEAM]
    rule = SR.PairRule([FULL], np.array([1.0]), teams, matcher=_matcher(PANEL))
    a = [rule.draw_pair(np.random.default_rng([1, 0]), np.random.default_rng([1, 2])) for _ in range(1)]
    pairs = []
    rt, rr = np.random.default_rng([7, 0]), np.random.default_rng([7, 2])
    for _ in range(200):
        pairs.append(rule.draw_pair(rt, rr))
    assert all(0 not in p and p[0] != p[1] for p in pairs) and rule.redraws > 0
    again = SR.PairRule([FULL], np.array([1.0]), teams, matcher=_matcher(PANEL))
    rt, rr = np.random.default_rng([7, 0]), np.random.default_rng([7, 2])
    assert [again.draw_pair(rt, rr) for _ in range(200)] == pairs
    assert a == [rule.draw_pair(np.random.default_rng([1, 0]), np.random.default_rng([1, 2]))]


def test_a_pool_where_every_pair_is_near_is_refused(monkeypatch):
    monkeypatch.setattr(SR, "MAX_REDRAWS", 20)
    rule = SR.PairRule([FULL], np.array([1.0]), [NEAR_TEAM, NEAR_TEAM], matcher=_matcher(PANEL))
    with pytest.raises(SR.RuleError):
        rule.draw_pair(np.random.default_rng(0), np.random.default_rng(1))


def test_the_profile_round_trips_with_its_weights(tmp_path):
    p = tmp_path / "prof.json.gz"
    body = {"schema": "gen3_x59s_reveal_profile_v1", "shapes": [[[[2, 0, 1], [0, 0, 0]], 3], [[[4, 1, 1]], 1]]}
    with gzip.open(p, "wt") as fh:
        json.dump(body, fh)
    shapes, probs, sha = SR.load_profile(p)
    assert shapes == [((2, 0, 1), (0, 0, 0)), ((4, 1, 1),)] and probs.tolist() == [0.75, 0.25] and len(sha) == 64
    bad = tmp_path / "bad.json.gz"
    with gzip.open(bad, "wt") as fh:
        json.dump({"schema": "other", "shapes": []}, fh)
    with pytest.raises(SR.RuleError):
        SR.load_profile(bad)


def test_candidate_battles_follow_the_scan_filters():
    rec = {"rated": True, "end": "ko", "win": "p1", "turns": 11, "rating": {"p1": 1300, "p2": None}}
    assert SR.is_candidate(rec, "2026-09-01/x.log")
    assert not SR.is_candidate({**rec, "turns": 10}, "2026-09-01/x.log")
    assert not SR.is_candidate(rec, "2026-09-21/x.log")
    assert not SR.is_candidate({**rec, "rating": {"p1": 1299, "p2": 1200}}, "2026-09-01/x.log")
    assert not SR.is_candidate(None, "2026-09-01/x.log")
