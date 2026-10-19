"""F2's legality rule, its pure parts (no Node, no core): the never-seen set's REDRAW until the rule accepts it (and a
set legal at its first draw consumes the rng exactly as before), the TRUE-TEAM GUARD that raises on a rule rejecting
the truth, the returned-world guard, the pad strip and the stale Hidden Power type field. The validator's own verdicts
and the builder on a real decision: ``world_legality_integration_test.py``."""
from __future__ import annotations

import numpy as np
import pytest

from agents.info_levers.consumers import SEARCH
from main.value_search import world_draws as WD
from main.value_search import world_guard as WG
from main.value_search import world_legality as WL
from main.value_search import world_screen as WS
from main.value_search import worlds as W


class _Rule:
    """A fake validator: a team is illegal iff one of its sets holds every move of a banned COMBINATION."""

    def __init__(self, *combos):
        self.combos = [set(c) for c in combos]
        self.asked = 0

    def problems(self, teams):
        self.asked += 1
        out = []
        for t in teams:
            bad = []
            for e in t.split("]"):
                mv = set(W.parse_set(e)[4].split(","))
                bad += [f"combo {sorted(c)}" for c in self.combos if c <= mv]
            out.append(tuple(bad))
        return out


class _Prior:
    """A one-species prior whose move marginals make the naive draw hold the banned pair often."""

    def name(self, sid):
        return "Tyranitar"

    def moves(self, sid):
        return {"dragondance": 0.7, "pursuit": 0.6, "rockslide": 0.9, "earthquake": 0.8, "crunch": 0.4,
                "icebeam": 0.3, "fireblast": 0.3}

    def items(self, sid):
        return {"leftovers": 0.6, "choiceband": 0.4}

    def ability(self, sid):
        return {"sandstream": 1.0}

    def spreads(self, sid):
        return {("adamant", (252, 252, 0, 0, 4, 0)): 0.5, ("jolly", (0, 252, 0, 0, 4, 252)): 0.5}


_OFF = SEARCH.validate("off")
_PAIR = ("dragondance", "pursuit")


def _moves(entry):
    return set(W.parse_set(entry)[4].split(","))


def test_the_naive_draw_holds_the_banned_pair_and_the_legal_draw_never_does():
    rule = WL.Legality(validator=_Rule(_PAIR))
    naive, legal, tally = [], [], {}
    for seed in range(200):
        naive.append(WD.new_set(np.random.default_rng(seed), _Prior(), "tyranitar", None, _OFF)[0])
        got = WD.new_set(np.random.default_rng(seed), _Prior(), "tyranitar", None, _OFF, legal=rule.set_ok,
                         tally=tally)
        assert got is not None
        legal.append(got[0])
    assert sum(set(_PAIR) <= _moves(e) for e in naive) >= 20, "the precondition: the naive draw makes the combo"
    assert not any(set(_PAIR) <= _moves(e) for e in legal)            # FAILS if the rule is not applied
    assert tally["illegal_set"] >= 20 and "illegal_exhausted" not in tally


def test_a_set_legal_at_its_first_draw_consumes_the_rng_exactly_as_without_the_rule():
    rule = WL.Legality(validator=_Rule(("nothing", "never")))
    for seed in range(50):
        a, b = np.random.default_rng(seed), np.random.default_rng(seed)
        ea = WD.new_set(a, _Prior(), "tyranitar", None, _OFF)
        eb = WD.new_set(b, _Prior(), "tyranitar", None, _OFF, legal=rule.set_ok, tally={})
        assert ea == eb and a.random() == b.random()


def test_a_rule_no_set_can_meet_fails_the_draw_named_never_loops():
    tally: dict = {}
    every = WL.Legality(validator=_Rule(("rockslide",), ("earthquake",), ("dragondance",), ("pursuit",)))
    hits = sum(WD.new_set(np.random.default_rng(s), _Prior(), "tyranitar", None, _OFF, legal=every.set_ok,
                          tally=tally) is None for s in range(5))
    assert hits >= 1 and tally["illegal_exhausted"] == hits


_TRUE = W.parse_set("Tyranitar||leftovers|sandstream|dragondance,rockslide,earthquake,pursuit|Adamant|"
                    "252,252,,,4,|||||")
_MATE = "Skarmory||leftovers|keeneye|spikes,whirlwind,drillpeck,rest|Impish|252,,252,,4,|||||"


def test_the_true_team_guard_raises_on_a_rule_that_rejects_the_truth_and_is_never_an_exclusion():
    team = "|".join(_TRUE) + "]" + _MATE
    WL.Legality(validator=_Rule(("spikes", "pursuit"))).require_true_team(team, "ok")     # the truth passes: silent
    broken = WL.Legality(validator=_Rule(_PAIR))                 # a deliberately BROKEN rule: the truth holds the pair
    with pytest.raises(WL.LegalityRuleError, match="TRUE team fails"):
        broken.require_true_team(team, "planted")
    assert not issubclass(WL.LegalityRuleError, W.WorldError), "a defect is never an exclusion"
    assert not issubclass(WL.IllegalWorldError, W.WorldError)


def test_the_returned_world_guard_raises_on_an_illegal_world():
    rule = WL.Legality(validator=_Rule(("spikes", "whirlwind")))
    WG.check_legal(["|".join(_TRUE)], "ok", legality=rule)
    with pytest.raises(WL.IllegalWorldError, match="1 of 2 returned worlds"):
        WG.check_legal(["|".join(_TRUE), _MATE], "planted", legality=rule)


def test_the_gender_pad_is_stripped_before_validation():
    assert WL.strip_pad("Zapdos||leftovers|pressure|thunderbolt|Timid||N*|||", "*") == \
        "Zapdos||leftovers|pressure|thunderbolt|Timid||N|||"
    assert WL.strip_pad("Zapdos||leftovers|pressure|thunderbolt|Timid||M|||", "*").endswith("||M|||")
    seen = []

    class _Echo:
        def problems(self, teams):
            seen.extend(teams)
            return [() for _ in teams]

    WL.Legality(validator=_Echo(), pad="*").set_ok("Zapdos||leftovers|pressure|thunderbolt|Timid||F*|||")
    assert seen == ["Zapdos||leftovers|pressure|thunderbolt|Timid||F|||"]


def test_a_stale_hidden_power_type_field_is_cleared_and_a_consistent_one_kept():
    f = W.parse_set("Salamence||choiceband|intimidate|dragonclaw,earthquake,rockslide,hiddenpowergrass|Adamant|"
                    ",248,,8,,252||30,31,31,30,31,31|||,Flying,")
    WS.clear_stale_hp_type(f)
    assert f[11] == ""                                               # Grass IVs: the truth's Flying field goes
    g = W.parse_set("Salamence||choiceband|intimidate|hiddenpowerflying,earthquake,rockslide,brickbreak|Adamant|"
                    ",252,,,4,252||30,30,30,30,30,31|||,Flying,")
    assert W.hp_type_of(W.set_ivs(g)) == "flying"
    WS.clear_stale_hp_type(g)
    assert g[11] == ",Flying,"                                       # it agrees with the IVs: kept verbatim


def test_a_seen_mon_entry_drops_the_truths_stale_type_field_unless_the_spread_is_told_true():
    true = ("Salamence||choiceband|intimidate|hiddenpowerflying,earthquake,rockslide,brickbreak|Adamant|"
            ",252,,,4,252||30,30,30,30,30,31|||,Flying,")
    f_true = W.parse_set(true)
    m = WS.MonSampler(key=WS.ScreenKey(slot=0, hp="must"), sid="salamence", f_true=f_true, items={}, abilities={},
                      spreads=WS.SpreadPrior({("adamant", (0, 252, 0, 0, 4, 252)): 1.0}), hp_prior={}, hp_any=1.0,
                      rng=np.random.default_rng(0), needed=0)
    e = m.entry(("choiceband", "intimidate", "adamant", (0, 252, 0, 0, 4, 252), "grass"), 0)
    assert "hiddenpowergrass" in e and W.parse_set(e)[11] == ""      # FAILS before: ",Flying," kept beside HP Grass
    told = WS.MonSampler(key=WS.ScreenKey(slot=0, hp="must", fixed=(("spread", "TRUE"),)), sid="salamence",
                         f_true=f_true, items={}, abilities={}, spreads=m.spreads, hp_prior={}, hp_any=1.0,
                         rng=np.random.default_rng(0), needed=0)
    assert told.entry(WS.true_state(f_true), 0) == true             # a TOLD-true spread: the true entry verbatim
