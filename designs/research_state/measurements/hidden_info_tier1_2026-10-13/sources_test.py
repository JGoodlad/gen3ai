"""Pure unit tests of the three world sources' prior tables (FN, POOLSETS, POOL): normalisation, the fallback counting,
FN taking F's seen-mon screen parameters. No engine, no model.

    PYTHONPATH=<worktree>/src python -m pytest designs/research_state/measurements/hidden_info_tier1_2026-10-13/sources_test.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import fn_worlds as FNW  # noqa: E402
import pool_worlds as PW  # noqa: E402
import poolsets_worlds as PSW  # noqa: E402
from main.value_search import world_screen as WS  # noqa: E402
from main.value_search import worlds as W  # noqa: E402


class StubBase:
    """A Smogon-shaped prior with fixed tables (the attributes the Prior subclasses copy)."""

    def __init__(self):
        self._p = self._s = None
        self.usage = {"skarmory": 0.3, "blissey": 0.2, "gengar": 0.1}
        self.pool = sorted(self.usage)
        self._cache = {}
        self.calls = []

    def moves(self, sid):
        self.calls.append(("moves", sid))
        return {"toxic": 0.9, "protect": 0.8, "spikes": 0.7, "roar": 0.6, "hiddenpowerfire": 0.2, "rest": 0.5}

    def items(self, sid):
        return {"leftovers": 1.0}

    def ability(self, sid):
        return {"keeneye": 1.0}

    def spreads(self, sid):
        return {("impish", (252, 0, 252, 0, 4, 0)): 0.6, ("careful", (252, 0, 4, 0, 252, 0)): 0.4}

    def hidden_power(self, sid):
        return {"fire": 0.7, "grass": 0.3}

    def mates(self, sid):
        return {"blissey": 0.5, "gengar": 0.1}

    def name(self, sid):
        return sid.capitalize()


# ---------------------------------------------------------------- POOLSETS
TEAM_A = ("Skarmory||leftovers|keeneye|spikes,protect,roar,toxic|Impish|252,,252,,4,||||||]"
          "Gengar||leftovers|levitate|thunderbolt,icepunch,hiddenpowerfire,explosion|Timid|,,,252,4,252||,30,,30,,30|||")
TEAM_B = ("Skarmory|||keeneye|spikes,protect,whirlwind,drillpeck|Careful|252,,4,,252,||||||]"
          "Gengar||choiceband|levitate|thunderbolt,icepunch,hiddenpowergrass,explosion|Timid|,,,252,4,252||30,,,30,,30|||")


def test_pool_tables_are_frequencies_and_normalised():
    t = PSW.pool_tables([TEAM_A, TEAM_B, TEAM_A])
    sk = t["skarmory"]
    assert t["__meta__"]["sets"]["skarmory"] == 3
    assert sk["moves"]["spikes"] == pytest.approx(1.0)
    assert sk["moves"]["roar"] == pytest.approx(2 / 3)
    assert sk["moves"]["whirlwind"] == pytest.approx(1 / 3)
    assert sum(sk["moves"].values()) == pytest.approx(4.0)                # inclusion frequencies of 4-move sets
    assert sk["items"] == {"leftovers": pytest.approx(2 / 3)}             # the item-less set is dropped, not renamed
    assert sum(sk["spreads"].values()) == pytest.approx(1.0)
    assert sk["hidden_power"] == {}
    g = t["gengar"]
    assert sum(g["hidden_power"].values()) == pytest.approx(1.0)
    assert g["hidden_power"]["fire"] == pytest.approx(2 / 3)
    assert g["moves"]["hiddenpowerfire"] == pytest.approx(2 / 3)
    assert "explosion" in g["moves"]


def test_pool_tables_drop_engine_banned_moves_and_items():
    bad = ("Skarmory||mail|keeneye|spikes,protect,roar,attract|Impish|252,,252,,4,||||||")
    t = PSW.pool_tables([bad])
    assert "attract" not in t["skarmory"]["moves"] and "mail" not in t["skarmory"]["items"]


def test_poolsets_prior_falls_back_to_smogon_and_counts_it():
    base = StubBase()
    P = PSW.PoolSetsPrior(PSW.pool_tables([TEAM_A, TEAM_B]), base)
    assert P.moves("skarmory")["spikes"] == pytest.approx(1.0)            # the pool's
    assert P.fallback[("moves", "skarmory")] == 0
    assert P.moves("blissey") == base.moves("blissey")                     # never in the pool: Smogon, counted
    assert P.fallback[("moves", "blissey")] == 1
    assert P.items("blissey") == base.items("blissey") and P.fallback[("items", "blissey")] == 1
    # a pool species whose sets never carry a Hidden Power: P(any) = 0 from its moves, the type table uniform
    assert P.source("skarmory", "hidden_power") == "pool"
    assert set(P.hidden_power("skarmory")) == set(W.HP_TYPES)
    assert not any(W.is_hp(m) for m in P.moves("skarmory"))
    # the species draw stays F's (Smogon)
    assert P.pool == base.pool and P.mates("skarmory") == base.mates("skarmory")


def test_poolsets_namespace_keeps_samplers_apart_and_shares_verdicts():
    S = PSW.PoolSetsSampler.__new__(PSW.PoolSetsSampler)
    cache = {("samplers", "d"): {"F": 1}, ("verdict", "d"): {"t##strict": (True, 1.0, None)}}
    sub = S.namespace(cache, "d")
    assert ("samplers", "d") not in sub
    assert sub[("verdict", "d")] is cache[("verdict", "d")]


# ---------------------------------------------------------------- FN
def _row(species=None, seen=None, hyp=None):
    return {"id": "d", "species": species or {}, "moves": {"seen": seen or {}, "hyp": hyp or {}}}


def test_fillable_counts_only_positive_runnable_non_hp_unrevealed_candidates():
    w = {"toxic": 0.5, "protect": 0.5, "hiddenpowerfire": 0.9, "attract": 0.9, "spikes": 0.0}
    assert FNW.fillable(w, set(), 2)
    assert not FNW.fillable(w, {"toxic"}, 2)
    assert not FNW.fillable(w, set(), 3)


def _num(sid):
    from agents.gen3_data import species

    return species.get(sid).num


def test_fn_prior_routes_seen_hyp_and_smogon_and_counts_fallbacks():
    base = StubBase()
    n_sk, n_gg, n_bl = _num("skarmory"), _num("gengar"), _num("blissey")
    seen_w = {str(n_sk): {"toxic": 0.6, "roar": 0.7, "whirlwind": 0.7, "hiddenpowerfire": 1.0}}
    hyp_w = {str(n_gg): {"thunderbolt": 0.9, "icepunch": 0.8, "explosion": 0.7, "willowisp": 0.6},
             str(n_bl): {"softboiled": 1.0}}                               # cannot fill 4 slots
    P = FNW.FNPrior(_row(seen=seen_w, hyp=hyp_w), {"skarmory": {"spikes"}}, base)
    assert P.move_src["skarmory"][0] == "net_seen"
    assert P.moves("skarmory") == dict(sorted(seen_w[str(n_sk)].items()))
    assert P.moves("gengar") == dict(sorted(hyp_w[str(n_gg)].items())) and P.move_src["gengar"][0] == "net_hyp"
    assert P.moves("blissey") == base.moves("blissey") and P.move_src["blissey"][0] == "smogon_fallback"
    assert P.moves("starmie") == base.moves("starmie") and P.move_src["starmie"][0] == "smogon"
    # every SET fact other than moves is Smogon's
    for f in ("items", "ability", "spreads", "hidden_power", "mates"):
        assert getattr(P, f)("skarmory") == getattr(base, f)("skarmory")


def test_fn_prior_seen_mon_without_presence_or_unfillable_is_smogon_counted():
    base = StubBase()
    seen_w = {str(_num("skarmory")): {"toxic": 0.6}}                       # 1 candidate for 3 free slots
    P = FNW.FNPrior(_row(seen=seen_w), {"skarmory": {"spikes"}, "blissey": set()}, base)
    assert P.move_src["skarmory"][0] == "smogon_fallback" and P.check["seen_unfillable"] == ["skarmory"]
    assert P.move_src["blissey"][0] == "smogon_fallback" and P.check["seen_no_presence"] == ["blissey"]


def test_fn_species_draw_follows_the_network_and_falls_back_to_F_weights_counted():
    from agents.info_levers.consumers import SEARCH

    base = StubBase()
    off = SEARCH.validate("off")
    P = FNW.FNPrior(_row(species={"gengar": 1.0, "skarmory": 0.5}), {"skarmory": set()}, base)
    picks = [FNW.fn_never_seen_species(np.random.default_rng(s), P, [3], {3: "blissey"}, {"skarmory"}, off)[0][3]
             for s in range(50)]
    assert set(picks) == {"gengar"}                                        # skarmory present; blissey has π = 0
    assert P.counts == {"species_draws": 50, "species_fallback": 0, "species_dropped_mass_states": 0}
    got, prov = FNW.fn_never_seen_species(np.random.default_rng(0), P, [3, 4], {3: "blissey", 4: "x"},
                                          {"skarmory"}, off)
    assert prov[3]["src"] == "net" and prov[4]["src"] == "smogon_fallback"   # gengar taken: π has no mass left
    assert P.counts["species_fallback"] == 1
    assert got[4] == "blissey"                                             # F's teammate weights (gengar present)
    # a prior that is not FN's takes F's own draw
    assert FNW.fn_never_seen_species is not FNW._ORIG_NSS


class _MS:
    def __init__(self, sid, base, hp_any):
        self.sid = sid
        self.items, self.abilities, self.hp_prior = base.items(sid), base.ability(sid), base.hidden_power(sid)
        self.spreads = WS.SpreadPrior(base.spreads(sid))
        self.hp_any = hp_any
        self.samples, self.steps, self.proposals = [], 0, 0


def test_fn_screen_puts_a_sampler_fn_builds_on_F_parameters():
    base = StubBase()
    own = {"kF": "F's sampler"}
    log = []
    g = FNW._FScreen(own, base, log)
    assert g["kF"] == "F's sampler"                                        # F's own samplers are read as they are
    m = _MS("skarmory", base, hp_any=0.93)                                 # the network's P(Hidden Power)
    g["kN"] = m
    assert m.hp_any == pytest.approx(FNW.smogon_hp_any(base, "skarmory")) == pytest.approx(0.2)
    assert log and log[0]["hp_any_network"] == 0.93


def test_fn_screen_refuses_a_sampler_whose_construction_differs_from_F():
    base = StubBase()
    g = FNW._FScreen({}, base, [])
    m = _MS("skarmory", base, hp_any=0.2)
    m.hp_prior = {"ice": 1.0}                                               # a network-derived Hidden Power type prior
    with pytest.raises(FNW.FNScreenMissing):
        g["k"] = m
    m2 = _MS("skarmory", base, hp_any=0.2)
    m2.samples = [("leftovers",)]
    with pytest.raises(FNW.FNScreenMissing):
        g["k2"] = m2


# ---------------------------------------------------------------- POOL
def test_pool_holds_matches_a_bare_hidden_power_to_any_typed_one():
    f = W.parse_set(TEAM_A.split("]")[1])
    assert PW.holds(f, {"thunderbolt", "hiddenpower"})
    assert PW.holds(f, {"hiddenpowerfire"})
    assert not PW.holds(f, {"hiddenpowergrass"})
    assert not PW.holds(f, {"shadowball"})


def test_pool_candidates_need_every_revealed_species_and_move():
    import pool_teams as PT

    pool = PT.Pool(teams=[TEAM_A, TEAM_B], sha256="x")
    assert PW.candidates(pool, {"skarmory": {"roar"}}) == [0]
    assert PW.candidates(pool, {"skarmory": set(), "gengar": {"hiddenpower"}}) == [0, 1]
    assert PW.candidates(pool, {"blissey": set()}) == []


def test_team_key_ignores_name_gender_and_order():
    import pool_teams as PT

    a, b = TEAM_A.split("]")
    b2 = b.replace("Gengar||", "Ghosty|Gengar|").replace("252||,30", "252|M|,30")
    assert W.parse_set(b2)[7] == "M" and W.parse_set(b2)[0] == "Ghosty"
    assert PT.team_key(a + "]" + b) == PT.team_key(b2 + "]" + a)
    assert PT.team_key(TEAM_A) != PT.team_key(TEAM_B)
