"""`hand_eval`: Foul Play's evaluator on the observation row.

Two kinds of test:

- UNIT (hand-built rows through the resolved layout, no Node, no Rust): every term of poke-engine's
  ``gen3/evaluate.rs`` at its verbatim constant, the floor, burn's physical-move scaling, Guts / Marvel Scale, the
  named deviations, and the side-swap symmetry.
- REAL ROWS (``sim`` + ``integration``: the Rust env core and the team validator): seeded fixture battles
  (``fixture_battles.play_rows``, fixed teams, one env), each p1 decision row scored, and the per-term facts
  checked against a tracker that reads ONLY the battle's protocol lines (HP, status, boosts, volatiles, Spikes,
  screens with their turns left, the opponent's item state, the unseen count, our active's PP). Its TEETH: every
  layout field the check covers, shifted by one cell, makes the check fail.
"""
from __future__ import annotations

import dataclasses
import json
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pytest

from main.hidden_info_study import hand_eval as H


@pytest.fixture(scope="module")
def L() -> H.HandEvalLayout:
    return H.resolve_layout()


# ------------------------------------------------------------------------------------------------
# hand-built rows
# ------------------------------------------------------------------------------------------------

def _row(L: H.HandEvalLayout) -> np.ndarray:
    return np.zeros(L.obs_dim, dtype=np.float32)


def _put_mon(r: np.ndarray, L: H.HandEvalLayout, side: int, slot: int, *, species: int = 1, hp: float = 1.0,
             status: Optional[str] = None, item: str = "held", moves: Tuple[Tuple[int, int, int, int], ...] = (),
             ability: int = 0, types: Tuple[int, int] = (1, 0), active: bool = False,
             base: Tuple[int, ...] = (100, 100, 100, 100, 100, 100)) -> None:
    """``moves``: (num, pp, category, type id). ``item``: held / none / unknown / consumed."""
    b = L.team_start[side] + slot * L.slot_dim
    r[b:b + L.slot_dim] = 0.0
    r[b + L.species_id] = species
    for j, v in enumerate(base):
        r[b + L.base_stats + j] = v / H.BASE_STAT_SCALE
    r[b + L.hp] = hp
    if status is not None:
        r[b + L.condition + L.status_index[status]] = 1.0
    if item in ("held", "consumed"):
        r[b + L.item_id] = 5
        r[b + L.item_known] = 1.0
    if item == "consumed":
        r[b + L.item_consumed] = 1.0
    for k, (num, pp, cat, typ) in enumerate(moves):
        m0 = b + L.moves[k]
        r[m0 + L.mv_id] = num
        r[m0 + L.mv_known] = 1.0
        r[m0 + L.mv_current_pp] = pp / L.max_pp
        r[m0 + L.mv_category] = cat
        r[m0 + L.mv_type] = typ
    r[b + L.ability_id1] = ability
    r[b + L.type1], r[b + L.type2] = types
    if active:
        r[b + L.active_flag] = 1.0


def _ctx(r: np.ndarray, L: H.HandEvalLayout, side: int) -> int:
    return L.context_start[side]


def _set_stage(r: np.ndarray, L: H.HandEvalLayout, side: int, stat: str, stage: int) -> None:
    c = _ctx(r, L, side)
    r[c + L.boost_pos[stat]] = max(stage, 0) / L.boost_scale
    r[c + L.boost_neg[stat]] = max(-stage, 0) / L.boost_scale


def _set_vol(r: np.ndarray, L: H.HandEvalLayout, side: int, name: str) -> None:
    r[_ctx(r, L, side) + L.vol_index[name]] = 1.0


def _set_spikes(r: np.ndarray, L: H.HandEvalLayout, side: int, layers: int) -> None:
    r[L.spikes[side]] = layers / L.max_spikes


def _set_screen(r: np.ndarray, L: H.HandEvalLayout, side: int, cond: str, turns: int) -> None:
    s = H.SIDES[side]
    r[L.screen_turns[(s, cond)]] = turns / L.screen_turn_scale
    r[L.screen_present[(s, cond)]] = 1.0 if turns > 0 else 0.0


def _full_teams(L: H.HandEvalLayout) -> np.ndarray:
    """Six seen, healthy, item-holding mons a side, slot 0 active, no moves: the evaluation is exactly 0."""
    r = _row(L)
    for s in (0, 1):
        for i in range(L.team_size):
            _put_mon(r, L, s, i, species=10 + i + 20 * s, active=(i == 0))
    return r


def _mon(t: Dict[str, Any], side: str, slot: int) -> Dict[str, Any]:
    return next(m for m in t[side]["mons"] if m["slot"] == slot)


PHYS, SPEC, STATUS_CAT = 1, 2, 0


# ------------------------------------------------------------------------------------------------
# UNIT: each term
# ------------------------------------------------------------------------------------------------

def test_layout_resolves_inside_the_blocks(L):
    assert L.team_start[1] - L.team_start[0] == L.team_size * L.slot_dim
    assert L.active_flag == L.slot_dim - 1, "the active flag is the slot's LAST dim"
    assert L.context_start[0] < L.context_start[1] < L.spikes[0] < L.spikes[1] < L.obs_dim
    assert set(L.boost_pos) == set(H.BOOST_WEIGHT) and L.boost_scale == 6.0
    assert all(0 <= v < L.obs_dim for v in L.screen_turns.values())
    assert L.flying_type in L.physical_type_ids, "Flying is a physical type in gen 3"


def test_a_healthy_mon_scores_140_and_balanced_teams_score_zero(L):
    r = _full_teams(L)
    t = H.hand_eval_terms(r, L)
    m = _mon(t, "ours", 3)
    assert (m["hp"], m["item_term"], m["alive_term"], m["score"]) == (100.0, 10.0, 30.0, 140.0)
    assert t["ours"]["total"] == t["theirs"]["total"] == 6 * 140.0
    assert H.hand_eval(r, L) == 0.0


@pytest.mark.parametrize("status,term", [("frz", -40.0), ("slp", -25.0), ("par", -25.0), ("tox", -30.0),
                                         ("psn", -10.0)])
def test_flat_status_terms(L, status, term):
    r = _full_teams(L)
    _put_mon(r, L, 0, 2, species=12, status=status)
    t = H.hand_eval_terms(r, L)
    assert _mon(t, "ours", 2)["status_term"] == term
    assert H.hand_eval(r, L) == pytest.approx(term)


def test_burn_scales_with_physical_moves_and_halves_for_special_attackers(L):
    phys2 = ((89, 16, PHYS, 9), (157, 16, PHYS, 13), (85, 24, SPEC, 5), (104, 24, STATUS_CAT, 1))
    r = _full_teams(L)
    _put_mon(r, L, 0, 1, species=11, status="brn", moves=phys2, base=(100, 130, 100, 90, 100, 100))
    assert _mon(H.hand_eval_terms(r, L), "ours", 1)["status_term"] == 2 * H.POKEMON_BURNED
    # Special Attack above Attack: the multiplier is halved (base stats here: the spread is unknown, D-STATS)
    _put_mon(r, L, 0, 1, species=11, status="brn", moves=phys2, base=(100, 90, 100, 130, 100, 100))
    m = _mon(H.hand_eval_terms(r, L), "ours", 1)
    assert m["status_term"] == 1 * H.POKEMON_BURNED and m["stats"]["from_spread"] is False
    # no physical move: burn costs nothing
    _put_mon(r, L, 0, 1, species=11, status="brn", moves=((85, 24, SPEC, 5),))
    assert _mon(H.hand_eval_terms(r, L), "ours", 1)["status_term"] == 0.0


def test_burn_uses_the_level_100_stats_when_the_spread_is_known(L):
    """Base Attack 100 > base SpA 95, but a Modest 0-Atk-EV / 252-SpA-EV spread puts SpA above Attack."""
    r = _full_teams(L)
    _put_mon(r, L, 0, 1, species=11, status="brn", moves=((89, 16, PHYS, 9), (157, 16, PHYS, 13)),
             base=(100, 100, 100, 95, 100, 100))
    b = L.team_start[0] + 1 * L.slot_dim
    for j in range(6):
        r[b + L.spread_ivs + j] = 1.0
    r[b + L.spread_evs + L.iv_ev_stats.index("spa")] = 1.0
    r[b + L.spread_known] = 1.0
    for j, s in enumerate(L.nature_stats):
        r[b + L.spread_nature + j] = {"atk": 0.9, "spa": 1.1}.get(s, 1.0)
    m = _mon(H.hand_eval_terms(r, L), "ours", 1)
    assert m["stats"] == {"atk": 212.0, "spa": 317.0, "from_spread": True}   # floor(236 * 0.9), floor(289 * 1.1)
    assert m["status_term"] == 1 * H.POKEMON_BURNED


@pytest.mark.parametrize("ability", ["guts", "marvelscale"])
def test_guts_and_marvel_scale_turn_burn_and_poison_positive(L, ability):
    num = L.ability_num[ability]
    for status, term in (("brn", 50.0), ("psn", 10.0), ("tox", 10.0)):
        r = _full_teams(L)
        _put_mon(r, L, 0, 4, species=14, status=status, ability=num, moves=((89, 16, PHYS, 9),))
        assert _mon(H.hand_eval_terms(r, L), "ours", 4)["status_term"] == term, (ability, status)
    r = _full_teams(L)
    _put_mon(r, L, 0, 4, species=14, status="par", ability=num)
    assert _mon(H.hand_eval_terms(r, L), "ours", 4)["status_term"] == -25.0


@pytest.mark.parametrize("pp,penalty", [(0, -30.0), (1, -27.0), (5, -15.0), (10, 0.0), (11, 0.0), (64, 0.0)])
def test_low_pp_penalty(L, pp, penalty):
    r = _full_teams(L)
    _put_mon(r, L, 0, 5, species=15, moves=((56, pp, SPEC, 3),))
    assert _mon(H.hand_eval_terms(r, L), "ours", 5)["pp_term"] == penalty
    assert H.hand_eval(r, L) == pytest.approx(penalty)


def test_the_floor_applies_before_the_alive_bonus(L):
    r = _full_teams(L)
    _put_mon(r, L, 0, 2, species=12, hp=0.05, status="frz", item="none", moves=((56, 1, SPEC, 3), (59, 2, SPEC, 6)))
    m = _mon(H.hand_eval_terms(r, L), "ours", 2)
    assert m["subtotal"] == pytest.approx(5.0 - 40.0 - 27.0 - 24.0) and m["floored"] is True
    assert m["score"] == H.POKEMON_ALIVE
    assert H.hand_eval(r, L) == pytest.approx(30.0 - 140.0)


def test_a_fainted_mon_scores_nothing(L):
    r = _full_teams(L)
    _put_mon(r, L, 1, 3, species=33, hp=0.0, status="par")
    _set_spikes(r, L, 1, 3)
    t = H.hand_eval_terms(r, L)
    m = _mon(t, "theirs", 3)
    assert m["alive"] is False and m["hazard"] == 0.0 and "score" not in m
    assert t["theirs"]["total"] == pytest.approx(5 * (140.0 - 25.0))


@pytest.mark.parametrize("stage", list(range(-6, 7)))
def test_boost_step_table_on_each_stat(L, stage):
    want = {6: 3.3, 5: 3.15, 4: 3.0, 3: 2.5, 2: 2.0, 1: 1.0, 0: 0.0}
    mult = want[stage] if stage >= 0 else -want[-stage]
    for stat, weight in (("atk", 30.0), ("def", 15.0), ("spa", 30.0), ("spd", 15.0), ("spe", 30.0)):
        r = _full_teams(L)
        _set_stage(r, L, 0, stat, stage)
        t = H.hand_eval_terms(r, L)
        assert t["ours"]["active"]["stages"][stat] == stage
        assert H.hand_eval(r, L) == pytest.approx(mult * weight), (stat, stage)


def test_their_boosts_count_against_us(L):
    r = _full_teams(L)
    _set_stage(r, L, 1, "spe", 2)
    assert H.hand_eval(r, L) == pytest.approx(-60.0)


@pytest.mark.parametrize("vol,term", [("substitute", 75.0), ("leechseed", -30.0), ("confusion", -20.0)])
def test_active_volatiles(L, vol, term):
    r = _full_teams(L)
    _set_vol(r, L, 0, vol)
    assert H.hand_eval(r, L) == pytest.approx(term)
    r = _full_teams(L)
    _set_vol(r, L, 1, vol)
    assert H.hand_eval(r, L) == pytest.approx(-term)


def test_a_fainted_active_takes_no_active_terms(L):
    r = _full_teams(L)
    _put_mon(r, L, 0, 0, species=10, hp=0.0, active=True)
    _set_vol(r, L, 0, "substitute")
    t = H.hand_eval_terms(r, L)
    assert t["ours"]["active"] == {}
    assert H.hand_eval(r, L) == pytest.approx(-140.0)


@pytest.mark.parametrize("layers,term", [(0, 0.0), (1, -12.0), (2, -16.0), (3, -25.0)])
def test_spikes_hit_every_alive_grounded_mon(L, layers, term):
    r = _full_teams(L)
    _set_spikes(r, L, 0, layers)
    assert H.hand_eval(r, L) == pytest.approx(6 * term)


def test_flying_and_levitate_are_not_grounded(L):
    r = _full_teams(L)
    _put_mon(r, L, 0, 1, species=11, types=(16, L.flying_type))                 # Steel / Flying
    _put_mon(r, L, 0, 2, species=12, ability=L.ability_num["levitate"])
    _set_spikes(r, L, 0, 3)
    t = H.hand_eval_terms(r, L)
    assert [_mon(t, "ours", i)["hazard"] for i in range(6)] == [-25.0, 0.0, 0.0, -25.0, -25.0, -25.0]


def test_unseen_opponent_slots_score_as_fresh_grounded_mons(L):
    """D-UNSEEN: 6 − seen slots, each 100 + 10 + 30 and its side's Spikes."""
    r = _row(L)
    for i in range(6):
        _put_mon(r, L, 0, i, species=10 + i, active=(i == 0))
    _put_mon(r, L, 1, 0, species=40, active=True)
    _put_mon(r, L, 1, 1, species=41)
    _set_spikes(r, L, 1, 2)
    t = H.hand_eval_terms(r, L)
    assert t["theirs"]["unseen"] == 4 and t["theirs"]["unseen_each"] == 140.0 - 16.0
    assert t["theirs"]["total"] == pytest.approx(6 * (140.0 - 16.0))
    assert H.hand_eval(r, L) == pytest.approx(6 * 16.0)


@pytest.mark.parametrize("cond,weight", [("reflect", 20.0), ("light_screen", 20.0)])
@pytest.mark.parametrize("turns", [1, 3, 5])
def test_screens_score_twenty_per_turn_left(L, cond, weight, turns):
    r = _full_teams(L)
    _set_screen(r, L, 0, cond, turns)
    assert H.hand_eval(r, L) == pytest.approx(turns * weight)
    r = _full_teams(L)
    _set_screen(r, L, 1, cond, turns)
    assert H.hand_eval(r, L) == pytest.approx(-turns * weight)


def test_a_screen_whose_turns_and_presence_disagree_raises(L):
    r = _full_teams(L)
    _set_screen(r, L, 0, "reflect", 3)
    r[L.screen_present[("ours", "reflect")]] = 0.0
    with pytest.raises(H.HandEvalError, match="reflect"):
        H.hand_eval(r, L)


def test_item_states(L):
    """Ours exact ([0,0,0] = empty hand); theirs: revealed and UNKNOWN score +10 (D-ITEM), consumed 0."""
    r = _full_teams(L)
    _put_mon(r, L, 0, 1, species=11, item="none")
    _put_mon(r, L, 0, 2, species=12, item="consumed")
    _put_mon(r, L, 1, 1, species=31, item="unknown")
    _put_mon(r, L, 1, 2, species=32, item="consumed")
    t = H.hand_eval_terms(r, L)
    assert [_mon(t, "ours", i)["item"] for i in (1, 2)] == ["none", "consumed"]
    assert [_mon(t, "theirs", i)["item"] for i in (1, 2)] == ["unknown", "consumed"]
    assert [_mon(t, "theirs", i)["item_term"] for i in (1, 2)] == [10.0, 0.0]
    assert H.hand_eval(r, L) == pytest.approx(-20.0 - (-10.0))


def test_bare_hidden_power_of_unknown_type_is_not_physical(L):
    """D-HP-TYPE: the bare id's category channel reads Normal's (physical); its type decides instead."""
    hp_num = L.hidden_power_num
    phys_type = min(L.physical_type_ids)
    for typ, mult in ((0, 0), (phys_type, 1)):
        r = _full_teams(L)
        _put_mon(r, L, 1, 2, species=32, status="brn", moves=((hp_num, 24, PHYS, typ),))
        assert _mon(H.hand_eval_terms(r, L), "theirs", 2)["status_term"] == mult * H.POKEMON_BURNED


def test_swapping_the_sides_negates_the_score(L):
    """Where the row allows it (every mon seen, every item revealed): ours <-> theirs negates the evaluation."""
    rng = np.random.default_rng(3)
    r = _row(L)
    for s in (0, 1):
        for i in range(6):
            st = [None, "brn", "par", "slp", "frz", "psn", "tox"][int(rng.integers(0, 7))]
            mv = tuple((50 + k, int(rng.integers(0, 20)), int(rng.integers(0, 3)), 1 + k) for k in range(4))
            _put_mon(r, L, s, i, species=10 + i + 20 * s, hp=float(rng.integers(0, 101)) / 100, status=st,
                     moves=mv, item=["held", "consumed"][int(rng.integers(0, 2))], active=(i == 1),
                     ability=[0, L.ability_num["guts"]][int(rng.integers(0, 2))],
                     types=(int(rng.integers(1, 18)), int(rng.integers(0, 18))),
                     base=tuple(int(x) for x in rng.integers(40, 160, 6)))
        _put_mon(r, L, s, 1, species=11 + 20 * s, hp=0.7, active=True)
        for stat in H.BOOST_WEIGHT:
            _set_stage(r, L, s, stat, int(rng.integers(-6, 7)))
        for v in ("substitute", "leechseed", "confusion"):
            if rng.integers(0, 2):
                _set_vol(r, L, s, v)
        _set_spikes(r, L, s, int(rng.integers(0, 4)))
        _set_screen(r, L, s, "reflect", int(rng.integers(0, 6)))
        _set_screen(r, L, s, "light_screen", int(rng.integers(0, 6)))
    sw = r.copy()
    t0, t1 = L.team_start
    n = L.team_size * L.slot_dim
    sw[t0:t0 + n], sw[t1:t1 + n] = r[t1:t1 + n], r[t0:t0 + n]
    c0, c1 = L.context_start
    sw[c0:c1], sw[c1:c1 + (c1 - c0)] = r[c1:c1 + (c1 - c0)], r[c0:c1]
    for a, b in [(L.spikes[0], L.spikes[1])] + [(L.screen_turns[("ours", c)], L.screen_turns[("theirs", c)])
                                                 for c in ("reflect", "light_screen")] + \
            [(L.screen_present[("ours", c)], L.screen_present[("theirs", c)]) for c in ("reflect", "light_screen")]:
        sw[a], sw[b] = r[b], r[a]
    v = H.hand_eval(r, L)
    assert abs(v) > 5.0, "the fixture must not be trivially balanced"
    assert H.hand_eval(sw, L) == pytest.approx(-v, abs=1e-3)


def test_guards_raise_on_a_contradictory_row(L):
    r = _full_teams(L)
    with pytest.raises(H.HandEvalError, match="shape"):
        H.hand_eval(r[:-1], L)
    bad = r.copy()
    bad[_ctx(bad, L, 0) + L.boost_pos["atk"]] = 0.5 / L.boost_scale
    with pytest.raises(H.HandEvalError, match="boost"):
        H.hand_eval(bad, L)
    bad = r.copy()
    _put_mon(bad, L, 0, 3, species=13, status="par")
    bad[L.team_start[0] + 3 * L.slot_dim + L.condition + L.status_index["slp"]] = 1.0
    with pytest.raises(H.HandEvalError, match="two statuses"):
        H.hand_eval(bad, L)


def test_foul_play_leaf_scale():
    assert H.foul_play_leaf(10.0, 10.0) == 0.5
    assert H.foul_play_leaf(200.0, 0.0) == pytest.approx(1.0 / (1.0 + np.exp(-2.5)))


# ------------------------------------------------------------------------------------------------
# REAL ROWS: seeded Rust-core battles, facts tracked from their protocol lines
# ------------------------------------------------------------------------------------------------

def _to_id(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def _cond(text: str) -> Tuple[int, int, Optional[str]]:
    """``"156/343"`` / ``"45/100 tox"`` / ``"0 fnt"`` -> (hp, maxhp, status)."""
    parts = text.split()
    if parts[0] == "0":
        return 0, 1, None
    hp, mx = (int(x) for x in parts[0].split("/"))
    st = parts[1] if len(parts) > 1 and parts[1] != "fnt" else None
    return hp, mx, st


class _Side:
    def __init__(self) -> None:
        self.mons: Dict[str, Dict[str, Any]] = {}
        self.active: Optional[str] = None
        self.boosts: Dict[str, int] = {}
        self.vols: set = set()
        self.spikes = 0
        self.screens: Dict[str, int] = {}
        self.bp = False


class ProtocolFacts:
    """The facts the evaluator scores, tracked from ONE side's protocol lines alone (no reading, no row)."""

    BOOSTS = ("atk", "def", "spa", "spd", "spe", "accuracy", "evasion")

    def __init__(self, me: str) -> None:
        self.me = me
        self.sides = {"p1": _Side(), "p2": _Side()}

    def _who(self, ident: str) -> Tuple[_Side, str, str]:
        side, name = ident.split(": ", 1)
        return self.sides[side[:2]], side[:2], name

    def _mon(self, ident: str) -> Dict[str, Any]:
        s, _, name = self._who(ident)
        return s.mons[name]

    def feed(self, line: str) -> None:
        f = line.split("|")
        if len(f) < 2:
            return
        kw = f[1]
        if kw in ("switch", "drag", "replace"):
            s, _, name = self._who(f[2])
            species = f[3].split(",")[0]
            hp, mx, st = _cond(f[4])
            m = s.mons.setdefault(name, {"species": _to_id(species), "item": None})
            m.update(hp=hp, max=mx, status=st)
            s.active = name
            if not s.bp:
                s.boosts, s.vols = {}, set()
            s.bp = False
        elif kw in ("-damage", "-heal", "-sethp"):
            hp, mx, st = _cond(f[3])
            m = self._mon(f[2])
            m.update(hp=hp, max=mx, status=st)
            self._item_from(f)
        elif kw == "faint":
            s, _, name = self._who(f[2])
            s.mons[name].update(hp=0)
            if s.active == name:
                s.boosts, s.vols = {}, set()
        elif kw == "-status":
            self._mon(f[2])["status"] = f[3]
        elif kw == "-curestatus":
            self._mon(f[2])["status"] = None
        elif kw == "-cureteam":
            s, _, _ = self._who(f[2])
            for m in s.mons.values():
                m["status"] = None
        elif kw in ("-boost", "-unboost"):
            s, _, _ = self._who(f[2])
            d = int(f[4]) * (1 if kw == "-boost" else -1)
            s.boosts[f[3]] = max(-6, min(6, s.boosts.get(f[3], 0) + d))
        elif kw == "-setboost":
            s, _, _ = self._who(f[2])
            s.boosts[f[3]] = int(f[4])
        elif kw == "-clearallboost":
            for s in self.sides.values():
                s.boosts = {}
        elif kw == "-copyboost":
            a, _, _ = self._who(f[2])
            b, _, _ = self._who(f[3])
            a.boosts = dict(b.boosts)
        elif kw in ("-clearboost", "-invertboost", "-swapboost", "-clearnegativeboost"):
            raise AssertionError(f"untracked boost line {line!r}")
        elif kw in ("-start", "-end"):
            s, _, _ = self._who(f[2])
            eff = _to_id(f[3].replace("move: ", ""))
            if eff in ("substitute", "leechseed", "confusion"):
                (s.vols.add if kw == "-start" else s.vols.discard)(eff)
        elif kw == "move":
            s, _, _ = self._who(f[2])
            if f[3] == "Baton Pass":
                s.bp = True
        elif kw in ("-sidestart", "-sideend"):
            s = self.sides[f[2][:2]]
            cond = _to_id(f[3].replace("move: ", ""))
            if cond == "spikes":
                s.spikes = min(3, s.spikes + 1) if kw == "-sidestart" else 0
            elif cond in ("reflect", "lightscreen"):
                if kw == "-sidestart":
                    s.screens[cond] = 5
                else:
                    s.screens.pop(cond, None)
        elif kw == "upkeep":
            for s in self.sides.values():
                s.screens = {k: v - 1 for k, v in s.screens.items()}
        elif kw == "-item":
            self._mon(f[2])["item"] = "held"
        elif kw == "-enditem":
            self._mon(f[2])["item"] = "consumed"
        elif kw in ("-transform", "-formechange", "detailschange"):
            raise AssertionError(f"untracked line {line!r}")

    def _item_from(self, f: List[str]) -> None:
        for x in f[4:]:
            if x.startswith("[from] item:"):
                m = self._mon(f[2])
                if m["item"] is None:
                    m["item"] = "held"


def _species_num(species_id: str) -> int:
    from agents import gen3_data

    return int(gen3_data.species.species_data(species_id).num)


def check_row(row: np.ndarray, facts: ProtocolFacts, req: dict, L: H.HandEvalLayout) -> List[str]:
    """Every per-term fact of ``hand_eval_terms(row)`` the protocol determines, compared; the mismatches."""
    try:
        t = H.hand_eval_terms(row, L)
    except H.HandEvalError as exc:
        return [f"raised: {exc}"]
    bad: List[str] = []
    for side, key in (("p1", "ours"), ("p2", "theirs")):
        ps = facts.sides[side]
        got = {m["slot"]: m for m in t[key]["mons"]}
        by_num = {}
        for name, m in ps.mons.items():
            by_num[_species_num(m["species"])] = (name, m)
        slots = {}
        for slot in got:
            b = L.team_start[0 if key == "ours" else 1] + slot * L.slot_dim
            slots[int(round(float(row[b + L.species_id])))] = slot
        seen = set(by_num) if key == "theirs" else set(slots)
        if key == "theirs" and set(slots) != seen:
            bad.append(f"{key}: seen nums {sorted(slots)} != protocol {sorted(seen)}")
            continue
        if key == "theirs" and t[key]["unseen"] != 6 - len(seen):
            bad.append(f"{key}: unseen {t[key]['unseen']} != {6 - len(seen)}")
        for num, (name, m) in by_num.items():
            if num not in slots:
                bad.append(f"{key}: {name} not in the row")
                continue
            g = got[slots[num]]
            want_hp = m["hp"] / m["max"]
            if abs(g["hp_fraction"] - want_hp) > 1e-4:
                bad.append(f"{key} {name}: hp {g['hp_fraction']:.4f} != {want_hp:.4f}")
            if g["alive"] != (m["hp"] > 0):
                bad.append(f"{key} {name}: alive {g['alive']}")
            if m["hp"] > 0 and g["status"] != m["status"]:
                bad.append(f"{key} {name}: status {g['status']} != {m['status']}")
            if m["hp"] > 0 and key == "theirs":
                want_item = m["item"] or "unknown"
                if g["item"] != want_item:
                    bad.append(f"{key} {name}: item {g['item']} != {want_item}")
            if m["hp"] > 0 and key == "ours" and m["item"] == "consumed" and g["item"] != "consumed":
                bad.append(f"{key} {name}: item {g['item']} != consumed")
            if g["active"] != (ps.active == name):
                bad.append(f"{key} {name}: active {g['active']}")
        act = t[key]["active"]
        if act:
            want = {s_: ps.boosts.get(s_, 0) for s_ in H.BOOST_WEIGHT}
            if act["stages"] != want:
                bad.append(f"{key}: stages {act['stages']} != {want}")
            for v, term in (("substitute", H.SUBSTITUTE), ("leechseed", H.LEECH_SEED), ("confusion", H.CONFUSION)):
                if act[v] != (term if v in ps.vols else 0.0):
                    bad.append(f"{key}: {v} {act[v]} vs protocol {v in ps.vols}")
        if t[key]["spikes"] != ps.spikes:
            bad.append(f"{key}: spikes {t[key]['spikes']} != {ps.spikes}")
        for c, pc in (("reflect", "reflect"), ("light_screen", "lightscreen")):
            want_turns = ps.screens.get(pc, 0)
            if t[key]["screens"][c]["turns"] != want_turns:
                bad.append(f"{key}: {c} turns {t[key]['screens'][c]['turns']} != {want_turns}")
    # our active's PP from the decision's own request (the server's line, not the reading)
    mv = (req.get("active") or [{}])[0].get("moves") or []
    if mv and all("pp" in m for m in mv) and len(mv) == 4:
        want_pp = sum(3.0 * m["pp"] - 30.0 for m in mv if m["pp"] <= 10)
        act_slot = next((m for m in t["ours"]["mons"] if m["active"] and m["alive"]), None)
        if act_slot is not None and act_slot["pp_term"] != want_pp:
            bad.append(f"ours: active pp_term {act_slot['pp_term']} != {want_pp}")
    return bad


#: the fixture: 16 battles at each seed, teams drawn from the WHOLE pool (fixed), p1 a seeded uniform draw, p2 the
#: episode-seeded bot. The two seeds were chosen (a coverage search over seeds 2..29) so that together they exercise
#: EVERY term the check covers: 23 brings Reflect / Light Screen, Spikes and Leech Seed; 26 Substitute and confusion.
REAL_SEEDS = (23, 26)
REAL_BATTLES = 16


@pytest.fixture(scope="module")
def real_rows():
    """[(row, facts snapshot, request)] for every p1 decision of the fixture battles."""
    import copy

    from utils.rust_env import ffi
    from utils.rust_env import fixture_battles as FB
    from utils.rust_env.build import ensure_built

    ensure_built("selfcheck")
    lib = ffi.load(ffi.default_path("selfcheck"), nan_poison=True)
    scripts: List[str] = []
    real_finished = ffi.FfiCore.finished

    def finished(self):
        out = real_finished(self)
        scripts.extend(f["script"] for f in out)
        return out

    games: list = []
    mp = pytest.MonkeyPatch()
    mp.setattr(ffi.FfiCore, "finished", finished)
    try:
        for seed in REAL_SEEDS:
            scripts_before = len(scripts)
            got = FB.play_rows(REAL_BATTLES, seed=seed, n_teams=None, n_envs=1, lib=lib)
            assert len(got) == REAL_BATTLES and len(scripts) - scripts_before >= REAL_BATTLES
            games += list(zip(got, scripts[scripts_before:scripts_before + REAL_BATTLES]))
    finally:
        mp.undo()
    out = []
    for g, ((obs, _mask), script) in enumerate(games):
        rec = ffi.trace(lib, script, commit="test", label=f"he{g}")["records"]["p1"].splitlines()[1:]
        facts = ProtocolFacts("p1")
        k = 0
        for ln in rec:
            text = json.loads(ln)["text"]
            if text.startswith("|request|"):
                req = json.loads(text[len("|request|"):])
                if req.get("wait") or req.get("teamPreview"):
                    continue
                assert k < len(obs), f"battle {g}: more decision requests than rows"
                out.append((obs[k], copy.deepcopy(facts), req))
                k += 1
                continue
            facts.feed(text)
        assert k == len(obs), f"battle {g}: {k} decision requests, {len(obs)} rows"
    return out


def _coverage(rows) -> Dict[str, int]:
    c = {"rows": len(rows), "status": 0, "boost": 0, "spikes": 0, "screen": 0, "substitute": 0, "leechseed": 0,
         "confusion": 0, "item_known": 0, "item_consumed": 0, "pp_low": 0, "hp_partial": 0}
    for _row_, f, req in rows:
        for s in f.sides.values():
            c["status"] += any(m["status"] and m["hp"] > 0 for m in s.mons.values())
            c["boost"] += any(s.boosts.get(x, 0) for x in H.BOOST_WEIGHT)
            c["spikes"] += s.spikes > 0
            c["screen"] += bool(s.screens)
            for v in ("substitute", "leechseed", "confusion"):
                c[v] += v in s.vols
            c["hp_partial"] += any(0 < m["hp"] < m["max"] for m in s.mons.values())
        opp = f.sides["p2"].mons.values()
        c["item_known"] += any(m["item"] == "held" for m in opp)
        c["item_consumed"] += any(m["item"] == "consumed" for m in opp)
        mv = (req.get("active") or [{}])[0].get("moves") or []
        c["pp_low"] += any(m.get("pp", 99) <= 10 for m in mv)
    return c


@pytest.mark.sim
@pytest.mark.integration
def test_real_rows_match_the_protocol_facts(real_rows, L):
    cov = _coverage(real_rows)
    # the precondition: every checked term is exercised by these battles (else the teeth below are vacuous)
    assert all(v > 0 for v in cov.values()), f"the fixture battles do not exercise every term: {cov}"
    bad = []
    for i, (row, facts, req) in enumerate(real_rows):
        bad += [f"row {i}: {b}" for b in check_row(row, facts, req, L)]
    assert not bad, f"{len(bad)} mismatches, first: {bad[:8]}"


def _shift(L: H.HandEvalLayout, field: str) -> H.HandEvalLayout:
    v = getattr(L, field)
    if isinstance(v, int):
        nv: Any = v + 1
    elif isinstance(v, tuple):
        nv = tuple(x + 1 for x in v)
    else:
        nv = {k: x + 1 for k, x in v.items()}
    return dataclasses.replace(L, **{field: nv})


#: every layout field the real-row check covers: shifting any one of them by ONE cell must break the check.
TEETH_FIELDS = ("team_start", "species_id", "hp", "condition", "item_known", "item_consumed", "active_flag",
                "moves", "mv_current_pp", "mv_known", "context_start", "boost_pos", "boost_neg", "vol_index",
                "spikes", "screen_turns", "screen_present")
#: the shifts the LAYOUT itself refuses (the active flag is the slot's last dim: one cell on leaves the slot).
LAYOUT_REFUSES = frozenset({"active_flag"})


@pytest.mark.sim
@pytest.mark.integration
@pytest.mark.parametrize("field", TEETH_FIELDS)
def test_real_row_check_has_teeth_one_cell_off(real_rows, L, field):
    if field in LAYOUT_REFUSES:
        with pytest.raises(H.HandEvalError, match="outside its block"):
            _shift(L, field)
        return
    shifted = _shift(L, field)
    n = sum(1 for row, facts, req in real_rows if check_row(row, facts, req, shifted))
    assert n > 0, f"shifting {field} by one cell went unnoticed on all {len(real_rows)} rows"
