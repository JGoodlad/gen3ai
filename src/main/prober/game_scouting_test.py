"""`engine.scouting` — `/game`'s scouting notes from a battle capture (pure: numpy dicts, no torch).

The capture is hand-built in the exact shape `model_capture.capture` (+ `ProbeModel.capture_battle`)
returns, so each rule is pinned on its own: the card order (active first, then slot order), the move rows
sorted by presence with Hidden Power as ONE entry carrying its type split, the verdict rule (a true move is
right at presence >= 0.5; an unlisted one is MISSED only when the floor proves it is below 0.5), the 0.10
change rule against the previous decision, the spread's range / position / band, the unseen slots, and
None (never zeros) for a head the model does not build or a truth the trace does not carry."""

from __future__ import annotations

import numpy as np
import pytest

from main.prober.engine import scouting as S

K = 8


@pytest.fixture(scope="module")
def n():
    from agents.gen3_data import items, moves, natures, species
    mv, it = moves.raw(), items.raw()
    out = {m: int(mv[m]["num"]) for m in ("spikes", "roar", "protect", "toxic", "drillpeck", "whirlwind", "taunt",
                                          "hiddenpower", "softboiled", "icebeam", "seismictoss", "thunderwave",
                                          "calmmind", "aromatherapy", "thief", "fireblast", "flamethrower")}
    out.update({i: int(it[i]["num"]) for i in ("leftovers", "shellbell", "lumberry")})
    out.update({s: int(species.get(s).num) for s in ("tyranitar", "gengar", "starmie", "swampert", "metagross")})
    out.update({x: int(natures.raw()[x]["num"]) for x in ("impish", "bold", "relaxed")})
    return out


def _moves_row(n, pairs):
    """K (num, p) pairs for one slot, padded with DISTINCT zero-presence fillers (a top-k never repeats a num)."""
    fill = [m for m in ("thief", "fireblast", "flamethrower", "calmmind", "aromatherapy")
            if n[m] not in {q[0] for q in pairs}]
    pairs = list(pairs) + [(n[m], 0.0) for m in fill[:K - len(pairs)]]
    return [p[0] for p in pairs], [p[1] for p in pairs]


def _cap(n, slots, *, opp, active, revealed, floors=None, items=None, spread=None, beliefs=True, hyp=True):
    """A capture of len(opp) decisions. ``slots[i]`` = {j: [(move num, p), …]} per decision."""
    N = len(opp)
    nums = np.zeros((N, 6, K), np.int32)
    ps = np.zeros((N, 6, K), np.float32)
    for i, by_slot in enumerate(slots):
        for j, pairs in by_slot.items():
            a, b = _moves_row(n, pairs)
            nums[i, j], ps[i, j] = a, b
    cap = {"teams": [{"our": ["zapdos"] * 6, "opp": list(o)} for o in opp],
           "opp_active": np.asarray(active, np.int8), "opp_revealed_moves": revealed}
    if beliefs:
        hp_t = np.zeros((N, 6, 3), np.int8)
        hp_t[..., :] = [10, 9, 14]                                   # ice, ground, steel
        hp_p = np.tile(np.asarray([0.6, 0.3, 0.1], np.float32), (N, 6, 1))
        inums = np.zeros((N, 6, 3), np.int32)
        inums[...] = [n["leftovers"], n["shellbell"], n["lumberry"]]
        ip = np.tile(np.asarray([0.71, 0.09, 0.06], np.float32), (N, 6, 1)) if items is None else items
        sp = np.tile(np.asarray([197.0, 319.0, 105.0, 254.0, 181.0], np.float32), (N, 6, 1)) if spread is None else spread
        cap.update({"belief_move_nums": nums, "belief_move_p": ps,
                    "belief_move_floor": (np.full((N, 6), 0.04, np.float32) if floors is None else floors),
                    "belief_move_hp_type": hp_t, "belief_move_hp_type_p": hp_p,
                    "belief_item_nums": inums, "belief_item_p": ip, "belief_item_floor": np.full((N, 6), 0.02, np.float32),
                    "belief_spread": sp,
                    "belief_nature_nums": np.tile(np.asarray([n["impish"], n["bold"], n["relaxed"]], np.int8), (N, 6, 1)),
                    "belief_nature_p": np.tile(np.asarray([0.41, 0.3, 0.1], np.float32), (N, 6, 1)),
                    "belief_hp_type": hp_t, "belief_hp_type_p": hp_p,
                    "hp_type_names": ["bug", "dark", "dragon", "electric", "fighting", "fire", "flying", "ghost", "grass",
                                      "ground", "ice", "poison", "psychic", "rock", "steel", "water"]})
    if hyp:
        hidden = np.asarray([[not s for s in o] for o in opp])
        cap.update({"slot_is_hyp": hidden,
                    "slot_species": np.tile(np.asarray([0, n["tyranitar"], n["gengar"], n["starmie"], n["swampert"],
                                                        n["metagross"]]),
                                            (N, 1)),
                    "slot_pi": np.tile(np.asarray([1.0, 0.49, 0.37, 0.27, 0.38, 0.2], np.float32), (N, 1)),
                    "tail_idx": np.zeros((N, 1), np.int64), "tail_p": np.zeros((N, 1), np.float32),
                    "other_mass": np.full(N, 0.3, np.float32), "other_live": np.ones(N, bool),
                    "other_any": np.full(N, 0.97, np.float32)})
    return cap


TRUTH = [{"species": "skarmory", "moves": ["spikes", "protect", "toxic", "whirlwind"], "item": "leftovers",
          "ability": "keeneye", "nature": "impish", "evs": {"hp": 252, "def": 252, "spe": 4},
          "ivs": {s: 31 for s in ("hp", "atk", "def", "spa", "spd", "spe")}},
         {"species": "blissey", "moves": ["softboiled", "icebeam", "hiddenpowerice", "toxic"], "item": "leftovers",
          "ability": "naturalcure", "nature": "bold", "evs": {"hp": 252, "def": 252},
          "ivs": {s: 31 for s in ("hp", "atk", "def", "spa", "spd", "spe")}},
         {"species": "gengar", "moves": [], "item": "", "evs": {}, "ivs": {}},
         {"species": "starmie", "moves": [], "item": "", "evs": {}, "ivs": {}}]


def _skarm(n, drillpeck=0.62, roar=0.64):
    return [(n["roar"], roar), (n["spikes"], 0.9995), (n["drillpeck"], drillpeck), (n["protect"], 0.9995),
            (n["toxic"], 0.31), (n["taunt"], 0.05)]


def _two(n, **kw):
    """Decision 0: only Skarmory (slot 0) seen and active. Decision 1: Blissey (slot 2) is in and active."""
    bliss = [(n["softboiled"], 0.97), (n["icebeam"], 0.72), (n["hiddenpower"], 0.55), (n["seismictoss"], 0.5000001),
             (n["thunderwave"], 0.32)]
    return _cap(n, [{0: _skarm(n)}, {0: _skarm(n, drillpeck=0.87, roar=0.30), 2: bliss}],
                opp=[["skarmory", "", "", "", "", ""], ["skarmory", "", "blissey", "", "", ""]], active=[0, 2],
                revealed=[[["spikes", "protect"], [], [], [], [], []],
                          [["spikes", "protect"], [], ["softboiled"], [], [], []]], **kw)


def test_the_cards_put_the_active_first_and_sort_each_moveset(n):
    cap = _two(n)
    v = S.scouting_view(cap, 1, TRUTH)
    assert [m["id"] for m in v["mons"]] == ["blissey", "skarmory"]
    assert [m["active"] for m in v["mons"]] == [True, False]
    for m in v["mons"]:
        ps = [r["p"] for r in m["moves"]]
        assert ps == sorted(ps, reverse=True) and len(ps) == K
        assert all(0.0 <= p <= 1.0 for p in ps)
    sk = v["mons"][1]
    assert [r["id"] for r in sk["moves"][:3]] == ["spikes", "protect", "drillpeck"]
    assert {r["id"]: r["seen"] for r in sk["moves"]}["spikes"] is True
    assert {r["id"]: r["seen"] for r in sk["moves"]}["roar"] is False
    hp = next(r for r in v["mons"][0]["moves"] if r["id"] == "hiddenpower")
    assert hp["name"] == "Hidden Power" and [t["type"] for t in hp["hp_type"]] == ["Ice", "Ground", "Steel"]
    assert all(r["hp_type"] is None for r in sk["moves"])
    assert [r["name"] for r in sk["item"]] == ["Leftovers", "Shell Bell", "Lum Berry"]
    assert sk["spread"]["nature"][0] == {"name": "Impish", "p": 0.41}


def test_the_verdict_counts_right_missed_and_false_moves(n):
    v = S.scouting_view(_two(n), 1, TRUTH)
    sk = next(m for m in v["mons"] if m["id"] == "skarmory")
    ver = sk["verdict"]
    # spikes, protect >= 0.5: right; toxic 0.31: missed WITH its p; whirlwind unlisted under a 0.04 floor: missed, p None
    assert (ver["moves_right"], ver["moves_total"], ver["moves_undetermined"]) == (2, 4, 0)
    assert ver["missed"] == [{"name": "Toxic", "p": 0.31}, {"name": "Whirlwind", "p": None}]
    assert [f["name"] for f in ver["false"]] == ["Drill Peck"]          # Roar fell to 0.30 at this decision
    assert ver["item_right"] is True and ver["item_p_true"] == pytest.approx(0.71)
    assert ver["spe_err"] == pytest.approx(181.0 - sk["truth"]["stats"]["spe"])
    assert sk["truth"]["move_names"] == ["Spikes", "Protect", "Toxic", "Whirlwind"]
    assert sk["truth"]["nature"] == "Impish" and sk["truth"]["item_name"] == "Leftovers"
    bl = next(m for m in v["mons"] if m["id"] == "blissey")
    # the typed true Hidden Power is the collapsed HP entry (0.55: right). Seismic Toss sits at 0.5000001 — within a
    # rounding error of the 0.5 boundary, so it is EXCLUDED from the assertion (owner's deterministic-checks rule).
    assert bl["verdict"]["moves_right"] == 3 and bl["verdict"]["moves_total"] == 4
    assert {f["name"] for f in bl["verdict"]["false"]} - {"Seismic Toss"} == set()
    assert [m["name"] for m in bl["verdict"]["missed"]] == ["Toxic"]


def test_an_unlisted_true_move_above_a_high_floor_is_undetermined_not_missed(n):
    cap = _two(n, floors=np.full((2, 6), 0.6, np.float32))
    ver = next(m for m in S.scouting_view(cap, 1, TRUTH)["mons"] if m["id"] == "skarmory")["verdict"]
    assert ver["moves_undetermined"] == 1 and [m["name"] for m in ver["missed"]] == ["Toxic"]


def test_the_change_list_keeps_moves_that_moved_ten_points_or_more(n):
    cap = _two(n)
    v0 = S.scouting_view(cap, 0, TRUTH)
    v1 = S.scouting_view(cap, 1, TRUTH, prev=v0)
    sk = next(m for m in v1["mons"] if m["id"] == "skarmory")
    # Roar 0.64 → 0.30 (|Δ| 0.34) and Drill Peck 0.62 → 0.87 (0.25), the larger first; nothing else moved
    assert sk["delta"]["moves"] == [{"name": "Roar", "from": 0.64, "to": 0.3, "unlisted": None},
                                    {"name": "Drill Peck", "from": 0.62, "to": 0.87, "unlisted": None}]
    assert sk["delta"]["item"] == []
    # Blissey was not seen at decision 0: no previous view of it, so no change list
    assert next(m for m in v1["mons"] if m["id"] == "blissey")["delta"] == {"moves": [], "item": []}
    assert all(m["delta"] == {"moves": [], "item": []} for m in v0["mons"])          # no prev at all


def test_a_change_under_ten_points_is_not_listed_and_an_entering_move_is_bounded_by_the_floor(n):
    cap = _cap(n, [{0: _skarm(n)}, {0: _skarm(n, drillpeck=0.70, roar=0.64) + [(n["whirlwind"], 0.55)]}],
               opp=[["skarmory"] + [""] * 5] * 2, active=[0, 0], revealed=[[[]] * 6] * 2)
    v1 = S.scouting_view(cap, 1, None, prev=S.scouting_view(cap, 0, None))
    d = v1["mons"][0]["delta"]["moves"]
    # Drill Peck 0.62 → 0.70 (0.08) is under the bar; Whirlwind entered the list from below the 0.04 floor
    assert d == [{"name": "Whirlwind", "from": 0.04, "to": 0.55, "unlisted": "from"}]


def test_the_spread_reads_its_place_in_the_species_range(n):
    sp = np.tile(np.asarray([197.0, 319.0, 105.0, 254.0, 181.0], np.float32), (2, 6, 1))
    sp[1, 0] = [400.0, 0.0, 150.0, 210.0, 181.0]
    cap = _two(n, spread=sp)
    s0 = S.scouting_view(cap, 0)["mons"][0]["spread"]
    lo, hi = s0["range"]["spe"]
    assert (lo, hi) == (158, 262)            # Skarmory, base Spe 70: (2·70+31+5)·0.9 → (2·70+31+63+5)·1.1
    assert s0["pos"]["spe"] == pytest.approx(round((181 - 158) / (262 - 158), 3)) and s0["band"]["spe"] == "low"
    assert s0["band"]["spd"] == "high"       # 254 in [158, 262]
    s1 = next(m for m in S.scouting_view(cap, 1)["mons"] if m["id"] == "skarmory")["spread"]
    assert s1["pos"]["atk"] == 1.0 and s1["band"]["atk"] == "high"        # clamped above the range
    assert s1["pos"]["def"] == 0.0 and s1["band"]["def"] == "low"         # clamped below
    assert s1["band"]["spa"] == "mid"                                     # 150 in [104, 196]
    assert S.band_of(0.2) == "low" and S.band_of(0.5) == "mid" and S.band_of(0.8) == "high"


def test_the_unseen_slots_list_the_hypotheses_and_the_truth(n):
    v = S.scouting_view(_two(n), 1, TRUTH)
    u = v["unseen"]
    assert u["n"] == 4 and u["other_any"] == pytest.approx(0.97)
    # the hidden slots 1, 3, 4, 5 hold Tyranitar .49, Starmie .27, Swampert .38, Metagross .20 — sorted by presence
    assert [(g["id"], g["p"]) for g in u["guesses"]] == [("tyranitar", 0.49), ("swampert", 0.38), ("starmie", 0.27),
                                                          ("metagross", 0.2)]
    assert [g["on_team"] for g in u["guesses"]] == [False, False, True, False]
    assert u["truth"] == ["Gengar", "Starmie"]


def test_no_truth_without_the_reconstruction(n):
    v = S.scouting_view(_two(n), 1, None)
    assert all(m["truth"] is None and m["verdict"] is None for m in v["mons"])
    assert v["unseen"]["truth"] is None
    assert all(g["on_team"] is None for g in v["unseen"]["guesses"])


def test_a_head_the_model_does_not_build_is_none_never_zeros(n):
    cap = _cap(n, [{0: _skarm(n)}], opp=[["skarmory"] + [""] * 5], active=[0], revealed=[[[]] * 6],
               beliefs=False, hyp=False)
    v = S.scouting_view(cap, 0, TRUTH)
    m = v["mons"][0]
    assert m["moves"] is None and m["item"] is None and m["spread"] is None and m["hp_type_head"] is None
    assert m["floor"] == {"moves": None, "item": None}
    assert m["truth"] is not None
    assert m["verdict"]["moves_right"] is None and m["verdict"]["item_right"] is None and m["verdict"]["spe_err"] is None
    assert v["unseen"]["guesses"] is None and v["unseen"]["other_any"] is None and v["unseen"]["n"] == 5
    # the nature head alone may be absent: the spread stays, its nature is None
    cap2 = _two(n)
    del cap2["belief_nature_nums"], cap2["belief_nature_p"]
    assert S.scouting_view(cap2, 0)["mons"][0]["spread"]["nature"] is None



def test_reveal_notes_quote_the_public_evidence_between_two_decisions():
    """The evidence beside a moved belief: what THEIR mon publicly did between the previous decision's turn
    and this one's — its moves, its arrival, an item / ability a line named — never ours, never another
    mon's, never chip damage from the field, oldest first and at most `REVEALS_SHOWN`."""
    from main.prober.engine.scouting import REVEALS_SHOWN, reveal_notes
    from main.prober.engine.turn_events import fold_turns
    from main.prober.web.fixture_story import STORY_LOG

    turns = fold_turns(tuple(STORY_LOG.splitlines()), trainee_side="p1")
    assert reveal_notes(turns, 1, 3, "Tyranitar") == ["it used Rock Slide"]
    assert reveal_notes(turns, 3, 4, "Celebi") == ["it came in", "its Leftovers showed"]
    assert reveal_notes(turns, 1, 3, "Skarmory") == [], "our own mon's events are not evidence about theirs"
    assert reveal_notes(turns, 4, 5, "Starmie") == ["it came in", "its Leftovers showed"]
    assert all(len(reveal_notes(turns, 0, 99, sp)) <= REVEALS_SHOWN for sp in ("Tyranitar", "Starmie"))
    assert "Sandstorm" not in " ".join(reveal_notes(turns, 0, 99, "Celebi")), "field chip is no reveal"


def test_the_readout_carries_the_evidence_beside_a_belief_that_moved(monkeypatch):
    """`battle_readout` joins `reveal_notes` onto every card whose belief moved (`delta.after`)."""
    import main.prober.session.game as sg

    calls = []

    def notes(turns, t_from, t_to, species):
        calls.append((t_from, t_to, species))
        return [f"evidence for {species}"]

    monkeypatch.setattr(sg, "reveal_notes", notes)
    rows = [{"turn": 1, "scouting": {"mons": []}},
            {"turn": 3, "scouting": {"mons": [{"species": "Celebi", "delta": {"moves": [{"name": "x"}], "item": []}},
                                              {"species": "Starmie", "delta": {"moves": [], "item": []}}]}}]
    sg._join_reveal_notes(rows, [{"turn": 1}, {"turn": 2}])
    assert calls == [(1, 3, "Celebi")], "the evidence went to the wrong card or window"
    assert rows[1]["scouting"]["mons"][0]["delta"]["after"] == ["evidence for Celebi"]
    assert "after" not in rows[1]["scouting"]["mons"][1]["delta"], "a card whose belief did not move got evidence"
