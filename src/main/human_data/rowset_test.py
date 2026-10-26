"""The rowset builder's row rules (``rowset.py``): hidden moves, the consistent set, the panel rule, the split.

Pure functions on synthetic lines; the reader path is exercised on the committed raw replay fixture
(``main/x50_r1/testdata/human_2642128456.log``) by the integration test at the bottom.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from main.human_data import rowset as R
from main.human_data import textpass as T

FIX = Path(__file__).resolve().parents[1] / "x50_r1" / "testdata" / "human_2642128456.log"


def _pts(lines):
    from main.x50_r1.points import decision_points

    return decision_points(lines, "p1")


def _battle(turn1):
    head = ["|t:|1", "|player|p1|A||1500", "|player|p2|B||1500", "|start",
            "|switch|p1a: Snorlax|Snorlax, M|100/100", "|switch|p2a: Gengar|Gengar, M|100/100", "|turn|1", "|t:|2"]
    return head + turn1 + ["|turn|2", "|t:|3", "|win|A"]


@pytest.mark.parametrize("turn1,want", [
    (["|cant|p1a: Snorlax|slp"], "cant"),
    (["|cant|p1a: Snorlax|recharge"], None),                                   # a recharge is a lock, not a choice
    (["|-activate|p1a: Snorlax|confusion", "|-damage|p1a: Snorlax|80/100|[from] confusion"], "confusion"),
    (["|move|p2a: Gengar|Thunderbolt|p1a: Snorlax", "|faint|p1a: Snorlax"], "faint"),
    (["|move|p2a: Gengar|Pursuit|p1a: Snorlax", "|faint|p1a: Snorlax"], "pursuit"),   # maybe a switch: excluded
    (["|move|p1a: Snorlax|Body Slam|p2a: Gengar"], None),                        # shown: not hidden
    (["|switch|p1a: Zapdos|Zapdos|100/100"], None),
    (["|move|p2a: Gengar|Thunderbolt|p1a: Snorlax"], None),                    # nothing proves anything
])
def test_hidden_reasons(turn1, want):
    lines = _battle(turn1)
    pts = _pts(lines)
    assert R.hidden_reasons(lines, "p1", pts)[0] == want


def test_hidden_reasons_never_on_a_forced_decision():
    lines = ["|t:|1", "|start", "|switch|p1a: Snorlax|Snorlax|100/100", "|switch|p2a: Gengar|Gengar|100/100",
             "|turn|1", "|t:|2", "|move|p2a: Gengar|Thunderbolt|p1a: Snorlax", "|faint|p1a: Snorlax", "|",
             "|t:|3", "|switch|p1a: Zapdos|Zapdos|100/100", "|cant|p1a: Zapdos|par", "|turn|2", "|t:|4", "|win|B"]
    pts = _pts(lines)
    assert [k for _, k in pts] == ["turn", "forced", "turn"]
    assert R.hidden_reasons(lines, "p1", pts)[:2] == ["faint", None]


def _req(active_moves):
    side = [{"ident": "p1: Snorlax", "details": "Snorlax, M", "active": True, "moves": active_moves},
            {"ident": "p1: Zapdos", "details": "Zapdos", "active": False, "moves": []},
            {"ident": "p1: Gengar", "details": "Gengar, M", "active": False, "moves": []}]
    return "|request|" + json.dumps({"active": [{"moves": []}], "side": {"pokemon": side}})


def test_consistent_mask_drops_drawn_moves_and_drawn_bench_mons():
    tokens = {0: "switch Zapdos", 1: "switch Gengar", 6: "move bodyslam", 7: "move curse", 8: "move hiddenpower",
              9: "move rest"}
    mask = np.zeros(11, np.int8)
    mask[[0, 1, 6, 7, 8, 9]] = 1
    reveals = {"snorlax": (["bodyslam", "hiddenpower"], None, None), "zapdos": ([], None, None)}
    c = R.consistent_mask(tokens, mask, _req(["bodyslam", "curse", "hiddenpowerghost", "rest"]), reveals)
    assert c.tolist() == [1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0]    # Gengar never seen; Curse / Rest never used
    assert R.moveset_mask(tokens, mask).tolist() == [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0]
    mask[8] = 0                                              # a closed action is never consistent
    assert R.consistent_mask(tokens, mask, _req([]), reveals)[8] == 0


def test_the_panel_rule_counts_only_mons_with_a_revealed_move():
    class _MS:
        def __init__(self, sp, moves, item="", ability=""):
            self.species, self.moves, self.item, self.ability = sp, tuple(moves), item, ability

    m = T.PanelMatcher.__new__(T.PanelMatcher)
    m.panels = ["px"]
    m._canon = {}
    m._sid = lambda s: s
    team = [_MS("snorlax", ["bodyslam", "curse"]), _MS("zapdos", ["thunderbolt"]), _MS("gengar", ["thunderbolt"]),
            _MS("skarmory", ["spikes"])]
    m.index = {ms.species: [("px", 0, ms)] for ms in team}
    side = {"snorlax": (["bodyslam"], None, None), "zapdos": (["thunderbolt"], None, None),
            "gengar": ([], None, None), "skarmory": (["struggle"], None, None)}
    assert m.side_max_fits(side) == {"px": 2}                # species-only and Struggle-only mons do not count
    side["gengar"] = (["thunderbolt"], None, None)
    assert m.side_max_fits(side)["px"] == T.NEAR             # three move-revealed fits: NEAR
    side["zapdos"] = (["thunderbolt"], "leftovers", None)
    assert m.side_max_fits(side)["px"] == 2                  # a revealed item the set lacks breaks the fit


def test_account_split_and_keys_are_stable():
    assert R.key64("a") == R.key64("a") != R.key64("b")
    keys = [R.key64(f"acct_{i:012x}") for i in range(5000)]
    share = np.mean([R.is_val_key(k) for k in keys])
    assert 0.01 < share < 0.03


def test_textpass_record_reads_the_ending_and_reveals():
    rec = T.record("\n".join(_battle(["|move|p1a: Snorlax|Body Slam|p2a: Gengar",
                                      "|-message|B forfeited."])).replace("|win|A", "|win|A"))
    assert rec["end"] == "forfeit" and rec["quit"] == "p2" and rec["win"] == "p1" and rec["turns"] == 2
    assert rec["reveals"]["p1"]["snorlax"][0] == ["bodyslam"] and rec["reveals"]["p2"]["gengar"][0] == []


@pytest.mark.integration
def test_seat_rows_on_a_real_replay(tmp_path):
    """The reader path end to end on the committed fixture: labels inside the consistent set, a hidden row's set is
    its legal moves, and the quitter rule leaves the counts it claims."""
    import collections

    from main.live.replay_scan import battle_lines
    from main.x50_r1 import extract as E

    text = FIX.read_text(encoding="utf-8", errors="replace")
    lines = battle_lines(text)
    rec = T.record(text)
    info = E.player_info(lines)
    meta = {"quit": rec["quit"], "end": rec["end"], "acct": {"p1": "acct_x", "p2": "acct_y"},
            "rating": {"p1": 1500, "p2": 1500}}
    got = 0
    for seat in ("p1", "p2"):
        c: collections.Counter = collections.Counter()
        d = R.seat_rows(lines, seat, info[seat]["name"], str(FIX), FIX.name, meta, rec["reveals"][seat], c)
        if not d:
            continue
        lab = d["act"] >= 0
        assert d["cons"][np.nonzero(lab)[0], d["act"][lab]].all()
        assert ((d["cons"] == 1) <= (d["mask"] == 1)).all()
        assert (d["moveset"][d["hidden"]].sum(1) > 0).all() and (d["act"][d["hidden"]] == -1).all()
        assert (d["mask"].sum(1) >= 2).all()
        got += len(d["act"])
    assert got == 16, "the fixture's kept rows moved: a row rule changed"
