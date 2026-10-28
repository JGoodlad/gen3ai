"""The live searched player's pure parts (S-U build B1): the HONEST input log a searched decision replays, the deal
key, the registered presets, and the decision's branching (which decisions search). No engine, no model: the search
itself is stubbed here; ``search_side_integration_test`` runs it on a live in-process battle."""
from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import numpy as np
import pytest

from main.anchors import search_presets as SP
from main.anchors import search_side as SS

TEAM1 = "Tyranitar||choiceband|sandstream|rockslide,earthquake,focuspunch,hiddenpowerbug|Adamant|252,252,,,,4|||||"
TEAM2 = "Swampert||leftovers|torrent|surf,earthquake,icebeam,protect|Relaxed|252,,252,,4,|||||"
US, THEM = SimpleNamespace(name="us"), SimpleNamespace(name="them")


def _battle(log: list[tuple[int, str, str]], req_n=None, seed=(1, 2, 3, 4)) -> SimpleNamespace:
    start = {"formatid": "gen3ou", "p1": {"name": "Gen3AIAnchor1", "team": TEAM1},
             "p2": {"name": "MetaKakuna1", "team": TEAM2}, "core_obs": {"sides": ["p1"]}}
    if seed is not None:
        start["seed"] = list(seed)
    return SimpleNamespace(tag="battle-gen3ou-1", start_doc=start, choice_log=list(log),
                           req_n=dict(req_n or {"p1": 3, "p2": 3}), index=0, conns={"p1": US, "p2": THEM})


def test_the_honest_log_orders_by_update_then_side_and_leaves_out_the_opponents_pending_choice():
    log = [(1, "p2", "CHOOSE p2 move surf"), (1, "p1", "CHOOSE p1 move earthquake"),
           (2, "p1", "CHOOSE p1 move rockslide"), (2, "p2", "CHOOSE p2 move icebeam"),
           (3, "p2", "CHOOSE p2 move protect")]                      # the opponent already chose for OUR update 3
    got, at, u = SS.honest_log(_battle(log), "p1")
    assert u == 3 and at == 4
    assert got["cmds"] == ["CHOOSE p1 move earthquake", "CHOOSE p2 move surf", "CHOOSE p1 move rockslide",
                           "CHOOSE p2 move icebeam"]
    assert got["seed"] == "1,2,3,4" and got["format_id"] == "gen3ou"
    assert got["names"] == ["Gen3AIAnchor1", "MetaKakuna1"] and got["teams"] == [TEAM1, TEAM2]


def test_our_own_rejected_choice_of_this_update_stays_and_keeps_its_order():
    # p2 (us) chose a switch the sim refused ([Unavailable choice]: trapped), then re-decides in the SAME update
    log = [(1, "p1", "CHOOSE p1 move rockslide"), (1, "p2", "CHOOSE p2 move surf"),
           (2, "p2", "CHOOSE p2 switch 2"), (2, "p1", "CHOOSE p1 move earthquake")]
    got, at, u = SS.honest_log(_battle(log, req_n={"p1": 2, "p2": 2}), "p2")
    assert u == 2 and got["cmds"] == ["CHOOSE p1 move rockslide", "CHOOSE p2 move surf", "CHOOSE p2 switch 2"]
    assert at == 3


def test_a_battle_without_a_seed_or_with_a_forfeit_is_refused():
    with pytest.raises(SS.LiveSearchError, match="seed-base"):
        SS.honest_log(_battle([], seed=None), "p1")
    with pytest.raises(SS.LiveSearchError, match="forfeit"):
        SS.honest_log(_battle([(1, "p2", "FORCELOSE p2")], req_n={"p1": 2, "p2": 2}), "p1")


def test_the_deal_key_is_the_battles_identity_and_our_side():
    a, b = _battle([]), _battle([(1, "p1", "CHOOSE p1 move earthquake")])
    assert SS.deal_key(a.start_doc, "p1") == SS.deal_key(b.start_doc, "p1")      # the commands are not the deal
    assert SS.deal_key(a.start_doc, "p1") != SS.deal_key(a.start_doc, "p2")
    c = _battle([], seed=(1, 2, 3, 5))
    assert SS.deal_key(a.start_doc, "p1") != SS.deal_key(c.start_doc, "p1")


def test_species_hit_counts_the_truths_species_a_world_holds():
    truth = TEAM1 + "]" + TEAM2
    world = TEAM1 + "]" + "Blissey||leftovers|naturalcure|softboiled,seismictoss,toxic,icebeam|Bold|252,,252,,4,|||||"
    assert SS.species_hit(world, truth) == 0.5
    assert SS.species_hit(truth, truth) == 1.0


def test_the_presets_parse_and_r0s4_is_r0s_search():
    p, t = SP.parse("r0s4@10")
    assert (p.n_worlds, p.per_action, p.depth, p.rule, p.pikl_lambda, p.reveal, p.dice, p.max_tries) == \
        (8, 8, 4, "pikl", 0.015, "off", "seed", 60)
    assert t == 10 and SP.parse("r0s4")[1] == SP.DEFAULT_FROM_TURN == 10
    assert SP.canonical("r0s4") == "r0s4@10" and SP.stamp("r0s4@12") == "+search=r0s4@12" and SP.stamp(None) == ""
    # r0s4_rolls: r0s4 with only the dice changed (the S-U gate's leak-reduced re-gate)
    r, tr = SP.parse("r0s4_rolls@10")
    assert tr == 10 and r.dice == "rolls" and SP.stamp("r0s4_rolls") == "+search=r0s4_rolls@10"
    assert {k: v for k, v in r.as_dict().items() if k not in ("name", "dice")} == \
        {k: v for k, v in p.as_dict().items() if k not in ("name", "dice")}
    for bad in ("r0s5@10", "r0s4@0", "r0s4@x", "", None, 3):
        with pytest.raises(SP.SearchSpecError):
            SP.parse(bad)


class _Policy:
    stochastic = False

    def __init__(self, a: int) -> None:
        self.a = a

    def decide(self, row, mask):
        return self.a, self.a


def _searcher(tmp_path, monkeypatch, picks):
    s = SS.LiveSearcher.__new__(SS.LiveSearcher)
    s.preset, s.from_turn = SP.parse("r0s4@10")
    s.search, s.policy, s.out = "r0s4@10", _Policy(3), tmp_path / "search_decisions.jsonl"
    s.stats = {"search": s.search, "decisions": 0, "from_turn_decisions": 0, "searched": 0, "excluded": 0,
               "deviations": 0, "search_cpu_s": 0.0, "search_wall_s": 0.0}
    calls = []

    def fake_search(log, at, side, row, mask, d, key, argmax):
        calls.append((at, side, key))
        return {"searched": True, "played": picks.pop(0), "cpu_s": 1.0, "wall_s": 1.0}

    monkeypatch.setattr(s, "_search", fake_search)
    return s, calls


def _frame(turn: int, n: int, mask) -> dict:
    return {"turn": turn, "n": n, "mask": mask, "tokens": {}, "frame": {"b64": ""}}


def test_only_decisions_from_the_first_turn_with_two_legal_actions_search(tmp_path, monkeypatch):
    s, calls = _searcher(tmp_path, monkeypatch, picks=[6])
    battle = _battle([(1, "p1", "CHOOSE p1 move earthquake"), (1, "p2", "CHOOSE p2 move surf")],
                     req_n={"p1": 2, "p2": 2})
    front = SimpleNamespace(battles={battle.tag: battle})
    two = np.array([0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0], dtype=np.int8)
    one = np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0], dtype=np.int8)
    row = np.zeros(4, dtype=np.float32)
    run = asyncio.run
    assert run(s.decide(front, battle.tag, _frame(9, 7, two), row, two, US)) == (3, 3)      # before turn 10: greedy
    assert run(s.decide(front, battle.tag, _frame(10, 8, one), row, one, US)) == (3, 3)     # one legal action: greedy
    assert not calls
    assert run(s.decide(front, battle.tag, _frame(10, 9, two), row, two, US)) == (6, 3)     # searched: the pick plays
    assert len(calls) == 1 and calls[0][0] == 2 and calls[0][1] == "p1"
    assert calls[0][2] == f"su_live_v1|{SS.deal_key(battle.start_doc, 'p1')}|t10|n9"
    recs = [json.loads(ln) for ln in s.out.read_text().splitlines()]
    assert [(r["turn"], r["searched"], r.get("why"), r["played"]) for r in recs] == \
        [(9, False, "before_turn", 3), (10, False, "one_legal", 3), (10, True, None, 6)]
    assert all(r["schema"] == SS.SCHEMA and r["search"] == "r0s4@10" for r in recs)
    assert s.stats["decisions"] == 3 and s.stats["from_turn_decisions"] == 2


def test_a_sampling_policy_is_refused():
    pol = _Policy(0)
    pol.stochastic = True
    with pytest.raises(SS.LiveSearchError, match="greedy"):
        SS.LiveSearcher(model=None, policy=pol, search="r0s4@10", out="/nonexistent/x.jsonl", forward=lambda r, m: r)


def test_a_peers_switch_is_respelled_the_way_the_core_names_it():
    teams = {"p1": TEAM1, "p2": "Mr. Mime||leftovers|soundproof|psychic|Modest|||||||]" + TEAM2 + "]"
                                + "Lax|Snorlax|leftovers|immunity|bodyslam|Adamant|||||||"}
    assert SS.core_command("CHOOSE p2 switch swampert", teams) == "CHOOSE p2 switch Swampert"
    assert SS.core_command("CHOOSE p2 switch mrmime", teams) == "CHOOSE p2 switch Mr. Mime"
    assert SS.core_command("CHOOSE p2 switch snorlax", teams) == "CHOOSE p2 switch Lax"      # by species, nicknamed
    assert SS.core_command("CHOOSE p2 switch Swampert", teams) == "CHOOSE p2 switch Swampert"
    for keep in ("CHOOSE p2 move earthquake", "CHOOSE p2 switch 3", "FORCELOSE p2"):
        assert SS.core_command(keep, teams) == keep
    with pytest.raises(SS.LiveSearchError, match="names no single mon"):
        SS.core_command("CHOOSE p2 switch blissey", teams)
    log = [(1, "p2", "CHOOSE p2 switch swampert"), (1, "p1", "CHOOSE p1 move earthquake")]
    got, _at, _u = SS.honest_log(_battle(log, req_n={"p1": 2, "p2": 2}), "p1")
    assert got["cmds"] == ["CHOOSE p1 move earthquake", "CHOOSE p2 switch Swampert"]
