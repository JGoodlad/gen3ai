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


#: the four earlier presets' registered dicts, byte for byte (the envelope fields left out at their defaults): a
#: change here moves every banked record's ``config`` and every searched read's ``expected_preset``
EARLIER = {
    "r0s4": {"name": "r0s4", "n_worlds": 8, "per_action": 8, "depth": 4, "rule": "pikl", "pikl_lambda": 0.015,
             "reveal": "off", "dice": "seed", "max_tries": 60},
    "r0s4_rolls": {"name": "r0s4_rolls", "n_worlds": 8, "per_action": 8, "depth": 4, "rule": "pikl",
                   "pikl_lambda": 0.015, "reveal": "off", "dice": "rolls", "max_tries": 60},
    "r0s4_honest": {"name": "r0s4_honest", "n_worlds": 8, "per_action": 8, "depth": 4, "rule": "pikl",
                    "pikl_lambda": 0.015, "reveal": "off", "dice": "fresh", "max_tries": 60},
    "r0s4_honest_inc": {"name": "r0s4_honest_inc", "n_worlds": 8, "per_action": 8, "depth": 4, "rule": "pikl",
                        "pikl_lambda": 0.015, "reveal": "off", "dice": "fresh_inc", "max_tries": 60}}


def test_the_earlier_presets_dicts_are_byte_identical_and_the_envelope_presets_change_one_lever():
    for name, want in EARLIER.items():
        p = SP.PRESETS[name]
        assert json.dumps(p.as_dict(), sort_keys=True) == json.dumps(want, sort_keys=True)
        assert p.budget is None and p.net_reserve_s is None
    ref = SP.PRESETS["r0s4_honest_inc"].as_dict()
    k4, cap = SP.PRESETS["r0s4_honest_inc_k4"].as_dict(), SP.PRESETS["r0s4_honest_inc_cap"].as_dict()
    diff = lambda d: {k: v for k, v in d.items() if ref.get(k) != v and k != "name"}       # noqa: E731
    assert diff(k4) == {"n_worlds": 4, "per_action": 4, "net_reserve_s": 10.0}
    assert diff(cap) == {"budget": "cap", "net_reserve_s": 10.0}
    assert SP.canonical("r0s4_honest_inc_k4") == "r0s4_honest_inc_k4@10"


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


def test_a_searched_or_excluded_record_carries_the_decisions_identity_the_kept_raw_decision_carries(tmp_path,
                                                                                                    monkeypatch):
    """B2: ``log_sha256`` / ``frame_sha256`` on every record past the early return, computed by the SAME cut and the
    SAME digests a kept raw decision of the same state yields (``input_log``'s record; ``forced_branch.log_sha256``),
    so the live identity guard matches a live decision to the banked offline row of the same state."""
    from main.anchors.forced_branch import log_sha256
    from main.anchors.input_log import InputLogRecorder, frame_sha256

    s, _calls = _searcher(tmp_path, monkeypatch, picks=[6])
    outs = [{"searched": True, "played": 6, "cpu_s": 1.0, "wall_s": 1.0},
            {"searched": False, "why": "excluded", "played": 3, "cpu_s": 1.0, "wall_s": 1.0}]
    monkeypatch.setattr(s, "_search", lambda *a: dict(outs.pop(0)))
    battle = _battle([(1, "p1", "CHOOSE p1 move earthquake"), (1, "p2", "CHOOSE p2 switch swampert"),
                      (2, "p2", "CHOOSE p2 move surf")], req_n={"p1": 2, "p2": 2})   # the peer's pending choice
    front = SimpleNamespace(battles={battle.tag: battle})
    two = np.array([0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0], dtype=np.int8)
    row = np.zeros(4, dtype=np.float32)
    frames = [dict(_frame(10, 9, two), frame={"b64": "QUJDRA=="}, tokens={"3": "move a", "6": "move b"}),
              dict(_frame(11, 10, two), frame={"b64": "RUZHSA=="}, tokens={"3": "move a", "6": "move b"})]
    kept = InputLogRecorder("10-30", tmp_path / "input_logs.jsonl")
    for d in frames:
        asyncio.run(s.decide(front, battle.tag, d, row, two, US))
        kept.record(front, battle.tag, d, two, US, 3)
    recs = [json.loads(ln) for ln in s.out.read_text().splitlines()]
    raws = [json.loads(ln) for ln in (tmp_path / "input_logs.jsonl").read_text().splitlines()]
    assert [r.get("why") for r in recs] == [None, "excluded"]
    want_log = log_sha256(SS.honest_log(battle, "p1")[0])
    for r, k, d in zip(recs, raws, frames):
        assert r["key"] == k["key"]
        assert r["frame_sha256"] == k["frame_sha256"] == frame_sha256(d["frame"]["b64"])
        assert r["log_sha256"] == log_sha256(k["log"]) == want_log
        assert "CHOOSE p2 move surf" not in k["log"]["cmds"]          # the cut both share leaves it out
    assert recs[0]["frame_sha256"] != recs[1]["frame_sha256"]
    # a decision that never reaches the search carries no identity (no log is cut there)
    asyncio.run(s.decide(front, battle.tag, _frame(9, 8, two), row, two, US))
    last = json.loads(s.out.read_text().splitlines()[-1])
    assert last["why"] == "before_turn" and "log_sha256" not in last and "frame_sha256" not in last


def test_the_hashed_rescore_sample_carries_what_an_offline_rescore_needs(tmp_path, monkeypatch):
    """1 in 16 searched-path decisions by key hash (outcome-blind): ``rescore_sample``, the honest log, the mask and
    the tokens; every other record carries none of them, and no choice changes."""
    import hashlib

    keys = [f"su_live_v1|d|t{t}|n{n}" for t in range(10, 60) for n in range(3)]
    want = [k for k in keys if int(hashlib.sha256(k.encode()).hexdigest()[:8], 16) % 16 == 0]
    assert [k for k in keys if SS.rescore_sampled(k)] == want and 0 < len(want) < len(keys) // 4
    s, _calls = _searcher(tmp_path, monkeypatch, picks=[6] * 200)
    battle = _battle([(1, "p1", "CHOOSE p1 move earthquake"), (1, "p2", "CHOOSE p2 move surf")],
                     req_n={"p1": 2, "p2": 2})
    front = SimpleNamespace(battles={battle.tag: battle})
    two = np.array([0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0], dtype=np.int8)
    row = np.zeros(4, dtype=np.float32)
    for t in range(10, 60):
        d = dict(_frame(t, t + 1, two), frame={"b64": "QUJD"}, tokens={"3": "move a", "6": "move b"})
        assert asyncio.run(s.decide(front, battle.tag, d, row, two, US)) == (6, 3)
    recs = [json.loads(ln) for ln in s.out.read_text().splitlines()]
    hit = [r for r in recs if r.get("rescore_sample")]
    assert [r["key"] for r in hit] == [r["key"] for r in recs if SS.rescore_sampled(r["key"])] and hit
    for r in hit:
        assert r["log"] == SS.honest_log(battle, "p1")[0] and r["mask"] == [int(x) for x in two]
        assert r["tokens"] == {"3": "move a", "6": "move b"} and {"at", "side", "turn", "n", "pi_argmax"} <= set(r)
    assert not any({"log", "mask", "tokens", "rescore_sample"} & set(r) for r in recs if r not in hit)


def _net_searcher(tmp_path, monkeypatch, walls):
    """A searcher of ``r0s4_honest_inc_k4@10`` (the safety net at 10 s) whose search returns pick 6, deviating, with
    the given walls in turn."""
    s, calls = _searcher(tmp_path, monkeypatch, picks=[6] * len(walls))
    s.preset, s.from_turn = SP.parse("r0s4_honest_inc_k4@10")
    s.search = "r0s4_honest_inc_k4@10"
    s.clocks = {}
    s.stats.update(net_reserve_s=10.0, net_cut=0, net_cut_path=0, net_forfeit=0)
    ws = list(walls)

    def fake_search(log, at, side, row, mask, d, key, argmax):
        calls.append(key)
        s.stats["deviations"] += 1
        return {"searched": True, "played": 6, "deviation": True, "cpu_s": 1.0, "wall_s": ws.pop(0)}

    monkeypatch.setattr(s, "_search", fake_search)
    return s


def test_the_safety_net_cuts_the_pick_on_the_bank_aware_deadline_and_keeps_the_searchs_view(tmp_path, monkeypatch):
    s = _net_searcher(tmp_path, monkeypatch, walls=[3.0, 300.0, 2.0])
    battle = _battle([(1, "p1", "CHOOSE p1 move earthquake"), (1, "p2", "CHOOSE p2 move surf")],
                     req_n={"p1": 2, "p2": 2})
    front = SimpleNamespace(battles={battle.tag: battle})
    two = np.array([0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0], dtype=np.int8)
    row = np.zeros(4, dtype=np.float32)
    run = asyncio.run
    assert run(s.decide(front, battle.tag, _frame(9, 0, two), row, two, US)) == (3, 3)      # off the path
    assert run(s.decide(front, battle.tag, _frame(10, 1, two), row, two, US)) == (6, 3)     # 3 s: the search plays
    assert run(s.decide(front, battle.tag, _frame(11, 2, two), row, two, US)) == (3, 3)     # 300 s: π plays
    assert run(s.decide(front, battle.tag, _frame(12, 3, two), row, two, US)) == (3, 3)     # the overrun runs on
    recs = [json.loads(ln) for ln in s.out.read_text().splitlines()]
    assert "net" not in recs[0]
    assert recs[1]["net"]["cut"] is False and recs[1].get("why") is None and recs[1]["played"] == 6
    cut = recs[2]
    assert cut["searched"] is True and cut["why"] == SS.NET_WHY == "deadline" and cut["played"] == 3
    assert cut["deviation"] is False and cut["net"]["search_played"] == 6 and cut["net"]["search_deviation"] is True
    assert cut["net"]["deadline_s"] == 139.0 and cut["net"]["need_s"] == 300.0
    assert recs[3]["why"] == "deadline" and recs[3]["net"]["need_s"] > 150.0
    # the re-score compares the search's view: the pick the search returned, no ``why``
    view = SS.search_view(cut)
    assert view["played"] == 6 and "why" not in view and view["deviation"] is True
    assert SS.search_view(recs[1]) is recs[1]
    assert (s.stats["searched"], s.stats["net_cut"], s.stats["net_cut_path"], s.stats["deviations"]) == (0, 2, 2, 1)
    # the net is the timer model's NET_POLICY replayed on the records: the same cuts
    from main.reads import timer_model as TM

    rows = TM.rows_of(recs)
    assert TM.simulate(rows, **TM.policy_kwargs(TM.NET_POLICY))["cut_searched"] == 2


def test_a_preset_without_a_net_writes_no_net_block(tmp_path, monkeypatch):
    s, _calls = _searcher(tmp_path, monkeypatch, picks=[6])
    battle = _battle([(1, "p1", "CHOOSE p1 move earthquake")], req_n={"p1": 2, "p2": 2})
    front = SimpleNamespace(battles={battle.tag: battle})
    two = np.array([0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0], dtype=np.int8)
    asyncio.run(s.decide(front, battle.tag, _frame(10, 0, two), np.zeros(4, dtype=np.float32), two, US))
    rec = json.loads(s.out.read_text().splitlines()[0])
    assert "net" not in rec and "net_cut" not in s.stats
