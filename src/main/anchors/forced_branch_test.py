"""The forced branch (``main.anchors --our-force``, the deviation-only live read's play path): the spec's own rules,
and G-BRANCH at the named decision on a fake front end whose battle carries a real honest input log. Each guard is
pinned on its own: the forced action is played only at the named decision and only when the branch game's honest log,
tokens, row and greedy action all equal the raw decision's; any one differing is a named failure, recorded first."""
from __future__ import annotations

import base64
import json
from types import SimpleNamespace
from typing import Any, Dict

import numpy as np
import pytest

from main.anchors import forced_branch as FB
from main.anchors.input_log import frame_sha256
from main.anchors.search_side import honest_log

TAG = "battle-gen3ou-3"
B64 = base64.b64encode(np.arange(4, dtype="<f4").tobytes()).decode()
TOKENS = {"2": "move earthquake", "3": "move protect", "6": "switch Skarmory"}
MASK = [0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0]


def _battle(extra: bool = False) -> Any:
    log = [(0, "p1", "CHOOSE p1 move earthquake"), (0, "p2", "CHOOSE p2 move surf"),
           (1, "p2", "CHOOSE p2 move toxic"), (1, "p1", "CHOOSE p1 move protect")]
    if extra:
        log.insert(2, (0, "p2", "CHOOSE p2 move rest"))
    start = {"formatid": "gen3ou", "seed": [1, 2, 3, 4], "p1": {"name": "Gen3AIAnchor1", "team": "A|x"},
             "p2": {"name": "MetaKakuna1", "team": "B|y"}}
    return SimpleNamespace(tag=TAG, index=2, start_doc=start, req_n={"p1": 2, "p2": 2}, choice_log=log, conns={})


def _front(battle: Any, conn: Any) -> Any:
    battle.conns["p1"] = conn
    return SimpleNamespace(battles={TAG: battle})


def _frame(turn: int = 12, n: int = 14, tokens: Dict[str, str] = TOKENS, b64: str = B64) -> Dict[str, Any]:
    return {"turn": turn, "n": n, "tokens": dict(tokens), "frame": {"b64": b64}}


def _spec(action: int = 3, pi: int = 2, **over: Any) -> Dict[str, Any]:
    conn = object()
    log, _at, _u = honest_log(_front(_battle(), conn).battles[TAG], "p1")
    rec = {"key": "su_live_v1|deal|t12|n14", "battle": TAG, "index": 2, "turn": 12, "n": 14, "pi_argmax": pi,
           "tokens": TOKENS, "frame_sha256": frame_sha256(B64), "log": log}
    return {**FB.spec_from_kept(rec, action, game=2), **over}


def _play(tmp_path, spec, frame, *, argmax: int = 2, battle: Any = None):
    conn = object()
    front = _front(battle or _battle(), conn)
    fb = FB.ForcedBranch(spec, tmp_path / "branch.jsonl")
    return fb, (lambda: fb.choose(front, TAG, frame, MASK, conn, argmax))


def test_the_spec_names_its_kind_by_the_action_and_refuses_what_cannot_be_played() -> None:
    assert _spec(3)["kind"] == "branch" and _spec(2)["kind"] == "null"
    with pytest.raises(FB.BranchError, match="null branch forces"):
        FB.validate(_spec(3, kind="null"))
    with pytest.raises(FB.BranchError, match="not a legal action"):
        FB.validate({**_spec(3), "action": 9})
    with pytest.raises(FB.BranchError, match="log_sha256"):
        FB.validate(_spec(3, log_sha256="zz"))
    with pytest.raises(FB.BranchError, match="not game 5"):
        FB.spec_from_kept({"index": 2, "key": "k"}, 3, game=5)
    assert FB.stamp(_spec(3)) == "+force=branch:a3@t12n14"


def test_the_forced_action_is_played_only_at_the_named_decision_after_g_branch(tmp_path) -> None:
    fb, go = _play(tmp_path, _spec(3), _frame(turn=11, n=13))
    assert go() == 2 and fb.summary()["forced"] == 0            # another decision: the greedy choice
    fb, go = _play(tmp_path, _spec(3), _frame())
    assert go() == 3
    rec = json.loads((tmp_path / "branch.jsonl").read_text().splitlines()[-1])
    assert rec["g_branch"] == "pass" and rec["forced"] == 3 and rec["mismatch"] == []
    s = fb.summary()
    assert s["forced"] == 1 and s["g_branch"] == "pass" and FB.check_half(s, _spec(3), 1) is None
    with pytest.raises(FB.BranchError, match="reached twice"):
        go()


@pytest.mark.parametrize("case,want", [("log", "log_sha256"), ("tokens", "tokens"), ("row", "frame_sha256"),
                                       ("argmax", "pi_argmax"), ("game", "game")])
def test_g_branch_refuses_a_branch_game_that_is_not_the_raw_prefix(tmp_path, case: str, want: str) -> None:
    spec = _spec(3)
    frame, argmax, battle = _frame(), 2, _battle()
    if case == "log":
        battle = _battle(extra=True)
    elif case == "tokens":
        frame = _frame(tokens={**TOKENS, "6": "switch Blissey"})
    elif case == "row":
        frame = _frame(b64=base64.b64encode(np.ones(4, dtype="<f4").tobytes()).decode())
    elif case == "argmax":
        argmax = 6
    elif case == "game":
        spec = {**spec, "game": 5}                             # the raw decision was game 2 of its series
    fb, go = _play(tmp_path, spec, frame, argmax=argmax, battle=battle)
    with pytest.raises(FB.BranchError, match="G-BRANCH"):
        go()
    rec = json.loads((tmp_path / "branch.jsonl").read_text().splitlines()[-1])
    assert rec["g_branch"] == "fail" and want in rec["mismatch"]
    assert FB.check_half(fb.summary(), spec, 1) is not None


def test_a_half_that_never_reached_the_named_decision_does_not_verify() -> None:
    spec = _spec(3)
    assert FB.check_half(None, spec, 0) is None                 # the deal's other half plays no game
    assert "forced 0 time" in FB.check_half({"key": spec["key"], "forced": 0, "g_branch": None}, spec, 1)
    assert "no forced-branch player" in FB.check_half(None, spec, 1)


@pytest.mark.parametrize("stats,verified", [(None, False), ({"forced": 0, "g_branch": None}, False),
                                            ({"forced": 1, "g_branch": "pass"}, True)])
def test_the_half_report_unverifies_a_half_that_did_not_force_its_branch(tmp_path, monkeypatch, stats,
                                                                         verified) -> None:
    """``runner._half_report``: a ``--our-force`` half that played its game must have forced the named decision once
    with G-BRANCH passing, else the unit never counts (``regime_verified_decisions = false``)."""
    from main.anchors import runner
    from main.anchors.peer_fault_test import CLEAN_LOG, FakePeer, _plan, _record
    from main.anchors.session import OurSideState

    spec = _spec(3)
    monkeypatch.setitem(runner.peers_mod.PEERS, "stub", SimpleNamespace(
        read_report=lambda _p: {"regime_verified_decisions": True, "peer_error_free": True}))
    monkeypatch.setattr(runner.peers_mod, "team_source_count", lambda _p: 1)
    plan = SimpleNamespace(opponent_kind="stub", our_side_is_bot=False, our_side_is_peer=False, our_transport="core",
                           our_temperature=None, our_search=None, our_belief_prior_mix=None, our_force_spec=spec)
    state = OurSideState()
    state.records.append(_record(1))
    state.force_stats = None if stats is None else {"key": spec["key"], **stats}
    _, report, _f = runner._half_report(plan, "peer_challenge", 1, _plan(tmp_path, CLEAN_LOG), FakePeer(0), state,
                                        None, None)
    assert report["regime_verified_decisions"] is verified
    assert ("force_mismatch" in report) is not verified
