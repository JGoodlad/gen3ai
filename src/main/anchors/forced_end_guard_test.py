"""The forfeit-win guard (S-U, 2026-10-27): a battle a FORCELOSE decided is no outcome, unless it is OUR side's own
turn-limit forfeit. Fast tier: ``runner.forced_end_verdict`` / ``apply_forced_end_guard`` on stub plans and rows,
and ``results.unit_counts`` on the summary fields. The live negative control (a blocked event loop dropping the
Kakuna peer) is ``peer_patience_integration_test.py`` (slow)."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from main.anchors import runner
from main.anchors.results import unit_counts
from main.anchors.server import FORCED_ENDS_NAME, arm_forced_ends
from utils.bridge.ws_frontend import FORCED_END_SCHEMA

LIMIT = 250


def _plan(tmp_path: Path, **kw):
    base = dict(server_impl="rust", started_server=True, our_username="Gen3AIAnchor", peer_username="MetaKakuna",
                forfeit_turn_limit=LIMIT, our_side_is_peer=False, out_dir=tmp_path)
    base.update(kw)
    return SimpleNamespace(**base)


def _rec(tag="battle-gen3ou-1", *, slot="p2", user="metakakuna1", cause="disconnect", turn=3):
    return {"schema": FORCED_END_SCHEMA, "tag": tag, "index": 0, "slot": slot, "user": user, "name": user,
            "cause": cause, "turn": turn, "time": 0.0}


def _row(tag="battle-gen3ou-1", result="win", cap=False):
    return SimpleNamespace(battle_tag=tag, result=result, hit_forfeit_limit=cap)


def test_a_peers_disconnect_that_booked_our_win_is_void(tmp_path):
    v = runner.forced_end_verdict(_plan(tmp_path), [_row()], [_rec()])
    assert v["guard"] == "armed"
    assert [(x["cause"], x["booked"]) for x in v["void"]] == [("disconnect", "win")]


def test_our_own_turn_limit_forfeit_is_the_one_forced_end_that_is_an_outcome(tmp_path):
    rec = _rec(slot="p1", user="gen3aianchor1", cause="forfeit", turn=LIMIT)
    v = runner.forced_end_verdict(_plan(tmp_path), [_row(result="loss", cap=True)], [rec])
    assert v["void"] == [] and v["forced_ends"] == [rec]


def test_every_other_forced_end_is_void(tmp_path):
    plan = _plan(tmp_path)
    cases = [
        _rec(cause="forfeit", user="metakakuna1"),                              # the PEER forfeits
        _rec(slot="p1", user="gen3aianchor2", cause="disconnect", turn=LIMIT),  # OUR side drops, even at the cap
        _rec(slot="p1", user="gen3aianchor1", cause="forfeit", turn=LIMIT - 1),  # our forfeit BEFORE the cap
        {**_rec(slot="p1", user="gen3aianchor1", cause="forfeit", turn=LIMIT), "schema": "other"},
    ]
    for rec in cases:
        assert runner.forced_end_verdict(plan, [_row(result="loss", cap=True)], [rec])["void"], rec
    # our cap forfeit on a row NOT stamped as the cap is not the cap
    cap = _rec(slot="p1", user="gen3aianchor1", cause="forfeit", turn=LIMIT)
    assert runner.forced_end_verdict(plan, [_row(result="loss", cap=False)], [cap])["void"]
    # a peer our-side, or no forfeit limit, forfeits nothing legitimately
    assert runner.forced_end_verdict(_plan(tmp_path, our_side_is_peer=True), [_row(cap=True)], [cap])["void"]
    assert runner.forced_end_verdict(_plan(tmp_path, forfeit_turn_limit=0), [_row(cap=True)],
                                     [{**cap, "turn": 0}])["void"]


def test_a_missing_record_is_blind_and_a_server_not_ours_is_unavailable(tmp_path):
    assert runner.forced_end_verdict(_plan(tmp_path), [_row()], None)["guard"] == "blind"
    for kw in ({"server_impl": "node"}, {"server_impl": "external"}, {"started_server": False}):
        assert runner.forced_end_verdict(_plan(tmp_path, **kw), [_row()], None)["guard"] == "unavailable"


def test_the_guard_drops_the_void_row_and_fails_the_series_by_name(tmp_path):
    plan = _plan(tmp_path)
    path = tmp_path / FORCED_ENDS_NAME
    arm_forced_ends(path)
    rows = [_row("battle-gen3ou-1", "win"), _row("battle-gen3ou-2", "loss")]
    kept, failure = runner.apply_forced_end_guard(plan, rows, report := {}, None)
    assert (kept, failure, report["forced_end_guard"], report["non_battle_endings"]) == (rows, None, "armed", [])
    import json
    path.write_text(json.dumps(_rec("battle-gen3ou-1")) + "\n")
    kept, failure = runner.apply_forced_end_guard(plan, rows, report := {}, None)
    assert [r.battle_tag for r in kept] == ["battle-gen3ou-2"]
    assert failure is not None and failure.cause == "non_battle_termination"
    assert report["non_battle_endings"][0]["booked"] == "win"


def test_a_missing_record_fails_the_series_blind(tmp_path):
    rows = [_row()]
    kept, failure = runner.apply_forced_end_guard(_plan(tmp_path), rows, report := {}, None)
    assert kept == rows and failure is not None and failure.cause == "forced_end_guard_blind"
    assert report["forced_end_guard"] == "blind"


def _summary(**kw):
    s = {"status": "OK", "n": 1, "regime_verified_decisions": True, "peer_clean": True,
         "team_source_asymmetry": False, "team_content_asymmetry": False}
    s.update(kw)
    return s


def test_unit_counts_refuses_a_non_battle_ending_and_an_unarmed_guard():
    assert unit_counts(_summary(), games=1).counts                      # an older summary: judged as before
    assert unit_counts(_summary(forced_end_guard="armed", non_battle_endings=[]), games=1).counts
    nbe = unit_counts(_summary(forced_end_guard="armed", non_battle_endings=[_rec()]), games=1)
    assert not nbe.counts and any("non_battle_endings" in r for r in nbe.reasons)
    for guard in ("blind", "unavailable", None):
        assert not unit_counts(_summary(forced_end_guard=guard, non_battle_endings=[]), games=1).counts
