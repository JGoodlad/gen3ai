"""The forfeit-win guard (S-U, 2026-10-27): a battle a FORCELOSE decided is no outcome, unless it is OUR side's own
turn-limit forfeit. Fast tier: ``runner.forced_end_verdict`` / ``apply_forced_end_guard`` on stub plans and rows,
and ``results.unit_counts`` on the summary fields. The live negative control (a blocked event loop dropping the
Kakuna peer) is ``peer_patience_integration_test.py`` (slow)."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

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


# ------------------------------------------------------------------- REVIEW 2026-10-27 (anchor_guards_review)
def test_a_peers_forfeit_at_the_cap_on_a_capped_row_is_void(tmp_path):
    """Pins the ``user in ours`` clause on its own (the review's mutant M1 survived every fast test): a forfeit at the
    turn limit by the PEER, on a row our side stamped ``hit_forfeit_limit``, is the peer's, never our cap."""
    rec = _rec(slot="p2", user="metakakuna1", cause="forfeit", turn=LIMIT)
    assert runner.forced_end_verdict(_plan(tmp_path), [_row(result="win", cap=True)], [rec])["void"]


@pytest.mark.xfail(strict=True, reason=(
    "REVIEW 2026-10-27 H3 (MINOR, designs/ops/reviews/anchor_guards_review_2026-10-27.md): ws_frontend._force_lose "
    "swallows an OSError on the forced-end append, so the armed-EMPTY file reads 'armed' with no void and a peer's "
    "drop counts as our win. Flips RED when a failed append makes the guard blind (fail closed)."))
async def test_a_forced_end_the_record_could_not_store_never_reads_armed_and_clean(tmp_path, monkeypatch):
    import builtins

    from utils.bridge import ws_frontend
    from utils.bridge.ws_frontend_test import _fake_battle, _front, _named

    path = tmp_path / FORCED_ENDS_NAME
    arm_forced_ends(path)

    def full_disk(file, *a, **kw):
        if str(file) == str(path) and a and a[0] == "a":
            raise OSError(28, "No space left on device")
        return builtins.open(file, *a, **kw)

    monkeypatch.setattr(ws_frontend, "open", full_disk, raising=False)
    front = _front(forced_ends_log=str(path))
    alpha, beta = await _named(front, "Gen3AIAnchor1"), await _named(front, "MetaKakuna1")
    battle = _fake_battle(front, alpha, beta)
    front._note_side_line(battle, "p2", "|turn|3")
    await front._drop(beta)                                    # the peer drops at turn 3: a FORCELOSE, OUR "win"
    assert len(front.forced_ends) == 1                         # the front end KNOWS
    v = runner.forced_end_verdict(_plan(tmp_path), [_row(battle.tag, "win")], runner.read_forced_ends(path))
    assert v["guard"] != "armed" or v["void"], v               # ...but the guard reads armed and clean today


@pytest.mark.xfail(strict=True, reason=(
    "REVIEW 2026-10-27 H1 (MAJOR): a summary without forced_end_guard (any unit played by a pin before bf5f1a8e, "
    "e.g. a registered read still playing at its old pin) is 'judged as before', even when its own capture shows "
    "the peer's FORCELOSE at turn 3. Flips RED when assemble re-derives the guard from the captures."))
def test_an_unguarded_summary_whose_capture_shows_a_peer_drop_does_not_count(tmp_path):
    import json

    from main.reads.assemble import anchor_units

    state = tmp_path / "kakuna"
    u = {"name": "p__home72__00", "label": "p", "arm": "p", "seed": 1, "teamset": "home72", "unit_seed": 0,
         "model": "m.zip", "model_sha256": "0" * 64, "games": 1}
    cell = state / "cells" / u["name"]
    (cell / "capture").mkdir(parents=True)
    (state / "plan.json").write_text(json.dumps({"units": [u]}))
    (cell / "summary.json").write_text(json.dumps(_summary(
        wins=1, losses=0, ties=0,
        cell={"opponent_version": "Kakuna@ckpt34", "forfeit_turn_limit": LIMIT, "our_regime": "greedy",
              "server_version": "ws_frontend[in-process]@de15dfc7b+rust:cargo:src/rust_sim"})))
    (cell / "games.jsonl").write_text(json.dumps({"half": "peer_challenge", "index": 1, "our_team": "t1.txt",
                                                  "result": "win", "turns": 3, "battle_tag": "battle-gen3ou-1"}) + "\n")
    start = {"formatid": "gen3ou", "p1": {"name": "Gen3AIAnchor1"}, "p2": {"name": "MetaKakuna1"}}
    (cell / "capture" / "battle-gen3ou-1.json").write_text(json.dumps({
        "tag": "battle-gen3ou-1", "seed": [1, 2, 3, 4],
        "commands": ["START " + json.dumps(start), "CHOOSE p1 move surf", "CHOOSE p2 move spikes", "FORCELOSE p2",
                     "END"],
        "chunks": [["p1", "|turn|1"], ["p2", "|turn|1"], ["p1", "|turn|3"], ["p2", "|turn|3"],
                   ["p1", "|win|Gen3AIAnchor1"], ["p2", "|win|Gen3AIAnchor1"]]}))
    (row,) = anchor_units(state)
    assert row["counted"] is False, row["reasons"]
