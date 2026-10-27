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


# ------------------------------------------------------------------- REVIEW 2026-10-27 (anchor_guards_review)
def test_a_peers_forfeit_at_the_cap_on_a_capped_row_is_void(tmp_path):
    """Pins the ``user in ours`` clause on its own (the review's mutant M1 survived every fast test): a forfeit at the
    turn limit by the PEER, on a row our side stamped ``hit_forfeit_limit``, is the peer's, never our cap."""
    rec = _rec(slot="p2", user="metakakuna1", cause="forfeit", turn=LIMIT)
    assert runner.forced_end_verdict(_plan(tmp_path), [_row(result="win", cap=True)], [rec])["void"]


async def test_a_forced_end_the_record_could_not_store_never_reads_armed_and_clean(tmp_path, monkeypatch):
    """REVIEW 2026-10-27 H3 (fixed): a forced end whose JSONL append fails (a full disk) moves the armed sink aside for
    good, so the guard reads BLIND (fail closed) and a later forced end never re-creates a clean-looking record."""
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
    assert len(front.forced_ends) == 1 and front.forced_ends_unrecorded   # the front end KNOWS
    v = runner.forced_end_verdict(_plan(tmp_path), [_row(battle.tag, "win")], runner.read_forced_ends(path))
    assert v["guard"] == "blind", v
    assert not path.exists() and (tmp_path / (FORCED_ENDS_NAME + ".unwritable")).exists()
    monkeypatch.setattr(ws_frontend, "open", builtins.open, raising=False)    # the disk frees up: still never clean
    gamma, delta = await _named(front, "Gen3AIAnchor2"), await _named(front, "MetaKakuna2")
    await front._drop(_fake_battle(front, gamma, delta).conns["p2"])
    assert len(front.forced_ends) == 2 and not path.exists()


def test_an_unrecorded_forced_end_makes_the_series_guard_blind(tmp_path):
    """The runner's in-process cross-check (H3): ``unrecorded`` reads no record at all, even an armed, clean one."""
    plan = _plan(tmp_path)
    arm_forced_ends(tmp_path / FORCED_ENDS_NAME)
    kept, failure = runner.apply_forced_end_guard(plan, [_row()], report := {}, None, unrecorded=True)
    assert failure is not None and failure.cause == "forced_end_guard_blind" and report["forced_end_guard"] == "blind"
    kept, failure = runner.apply_forced_end_guard(plan, [_row()], report := {}, None)
    assert failure is None and report["forced_end_guard"] == "armed"


# --------------------------------------------- REVIEW H1 (fixed): the guards RE-DERIVED at assemble time (rederive)
def _unguarded_unit(tmp_path: Path, *, result="win", turns=3, cap=False, forcelose="p2", capture=True,
                    peer_log: str = "battle 1 done\n", our_side: str = "model", limit: int = LIMIT) -> Path:
    """A kakuna state with ONE unit played by a pre-guard pin: its summary carries no ``forced_end_guard`` and no
    ``peer_fault_lines``; its capture and peer log are what every pinned unit writes."""
    import json

    state = tmp_path / "kakuna"
    u = {"name": "p__home72__00", "label": "p", "arm": "p", "seed": 1, "teamset": "home72", "unit_seed": 0,
         "model": "m.zip", "model_sha256": "0" * 64, "games": 1}
    cell = state / "cells" / u["name"]
    (cell / "capture").mkdir(parents=True)
    (cell / "peer_challenge").mkdir()
    (cell / "peer_challenge" / "peer_challenger.log").write_text(peer_log)
    (state / "plan.json").write_text(json.dumps({"units": [u]}))
    (cell / "summary.json").write_text(json.dumps(_summary(
        wins=int(result == "win"), losses=int(result == "loss"), ties=0,
        cell={"opponent_version": "Kakuna@ckpt34", "forfeit_turn_limit": limit, "our_regime": "greedy",
              "our_side": our_side, "server_version": "ws_frontend[in-process]@de15dfc7b+rust:cargo:src/rust_sim"})))
    (cell / "games.jsonl").write_text(json.dumps({"half": "peer_challenge", "index": 1, "our_team": "t1.txt",
                                                  "result": result, "turns": turns, "hit_forfeit_limit": cap,
                                                  "battle_tag": "battle-gen3ou-1"}) + "\n")
    if capture:
        start = {"formatid": "gen3ou", "p1": {"name": "Gen3AIAnchor1"}, "p2": {"name": "MetaKakuna1"}}
        cmds = ["START " + json.dumps(start), "CHOOSE p1 move surf", "CHOOSE p2 move spikes"]
        (cell / "capture" / "battle-gen3ou-1.json").write_text(json.dumps({
            "tag": "battle-gen3ou-1", "seed": [1, 2, 3, 4],
            "commands": cmds + ([f"FORCELOSE {forcelose}"] if forcelose else []) + ["END"],
            "chunks": [["p1", "|turn|1"], ["p2", "|turn|1"], ["p1", f"|turn|{turns}"], ["p2", f"|turn|{turns}"],
                       ["p1", "|win|Gen3AIAnchor1"], ["p2", "|win|Gen3AIAnchor1"]]}))
    return state


def test_an_unguarded_summary_whose_capture_shows_a_peer_drop_does_not_count(tmp_path):
    """THE H1 reproducer, fixed: under a spec's ``rederived_guards = count`` (every spec registered after the fix), the
    peer's FORCELOSE at turn 3 that booked OUR win, re-derived from the capture, makes the unit not count."""
    from main.reads.assemble import anchor_units

    (row,) = anchor_units(_unguarded_unit(tmp_path), rederived="count")
    assert row["counted"] is False, row["reasons"]
    assert any("rederived guards hit (forced_end" in r for r in row["reasons"])
    assert row["rederived"]["forced_end"] == "hit" and row["rederived"]["peer_fault"] == "clean"
    assert row["rederived_uncounted"] is True


def test_a_read_registered_before_the_rederivation_only_reports_the_hit(tmp_path):
    """``report`` (a spec registered before the fix, in flight): counting is unchanged, the hit is recorded."""
    from main.reads.assemble import anchor_units

    (row,) = anchor_units(_unguarded_unit(tmp_path), rederived="report")
    assert row["counted"] is True and row["rederived"]["hits"][0]["booked"] == "win"
    assert "rederived_uncounted" not in row


def test_our_own_turn_limit_forfeit_in_the_capture_is_the_one_clean_forced_end(tmp_path):
    from main.reads.assemble import anchor_units

    (row,) = anchor_units(_unguarded_unit(tmp_path, result="loss", turns=LIMIT, cap=True, forcelose="p1"),
                          rederived="count")
    assert row["counted"] is True and row["rederived"]["forced_end"] == "clean", row["rederived"]
    for kw in ({"turns": LIMIT - 1}, {"cap": False}, {"result": "tie"}, {"our_side": "metamon:Kakuna"},
               {"limit": 0}):                                            # each clause of the cap, on its own
        sub = tmp_path / str(sorted(kw.items()))
        args = {**dict(result="loss", turns=LIMIT, cap=True, forcelose="p1"), **kw}
        (row,) = anchor_units(_unguarded_unit(sub, **args), rederived="count")
        assert row["counted"] is False and row["rederived"]["forced_end"] == "hit", kw


def test_a_peer_long_tail_line_in_an_unguarded_units_log_is_a_hit(tmp_path):
    from main.anchors.session import PEER_FAULT_LINE
    from main.reads.assemble import anchor_units

    state = _unguarded_unit(tmp_path, forcelose=None, peer_log=f"battle 1 done\n{PEER_FAULT_LINE}")
    (row,) = anchor_units(state, rederived="count")
    assert row["counted"] is False and row["rederived"]["peer_fault"] == "hit"
    assert row["rederived"]["hits"][0]["lines"] == 1


def test_a_unit_without_its_evidence_is_unavailable_and_judged_as_before(tmp_path):
    import shutil

    from main.reads.assemble import anchor_units

    state = _unguarded_unit(tmp_path, capture=False)
    shutil.rmtree(state / "cells" / "p__home72__00" / "peer_challenge")
    (row,) = anchor_units(state, rederived="count")
    assert row["counted"] is True
    assert (row["rederived"]["forced_end"], row["rederived"]["peer_fault"]) == ("unavailable", "unavailable")
    assert all(row["rederived"]["why"])


def test_a_logged_mid_battle_drop_is_a_hit_even_without_a_capture(tmp_path):
    """The front end's log is a second witness (every pin logs ``disconnected mid-battle``): a unit played with
    capture off still shows the peer's drop. The capture and the log naming the same drop count once."""
    from main.reads.assemble import anchor_units

    line = ("2026-10-26 01:02:03,456 WARNING ws_frontend metakakuna2 disconnected mid-battle battle-gen3ou-1 — "
            "forfeiting p2\n")
    state = _unguarded_unit(tmp_path, capture=False)
    (state / "cells" / "p__home72__00" / "ws_frontend.log").write_text(line)
    (row,) = anchor_units(state, rederived="count")
    assert row["counted"] is False and row["rederived"]["forced_end"] == "hit"
    assert [(h["source"], h["booked"]) for h in row["rederived"]["hits"]] == [("ws_frontend.log", "win")]
    both = _unguarded_unit(tmp_path / "both")
    (both / "cells" / "p__home72__00" / "ws_frontend.log").write_text(line)
    (row,) = anchor_units(both, rederived="count")
    assert [h["source"] for h in row["rederived"]["hits"]] == ["capture"]
