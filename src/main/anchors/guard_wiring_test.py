"""REVIEW 2026-10-27 H7: the three guard WIRINGS no fast test pinned (``designs/ops/reviews/anchor_guards_review_2026-10-27.md``,
mutants M24, M10, M27). Each test drives the real call site with its processes and sockets stubbed, so deleting the one
call it pins turns it red:

* M24 -- ``runner.run_series`` applies the forfeit-win guard (before: only the slow ``peer_patience_integration_test``);
* M10 -- ``runner.run_peer_pair_half`` runs H21 on EITHER peer of a pair cell;
* M27 -- ``cli.main`` carries the ``forced_end`` block into ``summary.json`` (``results.unit_counts`` reads it).
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace

from main.anchors import runner
from main.anchors.peers import PeerPlan
from main.anchors.server import FORCED_ENDS_NAME, arm_forced_ends
from main.anchors.session import PEER_FAULT_LINE, BattleRecord
from utils.bridge.ws_frontend import FORCED_END_SCHEMA


def _record(i: int, won: bool = True) -> BattleRecord:
    return BattleRecord(battle_tag=f"battle-gen3ou-{i}", turns=3 if i == 1 else 40, won=won, finished=True,
                        hit_forfeit_limit=False, our_team="t", n_decisions=3, n_defaults=0, n_redecides=0,
                        t_finished=time.time())


def _series_plan(out: Path, **kw):
    base = dict(our_transport="ws", our_side_is_peer=False, opponent="metamon:Kakuna", regime="greedy",
                teamset="home72", mirrored_pairs=False, server_impl="rust", started_server=True, out_dir=out,
                forfeit_turn_limit=250, our_username="Gen3AIAnchor", peer_username="MetaKakuna",
                half_sizes=lambda: {"ours_challenge": 2})
    base.update(kw)
    return SimpleNamespace(**base)


async def test_run_series_applies_the_forfeit_win_guard(tmp_path, monkeypatch):
    """M24: a peer's drop at turn 3 that our side booked as a WIN is dropped from the rows and fails the series."""
    path = tmp_path / FORCED_ENDS_NAME
    arm_forced_ends(path)

    async def half(plan, cfg, half, n, front=None):
        path.write_text(json.dumps({"schema": FORCED_END_SCHEMA, "tag": "battle-gen3ou-1", "index": 0, "slot": "p2",
                                    "user": "metakakuna1", "name": "MetaKakuna1", "cause": "disconnect", "turn": 3,
                                    "time": 0.0}) + "\n")
        return [_record(1), _record(2, won=False)], {"regime_verified_decisions": True, "peer_clean": True}, None

    monkeypatch.setattr(runner, "run_half", half)
    monkeypatch.setattr(runner, "our_team_facts", lambda plan, cfg: (1, None, None))
    monkeypatch.setattr(runner, "cell_spec", lambda plan, report, n: "cell")
    rows, report, failure = await runner.run_series(_series_plan(tmp_path), None)
    assert [r.battle_tag for r in rows] == ["battle-gen3ou-2"]
    assert failure is not None and failure.cause == "non_battle_termination"
    assert report["forced_end_guard"] == "armed" and report["non_battle_endings"][0]["booked"] == "win"


async def test_the_pair_half_applies_h21_to_either_peer(tmp_path, monkeypatch):
    """M10: in an anchor-vs-anchor cell, OUR peer's long-tail line voids the half exactly as theirs does."""
    plans = {}
    for side, text in (("theirs", "clean\n"), ("ours", f"clean\n{PEER_FAULT_LINE}\n")):
        log = tmp_path / side / "peer.log"
        log.parent.mkdir()
        log.write_text(text)
        plans[side] = PeerPlan(label=f"metamon:{side}", argv=["x", "--results-dir", str(tmp_path / "res")],
                               cwd=tmp_path, env={}, ready_pattern="", log_path=log)

    async def noop(*_a, **_k):
        return None

    report = {"regime_verified_decisions": True, "peer_error_free": True}
    adapter = SimpleNamespace(read_report=lambda _pp: dict(report))
    monkeypatch.setattr(runner, "our_peer_plan", lambda *a: plans["ours"])
    monkeypatch.setattr(runner, "peer_plan", lambda *a: plans["theirs"])
    monkeypatch.setattr(runner, "start_peer", lambda pp, nice=0: SimpleNamespace(returncode=0))
    for name in ("await_peer_ready", "watch_peer_pair", "await_peer_exit"):
        monkeypatch.setattr(runner, name, noop)
    monkeypatch.setattr(runner, "stop_peer", lambda _p: None)
    monkeypatch.setattr(runner, "read_peer_battles", lambda _d: [_record(2)])
    monkeypatch.setattr(runner, "pair_adapters", lambda _plan: (adapter, adapter))
    monkeypatch.setattr(runner.peers_mod, "team_source_count", lambda _pp: 72)
    monkeypatch.setattr(runner, "team_set_digest_or_none", lambda _d: None)
    monkeypatch.setattr(runner, "pair_exit_note", lambda *a: None)
    plan = SimpleNamespace(our_username="KakunaHome", peer_username="KakunaAway", nice=0, peer_ready_timeout_s=1,
                           first_game_timeout_s=1, progress_timeout_s=1)
    records, rep, failure = await runner.run_peer_pair_half(plan, None, "ours_challenge", 1)
    assert failure is not None and failure.cause == "peer_long_tail_fault"
    assert rep["peer_clean"] is False and rep["peer_fault_lines"] == 1


def test_cli_carries_the_forced_end_block_into_the_summary(tmp_path, monkeypatch):
    """M27: the runner's ``forced_end_guard`` / ``forced_ends`` / ``non_battle_endings`` reach ``summarize``, so the
    summary carries what ``results.unit_counts`` checks."""
    from main.anchors import cli, results as results_mod, team_overlap as team_overlap_mod
    import main.live.halt as halt

    plan = SimpleNamespace(out_dir=tmp_path, started_server=False, our_transport="core", our_side="model",
                           half_sizes=lambda: {"ours_challenge": 1}, declared_team_cell=None, deal_game=None,
                           opponent_kind="metamon")
    report = {"forced_end_guard": "armed", "forced_ends": [{"tag": "t"}], "non_battle_endings": []}
    row = SimpleNamespace(cell=SimpleNamespace(our_team_count=1))
    got = {}

    async def series(_plan, _cfg):
        return [row], report, None

    def summarize(cell, rows, **kw):
        got.update(kw)
        return {"their_argmax_match_rates": [1.0]}

    monkeypatch.setattr(cli.config_mod, "load_config", lambda: SimpleNamespace(source="cfg"))
    monkeypatch.setattr(cli, "build_plan", lambda args, cfg: plan)
    monkeypatch.setattr(cli, "render_plan", lambda plan, cfg: "")
    monkeypatch.setattr(cli, "_release_port_claim", lambda plan: None)
    monkeypatch.setattr(halt, "refuse_if_halted", lambda _who: None)
    monkeypatch.setattr(team_overlap_mod, "read_overlap", lambda plan, cfg: {})
    monkeypatch.setattr(team_overlap_mod, "render_lines", lambda _o: [])
    monkeypatch.setattr(cli.runner_mod, "run_series", series)
    monkeypatch.setattr(cli.runner_mod, "cell_spec", lambda *a: row.cell)
    monkeypatch.setattr(results_mod, "summarize", summarize)
    monkeypatch.setattr(results_mod, "write_games", lambda *a: None)
    monkeypatch.setattr(results_mod, "write_summary", lambda *a: None)
    monkeypatch.setattr(results_mod, "render", lambda _s: "")
    assert cli.main(["--model", "m.zip", "--opponent", "metamon:Kakuna", "--out", str(tmp_path)]) == 0
    assert got["forced_end"] == {"guard": "armed", "forced_ends": [{"tag": "t"}], "void": []}
