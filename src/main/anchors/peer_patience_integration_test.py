"""THE PEER'S PATIENCE (S-U FINDING S-3, ``designs/endstate/design_search_u_headroom.md`` §8 A9): a searched decision
takes 30-90 s; does the Kakuna peer, or our in-process server, time out a decision that long?

Read first (2026-10-27). Our front end has NO battle timer (``ws_frontend``: ``/timer`` is accepted and does nothing).
The Metamon peer runs upstream poke-env 0.8, whose ``OpenAIGymEnv`` waits on its observation queue with no timeout,
and Metamon builds it with poke-env's defaults: a websocket client that PINGS every 20 s and closes the connection
when a pong is 20 s late (``ps_client.listen``: ``ws.connect(ping_interval=20, ping_timeout=20)``). So the one timer
is the keepalive: a decision may take as long as it likes PROVIDED this process's event loop keeps answering pings.
A search run ON the loop would block it; the front end would then see the peer drop, forfeit it, and book OUR win.
The live searcher therefore runs every search in a worker thread (``main.anchors.search_side``). Our own watchdog
(``--first-game-timeout`` / ``--progress-timeout``) is the other clock; a searched unit sizes it
(``main.x35.kakuna.UNIT_SEARCH_S``).

This test is the real check: one game against ``metamon:Kakuna`` in which ONE of our decisions takes 70 s (longer
than the ping interval plus its timeout), spent OFF the event loop exactly as a search is. The game must play on to
our forfeit turn with the peer clean and connected. Slow tier: it starts the real Metamon peer.

THE NEGATIVE CONTROL is the forfeit-win guard's test (``runner.forced_end_verdict``): the same 70 s spent ON the loop
drops the peer (``keepalive ping timeout``), the front end FORCELOSEs it and the sim books OUR win. That game must be
VOID: no row, a FAILED summary naming ``non_battle_termination``, a non-zero exit, and a unit that does not count.
Before the guard (the S-U build's smoke, 2026-10-27) the same run wrote ``status OK`` with a win.
"""
from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

import pytest

from main.anchors import core_side
from main.anchors.results import unit_counts
from main.x35 import kakuna
from main.anchors.anchors_integration_test import _save_current_generation_checkpoint, _skip_reason
from main.anchors.cli import main as anchors_main

pytestmark = [pytest.mark.integration, pytest.mark.slow]

DELAY_S = 70.0
FORFEIT_TURN = 8


class _SlowDecision:
    """A stand-in for the searcher: our policy's own choice, but decision ``at_n`` waits ``delay_s`` OFF the loop."""

    def __init__(self, policy, delay_s: float, at_n: int = 2, on_loop: bool = False) -> None:
        self.policy, self.delay_s, self.at_n, self.on_loop = policy, delay_s, at_n, on_loop
        self.slow_s = None

    async def decide(self, front, tag, d, row, mask, conn):
        if int(d["n"]) == self.at_n and self.slow_s is None:
            t0 = time.monotonic()
            if self.on_loop:
                time.sleep(self.delay_s)          # BLOCKS the event loop: no keepalive pong is answered
            else:
                await asyncio.to_thread(time.sleep, self.delay_s)
            self.slow_s = time.monotonic() - t0
        return self.policy.decide(row, mask)

    def summary(self):
        return {"search": None, "slow_s": self.slow_s}

    def close(self):
        pass


def _play_one(tmp_path, monkeypatch, *, on_loop: bool, name: str):
    reason = _skip_reason("rust")
    if reason:
        pytest.skip(reason)
    model = _save_current_generation_checkpoint(tmp_path / "current_gen_run")
    slow = {}
    real = core_side.build_slot

    def build_slot(*a, **k):
        slot = real(*a, **k)
        slot.searcher = slow["s"] = _SlowDecision(slot.policy, DELAY_S, on_loop=on_loop)
        return slot

    monkeypatch.setattr(core_side, "build_slot", build_slot)
    out = tmp_path / "out"
    # a CLAIMED 95xx port (the reads' held-flock claim): the two tests of this file may run at once under xdist, and an
    # unclaimed pick raced them both onto 9500
    port, claim = kakuna.claim_port()
    try:
        rc = _anchors(model, out, port, name)
    finally:
        claim.close()
    return rc, out, slow["s"]


def _anchors(model, out, port, name):
    return anchors_main(["--model", model, "--opponent", "metamon:Kakuna", "--regime", "greedy", "--teamset", "away",
                       "--games", "1", "--device", "cpu", "--out", str(out), "--seed-base", "7", "--team-seed", "7",
                       "--forfeit-turn-limit", str(FORFEIT_TURN), "--first-game-timeout", "1800",
                       "--progress-timeout", "1800", "--peer-ready-timeout", "1200",
                       "--port", str(port), "--username", f"Gen3AI{name}", "--peer-username", f"Meta{name}"])


def test_a_seventy_second_decision_off_the_loop_keeps_the_kakuna_peer_connected(tmp_path, monkeypatch):
    rc, out, slow = _play_one(tmp_path, monkeypatch, on_loop=False, name="Patience")
    summary = json.loads((out / "summary.json").read_text())
    rows = [json.loads(ln) for ln in (out / "games.jsonl").read_text().splitlines()]
    assert rc == 0, summary.get("failure")
    assert slow.slow_s is not None and slow.slow_s >= DELAY_S, "the slow decision never happened"
    assert len(rows) == 1 and rows[0]["hit_forfeit_limit"] and rows[0]["turns"] == FORFEIT_TURN, rows
    # peer_clean is not asserted: Metamon's teardown after OUR forfeit is its own (anchors_integration_test, H16);
    # the connection staying up THROUGH the slow decision is what this test is about
    assert summary["regime_verified_decisions"] is True
    assert summary["forced_end_guard"] == "armed" and summary["non_battle_endings"] == []
    # our own turn-limit forfeit IS a forced end, and the one that is an outcome
    assert [(f["cause"], f["turn"]) for f in summary["forced_ends"]] == [("forfeit", FORFEIT_TURN)]
    log = Path(out / "ws_frontend.log").read_text()
    assert "disconnected mid-battle" not in log, "the peer dropped during the slow decision"
    peer_log = "\n".join(p.read_text(errors="replace") for p in out.rglob("peer_*.log"))
    assert "keepalive ping timeout" not in peer_log


def test_the_negative_control_a_blocked_loop_drops_the_peer_and_its_forfeit_win_is_void(tmp_path, monkeypatch):
    """Fails on revert of the guard: without it this run wrote ``status OK``, rc 0, a WIN row (S-U build smoke)."""
    rc, out, slow = _play_one(tmp_path, monkeypatch, on_loop=True, name="NegCtl")
    assert slow.slow_s is not None and slow.slow_s >= DELAY_S, "the blocking decision never happened"
    log = Path(out / "ws_frontend.log").read_text()
    assert "disconnected mid-battle" in log, "the control did not drop the peer: it tests nothing"
    summary = json.loads((out / "summary.json").read_text())
    rows = [json.loads(ln) for ln in (out / "games.jsonl").read_text().splitlines() if ln.strip()]
    assert rc != 0
    assert summary["status"] == "FAILED" and summary["failure"]["cause"] == "non_battle_termination", summary
    assert not any(r["result"] == "win" for r in rows) and rows == [], rows
    (void,) = summary["non_battle_endings"]
    assert void["cause"] == "disconnect" and void["slot"] != "" and void["booked"] == "win"
    assert not unit_counts(summary, games=1).counts
