"""OUR side as a LIVE CLIENT on the Rust stack (P6 of the poke-env retirement; ``main.anchors.live_side``).

Two tiers:

* **unit** (no battle): the bot's stream seeds are the env core's route rule (the pinned Rust values); a
  checkpoint's per-decision verification (``stochastic``, argmax match, the stall forfeit not counted) lands on
  the battle's record with ``n_decisions`` PER GAME; the loader's ``auto`` / ``bare`` / ``foreign`` rule
  (``core_side.load_policy`` — the one loader both of our transports use) records which loader ran and never
  falls back silently.
* **sim**: a real series on our in-process front end — OUR side the Rust port of a roster bot on the
  ``bot_reader`` session (accepting), a seeded random live client challenging — played twice with the same seeds
  (the bot's choices are identical: its streams are SEEDED, not drawn from a process-wide ``random``) and once
  with a different ``--bot-seed`` (they differ: the seed reaches the bot). Every bot choice must be the frame's
  token at the bot's index, and every battle books one record.

The read against a real Metamon on Node / the front end: ``anchors_integration_test.py`` (``slow``); the identity
proof against the deleted legacy client: ``designs/research_state/measurements/pokeenv_p6_anchors_2026-10-07/``.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any, List

import numpy as np
import pytest

from main.anchors import core_side, live_side
from main.anchors.session import OurSideState


# ─────────────────────────────────────────────────────────────────────────────── unit ───
def test_the_bot_streams_are_the_env_cores_route_rule() -> None:
    """``opponents::stream_seed(5, env, k)`` — the values the Rust test pins (``a_bot_route_declares_its_seed``)."""
    from agents.training.rust_env_opponents import bot_stream_seed

    assert live_side.bot_streams(5, "ours_challenge")["choice"] == 4517933670823692284
    assert bot_stream_seed(11, 2, 1) == 5390792918547426617          # the twin, on the second pinned value
    assert live_side.bot_streams(11, "peer_challenge") == {"choice": bot_stream_seed(11, 1, 0),
                                                           "protect": bot_stream_seed(11, 1, 1)}


class _Core:
    stochastic = True

    def __init__(self, script):
        self.script = list(script)

    def decide(self, row, mask):
        return self.script.pop(0)


def _plan(**kw: Any) -> SimpleNamespace:
    base = dict(server_uri="ws://localhost:9599/showdown/websocket", battle_format="gen3ou", connect_timeout_s=30.0,
                progress_timeout_s=60.0, our_side="model", forfeit_turn_limit=250, team_seed=7, bot_seed=None,
                model_zip="/x.zip", device="cpu", model_loader="auto", regime="greedy", our_temperature=None,
                our_transport="live")
    base.update(kw)
    return SimpleNamespace(**base)


def test_a_checkpoints_decisions_are_verified_and_booked_per_game(monkeypatch) -> None:
    from main.live.client import BattleResult, Decision
    from main.live.reader import Frame

    monkeypatch.setattr(core_side, "load_policy", lambda path, device, loader, state: "model")
    monkeypatch.setattr(core_side, "CorePolicy",
                        lambda model, stochastic, temperature: _Core([(2, 2), (1, 3), (0, 0)]))
    state = OurSideState()
    client = live_side.build_client(_plan(), "ours_challenge", username="p6ours", team_spec={"kind": "pool"},
                                    state=state)
    fr = Frame(side="p1", row=np.zeros(4, np.float32), mask=np.ones(11, np.int8), tokens={}, turn=1, line=1,
               rqid=None, n=0)
    for k in range(3):
        idx = client.policy.choose(fr)
        client.on_decision(Decision(f"battle-gen3ou-{1 + (k == 2)}", fr, idx, "t"))
    client.on_decision(Decision("battle-gen3ou-2", fr, None, None))      # the stall forfeit: not a decision
    client.results = [BattleResult("battle-gen3ou-1", "p1", "p6ours", True, 31, 2, False),
                      BattleResult("battle-gen3ou-2", "p2", None, None, 251, 1, True)]
    for r in client.results:
        client._record(r)
    a, b = state.records
    assert (a.result, a.turns, a.n_decisions, a.our_argmax_matches, a.our_argmax_decisions) == ("win", 31, 2, 1, 2)
    assert (b.result, b.hit_forfeit_limit, b.n_decisions, b.our_argmax_matches, b.our_argmax_decisions) == (
        "tie", True, 1, 1, 1)
    assert (a.n_defaults, a.n_redecides) == (0, 0)
    assert state.stochastic_kwargs == [True] and len(state.decision_times) == 3
    assert len(state.draws) == 0, "no team was drawn: the client draws only at /utm"


def test_a_bot_client_never_forfeits_and_reports_no_argmax(monkeypatch) -> None:
    state = OurSideState()
    client = live_side.build_client(_plan(our_side="bot:staller", bot_seed=9), "peer_challenge", username="p6ours",
                                    team_spec={"kind": "pool"}, state=state)
    assert client.cfg.forfeit_turn_limit > 10 ** 6, "a bot carries no stall check (F-LF-5)"
    reader = client.reader_factory()
    assert (reader.bot, reader.seed, reader.env) == ("staller", 9, 1)
    from main.live.client import BattleResult

    client._record(BattleResult("battle-gen3ou-3", "p1", "p6ours", True, 260, 140, False))
    rec = state.records[0]
    assert rec.our_argmax_matches is None and rec.hit_forfeit_limit is True and rec.n_decisions == 140


@pytest.mark.parametrize("mode, expected", [("bare", "bare"), ("auto", "bare")])
def test_a_successful_bare_load_is_recorded_as_bare(monkeypatch, mode, expected) -> None:
    import agents.model.oracle_reveal as oracle_reveal
    import agents.model.snapshot as snapshot

    monkeypatch.setattr(oracle_reveal, "refuse_if_revealed", lambda *a, **k: None)
    monkeypatch.setattr(snapshot, "historical_load_kwargs", lambda path: {})
    monkeypatch.setattr(snapshot, "load_checkpoint_strict", lambda path, device, **k: ("model", path, device))
    state = OurSideState()
    assert core_side.load_policy("/s.zip", "cpu", mode, state) == ("model", "/s.zip", "cpu")
    assert state.model_loader == expected


def test_auto_falls_back_to_the_foreign_loader_and_says_which_one_ran(monkeypatch, capsys) -> None:
    """🚨 A fallback nobody can see is a cell whose policy nobody can identify: ``model_loader`` is set from what
    HAPPENED, not from the flag."""
    import agents.model.oracle_reveal as oracle_reveal
    import agents.model.snapshot as snapshot
    from main.anchors import session as session_mod

    def bare(path, device, **k):
        raise TypeError("ExtractorBuild.__init__() got an unexpected keyword argument 'threat_prob_outspeed'")

    monkeypatch.setattr(oracle_reveal, "refuse_if_revealed", lambda *a, **k: None)
    monkeypatch.setattr(snapshot, "historical_load_kwargs", lambda path: {})
    monkeypatch.setattr(snapshot, "load_checkpoint_strict", bare)
    monkeypatch.setattr(session_mod, "foreign_loader", lambda path, device: "foreign-model")
    state = OurSideState()
    assert core_side.load_policy("/s.zip", "cpu", "auto", state) == "foreign-model"
    assert state.model_loader == "foreign"
    out = capsys.readouterr().out
    assert "bare load failed" in out and "threat_prob_outspeed" in out


def test_bare_mode_does_NOT_silently_fall_back(monkeypatch) -> None:
    """``--model-load bare`` is the LADDER's semantics: play the model you were handed, or refuse."""
    import agents.model.oracle_reveal as oracle_reveal
    import agents.model.snapshot as snapshot
    from main.anchors import session as session_mod

    def bare(path, device, **k):
        raise TypeError("unexpected keyword argument 'threat_prob_outspeed'")

    called: List[str] = []
    monkeypatch.setattr(oracle_reveal, "refuse_if_revealed", lambda *a, **k: None)
    monkeypatch.setattr(snapshot, "historical_load_kwargs", lambda path: {})
    monkeypatch.setattr(snapshot, "load_checkpoint_strict", bare)
    monkeypatch.setattr(session_mod, "foreign_loader", lambda path, device: called.append(path) or "nope")
    with pytest.raises(TypeError):
        core_side.load_policy("/s.zip", "cpu", "bare", OurSideState())
    assert called == []


# ──────────────────────────────────────────────────────────────────────────────── sim ───
async def _bot_series(bot: str, bot_seed: int, n: int = 2, seed_base: int = 61_007):
    """``n`` battles on the in-process front end: our Rust bot (accepting) vs a seeded random live client."""
    from main.anchors.server import InProcessFrontEnd, pick_port
    from main.live.client import ClientConfig, LiveClient
    from main.live.policy import RandomPolicy
    from utils.team_sources import packed_teams

    teams = packed_teams("pool")
    port = pick_port([9500, 9599])
    front = InProcessFrontEnd(port, seed_base=seed_base)
    await front.__aenter__()
    state = OurSideState()
    ours = opp = None
    choices: List[Any] = []
    try:
        plan = _plan(server_uri=front.uri, our_side=f"bot:{bot}", bot_seed=bot_seed, team_seed=11)
        ours = live_side.build_client(plan, "peer_challenge", username="p6ours", team_spec={"kind": "pool"},
                                      state=state)
        opp = LiveClient(ClientConfig(uri=front.uri, username="p6opp", forfeit_turn_limit=250),
                         policy=RandomPolicy(3), team_fn=lambda: teams[5])
        await ours.connect()
        await opp.connect()
        acc = asyncio.ensure_future(ours.accept("p6opp", n))
        await asyncio.wait_for(opp.challenge("p6ours", n), timeout=600)
        await asyncio.wait_for(acc, timeout=120)
        choices = list(ours._reader.choices)
    finally:
        for c in (ours, opp):
            if c is not None:
                await c.close()
        await front.__aexit__(None, None, None)
    return state, choices


@pytest.mark.sim
@pytest.mark.integration
def test_a_rust_bot_plays_our_side_on_its_own_reading_and_its_streams_are_seeded(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("GEN3AI_LIVE_HALT_FILE", str(tmp_path / "halt.json"))
    s1, c1 = asyncio.run(_bot_series("random", 101))
    s2, c2 = asyncio.run(_bot_series("random", 101))
    s3, c3 = asyncio.run(_bot_series("random", 102))
    for st, ch in ((s1, c1), (s2, c2), (s3, c3)):
        assert len(st.records) == 2, [r.battle_tag for r in st.records]
        assert sum(r.n_decisions for r in st.records) == len(ch) > 0
        assert all(r.our_argmax_decisions is None for r in st.records)
        assert all(c["choice_words"] > 0 for c in ch[-1:]), "the random bot draws at every decision"
    assert c1 == c2, "the same --bot-seed must replay the same bot draws (a seeded stream, not process-wide random)"
    assert [c["token"] for c in c1] != [c["token"] for c in c3], "a different --bot-seed did not reach the bot"
    assert not (tmp_path / "halt.json").exists()
