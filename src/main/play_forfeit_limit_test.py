"""``play.py``'s forfeit limit is the TRAINER's, not a second copy of it.

A websocket head-to-head against a third-party bot (Metamon, Foul Play) only measures the agent
we train if a stalled game ends on the same turn a training episode's would. That number is
``agents.training.stall.StallConfig.threshold`` (== ``MAX_TURNS``, ``gen3_deadline_clock_v1``);
these tests fail if ``play.py`` ever grows its own.
"""

from agents.observation.constants import MAX_TURNS
from agents.training.stall import StallConfig
from main.play import DEFAULT_FORFEIT_TURN_LIMIT, build_parser


def test_default_is_the_trainers_own_constant():
    """No second 250. The default IS `StallConfig.threshold`, which IS `MAX_TURNS`."""
    assert DEFAULT_FORFEIT_TURN_LIMIT == StallConfig().threshold == MAX_TURNS


def test_parser_default_matches():
    args = build_parser().parse_args([])
    assert args.forfeit_turn_limit == DEFAULT_FORFEIT_TURN_LIMIT


def test_flag_overrides():
    args = build_parser().parse_args(["--forfeit-turn-limit", "120"])
    assert args.forfeit_turn_limit == 120


def test_the_flag_reaches_the_rust_clients_config(monkeypatch):
    """The plumbing, not just the parse — a flag that never reaches the client is a no-op, and a no-op here reads
    exactly like a limit that fired. The Rust client (`main.live.client`, the only client since P6) forfeits at
    `ClientConfig.forfeit_turn_limit`; `main_rust` must build every config from the flag."""
    import asyncio

    import main.live.client as live_client
    import main.play as play

    captured = []

    class _Stop(Exception):
        pass

    class _FakeClient:
        def __init__(self, cfg, **kw):
            captured.append(cfg)

        async def connect(self):
            raise _Stop()

        async def close(self):
            pass

    monkeypatch.setattr(live_client, "LiveClient", _FakeClient)
    monkeypatch.setattr(play, "rust_policy", lambda args, seed_offset=0: object())
    args = build_parser().parse_args(
        ["--mode", "challenge", "--model", "x.zip", "--opponent", "y", "--forfeit-turn-limit", "77"])
    try:
        asyncio.run(play.main_rust(args))
    except _Stop:
        pass
    assert [c.forfeit_turn_limit for c in captured] == [77]
