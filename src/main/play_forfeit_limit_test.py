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


def test_the_flag_reaches_the_players_stall_config():
    """The plumbing, not just the parse — a flag that never reaches `RLPlayer` is a no-op, and a
    no-op here reads exactly like a limit that fired."""
    captured = {}

    class _FakeModel:
        class policy:
            @staticmethod
            def modules():
                return []

    import main.play as play

    def _fake_rl_player(**kwargs):
        captured.update(kwargs)
        return object()

    def _fake_load(*a, **k):
        return _FakeModel()

    import sys
    import types

    # `build_model_player` imports these lazily INSIDE the function, so a stub module in
    # sys.modules is what the import actually resolves to.
    sb3_contrib = types.ModuleType("sb3_contrib")
    sb3_contrib.MaskablePPO = types.SimpleNamespace(load=_fake_load)
    inference_player = types.ModuleType("agents.inference.player")
    inference_player.RLPlayer = _fake_rl_player
    state_encoder = types.ModuleType("agents.observation.state_encoder")
    state_encoder.load_mappings = lambda: None

    saved = {k: sys.modules.get(k) for k in
             ("sb3_contrib", "agents.inference.player", "agents.observation.state_encoder")}
    sys.modules["sb3_contrib"] = sb3_contrib
    sys.modules["agents.inference.player"] = inference_player
    sys.modules["agents.observation.state_encoder"] = state_encoder
    try:
        args = build_parser().parse_args(
            ["--mode", "challenge", "--model", "x.zip", "--forfeit-turn-limit", "77"])
        play.build_model_player(args, teambuilder=None, server_config=None, account=None)
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v

    assert captured["stall_config"].threshold == 77
