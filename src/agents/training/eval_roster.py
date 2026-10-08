"""The eval ROSTER — the bot opponents, the schedule constants and the player builders, split out of
``eval_callback.py`` (2026-10-01).

The single source of truth for which bot CLASS stands behind each roster name (``_EVAL_OPPONENT_SPECS``),
their display names, the flat schedule (``EVAL_FREQ_STEPS`` / ``EVAL_GAMES`` / ``EVAL_SHARD_GAMES``, re-exported
from ``eval_schedule``) and the per-player concurrency ceiling, plus ``build_eval_opponents`` (the poke-env bot
players ``main.search_dividend`` and the bot calibration still build). ``eval_callback`` re-exports every public
name here. (``build_eval_players`` — the poke-env ``EvalRLPlayer`` trainee — was deleted with the Python eval
worker, poke-env retirement P6 slice 6c; eval plays on the Rust eval core.)
"""
from poke_env.player import RandomPlayer, SimpleHeuristicsPlayer
from poke_env.ps_client import AccountConfiguration

from agents.opponents import (
    Gen3StallerPlayer, Gen3AggressivePlayer, Gen3SetupSweepPlayer,
    Gen3StallerV2Player, Gen3AggressiveV2Player, Gen3SetupSweepV2Player,
    Gen3HeuristicV2Player,
)

# The schedule constants, the battle format and the roster's NAMES are poke-env-free and live in `eval_schedule`
# (P1 of the retirement); re-exported here so every historical import site resolves.
from agents.training.eval_schedule import (  # noqa: F401
    BATTLE_FORMAT, _EVAL_CONCURRENCY, EVAL_FREQ_STEPS, EVAL_GAMES, EVAL_SHARD_GAMES,
    RANDOM_OPPONENT_NAME as _SCHEDULE_RANDOM_NAME, _EVAL_ROSTER as _SCHEDULE_ROSTER, eval_opponent_names,
)

_OPPONENT_NAMES: dict[type, str] = {
    RandomPlayer: "random",
    SimpleHeuristicsPlayer: "heuristic",
    Gen3StallerPlayer: "staller",
    Gen3AggressivePlayer: "aggressive",
    Gen3SetupSweepPlayer: "setup_sweep",
    # V2 bots — in the full eval rotation (see _EVAL_OPPONENT_SPECS) and the training roster.
    Gen3HeuristicV2Player: "heuristic2",
    Gen3StallerV2Player: "staller_v2",
    Gen3AggressiveV2Player: "aggressive_v2",
    Gen3SetupSweepV2Player: "setup_sweep_v2",
}


def opponent_name(player_cls: type) -> str:
    """Return the display name for a player class (TensorBoard keys, TUI labels)."""
    return _OPPONENT_NAMES.get(player_cls, player_cls.__name__)


RANDOM_OPPONENT_NAME = opponent_name(RandomPlayer)
assert RANDOM_OPPONENT_NAME == _SCHEDULE_RANDOM_NAME, (RANDOM_OPPONENT_NAME, _SCHEDULE_RANDOM_NAME)

# The canonical eval roster: (display name, player class, account prefix). Every bot
# plays — both the v1 and v2 of each archetype, since they play differently and the
# extra diversity is the point. v1/v2 are paired for readable TUI/TensorBoard ordering.
# Single source of truth so the in-process selfplay path, the subprocess worker, and the
# orchestrator agree. Random leads (a cheap "is the model broken" floor; eval-only,
# excluded from `win_rate_vs_bots`).
_EVAL_OPPONENT_SPECS: list[tuple[str, type, str]] = [
    ("random", RandomPlayer, "CbRand"),
    ("heuristic", SimpleHeuristicsPlayer, "CbHeur"),
    ("heuristic2", Gen3HeuristicV2Player, "CbHeur2"),
    ("staller", Gen3StallerPlayer, "CbStall"),
    ("staller_v2", Gen3StallerV2Player, "CbStallV2"),
    ("aggressive", Gen3AggressivePlayer, "CbAggr"),
    ("aggressive_v2", Gen3AggressiveV2Player, "CbAggrV2"),
    ("setup_sweep", Gen3SetupSweepPlayer, "CbSetup"),
    ("setup_sweep_v2", Gen3SetupSweepV2Player, "CbSetupV2"),
]

# Full roster, in spec order. Random first (the broken-model floor). The poke-env-free copy of these names is
# `eval_schedule._EVAL_ROSTER` (what `eval_opponent_names()` serves); the two MUST agree, so a drift fails at import.
_EVAL_ROSTER = [name for (name, _cls, _prefix) in _EVAL_OPPONENT_SPECS]
assert _EVAL_ROSTER == _SCHEDULE_ROSTER, (
    f"eval_roster._EVAL_OPPONENT_SPECS names {_EVAL_ROSTER} != eval_schedule._EVAL_ROSTER {_SCHEDULE_ROSTER}")


def eval_opponent_class(name: str) -> type:
    """The player CLASS for one roster bot name — the single lookup, so a caller that needs to
    build a bot with its OWN account (e.g. an external-anchor read, where the peer must be told
    our username up front) does not re-derive the name->class table and drift from this one.

    Raises ``KeyError`` naming the roster, because a silently-unknown bot name is a cell that
    quietly measures a different opponent.
    """
    by_name = {n: cls for (n, cls, _prefix) in _EVAL_OPPONENT_SPECS}
    if name not in by_name:
        raise KeyError(f"unknown eval bot {name!r}; the roster is {_EVAL_ROSTER}")
    return by_name[name]


def build_eval_opponents(server_config, teambuilder, names, tag="", *, start_listening=True):
    """Construct the opponent players for `names`.

    Each Player opens its own Showdown connection on construction, so build only
    the names this caller actually needs. `tag` is appended to every account name
    and MUST be unique per concurrently-live set — under work stealing a worker
    builds a fresh set per claimed opponent, so the tag carries (cycle, worker,
    claim) to avoid username collisions on the shared server.

    `start_listening=False` (bridge eval) opens no websocket — the in-process
    `run_local_battles` driver supplies the transport instead.
    """
    by_name = {n: (cls, prefix) for (n, cls, prefix) in _EVAL_OPPONENT_SPECS}
    out = []
    for name in names:
        cls, prefix = by_name[name]
        out.append((name, cls(
            battle_format=BATTLE_FORMAT, team=teambuilder,
            server_configuration=server_config,
            account_configuration=AccountConfiguration(f"{prefix}{tag}", "password"),
            max_concurrent_battles=_EVAL_CONCURRENCY,  # opponent side high; RL side governs
            start_listening=start_listening,
        )))
    return out
