"""The eval ROSTER — the bot opponents, the schedule constants and the player builders, split out of
``eval_callback.py`` (2026-10-01).

The single source of truth for which bots an eval cycle plays (``_EVAL_OPPONENT_SPECS``), their display
names, the flat schedule (``EVAL_FREQ_STEPS`` / ``EVAL_GAMES`` / ``EVAL_SHARD_GAMES``) and the per-player
concurrency ceilings, plus ``build_eval_opponents`` / ``build_eval_players``. ``eval_callback``
re-exports every public name here.
"""
from poke_env.player import RandomPlayer, SimpleHeuristicsPlayer
from poke_env.ps_client import AccountConfiguration

from agents.opponents import (
    Gen3StallerPlayer, Gen3AggressivePlayer, Gen3SetupSweepPlayer,
    Gen3StallerV2Player, Gen3AggressiveV2Player, Gen3SetupSweepV2Player,
    Gen3HeuristicV2Player,
)
from agents.training.eval_player import EvalRLPlayer

BATTLE_FORMAT = "gen3ou"

# Single-player concurrency ceiling (kept for SelfPlayCallback's one-player path).
_EVAL_CONCURRENCY = 100
# PerOpponentEvalCallback evaluates every opponent concurrently — one RLPlayer per
# opponent, all gathered. Cap the AGGREGATE in-flight battles so N opponents don't


# Per-opponent in-flight battles in the subprocess eval worker. ONE game at a time.
# Eval inference is single-threaded (one Python thread does every forward in this
# process), so overlapping battles never parallelizes the bottleneck — it only piles
# on contention: extra Node sim procs on the bridge / extra load on the shared
# Showdown server, both fighting training's CPU-saturated env workers. Overlap
# measured slower, not faster. Cross-opponent parallelism still comes from the
# `--eval-workers` (3) subprocesses work-stealing the pool; each plays serially.
_EVAL_SUBPROCESS_CONCURRENCY = 1

# Flat eval schedule — one cadence, one game count, applied uniformly to every bot
# AND every self-play sentinel. No maturity tiers, no per-opponent caps: eval runs
# non-blocking in a subprocess and skips a cycle whenever the previous one is still
# running, so a heavier roster self-throttles (cadence just goes sparser) instead of
# needing hand-tuned ceilings.
EVAL_FREQ_STEPS = 2_000_000
EVAL_GAMES = 100

# Battle-level work-stealing: each opponent's EVAL_GAMES games are split into shard units of
# (at most) this many games, so any idle worker can drain a straggler's remaining games instead of
# one worker grinding a whole opponent alone. Smaller → finer tail collapse but more player builds
# (and, on the websocket transport, more connection churn — the bridge is preferred for fine
# shards). `>= EVAL_GAMES` ⇒ one shard per opponent == the original opponent-level behaviour.
# Default 25 → 4 shards/opponent: a ~4x shorter tail at modest cost. Tunable via
# `--eval-shard-games`.
EVAL_SHARD_GAMES = 25

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

# Full roster, in spec order. Random first (the broken-model floor).
_EVAL_ROSTER = [name for (name, _cls, _prefix) in _EVAL_OPPONENT_SPECS]


def eval_opponent_names() -> list[str]:
    """Ordered display names of the full eval roster (all bots + Random)."""
    return list(_EVAL_ROSTER)


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


def build_eval_players(model, names, teambuilder, mappings, server_config, concurrency,
                       tag="", *, start_listening=True, gamma: float = 0.99, reward_fn_factory):
    """One EvalRLPlayer per opponent name, sharing the (frozen) model + teambuilder. ``reward_fn_factory``
    is REQUIRED — pass ``functools.partial(Gen3RewardManager, config=RewardConfig.from_dict(...))`` so
    eval measures the run's actual reward, not a default one."""
    return {
        name: EvalRLPlayer(
            model=model, team=teambuilder, battle_format=BATTLE_FORMAT,
            server_configuration=server_config, mappings=mappings,
            account_configuration=AccountConfiguration(f"RLEv{tag}{i}", "password"),
            max_concurrent_battles=concurrency,
            stochastic=False,  # bot-eval measures the GREEDY policy
            start_listening=start_listening,
            gamma=gamma, reward_fn_factory=reward_fn_factory,
        )
        for i, name in enumerate(names)
    }
