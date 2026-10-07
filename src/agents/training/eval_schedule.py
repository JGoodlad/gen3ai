"""The eval SCHEDULE and the roster's NAMES — poke-env-free, split out of ``eval_roster.py`` (P1 of the poke-env
retirement, ``T27``).

The flat eval schedule (``EVAL_FREQ_STEPS`` / ``EVAL_GAMES`` / ``EVAL_SHARD_GAMES``), the battle format, the
per-player concurrency ceiling and the ORDERED roster of bot display names (``eval_opponent_names``). None of it
touches a poke-env object, but it lived beside the player classes (``RandomPlayer``, ``Gen3StallerPlayer``, …) in
``eval_roster``, so reading the schedule from the trainer's argument parser, ``main.h2h`` or ``main.plateau`` imported
the whole poke-env package. ``eval_roster`` (and through it ``eval_callback``) re-exports every name here, so every
historical import site still resolves.

**The roster is stated TWICE, deliberately.** ``eval_roster._EVAL_OPPONENT_SPECS`` pairs each name with its poke-env
player CLASS (and ``utils/rust_env/bot_inventory.py`` reads that literal list); :data:`_EVAL_ROSTER` here is the
poke-env-free list of the same names, in the same order. ``eval_roster`` asserts at import that the two agree and
``eval_schedule_test.py`` pins it, so a bot added to one and not the other fails at once rather than measuring a
different roster than the one named.
"""

BATTLE_FORMAT = "gen3ou"

# Single-player concurrency ceiling (kept for SelfPlayCallback's one-player path).
_EVAL_CONCURRENCY = 100
# PerOpponentEvalCallback evaluates every opponent concurrently — one RLPlayer per
# opponent, all gathered. Cap the AGGREGATE in-flight battles so N opponents don't


# Flat eval schedule — one cadence, one game count, applied uniformly to every bot
# AND every self-play sentinel. No maturity tiers, no per-opponent caps. The cycle plays blocking
# in the trainer's process (~1.5% of wall at N=256, m5_sizing PROGRESS.md O9), so the cadence is
# what bounds its cost.
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

#: The display name of the Random player (the broken-model floor; eval-only, excluded from `win_rate_vs_bots`).
RANDOM_OPPONENT_NAME = "random"

# Full roster display names, in spec order. Random first (the broken-model floor). The SAME names, in the SAME order,
# as ``eval_roster._EVAL_OPPONENT_SPECS`` (which adds each one's player class and account prefix).
_EVAL_ROSTER = [
    "random", "heuristic", "heuristic2", "staller", "staller_v2",
    "aggressive", "aggressive_v2", "setup_sweep", "setup_sweep_v2",
]


def eval_opponent_names() -> list[str]:
    """Ordered display names of the full eval roster (all bots + Random)."""
    return list(_EVAL_ROSTER)
