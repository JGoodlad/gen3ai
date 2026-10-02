"""THE PER-GAME SEED RULE (``gen3_eval_game_seed_v1``, M5 Lane H) — one eval GAME's randomness as a
pure function of the game.

An eval game is keyed ``(cycle seed, opponent key, game index)`` — the game index counts the
opponent's games in PLAN order (shard 0's games first), so it never depends on which env or worker
played the game, on the thread count, or on how many games ran before it. From the key:

* the TRAINEE's team — one ``yield_team()`` of the trainee builder with its draw RNG re-seeded
  ``random.Random(team_seed(key, TRAINEE))``;
* the OPPONENT's team — likewise, from the builder the opponent pilots (a bot's and a sampled
  sentinel's: the flat pool builder; a greedy sentinel's: the trainee's own builder —
  ``eval_worker._sentinel_tb``; a fixed opponent's: its pinned builder, else the flat pool);
* the BATTLE SEED — four 16-bit words, the core's ``ep_seed`` and the bridge's ``seed=``;
* a scripted bot's streams — re-seeded by the CORE at the game's start from the route seed and the
  battle seed (a ``"streams": "episode"`` route, ``rust_env_opponents.episode_bot_stream_seed``), so
  the Python path sets the SAME ``random.Random`` on the bot's ``_choice_rng`` / ``_protect_rng``;
* a SAMPLED policy opponent's (``--no-eval-sentinel-greedy``) draws — the keyed draw keyed by the
  game (not by env / episode), stream ``KD.STREAM_OPPONENT``.

Both eval paths call these functions — the Rust executor and the Python worker's per-game mode
(``eval_worker`` ``seed_rule = "per_game"``) — so "the same seed set on both paths" is one table.
A live Python eval passes no seed and keeps today's unseeded streams (byte-identical).
"""
from __future__ import annotations

import hashlib
import random
from typing import Any, Dict, List, Tuple

SCHEMA = "gen3_eval_game_seed_v1"

#: Which draw a derived seed is for (the second key component).
TRAINEE, OPPONENT, BATTLE, SAMPLE = "trainee", "opponent", "battle", "sample"

#: The seed every bot route of the eval core is declared with (its streams are per-episode, so this
#: only separates the eval core's bot streams from any other table's).
BOT_ROUTE_SEED = 0x6576616C  # "eval"


def bot_route_seed(bot: str) -> int:
    """The declared seed of eval bot ``bot``'s route in the eval core (its index in the eval roster)."""
    from agents.training.eval_callback import eval_opponent_names

    return BOT_ROUTE_SEED + eval_opponent_names().index(bot)


def bot_stream_seeds(bot: str, words: List[int]) -> Dict[str, int]:
    """The seeds of bot ``bot``'s streams in the game staged with battle seed ``words`` — what the core
    re-seeds at the game's start (``opponents::episode_stream_seed``), for the Python path's bot."""
    from agents.training.rust_env_opponents import episode_bot_stream_seed

    s = bot_route_seed(bot)
    return {name: episode_bot_stream_seed(s, words, k) for k, name in enumerate(("choice", "protect", "bait"))}


def _h(*parts: Any) -> int:
    d = hashlib.blake2b(":".join(str(p) for p in parts).encode(), digest_size=8).digest()
    return int.from_bytes(d, "big") & ((1 << 62) - 1)


def game_key(cycle_seed: int, item_key: str, game: int) -> Tuple[int, str, int]:
    return (int(cycle_seed), str(item_key), int(game))


def derived_seed(key: Tuple[int, str, int], what: str) -> int:
    """A 62-bit seed for draw ``what`` of the game ``key``."""
    return _h(SCHEMA, *key, what)


def pair_game(cycle_seed: int, item_key: str, game: int, mirrored: bool) -> Tuple[Tuple[int, str, int], bool]:
    """``(key, swapped)`` for game ``game`` of an item — THE MIRRORED-PAIR RULE (``gen3_mirrored_pairs_v1``).

    Unmirrored: the game's own key, never swapped (exactly :func:`game_key`). MIRRORED: games ``2k`` and
    ``2k+1`` are ONE team pairing played from both sides — both take the key of game ``2k`` (so both draw
    the SAME two teams, the SAME battle seed, the SAME bot streams and the SAME sampled-opponent seed), and
    game ``2k+1`` is ``swapped``: the trainee pilots the team the opponent drew and the opponent pilots the
    trainee's. The trainee keeps its seat (p1) in both games; only the TEAMS change hands."""
    if not mirrored:
        return game_key(cycle_seed, item_key, game), False
    first = int(game) - int(game) % 2
    return game_key(cycle_seed, item_key, first), bool(int(game) % 2)


def battle_seed(key: Tuple[int, str, int]) -> List[int]:
    """The game's four 16-bit battle-seed words (never all zero)."""
    h = derived_seed(key, BATTLE)
    w = [(h >> s) & 0xFFFF for s in (0, 16, 32, 48)]
    if not any(w):
        w[0] = 1
    return w


def draw_team(builder: Any, key: Tuple[int, str, int], side: str) -> str:
    """One ``yield_team()`` of ``builder`` with its draw RNG re-seeded for ``(key, side)``. The builder
    must carry no per-yield state (``team_pfsp`` off, block episodes 1 — eval's own builders)."""
    if int(getattr(builder, "_block_episodes", 1) or 1) > 1:
        raise ValueError("draw_team: a block-episodes builder carries state across yields; eval builders do not")
    if getattr(builder, "_team_pfsp", "off") != "off":
        raise ValueError("draw_team: a team-PFSP builder carries state across yields; eval builders do not")
    builder._rng = random.Random(derived_seed(key, side))
    return builder.yield_team()


def sample_seed(key: Tuple[int, str, int]) -> int:
    """The keyed-draw seed of a SAMPLED policy opponent's decisions in this game."""
    return derived_seed(key, SAMPLE) & 0x7FFFFFFFFFFF
