"""Feed ONE side's recorded protocol into a :class:`Gen3Battle` offline — no Player, no loop.

``gen3_offline_feed_v1``. The Rust Core Program's parity harness (``rust_core_parity``) compares
the core's events against ``Gen3Battle``'s reading of the SAME per-side text, and it must do that
without poke-env's async ``Player`` machinery (``designs/endstate/program_rust_core.md`` §3 — "no
players: both paths re-derive from a recorded input log").

🚨 **This is a DISPATCH MIRROR, and it mirrors exactly one function.** A live battle's log is
whatever ``poke_env.player.Player._handle_battle_message`` routes to the battle, line by line:
``request`` → ``parse_request``; ``win`` / ``tie`` → ``won_by`` / ``tied``; ``|error|[Unavailable
choice]`` → the out-of-band ``record_choice_rejected`` hook; ``bigerror`` and the player's own
``MESSAGES_TO_IGNORE`` are dropped; a bare ``|`` and everything else reach ``parse_message``. Every
branch below names the branch it mirrors, and the set of player-level keywords is READ from the
player class rather than copied, so an added ignore is picked up. What the mirror deliberately
omits is the player's side of the exchange (choosing, re-requesting on ``[Invalid choice]``,
``showteam``'s team-sheet reveal, which gen 3 never sends) — none of it writes the battle.

The live-vs-offline equality is gated end to end by ``offline_feed_test.py`` (a real bridge
battle's live log == this feed's log, both viewers).
"""

from __future__ import annotations

import logging
from typing import Iterable, Optional

import orjson
from poke_env.player.player import Player

from agents.battle.gen3_battle import Gen3Battle

_LOG = logging.getLogger("offline_feed")


class OfflineFeedError(RuntimeError):
    """A line the live dispatch would have handled in a way this mirror does not model."""


def player_names(lines: Iterable[str]) -> dict:
    """``{"p1": name, "p2": name}`` from the ``|player|pN|<name>|…`` framing lines."""
    out = {}
    for line in lines:
        if line.startswith("|player|"):
            parts = line.split("|")
            if len(parts) > 3 and parts[3]:
                out[parts[2]] = parts[3]
    return out


def new_battle(viewer: str, names: dict, battle_tag: str = "battle-gen3ou-offline",
               logger: Optional[logging.Logger] = None,
               packed_team: Optional[str] = None) -> Gen3Battle:
    """A fresh battle for ``viewer`` ("p1"/"p2"), named so the ``|player|`` lines resolve the role.

    ``AbstractBattle.parse_message``'s ``player`` branch sets the role by comparing each line's
    name to the battle's username, so the username must be the viewer's name as the stream spells
    it. The role is also set up front, which is what ``bc/log_reader`` does.

    ``packed_team`` mirrors ``Player._create_battle``'s ``battle._teambuilder_team =
    Teambuilder.parse_packed_team(self._current_packed_team)`` — the ONLY source of our own
    IVs / EVs / nature in a no-preview format (``parse_request`` backfills them from it). A feed
    that omits it reads every own spread as ``None``, which no live player ever does.
    """
    b = Gen3Battle(battle_tag, names[viewer], logger or _LOG, gen=3)
    b._player_role = viewer
    if packed_team:
        from poke_env.teambuilder.teambuilder import Teambuilder

        b._teambuilder_team = Teambuilder.parse_packed_team(packed_team)
    return b


def feed_chunk(battle: Gen3Battle, chunk: str) -> None:
    """Route one per-side chunk's lines exactly as ``Player._handle_battle_message`` does."""
    for line in chunk.split("\n"):
        feed_line(battle, line)


def feed_line(battle: Gen3Battle, line: str) -> None:
    sm = line.split("|")
    if len(sm) == 1:                        # `elif len(split_message) == 1:` (teampreview only)
        return
    kw = sm[1]
    if kw == "":                            # `elif split_message[1] == "":` -> parse_message
        battle.parse_message(sm)
    elif kw in Player.MESSAGES_TO_IGNORE:   # the PLAYER-level ignore set
        return
    elif kw == "request":
        if sm[2]:
            battle.parse_request(orjson.loads(sm[2]))
    elif kw == "showteam":
        raise OfflineFeedError("|showteam| is not a gen-3 line; the mirror does not model it")
    elif kw == "win":
        battle.won_by(sm[2])
    elif kw == "tie":
        battle.tied()
    elif kw == "error":
        if sm[2].startswith("[Unavailable choice]"):
            battle.record_choice_rejected(sm)
        # `[Invalid choice]` re-requests (a PLAYER action); anything else is logged. Neither
        # writes the battle.
    elif kw == "bigerror":
        return
    else:
        battle.parse_message(sm)
