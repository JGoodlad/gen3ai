"""Move CATEGORY of a legal action (`gen3_policy_spectrum_categories_v1`, M5 Lane S).

The six categories the owner named (2026-09-29): ``attack`` / ``status`` / ``setup`` / ``hazard`` /
``recovery`` / ``switch``. Every legal action of a banked decision is stamped with one, read from
the core's CHOICE TOKEN for that action index (``switch Dugtrio`` / ``move thunderbolt``) and the
move's dex record through the ``agents.gen3_data`` facade (never live poke-env).

Rules, in order (the first match wins):

- ``switch``    the token is a switch.
- ``attack``    the move deals direct damage: base power > 0, or one of the 0-base-power damaging
                moves in :data:`ZERO_BP_DAMAGING` (fixed / variable damage: Seismic Toss, Counter,
                Return, Hidden Power's bare id, the OHKO moves, …). The facade derives gen-3
                category from base power, so these read STATUS there; the set is pinned against
                Showdown's own ``category`` field by ``categories_test.py``. Struggle is an attack.
- ``setup``     raises the user's own stats (``is_boost``: Swords Dance, Dragon Dance, Calm Mind,
                Belly Drum, …) or Curse (the facade leaves Curse's type-conditional boost out; a
                Ghost-type Curse user is rare in gen-3 OU and is still read as setup here).
- ``hazard``    sets an entry hazard (gen 3: Spikes).
- ``recovery``  restores the user's HP (``is_heal``: Recover, Rest, Wish, Softboiled, Synthesis, …)
                or Pain Split (:data:`RECOVERY_EXTRA`).
- ``status``    every other status move. Its SUBTYPE (:func:`status_subtype`) is stamped beside it:
                ``inflict`` (Toxic, Thunder Wave, Will-O-Wisp, the sleep moves), ``protect``,
                ``phaze`` (Roar / Whirlwind), ``cure`` (Heal Bell / Aromatherapy / Refresh), ``other``
                (Substitute, Baton Pass, Leech Seed, Taunt, Encore, Haze, …).

A move id the facade does not know is a :class:`KeyError` — a banked decision is never stamped with a
guessed category.
"""

from __future__ import annotations

from typing import Optional

CATEGORIES = ("attack", "status", "setup", "hazard", "recovery", "switch")

#: 0-base-power moves that DO deal damage (Showdown ``category`` Physical/Special), pinned by test.
ZERO_BP_DAMAGING = frozenset({
    "bide", "counter", "dragonrage", "endeavor", "fissure", "flail", "frustration", "guillotine",
    "hiddenpower", "horndrill", "lowkick", "magnitude", "mirrorcoat", "nightshade", "present",
    "psywave", "return", "reversal", "seismictoss", "sheercold", "sonicboom", "spitup", "superfang",
})

RECOVERY_EXTRA = frozenset({"painsplit"})
SETUP_EXTRA = frozenset({"curse"})


def _move(move_id: str):
    from agents.gen3_data import moves as M

    d = M.get(move_id)
    if d is None:
        raise KeyError(f"move {move_id!r} is not in the gen-3 dex (agents.gen3_data.moves)")
    return d


def move_category(move_id: str) -> str:
    """The category of a MOVE id (never ``switch``)."""
    if move_id == "struggle":
        return "attack"
    d = _move(move_id)
    if d.base_power > 0 or move_id in ZERO_BP_DAMAGING:
        return "attack"
    if d.is_boost or move_id in SETUP_EXTRA:
        return "setup"
    if d.is_hazard:
        return "hazard"
    if d.is_heal or move_id in RECOVERY_EXTRA:
        return "recovery"
    return "status"


def status_subtype(move_id: str) -> Optional[str]:
    """The finer class of a ``status`` move (``None`` for any other category)."""
    if move_id == "struggle" or move_category(move_id) != "status":
        return None
    d = _move(move_id)
    if d.status_inflicted:
        return "inflict"
    if d.is_protect:
        return "protect"
    if d.is_phaze:
        return "phaze"
    if d.cures_self_status or d.cures_team_status:
        return "cure"
    return "other"


def token_category(token: str) -> str:
    """The category of one core CHOICE TOKEN (``switch <Species>`` / ``move <id>``)."""
    verb, _, arg = token.partition(" ")
    if verb == "switch":
        return "switch"
    if verb == "move":
        return move_category(arg.strip())
    raise ValueError(f"unknown choice token {token!r}")


def token_subtype(token: str) -> Optional[str]:
    verb, _, arg = token.partition(" ")
    return status_subtype(arg.strip()) if verb == "move" else None
