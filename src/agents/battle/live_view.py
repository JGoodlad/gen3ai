"""``LiveView`` — the current-board read-model. The single source of truth for
"what is true *right now*", with **no past-turn state**.

Why this exists
---------------
poke-env's ``Pokemon`` is a rich state tracker that mixes *current* facts (HP, status,
boosts, revealed moves) with *temporal* ones (``last_move``, ``last_cant_reason``,
``first_turn``, ``protect_counter`` …). For RL that mixing is a hazard: a consumer can
accidentally read a past-turn field as if it were current, and "what happened last
turn" ends up sourced from two places that can disagree.

We split the two concerns into two clean, separately-fuzzed surfaces:

* **History / "what happened, in order"**  → the event log + :class:`TurnView`.
* **Current board / "what is true now"**    → :class:`LiveView` (this module).

:class:`LiveView` is an immutable snapshot built from the battle on demand
(``battle.live_view()``). It holds **only primitives** — no reference back to the
``Pokemon`` object — so a consumer *physically cannot* reach ``last_move`` or any other
historical field through it. If you need history, you go to the event log. That
constraint is the point: it makes the well-fuzzed API the only path.

Scope (Gen 3 OU singles). Opponent fields are reveal-gated by what poke-env actually
knows: an unrevealed item reads ``None`` (not the ``unknown_item`` sentinel), an
unrevealed move simply isn't in ``moves``, and ``ability`` is ``None`` unless the
protocol disclosed it *or* it is uniquely inferable from the species (e.g. gen-3
Tyranitar ⇒ Sand Stream — public knowledge, not a leak).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Mapping, Optional, Tuple

from poke_env.data.gen_data import GenData

# Items poke-env represents as "not yet known". Treated as None in the live view.
_UNKNOWN_ITEMS = {None, GenData.UNKNOWN_ITEM}


def _enum_name(value) -> Optional[str]:
    """Lowercased ``.name`` of a poke-env enum (Status / Weather / …), or None."""
    return value.name.lower() if value is not None else None


def _id(value) -> Optional[str]:
    """Showdown id form of an enum name: lowercased, no separators
    (``Effect.LEECH_SEED`` -> ``leechseed``) so ids match the move/item/ability
    convention the rest of the codebase uses."""
    if value is None:
        return None
    return value.name.lower().replace("_", "")


@dataclass(frozen=True)
class LivePokemon:
    """Current-board snapshot of one Pokémon. Primitives only — no history, no
    back-reference to the ``Pokemon`` object."""

    species: str
    active: bool
    fainted: bool
    revealed: bool  # has this mon been seen on the field at all?
    hp_fraction: float  # 0.0–1.0
    status: Optional[str]  # 'brn'/'par'/'slp'/'frz'/'psn'/'tox'/'fnt' or None
    types: Tuple[str, ...]  # current types (lowercased)
    moves: Tuple[str, ...]  # REVEALED move ids, sorted (all 4 for our own side)
    item: Optional[str]  # revealed item id, else None (sentinel hidden)
    ability: Optional[str]  # known ability id, else None
    boosts: Mapping[str, int]  # current nonzero stat stages {stat: stage}
    volatiles: Tuple[str, ...]  # current volatile/effect ids, sorted (Sub, Leech Seed…)

    def has_volatile(self, name: str) -> bool:
        """True if this mon currently has the named volatile, matched in Showdown id
        form so ``"leechseed"`` / ``"Leech Seed"`` / ``"LEECH_SEED"`` all work."""
        return name.lower().replace("_", "").replace(" ", "") in self.volatiles

    @classmethod
    def from_pokemon(cls, mon, active: bool) -> "LivePokemon":
        """``active`` is supplied by the caller from poke-env's own
        ``active_pokemon`` accessor (the source of truth) rather than re-derived from
        ``mon.active`` — the latter can stay set on a just-fainted mon."""
        item = mon.item if mon.item not in _UNKNOWN_ITEMS else None
        return cls(
            species=mon.species,
            active=active,
            fainted=bool(mon.fainted),
            revealed=bool(mon.revealed),
            hp_fraction=float(mon.current_hp_fraction),
            status=_enum_name(mon.status),
            types=tuple(_enum_name(t) for t in mon.types if t is not None),
            moves=tuple(sorted(mon.moves.keys())),
            item=item,
            ability=mon.ability,
            boosts={k: v for k, v in mon.boosts.items() if v},
            volatiles=tuple(sorted(_id(e) for e in mon.effects)),
        )


@dataclass(frozen=True)
class LiveSide:
    """Current-board snapshot of one side."""

    team_size: int  # declared roster size (how many mons this side brought)
    active: Optional[LivePokemon]
    mons: Tuple[LivePokemon, ...]  # known mons (our full team; opp's REVEALED mons only)
    side_conditions: Mapping[str, int]  # hazards/screens: {'spikes': 1, 'reflect': …}

    @property
    def revealed_count(self) -> int:
        """How many of this side's mons have been revealed (== len(mons))."""
        return len(self.mons)

    @property
    def remaining(self) -> int:
        """Non-fainted known mons (a lower bound for the opponent until fully revealed)."""
        return sum(1 for m in self.mons if not m.fainted)

    def get(self, species: str) -> Optional[LivePokemon]:
        for m in self.mons:
            if m.species == species:
                return m
        return None


@dataclass(frozen=True)
class LiveView:
    """Immutable snapshot of the whole current board at one instant.

    Build with ``battle.live_view()``. Read ``ours`` / ``opp`` for per-side state and
    the turn-level fields for weather. Carries no temporal information by design.
    """

    turn: int
    weather: Optional[str]
    ours: LiveSide
    opp: LiveSide

    def mon(self, side: str, species: str) -> Optional[LivePokemon]:
        return (self.ours if side == "ours" else self.opp).get(species)

    @classmethod
    def from_battle(cls, battle) -> "LiveView":
        role = battle._player_role
        opp_role = "p2" if role == "p1" else "p1"
        sizes = getattr(battle, "_team_size", {}) or {}

        def side(team: Dict, conditions, declared_role, active_mon) -> LiveSide:
            # Identity-match poke-env's own active accessor so the active slot is
            # faithful even when it momentarily holds a just-fainted mon.
            built = {}
            active = None
            for raw in team.values():
                is_active = raw is active_mon
                lm = LivePokemon.from_pokemon(raw, active=is_active)
                built[id(raw)] = lm
                if is_active:
                    active = lm
            return LiveSide(
                team_size=int(sizes.get(declared_role, len(built))),
                active=active,
                mons=tuple(built.values()),
                side_conditions={
                    _enum_name(k): v for k, v in (conditions or {}).items()
                },
            )

        weather = next((_enum_name(w) for w in battle.weather), None)
        return cls(
            turn=battle.turn,
            weather=weather,
            ours=side(
                battle.team, battle.side_conditions, role, battle.active_pokemon
            ),
            opp=side(
                battle.opponent_team, battle.opponent_side_conditions, opp_role,
                battle.opponent_active_pokemon,
            ),
        )
