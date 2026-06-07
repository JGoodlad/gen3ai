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
from types import MappingProxyType
from typing import Any, Dict, List, Mapping, Optional, Tuple

from poke_env.data.gen_data import GenData
from poke_env.data.normalize import to_id_str

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
class LiveMove:
    """One revealed move with its current/maximum PP. Primitives only.

    ``id`` is the move's key in poke-env's ``moves`` dict (so ``move_ids`` stays identical
    to the old id-tuple shape, and a consumer could look the move back up). Note this can
    differ from the underlying ``Move.id`` for Hidden Power: in a live battle the server
    request re-keys a typed HP under the bare ``"hiddenpower"`` while the ``Move`` object
    keeps its typed ``.id`` — we follow the dict key, as the original ``LiveView`` did.

    Gen 3 Showdown does not track opponent PP, so an opponent's revealed move reports full
    PP (``current_pp == max_pp``) — faithful to what poke-env knows, not a guess."""

    id: str
    current_pp: int
    max_pp: int


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
    moves: Tuple[LiveMove, ...]  # REVEALED moves w/ PP, sorted by id (all 4 for own side)
    item: Optional[str]  # revealed item id, else None (sentinel hidden)
    ability: Optional[str]  # known ability id, else None
    boosts: Mapping[str, int]  # current nonzero stat stages {stat: stage}
    # current volatiles as {id: counter}. The counter is poke-env's per-effect int —
    # turns-active for action-countable effects (Rage), 0 for the rest. We keep the
    # FULL mapping, not just the id set, so graded "how long / how many" information is
    # preserved for the encoder to normalise rather than flattened to a presence bit.
    # (Most gen3 volatiles aren't action-countable, so their counter is 0; the genuinely
    # graded states — Perish/Stockpile level — arrive via distinct ids perishN/stockpileN
    # and are normalised in gen3_effects.encode_volatiles.) Dict keys preserve the old
    # tuple ergonomics: ``"leechseed" in mon.volatiles`` and ``for v in mon.volatiles``
    # still work.
    volatiles: Mapping[str, int]

    # ---- spread block (mirrors the obs-encoder's spread_known gating) ----
    # ``base_stats`` is a public dex fact, populated for BOTH sides once the species is
    # known. IVs / EVs / nature are private team-building choices: known for our own mons,
    # ``None`` for the opponent. ``spread_known`` is the single gate the encoder reads — it
    # distinguishes "unknown opponent" from "own Pokémon that happens to run 0 EVs".
    base_stats: Mapping[str, int] = field(default_factory=dict)
    ivs: Optional[Tuple[int, ...]] = None  # [HP, Atk, Def, SpA, SpD, Spe] — own side only
    evs: Optional[Tuple[int, ...]] = None  # [HP, Atk, Def, SpA, SpD, Spe] — own side only
    nature: Optional[str] = None  # own side only
    spread_known: bool = False  # True iff ivs/evs/nature are known (our own mons)

    # ---- consumed item / status counter (current-board facts, both sides) ----
    # ``consumed_item`` is revealed knowledge (a popped Berry, a Knock Off victim): the
    # *held* ``item`` is gone (None) but the identity of what was lost is retained, exactly
    # as the obs item encoder reads ``mon.consumed_item``. ``status_counter`` is poke-env's
    # turns-in-status counter (sleep turns / toxic severity) — a current counter, not history.
    consumed_item: Optional[str] = None
    status_counter: int = 0

    # ---- computed stats + integer HP (current-board facts) ----
    # ``stats`` is poke-env's EV/IV/nature-applied stat dict ({atk,def,spa,spd,spe}) — the REAL
    # battle stats, not the dex ``base_stats``. Populated for our own mons (we know the spread);
    # an opponent's are mostly ``None`` until inferable, so ``stats``/``current_hp`` are own-side
    # reliable. ``current_hp``/``max_hp`` are the integer HP poke-env tracks (exact for our mons;
    # an opponent's HP is %-based). The incoming-damage belief reads def/spd/spe + integer HP from
    # here so it never touches the raw ``Pokemon`` — keeping the obs+reward belief on the read-model.
    stats: Mapping[str, int] = field(default_factory=dict)
    current_hp: Optional[int] = None
    max_hp: Optional[int] = None

    @property
    def move_ids(self) -> Tuple[str, ...]:
        """Just the revealed move ids, sorted — the terse accessor for id-only call-sites
        (the old ``moves`` shape) now that ``moves`` carries PP per slot."""
        return tuple(m.id for m in self.moves)

    def has_volatile(self, name: str) -> bool:
        """True if this mon currently has the named volatile, matched in Showdown id
        form so ``"leechseed"`` / ``"Leech Seed"`` / ``"LEECH_SEED"`` all work."""
        return name.lower().replace("_", "").replace(" ", "") in self.volatiles

    @classmethod
    def from_pokemon(cls, mon, active: bool, is_own: bool = False) -> "LivePokemon":
        """``active`` is supplied by the caller from poke-env's own
        ``active_pokemon`` accessor (the source of truth) rather than re-derived from
        ``mon.active`` — the latter can stay set on a just-fainted mon.

        ``is_own`` gates the private spread block: our own mons carry IVs / EVs / nature
        (with ``spread_known=True``); the opponent's are ``None`` (``spread_known=False``).
        Own ``mon.ivs/evs/nature`` are populated by poke-env from the team we declared
        (``backfill_teambuilder_spread`` covers no-preview formats like gen3ou, where the
        protocol never echoes the spread).
        """
        item = mon.item if mon.item not in _UNKNOWN_ITEMS else None
        # Moves keyed + sorted by the moves-dict key (the original LiveView.moves shape),
        # so the slot order is stable across workers and `move_ids` is unchanged. PP comes
        # off the Move value at that key.
        moves = tuple(
            LiveMove(id=mid, current_pp=int(mv.current_pp), max_pp=int(mv.max_pp))
            for mid, mv in sorted(mon.moves.items())
        )
        # Spread: base_stats is public (both sides); ivs/evs/nature are own-side only.
        ivs = tuple(mon.ivs) if (is_own and mon.ivs is not None) else None
        evs = tuple(mon.evs) if (is_own and mon.evs is not None) else None
        nature = mon.nature if is_own else None
        # poke-env stores consumed_item verbatim from the protocol (display form, e.g.
        # "Salac Berry"); normalise to id-form so it matches `item` and the codebase
        # convention ("salacberry"), the same id the obs item encoder works in.
        consumed = mon.consumed_item
        consumed_item = to_id_str(consumed) if consumed else None
        return cls(
            species=mon.species,
            active=active,
            fainted=bool(mon.fainted),
            revealed=bool(mon.revealed),
            hp_fraction=float(mon.current_hp_fraction),
            status=_enum_name(mon.status),
            types=tuple(_enum_name(t) for t in mon.types if t is not None),
            moves=moves,
            item=item,
            ability=mon.ability,
            boosts={k: v for k, v in mon.boosts.items() if v},
            volatiles={_id(e): int(cnt) for e, cnt in mon.effects.items()},
            base_stats=dict(mon.base_stats),
            ivs=ivs,
            evs=evs,
            nature=nature,
            spread_known=bool(is_own),
            consumed_item=consumed_item,
            status_counter=int(getattr(mon, "status_counter", 0) or 0),
            stats=dict(mon.stats) if getattr(mon, "stats", None) else {},
            current_hp=(int(mon.current_hp) if getattr(mon, "current_hp", None) is not None else None),
            max_hp=(int(mon.max_hp) if getattr(mon, "max_hp", None) is not None else None),
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
class LiveWeather:
    """Current weather, folded from the WEATHER event log — NOT guessed from active-mon
    abilities. The protocol tells us the cause directly:

    * ``|-weather|Sandstorm|[from] ability: Sand Stream`` → ability-sourced ⇒ permanent
      (no timer in gen3) — ``is_permanent=True``.
    * ``|-weather|RainDance`` (bare) → move-sourced ⇒ lasts 5 turns —
      ``is_permanent=False``.
    * ``|-weather|Sandstorm|[upkeep]`` → a per-turn tick (continues, doesn't reset).
    * ``|-weather|none`` → cleared.
    """

    weather: Optional[str]  # 'sandstorm'/'raindance'/'sunnyday'/'hail' or None
    is_permanent: bool  # True iff ability-sourced (no countdown)
    turns_active: int  # turns since it was set (0 on the turn it was set)

    @property
    def turns_remaining(self) -> Optional[int]:
        """Turns left for move-set weather (gen3: 5 total), or ``None`` when permanent
        or absent — ``None`` is the honest 'no finite timer' signal, not a guess."""
        if self.weather is None or self.is_permanent:
            return None
        return max(0, 5 - self.turns_active)


_NO_WEATHER = LiveWeather(weather=None, is_permanent=False, turns_active=0)


def _fold_weather(events, now_turn: int) -> LiveWeather:
    """Derive current weather by folding WEATHER events in order. Single source of
    truth = the protocol; we never inspect mon abilities to guess permanence."""
    from agents.battle.battle_event import EventKind

    weather: Optional[str] = None
    is_permanent = False
    start_turn = now_turn
    for e in events:
        if e.kind is not EventKind.WEATHER:
            continue
        wid = e.value.get("weather")
        if wid in (None, "none"):
            weather, is_permanent = None, False
            continue
        if "[upkeep]" in e.raw:
            # a tick of the SAME weather — keep it (and its start turn) running
            if weather is None:  # upkeep seen without a prior set (mid-battle join)
                weather, start_turn = wid, e.turn
            continue
        # a fresh set: the cause decides permanence
        weather = wid
        is_permanent = str(e.value.get("from", "")).startswith("ability")
        start_turn = e.turn
    if weather is None:
        return _NO_WEATHER
    return LiveWeather(weather, is_permanent, max(0, now_turn - start_turn))


@dataclass(frozen=True)
class LiveView:
    """Immutable snapshot of the whole current board at one instant.

    Build with ``battle.live_view()``. Read ``ours`` / ``opp`` for per-side state and
    ``weather`` (a :class:`LiveWeather`) for field state. The meta fields
    (``turn`` / ``battle_tag`` / ``finished`` / ``won`` / ``lost``) mirror poke-env's
    battle-level accessors so a consumer never needs the raw battle for them. No
    temporal per-mon info by design.
    """

    turn: int
    weather: LiveWeather
    ours: LiveSide
    opp: LiveSide
    battle_tag: str = ""
    finished: bool = False
    won: Optional[bool] = None  # None until the battle ends
    lost: Optional[bool] = None  # None until the battle ends

    def mon(self, side: str, species: str) -> Optional[LivePokemon]:
        return (self.ours if side == "ours" else self.opp).get(species)

    @classmethod
    def from_battle(cls, battle) -> "LiveView":
        role = battle._player_role
        opp_role = "p2" if role == "p1" else "p1"
        sizes = getattr(battle, "_team_size", {}) or {}

        def side(team: Dict, conditions, declared_role, active_mon, is_own) -> LiveSide:
            # Identity-match poke-env's own active accessor so the active slot is
            # faithful even when it momentarily holds a just-fainted mon.
            built = {}
            active = None
            for raw in team.values():
                is_active = raw is active_mon
                lm = LivePokemon.from_pokemon(raw, active=is_active, is_own=is_own)
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

        # Weather is folded from our event log (cause-aware), not battle.weather —
        # which can be empty/lossy and carries no permanence info.
        # Prefer the battle's incrementally-folded weather (O(1)); fall back to scanning
        # the event log for a plain Battle that lacks the running state.
        if hasattr(battle, "live_weather"):
            weather = battle.live_weather()
        else:
            weather = _fold_weather(getattr(battle, "events", ()), battle.turn)
        return cls(
            turn=battle.turn,
            weather=weather,
            ours=side(
                battle.team, battle.side_conditions, role,
                battle.active_pokemon, True,
            ),
            opp=side(
                battle.opponent_team, battle.opponent_side_conditions, opp_role,
                battle.opponent_active_pokemon, False,
            ),
            battle_tag=battle.battle_tag,
            finished=bool(battle.finished),
            won=battle.won,
            lost=battle.lost,
        )


# --------------------------------------------------------------------------- #
# LegalActions — the server-authoritative decision surface.                     #
# --------------------------------------------------------------------------- #
# Action legality is NOT something we derive: Showdown's ``|request|`` JSON states it
# outright (per-move pp / disabled, forceSwitch, trapped/maybeTrapped, wait). poke-env
# parses that request verbatim into ``available_moves`` / ``available_switches`` /
# ``force_switch`` / ``trapped`` / ``wait`` (see ``battle.py:parse_request``). LegalActions
# bundles those already-parsed, known-correct fields into one immutable object so the action
# mask / mapper can read legality through the strict boundary instead of poking the battle.


@dataclass(frozen=True)
class LegalMove:
    """One move slot exactly as the server's request listed it. ``disabled`` is the
    server's word (Taunt / Disable / Choice lock / Imprison / no-PP), so the masker need
    not re-derive it."""

    id: str
    current_pp: int
    max_pp: int
    disabled: bool
    target: Optional[str]  # request 'target' field ('normal', 'self', 'allAdjacent', …)


@dataclass(frozen=True)
class LegalSwitch:
    """A bench mon the server says we may switch to, with its team-slot index (0–5) —
    the same ordering the 11-dim action space switches map onto."""

    species: str
    slot: int


@dataclass(frozen=True)
class LegalActions:
    """Immutable, server-authoritative snapshot of what is legal this decision.

    Built with :meth:`from_battle` (or ``battle.strict_view().legal``).

    **Hybrid sourcing — kept deliberately.** ``move_slots`` is *wire-truth*: it is read
    straight from the parsed server request (``last_request['active'][0]['moves']``), so
    which of the 4 move slots are legal — and their pp / disabled / target — is exactly what
    Showdown sent. The legality *flags* (``switches`` / ``force_switch`` / ``trapped`` /
    ``maybe_trapped`` / ``wait`` / ``struggle``), by contrast, are poke-env's *derived*
    interpretation of that request (``available_switches`` / ``force_switch`` /
    ``available_moves`` …). We keep them byte-identical rather than re-deriving from the raw
    request: they are the second poke-env-interpreted seam (alongside Choice→BattleOrder
    serialization) that a future fully-owned ``Player`` would re-derive — out of scope here.

    **Struggle is the flag, never a move slot.** When all PP is gone the server sends a lone
    ``struggle`` entry in the request moves; :meth:`from_battle` filters it out of
    ``move_slots`` and surfaces it ONLY as the :attr:`struggle` flag. That single source of
    truth is what removes the historical "struggle double-enabling" footgun where struggle
    lived in both a move slot and the dedicated bit.
    """

    move_slots: Tuple[LegalMove, ...]  # request 'active'[0]['moves'] order (slots 0–3), struggle excluded
    switches: Tuple[LegalSwitch, ...]  # bench mons we may switch to
    force_switch: bool  # the active mon must be replaced (it fainted / was phazed)
    trapped: bool  # cannot switch (Mean Look / Arena Trap / …)
    maybe_trapped: bool  # opponent *might* be trapping us (unconfirmed)
    wait: bool  # the server wants no action from us right now
    struggle: bool  # all PP gone → the server forces Struggle
    last_request: Optional[Mapping[str, Any]]  # read-only mirror of the raw request

    @property
    def move_ids(self) -> Tuple[str, ...]:
        """Move ids in request-slot order (slots 0–3)."""
        return tuple(m.id for m in self.move_slots)

    @property
    def switch_species(self) -> Tuple[str, ...]:
        return tuple(s.species for s in self.switches)

    @property
    def switch_slots(self) -> Tuple[int, ...]:
        return tuple(s.slot for s in self.switches)

    @classmethod
    def from_battle(cls, battle) -> "LegalActions":
        request = battle.last_request or None
        # During a force-switch the request carries no 'active' block; default to empty.
        active_block = (request.get("active") if request else None) or [{}]
        req_moves = active_block[0].get("moves", []) if active_block else []
        # Normalize struggle OUT of the move slots — it is surfaced only via the
        # `struggle` flag below (single source of truth). The server lists a lone
        # `struggle` entry here when all PP is gone; keeping it in move_slots would
        # re-introduce the two-representations footgun behind the historical
        # "struggle double-enabling" mask bug.
        move_slots = tuple(
            LegalMove(
                id=m.get("id", ""),
                current_pp=int(m.get("pp", 0)),
                max_pp=int(m.get("maxpp", 0)),
                disabled=bool(m.get("disabled", False)),
                target=m.get("target"),
            )
            for m in req_moves
            if m.get("id") != "struggle"
        )

        # Switch slots are indexed against the team ordering the action space uses
        # (list(battle.team.values())), so the mapper's slot→mon translation is faithful.
        available = battle.available_switches
        switches = tuple(
            LegalSwitch(species=mon.species, slot=i)
            for i, mon in enumerate(battle.team.values())
            if mon in available
        )

        struggle = any(m.id == "struggle" for m in battle.available_moves)
        return cls(
            move_slots=move_slots,
            switches=switches,
            force_switch=bool(battle.force_switch),
            trapped=bool(battle.trapped),
            maybe_trapped=bool(battle.maybe_trapped),
            wait=bool(battle.wait),
            struggle=struggle,
            last_request=MappingProxyType(request) if request else None,
        )
