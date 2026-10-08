"""``LiveView`` / ``LegalActions`` — the current-board and legality READ-MODELS, as frozen data classes.

The Rust core's reading builds them: its ``present()`` view and ``legal_actions()`` are copied field
for field into these classes by :mod:`agents.battle.core_view` (the prober's core walk,
``main.prober.core_walk``, is the consumer). Every reading rule is the Rust side's
(``src/rust_sim/src/present/``); nothing here computes a value.

The poke-env constructors (``LiveView.from_battle``, ``LivePokemon.from_pokemon``,
``LegalActions.from_battle``) and the weather fold they used are DELETED with the Python battle
layer (T27 P6 slice 6d-2).

The split this module was built around stays: :class:`LiveView` is "what is true NOW" and holds
only primitives — no past-turn state; "what happened" is the core's event record. (``protect_counter``
is the one borderline field: like ``status_counter`` it is the CURRENT value of an in-battle counter
that sets the next Protect's odds.) Opponent fields are reveal-gated: an unrevealed item reads
``None``, an unrevealed move is not in ``moves``, and ``ability`` is ``None`` unless disclosed or
uniquely inferable from the species.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Tuple


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
    # gen3_obs_facts_v1 (backlog E1): a public ``|move|`` line revealed this move (poke-env's own
    # reveal rule, ``Move.seen``). On the opponent's side every listed move is seen; on OURS it
    # is "the opponent has seen this move of ours".
    seen: bool = False


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
    # poke-env's ``protect_counter`` — how many Protect/Detect/Endure moves this mon has landed
    # in a row. A CURRENT-board fact (the present counter value that sets the NEXT Protect's
    # success odds), NOT history: poke-env resets it to 0 on a switch, faint, non-stall move, or
    # a failed roll. Exposed exactly like ``status_counter`` (a current in-battle counter); the
    # obs encoder turns it into the gen3 floored-doubling success probability
    # (``gen3_mechanics.protect_success_probability``) — keeping the gen3 mechanic in the encoder,
    # the same split as ``stats`` feeding the incoming-damage belief.
    protect_counter: int = 0

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

    # ---- what a PROTOCOL line has revealed (gen3_obs_facts_v1, backlog E1) ----
    # Set by the reading's protocol write paths only, never by a ``|request|``: on our side they
    # are "the opponent has seen our item / ability". ``item_public`` with ``item`` None is a known
    # EMPTY hand (a Knock Off / Thief victim, a consumed Berry).
    item_public: bool = False
    ability_public: bool = False

    @property
    def move_ids(self) -> Tuple[str, ...]:
        """Just the revealed move ids, sorted — the terse accessor for id-only call-sites
        (the old ``moves`` shape) now that ``moves`` carries PP per slot."""
        return tuple(m.id for m in self.moves)

    def has_volatile(self, name: str) -> bool:
        """True if this mon currently has the named volatile, matched in Showdown id
        form so ``"leechseed"`` / ``"Leech Seed"`` / ``"LEECH_SEED"`` all work."""
        return name.lower().replace("_", "").replace(" ", "") in self.volatiles


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
    # gen3_obs_facts_v1: this turn's end-of-turn residuals have run (its ``|upkeep|`` was read) —
    # true at a forced replacement after a faint, false at a turn-start or Baton Pass decision.
    # Showdown durations count residuals, so a turns-left read needs it.
    residual_done: bool = False

    def mon(self, side: str, species: str) -> Optional[LivePokemon]:
        return (self.ours if side == "ours" else self.opp).get(species)

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
    # Our active mon's TYPED Hidden Power id ("hiddenpowergrass"), or None when it has no HP.
    # gen3 has no team preview, so the wire request re-keys our HP to the bare "hiddenpower" in
    # `move_slots` (kept wire-truth for the mask/mapper/serialization) — but the Move object keeps
    # the IV-derived typed id, and we ALWAYS know our own HP type. Resolved off the active mon's
    # moveset in `from_battle` and surfaced ONLY for human/forensic LABELS via `display_move_ids`.
    # OUR side only (LegalActions IS the request, always ours) → no opponent-info leak.
    own_hp_typed_id: Optional[str] = None

    @property
    def move_ids(self) -> Tuple[str, ...]:
        """Move ids in request-slot order (slots 0–3). WIRE-TRUTH (bare ``hiddenpower``) — the
        mask/mapper/serialization key on this; for a typed-HP display label use
        :attr:`display_move_ids`."""
        return tuple(m.id for m in self.move_slots)

    @property
    def display_move_ids(self) -> Tuple[str, ...]:
        """:attr:`move_ids` with OUR bare ``hiddenpower`` shown as its TYPED id
        (``hiddenpowergrass``) — for human/forensic LABELS only (the recorder's action labels),
        NEVER the mask/mapper/serialization, which stay on the wire-truth :attr:`move_ids`. A no-op
        unless our active mon carries a Hidden Power (``own_hp_typed_id`` set). We always know our
        own HP type, so a label should show it; an opponent's un-revealed HP stays bare (no leak)."""
        if not self.own_hp_typed_id:
            return self.move_ids
        return tuple(self.own_hp_typed_id if m == "hiddenpower" else m for m in self.move_ids)

    @property
    def switch_species(self) -> Tuple[str, ...]:
        return tuple(s.species for s in self.switches)

    @property
    def switch_slots(self) -> Tuple[int, ...]:
        return tuple(s.slot for s in self.switches)
