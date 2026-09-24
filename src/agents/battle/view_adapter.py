"""``view_adapter`` — build the read-models from the port's ONE-SIDED VIEW payload
(``gen3_one_sided_view_v1``).

The materializer's per-successor cost today is a **re-derivation**: the Rust engine holds the
successor's board, renders it as one-sided protocol text, and Python replays that text through
poke-env's state tracker to get the board back. ``src/rust_sim/src/view.rs`` emits the board
directly instead; this module turns that payload into the SAME
:class:`~agents.battle.live_view.LiveView` / :class:`~agents.battle.live_view.LegalActions`
objects ``battle.strict_view()`` hands out, so **no sub-encoder changes and the strict boundary
is untouched** — ``strict_api_lock_test.py`` sees the same two classes it always did.

The contract (which field comes from where, and what is deferred) is owned by
``designs/rust_sim/one_sided_view.md``. The split it rests on:

* the **payload** carries SIM FACTS plus REVEAL FLAGS — the port's own board, projected onto
  what the side has been told;
* **this module** applies poke-env's PRESENTATION rules on top, because they are properties of
  the library being mirrored rather than of the simulator. There are three, each named at its
  site below: the single-possible-ability inference, the all-``None`` ``stats`` dict an
  opponent carries, and the request→``LegalActions`` derivation.

🚨 **This is a TRANSPORT, not a second encoder.** Nothing here may compute an observation
value, normalise a stat, or decide a legality — every one of those already has exactly one home,
and a second one is how two paths come to disagree. If a field cannot be built faithfully it is
listed as a DEFERRAL in the contract doc and the differential gate reports it, rather than being
approximated here.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from agents.battle.live_view import (LegalActions, LegalMove, LegalSwitch, LiveMove,
                                     LivePokemon, LiveSide, LiveView, LiveWeather)

#: poke-env's per-mon ``_stats`` dict before any request has filled it — the shape an
#: OPPONENT's stays in for the whole battle (``_update_from_request`` is our side only). The
#: payload spells that state as ``"stats": null``; ``LiveView`` then holds a dict of ``None``s,
#: NOT an empty dict, because ``dict(mon.stats) if mon.stats else {}`` sees a truthy dict.
_EMPTY_STATS: Dict[str, Optional[int]] = {
    "hp": None, "atk": None, "def": None, "spa": None, "spd": None, "spe": None,
}

_STAT_ORDER = ("hp", "atk", "def", "spa", "spd", "spe")


def _possible_abilities(species: str) -> List[str]:
    """poke-env's own view of a species' abilities, in poke-env's own id form.

    Read from ``GenData`` rather than the ``gen3_data`` facade ON PURPOSE: the rule below
    mirrors ``Pokemon._update_from_pokedex``, so it has to consult the same table that method
    does. A disagreement between the two dexes would be a finding about the dexes, and reading
    the facade here would hide it inside this adapter instead."""
    from poke_env.data.gen_data import GenData
    from poke_env.data.normalize import to_id_str

    entry = GenData.from_gen(3).pokedex.get(species)
    if not entry:
        return []
    return [to_id_str(a) for a in entry.get("abilities", {}).values()]


def _ability(species: str, revealed: Optional[str], events=(),
             base_slot: Optional[str] = None) -> Optional[str]:
    """poke-env PRESENTATION RULE 1 — the ability the tracker would be holding.

    Three poke-env behaviours, replayed here because all three are the library's and none is the
    simulator's:

    1. **The single-possible-ability inference.** ``Pokemon._update_from_pokedex`` sets
       ``_ability`` outright when the species has exactly ONE possible ability and ``gen >= 3``,
       so a gen-3 Tyranitar reads ``sandstream`` before anything disclosed it — public knowledge
       (the dex says so), not a leak.
    2. **TWO SLOTS, and the getter prefers the temporary one.** The ``ability`` setter writes
       ``_ability`` only while it is ``None`` and ``_temporary_ability`` thereafter — so whether
       an announcement is "the" ability or an overlay depends on whether rule 1 already fired.
    3. **A switch-out clears the temporary slot.** A Porygon2 that Traced Magnet Pull reads
       ``magnetpull`` while it is out and reverts to ``trace`` the moment it pivots — measured, and
       only on the THIRD fresh seed.

    ``revealed`` is the OWN side's engine value (our own ``|request|`` states it); ``events`` is
    the watched side's announcement list, an empty ``id`` marking a switch-out or a faint (both
    clear the temporary slot). ``base_slot`` seeds the base slot instead of the inference — our
    OWN mon's, which poke-env fills from the request's ``baseAbility``; it is how a watched
    move's Pressure is decided against our own mon's ability AT USE TIME."""
    if revealed:
        return revealed
    if base_slot is not None:
        base: Optional[str] = base_slot
    else:
        poss = _possible_abilities(species)
        base = poss[0] if len(poss) == 1 else None
    temp: Optional[str] = None

    def assign(value: str) -> None:
        nonlocal base, temp
        if base is None:
            base = value
        else:
            temp = value

    for ev in events or ():
        eid = str(ev.get("id") or "")
        if not eid:
            temp = None                      # `Pokemon.switch_out`
            continue
        if ev.get("if_unknown"):
            # The `-activate|X|ability: A` handler: `if holder_mon.ability is None:
            # holder_mon.ability = A` — the getter (temporary first), inference included.
            if (temp if temp is not None else base) is None:
                assign(eid)
            continue
        if ev.get("trace") and (temp or base) != "trace":
            # The `-ability` handler's Trace special case: it un-sets whichever slot is occupied,
            # assigns "trace", and THEN assigns the copied ability.
            if temp is not None:
                temp = None
            elif base is not None:
                base = None
            assign("trace")
        assign(eid)
    return temp if temp is not None else base


def _volatiles(announced) -> Dict[str, int]:
    """poke-env PRESENTATION RULE 2 — the announced volatiles → ``LiveView.volatiles``.

    ``LivePokemon.volatiles`` is ``{_id(effect): counter}`` over ``Pokemon.effects``, which is a
    FOLD over ``|-start|`` / ``|-end|`` / ``|-activate|`` / ``|-singleturn|`` / ``|-singlemove|``
    (plus the silent Minimize), cleared on switch-out and faint. The port folds the same lines and
    sends each volatile's raw name with its ANNOUNCEMENT HISTORY — the ``|turn|`` tick of every
    announcement and the tick now — and this REPLAYS poke-env's lifecycle over it (reading rule
    V4): ``start_effect`` creates at 0 or counts an ``is_action_countable`` restart,
    ``end_turn`` counts an ``is_turn_countable`` one and deletes an ``ends_on_turn`` one — so a
    Protect re-announced a turn after its first is a fresh effect. A Baton-Passed history
    (``bp_carried``) survives only for ``BATON_PASS_COPIED_EFFECTS`` (``apply_baton_pass``).
    The poke-env-owned rules applied here are properties of the ``Effect`` enum rather than of
    the simulator:

    * ``Effect.from_showdown_message`` — the name normalisation (strips ``move:`` / ``ability:``
      prefixes, maps aliases onto the enum), then the id form ``_id`` uses
      (``Effect.LEECH_SEED`` → ``leechseed``);
    * ``ends_on_turn`` — ``Pokemon.end_turn`` DELETES these at the next ``|turn|``. Focus Punch is
      the gen-3 one that bites: without this rule a Snorlax carried ``focuspunch`` forever while
      poke-env had dropped it the same turn;
    * the COUNTER — ``end_turn`` increments an ``is_turn_countable`` effect once per turn (Taunt,
      Disable, Encore, the partial traps), ``start_effect`` increments an ``is_action_countable``
      one per re-announcement (Rage, Stockpile), and everything else stays 0.
    """
    from poke_env.battle.effect import BATON_PASS_COPIED_EFFECTS, Effect

    out: Dict[str, int] = {}
    for v in announced:
        if isinstance(v, str):          # tolerate the bare-name form: one start, now
            name, starts, now, carried = v, [0], 0, 0
        else:
            name = str(v["name"])
            starts = [int(t) for t in v.get("starts", ())]
            now = int(v.get("now", starts[-1] if starts else 0))
            carried = int(v.get("bp_carried", 0))
        effect = Effect.from_showdown_message(name)
        if carried and effect not in BATON_PASS_COPIED_EFFECTS:
            starts = starts[carried:]      # the pass did not copy it; later announcements stand
        present, counter, last = False, 0, None

        def ticks(n: int) -> None:
            nonlocal present, counter
            for _ in range(n):
                if not present:
                    return
                if effect.is_turn_countable:
                    counter += 1
                if effect.ends_on_turn:
                    present = False

        for s in starts:
            if last is not None:
                ticks(s - last)
            if not present:
                present, counter = True, 0
            elif effect.is_action_countable:
                counter += 1
            last = s
        if last is not None:
            ticks(now - last)
        if present:
            out[effect.name.lower().replace("_", "")] = counter
    return out


class _AbilityAt:
    """poke-env's view of a mon's ability AT A PAST MOMENT — the mon's ability events replayed up
    to the index a sighting recorded (reading rule V3: ``_pressure_on`` reads ``target.ability``
    when the move is USED). Rows are looked up per SIDE, so a mirror match cannot cross them."""

    def __init__(self, payload: Mapping[str, Any]) -> None:
        self._rows = {
            own: {str(r["species"]): r for r in (payload.get(which) or {}).get("mons", ())}
            for own, which in ((True, "ours"), (False, "opp"))}

    def __call__(self, species: str, own: bool, k: int) -> Optional[str]:
        row = self._rows[own].get(species)
        if row is None:
            return None
        events = list(row.get("ability_events") or ())[:k]
        return _ability(species, None, events,
                        base_slot=row.get("base_ability") if own else None)


def _move(m: Mapping[str, Any], ability_at: "Optional[_AbilityAt]" = None) -> LiveMove:
    """One move slot. An OWN mon's row carries the true ``current_pp``; a WATCHED mon's carries
    the SIGHTING COUNT instead, because poke-env's ``Move.current_pp`` for an opponent is a
    counter it keeps itself (``Pokemon.moved`` calls ``move.use()`` for either side), not
    anything the wire states. The port must not send the engine's PP there — it is privileged,
    and it moves on decrements the watcher never saw.

    poke-env decrements by TWO when the TARGET has Pressure (``_pressure_on``), so the payload
    carries the sightings keyed by target species and the extra decrement is applied here — the
    ability that decides it is the read-model's, inference included (a gen-3 Zapdos reads
    ``pressure`` before anything disclosed it, which is exactly the case a sim-side rule would
    have missed).

    ``_pressure_on`` also requires the target to be un-fainted, but AT THE TIME of the use — and a
    mon cannot be targeted while fainted, so the condition is always true for a sighting that
    happened. Re-checking it at READ time is the wrong question and was measured wrong: every
    late-game board whose Pressure holder had since died lost the whole correction (30 divergences
    over 80 comparisons, all of them foe-targeting moves and none of the `self`/`foeSide` ones)."""
    if "uses" not in m:
        return LiveMove(id=str(m["id"]), current_pp=int(m["current_pp"]), max_pp=int(m["max_pp"]))
    max_pp = int(m["max_pp"])
    dec = int(m["uses"])
    # Every sighting that may cost ONE MORE PP: `_pressure_on(user, move, presumed_target)`,
    # decided with the ability poke-env held AT USE TIME. A plain sighting's base PP is already
    # in `uses`; a CALLED one (`[from]move: Sleep Talk` / Metronome) costs the caller only this
    # extra — `Move.use(pressure, overridden=True)` = `1 + pressure − 1`.
    for s in m.get("sightings") or ():
        if ability_at is None:
            break
        ttype = _target_type(str(s["mv"]))
        if not _pressure_target_type(str(s["mv"]), ttype):
            continue
        # `Battle._get_target_mon`: an `all`-target move ignores the named target and takes the
        # OTHER side's active — the `d` default, which is also what a line with no target named.
        if ttype == "all":
            species, own, k = str(s["d"]), True, int(s["d_k"])
        else:
            species, own, k = str(s["t"]), bool(s["t_own"]), int(s["t_k"])
        if species and ability_at(species, own, k) == "pressure":
            dec += int(s["n"])
    return LiveMove(id=str(m["id"]), current_pp=max(max_pp - dec, 0), max_pp=max_pp)


#: The ``target`` kinds ``_pressure_on`` treats as foe-directed. Declared rather than inlined so
#: the list is comparable with poke-env's by eye.
_PRESSURE_TARGETS = frozenset(
    {"all", "allAdjacent", "allAdjacentFoes", "any", "normal", "randomNormal", "scripted"})


def _move_entry(move_id: str) -> Optional[Mapping[str, Any]]:
    from poke_env.battle.move import Move
    from poke_env.data.gen_data import GenData

    return GenData.from_gen(3).moves.get(Move.retrieve_id(move_id))


def _target_type(move_id: str) -> Optional[str]:
    entry = _move_entry(move_id)
    return entry["target"] if entry else None


def _pressure_target_type(move_id: str, ttype: Optional[str]) -> bool:
    """The second half of ``_pressure_on``: a foe-directed target type, or ``mustpressure``."""
    entry = _move_entry(move_id)
    if not entry:
        return False
    return ttype in _PRESSURE_TARGETS or "mustpressure" in entry.get("flags", {})


def _targets_a_foe(move_id: str) -> bool:
    return _pressure_target_type(move_id, _target_type(move_id))


def _mon(row: Mapping[str, Any], own: bool, ability_at: "Optional[_AbilityAt]" = None) -> LivePokemon:
    species = str(row["species"])
    moves = tuple(sorted((_move(m, ability_at) for m in row.get("moves", [])), key=lambda m: m.id))
    raw_stats = row.get("stats")
    stats: Dict[str, Optional[int]] = (
        {k: int(v) for k, v in raw_stats.items()} if raw_stats else dict(_EMPTY_STATS)
    )
    # A fainted mon's stages are the sim's: none (its faint `clearVolatile` zeroed them). This was
    # reading rule V10 — reproducing poke-env's keep-until-switch-out — until the fork was fixed
    # (`gen3_pe_reading_fixes_v1`, PE-V10); the payload's `boosts` is now read as it stands.
    boosts = row.get("boosts") or {}
    ivs = row.get("ivs")
    evs = row.get("evs")
    return LivePokemon(
        species=species,
        active=bool(row["active"]),
        fainted=bool(row["fainted"]),
        revealed=bool(row["revealed"]),
        hp_fraction=float(row["hp_fraction"]),
        status=row.get("status"),
        types=tuple(str(t) for t in row.get("types", ())),
        moves=moves,
        item=row.get("item"),
        ability=_ability(species, row.get("ability"), row.get("ability_events")),
        boosts={k: int(v) for k, v in boosts.items()},
        volatiles=_volatiles(row.get("volatiles") or ()),
        base_stats={k: int(v) for k, v in (row.get("base_stats") or {}).items()},
        ivs=tuple(int(v) for v in ivs) if ivs is not None else None,
        evs=tuple(int(v) for v in evs) if evs is not None else None,
        nature=row.get("nature"),
        spread_known=bool(row.get("spread_known", False)),
        consumed_item=row.get("consumed_item"),
        status_counter=int(row.get("status_counter", 0) or 0),
        protect_counter=int(row.get("protect_counter", 0) or 0),
        stats=stats,
        current_hp=(int(row["current_hp"]) if row.get("current_hp") is not None else None),
        max_hp=(int(row["max_hp"]) if row.get("max_hp") is not None else None),
    )


def _side(block: Mapping[str, Any], own: bool,
          ability_at: "Optional[_AbilityAt]" = None) -> LiveSide:
    mons = tuple(_mon(r, own, ability_at) for r in block.get("mons", []))
    active = next((m for m in mons if m.active), None)
    return LiveSide(
        team_size=int(block.get("team_size", len(mons))),
        active=active,
        mons=mons,
        side_conditions={k: int(v) for k, v in (block.get("side_conditions") or {}).items()},
    )


def live_view_from_payload(payload: Mapping[str, Any], *, battle_tag: str = "") -> LiveView:
    """The current board, as the payload's side observed it.

    ``battle_tag`` is the caller's — it is a poke-env room name with no counterpart in the
    simulator, and the obs encoder reads it only as the assembler's cache identity."""
    w = payload.get("weather") or {}
    weather = LiveWeather(
        weather=w.get("weather"),
        is_permanent=bool(w.get("is_permanent", False)),
        turns_active=int(w.get("turns_active", 0) or 0),
    )
    # Pressure needs the ABILITY of each sighting's target AS POKE-ENV HELD IT AT USE TIME
    # (inference, a Trace overlay, a switch-out clearing it) — resolved per side, per sighting.
    ability_at = _AbilityAt(payload)
    return LiveView(
        turn=int(payload["turn"]),
        weather=weather,
        ours=_side(payload["ours"], True, ability_at),
        opp=_side(payload["opp"], False, ability_at),
        battle_tag=battle_tag,
        finished=bool(payload.get("finished", False)),
        won=payload.get("won"),
        lost=payload.get("lost"),
    )


def legal_actions_from_payload(payload: Mapping[str, Any],
                               live: "Optional[LiveView]" = None) -> Optional[LegalActions]:
    """poke-env PRESENTATION RULE 3 — the request → :class:`LegalActions` derivation.

    ``LegalActions.from_battle`` takes ``move_slots`` from the raw request (wire-truth) and the
    six legality FLAGS from poke-env's parse of the same request. That parse is four lines of
    ``Battle.parse_request`` and is reproduced here against the identical bytes:

    ==================  ========================================================
    ``wait``            ``request["wait"]``, truthy
    ``force_switch``    ``request["forceSwitch"][0]``
    ``trapped``         ``request["active"][0]["trapped"]``, truthy
    ``maybe_trapped``   ``request["active"][0]["maybeTrapped"]``, truthy
    ``switches``        when NOT trapped: every roster entry that is neither
                        ``active`` nor fainted, at its index in the STABLE team
                        order. 🚨 That is **not** the request's own order:
                        Showdown floats the ACTIVE mon to ``side.pokemon[0]``
                        on every request, while poke-env's ``battle.team`` is a
                        dict fixed at the FIRST request and never reordered —
                        and ``LegalSwitch.slot`` is an index into the latter,
                        which the 11-dim action space maps onto. The stable
                        order is the payload's own ``ours.mons``
    ``struggle``        the lone ``struggle`` entry the server sends when all PP
                        is gone (and it is filtered OUT of ``move_slots``)
    ==================  ========================================================

    Returns ``None`` when the side has no outstanding request (the battle ended, or this side
    is not being asked) — the caller must treat that as "no decision here", never as an empty
    legality (an all-zero mask is a DEFERRED decision in the live player, not a forfeit)."""
    request = payload.get("request")
    if not request:
        return None
    active_block = (request.get("active") or [{}])
    active = active_block[0] if active_block else {}
    req_moves = active.get("moves", []) or []
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
    trapped = bool(active.get("trapped"))
    roster = (request.get("side") or {}).get("pokemon") or []
    stable = [m.species for m in live.ours.mons] if live is not None else []
    switches: Tuple[LegalSwitch, ...] = ()
    if not trapped:
        rows = []
        for p in roster:
            if p.get("active") or "fnt" in str(p.get("condition", "")):
                continue
            sp = _roster_species(p)
            rows.append(LegalSwitch(species=sp, slot=stable.index(sp) if sp in stable else -1))
        switches = tuple(sorted(rows, key=lambda r: r.slot))
    return LegalActions(
        move_slots=move_slots,
        switches=switches,
        force_switch=bool((request.get("forceSwitch") or [False])[0]),
        trapped=trapped,
        maybe_trapped=bool(active.get("maybeTrapped")),
        wait=bool(request.get("wait")),
        struggle=any(m.get("id") == "struggle" for m in req_moves),
        last_request=MappingProxyType(dict(request)),
        own_hp_typed_id=_own_hp_typed_id(roster),
    )


def _roster_species(entry: Mapping[str, Any]) -> str:
    """The roster row's species in poke-env's id form. poke-env takes it from the ``details``
    string (``"Blissey, F"``), NOT from the ident (which is the NICKNAME)."""
    from poke_env.data.normalize import to_id_str

    details = str(entry.get("details", ""))
    return to_id_str(details.split(",")[0].strip())


def _own_hp_typed_id(roster) -> Optional[str]:
    """OUR active mon's TYPED Hidden Power id, or ``None``.

    ``LegalActions._own_hp_typed_id`` reads it off the live ``Move`` objects, which keep the
    IV-derived typed id even though the request's ``active`` block re-keys the slot bare. The
    port's request ROSTER already carries the typed id (``gen3_own_typed_hp_request_roster_v1``),
    so it is read from there. Ours only — the roster IS our side."""
    for entry in roster:
        if entry.get("active"):
            typed = [m for m in entry.get("moves", []) if str(m).startswith("hiddenpower")]
            return typed[0] if len(typed) == 1 else None
    return None


# ---------------------------------------------------------------------------
# The encoder-facing shim
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ViewMon:
    """What the observation encoder still reads off a poke-env ``Pokemon`` rather than off the
    read-model — supplied here from the SAME :class:`LivePokemon`, so no value has a second
    source.

    **That this class needs more than `(species, active)` is the headline FINDING of
    `gen3_one_sided_view_v1`, and it is about the ENCODERS, not about the port.** Two distinct
    gaps, both recorded in ``designs/rust_sim/one_sided_view.md``:

    * **D1 — the SLOT comes from the battle, the VALUES come from the view.**
      ``state_encoder.encode`` walks ``base.get_team_list(battle, …)`` (``battle.team`` /
      ``battle.opponent_team``) to decide which 122-dim slot each mon occupies, and only then
      looks it up in the ``LiveView`` (``live.ours.get(species)``). So an ordered identity list
      is required whatever else happens.
    * **D2 — three per-mon sub-encoders never took a ``live_mon`` at all.**
      ``items.ItemsEncoder.encode`` reads ``mon.item`` / ``mon.consumed_item``,
      ``abilities.AbilitiesEncoder.encode`` reads ``mon.ability``, and
      ``types.TypeEncoder.encode`` reads ``mon.type_1`` / ``mon.type_2`` — unconditionally, with
      no read-model branch, even though ``LivePokemon`` carries every one of those fields. The
      rest of ``pokemon.encode`` is already ``if live_mon is not None:`` throughout. Migrating
      those three onto ``live_mon`` is the right fix and is a value-neutral refactor that owes
      the obs-build benchmark; until then they are fed from here.

    Every field below is a DIRECT copy of its ``LivePokemon`` counterpart. Nothing is derived,
    defaulted or inferred — that is the line between a transport and a second encoder."""

    species: str
    active: bool
    fainted: bool
    #: ``LivePokemon.item`` — already sentinel-gated (an unrevealed opponent item is ``None``,
    #: never poke-env's ``unknownitem``), which ``ItemsEncoder`` treats identically.
    item: Optional[str]
    consumed_item: Optional[str]
    #: ``LivePokemon.ability`` — ``None`` when undisclosed, which is what sends
    #: ``AbilitiesEncoder`` to its Smogon priors, exactly as the raw path does.
    ability: Optional[str]
    #: poke-env ``PokemonType`` members, because ``TypeEncoder`` sorts on ``.name``. Rebuilt from
    #: ``LivePokemon.types`` (lowercased names) through ``PokemonType.from_name``, which is the
    #: inverse of the ``_enum_name`` the read-model applied.
    type_1: Any = None
    type_2: Any = None
    #: ``{moves-dict key: poke-env Move}`` — what ``base.get_sorted_moves`` walks.
    #: ``MovesEncoder`` is the fourth encoder in the D2 list: it takes no ``live_mon`` and reads
    #: ``mon.moves.values()`` for each slot's id, category and PP. The ``Move`` objects are
    #: poke-env's own class built from the id the payload states, with ``current_pp`` set to the
    #: value the read-model already agreed on — no property is re-derived here.
    #: 🚨 The KEY and the ``Move.id`` differ for a typed Hidden Power (the wire re-keys it bare),
    #: and the two are sorted on by DIFFERENT consumers, so the payload states both.
    moves: Any = None

    @classmethod
    def of(cls, m: LivePokemon, row: Optional[Mapping[str, Any]] = None) -> "ViewMon":
        from poke_env.battle.move import Move
        from poke_env.battle.pokemon_type import PokemonType

        types = [PokemonType.from_name(t) for t in m.types]
        real_id = {str(r["id"]): str(r.get("move_id") or r["id"]) for r in (row or {}).get("moves", ())}
        moves = {}
        for lm in m.moves:
            mid = real_id.get(lm.id, lm.id)
            mv = Move(move_id=Move.retrieve_id(mid), raw_id=mid, gen=3)
            mv._current_pp = lm.current_pp
            moves[lm.id] = mv
        return cls(
            species=m.species,
            active=m.active,
            fainted=m.fainted,
            item=m.item,
            consumed_item=m.consumed_item,
            ability=m.ability,
            type_1=types[0] if types else None,
            type_2=types[1] if len(types) > 1 else None,
            moves=moves,
        )


class _ViewStrict:
    """What ``battle.strict_view()`` returns, narrowed to the two members ``encode`` reads."""

    __slots__ = ("live", "legal")

    def __init__(self, live: LiveView, legal: Optional[LegalActions]) -> None:
        self.live = live
        self.legal = legal

    def __getattr__(self, name: str):  # pragma: no cover - diagnostic path
        raise AttributeError(
            f"the one-sided VIEW path exposes only `.live` and `.legal`; `{name}` would need a "
            f"poke-env battle. If the encoder started reading it, the view payload must carry "
            f"it — see designs/rust_sim/one_sided_view.md"
        )


def _rows_by_species(payload: Optional[Mapping[str, Any]],
                     which: str) -> Dict[str, Mapping[str, Any]]:
    """``{species: payload row}`` for ONE side, so :meth:`ViewMon.of` can read the per-move
    ``move_id`` the read-model does not carry (``LiveMove.id`` is the dict KEY).

    🚨 Keyed per SIDE on purpose. A gen3ou MIRROR is common (both teams running Metagross), and
    one dict over both sides let the opponent's row overwrite ours — so our own Metagross took the
    opponent's bare ``hiddenpower`` instead of its typed ``hiddenpowerfire``, and its move-slot id
    encoded as dex num 237 instead of 365."""
    return {str(r["species"]): r
            for r in ((payload or {}).get(which) or {}).get("mons", ())}


class ViewBattle:
    """A stand-in for the poke-env ``Battle`` that ``Gen3ObservationEncoder.encode`` accepts.

    ``encode`` touches the battle object in exactly four places — ``strict_view()``,
    ``live_view()``, ``opponent_active_pokemon`` and ``get_team_list``'s two team dicts — and
    every per-mon VALUE it uses comes from the ``LiveView`` it built from the first. This class
    supplies those four and nothing else, so an encoder that starts reading a fifth fails LOUDLY
    here instead of silently reading a default.

    🚨 **Team ORDER is load-bearing and is the payload's, not this class's.** ``team`` is roster
    order; ``opponent_team`` is REVEAL order, which is what poke-env's dict holds and therefore
    which 122-dim slot each opposing mon occupies. The port emits the rows already ordered.
    """

    __slots__ = ("_live", "_legal", "team", "opponent_team", "events")


    def __init__(self, live: LiveView, legal: Optional[LegalActions] = None,
                 payload: Optional[Mapping[str, Any]] = None,
                 events: Sequence[Any] = ()) -> None:
        self._live = live
        self._legal = legal
        #: The successor's WHOLE-BATTLE event log — the root's, plus the ply folded by
        #: :class:`~agents.battle.event_fold.ViewEventFolder` (`gen3_view_successor_v1`). TWO
        #: sub-encoders fold it and they were `gen3_one_sided_view_v1`'s last two obs
        #: DEFERRALS: ``reactive.encode`` → ``wish_belief.build_wish_pending`` (D3, the pending-
        #: Wish pair) and ``state_encoder.encode`` → ``sleep_belief.build_sleep_sources`` (D4,
        #: the 3-dim sleep-wake belief). Both read ``battle.events`` and ``battle.turn`` and
        #: nothing else, so supplying the log closes both — the payload was never the problem,
        #: the missing log was. Empty ⇒ both fold to their no-information answer, which is what
        #: a caller with no event log gets and is why this defaults rather than raising.
        self.events: Sequence[Any] = events
        ours = _rows_by_species(payload, "ours")
        opp = _rows_by_species(payload, "opp")
        self.team = {m.species: ViewMon.of(m, ours.get(m.species)) for m in live.ours.mons}
        self.opponent_team = {
            m.species: ViewMon.of(m, opp.get(m.species)) for m in live.opp.mons}

    @property
    def turn(self) -> int:
        """The board's turn — read by ``build_wish_pending`` beside ``events``."""
        return int(self._live.turn)

    @property
    def opponent_active_pokemon(self) -> Optional[ViewMon]:
        return next((m for m in self.opponent_team.values() if m.active), None)

    @property
    def active_pokemon(self) -> "Optional[ViewMon]":
        """D3 — ``reactive._request_slot_moves`` resolves each REQUEST slot to a live poke-env
        ``Move`` off ``battle.active_pokemon.moves``, because the reactive block's
        ``active_req_moves`` needs the move's dex num and its RESOLVED type (and, for our own
        Hidden Power, the typed variant the wire hides). ``legal.move_slots`` supplies the ids
        and ``legal.own_hp_typed_id`` the typed HP, so the moveset is rebuilt from the read-model
        — using poke-env's OWN ``Move`` class, which is what poke-env itself would construct from
        the same id. Nothing here re-derives a move property."""
        return next((m for m in self.team.values() if m.active), None)

    def strict_view(self) -> _ViewStrict:
        return _ViewStrict(self._live, self._legal)

    def live_view(self) -> LiveView:
        return self._live

    def __getattr__(self, name: str):  # pragma: no cover - diagnostic path
        raise AttributeError(
            f"ViewBattle has no `{name}`: the one-sided VIEW path carries a projected board, not "
            f"a poke-env battle. See designs/rust_sim/one_sided_view.md for what the payload "
            f"holds and what is deferred."
        )


def read_models_from_payload(
    payload: Mapping[str, Any], *, battle_tag: str = "", events: Sequence[Any] = (),
) -> Tuple[LiveView, Optional[LegalActions], ViewBattle]:
    """The whole adapter in one call: ``(live, legal, battle)`` ready for
    ``Gen3ObservationEncoder.encode(battle, legal=legal, …)``.

    ``events`` is the successor's whole-battle event log when the caller has one (see
    :attr:`ViewBattle.events`); omitting it leaves the Wish pair and the sleep-wake belief at
    their no-information values, which is what every board-only caller wants.

    A FAINTED mon's stages are the payload's — none, the sim's truth — on every caller: the fork
    clears them at the faint (`gen3_pe_reading_fixes_v1`, PE-V10), which retired the ply-folded
    boost ledger this function used to restore them from at a D10 board."""
    live = live_view_from_payload(payload, battle_tag=battle_tag)
    legal = legal_actions_from_payload(payload, live)
    return live, legal, ViewBattle(live, legal, payload, events=events)
