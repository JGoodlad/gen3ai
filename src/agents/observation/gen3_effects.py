"""Source-derived gen3ou effect allowlists + **crash-don't-drop** encoders.

Guiding principle (user requirement): never *silently drop* battle state. If a volatile
or a ``|cant|`` reason appears that we have no encoding slot for, we **raise** (crash)
rather than ignore it — the same tripwire philosophy as the event log's
``UnsupportedMessageType``. A new effect means our coverage is stale; failing loudly
forces classification before it can corrupt training with dropped signal.

Both allowlists are **derived from data, not memory**, and ``gen3_effects_test.py``
re-derives the volatile set so the curated list can't silently drift:

* **Volatiles** — every ``volatileStatus`` / ``addVolatile`` a gen3-legal move can set
  (``deps/pokemon-showdown/data/moves.ts`` ∩ ``data/pokemon/gen3_moves.json``),
  **filtered to ids poke-env's ``Effect`` enum can actually surface** (only those reach
  ``mon.effects`` → ``LiveView.volatiles``), proven a superset by the e2e fuzz.
* **Cant reasons** — every ``this.add('cant', …)`` reachable in gen3.

Counters / trap variants collapse to one slot each (clarity / signal-to-noise): Perish
Song → one normalised 0–1 counter, Stockpile → one 0–1 counter, all partial-trap moves
→ the single ``partiallytrapped`` state. Missing data is ``0`` (never a guess).
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np


class UnknownVolatileError(Exception):
    """A volatile appeared that isn't in :data:`GEN3_VOLATILE_TO_SLOT`. Crash-don't-drop:
    classify it (binary slot, or a counter/trap variant) before training proceeds — we
    will not silently omit a real battle fact."""


class UnknownCantReasonError(Exception):
    """A ``|cant|`` reason appeared that isn't in :data:`CANT_REASONS`. Crash-don't-drop
    — flinch vs sleep vs a damaged Focus Punch are strategically distinct and must each
    have a slot rather than folding into a generic 'could not move' bit."""


# --------------------------------------------------------------------------- #
# Volatiles                                                                    #
# --------------------------------------------------------------------------- #
# Binary single-instance volatiles, poke-env id-form (``Effect.name.lower()`` minus
# ``_``), exactly what ``LiveView.volatiles`` emits. This is the authoritative set:
# every volatile a gen3-legal MOVE or ABILITY can set (moves.ts + abilities.ts
# ``volatileStatus``/``addVolatile`` ∩ gen3_moves.json / gen3_abilities.json)
# **filtered to ids the poke-env ``Effect`` enum represents** — only those can reach
# ``mon.effects``. Ids Showdown declares but poke-env does not enum (counter,
# mirrorcoat, furycutter, rollout/iceball, twoturnmove, perishsong, stall, truant) are
# excluded: they'd be dead slots, and the crash-guard catches them if they ever surface
# as ``Effect.UNKNOWN``. ``partiallytrapped`` is the single id poke-env uses for every
# partial-trap move; ``struggle`` is engine-set (out of PP); ``flashfire`` is the
# Flash Fire ability's "absorbed a Fire move" boosted state — all confirmed by the e2e
# fuzz. The derivation test asserts this stays a superset of the move+ability set.
# NOTE: two-turn-move *charge* (Solar Beam / Fly / Dig) is NOT a volatile in poke-env
# (tracked via ``mon.preparing``) — surfaced by LiveView separately, not here.
# BUT Future Sight / Doom Desire ARE surfaced as volatiles ON THE USER: although they
# mechanically use ``addSlotCondition(target, 'futuremove')``, both moves also emit
# ``this.add('-start', source, 'Doom Desire' / 'move: Future Sight')`` (moves.ts), which
# poke-env parses into ``Effect.DOOM_DESIRE`` / ``Effect.FUTURE_SIGHT`` on the user's
# ``effects`` → ``LiveView.volatiles``. The static derivation scan (volatileStatus /
# addVolatile) can't see the ``-start`` path, so they're classified here as binary
# intentional-extras (like struggle / flashfire) — confirmed by the training smoke
# crash-don't-drop tripwire. Encoded binary, consistent with the other turn-countable
# volatiles (disable / encore / bide).
#
# ABILITY-ACTIVATION VOLATILES (the second class the fuzz surfaced). poke-env's
# ``-activate`` / ``-start`` parse path calls ``start_effect(effect)`` for ANY
# ``|-activate|mon|ability: X`` line (abstract_battle.py), so a gen3 ability that emits
# ``this.add('-activate', pokemon, 'ability: X')`` when it fires (status-cure /
# anti-status abilities) records ``Effect.X`` into ``mon.effects`` → ``LiveView.volatiles``.
# These are NOT in the volatileStatus/addVolatile scan, so they are intentional-extras.
# The full gen3 set (gen3_abilities.json ∩ self-emitting -activate/-start ∩ poke-env
# Effect enum) is the eleven below; ``flashfire`` is handled separately (it's a persistent
# boosted STATE — "absorbed a Fire move" — not a one-shot activation, so it keeps its own
# slot in the binary list above). ``magmaarmor`` required adding ``Effect.MAGMA_ARMOR`` to
# poke-env's enum (it previously mapped to ``Effect.UNKNOWN`` → id ``'unknown'``).
#
# COLLAPSED TO ONE SLOT (``ability_activated``): these eleven are anti-status / status-cure
# ability activations (Immunity cured poison, Synchronize reflected a status, Oblivious
# shrugged off attract, …). Their IDENTITY is now captured PERSISTENTLY in the per-mon
# ability block — the ``-activate`` line reveals the ability and poke-env records it
# (abstract_battle ``-activate`` handler → ``mon.ability`` → obs ability block flips
# known=1 with the specific id). So a per-ability volatile slot would just duplicate the
# ability block; we keep a single shared "an anti-status ability fired this window" bit
# (mild tempo signal) instead of 11 redundant slots. Same collapse pattern as the trap
# variants → ``partiallytrapped``. We still must CLASSIFY each id (crash-don't-drop), so
# they all map to the one shared slot rather than being dropped.
_ABILITY_ACTIVATION_VOLATILES: Tuple[str, ...] = (
    "immunity", "insomnia", "limber", "magmaarmor", "oblivious", "owntempo",
    "shedskin", "stickyhold", "suctioncups", "synchronize", "vitalspirit",
)
_ABILITY_ACTIVATED_SLOT = "ability_activated"
_BINARY_VOLATILES: Tuple[str, ...] = (
    "attract", "bide", "charge", "confusion", "curse", "defensecurl",
    "destinybond", "disable", "doomdesire", "encore", "endure", "flashfire",
    "flinch", "focusenergy", "focuspunch", "followme", "foresight", "futuresight",
    "grudge", "helpinghand", "imprison", "ingrain", "leechseed", "lockedmove",
    "lockon", "magiccoat", "minimize", "mustrecharge", "nightmare",
    "partiallytrapped", "protect", "pursuit", "rage", "snatch", "struggle",
    "substitute", "taunt", "torment", "trapped", "uproar", "yawn",
)

# Counter volatiles — every stage maps to the one normalised counter slot. poke-env
# id-forms (confirmed against the Effect enum): Perish ``perish0..3``, Stockpile
# ``stockpile`` + ``stockpile1..3``.
_PERISH_LEVEL: Dict[str, float] = {f"perish{n}": (n + 1) / 4 for n in range(4)}
_STOCKPILE_LEVEL: Dict[str, float] = {
    "stockpile": 1 / 3, "stockpile1": 1 / 3, "stockpile2": 2 / 3, "stockpile3": 1.0
}
# Trap-move id variants poke-env *might* emit instead of the collapsed id — mapped to
# the same slot for belt-and-suspenders (the live protocol uses ``partiallytrapped``).
_TRAP_VARIANTS: Tuple[str, ...] = (
    "wrap", "bind", "clamp", "whirlpool", "firespin", "sandtomb",
)

# Encoded slot layout (order is part of the obs layout — append-only is safe, reordering
# is a retrain). Binary slots, the single collapsed ability-activation slot, then the two
# collapsed counter slots.
VOLATILE_SLOTS: Tuple[str, ...] = (
    _BINARY_VOLATILES + (_ABILITY_ACTIVATED_SLOT, "perish", "stockpile")
)
VOLATILE_DIM: int = len(VOLATILE_SLOTS)
_SLOT_INDEX: Dict[str, int] = {name: i for i, name in enumerate(VOLATILE_SLOTS)}

# id-form (as LiveView emits) -> (slot_name, value to write). Binary → 1.0; counters →
# normalised level; ability activations → the one shared ability_activated slot.
GEN3_VOLATILE_TO_SLOT: Dict[str, Tuple[str, float]] = {}
for _v in _BINARY_VOLATILES:
    GEN3_VOLATILE_TO_SLOT[_v] = (_v, 1.0)
for _v in _ABILITY_ACTIVATION_VOLATILES:
    GEN3_VOLATILE_TO_SLOT[_v] = (_ABILITY_ACTIVATED_SLOT, 1.0)
for _v in _TRAP_VARIANTS:
    GEN3_VOLATILE_TO_SLOT[_v] = ("partiallytrapped", 1.0)
for _v, _lvl in _PERISH_LEVEL.items():
    GEN3_VOLATILE_TO_SLOT[_v] = ("perish", _lvl)
for _v, _lvl in _STOCKPILE_LEVEL.items():
    GEN3_VOLATILE_TO_SLOT[_v] = ("stockpile", _lvl)


def encode_volatiles(volatiles) -> np.ndarray:
    """Encode an iterable of id-form volatile strings (``LivePokemon.volatiles``) into a
    fixed :data:`VOLATILE_DIM` vector. **Raises** :class:`UnknownVolatileError` on any
    volatile without a slot — we crash rather than drop. Empty input → all zeros."""
    vec = np.zeros(VOLATILE_DIM, dtype=np.float32)
    for vid in volatiles:
        mapping = GEN3_VOLATILE_TO_SLOT.get(vid)
        if mapping is None:
            raise UnknownVolatileError(
                f"volatile {vid!r} has no gen3 encoding slot — classify it in "
                f"gen3_effects.py (binary / trap / counter) before training. "
                f"Known: {sorted(GEN3_VOLATILE_TO_SLOT)}"
            )
        slot, value = mapping
        idx = _SLOT_INDEX[slot]
        vec[idx] = max(vec[idx], value)  # counters keep max level; binaries set the bit
    return vec


# --------------------------------------------------------------------------- #
# |cant| reasons                                                              #
# --------------------------------------------------------------------------- #
# Normalised cant-reason id -> meaning. Derived from Showdown add('cant', …) reachable
# in gen3 (conditions.ts + gen3 move callbacks). Each is a strategically distinct
# "couldn't act" cause — kept separate, never collapsed into one bit.
CANT_REASONS: Tuple[str, ...] = (
    "slp",          # fully asleep
    "frz",          # frozen solid
    "par",          # fully paralysed
    "flinch",       # flinched (only possible if hit first — a tempo signal)
    "recharge",     # must recharge (Hyper Beam)
    "attract",      # immobilised by love
    "disable",      # chose a disabled move
    "taunt",        # chose a status move under Taunt
    "imprison",     # chose a move the opponent sealed
    "focuspunch",   # Focus Punch disrupted by taking damage (the "took damage" case)
    "nopp",         # chosen move hit 0 PP
    "truant",       # loafing (Truant ability)
)
CANT_DIM: int = len(CANT_REASONS)
_CANT_INDEX: Dict[str, int] = {r: i for i, r in enumerate(CANT_REASONS)}
_CANT_RAW_TO_ID: Dict[str, str] = {r: r for r in CANT_REASONS}


def normalize_cant_reason(reason) -> str:
    """Normalise a raw ``|cant|`` reason to a stable id in :data:`CANT_REASONS`.

    Strips the protocol's ``move:`` / ``ability:`` prefixes and lowercases to id-form
    (``"move: Taunt"`` → ``"taunt"``, ``"Focus Punch"`` → ``"focuspunch"``,
    ``"ability: Truant"`` → ``"truant"``). **Raises** :class:`UnknownCantReasonError`
    on anything unrecognised — crash-don't-drop."""
    if reason is None:
        raise UnknownCantReasonError("cant reason is None — caller must gate on present")
    raw = str(reason)
    for prefix in ("move:", "ability:"):
        if raw.lower().startswith(prefix):
            raw = raw.split(":", 1)[1]
            break
    rid = "".join(c for c in raw.lower() if c.isalnum())
    norm = _CANT_RAW_TO_ID.get(rid)
    if norm is None:
        raise UnknownCantReasonError(
            f"cant reason {reason!r} (id {rid!r}) is not a known gen3 cause — add it to "
            f"gen3_effects.CANT_REASONS before training. Known: {list(CANT_REASONS)}"
        )
    return norm


def encode_cant_reason(reason) -> np.ndarray:
    """One-hot a normalised cant reason into a :data:`CANT_DIM` vector. ``reason=None``
    (the side acted normally / switched) → all zeros, NOT a crash."""
    vec = np.zeros(CANT_DIM, dtype=np.float32)
    if reason is None:
        return vec
    vec[_CANT_INDEX[normalize_cant_reason(reason)]] = 1.0
    return vec


def describe_volatiles(vec: np.ndarray) -> List[str]:
    """Human-readable nonzero volatile slots (for describe_vector / probe)."""
    return [
        f"{VOLATILE_SLOTS[i]}={vec[i]:.2f}" for i in range(VOLATILE_DIM) if vec[i] != 0
    ]


def describe_cant(vec: np.ndarray) -> str:
    """Human-readable cant reason from a one-hot (for describe_vector / probe)."""
    nz = np.nonzero(vec)[0]
    return CANT_REASONS[int(nz[0])] if len(nz) else "none"
