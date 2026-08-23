"""Tests for the source-derived gen3 effect allowlists + crash-don't-drop encoders.

The derivation test re-derives the volatile set from Showdown move data ∩ the project's
gen3 move set (filtered to poke-env's Effect enum), so the curated allowlist can't
silently drift from the data. The behaviour tests pin crash-don't-drop: an unknown
volatile / cant reason RAISES rather than being dropped.
"""

import json
import re

import numpy as np
import pytest

from agents.observation.gen3_effects import (
    CANT_DIM,
    CANT_REASONS,
    GEN3_VOLATILE_TO_SLOT,
    VOLATILE_DIM,
    VOLATILE_SLOTS,
    UnknownCantReasonError,
    UnknownVolatileError,
    encode_cant_reason,
    encode_volatiles,
    normalize_cant_reason,
)

from utils.paths import repo_root

_ROOT = repo_root()


def _scan_volatiles(ts_path, allowed_ids) -> set[str]:
    """Volatiles set by any entry in ``ts_path`` whose id is in ``allowed_ids``,
    via ``volatileStatus: 'x'`` or ``addVolatile('x')``. Deterministic top-level
    block split (``\\n\\tID: {`` keys)."""
    txt = (_ROOT / ts_path).read_text(encoding="utf-8", errors="replace")
    keys = list(re.finditer(r"\n\t([a-z0-9]+): \{", txt))
    found = set()
    for i, k in enumerate(keys):
        if k.group(1) not in allowed_ids:
            continue
        body = txt[k.end():(keys[i + 1].start() if i + 1 < len(keys) else len(txt))]
        for vm in re.finditer(r"volatileStatus: '([a-z0-9]+)'", body):
            found.add(vm.group(1))
        for vm in re.finditer(r"addVolatile\('([a-z0-9]+)'", body):
            found.add(vm.group(1))
    return found


def _gen3_move_driven_volatiles() -> set[str]:
    """Re-derive every volatile a gen3-legal MOVE or ABILITY can set, filtered to ids
    poke-env's ``Effect`` enum can surface (only those reach ``mon.effects`` →
    LiveView.volatiles). Scanning abilities too is essential — Flash Fire's volatile
    comes from abilities.ts, not moves.ts (the gap the e2e fuzz caught)."""
    from poke_env.battle.effect import Effect

    enum_ids = {e.name.lower().replace("_", "") for e in Effect}
    move_ids = set(json.load(open(_ROOT / "data/pokemon/gen3_moves.json")).keys())
    ability_ids = set(json.load(open(_ROOT / "data/pokemon/gen3_abilities.json")).keys())
    found = _scan_volatiles("deps/pokemon-showdown/data/moves.ts", move_ids)
    found |= _scan_volatiles("deps/pokemon-showdown/data/abilities.ts", ability_ids)
    return {v for v in found if v in enum_ids}


def _gen3_ability_activation_volatiles() -> set[str]:
    """Re-derive the ability-activation volatile class FROM SOURCE — the second class,
    which the ``volatileStatus``/``addVolatile`` scan above structurally cannot see.

    An anti-status / status-cure ability surfaces as a volatile via a DIFFERENT poke-env
    path: when it fires it emits ``this.add('-activate', mon, 'ability: <self>')`` in
    ``abilities.ts``, and poke-env's ``-activate`` handler (abstract_battle.py) calls
    ``mon.start_effect('ability: <self>')`` for any such line → ``mon.effects`` →
    ``LiveView.volatiles``. So the source-of-truth for this class is: every gen3 ability
    that **self-emits** ``-activate|ability:<itself>``, intersected with the ``Effect``
    enum (only enum-representable ids reach ``mon.effects``; the rest map to
    ``Effect.UNKNOWN`` and would be a separate gap). This is what makes a future
    poke-env/Showdown change (a new gen3 burn/poison/sleep-immunity ability, a rename)
    fail CI here instead of crashing 6 hours into training. ``flashfire`` is excluded by
    construction: it emits ``-start``, not ``-activate`` (it's a persistent boosted state,
    not a one-shot activation), and lives in the binary list."""
    from poke_env.battle.effect import Effect

    def _to_id(s: str) -> str:
        return "".join(c for c in s.lower() if c.isalnum())

    enum_ids = {e.name.lower().replace("_", "") for e in Effect}
    ability_ids = set(json.load(open(_ROOT / "data/pokemon/gen3_abilities.json")).keys())
    txt = (_ROOT / "deps/pokemon-showdown/data/abilities.ts").read_text(
        encoding="utf-8", errors="replace"
    )
    keys = list(re.finditer(r"\n\t([a-z0-9]+): \{", txt))
    derived = set()
    for i, k in enumerate(keys):
        aid = k.group(1)
        if aid not in ability_ids:
            continue
        body = txt[k.end():(keys[i + 1].start() if i + 1 < len(keys) else len(txt))]
        for m in re.finditer(
            r"this\.add\(\s*'-activate'\s*,\s*\w+\s*,\s*'ability:\s*([^']+)'", body
        ):
            if _to_id(m.group(1)) == aid:  # SELF-activation only
                derived.add(aid)
    return {v for v in derived if v in enum_ids}


# --------------------------------------------------------------------------- #
# Derivation: the allowlist covers every gen3 move-driven (enum-surfaceable) volatile.
# --------------------------------------------------------------------------- #
@pytest.mark.integration  # needs deps/pokemon-showdown checked out
def test_allowlist_covers_every_gen3_move_volatile():
    derived = _gen3_move_driven_volatiles()
    known = set(GEN3_VOLATILE_TO_SLOT)
    missing = derived - known
    assert not missing, (
        f"gen3-legal moves can set these volatiles that gen3_effects.py does not "
        f"classify (would crash-don't-drop at runtime): {sorted(missing)}"
    )


@pytest.mark.integration  # needs deps/pokemon-showdown checked out
def test_allowlist_covers_every_gen3_ability_activation_volatile():
    """THE REGRESSION GUARD FOR THE WATERVEIL CLASS. The ability-activation volatiles
    (Immunity / Synchronize / Magma Armor / Water Veil / …) reach the encoder through a
    poke-env code path (``-activate|ability:X`` → ``start_effect``) that the
    move-volatile scan structurally cannot see, so they used to be a HAND-MAINTAINED list
    — and that list silently missed ``waterveil`` until it crashed training 6 hours in
    (the 3rd time this class bit: doomdesire, magmaarmor, now waterveil). This re-derives
    the whole class from ``abilities.ts`` and asserts every member is classified AND
    encodes to the shared ``ability_activated`` slot without raising. A future gen3
    status-immunity ability now fails HERE, in CI, not in training."""
    from agents.observation.gen3_effects import (
        _ABILITY_ACTIVATION_VOLATILES, _ABILITY_ACTIVATED_SLOT,
    )

    derived = _gen3_ability_activation_volatiles()
    assert derived, "source scan found no gen3 ability-activation volatiles — scan broke"
    # The curated runtime list must be EXACTLY the source-derived set (no drift either way):
    # missing → crash-don't-drop at runtime; extra → a dead id masquerading as derived.
    assert set(_ABILITY_ACTIVATION_VOLATILES) == derived, (
        f"_ABILITY_ACTIVATION_VOLATILES has drifted from abilities.ts.\n"
        f"  missing (would crash training): {sorted(derived - set(_ABILITY_ACTIVATION_VOLATILES))}\n"
        f"  extra (dead / not source-derived): {sorted(set(_ABILITY_ACTIVATION_VOLATILES) - derived)}"
    )
    # And every derived id must actually encode (to the one shared slot), never raise.
    slot_idx = VOLATILE_SLOTS.index(_ABILITY_ACTIVATED_SLOT)
    for vid in sorted(derived):
        vec = encode_volatiles([vid])  # must not raise UnknownVolatileError
        assert vec[slot_idx] == 1.0 and vec.sum() == 1.0, (
            f"{vid!r} did not collapse to the single ability_activated slot"
        )


@pytest.mark.integration
def test_no_dead_volatile_slots_beyond_known_extras():
    """Every classified id is either a derived gen3 volatile (move-driven OR
    ability-activation, both derived from source) or an intentional counter/trap variant
    or a fuzz-confirmed engine extra. No silent dead entries."""
    derived = _gen3_move_driven_volatiles() | _gen3_ability_activation_volatiles()
    intentional_extras = {
        "perish0", "perish1", "perish2", "perish3",
        "stockpile", "stockpile1", "stockpile2", "stockpile3",
        "wrap", "bind", "clamp", "whirlpool", "firespin", "sandtomb",
        "struggle",  # engine-set when out of usable moves; confirmed by e2e fuzz
        # Future Sight / Doom Desire emit `-start` on the user (moves.ts) even though
        # they mechanically use addSlotCondition — poke-env surfaces them as user
        # volatiles. The volatileStatus/addVolatile scan can't see the `-start` path,
        # so they're explicit extras. Confirmed by the training smoke tripwire.
        "doomdesire", "futuresight",
    }
    for vid in GEN3_VOLATILE_TO_SLOT:
        assert vid in derived or vid in intentional_extras, (
            f"{vid!r} is classified but is neither a derived gen3 volatile nor a known "
            f"variant/extra — is it dead?"
        )


# --------------------------------------------------------------------------- #
# Volatile encoding + crash-don't-drop                                         #
# --------------------------------------------------------------------------- #
def test_encode_volatiles_basic():
    vec = encode_volatiles(["leechseed", "substitute", "taunt"])
    assert vec.shape == (VOLATILE_DIM,)
    assert vec[VOLATILE_SLOTS.index("leechseed")] == 1.0
    assert vec[VOLATILE_SLOTS.index("substitute")] == 1.0
    assert vec[VOLATILE_SLOTS.index("taunt")] == 1.0
    assert vec.sum() == 3.0


def test_empty_volatiles_is_zeros_not_crash():
    assert encode_volatiles([]).sum() == 0.0


def test_partially_trapped_is_one_slot():
    # poke-env emits the collapsed id; the variant move ids map to the same slot
    for vid in ("partiallytrapped", "wrap", "bind", "clamp", "whirlpool"):
        vec = encode_volatiles([vid])
        assert vec[VOLATILE_SLOTS.index("partiallytrapped")] == 1.0
        assert vec.sum() == 1.0


def test_perish_counter_normalised():
    # perishN normalises to (N+1)/4: perish3 (3 turns left) -> 1.0, perish0 -> 0.25
    assert encode_volatiles(["perish3"])[VOLATILE_SLOTS.index("perish")] == 1.0
    assert encode_volatiles(["perish1"])[VOLATILE_SLOTS.index("perish")] == pytest.approx(0.5)
    assert encode_volatiles(["perish0"])[VOLATILE_SLOTS.index("perish")] == pytest.approx(0.25)


def test_stockpile_counter_normalised():
    assert encode_volatiles(["stockpile3"])[VOLATILE_SLOTS.index("stockpile")] == 1.0
    assert encode_volatiles(["stockpile1"])[VOLATILE_SLOTS.index("stockpile")] == pytest.approx(1 / 3)


def test_unknown_volatile_crashes():
    with pytest.raises(UnknownVolatileError):
        encode_volatiles(["dynamax"])  # gen8 — must never silently drop
    with pytest.raises(UnknownVolatileError):
        encode_volatiles(["leechseed", "totallybogus"])


def test_focuspunch_and_struggle_have_slots():
    """The two engine-set volatiles the e2e fuzz caught must encode, not crash."""
    assert encode_volatiles(["focuspunch"]).sum() == 1.0
    assert encode_volatiles(["struggle"]).sum() == 1.0


def test_future_move_volatiles_have_slots():
    """Future Sight / Doom Desire surface as user volatiles via their `-start` line;
    the training smoke crash-don't-drop tripwire caught doomdesire. Must encode binary."""
    assert encode_volatiles(["doomdesire"])[VOLATILE_SLOTS.index("doomdesire")] == 1.0
    assert encode_volatiles(["futuresight"])[VOLATILE_SLOTS.index("futuresight")] == 1.0
    # paired with another volatile, both bits set, no crash
    assert encode_volatiles(["futuresight", "leechseed"]).sum() == 2.0


def test_ability_activation_volatiles_collapse_to_one_slot():
    """gen3 ability-activation effects (Immunity / Synchronize / Oblivious / …) surface
    as volatiles via poke-env's -activate path. `immunity` was caught by the 5m fuzz.
    They all COLLAPSE to the single `ability_activated` slot — their identity is now
    captured persistently in the per-mon ability block (the activation reveals the
    ability), so a per-ability slot would just duplicate it."""
    from agents.observation.gen3_effects import (
        _ABILITY_ACTIVATION_VOLATILES, _ABILITY_ACTIVATED_SLOT,
    )
    slot_idx = VOLATILE_SLOTS.index(_ABILITY_ACTIVATED_SLOT)
    for vid in _ABILITY_ACTIVATION_VOLATILES:
        vec = encode_volatiles([vid])
        assert vec[slot_idx] == 1.0, f"{vid} did not map to ability_activated"
        assert vec.sum() == 1.0, f"{vid} set more than the one shared slot"
    # No per-ability slot exists any more (collapsed)
    assert "immunity" not in VOLATILE_SLOTS
    assert "synchronize" not in VOLATILE_SLOTS


def test_waterveil_encodes_to_ability_activated_not_crash():
    """Regression for the crash-don't-drop tripwire that fired in training: Water Veil
    (the burn-immunity ability) reached LivePokemon.volatiles from a real gen3ou battle
    and had no slot, raising UnknownVolatileError. It is the exact sibling of immunity /
    limber / magmaarmor — a status-immunity ability that emits |-activate|mon|ability: X —
    so it must collapse to the shared `ability_activated` slot, never raise. No source
    files needed (pure unit), so this guards the fix even without the submodule."""
    from agents.observation.gen3_effects import _ABILITY_ACTIVATED_SLOT
    slot_idx = VOLATILE_SLOTS.index(_ABILITY_ACTIVATED_SLOT)
    vec = encode_volatiles(["waterveil"])  # was UnknownVolatileError
    assert vec[slot_idx] == 1.0
    assert vec.sum() == 1.0
    # Paired with a real binary volatile from the same window: both bits, no crash.
    assert encode_volatiles(["waterveil", "leechseed"]).sum() == 2.0


def test_status_immunity_abilities_all_have_volatile_slot():
    """LOCKSTEP INVARIANT — the waterveil bug, generalized. An ability that grants full
    immunity to a major status (``ABILITY_STATUS_IMMUNITY`` in gen3_mechanics) BLOCKS that
    status by *activating*: poke-env records the ``-immune [from] ability:`` / ``-activate``
    as a volatile on the holder, which ``encode_volatiles`` must classify or it crashes
    (crash-don't-drop). So **every status-immunity ability MUST also be an ability-activation
    volatile** — ``waterveil`` lived in the mechanics map but NOT this allowlist, and crashed
    training the moment Water Veil blocked a burn (6 hours in). This guard catches that class
    INSTANTLY: pure unit, no server, runs in the default suite. The reverse is deliberately
    NOT required — the activation set also holds non-status activations (oblivious / owntempo
    / shedskin / stickyhold / suctioncups / synchronize), which legitimately aren't statuses.

    Dependency direction is obs→mechanics (the consumer reads the producer), never the
    reverse, so this lives here in the obs test, not in gen3_mechanics_test."""
    from agents.gen3_mechanics import ABILITY_STATUS_IMMUNITY
    from agents.observation.gen3_effects import _ABILITY_ACTIVATION_VOLATILES

    missing = set(ABILITY_STATUS_IMMUNITY) - set(_ABILITY_ACTIVATION_VOLATILES)
    assert not missing, (
        f"status-immunity abilities with NO ability_activated volatile slot — "
        f"encode_volatiles will crash-don't-drop the moment one of them blocks a status "
        f"(the waterveil incident): {sorted(missing)}. Add each to "
        f"_ABILITY_ACTIVATION_VOLATILES in gen3_effects.py."
    )


def test_ability_activation_volatiles_resolve_to_effect_enum():
    """Every ability-activation id must resolve to a REAL poke-env ``Effect`` member, never
    ``Effect.UNKNOWN`` — checked two ways: (1) as the id ``LiveView.volatiles`` actually
    emits (``Effect.name`` id-form), and (2) via poke-env's own protocol resolution of the
    line the ability emits, ``Effect.from_showdown_message('ability: <Name>')``. If an id
    falls through to ``Effect.UNKNOWN`` it surfaces as volatile id ``'unknown'`` and the
    allowlist's slot never matches → crash-don't-drop. This is the ``magmaarmor`` lesson
    (it once mapped to ``Effect.UNKNOWN`` until ``Effect.MAGMA_ARMOR`` was added to the fork
    enum) turned into a standing CI guard. Pure unit — needs only data/ + the fork enum."""
    from agents.observation.gen3_effects import _ABILITY_ACTIVATION_VOLATILES
    from poke_env.battle.effect import Effect
    from agents import gen3_data

    enum_ids = {e.name.lower().replace("_", "") for e in Effect}
    for vid in _ABILITY_ACTIVATION_VOLATILES:
        assert vid in enum_ids, (
            f"{vid!r} matches no poke-env Effect member id — LiveView would surface it as "
            f"'unknown' (Effect.UNKNOWN); add the member to the fork enum (effect.py)."
        )
        ab = gen3_data.abilities.get(vid)
        assert ab is not None, f"{vid!r} is not a known gen3 ability (data/pokemon)"
        # The exact path poke-env walks for |-activate|mon|ability: <Name>.
        eff = Effect.from_showdown_message(f"ability: {ab.name}")
        assert eff is not Effect.UNKNOWN, (
            f"poke-env resolves '-activate|...|ability: {ab.name}' to Effect.UNKNOWN — the "
            f"fork enum is missing this ability; add it to effect.py (the magmaarmor fix)."
        )


# --------------------------------------------------------------------------- #
# Cant reason encoding + crash-don't-drop                                      #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("raw,expected", [
    ("slp", "slp"),
    ("frz", "frz"),
    ("flinch", "flinch"),
    ("move: Taunt", "taunt"),
    ("move: Imprison", "imprison"),
    ("Focus Punch", "focuspunch"),
    ("ability: Truant", "truant"),
    ("Attract", "attract"),
    ("recharge", "recharge"),
])
def test_normalize_cant_reason(raw, expected):
    assert normalize_cant_reason(raw) == expected


def test_cant_onehot_and_none():
    vec = encode_cant_reason("flinch")
    assert vec.shape == (CANT_DIM,)
    assert vec[CANT_REASONS.index("flinch")] == 1.0
    assert vec.sum() == 1.0
    assert encode_cant_reason(None).sum() == 0.0  # acted / switched -> zeros, not crash


def test_flinch_distinct_from_sleep():
    """The whole point: distinct causes get distinct slots (the user's example)."""
    assert not np.array_equal(encode_cant_reason("flinch"), encode_cant_reason("slp"))


def test_unknown_cant_reason_crashes():
    with pytest.raises(UnknownCantReasonError):
        normalize_cant_reason("move: Heal Block")  # gen4+ — never in gen3
    with pytest.raises(UnknownCantReasonError):
        encode_cant_reason("xyzzy")
