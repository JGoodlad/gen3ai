"""Tests for the source-derived gen3 effect allowlists + crash-don't-drop encoders.

The derivation test re-derives the volatile set from Showdown move data ∩ the project's
gen3 move set (filtered to poke-env's Effect enum), so the curated allowlist can't
silently drift from the data. The behaviour tests pin crash-don't-drop: an unknown
volatile / cant reason RAISES rather than being dropped.
"""

import json
import pathlib
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

_ROOT = pathlib.Path(__file__).resolve().parents[3]


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


@pytest.mark.integration
def test_no_dead_volatile_slots_beyond_known_extras():
    """Every classified id is either a derived gen3 volatile or an intentional
    counter/trap variant or a fuzz-confirmed engine extra. No silent dead entries."""
    derived = _gen3_move_driven_volatiles()
    intentional_extras = {
        "perish0", "perish1", "perish2", "perish3",
        "stockpile", "stockpile1", "stockpile2", "stockpile3",
        "wrap", "bind", "clamp", "whirlpool", "firespin", "sandtomb",
        "struggle",  # engine-set when out of usable moves; confirmed by e2e fuzz
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
