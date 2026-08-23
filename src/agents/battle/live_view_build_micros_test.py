"""Gates for the pure-function memos on the `LiveView` build path
(`gen3_live_view_build_micros_v1`).

The board build (`LiveView.from_battle`) measures **17% of per-decision worker CPU** — the single
largest item in the budget once `gen3_live_view_memo_v1` collapsed the five per-decision rebuilds
into one. This pass did not reduce that ONE build's count; it reduced its cost, by memoizing
three derivations that are pure functions of IMMUTABLE inputs:

* ``Move.entry``  — a dex lookup keyed by ``(_id, gen)``, both write-once in ``Move.__init__``
* ``Move.max_pp`` — the same, plus ``_from_transform``
* ``live_view._enum_name`` / ``_id`` — the ``.name`` of a process-wide enum SINGLETON

Each is exact by construction, so these tests pin the CONSTRUCTION rather than sampling outputs:
the memo must agree with a COLD instance (whose first read *is* the computation), and the enum
memo's key safety is a property of the enum classes rather than of our code — so it is asserted
directly, on them.
"""
from __future__ import annotations

import copy
import itertools

import pytest

from poke_env.battle.effect import Effect
from poke_env.battle.field import Field
from poke_env.battle.move import Move
from poke_env.battle.pokemon_type import PokemonType
from poke_env.battle.side_condition import SideCondition
from poke_env.battle.status import Status
from poke_env.battle.weather import Weather
from poke_env.data.gen_data import GenData

from agents.battle.live_view import _enum_name, _id

# Every enum that reaches `_enum_name` / `_id` from `LivePokemon.from_pokemon` or
# `LiveView.from_battle`: status, types, volatiles (Effect), side conditions — plus Weather and
# Field, which share the helpers elsewhere in the read-model.
_LIVE_VIEW_ENUMS = (Status, PokemonType, Effect, SideCondition, Weather, Field)


# --------------------------------------------------------------------------- #
# The enum-name memos                                                          #
# --------------------------------------------------------------------------- #
def test_equal_enum_members_across_classes_are_always_identical():
    """The KEY-SAFETY property the two enum memos depend on, asserted on the enums themselves.

    Both caches are plain dicts keyed by the enum MEMBER, so two members that compare equal share
    a cache slot. For a plain ``Enum`` that is identity and the caches are safe. Turn any of these
    into an ``IntEnum`` / ``StrEnum`` and members of DIFFERENT classes with the same value begin
    comparing equal — ``Status.BRN`` and ``Weather.SUNNYDAY`` would share a slot and one would
    answer with the other's name, silently, on every board build. Nothing else in the tree would
    notice, so the guard lives here.
    """
    members = [m for e in _LIVE_VIEW_ENUMS for m in e]
    assert len(members) > 100, f"suspiciously few enum members to check: {len(members)}"
    for a, b in itertools.combinations(members, 2):
        if a == b:
            assert a is b, (
                f"{type(a).__name__}.{a.name} == {type(b).__name__}.{b.name} but is not the same "
                f"object — the _enum_name/_id memos are keyed by member and would collide."
            )


@pytest.mark.parametrize("enum_cls", _LIVE_VIEW_ENUMS, ids=lambda e: e.__name__)
def test_enum_name_and_id_memos_match_the_uncached_derivation(enum_cls):
    for member in enum_cls:
        assert _enum_name(member) == member.name.lower()
        assert _id(member) == member.name.lower().replace("_", "")
        # The SECOND read is the memoized path — it must give the same answer, not merely a value.
        assert _enum_name(member) == member.name.lower()
        assert _id(member) == member.name.lower().replace("_", "")


def test_enum_helpers_still_pass_none_through():
    assert _enum_name(None) is None
    assert _id(None) is None


def test_the_two_enum_memos_do_not_share_a_cache():
    """``_enum_name`` keeps separators, ``_id`` strips them, so a member whose name HAS an
    underscore is the only place their answers differ — and therefore the only place a shared
    cache would be invisible. Pick those deliberately."""
    underscored = [m for e in _LIVE_VIEW_ENUMS for m in e if "_" in m.name]
    assert underscored, "no underscored member left to distinguish the two derivations"
    for member in underscored[:25]:
        assert _enum_name(member) != _id(member)
        assert _enum_name(member) == member.name.lower()
        assert _id(member) == member.name.lower().replace("_", "")


# --------------------------------------------------------------------------- #
# The Move.entry / Move.max_pp memos                                           #
# --------------------------------------------------------------------------- #
def _uncached_max_pp(move_id: str, gen: int, from_transform: bool = False) -> int:
    """The pre-memo formula, spelled out. Deliberately a DUPLICATE of the property's body: an
    oracle that read the property back would pass no matter what the property did."""
    entry = GenData.from_gen(gen).moves[move_id]
    max_pp = entry["pp"] * 8 // 5
    if gen >= 5 and from_transform:
        return min(5, max_pp)
    elif gen < 3:
        max_pp = min(max_pp, 61)
    return max_pp


def test_max_pp_memo_matches_the_formula_over_the_whole_gen3_move_universe():
    ids = sorted(GenData.from_gen(3).moves)
    assert len(ids) > 300, f"suspiciously small gen3 move table: {len(ids)}"
    for move_id in ids:
        mv = Move(move_id, gen=3)
        first = mv.max_pp                      # COLD — this read IS the computation
        assert first == _uncached_max_pp(move_id, 3), move_id
        assert mv.max_pp == first, move_id     # WARM — the memoized read
        assert mv.max_pp == first, move_id


@pytest.mark.parametrize("gen", [1, 2, 3, 4, 5, 8, 9])
def test_max_pp_memo_takes_the_same_branch_as_the_formula_in_every_gen(gen):
    """The property has three branches (the ``from_transform`` cap at gen>=5, the gen<3 clamp,
    and plain), and a memo that cached the value from before a branch would be wrong in exactly
    one of them."""
    checked = 0
    for move_id in ("tackle", "thunderbolt", "recover"):
        if move_id not in GenData.from_gen(gen).moves:
            continue
        checked += 1
        assert Move(move_id, gen=gen).max_pp == _uncached_max_pp(move_id, gen)
        transformed = Move(move_id, gen=gen, from_transform=True)
        assert transformed.max_pp == _uncached_max_pp(move_id, gen, from_transform=True)
    assert checked, f"no probe move exists in gen {gen} — the parametrization proves nothing"


def test_the_memo_is_PER_INSTANCE_and_never_leaks_between_moves():
    """A cache keyed on anything coarser than the instance would serve one move's pp for another.
    Two moves with DIFFERENT max pp, read interleaved — then the same id across two gens, where
    the clamp differs."""
    a, b = Move("tackle", gen=3), Move("recover", gen=3)   # 35 pp vs 5 pp base
    assert a.max_pp != b.max_pp
    for _ in range(3):
        assert a.max_pp == _uncached_max_pp("tackle", 3)
        assert b.max_pp == _uncached_max_pp("recover", 3)
    g1, g3 = Move("tackle", gen=1), Move("tackle", gen=3)
    assert g1.max_pp == _uncached_max_pp("tackle", 1)
    assert g3.max_pp == _uncached_max_pp("tackle", 3)
    plain, transformed = Move("tackle", gen=5), Move("tackle", gen=5, from_transform=True)
    assert plain.max_pp != transformed.max_pp
    assert plain.max_pp == _uncached_max_pp("tackle", 5)
    assert transformed.max_pp == _uncached_max_pp("tackle", 5, from_transform=True)


def test_entry_memo_returns_the_dex_row_itself_every_time():
    mv = Move("thunderbolt", gen=3)
    dex_row = GenData.from_gen(3).moves["thunderbolt"]
    assert mv.entry is dex_row
    assert mv.entry is dex_row


def test_entry_memo_covers_the_synthetic_recharge_branch_and_is_stable():
    """``recharge`` / ``fight`` are not in the dex — the property SYNTHESISES a dict. Before the
    memo each read built a fresh one; now every read returns the same object. Both are correct
    (the dex branch has always returned a SHARED row, so no caller may mutate an entry), but the
    identity is now load-bearing, so pin it."""
    mv = Move("recharge", gen=3)
    first = mv.entry
    assert first == {"pp": 1, "type": "normal", "category": "Special", "accuracy": 1}
    assert mv.entry is first
    assert mv.max_pp == 1 * 8 // 5


def test_a_deep_copied_move_still_carries_no_reference_to_the_dex():
    """The materializer's per-arm `deepcopy` of the whole battle graph is justified in
    `obs_materializer._PlayerSnapshot` by exactly one sentence — *"`Pokemon`/`Move` carry an int
    `_gen` and look entries up on demand"*. Caching `entry` ON THE INSTANCE would break that: a
    dex row would ride inside every `Move` and each arm's deepcopy would duplicate it. That is
    why the entry memo is keyed `(gen, id)` at MODULE scope while `max_pp` (a plain int) is not.
    """
    mv = Move("thunderbolt", gen=3)
    _warm = (mv.entry, mv.max_pp)                        # noqa: F841 — populate both memos
    clone = copy.deepcopy(mv)
    assert clone.entry is GenData.from_gen(3).moves["thunderbolt"], (
        "a deep-copied Move produced its OWN dex row — the entry memo has migrated onto the "
        "instance and the materializer's clone contract is broken."
    )
    assert clone.max_pp == mv.max_pp


def test_an_unknown_move_still_raises_and_the_error_path_caches_nothing():
    """The raise happens in the CONSTRUCTOR — ``__init__`` reads ``max_pp`` → ``entry`` — so no
    instance with a poisoned cache can exist at all. That is pre-existing behaviour and the memo
    must not change it: nothing is written on the error path, and the object never escapes."""
    with pytest.raises(ValueError, match="Unknown move"):
        Move("thismoveisnotreal", gen=3)


def test_init_reads_max_pp_before_the_cache_slot_exists():
    """``Move.__init__`` sets ``_current_pp`` from ``self.max_pp`` — the property runs while the
    memo slot is still UNSET. ``Move`` uses ``__slots__``, so an unset slot raises AttributeError;
    this pins that the lazy path handles that rather than the constructor blowing up, which is
    exactly how a naive ``self._cache is None`` memo fails on a slotted class."""
    mv = Move("tackle", gen=3)
    assert mv.current_pp == mv.max_pp
    transformed = Move("tackle", gen=5, from_transform=True)
    assert transformed.current_pp == min(5, transformed.max_pp)
