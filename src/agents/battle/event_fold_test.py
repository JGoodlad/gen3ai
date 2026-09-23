"""Unit pins for ``ViewEventFolder``'s light board against ``Gen3Battle`` on constructed protocol.

``event_fold_parity_fuzz_test.py`` is the differential over real search plies, but it seeds every
fold at DEPTH 1 from a real battle and folds ONE ply, so any rule that only bites once a fold spans
a ``|turn|`` or continues after ``branch()`` is invisible to it. These pins construct exactly those
shapes and compare the fold's events with the live ``Gen3Battle``'s, field by field.
"""

from __future__ import annotations

from typing import List

from agents.battle.event_fold import ViewEventFolder
from agents.battle.offline_feed import feed_chunk, new_battle

_FIELDS = ("seq", "turn", "kind", "side", "actor_species", "target_species", "value", "raw")

_PREFIX = "\n".join([
    "|player|p1|me||",
    "|player|p2|foe||",
    "|teamsize|p1|2",
    "|teamsize|p2|2",
    "|start",
    "|switch|p1a: Metagross|Metagross|301/301",
    "|switch|p2a: Gyarados|Gyarados, M|100/100",
    "|turn|1",
])

#: The reproduction of Phase-0 finding F2 (`rust_core_phase0_2026-09-23` §3): OUR move opens the
#: move scope on turn 1; at the top of turn 2 the foe switches in an Intimidate user whose drop our
#: Clear Body blocks. poke-env's `end_turn` resets `_current_move_user_side` at `|turn|`, so the
#: live `-fail` is owned by NO side; a fold that never resets reads it as OURS (actor Metagross).
_PLY_1 = "\n".join([
    "|",
    "|move|p1a: Metagross|Meteor Mash|p2a: Gyarados",
    "|-damage|p2a: Gyarados|60/100",
    "|",
    "|upkeep",
    "|turn|2",
])
_PLY_2 = "\n".join([
    "|",
    "|switch|p2a: Salamence|Salamence, M|100/100",
    "|-ability|p2a: Salamence|Intimidate|boost",
    "|-fail|p1a: Metagross|unboost|[from] ability: Clear Body|[of] p1a: Metagross",
])


def _assert_same(live, folded) -> None:
    assert [e.kind for e in live] == [e.kind for e in folded]
    for a, b in zip(live, folded):
        for f in _FIELDS:
            assert getattr(a, f) == getattr(b, f), (f, a, b)


def _live(chunks: List[str]):
    b = new_battle("p1", {"p1": "me", "p2": "foe"})
    feed_chunk(b, _PREFIX)
    cursor = b.event_cursor
    for c in chunks:
        feed_chunk(b, c)
    return b, b.events_since(cursor)


def test_a_fold_spanning_turn_resets_the_move_owner_like_poke_env_end_turn():
    """F2, depth 1: ONE fold covers the `|turn|` and the start-of-turn outcome line."""
    live_battle, live = _live([_PLY_1, _PLY_2])
    fail = [e for e in live if e.kind.name == "FAIL"]
    assert fail and fail[0].side is None and fail[0].actor_species is None, \
        "precondition: the LIVE reading owns the start-of-turn -fail to nobody"
    seed = new_battle("p1", {"p1": "me", "p2": "foe"})
    feed_chunk(seed, _PREFIX)
    folder = ViewEventFolder.seed_from(seed)
    _assert_same(live, folder.fold([_PLY_1, _PLY_2]))


def test_a_branched_second_ply_starts_with_no_move_owner():
    """F2, depth 2: fold ply 1, ``branch()`` (search's deeper ply), fold ply 2."""
    _, live = _live([_PLY_1, _PLY_2])
    seed = new_battle("p1", {"p1": "me", "p2": "foe"})
    feed_chunk(seed, _PREFIX)
    folder = ViewEventFolder.seed_from(seed)
    first = folder.fold([_PLY_1])
    second = folder.branch().fold([_PLY_2])
    _assert_same(live, list(first) + list(second))


def _fold_vs_live(chunks: List[str]) -> None:
    _, live = _live(chunks)
    seed = new_battle("p1", {"p1": "me", "p2": "foe"})
    feed_chunk(seed, _PREFIX)
    _assert_same(live, ViewEventFolder.seed_from(seed).fold(chunks))


def test_cureteam_cures_only_the_named_sides_living_mons():
    """``-cureteam`` is poke-env's ``team.cure_status()`` over the NAMED side's team only (and a
    fainted mon keeps FNT). The light board cured every mon on both sides, so a later move into a
    paralysed FOE read ``target_status=None``."""
    _fold_vs_live(["\n".join([
        "|",
        "|move|p2a: Gyarados|Thunder Wave|p1a: Metagross",
        "|-status|p1a: Metagross|par",
        "|move|p1a: Metagross|Toxic|p2a: Gyarados",
        "|-status|p2a: Gyarados|tox",
        "|",
        "|upkeep",
        "|turn|2",
        "|",
        "|move|p2a: Gyarados|Aromatherapy|p2a: Gyarados",
        "|-cureteam|p2a: Gyarados|[from] move: Aromatherapy",
        "|move|p2a: Gyarados|Earthquake|p1a: Metagross",
        "|-damage|p1a: Metagross|200/301 par",
    ])])


def test_a_fainted_mon_reads_fnt_like_poke_env_faint():
    """``Pokemon.faint`` (and ``set_hp_status('0 fnt')``) sets status FNT; the light board cleared
    it to None, so a move line naming the fainted active read a different ``target_status``."""
    _fold_vs_live(["\n".join([
        "|",
        "|move|p1a: Metagross|Explosion|p2a: Gyarados",
        "|-damage|p2a: Gyarados|0 fnt",
        "|faint|p2a: Gyarados",
        "|move|p1a: Metagross|Earthquake|p2a: Gyarados",
    ])])


def test_a_forme_change_does_not_rename_the_species():
    """``-formechange`` is ``Pokemon.forme_change`` → ``_update_from_pokedex(store_species=False)``:
    poke-env keeps the BASE species, so every later event on a Forecast Castform names
    ``castform``. The light board renamed it (``castformsunny``)."""
    _live_chunks = ["\n".join([
        "|",
        "|switch|p2a: Castform|Castform, M|100/100",
        "|-formechange|p2a: Castform|Castform-Sunny|[msg]|[from] ability: Forecast",
        "|move|p1a: Metagross|Meteor Mash|p2a: Castform",
        "|-damage|p2a: Castform|40/100",
    ])]
    _fold_vs_live(_live_chunks)


def test_a_statusless_hp_line_clears_the_status_like_set_hp_status():
    """``set_hp_status`` writes the status from the HP field and CLEARS it when the field carries
    none; the light board only ever SET it, so the two disagreed on a later ``target_status``."""
    _fold_vs_live(["\n".join([
        "|",
        "|move|p2a: Gyarados|Thunder Wave|p1a: Metagross",
        "|-status|p1a: Metagross|par",
        "|-heal|p1a: Metagross|301/301",
        "|move|p2a: Gyarados|Earthquake|p1a: Metagross",
    ])])
