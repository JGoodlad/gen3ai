"""Unit tests for the own-side imputation meter's REVEAL RULES.

The meter's whole claim rests on one thing: that :func:`track_own_reveals` models what a
public replay actually shows, no more and no less. If it over-reveals, the measured error is
too small and the memo's #1 risk reads as safer than it is; if it under-reveals, we would
scrap a viable data source on a number we manufactured. So the rules are pinned here against
literal protocol lines rather than exercised only through a battle.

Fast (no battles, no bridge, milliseconds) and unmarked — this runs in the inner loop.
"""

from __future__ import annotations

import pytest

from agents.training.replay_imputation_probe import (
    OwnSideReveals,
    impute_item,
    impute_moves,
    impute_spread,
    phase_of,
    track_own_reveals,
)

P1_LEAD = "|switch|p1a: Sal|Salamence, L100, M|100/100"
P2_LEAD = "|switch|p2a: Tar|Tyranitar, L100, F|100/100"


def reveals(*lines: str, our_tag: str = "p1") -> OwnSideReveals:
    return track_own_reveals([P1_LEAD, P2_LEAD, *lines], our_tag)


# --------------------------------------------------------------------------- #
# moves
# --------------------------------------------------------------------------- #


def test_a_move_we_use_is_revealed():
    r = reveals("|move|p1a: Sal|Dragon Claw|p2a: Tar")
    assert r.moves_of("salamence") == {"dragonclaw"}


def test_the_opponents_move_never_reveals_ours():
    r = reveals("|move|p2a: Tar|Rock Slide|p1a: Sal")
    assert r.moves_of("salamence") == set()
    assert r.moves_of("tyranitar") == set()  # we track OUR side only


def test_the_species_comes_from_the_switch_details_not_the_nickname():
    """This pool carries localized nicknames, so a nickname-keyed read names nothing —
    the same trap ``determinize.reveal_events`` documents."""
    r = track_own_reveals(
        ["|switch|p1a: Triopikeur|Dugtrio, L100, M|100/100",
         "|move|p1a: Triopikeur|Earthquake|p2a: Tar"], "p1")
    assert r.moves_of("dugtrio") == {"earthquake"}
    assert r.moves_of("triopikeur") == set()


def test_struggle_is_never_a_set_move():
    r = reveals("|move|p1a: Sal|Struggle|p2a: Tar")
    assert r.moves_of("salamence") == set()


def test_sleep_talk_reveals_BOTH_itself_and_its_callee():
    """gen3 Sleep Talk draws from the USER's own set, so the callee is a real slot."""
    r = reveals("|move|p1a: Sal|Dragon Claw|p2a: Tar|[from] Sleep Talk")
    assert r.moves_of("salamence") == {"sleeptalk", "dragonclaw"}


def test_metronome_reveals_ONLY_metronome():
    """Metronome picks from the whole move pool — the callee says nothing about the set."""
    r = reveals("|move|p1a: Sal|Fissure|p2a: Tar|[from] Metronome")
    assert r.moves_of("salamence") == {"metronome"}


def test_mirror_move_reveals_ONLY_mirror_move():
    """Mirror Move copies the TARGET's last move; it is not ours."""
    r = reveals("|move|p1a: Sal|Rock Slide|p2a: Tar|[from] Mirror Move")
    assert r.moves_of("salamence") == {"mirrormove"}


def test_a_locked_move_continuation_still_reveals_the_move():
    """``[from] lockedmove`` marks a multi-turn move the user genuinely selected."""
    r = reveals("|move|p1a: Sal|Outrage|p2a: Tar|[from] lockedmove")
    assert "outrage" in r.moves_of("salamence")


def test_pursuit_self_tag_reveals_pursuit_once():
    """Pursuit hitting a switching target tags its own line ``[from] Pursuit`` — a
    same-move source, not a call."""
    r = track_own_reveals(
        ["|switch|p1a: Tar|Tyranitar, L100, M|100/100",
         "|move|p1a: Tar|Pursuit|p2a: X|[from] Pursuit"], "p1")
    assert r.moves_of("tyranitar") == {"pursuit"}


def test_reveals_accumulate_across_switch_outs_and_ins():
    r = reveals(
        "|move|p1a: Sal|Dragon Claw|p2a: Tar",
        "|switch|p1a: Sui|Suicune, L100|100/100",
        "|move|p1a: Sui|Surf|p2a: Tar",
        "|switch|p1a: Sal|Salamence, L100, M|80/100",
        "|move|p1a: Sal|Brick Break|p2a: Tar",
    )
    assert r.moves_of("salamence") == {"dragonclaw", "brickbreak"}
    assert r.moves_of("suicune") == {"surf"}


# --------------------------------------------------------------------------- #
# items
# --------------------------------------------------------------------------- #


def test_an_unactivated_item_is_NOT_revealed():
    r = reveals("|move|p1a: Sal|Dragon Claw|p2a: Tar")
    assert not r.item_shown("salamence")


def test_a_consumed_berry_reveals_the_item():
    r = reveals("|-enditem|p1a: Sal|Liechi Berry|[eat]")
    assert r.item_shown("salamence")


def test_leftovers_healing_reveals_the_item_through_the_from_clause():
    r = reveals("|-heal|p1a: Sal|94/100|[from] item: Leftovers")
    assert r.item_shown("salamence")


def test_the_modern_unspaced_from_clause_spelling_also_reveals():
    r = reveals("|-heal|p1a: Sal|94/100|[from]item: Leftovers")
    assert r.item_shown("salamence")


def test_an_item_given_by_trick_is_revealed():
    r = reveals("|-item|p1a: Sal|Choice Band|[from] move: Trick")
    assert r.item_shown("salamence")


def test_knock_off_on_the_OPPONENT_does_not_reveal_ours():
    """The actor of the ``-enditem`` is the mon LOSING the item; an ``[of]`` clause naming
    us is the aggressor, not a disclosure of our own item."""
    r = reveals("|-enditem|p2a: Tar|Leftovers|[from] move: Knock Off|[of] p1a: Sal")
    assert not r.item_shown("salamence")


def test_knock_off_on_US_does_reveal_ours():
    r = reveals("|-enditem|p1a: Sal|Leftovers|[from] move: Knock Off|[of] p2a: Tar")
    assert r.item_shown("salamence")


def test_an_item_activate_line_reveals():
    r = reveals("|-activate|p1a: Sal|item: Focus Band")
    assert r.item_shown("salamence")


def test_an_ability_from_clause_does_not_reveal_an_item():
    r = reveals("|-weather|Sandstorm|[from] ability: Sand Stream|[of] p1a: Sal")
    assert not r.item_shown("salamence")


# --------------------------------------------------------------------------- #
# side orientation + spreads
# --------------------------------------------------------------------------- #


def test_our_tag_p2_flips_which_side_is_tracked():
    lines = [P1_LEAD, P2_LEAD, "|move|p2a: Tar|Rock Slide|p1a: Sal",
             "|-enditem|p2a: Tar|Lum Berry"]
    r = track_own_reveals(lines, "p2")
    assert r.moves_of("tyranitar") == {"rockslide"}
    assert r.item_shown("tyranitar")
    assert r.moves_of("salamence") == set()


def test_only_mons_that_appeared_are_seen():
    r = reveals("|switch|p1a: Sui|Suicune, L100|100/100")
    assert r.seen == {"salamence", "suicune"}


def test_a_spread_is_never_revealed_by_anything():
    """There is no protocol line that discloses EVs or nature — the meter's model of that
    is the absence of any field for it, and this test is what stops one being added."""
    r = reveals(
        "|move|p1a: Sal|Dragon Claw|p2a: Tar",
        "|-enditem|p1a: Sal|Liechi Berry|[eat]",
        "|-damage|p2a: Tar|41/100",
    )
    assert not hasattr(r, "spreads")
    assert set(vars(r)) == {"moves", "items", "seen"}


def test_non_protocol_and_short_lines_are_ignored_not_crashed():
    r = track_own_reveals([P1_LEAD, "", "not a protocol line", "|turn|", "|upkeep"], "p1")
    assert r.seen == {"salamence"}


# --------------------------------------------------------------------------- #
# the imputation itself (Naive-equivalent, from the live Smogon priors)
# --------------------------------------------------------------------------- #


def test_imputed_moves_are_learnset_legal_and_exclude_the_revealed_ones():
    got = impute_moves("snorlax", {"bodyslam"}, 3)
    assert len(got) == 3
    assert "bodyslam" not in got
    legal = __import__("agents.gen3_data", fromlist=["learnset"]).learnset
    for mid in got:
        assert legal.is_legal("snorlax", mid), mid


def test_imputing_zero_slots_returns_nothing():
    assert impute_moves("snorlax", set(), 0) == []


def test_the_top_item_prior_is_the_naive_item_guess():
    assert impute_item("snorlax") == "leftovers"


def test_an_unknown_species_imputes_nothing_rather_than_raising():
    assert impute_moves("notapokemon", set(), 4) == []
    assert impute_item("notapokemon") is None
    assert impute_spread("notapokemon") is None


def test_the_spread_guess_is_a_nature_plus_six_evs():
    got = impute_spread("snorlax")
    assert got is not None
    nature, evs = got
    assert nature.islower() and len(evs) == 6
    assert all(isinstance(e, int) for e in evs)


# --------------------------------------------------------------------------- #
# snapshot / restore — the meter mutates the LIVE battle, so the restore has to be total
# --------------------------------------------------------------------------- #


def _bare_mon(species: str = "snorlax"):
    from poke_env.battle.pokemon import Pokemon
    from poke_env.teambuilder.teambuilder_pokemon import TeambuilderPokemon
    mon = Pokemon(gen=3, species=species)
    mon._update_from_teambuilder(TeambuilderPokemon(
        species=species, item="leftovers", ability="immunity",
        moves=["bodyslam", "curse", "rest", "sleeptalk"],
        evs=[4, 252, 0, 0, 252, 0], ivs=[31] * 6, nature="Adamant", level=100))
    return mon


_MUTATED_FIELDS = ("_moves", "_item", "_consumed_item", "_ivs", "_evs", "_nature",
                   "_stats", "_max_hp", "_current_hp")


def test_the_snapshot_covers_every_field_the_imputation_writes():
    """A restore that misses one field contaminates every LATER decision's 'truth'. The
    probe's live gate re-encodes and compares; this pins the same claim structurally, so a
    new imputed field cannot be added without also being snapshotted."""
    from agents.training.replay_imputation_probe import _MonSnapshot
    snap_fields = {f"_{n}" for n in _MonSnapshot.__dataclass_fields__ if n != "mon"}
    assert snap_fields == set(_MUTATED_FIELDS)


def test_snapshot_then_restore_is_a_no_op_after_arbitrary_mutation():
    from poke_env.battle.move import Move, MoveSet
    from agents.training.replay_imputation_probe import _restore, _snapshot

    mon = _bare_mon()
    before = {f: repr(getattr(mon, f)) for f in _MUTATED_FIELDS}
    snap = _snapshot(mon)

    mon._moves = MoveSet({"tackle": Move("tackle", gen=3)})
    mon._item = "choiceband"
    mon._consumed_item = "lumberry"
    mon._ivs = [0] * 6
    mon._evs = [0] * 6
    mon._nature = "timid"
    mon._stats = {"hp": 1, "atk": 1, "def": 1, "spa": 1, "spd": 1, "spe": 1}
    mon._max_hp = 1
    mon._current_hp = 1
    assert any(repr(getattr(mon, f)) != before[f] for f in _MUTATED_FIELDS)

    _restore(snap)
    for f in _MUTATED_FIELDS:
        assert repr(getattr(mon, f)) == before[f], f


def test_the_snapshot_deep_copies_the_lists_it_holds():
    """``_ivs`` / ``_evs`` / ``_stats`` are mutable containers on the Pokémon; a shallow
    snapshot would hold the SAME object the imputation then overwrites in place, and the
    restore would put back the mutated value while looking correct."""
    from agents.training.replay_imputation_probe import _restore, _snapshot

    mon = _bare_mon()
    snap = _snapshot(mon)
    mon._evs[1] = 0
    mon._stats["atk"] = 0
    _restore(snap)
    assert mon._evs[1] == 252
    assert mon._stats["atk"] != 0


@pytest.mark.parametrize("turn,expect", [
    (1, "turns_1_5"), (5, "turns_1_5"), (6, "turns_6_15"),
    (15, "turns_6_15"), (16, "turns_16plus"), (200, "turns_16plus"),
])
def test_phase_buckets(turn, expect):
    assert phase_of(turn) == expect
