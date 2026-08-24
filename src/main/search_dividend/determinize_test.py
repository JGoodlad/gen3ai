"""Determinization VALIDITY — a sampled world must be consistent with everything revealed.

The pure tests pin the rules; the ``sim``-marked one pins the CLAIM, against a real battle: play
a reproducible bridge battle, sample worlds from the opponent's actual mid-battle information set,
and assert that every world (a) keeps every revealed species, (b) changes at least one hidden one,
and (c) REPRODUCES the our-side protocol prefix byte-for-byte through the real replay driver.

(c) is the load-bearing one and it is what makes the honest arm a measurement rather than a hope.
The hidden-info-floor precedent ran this gate over 535 and 615 worlds with zero mismatches, so a
failure here is news.
"""

from __future__ import annotations

import random

import pytest

from main.search_dividend import determinize as dz

# Hand-built packed mons. The field order is the real one and the INDICES matter — gender is
# field 7, so an off-by-one here silently tests the IVs field instead (which is how the first
# draft of this file passed a gender assertion it was not making).
# name|species|item|ability|moves|nature|evs|gender|ivs|shiny|level|happiness
def _mon(species, moves="tackle", gender=""):
    return f"|{species}|leftovers|pressure|{moves}|Serious||{gender}|||||"


def _team(*species):
    return "]".join(_mon(s) for s in species)


GENDER = {"skarmory": "", "magneton": "N", "blissey": "F", "celebi": "N",
          "salamence": "", "tyranitar": "", "swampert": "", "milotic": ""}


def _split(packed):
    return dz.split_team(packed, GENDER)


# -- packed-team surgery ------------------------------------------------------


def test_split_join_is_lossless():
    packed = _team("skarmory", "blissey", "celebi")
    assert dz.join_team(_split(packed)) == packed


def test_species_is_read_from_the_species_field():
    (m,) = _split(_mon("skarmory"))
    assert m.species == "skarmory"


def test_gender_draw_count_is_what_rolls_gender_tracks():
    """`Pokemon`'s constructor samples ['M','F'] only for a blank packed gender on a species with
    no fixed dex gender — the ONE construction draw whose count depends on WHICH mon fills a
    slot. Get this wrong and the whole downstream dice stream shifts."""
    (skarm,) = _split(_mon("skarmory"))              # no dex gender, blank field -> rolls
    (magne,) = _split(_mon("magneton"))              # genderless -> no roll
    (pinned,) = _split(_mon("skarmory", gender="M"))  # explicit -> no roll
    assert skarm.rolls_gender is True
    assert magne.rolls_gender is False
    assert pinned.rolls_gender is False


def test_a_donor_is_gender_matched_to_keep_the_draw_count_identical():
    (roller,) = _split(_mon("skarmory"))
    (genderless,) = _split(_mon("magneton"))
    # replacing a ROLLER: a genderless donor cannot match, an ambiguous one is blanked
    assert dz._gender_matched(genderless, roller, GENDER) is None
    (other_roller,) = _split(_mon("salamence"))
    assert dz._gender_matched(other_roller, roller, GENDER).fields[7] == ""
    # replacing a NON-roller: an ambiguous donor must be PINNED so it does not start rolling
    assert dz._gender_matched(other_roller, genderless, GENDER).fields[7] == "M"
    assert dz._gender_matched(genderless, genderless, GENDER).fields[7] == ""


# -- revealed detection -------------------------------------------------------

_LINES = [
    "|t:|1700000000",
    "|switch|p2a: Steel Bird|Skarmory, M|100/100",
    "|switch|p1a: Mence|Salamence, F|100/100",
    "|turn|1",
    "|move|p2a: Steel Bird|Spikes|p1a: Mence",
    "|turn|2",
    "|switch|p2a: Egg|Blissey, F|100/100",
    "|move|p2a: Egg|Seismic Toss|p1a: Mence",
    "|turn|3",
]


def test_revealed_species_reads_DETAILS_not_the_nickname():
    """This pool carries localized nicknames (Triopikeur = Dugtrio); a nickname-keyed read names
    nothing at all."""
    assert dz.revealed_species(_LINES, "p2") == {"skarmory", "blissey"}
    assert dz.revealed_species(_LINES, "p1") == {"salamence"}


def test_used_moves_track_the_nickname_to_species_binding():
    used = dz.used_moves_by_species(_LINES, "p2")
    assert used == {"skarmory": {"spikes"}, "blissey": {"seismictoss"}}


def test_hidden_power_normalizes_onto_the_bare_id():
    """The protocol says `Hidden Power` while the packed set says `hiddenpowerflying`, so an
    id-equality 'was this used?' test answers NO for every HP ever thrown. The precedent measured
    440 of 615 worlds failing the gate on exactly this."""
    assert dz.norm_move("hiddenpowerflying") == "hiddenpower"
    assert dz.norm_move("hiddenpower") == "hiddenpower"
    assert dz.norm_move("earthquake") == "earthquake"


def test_strip_ts_removes_only_the_wall_clock_lines():
    assert dz.strip_ts(_LINES)[0] == "|switch|p2a: Steel Bird|Skarmory, M|100/100"
    assert len(dz.strip_ts(_LINES)) == len(_LINES) - 1


# -- the determinization itself -----------------------------------------------


def _pool_mons():
    return [_split(_team("skarmory", "blissey", "tyranitar")),
            _split(_team("skarmory", "blissey", "swampert")),
            _split(_team("magneton", "milotic", "celebi")),
            _split(_team("skarmory", "celebi", "milotic"))]


def test_every_revealed_slot_survives_and_only_hidden_ones_move():
    """THE consistency property: a world may disagree with the truth only where we have not
    looked."""
    base = _split(_team("skarmory", "blissey", "salamence"))
    revealed = {"skarmory", "blissey"}
    dets, stats = dz.build_determinizations(base, revealed, _pool_mons(), GENDER,
                                            k=3, rng=random.Random(0))
    assert stats["n_hidden"] == 1 and stats["hidden_slots"] == [2]
    assert dets, "the pool should be able to complete one hidden slot"
    for d in dets:
        species = [m.species for m in _split(d["packed"])]
        assert species[0] == "skarmory" and species[1] == "blissey"
        assert len(set(species)) == 3, "the species clause must hold in every world"


def test_tier_1_donors_are_preferred_and_the_tier_is_reported():
    """Tier 1 = donors whose roster CONTAINS every revealed species — uniform over those IS the
    posterior, since eval draws the opponent's team uniformly from this pool. A world built from
    a weaker tier is still usable but must say so."""
    base = _split(_team("skarmory", "blissey", "salamence"))
    dets, stats = dz.build_determinizations(base, {"skarmory", "blissey"}, _pool_mons(), GENDER,
                                            k=4, rng=random.Random(1))
    assert stats["tier1_donors"] == 2
    assert dets[0]["tier"] == 1


def test_no_hidden_slots_yields_no_worlds():
    base = _split(_team("skarmory", "blissey"))
    dets, stats = dz.build_determinizations(base, {"skarmory", "blissey"}, _pool_mons(), GENDER,
                                            k=4, rng=random.Random(2))
    assert dets == [] and stats["n_hidden"] == 0


def test_worlds_are_DISTINCT_completions():
    base = _split(_team("skarmory", "salamence", "tyranitar"))
    dets, _ = dz.build_determinizations(base, {"skarmory"}, _pool_mons(), GENDER,
                                        k=5, rng=random.Random(3))
    sigs = [tuple(d["hidden_species"]) for d in dets]
    assert len(sigs) == len(set(sigs)), "a repeated world would double-count in the average"


def test_exclude_true_drops_the_real_completion():
    """The FLOOR probe wanted alternatives only; a SEARCH wants the posterior it believes, so the
    default keeps the truth in the sample. Both behaviours have to be reachable and distinct."""
    base = _split(_team("skarmory", "blissey", "tyranitar"))
    kw = dict(revealed={"skarmory"}, pool_teams=_pool_mons(), gender_tbl=GENDER, k=9)
    with_true = dz.build_determinizations(base, rng=random.Random(4), **kw)[0]
    without = dz.build_determinizations(base, rng=random.Random(4), exclude_true=True, **kw)[0]
    true_sig = ["blissey", "tyranitar"]
    assert any(sorted(d["hidden_species"]) == true_sig for d in with_true)
    assert all(sorted(d["hidden_species"]) != true_sig for d in without)


def test_swap_unused_moves_never_touches_a_move_that_was_USED():
    """Axis M is only safe because nothing in the one-sided stream depends on an unused move. A
    USED move that got swapped would invalidate the recorded command that threw it and stall the
    reconstruction outright."""
    team = _split("|Skarmory|leftovers|pressure|spikes,protect,toxic,drillpeck|Serious|||||")
    bank = {"skarmory": [["roar", "whirlwind", "rest", "curse"]]}
    out, n = dz.swap_unused_moves(team, {"skarmory"}, {"skarmory": {"spikes"}}, bank,
                                  random.Random(0))
    moves = out[0].moves
    assert moves[0] == "spikes", "the USED move must survive verbatim"
    assert n == 3 and moves[1:] != ["protect", "toxic", "drillpeck"]


def test_swap_unused_moves_never_injects_a_hidden_power():
    """HP's type rides the set's IVs, which stay fixed here — injecting one builds a set we did
    not mean to build."""
    team = _split("|Skarmory|leftovers|pressure|spikes,protect|Serious|||||")
    bank = {"skarmory": [["hiddenpowerfire", "roar"]]}
    out, _ = dz.swap_unused_moves(team, {"skarmory"}, {"skarmory": {"spikes"}}, bank,
                                  random.Random(0))
    assert not any(m.startswith("hiddenpower") for m in out[0].moves)


def test_an_unrevealed_mon_keeps_its_moves_under_axis_M():
    team = _split(_team("skarmory", "blissey"))
    out, n = dz.swap_unused_moves(team, {"skarmory"}, {}, {"blissey": [["softboiled"]]},
                                  random.Random(0))
    assert out[1].moves == ["tackle"] and out[1].species == "blissey"


# -- record surgery -----------------------------------------------------------


def test_record_with_team_touches_only_the_target_player_line():
    from utils.bridge.reconstruction import ReconstructionRecord

    rec = ReconstructionRecord(
        format_id="gen3ou", prng_seed="sodium,ab",
        input_log=('>start {"formatid":"gen3ou","seed":"sodium,ab"}',
                   '>player p1 {"name": "A", "team": "OLD1"}',
                   '>player p2 {"name": "B", "team": "OLD2"}'),
        commands=(("p1", "move 1"), ("p2", "switch 2")), battle_tag="battle-1")
    out = dz.record_with_team(rec, "p2", "NEW2", "-w0")
    assert out.packed_team("p2") == "NEW2"
    assert out.packed_team("p1") == "OLD1"
    assert out.start_options()["seed"] == "sodium,ab"
    assert out.commands == rec.commands, "the commands' referents are unchanged"
    assert out.battle_tag == "battle-1-w0", "each world needs its own post-divergence dice salt"
    assert rec.packed_team("p2") == "OLD2", "the source record must not be mutated"


def test_prefix_matches_ignores_only_the_timestamp_lines():
    observed = ["|switch|p2a: X|Skarmory, M|100/100", "|turn|1"]
    assert dz.prefix_matches(observed, ["|t:|17\n|switch|p2a: X|Skarmory, M|100/100\n|turn|1"])
    assert not dz.prefix_matches(observed, ["|switch|p2a: X|Skarmory, M|99/100\n|turn|1"])


def test_the_gate_is_TURN_SCOPED_on_both_sides():
    """The live stream has run past the decision (it carries this turn's request) while a search
    root stops at the start of turn T. Without scoping, the gate would fire on trailing content
    that is not a disagreement — and a gate that fires on everything is the same as no gate,
    because the arm then falls back on every decision and silently becomes the control."""
    live = ["|switch|p2a: X|Skarmory, M|100/100", "|turn|1", "|move|p1a: Y|Surf|p2a: X",
            "|turn|2", "|request|{...}"]
    root = ["|switch|p2a: X|Skarmory, M|100/100\n|turn|1"]
    assert not dz.prefix_matches(live, root)              # unscoped: trailing content differs
    assert dz.prefix_matches(live, root, turn=1)          # scoped: identical through turn 1


def test_a_missing_turn_marker_is_a_MISMATCH_not_a_pass():
    """Failing open here would keep a world whose replay never reached the decision at all."""
    live = ["|switch|p2a: X|Skarmory, M|100/100", "|turn|1", "|turn|2"]
    assert not dz.prefix_matches(live, ["|switch|p2a: X|Skarmory, M|100/100\n|turn|1"], turn=2)
    assert not dz.prefix_matches(live, ["|turn|9"], turn=9)


def test_turn_markers_match_EXACTLY_never_by_prefix():
    lines = ["|turn|1", "|turn|12", "|turn|2"]
    assert dz.turn_marker_index(lines, 1) == 0
    assert dz.turn_marker_index(lines, 2) == 2, "|turn|1 must not be found inside |turn|12"
    assert dz.turn_marker_index(lines, 3) is None
    assert dz.prefix_through_turn(lines, 12) == ["|turn|1", "|turn|12"]


# -- the real thing -----------------------------------------------------------


@pytest.mark.sim
def test_a_sampled_world_reproduces_a_REAL_battles_prefix_byte_for_byte(tmp_path):
    """The claim, against a real battle: replacing the opponent's never-revealed slots leaves our
    entire observed protocol prefix unchanged.

    Reproducible by construction (a fixed key, fixed teams, a fixed sim seed) — a wall-clock-
    seeded fixture eventually draws the battle that skips its own assertion, which this tree has
    already chased three times as a flake."""
    from agents.training.obs_roundtrip_fuzz_test import record_fixture_battle
    from utils.bridge.reconstruction import replay_battle

    pool, gender = dz.load_pool()
    pool_mons = [dz.split_team(p, gender) for p in pool]

    # The information set is TURN-SCOPED, exactly as the live search's is: by the end of a
    # random-vs-random game the opponent has usually shown all six, so an end-of-battle reveal
    # set has nothing left to determinize. Pick the earliest turn that still hides a mon.
    # Redrawing over a BOUNDED FIXED key sequence keeps this reproducible across processes, and
    # failing (rather than skipping) when none qualifies keeps it from passing vacuously.
    record = our = opp = our_lines = revealed = base = hidden = None
    turn = 0
    for key in (3, 5, 8, 11):
        rec, _summary, _npz = record_fixture_battle(str(tmp_path), key=key, tag=f"SDdz{key}")
        side = rec.side_of(rec.trainee_username)
        other = "p2" if side == "p1" else "p1"
        truth = replay_battle(rec, impl="node")
        lines = dz.chunks_to_lines(truth.p1_chunks if side == "p1" else truth.p2_chunks)
        team = dz.split_team(rec.packed_team(other), gender)
        for t in range(2, 12):
            upto = dz.prefix_through_turn(lines, t)
            if upto is None:
                break
            seen = dz.revealed_species(upto, other)
            unseen = {m.species for m in team} - seen
            if seen and unseen:
                record, our, opp, turn = rec, side, other, t
                our_lines, revealed, base, hidden = lines, seen, team, unseen
                break
        if record is not None:
            break
    assert record is not None, (
        "no fixture battle in keys (3, 5, 8, 11) had a turn with the opponent partly hidden — "
        "this test cannot be run vacuously, so widen the key sequence rather than skipping")

    dets, stats = dz.build_determinizations(base, revealed, pool_mons, gender, k=4,
                                            rng=random.Random(7), exclude_true=True)
    assert dets, f"no world built from a {len(pool)}-team pool (stats={stats})"

    # The gate runs through `open_root`, i.e. THE EXACT PATH the live search uses, not through a
    # full `replay_battle`. That distinction is a finding, not a detail: a determinized world is
    # only claimed to reproduce the PREFIX, and a full replay of one legitimately diverges the
    # moment a swapped mon comes in — measured here as "replayed all 239 commands but battle has
    # not ended (turn 40)". `buildToTurn` stops at the decision turn, which is the whole scope of
    # the claim.
    from utils.bridge.search_session import SearchSession

    with SearchSession(impl="node") as ss:
        for i, d in enumerate(dets):
            world_species = {m.species for m in dz.split_team(d["packed"], gender)}
            # (a) every revealed species survives
            assert revealed <= world_species, f"world {i} lost a revealed mon"
            # (b) the world genuinely differs where we have not looked
            assert world_species - revealed != hidden, f"world {i} is the truth in disguise"
            # (c) THE GATE — the root's our-side protocol is byte-for-byte what we observed.
            wrec = dz.record_with_team(record, opp, d["packed"], f"-w{i}")
            root = ss.open_root(turn, record=wrec)
            chunks = root.prefix_p1_chunks if our == "p1" else root.prefix_p2_chunks
            assert dz.prefix_matches(our_lines, chunks, turn=turn), (
                f"world {i} (tier {d['tier']}, hidden={d['hidden_species']}) changed the our-side "
                f"protocol through turn {turn} — the determinization is NOT information-set "
                "consistent")
