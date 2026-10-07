"""Team validation against the gen3ou format spec: a planted illegal team is flagged, by the right rule, and the
pools we train on are legal under the LADDER's rules (each test fails on revert of the rule it names)."""
from __future__ import annotations

import pytest

from agents.gen3_data import format_spec as fs
from agents.gen3_data import team_legality as tl

LEGAL = """Tyranitar @ Leftovers
Ability: Sand Stream
EVs: 252 HP / 252 Atk / 4 SpD
Adamant Nature
- Rock Slide
- Earthquake
- Pursuit
- Substitute

Skarmory @ Leftovers
Ability: Keen Eye
- Spikes
- Roar
- Toxic
- Drill Peck

Metagross (M) @ Choice Band
Ability: Clear Body
- Meteor Mash
- Earthquake
- Explosion
- Rock Slide

Jirachi @ Leftovers
Ability: Serene Grace
- Calm Mind
- Psychic
- Fire Punch
- Wish

Blissey (F) @ Leftovers
Ability: Natural Cure
- Seismic Toss
- Soft-Boiled
- Aromatherapy
- Toxic

Mr. Rime (Gengar) @ Leftovers
Ability: Levitate
- Thunderbolt
- Ice Punch
- Will-O-Wisp
- Hypnosis
"""


def _team(swap):
    """LEGAL with one set's fields replaced: swap = {slot: (species, item, ability, moves)}."""
    sets = tl.parse_paste(LEGAL)
    for slot, (sp, it, ab, mv) in swap.items():
        sets[slot] = tl.TeamSet(sp, it, ab, tuple(mv))
    return sets


def _rules(team):
    return sorted((p.rule, p.slot, p.ladder_only) for p in tl.validate_team(team))


def test_the_paste_parser_reads_nicknames_gender_and_items():
    sets = tl.parse_paste(LEGAL)
    assert [s.species for s in sets] == ["tyranitar", "skarmory", "metagross", "jirachi", "blissey", "gengar"]
    assert sets[2].item == "choiceband" and sets[5].ability == "levitate"
    assert sets[4].moves == ("seismictoss", "softboiled", "aromatherapy", "toxic")
    assert tl.validate_team(sets) == []


def test_the_packed_parser():
    packed = ("|tyranitar|leftovers|sandstream|rockslide,earthquake|Adamant|252,252,,,4,|||||]"
              "Nick|skarmory|quickclaw|keeneye|spikes,roar|Impish|||||")
    sets = tl.parse_packed(packed)
    assert [(s.species, s.item) for s in sets] == [("tyranitar", "leftovers"), ("skarmory", "quickclaw")]


@pytest.mark.parametrize("swap,want", [
    ({1: ("skarmory", "quickclaw", "keeneye", ("spikes", "roar"))}, [("banlist", 1, True)]),
    ({1: ("skarmory", "brightpowder", "keeneye", ("spikes", "roar"))}, [("banlist", 1, False)]),
    ({0: ("tyranitar", "leftovers", "sandstream", ("swagger", "earthquake"))}, [("banlist", 0, False)]),
    ({0: ("tyranitar", "leftovers", "sandstream", ("fissure",))}, [("banlist", 0, False)]),
    ({3: ("mewtwo", "leftovers", "pressure", ("psychic",))}, [("Uber", 3, False)]),
    ({3: ("exploud", "leftovers", "soundproof", ("return",))},
     [("Obtainable Abilities", 3, False), ("banlist", 3, False)]),
    ({3: ("smeargle", "leftovers", "owntempo", ("ingrain", "spore"))}, [("banlist", 3, False)]),
    ({3: ("umbreon", "leftovers", "synchronize", ("batonpass", "meanlook"))},
     [("Baton Pass + Mean Look", 3, False)]),
    ({3: ("dugtrio", "choiceband", "arenatrap", ("earthquake", "sandattack"))},
     [("Accuracy Trap Clause", 3, False)]),
    ({3: ("ninjask", "leftovers", "speedboost", ("batonpass", "protect"))}, [("Speed Pass Clause", 3, False)]),
    ({3: ("blissey", "leftovers", "naturalcure", ("softboiled",))}, [("Species Clause", 4, False)]),
])
def test_a_planted_illegal_set_is_flagged_by_its_rule(swap, want):
    assert _rules(_team(swap)) == sorted(want)


def test_one_boost_passer_clause_and_the_master_only_recycle():
    two = {3: ("celebi", "leftovers", "naturalcure", ("batonpass", "calmmind")),
           1: ("vaporeon", "leftovers", "waterabsorb", ("batonpass", "acidarmor"))}
    assert _rules(_team(two)) == [("One Boost Passer Clause", 3, False)]
    multi = {3: ("celebi", "leftovers", "naturalcure", ("batonpass", "calmmind", "swordsdance"))}
    assert _rules(_team(multi)) == [("One Boost Passer Clause", 3, False)]
    # Recycle counts on MASTER only (PINNED_DIFFERENCES): a passer of CM + Recycle is ladder-illegal alone
    recycle = {3: ("celebi", "leftovers", "naturalcure", ("batonpass", "calmmind", "recycle"))}
    assert _rules(_team(recycle)) == [("One Boost Passer Clause", 3, True)]


def test_a_spec_without_the_clause_does_not_flag_it():
    sp = {3: ("ninjask", "leftovers", "speedboost", ("batonpass", "protect"))}
    spec = fs.without_rule(fs.GEN3OU, "speedpassclause")
    assert tl.validate_team(_team(sp), spec) == []


def test_the_training_pool_is_legal_on_the_ladder():
    """Every team TeamLoader serves (data/teams, `valid` not false) passes the LADDER's format rules — a newly
    added illegal team fails here; the full per-pool report is `python -m main.team_legality`."""
    from main import team_legality as cli
    recs = cli.scan()
    training = [r for r in recs if r.in_training]
    assert len(training) >= 700
    bad = [(r.path, r.problems) for r in training if r.illegal]
    assert bad == []
    assert all(not r.illegal for r in recs if r.pool in ("smogon_sample", "specialist"))
