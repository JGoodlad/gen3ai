"""The packed-set reader: a name variant in a team is read as the move it is (GIGO 2026-10-15: one pool team packs
Will-O-Wisp as ``wisp``), a token the gen-3 data cannot resolve THROWS, and the whole training pool reads clean."""
from collections import Counter

import pytest

from agents.info_levers import packed as P

GENGAR_WISP = "Gengar||leftovers|levitate|thunderbolt,icepunch,wisp,explosion|Timid|,,,252,4,252|M||||"
GENGAR_CANON = GENGAR_WISP.replace("wisp", "willowisp")


def test_a_name_variant_is_read_as_the_move_it_is():
    f = P.parse_set(GENGAR_WISP)
    assert P.typed_moves(f) == ["thunderbolt", "icepunch", "willowisp", "explosion"]
    assert P.typed_moves(f) == P.typed_moves(P.parse_set(GENGAR_CANON))
    # a typed shorthand keeps its type; a bare Hidden Power is typed by the IVs (a 30/30 Atk/SpA set is Grass)
    g = P.parse_set("Jolteon||leftovers|voltabsorb|thunderbolt,hpice,hiddenpower,batonpass|Timid|||31,30,31,30,31,31|||")
    assert P.typed_moves(g)[1] == "hiddenpowerice" and P.typed_moves(g)[2] == "hiddenpowergrass"


def test_the_id_readers_are_one_definition():
    f = P.parse_set("Mr. Mime Nick|Starmie|Leftovers|Natural Cure|surf|Timid|||||||")
    assert (P.species_id(f), P.item_id(f), P.ability_id(f)) == ("starmie", "leftovers", "naturalcure")
    from utils.showdown_id import to_id_str

    assert P.toid("King's Rock") == to_id_str("King's Rock") == "kingsrock"
    assert P.species_id(P.parse_set("Snorlax|||||curse||||||")) == "snorlax"      # an empty species field: the name


@pytest.mark.parametrize("entry, needle", [
    ("Gengar||cb|levitate|icepunch|Timid|||||||", "item 'cb'"),                  # an item name variant (Choice Band)
    ("Gengar||leftovers|intim|icepunch|Timid|||||||", "ability 'intim'"),        # an ability name variant
    ("Deon||leftovers|pressure|recover|Timid|||||||", "species 'deon'"),          # a species name variant (Deoxys)
    ("Gengar||leftovers|levitate|icepunch,notamove|Timid|||||||", "move 'notamove'"),
])
def test_an_id_the_data_does_not_know_throws(entry, needle):
    assert any(needle in u for u in P.unknown_ids(P.parse_set(entry)))
    with pytest.raises(P.PackedError, match=needle):
        P.require_known("Skarmory||leftovers|keeneye|spikes|Impish|||||||]" + entry, "k")


def test_a_team_with_a_move_variant_is_known_and_a_clean_team_passes():
    P.require_known(GENGAR_WISP)               # the variant RESOLVES: not an unknown id
    assert P.unknown_ids(P.parse_set(GENGAR_WISP)) == []
    P.require_known("Skarmory||leftovers|keeneye|spikes,whirlwind,drillpeck,rest|Impish|||||||")


def test_the_whole_training_pool_reads_clean_and_its_only_move_variant_is_the_one_known():
    """THE CENSUS (719 teams, 2026-10-15): every species / item / ability / move token of every pool team is a
    gen-3 id once moves are resolved, and the single name variant in the pool is Gengar's ``wisp`` (team 555, from
    ``data/teams/others/giraffe/dc99ab0afa2b1da9.txt``). A new variant appearing is a deliberate decision: read it
    through ``packed``, then update this count."""
    from utils.team_loader import TeamLoader
    from utils.team_packing import ConstantTeambuilder

    raw = TeamLoader().get_all_teams()
    assert len(raw) == 719
    variants: Counter = Counter()
    for r in raw:
        team = ConstantTeambuilder(r).yield_team()
        P.require_known(team, "pool")
        for e in team.split("]"):
            for tok in P.parse_set(e)[4].split(","):
                if tok and P.toid(tok) != P.move_id(tok):
                    variants[P.toid(tok)] += 1
    assert variants == Counter({"wisp": 1})
