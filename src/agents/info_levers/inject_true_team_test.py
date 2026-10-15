"""``TrueTeam.of`` reads the opponent's packed team through ``packed``: a move name variant resolves to the move's num
(it used to raise ``unknown move 'wisp'`` on the one pool team that packs Will-O-Wisp so), and an item the data cannot
resolve THROWS instead of reading as "no item" (num 0)."""
import pytest

from agents.gen3_data import moves
from agents.info_levers import inject as I
from agents.info_levers.packed import PackedError

GENGAR = "Gengar||leftovers|levitate|thunderbolt,icepunch,{wisp},explosion|Timid|,,,252,4,252|M||||"


def test_a_move_name_variant_has_the_moves_num():
    a = I.TrueTeam.of(GENGAR.format(wisp="wisp"))
    b = I.TrueTeam.of(GENGAR.format(wisp="willowisp"))
    assert a == b
    (mon,) = a.mons.values()
    assert int(moves.get("willowisp").num) in mon.moves


def test_an_item_the_data_cannot_resolve_throws_rather_than_reading_as_no_item():
    with pytest.raises(PackedError, match="item 'cb'"):
        I.TrueTeam.of(GENGAR.format(wisp="wisp").replace("leftovers", "cb"))
