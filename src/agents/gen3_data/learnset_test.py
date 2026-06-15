"""Unit tests for the gen3 legal-movepool facade (`gen3_data.learnset`)."""
import pytest

from agents import gen3_data
from agents.gen3_data import learnset


def test_known_species_legal_and_illegal_moves():
    # Skarmory: Spikes / Drill Peck / Roar are its staples; it cannot learn Fire Blast or Surf.
    assert learnset.is_legal("skarmory", "spikes")
    assert learnset.is_legal("skarmory", "drillpeck")
    assert not learnset.is_legal("skarmory", "fireblast")
    assert not learnset.is_legal("skarmory", "surf")
    # Snorlax: Body Slam / Earthquake / Rest legal; Spikes illegal.
    assert learnset.is_legal("snorlax", "bodyslam")
    assert learnset.is_legal("snorlax", "earthquake")
    assert not learnset.is_legal("snorlax", "spikes")


def test_hidden_power_is_legal_for_hp_users():
    # HP users carry the bare 'hiddenpower' learnset entry (the typed variant is an IV choice).
    assert learnset.is_legal("zapdos", "hiddenpower")


def test_get_legal_moves_returns_set_for_known_species():
    moves = learnset.get_legal_moves("salamence")
    assert moves is not None
    assert "fireblast" in moves and "brickbreak" in moves
    assert "spikes" not in moves
    # Every legal id is a real gen3 move (no later-gen / typo leak through the builder).
    for mid in moves:
        assert gen3_data.moves.get(mid) is not None


def test_unknown_species_is_tolerant_no_constraint():
    # The tolerance contract: unknown species → None / True, so the gate never wrongly prunes.
    assert learnset.get_legal_moves("notarealmon") is None
    assert learnset.is_legal("notarealmon", "anything")


def test_legal_moves_for_species_raises_on_unknown():
    with pytest.raises(KeyError):
        learnset.legal_moves_for_species("notarealmon")


def test_full_gen3_pokedex_covered():
    # Every gen3 species with a movepool should be present (386 base forms in the pokedex).
    raw = learnset.raw()
    assert len(raw) == 386
    # A representative sample of OU mons all have non-empty movepools.
    for sid in ("tyranitar", "blissey", "metagross", "suicune", "gengar"):
        assert len(raw[sid]) > 0
