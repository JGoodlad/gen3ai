"""Tests for the Pokémon data extractor.

Guard two invariants for Gen 3:
  - abilities/moves reproduce the committed data/pokemon/ mappings (so a
    regeneration is a no-op diff aside from intentional field additions), and
  - every move carries an `accuracy` value in the expected shape.

The abilities builder reads the Showdown source tree (deps/pokemon-showdown),
so those tests are marked `integration`; the moves builder only needs the
poke-env static data shipped in src/, so its tests are plain unit tests.
"""

import json
import os

import pytest

# Fully-qualified import (tools is a PEP 420 namespace package; repo root is on sys.path under
# `python -m pytest`) so the module is `tools.pokemon_data_extractor.sync` and never collides
# with the other tools' identically-named `sync.py` during a combined collection.
import tools.pokemon_data_extractor.sync as sync

DATA_DIR = os.path.join(sync.REPO_ROOT, "data", "pokemon")


def _committed(name):
    with open(os.path.join(DATA_DIR, name)) as f:
        return json.load(f)


@pytest.mark.integration
def test_abilities_reproduce_committed_gen3():
    built = sync.build_abilities(3)
    committed = _committed("gen3_abilities.json")
    # Order matters: the encoder relies on a stable id ordering.
    assert list(built.items()) == list(committed.items())


@pytest.mark.integration
def test_tangled_feet_excluded_from_gen3():
    # Tangled Feet (num 77) is the first Gen 4 ability; the live pokedex lists it
    # on the Pidgey line, so the gen ceiling must keep it out.
    assert "tangledfeet" not in sync.build_abilities(3)


def test_moves_match_committed_base_fields():
    built = sync.build_moves(3)
    committed = _committed("gen3_moves.json")
    assert list(built) == list(committed)  # same ids, same order
    accuracy_fields = {"accuracy", "never_miss"}
    for move_id, entry in built.items():
        stripped = {k: v for k, v in entry.items() if k not in accuracy_fields}
        committed_stripped = {k: v for k, v in committed[move_id].items() if k not in accuracy_fields}
        assert stripped == committed_stripped, move_id


def test_every_move_has_numeric_accuracy_and_flag():
    built = sync.build_moves(3)
    for move_id, entry in built.items():
        acc = entry["accuracy"]
        # Always a numeric percentage now; never-miss moves are 100 + the flag.
        assert isinstance(acc, int) and not isinstance(acc, bool), (move_id, acc)
        assert 0 < acc <= 100, (move_id, acc)
        assert isinstance(entry["never_miss"], bool), (move_id, entry["never_miss"])


def test_known_accuracy_values():
    built = sync.build_moves(3)
    assert built["hydropump"]["accuracy"] == 80
    assert built["hydropump"]["never_miss"] is False
    assert built["thunder"]["accuracy"] == 70
    # Never-miss: numeric accuracy 100 plus the flag.
    assert built["swift"]["accuracy"] == 100
    assert built["swift"]["never_miss"] is True


# --- gen3_learnset_prevo_chain_v1: the learnset is the union over the PRE-EVOLUTION chain ----------- #
# (These read only the frozen poke-env static data, so they are plain unit tests.)

def _static_pokedex_and_species():
    with open(sync._static("pokedex", "gen3pokedex.json")) as f:
        pokedex = json.load(f)
    legal = {sid for sid, e in sync.build_species(3).items() if "baseSpecies" not in e}
    return pokedex, legal


def test_prevo_chain_follows_the_pokedex_and_stops_at_a_later_generation():
    pokedex, legal = _static_pokedex_and_species()
    assert sync._prevo_chain("tyranitar", pokedex, legal) == ["tyranitar", "pupitar", "larvitar"]
    assert sync._prevo_chain("azumarill", pokedex, legal) == ["azumarill", "marill", "azurill"]
    assert sync._prevo_chain("wobbuffet", pokedex, legal) == ["wobbuffet", "wynaut"]
    assert sync._prevo_chain("larvitar", pokedex, legal) == ["larvitar"]
    # Munchlax / Happiny / Chingling are gen 4: Showdown's `learnsetParent` returns null for `prevo.gen > 3`.
    assert sync._prevo_chain("snorlax", pokedex, legal) == ["snorlax"]
    assert sync._prevo_chain("blissey", pokedex, legal) == ["blissey", "chansey"]
    assert sync._prevo_chain("chimecho", pokedex, legal) == ["chimecho"]


def test_prevo_chain_survives_a_cycle():
    pokedex = {"a": {"prevo": "B"}, "b": {"prevo": "A"}}
    assert sync._prevo_chain("a", pokedex, {"a", "b"}) == ["a", "b"]


def test_built_learnset_is_the_union_over_the_chain():
    """Dragon Dance on Tyranitar (the named case): the species' own static entry has no gen-3 source for it, its
    pre-evolution Larvitar has `3E` + `3S0`."""
    with open(sync._static("learnset.json")) as f:
        raw = json.load(f)
    assert not any(str(c).startswith("3") for c in raw["tyranitar"]["learnset"]["dragondance"]), \
        "precondition: Tyranitar's OWN entry carries no gen-3 Dragon Dance source"
    assert {"3E", "3S0"} <= set(raw["larvitar"]["learnset"]["dragondance"])
    built = sync.build_learnset(3)
    assert "dragondance" in built["tyranitar"] and "dragondance" in built["pupitar"]
    pokedex, legal = _static_pokedex_and_species()
    for sid, moves in built.items():
        for link in sync._prevo_chain(sid, pokedex, legal):
            assert set(built[link]) <= set(moves), (sid, link)


def test_a_sketch_learner_gets_every_sketchable_move_and_never_a_typed_hidden_power():
    built = sync.build_learnset(3)
    with open(sync._static("moves", "gen3moves.json")) as f:
        static_moves = json.load(f)
    nosketch = {m for m, e in static_moves.items() if (e.get("flags") or {}).get("nosketch")}
    assert {"sketch", "struggle"} <= nosketch
    typed_hp = {m for m in sync.build_moves(3) if m.startswith("hiddenpower") and m != "hiddenpower"}
    assert len(typed_hp) == 16
    want = set(sync.build_moves(3)) - nosketch - typed_hp
    got = set(built["smeargle"])
    assert want - {"sketch"} <= got and "struggle" not in got and not (got & typed_hp)
    assert "spore" in got and "spikes" in got and "explosion" in got
