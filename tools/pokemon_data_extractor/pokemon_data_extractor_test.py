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
