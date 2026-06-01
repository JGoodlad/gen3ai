"""Extractor ↔ upstream parity — the drift guard.

Proves the committed ``data/pokemon/`` files are genuinely *derived* and reproducible: each file
equals its upstream source of truth, and re-running the builders reproduces the committed file.
If any committed file is hand-edited away from what the extractor would produce, these fail.

Builders that read only poke-env static data (always present in the repo) are unit tests; those
that read the Showdown submodule (``deps/pokemon-showdown``) are integration tests.
"""
import importlib.util
import json
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[3]


def _load_sync():
    spec = importlib.util.spec_from_file_location(
        "pde_sync", _REPO / "tools" / "pokemon_data_extractor" / "sync.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _committed(name):
    with open(_REPO / "data" / "pokemon" / name) as f:
        return json.load(f)


# --- committed file == upstream source of truth (poke-env static; unit-safe) --------------- #

def test_type_chart_matches_gendata():
    from poke_env.data import GenData
    assert _committed("gen3_type_chart.json") == GenData.from_gen(3).type_chart


def test_natures_match_pokeenv():
    with open(_REPO / "src" / "poke_env" / "data" / "static" / "natures.json") as f:
        upstream = json.load(f)
    assert _committed("gen3_natures.json") == upstream


# --- builder reproduces the committed file (the reproducibility proof) ---------------------- #

def test_species_builder_reproduces_committed():
    assert _load_sync().build_species(3) == _committed("gen3_species.json")


def test_moves_builder_reproduces_committed():
    assert _load_sync().build_moves(3) == _committed("gen3_moves.json")


@pytest.mark.integration
def test_abilities_builder_reproduces_committed():
    # build_abilities reads the Showdown submodule (deps/pokemon-showdown/data/abilities.ts).
    assert _load_sync().build_abilities(3) == _committed("gen3_abilities.json")


@pytest.mark.integration
def test_items_builder_reproduces_committed():
    # build_items reads the Showdown submodule (deps/pokemon-showdown/data/items.ts).
    assert _load_sync().build_items(3) == _committed("gen3_items.json")
