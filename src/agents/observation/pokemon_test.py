from .pokemon import PokemonEncoder
from .species import SpeciesEncoder
from .items import ItemsEncoder
from .types import TypeEncoder
from .abilities import AbilitiesEncoder
from .moves import MovesEncoder
from .state_encoder import load_mappings

# Minimal natures dict for unit tests — no file I/O needed
_TEST_NATURES = {
    "adamant": {"atk": 1.1, "def": 1.0, "spa": 0.9, "spd": 1.0, "spe": 1.0},
    "modest":  {"atk": 0.9, "def": 1.0, "spa": 1.1, "spd": 1.0, "spe": 1.0},
    "timid":   {"atk": 0.9, "def": 1.0, "spa": 1.0, "spd": 1.0, "spe": 1.1},
    "hardy":   {"atk": 1.0, "def": 1.0, "spa": 1.0, "spd": 1.0, "spe": 1.0},
    "serious": {"atk": 1.0, "def": 1.0, "spa": 1.0, "spd": 1.0, "spe": 1.0},
}


def _make_encoder(natures=None):
    mappings = load_mappings()
    return PokemonEncoder(
        SpeciesEncoder(mappings["species"]),
        ItemsEncoder(mappings["items"]),
        TypeEncoder(),
        AbilitiesEncoder(mappings["abilities"]),
        MovesEncoder(mappings["moves"]),
        natures=natures if natures is not None else _TEST_NATURES,
    )


def test_pokemon_encoder_dimension():
    assert _make_encoder().dimension == 119  # gen3_pair_history_v1: 113 + 6 last-action
