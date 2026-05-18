import pytest
import numpy as np
from unittest.mock import MagicMock
from .pokemon import PokemonEncoder
from .species import SpeciesEncoder
from .items import ItemsEncoder
from .types import TypeEncoder
from .abilities import AbilitiesEncoder
from .moves import MovesEncoder
from .state_encoder import load_mappings
from .constants import POKEMON_COUNTER_OFFSET
from poke_env.battle.status import Status


def _make_encoder():
    mappings = load_mappings()
    return PokemonEncoder(
        SpeciesEncoder(mappings["species"]),
        ItemsEncoder(mappings["items"]),
        TypeEncoder(),
        AbilitiesEncoder(mappings["abilities"]),
        MovesEncoder(mappings["moves"]),
    )


def test_pokemon_encoder_dimension():
    assert _make_encoder().dimension == 60


def test_pokemon_encoder_empty():
    encoder = _make_encoder()
    vec = encoder.encode(None, None)
    assert len(vec) == 60
    assert np.all(vec == 0)


def test_sleep_counter_normalized():
    encoder = _make_encoder()
    mon = MagicMock()
    mon.status = Status.SLP
    mon.status_counter = 3
    mon.boosts = {}
    mon.effects = {}
    mon.moves = {}
    mon.item = None
    mon.ability = None
    mon.type_1 = None
    mon.type_2 = None
    mon.species = "pikachu"
    mon.current_hp_fraction = 1.0
    vec = encoder.encode(mon, None)
    assert vec[POKEMON_COUNTER_OFFSET] == pytest.approx(3 / 4.0)
    assert vec[POKEMON_COUNTER_OFFSET + 1] == 0.0  # not toxic


def test_sleep_counter_clamped():
    encoder = _make_encoder()
    mon = MagicMock()
    mon.status = Status.SLP
    mon.status_counter = 10  # beyond Gen 3 max
    mon.boosts = {}
    mon.effects = {}
    mon.moves = {}
    mon.item = None
    mon.ability = None
    mon.type_1 = None
    mon.type_2 = None
    mon.species = "pikachu"
    mon.current_hp_fraction = 1.0
    vec = encoder.encode(mon, None)
    assert vec[POKEMON_COUNTER_OFFSET] == pytest.approx(1.0)


def test_toxic_counter_normalized():
    encoder = _make_encoder()
    mon = MagicMock()
    mon.status = Status.TOX
    mon.status_counter = 4
    mon.boosts = {}
    mon.effects = {}
    mon.moves = {}
    mon.item = None
    mon.ability = None
    mon.type_1 = None
    mon.type_2 = None
    mon.species = "blissey"
    mon.current_hp_fraction = 0.8
    vec = encoder.encode(mon, None)
    assert vec[POKEMON_COUNTER_OFFSET + 1] == pytest.approx(4 / 8.0)
    assert vec[POKEMON_COUNTER_OFFSET] == 0.0  # not sleeping


def test_other_status_leaves_counters_zero():
    encoder = _make_encoder()
    mon = MagicMock()
    mon.status = Status.BRN
    mon.status_counter = 5
    mon.boosts = {}
    mon.effects = {}
    mon.moves = {}
    mon.item = None
    mon.ability = None
    mon.type_1 = None
    mon.type_2 = None
    mon.species = "tauros"
    mon.current_hp_fraction = 0.6
    vec = encoder.encode(mon, None)
    assert vec[POKEMON_COUNTER_OFFSET] == 0.0
    assert vec[POKEMON_COUNTER_OFFSET + 1] == 0.0


def test_no_status_leaves_counters_zero():
    encoder = _make_encoder()
    mon = MagicMock()
    mon.status = None
    mon.status_counter = 0
    mon.boosts = {}
    mon.effects = {}
    mon.moves = {}
    mon.item = None
    mon.ability = None
    mon.type_1 = None
    mon.type_2 = None
    mon.species = "gengar"
    mon.current_hp_fraction = 1.0
    vec = encoder.encode(mon, None)
    assert vec[POKEMON_COUNTER_OFFSET] == 0.0
    assert vec[POKEMON_COUNTER_OFFSET + 1] == 0.0
