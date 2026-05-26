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
from .constants import (
    POKEMON_COUNTER_OFFSET,
    POKEMON_SPREAD_OFFSET,
    POKEMON_SPREAD_DIM,
    POKEMON_HP_REVEALED_OFFSET,
    POKEMON_HP_PROBS_OFFSET,
)
from poke_env.battle.status import Status
from poke_env.battle.pokemon_type import PokemonType
from agents.training.hidden_power_tracker import HIDDEN_POWER_TYPE_ORDER

_HP_IDX = {t.name: i for i, t in enumerate(HIDDEN_POWER_TYPE_ORDER)}

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


def _make_mon(nature="hardy", ivs=None, evs=None, status=None, status_counter=0,
              species="pikachu", hp_fraction=1.0):
    """Build a minimal mock Pokemon for spread-encoding tests."""
    mon = MagicMock()
    mon.nature = nature
    mon.ivs = ivs if ivs is not None else [31] * 6
    mon.evs = evs if evs is not None else [0] * 6
    mon.status = status
    mon.status_counter = status_counter
    mon.boosts = {}
    mon.effects = {}
    mon.moves = {}
    mon.item = None
    mon.ability = None
    mon.type_1 = None
    mon.type_2 = None
    mon.species = species
    mon.current_hp_fraction = hp_fraction
    return mon


def test_pokemon_encoder_dimension():
    assert _make_encoder().dimension == 96


def test_pokemon_encoder_empty():
    encoder = _make_encoder()
    vec = encoder.encode(None, None)
    assert len(vec) == 96
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


# ---------------------------------------------------------------------------
# Spread block tests
# ---------------------------------------------------------------------------

def test_spread_block_opponent_all_zeros():
    """Opponent slots: entire 18-dim spread block must be zero (including spread_known)."""
    encoder = _make_encoder()
    mon = _make_mon(nature="adamant", ivs=[31]*6, evs=[252, 252, 4, 0, 0, 0])
    vec = encoder.encode(mon, None, is_own=False)
    spread = vec[POKEMON_SPREAD_OFFSET : POKEMON_SPREAD_OFFSET + POKEMON_SPREAD_DIM]
    assert np.all(spread == 0.0), f"Expected all-zero spread for opponent, got: {spread}"


def test_spread_block_own_standard():
    """Own team: IVs, EVs, spread_known=1, and adamant nature modifiers correctly encoded."""
    encoder = _make_encoder()
    mon = _make_mon(nature="adamant", ivs=[31]*6, evs=[252, 252, 4, 0, 0, 0])
    vec = encoder.encode(mon, None, is_own=True)
    off = POKEMON_SPREAD_OFFSET
    # IVs: all 31 → all 1.0
    assert np.allclose(vec[off:off+6], [1.0]*6), f"IVs: {vec[off:off+6]}"
    # EVs: [252, 252, 4, 0, 0, 0] normalized
    assert np.allclose(vec[off+6:off+12], [1.0, 1.0, 4/252, 0.0, 0.0, 0.0], atol=1e-5)
    # spread_known = 1.0
    assert vec[off+12] == pytest.approx(1.0)
    # adamant: atk=1.1, def=1.0, spa=0.9, spd=1.0, spe=1.0
    assert np.allclose(vec[off+13:off+18], [1.1, 1.0, 0.9, 1.0, 1.0], atol=1e-5)


def test_spread_block_own_hp_iv_spread():
    """Non-31 IVs (e.g. Hidden Power Ice spread) encode correctly."""
    encoder = _make_encoder()
    ivs = [31, 30, 31, 30, 31, 30]  # common HP Ice spread
    mon = _make_mon(nature="modest", ivs=ivs, evs=[0]*6)
    vec = encoder.encode(mon, None, is_own=True)
    off = POKEMON_SPREAD_OFFSET
    expected_ivs = [iv / 31.0 for iv in ivs]
    assert np.allclose(vec[off:off+6], expected_ivs, atol=1e-5)
    # modest: atk=0.9, def=1.0, spa=1.1, spd=1.0, spe=1.0
    assert np.allclose(vec[off+13:off+18], [0.9, 1.0, 1.1, 1.0, 1.0], atol=1e-5)


def test_spread_block_own_zero_evs_distinguishable_from_opp():
    """Own slot with 0 EVs has spread_known=1.0 → distinguishable from opponent (spread_known=0)."""
    encoder = _make_encoder()
    mon = _make_mon(nature="hardy", ivs=[31]*6, evs=[0]*6)
    own_vec  = encoder.encode(mon, None, is_own=True)
    opp_vec  = encoder.encode(mon, None, is_own=False)
    own_spread = own_vec[POKEMON_SPREAD_OFFSET : POKEMON_SPREAD_OFFSET + POKEMON_SPREAD_DIM]
    opp_spread = opp_vec[POKEMON_SPREAD_OFFSET : POKEMON_SPREAD_OFFSET + POKEMON_SPREAD_DIM]
    # spread_known is at offset 12 within the block
    assert own_spread[12] == pytest.approx(1.0), "Own slot: spread_known should be 1.0"
    assert opp_spread[12] == pytest.approx(0.0), "Opp slot: spread_known should be 0.0"
    # EVs are all zero for own slot but spread_known still distinguishes them
    assert not np.array_equal(own_spread, opp_spread)


def test_spread_block_own_neutral_nature_all_ones():
    """Neutral nature (hardy/serious) → all 5 nature modifiers are 1.0."""
    encoder = _make_encoder()
    mon = _make_mon(nature="hardy", ivs=[31]*6, evs=[0]*6)
    vec = encoder.encode(mon, None, is_own=True)
    off = POKEMON_SPREAD_OFFSET
    assert np.allclose(vec[off+13:off+18], [1.0]*5, atol=1e-5)


def test_spread_block_is_own_default_false():
    """is_own defaults to False → opponent encoding (spread block all zeros)."""
    encoder = _make_encoder()
    mon = _make_mon(nature="adamant", ivs=[31]*6, evs=[252]*6)
    vec = encoder.encode(mon, None)  # no is_own kwarg
    spread = vec[POKEMON_SPREAD_OFFSET : POKEMON_SPREAD_OFFSET + POKEMON_SPREAD_DIM]
    assert np.all(spread == 0.0)


def test_spread_block_own_none_ivs_fallback():
    """If mon.ivs is None (poke_env edge case), IVs fall back to all-31, not all-0.
    Encoding 0.0 for unknown IVs would silently tell the model 'this Pokémon has 0 in
    every IV' — a lie. All-31 is the correct competitive assumption."""
    encoder = _make_encoder()
    mon = _make_mon(nature="adamant")
    mon.ivs = None  # simulate poke_env not having set IVs
    mon.evs = [0] * 6
    vec = encoder.encode(mon, None, is_own=True)
    off = POKEMON_SPREAD_OFFSET
    # IVs should be 1.0 (all-31 fallback), NOT 0.0
    assert np.allclose(vec[off:off+6], [1.0]*6), f"Expected all-31 fallback, got: {vec[off:off+6]}"
    # spread_known still 1.0 — it's our own team
    assert vec[off+12] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# poke_env layer: _update_from_teambuilder with all-zero EVs
# ---------------------------------------------------------------------------

def test_poke_env_all_zero_evs_stores_ivs_and_nature():
    """Regression: poke_env must store ivs/evs/nature even when all EVs are zero.
    The old guard `if not all(e == 0 for e in tb.evs)` left _ivs/_nature as None,
    causing the spread encoder to silently use wrong values."""
    from poke_env.teambuilder.teambuilder import Teambuilder
    from poke_env.battle.pokemon import Pokemon

    # Team paste with explicit 0 EVs and a non-trivial IV spread + non-neutral nature
    team_paste = """Shedinja @ Lum Berry
Ability: Wonder Guard
EVs: 0 HP / 0 Atk / 0 Def / 0 SpA / 0 SpD / 0 Spe
Jolly Nature
IVs: 31 HP / 31 Atk / 31 Def / 31 SpA / 31 SpD / 31 Spe
- Shadow Ball
- Silver Wind
- Return
- Shadow Ball
"""
    parsed = Teambuilder.parse_showdown_team(team_paste)
    assert len(parsed) == 1
    tb = parsed[0]
    assert tb.evs == [0, 0, 0, 0, 0, 0], "EVs should all be zero"
    assert tb.ivs == [31, 31, 31, 31, 31, 31]
    assert tb.nature is not None and tb.nature.lower() == "jolly"

    mon = Pokemon(gen=3)
    mon._update_from_teambuilder(tb)

    # After fix: these must NOT be None
    assert mon.ivs is not None, "ivs should be stored even with all-zero EVs"
    assert mon.evs is not None, "evs should be stored even with all-zero EVs"
    assert mon.nature is not None, "nature should be stored even with all-zero EVs"
    assert mon.ivs == [31, 31, 31, 31, 31, 31]
    assert mon.evs == [0, 0, 0, 0, 0, 0]
    assert mon.nature == "jolly"


def test_poke_env_no_ev_line_stores_defaults():
    """Team paste with no EV line → tb.evs=[0]*6 → _update_from_teambuilder must
    still store the defaults (ivs=[31]*6, evs=[0]*6, nature from paste or 'serious')."""
    from poke_env.teambuilder.teambuilder import Teambuilder
    from poke_env.battle.pokemon import Pokemon

    team_paste = """Gengar @ Leftovers
Ability: Levitate
Timid Nature
- Shadow Ball
- Thunderbolt
- Ice Punch
- Destiny Bond
"""
    parsed = Teambuilder.parse_showdown_team(team_paste)
    tb = parsed[0]
    assert tb.evs == [0, 0, 0, 0, 0, 0]  # default
    assert tb.ivs == [31, 31, 31, 31, 31, 31]  # default

    mon = Pokemon(gen=3)
    mon._update_from_teambuilder(tb)

    assert mon.ivs == [31, 31, 31, 31, 31, 31], "Default all-31 IVs should be stored"
    assert mon.evs == [0, 0, 0, 0, 0, 0], "Default all-0 EVs should be stored"
    assert mon.nature == "timid"


def test_spread_encoder_all_zero_evs_end_to_end():
    """End-to-end: a real Pokemon with all-zero EVs (from poke_env) must encode
    correctly — IVs as 1.0, EVs as 0.0, correct nature modifiers, spread_known=1."""
    from poke_env.teambuilder.teambuilder import Teambuilder
    from poke_env.battle.pokemon import Pokemon

    team_paste = """Shedinja @ Lum Berry
Ability: Wonder Guard
EVs: 0 HP / 0 Atk / 0 Def / 0 SpA / 0 SpD / 0 Spe
Adamant Nature
IVs: 31 HP / 31 Atk / 31 Def / 31 SpA / 31 SpD / 31 Spe
- Shadow Ball
- Silver Wind
- Return
- Shadow Ball
"""
    parsed = Teambuilder.parse_showdown_team(team_paste)
    mon = Pokemon(gen=3)
    mon._update_from_teambuilder(parsed[0])

    mappings = load_mappings()
    encoder = _make_encoder(natures=mappings["natures"])
    vec = encoder.encode(mon, None, is_own=True)
    off = POKEMON_SPREAD_OFFSET

    assert np.allclose(vec[off:off+6],   [1.0]*6,              atol=1e-5), f"IVs wrong: {vec[off:off+6]}"
    assert np.allclose(vec[off+6:off+12],[0.0]*6,              atol=1e-5), f"EVs wrong: {vec[off+6:off+12]}"
    assert vec[off+12] == pytest.approx(1.0),                               "spread_known wrong"
    # adamant: atk=1.1, def=1.0, spa=0.9, spd=1.0, spe=1.0
    assert np.allclose(vec[off+13:off+18],[1.1, 1.0, 0.9, 1.0, 1.0], atol=1e-5), f"Nature wrong: {vec[off+13:off+18]}"


# ---------------------------------------------------------------------------
# Hidden Power block (offset POKEMON_HP_REVEALED_OFFSET, 17 dims)
# ---------------------------------------------------------------------------

def _hp_block(vec):
    """Slice the 17-dim HP block out of a Pokemon vector."""
    return vec[POKEMON_HP_REVEALED_OFFSET : POKEMON_HP_REVEALED_OFFSET + 17]


def _hp_move(move_id, type_name=None):
    """Build a mock Hidden Power move. type_name is the PokemonType .name string
    (e.g. 'GRASS'); pass None for the bare-HP case where type is unknown."""
    m = MagicMock()
    m.id = move_id
    m.current_pp = 24
    m.max_pp = 24
    if type_name is not None:
        m.type = MagicMock()
        m.type.name = type_name
    else:
        m.type = MagicMock()
        m.type.name = "NORMAL"  # default for bare HP entry
    return m


def test_hp_block_own_with_hp_grass_writes_one_hot():
    """Own mon carrying HP-Grass: hp_revealed=1, one-hot at Grass index."""
    encoder = _make_encoder()
    mon = _make_mon()
    mon.moves = {"hiddenpowergrass": _hp_move("hiddenpowergrass", "GRASS")}

    vec = encoder.encode(mon, None, is_own=True)
    block = _hp_block(vec)
    assert block[0] == 1.0, "hp_revealed must be 1.0 for our own team"
    assert block[1 + _HP_IDX["GRASS"]] == 1.0, "Grass slot must hold 1.0"
    # Everything else in the 16-dim probability vector must be zero
    other_mass = block[1:].sum() - 1.0
    assert other_mass == pytest.approx(0.0), f"non-Grass slots leaked: {block[1:]}"


def test_hp_block_own_with_hp_fire_writes_one_hot():
    """Different type to confirm the helper isn't constant."""
    encoder = _make_encoder()
    mon = _make_mon()
    mon.moves = {"hiddenpowerfire": _hp_move("hiddenpowerfire", "FIRE")}

    vec = encoder.encode(mon, None, is_own=True)
    block = _hp_block(vec)
    assert block[0] == 1.0
    assert block[1 + _HP_IDX["FIRE"]] == 1.0
    assert block[1 + _HP_IDX["GRASS"]] == 0.0


def test_hp_block_own_without_hp_marks_no_hp():
    """Own mon without Hidden Power: hp_revealed=1, all-zero probs.
    Distinguishes from 'unknown opponent' which has hp_revealed=0."""
    encoder = _make_encoder()
    mon = _make_mon()
    # No HP move — just a regular move
    regular = MagicMock()
    regular.id = "surf"
    regular.current_pp = 24
    regular.max_pp = 24
    mon.moves = {"surf": regular}

    vec = encoder.encode(mon, None, is_own=True)
    block = _hp_block(vec)
    assert block[0] == 1.0, "hp_revealed must be 1.0 even without HP"
    assert np.all(block[1:] == 0.0), "probs must be all zero when no HP"


def test_hp_block_own_with_bare_hp_falls_back_to_no_hp():
    """Defensive: if the raw_id patch was bypassed and our mon has bare
    'hiddenpower', we have no type info — treat as 'no derivable HP type'.
    hp_revealed=1.0 (we DO know the state of this side) and probs zero."""
    encoder = _make_encoder()
    mon = _make_mon()
    mon.moves = {"hiddenpower": _hp_move("hiddenpower", "NORMAL")}

    vec = encoder.encode(mon, None, is_own=True)
    block = _hp_block(vec)
    assert block[0] == 1.0
    assert np.all(block[1:] == 0.0)


def test_hp_block_opponent_unknown_all_zero():
    """Default opponent (hp_known=False): entire block is zero."""
    encoder = _make_encoder()
    mon = _make_mon()
    mon.moves = {}

    vec = encoder.encode(mon, None, is_own=False, hp_probs=None, hp_known=False)
    block = _hp_block(vec)
    assert np.all(block == 0.0)


def test_hp_block_opponent_narrowed_sparse_probs():
    """Opponent narrowed via observation: hp_revealed=1, sparse non-zero probs."""
    encoder = _make_encoder()
    mon = _make_mon()
    mon.moves = {}

    probs = np.zeros(16, dtype=np.float32)
    probs[_HP_IDX["FIGHTING"]] = 1.0
    vec = encoder.encode(mon, None, is_own=False, hp_probs=probs, hp_known=True)
    block = _hp_block(vec)
    assert block[0] == 1.0
    assert block[1 + _HP_IDX["FIGHTING"]] == 1.0
    assert block[2:].sum() <= 1.0


def test_hp_block_opponent_ruled_out_zero_probs_with_flag():
    """Opponent ruled out (4 moves revealed, no HP): hp_revealed=1, probs all zero.
    This is the new state distinguishing 'no HP' from 'haven't seen any HP yet'."""
    encoder = _make_encoder()
    mon = _make_mon()
    mon.moves = {}

    probs = np.zeros(16, dtype=np.float32)
    vec = encoder.encode(mon, None, is_own=False, hp_probs=probs, hp_known=True)
    block = _hp_block(vec)
    assert block[0] == 1.0, "hp_revealed must signal 'state known'"
    assert np.all(block[1:] == 0.0), "probs must all be zero"


def test_hp_block_real_pokemon_with_hp_grass_end_to_end():
    """End-to-end: a real Pokemon parsed from a team paste with Hidden Power
    [Grass] must encode hp_revealed=1 + one-hot at Grass. Exercises the full
    poke-env raw_id patch path."""
    from poke_env.teambuilder.teambuilder import Teambuilder
    from poke_env.battle.pokemon import Pokemon

    team_paste = """Celebi @ Leftovers
Ability: Natural Cure
EVs: 252 HP / 124 Def / 40 SpA / 60 SpD / 32 Spe
Bold Nature
IVs: 2 Atk / 30 SpA
- Leech Seed
- Recover
- Perish Song
- Hidden Power [Grass]
"""
    parsed = Teambuilder.parse_showdown_team(team_paste)
    mon = Pokemon(gen=3)
    mon._update_from_teambuilder(parsed[0])

    mappings = load_mappings()
    encoder = _make_encoder(natures=mappings["natures"])
    vec = encoder.encode(mon, None, is_own=True)
    block = _hp_block(vec)

    assert block[0] == 1.0, "hp_revealed must be 1.0"
    assert block[1 + _HP_IDX["GRASS"]] == 1.0, (
        "Grass one-hot must be set — confirms raw_id flows through poke-env"
    )
    # Total mass exactly 1.0 — pure one-hot
    assert block[1:].sum() == pytest.approx(1.0)
