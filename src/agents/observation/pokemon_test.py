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


# ---------------------------------------------------------------------------
# Spread block tests
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Hidden Power block (offset POKEMON_HP_REVEALED_OFFSET, 17 dims)
# ---------------------------------------------------------------------------
