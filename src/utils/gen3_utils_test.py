import re

from utils.gen3_utils import fix_gen3_hp_ivs, GEN3_HP_IVS
from utils.paths import repo_path
from utils.team_packing import TeambuilderPokemon

def test_hp_bug_fix():
    # Setup a Tyranitar with Hidden Power [Bug] and default IVs
    mon = TeambuilderPokemon(
        nickname="Tyranitar",
        moves=["Focus Punch", "Rock Slide", "Hidden Power [Bug]", "Earthquake"],
        ivs=[31] * 6
    )
    
    fixed_mons = fix_gen3_hp_ivs([mon])
    fixed_mon = fixed_mons[0]
    
    # Check if IVs were updated according to GEN3_HP_IVS["Bug"]
    # indices: hp=0, atk=1, def=2, spa=3, spd=4, spe=5
    assert fixed_mon.ivs[1] == GEN3_HP_IVS["Bug"].get("atk", 31)
    assert fixed_mon.ivs[2] == GEN3_HP_IVS["Bug"].get("def", 31)
    assert fixed_mon.ivs[4] == GEN3_HP_IVS["Bug"].get("spd", 31)

def test_hp_ice_fix():
    mon = TeambuilderPokemon(
        nickname="Starmie",
        moves=["Surf", "Thunderbolt", "Hidden Power [Ice]", "Recover"],
        ivs=[31] * 6
    )
    
    fixed_mons = fix_gen3_hp_ivs([mon])
    fixed_mon = fixed_mons[0]
    
    assert fixed_mon.ivs[1] == GEN3_HP_IVS["Ice"].get("atk", 31)
    assert fixed_mon.ivs[2] == GEN3_HP_IVS["Ice"].get("def", 31)

def test_no_hidden_power():
    # Mon without Hidden Power should stay at 31s
    mon = TeambuilderPokemon(
        nickname="Skarmory",
        moves=["Spikes", "Roar", "Drill Peck", "Rest"],
        ivs=[31] * 6
    )
    
    fixed_mons = fix_gen3_hp_ivs([mon])
    assert fixed_mons[0].ivs == [31] * 6

def test_all_types():
    # Test every type in our mapping to ensure no crashes
    for hp_type in GEN3_HP_IVS.keys():
        mon = TeambuilderPokemon(
            nickname="TestMon",
            moves=[f"Hidden Power [{hp_type}]"],
            ivs=[31] * 6
        )
        fixed_mons = fix_gen3_hp_ivs([mon])
        assert fixed_mons[0].ivs != [31] * 6 or hp_type == "Dark"


_HP_TYPES_BY_INDEX = ("Fighting", "Flying", "Poison", "Ground", "Rock", "Bug", "Ghost", "Steel", "Fire", "Water",
                      "Grass", "Electric", "Psychic", "Ice", "Dragon", "Dark")
_STATS = ("hp", "atk", "def", "spa", "spd", "spe")


def _showdown_hpivs() -> dict:
    """``fillStats(typechart.ts HPivs, 31)`` per type (the validator's fill), parsed from the pinned submodule."""
    src = repo_path("deps", "pokemon-showdown", "data", "typechart.ts").read_text()
    out = {}
    for name, body in re.findall(r"^\t(\w+): \{(.*?)^\t\},", src, re.MULTILINE | re.DOTALL):
        m = re.search(r"HPivs: \{([^}]*)\}", body)
        if m is None or name in ("normal", "fairy", "stellar"):
            continue
        ivs = {k: 31 for k in _STATS}
        ivs.update({k: int(v) for k, v in re.findall(r"(\w+): (\d+)", m.group(1))})
        out[name.capitalize()] = ivs
    return out


def test_gen3_hp_ivs_is_showdowns_validator_fill_for_all_16_types():
    # Fails on revert of the Ghost row (30 Atk / 30 SpD vs Showdown's 30 Def / 30 SpD).
    found = _showdown_hpivs()
    assert len(found) == 16
    assert GEN3_HP_IVS == found


def test_gen3_hp_ivs_give_their_type_at_power_70():
    # Independent of Showdown: the gen-3 formula (Smogon / Bulbapedia). Type bits are each IV's low bit in the
    # order HP, Atk, Def, Spe, SpA, SpD; power bits the second-lowest bit (IV % 4 >= 2) in the same order.
    assert len(GEN3_HP_IVS) == 16
    for typ, ivs in GEN3_HP_IVS.items():
        order = [ivs[k] for k in ("hp", "atk", "def", "spe", "spa", "spd")]
        type_bits = sum((iv & 1) << i for i, iv in enumerate(order))
        power_bits = sum(((iv >> 1) & 1) << i for i, iv in enumerate(order))
        assert _HP_TYPES_BY_INDEX[type_bits * 15 // 63] == typ
        assert power_bits * 40 // 63 + 30 == 70
