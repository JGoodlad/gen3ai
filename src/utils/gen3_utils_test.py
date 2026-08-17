from utils.gen3_utils import fix_gen3_hp_ivs, GEN3_HP_IVS
from poke_env.teambuilder import TeambuilderPokemon

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
