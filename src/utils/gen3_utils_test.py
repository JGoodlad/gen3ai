from utils.gen3_utils import fix_gen3_hp_ivs, GEN3_HP_IVS
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


def test_gen3_hp_ivs_is_the_one_table():
    # The teambuilder's fill IS agents.gen3_data.hidden_power.HP_IVS (pinned there to typechart.ts, all 16 types),
    # not a copy that can drift (the Ghost row did, until 2026-10-19).
    from agents.gen3_data import hidden_power

    assert GEN3_HP_IVS is hidden_power.HP_IVS
    assert len(GEN3_HP_IVS) == 16


def test_a_hidden_power_type_line_without_the_move_fills_the_ivs_as_showdown_does():
    """gen3_team_pack_hptype_ivs_v1 (2026-10-26 GIGO audit C-3): Showdown's validator fills a `Hidden Power: <Type>`
    line's IVs when every IV is 31 even WITHOUT the move (team-validator.ts, `set.hpType && maxedIVs`); the authored
    team plays those IVs. Ours left them at 31 (±1 stat point against the authored team)."""
    mon = TeambuilderPokemon(nickname="Salamence", moves=["Dragon Dance", "Earthquake", "Rock Slide", "Fire Blast"],
                             ivs=[31] * 6, hiddenpowertype="Flying")
    fixed = fix_gen3_hp_ivs([mon])[0]
    want = [GEN3_HP_IVS["Flying"].get(k, 31) for k in ("hp", "atk", "def", "spa", "spd", "spe")]
    assert want != [31] * 6 and fixed.ivs == want
    # a set that already chose its IVs is left alone, as Showdown leaves it (not maxed)
    mon2 = TeambuilderPokemon(nickname="Salamence", moves=["Earthquake"], ivs=[31, 0, 31, 31, 31, 31],
                              hiddenpowertype="Flying")
    assert fix_gen3_hp_ivs([mon2])[0].ivs == [31, 0, 31, 31, 31, 31]
