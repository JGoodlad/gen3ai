import pytest
from utils.teambuilder import Gen3Teambuilder

def test_gen3_teambuilder_initialization():
    team_str = """
    Tyranitar @ Leftovers
    Ability: Sand Stream
    EVs: 252 HP / 252 Atk / 4 SpD
    Adamant Nature
    - Hidden Power [Bug]
    - Rock Slide
    - Focus Punch
    - Earthquake
    """
    
    # Test initialization with a single team
    tb = Gen3Teambuilder(team_str)
    assert len(tb.packed_teams) == 1
    
    # Test yield_team returns a string
    team = tb.yield_team()
    assert isinstance(team, str)
    # Packed team should contain Tyranitar (usually first part of packed string)
    assert "Tyranitar" in team

def test_gen3_teambuilder_multiple_teams():
    team_1 = "Pikachu @ Light Ball\n- Thunderbolt"
    team_2 = "Snorlax @ Leftovers\n- Body Slam"
    
    tb = Gen3Teambuilder([team_1, team_2])
    assert len(tb.packed_teams) == 2
    
    # Ensure yield_team can return both
    teams_yielded = {tb.yield_team() for _ in range(20)}
    assert len(teams_yielded) == 2
