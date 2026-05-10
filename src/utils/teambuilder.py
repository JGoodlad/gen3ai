from poke_env.teambuilder import Teambuilder
from utils.gen3_utils import fix_gen3_hp_ivs
import random

class Gen3Teambuilder(Teambuilder):
    """
    A specialized Teambuilder for Gen 3 (ADV) that automatically handles 
    validation nuances like Hidden Power IV mappings.
    """
    
    def __init__(self, teams):
        """
        Initialize with a single team string or a list of team strings.
        """
        if isinstance(teams, str):
            teams = [teams]
        
        self.packed_teams = []
        for team_str in teams:
            # 1. Parse the showdown string
            parsed_team = self.parse_showdown_team(team_str)
            
            # 2. Automatically fix IVs for Hidden Power and other Gen 3 quirks
            fixed_team = fix_gen3_hp_ivs(parsed_team)
            
            # 3. Join and store the packed format
            self.packed_teams.append(self.join_team(fixed_team))

    def yield_team(self):
        """Returns a random team from the pool (or the only team if pool size is 1)."""
        return random.choice(self.packed_teams)
