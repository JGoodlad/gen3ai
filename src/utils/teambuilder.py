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
            packed = self.join_team(fixed_team)
            
            # 3. STRICT VALIDATION: Ensure the team is legal for Gen 3 OU
            from utils.bridge.team_validator import validate_team_locally
            validation = validate_team_locally("gen3ou", team_str)
            if not validation.get("valid"):
                errors = ", ".join(validation.get("errors", ["Unknown error"]))
                raise ValueError(f"Team Validation Failed: {errors}\nTeam:\n{team_str}")
                
            self.packed_teams.append(packed)

    def yield_team(self):
        """Returns a random team from the pool (or the only team if pool size is 1)."""
        return random.choice(self.packed_teams)
