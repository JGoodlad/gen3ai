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
        
        # STRICT VALIDATION: Ensure all teams are legal for Gen 3 OU in one batch
        from utils.bridge.team_validator import validate_team_locally
        
        validations = validate_team_locally("gen3ou", teams)
        if not isinstance(validations, list):
            validations = [validations]
            
        self.packed_teams = []
        for i, team_str in enumerate(teams):
            if not validations[i].get("valid"):
                errors = ", ".join(validations[i].get("errors", ["Unknown error"]))
                # Log the error but don't necessarily crash if we have other valid teams?
                # Actually, user wants fail-fast.
                raise ValueError(f"Team Validation Failed: {errors}\nTeam:\n{team_str}")
            
            # Parse and Pack
            parsed_team = self.parse_showdown_team(team_str)
            fixed_team = fix_gen3_hp_ivs(parsed_team)
            self.packed_teams.append(self.join_team(fixed_team))

    def yield_team(self):
        """Returns a random team from the pool (or the only team if pool size is 1)."""
        return random.choice(self.packed_teams)
