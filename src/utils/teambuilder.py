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
        from utils.bridge.team_validator import validate_teams_locally
        
        validations = validate_teams_locally("gen3ou", teams)
            
        valid_indices = []
        for i, res in enumerate(validations):
            if res.get("valid"):
                valid_indices.append(i)
            else:
                errors = ", ".join(res.get("errors", ["Unknown error"]))
                # We skip illegal teams but don't crash unless the whole pool is bad.

        if not valid_indices:
            raise ValueError(f"No valid teams found in the provided list! (First error: {validations[0].get('errors')})")

        self.packed_teams = []
        for idx in valid_indices:
            team_str = teams[idx]
            # Parse and Pack
            parsed_team = self.parse_showdown_team(team_str)
            fixed_team = fix_gen3_hp_ivs(parsed_team)
            self.packed_teams.append(self.join_team(fixed_team))

    def yield_team(self):
        """Returns a random team from the pool (or the only team if pool size is 1)."""
        return random.choice(self.packed_teams)
