from poke_env.teambuilder import Teambuilder
from utils.gen3_utils import fix_gen3_hp_ivs
import random

class Gen3Teambuilder(Teambuilder):
    """
    A specialized Teambuilder for Gen 3 (ADV) that automatically handles
    validation nuances like Hidden Power IV mappings.
    """

    def __init__(self, teams, bias_teams=None, bias_prob=0.0):
        """
        Initialize with a single team string or a list of team strings.

        bias_teams: optional secondary pool; when provided, yield_team picks from
                    it with probability bias_prob (and from the main pool otherwise).
        bias_prob:  float in [0, 1].  0.0 = always use main pool (default).
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

        self.bias_prob = bias_prob
        self.bias_packed_teams = []
        if bias_teams and bias_prob > 0.0:
            if isinstance(bias_teams, str):
                bias_teams = [bias_teams]
            bias_validations = validate_teams_locally("gen3ou", bias_teams)
            for i, res in enumerate(bias_validations):
                if res.get("valid"):
                    parsed = self.parse_showdown_team(bias_teams[i])
                    fixed = fix_gen3_hp_ivs(parsed)
                    self.bias_packed_teams.append(self.join_team(fixed))

    def yield_team(self):
        """Returns a random team from the pool, biased toward bias_packed_teams if configured."""
        if self.bias_packed_teams and random.random() < self.bias_prob:
            return random.choice(self.bias_packed_teams)
        return random.choice(self.packed_teams)
