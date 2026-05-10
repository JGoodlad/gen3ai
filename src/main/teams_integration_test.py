import pytest
import json
import os
from src.utils.bridge.team_validator import validate_team_locally

# This test integrates with the local Pokemon Showdown sim library via a Node bridge.
# It is much faster than server-based testing and doesn't require a running instance.

@pytest.mark.integration
def test_all_downloaded_teams_validity():
    """
    Validates all downloaded sample teams using the local Showdown validator bridge.
    This ensures that the teams we synchronized are legal for the Gen 3 OU format.
    """
    metadata_path = "data/teams/teams.json"
    if not os.path.exists(metadata_path):
        pytest.skip("data/teams/teams.json not found. Run sync-teams first.")

    with open(metadata_path, 'r') as f:
        teams_metadata = json.load(f)

    if not teams_metadata:
        pytest.skip("No teams found in teams.json.")

    print(f"Validating {len(teams_metadata)} teams locally...")
    
    invalid_teams = []
    for team_info in teams_metadata:
        team_id = team_info['id']
        team_name = team_info['name']
        team_file = os.path.join("data", team_info['file'])
        
        if not os.path.exists(team_file):
            print(f"Warning: Team file missing for {team_name} ({team_id})")
            continue

        with open(team_file, 'r') as f:
            raw_team = f.read()

        # Validate using our local Node bridge
        result = validate_team_locally("gen3ou", raw_team)
        
        if not result["valid"]:
            invalid_teams.append({
                "name": team_name,
                "id": team_id,
                "errors": result["errors"]
            })

    if invalid_teams:
        error_msg = "\n".join([f"- {it['name']} ({it['id']}): {it['errors']}" for it in invalid_teams])
        pytest.fail(f"Found {len(invalid_teams)} invalid teams:\n{error_msg}")
    
    print(f"Successfully validated {len(teams_metadata)} teams.")
