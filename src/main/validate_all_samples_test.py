import json
import os
import pytest
from src.utils.team_validator import validate_team_locally

def test_validate_all_sample_teams_locally():
    """
    Uses the local Node bridge to validate all downloaded sample teams.
    This is much faster than the full integration test and doesn't require a running server.
    """
    metadata_path = "data/teams/teams.json"
    if not os.path.exists(metadata_path):
        pytest.skip("No sample teams found. Run sync-teams first.")

    with open(metadata_path, 'r') as f:
        teams = json.load(f)

    print(f"Validating {len(teams)} teams locally...")
    
    invalid_teams = []
    for team_meta in teams:
        team_file = os.path.join("data", team_meta["file"])
        with open(team_file, 'r') as f:
            team_text = f.read()
            
        result = validate_team_locally("gen3ou", team_text)
        if not result["valid"]:
            invalid_teams.append({
                "name": team_meta["name"],
                "id": team_meta["id"],
                "errors": result["errors"]
            })

    if invalid_teams:
        print("\nInvalid teams found:")
        for it in invalid_teams:
            print(f"- {it['name']} ({it['id']}): {it['errors']}")
            
    assert len(invalid_teams) == 0, f"Found {len(invalid_teams)} invalid teams."
