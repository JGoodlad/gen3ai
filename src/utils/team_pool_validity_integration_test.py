"""EVERY team in the pool parses and packs through ``Gen3Teambuilder`` (719 teams, ~1.2 s each: ~15 min).

Lived as ``tests/test_teams_validity.py``, a path no collection root named and a filename ``python_files`` does not match,
so it never ran; ``src/uncollected_tests_gate_test.py`` found it. It is ``slow`` by its measured cost: run it when the
team pool changes (``pytest src/utils/team_pool_validity_integration_test.py -m slow -q``).
"""
import pytest
from utils.team_loader import TeamLoader
from utils.teambuilder import Gen3Teambuilder

pytestmark = [pytest.mark.slow, pytest.mark.integration]


def test_all_teams_validity():
    """
    Test that every team discovered by TeamLoader can be successfully 
    parsed and packed by Gen3Teambuilder.
    """
    loader = TeamLoader()
    all_teams = loader.get_all_teams()
    
    assert len(all_teams) > 0, "No teams were loaded!"
    
    failed_teams = []
    for i, team_str in enumerate(all_teams):
        try:
            # Gen3Teambuilder performs parsing and IV fixing
            tb = Gen3Teambuilder(team_str)
            # Check if we actually got packed teams
            assert len(tb.packed_teams) > 0
            assert isinstance(tb.packed_teams[0], str)
        except Exception as e:
            # Find which file it was (if possible)
            # The loader doesn't store paths with team texts, but we can 
            # find the team by its name if we adjust the loader.
            # For now, just report the error and index.
            failed_teams.append((i, str(e)))
            
    if failed_teams:
        error_msg = "\n".join([f"Team {idx} failed: {err}" for idx, err in failed_teams])
        pytest.fail(f"Validation failed for {len(failed_teams)} teams:\n{error_msg}")

if __name__ == "__main__":
    # If run directly, just run the logic
    loader = TeamLoader()
    teams = loader.get_all_teams()
    print(f"Validating {len(teams)} teams...")
    for i, t in enumerate(teams):
        try:
            Gen3Teambuilder(t)
        except Exception as e:
            print(f"FAILED Team {i}: {e}")
    print("Validation finished.")
