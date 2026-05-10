import pytest
import os
import json
import tempfile
import shutil
from src.utils.team_loader.loader import TeamLoader

@pytest.fixture
def mock_teams_structure():
    """
    Creates a temporary directory structure that mimics the data/teams folder.
    """
    temp_dir = tempfile.mkdtemp()
    
    # We need to mimic the 'data/teams' structure because the loader 
    # uses 'data' as a hardcoded prefix for file resolution.
    data_dir = os.path.join(temp_dir, "data")
    teams_dir = os.path.join(data_dir, "teams")
    os.makedirs(teams_dir)
    
    # Sample teams
    sample_dir = os.path.join(teams_dir, "sample")
    os.makedirs(sample_dir)
    with open(os.path.join(sample_dir, "team_s1.txt"), "w") as f:
        f.write("Sample Team 1 Content")
    with open(os.path.join(sample_dir, "teams.json"), "w") as f:
        json.dump([{"file": "teams/sample/team_s1.txt"}], f)
        
    # Other teams
    others_dir = os.path.join(teams_dir, "others", "johnnyg2")
    os.makedirs(others_dir)
    with open(os.path.join(others_dir, "team_o1.txt"), "w") as f:
        f.write("Other Team 1 Content")
    with open(os.path.join(others_dir, "teams.json"), "w") as f:
        json.dump([{"file": "teams/others/johnnyg2/team_o1.txt"}], f)
        
    # Store the original cwd to restore later
    old_cwd = os.getcwd()
    os.chdir(temp_dir)
    
    yield teams_dir
    
    os.chdir(old_cwd)
    shutil.rmtree(temp_dir)

def test_loader_finds_all_teams(mock_teams_structure):
    loader = TeamLoader(base_dir=mock_teams_structure)
    
    assert len(loader.get_sample_teams()) == 1
    assert len(loader.get_other_teams()) == 1
    assert len(loader.get_all_teams()) == 2
    
    assert "Sample Team 1 Content" in loader.get_sample_teams()
    assert "Other Team 1 Content" in loader.get_other_teams()

def test_loader_handles_missing_base_dir():
    # Should not crash if directory doesn't exist
    loader = TeamLoader(base_dir="/non/existent/path")
    assert len(loader.get_all_teams()) == 0

def test_loader_handles_invalid_json(mock_teams_structure):
    # Add an invalid JSON file
    invalid_json_dir = os.path.join(mock_teams_structure, "invalid")
    os.makedirs(invalid_json_dir)
    with open(os.path.join(invalid_json_dir, "teams.json"), "w") as f:
        f.write("not a json")
        
    # Should not crash
    loader = TeamLoader(base_dir=mock_teams_structure)
    # Should still find the other valid teams
    assert len(loader.get_all_teams()) >= 2

def test_loader_handles_missing_team_files(mock_teams_structure):
    # Add a JSON entry pointing to a missing file
    missing_file_dir = os.path.join(mock_teams_structure, "missing")
    os.makedirs(missing_file_dir)
    with open(os.path.join(missing_file_dir, "teams.json"), "w") as f:
        json.dump([{"file": "teams/missing/ghost.txt"}], f)
        
    # Should not crash, just skip the missing file
    loader = TeamLoader(base_dir=mock_teams_structure)
    # ghost.txt doesn't exist, so it shouldn't be added to any list
    for team in loader.get_all_teams():
        assert "ghost" not in team
