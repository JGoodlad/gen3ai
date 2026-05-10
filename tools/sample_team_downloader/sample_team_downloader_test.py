import pytest
from unittest.mock import patch, MagicMock
from tools.sample_team_downloader.sync import sync_teams
import os

@patch('requests.get')
def test_sync_teams_logic(mock_get):
    # Mock Smogon Thread
    mock_thread_res = MagicMock()
    mock_thread_res.status_code = 200
    mock_thread_res.text = """
    <html>
        <body>
            <div class="message-inner">
                <div class="bbWrapper">
                    <b>Big 5 Team</b>: <a href="https://pokepast.es/test1">https://pokepast.es/test1</a>
                    <br>
                    <b>Superman Team</b>: <a href="https://pokepast.es/test2">Link</a>
                </div>
            </div>
        </body>
    </html>
    """
    
    # Mock PokePaste Raws
    mock_raw_res = MagicMock()
    mock_raw_res.status_code = 200
    mock_raw_res.text = "Pokemon 1 @ Item\n- Move 1"
    
    mock_get.side_effect = [mock_thread_res, mock_raw_res, mock_raw_res]
    
    # Ensure data/teams exists for test
    os.makedirs("data/teams", exist_ok=True)
    
    # Run the sync (with small timeout/sleep mocks if needed, but here it's fine)
    with patch('time.sleep', return_value=None):
        sync_teams()
    
    # Check if files were created
    # Based on our regex/logic, names should be derived from link text or siblings
    # In the mock:
    # 1. link text is URL, previous sibling is "Big 5 Team: "
    # 2. link text is "Link", previous sibling is "Superman Team: "
    
    files = os.listdir("data/teams")
    assert any("big_5_team_test1" in f for f in files)
    assert any("superman_team_test2" in f for f in files)
