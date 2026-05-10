import pytest
from unittest.mock import MagicMock
from bs4 import BeautifulSoup
from tools.sample_team_downloader.sync import extract_metadata

def test_extract_metadata_logic():
    html = """
    <div class="message-inner">
        <div class="bbWrapper">
            <b>Balance</b>
            <br>
            <a href="https://pokepast.es/team1"><img src="sprite1.png"></a>
            <br>
            Big 5 + Starmie – by UD
            <br>
            <b>Offense</b>
            <br>
            <a href="https://pokepast.es/team2"><img src="sprite2.png"></a>
            <br>
            Superman TSS – by ADV Community
        </div>
    </div>
    """
    soup = BeautifulSoup(html, 'lxml')
    first_post = soup.select_one('.bbWrapper')
    
    teams = extract_metadata(first_post)
    
    assert len(teams) == 2
    
    # Team 1
    assert teams[0]["url"] == "https://pokepast.es/team1"
    assert teams[0]["name"] == "Big 5 + Starmie"
    assert teams[0]["author"] == "UD"
    assert teams[0]["category"] == "Balance"
    
    # Team 2
    assert teams[1]["url"] == "https://pokepast.es/team2"
    assert teams[1]["name"] == "Superman TSS"
    assert teams[1]["author"] == "ADV Community"
    assert teams[1]["category"] == "Offense"

def test_extract_metadata_with_author_link():
    html = """
    <div class="bbWrapper">
        <b>Stall</b>
        <br>
        <a href="https://pokepast.es/team3"></a>
        <br>
        Triple Natural Cure – by <a href="/members/abr.123/">ABR</a>
    </div>
    """
    soup = BeautifulSoup(html, 'lxml')
    first_post = soup.select_one('.bbWrapper')
    
    teams = extract_metadata(first_post)
    
    assert len(teams) == 1
    assert teams[0]["name"] == "Triple Natural Cure"
    assert teams[0]["author"] == "ABR"
    assert teams[0]["category"] == "Stall"
