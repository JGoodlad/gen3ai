import asyncio
import json
import os
from poke_env.player import RandomPlayer
from poke_env.ps_client import LocalhostServerConfiguration
from utils.teambuilder import Gen3Teambuilder

async def test_all_teams():
    with open("data/teams/teams.json", "r") as f:
        teams_meta = json.load(f)
    
    all_valid = []
    all_invalid = []
    
    for team_info in teams_meta:
        team_name = team_info["name"]
        print(f"Testing legality of: {team_name}...")
        with open(os.path.join("data", team_info["file"]), "r") as f:
            team_text = f.read()
        
        player1 = RandomPlayer(
            battle_format="gen3ou",
            team=Gen3Teambuilder(team_text),
            server_configuration=LocalhostServerConfiguration,
        )
        player2 = RandomPlayer(
            battle_format="gen3ou",
            team=Gen3Teambuilder(team_text),
            server_configuration=LocalhostServerConfiguration,
        )
        
        try:
            # Try a single battle with a 5s timeout
            await asyncio.wait_for(player1.battle_against(player2, n_battles=1), timeout=5.0)
            print(f"  SUCCESS: {team_name}")
            all_valid.append(team_info)
        except asyncio.TimeoutError:
            print(f"  FAILED: {team_name} - Timeout (likely illegal)")
            all_invalid.append({"name": team_name, "error": "Timeout"})
        except Exception as e:
            print(f"  FAILED: {team_name} - {str(e)}")
            all_invalid.append({"name": team_name, "error": str(e)})
            
    print("\nResults:")
    print(f"Valid: {len(all_valid)}")
    print(f"Invalid: {len(all_invalid)}")
    
    with open("data/teams/valid_teams.json", "w") as f:
        json.dump(all_valid, f, indent=2)

if __name__ == "__main__":
    asyncio.run(test_all_teams())
