import asyncio
import json
import os
import random
from poke_env.player import RandomPlayer
from utils.teambuilder import Gen3Teambuilder
from poke_env.ps_client import LocalhostServerConfiguration

from utils.bridge.team_validator import validate_team_locally

async def get_valid_random_team():
    """Selects a random team and validates it. Retries until a valid team is found."""
    metadata_path = "data/teams/teams.json"
    if not os.path.exists(metadata_path):
        raise FileNotFoundError("Sample teams not found. Run 'npm run sync-teams' first.")

    with open(metadata_path, 'r') as f:
        teams = json.load(f)

    if not teams:
        raise ValueError("No teams found in teams.json.")

    while True:
        team_meta = random.choice(teams)
        team_file = os.path.join("data", team_meta["file"])
        
        with open(team_file, 'r') as f:
            team_text = f.read()
            
        # Validate the team locally using the bridge
        result = validate_team_locally("gen3ou", team_text)
        if result["valid"]:
            return team_meta, team_text
        else:
            print(f" Skipping invalid team: {team_meta['name']} - {result['errors'][0]}")

async def main():
    # Pick a random team for Player 1
    meta1, text1 = await get_valid_random_team()
    builder1 = Gen3Teambuilder(text1)
    
    # Pick a random team for Player 2
    meta2, text2 = await get_valid_random_team()
    builder2 = Gen3Teambuilder(text2)

    player_1 = RandomPlayer(
        battle_format="gen3ou",
        team=builder1,
        server_configuration=LocalhostServerConfiguration,
        max_concurrent_battles=1
    )
    
    player_2 = RandomPlayer(
        battle_format="gen3ou",
        team=builder2,
        server_configuration=LocalhostServerConfiguration,
        max_concurrent_battles=1
    )

    print(f"Matchup:")
    print(f"P1: [{meta1['category']}] {meta1['name']} by {meta1['author']}")
    print(f"P2: [{meta2['category']}] {meta2['name']} by {meta2['author']}")
    print("-" * 40)

    # Run the battle
    await player_1.battle_against(player_2, n_battles=1)

    print("-" * 40)
    print(f"Battle Finished!")
    if player_1.n_won_battles > 0:
        print("Winner: Player 1")
    else:
        print("Winner: Player 2")

if __name__ == "__main__":
    asyncio.run(main())
