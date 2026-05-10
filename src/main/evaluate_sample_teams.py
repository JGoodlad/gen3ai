import asyncio
import json
import os
import argparse
import time
from typing import Dict, List
from poke_env.player import RandomPlayer, cross_evaluate
from tabulate import tabulate
from utils.teambuilder import Gen3Teambuilder
from utils.bridge.team_validator import validate_team_locally
from poke_env.ps_client import LocalhostServerConfiguration

class Evaluator(RandomPlayer):
    def __init__(self, name, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_name = name
        self.n_ties = 0

    def on_battle_finished(self, battle):
        if battle.tied:
            self.n_ties += 1

async def main():
    parser = argparse.ArgumentParser(description="Evaluate Gen 3 OU Sample Teams")
    parser.add_argument("--n", type=int, default=20, help="Number of challenges per pair")
    args = parser.parse_args()

    # 1. Load and validate all teams
    metadata_path = "data/teams/teams.json"
    if not os.path.exists(metadata_path):
        print("Error: data/teams/teams.json not found. Run sync-teams first.")
        return

    with open(metadata_path, 'r') as f:
        teams_metadata = json.load(f)

    print("Validating teams locally...")
    valid_teams = {} # name -> raw_text
    for meta in teams_metadata:
        team_file = os.path.join("data", meta["file"])
        if not os.path.exists(team_file): continue
        with open(team_file, 'r') as f:
            text = f.read()
        if validate_team_locally("gen3ou", text)["valid"]:
            valid_teams[meta["name"]] = text
    
    num_teams = len(valid_teams)
    num_pairs = (num_teams * (num_teams - 1)) // 2
    total_expected_games = num_pairs * args.n
    
    print(f"Found {num_teams} valid teams.")
    print(f"Starting cross-evaluation: {num_pairs} pairs, {args.n} games/pair (~{total_expected_games} total games).")

    # 2. Setup Players
    players = []
    for name, text in valid_teams.items():
        builder = Gen3Teambuilder(text)
        p = Evaluator(
            name=name,
            battle_format="gen3ou",
            team=builder,
            server_configuration=LocalhostServerConfiguration,
            max_concurrent_battles=10
        )
        players.append(p)

    # 3. Cross Evaluation
    start_time = time.time()
    results = await cross_evaluate(players, n_challenges=args.n)
    end_time = time.time()
    
    duration = end_time - start_time
    bps = total_expected_games / duration if duration > 0 else 0

    # 4. Report Results
    # results is Dict[p1_username, Dict[p2_username, win_rate]]
    table_data = []
    for p1 in players:
        total_wr = 0
        count = 0
        for p2 in players:
            if p1 == p2: continue
            wr = results[p1.username][p2.username]
            if wr is not None:
                total_wr += wr
                count += 1
        
        avg_wr = (total_wr / count * 100) if count > 0 else 0
        table_data.append([
            p1.eval_name, 
            f"{avg_wr:.2f}%", 
            p1.n_ties
        ])

    # Sort results by average win rate
    table_data.sort(key=lambda x: float(x[1].replace("%", "")), reverse=True)

    # Add Headers
    headers = ["Team Name", "Avg Win Rate", "Ties"]
    
    print("\n" + "="*80)
    print("GEN 3 OU SAMPLE TEAM EVALUATION RESULTS")
    print("="*80)
    print(tabulate(table_data, headers=headers, tablefmt="grid"))
    print("="*80)
    print(f"PERFORMANCE STATS")
    print(f"Total Duration: {duration:.2f} seconds")
    print(f"Total Games:    {total_expected_games}")
    print(f"Avg Speed:      {bps:.2f} battles/second")
    print("="*80)

if __name__ == "__main__":
    asyncio.run(main())
