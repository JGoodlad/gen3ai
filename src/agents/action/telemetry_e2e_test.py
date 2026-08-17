import asyncio
import numpy as np
import sys
import os
import time

# Ensure project root is in PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../src")))

from poke_env.player import Player, RandomPlayer
from poke_env import AccountConfiguration
from agents.action.mapper import Gen3ActionMapper
from agents.action.mask_generator import Gen3ActionMasker
from utils.team_loader.loader import TeamLoader
from utils.teambuilder import Gen3Teambuilder

class TelemetryDiagnosticPlayer(Player):
    """
    Plays real games and detects if the server state changes 
    during the model's decision window.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.silent_updates_detected = 0
        self.total_turns = 0

    def choose_move(self, battle):
        self.total_turns += 1
        if self.total_turns % 10 == 0:
            print(f"  [BATTLE PROGRESS] Battle {battle.battle_tag}, Turn {battle.turn}")
        
        # 1. Compare Current Request vs. Alphabetical
        current_request = battle.last_request
        mask = Gen3ActionMasker.get_mask(battle)
        if current_request:
            active_request = current_request.get("active", [{}])[0]
            server_moves = [m.get("id") for m in active_request.get("moves", [])]
            alpha_moves = sorted(battle.active_pokemon.moves.keys())
            
            if server_moves and server_moves != alpha_moves:
                self.silent_updates_detected += 1 # We'll repurpose this for 'desyncs detected'
                if self.silent_updates_detected <= 3:
                    print(f"\n📢 REAL-GAME DESYNC DETECTED (Turn {battle.turn}):")
                    print(f"  Server Slots: {server_moves}")
                    print(f"  Alpha Order:  {alpha_moves}")

        # 4. EXECUTE
        # Pick random valid action
        valid_indices = np.where(mask == 1)[0]
        choice = np.random.choice(valid_indices)
        return Gen3ActionMapper.action_to_order(int(choice), battle)

async def run_telemetry(n_battles=10):
    print(f"🛰️ Starting Real-Game Telemetry: {n_battles} battles...")
    
    loader = TeamLoader()
    teams = loader.get_all_teams()
    teambuilder = Gen3Teambuilder(teams)
    
    ts = int(time.time())
    
    # Configure a custom server config for port 8001
    from poke_env.ps_client.server_configuration import ServerConfiguration
    ShadowServer = ServerConfiguration("ws://localhost:8001/showdown/websocket", None)
    
    player = TelemetryDiagnosticPlayer(
        battle_format="gen3ou",
        team=teambuilder,
        server_configuration=ShadowServer,
        account_configuration=AccountConfiguration(f"RLDiag{ts}", None),
        max_concurrent_battles=1
    )
    
    opponent = RandomPlayer(
        battle_format="gen3ou",
        team=teambuilder,
        server_configuration=ShadowServer,
        account_configuration=AccountConfiguration(f"OppDiag{ts}", None),
        max_concurrent_battles=1
    )

    await player.battle_against(opponent, n_battles=n_battles)
    
    print("\n" + "="*40, flush=True)
    print("TELEMETRY RESULTS", flush=True)
    print(f"Total Turns Played: {player.total_turns}", flush=True)
    print(f"Silent Updates Detected: {player.silent_updates_detected}", flush=True)
    if player.total_turns > 0:
        print(f"Empirical Desync Rate: {(player.silent_updates_detected / player.total_turns * 100):.2f}%", flush=True)
    print("="*40, flush=True)

if __name__ == "__main__":
    asyncio.run(run_telemetry())
