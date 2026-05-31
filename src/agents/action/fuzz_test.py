import asyncio
import numpy as np
import sys
import os
import traceback

# Ensure project root is in PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../src")))

import time
from poke_env import AccountConfiguration
from poke_env.player import Player, RandomPlayer
from poke_env.ps_client.server_configuration import LocalhostServerConfiguration
from agents.action.mapper import Gen3ActionMapper
from agents.action.mask_generator import Gen3ActionMasker
from agents.action.serialize import choice_to_order
from agents.battle.live_view import LegalActions
from utils.team_loader.loader import TeamLoader
from utils.teambuilder import Gen3Teambuilder
from utils.bridge.local_battle_runner import run_local_battles

class IntegrityFuzzPlayer(Player):
    """
    A specialized agent for fuzz-testing the Gen 3 action system.
    It performs exhaustive validation of the mask and mapper on every turn.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.total_turns_validated = 0
        self.battles_processed = 0

    def choose_move(self, battle):
        try:
            if battle.turn == 1:
                print(f"  [BATTLE START] {battle.battle_tag}")
            
            # 1. Snapshot the server-authoritative legality and build the mask from it.
            legal = LegalActions.from_battle(battle)
            mask = Gen3ActionMasker.get_mask(battle, legal=legal)
            self.total_turns_validated += 1
            
            if self.total_turns_validated % 100 == 0:
                print(f"  [PROGRESS] Validated {self.total_turns_validated} turns...")
            # (Moves 6-9, Struggle 10)
            if battle.available_moves and not battle.force_switch:
                if np.sum(mask[6:11]) == 0:
                    raise RuntimeError(f"INTEGRITY FAILURE: Moves available but none masked! Turn {battle.turn}")
            
            # (Switches 0-5)
            if battle.available_switches:
                if np.sum(mask[0:6]) == 0:
                    raise RuntimeError(f"INTEGRITY FAILURE: Switches available but none masked! Turn {battle.turn}")

            # 3. Exhaustive Mapping Check
            # We try to map EVERY valid action index to a BattleOrder.
            # This ensures that any action the model MIGHT take is guaranteed to work.
            valid_indices = np.where(mask == 1)[0]
            if len(valid_indices) == 0:
                raise RuntimeError(f"INTEGRITY FAILURE: Action mask is completely empty! Turn {battle.turn}")

            for idx in valid_indices:
                try:
                    # Pure decode against the captured snapshot, then serialize — the
                    # exact two-stage seam the env/player use.
                    decoded = Gen3ActionMapper.action_to_choice(int(idx), legal)
                    order = choice_to_order(decoded, battle)
                    if order is None:
                        raise RuntimeError(f"INTEGRITY FAILURE: Action {idx} returned None order!")
                except Exception as e:
                    raise RuntimeError(f"INTEGRITY FAILURE: Action {idx} masked as valid but mapper failed: {e}")

            # 4. Stochastic Action Selection
            # We pick a random valid action to drive the battle forward (via the
            # convenience that composes decode + serialize against the same snapshot).
            choice = np.random.choice(valid_indices)
            return Gen3ActionMapper.action_to_order(int(choice), battle, legal=legal, mask=mask)

        except Exception as e:
            print(f"\n🛑 [FUZZ TEST CRITICAL FAILURE] 🛑")
            print(f"Battle: {battle.battle_tag}, Turn: {battle.turn}")
            print(f"Error: {e}")
            traceback.print_exc()
            # Stop immediately on failure
            os._exit(1)

async def run_fuzz_test(n_battles=20):
    print(f"🚀 Starting Action System Fuzz Test: {n_battles} battles...")
    
    # Load valid Gen 3 teams to avoid server rejections
    loader = TeamLoader()
    teams = loader.get_all_teams()
    if not teams:
        print("Warning: No teams found, using a basic placeholder (might fail server validation)")
        teams = ["Pikachu|||Thunderbolt|"] # Not a real ADV team but for structural test
    
    teambuilder = Gen3Teambuilder(teams)
    
    ts = int(time.time())
    
    player = IntegrityFuzzPlayer(
        battle_format="gen3ou",
        team=teambuilder,
        server_configuration=LocalhostServerConfiguration, start_listening=False,
        account_configuration=AccountConfiguration(f"RLAgentFuzz{ts}", "password"),
        max_concurrent_battles=10
    )
    
    opponent = RandomPlayer(
        battle_format="gen3ou",
        team=teambuilder,
        server_configuration=LocalhostServerConfiguration, start_listening=False,
        account_configuration=AccountConfiguration(f"OppFuzz{ts}", "password"),
        max_concurrent_battles=10
    )

    start_time = asyncio.get_event_loop().time()
    await run_local_battles(player, opponent, n_battles)
    duration = asyncio.get_event_loop().time() - start_time
    
    print("\n" + "✅" * 20)
    print("ACTION SYSTEM INTEGRITY VERIFIED")
    print(f"Battles Completed: {n_battles}")
    print(f"Total Turns Hardened: {player.total_turns_validated}")
    print(f"Total Time: {duration:.2f}s")
    print("✅" * 20 + "\n")

if __name__ == "__main__":
    # Optional: allow setting number of battles via CLI
    num = 50
    if len(sys.argv) > 1:
        num = int(sys.argv[1])
    asyncio.run(run_fuzz_test(num))
