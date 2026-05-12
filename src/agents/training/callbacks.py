import os
import json
import asyncio
import threading
import numpy as np
import torch
from datetime import datetime
from stable_baselines3.common.callbacks import BaseCallback
from poke_env.player import Player
from poke_env.player.battle_order import BattleOrder, DefaultBattleOrder
from poke_env.environment.singles_env import SinglesEnv
from poke_env.player import Player, RandomPlayer, SimpleHeuristicsPlayer
from poke_env.ps_client import LocalhostServerConfiguration, AccountConfiguration
from agents.rl_agent import RLPlayer
from agents.action.mask_generator import Gen3ActionMasker

BATTLE_FORMAT = "gen3ou"

def init_stats():
    """Helper to initialize stats for a battle."""
    return {
        "switches_made": 0,
        "moves": {} # Species -> {Move -> Count}
    }

class StatTrackingRLPlayer(RLPlayer):
    """Extends RLPlayer to log granular statistics and action distributions."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.battle_summaries = {} 

    def choose_move(self, battle):
        if battle.battle_tag not in self.battle_summaries:
            self.battle_summaries[battle.battle_tag] = init_stats()
            self.battle_summaries[battle.battle_tag]["turn_log"] = {}
        
        # 1. Use centralized prediction logic from parent RLPlayer
        idx, probs, mask = self._predict_best_action(battle)
            
        # 2. Log probabilities and mask for this turn
        stats = self.battle_summaries[battle.battle_tag]
        stats["turn_log"][battle.turn] = {
            "action_idx": idx,
            "probabilities": [round(float(p), 4) for p in probs],
            "mask": mask.tolist()
        }
        
        # 3. Track Stats
        if idx < 6:
            stats["switches_made"] += 1
        else:
            move_slot = idx - 6
            available = battle.available_moves
            if move_slot < len(available):
                mon_name = battle.active_pokemon.species
                move_name = available[move_slot].id
                if mon_name not in stats["moves"]: stats["moves"][mon_name] = {}
                stats["moves"][mon_name][move_name] = stats["moves"][mon_name].get(move_name, 0) + 1

        # 4. Use Centralized Absolute Mapping
        return self.action_to_order(idx, battle)

class StatTrackingHeuristicPlayer(SimpleHeuristicsPlayer):
    """Extends SimpleHeuristicsPlayer to capture opponent behavior statistics."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.battle_summaries = {}

    def choose_move(self, battle):
        if battle.battle_tag not in self.battle_summaries:
            self.battle_summaries[battle.battle_tag] = init_stats()
        
        order = super().choose_move(battle)
        stats = self.battle_summaries[battle.battle_tag]
        
        from poke_env.battle.move import Move
        
        if isinstance(order.order, Move):
            mon_name = battle.active_pokemon.species
            move_name = order.order.id
            if mon_name not in stats["moves"]: stats["moves"][mon_name] = {}
            stats["moves"][mon_name][move_name] = stats["moves"][mon_name].get(move_name, 0) + 1
        elif not isinstance(order, DefaultBattleOrder):
            stats["switches_made"] += 1
        return order

class ReplayCallback(BaseCallback):
    """
    Captures full battle replays and detailed JSON summaries at specific milestones.
    Saves them to the model directory in a /replays subfolder.
    """
    def __init__(self, model_dir, mappings, trainee_teambuilder, opponent_teambuilder, save_freq=100000, n_replays=3, verbose=0):
        super().__init__(verbose)
        self.model_dir = model_dir
        self.mappings = mappings
        self.trainee_teambuilder = trainee_teambuilder
        self.opponent_teambuilder = opponent_teambuilder
        self.save_freq = save_freq 
        self.n_replays = n_replays
        self.replay_dir = os.path.join(model_dir, "replays")
        os.makedirs(self.replay_dir, exist_ok=True)
        self.last_save = 0

    def _on_step(self) -> bool:
        # HUMAN MILESTONE RAMP
        if self.num_timesteps < 1_000_000:
            interval = 200_000
        elif self.num_timesteps < 10_000_000:
            interval = 1_000_000
        else:
            interval = 2_000_000
            
        trigger = False
        if self.last_save == 0 and self.num_timesteps > 0:
            trigger = True # First update proof-of-life
        elif (self.num_timesteps // interval) > (self.last_save // interval):
            trigger = True

        if trigger:
            self.last_save = self.num_timesteps
            step_dir = os.path.join(self.replay_dir, f"step_{self.num_timesteps}")
            os.makedirs(step_dir, exist_ok=True)
            print(f"\n🎥 [REPLAY] Step {self.num_timesteps}: Recording {self.n_replays} games to {step_dir}...")
            
            # Use threading to run async battles without blocking SB3's main loop
            thread = threading.Thread(target=self._run_async_battles, args=(step_dir,))
            thread.start()
            # We join here to ensure replays are fully recorded before continuing
            thread.join() 
            
        return True

    def _run_async_battles(self, step_dir):
        async def run_it():
            ts = datetime.now().strftime('%H%M%S')
            
            replay_player = StatTrackingRLPlayer(
                model=self.model,
                team=self.trainee_teambuilder,
                battle_format=BATTLE_FORMAT,
                server_configuration=LocalhostServerConfiguration,
                mappings=self.mappings,
                account_configuration=AccountConfiguration(f"Replay{ts}", "password"),
                save_replays=step_dir
            )
            
            replay_opp = StatTrackingHeuristicPlayer(
                battle_format=BATTLE_FORMAT,
                team=self.opponent_teambuilder,
                account_configuration=AccountConfiguration(f"RepOpp{ts}", "password"),
                server_configuration=LocalhostServerConfiguration,
            )
            
            await replay_player.battle_against(replay_opp, n_battles=self.n_replays)
            
            # Combined Summary Generation
            import glob
            html_files = sorted(glob.glob(os.path.join(step_dir, "*.html")))
            for i, tag in enumerate(replay_player.battle_summaries.keys()):
                battle = replay_player._battles.get(tag)
                if not battle: continue
                
                summary = {
                    "step": self.num_timesteps,
                    "battle_id": tag,
                    "winner": "US" if battle.won else "THEM",
                    "total_turns": battle.turn,
                    "our_team_stats": {
                        "total_switches": replay_player.battle_summaries[tag]["switches_made"],
                        "moves_per_pokemon": replay_player.battle_summaries[tag]["moves"],
                        "final_health": {m.species: f"{m.current_hp_fraction*100:.1f}%" for m in battle.team.values()},
                        "turn_log": replay_player.battle_summaries[tag]["turn_log"]
                    },
                    "opp_team_stats": {
                        "total_switches": replay_opp.battle_summaries[tag]["switches_made"] if tag in replay_opp.battle_summaries else 0,
                        "moves_per_pokemon": replay_opp.battle_summaries[tag]["moves"] if tag in replay_opp.battle_summaries else {},
                        "final_health": {m.species: f"{m.current_hp_fraction*100:.1f}%" for m in battle.opponent_team.values()}
                    }
                }
                
                with open(os.path.join(step_dir, f"battle_{i+1}_summary.json"), "w") as f:
                    json.dump(summary, f, indent=4)
                
                if i < len(html_files):
                    os.rename(html_files[i], os.path.join(step_dir, f"battle_{i+1}_replay.html"))

        new_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(new_loop)
        new_loop.run_until_complete(run_it())
        new_loop.close()
