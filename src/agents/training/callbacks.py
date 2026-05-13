import os
import json
import glob
import asyncio
import threading
import traceback
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
from utils.gen3_utils import SwitchDetection

BATTLE_FORMAT = "gen3ou"

def init_stats():
    """Helper to initialize stats for a battle."""
    return {
        "voluntary_switches": 0,
        "forced_switches": 0,
        "unknown_switches": 0,
        "moves": {} # Species -> {Move -> Count}
    }

class StatTrackingRLPlayer(RLPlayer):
    """Extends RLPlayer to log granular statistics and action distributions."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.battle_summaries = {} 

    def choose_move(self, battle):
        if battle.battle_tag not in self.battle_summaries:
            stats = init_stats()
            stats["turn_log"] = {}
            stats["switch_events"] = {} # Turn -> "From -> To"
            stats["last_mon"] = "NULL"
            stats["last_mask"] = None
            stats["last_opp_faints"] = 0
            stats["last_our_faints"] = 0
            self.battle_summaries[battle.battle_tag] = stats
        
        stats = self.battle_summaries[battle.battle_tag]
        # Treat fainted as NONE to prevent ghost voluntary switches
        is_fainted = battle.active_pokemon.fainted if battle.active_pokemon else False
        current_mon = "NONE" if is_fainted or not battle.active_pokemon else battle.active_pokemon.species
        
        # 1. Retroactive Result Logging (Record what happened on the PREVIOUS turn)
        prev_turn = battle.turn - 1
        if prev_turn in stats["turn_log"]:
            results = []
            opp_faints = len([m for m in battle.opponent_team.values() if m.fainted])
            our_faints = len([m for m in battle.team.values() if m.fainted])
            
            if opp_faints > stats["last_opp_faints"]: results.append("Opponent Fainted")
            if our_faints > stats["last_our_faints"]: results.append("Our Mon Fainted")
            
            stats["turn_log"][prev_turn]["result"] = ", ".join(results) if results else "None"
            stats["last_opp_faints"] = opp_faints
            stats["last_our_faints"] = our_faints

        # 2. Use centralized prediction logic from parent RLPlayer
        idx, probs, mask = self._predict_best_action(battle, stochastic=True)
            
        # 2. Detect and log ACTUAL Switch Transitions (the source of truth)
        has_switched, is_real, is_voluntary = SwitchDetection.get_switch_type(
            stats["last_mon"], current_mon, stats["last_mask"]
        )
        
        if has_switched and is_real:
            if is_voluntary is True:
                label = "voluntary"
                stats["voluntary_switches"] += 1
            elif is_voluntary is False:
                label = "forced"
                stats["forced_switches"] += 1
            else:
                label = "unknown"
                stats["unknown_switches"] += 1
            
            event_str = f"{stats['last_mon']} -> {current_mon} ({label})"
            stats["switch_events"][str(battle.turn)] = event_str

        # 3. Enhanced Turn Logging (Forensic Metadata)
        available_moves = list(battle.available_moves)
        
        # Map the chosen action to a label
        action_label = "unknown"
        
        team_list = list(battle.team.values())
        if idx < 6:
            mon_name = team_list[idx].species.capitalize() if idx < len(team_list) else f"Slot_{idx}"
            action_label = f"switch_{mon_name.lower()}"
        elif idx < 10:
            move_idx = idx - 6
            move_id = available_moves[move_idx].id if move_idx < len(available_moves) else f"move_{move_idx}"
            action_label = move_id
        elif idx == 10:
            action_label = "struggle"
            
        # Construct mapped options for easier debugging
        options = {}
        for i in range(11):
            label = "unknown"
            if i < 6: 
                label = f"switch_{team_list[i].species}" if i < len(team_list) else f"slot_{i}"
            elif i < 10: 
                label = available_moves[i-6].id if (i-6) < len(available_moves) else f"move_{i-6}"
            elif i == 10: 
                label = "struggle"
            
            status = "OK" if mask[i] else "NO"
            options[label] = f"{probs[i]:.4f} [{status}]"

        our_mon = "unknown"
        if battle.active_pokemon:
            our_hp = battle.active_pokemon.current_hp_fraction * 100
            our_mon = f"{battle.active_pokemon.species} ({our_hp:.0f}%)"

        opp_mon = "unknown"
        if battle.opponent_active_pokemon:
            hp_pct = battle.opponent_active_pokemon.current_hp_fraction * 100
            opp_mon = f"{battle.opponent_active_pokemon.species} ({hp_pct:.0f}%)"

        stats["turn_log"][battle.turn] = {
            "action": action_label,
            "active_mon": our_mon,
            "opponent_mon": opp_mon,
            "opponent_revealed_moves": [m.id for m in battle.opponent_active_pokemon.moves.values()] if battle.opponent_active_pokemon else [],
            "action_idx": int(idx),
            "result": "Pending...",
            "options": options
        }
        
        # 4. Track Move Frequency Stats
        if (idx >= 6 and idx < 10) or idx == 10:
            if idx == 10:
                move_name = "struggle"
            else:
                move_idx = idx - 6
                move_name = available_moves[move_idx].id if move_idx < len(available_moves) else f"move_{move_idx}"
                
            if current_mon not in stats["moves"]: stats["moves"][current_mon] = {}
            stats["moves"][current_mon][move_name] = stats["moves"][current_mon].get(move_name, 0) + 1

        # Update trackers for next turn
        stats["last_mon"] = current_mon
        stats["last_mask"] = mask.copy()

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
            stats["voluntary_switches"] += 1
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
            # daemon=True ensures the thread dies if the main process exits
            thread = threading.Thread(target=self._run_async_battles, args=(step_dir,), daemon=True)
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
            html_files = sorted(glob.glob(os.path.join(step_dir, "*.html")))
            for i, tag in enumerate(replay_player.battle_summaries.keys()):
                battle = replay_player._battles.get(tag)
                if not battle: continue
                
                # Final Turn Result Patching
                stats = replay_player.battle_summaries[tag]
                last_turn = battle.turn
                if last_turn in stats["turn_log"]:
                    results = []
                    opp_faints = len([m for m in battle.opponent_team.values() if m.fainted])
                    our_faints = len([m for m in battle.team.values() if m.fainted])
                    if opp_faints > stats["last_opp_faints"]: results.append("Opponent Fainted")
                    if our_faints > stats["last_our_faints"]: results.append("Our Mon Fainted")
                    stats["turn_log"][last_turn]["result"] = ", ".join(results) if results else "None"

                summary = {
                    "step": self.num_timesteps,
                    "battle_id": tag,
                    "winner": "US" if battle.won else "THEM",
                    "total_turns": battle.turn,
                    "our_team_stats": {
                        "switches": {
                            "total": (replay_player.battle_summaries[tag]["voluntary_switches"] + 
                                      replay_player.battle_summaries[tag]["forced_switches"] +
                                      replay_player.battle_summaries[tag]["unknown_switches"]),
                            "voluntary": replay_player.battle_summaries[tag]["voluntary_switches"],
                            "forced": replay_player.battle_summaries[tag]["forced_switches"],
                            "unknown": replay_player.battle_summaries[tag]["unknown_switches"],
                            "log": replay_player.battle_summaries[tag]["switch_events"]
                        },
                        "moves_per_pokemon": replay_player.battle_summaries[tag]["moves"],
                        "final_health": {m.species: f"{m.current_hp_fraction*100:.1f}%" for m in battle.team.values()},
                        "turn_log": replay_player.battle_summaries[tag]["turn_log"]
                    },
                    "opp_team_stats": {
                        "switches": {
                            "total": replay_opp.battle_summaries[tag]["voluntary_switches"] + replay_opp.battle_summaries[tag]["forced_switches"] if tag in replay_opp.battle_summaries else 0,
                            "voluntary": replay_opp.battle_summaries[tag]["voluntary_switches"] if tag in replay_opp.battle_summaries else 0,
                            "forced": replay_opp.battle_summaries[tag]["forced_switches"] if tag in replay_opp.battle_summaries else 0
                        },
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
        
        def exception_handler(loop, context):
            exception = context.get("exception")
            msg = exception if exception else context["message"]
            print(f"\n🛑 [REPLAY FATAL] Background task failed: {msg}")
            if exception:
                traceback.print_stack()
            os._exit(1)

        new_loop.set_exception_handler(exception_handler)

        try:
            new_loop.run_until_complete(run_it())
        except Exception as e:
            print(f"\n🛑 [REPLAY CRASH] Step {self.num_timesteps} failed: {e}")
            traceback.print_exc()
            os._exit(1)
        finally:
            new_loop.close()
