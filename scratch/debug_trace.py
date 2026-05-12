import sys
sys.path.append('src')
import numpy as np
from agents.observation.state_encoder import load_mappings
from utils.team_loader import TeamLoader
from utils.teambuilder import Gen3Teambuilder
from main.train_rl_agent import Gen3Env, Gen3SingleAgentWrapper
from poke_env.player import SimpleHeuristicsPlayer
from poke_env import AccountConfiguration, LocalhostServerConfiguration
from datetime import datetime

class TraceGen3Env(Gen3Env):
    def embed_battle(self, battle):
        res = super().embed_battle(battle)
        is_p1 = (battle is self.battle1)
        mask = self._synced_masks[(battle.battle_tag, is_p1)]
        print(f"DEBUG: [Env] embed_battle for {battle.battle_tag} (P1={is_p1}). Mask: {mask}")
        return res

    def get_action_mask(self, battle):
        mask = super().get_action_mask(battle)
        is_p1 = (battle is self.battle1)
        print(f"DEBUG: [Env] get_action_mask for {battle.battle_tag} (P1={is_p1}). Returning: {mask}")
        return mask

    def action_to_order(self, action, battle, **kwargs):
        is_p1 = (battle is self.battle1)
        print(f"DEBUG: [Env] action_to_order for {battle.battle_tag} (P1={is_p1}). Action: {action}")
        return super().action_to_order(action, battle, **kwargs)

class TraceWrapper(Gen3SingleAgentWrapper):
    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        print(f"DEBUG: [Wrapper] reset updated _last_action_mask to: {self._last_action_mask}")
        return obs, info

    def step(self, action):
        print(f"DEBUG: [Wrapper] step(action={action}) called. Current _last_action_mask: {self._last_action_mask}")
        obs, reward, term, trunc, info = super().step(action)
        print(f"DEBUG: [Wrapper] step returned. Updated _last_action_mask to: {self._last_action_mask}")
        return obs, reward, term, trunc, info

    def action_masks(self):
        mask = super().action_masks()
        print(f"DEBUG: [Wrapper] action_masks() called. Returning: {mask}")
        return mask

loader = TeamLoader()
teams = loader.get_sample_teams()
tb = Gen3Teambuilder(teams)
mappings = load_mappings()

ts = datetime.now().strftime('%H%M%S')
env = TraceGen3Env(mappings, battle_format="gen3ou", team=tb, server_configuration=LocalhostServerConfiguration, account_configuration1=AccountConfiguration(f"Test{ts}", "password"))
opp = SimpleHeuristicsPlayer(battle_format="gen3ou", team=tb, server_configuration=LocalhostServerConfiguration, account_configuration=AccountConfiguration(f"Opp{ts}", "password"))

wrapped = TraceWrapper(env, opp)
obs, info = wrapped.reset()

print("Initial observation generated.")
for i in range(10):
    mask = wrapped.action_masks()
    valid_actions = np.where(mask == 1)[0]
    # Pick a valid action, but let's try to find one that might desync
    action = valid_actions[0]
    print(f"--- Iteration {i} ---")
    try:
        obs, reward, term, trunc, info = wrapped.step(action)
    except Exception as e:
        print(f"CRASH: {e}")
        break
    if term or trunc:
        break
