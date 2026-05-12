import sys
sys.path.append('src')
from agents.observation.state_encoder import load_mappings
from utils.team_loader import TeamLoader
from utils.teambuilder import Gen3Teambuilder
from main.train_rl_agent import Gen3Env, Gen3SingleAgentWrapper
from poke_env.player import SimpleHeuristicsPlayer
from poke_env import AccountConfiguration, LocalhostServerConfiguration
from datetime import datetime
from sb3_contrib import MaskablePPO
import torch
import numpy as np

loader = TeamLoader()
teams = loader.get_sample_teams()
tb = Gen3Teambuilder(teams)
mappings = load_mappings()

ts = datetime.now().strftime('%H%M%S')
env = Gen3Env(mappings, battle_format="gen3ou", team=tb, server_configuration=LocalhostServerConfiguration, account_configuration1=AccountConfiguration(f"Test{ts}", "password"))
opp = SimpleHeuristicsPlayer(battle_format="gen3ou", team=tb, server_configuration=LocalhostServerConfiguration, account_configuration=AccountConfiguration(f"Opp{ts}", "password"))

wrapped = Gen3SingleAgentWrapper(env, opp)
obs, info = wrapped.reset()

model = MaskablePPO("MultiInputPolicy", wrapped, verbose=0, device="cpu")

# Get mask
mask = wrapped.action_masks()
obs_dict = {
    "observation": torch.tensor(obs["observation"]).unsqueeze(0),
    "action_mask": torch.tensor(obs["action_mask"]).unsqueeze(0)
}

with torch.no_grad():
    actions, values, log_probs = model.policy(obs_dict, action_masks=torch.tensor(mask).unsqueeze(0))
    dist = model.policy.get_distribution(obs_dict)
    logits = dist.distribution.logits
    print("Original Logits:", logits)
    masked_logits = logits + (torch.tensor(mask).unsqueeze(0) - 1.0) * 1e9
    print("Masked Logits:", masked_logits)
    print("Action taken:", actions.item())
