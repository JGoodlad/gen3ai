import sys
sys.path.append('src')
from agents.observation.state_encoder import load_mappings
from utils.team_loader import TeamLoader
from utils.teambuilder import Gen3Teambuilder
from main.train_rl_agent import Gen3Env
from poke_env.player import SimpleHeuristicsPlayer
from poke_env import AccountConfiguration, LocalhostServerConfiguration
from datetime import datetime

loader = TeamLoader()
teams = loader.get_sample_teams()
tb = Gen3Teambuilder(teams)
mappings = load_mappings()

ts = datetime.now().strftime('%H%M%S')
env = Gen3Env(mappings, battle_format="gen3ou", team=tb, server_configuration=LocalhostServerConfiguration, account_configuration1=AccountConfiguration(f"Test{ts}", "password"))

obs_raw, info = env.reset()
print("Raw obs keys:", obs_raw.keys())
player_obs = obs_raw[env.agent1.username]
print("player_obs keys:", player_obs.keys())
print("player_obs['observation'] type:", type(player_obs['observation']))
if isinstance(player_obs['observation'], dict):
    print("player_obs['observation'] keys:", player_obs['observation'].keys())
print("player_obs['action_mask'] shape:", player_obs['action_mask'].shape)

