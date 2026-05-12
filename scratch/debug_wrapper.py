import sys
sys.path.append('src')
from agents.observation.state_encoder import load_mappings
from utils.team_loader import TeamLoader
from utils.teambuilder import Gen3Teambuilder
from main.train_rl_agent import Gen3Env
from poke_env.player import SimpleHeuristicsPlayer
from poke_env import AccountConfiguration, LocalhostServerConfiguration
from datetime import datetime
from agents.rl_agent import SingleAgentWrapper

loader = TeamLoader()
teams = loader.get_sample_teams()
tb = Gen3Teambuilder(teams)
mappings = load_mappings()

ts = datetime.now().strftime('%H%M%S')
env = Gen3Env(mappings, battle_format="gen3ou", team=tb, server_configuration=LocalhostServerConfiguration, account_configuration1=AccountConfiguration(f"Test{ts}", "password"))
opp = SimpleHeuristicsPlayer(battle_format="gen3ou", team=tb, server_configuration=LocalhostServerConfiguration, account_configuration=AccountConfiguration(f"Opp{ts}", "password"))

wrapped = SingleAgentWrapper(env, opp)
obs, info = wrapped.reset()

print("KEYS:", obs.keys())
print("obs['action_mask']:", obs['action_mask'])
print("obs['action_mask'] shape:", obs['action_mask'].shape)
