"""The training-transport envs the SOAK (and, with `--obs-source core`, slice N) drive.

The soak env is the bridge integration test's shape (`utils/bridge/bridge_session_integration_test`):
a real ``Gen3Env`` over ONE persistent Rust ``sim_bridge`` child (``attach_bridge_transport(impl=
"rust")`` — the self-check build under ``POKESIM_EMISSION_SELFCHECK=1``), a ``RandomPlayer``
opponent, pool teams; ``recycle_every`` above the unit's episode count, so a child REPLACEMENT inside
a unit is a crash, never routine recycling.
"""
from __future__ import annotations

from typing import Tuple


def build_soak_env(tag: str, obs_source: str = "python", recycle_every: int = 100_000) -> Tuple[object, object]:
    from poke_env import AccountConfiguration
    from poke_env.environment.single_agent_wrapper import SingleAgentWrapper
    from poke_env.player import RandomPlayer

    from agents.observation.state_encoder import load_mappings
    from agents.training.gen3_env import Gen3Env
    from utils.bridge.bridge_session import attach_bridge_transport
    from utils.team_sources import team_list
    from utils.teambuilder import Gen3Teambuilder

    teams = team_list("pool")
    kw = {}
    if obs_source != "python":
        kw["obs_source"] = obs_source
    env = Gen3Env(load_mappings(), battle_format="gen3ou", team=Gen3Teambuilder(teams),
                  account_configuration1=AccountConfiguration(f"{tag}e"[:18], None),
                  start_listening=False, **kw)
    bkw = {"core_obs": True} if obs_source != "python" else {}
    session = attach_bridge_transport(env, battle_format="gen3ou", persistent=True,
                                      recycle_every=recycle_every, impl="rust", **bkw)
    opponent = RandomPlayer(battle_format="gen3ou", team=Gen3Teambuilder(teams),
                            account_configuration=AccountConfiguration(f"{tag}o"[:18], None),
                            start_listening=False)
    wrapped = SingleAgentWrapper(env, opponent)
    wrapped.action_space = env.action_space
    wrapped.observation_space = env.observation_space
    return wrapped, session


def run_envn(stream, i, out):  # slice N — built with `--obs-source core` (phase C)
    raise NotImplementedError("slice N needs `--obs-source core` in this pin")
