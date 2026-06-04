"""Unit guards for the bridge transport swap (no server, no battle, no Node needed).

These pin the *contract* of ``attach_bridge_transport``: after it runs, the env's two
``_EnvPlayer`` agents talk to a ``BattleStreamClient`` (the bridge), expose no ``websocket``
attribute, and route battle start through the session instead of the ``/challenge`` handshake.
A regression here (e.g. the flag silently leaving the websocket transport in place) is exactly
the class of bug ``server_port_threading_test`` guards for the websocket path.
"""

from poke_env import AccountConfiguration

from agents.observation.state_encoder import load_mappings
from agents.training.gen3_env import Gen3Env
from utils.bridge.battle_stream_client import BattleStreamClient
from utils.bridge.bridge_session import BridgeSession, attach_bridge_transport
from utils.team_loader.loader import TeamLoader
from utils.teambuilder import Gen3Teambuilder


def _bridge_env(idx: int) -> Gen3Env:
    loader = TeamLoader()
    teams = loader.get_sample_teams() or loader.get_all_teams()
    env = Gen3Env(
        load_mappings(),
        battle_format="gen3ou",
        team=Gen3Teambuilder(teams),
        account_configuration1=AccountConfiguration(f"BridgeUnit{idx}", None),
        start_listening=False,
    )
    attach_bridge_transport(env, battle_format="gen3ou")
    return env


def test_attach_swaps_both_agents_to_bridge_clients():
    env = _bridge_env(1)
    try:
        assert isinstance(env.agent1.ps_client, BattleStreamClient)
        assert isinstance(env.agent2.ps_client, BattleStreamClient)
        # The two sides are distinct clients (independent listen topology) on the right sides.
        assert env.agent1.ps_client is not env.agent2.ps_client
        assert env.agent1.ps_client._side == "p1"
        assert env.agent2.ps_client._side == "p2"
        # Both bridge clients bind to the env's own loop (where the queues live).
        assert env.agent1.ps_client.loop is env._loop
        assert env.agent2.ps_client.loop is env._loop
    finally:
        env.close()


def test_bridge_client_exposes_no_websocket():
    # poke-env guards (`hasattr(ps_client, "websocket")`) must skip cleanly on the bridge.
    env = _bridge_env(2)
    try:
        assert not hasattr(env.agent1.ps_client, "websocket")
        assert not hasattr(env.agent2.ps_client, "websocket")
    finally:
        env.close()


def test_attach_intercepts_battle_start_and_stashes_session():
    env = _bridge_env(3)
    try:
        session = env._bridge_session
        assert isinstance(session, BridgeSession)
        # reset() kicks battles off via agent1.battle_against — it must now be the session's
        # bridge-start, not the websocket /challenge handshake.
        assert env.agent1.battle_against == session._battle_against
    finally:
        env.close()
