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


def test_dispatch_routes_recon_to_single_slot_not_clients():
    """A ``__RECON__`` frame (full-information record) must never be fed to a
    player client — it lands in the single-slot ``last_recon`` stash. A
    regression here either crashes the reader (parsed as a side chunk) or, far
    worse, leaks referee-view data into a player's one-sided message stream."""
    env = _bridge_env(4)
    try:
        session = env._bridge_session
        session._tag = "battle-gen3ou-recontest"

        def _boom(*a, **k):  # any feed attempt = the wall was breached
            raise AssertionError("__RECON__ frame was fed to a player client")

        session.c1.feed = _boom
        session.c2.feed = _boom
        session._dispatch("__RECON__ eyJ2IjogMX0=")  # b64 of {"v": 1}
        assert session.last_recon == ("battle-gen3ou-recontest", "eyJ2IjogMX0=")
    finally:
        env.close()


def test_child_error_wakes_a_blocked_queue_get_instead_of_hanging():
    """A dead bridge child must WAKE an in-flight ``step()``/``reset()``, not let it sit out
    poke-env's watchdog (`gen3_bridge_child_error_wakes_step_v1`).

    Latching ``_child_error`` alone only makes the NEXT ``reset()`` raise; a ``step()`` already
    parked in ``battle_queue.race_get`` waits for a request the now-dead reader can never
    deliver. poke-env already solves this shape for a dropped websocket — ``listen`` sets
    ``_disconnected`` and ``_AsyncQueue`` races its gets against it — so the bridge reuses that
    signal.

    THE LOAD-BEARING DETAIL this pins: the queues bind ``_disconnected`` at CONSTRUCTION, from
    the ORIGINAL ps_client, and ``attach()`` then swaps in a ``BattleStreamClient`` with its own
    fresh event. Signalling the NEW client's event would wake nothing. So we assert the events
    the session signals are the ones the QUEUES actually race on — an implementation that
    signalled the client instead passes a naive "is_set" check on the client and still hangs.
    """
    env = _bridge_env(5)
    try:
        session = env._bridge_session
        q1, q2 = env.agent1.battle_queue, env.agent2.battle_queue
        assert q1._disconnected is not None and q2._disconnected is not None
        # The session must hold the QUEUES' events (identity, not just equality).
        assert any(ev is q1._disconnected for ev in session._disconnect_events)
        assert any(ev is q2._disconnected for ev in session._disconnect_events)
        # ...and those must NOT be the swapped-in bridge client's own fresh event, which is what
        # a plausible-but-wrong implementation would have signalled.
        assert q1._disconnected is not session.c1._disconnected

        assert not q1._disconnected.is_set() and not q2._disconnected.is_set()
        session._signal_transport_dead()
        assert q1._disconnected.is_set(), "a dead child must wake agent1's blocked get"
        assert q2._disconnected.is_set(), "a dead child must wake agent2's blocked get"
        session._signal_transport_dead()  # idempotent — a second fatal path must not throw
    finally:
        env.close()
