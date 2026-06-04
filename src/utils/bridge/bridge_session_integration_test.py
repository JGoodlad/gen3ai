"""Integration test: drive a real ``Gen3Env`` through the in-process bridge transport.

Unlike ``local_sim_bridge_integration_test`` (which drives whole battles via the *synchronous*
``run_local_battles``), this exercises the **gym** inversion of control: actions arrive from
*outside* the env via ``step()``, and the bridge must pump battle state into each agent's
``battle_queue`` in the background so ``reset()`` / ``step()`` can read it. It builds a
``Gen3Env`` with ``start_listening=False``, attaches ``BridgeSession`` (no websocket / server /
port), wraps it with ``SingleAgentWrapper`` + a ``RandomPlayer`` opponent, and plays full
episodes with random *legal* actions — asserting each episode completes with a finished battle.

Marked ``integration``: needs Node + ``deps/pokemon-showdown`` (no live server).
"""

import asyncio
import os
import signal

import numpy as np
import pytest

from poke_env import AccountConfiguration
from poke_env.player import RandomPlayer
from poke_env.environment.single_agent_wrapper import SingleAgentWrapper

from agents.observation.state_encoder import load_mappings
from agents.training.gen3_env import Gen3Env
from utils.bridge.bridge_session import attach_bridge_transport
from utils.team_loader.loader import TeamLoader
from utils.teambuilder import Gen3Teambuilder


def _teams():
    loader = TeamLoader()
    return loader.get_sample_teams() or loader.get_all_teams()


def _build_bridge_env(teams, idx: int, persistent: bool = True, recycle_every: int = 1000):
    """A Gen3Env wired to the bridge transport + a RandomPlayer opponent, like the training
    factory's bridge path but minimal (no reward/self-play/eval machinery needed here)."""
    env = Gen3Env(
        load_mappings(),
        battle_format="gen3ou",
        team=Gen3Teambuilder(teams),
        account_configuration1=AccountConfiguration(f"BridgeEnv{idx}", None),
        start_listening=False,  # no websocket — the bridge is the transport
    )
    attach_bridge_transport(
        env, battle_format="gen3ou", persistent=persistent, recycle_every=recycle_every
    )
    opponent = RandomPlayer(
        battle_format="gen3ou",
        team=Gen3Teambuilder(teams),
        account_configuration=AccountConfiguration(f"BridgeOpp{idx}", None),
        start_listening=False,
    )
    wrapped = SingleAgentWrapper(env, opponent)
    # SingleAgentWrapper hardcodes Discrete(10) for gen3ou; Gen3Env uses 11 (see train_rl_agent).
    wrapped.action_space = env.action_space
    wrapped.observation_space = env.observation_space
    return wrapped


def _play_episode(wrapped, rng: np.random.Generator, max_steps: int = 600) -> int:
    obs, _ = wrapped.reset()
    steps = 0
    for _ in range(max_steps):
        mask = np.asarray(obs["action_mask"]).astype(bool)
        legal = np.flatnonzero(mask)
        action = int(rng.choice(legal)) if legal.size else 0
        obs, _reward, terminated, truncated, _info = wrapped.step(action)
        steps += 1
        if terminated or truncated:
            return steps
    raise AssertionError(f"episode did not finish within {max_steps} steps")


@pytest.mark.integration
@pytest.mark.parametrize("persistent", [True, False], ids=["persistent", "spawn_per_battle"])
def test_bridge_env_plays_full_episodes(persistent):
    teams = _teams()
    rng = np.random.default_rng(0)
    wrapped = _build_bridge_env(teams, idx=1, persistent=persistent)
    try:
        finished = 0
        for _ in range(3):
            steps = _play_episode(wrapped, rng)
            assert steps > 0
            battle = wrapped.env.battle1
            assert battle is not None
            assert battle.finished
            assert battle.turn > 0
            assert battle.player_role == "p1"  # role resolved from |player| vs username
            finished += 1
        # Every episode produced a decided battle (poke-env clears _battles per reset, so
        # count completions ourselves rather than reading n_finished_battles).
        assert finished == 3
    finally:
        wrapped.close()


@pytest.mark.integration
@pytest.mark.parametrize("persistent", [True, False], ids=["persistent", "spawn_per_battle"])
def test_bridge_env_resets_are_independent(persistent):
    """Back-to-back episodes must each get a fresh, uniquely-tagged battle (no stale-team reuse
    crash — the same failure class as ``test_repeated_calls_use_unique_tags`` for the runner).
    In persistent mode this also proves one reused child runs many battles with correct tag
    swaps; a single child PID across the episodes confirms no per-episode respawn."""
    teams = _teams()
    rng = np.random.default_rng(7)
    wrapped = _build_bridge_env(teams, idx=2, persistent=persistent)
    try:
        tags = set()
        pids = set()
        for _ in range(4):
            _play_episode(wrapped, rng)
            tags.add(wrapped.env.battle1.battle_tag)
            proc = wrapped.env._bridge_session._proc
            if proc is not None:
                pids.add(proc.pid)
        assert len(tags) == 4  # four distinct battle rooms, no tag collision
        if persistent:
            assert len(pids) == 1  # one reused child across all four episodes
    finally:
        wrapped.close()


@pytest.mark.integration
def test_persistent_child_recycles_after_n_battles():
    """A HEALTHY persistent child is swapped for a fresh one every ``recycle_every`` battles
    (bounds V8-heap growth over a long run). With cap=3 over 10 episodes we expect 3 recycles →
    4 distinct child PIDs, and every episode still completes correctly across the swaps."""
    teams = _teams()
    rng = np.random.default_rng(11)
    wrapped = _build_bridge_env(teams, idx=3, persistent=True, recycle_every=3)
    try:
        pids = []
        for _ in range(10):
            _play_episode(wrapped, rng)
            assert wrapped.env.battle1.finished
            assert wrapped.env.battle1.turn > 0
            pids.append(wrapped.env._bridge_session._proc.pid)
        session = wrapped.env._bridge_session
        # cap=3 → recycle at episodes 4, 7, 10 → 4 distinct children, 3 recycles.
        assert len(set(pids)) == 4, f"expected 4 children from recycling, got {sorted(set(pids))}"
        assert session._n_recycles == 3
    finally:
        wrapped.close()


@pytest.mark.integration
def test_dead_persistent_child_crashes_no_recovery():
    """A child that dies mid-run must CRASH the env (no in-place recovery) — resuming risks a
    corrupted transition, so the launcher restarts from checkpoint instead. We kill the (idle)
    child after a clean episode; the next reset() must raise, not respawn or hang."""
    teams = _teams()
    rng = np.random.default_rng(13)
    wrapped = _build_bridge_env(teams, idx=4, persistent=True, recycle_every=0)
    try:
        _play_episode(wrapped, rng)  # one clean episode; the persistent child is now idle
        session = wrapped.env._bridge_session
        first_pid = session._proc.pid
        os.kill(first_pid, signal.SIGKILL)  # simulate a child crash
        # The next reset() drives _start_persistent_battle, which must detect the dead child and
        # raise (crash-over-corruption) — NOT silently respawn or hang for 180s.
        with pytest.raises(Exception):
            obs, _ = wrapped.reset()
            # If reset somehow returned, stepping would surface the broken transport.
            for _ in range(5):
                mask = np.asarray(obs["action_mask"]).astype(bool)
                legal = np.flatnonzero(mask)
                action = int(rng.choice(legal)) if legal.size else 0
                obs, _r, term, trunc, _i = wrapped.step(action)
                if term or trunc:
                    break
        # It crashed on the same (dead) child — no fresh PID was silently spun up.
        assert session._proc.pid == first_pid
        assert session._n_recycles == 0
    finally:
        wrapped.close()


if __name__ == "__main__":
    # Run directly (script mode) for a quick manual check, like the other bridge tests.
    _t = _teams()
    _rng = np.random.default_rng(0)
    _w = _build_bridge_env(_t, idx=99)
    try:
        for _i in range(3):
            _n = _play_episode(_w, _rng)
            print(f"episode {_i}: {_n} steps, finished={_w.env.battle1.finished}, "
                  f"turns={_w.env.battle1.turn}")
        print("OK — bridge env played 3 episodes")
    finally:
        _w.close()
