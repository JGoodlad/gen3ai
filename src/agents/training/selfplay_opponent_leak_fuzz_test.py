#!/usr/bin/env python3
"""Regression fuzz test: the self-play OPPONENT must not leak per-battle state.

Background — the bug this guards against
----------------------------------------
``Gen3Player`` keeps two per-battle-tag caches, ``_trackers`` (one ``EpisodeTracker``,
which holds the obs + turn-delta history) and ``_stall_loggers`` (one ``StallLogger``).
Their normal eviction point is ``_battle_finished_callback`` — but that only fires for a
*networked* player. A ``--self-play`` opponent (``pool_player``) is built
``start_listening=False`` and used as a pure DECISION FUNCTION over the env's battle, so it
never receives that callback. The training wrapper DOES call ``opponent.reset_battles()``
every episode, and the bridge mints a process-unique tag per battle, so before the fix every
single battle left a permanent ``EpisodeTracker`` (+ history) and ``StallLogger`` behind —
~1 MB/battle × thousands of battles × 24 workers → the env-worker RSS climbing ~30 GB/hr to
an ~82 GB ceiling each 3 h restart cycle.

The fix: ``Gen3Player.reset_battles()`` prunes both caches to the still-live battle tags.

This test runs real self-play battles in-process via the local bridge (no server) — the
trainee plays random legal actions, the opponent is a real (untrained) ``RLPlayer`` driven
through the actual ``MaskableAgentWrapper`` self-play path — and asserts the opponent's
per-tag caches stay BOUNDED across battles. On the pre-fix code ``_trackers`` grows +1 per
battle and this trips within a handful of episodes.

Run:
    PYTHONPATH=$PYTHONPATH:src python3 src/agents/training/selfplay_opponent_leak_fuzz_test.py [n_battles]
"""
import sys
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from poke_env.player import SimpleHeuristicsPlayer
from poke_env import AccountConfiguration, LocalhostServerConfiguration

from agents.inference.player import RLPlayer
from agents.model.features_extractor import Gen3FeaturesExtractor, NET_ARCH
from agents.model.policy import Gen3DualHeadMaskablePolicy
from agents.observation.state_encoder import Gen3ObservationEncoder, load_mappings
from agents.training.gen3_env import Gen3Env
from agents.training.wrappers import MaskableAgentWrapper
from agents.training.reward_manager import Gen3RewardManager
from agents.training.stall import StallConfig
from utils.bridge.bridge_session import attach_bridge_transport
from utils.teambuilder import Gen3Teambuilder
from utils.team_loader import TeamLoader
from utils.logging.levels import LogLevel

BATTLE_FORMAT = "gen3ou"
# At most one in-flight battle on the opponent at a time (the trainee plays it move-by-move),
# so after each episode's reset the caches should be ~0. Allow generous slack for an in-flight
# tag; the pre-fix bug makes these grow to == n_battles, far past any sane bound.
MAX_CACHE = 3


class _DummyMaskedEnv(gym.Env):
    """Minimal env with the real Gen3 spaces — just enough to construct a model."""

    def __init__(self, obs_dim: int):
        self.observation_space = spaces.Dict({
            "observation": spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32),
            "action_mask": spaces.Box(0, 1, shape=(11,), dtype=np.int8),
        })
        self.action_space = spaces.Discrete(11)
        self._dim = obs_dim

    def _obs(self):
        return {"observation": np.zeros(self._dim, np.float32),
                "action_mask": np.ones(11, np.int8)}

    def action_masks(self):
        return np.ones(11, np.int8)

    def reset(self, *, seed=None, options=None):
        return self._obs(), {}

    def step(self, action):
        return self._obs(), 0.0, True, False, {}


def _build_opponent_model(enc, device="cpu"):
    policy_kwargs = {
        "features_extractor_class": Gen3FeaturesExtractor,
        "features_extractor_kwargs": enc.get_features_extractor_kwargs(),
        "net_arch": NET_ARCH,
    }
    dummy = DummyVecEnv([lambda: _DummyMaskedEnv(enc.dimension)])
    return MaskablePPO(Gen3DualHeadMaskablePolicy, dummy, n_steps=16, batch_size=16,
                       n_epochs=1, device=device, policy_kwargs=policy_kwargs)


class _FixedPool:
    """Stub SnapshotPool that always serves the one untrained opponent model — keeps the
    test off disk/version machinery while exercising the real wrapper self-play path."""

    class _Entry:
        path = "fixed"
        step = 0

    def __init__(self, model):
        self._model = model

    def _scan(self):
        pass

    def is_empty(self):
        return False

    def sample(self):
        return self._Entry()

    def load_model(self, entry):
        return self._model


def main(n_battles: int = 30) -> None:
    mappings = load_mappings()
    enc = Gen3ObservationEncoder(mappings)
    opp_model = _build_opponent_model(enc)

    loader = TeamLoader()
    all_teams = loader.get_all_teams()
    sample_teams = loader.get_sample_teams()
    trainee_tb = Gen3Teambuilder(all_teams, bias_teams=sample_teams, bias_prob=0.1)
    opp_tb = Gen3Teambuilder(all_teams)

    env = Gen3Env(
        mappings, battle_format=BATTLE_FORMAT, team=trainee_tb, log_level=LogLevel.QUIET,
        stall_config=StallConfig(output_dir="/tmp/selfplay_leak_stalls"),
        reward_fn=Gen3RewardManager(log_level=LogLevel.QUIET),
        server_configuration=LocalhostServerConfiguration,
        account_configuration1=AccountConfiguration("LeakRL0", "password"),
        start_listening=False,
    )
    attach_bridge_transport(env, battle_format=BATTLE_FORMAT)

    # The self-play opponent: a decision-function RLPlayer (start_listening=False) — the
    # exact configuration whose _battle_finished_callback never fires.
    pool_player = RLPlayer(
        model=None, team=opp_tb, battle_format=BATTLE_FORMAT,
        server_configuration=LocalhostServerConfiguration, mappings=mappings,
        account_configuration=AccountConfiguration("LeakOpp0", "password"),
        start_listening=False, stochastic=True, temperature=1.0,
    )
    fallback = SimpleHeuristicsPlayer(
        battle_format=BATTLE_FORMAT, team=opp_tb,
        server_configuration=LocalhostServerConfiguration,
        account_configuration=AccountConfiguration("LeakFb0", "password"),
        start_listening=False,
    )
    wrapped = MaskableAgentWrapper(
        env, heuristic_opponents=[fallback], pool=_FixedPool(opp_model),
        pool_player=pool_player, self_play_fraction=1.0, rng_seed=0,
    )
    wrapped.action_space = env.action_space
    wrapped.observation_space = env.observation_space
    m = Monitor(wrapped)

    rng = np.random.default_rng(0)
    obs, _ = m.reset()
    eps = 0
    peak = 0
    while eps < n_battles:
        legal = np.nonzero(np.asarray(obs["action_mask"]))[0]
        a = int(rng.choice(legal)) if len(legal) else 0
        obs, _, term, trunc, _ = m.step(a)
        if term or trunc:
            eps += 1
            obs, _ = m.reset()
            nt = len(pool_player._trackers)
            ns = len(pool_player._stall_loggers)
            peak = max(peak, nt, ns)
            if nt > MAX_CACHE or ns > MAX_CACHE:
                raise AssertionError(
                    f"Self-play opponent leaked per-battle state after {eps} battles: "
                    f"_trackers={nt}, _stall_loggers={ns} (bound {MAX_CACHE}). "
                    f"reset_battles() is not pruning Gen3Player's per-tag caches — the "
                    f"env-worker memory leak has regressed (see this file's docstring)."
                )

    print(f"✓ ran {n_battles} self-play battles via bridge; opponent per-battle caches "
          f"stayed bounded (peak _trackers/_stall_loggers={peak} ≤ {MAX_CACHE}). No leak.")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    main(n)
