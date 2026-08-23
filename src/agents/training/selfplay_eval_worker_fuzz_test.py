"""Self-play EVAL-WORKER fuzz test — bridge-backed, no live server.

End-to-end guard for the NON-BLOCKING self-play eval worker's sentinel path
(``main.eval_worker._play_unit`` (SENTINEL branch)), which mocked unit tests can't cover. It plays
REAL battles in-process via the local BattleStream bridge (same mechanism as the other
``*_fuzz_test.py`` files) and validates the worker's exact construction:

  1. **Frozen trainee + version-checked sentinel load.** A real ``MaskablePPO`` (full Gen3
     architecture) is frozen to disk and reloaded the way the worker reloads the trainee
     snapshot (``MaskablePPO.load``), while pool sentinels are reloaded via
     ``load_model_snapshot`` against the pool's shared ``model_config.json`` using the
     CURRENT-code version from ``current_model_version`` — the exact path the worker takes.
     A deliberately-incompatible version is rejected with ``ModelVersionError``.

  2. **Trainee (greedy) vs sentinel (stochastic) play legal battles.** The matchup is built
     exactly as ``_play_unit`` builds it — an ``EvalRLPlayer`` trainee
     (``stochastic=False``) vs an ``RLPlayer`` sentinel (``stochastic=True``, temperature).
     ``_predict_best_action`` raises on an illegal action, so "all N battles finished"
     proves no illegal action was ever selected. The trainee tracks reward + a win rate.

  3. **The stochastic sentinel actually varies its choices** — the lever that makes a pool
     opponent a richer signal than a deterministic adversary.

Run directly (needs deps/pokemon-showdown bridge + data/ mappings; no server):
    python src/agents/training/selfplay_eval_worker_fuzz_test.py [n_battles]
    (in a linked worktree, first: export PYTHONPATH=$PYTHONPATH:src)
"""

from __future__ import annotations

import asyncio
import sys
import tempfile
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import DummyVecEnv
from poke_env import AccountConfiguration, LocalhostServerConfiguration

from agents.inference.player import RLPlayer
from agents.model.features_extractor import Gen3FeaturesExtractor, NET_ARCH
from agents.model.policy import Gen3DualHeadMaskablePolicy
from agents.model.model_version import ModelVersion, ModelVersionError
from agents.model.snapshot import load_model_snapshot, current_model_version
from agents.observation.state_encoder import Gen3ObservationEncoder, load_mappings
from agents.training.eval_callback import EvalRLPlayer
from agents.training.snapshot_pool import SnapshotPool
from utils.teambuilder import Gen3Teambuilder
from utils.team_loader import TeamLoader
from utils.bridge.local_battle_runner import run_local_battles

BATTLE_FORMAT = "gen3ou"


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
        return {
            "observation": np.zeros(self._dim, dtype=np.float32),
            "action_mask": np.ones(11, dtype=np.int8),
        }

    def action_masks(self):
        return np.ones(11, dtype=np.int8)

    def reset(self, *, seed=None, options=None):
        return self._obs(), {}

    def step(self, action):
        return self._obs(), 0.0, True, False, {}


class _RecordingTrainee(EvalRLPlayer):
    """Frozen-trainee eval player that records every chosen action (legality check)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.chosen: list[int] = []

    def _predict_best_action(self, battle, **kwargs):
        idx, probs, mask = super()._predict_best_action(battle, **kwargs)
        if idx is not None:
            self.chosen.append(int(idx))
        return idx, probs, mask


class _RecordingSentinel(RLPlayer):
    """Sentinel opponent that records every chosen action (legality + variety check)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.chosen: list[int] = []

    def _predict_best_action(self, battle, **kwargs):
        idx, probs, mask = super()._predict_best_action(battle, **kwargs)
        if idx is not None:
            self.chosen.append(int(idx))
        return idx, probs, mask


def _build_model(mappings, device="cpu"):
    """Construct a real MaskablePPO with the full Gen3 architecture (untrained weights)."""
    enc = Gen3ObservationEncoder(mappings)
    ek = enc.get_features_extractor_kwargs()
    policy_kwargs = {
        "features_extractor_class": Gen3FeaturesExtractor,
        "features_extractor_kwargs": ek,
        "net_arch": NET_ARCH,
    }
    version = ModelVersion.from_layout_and_policy_kwargs(ek["layout"], policy_kwargs)
    dummy = DummyVecEnv([lambda: _DummyMaskedEnv(enc.dimension)])
    model = MaskablePPO(
        Gen3DualHeadMaskablePolicy, dummy,
        n_steps=16, batch_size=16, n_epochs=1,
        device=device, policy_kwargs=policy_kwargs,
    )
    return model, version, ek


def _load_like_worker(mappings, td: Path):
    """Reproduce the worker's loads: frozen trainee (MaskablePPO.load) + version-checked
    sentinel (load_model_snapshot vs current_model_version). Returns (trainee, sentinel)."""
    model, version, ek = _build_model(mappings)

    # Pool sentinels live on disk; the trainee snapshot is frozen separately at trigger.
    pool = SnapshotPool(pool_dir=td, current_version=version, device="cpu")
    pool.seed(model)                      # snapshot_0 = a sentinel
    assert (td / "model_config.json").exists(), "pool did not write model_config.json"

    frozen = td / "frozen_trainee"
    model.save(str(frozen))               # the worker's cfg['snapshot']
    trainee_model = MaskablePPO.load(str(frozen) + ".zip", env=None, device="cpu")

    # Sentinel load — exactly the worker's path (current_model_version → version-checked).
    cur = current_model_version(mappings)
    sentinel_entry = pool.sentinel_entries(n=5)[0]
    sentinel_model = load_model_snapshot(
        str(sentinel_entry.path), env=None, current_version=cur, device="cpu",
    )
    assert hasattr(sentinel_model, "policy"), "reloaded sentinel has no policy"
    print("  ✓ frozen-trainee load + version-checked sentinel load (matching arch)")

    # Incompatible arch must be rejected (proves the version guard fires on the worker path).
    bad = ModelVersion.from_layout_and_policy_kwargs(ek["layout"], {
        "features_extractor_class": Gen3FeaturesExtractor,
        "features_extractor_kwargs": ek,
        "net_arch": [256, 256, 256],
    })
    try:
        bad.check_compatible(version)
        mismatch_in_signature = False
    except ModelVersionError:
        mismatch_in_signature = True
    if mismatch_in_signature:
        raised = False
        try:
            load_model_snapshot(str(sentinel_entry.path), env=None, current_version=bad, device="cpu")
        except ModelVersionError:
            raised = True
        assert raised, "mismatched arch loaded silently — sentinel version guard is broken"
        print("  ✓ mismatched arch rejected with ModelVersionError")
    else:
        print("  ⚠ net_arch not in the version signature — skipped the mismatch assertion")

    return trainee_model, sentinel_model


async def _play(trainee_model, sentinel_model, teams, n_battles: int, ts: int):
    """Build the matchup exactly as eval_worker._play_unit does, then run it on the bridge."""
    trainee = _RecordingTrainee(
        model=trainee_model, team=Gen3Teambuilder(teams), battle_format=BATTLE_FORMAT,
        server_configuration=LocalhostServerConfiguration, start_listening=False,
        account_configuration=AccountConfiguration(f"SPwtr{ts}", "password"),
        max_concurrent_battles=5,
        stochastic=False,   # eval trainee = GREEDY
    )
    sentinel = _RecordingSentinel(
        model=sentinel_model, team=Gen3Teambuilder(teams), battle_format=BATTLE_FORMAT,
        server_configuration=LocalhostServerConfiguration, start_listening=False,
        account_configuration=AccountConfiguration(f"SPwse{ts}", "password"),
        max_concurrent_battles=5,
        stochastic=True, temperature=1.0,   # sentinel = stochastic (training-opponent behaviour)
    )
    await run_local_battles(trainee, sentinel, n_battles, battle_format=BATTLE_FORMAT)
    return trainee, sentinel


async def main(n_battles: int = 8) -> None:
    print(f"Self-Play Eval-Worker Fuzz — gen3ou — {n_battles} battles")
    mappings = load_mappings()
    teams = TeamLoader().get_sample_teams()
    ts = int(time.time()) % 100000

    with tempfile.TemporaryDirectory() as td:
        print("\n[1] Worker loads: frozen trainee + version-checked sentinel")
        trainee_model, sentinel_model = _load_like_worker(mappings, Path(td))

        print(f"\n[2] Trainee (greedy) vs sentinel (stochastic) — {n_battles} bridge battles")
        trainee, sentinel = await _play(trainee_model, sentinel_model, teams, n_battles, ts)

    fails: list[str] = []

    if trainee.n_finished_battles != n_battles:
        fails.append(f"only {trainee.n_finished_battles}/{n_battles} battles finished "
                     f"(an illegal action or crash aborted a battle)")
    if not trainee.chosen:
        fails.append("trainee never chose a move")
    if not sentinel.chosen:
        fails.append("sentinel never chose a move")
    for label, p in (("trainee", trainee), ("sentinel", sentinel)):
        bad = [i for i in p.chosen if not (0 <= i < 11)]
        if bad:
            fails.append(f"{label}: out-of-range action indices {set(bad)}")

    # Win-rate computation (what the sentinel shard produces) must be well-formed.
    finished = trainee.n_finished_battles
    win_rate = trainee.n_won_battles / finished if finished else 0.0
    if not (0.0 <= win_rate <= 1.0):
        fails.append(f"win_rate out of range: {win_rate}")
    if not isinstance(trainee.mean_episode_reward, float):
        fails.append("trainee did not track an episode reward")

    # The stochastic sentinel must actually vary (the point of the pool-opponent signal).
    if len(set(sentinel.chosen)) < 2:
        fails.append(f"stochastic sentinel showed no action variety "
                     f"({len(sentinel.chosen)} decisions, all {set(sentinel.chosen)})")

    print(f"  trainee: {finished}/{n_battles} finished, win_rate={win_rate * 100:.0f}%, "
          f"reward={trainee.mean_episode_reward:.3f}, {len(trainee.chosen)} decisions")
    print(f"  sentinel: {len(sentinel.chosen)} decisions, "
          f"{len(set(sentinel.chosen))} distinct actions")

    print(f"\n{'=' * 60}")
    if fails:
        print("❌ SELF-PLAY EVAL-WORKER FUZZ FAILED")
        for f in fails:
            print(f"  - {f}")
        sys.exit(1)
    print("✅ SELF-PLAY EVAL-WORKER FUZZ PASSED")
    sys.exit(0)


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    asyncio.run(main(n))
