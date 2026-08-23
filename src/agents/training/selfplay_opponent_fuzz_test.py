"""Self-play OPPONENT fuzz test — bridge-backed, no live server.

This is the end-to-end guard for the Phase-1 self-play opponent path, which unit tests
(mocked pool / mocked model) cannot cover. It plays REAL battles in-process via the local
BattleStream bridge (same mechanism as the other ``*_fuzz_test.py`` files) and validates:

  1. **Pool round-trip + version check.** A real ``MaskablePPO`` (full Gen3 architecture) is
     saved into a ``SnapshotPool`` and reloaded via ``load_model_snapshot`` — the exact path
     the training-env opponents now use. The pool writes a shared ``model_config.json``, so
     the architecture compatibility check is REAL (matching version → loads clean). An
     intentionally-incompatible version is rejected with ``ModelVersionError``.

  2. **The opponent plays legal moves.** An ``RLPlayer`` wrapping the reloaded snapshot plays
     real battles vs ``SimpleHeuristicsPlayer`` in BOTH modes:
       - deterministic (argmax) — the eval-sentinel behaviour, and
       - stochastic (temperature-sampled) — the training-opponent behaviour.
     ``RLPlayer._predict_best_action`` raises on an illegal action, so "all N battles finished"
     proves no illegal action was ever selected.

  3. **The stochastic opponent actually varies its choices** — the diversity lever that makes
     self-play a richer, less-exploitable signal than a deterministic adversary.

Run directly (needs deps/pokemon-showdown bridge + data/ mappings; no server):
    python src/agents/training/selfplay_opponent_fuzz_test.py [n_battles]
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
from poke_env.player import SimpleHeuristicsPlayer
from poke_env import AccountConfiguration, LocalhostServerConfiguration

from agents.inference.player import RLPlayer
from agents.model.features_extractor import Gen3FeaturesExtractor, NET_ARCH
from agents.model.policy import Gen3DualHeadMaskablePolicy
from agents.model.model_version import ModelVersion, ModelVersionError
from agents.model.snapshot import load_model_snapshot
from agents.observation.state_encoder import Gen3ObservationEncoder, load_mappings
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


class _RecordingRLPlayer(RLPlayer):
    """RLPlayer that records every chosen action index for legality/variety checks."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.chosen: list[int] = []

    def _predict_best_action(self, battle, **kwargs):
        idx, probs, mask = super()._predict_best_action(battle, **kwargs)
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


def _check_pool_and_version(mappings, tmp_dir: Path) -> tuple[object, ModelVersion]:
    """Part 1: save → reload (version-checked) round-trip, incl. the mismatch path."""
    model, version, ek = _build_model(mappings)
    pool = SnapshotPool(pool_dir=tmp_dir, current_version=version, device="cpu")
    pool.seed(model)
    pool.add(model, step=1000)

    config = tmp_dir / "model_config.json"
    assert config.exists(), f"pool did not write {config} — opponent version check would be skipped"
    assert len(pool) == 2, f"expected seed + 1 promotion, got {len(pool)}"

    seed_entry = sorted(pool._entries, key=lambda e: e.step)[0]

    # Matching version → loads clean (this is the training-env opponent path).
    opp_model = load_model_snapshot(
        str(seed_entry.path), env=None, current_version=version, device="cpu"
    )
    assert hasattr(opp_model, "policy"), "reloaded opponent has no policy"
    print("  ✓ pool round-trip + version-checked load (matching arch)")

    # Incompatible version → must be rejected (proves the check actually fires on the
    # opponent path). Build a different arch signature via a different net_arch.
    bad_pk = {
        "features_extractor_class": Gen3FeaturesExtractor,
        "features_extractor_kwargs": ek,
        "net_arch": [256, 256, 256],
    }
    bad_version = ModelVersion.from_layout_and_policy_kwargs(ek["layout"], bad_pk)
    try:
        bad_version.check_compatible(version)
        incompatible = False
    except ModelVersionError:
        incompatible = True

    if incompatible:
        raised = False
        try:
            load_model_snapshot(str(seed_entry.path), env=None, current_version=bad_version, device="cpu")
        except ModelVersionError:
            raised = True
        assert raised, "mismatched arch loaded silently — opponent version guard is broken"
        print("  ✓ mismatched arch rejected with ModelVersionError")
    else:
        print("  ⚠ net_arch not part of the version signature — skipped the mismatch assertion")

    return opp_model, version


async def _play(opp_model, teams, n_battles: int, ts: int) -> dict:
    det = _RecordingRLPlayer(
        model=opp_model, team=Gen3Teambuilder(teams), battle_format=BATTLE_FORMAT,
        server_configuration=LocalhostServerConfiguration, start_listening=False,
        account_configuration=AccountConfiguration(f"SPdet{ts}", "password"),
        max_concurrent_battles=5, stochastic=False,
    )
    sto = _RecordingRLPlayer(
        model=opp_model, team=Gen3Teambuilder(teams), battle_format=BATTLE_FORMAT,
        server_configuration=LocalhostServerConfiguration, start_listening=False,
        account_configuration=AccountConfiguration(f"SPsto{ts}", "password"),
        max_concurrent_battles=5, stochastic=True, temperature=1.0,
    )
    heur_d = SimpleHeuristicsPlayer(
        battle_format=BATTLE_FORMAT, team=Gen3Teambuilder(teams),
        server_configuration=LocalhostServerConfiguration, start_listening=False,
        account_configuration=AccountConfiguration(f"SPhd{ts}", "password"),
        max_concurrent_battles=5,
    )
    heur_s = SimpleHeuristicsPlayer(
        battle_format=BATTLE_FORMAT, team=Gen3Teambuilder(teams),
        server_configuration=LocalhostServerConfiguration, start_listening=False,
        account_configuration=AccountConfiguration(f"SPhs{ts}", "password"),
        max_concurrent_battles=5,
    )

    await run_local_battles(det, heur_d, n_battles, battle_format=BATTLE_FORMAT)
    await run_local_battles(sto, heur_s, n_battles, battle_format=BATTLE_FORMAT)
    return {"det": det, "sto": sto}


async def main(n_battles: int = 12) -> None:
    print(f"Self-Play Opponent Fuzz — gen3ou — {n_battles} battles per mode")
    mappings = load_mappings()
    teams = TeamLoader().get_sample_teams()
    ts = int(time.time()) % 100000

    with tempfile.TemporaryDirectory() as td:
        print("\n[1] Pool save → version-checked reload")
        opp_model, _version = _check_pool_and_version(mappings, Path(td))

        print(f"\n[2] Opponent plays {n_battles} battles vs Heuristic (deterministic + stochastic)")
        players = await _play(opp_model, teams, n_battles, ts)

    det, sto = players["det"], players["sto"]
    fails: list[str] = []

    for label, p in (("deterministic", det), ("stochastic", sto)):
        if p.n_finished_battles != n_battles:
            fails.append(
                f"{label}: only {p.n_finished_battles}/{n_battles} battles finished "
                f"(an illegal action or crash aborted a battle)"
            )
        if not p.chosen:
            fails.append(f"{label}: no actions recorded — opponent never chose a move")
        bad = [i for i in p.chosen if not (0 <= i < 11)]
        if bad:
            fails.append(f"{label}: out-of-range action indices {set(bad)}")
        print(f"  {label}: {p.n_finished_battles}/{n_battles} battles, "
              f"{len(p.chosen)} decisions, {len(set(p.chosen))} distinct actions")

    # The stochastic opponent must actually vary (the whole point of the change).
    if len(set(sto.chosen)) < 2:
        fails.append(
            f"stochastic opponent showed no action variety "
            f"({len(sto.chosen)} decisions, all action {set(sto.chosen)}) — "
            f"sampling is not taking effect"
        )

    print(f"\n{'=' * 60}")
    if fails:
        print("❌ SELF-PLAY OPPONENT FUZZ FAILED")
        for f in fails:
            print(f"  - {f}")
        sys.exit(1)
    print("✅ SELF-PLAY OPPONENT FUZZ PASSED")
    sys.exit(0)


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    asyncio.run(main(n))
