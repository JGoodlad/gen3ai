"""Stable (cross-run) OPPONENT fuzz test — bridge-backed, no live server.

End-to-end guard for the stable-opponent path (``designs/ai_v5/design_stable_opponents.md``,
Stage 0/1) that mocked unit tests cannot cover. It plays REAL battles in-process via the local
BattleStream bridge (same mechanism as the other ``*_fuzz_test.py`` files) and validates:

  1. **CLI resolution + the arch (observation-family) gate.** A real ``MaskablePPO`` (full Gen3
     architecture) is saved into a fake "foreign run" dir (weights + ``model_config.json``), then
     resolved through ``resolve_stable_opponents`` — the exact ``--stable-opponents`` startup path.
     A matching ``arch_signature`` resolves clean; a DIFFERENT ``arch_signature`` (= different obs
     layout) is rejected with ``ModelVersionError`` — the surfaced-at-startup FATAL.

  2. **A foreign model loads inference-only, bypassing ``check_compatible``.** ``load_foreign_opponent``
     loads against the foreign config (NOT the live trainee), so a ``use_popart`` / weight-field
     difference that ``check_compatible`` would FATAL on is irrelevant to the opponent forward.

  3. **The stable opponent plays legal moves** — an ``RLPlayer`` wrapping the loaded foreign model
     plays real battles vs ``SimpleHeuristicsPlayer`` STOCHASTICALLY (the stable-opponent regime).
     ``_predict_best_action`` raises on an illegal action, so "all N battles finished" proves no
     illegal action was ever selected; the opponent must also vary its choices (stochastic).

Run directly (needs deps/pokemon-showdown bridge + data/ mappings; no server):
    python src/agents/training/stable_opponent_fuzz_test.py [n_battles]
    (in a linked worktree, first: export PYTHONPATH=$PYTHONPATH:src)
"""

from __future__ import annotations

import asyncio
import dataclasses
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
from agents.model.snapshot import save_model_snapshot, load_foreign_opponent
from agents.observation.state_encoder import Gen3ObservationEncoder, load_mappings
from agents.training.fixed_opponent_pool import resolve_stable_opponents
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
        return {"observation": np.zeros(self._dim, dtype=np.float32),
                "action_mask": np.ones(11, dtype=np.int8)}

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
    model = MaskablePPO(Gen3DualHeadMaskablePolicy, dummy, n_steps=16, batch_size=16,
                        n_epochs=1, device=device, policy_kwargs=policy_kwargs)
    return model, version


def _save_foreign_run(run_dir: Path, model, version) -> None:
    """Write a fake finished run: best_model/best_model.zip + a run-level model_config.json."""
    (run_dir / "best_model").mkdir(parents=True, exist_ok=True)
    model.save(str(run_dir / "best_model" / "best_model.zip"))
    save_model_snapshot(str(run_dir), version, git_hash="foreign")


def _check_resolution_and_gate(mappings, tmp: Path) -> tuple[object, ModelVersion]:
    """Part 1+2: resolve a same-arch foreign run (loads), and reject a mismatched-arch one."""
    model, version = _build_model(mappings)

    same = tmp / "foreign_same_arch"
    _save_foreign_run(same, model, version)
    entries = resolve_stable_opponents(f"{same}:champ", version, default_temperature=1.0)
    assert len(entries) == 1 and entries[0].label == "ext_champ", entries
    assert entries[0].arch_signature == version.arch_signature
    opp_model, foreign = load_foreign_opponent(
        entries[0].zip_path, current_version=version, device="cpu",
        config_path=entries[0].config_path)
    assert hasattr(opp_model, "policy"), "loaded foreign opponent has no policy"
    print(f"  ✓ same-arch foreign run resolves + loads ({entries[0].label})")

    # A DIFFERENT arch_signature (= different obs family) must be the surfaced-at-startup FATAL.
    mism = tmp / "foreign_diff_arch"
    _save_foreign_run(mism, model, dataclasses.replace(version, arch_signature="gen3_old_arch_v1"))
    raised = False
    try:
        resolve_stable_opponents(str(mism), version)
    except ModelVersionError:
        raised = True
    assert raised, "arch-mismatched stable opponent loaded silently — the FATAL gate is broken"
    print("  ✓ arch-mismatched foreign run rejected with ModelVersionError (startup FATAL)")

    return opp_model, version


async def _play(opp_model, teams, n_battles: int, ts: int) -> _RecordingRLPlayer:
    """The stable opponent plays STOCHASTICALLY (its regime) vs Heuristic over the bridge."""
    opp = _RecordingRLPlayer(
        model=opp_model, team=Gen3Teambuilder(teams), battle_format=BATTLE_FORMAT,
        server_configuration=LocalhostServerConfiguration, start_listening=False,
        account_configuration=AccountConfiguration(f"SOopp{ts}", "password"),
        max_concurrent_battles=5, stochastic=True, temperature=1.0,
    )
    heur = SimpleHeuristicsPlayer(
        battle_format=BATTLE_FORMAT, team=Gen3Teambuilder(teams),
        server_configuration=LocalhostServerConfiguration, start_listening=False,
        account_configuration=AccountConfiguration(f"SOheu{ts}", "password"),
        max_concurrent_battles=5,
    )
    await run_local_battles(opp, heur, n_battles, battle_format=BATTLE_FORMAT)
    return opp


async def main(n_battles: int = 12) -> None:
    print(f"Stable Opponent Fuzz — gen3ou — {n_battles} battles")
    mappings = load_mappings()
    teams = TeamLoader().get_sample_teams()
    ts = int(time.time()) % 100000

    with tempfile.TemporaryDirectory() as td:
        print("\n[1] Resolve foreign run + arch gate (matching + mismatched)")
        opp_model, _version = _check_resolution_and_gate(mappings, Path(td))

        print(f"\n[2] Stable opponent plays {n_battles} battles vs Heuristic (stochastic)")
        opp = await _play(opp_model, teams, n_battles, ts)

    fails: list[str] = []
    if opp.n_finished_battles != n_battles:
        fails.append(f"only {opp.n_finished_battles}/{n_battles} battles finished "
                     "(an illegal action or crash aborted a battle)")
    if not opp.chosen:
        fails.append("no actions recorded — opponent never chose a move")
    bad = [i for i in opp.chosen if not (0 <= i < 11)]
    if bad:
        fails.append(f"out-of-range action indices {set(bad)}")
    if len(set(opp.chosen)) < 2:
        fails.append(f"stochastic opponent showed no action variety ({set(opp.chosen)})")
    print(f"  {opp.n_finished_battles}/{n_battles} battles, {len(opp.chosen)} decisions, "
          f"{len(set(opp.chosen))} distinct actions")

    print(f"\n{'=' * 60}")
    if fails:
        print("❌ STABLE OPPONENT FUZZ FAILED")
        for f in fails:
            print(f"  - {f}")
        sys.exit(1)
    print("✅ STABLE OPPONENT FUZZ PASSED")
    sys.exit(0)


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    asyncio.run(main(n))
