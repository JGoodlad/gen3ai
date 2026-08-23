"""TD-residual parity fuzz (#4) — real battles via the local BattleStream bridge (no server).

Proves the LIVE per-decision critic surprise the eval cycle folds into ``eval/td_resid_tail_*``
equals what the prober would recompute OFFLINE from the same saved trace, using the prober's
single-source-of-truth formula δ(t) = r(t) + γ·V(s_{t+1}) − V(s_t)
(``main/prober/session/core.py::ProbeSession._td``).

It guards the three things a unit test with hand-fed numbers can't: that on a *real* battle (a)
the scalar reward ``BattleRecorder`` closes each transition with is the SAME number stored in the
invocation outcome the prober reads (``reward.total``), (b) the V(s) / V(s') alignment (off-by-one)
is right, and (c) the last decision correctly yields no δ. Follows the project's fuzz pattern: an
``EvalRLPlayer`` plays N random battles with forensics armed; in ``_battle_finished_callback`` we
peek the recorder, recompute δ offline from its stored arrays, and assert it matches the recorder's
live ``td_residuals()`` — any divergence raises immediately.

    python src/agents/training/td_residual_parity_fuzz_test.py [n_battles]
"""

import asyncio
import sys
import tempfile
import time

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import DummyVecEnv
from poke_env.player import SimpleHeuristicsPlayer
from poke_env import AccountConfiguration

from agents.model.features_extractor import Gen3FeaturesExtractor, NET_ARCH
from agents.model.policy import Gen3DualHeadMaskablePolicy
from agents.model.model_version import ModelVersion
from agents.observation.state_encoder import Gen3ObservationEncoder, load_mappings
from agents.training.eval_callback import EvalRLPlayer, BATTLE_FORMAT
from utils.teambuilder import Gen3Teambuilder
from utils.team_loader import TeamLoader
from utils.bridge.local_battle_runner import run_local_battles

GAMMA = 0.9999     # production discount — parity is γ-independent, but exercise the real value
TOL = 1e-4


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


def _build_model(mappings, device="cpu"):
    enc = Gen3ObservationEncoder(mappings)
    ek = enc.get_features_extractor_kwargs()
    policy_kwargs = {
        "features_extractor_class": Gen3FeaturesExtractor,
        "features_extractor_kwargs": ek,
        "net_arch": NET_ARCH,
    }
    ModelVersion.from_layout_and_policy_kwargs(ek["layout"], policy_kwargs)  # validate arch
    dummy = DummyVecEnv([lambda: _DummyMaskedEnv(enc.dimension)])
    return MaskablePPO(
        Gen3DualHeadMaskablePolicy, dummy, n_steps=16, batch_size=16, n_epochs=1,
        device=device, policy_kwargs=policy_kwargs,
    )


def _offline_td(rec) -> list[float]:
    """Recompute δ from the recorder's STORED arrays exactly as the prober would off-disk:
    V(s) from the per-decision states, reward(t) from invocation[t].outcome.reward(.total)."""
    vals = [s.get("value") if s else None for s in rec._states]
    out = []
    for t, inv in enumerate(rec._invocations):           # committed decisions (last is pending)
        v, v_next = vals[t], vals[t + 1] if t + 1 < len(vals) else None
        if v is None or v_next is None:
            continue
        r = inv["outcome"]["reward"]
        r = r["total"] if isinstance(r, dict) else r
        out.append(float(r) + rec._gamma * float(v_next) - float(v))
    return out


class _ParityTrainee(EvalRLPlayer):
    """EvalRLPlayer that, at each battle finish, asserts the recorder's live residuals match the
    offline recompute before the base class harvests + drops the recorder."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.checked = 0
        self.total_residuals = 0
        self.failures: list[str] = []

    def _battle_finished_callback(self, battle):
        rec = self._recorders.get(battle.battle_tag)
        if rec is not None and rec.td_residuals():
            live = list(rec.td_residuals())
            offline = _offline_td(rec)
            self.checked += 1
            self.total_residuals += len(live)
            if len(live) != len(offline):
                self.failures.append(
                    f"{battle.battle_tag}: live {len(live)} residuals != offline {len(offline)}")
            else:
                for i, (a, b) in enumerate(zip(live, offline)):
                    if abs(a - b) > TOL:
                        self.failures.append(
                            f"{battle.battle_tag} δ[{i}]: live {a:.6f} != offline {b:.6f}")
                        break
        super()._battle_finished_callback(battle)


async def _run(n_battles: int) -> _ParityTrainee:
    mappings = load_mappings()
    teams = TeamLoader().get_sample_teams() or TeamLoader().get_all_teams()
    model = _build_model(mappings)
    ts = int(time.time()) % 100000

    trainee = _ParityTrainee(
        model=model, team=Gen3Teambuilder(teams), battle_format=BATTLE_FORMAT,
        server_configuration=None, mappings=mappings,
        account_configuration=AccountConfiguration(f"TDpar{ts}", "password"),
        max_concurrent_battles=1, stochastic=False, start_listening=False,
        gamma=GAMMA, win_quota=10_000, loss_quota=10_000,   # capture every battle → max signal
    )
    opp = SimpleHeuristicsPlayer(
        battle_format=BATTLE_FORMAT, team=Gen3Teambuilder(teams),
        account_configuration=AccountConfiguration(f"TDopp{ts}", "password"),
        max_concurrent_battles=1, start_listening=False,
    )
    # A real dir ARMS capture (need_aux=True → V(s) recorded); high quotas keep every battle.
    with tempfile.TemporaryDirectory() as tmp:
        trainee.begin_forensic_cycle(forensic_dir=tmp, step=0)
        await run_local_battles(trainee, opp, n_battles, battle_format=BATTLE_FORMAT)
    return trainee


def main(n_battles: int = 12) -> None:
    print(f"TD-residual parity fuzz — gen3ou — {n_battles} bridge battles (γ={GAMMA})")
    trainee = asyncio.run(_run(n_battles))
    print(f"  battles with residuals: {trainee.checked}/{n_battles}  "
          f"(total δ checked: {trainee.total_residuals})")
    if trainee.failures:
        for f in trainee.failures[:10]:
            print(f"  ✗ {f}")
        raise SystemExit(f"FAIL — {len(trainee.failures)} parity mismatch(es): live δ diverged "
                         f"from the prober's offline recompute")
    if trainee.checked == 0 or trainee.total_residuals == 0:
        raise SystemExit("FAIL — no residuals were produced (capture never armed?) — nothing tested")
    print(f"PASS — every live δ matched the prober's offline _td within {TOL} over "
          f"{trainee.total_residuals} decisions in {trainee.checked} battles.")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    main(n)
