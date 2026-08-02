"""Regression guard for ledger M1: the identity-at-init contract must hold on a REAL policy.

THE BUG THIS PINS. SB3's `ActorCriticPolicy._build()` runs
``self.features_extractor.apply(partial(self.init_weights, gain=sqrt(2)))``
(stable_baselines3/common/policies.py:617-631), and `init_weights` orthogonally re-initialises EVERY
`nn.Linear` it finds. `ortho_init` defaults True and nothing in this repo overrides it, so until
2026-08-01 every deliberate zero-init INSIDE the extractor was destroyed the moment the policy was
built — in every real training run. Measured before the fix: `refine_proj` 0.470, `outgoing_proj`
0.355, `status_in_proj` 0.416, `status_out_proj` 0.369, `film_pi` 0.211, `film_vf` 0.185, plus the
belief heads whose zero-init is what makes the cold-start posterior EQUAL the Smogon prior.

WHY IT SURVIVED SO LONG — and the rule this file enforces. Every existing test builds the module, or
a bare `Gen3FeaturesExtractor`, DIRECTLY. The zero-init survives there; only SB3-wrapped construction
destroys it. **An invariant asserted only on a construction path the production code does not use is
not an invariant.** So this file deliberately pays the cost of building a real `MaskablePPO`.
"""
import inspect

import gymnasium as gym
import numpy as np
import pytest
import torch
from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import DummyVecEnv

from agents.action.constants import ACTION_SPACE_SIZE
from agents.model.features_extractor import Gen3FeaturesExtractor
from agents.model.policy import Gen3DualHeadMaskablePolicy
from agents.observation.state_encoder import Gen3ObservationEncoder, load_mappings

# The toggles that OWN the documented zero-init modules — refine/threat/status (v31/v33/v36/v37),
# FiLM (v44), and the belief heads whose cold-start is supposed to equal the prior.
_ZERO_INIT_TOGGLES = dict(
    attend_unrevealed_opponents=True, opp_belief_slots=True,
    move_belief_mode="revealed", move_prior_fusion=True, move_latent=True,
    damage_op=True, damage_outgoing=True, damage_refine_rounds=2,
    threat_refine_outgoing=True, threat_status_refine=True,
    spread_belief=True, hp_type_belief_mode="learned",
    zarch_film="heads", zarch_dim=32,
)


class _Env(gym.Env):
    def __init__(self, dim):
        self.observation_space = gym.spaces.Dict(
            {"observation": gym.spaces.Box(0.0, 1.0, (dim,), np.float32)})
        self.action_space = gym.spaces.Discrete(ACTION_SPACE_SIZE)
        self._dim = dim

    def reset(self, **kw):
        return {"observation": np.zeros(self._dim, np.float32)}, {}

    def step(self, a):
        return {"observation": np.zeros(self._dim, np.float32)}, 0.0, True, False, {}

    def action_masks(self):
        return np.ones(ACTION_SPACE_SIZE, bool)


def _build_real_policy(**overrides):
    """Construct through the SAME path training uses — MaskablePPO -> ActorCriticPolicy._build()."""
    enc = Gen3ObservationEncoder(load_mappings())
    ek = enc.get_features_extractor_kwargs()
    sig = set(inspect.signature(Gen3FeaturesExtractor.__init__).parameters)
    kw = {**ek, **{k: v for k, v in {**_ZERO_INIT_TOGGLES, **overrides}.items() if k in sig}}
    torch.manual_seed(0)
    model = MaskablePPO(
        Gen3DualHeadMaskablePolicy, DummyVecEnv([lambda: _Env(enc.dimension)]),
        n_steps=16, batch_size=16, n_epochs=1, device="cpu",
        policy_kwargs={"features_extractor_class": Gen3FeaturesExtractor,
                       "features_extractor_kwargs": kw,
                       "net_arch": dict(pi=[64], vf=[64])},
    )
    return model, enc


def test_sb3_ortho_init_still_clobbers_the_extractor():
    """Pin the UPSTREAM behaviour this guard defends against, so an SB3 upgrade that silently changes
    it is noticed rather than leaving a now-pointless workaround in place."""
    assert MaskablePPO is not None
    model, _ = _build_real_policy()
    assert model.policy.ortho_init is True, (
        "SB3 no longer defaults ortho_init=True — re-evaluate whether restore_identity_init is still "
        "needed, and whether anything now depends on it NOT running."
    )


def test_zero_init_modules_are_still_zero_after_policy_build():
    """THE regression: every Linear the extractor deliberately zero-initialised must still be zero
    once SB3 has finished building the policy."""
    model, _ = _build_real_policy()
    fe = model.policy.features_extractor
    tracked = fe._identity_init_zeroed
    assert len(tracked) >= 6, (
        f"only {len(tracked)} identity-init Linears tracked — expected at least the six documented "
        f"ones (refine_proj, outgoing_proj, status_in/out_proj, film_pi, film_vf). Did the snapshot "
        f"in __init__ move before the modules are built?"
    )
    mods = dict(fe.named_modules())
    nonzero = {n: float(mods[n].weight.abs().max()) for n in tracked
               if float(mods[n].weight.abs().max()) != 0.0}
    assert not nonzero, (
        f"identity-at-init VIOLATED after policy build for {len(nonzero)} module(s): {nonzero}. "
        f"SB3's ortho init clobbered them and restore_identity_init did not undo it."
    )


@pytest.mark.parametrize("name", [
    "refine_proj", "outgoing_proj", "status_in_proj", "status_out_proj", "film_pi", "film_vf",
])
def test_each_documented_zero_init_module_by_name(name):
    """Name the six modules whose docs explicitly promise identity-at-init, so a rename or a dropped
    zero-init is caught by NAME rather than only by the generic sweep above."""
    model, _ = _build_real_policy()
    fe = model.policy.features_extractor
    mod = dict(fe.named_modules()).get(name)
    assert mod is not None, f"{name} absent — the toggle set in this test no longer builds it"
    assert float(mod.weight.abs().max()) == 0.0, f"{name} is not zero after policy build"
    assert float(mod.bias.abs().max()) == 0.0, f"{name} bias is not zero after policy build"


def test_belief_heads_cold_start_equals_prior():
    """The belief heads are zero-init for a SEMANTIC reason, not just numerical hygiene: it is what
    makes the cold-start posterior EQUAL the Smogon prior (v20 prior-fusion, v38, v40). If SB3
    clobbers them the belief starts at prior ⊕ noise, which is a different experiment."""
    model, _ = _build_real_policy()
    fe = model.policy.features_extractor
    mods = dict(fe.named_modules())
    checked = 0
    for name in ("move_belief.move_head", "spread_belief.stat_head", "spread_belief.nature_head",
                 "spread_belief.ev_head", "hp_type_belief.type_head"):
        mod = mods.get(name)
        if mod is None:          # flag-gated: not every head exists under every toggle set
            continue
        checked += 1
        assert float(mod.weight.abs().max()) == 0.0, (
            f"{name} is not zero after policy build — its cold-start posterior no longer equals the "
            f"prior, silently changing every belief experiment's starting point."
        )
    assert checked >= 1, "no belief head was present to check — the toggle set regressed"


def test_restore_is_idempotent_and_reports_its_work():
    """Calling it twice must be a no-op, and it must report a plausible count (a silent 0 would mean
    the snapshot was empty and the guard is doing nothing)."""
    model, _ = _build_real_policy()
    fe = model.policy.features_extractor
    n1 = fe.restore_identity_init()
    n2 = fe.restore_identity_init()
    assert n1 == n2 >= 6, f"restore reported {n1}/{n2} modules — expected >= 6 and stable"
