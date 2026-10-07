"""gen3_value_threat_inject_off_v1 (v142, architecture audit F10) — `--value-threat-inject off` as a ONE-lever arm.

F10: the critic reads the damage operator's incoming rows three ways — through the trunk (`prefuse_proj`),
through `value_entity_pool`'s op-row source, and through `value_threat_inject`'s token content on the value
pool's copy. The approved screen deletes the third route and reads critic discrimination. For that read to be
about the route and nothing else, the OFF arm must differ from production (ON) by the route ALONE.

What must hold, each failing on revert:

  * **`off` builds no route, on a policy built the way training builds it** (`MaskablePPO` → `_build`): the
    projection is RETIRED — no state_dict key, no optimizer slot, no parameter — and a bare OFF extractor's
    forward never reads it (it is constructed NOT LIVE).
  * **every other parameter's initial bytes equal production's** — the one-lever property. Before v142 an OFF
    build skipped the Linear and moved ~185 later tensors' init draws (the revert this pins).
  * **identity at init**: production's projection is zero-init, so the two arms' step-0 forwards are bitwise
    equal in BOTH halves — the arm starts where production starts.
  * **it composes** with the four model flags already on main (`--token-encoding static`, `--policy-readout
    trunk`, `--move-resolution on`, `--belief-tokens fixed_mass`), each judged by the same three properties.
  * **production is untouched**: `on` keeps a LIVE projection and `retire_value_threat_inject` is a no-op.
"""
from __future__ import annotations

import inspect
import json
from typing import Any, Dict, Tuple

import numpy as np
import pytest
import torch

from agents.model.arch_constants import D_MODEL
from agents.model.features_extractor import Gen3FeaturesExtractor
from agents.model.value_threat_inject import value_threat_inject_dim
from agents.observation.state_encoder import Gen3ObservationEncoder, load_mappings
from utils.paths import repo_path, src_path

_ROUTE = "cls_pool.value_threat_proj."


def _prod_kwargs(**over: Any) -> Dict[str, Any]:
    from agents.model.damage_tables import sanitize_historical_move_floor
    cfg = json.load(open(repo_path("designs", "production_config.json")))
    sig = set(inspect.signature(Gen3FeaturesExtractor.__init__).parameters)
    kw = {k: v for k, v in cfg.items() if k in sig}
    sanitize_historical_move_floor(kw)
    kw.update(over)
    return kw


def _policy(**over: Any):
    from sb3_contrib import MaskablePPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    from agents.model.identity_init_test import _Env
    from agents.model.policy import Gen3DualHeadMaskablePolicy
    enc = Gen3ObservationEncoder(load_mappings())
    kw = {**enc.get_features_extractor_kwargs(), **_prod_kwargs(**over)}
    torch.manual_seed(0)
    return MaskablePPO(
        Gen3DualHeadMaskablePolicy, DummyVecEnv([lambda: _Env(enc.dimension)]),
        n_steps=16, batch_size=16, n_epochs=1, device="cpu",
        policy_kwargs={"features_extractor_class": Gen3FeaturesExtractor,
                       "features_extractor_kwargs": kw}).policy


def _pair(**base: Any) -> Tuple[Any, Any]:
    """(ON, OFF) real policies on the same base config, ON being that config's production value."""
    assert _prod_kwargs()["value_threat_inject"] is True, "production must carry the route for F10 to delete it"
    return _policy(**base), _policy(**base, value_threat_inject=False)


@pytest.fixture(scope="module")
def obs() -> torch.Tensor:
    return torch.as_tensor(np.load(src_path("agents", "model", "compile_parity_obs.npz"))["obs"])[:8]


@pytest.fixture(scope="module")
def prod_pair():
    return _pair()


def _route_keys(sd: Dict[str, torch.Tensor]) -> set:
    return {k for k in sd if _ROUTE in k}


def _assert_one_lever(on, off) -> None:
    """The three structural properties every (ON, OFF) pair must hold."""
    fe_on, fe_off = on.features_extractor, off.features_extractor
    # ON: production's live route, untouched by the retire hook.
    assert fe_on.cls_pool.value_threat_live and fe_on.cls_pool.value_threat_proj is not None
    assert fe_on.retire_value_threat_inject() is False
    # OFF: no module, no key, no optimizer slot, no parameter.
    assert fe_off.cls_pool.value_threat_proj is None and not fe_off.cls_pool.value_threat_live
    a, b = on.state_dict(), off.state_dict()
    assert not _route_keys(b), sorted(_route_keys(b))[:3]
    assert set(a) - set(b) == _route_keys(a) and _route_keys(a), sorted(set(a) - set(b))[:5]
    assert not set(b) - set(a), f"OFF carries keys production lacks: {sorted(set(b) - set(a))[:5]}"
    n_on = sum(p.numel() for p in on.parameters())
    n_off = sum(p.numel() for p in off.parameters())
    assert n_on - n_off == value_threat_inject_dim() * D_MODEL + D_MODEL
    opt = {id(p) for g in off.optimizer.param_groups for p in g["params"]}
    assert opt == {id(p) for p in off.parameters()}, "the optimizer must hold exactly OFF's parameters"
    # The ONE-lever property: every shared tensor starts bit-identical.
    moved = [k for k in b if not torch.equal(a[k], b[k])]
    assert not moved, f"OFF moved other parameters' initial bytes: {moved[:5]}"


def _assert_identity_at_init(on, off, obs: torch.Tensor) -> None:
    with torch.no_grad():
        pi_on, vf_on = on.features_extractor({"observation": obs})
        pi_off, vf_off = off.features_extractor({"observation": obs})
    assert torch.isfinite(vf_off).all() and torch.isfinite(pi_off).all()
    assert torch.equal(pi_on, pi_off), "policy features must be bitwise equal at init"
    assert torch.equal(vf_on, vf_off), "value features must be bitwise equal at init (zero-init route)"


# ------------------------------------------------------------------------------------------ production
def test_off_retires_the_route_and_moves_no_other_init_byte(prod_pair):
    _assert_one_lever(*prod_pair)


def test_off_starts_where_production_starts(prod_pair, obs):
    _assert_identity_at_init(*prod_pair, obs)


def test_a_bare_off_extractor_never_reads_the_projection(obs):
    """Before the policy retires it, the OFF projection exists but is NOT LIVE: an arbitrary weight in it must
    leave both halves bit-identical, while the same weight in an ON extractor moves the critic (so the test can
    see the route at all)."""
    import gymnasium as gym
    mappings = load_mappings()
    layout = Gen3ObservationEncoder(mappings).get_layout()
    space = gym.spaces.Box(0.0, 1.0, shape=(layout["total_dim"],), dtype=np.float32)
    outs = {}
    for live in (False, True):
        torch.manual_seed(0)
        fe = Gen3FeaturesExtractor(space, layout=layout, mappings=mappings,
                                   **_prod_kwargs(value_threat_inject=live)).eval()
        assert fe.cls_pool.value_threat_proj is not None and fe.cls_pool.value_threat_live is live
        with torch.no_grad():
            before = fe({"observation": obs})
            g = torch.Generator().manual_seed(7)
            proj = fe.cls_pool.value_threat_proj.proj
            proj.weight.copy_(torch.randn(proj.weight.shape, generator=g) * 3.0)
            proj.bias.copy_(torch.randn(proj.bias.shape, generator=g) * 3.0)
            after = fe({"observation": obs})
        outs[live] = (before, after)
    (pi0, vf0), (pi1, vf1) = outs[False]
    assert torch.equal(pi0, pi1) and torch.equal(vf0, vf1), "OFF read the not-live projection"
    (_, vf_on0), (_, vf_on1) = outs[True]
    assert not torch.equal(vf_on0, vf_on1), "the same weight must move an ON critic, or this test is blind"


def test_off_keeps_the_ops_production_rung_and_skips_the_reducer():
    """OFF leaves the op at R0 `hard_max` (no reducer, nothing stashed): the reduced rows fed ONLY this route,
    and R1 is parameter-free, so the rung costs no init byte either way."""
    import gymnasium as gym
    mappings = load_mappings()
    layout = Gen3ObservationEncoder(mappings).get_layout()
    space = gym.spaces.Box(0.0, 1.0, shape=(layout["total_dim"],), dtype=np.float32)
    torch.manual_seed(0)
    fe = Gen3FeaturesExtractor(space, layout=layout, mappings=mappings, **_prod_kwargs(value_threat_inject=False))
    assert fe.damage_op.reduce_how == "hard_max" and fe.damage_op.pair_reducer is None


# ----------------------------------------------------------------------------------------- composition
_COMPOSE = {
    "static": {"token_encoding": "static"},
    "trunk": {"policy_readout": "trunk"},
    "move_resolution": {"move_resolution": "on"},
    "fixed_mass": {"belief_tokens": "fixed_mass"},
    "static+trunk+fixed_mass": {"token_encoding": "static", "policy_readout": "trunk",
                                "belief_tokens": "fixed_mass"},
}


@pytest.mark.parametrize("name", sorted(_COMPOSE))
def test_off_composes_with_the_other_model_flags(name, obs):
    on, off = _pair(**_COMPOSE[name])
    _assert_one_lever(on, off)
    _assert_identity_at_init(on, off, obs)
