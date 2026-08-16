"""gen3_unified_value_readout_v1 (v80) — the Stage-3 critic entity pool's contract, pinned.

The four claims a delivery-route flag must prove: OFF builds NOTHING (byte-identical baseline);
ON contributes EXACTLY zero at init (identity-at-init, surviving SB3's ortho clobber via the
end-of-__init__ zero-Linear sweep); the policy half is untouched at ANY weight, not merely at
init (vf-only by placement); and a masked entity row gets zero attention. Plus the fail-loud
op-rows requirement and the v80 migration stamp.
"""
import inspect

import numpy as np
import pytest
import torch

pytest.importorskip("sb3_contrib")

from agents.model.arch_constants import UVR_DIM, UVR_K, UVR_OUT_DIM, D_MODEL
from agents.model.features_extractor import UnifiedValueReadout
from agents.model.identity_init_test import _build_real_policy
from agents.model.model_version import MODEL_CONFIG_VERSION, _migrate_config


@pytest.fixture(scope="module")
def model_and_enc():
    return _build_real_policy(value_entity_pool=True)


def _obs(enc, n=4, seed=0):
    rng = np.random.default_rng(seed)
    return {"observation": torch.as_tensor(rng.random((n, enc.dimension), dtype=np.float32)),
            "action_mask": torch.ones(n, 11)}


def test_off_builds_nothing(model_and_enc):
    off, _ = _build_real_policy()          # flag absent == OFF
    assert not any("value_entity_pool" in k for k in off.policy.state_dict()), \
        "OFF must not construct the module — the byte-identical baseline"
    on, _ = model_and_enc
    assert any("value_entity_pool.out_proj" in k for k in on.policy.state_dict())


def test_zero_init_survives_policy_build_and_contributes_zero(model_and_enc):
    """The out projection must still be ZERO after SB3's ortho-init pass (the identity-init
    sweep must have picked it up), so the module's forward output is exactly 0 cold."""
    model, _ = model_and_enc
    fe = model.policy.features_extractor
    uvr = fe.value_entity_pool
    assert any(n.endswith("value_entity_pool.out_proj")
               for n in fe._identity_init_zeroed), \
        "out_proj not in the identity-init sweep — SB3 ortho-init would clobber the zero"
    assert float(uvr.out_proj.weight.abs().max()) == 0.0
    assert float(uvr.out_proj.bias.abs().max()) == 0.0
    out = uvr(torch.randn(3, 6, D_MODEL), torch.randn(3, 6, D_MODEL),
              torch.zeros(3, 12, dtype=torch.bool),
              torch.randn(3, 6, uvr.op_proj.in_features), torch.ones(3, 6))
    assert out.shape == (3, UVR_OUT_DIM) and float(out.abs().max()) == 0.0


def test_pi_untouched_at_any_weight_and_vf_fires(model_and_enc):
    """vf-ONLY by placement: randomizing the pool's output weights must move vf and leave pi
    bit-identical — the intent_value_reduce structural property, not an init coincidence."""
    model, enc = model_and_enc
    fe = model.policy.features_extractor
    obs = _obs(enc)
    with torch.no_grad():
        pi0, vf0 = fe(obs)
        torch.nn.init.normal_(fe.value_entity_pool.out_proj.weight, std=1.0)
        pi1, vf1 = fe(obs)
        torch.nn.init.zeros_(fe.value_entity_pool.out_proj.weight)   # restore for other tests
    assert torch.equal(pi0, pi1), "the policy half read the value-only pool"
    assert not torch.allclose(vf0, vf1), "randomized out_proj moved nothing — the pool is dead"


def test_masked_rows_get_zero_attention_and_all_masked_is_finite():
    uvr = UnifiedValueReadout(0)                       # token-only variant (no op)
    fainted = torch.zeros(2, 12, dtype=torch.bool)
    fainted[:, 3] = True
    uvr(torch.randn(2, 6, D_MODEL), torch.randn(2, 6, D_MODEL), fainted)
    assert uvr.last_att.shape == (2, UVR_K, 12)
    assert float(uvr.last_att[:, :, 3].abs().max()) < 1e-6
    # the dummy discovery forward's board: every row masked → uniform average, never NaN
    out = uvr(torch.zeros(1, 6, D_MODEL), torch.zeros(1, 6, D_MODEL),
              torch.ones(1, 12, dtype=torch.bool))
    assert bool(torch.isfinite(out).all())


def test_op_variant_fails_loud_without_rows():
    uvr = UnifiedValueReadout(17)
    with pytest.raises(ValueError, match="supplied no op rows"):
        uvr(torch.randn(1, 6, D_MODEL), torch.randn(1, 6, D_MODEL),
            torch.zeros(1, 12, dtype=torch.bool))


def test_v80_migration_stamps_and_defaults_off():
    data = {"config_version": 79, "obs_dim": 1, "n_actions": 11}
    out = _migrate_config(dict(data))
    assert out["config_version"] == MODEL_CONFIG_VERSION == 80
    assert out["value_entity_pool"] is False
