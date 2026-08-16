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


# ---------------------------------------------------------------- the v80 x v74 interaction
# The defect these pin: `forward_internal`'s intent-value-reduce DISCOVERY branch used to
# `return` the pair outright. Every value part appended BELOW it was therefore invisible to the
# dummy forward that sizes `value_pre_norm` — and v80's entity pool landed exactly there. The
# critic was built UVR_OUT_DIM short and died on the first real forward with
# `normalized_shape=[1241] ... got [*, 1369]`.
#
# Why nothing caught it: every test above builds `value_entity_pool=True` ALONE, and the intent
# tests build `intent_value_reduce=True` alone. The bug lives only in the intersection, and an
# intersection nobody constructs is an intersection nobody tests. Production wanted both on the
# very next run.

def _build_both():
    """Both value parts on — the production gen-12 shape.

    Reuses `intent_value_reduce_test._build`'s base, which is the known-good set for that half
    (the reduce needs opp_intent for a distribution, the op's top-K cells to weight, and both
    damage matrices to compute them), and adds the v80 pool on top. Borrowed rather than re-typed
    so this test cannot drift from the config the intent tests prove.
    """
    from agents.model.intent_value_reduce_test import _build
    return _build(intent_value_reduce=True, value_entity_pool=True)


def test_the_vf_projection_is_sized_for_BOTH_value_parts():
    """The direct assertion: the discovered width must equal what a real forward produces."""
    model, enc = _build_both()
    fe = model.policy.features_extractor
    if fe.intent_value_reduce is None or fe.value_entity_pool is None:
        pytest.skip("this build resolved without both value parts; nothing to intersect")
    with torch.no_grad():
        _pi, vf = fe.forward_internal(_obs(enc, n=3))
    assert vf.shape[1] == fe.value_projection_input_dim, (
        f"vf width {vf.shape[1]} != discovered {fe.value_projection_input_dim} "
        f"(delta {vf.shape[1] - fe.value_projection_input_dim}); the discovery forward skipped a "
        f"value part appended after the intent-reduce branch")
    assert fe.value_pre_norm.normalized_shape[0] == vf.shape[1]


def test_a_real_forward_through_the_policy_does_not_raise(model_and_enc):
    """The end-to-end shape: whatever the discovery did, the built critic must actually run."""
    model, enc = _build_both()
    with torch.no_grad():
        pi, vf = model.policy.features_extractor(_obs(enc, n=5))
    assert pi.shape[0] == vf.shape[0] == 5
    assert torch.isfinite(vf).all()


def test_the_discovery_branch_falls_through_rather_than_returning():
    """Pin the SHAPE of the fix, not just its effect — a future value part appended below the
    intent-reduce branch must be reached too, and only a fall-through guarantees that.

    Written this way after a first cut PASSED against the reintroduced bug: it split on `else:`
    and, with no `else:` in the branch, silently took the whole window as the prefix. So the
    `else:` is now REQUIRED to exist, which is the actual structural claim — an if/else, not an
    early exit — and its absence fails instead of being absorbed.
    """
    import agents.model.features_extractor as fx
    src = inspect.getsource(fx.Gen3FeaturesExtractor.forward_internal)
    # Anchor on INTENT_VALUE_REDUCE_DIM, which appears only in this branch. `_intent_reduce_
    # discovering` does NOT identify it — `intent_move_cell` guards on the same flag and comes
    # FIRST, so an earlier cut of this test silently inspected that branch instead and passed
    # against the reintroduced bug.
    i = src.find("INTENT_VALUE_REDUCE_DIM")
    assert i > 0, "the intent-value-reduce discovery branch moved; update this test"
    i = src.rfind("_intent_reduce_discovering", 0, i)
    # CODE only — the branch's own comment says the word "return" (it documents that it must not
    # use one), and scanning raw text made the test fail against the very fix it guards.
    window = "\n".join(ln for ln in src[i:i + 1600].splitlines()
                       if not ln.lstrip().startswith("#"))
    j = window.find("\n            else:")
    assert j > 0, (
        "the intent-reduce discovery branch is not an if/else — it must not exit early, or every "
        "value part appended below it (v80's entity pool, and whatever comes next) is invisible "
        "to the forward that sizes value_pre_norm")
    assert "return" not in window[:j], (
        "the discovery branch returns before its else; it must fall through")
