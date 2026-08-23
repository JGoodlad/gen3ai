"""gen3_unified_value_readout_v1 (v80) — the Stage-3 critic entity pool's contract, pinned.

The four claims a delivery-route flag must prove: OFF builds NOTHING (byte-identical baseline);
ON contributes EXACTLY zero at init (identity-at-init, surviving SB3's ortho clobber via the
end-of-__init__ zero-Linear sweep); the policy half is untouched at ANY weight, not merely at
init (vf-only by placement); and a masked entity row gets zero attention. Plus the fail-loud
op-rows requirement and the v80 migration stamp.
"""

import numpy as np
import pytest
import torch

pytest.importorskip("sb3_contrib")

from agents.model.arch_constants import UVR_K, D_MODEL
from agents.model.features_extractor import UnifiedValueReadout
from agents.model.identity_init_test import _build_real_policy
from agents.model.model_version import _migrate_config


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
    assert out.shape == (3, D_MODEL) and float(out.abs().max()) == 0.0


def test_pi_untouched_at_any_weight_and_vf_fires(model_and_enc):
    """vf-ONLY by placement: randomizing the pool's output weights must move vf and leave pi
    bit-identical — a structural property, not an init coincidence."""
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
    # a degenerate all-fainted board: every row masked → uniform average, never NaN
    out = uvr(torch.zeros(1, 6, D_MODEL), torch.zeros(1, 6, D_MODEL),
              torch.ones(1, 12, dtype=torch.bool))
    assert bool(torch.isfinite(out).all())


def test_op_variant_fails_loud_without_rows():
    uvr = UnifiedValueReadout(17)
    with pytest.raises(ValueError, match="supplied no op rows"):
        uvr(torch.randn(1, 6, D_MODEL), torch.randn(1, 6, D_MODEL),
            torch.zeros(1, 12, dtype=torch.bool))


def test_v80_config_is_refused_below_the_migration_floor():
    """gen3_frame_deletion_v1 raised MIGRATION_FLOOR to 90, so this pre-floor config is now
    REFUSED rather than migrated — the floor's stated purpose ("refuses pre-floor configs outright
    instead of walking dead branches"). The assertion follows the behaviour: what must hold is that
    the old version is rejected with a diagnosis, not that a dead branch still defaults a field."""
    from agents.model.model_version import ModelVersionError
    with pytest.raises(ModelVersionError, match="PRE-GENERATION|floor"):
        _migrate_config({"config_version": 79, "obs_dim": 1, "n_actions": 11})


# ---------------------------------------------------------------- the v80 x v74 interaction
# The defect these pin: `forward_internal`'s intent-value-reduce DISCOVERY branch used to
# `return` the pair outright. Every value part appended BELOW it was therefore invisible to the
# dummy forward that sizes `value_pre_norm` — and v80's entity pool landed exactly there. The
# critic was built one pool-width short and died on the first real forward with
# `normalized_shape=[1241] ... got [*, 1369]`.
#
# Why nothing caught it: every test above built `value_entity_pool=True` ALONE, and the intent
# tests built their own flag alone. The bug lived only in the intersection, and an intersection
# nobody constructs is an intersection nobody tests. Production wanted both on the very next run.
#
# The intersection partner has CHANGED. `intent_value_reduce` — the other value part at the time —
# was deleted by the critic-route wave (dV 0.3176 at 2x sample, below the 0.39 bar). The surviving
# critic surface is the entity pool (the seam) plus the two `CLSPool` TOKEN-CONTENT injections, so
# that is what these now intersect. The claim is unchanged and, if anything, stronger: no
# combination of critic enrichments moves any projection width, because there is no vf concat left
# for one to be appended to.

_CRITIC_STACK = dict(
    attend_unrevealed_opponents=True, move_belief_mode="revealed", move_prior_fusion=True,
    move_latent=True, damage_op=True, damage_outgoing=True, damage_matrices_incoming=True,
    damage_topk_k=6, entity_topk_seats=6, opp_intent=True,
)


def _build_both(**over):
    """Every surviving critic route on at once — the production shape."""
    kw = dict(_CRITIC_STACK, value_entity_pool=True, value_entity_pool_full=True,
              value_threat_inject=True, intent_threshold=True)
    kw.update(over)
    return _build_real_policy(**kw)


def test_the_vf_projection_is_sized_for_BOTH_value_parts():
    """The direct assertion: the static width must equal what a real forward produces
    (gen3_static_widths_v1 — the broad flag sweep lives in `projection_width_test.py`)."""
    model, enc = _build_both()
    fe = model.policy.features_extractor
    # NOT a skip. `_build_both()` passes value_entity_pool=True, value_entity_pool_full=True AND
    # value_threat_inject=True, so "this build resolved without both value parts" describes a
    # resolution BUG — and skipping on it retires the only direct check that the static vf width
    # matches a real forward.
    assert fe.value_entity_pool is not None and fe.cls_pool.value_threat_proj is not None, (
        f"_build_both() asked for both value parts but got "
        f"value_entity_pool={'present' if fe.value_entity_pool is not None else 'MISSING'}, "
        f"value_threat_proj={'present' if fe.cls_pool.value_threat_proj is not None else 'MISSING'}"
        f" — flag resolution dropped one; fix that rather than skipping the width gate")
    with torch.no_grad():
        _pi, vf = fe.forward_internal(_obs(enc, n=3))
    assert vf.shape[1] == fe.value_projection_input_dim, (
        f"vf width {vf.shape[1]} != computed {fe.value_projection_input_dim} "
        f"(delta {vf.shape[1] - fe.value_projection_input_dim}); a value part was appended "
        f"outside compute_projection_widths' arithmetic")
    assert fe.value_pre_norm.normalized_shape[0] == vf.shape[1]


def test_a_real_forward_through_the_policy_does_not_raise(model_and_enc):
    """The end-to-end shape: whatever the width arithmetic says, the built critic must run."""
    model, enc = _build_both()
    with torch.no_grad():
        pi, vf = model.policy.features_extractor(_obs(enc, n=5))
    assert pi.shape[0] == vf.shape[0] == 5
    assert torch.isfinite(vf).all()


def test_route_availability_is_width_neutral_by_construction():
    """gen3_value_pooled_routes_v1 (v89) replaced the old fall-through pin this test used to be.

    The ede5a88 bug class — a discovery branch exiting early and hiding a value part from the
    forward that sizes `value_pre_norm` — is now UNREPRESENTABLE: every value route injects
    additively into `value_pooled`, so no combination of routes changes any projection width.
    Since the critic-route deletion wave the claim is stronger still: `vf_combined IS
    value_pooled`, so the vf projection input is `D_MODEL` FLAT — not merely equal across
    route combinations, but a constant no flag can move."""
    from agents.model.arch_constants import D_MODEL as _D
    model_on, _ = _build_both()
    model_off, _ = _build_real_policy(**_CRITIC_STACK)
    fe_on = model_on.policy.features_extractor
    fe_off = model_off.policy.features_extractor
    assert fe_on.value_projection_input_dim == fe_off.value_projection_input_dim == _D
    assert (fe_on.value_pre_norm.normalized_shape[0]
            == fe_off.value_pre_norm.normalized_shape[0] == _D)


# ------------------------------------------------------------------ v82: the FULL row set

def test_full_requires_base_flag():
    with pytest.raises(ValueError, match="requires value_entity_pool=True"):
        _build_real_policy(value_entity_pool_full=True)


def test_full_pool_adds_global_and_belief_rows_and_stays_zero_init():
    """The complete Stage-3 row set: 12 team + 6 op + 1 global (+K belief when the hidden-opp
    pool exists) — attended (last_att covers every row, global never masked), still exactly
    zero cold, and the v80 3-row table is untouched when full=False (gen-12 compat)."""
    model, enc = _build_real_policy(value_entity_pool=True, value_entity_pool_full=True,
                                    opp_belief_cls_k=6, attend_unrevealed_opponents=True)
    fe = model.policy.features_extractor
    uvr = fe.value_entity_pool
    assert uvr.full and uvr.source_emb.shape[0] == 5
    assert float(uvr.out_proj.weight.abs().max()) == 0.0
    obs = _obs(enc, n=3)
    with torch.no_grad():
        pi, vf = fe(obs)
    assert torch.isfinite(vf).all()
    n_rows = uvr.last_att.shape[-1]
    assert n_rows == 12 + 6 + 1 + 6, n_rows          # team + op + global + K=6 belief
    # the global row (index 18) is never masked: it must carry attention mass somewhere
    assert float(uvr.last_att[:, :, 18].sum()) > 0.0
    # v80 compat: the base build keeps the 3-row table byte-shape
    base, _ = _build_real_policy(value_entity_pool=True)
    assert base.policy.features_extractor.value_entity_pool.source_emb.shape[0] == 3


def test_v82_config_is_refused_below_the_migration_floor():
    """gen3_frame_deletion_v1 raised MIGRATION_FLOOR to 90, so this pre-floor config is now
    REFUSED rather than migrated — the floor's stated purpose ("refuses pre-floor configs outright
    instead of walking dead branches"). The assertion follows the behaviour: what must hold is that
    the old version is rejected with a diagnosis, not that a dead branch still defaults a field."""
    from agents.model.model_version import ModelVersionError
    with pytest.raises(ModelVersionError, match="PRE-GENERATION|floor"):
        _migrate_config({"config_version": 81, "obs_dim": 1, "n_actions": 11})
