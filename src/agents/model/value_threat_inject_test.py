"""Gates V0-V3 for the critic threat-injection route (gen3_value_threat_inject_v1, v64).

`design_opponent_intent.md` §7a.2c. Two of these carry the whole arm:

  V1 — `pi` is bit-identical ON vs OFF for an ARBITRARY `W_inj`, not merely at init. This is the
       executable form of "vf-only". Without it, an ELO move could be a policy change wearing a
       critic change's name, and the experiment would answer nothing.
  V2 — permuting OUR six mons leaves `value_pooled` INVARIANT. This is why the route was chosen
       over the deleted flat concat, whose meaning was slot-order-dependent by construction.

The others are the standard structural-toggle contract: OFF byte-identical (V0), a version gate
that FATALs on a flip (V3), and the identity-init guard actually covering the new Linear (M1 — SB3's
ortho pass destroys extractor zero-inits, which silently falsified six shipped features' cold-start
claims before it was found).
"""
import numpy as np
import pytest
import torch


def _space(layout):
    import gymnasium as gym
    return gym.spaces.Dict({
        "observation": gym.spaces.Box(-np.inf, np.inf, (layout["total_dim"],), np.float32),
        "action_mask": gym.spaces.Box(0, 1, (11,), np.float32)})


def _common(layout):
    return dict(layout=layout, move_belief_mode="revealed", damage_op=True,
                attend_unrevealed_opponents=True)


def _pair(seed=0):
    """A WEIGHT-MATCHED (OFF, ON) extractor pair.

    Seeding both constructions identically is NOT enough: building `ValueThreatInject` draws from
    the RNG stream (a `Linear`'s kaiming init runs before we zero it), so every module built after
    it would receive different draws in ON than in OFF, and a forward comparison would report a
    difference that has nothing to do with the injection. So the shared weights are copied across
    explicitly — which is also the sharper claim: with the SAME body, does the injection change
    anything at zero-init?
    """
    from agents.model.damage_op_test import _make_layout
    from agents.model.features_extractor import Gen3FeaturesExtractor
    layout = _make_layout()
    space, common = _space(layout), _common(layout)
    torch.manual_seed(seed); off = Gen3FeaturesExtractor(space, **common)
    torch.manual_seed(seed); on = Gen3FeaturesExtractor(space, **common, value_threat_inject=True)
    shared = {k: v for k, v in on.state_dict().items()
              if not k.startswith("cls_pool.value_threat_proj.")}
    missing, unexpected = off.load_state_dict(shared, strict=False)
    assert not unexpected, f"ON carries keys OFF lacks beyond the injection: {unexpected}"
    assert not missing, f"OFF has keys ON lacks: {missing}"
    return off, on, layout


def _obs(layout, batch=4, seed=1):
    g = torch.Generator().manual_seed(seed)
    return {"observation": torch.rand(batch, layout["total_dim"], generator=g),
            "action_mask": torch.ones(batch, 11)}


# ---------------------------------------------------------------- module-level

def test_projection_is_zero_init_and_shared_over_our_mons():
    from agents.model.value_threat_inject import ValueThreatInject, value_threat_inject_dim
    dim = value_threat_inject_dim()
    m = ValueThreatInject(dim, 128)
    assert not m.proj.weight.any() and not m.proj.bias.any(), "W_inj must be zero-init"
    tok = torch.randn(3, 6, 128)
    rows = torch.randn(3, 6, dim)
    assert torch.equal(m(tok, rows), tok), "zero-init ⇒ the augmentation is the identity"
    # ONE Linear over all six rows: feeding the same row in two slots must give the same delta.
    m.proj.weight.data.normal_()
    rows2 = rows.clone(); rows2[:, 1] = rows2[:, 0]
    out = m(torch.zeros(3, 6, 128), rows2)
    assert torch.allclose(out[:, 0], out[:, 1], atol=1e-6), \
        "the projection must be SHARED over j — a per-slot map would be positional"


def test_width_guard_rejects_a_mismatched_row():
    from agents.model.value_threat_inject import ValueThreatInject
    m = ValueThreatInject(13, 128)
    with pytest.raises(ValueError, match="width 13"):
        m(torch.randn(2, 6, 128), torch.randn(2, 6, 16))


def test_inject_dim_matches_the_rung_the_op_will_build():
    """The pre-op helper and the real reducer must agree — the assert in __init__ depends on it."""
    from agents.model.damage_op import _PAIR_REDUCE_N_CHANNELS
    from agents.model.pair_reduce import PairReducer
    from agents.model.value_threat_inject import (VALUE_THREAT_INJECT_REDUCE_HOW,
                                                  value_threat_inject_dim)
    built = PairReducer(VALUE_THREAT_INJECT_REDUCE_HOW, _PAIR_REDUCE_N_CHANNELS).extra_dim
    assert value_threat_inject_dim() == built == 13


# ---------------------------------------------------------------- V0 / V1 / V2

def test_v0_off_adds_no_module_and_on_adds_only_its_own_keys():
    off, on, _ = _pair()
    assert off.cls_pool.value_threat_proj is None
    assert on.cls_pool.value_threat_proj is not None
    new = set(on.state_dict()) - set(off.state_dict())
    assert new and all(k.startswith("cls_pool.value_threat_proj.") for k in new), sorted(new)[:5]
    assert not (set(off.state_dict()) - set(on.state_dict())), "OFF must not have keys ON lacks"


def test_v0_on_is_bitwise_identical_to_off_at_init():
    """Zero-init ⇒ step 0 is the OFF model exactly, in BOTH heads."""
    off, on, layout = _pair()
    off.eval(); on.eval()
    obs = _obs(layout)
    with torch.no_grad():
        pi_off, vf_off = off(obs)
        pi_on, vf_on = on(obs)
    assert torch.equal(pi_off, pi_on), "policy features must be bitwise identical at init"
    assert torch.equal(vf_off, vf_on), "value features must be bitwise identical at init"


def test_v1_policy_is_bit_identical_for_an_ARBITRARY_projection():
    """THE vf-only gate. Not 'identical at init' — identical at any weight, while vf MOVES.

    A large random `W_inj` is used deliberately: a small one could hide a leak inside float noise
    and let this pass for the wrong reason.
    """
    off, on, layout = _pair()
    off.eval(); on.eval()
    obs = _obs(layout)
    with torch.no_grad():
        pi_off, vf_off = off(obs)
        on.cls_pool.value_threat_proj.proj.weight.data.normal_(0.0, 5.0)
        on.cls_pool.value_threat_proj.proj.bias.data.normal_(0.0, 5.0)
        pi_on, vf_on = on(obs)
    assert torch.equal(pi_off, pi_on), (
        "POLICY LEAK: pi changed when only the critic's injection weights moved — the augmented "
        "tokens have escaped the value pool")
    assert not torch.allclose(vf_off, vf_on), (
        "the critic did NOT move under a large injection — the route is inert, so V1 passes "
        "vacuously and the arm would measure nothing")


def test_v2_value_pool_is_invariant_to_permuting_our_mons():
    """Equivariance: the row rides its own mon's token and the pool is permutation-invariant.

    Exercised on CLSPool directly so the permutation is exact — permuting the OBS would also
    permute the op's internals, which tests a different (and weaker) claim.
    """
    from agents.model.features_extractor import CLSPool
    from agents.model.value_threat_inject import value_threat_inject_dim
    from agents.model.damage_op_test import _make_layout
    torch.manual_seed(0)
    pool = CLSPool(_make_layout(), value_threat_inject_dim=value_threat_inject_dim()).eval()
    pool.value_threat_proj.proj.weight.data.normal_(0.0, 1.0)   # a real, non-identity injection

    B, T, D = 2, 6, 128
    ours = torch.randn(B, T, D)
    theirs = torch.randn(B, T, D)
    rows = torch.randn(B, T, value_threat_inject_dim())

    class _Ctx:
        batch_size, device = B, torch.device("cpu")
        our_active_idx = torch.zeros(B, dtype=torch.long)
        fainted_mask_ours = torch.zeros(B, T, dtype=torch.bool)
        fainted_mask_opp = torch.zeros(B, T, dtype=torch.bool)
        all_fainted = torch.zeros(B, 2 * T, dtype=torch.bool)

    with torch.no_grad():
        _, _, _, v_ref = pool(ours, theirs, _Ctx(), threat_rows=rows)
        perm = torch.tensor([3, 1, 5, 0, 4, 2])
        _, _, _, v_perm = pool(ours[:, perm], theirs, _Ctx(), threat_rows=rows[:, perm])
    assert torch.allclose(v_ref, v_perm, atol=1e-5), (
        f"value_pooled is NOT permutation-invariant over our mons "
        f"(max|Δ| = {(v_ref - v_perm).abs().max():.3e}) — the injection has become slot-indexed")


def test_v2b_permuting_only_the_rows_DOES_change_the_pool():
    """The counterpart that stops V2 passing vacuously: mismatching rows to tokens MUST matter.

    If this also came out invariant, the injection would be carrying no per-entity information and
    V2 would be measuring an inert path.
    """
    from agents.model.features_extractor import CLSPool
    from agents.model.value_threat_inject import value_threat_inject_dim
    from agents.model.damage_op_test import _make_layout
    torch.manual_seed(0)
    pool = CLSPool(_make_layout(), value_threat_inject_dim=value_threat_inject_dim()).eval()
    pool.value_threat_proj.proj.weight.data.normal_(0.0, 1.0)
    B, T, D = 2, 6, 128
    ours, theirs = torch.randn(B, T, D), torch.randn(B, T, D)
    rows = torch.randn(B, T, value_threat_inject_dim())

    class _Ctx:
        batch_size, device = B, torch.device("cpu")
        our_active_idx = torch.zeros(B, dtype=torch.long)
        fainted_mask_ours = torch.zeros(B, T, dtype=torch.bool)
        fainted_mask_opp = torch.zeros(B, T, dtype=torch.bool)
        all_fainted = torch.zeros(B, 2 * T, dtype=torch.bool)

    with torch.no_grad():
        _, _, _, v_ref = pool(ours, theirs, _Ctx(), threat_rows=rows)
        _, _, _, v_bad = pool(ours, theirs, _Ctx(), threat_rows=rows[:, [3, 1, 5, 0, 4, 2]])
    assert not torch.allclose(v_ref, v_bad, atol=1e-5), \
        "re-pairing rows to the WRONG mons changed nothing — the injection carries no entity info"


# ---------------------------------------------------------------- guards / V3

def test_identity_init_guard_covers_the_projection():
    """M1: SB3's ortho pass clobbers extractor zero-inits; the guard must re-zero this one."""
    _, on, _ = _pair()
    assert any(n.endswith("cls_pool.value_threat_proj.proj") for n in on._identity_init_zeroed), \
        f"W_inj is not in the identity-init capture set: {on._identity_init_zeroed}"
    on.cls_pool.value_threat_proj.proj.weight.data.normal_()   # simulate the ortho clobber
    assert on.restore_identity_init() >= 1
    assert not on.cls_pool.value_threat_proj.proj.weight.any(), "guard failed to re-zero W_inj"


def test_enabling_without_the_damage_op_fails_loud():
    from agents.model.damage_op_test import _make_layout
    from agents.model.features_extractor import Gen3FeaturesExtractor
    layout = _make_layout()
    with pytest.raises(ValueError, match="requires damage_op"):
        Gen3FeaturesExtractor(_space(layout), layout=layout, move_belief_mode="revealed",
                              attend_unrevealed_opponents=True,
                              damage_op=False, value_threat_inject=True)


def test_on_forces_the_belief_mean_rung_and_populates_the_rows():
    """OFF must stay on R0 (no reducer at all); ON must build R1 and stash real rows."""
    from agents.model.value_threat_inject import VALUE_THREAT_INJECT_REDUCE_HOW
    off, on, layout = _pair()
    assert off.damage_op.reduce_how == "hard_max" and off.damage_op.pair_reducer is None
    assert on.damage_op.reduce_how == VALUE_THREAT_INJECT_REDUCE_HOW
    assert on.damage_op.pair_reducer is not None
    on.eval()
    with torch.no_grad():
        on(_obs(layout))
    rows = on.damage_op.last_reduced_extra
    assert rows is not None and rows.shape[1:] == (6, 13), f"unexpected rows: {None if rows is None else rows.shape}"


def test_v0_real_policy_survives_sb3_ortho_init():
    """V0 on the REAL construction path — MaskablePPO → ActorCriticPolicy._build().

    The bare-extractor tests above cannot see this failure mode at all: SB3's ortho pass only runs
    on a policy build, and it is what silently destroyed six earlier features' zero-inits (M1). A
    clobbered `W_inj` would mean the arm starts as a RANDOM perturbation of the critic rather than
    as the OFF model — invisible in training, and it would poison the generation comparison.
    """
    from agents.model.identity_init_test import _build_real_policy
    model, _ = _build_real_policy(value_threat_inject=True)
    fe = model.policy.features_extractor
    assert fe.value_threat_inject and fe.cls_pool.value_threat_proj is not None, \
        "the override did not reach the extractor — this test would pass vacuously"
    w = fe.cls_pool.value_threat_proj.proj.weight
    assert not w.any(), (
        f"W_inj was clobbered by SB3's ortho init and not restored (max|w| = "
        f"{float(w.abs().max()):.3e}) — the ON arm would not start at the OFF model")


def test_v64_migration_defaults_off():
    """The v64 default-injection branch is pre-floor (MIGRATION_FLOOR): a v63 config is a
    pre-generation checkpoint and is refused outright instead of migrating to OFF."""
    from agents.model.model_version import _migrate_config, MODEL_CONFIG_VERSION, ModelVersionError
    assert MODEL_CONFIG_VERSION >= 64
    with pytest.raises(ModelVersionError, match="PRE-GENERATION"):
        _migrate_config({"config_version": 63})


def test_v3_version_gate_rejects_a_toggle_flip():
    import dataclasses
    from agents.model.model_version import ModelVersionError
    from agents.observation.state_encoder import load_mappings
    from agents.model.snapshot import current_model_version
    base = current_model_version(load_mappings())
    on = dataclasses.replace(base, value_threat_inject=True)
    with pytest.raises(ModelVersionError, match="value_threat_inject"):
        on.check_compatible(base)
    with pytest.raises(ModelVersionError, match="value_threat_inject"):
        base.check_compatible(on)
    on.check_compatible(on)          # matching pair is fine
