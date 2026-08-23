"""gen3_intent_threshold_v1 (v84) — the α-weighted threshold operator's gates.

What must hold (design_conditional_execution.md §6 G0/G1 + the house rules):
  * OFF is the default and builds NOTHING — no modules, no extra dims, no state_dict keys.
  * ON contributes EXACTLY zero to the pointer move cells at init (the projection is
    zero-init, captured by the identity-init sweep — ledger M1).
  * ⚠️ THE POLICY HALF SURVIVES THE CRITIC-ROUTE DELETION WAVE. v84 built TWO consumers off
    one flag; the wave deleted the p_KO vf route (`IntentThresholdValue`, dV 0.155 then 0.136
    against a 0.39 bar) and KEPT the pointer MOVE cell, which is a per-action channel measured
    in KL/flips, not |dV|. `test_the_p_ko_policy_context_survives_the_vf_route_deletion` is the
    pin: p_KO must still reach the move logits, and moving it must still move them.
  * The contraction math: p_KO is the α-weighted sum of the op's per-candidate KO ramps;
    SWITCH mass shrinks every threshold probability toward zero (unrenormalized slice); a
    threshold on the roll distribution is not a function of its mean (the §3.0 point).
  * Seat-permutation invariance: jointly permuting α's move seats and the operand columns
    leaves every probability unchanged.
  * An axis-width mismatch and a missing-stash forward FAIL LOUD (the `op move-order` class).
  * The v84 version machinery: migration default OFF + the check_compatible gate.
"""
import dataclasses

import gymnasium as gym
import numpy as np
import pytest
import torch

from agents.model.arch_constants import INTENT_THRESH_MOVE_DIM
from agents.model.features_extractor import Gen3FeaturesExtractor, TEAM_SIZE
from agents.model.intent_threshold import (
    IntentThresholdMoveCell, threshold_probs,
)
from agents.model.model_version import (
    MODEL_CONFIG_VERSION, ModelVersion, ModelVersionError, _migrate_config,
)
from agents.observation.state_encoder import Gen3ObservationEncoder, load_mappings

_ON_KWARGS = dict(
    attend_unrevealed_opponents=True, move_belief_mode="revealed", move_prior_fusion=True,
    move_latent=True, damage_op=True, damage_outgoing=True, damage_matrices_incoming=True,
    damage_topk_k=6, entity_topk_seats=6, opp_intent=True, intent_threshold=True,
)


def _build(**kwargs):
    mappings = load_mappings()
    layout = Gen3ObservationEncoder(mappings).get_layout()
    space = gym.spaces.Box(0.0, 1.0, shape=(layout["total_dim"],), dtype=np.float32)
    torch.manual_seed(7)
    fe = Gen3FeaturesExtractor(space, layout=layout, mappings=mappings, **kwargs)
    fe.eval()
    return fe, layout


def _obs(layout, b=3):
    torch.manual_seed(11)
    return {"observation": torch.rand(b, layout["total_dim"])}


# ------------------------------------------------------------------- the pure contraction


def _hand_cells(B=2, K=3, ko=None, high=None, acc=None):
    """[B,6,K,6] cells in the op's channel order [low, high, crit, ko, acc, is_phys]."""
    cells = torch.zeros(B, TEAM_SIZE, K, 6)
    cells[..., 1] = torch.tensor(high if high is not None else [0.5] * K)
    cells[..., 3] = torch.tensor(ko if ko is not None else [0.3] * K)
    cells[..., 4] = torch.tensor(acc if acc is not None else [1.0] * K)
    return cells


def _uniform_alpha_logits(B=2, K=3, switch_logit=0.0):
    lg = torch.zeros(B, K + 1)
    lg[:, -1] = switch_logit
    return lg


def test_p_ko_is_the_alpha_weighted_sum():
    K = 3
    ko = [0.9, 0.1, 0.5]
    cells = _hand_cells(K=K, ko=ko)
    gate = torch.ones(2, TEAM_SIZE, 1)
    active = torch.zeros(2, dtype=torch.long)
    lg = torch.zeros(2, K + 1)
    lg[:, 0] = 10.0                                   # α ≈ all on seat 0 (SWITCH ≈ 0)
    p_ko, _, _ = threshold_probs(lg, cells, gate, active)
    assert torch.allclose(p_ko, torch.full((2, 1), 0.9), atol=1e-3)
    # uniform α over 3 seats + SWITCH: each move seat gets 1/4 (unrenormalized slice)
    p_ko_u, _, _ = threshold_probs(_uniform_alpha_logits(K=K), cells, gate, active)
    want = sum(ko) / 4.0
    assert torch.allclose(p_ko_u, torch.full((2, 1), want), atol=1e-5)


def test_switch_mass_shrinks_every_threshold_toward_zero():
    """The unrenormalized slice: a certain SWITCH read means no damage this turn."""
    cells = _hand_cells(ko=[0.9, 0.9, 0.9], high=[0.9, 0.9, 0.9])
    gate = torch.ones(2, TEAM_SIZE, 1)
    active = torch.zeros(2, dtype=torch.long)
    certain_switch = _uniform_alpha_logits(switch_logit=20.0)
    p_ko, p_sub, p_fp = threshold_probs(certain_switch, cells, gate, active)
    for p in (p_ko, p_sub, p_fp):
        assert float(p.abs().max()) < 1e-4


def test_threshold_is_not_a_function_of_the_mean():
    """§3.0's structural point: two candidate sets with the SAME mean damage but different
    spreads must produce different sub-break readings — max/mean cannot represent this."""
    gate = torch.ones(1, TEAM_SIZE, 1)
    active = torch.zeros(1, dtype=torch.long)
    lg = _uniform_alpha_logits(B=1, K=2)[:, :3]                     # 2 seats + SWITCH
    tight = _hand_cells(B=1, K=2, high=[0.24, 0.24])                # both under the sub's 25%
    wide = _hand_cells(B=1, K=2, high=[0.04, 0.44])                 # same mean, one breaks
    _, p_sub_tight, _ = threshold_probs(lg, tight, gate, active)
    _, p_sub_wide, _ = threshold_probs(lg, wide, gate, active)
    assert float(p_sub_tight) < 1e-4
    assert float(p_sub_wide) > 0.2


def test_status_and_immune_candidates_do_not_break_focus_punch():
    """BP-0/immune candidates have high == 0 ⇒ they cannot break the punch — the immunity
    term the design flagged as the likeliest G0 mistake falls out of the physics."""
    cells = _hand_cells(ko=[0.0, 0.0, 0.0], high=[0.0, 0.0, 0.6])
    gate = torch.ones(1, TEAM_SIZE, 1)
    active = torch.zeros(1, dtype=torch.long)
    lg = torch.zeros(1, 4)
    lg[:, 0] = 10.0                                                  # α on the status candidate
    _, _, p_fp = threshold_probs(lg, cells[:1], gate, active)
    assert float(p_fp) < 1e-3
    lg2 = torch.zeros(1, 4)
    lg2[:, 2] = 10.0                                                 # α on the damaging one
    _, _, p_fp2 = threshold_probs(lg2, cells[:1], gate, active)
    assert float(p_fp2) > 0.99


def test_seat_permutation_invariance():
    torch.manual_seed(3)
    K = 4
    cells = torch.rand(2, TEAM_SIZE, K, 6)
    gate = torch.ones(2, TEAM_SIZE, 1)
    active = torch.randint(0, TEAM_SIZE, (2,))
    lg = torch.randn(2, K + 1)
    perm = torch.randperm(K)
    lg_p = torch.cat([lg[:, perm], lg[:, -1:]], dim=1)
    out = threshold_probs(lg, cells, gate, active)
    out_p = threshold_probs(lg_p, cells[:, :, perm], gate, active)
    for a, b in zip(out, out_p):
        assert torch.allclose(a, b, atol=1e-6)


def test_axis_width_mismatch_raises():
    cells = _hand_cells(K=3)
    with pytest.raises(ValueError, match="SAME axis"):
        threshold_probs(torch.zeros(2, 5), cells, torch.ones(2, TEAM_SIZE, 1),
                        torch.zeros(2, dtype=torch.long))


# ------------------------------------------------------------------- the two heads


def test_move_cell_gates_route_each_mechanic():
    m = IntentThresholdMoveCell(6)
    with torch.no_grad():                            # identity read: raw channels straight through
        m.proj.weight.copy_(torch.eye(6))
    p_ko = torch.tensor([[0.8]])
    p_sub = torch.tensor([[0.3]])
    p_fp = torch.tensor([[0.4]])
    ids = torch.tensor([[264, 164, 203, 283]])       # focuspunch, substitute, endure, endeavor
    out = m(p_ko, p_sub, p_fp, ids)                  # [1,4,6]
    assert torch.allclose(out[0, 0, 0], torch.tensor(0.6), atol=1e-6)   # fp executes 1−0.4
    assert torch.allclose(out[0, 1, 1], torch.tensor(0.7), atol=1e-6)   # sub survives 1−0.3
    assert torch.allclose(out[0, 2, 2], torch.tensor(0.8), atol=1e-6)   # endure·p_KO
    assert torch.allclose(out[0, 3, 4], torch.tensor(0.2), atol=1e-6)   # endeavor 1−p_KO
    assert torch.allclose(out[..., 5], torch.full((1, 4), 0.8))         # p_KO context everywhere
    # a non-mechanic slot's gated channels stay zero
    out2 = m(p_ko, p_sub, p_fp, torch.tensor([[89, 0, 0, 0]]))          # earthquake + empties
    assert float(out2[..., :5].abs().max()) == 0.0


def test_the_move_cell_is_zero_init():
    m = IntentThresholdMoveCell(INTENT_THRESH_MOVE_DIM)
    assert float(m.proj.weight.abs().max()) == 0.0
    out = m(torch.rand(2, 1), torch.rand(2, 1), torch.rand(2, 1),
            torch.tensor([[264, 164, 203, 194]] * 2))
    assert float(out.abs().max()) == 0.0


# ------------------------------------------------------------------- extractor wiring


def test_off_builds_no_module_and_no_extra_dims():
    fe_off, _ = _build(**{**_ON_KWARGS, "intent_threshold": False})
    fe_on, _ = _build(**_ON_KWARGS)
    assert fe_off.intent_threshold_move is None
    assert not any("intent_threshold" in k for k in fe_off.state_dict())
    assert fe_on.pointer_move_cell_dim == fe_off.pointer_move_cell_dim + INTENT_THRESH_MOVE_DIM
    # Both HEAD widths are untouched: the cell widens the pointer stash, not pi/vf.
    assert fe_on.value_projection.in_features == fe_off.value_projection.in_features
    assert fe_on.projection.in_features == fe_off.projection.in_features


def test_on_forward_runs_and_contributes_zero_at_init():
    fe, layout = _build(**_ON_KWARGS)
    obs = _obs(layout)
    pi, vf = fe(obs)
    assert pi.shape[1] == vf.shape[1]
    assert fe._thresh_probs is not None
    # the projection is in the identity-init capture set (M1: the sweep re-zeros it after
    # SB3's ortho pass on a real policy build)
    assert "intent_threshold_move.proj" in fe._identity_init_zeroed


def test_missing_stash_fails_loud():
    with pytest.raises((RuntimeError, ValueError)):
        fe, layout = _build(**{**_ON_KWARGS, "damage_matrices_incoming": False,
                               "damage_topk_k": 0})
        fe(_obs(layout))


def test_requires_opp_intent_and_damage_op():
    with pytest.raises(ValueError, match="opp_intent"):
        _build(**{**_ON_KWARGS, "opp_intent": False})
    with pytest.raises(ValueError, match="damage_op"):
        _build(**{**_ON_KWARGS, "damage_op": False, "damage_outgoing": False,
                  "damage_matrices_incoming": False, "damage_topk_k": 0,
                  "opp_intent": True})


def test_every_surviving_value_flag_together_still_builds_and_runs():
    """The ede5a88 pin's descendant. The bug it guarded (a construction-time discovery forward
    whose early return hid every vf part below it) is now structurally impossible — widths are
    static arithmetic and every value route injects additively — so what is left to assert is the
    composition: the surviving route stack plus intent_threshold must build and run."""
    fe, layout = _build(**{**_ON_KWARGS, "opp_belief_slots": True,
                           "value_entity_pool": True, "value_threat_inject": True})
    pi, vf = fe(_obs(layout))
    assert pi.shape == vf.shape


# -------------------------------------------- THE PIN: the POLICY half survives the wave


def test_the_p_ko_policy_context_survives_the_vf_route_deletion():
    """`--intent-threshold` keeps its POINTER MOVE CELL. Do not sweep it up with the vf route.

    The critic-route deletion wave retired `IntentThresholdValue` — the p_KO vf injection —
    because it measured dV 0.155 (gen-13) and 0.136 (gen-14, n=12,391) against a 0.39 bar. The
    POLICY-side p_KO context on the move cells (v84) is a DIFFERENT object measured in DIFFERENT
    units, was never part of that verdict, and is ON in production.

    Three facts, each of which would have to be false for the policy half to have died with the
    vf half:
      1. the module exists and widens the pointer MOVE cell by `INTENT_THRESH_MOVE_DIM`;
      2. nothing named `intent_threshold_value` is left anywhere on the extractor — so a future
         reader cannot resurrect the vf route by accident;
      3. **it is LIVE, not merely present**: perturbing the threshold probabilities the cell
         consumes moves the pointer MOVE logits. Asserted with a real forward + a non-zero
         projection (zero-init would make any pin vacuous), which is the difference between
         "the wiring is there" and "the wiring carries something".
    """
    fe, layout = _build(**_ON_KWARGS)
    assert fe.intent_threshold_move is not None
    assert not hasattr(fe, "intent_threshold_value")
    assert not any("intent_threshold_value" in k for k in fe.state_dict())

    # (3) — de-zero the cell's projection so its contribution is observable at all, then move
    # p_KO and watch the MOVE logits follow.
    torch.nn.init.normal_(fe.intent_threshold_move.proj.weight, std=0.5)
    obs = _obs(layout)
    fe.eval()

    from agents.model.intent_threshold import threshold_probs as _tp

    def _logits(scale):
        # ⚠️ Patch the module that HOLDS `forward_internal`, not the `features_extractor` hub.
        # Python resolves a global at CALL time against the defining module's namespace, so a
        # patch on the hub reaches the forward only while the forward lives there — it stopped
        # doing so when the extractor was decomposed into a base-class chain on 2026-08-23. This
        # one failed loudly (the pin is `assert not equal`, so a patch that lands nowhere reads as
        # "the cell is dead"); the same drift on an `assert equal` pin would have passed for the
        # wrong reason forever, which is why the module is named through `forward_internal` itself.
        import importlib
        import inspect
        _fx = importlib.import_module(
            inspect.getmodule(type(fe).forward_internal).__name__)
        real = _fx.threshold_probs

        def patched(*a, **k):
            pr = real(*a, **k)
            return type(pr)(p_ko=pr.p_ko * scale, p_sub_broken=pr.p_sub_broken,
                            p_fp_broken=pr.p_fp_broken)
        _fx.threshold_probs = patched
        try:
            with torch.no_grad():
                fe(obs)
            return fe.last_pointer_inputs.move_cells.clone()
        finally:
            _fx.threshold_probs = real

    assert _tp is not None                      # the producer is still importable
    base = _logits(1.0)
    moved = _logits(0.0)
    assert not torch.equal(base, moved), (
        "zeroing p_KO left the pointer MOVE cells bit-identical — the POLICY-side threshold "
        "context is dead, which is exactly what this pin exists to catch.")


# ------------------------------------------------------------------- version machinery


def test_pre_floor_config_is_refused():
    """gen3_frame_deletion_v1 raised MIGRATION_FLOOR to 90, so this pre-floor config is now
    REFUSED rather than migrated — the floor's stated purpose ("refuses pre-floor configs outright
    instead of walking dead branches"). The assertion follows the behaviour: what must hold is that
    the old version is rejected with a diagnosis, not that a dead branch still defaults a field."""
    with pytest.raises(ModelVersionError, match="PRE-GENERATION|floor"):
        _migrate_config({"config_version": 83})
    assert MODEL_CONFIG_VERSION >= 84


def test_check_compatible_gates_the_flag():
    mappings = load_mappings()
    layout = Gen3ObservationEncoder(mappings).get_layout()
    b = ModelVersion.from_layout_and_policy_kwargs(layout, {"features_extractor_kwargs": {}})
    a = dataclasses.replace(b, intent_threshold=True)
    with pytest.raises(ModelVersionError, match="intent_threshold"):
        a.check_compatible(b)


# ------------------------------------------------------------------- G0: through the REAL op


def test_g0_immunity_and_category_through_the_real_physics():
    """A constructed scenario through the FULL op forward (the G0 discipline): a Flying
    defender believed to face Earthquake (immune) and Thunderbolt (hits). α on the Earthquake
    seat must read p_fp_broken ≈ 0 AND p_KO ≈ 0 — the immunity term §3.5 flags as the
    likeliest mistake, falling out of the op's own type chart rather than any rule we wrote.
    α on the Thunderbolt seat must read p_fp_broken ≈ its accuracy (1.0)."""
    from agents.model import damage_op_test as DT
    K = 5
    op, layout = DT._op_and_layout_topk(K)
    op.stash_pair_cells = True
    from agents.observation.types import TypeEncoder
    T = TypeEncoder.TYPE_TO_IDX
    eq, tb = DT._move_num("earthquake"), DT._move_num("thunderbolt")
    ctx = DT._topk_ctx(op, defenders=[(227, T["STEEL"], T["FLYING"])] + [(0, 0, 0)] * 5)
    op(ctx, DT._logits_moves(layout["max_moves"], [eq, tb]), None, DT._synth_latent(layout))
    idx = op.last_topk_idx[0].tolist()
    assert eq in idx and tb in idx
    for target, want_break in ((eq, False), (tb, True)):
        lg = torch.full((1, K + 1), -20.0)
        lg[0, idx.index(target)] = 20.0                        # α ≈ all on this seat
        p_ko, _, p_fp = threshold_probs(lg, op.last_pair_cells, op.last_pair_gate,
                                        torch.zeros(1, dtype=torch.long))
        if want_break:
            assert float(p_fp) > 0.95, f"Thunderbolt must break the punch (got {float(p_fp)})"
        else:
            assert float(p_fp) < 1e-3, f"an IMMUNE Earthquake must not (got {float(p_fp)})"
            assert float(p_ko) < 1e-3
