"""gen3_pair_value_route_v1 (v95) — PV's gates.

`design_opponent_intent.md` §7a(2). What must hold:

  * **vf-ONLY at ANY weight, structurally.** The augmentation happens inside `CLSPool` on a LOCAL
    copy, so `pi` is bit-identical for an ARBITRARY projection — not merely at init. That is what
    makes the arm interpretable: a policy change cannot confound the critic result.
  * **The injected row IS `reduce_pair_in_all` under the R1 rung, exactly** — the same function
    Phase B's switch cell uses, not a second spelling of it.
  * **α is R1 by ORDERING, not by fallback.** `value_cls` pools BEFORE the α/β heads are scored, so
    the route cannot read the publication even when `--opp-intent` is on — and the test proves the
    rows are byte-identical across that flag rather than trusting the comment.
  * **The critic's gradient reaches the projection** under BOTH parameterizations (the v89
    dead-tail bug class; this route is not in the `_value_pooled_routes` seam by design, so it does
    NOT inherit that guard and needs its own).
  * **It stacks with `--value-threat-inject` without either becoming the other.**
  * OFF is byte-identical; ON is identity-at-init on a REAL `MaskablePPO` policy (ledger M1).
  * The v95 version machinery + the delivery-graph edge.

⚠️ The C4 RE-ENTRY CONDITION is a POLICY, not a testable property, so it is not asserted here — it
lives in the flag registry, the CLI help, the module docstring and the CHANGELOG entry: *any α/β-
critic route may be BUILT opt-in but its ENABLING owes the C4-style offline gate first.*
"""
import dataclasses

import gymnasium as gym
import numpy as np
import pytest
import torch

from agents.model.arch_constants import D_MODEL, PAIR_VALUE_ROUTE_DIM, _PAIR_OUTCOME_RAW
from agents.model.features_extractor import Gen3FeaturesExtractor, TEAM_SIZE
from agents.model.model_version import (
    MODEL_CONFIG_VERSION, ModelVersion, ModelVersionError, _migrate_config,
)
from agents.model.pair_outcome import pair_alpha, reduce_pair_in_all
from agents.model.pair_value_route import PairValueInject
from agents.observation.state_encoder import Gen3ObservationEncoder, load_mappings

_BASE_KWARGS = dict(
    attend_unrevealed_opponents=True, move_belief_mode="revealed", move_prior_fusion=True,
    move_latent=True, damage_op=True, damage_outgoing=True, damage_matrices_incoming=True,
    damage_topk_k=6, entity_topk_seats=6,
)
_ON_KWARGS = {**_BASE_KWARGS, "pair_value_route": True}


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


def _capture_rows(fe):
    """Record the rows the route is handed on the next forward, AND whether the α publication even
    existed at that moment — the second is the sharp form of the ordering claim."""
    seen = {}
    orig = fe.cls_pool.pair_value_proj.forward

    def spy(tokens, rows):
        seen["rows"] = rows.detach().clone()
        seen["alpha_logits_at_injection"] = fe.last_alpha_logits
        return orig(tokens, rows)
    fe.cls_pool.pair_value_proj.forward = spy
    return seen


# ------------------------------------------------------------------------------- the module


def test_the_row_width_is_phase_As_row_not_a_number_written_twice():
    assert PAIR_VALUE_ROUTE_DIM == _PAIR_OUTCOME_RAW
    m = PairValueInject(PAIR_VALUE_ROUTE_DIM, D_MODEL)
    assert m.proj.in_features == _PAIR_OUTCOME_RAW and m.proj.out_features == D_MODEL


def test_the_module_is_zero_init_and_adds_exactly_zero():
    m = PairValueInject(_PAIR_OUTCOME_RAW, D_MODEL)
    assert float(m.proj.weight.abs().max()) == 0.0 and float(m.proj.bias.abs().max()) == 0.0
    tok = torch.randn(2, TEAM_SIZE, D_MODEL)
    out = m(tok, torch.rand(2, TEAM_SIZE, _PAIR_OUTCOME_RAW))
    assert torch.equal(out, tok)


def test_the_module_is_one_SHARED_linear_over_j_so_the_map_is_equivariant():
    """A per-slot projection would reintroduce exactly the positional dependence the entity re-home
    deleted. One shared `Linear` ⇒ permuting (tokens, rows) permutes the output."""
    m = PairValueInject(_PAIR_OUTCOME_RAW, D_MODEL)
    torch.nn.init.normal_(m.proj.weight, std=0.5)
    tok = torch.randn(1, TEAM_SIZE, D_MODEL)
    rows = torch.rand(1, TEAM_SIZE, _PAIR_OUTCOME_RAW)
    p = torch.tensor([4, 0, 5, 2, 1, 3])
    assert torch.allclose(m(tok, rows)[:, p], m(tok[:, p], rows[:, p]), atol=1e-6)
    assert [n for n, _ in m.named_parameters()] == ["proj.weight", "proj.bias"]


def test_a_drifted_row_width_fails_loud():
    m = PairValueInject(_PAIR_OUTCOME_RAW, D_MODEL)
    with pytest.raises(ValueError, match="drifted"):
        m(torch.randn(1, TEAM_SIZE, D_MODEL), torch.rand(1, TEAM_SIZE, _PAIR_OUTCOME_RAW - 1))
    with pytest.raises(ValueError, match="shape mismatch"):
        m(torch.randn(1, TEAM_SIZE, D_MODEL), torch.rand(1, 5, _PAIR_OUTCOME_RAW))


def test_the_pool_refuses_to_silently_skip_when_rows_are_missing():
    """A silent skip would make the flag a no-op that still passes every shape test — which is
    indistinguishable from a null RESULT, the failure this whole substrate exists to prevent."""
    from agents.model.pools import CLSPool
    mappings = load_mappings()
    layout = Gen3ObservationEncoder(mappings).get_layout()
    pool = CLSPool(layout, pair_value_row_dim=_PAIR_OUTCOME_RAW)
    assert pool.pair_value_proj is not None
    with pytest.raises(ValueError, match="pair_value_route is built"):
        pool.forward(torch.randn(1, TEAM_SIZE, D_MODEL), torch.randn(1, TEAM_SIZE, D_MODEL),
                     _DummyCtx(), None, None)


class _DummyCtx:
    batch_size = 1
    device = torch.device("cpu")
    fainted_mask_ours = torch.zeros(1, TEAM_SIZE, dtype=torch.bool)
    fainted_mask_opp = torch.zeros(1, TEAM_SIZE, dtype=torch.bool)
    all_fainted = torch.zeros(1, 2 * TEAM_SIZE, dtype=torch.bool)
    our_active_idx = torch.zeros(1, dtype=torch.long)


# ------------------------------------------------------------------- the row, and where α is from


def test_the_injected_row_is_reduce_pair_in_all_under_the_R1_rung_EXACTLY():
    """Not "approximately the same reduction" — the SAME function Phase B's switch cell calls, on
    the SAME α ladder. Two spellings of one distribution is how they drift apart."""
    fe, layout = _build(**_ON_KWARGS)
    seen = _capture_rows(fe)
    fe(_obs(layout))
    op = fe.damage_op
    want = reduce_pair_in_all(
        pair_alpha(None, op.last_topk_w, op.last_pair_seat_live),
        op.last_pair_in, op.last_pair_gate)
    assert seen["rows"].shape == (3, TEAM_SIZE, _PAIR_OUTCOME_RAW)
    assert torch.equal(seen["rows"], want)


def test_alpha_is_the_R1_rung_even_when_the_INTENT_HEAD_IS_ON():
    """⚠️ ORDERING, not preference. `value_cls` pools at T2 BEFORE the α/β heads are scored, so the
    publication does not exist when this route runs — and that is a fact about the phase chain, not
    a fallback that fires when a head is absent. Asserted rather than commented: the injected rows
    must be BYTE-IDENTICAL to the R1 reduction with `--opp-intent` on, which they cannot be if the
    publication ever leaks in."""
    fe, layout = _build(**_ON_KWARGS, opp_intent=True)
    assert fe.alpha_head is not None
    seen = _capture_rows(fe)
    fe(_obs(layout))
    # THE sharp form, and the one that would break if the injection were ever moved downstream of
    # the heads: at the moment PV runs, the publication does not exist. (By the END of the same
    # forward it does — asserted below — so this is a statement about ORDER, not about the config.)
    assert seen["alpha_logits_at_injection"] is None, (
        "the α publication already existed when PV ran — the injection has moved downstream of the "
        "α head, which changes which distribution this route CAN take and invalidates the "
        "DELIVERY-vs-DISTRIBUTION split the arm is built on")
    assert fe.last_alpha_logits is not None
    op = fe.damage_op
    r1 = reduce_pair_in_all(pair_alpha(None, op.last_topk_w, op.last_pair_seat_live),
                            op.last_pair_in, op.last_pair_gate)
    assert torch.equal(seen["rows"], r1)
    # ...and the publication, which DOES exist by the end of the forward, is a different object —
    # so this is a live distinction rather than a vacuous one.
    published = pair_alpha(fe.last_alpha_logits, op.last_topk_w, op.last_pair_seat_live)
    plain = pair_alpha(None, op.last_topk_w, op.last_pair_seat_live)
    assert not torch.equal(published, plain), (
        "the two rungs happened to coincide — the ordering claim is untested on this seed")


def test_the_row_carries_the_STATUS_currency_the_critic_has_no_other_route_to():
    """The whole point of PV over `--value-threat-inject`: v64 sends a 13-wide DAMAGE summary; this
    sends Phase A's unified row, whose last eight coordinates are the six status identities,
    `neutralization` and `tempo_cost`. Incoming status reaches vf only as the `s3` edge family's
    softmax-normalised RATIO otherwise."""
    from agents.model.pair_outcome import PAIR_OUTCOME_COORDS
    from agents.model.value_threat_inject import value_threat_inject_dim
    assert PAIR_OUTCOME_COORDS[6:] == (
        "p_par", "p_brn", "p_frz", "p_slp", "p_psn", "p_tox", "neutralization", "tempo_cost")
    assert PAIR_VALUE_ROUTE_DIM != value_threat_inject_dim(), (
        "if the two rows ever became the same width, check they have not become the same object")


# --------------------------------------------------------------------- vf-ONLY, at ANY weight


def test_pi_is_bit_identical_at_an_ARBITRARY_weight_not_merely_at_init():
    """V1, the property the injection site was CHOSEN for. `our_cls`, `our_active_refined` and the
    pointer head all read the unaugmented tokens, so this is structural — a large random projection
    must move vf and leave pi byte-for-byte."""
    fe, layout = _build(**_ON_KWARGS)
    obs = _obs(layout)
    with torch.no_grad():
        pi0, vf0 = fe(obs)
        ptr0 = fe.last_pointer_inputs.team_tokens.clone()
        fe.cls_pool.pair_value_proj.proj.weight.normal_(std=3.0)
        fe.cls_pool.pair_value_proj.proj.bias.normal_(std=3.0)
        pi1, vf1 = fe(obs)
        ptr1 = fe.last_pointer_inputs.team_tokens.clone()
    assert torch.equal(pi0, pi1), "PV leaked into the POLICY half"
    assert torch.equal(ptr0, ptr1), "PV leaked into the pointer head's team tokens"
    assert not torch.equal(vf0, vf1), "PV at a large random weight did not move vf"


def test_the_critic_gradient_reaches_the_projection_under_both_parameterizations():
    """The v89 dead-tail bug class. This route is deliberately NOT in the `_value_pooled_routes`
    seam (a post-pool additive route would have to collapse the J axis), so it does not inherit
    that seam's gradient guard and gets its own — under the scalar critic AND under the dist head,
    which is the parameterization that made the vf-tail concat structurally dead."""
    for dist in (False, True):
        kw = dict(_ON_KWARGS)
        if dist:
            kw.update(value_dist_mode="shaping", value_dist_bins=51,
                      value_dist_vmin=-12.0, value_dist_vmax=12.0)
        fe, layout = _build(**kw)
        fe.train()
        pi, vf = fe(_obs(layout))
        loss = fe.last_value_dist_logits.sum() if dist else vf.sum()
        loss.backward()
        g = fe.cls_pool.pair_value_proj.proj.weight.grad
        assert g is not None and float(g.abs().max()) > 0.0, (
            f"PV received NO gradient from the {'dist-head' if dist else 'scalar'} critic — it is "
            "structurally disconnected")


def test_it_stacks_with_value_threat_inject_without_either_becoming_the_other():
    """Two injections, one local copy, each with its own zero-init delta — so enabling one says
    nothing about the other and an ablation can attribute."""
    fe, layout = _build(**_ON_KWARGS, value_threat_inject=True)
    assert fe.cls_pool.value_threat_proj is not None and fe.cls_pool.pair_value_proj is not None
    assert fe.cls_pool.value_threat_proj.extra_dim != fe.cls_pool.pair_value_proj.row_dim
    obs = _obs(layout)
    with torch.no_grad():
        pi0, vf0 = fe(obs)
        fe.cls_pool.pair_value_proj.proj.weight.normal_(std=2.0)
        pi1, vf1 = fe(obs)
        fe.cls_pool.value_threat_proj.proj.weight.normal_(std=2.0)
        pi2, vf2 = fe(obs)
    assert torch.equal(pi0, pi1) and torch.equal(pi1, pi2)
    assert not torch.equal(vf0, vf1) and not torch.equal(vf1, vf2)


# ---------------------------------------------------------------------- OFF / ON, and the widths


def test_off_builds_no_module_and_moves_no_width_anywhere():
    """PV injects ADDITIVELY into a token, so — unlike every pointer-cell flag in this substrate —
    NOTHING shape-based can see the difference except the extra state_dict key. That is exactly why
    the version gate carries this one."""
    fe_off, _ = _build(**_BASE_KWARGS)
    fe_on, _ = _build(**_ON_KWARGS)
    assert fe_off.cls_pool.pair_value_proj is None
    assert fe_on.cls_pool.pair_value_proj is not None
    assert not any("pair_value_proj" in k for k in fe_off.state_dict())
    assert fe_on.projection.in_features == fe_off.projection.in_features
    assert fe_on.value_projection.in_features == fe_off.value_projection.in_features
    assert fe_on.pointer_switch_cell_dim == fe_off.pointer_switch_cell_dim
    assert fe_on.pointer_move_cell_dim == fe_off.pointer_move_cell_dim


def test_off_is_byte_identical():
    fe_off, layout = _build(**_BASE_KWARGS)
    obs = _obs(layout)
    pi, vf = fe_off(obs)
    pi2, vf2 = fe_off(obs)
    assert torch.equal(pi, pi2) and torch.equal(vf, vf2)
    assert fe_off.damage_op.stash_pair_outcome is False


def test_on_at_init_leaves_value_pooled_bit_identical():
    """Zero-init ⇒ ON adds exactly 0 to every token, so the critic's read surface is bit-identical
    to the same network with the route removed.

    The comparison is against THIS build with the module detached, not against a separately
    constructed OFF build: constructing the extra `Linear` draws from the seeded RNG, so two builds
    differ in every OTHER weight and a cross-build equality would be testing the RNG."""
    fe, layout = _build(**_ON_KWARGS)
    obs = _obs(layout)
    with torch.no_grad():
        fe(obs)
        pooled_on = fe.last_value_pooled.clone()
        pi_on, vf_on = fe(obs)
        held, fe.cls_pool.pair_value_proj = fe.cls_pool.pair_value_proj, None
        fe(obs)
        pooled_detached = fe.last_value_pooled.clone()
        pi_off, vf_off = fe(obs)
        fe.cls_pool.pair_value_proj = held
    assert torch.equal(pooled_on, pooled_detached)
    assert torch.equal(pi_on, pi_off) and torch.equal(vf_on, vf_off)


def test_requires_damage_op():
    with pytest.raises(ValueError, match="damage_op"):
        _build(**{**_ON_KWARGS, "damage_op": False, "damage_outgoing": False,
                  "damage_matrices_incoming": False, "damage_topk_k": 0})


def test_missing_topk_stash_fails_loud_rather_than_contributing_zeros():
    with pytest.raises((RuntimeError, ValueError)):
        fe, layout = _build(**{**_ON_KWARGS, "damage_matrices_incoming": False,
                               "damage_topk_k": 0})
        fe(_obs(layout))


# ------------------------------------------------------- identity-at-init on a REAL policy (M1)


def test_zero_init_survives_a_real_MaskablePPO_build():
    from agents.model.identity_init_test import _build_real_policy
    model, _ = _build_real_policy(damage_matrices_incoming=True, damage_topk_k=6,
                                  entity_topk_seats=6, move_latent=True,
                                  pair_value_route=True)
    fe = model.policy.features_extractor
    assert fe.cls_pool.pair_value_proj is not None
    assert float(fe.cls_pool.pair_value_proj.proj.weight.abs().max()) == 0.0
    assert float(fe.cls_pool.pair_value_proj.proj.bias.abs().max()) == 0.0
    assert "cls_pool.pair_value_proj.proj" in fe._identity_init_zeroed


# ------------------------------------------------------------------------------ version machinery


def test_a_pre_floor_config_is_REFUSED_not_defaulted():
    """The critic-route deletion wave bumped ARCH_SIGNATURE, so MIGRATION_FLOOR rose to 96
    and this v94 config is now refused outright rather than walked through the v95 branch
    that defaults `pair_value_route`. That is the floor's stated purpose ("refuses pre-floor
    configs outright instead of walking dead branches"), and the assertion follows the
    BEHAVIOUR: what must hold is that a stale config is rejected with a diagnosis, not that
    an unreachable branch still defaults a field."""
    with pytest.raises(ModelVersionError, match="PRE-GENERATION|floor"):
        _migrate_config({"config_version": 94})
    assert MODEL_CONFIG_VERSION >= 95


def test_check_compatible_gates_the_flag():
    mappings = load_mappings()
    layout = Gen3ObservationEncoder(mappings).get_layout()
    b = ModelVersion.from_layout_and_policy_kwargs(layout, {"features_extractor_kwargs": {}})
    a = dataclasses.replace(b, pair_value_route=True)
    with pytest.raises(ModelVersionError, match="pair_value_route"):
        a.check_compatible(b)


def test_the_flag_round_trips_through_the_snapshot_kwargs():
    from agents.model.snapshot import current_model_version
    v = current_model_version(load_mappings(), pair_value_route=True)
    assert v.pair_value_route is True
    assert current_model_version(load_mappings()).pair_value_route is False


# -------------------------------------------------------------------------------- delivery graph


def test_the_route_is_drawn_when_it_is_ON():
    import json
    import os
    import tempfile

    from agents.model.delivery_graph import _DEFAULT_CONFIG, build_graph
    cfg = json.load(open(_DEFAULT_CONFIG))
    cfg["pair_value_route"] = True
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "cfg.json")
        with open(path, "w") as fh:
            json.dump(cfg, fh)
        graph = build_graph(path)
    drawn = [e for e in graph["edges"] if "pair_value_proj" in (e.get("via") or "")]
    assert len(drawn) == 1 and drawn[0]["dst"] == "vf_projection"
    assert drawn[0]["type"] == "content" and drawn[0]["zero_init"] and drawn[0]["pooled"]
    assert drawn[0]["width"] == PAIR_VALUE_ROUTE_DIM
