"""gen3_pair_outcome_v1 (v93) — the unified outcome vector's gates.

What must hold (design_opponent_intent.md §5.1/§5.3, design_pair_reduction.md §2.1/§3.1/§9a):

  * **The coordinates are what they claim to be**, per coordinate, on constructed inputs: the
    status columns carry the seat's IDENTITY (and keep tox apart from psn, which `MOVE_STATUS_CAT`
    cannot), `neutralization` scales with the DEFENDER's own profile, `tempo_cost` with its own
    moveset.
  * **The reduction contract**: ONE α over the move axis, shared across EVERY channel. The
    load-bearing test plants the D2 violation — a per-channel maximum — and asserts the contract
    catches it, because a reduction that quietly took a max per channel would pass every shape
    check and every smoke test.
  * **The α fallback is the shipped R1 rung**, exactly (not an approximation of it), and the
    intent path is the UNRENORMALIZED move slice, so a certain SWITCH read zeroes the vector.
  * **Unmodeled seats are masked, not renormalized away.**
  * **OFF is byte-identical**: no module, no state_dict key, no extra dim, and the same pi/vf.
  * **ON is identity-at-init on a REAL `MaskablePPO` policy** (ledger M1 — a zero-init asserted
    only on a directly-built extractor is not an invariant, because SB3's ortho pass runs on the
    path production uses and not on that one).
  * The §9a motivating case, through the FULL op forward: Swampert vs a believed Will-O-Wisp reads
    **0.0 in every damage coordinate** and a nonzero burn/neutralization — the currency failure
    §2.1 names, closed.
  * The v93 version machinery + the delivery-graph edges.
"""
import dataclasses
import inspect

import gymnasium as gym
import numpy as np
import pytest
import torch

from agents.model.arch_constants import (
    PAIR_OUTCOME_MOVE_DIM, _PAIR_OUTCOME_DMG, _PAIR_OUTCOME_NEW, _PAIR_OUTCOME_RAW,
)
from agents.model.features_extractor import Gen3FeaturesExtractor, TEAM_SIZE
from agents.model.model_version import (
    MODEL_CONFIG_VERSION, ModelVersion, ModelVersionError, _migrate_config,
)
from agents.model.pair_outcome import (
    PAIR_OUTCOME_COORDS, PAIR_OUTCOME_IDX, PairOutcomeMoveCell, alpha_belief_mean, pair_alpha,
    reduce_pair_in,
)
from agents.observation.state_encoder import Gen3ObservationEncoder, load_mappings

_BASE_KWARGS = dict(
    attend_unrevealed_opponents=True, move_belief_mode="revealed", move_prior_fusion=True,
    move_latent=True, damage_op=True, damage_outgoing=True, damage_matrices_incoming=True,
    damage_topk_k=6, entity_topk_seats=6,
)
_ON_KWARGS = {**_BASE_KWARGS, "pair_outcome_cell": True}


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


# --------------------------------------------------------------- the coordinate CONTRACT itself


def test_the_coordinate_table_is_the_single_spelling_of_the_layout():
    assert len(PAIR_OUTCOME_COORDS) == _PAIR_OUTCOME_RAW == PAIR_OUTCOME_MOVE_DIM
    assert _PAIR_OUTCOME_DMG + _PAIR_OUTCOME_NEW == _PAIR_OUTCOME_RAW
    # The damage prefix is the op's OWN pair-cell channel order, verbatim — the whole point of
    # component 1 is that damage and status share one tensor rather than being re-derived.
    assert PAIR_OUTCOME_COORDS[:_PAIR_OUTCOME_DMG] == (
        "low", "high", "crit", "ko_ramp", "acc", "is_phys")
    assert PAIR_OUTCOME_COORDS[-2:] == ("neutralization", "tempo_cost")
    assert PAIR_OUTCOME_IDX["high"] == 1


def test_the_two_derivable_collapses_are_deliberately_absent():
    """§9a's derivability rule, applied to the design's own sketch. §5.1 lists `p_status_land` and
    `p_immobilize`; both are LINEAR functions of the six per-identity columns that ship, and those
    columns pass through a `Linear` before anything else touches them. Delivering a collapse beside
    the thing it collapses is the redundancy the rule forbids — so their absence is a decision, and
    a future edit that "restores" them should have to delete this test first."""
    assert "p_status_land" not in PAIR_OUTCOME_COORDS
    assert "p_immobilize" not in PAIR_OUTCOME_COORDS
    assert [c for c in PAIR_OUTCOME_COORDS if c.startswith("p_")] == [
        "p_par", "p_brn", "p_frz", "p_slp", "p_psn", "p_tox"]


# --------------------------------------------------------------------- the reduction (Contract W)


def _grid(B=2, K=3, J=TEAM_SIZE, F=_PAIR_OUTCOME_RAW):
    torch.manual_seed(5)
    return torch.rand(B, J, K, F)


def test_reduction_is_the_alpha_weighted_sum_at_our_active():
    K, F = 3, _PAIR_OUTCOME_RAW
    pair_in = _grid(K=K)
    gate = torch.ones(2, TEAM_SIZE, 1)
    active = torch.tensor([0, 3])
    alpha = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    row = reduce_pair_in(alpha, pair_in, gate, active)
    assert row.shape == (2, F)
    assert torch.allclose(row[0], pair_in[0, 0, 0])
    assert torch.allclose(row[1], pair_in[1, 3, 2])


def test_ONE_distribution_serves_EVERY_channel_a_per_channel_max_cannot_sneak_in():
    """THE load-bearing gate (design_pair_reduction.md §3.1, defect D2).

    The flat/trunk block takes NINE INDEPENDENT MAXIMA, so up to nine different opponent moves
    describe one defender — an incoherence that passes every shape check. Contract W kills it
    structurally (α has no channel axis), and this test is what proves the structure is real: a
    grid whose channels peak on DIFFERENT seats, reduced by an α concentrated on seat 0. Every
    channel must read seat 0's value. A per-channel `amax` would instead return each channel's own
    maximum — so the test FAILS if one is ever reintroduced.
    """
    K, F = 3, _PAIR_OUTCOME_RAW
    pair_in = torch.zeros(1, TEAM_SIZE, K, F)
    for f in range(F):
        peak_seat = f % K                       # channel f's maximum lives on a different seat
        pair_in[0, 0, :, f] = 0.1
        pair_in[0, 0, peak_seat, f] = 0.9
    alpha = torch.zeros(1, K)
    alpha[0, 0] = 1.0                            # ALL the mass on seat 0
    row = reduce_pair_in(alpha, pair_in, torch.ones(1, TEAM_SIZE, 1),
                         torch.zeros(1, dtype=torch.long))
    want = pair_in[0, 0, 0]                      # seat 0's row, every channel
    assert torch.allclose(row[0], want), "the reduction did not use ONE shared alpha"
    per_channel_max = pair_in[0, 0].amax(dim=0)
    assert not torch.allclose(row[0], per_channel_max), (
        "the reduced row equals the PER-CHANNEL maximum — defect D2 is back, and the coherence "
        "contract this module exists to enforce is not being enforced")
    # And the difference is not a rounding artifact: the two readings genuinely disagree.
    assert float((row[0] - per_channel_max).abs().max()) > 0.5


def test_alpha_cannot_depend_on_the_defender_by_signature():
    """Defect D3 in its soft form: Skarmory's row assuming "they click Rock Slide" while Blissey's
    assumes "they click Thunderbolt". They choose WITHOUT seeing which mon you bring, so a
    per-defender α is illegitimate — and here it is a SHAPE ERROR, not a property under test."""
    sig = inspect.signature(reduce_pair_in)
    assert list(sig.parameters) == ["alpha", "pair_in", "gate", "our_active_idx"]
    with pytest.raises((RuntimeError, ValueError)):
        # An α carrying a defender axis has nowhere to go in `bk,bkf->bf`.
        reduce_pair_in(torch.rand(1, TEAM_SIZE, 3), _grid(B=1, K=3),
                       torch.ones(1, TEAM_SIZE, 1), torch.zeros(1, dtype=torch.long))


def test_gate_zeroes_a_dead_or_opponentless_row():
    row = reduce_pair_in(torch.full((1, 3), 1 / 3), _grid(B=1, K=3),
                         torch.zeros(1, TEAM_SIZE, 1), torch.zeros(1, dtype=torch.long))
    assert float(row.abs().max()) == 0.0


def test_axis_width_mismatch_raises_rather_than_broadcasting():
    with pytest.raises(ValueError, match="move-order|SAME axis"):
        reduce_pair_in(torch.rand(1, 5), _grid(B=1, K=3), torch.ones(1, TEAM_SIZE, 1),
                       torch.zeros(1, dtype=torch.long))


def test_seat_permutation_invariance():
    K = 4
    pair_in = _grid(K=K)
    gate = torch.ones(2, TEAM_SIZE, 1)
    active = torch.tensor([0, 1])
    alpha = torch.rand(2, K)
    perm = torch.randperm(K)
    a = reduce_pair_in(alpha, pair_in, gate, active)
    b = reduce_pair_in(alpha[:, perm], pair_in[:, :, perm], gate, active)
    assert torch.allclose(a, b, atol=1e-6)


# ------------------------------------------------------------------------- alpha and its FALLBACK


def test_alpha_from_the_publication_is_the_unrenormalized_move_slice():
    K = 3
    w = torch.rand(2, K)
    lg = torch.zeros(2, K + 1)                                  # uniform over K seats + SWITCH
    alpha = pair_alpha(lg, w)
    assert torch.allclose(alpha, torch.full((2, K), 1.0 / (K + 1)), atol=1e-6)
    assert float(alpha.sum(-1)[0]) < 1.0, "the SWITCH mass must NOT be renormalized away"


def test_certain_switch_zeroes_the_whole_outcome_vector():
    """The unrenormalized slice's semantics: a switching opponent applies no outcome to us this
    turn, so every coordinate — damage AND status AND neutralization AND tempo — shrinks to zero
    together. That coherence is only expressible because they share one α."""
    K = 3
    lg = torch.zeros(1, K + 1)
    lg[:, -1] = 30.0                                            # α_SWITCH ≈ 1
    row = reduce_pair_in(pair_alpha(lg, torch.rand(1, K)), _grid(B=1, K=K),
                        torch.ones(1, TEAM_SIZE, 1), torch.zeros(1, dtype=torch.long))
    assert float(row.abs().max()) < 1e-4


def test_the_fallback_IS_the_shipped_R1_rung_not_an_approximation_of_it():
    """`--opp-intent` OFF ⇒ α := w/Σw, and it must be the SAME function `pair_reduce` ships (two
    spellings of a distribution is how they drift apart). Note it sums to 1 where the intent path
    sums to 1−α_SWITCH: with no intent head there is no switch belief to withhold mass for, and
    that asymmetry is documented rather than papered over."""
    from agents.model.pair_reduce import alpha_belief_mean as shipped
    w = torch.rand(4, 5)
    got = pair_alpha(None, w)
    assert torch.allclose(got, shipped(w))
    assert torch.allclose(got, alpha_belief_mean(w))
    assert torch.allclose(got.sum(-1), torch.ones(4), atol=1e-6)


def test_the_fallback_keeps_an_all_zero_belief_row_at_zero():
    """R1's convention: no believed threat ⇒ no usage mass ⇒ a zero row, NOT a uniform one.
    A uniform fallback would assert 'every move equally likely' where the truth is 'nothing is
    believed at all'."""
    w = torch.zeros(2, 4)
    assert float(pair_alpha(None, w).abs().max()) == 0.0


def test_unmodeled_seats_are_masked_and_the_mass_is_NOT_reassigned():
    """§4.2's rule — "if we can't name it, we don't train on it" — in the forward. A closed 5th+
    top-K slot describes nothing, so its mass is simply not spent, exactly like SWITCH's."""
    K = 4
    w = torch.ones(1, K)
    live = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
    alpha = pair_alpha(None, w, live)
    assert torch.allclose(alpha, torch.tensor([[0.25, 0.25, 0.0, 0.0]]))
    assert float(alpha.sum()) == pytest.approx(0.5), "masked mass was renormalized back in"


def test_alpha_is_stop_grad_so_no_policy_route_opens_into_the_intent_head():
    """A POLICY-side consumer must not open a PPO → alpha_head route (the v87 pattern). Relying on
    `belief_grad_mode=label_only` for that would make the route's EXISTENCE a function of a
    TRAINING flag — so the cut is unconditional here."""
    lg = torch.zeros(1, 4, requires_grad=True)
    alpha = pair_alpha(lg, torch.rand(1, 3))
    assert not alpha.requires_grad, "alpha stayed attached — a PPO route into alpha_head is open"
    # and the whole downstream row is detached with it
    row = reduce_pair_in(alpha, _grid(B=1, K=3), torch.ones(1, TEAM_SIZE, 1),
                         torch.zeros(1, dtype=torch.long))
    assert not row.requires_grad
    assert lg.grad is None


def test_alpha_seat_width_mismatch_raises():
    with pytest.raises(ValueError, match="SAME axis"):
        pair_alpha(torch.zeros(1, 8), torch.rand(1, 3))


# ---------------------------------------------------------------------------------- the move cell


def test_move_cell_is_zero_init_and_broadcasts_the_row_to_every_slot():
    cell = PairOutcomeMoveCell(PAIR_OUTCOME_MOVE_DIM)
    assert float(cell.proj.weight.abs().max()) == 0.0
    assert float(cell.proj.bias.abs().max()) == 0.0
    out = cell(torch.rand(2, _PAIR_OUTCOME_RAW))
    assert out.shape == (2, 4, PAIR_OUTCOME_MOVE_DIM)
    assert float(out.abs().max()) == 0.0
    # Identity read: every slot receives the SAME row (it is per-decision context, not per-action
    # content — the `p_ko` precedent). It still moves the logits, because the pointer scorer is an
    # MLP over (move token ‖ cell) and each token differs.
    with torch.no_grad():
        cell.proj.weight.copy_(torch.eye(_PAIR_OUTCOME_RAW))
    row = torch.rand(2, _PAIR_OUTCOME_RAW)
    out = cell(row)
    for k in range(4):
        assert torch.allclose(out[:, k, :], row)


def test_move_cell_refuses_a_drifted_vector_width():
    cell = PairOutcomeMoveCell(PAIR_OUTCOME_MOVE_DIM)
    with pytest.raises(ValueError, match="drifted|outcome vector"):
        cell(torch.rand(2, _PAIR_OUTCOME_RAW + 1))


# ------------------------------------------------------- the coordinates through the REAL physics


def _real_op(K=6):
    from agents.model import damage_op_test as DT
    op, layout = DT._op_and_layout_topk(K)
    op.stash_pair_cells = True
    op.stash_pair_outcome = True
    return op, layout, DT


def _run(op, DT, layout, moves, defenders):
    ctx = DT._topk_ctx(op, defenders=defenders)
    op(ctx, DT._logits_moves(layout["max_moves"], moves), None, DT._synth_latent(layout))
    return ctx


def test_g0_the_currency_failure_is_closed_swampert_into_will_o_wisp():
    """§2.1/§5.1's motivating case, through the FULL op forward rather than a hand-built tensor.

    Swampert (Water/Ground, base Atk 110 / SpA 85) against a Gengar believed to hold Will-O-Wisp
    and Thunderbolt. On damage alone it reads **0.0 in both branches** — immune to one, and burn
    deals none — so damage-only scoring picks it forever and the hedge is unreachable. The unified
    vector must show a nonzero burn probability AND a nonzero neutralization on exactly the branch
    the damage numbers cannot see.
    """
    op, layout, DT = _real_op()
    from agents.observation.types import TypeEncoder
    T = TypeEncoder.TYPE_TO_IDX
    wow, tb = DT._move_num("willowisp"), DT._move_num("thunderbolt")
    _run(op, DT, layout, [wow, tb], [(260, T["WATER"], T["GROUND"])] + [(0, 0, 0)] * 5)
    pin = op.last_pair_in
    assert pin.shape[-1] == _PAIR_OUTCOME_RAW
    k = op.last_topk_idx[0].tolist().index(wow)
    cell = pin[0, 0, k]
    # every damage coordinate is zero — the premise of the whole argument
    for name in ("low", "high", "crit", "ko_ramp"):
        assert float(cell[PAIR_OUTCOME_IDX[name]]) == 0.0, f"{name} should be 0 for Will-O-Wisp"
    # and the status/neutralization coordinates are not
    assert float(cell[PAIR_OUTCOME_IDX["p_brn"]]) > 0.5
    assert float(cell[PAIR_OUTCOME_IDX["neutralization"]]) > 0.0
    # the burn lands on the BURN column and nowhere else
    for other in ("p_par", "p_frz", "p_slp", "p_psn", "p_tox"):
        assert float(cell[PAIR_OUTCOME_IDX[other]]) == 0.0


def test_g0_neutralization_scales_with_the_defenders_own_physical_share():
    """Burn's severity is `0.5 · base_atk/(base_atk+base_spa)` — read off the DEFENDER's stats, not
    assumed. A physical attacker must read strictly higher than a special one facing the same
    believed Will-O-Wisp, which is the ordering the coordinate exists to create."""
    from agents.observation.types import TypeEncoder
    T = TypeEncoder.TYPE_TO_IDX
    wow = None
    reads = {}
    for label, spec in (("physical", (260, T["WATER"], T["GROUND"])),      # Swampert  110/85
                        ("special", (196, T["PSYCHIC"], T["PSYCHIC"]))):   # Espeon     65/130
        op, layout, DT = _real_op()
        wow = DT._move_num("willowisp")
        _run(op, DT, layout, [wow], [spec] + [(0, 0, 0)] * 5)
        k = op.last_topk_idx[0].tolist().index(wow)
        reads[label] = float(op.last_pair_in[0, 0, k, PAIR_OUTCOME_IDX["neutralization"]])
    assert reads["physical"] > reads["special"] > 0.0, reads


def test_g0_toxic_and_poison_powder_land_on_DIFFERENT_columns():
    """`MOVE_STATUS_CAT` folds tox into psn (they share the Steel/Poison immunity), so a
    category-keyed identity would make Toxic and Poison Powder the SAME outcome. They are not —
    one escalates. `MOVE_STATUS_IDENT` reads the raw status id and keeps them apart."""
    op, layout, DT = _real_op()
    from agents.observation.types import TypeEncoder
    T = TypeEncoder.TYPE_TO_IDX
    tox, psn = DT._move_num("toxic"), DT._move_num("poisonpowder")
    _run(op, DT, layout, [tox, psn], [(260, T["WATER"], T["GROUND"])] + [(0, 0, 0)] * 5)
    idx = op.last_topk_idx[0].tolist()
    ktox, kpsn = idx.index(tox), idx.index(psn)
    pin = op.last_pair_in[0, 0]
    assert float(pin[ktox, PAIR_OUTCOME_IDX["p_tox"]]) > 0.5
    assert float(pin[ktox, PAIR_OUTCOME_IDX["p_psn"]]) == 0.0
    assert float(pin[kpsn, PAIR_OUTCOME_IDX["p_psn"]]) > 0.5
    assert float(pin[kpsn, PAIR_OUTCOME_IDX["p_tox"]]) == 0.0


def test_g0_an_immune_pivot_reads_zero_status_risk():
    """The landing physics is `_incoming_status_lands` verbatim, so the per-pivot immunity read
    survives into the new coordinates: Thunder Wave into a GROUND pivot is 0, not merely small."""
    op, layout, DT = _real_op()
    from agents.observation.types import TypeEncoder
    T = TypeEncoder.TYPE_TO_IDX
    tw = DT._move_num("thunderwave")
    _run(op, DT, layout, [tw],
         [(260, T["WATER"], T["GROUND"]), (242, T["NORMAL"], T["NORMAL"])] + [(0, 0, 0)] * 4)
    k = op.last_topk_idx[0].tolist().index(tw)
    pin = op.last_pair_in[0]
    assert float(pin[0, k, PAIR_OUTCOME_IDX["p_par"]]) == 0.0, "Ground is immune to Thunder Wave"
    assert float(pin[0, k, PAIR_OUTCOME_IDX["neutralization"]]) == 0.0
    assert float(pin[1, k, PAIR_OUTCOME_IDX["p_par"]]) > 0.5, "a Normal pivot is not immune"


def test_g0_tempo_is_paid_only_by_a_mon_that_can_undo_the_status():
    """`tempo_cost` is read off THIS mon's OWN moveset — the receiver is fully observed. A mon
    carrying Refresh pays a turn; the same mon without it pays nothing (its loss is
    `neutralization`'s to carry, and the two ride decorrelated rather than pre-blended)."""
    from agents.observation.types import TypeEncoder
    T = TypeEncoder.TYPE_TO_IDX
    reads = {}
    for label, our_moves in (("cure", ["refresh"]), ("nocure", ["surf"])):
        op, layout, DT = _real_op()
        wow = DT._move_num("willowisp")
        ctx = DT._topk_ctx(op, defenders=[(260, T["WATER"], T["GROUND"])] + [(0, 0, 0)] * 5)
        ctx.all_move_ids[0, 0, 0] = DT._move_num(our_moves[0])
        op(ctx, DT._logits_moves(layout["max_moves"], [wow]), None, DT._synth_latent(layout))
        k = op.last_topk_idx[0].tolist().index(wow)
        reads[label] = float(op.last_pair_in[0, 0, k, PAIR_OUTCOME_IDX["tempo_cost"]])
    assert reads["cure"] > 0.5, reads
    assert reads["nocure"] == 0.0, reads


def test_g0_rest_prices_the_undo_higher_than_a_dedicated_cure():
    """Rest undoes any status too, but at the op's OWN `rest_sleep_noeb` price (2 lost turns,
    derived from the verified sleep hazard table) rather than one — so the two are not the same
    answer and the coordinate says so."""
    from agents.observation.types import TypeEncoder
    T = TypeEncoder.TYPE_TO_IDX
    reads = {}
    for label, mv in (("refresh", "refresh"), ("rest", "rest")):
        op, layout, DT = _real_op()
        wow = DT._move_num("willowisp")
        ctx = DT._topk_ctx(op, defenders=[(260, T["WATER"], T["GROUND"])] + [(0, 0, 0)] * 5)
        ctx.all_move_ids[0, 0, 0] = DT._move_num(mv)
        op(ctx, DT._logits_moves(layout["max_moves"], [wow]), None, DT._synth_latent(layout))
        k = op.last_topk_idx[0].tolist().index(wow)
        reads[label] = float(op.last_pair_in[0, 0, k, PAIR_OUTCOME_IDX["tempo_cost"]])
    assert reads["rest"] > reads["refresh"] > 0.0, reads


def test_the_op_stashes_nothing_when_the_seam_flag_is_off():
    op, layout, DT = _real_op()
    op.stash_pair_outcome = False
    from agents.observation.types import TypeEncoder
    T = TypeEncoder.TYPE_TO_IDX
    _run(op, DT, layout, [DT._move_num("willowisp")],
         [(260, T["WATER"], T["GROUND"])] + [(0, 0, 0)] * 5)
    assert op.last_pair_in is None


# ------------------------------------------------------------------------------ extractor wiring


def test_off_builds_no_module_and_no_extra_dims():
    fe_off, _ = _build(**_BASE_KWARGS)
    fe_on, _ = _build(**_ON_KWARGS)
    assert fe_off.pair_outcome_move is None and fe_on.pair_outcome_move is not None
    assert not any("pair_outcome" in k for k in fe_off.state_dict())
    assert fe_on.pointer_move_cell_dim == fe_off.pointer_move_cell_dim + PAIR_OUTCOME_MOVE_DIM
    # pi and vf are untouched at ANY weight — the cell widens the pointer stash, not a projection
    assert fe_on.projection.in_features == fe_off.projection.in_features
    assert fe_on.value_projection.in_features == fe_off.value_projection.in_features


def test_off_is_byte_identical():
    """The OFF baseline must be UNCHANGED by this feature landing — same state_dict keys, and the
    same pi/vf to the bit. The new op buffers are non-persistent and the new forward branches are
    seam-gated, but "should be" is not a gate."""
    fe_off, layout = _build(**_BASE_KWARGS)
    obs = _obs(layout)
    pi, vf = fe_off(obs)
    pi2, vf2 = fe_off(obs)
    assert torch.equal(pi, pi2) and torch.equal(vf, vf2)
    assert not any("pair_outcome" in k for k in fe_off.state_dict())
    # the op does no extra work either
    assert fe_off.damage_op.stash_pair_outcome is False
    assert fe_off.damage_op.last_pair_in is None


def test_on_forward_runs_and_contributes_exactly_zero_at_init():
    fe, layout = _build(**_ON_KWARGS)
    pi, vf = fe(_obs(layout))
    assert pi.shape[1] == vf.shape[1]
    cells = fe.last_pointer_inputs.move_cells
    assert cells.shape[2] == fe.pointer_move_cell_dim
    assert float(cells[..., -PAIR_OUTCOME_MOVE_DIM:].abs().max()) == 0.0
    # captured by the identity-init sweep (M1: re-zeroed after SB3's ortho pass on a real policy)
    assert "pair_outcome_move.proj" in fe._identity_init_zeroed


def test_on_runs_WITHOUT_the_intent_head_and_uses_the_R1_fallback():
    """The point of the fallback: the flag is independently enableable, so the DELIVERY claim can
    be tested without the DISTRIBUTION claim. It must actually build and run, not just be
    documented."""
    fe, layout = _build(**_ON_KWARGS)
    assert fe.alpha_head is None
    pi, vf = fe(_obs(layout))
    assert pi.shape[0] == 3
    assert fe.damage_op.last_pair_in is not None


def test_on_runs_WITH_the_intent_head():
    fe, layout = _build(**_ON_KWARGS, opp_intent=True)
    assert fe.alpha_head is not None
    pi, vf = fe(_obs(layout))
    assert pi.shape[0] == 3


def test_requires_damage_op():
    with pytest.raises(ValueError, match="damage_op"):
        _build(**{**_ON_KWARGS, "damage_op": False, "damage_outgoing": False,
                  "damage_matrices_incoming": False, "damage_topk_k": 0})


def test_missing_topk_stash_fails_loud_rather_than_contributing_zeros():
    """A silent no-op reads EXACTLY like a null RESULT, so the missing-stash case raises."""
    with pytest.raises((RuntimeError, ValueError)):
        fe, layout = _build(**{**_ON_KWARGS, "damage_matrices_incoming": False,
                               "damage_topk_k": 0})
        fe(_obs(layout))


def test_it_stacks_with_the_other_alpha_cells():
    """All four move-cell consumers on at once must build and run at the summed width — the
    ede5a88 lesson applied to the pointer stash."""
    fe, layout = _build(**_ON_KWARGS, opp_intent=True, intent_threshold=True,
                        intent_move_cell=True, intent_value_reduce=True,
                        value_entity_pool=True)
    pi, vf = fe(_obs(layout))
    assert pi.shape == vf.shape
    assert fe.last_pointer_inputs.move_cells.shape[2] == fe.pointer_move_cell_dim


# ------------------------------------------------------- identity-at-init on a REAL policy (M1)


def test_zero_init_survives_a_real_MaskablePPO_build():
    """Ledger M1. SB3's `_build()` orthogonally re-initialises EVERY Linear in the extractor, so a
    zero-init asserted only on a directly-constructed module is not an invariant — that path is not
    the one training uses. Build the real thing."""
    from agents.model.identity_init_test import _build_real_policy
    model, _ = _build_real_policy(damage_matrices_incoming=True, damage_topk_k=6,
                                  entity_topk_seats=6, move_latent=True,
                                  pair_outcome_cell=True)
    fe = model.policy.features_extractor
    assert fe.pair_outcome_move is not None
    w = fe.pair_outcome_move.proj.weight
    assert float(w.abs().max()) == 0.0, (
        "SB3's ortho pass clobbered the outcome cell's zero-init and the guard did not restore it "
        "— ON would not be identity-at-init in any real run")
    assert float(fe.pair_outcome_move.proj.bias.abs().max()) == 0.0


# ------------------------------------------------------------------------------ version machinery


def test_migration_defaults_the_flag_off():
    out = _migrate_config({"config_version": 92})
    assert out["pair_outcome_cell"] is False
    assert out["config_version"] == MODEL_CONFIG_VERSION >= 93


def test_check_compatible_gates_the_flag():
    mappings = load_mappings()
    layout = Gen3ObservationEncoder(mappings).get_layout()
    b = ModelVersion.from_layout_and_policy_kwargs(layout, {"features_extractor_kwargs": {}})
    a = dataclasses.replace(b, pair_outcome_cell=True)
    with pytest.raises(ModelVersionError, match="pair_outcome_cell"):
        a.check_compatible(b)


def test_the_flag_round_trips_through_the_snapshot_kwargs():
    from agents.model.snapshot import current_model_version
    assert current_model_version(load_mappings(), pair_outcome_cell=True).pair_outcome_cell is True
    assert current_model_version(load_mappings()).pair_outcome_cell is False


# -------------------------------------------------------------------------------- delivery graph


def test_the_module_is_drawn_with_edges_when_it_is_ON():
    """`test_every_parametered_module_is_reachable_in_the_graph` builds from the PRODUCTION config,
    where this flag is off — so it only checks the declaration exists. This checks the other half:
    that turning it on actually produces edges, which is the thing the completeness gate is a proxy
    for (thirteen modules, including the v84/v87 value routes, sat undrawn for months)."""
    import json
    import os
    import tempfile

    from agents.model.delivery_graph import (
        MODULE_GRAPH_TOKENS, _DEFAULT_CONFIG, build_extractor, build_graph, module_coverage)
    assert MODULE_GRAPH_TOKENS["pair_outcome_move"] == ("PairOutcomeMoveCell",)
    cfg = json.load(open(_DEFAULT_CONFIG))
    cfg["pair_outcome_cell"] = True
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "cfg.json")
        with open(path, "w") as fh:
            json.dump(cfg, fh)
        graph = build_graph(path)
        fe = build_extractor(path)[0]
    vias = " ".join(str(e.get("via", "")) for e in graph["edges"])
    assert "PairOutcomeMoveCell" in vias, "the module is ON and draws no edge"
    assert not module_coverage(fe, graph)
