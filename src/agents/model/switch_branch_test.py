"""gen3_switch_branch_v1 (v94) — OA2, the Rapid-Spin spinblock, and Protect's α-conditioning.

`design_conditional_opponent_cells.md` §2 + the owner's two mechanics. Everything here is one
shape — `Σ over their options of (usage probability) × (a property of the option)` — so the gates
are mostly EXACT ARITHMETIC on constructed inputs, plus the invariances §5 gate 5 names.

What must hold:

  * **OA2 contracts over β and keeps the branches DECORRELATED** (§2.3): the switch branch ships
    raw beside the (already-delivered) stay branch, never the collapsed `(1−p)·stay + p·switch`,
    and `α_SWITCH` rides as ONE shared scalar because gen-3 is simultaneous-move (§2.1).
  * **`p_spin_blocked` is the Pursuit mirror**, with `P(arrival is Ghost)` sourced leak-free — the
    revealed types where revealed, the species posterior where not — and gated to the Rapid Spin
    request slot only.
  * **Protect's `attack_mass` is a MASS, not v85's magnitude**, and it is typed from the DATA
    facade so an immune damaging move cannot masquerade as a status move.
  * **No legal switch-in ⇒ the β-contracted coordinates are exactly 0**, not a uniform arrival
    (a uniform belief is a claim; absence is not).
  * **Both invariances**: seat permutation (α's axis) and THEIR-bench permutation (β's axis).
  * α and β are **stop-grad**; the seat-axis mismatch is **fail-loud**.
  * **OFF is byte-identical**; **ON is identity-at-init on a real `MaskablePPO` policy** (M1).
  * The v94 version machinery + the delivery-graph edges.
"""
import dataclasses

import gymnasium as gym
import numpy as np
import pytest
import torch

from agents.model.arch_constants import SWITCH_BRANCH_MOVE_DIM, _SWITCH_BRANCH_RAW
from agents.model.damage_op_layout import (
    _DMG_OMX_CELL, _DMG_OMX_IDX_HIGH, _DMG_OMX_IDX_MULT, _DMG_OMX_IDX_PKO,
)
from agents.model.features_extractor import Gen3FeaturesExtractor, TEAM_SIZE
from agents.model.model_version import (
    MODEL_CONFIG_VERSION, ModelVersion, ModelVersionError, _migrate_config,
)
from agents.model.pair_outcome import rapid_spin_num
from agents.model.switch_branch import (
    SWITCH_BRANCH_COORDS, SWITCH_BRANCH_IDX, SwitchBranchMoveCell, _PROTECT_ONLY,
)
from agents.observation.state_encoder import Gen3ObservationEncoder, load_mappings

_BASE_KWARGS = dict(
    attend_unrevealed_opponents=True, move_belief_mode="revealed", move_prior_fusion=True,
    move_latent=True, damage_op=True, damage_outgoing=True, damage_matrices_incoming=True,
    damage_matrices_outgoing=True, damage_topk_k=6, entity_topk_seats=6, opp_intent=True,
)
_ON_KWARGS = {**_BASE_KWARGS, "switch_branch_cell": True}

_SPIN = rapid_spin_num()
_PROTECT = _PROTECT_ONLY[0]


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


def _move_num(name):
    from agents import gen3_data
    return int(gen3_data.moves.get(name).num)


# --------------------------------------------------------------------- the coordinate CONTRACT


def test_the_coordinate_table_is_the_single_spelling_of_the_layout():
    assert len(SWITCH_BRANCH_COORDS) == _SWITCH_BRANCH_RAW == SWITCH_BRANCH_MOVE_DIM
    assert SWITCH_BRANCH_COORDS[:5] == (
        "e_high_switch", "e_pko_switch", "e_mult_switch", "wasted_ko", "a_switch")
    assert SWITCH_BRANCH_COORDS[5:] == (
        "p_spin_blocked", "spin_value_lost", "protect_attack_mass", "protect_blocked_mass")


def test_the_STAY_branch_is_NOT_duplicated_here():
    """§2.3's rule, as a pinning test. The stay branch already rides the op's own move cell
    (`[low, high, crit, pko]` vs their active); re-shipping it would be redundancy, and shipping
    the COLLAPSED mixture instead of the raw branches is the anti-pattern the section forbids by
    name. The only place `pko_stay` appears is inside the `wasted_ko` interaction."""
    assert not any(c.endswith("_stay") for c in SWITCH_BRANCH_COORDS)
    assert not any("blend" in c or "mix" in c for c in SWITCH_BRANCH_COORDS)
    assert "wasted_ko" in SWITCH_BRANCH_COORDS


def test_endure_is_deliberately_not_in_the_protect_gate():
    """Endure does not BLOCK an attack, it survives one — so "will they attack at all" is not the
    question its value asks (that is the v84 `p_KO` branch). Same carve-out v85 made."""
    assert _PROTECT_ONLY == (182, 197)
    assert _move_num("endure") not in _PROTECT_ONLY


# --------------------------------------------------------------------------- EXACT arithmetic


def _readable():
    cell = SwitchBranchMoveCell(SWITCH_BRANCH_MOVE_DIM)
    with torch.no_grad():
        cell.proj.weight.copy_(torch.eye(_SWITCH_BRANCH_RAW))
        cell.proj.bias.zero_()
    return cell


def _omx(high=None, pko=None, mult=None):
    """`[1,4,6,5]` — `[low, high, crit, pko, type_mult]` per (our request move, their mon)."""
    g = torch.zeros(1, 4, TEAM_SIZE, _DMG_OMX_CELL)
    for k in range(4):
        for j in range(TEAM_SIZE):
            g[0, k, j, _DMG_OMX_IDX_HIGH] = (high or (lambda k, j: 10.0 * k + j))(k, j)
            g[0, k, j, _DMG_OMX_IDX_PKO] = (pko or (lambda k, j: 0.01 * (10 * k + j)))(k, j)
            g[0, k, j, _DMG_OMX_IDX_MULT] = (mult or (lambda k, j: 1.0 + 0.5 * j))(k, j)
    return g


#: α = [0.1, 0.2, 0.3] over three seats + 0.4 on SWITCH. `seat_live` closes seat 2, so the shipped
#: α is [0.1, 0.2, 0.0]: `a_stay` = 0.3 and `a_stay + a_switch = 0.7 < 1` — the masked mass is NOT
#: reassigned, exactly as `pair_alpha_full` documents.
_ALPHA_LOGITS = torch.log(torch.tensor([[0.1, 0.2, 0.3, 0.4]]))
_SEAT_LIVE = torch.tensor([[1.0, 1.0, 0.0]])
#: β puts 0.25 on their slot 1 and 0.75 on their slot 3; every other slot is an illegal target.
_BETA_LOGITS = torch.full((1, TEAM_SIZE), float("-inf"))
_BETA_LOGITS[0, 1] = float(np.log(0.25))
_BETA_LOGITS[0, 3] = float(np.log(0.75))


def _run(cell=None, *, omx=None, p_ghost=None, req=None, protect_odds=0.5, our_hazards=1.0 / 3.0,
         topk_nums=None, alpha_logits=None, beta_logits=None, opp_active=0):
    cell = cell or _readable()
    if p_ghost is None:
        p_ghost = torch.zeros(1, TEAM_SIZE)
        p_ghost[0, 3] = 1.0                       # their slot 3 is a Ghost (β's 0.75 mass)
    if topk_nums is None:                          # seat 0 damaging, seat 1 status, seat 2 damaging
        topk_nums = torch.tensor([[_move_num("tackle"), _move_num("toxic"), _move_num("surf")]])
    if req is None:
        req = torch.tensor([[_SPIN, _PROTECT, 0, 0]])
    return cell(alpha_logits if alpha_logits is not None else _ALPHA_LOGITS,
                beta_logits if beta_logits is not None else _BETA_LOGITS,
                _SEAT_LIVE, topk_nums, omx if omx is not None else _omx(), p_ghost,
                torch.tensor([opp_active]), req, torch.tensor([[protect_odds]]),
                torch.tensor([[our_hazards]]))


def test_oa2_contracts_the_outgoing_grid_over_beta_exactly():
    out = _run()[0]                                                          # [4, RAW]
    g = _omx()[0]
    for k in range(4):
        want_h = 0.25 * float(g[k, 1, _DMG_OMX_IDX_HIGH]) + 0.75 * float(g[k, 3, _DMG_OMX_IDX_HIGH])
        want_p = 0.25 * float(g[k, 1, _DMG_OMX_IDX_PKO]) + 0.75 * float(g[k, 3, _DMG_OMX_IDX_PKO])
        want_m = 0.25 * float(g[k, 1, _DMG_OMX_IDX_MULT]) + 0.75 * float(g[k, 3, _DMG_OMX_IDX_MULT])
        assert float(out[k, SWITCH_BRANCH_IDX["e_high_switch"]]) == pytest.approx(want_h, abs=1e-5)
        assert float(out[k, SWITCH_BRANCH_IDX["e_pko_switch"]]) == pytest.approx(want_p, abs=1e-6)
        assert float(out[k, SWITCH_BRANCH_IDX["e_mult_switch"]]) == pytest.approx(want_m, abs=1e-6)


def test_wasted_ko_is_the_stay_branch_KO_times_the_switch_mass():
    """§2.3's named interaction — *"don't click the KO into the obvious switch."* A product of two
    delivered numbers, and it ships because a thin shared `tanh` scorer does not form the product
    of two of its own inputs (the §9a counter-rule: derivable in principle is not derivable in
    practice)."""
    out = _run()[0]
    g = _omx()[0]
    for k in range(4):
        want = float(g[k, 0, _DMG_OMX_IDX_PKO]) * 0.4      # their active is slot 0; α_SWITCH = 0.4
        assert float(out[k, SWITCH_BRANCH_IDX["wasted_ko"]]) == pytest.approx(want, abs=1e-6)


def test_a_switch_is_ONE_scalar_broadcast_over_every_request_slot():
    """Gen-3 is simultaneous-move: they commit without seeing our move, so P(they switch) cannot be
    per-move (§2.1). A per-slot value here would be a claim about a mechanic the game does not
    have."""
    out = _run()[0]
    col = out[:, SWITCH_BRANCH_IDX["a_switch"]]
    assert torch.allclose(col, torch.full((4,), 0.4), atol=1e-6)


def test_spinblock_is_the_alpha_switch_times_beta_times_slot_property_contraction():
    """`p_spin_blocked = is_ghost(their active)·a_stay + α_SWITCH·Σ_j β_j·P(slot j is Ghost)`.

    Here their active (slot 0) is not a Ghost and their slot 3 is, carrying β's 0.75 — so the value
    is `0·0.3 + 0.4·0.75 = 0.30`, and it appears ONLY on the Rapid Spin request slot."""
    out = _run()[0]
    i, v = SWITCH_BRANCH_IDX["p_spin_blocked"], SWITCH_BRANCH_IDX["spin_value_lost"]
    assert float(out[0, i]) == pytest.approx(0.30, abs=1e-6)
    assert float(out[1:, i].abs().max()) == 0.0, "the coordinate leaked onto a non-spin slot"
    # the currency half: the probability times the stake it destroys (our-side hazards = 1/3)
    assert float(out[0, v]) == pytest.approx(0.30 / 3.0, abs=1e-6)


def test_spinblock_counts_a_ghost_ALREADY_IN_on_the_stay_branch():
    """The other half of the formula, and the one a β-only reading would drop: if their ACTIVE is
    already a Ghost, the spin fails on the branch where they do nothing at all."""
    pg = torch.zeros(1, TEAM_SIZE)
    pg[0, 0] = 1.0                                  # their active IS the Ghost
    out = _run(p_ghost=pg)[0]
    # is_ghost(active)=1 × a_stay 0.3, plus α_SWITCH 0.4 × β-weighted P(ghost arrival)=0
    assert float(out[0, SWITCH_BRANCH_IDX["p_spin_blocked"]]) == pytest.approx(0.30, abs=1e-6)


def test_spinblock_reads_a_FRACTIONAL_ghost_probability_from_an_unrevealed_slot():
    """`P(slot is Ghost)` is a posterior, not a bit — an unrevealed arrival contributes its belief
    mass. A predicate that only accepted 0/1 would read every hidden slot as "definitely not a
    spinblocker", which is the revealed-gating GIGO class §4.1 names."""
    pg = torch.zeros(1, TEAM_SIZE)
    pg[0, 1] = 0.20                                 # unrevealed, 20% Ghost, carries β 0.25
    pg[0, 3] = 0.40                                 # unrevealed, 40% Ghost, carries β 0.75
    out = _run(p_ghost=pg)[0]
    want = 0.4 * (0.25 * 0.20 + 0.75 * 0.40)        # α_SWITCH × Σ β·p_ghost
    assert float(out[0, SWITCH_BRANCH_IDX["p_spin_blocked"]]) == pytest.approx(want, abs=1e-6)


def test_protect_carries_the_attack_MASS_and_its_product_with_the_decay():
    """Seat 0 is Tackle (damaging, α 0.1), seat 1 is Toxic (status, α 0.2), seat 2 is masked. So
    `attack_mass = 0.1` — and NOT 0.3, and NOT `1 − α_SWITCH` = 0.6. The gap between those three
    numbers is the whole content of the coordinate."""
    out = _run()[0]
    m, b = SWITCH_BRANCH_IDX["protect_attack_mass"], SWITCH_BRANCH_IDX["protect_blocked_mass"]
    assert float(out[1, m]) == pytest.approx(0.1, abs=1e-6)
    assert float(out[1, b]) == pytest.approx(0.1 * 0.5, abs=1e-6)     # × p_success 0.5
    assert float(out[[0, 2, 3]][:, m].abs().max()) == 0.0, "leaked onto a non-Protect slot"


def test_attack_mass_is_typed_from_the_DATA_facade_not_from_the_damage_numbers():
    """An immune damaging move reads 0 damage. If `is_damaging` were `high > 0`, a Ground pivot
    facing a believed Thunderbolt would count the Thunderbolt as a STATUS move and Protect's
    attack mass would collapse exactly when the read matters. The op's own status predicate made
    the same choice for the same reason."""
    cell = _readable()
    assert float(cell.damaging_num[_move_num("thunderbolt")]) == 1.0
    assert float(cell.damaging_num[_move_num("toxic")]) == 0.0
    assert float(cell.damaging_num[_move_num("swordsdance")]) == 0.0


def test_no_legal_switch_in_zeroes_the_beta_contracted_coordinates():
    """A softmax over an all-`-inf` row is a UNIFORM arrival distribution, and a uniform belief is
    a claim rather than an absence. Trapped, or five fainted ⇒ these coordinates are exactly 0."""
    out = _run(beta_logits=torch.full((1, TEAM_SIZE), float("-inf")))[0]
    for name in ("e_high_switch", "e_pko_switch", "e_mult_switch"):
        assert float(out[:, SWITCH_BRANCH_IDX[name]].abs().max()) == 0.0, name
    # the spinblock's β half goes with it; the stay half (their active) is unaffected
    assert float(out[0, SWITCH_BRANCH_IDX["p_spin_blocked"]]) == 0.0
    # ...but the per-turn switch scalar is α's, not β's, and stays
    assert float(out[0, SWITCH_BRANCH_IDX["a_switch"]]) == pytest.approx(0.4, abs=1e-6)


# ----------------------------------------------------------------------------- the invariances


def test_contracted_columns_are_INVARIANT_to_permuting_THEIR_bench():
    """§5 gate 5: *"the contracted OA2 column is invariant to their bench permutation."* Permuting
    their six slots permutes β and the grid's defender axis together, so every contraction must be
    unchanged — anything that is not is reading a POSITION."""
    perm = torch.tensor([0, 3, 2, 1, 5, 4])          # keeps their ACTIVE at slot 0
    g = _omx()
    pg = torch.zeros(1, TEAM_SIZE)
    pg[0, 3] = 1.0
    base = _run(omx=g, p_ghost=pg)[0]
    perm_beta = _BETA_LOGITS[:, perm]
    out = _run(omx=g[:, :, perm], p_ghost=pg[:, perm], beta_logits=perm_beta)[0]
    assert torch.allclose(base, out, atol=1e-6)


def test_the_cell_is_INVARIANT_to_permuting_their_believed_move_SEATS():
    """The only seat-indexed computation is `Σ_k α_k · f_k`, so a JOINT permutation of (α's seats,
    the candidate nums, the liveness gate) must leave everything unchanged."""
    perm = [2, 0, 1]
    nums = torch.tensor([[_move_num("tackle"), _move_num("toxic"), _move_num("surf")]])
    base = _run(topk_nums=nums)[0]
    cell = _readable()
    out = cell(_ALPHA_LOGITS[:, perm + [3]], _BETA_LOGITS, _SEAT_LIVE[:, perm], nums[:, perm],
               _omx(), _p_ghost_default(), torch.tensor([0]),
               torch.tensor([[_SPIN, _PROTECT, 0, 0]]), torch.tensor([[0.5]]),
               torch.tensor([[1.0 / 3.0]]))[0]
    assert torch.allclose(base, out, atol=1e-6)


def _p_ghost_default():
    pg = torch.zeros(1, TEAM_SIZE)
    pg[0, 3] = 1.0
    return pg


# ------------------------------------------------------------------ contracts and fail-louds


def test_alpha_and_beta_are_both_stop_grad():
    """A POLICY-side consumer must not open a PPO route into either supervised head — and it must
    not depend on `--belief-grad-mode label_only` to cut it, or the route's EXISTENCE becomes a
    function of a training flag (the v87 pattern)."""
    a = _ALPHA_LOGITS.clone().requires_grad_(True)
    b = _BETA_LOGITS.clone().requires_grad_(True)
    _run(alpha_logits=a, beta_logits=b).sum().backward()
    # (the output DOES require grad — through the projection's own weights, which is the route the
    # cell is supposed to train. The claim is only that no gradient reaches the two heads.)
    assert a.grad is None, "a PPO route into alpha_head is open"
    assert b.grad is None, "a PPO route into beta_head is open"
    # and the RAW coordinates are detached before the projection ever sees them
    cell = SwitchBranchMoveCell(SWITCH_BRANCH_MOVE_DIM)
    for p in cell.parameters():
        p.requires_grad_(False)
    assert not _run(cell=cell, alpha_logits=a, beta_logits=b).requires_grad


def test_seat_axis_mismatch_fails_loud():
    with pytest.raises(ValueError, match="SAME axis"):
        _run(topk_nums=torch.zeros(1, 5, dtype=torch.long))


def test_a_grid_with_the_wrong_move_axis_fails_loud():
    """The cell is per-ACTION; a grid whose attacker axis is not the 4 request slots would score
    each move from another move's row while every shape check still passed."""
    with pytest.raises(ValueError, match="per-ACTION|request"):
        _run(omx=torch.zeros(1, 6, TEAM_SIZE, _DMG_OMX_CELL))


def test_the_cell_is_zero_init():
    cell = SwitchBranchMoveCell(SWITCH_BRANCH_MOVE_DIM)
    assert float(cell.proj.weight.abs().max()) == 0.0
    assert float(cell.proj.bias.abs().max()) == 0.0
    assert float(_run(cell=cell).abs().max()) == 0.0


# --------------------------------------------------------------- the op-side GHOST marginal


def test_op_ghost_marginal_is_exact_where_revealed_and_a_posterior_where_not():
    """`opp_p_ghost` is the op's, because the op owns both the species posterior and the type
    table. A REVEALED Gengar on their bench must read exactly 1.0; a hidden slot must read the
    Smogon-prior marginal, which is strictly between 0 and 1 (gen3ou has Ghosts, and it does not
    have only Ghosts)."""
    from agents.model import damage_op_test as DT
    from agents.model.damage_op import DamageOperator
    from agents.model.damage_tables import _T2I
    layout = Gen3ObservationEncoder(load_mappings()).get_layout()
    op = DamageOperator(layout, outgoing=True, matrices_outgoing=True)
    op.stash_opp_ghost = True
    ctx = DT._ctx_mtx(our_species=376, our_t1=_T2I["STEEL"], our_t2=_T2I["PSYCHIC"],
                      our_moves=[_move_num("earthquake"), 0, 0, 0],
                      our_move_types=[_T2I["GROUND"], 0, 0, 0],
                      opp_active=(0, _T2I["ELECTRIC"], 0),
                      bench_revealed=(94, _T2I["GHOST"], _T2I["POISON"]),      # Gengar, revealed
                      move_mask=[1, 0, 0, 0])
    op(ctx, torch.full((1, TEAM_SIZE, layout["max_moves"]), -10.0))
    pg = op.last_opp_p_ghost[0]
    assert float(pg[0]) == 0.0, "their revealed Electric active is not a Ghost"
    assert float(pg[1]) == 1.0, "a REVEALED Gengar must be exact, not a prior"
    assert 0.0 < float(pg[2]) < 0.5, ("an unrevealed slot carries the species-posterior marginal",
                                      float(pg[2]))


def test_the_op_stashes_nothing_when_the_ghost_seam_is_off():
    from agents.model import damage_op_test as DT
    from agents.model.damage_op import DamageOperator
    from agents.model.damage_tables import _T2I
    layout = Gen3ObservationEncoder(load_mappings()).get_layout()
    op = DamageOperator(layout, outgoing=True, matrices_outgoing=True)
    ctx = DT._ctx_mtx(our_species=376, our_t1=_T2I["STEEL"], our_t2=_T2I["PSYCHIC"],
                      our_moves=[0, 0, 0, 0], our_move_types=[0, 0, 0, 0],
                      opp_active=(0, _T2I["ELECTRIC"], 0), move_mask=[0, 0, 0, 0])
    op(ctx, torch.full((1, TEAM_SIZE, layout["max_moves"]), -10.0))
    assert op.last_opp_p_ghost is None


def test_out_cells_and_out_pko_are_ONE_tensor():
    """`out_pko` is a VIEW of `out_cells`, not a second reshape — so the OA2 magnitudes and v85's
    boom trade can never describe different worlds."""
    from agents.model import damage_op_test as DT
    from agents.model.damage_op import DamageOperator
    from agents.model.damage_tables import _T2I
    layout = Gen3ObservationEncoder(load_mappings()).get_layout()
    op = DamageOperator(layout, outgoing=True, matrices_outgoing=True)
    ctx = DT._ctx_mtx(our_species=376, our_t1=_T2I["STEEL"], our_t2=_T2I["PSYCHIC"],
                      our_moves=[_move_num("earthquake"), 0, 0, 0],
                      our_move_types=[_T2I["GROUND"], 0, 0, 0],
                      opp_active=(0, _T2I["ELECTRIC"], 0), move_mask=[1, 0, 0, 0])
    op(ctx, torch.full((1, TEAM_SIZE, layout["max_moves"]), -10.0))
    assert op.last_out_cells.shape == (1, 4, TEAM_SIZE, _DMG_OMX_CELL)
    assert torch.equal(op.last_out_pko, op.last_out_cells[..., _DMG_OMX_IDX_PKO])


# ------------------------------------------------------------------------------ extractor wiring


def test_off_builds_no_module_and_no_extra_dims():
    fe_off, _ = _build(**_BASE_KWARGS)
    fe_on, _ = _build(**_ON_KWARGS)
    assert fe_off.switch_branch is None and fe_on.switch_branch is not None
    assert not any("switch_branch" in k for k in fe_off.state_dict())
    assert fe_on.pointer_move_cell_dim == fe_off.pointer_move_cell_dim + SWITCH_BRANCH_MOVE_DIM
    assert fe_on.pointer_switch_cell_dim == fe_off.pointer_switch_cell_dim
    assert fe_on.projection.in_features == fe_off.projection.in_features
    assert fe_on.value_projection.in_features == fe_off.value_projection.in_features


def test_off_is_byte_identical():
    fe_off, layout = _build(**_BASE_KWARGS)
    obs = _obs(layout)
    pi, vf = fe_off(obs)
    pi2, vf2 = fe_off(obs)
    assert torch.equal(pi, pi2) and torch.equal(vf, vf2)
    assert not any("switch_branch" in k for k in fe_off.state_dict())
    assert fe_off.damage_op.stash_opp_ghost is False
    assert fe_off.damage_op.last_opp_p_ghost is None


def test_on_forward_runs_and_contributes_exactly_zero_at_init():
    fe, layout = _build(**_ON_KWARGS)
    fe(_obs(layout))
    cells = fe.last_pointer_inputs.move_cells
    assert cells.shape[2] == fe.pointer_move_cell_dim
    assert float(cells[..., -SWITCH_BRANCH_MOVE_DIM:].abs().max()) == 0.0
    assert "switch_branch.proj" in fe._identity_init_zeroed
    assert fe.damage_op.last_opp_p_ghost is not None


def test_requires_the_intent_head_because_alpha_SWITCH_and_beta_have_no_fallback():
    """The asymmetry with the pair-outcome flags is substantive, not caution. The R1 `belief_mean`
    rung is a PRESENCE belief over their MOVES and carries no switch class at all, so α_SWITCH
    would be identically 0 and every coordinate here would assert "they never switch" — a claim,
    not an absence. A flag whose fallback silently states something false is worse than one that
    says it needs the head."""
    with pytest.raises(ValueError, match="opp_intent"):
        _build(**{**_ON_KWARGS, "opp_intent": False})


def test_requires_the_outgoing_matrix():
    with pytest.raises(ValueError, match="damage_matrices_outgoing"):
        _build(**{**_ON_KWARGS, "damage_matrices_outgoing": False})


def test_missing_stashes_fail_loud_rather_than_contributing_zeros():
    with pytest.raises((RuntimeError, ValueError)):
        fe, layout = _build(**{**_ON_KWARGS, "damage_matrices_incoming": False,
                               "damage_topk_k": 0})
        fe(_obs(layout))


def test_it_stacks_with_every_other_move_cell_rider():
    """All five move-cell consumers plus the switch-cell one, at the summed width — the ede5a88
    lesson (a width that is only wrong when two flags MEET)."""
    fe, layout = _build(**_ON_KWARGS, intent_threshold=True, intent_move_cell=True,
                        intent_conditional=True, pair_outcome_cell=True,
                        pair_outcome_switch=True)
    pi, vf = fe(_obs(layout))
    assert pi.shape == vf.shape
    assert fe.last_pointer_inputs.move_cells.shape[2] == fe.pointer_move_cell_dim
    assert fe.last_pointer_inputs.switch_cells.shape[2] == fe.pointer_switch_cell_dim


# ------------------------------------------------------- identity-at-init on a REAL policy (M1)


def test_zero_init_survives_a_real_MaskablePPO_build():
    from agents.model.identity_init_test import _build_real_policy
    model, _ = _build_real_policy(damage_matrices_incoming=True, damage_matrices_outgoing=True,
                                  damage_topk_k=6, entity_topk_seats=6, move_latent=True,
                                  damage_outgoing=True, opp_intent=True,
                                  switch_branch_cell=True)
    fe = model.policy.features_extractor
    assert fe.switch_branch is not None
    assert float(fe.switch_branch.proj.weight.abs().max()) == 0.0
    assert float(fe.switch_branch.proj.bias.abs().max()) == 0.0


# ------------------------------------------------------------------------------ version machinery


def test_a_pre_floor_config_is_REFUSED_not_defaulted():
    """The critic-route deletion wave bumped ARCH_SIGNATURE, so MIGRATION_FLOOR rose to 96
    and this v93 config is now refused outright rather than walked through the v94 branch
    that defaults `switch_branch_cell`. That is the floor's stated purpose ("refuses pre-floor
    configs outright instead of walking dead branches"), and the assertion follows the
    BEHAVIOUR: what must hold is that a stale config is rejected with a diagnosis, not that
    an unreachable branch still defaults a field."""
    with pytest.raises(ModelVersionError, match="PRE-GENERATION|floor"):
        _migrate_config({"config_version": 93})
    assert MODEL_CONFIG_VERSION >= 94


def test_check_compatible_gates_the_flag():
    mappings = load_mappings()
    layout = Gen3ObservationEncoder(mappings).get_layout()
    b = ModelVersion.from_layout_and_policy_kwargs(layout, {"features_extractor_kwargs": {}})
    a = dataclasses.replace(b, switch_branch_cell=True)
    with pytest.raises(ModelVersionError, match="switch_branch_cell"):
        a.check_compatible(b)


def test_the_flag_round_trips_through_the_snapshot_kwargs():
    from agents.model.snapshot import current_model_version
    assert current_model_version(load_mappings(),
                                 switch_branch_cell=True).switch_branch_cell is True
    assert current_model_version(load_mappings()).switch_branch_cell is False


# -------------------------------------------------------------------------------- delivery graph


def test_the_module_is_drawn_with_edges_when_it_is_ON():
    import json
    import os
    import tempfile

    from agents.model.delivery_graph import (
        MODULE_GRAPH_TOKENS, _DEFAULT_CONFIG, build_extractor, build_graph, module_coverage)
    assert MODULE_GRAPH_TOKENS["switch_branch"] == ("SwitchBranchMoveCell",)
    cfg = json.load(open(_DEFAULT_CONFIG))
    cfg["switch_branch_cell"] = True
    cfg["opp_intent"] = True
    cfg["damage_matrices_outgoing"] = True
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "cfg.json")
        with open(path, "w") as fh:
            json.dump(cfg, fh)
        graph = build_graph(path)
        fe = build_extractor(path)[0]
    vias = " ".join(str(e.get("via", "")) for e in graph["edges"])
    assert "SwitchBranchMoveCell" in vias, "the module is ON and draws no edge"
    # β must be drawn reaching the move logits — OA2's whole subject is what they bring in
    assert any(str(e["src"]) == "beta_head" and str(e["dst"]).startswith("pointer.move_logit")
               for e in graph["edges"])
    assert not module_coverage(fe, graph)
