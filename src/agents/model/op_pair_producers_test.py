"""The OP-SIDE PRODUCERS the move-resolution family reads — ported AHEAD of legacy deletion U3 out of the seven per-action
blocks' tests that U3 deletes with the blocks (manifest finding F-LD-6: `designs/ops/legacy_deletion_manifest_2026-10-12.md`
§6; the inventory: `designs/research_state/measurements/legacy_deletion_u3_init_2026-10-15/F-LD-6_inventory.md`).

`move_resolution.gather_ops` reads, from the damage operator, `last_pair_in` (the unified outcome grid
`[B,6,K,14]`: the damage prefix `[low, high, crit, ko_ramp, acc, is_phys]` ⊕ six per-identity status columns ⊕ the
two status FACTS in columns 12-13), `last_pair_gate`, `last_pair_seat_live`, `last_pair_type_mult`,
`last_topk_idx`, `last_out_cells` and `stash.opp_species_post` (under X5: the same stashes with OTHER's column
appended, `FlatConsumerOps`), the α split `pair_outcome.pair_alpha_full`, and `pointer_intent_status_operands`
(the c2 operands `imc_ops`). Before this port most of the coverage of those producers lived in the seven blocks' test files
(`pair_outcome_test`, `pair_outcome_switch_test`, `switch_branch_test`, `conditional_threat_test`,
`intent_move_cell_test`, `intent_threshold_test`, `intent_conditional_test`); deleting the files with the blocks
would have dropped it silently. Each case below keeps its original constructed board and hand-derived expectation
and imports ONLY surviving code (the op, `pair_outcome`'s surviving names, `move_resolution`, the shared op
fixtures in `damage_op_test`). Every check is deterministic (root rule 8): exact zeros where the physics gives an
exact zero, and margins far from any decision boundary elsewhere.
"""
from __future__ import annotations

import inspect
import json
from typing import Any, Dict

import gymnasium as gym
import numpy as np
import pytest
import torch

from agents.model import damage_op_test as DT
from agents.model.arch_constants import _PAIR_OUTCOME_DMG, _PAIR_OUTCOME_NEW, _PAIR_OUTCOME_RAW
from agents.model.damage_op import DamageOperator
from agents.model.damage_op_layout import _DMG_OMX_CELL, _DMG_OMX_IDX_PKO
from agents.model.pair_outcome import GHOST_TYPE_IDX, PAIR_OUTCOME_IDX, STATUS_FACT_COORDS, pair_alpha_full
from agents.observation.constants import TEAM_SIZE
from agents.observation.types import TypeEncoder

T = TypeEncoder.TYPE_TO_IDX


# ------------------------------------------------------------------------------ the grid's coordinate contract


def test_the_outcome_grid_layout_is_the_one_move_resolution_slices() -> None:
    """The column order `move_resolution` hard-slices (`pair_in[..., :12]` the per-identity rows, `[..., 12:14]` the
    two facts, index 1 / 2 / 3 = high / crit / ko_ramp in `switch_facts`) is `PAIR_OUTCOME_IDX`'s. A reorder, a third
    fact, or a damage prefix that is no longer the op's own pair-cell channel order breaks every move-resolution
    read silently (the slices still have the right width). Ported from `pair_outcome_test.
    test_the_coordinate_table_is_the_single_spelling_of_the_layout`."""
    order = sorted(PAIR_OUTCOME_IDX, key=PAIR_OUTCOME_IDX.__getitem__)
    assert list(PAIR_OUTCOME_IDX.values()) == list(range(len(order)))
    assert len(order) == _PAIR_OUTCOME_RAW == _PAIR_OUTCOME_DMG + _PAIR_OUTCOME_NEW == 14
    assert tuple(order[:_PAIR_OUTCOME_DMG]) == ("low", "high", "crit", "ko_ramp", "acc", "is_phys")
    assert tuple(order[-2:]) == STATUS_FACT_COORDS == ("e_burn_dmg_lost", "e_par_outspeed_lost")
    assert [PAIR_OUTCOME_IDX[n] for n in STATUS_FACT_COORDS] == [12, 13]
    assert (PAIR_OUTCOME_IDX["high"], PAIR_OUTCOME_IDX["crit"], PAIR_OUTCOME_IDX["ko_ramp"]) == (1, 2, 3)


def test_the_two_derivable_collapses_are_deliberately_absent() -> None:
    """§9a's derivability rule: `p_status_land` / `p_immobilize` are LINEAR in the six per-identity columns, so
    delivering them beside those columns is the redundancy the rule forbids. A future edit that "restores" one
    must delete this test first. Ported from `pair_outcome_test.test_the_two_derivable_collapses_are_deliberately_
    absent`."""
    assert "p_status_land" not in PAIR_OUTCOME_IDX and "p_immobilize" not in PAIR_OUTCOME_IDX
    order = sorted(PAIR_OUTCOME_IDX, key=PAIR_OUTCOME_IDX.__getitem__)
    assert [c for c in order if c.startswith("p_")] == ["p_par", "p_brn", "p_frz", "p_slp", "p_psn", "p_tox"]
    assert [PAIR_OUTCOME_IDX[c] for c in ("p_par", "p_brn", "p_frz", "p_slp", "p_psn", "p_tox")] == list(range(6, 12))


def test_the_ghost_type_id_is_resolved_from_the_encoder_not_written_as_a_number() -> None:
    """`damage_op` reads `GHOST_TYPE_IDX` (Curse's Ghost branch); a hard-coded `14` would keep passing after a
    TypeEncoder renumbering while pricing the wrong type. Ported from `pair_outcome_switch_test.
    test_the_ghost_type_id_is_resolved_from_the_encoder_not_written_as_a_number`."""
    assert GHOST_TYPE_IDX == TypeEncoder.TYPE_TO_IDX["GHOST"]


# ------------------------------------------------------------------------------------ the α split (surviving)


def test_pair_alpha_full_splits_the_publication_without_reassigning_masked_mass() -> None:
    """`a_stay` is `Σ_k α_k`, NOT `1 − a_switch`: the two differ exactly when a seat is masked, and an unmodeled
    seat's usage mass is simply not spent. Breaks if the split renormalises the masked seat's mass back in.
    Ported from `pair_outcome_switch_test.test_pair_alpha_full_splits_the_publication_without_reassigning_masked_mass`
    (and `pair_outcome_test.test_unmodeled_seats_are_masked_and_the_mass_is_NOT_reassigned`)."""
    lg = torch.log(torch.tensor([[0.1, 0.2, 0.3, 0.4]]))
    alpha, a_switch, a_stay = pair_alpha_full(lg, torch.tensor([[1.0, 1.0, 0.0]]))
    assert torch.allclose(alpha, torch.tensor([[0.1, 0.2, 0.0]]), atol=1e-6)
    assert float(a_switch) == pytest.approx(0.4, abs=1e-6)
    assert float(a_stay) == pytest.approx(0.3, abs=1e-6), "masked mass was renormalized back in"
    assert float(a_stay + a_switch) < 1.0


def test_pair_alpha_full_is_the_unrenormalized_move_slice_and_a_certain_switch_empties_it() -> None:
    """The move slice is the UNRENORMALIZED softmax: uniform logits over K seats + SWITCH give 1/(K+1) per seat, and a
    certain SWITCH drives every seat's α to 0 — 'they are leaving' must not be priced as 'they attack'. Ported from
    `pair_outcome_test.test_alpha_from_the_publication_is_the_unrenormalized_move_slice` and
    `test_certain_switch_zeroes_the_whole_outcome_vector` (the `pair_alpha` / `reduce_pair_in` spellings go with the
    block; `pair_alpha_full` is the one move resolution reads)."""
    K = 3
    alpha, a_switch, a_stay = pair_alpha_full(torch.zeros(2, K + 1))
    assert torch.allclose(alpha, torch.full((2, K), 1.0 / (K + 1)), atol=1e-6)
    assert torch.allclose(a_switch, torch.full((2, 1), 1.0 / (K + 1)), atol=1e-6)
    assert float(a_stay.max()) < 1.0, "the SWITCH mass must NOT be renormalized away"
    lg = torch.zeros(1, K + 1)
    lg[:, -1] = 30.0
    alpha, a_switch, _ = pair_alpha_full(lg)
    assert float(alpha.max()) < 1e-4 and float(a_switch) > 1.0 - 1e-4


def test_pair_alpha_full_is_stop_grad() -> None:
    """A policy-side consumer must not open a PPO → α-head route; the cut is unconditional, not a function of
    `--belief-grad-mode`. Ported from `pair_outcome_switch_test.test_pair_alpha_full_is_stop_grad` (and
    `pair_outcome_test.test_alpha_is_stop_grad_so_no_policy_route_opens_into_the_intent_head`)."""
    lg = torch.zeros(1, 4, requires_grad=True)
    for t in pair_alpha_full(lg, torch.ones(1, 3)):
        assert not t.requires_grad, "a policy-side consumer opened a PPO route into the intent head"


# ------------------------------------------------------- the outcome grid through the REAL op forward (G0)


def _real_op(K: int = 6) -> Any:
    """The bare incoming op at top-K with the move-resolution seams on (the flags `extractor_build` sets for
    `move_resolution on`: pair cells + the outcome grid + the type multiplier)."""
    op, layout = DT._op_and_layout_topk(K)
    op.stash_pair_cells = True
    op.stash_pair_outcome = True
    op.stash_pair_type_mult = True
    return op, layout


def _run(op: Any, layout: Dict[str, Any], moves: list, defenders: list, **ctx_kw: Any) -> Any:
    ctx = DT._topk_ctx(op, defenders=defenders, **ctx_kw)
    op(ctx, DT._logits_moves(layout["max_moves"], moves), None, DT._synth_latent(layout))
    return ctx


def test_g0_the_currency_failure_is_closed_swampert_into_will_o_wisp() -> None:
    """Swampert (Water/Ground) vs a believed Will-O-Wisp + Thunderbolt: on damage alone it reads 0.0 in both
    branches, so the grid must carry the burn on exactly the seat the damage numbers cannot see — and on the BURN
    column only. Breaks if the status split is lost, mis-identified, or the damage prefix leaks into it. Ported from
    `pair_outcome_test.test_g0_the_currency_failure_is_closed_swampert_into_will_o_wisp`."""
    op, layout = _real_op()
    wow, tb = DT._move_num("willowisp"), DT._move_num("thunderbolt")
    _run(op, layout, [wow, tb], [(260, T["WATER"], T["GROUND"])] + [(0, 0, 0)] * 5)
    pin = op.last_pair_in
    assert pin is not None and pin.shape[-1] == _PAIR_OUTCOME_RAW
    k = op.last_topk_idx[0].tolist().index(wow)
    cell = pin[0, 0, k]
    for name in ("low", "high", "crit", "ko_ramp"):
        assert float(cell[PAIR_OUTCOME_IDX[name]]) == 0.0, f"{name} should be 0 for Will-O-Wisp"
    assert float(cell[PAIR_OUTCOME_IDX["p_brn"]]) > 0.5
    for other in ("p_par", "p_frz", "p_slp", "p_psn", "p_tox"):
        assert float(cell[PAIR_OUTCOME_IDX[other]]) == 0.0


def test_g0_toxic_and_poison_powder_land_on_DIFFERENT_columns() -> None:
    """`MOVE_STATUS_CAT` folds tox into psn; the grid's identity split (`MOVE_STATUS_IDENT`) must keep them apart —
    one escalates. Breaks if the producer keys the identity on the category. Ported from
    `pair_outcome_test.test_g0_toxic_and_poison_powder_land_on_DIFFERENT_columns`."""
    op, layout = _real_op()
    tox, psn = DT._move_num("toxic"), DT._move_num("poisonpowder")
    _run(op, layout, [tox, psn], [(260, T["WATER"], T["GROUND"])] + [(0, 0, 0)] * 5)
    idx = op.last_topk_idx[0].tolist()
    ktox, kpsn = idx.index(tox), idx.index(psn)
    pin = op.last_pair_in[0, 0]
    assert float(pin[ktox, PAIR_OUTCOME_IDX["p_tox"]]) > 0.5
    assert float(pin[ktox, PAIR_OUTCOME_IDX["p_psn"]]) == 0.0
    assert float(pin[kpsn, PAIR_OUTCOME_IDX["p_psn"]]) > 0.5
    assert float(pin[kpsn, PAIR_OUTCOME_IDX["p_tox"]]) == 0.0


def test_g0_an_immune_pivot_reads_zero_status_risk_and_zero_status_facts() -> None:
    """The grid's landing physics is `_incoming_status_lands`, so the per-pivot immunity survives into it: Thunder
    Wave into a Ground pivot is EXACTLY 0 — and so are both status facts (each is P(that status lands) × its effect).
    A Normal pivot is not immune. Breaks if the landing is re-derived without the defender's types, or a fact is not
    gated by its landing probability. Ported from `pair_outcome_test.test_g0_an_immune_pivot_reads_zero_status_risk`."""
    op, layout = _real_op()
    tw = DT._move_num("thunderwave")
    _run(op, layout, [tw], [(260, T["WATER"], T["GROUND"]), (242, T["NORMAL"], T["NORMAL"])] + [(0, 0, 0)] * 4)
    k = op.last_topk_idx[0].tolist().index(tw)
    pin = op.last_pair_in[0]
    assert float(pin[0, k, PAIR_OUTCOME_IDX["p_par"]]) == 0.0, "Ground is immune to Thunder Wave"
    for name in STATUS_FACT_COORDS:
        assert float(pin[0, k, PAIR_OUTCOME_IDX[name]]) == 0.0, name
    assert float(pin[1, k, PAIR_OUTCOME_IDX["p_par"]]) > 0.5, "a Normal pivot is not immune"


def test_g0_each_defenders_row_is_its_own_and_the_gate_and_seat_mask_are_stashed_on_the_grids_axes() -> None:
    """The per-defender grid must not become a per-defender BELIEF: two of our mons with different immunities read
    different rows (Ground: 0 paralysis, Normal: > 0), while `last_pair_gate` [B,6,1] and `last_pair_seat_live` [B,K]
    sit on the grid's own defender / seat axes, so `move_resolution.switch_facts`' contraction
    `Σ_k α_k · pair_in[j,k,:] · gate_j` is one α against six columns. Breaks if the gate or the seat mask is
    stashed on another axis or the rows are pooled. Ported from `pair_outcome_switch_test.
    test_g0_each_defenders_row_is_its_own_and_the_reduction_is_still_one_alpha` (its `reduce_pair_in_all` /
    `pair_alpha` spellings go with the block; the contraction is written out)."""
    op, layout = _real_op()
    tw = DT._move_num("thunderwave")
    _run(op, layout, [tw], [(260, T["WATER"], T["GROUND"]), (242, T["NORMAL"], T["NORMAL"])] + [(0, 0, 0)] * 4)
    pin, gate, live, w = op.last_pair_in, op.last_pair_gate, op.last_pair_seat_live, op.last_topk_w
    K = op.last_topk_idx.shape[-1]
    assert pin.shape == (1, TEAM_SIZE, K, _PAIR_OUTCOME_RAW)
    assert gate.shape == (1, TEAM_SIZE, 1) and live.shape == (1, K)
    assert float(gate[0, 0, 0]) == 1.0 and float(gate[0, 1, 0]) == 1.0, "our two live defenders are gated ON"
    alpha = (w * live) / (w * live).sum(-1, keepdim=True)                # any one distribution over the seats
    rows = torch.einsum("bk,bjkf->bjf", alpha, pin) * gate               # move_resolution's contraction
    i = PAIR_OUTCOME_IDX["p_par"]
    assert float(rows[0, 0, i]) == 0.0, "Ground is immune to Thunder Wave"
    assert float(rows[0, 1, i]) > 0.0


def test_the_damage_prefix_of_the_grid_IS_the_ops_pair_cells() -> None:
    """The grid's first six columns are the op's own aligned pair cells, verbatim — damage and status share ONE
    tensor on ONE (defender, seat) grid, which is what lets one α weight both. Breaks if the prefix is re-derived
    (or re-ordered) instead of concatenated. Ported from the premise of `pair_outcome_test.
    test_the_coordinate_table_is_the_single_spelling_of_the_layout` ("the damage prefix is the op's OWN pair-cell
    channel order, verbatim")."""
    op, layout = _real_op()
    eq, tb = DT._move_num("earthquake"), DT._move_num("thunderbolt")
    _run(op, layout, [eq, tb], [(143, T["NORMAL"], T["NORMAL"]), (227, T["STEEL"], T["FLYING"])] + [(0, 0, 0)] * 4,
         attacker_num=376, attacker_t1=T["STEEL"], attacker_t2=T["PSYCHIC"])
    assert torch.equal(op.last_pair_in[..., :_PAIR_OUTCOME_DMG], op.last_pair_cells)
    k = op.last_topk_idx[0].tolist().index(eq)
    assert float(op.last_pair_cells[0, 0, k, PAIR_OUTCOME_IDX["high"]]) > 0.0, "PRECONDITION: a live damage cell"


def test_g0_immunity_and_category_in_the_pair_cells() -> None:
    """A Steel/Flying defender facing a believed Earthquake (IMMUNE) and Thunderbolt (hits): the immune seat's cell
    reads EXACTLY 0 in `high` and `ko_ramp` — falling out of the op's own type chart — while Thunderbolt's reads a
    positive `high` at its accuracy (1.0) and is special (`is_phys` 0). Breaks if the immunity or the category is
    lost from the cells move resolution's hit / KO facts read. Ported from `intent_threshold_test.
    test_g0_immunity_and_category_through_the_real_physics` (its `threshold_probs` contraction goes with the block;
    the cell-level physics it rested on is pinned here)."""
    K = 5
    op, layout = _real_op(K)
    eq, tb = DT._move_num("earthquake"), DT._move_num("thunderbolt")
    _run(op, layout, [eq, tb], [(227, T["STEEL"], T["FLYING"])] + [(0, 0, 0)] * 5)
    idx = op.last_topk_idx[0].tolist()
    assert eq in idx and tb in idx
    c = op.last_pair_cells[0, 0]
    ke, kt = idx.index(eq), idx.index(tb)
    assert float(c[ke, PAIR_OUTCOME_IDX["high"]]) == 0.0, "an IMMUNE Earthquake deals nothing"
    assert float(c[ke, PAIR_OUTCOME_IDX["ko_ramp"]]) == 0.0
    assert float(c[kt, PAIR_OUTCOME_IDX["high"]]) > 0.0, "Thunderbolt hits a Flying type"
    assert float(c[kt, PAIR_OUTCOME_IDX["acc"]]) == pytest.approx(1.0, abs=1e-6)
    assert float(c[kt, PAIR_OUTCOME_IDX["is_phys"]]) == 0.0 and float(c[ke, PAIR_OUTCOME_IDX["is_phys"]]) == 1.0


def test_the_op_stashes_the_type_multiplier_only_when_asked_and_at_alphas_alignment() -> None:
    """The multiplier must arrive on α's OWN seat axis (re-deriving it at the consumer from `last_topk_idx` + the
    chart is the `op move-order` bug class): Earthquake into an Electric mon reads EXACTLY 2.0, into a Flying mon
    EXACTLY 0.0 (a gen-3 IMMUNITY); the seam is off by default. Breaks if the stash is mis-aligned, folded
    differently, or always on. Ported from `conditional_threat_test.
    test_the_op_stashes_the_type_multiplier_only_when_asked_and_at_alphas_alignment`."""
    K = 6
    eq = DT._move_num("earthquake")
    defenders = [(0, T["ELECTRIC"], 0), (0, T["FLYING"], 0)] + [(0, 0, 0)] * 4
    op, layout = DT._op_and_layout_topk(K)
    _run(op, layout, [eq], defenders)
    assert op.last_pair_type_mult is None, "the seam is OFF by default — no stash, no cost"
    op, layout = DT._op_and_layout_topk(K)
    op.stash_pair_type_mult = True
    _run(op, layout, [eq], defenders)
    tm = op.last_pair_type_mult
    assert tm is not None and tm.shape == (1, TEAM_SIZE, K)
    k = op.last_topk_idx[0].tolist().index(eq)
    assert float(tm[0, 0, k]) == pytest.approx(2.0, abs=1e-6)
    assert float(tm[0, 1, k]) == 0.0, "Ground vs Flying is a gen3 IMMUNITY, exactly 0"


def test_the_op_stashes_no_outcome_grid_when_the_seam_flag_is_off() -> None:
    """The grid is built only when asked (`stash_pair_outcome`; `extractor_build` sets it for move resolution).
    Breaks if the producer runs (and costs) unconditionally. Ported from `pair_outcome_test.
    test_the_op_stashes_nothing_when_the_seam_flag_is_off`."""
    op, layout = _real_op()
    op.stash_pair_outcome = False
    _run(op, layout, [DT._move_num("willowisp")], [(260, T["WATER"], T["GROUND"])] + [(0, 0, 0)] * 5)
    assert op.last_pair_in is None


def test_a_grid_without_a_topk_axis_fails_loud_rather_than_contributing_zeros() -> None:
    """The cells / grid seams with no top-K selection (no incoming matrix) RAISE — a silent no-op reads exactly like
    a null result. Breaks if the producer substitutes a different seat axis or skips quietly. Ported (to the op
    level) from `pair_outcome_test.test_missing_topk_stash_fails_loud_rather_than_contributing_zeros` /
    `pair_outcome_switch_test` / `intent_threshold_test.test_missing_stash_fails_loud`."""
    layout = DT._make_layout()
    op = DamageOperator(layout)                                  # no incoming matrix, no top-K
    op.stash_pair_cells = True
    op.stash_pair_outcome = True
    ctx = DT._topk_ctx(op, defenders=[(260, T["WATER"], T["GROUND"])] + [(0, 0, 0)] * 5)
    with pytest.raises(RuntimeError, match="top-K"):
        op(ctx, DT._logits_moves(layout["max_moves"], [DT._move_num("willowisp")]), None, DT._synth_latent(layout))


# ------------------------------------------------------------------------- the outgoing cells (move resolution)


def test_out_cells_and_out_pko_are_ONE_tensor() -> None:
    """`out_pko` is a VIEW of `out_cells` (the grid move resolution reads as `out_cells`), not a second reshape — so
    the two can never describe different worlds. Breaks if either is rebuilt separately or the cell layout drifts.
    Ported from `switch_branch_test.test_out_cells_and_out_pko_are_ONE_tensor`."""
    from agents.model.damage_tables import _T2I
    layout = DT._make_layout()
    op = DamageOperator(layout, outgoing=True, matrices_outgoing=True)
    ctx = DT._ctx_mtx(our_species=376, our_t1=_T2I["STEEL"], our_t2=_T2I["PSYCHIC"],
                      our_moves=[DT._move_num("earthquake"), 0, 0, 0],
                      our_move_types=[_T2I["GROUND"], 0, 0, 0],
                      opp_active=(0, _T2I["ELECTRIC"], 0), move_mask=[1, 0, 0, 0])
    op(ctx, torch.full((1, TEAM_SIZE, layout["max_moves"]), -10.0))
    assert op.last_out_cells.shape == (1, 4, TEAM_SIZE, _DMG_OMX_CELL)
    assert torch.equal(op.last_out_pko, op.last_out_cells[..., _DMG_OMX_IDX_PKO])


# ------------------------------------------------------------------ the c2 operands (`imc_ops`, move resolution)


def _operands_op(K: int = 3) -> Any:
    layout = DT._make_layout()
    op = DamageOperator(layout, outgoing=True, matrices_incoming=True, topk_k=K)
    eq, tb = DT._move_num("earthquake"), DT._move_num("thunderbolt")
    ctx = DT._topk_ctx(op, defenders=[(143, T["NORMAL"], T["NORMAL"])] + [(0, 0, 0)] * 5,
                       attacker_num=376, attacker_t1=T["STEEL"], attacker_t2=T["PSYCHIC"])
    ctx.our_active_req_move_ids = torch.tensor([[DT._move_num("willowisp"), DT._move_num("spore"),
                                                 DT._move_num("thunderwave"), 0]])
    logits = DT._logits_moves(layout["max_moves"], [eq, tb])
    return op, layout, ctx, logits, eq, tb


def test_the_status_operands_fail_loud_without_the_topk_stash() -> None:
    """`pointer_intent_status_operands` aligns its seat axis to α by the op's own `last_topk_idx`; with no top-K stash
    it RAISES rather than substituting another selection (the `op move-order` class). NEW at the op level: the
    retired blocks pinned this only through their own extractor forward (`intent_move_cell_test.
    test_requires_opp_intent_and_damage_op`, the missing-stash cases)."""
    op, _layout, ctx, logits, _eq, _tb = _operands_op()
    with pytest.raises(RuntimeError, match="last_topk_idx"):
        op.pointer_intent_status_operands(ctx, logits, k_cand=3)


def test_the_status_operands_price_burn_on_the_physical_seat_only_on_alphas_axis() -> None:
    """The burn delta is `high(atk/2) − high(atk)` on OUR active per seat: STRICTLY negative for a physical seat
    (Earthquake), EXACTLY 0 for a special one (Thunderbolt — only Attack halves), never below the sleep delta
    `−high` (a burn removes part of the hit, sleep all of it); both live on the op's top-K seat axis; our request
    slots' masks name Will-O-Wisp as the burn and Spore as the sleep. Breaks if the axis drifts from
    `last_topk_idx`, the burn halves SpA, or the category masks move. NEW at the op level: these operands' values
    were reached in the retired tests only through `IntentMoveCell` (`intent_move_cell_test`)."""
    op, layout, ctx, logits, eq, tb = _operands_op()
    op(ctx, logits, None, DT._synth_latent(layout))
    base, d_burn, d_slp, is_brn, is_slp = op.pointer_intent_status_operands(ctx, logits, k_cand=3)
    K = op.last_topk_idx.shape[-1]
    assert d_burn.shape == (1, K) and d_slp.shape == (1, K) and base.shape == (1, 4, 4)
    idx = op.last_topk_idx[0].tolist()
    ke, kt = idx.index(eq), idx.index(tb)
    assert float(d_slp[0, ke]) < 0.0 and float(d_slp[0, kt]) < 0.0, "PRECONDITION: both seats hit our Snorlax"
    assert float(d_burn[0, ke]) < 0.0, "a burn halves the physical seat's damage"
    assert float(d_burn[0, kt]) == 0.0, "a burn does not touch a special seat"
    assert float(d_burn[0, ke]) > float(d_slp[0, ke]), "the burn delta cannot exceed the whole hit"
    assert is_brn[0].tolist() == [1.0, 0.0, 0.0, 0.0]
    assert is_slp[0].tolist() == [0.0, 1.0, 0.0, 0.0]


# ------------------------------------------------- the producers inside the PRODUCTION extractor (move resolution on)


def _prod_extractor() -> Any:
    from agents.model.damage_tables import sanitize_historical_move_floor
    from agents.model.features_extractor import Gen3FeaturesExtractor
    from agents.observation.state_encoder import Gen3ObservationEncoder, load_mappings
    from utils.paths import repo_path
    with open(repo_path("designs", "production_config.json")) as fh:
        cfg = json.load(fh)
    sig = set(inspect.signature(Gen3FeaturesExtractor.__init__).parameters)
    kw = {k: v for k, v in cfg.items() if k in sig}
    sanitize_historical_move_floor(kw)
    mappings = load_mappings()
    layout = Gen3ObservationEncoder(mappings).get_layout()
    space = gym.spaces.Box(0.0, 1.0, shape=(layout["total_dim"],), dtype=np.float32)
    torch.manual_seed(0)
    return Gen3FeaturesExtractor(space, layout=layout, mappings=mappings, **kw).eval(), layout


def test_production_sets_every_seam_move_resolution_reads_and_each_stash_lands_on_its_axis() -> None:
    """Production (`move_resolution on`, X5) turns on the four op seams `gather_ops` needs, and one forward on the
    real parity rows fills every stash on the axes the family indexes: the grid `[B,6,K,14]` (the per-identity
    status columns are probabilities whose sum per (defender, seat) is ≤ 1, the facts ≥ 0), the type multiplier
    `[B,6,K]`, the gate `[B,6,1]`, the seat mask `[B,K]`, the top-K nums `[B,K]`, the species posterior, the outgoing
    cells; and the X5 re-expression's grid is the op's grid with ONE column (OTHER_move's) appended, bit-exact.
    Breaks if a seam stops being set (the family would raise at run time, not at build), a stash changes axis, or
    the X5 append re-derives the named seats. Ported from the retired blocks' `test_on_forward_runs_*` cases
    (`pair_outcome_test`, `pair_outcome_switch_test`, `conditional_threat_test`: "the op's last_pair_in /
    last_pair_type_mult is not None when ON") — now pinned on the build that ships."""
    from agents.model.compile_parity_fixture import load_parity_rows
    fe, layout = _prod_extractor()
    assert fe.move_resolution_cell is not None, "PRECONDITION: production runs move resolution"
    op = fe.damage_op
    assert op.stash_pair_cells and op.stash_pair_outcome and op.stash_pair_type_mult and op.stash_species_post
    obs, _mask = load_parity_rows(layout["total_dim"])
    x = torch.as_tensor(obs)
    with torch.no_grad():
        fe({"observation": x})
    B, K = x.shape[0], fe.entity_topk_seats
    pin = op.last_pair_in
    assert pin is not None and pin.shape == (B, TEAM_SIZE, K, _PAIR_OUTCOME_RAW)
    assert op.last_pair_type_mult.shape == (B, TEAM_SIZE, K)
    assert op.last_pair_gate.shape == (B, TEAM_SIZE, 1)
    assert op.last_pair_seat_live.shape == (B, K) and op.last_topk_idx.shape == (B, K)
    assert op.stash.opp_species_post is not None and op.last_out_cells is not None
    assert torch.isfinite(pin).all()
    st = pin[..., 6:12]
    assert float(st.min()) >= 0.0 and float(st.sum(-1).max()) <= 1.0 + 1e-5
    assert float(pin[..., 12:14].min()) >= 0.0
    assert float(st.max()) > 0.0, "PRECONDITION: some status seat lands in the fixture"
    x5 = fe.stash.flat_consumer_ops
    assert x5 is not None and x5.pair_in.shape == (B, TEAM_SIZE, K + 1, _PAIR_OUTCOME_RAW)
    assert torch.equal(x5.pair_in[:, :, :K], pin), "the named seats must be the op's grid, as given"
    assert torch.equal(x5.type_mult[:, :, :K], op.last_pair_type_mult)
    assert torch.equal(x5.seat_live[:, :K], op.last_pair_seat_live)
