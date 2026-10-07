"""X5 U4: the FLAT opponent pointer (`gen3_x5_flat_pointer_v1`; design_x5_belief_tokens.md §3.7, §3.5's
pointer I2, §8.3 U4).

Pure label / re-expression / consumer tests on synthetic tensors, plus reads on a REAL fixed_mass policy
over the K9 golden buffer (deterministic: fixed rows, fixed build seed):

* a belief MISS is an OTHER label (OTHER_move / OTHER_species), never a masked row; a typed Hidden
  Power label names a REVEALED Hidden Power's seat (num 237);
* the candidate set and its STRUCTURAL mask (K seats, OTHER_move, the six switch targets, OTHER_species);
* the pointer's I2 ("mass of copies"): splitting a candidate into two copies of presence w/2 leaves the
  EVENT's probability and every other candidate's unchanged;
* the consumers' re-expression is the flat distribution EXACTLY, finite in value AND gradient on a row
  with no switch candidate;
* OTHER's columns in `out_cells` / `opp_p_ghost` and OTHER_move's seat-axis columns are PRICED (never a
  zero row), checked against independent per-move / per-species oracles;
* `seat_live` is applied by all four α consumers (F-X5-15);
* `opp_intent/other_label_rate` is computed (F-X5-8); the fold traces fullgraph;
* the flat loss trains the head and never δ_θ (M10).

Each test fails on the revert it names.
"""
from __future__ import annotations

import copy
import math

import pytest
import torch

from agents.model.flat_intent import (LABEL_NONCHOICE, LABEL_OTHER_MOVE, LABEL_OTHER_SPECIES,
                                      LABEL_SEAT, LABEL_SLOT, LABEL_UNMODELED, FlatIntentHead,
                                      FlatIntentInputs, compat_intent_logits, flat_intent_targets,
                                      flat_width, other_move_col, other_species_col, slot_col)
from agents.model.opp_intent import INTENT_IGNORE

K, M, S = 6, 400, 400


# ------------------------------------------------------------------------------- synthetic labels
def _fi(B: int) -> FlatIntentInputs:
    """Row layout: seats [89, 57, 237(revealed HP), 0, 0, 0] with the first 3 on; OTHER_move members
    {94, 85}; slot 4 holds hypothesis Snorlax (143), slot 5 hypothesis Skarmory (227); the tail holds
    Blissey (242) and Gengar (94); every candidate live except where a test switches one off."""
    seat_nums = torch.tensor([[89, 57, 237, 0, 0, 0]]).expand(B, -1).clone()
    seat_on = torch.tensor([[True, True, True, False, False, False]]).expand(B, -1).clone()
    hp_seat = torch.tensor([[False, False, True, False, False, False]]).expand(B, -1).clone()
    beyond = torch.zeros(B, M, dtype=torch.bool)
    beyond[:, [94, 85]] = True
    slot_species = torch.tensor([[0, 0, 0, 0, 143, 227]]).expand(B, -1).clone()
    hyp = slot_species > 0
    in_tail = torch.zeros(B, S, dtype=torch.bool)
    in_tail[:, [242, 94]] = True
    live = torch.ones(B, flat_width(K), dtype=torch.bool)
    live[:, 3:K] = False
    live[:, slot_col(K, 0)] = False                  # slot 0 is the active
    return FlatIntentInputs(k=K, live=live, cand_ids=torch.zeros(B, flat_width(K), dtype=torch.long),
                            log_pi=torch.zeros(B, flat_width(K)), seat_nums=seat_nums, seat_on=seat_on,
                            hp_seat=hp_seat, beyond=beyond, slot_species=slot_species,
                            slot_is_hypothesis=hyp, in_tail=in_tail)


def _labels(rows):
    kind = torch.tensor([r[0] for r in rows])
    num = torch.tensor([r[1] for r in rows])
    slot = torch.tensor([r[2] for r in rows])
    sp = torch.tensor([r[3] for r in rows])
    return kind, num, slot, sp


def test_a_belief_miss_is_an_OTHER_label_never_a_masked_row():
    """Revert (OTHER branches → INTENT_IGNORE, the blob's mask-on-miss): rows 1 and 5 fail."""
    rows = [(0, 57, -100, 0),        # 0 a move in the seats               -> seat 1
            (0, 94, -100, 0),        # 1 a move BEYOND the seats           -> OTHER_move
            (0, 165, -100, 0),       # 2 Struggle: outside every candidate -> masked, UNMODELED
            (1, 0, 2, 0),            # 3 a switch to revealed slot 2       -> slot 2
            (1, 0, -100, 227),       # 4 a hidden switch-in = hypothesis   -> slot 5
            (1, 0, -100, 242),       # 5 a hidden switch-in in the TAIL    -> OTHER_species
            (1, 0, -100, 25),        # 6 a hidden species outside V's tail -> masked, UNMODELED
            (2, 0, -100, 0),         # 7 not a choice                      -> masked, NONCHOICE
            (0, 362, -100, 0)]       # 8 typed HP (Ice) vs the REVEALED HP seat (237) -> seat 2
    fi = _fi(len(rows))
    tgt, cls = flat_intent_targets(fi, *_labels(rows))
    assert tgt.tolist() == [1, other_move_col(K), INTENT_IGNORE, slot_col(K, 2), slot_col(K, 5),
                            other_species_col(K), INTENT_IGNORE, INTENT_IGNORE, 2]
    assert cls.tolist() == [LABEL_SEAT, LABEL_OTHER_MOVE, LABEL_UNMODELED, LABEL_SLOT, LABEL_SLOT,
                            LABEL_OTHER_SPECIES, LABEL_UNMODELED, LABEL_NONCHOICE, LABEL_SEAT]


def test_a_label_on_a_dead_candidate_is_masked_the_reach_rule():
    fi = _fi(3)
    live = fi.live.clone()
    live[:, other_move_col(K)] = False               # OTHER_move masked (all 4 revealed)
    live[:, other_species_col(K)] = False
    live[:, slot_col(K, 2)] = False
    import dataclasses
    fi = dataclasses.replace(fi, live=live)
    tgt, cls = flat_intent_targets(fi, *_labels([(0, 94, -100, 0), (1, 0, -100, 242), (1, 0, 2, 0)]))
    assert tgt.tolist() == [INTENT_IGNORE] * 3
    assert cls.tolist() == [LABEL_UNMODELED] * 3


# ----------------------------------------------------------------------------------- the head / I2
def _head_inputs(B: int = 5, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    D, C = 16, 8
    tok = torch.randn(B, flat_width(K), D, generator=g, dtype=torch.float64)
    ctx = torch.randn(B, C, generator=g, dtype=torch.float64)
    lp = torch.log(torch.rand(B, flat_width(K), generator=g, dtype=torch.float64) * 0.9 + 0.05)
    live = torch.ones(B, flat_width(K), dtype=torch.bool)
    live[:, 4] = False
    return tok, ctx, lp, live


def test_I2_for_the_pointer_the_mass_of_copies():
    """Splitting move seat j (presence w) into two copies of w/2 — through the head's real forward, one
    more seat — leaves P(event j) = the copies' sum and every other candidate's probability unchanged;
    the output vector changes shape. Revert (a learned scale on log π, or no log π bias): it fails."""
    tok, ctx, lp, live = _head_inputs()
    head = FlatIntentHead(tok.shape[-1], ctx.shape[-1]).double()
    p = torch.softmax(head(tok, ctx, lp, live, K), -1)
    j = 2
    ins = lambda x, v: torch.cat([x[:, :j + 1], v, x[:, j + 1:]], dim=1)   # noqa: E731 copy after j
    tok2 = ins(tok, tok[:, j:j + 1])
    lp2 = lp.clone()
    lp2[:, j] = lp[:, j] - math.log(2.0)
    lp2 = ins(lp2, lp2[:, j:j + 1])
    live2 = ins(live, live[:, j:j + 1])
    p2 = torch.softmax(head(tok2, ctx, lp2, live2, K + 1), -1)
    assert p2.shape[-1] == p.shape[-1] + 1
    assert torch.allclose(p2[:, j] + p2[:, j + 1], p[:, j], atol=1e-12, rtol=0)
    rest = [c for c in range(p.shape[-1]) if c != j]
    rest2 = [c if c < j else c + 1 for c in rest]
    assert torch.allclose(p2[:, rest2], p[:, rest], atol=1e-12, rtol=0)


def test_I1_a_zero_presence_candidate_equals_the_candidate_masked():
    tok, ctx, lp, live = _head_inputs()
    head = FlatIntentHead(tok.shape[-1], ctx.shape[-1]).double()
    lp0 = lp.clone()
    lp0[:, 1] = float("-inf")
    l0 = head(tok, ctx, torch.where(torch.isinf(lp0), torch.full_like(lp0, -1e300), lp0), live, K)
    off = live.clone()
    off[:, 1] = False
    lm = head(tok, ctx, lp, off, K)
    assert torch.allclose(torch.softmax(l0, -1), torch.softmax(lm, -1), atol=1e-12, rtol=0)


def test_a_row_with_no_live_candidate_is_finite():
    tok, ctx, lp, live = _head_inputs(2)
    live[1] = False
    head = FlatIntentHead(tok.shape[-1], ctx.shape[-1]).double()
    out = head(tok, ctx, lp, live, K)
    assert bool(torch.isfinite(out[1]).all()) and bool(torch.isneginf(out[0, 4]))


# ---------------------------------------------------------------------- the consumers' re-expression
def test_the_reexpression_is_the_flat_distribution_and_finite_with_no_switch():
    """α's softmax = the flat move columns + α_SWITCH; β's = p(switch → j) / α_SWITCH. A row with NO
    switch candidate: α_SWITCH = 0 and the GRADIENT is finite. Revert (an unguarded logsumexp): the
    gradient of that row is NaN."""
    g = torch.Generator().manual_seed(1)
    flat = torch.randn(3, flat_width(K), generator=g, dtype=torch.float64)
    flat[:, 4] = float("-inf")
    flat[2, K + 1:] = float("-inf")                  # row 2: no live switch target, no OTHER
    flat.requires_grad_(True)
    a, b = compat_intent_logits(flat, K)
    p = torch.softmax(flat, -1)
    pa = torch.softmax(a, -1)
    assert torch.allclose(pa[:, :K + 1], p[:, :K + 1], atol=1e-14, rtol=0)
    assert torch.allclose(pa[:, -1], p[:, K + 1:].sum(-1), atol=1e-14, rtol=0)
    pb = torch.softmax(b[:2], -1)
    assert torch.allclose(pb, p[:2, K + 1:] / p[:2, K + 1:].sum(-1, keepdim=True), atol=1e-14, rtol=0)
    assert float(pa[2, -1]) == 0.0
    (pa[:, :K].sum() + pa[:, -1].sum()).backward()
    assert bool(torch.isfinite(flat.grad).all())


# --------------------------------------------------------------------------- F-X5-15: seat_live
def test_seat_live_is_applied_by_every_alpha_consumer():
    """A seat whose `seat_live` is 0 carries NO α mass in `threshold_probs`, `IntentMoveCell` and
    `IntentConditionalMoveCell` — equal to the same call with that seat's operands zeroed (the
    reference spends no mass there by construction; `pair_alpha` already masked). Revert (drop the
    `seat_live` multiply in any one): its assert fails, and the no-`seat_live` control differs."""
    from agents.model.intent_conditional import IntentConditionalMoveCell
    from agents.model.intent_move_cell import IntentMoveCell
    from agents.model.intent_threshold import threshold_probs
    g = torch.Generator().manual_seed(3)
    B, k = 4, K + 1                                   # the flat re-expression's K seats + OTHER_move
    al = torch.randn(B, k + 1, generator=g)
    live = torch.ones(B, k)
    live[:, 3] = 0.0
    cells = torch.rand(B, 6, k, 6, generator=g)
    cells0 = cells.clone()
    cells0[:, :, 3] = 0.0
    gate = torch.ones(B, 6, 1)
    idx = torch.zeros(B, dtype=torch.long)
    # threshold_probs
    t_live = threshold_probs(al, cells, gate, idx, seat_live=live)
    t_zero = threshold_probs(al, cells0, gate, idx)
    for a_, b_ in zip(t_live, t_zero):
        assert torch.allclose(a_, b_, atol=1e-6)
    assert not torch.allclose(threshold_probs(al, cells, gate, idx).p_ko, t_zero.p_ko, atol=1e-6)
    # IntentMoveCell (is_status 0 everywhere, so the pure-mass channel alpha_stay is 0 in both)
    imc = IntentMoveCell(8)
    torch.nn.init.normal_(imc.proj.weight, generator=g)
    base = torch.rand(B, 4, 4, generator=g)
    base[..., 0] = 0.0
    db, ds = torch.rand(B, k, generator=g), torch.rand(B, k, generator=g)
    db0, ds0 = db.clone(), ds.clone()
    db0[:, 3], ds0[:, 3] = 0.0, 0.0
    isb, iss = torch.ones(B, 4), torch.ones(B, 4)
    assert torch.allclose(imc(al, base, db, ds, isb, iss, seat_live=live),
                          imc(al, base, db0, ds0, isb, iss), atol=1e-6)
    assert not torch.allclose(imc(al, base, db, ds, isb, iss), imc(al, base, db0, ds0, isb, iss), atol=1e-6)
    # IntentConditionalMoveCell (no boom move requested, so the pure-mass a_stay channel is unused)
    icm = IntentConditionalMoveCell(8)
    torch.nn.init.normal_(icm.proj.weight, generator=g)
    nums = torch.tensor([[89, 57, 85, 182, 94, 1]]).expand(B, -1)    # seat 3 = Protect
    nums0 = nums.clone()
    nums0[:, 3] = 0
    other_u = torch.zeros(B, M)
    other_u[:, 85] = 1.0
    rest = (torch.rand(B, 4, generator=g), torch.rand(B, 1, generator=g), torch.rand(B, 4, generator=g),
            torch.tensor([[68, 243, 228, 182]]).expand(B, -1),         # Counter, Mirror Coat, Pursuit, Protect
            torch.rand(B, 1, generator=g), torch.randn(B, 7, generator=g),
            torch.rand(B, 4, 7, generator=g), torch.zeros(B, dtype=torch.long))
    o_live = icm(al, cells, gate, idx, nums, *rest, seat_live=live, other_u=other_u)
    o_zero = icm(al, cells0, gate, idx, nums0, *rest, other_u=other_u)
    assert torch.allclose(o_live, o_zero, atol=1e-6)
    assert not torch.allclose(icm(al, cells, gate, idx, nums, *rest, other_u=other_u), o_zero, atol=1e-6)


# ------------------------------------------------------------------------- F-X5-8: other_label_rate
def test_other_label_rate_is_computed_over_choices_only():
    from agents.training.instrumented_ppo.flat_intent_fold import flat_intent_fold_tensors
    rows = [(0, 57, -100, 0), (0, 94, -100, 0), (0, 165, -100, 0), (1, 0, 2, 0), (1, 0, -100, 227),
            (1, 0, -100, 242), (2, 0, -100, 0), (2, 0, -100, 0)]
    fi = _fi(len(rows))
    kind, num, slot, sp = _labels(rows)
    flat = torch.zeros(len(rows), flat_width(K), requires_grad=True)
    out = flat_intent_fold_tensors(flat, fi, kind=kind, num=num, switch_slot=slot, switch_species=sp,
                                   intent_coef=0.05, bot_label_weight=1.0)
    m = {k: (float(v), float(w)) for k, (v, w) in out.metrics.items()}
    assert m["opp_intent/other_label_rate"] == (pytest.approx(2 / 6), 1.0)         # 6 choices, 2 OTHER
    assert m["opp_intent/other_move_label_rate"] == (pytest.approx(1 / 3), 1.0)
    assert m["opp_intent/other_species_label_rate"] == (pytest.approx(1 / 3), 1.0)
    assert m["opp_intent/flat_unmodeled_rate"] == (pytest.approx(1 / 6), 1.0)
    assert m["opp_intent/flat_n_supervised"] == (5.0, 1.0)
    out.intent_term.backward()
    assert float(flat.grad[1, other_move_col(K)]) < 0.0 and float(flat.grad[5, other_species_col(K)]) < 0.0


def test_every_flat_fold_key_of_every_opponent_class_renders_uniquely_in_the_stdout_table():
    """F-XC-5's launch blocker: the stdout table (`train_logger.HumanOutputFormat`, sb3's, 36 columns)
    truncates a key and RAISES when two truncate alike — `flat_switch_target_recall_top1_{bot,pool}` killed
    every fixed_mass self-play run at its first dump. Every key the fold emits, with every opponent class
    present, must render as a distinct row (the key is now `flat_switch_tgt_top1`)."""
    import io

    from agents.model.opp_intent import OPP_CLASS_NAMES
    from agents.training import train_logger as TL
    from agents.training.instrumented_ppo.flat_intent_fold import flat_intent_fold_tensors
    rows = [(0, 57, -100, 0), (1, 0, 2, 0), (1, 0, -100, 227), (0, 94, -100, 0)] * (2 * len(OPP_CLASS_NAMES))
    fi = _fi(len(rows))
    kind, num, slot, sp = _labels(rows)
    oc = torch.tensor(sorted(OPP_CLASS_NAMES) * (len(rows) // len(OPP_CLASS_NAMES)))
    out = flat_intent_fold_tensors(torch.zeros(len(rows), flat_width(K)), fi, kind=kind, num=num,
                                   switch_slot=slot, switch_species=sp, opp_class=oc, intent_coef=0.05,
                                   bot_label_weight=1.0)
    log = TL.Logger(None, [TL.HumanOutputFormat(io.StringIO())])
    for key, (v, _w) in out.metrics.items():
        log.record(key, float(v))
    log.dump(step=1)                      # raises ValueError on a truncation collision
    per_class = [k for k in out.metrics if k.endswith(tuple(f"_{n}" for n in OPP_CLASS_NAMES.values()))]
    assert len(per_class) >= 4 * len(OPP_CLASS_NAMES), per_class      # every class's keys were rendered


# --------------------------------------------------------------------- the real fixed_mass policy
@pytest.fixture(scope="module")
def fm():
    from agents.model.hypothesis_set_test import _obs_from_golden, _unperturbed_learner
    from main.train.production_args import production_args
    a = production_args()
    m = _unperturbed_learner(a)
    pol = copy.deepcopy(m.policy).eval()
    fe = pol.features_extractor
    obs = _obs_from_golden(64)
    with torch.no_grad():
        fe(obs)
    return dict(pol=pol, fe=fe, obs=obs, ctx=fe.unpack(obs), hs=fe.last_hypothesis)


@pytest.mark.integration
def test_the_candidate_set_and_its_structural_mask_on_a_real_forward(fm):
    fe, ctx, hs = fm["fe"], fm["ctx"], fm["hs"]
    assert not hasattr(fe, "alpha_head") and not hasattr(fe, "beta_head") and fe.flat_intent_head is not None
    fi, fl = fe.last_flat_intent, fe.last_flat_intent_logits
    k = fe.entity_topk_seats
    assert fi.k == k and fl.shape == (64, flat_width(k))
    mv = hs.moves
    act = ctx.hp_and_active[:, 6:12, -1] > 0.5
    slot_live = ctx.opp_addressable & ~act & (~hs.slot_is_hypothesis | (hs.slot_species > 0))
    assert torch.equal(fi.live[:, :k], mv.seat_live & (mv.seat_revealed | (mv.presence.k > 0)[:, None]))
    assert torch.equal(fi.live[:, other_move_col(k)], mv.other_live)
    assert torch.equal(fi.live[:, slot_col(k):slot_col(k) + 6], slot_live)
    assert torch.equal(fi.live[:, other_species_col(k)], hs.other_live)
    # every hidden slot is a live switch target holding its hypothesis
    assert torch.equal(fi.live[:, slot_col(k):slot_col(k) + 6][hs.slot_is_hypothesis],
                       torch.ones_like(hs.slot_species[hs.slot_is_hypothesis], dtype=torch.bool))
    ids = fi.cand_ids
    assert torch.equal(ids[:, :k], mv.seat_nums)
    assert torch.equal(ids[:, slot_col(k):slot_col(k) + 6],
                       torch.where(hs.slot_is_hypothesis, hs.slot_species, ctx.species_ids[:, 6:12]))
    # −inf exactly where not live; finite where live
    assert torch.equal(torch.isneginf(fl), ~fi.live)
    assert bool(torch.isfinite(fl[fi.live]).all())
    assert bool(fi.live[:, other_species_col(k)].any()) and bool(fi.live[:, other_move_col(k)].any())


@pytest.mark.integration
def test_OTHERs_columns_are_priced_never_a_zero_row(fm):
    """OTHER_species' column in `out_cells` is the OTHER-mode D1 pass (the tail-averaged defender: never
    IMMUNE, P(KO) nulled); in `opp_p_ghost` it is other_tail_probs @ SPECIES_IS_GHOST. OTHER_move's
    seat-axis columns equal the tail contraction of per-move ORACLES (MOVE_ACCURACY, MOVE_PHYS, the type
    chart). Revert (a zero OTHER column): every 'priced' assert fails."""
    fe, hs = fm["fe"], fm["hs"]
    op = fe.damage_op
    x = fe.stash.flat_consumer_ops
    assert x is not None
    k = fe.entity_topk_seats
    live_o = hs.other_live
    oc = x.out_cells                                                          # [B,4,7,5]
    assert oc.shape[2] == 7
    from agents.model.damage_op_layout import _DMG_OMX_IDX_MULT, _DMG_OMX_IDX_PKO
    assert bool((oc[live_o][:, :, 6, _DMG_OMX_IDX_MULT] > 0).any())
    assert float(oc[live_o][..., 6, _DMG_OMX_IDX_PKO].abs().max()) == 0.0     # OTHER keeps the null
    # (a MASKED OTHER's column is whatever slot `other_col` points at — finite, and β puts no mass on it)
    assert bool(torch.isfinite(oc).all()) and bool(torch.isneginf(x.beta[~live_o][:, 6]).all())
    from agents.model.hypothesis_tokens import other_column
    ro = op.stash.x5
    op.stash.x5 = ro.other
    try:
        with torch.no_grad():
            d1 = other_column(op.pairwise_outgoing(_hctx(fm), fe.last_spread_belief,
                                                   species_probs=ro.other.species_probs), ro.other_col, 2)
    finally:
        op.stash.x5 = ro
    assert torch.equal(oc[:, :, 6:7], d1[..., :oc.shape[-1]].to(oc.dtype))
    ghost = x.opp_p_ghost
    assert torch.allclose(ghost[:, 6], hs.other_tail_probs.to(op.SPECIES_IS_GHOST.dtype) @ op.SPECIES_IS_GHOST)
    # OTHER_move: the tail contraction of independent per-move oracles
    u = x.other_u.double()
    lm = fm["fe"].damage_op
    acc = (u * lm.MOVE_ACCURACY.double()[None, :u.shape[1]]).sum(-1)
    phys = (u * lm.MOVE_PHYS.double()[None, :u.shape[1]]).sum(-1)
    pc = x.pair_cells                                                         # [B,6,K+1,6]
    assert pc.shape[2] == k + 1
    mo = fm["fe"].last_flat_intent.live[:, other_move_col(k)]
    assert bool(mo.any())
    assert torch.allclose(pc[mo][:, :, k, 4].double(), acc[mo][:, None].expand(-1, 6), atol=1e-6)
    assert torch.allclose(pc[mo][:, :, k, 5].double(), phys[mo][:, None].expand(-1, 6), atol=1e-6)
    ctx = _hctx(fm)
    t1, t2 = ctx.type1_ids[:, :6], ctx.type2_ids[:, :6]
    amul = lm.ABILITY_DAMAGE_MULT[ctx.ability1_ids[:, :6]]
    mty = lm.MOVE_TYPE_IDX[:u.shape[1]].long()
    per_move = (lm.CHART[t1][:, :, mty] * lm.CHART[t2][:, :, mty]
                * amul[:, :, mty]).double()                                   # [B,6,M]
    tm_oracle = (per_move * u[:, None, :]).sum(-1)
    assert torch.allclose(x.type_mult[:, :, k].double(), tm_oracle, atol=1e-6)
    assert bool((x.type_mult[mo][:, :, k] > 0).all())                         # never IMMUNE as a whole
    assert x.pair_in.shape[2] == k + 1 and torch.equal(x.pair_in[:, :, k, :6], pc[:, :, k])
    assert torch.equal(x.seat_live[:, k], mo.to(x.seat_live.dtype))
    # masked OTHER_move: an exact-zero column (its α mass is −inf there anyway)
    assert float(x.pair_cells[~mo][:, :, k].abs().max()) == 0.0


def _hctx(fm):
    from agents.model.hypothesis_tokens import hypothesis_ctx
    return hypothesis_ctx(fm["ctx"], fm["hs"], fm["fe"].layout)


@pytest.mark.integration
def test_the_flat_loss_trains_the_head_and_never_delta_theta(fm):
    """M10 through the INTENT loss: the flat CE's gradient reaches the pointer's scorer and none of δ_θ
    (the log π bias is detached at its source and again in the head). Teeth: feed the head
    `hs.species.logits`-derived log π and the δ_θ half fails."""
    from agents.training.instrumented_ppo.flat_intent_fold import flat_intent_fold
    pol = copy.deepcopy(fm["pol"]).train()
    fe = pol.features_extractor
    with torch.no_grad():
        fe.hypothesis_builder.delta_out.weight.normal_(0.0, 0.1)
    obs = fm["obs"]
    pol.zero_grad(set_to_none=True)
    fe(obs)
    out = flat_intent_fold(fe, obs, fe.belief_supervision("flat_intent_logits"), intent_coef=0.05,
                           bot_label_weight=0.25)
    assert out is not None and bool(out.intent_present)
    out.intent_term.backward()
    g_head = fe.flat_intent_head.hidden.weight.grad
    assert g_head is not None and float(g_head.abs().max()) > 0.0
    for n, p in fe.hypothesis_builder.named_parameters():
        if n.startswith("delta_"):
            assert p.grad is None or float(p.grad.abs().max()) == 0.0, n


@pytest.mark.integration
def test_production_builds_the_flat_pointer_and_the_alpha_beta_readout_is_gone():
    """The X5 version break deleted the blob path's α / β readout: on a built policy the heads are retired,
    the forward publishes the flat pointer, and no α / β stash is written. FAILS if α / β come back."""
    from agents.model.hypothesis_set_test import _obs_from_golden, _unperturbed_learner
    from main.train.production_args import production_args
    m = _unperturbed_learner(production_args())
    fe = m.policy.features_extractor
    assert fe.flat_intent_head is not None and not hasattr(fe, "alpha_head") and not hasattr(fe, "beta_head")
    assert not any(k.startswith(("features_extractor.alpha_head.", "features_extractor.beta_head."))
                   for k in m.policy.state_dict())
    with torch.no_grad():
        fe(_obs_from_golden(8))
    assert fe.last_flat_intent_logits is not None
    assert not hasattr(fe, "last_alpha_logits") and not hasattr(fe, "last_beta_logits")


# ------------------------------------------------------------------------------ the fullgraph trace
_ON_28 = torch.__version__.startswith("2.8")


@pytest.mark.integration
@pytest.mark.skipif(not _ON_28, reason="fullgraph compile is only required on torch 2.8 (the target)")
def test_the_flat_fold_traces_fullgraph_and_equals_eager(fm):
    """The learner micro-step is compile region R1: the flat fold (targets + CE + every metric) must
    trace with fullgraph=True on the REAL stash shapes and equal eager."""
    import dataclasses

    import torch._dynamo

    from agents.training.instrumented_ppo.flat_intent_fold import flat_intent_fold_tensors
    fe, obs = fm["fe"], fm["obs"]
    fi0 = fe.last_flat_intent
    names = [f.name for f in dataclasses.fields(fi0) if f.name != "k"]

    def fold(flat, kind, num, slot, sp, oc, ci, *fields):
        fi = FlatIntentInputs(k=fi0.k, **dict(zip(names, fields)))
        o = flat_intent_fold_tensors(flat, fi, kind=kind, num=num, switch_slot=slot, switch_species=sp,
                                     opp_class=oc, intent_coef=ci, bot_label_weight=0.25)
        return o.intent_term, o.intent_present, o.metrics

    flat = fe.last_flat_intent_logits.detach().clone().requires_grad_(True)
    flat2 = flat.detach().clone().requires_grad_(True)
    lab = [obs[k].reshape(-1) for k in ("opp_action_kind", "opp_action_num", "opp_switch_slot",
                                        "opp_switch_species", "opp_class")]
    fields = [getattr(fi0, n) for n in names]
    ci = torch.tensor(0.05)
    compiled = torch.compile(fold, fullgraph=True, backend="aot_eager", dynamic=False)
    try:
        t_e, p_e, m_e = fold(flat, *lab, ci, *fields)
        t_c, p_c, m_c = compiled(flat2, *lab, ci, *fields)
        assert abs(float(t_c) - float(t_e)) <= 1e-6 * max(1.0, abs(float(t_e))) and bool(p_c) == bool(p_e)
        assert set(m_c) == set(m_e)
        for key in m_e:
            assert float(m_c[key][1]) == float(m_e[key][1]), key
            assert abs(float(m_c[key][0]) - float(m_e[key][0])) <= 1e-6, key
        ge, = torch.autograd.grad(t_e, [flat])
        gc, = torch.autograd.grad(t_c, [flat2])
        assert torch.allclose(gc, ge, atol=1e-7)
    finally:
        torch._dynamo.reset()
