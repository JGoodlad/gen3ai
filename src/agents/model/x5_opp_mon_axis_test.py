"""X5 U3 part 3: the damage op's opponent-MON axis under `--belief-tokens fixed_mass`
(`gen3_x5_belief_tokens_v1`; design_x5_belief_tokens.md §3.4 "Physics", §3.5, §8.3 U3 hand-off).

On a REAL fixed_mass policy and the K9 golden buffer (deterministic: fixed rows, fixed build seed):

* a hypothesis DEFENDER is priced as its species (the per-slot one-hot: its expected type x ability
  multiplier is THAT species' row), and its P(KO) is UN-nulled (a pristine concrete species);
* a hypothesis ATTACKER is live in every attacker family, gated by `opp_addressable` and never by its
  HP cell (zeroing the HP cell moves no gate; dropping addressability zeroes its column);
* the per-mon candidates are each mon's ONE order over its fixed-mass move presence (the active's row
  IS the move group's `w_all`), and the cut joins the rule-8 exclusion;
* `p_pur_vs_us` is a presence-scaled max over the mons (§9 M2 = C); Beat Up's opponent party sum reads
  the hidden-team marginal π / k over EVERY candidate; the bench E5 seats are presence-aware;
* the opponent active's move reinjection reads its DETACHED fixed-mass presence (F-X5-26).

Each test fails on the revert it names.
"""
from __future__ import annotations

import copy
import dataclasses

import pytest
import torch

from agents.model.hypothesis_set import near_tie_rows
from agents.model.hypothesis_tokens import bench_tail_cells, fixed_mass_moves, hypothesis_ctx

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def fm():
    from agents.model.hypothesis_set_test import _obs_from_golden, _unperturbed_learner
    from main.train.production_args import production_args
    a = production_args()
    a.belief_tokens = "fixed_mass"
    m = _unperturbed_learner(a)
    fe = copy.deepcopy(m.policy).features_extractor.eval()
    obs = _obs_from_golden(64)
    with torch.no_grad():
        fe(obs)
    ctx = fe.unpack(obs)
    hs = fe.last_hypothesis
    return dict(fe=fe, obs=obs, ctx=ctx, hs=hs, hctx=hypothesis_ctx(ctx, hs, fe.layout),
                op=fe.damage_op, x5=fe.damage_op.stash.x5)


def _with_roster(op, ro):
    op.stash.x5 = ro
    return op


def test_the_roster_is_wired_from_addressability_and_the_hypothesis_set(fm):
    ctx, hs, x5 = fm["ctx"], fm["hs"], fm["x5"]
    assert x5 is not None
    assert torch.equal(x5.alive.bool(), ctx.opp_addressable)
    assert torch.equal(x5.hyp, hs.slot_is_hypothesis) and bool(x5.hyp.any())
    assert torch.equal(x5.slot_pi[~x5.hyp], torch.ones_like(x5.slot_pi[~x5.hyp]))
    assert bool((x5.slot_pi[x5.hyp] < 1).all()) and bool((x5.slot_pi[x5.hyp] > 0).all())
    for t in (x5.move_w, x5.team_probs, x5.species_probs, x5.slot_pi):
        assert not t.requires_grad                                     # M10: presence is detached


def test_a_hypothesis_defender_is_priced_as_its_species_and_its_pko_is_unnulled(fm):
    """Revert to the T0 marginal (the blob's `species_probs`) and the multiplier is the team average,
    not the species row; revert the P(KO) gate to `revealed` and every hypothesis pko reads 0."""
    from agents.model.damage_kinds import gather_bp, is_priced, typeless_move_type
    op, ctx, hs = fm["op"], fm["ctx"], fm["hs"]
    oc = op.stash.out_cells                                            # [B,4,6,5]
    ids = ctx.our_active_req_move_ids
    mty = typeless_move_type(op, ids, ctx.our_active_req_move_type_ids)
    ar = torch.arange(ctx.batch_size)
    bp = torch.where(ids == op.hp_num, torch.full_like(mty, op.hp_bp, dtype=torch.float32), op.MOVE_BP[ids])
    bp = gather_bp(op, ids, ctx.hp_and_active[ar, ctx.our_active_idx, 0][:, None], bp=bp)
    usable = ctx.our_active_req_move_legal * is_priced(op, ids, bp)                     # [B,4]
    gate = (ctx.hp_and_active[:, 6:, -1].any(1).float()
            * (ctx.hp_and_active[ar, ctx.our_active_idx, 0] > 0).float())               # [B]
    hyp = hs.slot_is_hypothesis
    em = op.SPECIES_EXP_MULT[hs.slot_species]                                            # [B,6,T]
    want = torch.gather(em[:, None].expand(-1, 4, -1, -1), 3,
                        mty[:, :, None, None].expand(-1, -1, 6, 1).long()).squeeze(-1)  # [B,4,6]
    want = want * usable[:, :, None] * gate[:, None, None]
    m = hyp[:, None, :].expand_as(want)
    assert bool(m.any())
    assert torch.allclose(oc[..., 4][m], want[m], atol=1e-6, rtol=1e-6)
    assert bool((oc[..., 3][m] > 0).any())                             # P(KO | species, full HP) is live


def _hp_zeroed(c, hyp):
    """The hypothesis context with every hypothesis slot's HP CELL forced to 0 (nothing else moves)."""
    hp = c.hp_and_active.clone()
    opp_hp = hp[:, 6:, 0]
    hp[:, 6:, 0] = torch.where(hyp, torch.zeros_like(opp_hp), opp_hp)
    return dataclasses.replace(c, hp_and_active=hp)


def test_no_hidden_mon_is_dropped_by_an_hp_gate(fm):
    """Every opponent-axis family keeps a hypothesis slot when its HP cell reads 0 (the gates read
    `opp_addressable`). HP-free families are bit-identical; the attacker families stay live. Revert any
    gate to `hp > 0` and its family loses the hypothesis column here."""
    op, hctx, x5, fe = fm["op"], fm["hctx"], fm["x5"], fm["fe"]
    hyp = x5.hyp
    z = _hp_zeroed(hctx, hyp)
    mb = fe.last_move_belief_logits
    sb = fe.last_spread_belief
    _with_roster(op, x5)
    with torch.no_grad():
        for name, f in (("v", lambda c: op.pairwise_speed(c, sb)), ("t", op.pairwise_trap),
                        ("g", lambda c: op.pairwise_schedule(c)[1]), ("x", lambda c: op.pairwise_entry(c, mb)[1]),
                        ("s1", lambda c: op.discrete_outgoing_status(c, per_pair=True))):
            a, b = f(hctx), f(z)
            assert torch.equal(a, b), name
        # C1b / C2 / C3 share `_believed_attackers`' gate (their rows also need OUR setup / status /
        # recovery slot, which the golden rows may lack): the gate itself is addressability.
        att_gate = op._believed_attackers(z, mb, 6)[7]
        assert torch.equal(att_gate, x5.alive) and bool((att_gate[hyp] > 0).all())
        for name, f, col in (("d4", lambda c: op.pairwise_bench_incoming(c, mb, k_bench=6), 2),
                             ("v", lambda c: op.pairwise_speed(c, sb), 2)):
            cells = f(z)                                               # opp mon on axis `col`
            # the gate column for a hypothesis is open: its attacker/defender cells are not forced 0
            g = cells.abs().movedim(col, 1).flatten(2).amax(-1)        # [B,6]
            assert bool((g[hyp] > 0).any()), name


def test_the_attacker_gate_is_addressability(fm):
    """Drop ONE hypothesis slot's addressability in the roster and its d4 column is exactly 0; keep it
    and the column is live (the gate is `roster.alive`, nothing else)."""
    op, hctx, x5, fe = fm["op"], fm["hctx"], fm["x5"], fm["fe"]
    mb = fe.last_move_belief_logits
    act = torch.nn.functional.one_hot(hctx.opp_active_local, 6).bool()
    cand = x5.hyp & ~act
    b, j = [int(v) for v in torch.nonzero(cand)[0]]
    with torch.no_grad():
        live = _with_roster(op, x5).pairwise_bench_incoming(hctx, mb, k_bench=6)
        alive = x5.alive.clone()
        alive[b, j] = 0.0
        dead = _with_roster(op, dataclasses.replace(x5, alive=alive)).pairwise_bench_incoming(hctx, mb, k_bench=6)
    _with_roster(op, x5)
    assert float(live[b, :, j].abs().max()) > 0
    assert float(dead[b, :, j].abs().max()) == 0.0
    keep = torch.ones_like(alive, dtype=torch.bool)
    keep[b, j] = False
    assert torch.equal(live.movedim(2, 1)[keep], dead.movedim(2, 1)[keep])


def test_attacker_candidates_are_each_mons_one_order_over_its_fixed_mass_presence(fm):
    """Revert `_believed_attackers` to `topk` over sigmoid(logits) and the weights stop being the
    fixed-mass presence (a revealed move reads 1 exactly; the active's row is the move group's)."""
    op, hctx, x5, fe, hs = fm["op"], fm["hctx"], fm["x5"], fm["fe"], fm["hs"]
    _with_roster(op, x5)
    with torch.no_grad():
        w_k = op._believed_attackers(hctx, fe.last_move_belief_logits, 6)[0]
    assert torch.equal(w_k, x5.move_w.gather(-1, x5.move_order[..., :6]))
    # the kernel SELECTS by the roster's order (never re-ranks by weight): swap position 0 with the
    # 7th of every row and the swapped-in candidate is what it prices (a topk would ignore the order)
    o2 = x5.move_order.clone()
    o2[..., 0], o2[..., 6] = x5.move_order[..., 6], x5.move_order[..., 0]
    with torch.no_grad():
        w2 = _with_roster(op, dataclasses.replace(x5, move_order=o2))._believed_attackers(
            hctx, fe.last_move_belief_logits, 6)[0]
        d4a = op.pairwise_bench_incoming(hctx, fe.last_move_belief_logits, k_bench=6)
    _with_roster(op, x5)
    assert torch.equal(w2, x5.move_w.gather(-1, o2[..., :6]))
    with torch.no_grad():
        d4b = op.pairwise_bench_incoming(hctx, fe.last_move_belief_logits, k_bench=6)
    assert not torch.equal(d4a, d4b)                                    # d4 reads the same order
    bi = torch.arange(hctx.batch_size)
    fmv = fixed_mass_moves(hs.moves, fe.last_move_belief_logits[bi, hctx.opp_active_local])
    assert torch.equal(x5.move_w[bi, hctx.opp_active_local], fmv.w_all)
    # per mon the presence sums to its group mass 4 over the live candidates + revealed (HP aside)
    rev = fm["ctx"].all_move_ids[:, 6:, :]
    no_hp = ~(rev == 237).any(-1)
    s = x5.move_w.sum(-1)
    assert torch.allclose(s[no_hp], torch.full_like(s[no_hp], 4.0), atol=1e-4)


def test_p_pur_vs_us_is_a_presence_scaled_max_over_the_opponent_mons(fm):
    """§9 M2 = C over MONS: max_j slot_pi_j · P(Pursuit | j) · alive_j. Revert to the blob's sigmoid
    max and a hypothesis counts as a whole mon."""
    from agents.model.damage_tables import _pursuit_num
    op, hctx, x5, fe = fm["op"], fm["hctx"], fm["x5"], fm["fe"]
    pur = _pursuit_num()
    with torch.no_grad():
        our, _ = _with_roster(op, x5).pairwise_entry(hctx, fe.last_move_belief_logits)
    want = (x5.slot_pi * x5.move_w[:, :, pur] * x5.alive).amax(-1)
    alive_i = (hctx.hp_and_active[:, :6, 0] > 0).float()
    assert torch.equal(our[..., 1], want[:, None] * alive_i)
    # presence matters: with every hypothesis at presence 1 the read can only rise
    with torch.no_grad():
        our1, _ = _with_roster(op, dataclasses.replace(x5, slot_pi=torch.ones_like(x5.slot_pi))).pairwise_entry(
            hctx, fe.last_move_belief_logits)
    _with_roster(op, x5)
    assert bool((our1[..., 1] >= our[..., 1]).all())


def test_beat_up_reads_the_hidden_team_marginal_over_every_candidate(fm):
    """Σ_hidden = Σ_s π_s · baseAtk(s) over hypotheses AND the tail. Revert the override and the sum is
    over the six hypothesis one-hots only (the tail's mass drops out)."""
    from agents.model.damage_kinds import _BS_ATK_COL, _healthy, beatup_party_opp
    op, hctx, x5, hs = fm["op"], fm["hctx"], fm["x5"], fm["hs"]
    _with_roster(op, x5)
    S, N = beatup_party_opp(op, hctx, x5.species_probs)
    hidden = hctx.opp_believed_mask.float()
    ok = _healthy(hctx, slice(6, 12)) * (1.0 - hidden)
    atk = op.BASE_STATS[hctx.species_ids[:, 6:12]][..., _BS_ATK_COL]
    want = (ok * atk).sum(-1) + hs.species.pi.to(atk.dtype) @ op.BASE_STATS[:, _BS_ATK_COL]
    assert torch.allclose(S, want, rtol=1e-5)
    assert torch.equal(N, ok.sum(-1) + hidden.sum(-1))
    hyp_only = (ok * atk).sum(-1) + (x5.species_probs.sum(1) @ op.BASE_STATS[:, _BS_ATK_COL])
    k_pos = hidden.sum(-1) > 0
    assert not torch.allclose(S[k_pos], hyp_only[k_pos], rtol=1e-4)


def test_bench_e5_seats_are_presence_aware(fm):
    """Each mon's tail = its moves beyond rank K of ITS one order, summed presence (unclamped) and a
    presence-scaled worst case. Revert to the blob's sigmoid top-K and p_tail is clamped / sigmoid."""
    op, x5 = fm["op"], fm["x5"]
    cells = bench_tail_cells(x5, 6, op.MOVE_BP, op.MOVE_ACCURACY, op.MOVE_PHYS)
    tail = x5.move_rank >= 6
    assert torch.allclose(cells[..., 0], torch.where(tail, x5.move_w, torch.zeros_like(x5.move_w)).sum(-1))
    # a mon with fewer than four revealed moves at k = 4 − r carries tail mass (the expected count)
    assert bool((cells[..., 0][x5.hyp] > 0).all())


def test_the_per_mon_selection_cut_joins_the_rule8_exclusion(fm):
    hs = fm["hs"]
    assert hs.slot_moves_tie_gap is not None and not bool(near_tie_rows(hs).any())
    tied = dataclasses.replace(hs, slot_moves_tie_gap=torch.where(
        torch.arange(hs.slot_moves_tie_gap.shape[0]) == 3, torch.zeros_like(hs.slot_moves_tie_gap),
        hs.slot_moves_tie_gap))
    assert near_tie_rows(tied).tolist() == [i == 3 for i in range(hs.slot_moves_tie_gap.shape[0])]


def test_the_active_reinjection_reads_its_detached_fixed_mass_presence(fm, monkeypatch):
    """F-X5-26 (ORCHESTRATOR): the opponent active's reinjection row is `FixedMassMoves.w_all`
    (detached); every other row stays sigmoid(logits) — so a backward from the reinjected tokens puts
    ZERO gradient into the move posterior's active row. Revert (weights=None) and it is nonzero."""
    fe, obs = fm["fe"], fm["obs"]
    seen = {}
    real = fe.move_belief.reinject_moves

    def spy(opp_tokens, apply_mask, emb, logits, weights=None):
        seen["logits"], seen["weights"] = logits, weights
        return real(opp_tokens, apply_mask, emb, logits, weights=weights)

    monkeypatch.setattr(fe.move_belief, "reinject_moves", spy)
    with torch.no_grad():
        fe(obs)
    ctx = fe.unpack(obs)
    bi = torch.arange(ctx.batch_size)
    w = seen["weights"]
    assert w is not None and not w.requires_grad
    fmv = fixed_mass_moves(fe.last_hypothesis.moves, seen["logits"][bi, ctx.opp_active_local])
    assert torch.equal(w[bi, ctx.opp_active_local], fmv.w_all.to(w.dtype))
    other = torch.ones(ctx.batch_size, 6, dtype=torch.bool)
    other[bi, ctx.opp_active_local] = False
    assert torch.equal(w[other], torch.sigmoid(seen["logits"])[other])
    # gradient: the reinjected tokens do not depend on the active row's logits
    lg = seen["logits"].detach().clone().requires_grad_(True)
    weights = torch.where(torch.nn.functional.one_hot(ctx.opp_active_local, 6).bool().unsqueeze(-1),
                          fmv.w_all.unsqueeze(1), torch.sigmoid(lg))
    tok = torch.randn(ctx.batch_size, 6, fe.move_belief.norm.normalized_shape[0])
    out = real(tok, torch.ones(ctx.batch_size, 6, dtype=torch.bool), fe.embeddings.move_embedding, lg,
               weights=weights)
    out.sum().backward()
    assert float(lg.grad[bi, ctx.opp_active_local].abs().max()) == 0.0
    assert float(lg.grad[other].abs().max()) > 0.0


# ============================================================================ OTHER_species' physics (M3 (c))

def _other_cells(fm):
    fe = fm["fe"]
    return fe._other_edge_cells(fm["hctx"], fm["x5"], fe.last_spread_belief, fe.edge_bias.families)


def test_other_any_is_one_minus_the_product_and_enters_the_pursuit_max(fm):
    """F4 (b): OTHER's max-site presence is 1 − Π_tail(1 − π) (fp64 reference), in [0, 1], 0 when masked;
    `p_pur_vs_us` takes it with OTHER's averaged Pursuit presence. Revert to the mass and it leaves [0,1]."""
    from agents.model.damage_tables import _pursuit_num
    hs, x5, op, fe = fm["hs"], fm["x5"], fm["op"], fm["fe"]
    sp = hs.species
    in_tail = sp.cand & (hs.rank >= sp.k.unsqueeze(-1)) & (sp.k > 0).unsqueeze(-1)
    ref = 1.0 - torch.where(in_tail, 1.0 - sp.pi.double(), torch.ones_like(sp.pi.double())).prod(-1)
    ref = torch.where(hs.other_live, ref, torch.zeros_like(ref))
    assert torch.allclose(x5.other_any.double(), ref, atol=1e-6)
    assert bool((x5.other_any >= 0).all()) and bool((x5.other_any <= 1).all())
    assert bool((hs.other_mass[hs.other_live] > 1).any())             # the mass is NOT a presence
    with torch.no_grad():
        our, _ = _with_roster(op, x5).pairwise_entry(fm["hctx"], fe.last_move_belief_logits)
    alive_i = (fm["hctx"].hp_and_active[:, :6, 0] > 0).float()
    want = torch.maximum((x5.slot_pi * x5.move_w[:, :, _pursuit_num()] * x5.alive).amax(-1),
                         x5.other_any * x5.other_pursuit)
    assert torch.equal(our[..., 1], want[:, None] * alive_i)


def test_other_defender_is_the_averaged_tail_construction_and_never_immune(fm):
    """OTHER's D1 column = `_outgoing_matrix`'s expected-latent defender on P_tail: its type x ability
    multiplier is P_tail @ SPECIES_EXP_MULT (never 0 = IMMUNE), its P(KO) NULLED (the blob's rule for an
    averaged defender), its revealed bit 0. Revert OTHER's `species_probs` to a hypothesis one-hot and the
    multiplier is that species', not the tail's."""
    from agents.model.damage_kinds import gather_bp, is_priced, typeless_move_type
    op, ctx, hs, x5 = fm["op"], fm["ctx"], fm["hs"], fm["x5"]
    with torch.no_grad():
        oc = _other_cells(fm)
    assert set(oc) == {"d1", "c1", "c3", "d4", "v"}
    assert fm["op"].stash.x5 is x5                                     # the per-forward roster is restored
    d1 = oc["d1"][:, :, 0, :]                                          # [B,4,6] cells
    ids = ctx.our_active_req_move_ids
    mty = typeless_move_type(op, ids, ctx.our_active_req_move_type_ids)
    ar = torch.arange(ctx.batch_size)
    bp = torch.where(ids == op.hp_num, torch.full_like(mty, op.hp_bp, dtype=torch.float32), op.MOVE_BP[ids])
    bp = gather_bp(op, ids, ctx.hp_and_active[ar, ctx.our_active_idx, 0][:, None], bp=bp)
    usable = ctx.our_active_req_move_legal * is_priced(op, ids, bp)
    gate = (ctx.hp_and_active[:, 6:, -1].any(1).float() * (ctx.hp_and_active[ar, ctx.our_active_idx, 0] > 0).float())
    em = hs.other_tail_probs @ op.SPECIES_EXP_MULT                     # [B,T]
    want = em.gather(1, mty.long()) * usable * gate[:, None]
    live = hs.other_live[:, None].expand_as(want)
    assert torch.allclose(d1[..., 4][live], want[live], atol=1e-6, rtol=1e-5)
    on = live & (usable * gate[:, None] > 0)
    assert bool(on.any()) and bool((d1[..., 4][on] > 0).all())         # never IMMUNE
    lr = hs.other_live                                                 # (a masked OTHER's edge is zeroed by EdgeBias)
    assert float(d1[lr][..., 3].abs().max()) == 0.0                    # P(KO) nulled (averaged defender)
    assert float(d1[lr][..., 5].abs().max()) == 0.0                    # not revealed


def test_other_is_a_strict_refinement_one_hot_tail_equals_the_species_column(fm):
    """The averaged construction's PARITY: with P_tail = ONE species s′ — deliberately NOT the hypothesis in
    OTHER's column — and OTHER's moves = that column's move row, OTHER's attacker (D4) and defender (D1, but
    its nulled P(KO)) columns equal the column computed with s′ AS the hypothesis there (its dex row in the
    hypothesis context). So E_tail[base], E_tail[STAB] and the averaged defender are the species-exact
    computation on a one-hot, and agree with the hypothesis context's types / stats. Revert the attacker's
    base stats or STAB to the slot's gathered values and D4 reads the slot's species s, not s′."""
    from agents.model.hypothesis_tokens import other_column, other_roster
    op, ctx, hs, x5, fe = fm["op"], fm["ctx"], fm["hs"], fm["x5"], fm["fe"]
    col = x5.other_col
    bi = torch.arange(ctx.batch_size)
    S = hs.other_tail_probs.shape[-1]
    s = hs.slot_species[bi, col]
    valid = fe.hypothesis_builder.species_valid
    s2 = torch.where(valid[(s % 386) + 1], (s % 386) + 1, torch.full_like(s, 143))      # a different species
    s2 = torch.where(s2 == s, torch.full_like(s, 248), s2)
    # the reference: s′ as the hypothesis in OTHER's column (its dex row; every mask the real one)
    oh_col = torch.nn.functional.one_hot(col, 6).bool()
    hs_ref = dataclasses.replace(hs, slot_species=torch.where(oh_col, s2[:, None], hs.slot_species),
                                 slot_rows=torch.where(oh_col[..., None], fe.hypothesis_builder.dex_rows[s2][:, None],
                                                       hs.slot_rows))
    hctx_ref = hypothesis_ctx(ctx, hs_ref, fe.layout)
    sp_ref = torch.where(oh_col[..., None], torch.nn.functional.one_hot(s2, S).to(x5.species_probs.dtype)[:, None],
                         x5.species_probs)
    ro_ref = dataclasses.replace(x5, species_probs=sp_ref)
    # OTHER on the ORIGINAL hypothesis context, tail = s′, moves = the column's move row
    hs1 = dataclasses.replace(hs, other_tail_probs=torch.nn.functional.one_hot(s2, S).to(hs.other_tail_probs.dtype))
    o = other_roster(x5, hs1, fe.hypothesis_builder, fe.move_belief, op.BASE_STATS, op.SPECIES_TYPE,
                     op.SPECIES_SPREAD_PRIOR, int(op.CHART.shape[-1]), 4).other
    h3 = x5.hyp.unsqueeze(-1)
    o = dataclasses.replace(o, move_w=torch.where(h3, x5.move_w[bi, col].unsqueeze(1), o.move_w),
                            move_order=torch.where(h3, x5.move_order[bi, col].unsqueeze(1), o.move_order))
    mb, sb, live = fe.last_move_belief_logits, fe.last_spread_belief, hs.other_live
    with torch.no_grad():
        d4_r = other_column(_with_roster(op, ro_ref).pairwise_bench_incoming(hctx_ref, mb, k_bench=6), col, 2)
        d1_r = other_column(op.pairwise_outgoing(hctx_ref, sb, species_probs=sp_ref), col, 2)
        d4_o = other_column(_with_roster(op, o).pairwise_bench_incoming(fm["hctx"], mb, k_bench=6,
                                                                        species_probs=o.species_probs), col, 2)
        d1_o = other_column(op.pairwise_outgoing(fm["hctx"], sb, species_probs=o.species_probs), col, 2)
    _with_roster(op, x5)
    assert bool(live.any()) and bool((s2 != s).all())
    assert torch.allclose(d4_o[live], d4_r[live], atol=1e-5, rtol=1e-5)
    assert not torch.allclose(d4_o[live], other_column(op.pairwise_bench_incoming(fm["hctx"], mb, k_bench=6),
                                                       col, 2)[live], atol=1e-5)   # and differs from species s
    keep = [0, 1, 2, 4]                                                # low, high, crit, type_mult (not pko / revealed)
    assert torch.allclose(d1_o[live][..., keep], d1_r[live][..., keep], atol=1e-5, rtol=1e-5)
    # OTHER's P(KO) is nulled by construction: the OTHER-mode roster marks its slots NOT concrete
    assert float(x5.other.concrete[x5.hyp].abs().max()) == 0.0
    assert torch.equal(x5.other.concrete[~x5.hyp], torch.ones_like(x5.other.concrete[~x5.hyp]))


def test_other_alone_can_carry_the_pursuit_max(fm):
    """With no slot carrying Pursuit, `p_pur_vs_us` IS OTHER's other_any · P(Pursuit | OTHER). Revert
    (OTHER out of the max) and it reads 0."""
    from agents.model.damage_tables import _pursuit_num
    op, hctx, x5, fe = fm["op"], fm["hctx"], fm["x5"], fm["fe"]
    pur = _pursuit_num()
    mw = x5.move_w.clone()
    mw[:, :, pur] = 0.0
    with torch.no_grad():
        our, _ = _with_roster(op, dataclasses.replace(x5, move_w=mw)).pairwise_entry(hctx, fe.last_move_belief_logits)
    _with_roster(op, x5)
    alive_i = (hctx.hp_and_active[:, :6, 0] > 0).float()
    want = (x5.other_any * x5.other_pursuit)[:, None] * alive_i
    assert torch.equal(our[..., 1], want) and float(want.max()) > 0


def test_other_edges_are_written_at_the_other_seat_and_gated_by_liveness():
    """EdgeBias writes OTHER's cells with the family's OWN map at (the family's seats, the OTHER seat) and
    the transpose; a masked OTHER contributes exactly 0. Revert (no OTHER loop) and the column stays 0."""
    from agents.model.team_transformer import EdgeBias, _EDGE_FAMILIES
    from agents.model.arch_constants import TRANSFORMER_N_HEADS as H
    torch.manual_seed(0)
    eb = EdgeBias("d1,d4")
    for lin in (eb.d1_map, eb.d4_map):
        torch.nn.init.normal_(lin.weight)
        torch.nn.init.normal_(lin.bias)
    B, n, base, oi = 3, 40, 13, 30
    oc = {"d1": torch.randn(B, 4, 1, _EDGE_FAMILIES["d1"]), "d4": torch.randn(B, 6, 1, _EDGE_FAMILIES["d4"])}
    live = torch.tensor([True, False, True])
    with torch.no_grad():
        out = eb(torch.zeros(B, H, n, n), base, {}, None, other_cells=oc, other_index=oi, other_live=live)
        m1 = eb.d1_map(oc["d1"]).permute(0, 3, 1, 2)                   # [B,2H,4,1]
        m4 = eb.d4_map(oc["d4"]).permute(0, 3, 1, 2)                   # [B,2H,6,1]
    lv = live.float()[:, None, None, None]
    assert torch.allclose(out[:, :, base:base + 4, oi:oi + 1], m1[:, :H] * lv)
    assert torch.allclose(out[:, :, oi:oi + 1, base:base + 4], (m1[:, H:] * lv).transpose(-1, -2))
    assert torch.allclose(out[:, :, 0:6, oi:oi + 1], m4[:, :H] * lv)
    assert float(out[1].abs().max()) == 0.0
