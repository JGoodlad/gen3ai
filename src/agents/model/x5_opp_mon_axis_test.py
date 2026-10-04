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
