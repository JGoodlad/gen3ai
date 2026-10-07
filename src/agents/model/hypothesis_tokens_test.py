"""X5 U3's class-E presence semantics (`gen3_x5_belief_tokens_v1`; design_x5_belief_tokens.md §3.5).

Every EXPECTATION-class reduction over opponent tokens adds ``log π`` of each key to every query's
logit (ToMe's proportional attention). Two invariances pin each site, deterministically:

* **I1, zero presence = masking.** A key at π = 0 (``log π`` = ``MASKED_LOG_PRESENCE``) gives the
  SAME output as that key masked — BIT-identical in fp32 and fp64 (both weights underflow to exactly
  0, so there is no boundary).
* **I2, split = whole ("mass of copies").** One key at presence ``w`` gives the same output as two
  IDENTICAL copies whose presences sum to ``w`` (w/2 each): the copies' total MASS equals the whole's.
  To 1e-12 in fp64 and 1e-5 (relative to the output scale) in fp32. The second copy is placed in a
  hypothesis slot holding OTHER's own token, so the test goes through each site's REAL forward API.

Sites: the TeamTransformer's key bias, CLSPool's `their_cls` and `value_cls`, HiddenOppBeliefPool,
`value_entity_pool` (UnifiedValueReadout). Reverting any site's bias (dropping the log-presence)
fails its I2 test; reverting the float mask to the bool one fails its I1 / I2 test (the presence is
then never read). Plus the extractor-level wiring on a REAL fixed_mass policy: the tokens, OTHER's
seat, the species-specific T0 reads, M10 (π detached), and blob untouched.
"""
from __future__ import annotations

import copy
import math
from types import SimpleNamespace

import pytest
import torch

from agents.model.arch_constants import D_MODEL
from agents.model.hypothesis_set import MASKED_LOG_PRESENCE
from agents.model.hypothesis_tokens import OppPresence
from agents.model.pools import CLSPool, HiddenOppBeliefPool
from agents.model.value_readouts import UnifiedValueReadout

B = 16


def _ctx(dtype, our_fainted=5, opp_fainted=4, gin=50):
    g = torch.Generator().manual_seed(7)
    fm_our = torch.zeros(B, 6, dtype=torch.bool)
    fm_our[:, our_fainted] = True
    fm_opp = torch.zeros(B, 6, dtype=torch.bool)
    fm_opp[:, opp_fainted] = True
    return SimpleNamespace(
        batch_size=B, device=torch.device("cpu"), fainted_mask_ours=fm_our, fainted_mask_opp=fm_opp,
        all_fainted=torch.cat([fm_our, fm_opp], 1), our_active_idx=torch.zeros(B, dtype=torch.long),
        our_ctx_raw=torch.randn(B, gin, generator=g, dtype=dtype),
        opp_ctx_raw=torch.randn(B, gin, generator=g, dtype=dtype),
        non_matchup_rest=torch.randn(B, 1, generator=g, dtype=dtype))


def _tokens(dtype, seed=0):
    g = torch.Generator().manual_seed(seed)
    our = torch.randn(B, 6, D_MODEL, generator=g, dtype=dtype)
    their = torch.randn(B, 6, D_MODEL, generator=g, dtype=dtype)
    other = torch.randn(B, D_MODEL, generator=g, dtype=dtype)
    lp = torch.log(torch.rand(B, 6, generator=g, dtype=dtype).clamp(min=0.05))
    lp[:, 0] = 0.0                                                     # a revealed slot
    w = torch.rand(B, generator=g, dtype=dtype) * 3.0 + 0.2            # OTHER's mass (may exceed 1)
    return our, their, other, lp, w


def _close(a, b, dtype):
    if dtype == torch.float64:
        return float((a - b).abs().max()) <= 1e-12 * max(1.0, float(a.abs().max()))
    return float((a - b).abs().max()) <= 1e-5 * max(1.0, float(a.abs().max()))


def _split_configs(their, other, lp, w, j=2):
    """(A, B): A = slot j masked + OTHER at mass w; B = slot j := a copy of OTHER at w/2 + OTHER at
    w/2. The slot-j mask flip is passed back so each site masks the same key in A."""
    live = torch.ones(B, dtype=torch.bool)
    pa = OppPresence(slot_log_pi=lp, other_out=other, other_log_mass=torch.log(w), other_live=live)
    their_b = their.clone()
    their_b[:, j] = other
    lp_b = lp.clone()
    lp_b[:, j] = torch.log(w / 2)
    pb = OppPresence(slot_log_pi=lp_b, other_out=other, other_log_mass=torch.log(w / 2), other_live=live)
    return pa, their_b, pb


# ----------------------------------------------------------------------------------- CLSPool
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_cls_pools_I1_zero_presence_equals_masking(dtype):
    torch.manual_seed(0)
    pool = CLSPool({}).to(dtype).eval()
    ctx = _ctx(dtype)
    our, their, other, lp, w = _tokens(dtype)
    j = 3
    live = torch.ones(B, dtype=torch.bool)
    lp0 = lp.clone()
    lp0[:, j] = MASKED_LOG_PRESENCE
    p0 = OppPresence(lp0, other, torch.log(w), live)
    ctx_m = copy.copy(ctx)
    ctx_m.fainted_mask_opp = ctx.fainted_mask_opp.clone()
    ctx_m.fainted_mask_opp[:, j] = True
    ctx_m.all_fainted = torch.cat([ctx.fainted_mask_ours, ctx_m.fainted_mask_opp], 1)
    with torch.no_grad():
        a = pool(our, their, ctx, presence=p0)
        b = pool(our, their, ctx_m, presence=OppPresence(lp, other, torch.log(w), live))
    assert torch.equal(a[1], b[1]) and torch.equal(a[3], b[3])        # their_cls, value_cls
    # OTHER at π = 0 equals OTHER masked, too
    with torch.no_grad():
        c = pool(our, their, ctx, presence=OppPresence(lp, other, torch.full_like(w, MASKED_LOG_PRESENCE), live))
        d = pool(our, their, ctx, presence=OppPresence(lp, other, torch.log(w), ~live))
    assert torch.equal(c[1], d[1]) and torch.equal(c[3], d[3])


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_cls_pools_I2_split_equals_whole(dtype):
    torch.manual_seed(0)
    pool = CLSPool({}).to(dtype).eval()
    ctx = _ctx(dtype)
    our, their, other, lp, w = _tokens(dtype)
    j = 2
    pa, their_b, pb = _split_configs(their, other, lp, w, j)
    ctx_a = copy.copy(ctx)
    ctx_a.fainted_mask_opp = ctx.fainted_mask_opp.clone()
    ctx_a.fainted_mask_opp[:, j] = True
    ctx_a.all_fainted = torch.cat([ctx.fainted_mask_ours, ctx_a.fainted_mask_opp], 1)
    with torch.no_grad():
        a = pool(our, their, ctx_a, presence=pa)
        b = pool(our, their_b, ctx, presence=pb)
    assert _close(a[1], b[1], dtype) and _close(a[3], b[3], dtype)
    # teeth: without the presence (the blob's bool mask) the split is NOT the whole
    with torch.no_grad():
        nb = pool(our, their_b, ctx, presence=OppPresence(torch.zeros_like(lp), other,
                                                          torch.zeros_like(w), pb.other_live))
    assert not _close(a[1], nb[1], dtype)


# ----------------------------------------------------------------------------------- HiddenOppBeliefPool
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_hidden_opp_belief_pool_I1_and_I2(dtype):
    torch.manual_seed(0)
    pool = HiddenOppBeliefPool(6).to(dtype).eval()
    ctx = _ctx(dtype)
    our, their, other, lp, w = _tokens(dtype)
    live = torch.ones(B, dtype=torch.bool)
    j = 3
    lp0 = lp.clone()
    lp0[:, j] = MASKED_LOG_PRESENCE
    af_m = ctx.all_fainted.clone()
    af_m[:, 6 + j] = True
    allt = torch.cat([our, their], 1)
    with torch.no_grad():
        a = pool(allt, ctx.all_fainted, B, presence=OppPresence(lp0, other, torch.log(w), live))
        b = pool(allt, af_m, B, presence=OppPresence(lp, other, torch.log(w), live))
    assert torch.equal(a, b)                                            # I1
    j = 2
    pa, their_b, pb = _split_configs(their, other, lp, w, j)
    af_a = ctx.all_fainted.clone()
    af_a[:, 6 + j] = True
    with torch.no_grad():
        a = pool(allt, af_a, B, presence=pa)
        b = pool(torch.cat([our, their_b], 1), ctx.all_fainted, B, presence=pb)
    assert _close(a, b, dtype)                                          # I2


# ----------------------------------------------------------------------------------- value_entity_pool
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_value_entity_pool_I1_and_I2(dtype):
    torch.manual_seed(0)
    uvr = UnifiedValueReadout(12, full=True).to(dtype).eval()
    torch.nn.init.normal_(uvr.out_proj.weight)        # zero-init would make every output 0 (vacuous)
    ctx = _ctx(dtype)
    our, their, other, lp, w = _tokens(dtype)
    g = torch.Generator().manual_seed(3)
    op_rows = torch.randn(B, 6, 12, generator=g, dtype=dtype)
    op_alive = torch.ones(B, 6, dtype=dtype)
    glob = torch.randn(B, D_MODEL, generator=g, dtype=dtype)
    bel = torch.randn(B, 6, D_MODEL, generator=g, dtype=dtype)
    live = torch.ones(B, dtype=torch.bool)
    kw = dict(global_row=glob, belief_rows=bel)
    j = 3
    lp0 = lp.clone()
    lp0[:, j] = MASKED_LOG_PRESENCE
    af_m = ctx.all_fainted.clone()
    af_m[:, 6 + j] = True
    with torch.no_grad():
        a = uvr(our, their, ctx.all_fainted, op_rows, op_alive,
                presence=OppPresence(lp0, other, torch.log(w), live), **kw)
        b = uvr(our, their, af_m, op_rows, op_alive,
                presence=OppPresence(lp, other, torch.log(w), live), **kw)
    assert torch.equal(a, b)                                            # I1
    j = 2
    pa, their_b, pb = _split_configs(their, other, lp, w, j)
    af_a = ctx.all_fainted.clone()
    af_a[:, 6 + j] = True
    with torch.no_grad():
        a = uvr(our, their, af_a, op_rows, op_alive, presence=pa, **kw)
        b = uvr(our, their_b, ctx.all_fainted, op_rows, op_alive, presence=pb, **kw)
    assert _close(a, b, dtype)                                          # I2


# ----------------------------------------------------------------------------------- TeamTransformer
@pytest.fixture(scope="module")
def layout():
    from agents.observation.state_encoder import Gen3ObservationEncoder, load_mappings
    return Gen3ObservationEncoder(load_mappings()).get_layout()


def _transformer(layout, dtype):
    from agents.model.team_transformer import TeamTransformer
    torch.manual_seed(0)
    tt = TeamTransformer(layout).to(dtype).eval()
    return tt


def _tt_inputs(tt, dtype):
    gin = tt._global_token_input_dim
    ctx = _ctx(dtype, gin=1)
    g = torch.Generator().manual_seed(11)
    ctx.our_ctx_raw = torch.randn(B, gin // 2, generator=g, dtype=dtype)
    ctx.opp_ctx_raw = torch.randn(B, gin // 2, generator=g, dtype=dtype)
    ctx.non_matchup_rest = torch.randn(B, gin - 2 * (gin // 2), generator=g, dtype=dtype)
    role = torch.randn(B, 12, D_MODEL, generator=g, dtype=dtype)
    other = torch.randn(B, 1, D_MODEL, generator=g, dtype=dtype)
    return ctx, role, other


def _tt_run(tt, role, ctx, other, other_pad, klp):
    from agents.model.extractor_ctx import TOKEN_TYPE_THEIR_TEAM
    types = torch.full((1,), TOKEN_TYPE_THEIR_TEAM, dtype=torch.long)
    with torch.no_grad():
        return tt(role, ctx, None, extra=(other, types, other_pad), key_log_presence=klp)


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_transformer_key_bias_I1_zero_presence_equals_masking(layout, dtype):
    tt = _transformer(layout, dtype)
    ctx, role, other = _tt_inputs(tt, dtype)
    j = 3
    n = 2 * 6 + 1 + 1
    klp = torch.zeros(B, n, dtype=dtype)
    klp[:, 6:12] = torch.log(torch.rand(B, 6, dtype=dtype).clamp(min=0.05))
    klp[:, 6 + j] = MASKED_LOG_PRESENCE
    klp[:, 13] = math.log(1.7)
    pad = torch.zeros(B, 1, dtype=torch.bool)
    a = _tt_run(tt, role, ctx, other, pad, klp)
    ctx_m = copy.copy(ctx)
    ctx_m.fainted_mask_opp = ctx.fainted_mask_opp.clone()
    ctx_m.fainted_mask_opp[:, j] = True
    klp_m = klp.clone()
    klp_m[:, 6 + j] = 0.3                                              # ignored under the mask
    b = _tt_run(tt, role, ctx_m, other, pad, klp_m)
    keep = [i for i in range(6) if i != j]
    assert torch.equal(a[0], b[0]) and torch.equal(a[1][:, keep], b[1][:, keep]) and torch.equal(a[2], b[2])


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_transformer_key_bias_I2_split_equals_whole(layout, dtype):
    tt = _transformer(layout, dtype)
    ctx, role, other = _tt_inputs(tt, dtype)
    n = 2 * 6 + 1 + 1
    w = torch.rand(B, dtype=dtype) * 3.0 + 0.2
    base = torch.zeros(B, n, dtype=dtype)
    base[:, 6:12] = torch.log(torch.rand(B, 6, dtype=dtype).clamp(min=0.05))
    j = 2
    # A: slot j masked, OTHER (the extra seat) at mass w
    ctx_a = copy.copy(ctx)
    ctx_a.fainted_mask_opp = ctx.fainted_mask_opp.clone()
    ctx_a.fainted_mask_opp[:, j] = True
    ka = base.clone()
    ka[:, 13] = torch.log(w)
    pad = torch.zeros(B, 1, dtype=torch.bool)
    a = _tt_run(tt, role, ctx_a, other, pad, ka)
    # B: slot j := OTHER's token at w/2, OTHER at w/2 (same token type, so identical inputs)
    role_b = role.clone()
    role_b[:, 6 + j] = other[:, 0]
    kb = base.clone()
    kb[:, 6 + j] = torch.log(w / 2)
    kb[:, 13] = torch.log(w / 2)
    b = _tt_run(tt, role_b, ctx, other, pad, kb)
    keep = [i for i in range(6) if i != j]
    assert _close(a[0], b[0], dtype) and _close(a[1][:, keep], b[1][:, keep], dtype)
    assert _close(a[2], b[2], dtype)                                   # OTHER's own output
    assert _close(b[1][:, j], b[2][:, 0], dtype)                       # the two copies agree
    # teeth: with no key bias the split is NOT the whole
    a0 = _tt_run(tt, role, ctx_a, other, pad, None)
    b0 = _tt_run(tt, role_b, ctx, other, pad, None)
    assert not _close(a0[0], b0[0], dtype)


# ----------------------------------------------------------------------------------- the margin rule
def test_sort_head_margin_counts_only_genuine_adjacent_pi_pairs():
    from agents.model.selection_sites import Rule
    from agents.training.rust_rollout.tie_margins import site_margin
    inf = float("inf")
    # row 0: revealed keys −2 tie exactly (structural) then −0.5, −0.5 + 1e-7 (a genuine near-tie);
    # row 1: two exact zeros (a k = 0 row, structural) then non-candidates; row 2: well separated.
    neg = torch.tensor([[-2.0, -2.0, -0.5, -0.5 + 1e-7, -0.1, inf, inf, inf],
                        [0.0, 0.0, inf, inf, inf, inf, inf, inf],
                        [-0.9, -0.6, -0.3, -0.1, inf, inf, inf, inf]], dtype=torch.float64)
    g = site_margin(Rule("sort_head", head=7, zero_exact=True), "argsort", (neg,), {"dim": -1})
    per = g.amin(-1)
    assert float(per[0]) < 1e-6 and math.isinf(float(per[1])) and float(per[2]) > 0.3


# ----------------------------------------------------------------------------------- the real policy
@pytest.fixture(scope="module")
def x5():
    """The production learner (X5's hypothesis tokens, the only belief representation), UNPERTURBED."""
    from agents.model.hypothesis_set_test import _unperturbed_learner
    from main.train.production_args import production_args
    return _unperturbed_learner(production_args())


def _golden(n=64):
    from agents.model.hypothesis_set_test import _obs_from_golden
    return _obs_from_golden(n)


def test_the_hypothesis_context_substitutes_rows_and_keeps_every_real_mask(x5):
    from agents.model.hypothesis_tokens import hypothesis_ctx
    m_fm = x5
    fe = copy.deepcopy(m_fm.policy).features_extractor
    obs = _golden()
    with torch.no_grad():
        fe(obs)
    hs = fe.last_hypothesis
    ctx = fe.unpack(obs)
    hctx = hypothesis_ctx(ctx, hs, fe.layout)
    hid = hs.slot_is_hypothesis
    assert bool(hid.any())
    # the hidden slots now carry their hypothesis's species and dex row; nothing else moves
    assert torch.equal(hctx.species_ids[:, 6:][hid], hs.slot_species[hid])
    assert torch.equal(hctx.pokemon_part[:, 6:][hid], hs.slot_rows[hid])
    assert torch.equal(hctx.pokemon_part[:, :6], ctx.pokemon_part[:, :6])
    assert torch.equal(hctx.pokemon_part[:, 6:][~hid], ctx.pokemon_part[:, 6:][~hid])
    for name in ("opp_believed_mask", "opp_addressable", "fainted_mask_opp", "fainted_mask_ours",
                 "all_fainted", "our_active_idx", "opp_active_local"):
        assert torch.equal(getattr(hctx, name), getattr(ctx, name)), name


def test_x5_reads_the_hypothesis_set_but_no_policy_or_value_gradient_reaches_delta_theta(x5):
    """M10 extended to U3's consumers: δ_θ CHANGES the forward (π drives the key bias, the selection
    and OTHER), yet the PPO-side outputs put exactly zero gradient into it; the consumed X5 parameters
    (`hypothesis_marker`, OTHER's `other_rest` / `other_map`) DO receive gradient. Teeth: compute π from
    `logits` instead of its detached copy and the zero-gradient half fails."""
    m_fm = x5
    pol = copy.deepcopy(m_fm.policy).eval()
    fe = pol.features_extractor
    hb = fe.hypothesis_builder
    obs = _golden()
    with torch.no_grad():
        l0 = pol.get_distribution(obs).distribution.logits.clone()
        v0 = pol.predict_values(obs).clone()
        hb.delta_out.weight.normal_(0.0, 0.5)
        l1 = pol.get_distribution(obs).distribution.logits
        v1 = pol.predict_values(obs)
    fin = torch.isfinite(l0) & torch.isfinite(l1)
    assert not torch.equal(l0[fin], l1[fin]) or not torch.equal(v0, v1)
    pol.train()
    pol.zero_grad(set_to_none=True)
    logits = pol.get_distribution(obs).distribution.logits
    v = pol.predict_values(obs)
    (logits.float().nan_to_num(0.0).sum() + v.sum()).backward()
    grads = {n: p.grad for n, p in hb.named_parameters()}
    for n, g in grads.items():
        if n.startswith("delta_"):
            assert g is None or float(g.abs().max()) == 0.0, n
    for n in ("hypothesis_marker", "other_rest", "other_map.weight"):
        assert grads[n] is not None and float(grads[n].abs().max()) > 0.0, n


def test_moves_on_hypothesis_seats_are_supervised_iff_the_species_is_present():
    from agents.training.belief_bank import move_belief_loss
    from agents.training.belief_bank_static import move_belief_terms
    M = 400
    torch.manual_seed(0)
    ml = torch.randn(2, 6, M, requires_grad=True)
    hs = SimpleNamespace(slot_species=torch.tensor([[0, 25, 143, 0, 0, 0], [0, 0, 0, 0, 0, 0]]),
                         slot_is_hypothesis=torch.tensor([[False, True, True, False, False, False],
                                                          [False] * 6]))
    bs = torch.full((2, 6), -1, dtype=torch.long)
    bs[0, 3] = 143                       # Snorlax IS on the true unseen team (label slot 3); Pikachu is not
    bm = torch.full((2, 6, 4), -1, dtype=torch.long)
    bm[0, 3] = torch.tensor([156, 34, 89, -1])
    km = torch.full((2, 6, 4), -1, dtype=torch.long)
    loss, mets = move_belief_loss(ml, km, bm, "both", hypothesis=hs, belief_species=bs)
    assert mets["unrevealed_slots"] == 1.0               # only the Snorlax seat
    tgt = torch.zeros(M)
    tgt[[156, 34, 89]] = 1.0
    ref = torch.nn.functional.binary_cross_entropy_with_logits(ml[0, 2], tgt)
    assert torch.allclose(loss, ref)
    st = move_belief_terms(ml, km, bm, "both", hypothesis=hs, belief_species=bs)
    assert bool(st.present) and torch.allclose(st.loss, ref)
    # blob (no hypothesis): the Hungarian path, unchanged — the hidden label slot is scored
    loss_b, mets_b = move_belief_loss(ml, km, bm, "both")
    assert mets_b["unrevealed_slots"] == 1.0 and not torch.allclose(loss_b, ref)
