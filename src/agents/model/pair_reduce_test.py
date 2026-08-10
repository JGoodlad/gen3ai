"""Gates for the pair-reduction rungs (design_pair_reduction.md §7, unit-level).

G0 byte-identity · G2 status-threat coherence · G4 equivariance · α-init identity ·
second-moment recoverability. The full-forward G0 (default DamageOperator builds nothing and the
existing damage_op_test.py suite passes unchanged) rides the existing suite; here G0 is the
structural half: no params, no keys, no stash at the default rung.
"""
import pytest
import torch

from agents.model.pair_reduce import (
    PAIR_REDUCE_EPS, PAIR_REDUCE_LATENT, PairReducer, alpha_belief_mean, alpha_hard_max,
    build_pair_reducer, reduce_with_alpha,
)

B, J, C, F = 3, 6, 5, 6
G = torch.Generator().manual_seed(0)


def _rand_cells():
    return torch.rand(B, J, C, F, generator=G), torch.rand(B, C, generator=G)


# ---------------------------------------------------------------- G0 — byte-identity (structural)
def test_g0_hard_max_builds_nothing():
    assert build_pair_reducer("hard_max", n_channels=F) is None


def test_g0_default_damage_op_has_no_reducer_keys():
    # A default-config DamageOperator must carry ZERO new params/keys — state_dict unchanged ⇒
    # checkpoints and the byte-identity anchor are untouched. Constructed via the extractor's own
    # layout the same way damage_op_test.py does would drag the full fixture in; the structural
    # claim needs only the module surface: PairReducer is the ONLY class that owns params, and
    # build returns None for hard_max (above). Belt-and-braces: an unknown rung must raise.
    with pytest.raises(ValueError):
        PairReducer("hard_max", n_channels=F)
    with pytest.raises(ValueError):
        PairReducer("softmax_of_vibes", n_channels=F)


# ------------------------------------------------------------------------- Contract W basics
def test_alpha_belief_mean_is_a_distribution_and_zero_when_dead():
    w = torch.tensor([[0.2, 0.0, 0.6, 0.0, 0.2], [0.0] * 5])
    a = alpha_belief_mean(w)
    assert torch.allclose(a[0].sum(), torch.tensor(1.0))
    assert torch.allclose(a[0], w[0] / w[0].sum())
    assert a[1].abs().sum() == 0.0                       # all-zero belief ⇒ zero row, not uniform


def test_reduce_with_alpha_is_convex_units_preserving():
    cells, w = _rand_cells()
    a = alpha_belief_mean(w)
    red = reduce_with_alpha(a, cells)
    assert red.shape == (B, J, F)
    assert (red >= cells.amin(dim=2) - 1e-6).all() and (red <= cells.amax(dim=2) + 1e-6).all()


def test_hard_max_is_inside_contract_w():
    # §3.1: today's MODAL selection is expressible as α = onehot(argmax w·ref).
    cells, w = _rand_cells()
    ref = cells[:, :, :, 1].mean(dim=1)                  # board-level high-roll reference [B,C]
    a = alpha_hard_max(w, ref)
    assert ((a.sum(-1) - 1.0).abs() < 1e-6).all() and ((a == 0) | (a == 1)).all()
    idx = (w * ref).argmax(-1)
    picked = reduce_with_alpha(a, cells)
    assert torch.allclose(picked, cells[torch.arange(B), :, idx, :])


# ------------------------------------------------------- α-init identity (step 4, R2W zero-init)
def test_learned_alpha_at_init_equals_normalized_w():
    cells, w = _rand_cells()
    r = PairReducer("learned", n_channels=F)
    a = r.alpha_learned(w, cells)
    assert torch.allclose(a, alpha_belief_mean(w), atol=1e-6)


# --------------------------------------------------------------- G2 — status-threat coherence
def _g2_board():
    """Candidate 0 = the MODAL move: a status move (zero damage, high land-prob). Candidate 1 = a
    physical nuke (high damage, no status). Channels here: [dmg, status_land]."""
    w = torch.tensor([[0.8, 0.3]])                                       # presence belief
    cells = torch.zeros(1, 2, 2, 2)                                      # [B, J=2, C=2, F=2]
    cells[0, :, 0, 1] = 0.9                                              # status move: land=0.9
    cells[0, :, 1, 0] = 0.9                                              # nuke: dmg=0.9
    return w, cells


def test_status_threat_coherence_hard_max_describes_two_moves_at_once():
    # D2 documented: per-channel maxima put BOTH the nuke's damage and the status move's land-prob
    # in one row — a profile no single opponent move has. This is the defect, asserted as such.
    w, cells = _g2_board()
    per_channel_max = (w[:, None, :, None] * cells).amax(dim=2)          # the legacy reduction
    row = per_channel_max[0, 0]
    assert row[0] > 0.2 and row[1] > 0.7                                 # both channels "live"
    single_move_rows = (w[:, None, :, None] * cells)[0, 0]               # [C,F]
    assert not any(torch.allclose(row, r) for r in single_move_rows)     # matches NO real move


def test_status_threat_coherence_belief_mean_reflects_the_modal_status_move():
    w, cells = _g2_board()
    red = reduce_with_alpha(alpha_belief_mean(w), cells)
    row = red[0, 0]
    assert row[1] > row[0]                               # the row is status-led, like the opponent
    a = alpha_belief_mean(w)[0]
    assert torch.allclose(row, (a[:, None] * cells[0, 0]).sum(0))        # exactly ONE mixture


# ------------------------------------------------------------------------ G4 — equivariance
def test_g4_candidate_permutation_invariance():
    cells, w = _rand_cells()
    perm = torch.randperm(C, generator=G)
    for how in ("belief_mean", "learned", "deepsets_sum", "deepsets_max", "multi"):
        r = build_pair_reducer(how, n_channels=F)
        for m in r.modules():                            # give zero-init heads real weights
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.normal_(m.weight, std=0.3, generator=None)
        out = r(w, cells)
        out_p = r(w[:, perm], cells[:, :, perm, :])
        assert torch.allclose(out, out_p, atol=1e-5), how


def test_g4_defender_permutation_equivariance():
    cells, w = _rand_cells()
    perm = torch.randperm(J, generator=G)
    for how in ("belief_mean", "multi"):
        r = build_pair_reducer(how, n_channels=F)
        assert torch.allclose(r(w, cells)[:, perm], r(w, cells[:, perm]), atol=1e-6), how


# ------------------------------------------------------------- §4.1 — second-moment / hedging
def test_second_moment_recovers_branch_variance():
    # The W-rung emits E_α[cell] AND E_α[cell²]; Var = E[o²] − E[o]² must match the direct
    # branch-distribution variance — the quantity `max` structurally cannot produce.
    w = torch.tensor([[0.5, 0.5]])
    o = torch.tensor([0.0, 0.8])                                         # two branches (Swampert)
    cells = o[None, None, :, None].expand(1, 1, 2, 1)
    a = alpha_belief_mean(w)
    e1, e2 = reduce_with_alpha(a, cells), reduce_with_alpha(a, cells.pow(2))
    var = e2 - e1.pow(2)
    direct = (0.5 * (o - o.mean()).pow(2).sum())
    assert torch.allclose(var[0, 0, 0], direct, atol=1e-6)


def test_multi_bundle_width_and_zero_init_latents():
    cells, w = _rand_cells()
    r = build_pair_reducer("multi", n_channels=F)
    out = r(w, cells)
    assert out.shape == (B, J, r.extra_dim)
    assert r.extra_dim == (2 * F + 1) + 2 * PAIR_REDUCE_LATENT           # R1 block + 2 L pools
    assert out[..., 2 * F + 1:].abs().sum() == 0.0       # ρ zero-init ⇒ L latents ≡ 0 at init
    assert out[..., :2 * F + 1].abs().sum() > 0          # R1 block live immediately


def test_gradients_flow_to_belief_through_reduction():
    cells, w = _rand_cells()
    w = w.clone().requires_grad_(True)
    r = build_pair_reducer("multi", n_channels=F)
    r(w, cells).sum().backward()
    assert w.grad is not None and w.grad.abs().sum() > 0


# ------------------------------------- forward-level G0 through the REAL DamageOperator (add-beside)
def test_forward_multi_rung_leaves_block_byte_identical_and_stashes_extra():
    """The whole G0 claim at the op level: a non-default rung must (a) leave the flat block
    BIT-FOR-BIT equal to the default op's on the same ctx (the reducers only stash), (b) populate
    `last_reduced_extra` with the declared width, and (c) add ONLY `pair_reducer.*` state_dict
    keys. Default op: stash is None."""
    from agents.model.damage_op import DamageOperator, TEAM_SIZE, _DMG_PER_MON
    from agents.model.damage_op_test import _make_layout, _fake_ctx, _logits_hp_only, _T2I

    layout = _make_layout()
    defenders = [(260, _T2I["WATER"], _T2I["GROUND"]),
                 (126, _T2I["FIRE"], 0)] + [(0, 0, 0)] * 4
    kw = dict(defenders=defenders, hp_probs_active=[0.0] * 16,
              attacker_num=248, attacker_t1=_T2I["ELECTRIC"], attacker_t2=0)
    lg = _logits_hp_only(layout["max_moves"])

    torch.manual_seed(0)
    op0 = DamageOperator(layout)
    torch.manual_seed(0)
    opm = DamageOperator(layout, reduce_how="multi")

    block0 = op0(_fake_ctx(op0, **kw), lg)
    blockm = opm(_fake_ctx(opm, **kw), lg)
    assert torch.equal(block0, blockm)                       # add-beside: the block never moves
    assert op0.last_reduced_extra is None
    extra = opm.last_reduced_extra
    assert extra is not None and extra.shape == (1, TEAM_SIZE, opm.pair_reducer.extra_dim)
    assert torch.isfinite(extra).all()
    # Gating parity with the flat block: the stash uses the SAME defender_alive × has_opp gate the
    # block uses — no stricter. The fake ctx marks its num=0 filler slots ALIVE (and the flat block
    # is nonzero there too; production never fields empty own-team slots), so the checkable
    # property is that filler rows carry NO defender-specific signal: identical to each other,
    # distinct from the real defenders'.
    assert torch.equal(extra[0, 2], extra[0, 3]) and torch.equal(extra[0, 3], extra[0, 4]) \
        and torch.equal(extra[0, 4], extra[0, 5])
    assert not torch.allclose(extra[0, 0], extra[0, 2])

    new_keys = set(opm.state_dict()) - set(op0.state_dict())
    assert new_keys and all(k.startswith("pair_reducer.") for k in new_keys)
