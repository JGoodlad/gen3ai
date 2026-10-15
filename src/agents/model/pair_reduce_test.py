"""Gates for the surviving Contract-W primitives (design_pair_reduction.md §7, unit-level): the R1 rung
`alpha_belief_mean` and `reduce_with_alpha` — G2 status-threat coherence, G4 equivariance, second-moment
recoverability. (The reducer RUNGS and their G0 / α-init / bundle tests went with `--value-threat-inject`, their only
builder: legacy deletion U3.)
"""
import torch

from agents.model.pair_reduce import alpha_belief_mean, reduce_with_alpha

B, J, C, F = 3, 6, 5, 6
G = torch.Generator().manual_seed(0)


def _rand_cells():
    return torch.rand(B, J, C, F, generator=G), torch.rand(B, C, generator=G)


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


# ------------------------------------------------------------------------ G4 — equivariance
def test_g4_candidate_permutation_invariance_and_defender_equivariance():
    cells, w = _rand_cells()
    a = alpha_belief_mean(w)
    out = reduce_with_alpha(a, cells)
    pc = torch.randperm(C, generator=G)
    assert torch.allclose(out, reduce_with_alpha(alpha_belief_mean(w[:, pc]), cells[:, :, pc, :]), atol=1e-6)
    pj = torch.randperm(J, generator=G)
    assert torch.allclose(out[:, pj], reduce_with_alpha(a, cells[:, pj]), atol=1e-6)
