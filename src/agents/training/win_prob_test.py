"""Win-probability loss (the future-outcome plumbing)."""


import torch

from agents.training.instrumented_ppo import InstrumentedMaskablePPO


# ── _win_prob_loss ──────────────────────────────────────────────────────────────

def test_loss_masks_unknown_transitions():
    """A masked transition (in-progress episode) must be EXCLUDED — even one whose unmasked loss
    would be huge — so the head is never trained toward a fabricated label."""
    logits = torch.tensor([[2.0], [-2.0], [0.0], [5.0]])
    target = torch.tensor([[1.0], [0.0], [1.0], [0.0]])
    mask = torch.tensor([[1.0], [1.0], [1.0], [0.0]])      # 4th unknown
    loss, m = InstrumentedMaskablePPO._win_prob_loss(logits, target, mask)
    assert abs(m["coverage"] - 0.75) < 1e-6
    assert abs(m["pred_mean"] - 0.5) < 1e-6          # mean over the 3 known (excludes the 0.993 of #4)
    assert abs(m["label_mean"] - 2 / 3) < 1e-6
    # loss == mean BCE over the 3 known, the 4th must not appear
    import torch.nn.functional as F
    expect = F.binary_cross_entropy_with_logits(logits[:3].reshape(-1), target[:3].reshape(-1))
    assert abs(float(loss) - float(expect)) < 1e-6


def test_loss_margin_stratifies_and_scores_skill():
    """With the material margin, the loss adds closeness-stratified Brier/acc + a skill-vs-material
    score — the 'value on close games, beyond counting mons' signal the aggregate Brier hides."""
    # 2 blowouts (|margin| 0.9, head trivially right) + 2 close (|margin| 0.05, head still right).
    logits = torch.tensor([[4.0], [-4.0], [1.0], [-1.0]])   # P ≈ .98, .02, .73, .27
    target = torch.tensor([[1.0], [0.0], [1.0], [0.0]])
    mask = torch.ones(4, 1)
    margin = torch.tensor([[0.9], [-0.9], [0.05], [-0.05]])
    _, m = InstrumentedMaskablePPO._win_prob_loss(logits, target, mask, margin)
    assert "brier_contested" in m and "skill_vs_material" in m
    assert abs(m["contested_frac"] - 0.5) < 1e-6          # 2 of 4 are |margin|<0.25
    assert abs(m["acc_contested"] - 1.0) < 1e-6           # both close calls predicted correctly
    assert m["brier_contested"] < 0.25                    # below a 50/50 game's no-skill floor → real skill
    assert m["skill_vs_material"] > 0.0                   # beats the material-only baseline
    assert abs(m["contested_label_mean"] - 0.5) < 1e-6    # the close band is genuinely even


def test_loss_no_margin_omits_stratified():
    """Without the margin (old config / margin absent), only the aggregate metrics are reported."""
    logits = torch.zeros(2, 1)
    _, m = InstrumentedMaskablePPO._win_prob_loss(logits, torch.tensor([[1.0], [0.0]]), torch.ones(2, 1), None)
    assert "brier" in m and "brier_contested" not in m and "skill_vs_material" not in m


def test_a_FLAT_margin_is_treated_as_absent():
    """`gen3_tb_relevance_v1`: a margin with no SPREAD cannot stratify, so the six tags it would
    produce are copies of their pooled siblings plus two constants — worse than publishing
    nothing, because all six READ AS MEASUREMENTS. This is the exact shape the pre-fix win-prob
    arm shipped (`win_margin` pinned at 0.0 by the since-deleted `_fold_material_pbrs`'s early return)."""
    logits = torch.tensor([[2.0], [-2.0], [1.0], [-1.0]])
    target = torch.tensor([[1.0], [0.0], [1.0], [0.0]])
    _, m = InstrumentedMaskablePPO._win_prob_loss(
        logits, target, torch.ones(4, 1), torch.zeros(4, 1))
    assert "contested_frac" not in m and "skill_vs_material" not in m


def test_a_REAL_margin_spread_selects_a_STRICT_SUBSET():
    """`gen3_obs_margin_unconditional_v1` — the consumer half of the fix, driven by margins the REAL
    `material_margin` rule computes off board views rather than by hand-typed floats. (It formerly read
    them off the per-turn reward manager, which only wrapped this rule — deleted with the Python battle
    layer, T27 P6 slice 6d-2; training's `win_margin` label is the Rust core's twin, `labels/margin.rs`.)

    Pre-fix, every one of these read 0.0, so `|margin| < tau` was always true: `contested_frac`
    was a flat 1.0, every `*_contested` tag was a byte-identical copy of its pooled sibling, and
    `P_mat` was a constant 0.5 — `skill_vs_material` scored the head against a coin flip."""
    from agents.training.material_margin import material_margin
    from agents.training.reward_test_fakes import _full_team_live

    margins = [[material_margin(_full_team_live(our_alive=ours, opp_alive=opp))]
               for ours, opp in ((6, 1), (6, 2), (4, 4), (3, 3), (2, 6), (1, 6))]

    margin = torch.tensor(margins)
    n = margin.shape[0]
    # A board-derived spread, not a hand-built one — the fix's whole point.
    assert float(margin.max() - margin.min()) > 0.0, "the material-margin rule produced a flat margin"

    logits = torch.zeros(n, 1)
    target = (margin > 0).float()
    _, m = InstrumentedMaskablePPO._win_prob_loss(logits, target, torch.ones(n, 1), margin)

    assert "contested_frac" in m, "the contested family is missing on a real spread"
    assert 0.0 < m["contested_frac"] < 1.0, (
        f"the contested split selected {m['contested_frac']:.3f} of the batch — a STRICT SUBSET "
        "is what makes it a split rather than a copy of the pooled metrics")
    # The material baseline is no longer the degenerate constant 0.5, so the skill score is a
    # comparison against 'count the mons' rather than against a coin flip.
    assert abs(m["brier_material"] - 0.25) > 1e-6, "P_mat collapsed to the constant 0.5"


def test_loss_none_guards():
    z = torch.zeros(3, 1)
    assert InstrumentedMaskablePPO._win_prob_loss(None, z, z) is None
    assert InstrumentedMaskablePPO._win_prob_loss(z, None, z) is None
    assert InstrumentedMaskablePPO._win_prob_loss(z, z, torch.zeros(3, 1)) is None  # all masked


def test_loss_grad_flows_to_logits():
    logits = torch.zeros(4, 1, requires_grad=True)
    target = torch.tensor([[1.0], [0.0], [1.0], [0.0]])
    mask = torch.ones(4, 1)
    loss, _ = InstrumentedMaskablePPO._win_prob_loss(logits, target, mask)
    loss.backward()
    assert logits.grad is not None and float(logits.grad.abs().sum()) > 0
