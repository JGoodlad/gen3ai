"""`QWinProbHead`, the class that survives deletion pass L4 (config v134).

The extractor-built per-action head (`--q-winprob-mode`, its stash, its two coefficients and the
ModelVersion gate) was DELETED with the counterfactual training half, and so were the tests that built
it THROUGH `Gen3FeaturesExtractor`. The CLASS stays because the detached ride-along A head
(`ridealong_heads.AdvantageEnsemble`) is built from it; its stop-grad is the caller's seam
(`RideAlongBatch.detached`, pinned in `ridealong_heads_test`), not a property of this class. What is
pinned here is the module's own contract:

1. **The column order IS the action space.** A head whose index `a` did not mean action `a` would be
   the order-mismatch bug class with no shape error anywhere to catch it.
2. **The zero-init cold start is a total tie.** P(win|s,a) = 0.5 exactly for every action.
3. **ONE shared scorer, hence equivariance within a family.** Permuting our team permutes the six
   switch values with it; a head that memorised "slot 0" would be worthless.
"""
from __future__ import annotations

import torch

from agents.action.constants import (ACTION_SPACE_SIZE, MOVE_START, N_MOVE_SLOTS, N_SWITCH_SLOTS,
                                     STRUGGLE, SWITCH_START)
from agents.model.arch_constants import D_MODEL
from agents.model.q_winprob_head import QWinProbHead

_MC, _SC = 3, 5           # arbitrary nonzero pointer-cell widths for the standalone module tests


def _head(seed=0, **kw):
    torch.manual_seed(seed)
    return QWinProbHead(move_token_dim=D_MODEL, d_model=D_MODEL, ctx_dim=D_MODEL,
                        move_cell_dim=_MC, switch_cell_dim=_SC, **kw)


def _inputs(B=4, move_valid=None, team=None, generator=None):
    g = generator
    r = (lambda *s: torch.randn(*s, generator=g)) if g is not None else torch.randn
    return dict(
        ctx_vec=r(B, D_MODEL),
        move_tokens_req=r(B, N_MOVE_SLOTS, D_MODEL),
        move_valid=torch.ones(B, N_MOVE_SLOTS) if move_valid is None else move_valid,
        team_tokens=r(B, N_SWITCH_SLOTS, D_MODEL) if team is None else team,
        move_cells=r(B, N_MOVE_SLOTS, _MC),
        switch_cells=r(B, N_SWITCH_SLOTS, _SC),
    )


# ── the module's own contract ─────────────────────────────────────────────────

def test_the_output_is_the_action_space_in_order():
    """Index `a` IS action `a`. The three families land in `[switch x6, move x4, struggle]`, which
    is what makes a label row's `action` field, the policy's logits and the action mask the same
    index — the whole interface, and unrepresentable to check any other way than by construction."""
    head = _head()
    out = head(**_inputs())
    assert out.shape[-1] == ACTION_SPACE_SIZE
    assert (SWITCH_START, N_SWITCH_SLOTS, MOVE_START, N_MOVE_SLOTS, STRUGGLE) == (0, 6, 6, 4, 10)
    # A move slot marked invalid must zero exactly ITS column and no other. That pins the layout
    # from the inside: a head whose move block sat anywhere else would move the wrong zero.
    mv = torch.ones(2, N_MOVE_SLOTS)
    mv[:, 1] = 0.0
    torch.manual_seed(1)
    trained = _head()
    torch.nn.init.normal_(trained.q_score.weight)     # break the zero-init so 0 is informative
    torch.nn.init.normal_(trained.q_score.bias)
    got = trained(**_inputs(B=2, move_valid=mv))
    assert torch.all(got[:, MOVE_START + 1] == 0.0)
    assert torch.all(got[:, MOVE_START] != 0.0)


def test_zero_init_makes_every_action_exactly_one_half():
    """The honest cold start: a head that has seen no label ranks nothing. EXACT equality, not
    approximate — the scorer's weight AND bias are zeroed, so the logits are literally 0."""
    probs = torch.sigmoid(_head()(**_inputs(B=8)))
    assert torch.equal(probs, torch.full_like(probs, 0.5))


def test_one_shared_scorer_not_three():
    """The parameter sharing IS the equivariance claim. Three per-family scorers (the pointer
    head's shape, where they are correct because each family's logit has its own semantics) would
    let the Q head learn a different value function per family from the same evidence."""
    head = _head()
    scorers = [n for n, _ in head.named_parameters() if "q_score" in n]
    assert sorted(scorers) == ["q_score.bias", "q_score.weight"]


def test_switch_scores_are_EQUIVARIANT_under_permuting_our_team():
    """Permute our team tokens ⇒ the six switch Q values permute with them. A head that memorised
    "slot 0 is usually best" from an ordering that means nothing would be useless as a search leaf,
    and it is exactly what a flat `Linear(ctx, 11)` learns."""
    g = torch.Generator().manual_seed(7)
    head = _head(seed=3)
    # A zero-init scorer is trivially equivariant, so break it first: the property has to hold for
    # a head that actually discriminates.
    torch.nn.init.normal_(head.q_score.weight, std=0.5)
    torch.nn.init.normal_(head.q_score.bias, std=0.5)
    base = _inputs(B=2, generator=g)
    perm = torch.tensor([3, 0, 5, 1, 4, 2])
    swapped = dict(base)
    swapped["team_tokens"] = base["team_tokens"][:, perm]
    swapped["switch_cells"] = base["switch_cells"][:, perm]
    a = head(**base)[:, SWITCH_START:N_SWITCH_SLOTS]
    b = head(**swapped)[:, SWITCH_START:N_SWITCH_SLOTS]
    assert torch.allclose(a[:, perm], b, atol=1e-6)
