"""Per-seed QUANTILE assignment — give each value seed a DIFFERENT job (gen3_seed_quantile_v1).

The diagnosis behind this module is the one the VICReg attempt only treated symptomatically.
`MultiSeedValueReadout`'s k seeds collapse (gen-5: `out_effective_rank` = 1.000 for 24M steps)
because **the objective they serve carries one dimension of information**: k parallel reads feed
one scalar critic, so nothing in the loss ever preferred four different summaries over four copies
of the best one. Collapse is not a pathology there — it is the optimum.

That is the SAME disease the FitNets result already diagnosed elsewhere in this codebase:
distilling a teacher's SCALAR V *crystallized* the critic (`value_cls` effective rank 4.15 → 3.55),
and the fix was not a repulsion penalty but a RICHER TARGET (the 128-dim `value_pooled` hint).

VICReg (`seed_vicreg.py`) attacks collapse NEGATIVELY: "be different", with no statement about
what the differences should mean — it can only manufacture decorrelated directions, which may
carry nothing. This module attacks it POSITIVELY: **give seed k the job of predicting quantile
τ_k of the return distribution.** The τ's differ by construction, so producing them requires
different reads of the board; collapse stops being unpenalized and becomes strictly
LOSS-INCREASING. (Structurally this is QR-DQN's quantile regression — Dabney et al. 2018 — with
the quantile index carried by an existing seat rather than a new network.)

The decomposition is also SEMANTIC, which is the part a decorrelation penalty can never give:
the τ=0.1 seed has to find what makes the downside (which of my mons is about to be removed), the
τ=0.9 seed what makes the upside (which of my mons can take over) — genuinely different reads of
the same six per-our-mon rows, which is what the multi-seed window was FOR.

⚠️ **THE SHARED READOUT IS LOAD-BEARING — do not give each seed its own projection.** The τ_k
predictions come from ONE `nn.Linear(dim, 1)` applied to every seed. With a shared readout the
only way to emit k different numbers is for the k SEED OUTPUTS to differ, so the pressure lands
where we want it. Per-seed projections would let the HEAD manufacture the spread from four
identical inputs — the feature would report success while the seeds stayed collapsed, the exact
silent-failure shape of the z_arch covariance term. `seed_quantile_test.py::
test_shared_readout_makes_collapse_strictly_worse` is the proof.

Wiring: `--seed-quantile-coef` (0.0 = OFF, byte-identical — no module, no state_dict keys). The
loss is folded in `instrumented_ppo.train()` against the SAME rollout return the critic regresses
(PopArt-normalized when the critic is, so the τ's live in the critic's own units). It is an AUX on
the seed block — the value path is untouched, so it cannot damage the critic the way replacing the
scalar head would.

**Honest scope (do not overclaim).** Ledger K1 killed the DISTRIBUTIONAL CRITIC as a win-rate
lever (return residuals sub-Gaussian — no tail worth modeling). This module makes a DIFFERENT
claim: not "modeling the return distribution improves the value estimate", but "a k-dimensional
target prevents a k-way readout from collapsing". K1 does not refute that — but it does warn that
un-collapsing is a MEANS, and gen-5 (collapsed) already matched gen-4 on ELO, so the multiplicity
may still be worth ~0. Judge it on `value_seeds/out_effective_rank` first and anchored ELO second.
"""
from __future__ import annotations

from typing import Dict, Sequence, Tuple

import torch

# One target quantile per seed, spread over the return distribution. Ordered ascending so the
# per-seed predictions carry a CHECKABLE invariant (pred_0 < pred_1 < … — see the crossing_rate
# metric). Length must equal VALUE_SEED_K; asserted at construction.
SEED_QUANTILE_TAUS: Tuple[float, ...] = (0.1, 0.35, 0.65, 0.9)

# ⚠️ PURE pinball, NOT QR-DQN's Huber-softened variant — and that is a deliberate reversal of the
# obvious default. Huber caps the gradient inside |u| < κ, which pulls every quantile estimate
# toward the median: MEASURED on N(0,2) targets at κ=1, the fitted quantiles came out ±2.18/±0.67
# against true ±2.50/±0.78. QR-DQN wants that softening because it regresses a bootstrapped TD
# target; we regress a Monte-Carlo return that is already PopArt-normalized, so stability is not
# the binding concern — and the shrinkage attacks the ONE thing this module exists to create:
# SEPARATION between the per-seed targets. Pinball's gradient is bounded by construction anyway
# (magnitude ≤ 1 per sample), so nothing is lost.


class SeedQuantileHead(torch.nn.Module):
    """ONE shared `Linear(dim, 1)` mapping every seed output to its quantile prediction.

    Shared on purpose (see the module docstring): it is what forces the differentiation into the
    SEEDS instead of letting a per-seed head fake it.
    """

    def __init__(self, dim: int, n_seeds: int, taus: Sequence[float] = SEED_QUANTILE_TAUS):
        super().__init__()
        assert len(taus) == n_seeds, (
            f"need one tau per seed: got {len(taus)} taus for {n_seeds} seeds — "
            "SEED_QUANTILE_TAUS must match VALUE_SEED_K"
        )
        assert all(0.0 < t < 1.0 for t in taus), f"taus must be in (0,1): {taus}"
        assert list(taus) == sorted(taus), f"taus must be ascending (the ordering invariant): {taus}"
        self.proj = torch.nn.Linear(dim, 1)
        self.register_buffer("taus", torch.tensor(list(taus), dtype=torch.float32))

    def forward(self, seed_outputs: torch.Tensor) -> torch.Tensor:
        """[B, k, D] → [B, k] — seed k's prediction of quantile τ_k of the return."""
        return self.proj(seed_outputs).squeeze(-1)


def seed_quantile_loss(
    preds: torch.Tensor, returns: torch.Tensor, taus: torch.Tensor
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Huber quantile (pinball) regression of the k seed predictions onto the return.

    preds:   ``[B, k]``  — `SeedQuantileHead(seed_outputs)`, WITH grad.
    returns: ``[B]``     — the same target the critic regresses (normalized frame under PopArt).
    taus:    ``[k]``     — the per-seed target quantiles.

    Asymmetric by construction: a residual above the prediction is weighted τ, below it (1−τ), so
    the minimizer of seed k is the τ_k-quantile of the conditional return distribution. Four
    different τ ⇒ four different minimizers ⇒ four identical seeds cannot all be optimal.
    """
    assert preds.dim() == 2, f"preds must be [B, k], got {tuple(preds.shape)}"
    assert returns.dim() == 1 and returns.shape[0] == preds.shape[0], (
        f"returns must be [B] matching preds: {tuple(returns.shape)} vs {tuple(preds.shape)}"
    )
    assert taus.shape[0] == preds.shape[1], "one tau per seed"

    u = returns[:, None] - preds                                   # [B, k] residual
    weight = (taus[None, :] - (u.detach() < 0).float()).abs()      # τ above, (1−τ) below
    loss = (weight * u.abs()).mean()                               # PURE pinball — see below

    with torch.no_grad():
        per_seed = preds.mean(dim=0)                               # [k]
        # THE ORDERING INVARIANT: ascending taus ⇒ ascending predictions. A collapsed readout
        # emits k equal numbers, so the spread → 0 and crossings pile up. This is the cheap
        # per-step read of "are the seeds doing different jobs".
        d = per_seed[1:] - per_seed[:-1]
        crossing = float((d <= 0).float().mean()) if d.numel() else 0.0
        metrics = {
            "value_seeds/quantile_loss": float(loss.detach()),
            "value_seeds/quantile_spread": float(per_seed.max() - per_seed.min()),
            "value_seeds/quantile_crossing_rate": crossing,
        }
        for i in range(per_seed.shape[0]):
            metrics[f"value_seeds/quantile_pred_{i}"] = float(per_seed[i])
    return loss, metrics
