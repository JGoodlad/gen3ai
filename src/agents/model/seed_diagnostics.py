"""Seed-collapse diagnostics for the multi-seed critic readout (OpTensors step 4).

The k-seed value readout (k learned queries cross-attending over the `our_mon` tensor —
design_op_tensors.md §3.3) buys the critic readout MULTIPLICITY. Its known failure mode is
SEED COLLAPSE: the k queries converge and you pay for k reads but receive one. The house
precedent is `z_arch` (ledger: ≈2/3 of the code's energy sat in ONE shared direction because
the VICReg covariance term was specified but never wired — discovered post-hoc by a probe,
not during training). These diagnostics exist so that failure is visible on TensorBoard from
step 0 instead.

TB CONTRACT (the step-4 module MUST record these once per `train()` via the instrumented-PPO
diagnostics path, same cadence as `popart/*` and `film/*`):

    seeds/query_cos        mean off-diagonal |cosine| of the k seed QUERIES
                           (weights — collapse cause)
    seeds/out_cos          mean off-diagonal |cosine| of the k OUTPUTS, batch-averaged
                           (activations — collapse effect)
    seeds/out_effective_rank   participation ratio (Σλ)²/Σλ² of the k outputs' UNCENTERED
                               gram, batch-averaged — "how many distinct readout directions
                               am I really getting"; healthy ≈ k, collapsed → 1. Uncentered
                               on purpose: centering over seeds folds two distinct clusters
                               into one difference direction (a 2-cluster set would read
                               rank 1 centered, rank 2 uncentered — the latter is the
                               multiplicity question)
    seeds/out_var          mean per-dim variance across seeds — the VICReg variance target

PRE-REGISTERED VICReg TRIGGER (decide from the plot, not vibes): wire the VICReg
variance+covariance floor onto the seed outputs IF, after the run's first ~2M steps,
`value_seeds/query_cos` sustains > 0.6 OR `value_seeds/out_effective_rank` sustains < k/2. Below those,
the seeds are differentiating on their own and the regularizer is dead weight. (The z_arch
numbers for calibration: its collapsed state read ~2/3 energy in one direction ≈ effective
rank ~1.6 of a 32-dim code.)

⚠️ THE REGULARIZER'S TARGETS MUST BE SCALE-RELATIVE. Gen-6's first launch used an ABSOLUTE std
hinge (γ=1.0) against a readout whose entire signal has RMS ≈0.207 and whose cross-seed spread is
structurally bounded by ≈0.14 (each seed output is a convex combination of the same six kv rows).
`out_effective_rank` sat at exactly 1.000 for 2M steps while the hinge saturated at 0.997 — the
term pushed constantly toward something unreachable. `seed_vicreg.py` now hinges on the RATIO
(cross-seed std ÷ the dim's own RMS) and penalises cross-seed CORRELATION, and logs
`value_seeds/out_rms` so the one degenerate response (shrink the feature instead of
differentiating it) is visible rather than silent.

TRIGGER FIRED — 2026-08-10, gen-5 (`ai_v9_06_gen5_no_concat_0809`): `out_effective_rank` = 1.0
and `out_cos` = 1.000 sustained from 196k through 15M+ steps (`out_var` ≈ 5e-6; `query_cos` 0.33
— distinct queries, identical attention patterns). The wiring lives in `seed_vicreg.py`
(`--value-seed-vicreg-coef`, v62 resume-immutable, OFF for gen-5) — enable at the gen-6 launch
and judge by `out_effective_rank` rising toward k. (These TB tags were `seeds/*` on gen-5's
board; renamed to `value_seeds/*` in the same pass to disambiguate from RNG seeds.)

Pure torch, no side effects — unit-tested in `seed_diagnostics_test.py`.
"""
from __future__ import annotations

from typing import Dict

import torch


def _off_diag_abs_cos(x: torch.Tensor) -> torch.Tensor:
    """Mean absolute off-diagonal cosine similarity along the second-to-last axis.

    x: [k, D] or [B, k, D] → scalar tensor."""
    xn = torch.nn.functional.normalize(x, dim=-1, eps=1e-8)
    g = xn @ xn.transpose(-1, -2)                                  # [..., k, k]
    k = g.shape[-1]
    if k < 2:
        return torch.zeros((), device=x.device)
    eye = torch.eye(k, device=x.device, dtype=torch.bool)
    off = g.masked_select(~eye.expand_as(g).to(torch.bool))
    return off.abs().mean()


def seed_collapse_diagnostics(queries: torch.Tensor, outputs: torch.Tensor) -> Dict[str, float]:
    """→ the four TB scalars (see the module docstring for the contract and trigger).

    queries: the k learned seed queries, ``[k, D]`` (weights).
    outputs: the k per-seed readout vectors for a batch, ``[B, k, D]`` (activations).
    """
    assert queries.dim() == 2, f"queries must be [k, D], got {tuple(queries.shape)}"
    assert outputs.dim() == 3, f"outputs must be [B, k, D], got {tuple(outputs.shape)}"
    k = queries.shape[0]
    assert outputs.shape[1] == k, "outputs' seed axis must match queries'"

    with torch.no_grad():
        query_cos = _off_diag_abs_cos(queries)
        out_cos = _off_diag_abs_cos(outputs)

        # Participation ratio of the seed axis on the UNCENTERED [k, D] outputs (see the
        # contract note: centered PR folds distinct clusters into difference directions).
        # PR = (Σλ)² / Σλ², λ = σ² of the singular values.
        s = torch.linalg.svdvals(outputs.float())                  # [B, min(k, D)]
        lam = s.pow(2)
        pr = lam.sum(-1).pow(2) / lam.pow(2).sum(-1).clamp_min(1e-12)
        degenerate = lam.sum(-1) < 1e-10                           # all-zero outputs: report 1
        pr = torch.where(degenerate, torch.ones_like(pr), pr)

        # The VICReg variance target stays CENTERED — it asks "do the seeds differ", which is
        # exactly the deviation-from-the-shared-mean question.
        centered = outputs - outputs.mean(dim=1, keepdim=True)
        out_var = centered.var(dim=1, unbiased=False).mean()

    return {
        "value_seeds/query_cos": float(query_cos),
        "value_seeds/out_cos": float(out_cos),
        "value_seeds/out_effective_rank": float(pr.mean()),
        "value_seeds/out_var": float(out_var),
    }
