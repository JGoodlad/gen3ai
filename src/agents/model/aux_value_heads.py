"""Aux value readouts off value_pooled: WinProbHead and the distributional ValueDistHead.

Split out of `features_extractor.py` 2026-08-16 (one responsibility per file); that module
re-exports every name here, so historical import paths still resolve.
"""
import torch
from agents.model.arch_constants import (D_MODEL,
)




class WinProbHead(torch.nn.Module):
    """Auxiliary WIN-PROBABILITY readout — a calibrated P(win | state) the shaped critic can't give.

    The dual-head value (`value_pooled`) estimates expected *shaped* return (material Φ + PBRS terms +
    terminal, PopArt-normalised) — NOT a probability and not interpretable as win odds. This head reads
    the same whole-board `value_pooled` summary and emits ONE logit; sigmoid(logit) = P(win). It is
    supervised (in `instrumented_ppo`) by the Monte-Carlo episode OUTCOME (win=1 / loss=0) propagated to
    every step of the episode, so it learns the actual probability the current state leads to a win — and
    ΔP(win) across a decision is a directly legible "how much did this move change my win odds".

    SIDE readout, leak-safe: the logit is stashed at `features_extractor.last_win_prob_logits` and read
    ONLY by the aux loss + the offline prober/eval — NEVER concatenated into pi/vf, so the privileged
    future OUTCOME label can never reach the acting path. The tri-state `win_prob_mode` controls the
    GRADIENT at the call site (`read_only` feeds a STOP-GRAD `value_pooled` — the head trains its OWN
    params as a pure, risk-free diagnostic that can't perturb the policy; `shaping` feeds it live so the
    win-prediction objective also shapes the shared trunk). `none` = this module is not built (the chain
    is byte-for-byte the baseline)."""

    def __init__(self) -> None:
        super().__init__()
        # Small MLP off the value pool: LayerNorm → Linear → ReLU → Linear(→1). A bottleneck (not a bare
        # linear) so `read_only` reports "decodable by a small head" — fairer to the nonlinear trunk.
        self.net = torch.nn.Sequential(
            torch.nn.LayerNorm(D_MODEL),
            torch.nn.Linear(D_MODEL, D_MODEL),
            torch.nn.ReLU(),
            torch.nn.Linear(D_MODEL, 1),
        )

    def forward(self, value_pooled: torch.Tensor) -> torch.Tensor:
        """value_pooled [B, D_MODEL] → win-probability logit [B, 1] (sigmoid ⇒ P(win))."""
        return self.net(value_pooled)  # type: ignore[no-any-return]


class ValueDistHead(torch.nn.Module):
    """Distributional VALUE readout — an INTERPRETABILITY side head over the return distribution.

    The scalar critic emits one number, E[Z] (expected shaped return). This head reads the same
    whole-board `value_pooled` summary and emits `bins` logits over a FIXED atom support
    `linspace(vmin, vmax, bins)`: `softmax(logits)` is the critic's predicted return DISTRIBUTION,
    not just its mean. That distribution is what makes "how is the model predicting" legible — a
    sharp spike = confident, a wide spread = uncertain, a bimodal shape = the critic sees a coinflip
    (e.g. "I win if this move hits, else I lose") — all invisible in the scalar V that collapses
    every shape to one mean. The categorical HL-Gauss parameterization (Phase A side head) is the
    `WinProbHead` pattern applied to the value target. Design:
    `designs/ai_v6/design_distributional_value_critic.md`.

    SIDE readout, leak-safe: the logits are stashed at `features_extractor.last_value_dist_logits`
    and read ONLY by the (future) aux loss + the offline prober/eval — NEVER concatenated into pi/vf,
    so the projection dims are unchanged either way (off byte-for-byte). The tri-state
    `value_dist_mode` controls the GRADIENT at the call site (`read_only` feeds a STOP-GRAD
    `value_pooled` — a pure, risk-free diagnostic that can't perturb the policy; `shaping` feeds it
    live so the distributional objective also shapes the shared trunk). `none` = this module is not
    built (the chain is byte-for-byte the baseline). The `atoms` buffer is non-persistent
    (deterministic from `bins`/`vmin`/`vmax`) so it stays out of the state_dict — only the head's
    params (whose final Linear is `bins`-wide) define the loadable shape."""

    def __init__(self, bins: int, vmin: float, vmax: float):
        super().__init__()
        if bins <= 0:
            raise ValueError(f"ValueDistHead bins must be > 0, got {bins}")
        if not vmax > vmin:
            raise ValueError(f"ValueDistHead requires vmax > vmin, got vmin={vmin}, vmax={vmax}")
        self.bins = bins
        # Small MLP off the value pool: LayerNorm → Linear → ReLU → Linear(→bins) — the WinProbHead
        # bottleneck, widened from 1 logit to `bins` (a categorical head over the return support).
        self.net = torch.nn.Sequential(
            torch.nn.LayerNorm(D_MODEL),
            torch.nn.Linear(D_MODEL, D_MODEL),
            torch.nn.ReLU(),
            torch.nn.Linear(D_MODEL, bins),
        )
        # Fixed atom support, non-persistent (deterministic from bins+range → out of the state_dict,
        # like the damage_tables buffers). Read by the loss (target projection) + the prober (atoms →
        # return units) + `mean()` below; the head's forward only needs the net.
        self.register_buffer("atoms", torch.linspace(vmin, vmax, bins), persistent=False)

    def forward(self, value_pooled: torch.Tensor) -> torch.Tensor:
        """value_pooled [B, D_MODEL] → per-atom logits [B, bins] (softmax ⇒ return distribution)."""
        return self.net(value_pooled)  # type: ignore[no-any-return]

    def mean(self, logits: torch.Tensor) -> torch.Tensor:
        """E[Z] = Σ atomsᵢ·softmax(logits)ᵢ — the scalar the distribution implies, [B, 1]. (Used by
        the prober / diagnostics; the Phase-A side head does NOT feed this into the scalar critic.)"""
        return (torch.softmax(logits, dim=-1) * self.atoms).sum(-1, keepdim=True)  # type: ignore[no-any-return]
