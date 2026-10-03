"""Aux value readout off value_pooled: `WinProbHead`.

Split out of `features_extractor.py` 2026-08-16 (one responsibility per file); that module
re-exports the name, so historical import paths still resolve. (The evidential `CfEvidentialHead`
and the passive `ShadowValueHead` that lived here were DELETED with the counterfactual training
half, deletion pass L4.)
"""
import torch
from agents.model.arch_constants import D_MODEL


class WinProbHead(torch.nn.Module):
    """Auxiliary WIN-PROBABILITY readout — a calibrated P(win | state) the shaped critic can't give.

    The dual-head value (`value_pooled`) estimates expected return in the run's reward units under the
    shaped critic — NOT a probability and not interpretable as win odds. This head reads
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
