"""Aux value readouts off value_pooled: WinProbHead, the distributional ValueDistHead, and the
EVIDENTIAL CfEvidentialHead.

Split out of `features_extractor.py` 2026-08-16 (one responsibility per file); that module
re-exports every name here, so historical import paths still resolve.
"""
from typing import Tuple

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


class CfEvidentialHead(torch.nn.Module):
    """EVIDENTIAL readout — a **Beta posterior** over P(win|state), not a point estimate.

    WHY IT EXISTS, precisely. The G0 bias map (2026-08-22, 2,204 tight-MC labels) found that the
    scalar `WinProbHead`'s defect is **RESOLUTION, not an optimism offset**: population-mean
    predicted−MC gaps are |0.05|–|0.07| while the TRUE within-decile spread of P(win) is 0.11–0.36
    (`sd_true_excess`, the binomial floor already subtracted). The head cannot separate states it
    puts in the same bin. This head **cannot fix that blur** — it reads the same `value_pooled` and
    has no information the scalar head lacks. What it can do is **CONFESS** it: emit a WIDE Beta
    where the states behind a confidence bin are unresolved, and a narrow one where they are not.
    A confessed width is readable by the label factory's priority sampler (label the states the
    critic knows it cannot separate) and by the awareness stack, neither of which can act on a
    point estimate that is silently wrong.

    THE PARAMETERIZATION. Two outputs, mapped by ``softplus(·) + 1`` so **α, β ≥ 1**:
      * the Beta stays UNIMODAL (α<1 or β<1 puts mass at the endpoints and turns "uncertain" into
        "certain of both extremes", which is not the shape the readout is claiming), and
      * the uniform ``Beta(1, 1)`` — maximum ignorance — is exactly REACHABLE, so an untrained or
        genuinely-unresolved state has an honest place to sit.
    ``α + β`` is the **evidence / precision** (how many pseudo-observations the head thinks it has),
    ``α/(α+β)`` is the mean, and the std below is the epistemic width.

    SIDE readout, and ALWAYS-DETACHED at the call site. It feeds nothing forward — no pi, no vf, no
    other head — and its input is `value_pooled.detach()` unconditionally, so it is a pure
    supervised READOUT that cannot shape the trunk at any coefficient. That is a stronger contract
    than `WinProbHead`/`ValueDistHead`, whose tri-state modes allow a shaping variant; here there is
    deliberately no such mode. The module is not called from the extractor forward at all — the
    training-side loss applies it to the stashed `value_pooled` — so building it changes the
    state_dict and NOTHING else, and it is built LAST in `__init__` so no earlier module's
    initialization RNG draw moves (the append-never-insert rule).
    """

    def __init__(self) -> None:
        super().__init__()
        # The WinProbHead bottleneck, widened from 1 logit to 2 (the Beta's two raw parameters).
        self.net = torch.nn.Sequential(
            torch.nn.LayerNorm(D_MODEL),
            torch.nn.Linear(D_MODEL, D_MODEL),
            torch.nn.ReLU(),
            torch.nn.Linear(D_MODEL, 2),
        )

    def forward(self, value_pooled: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """value_pooled [B, D_MODEL] → ``(alpha [B], beta [B])``, each ≥ 1."""
        raw = torch.nn.functional.softplus(self.net(value_pooled)) + 1.0     # [B, 2]
        return raw[..., 0], raw[..., 1]

    # ---- the two closed forms the loss is built from (STATIC: unit-testable with no module) ----

    @staticmethod
    def _log_beta_fn(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """log B(a, b) = lgamma(a) + lgamma(b) − lgamma(a+b)."""
        return torch.lgamma(a) + torch.lgamma(b) - torch.lgamma(a + b)

    @classmethod
    def beta_binomial_nll(cls, alpha: torch.Tensor, beta: torch.Tensor,
                          wins: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
        """PER-ROW negative log-likelihood of ``w`` wins in ``n`` rollouts under Beta(α, β).

        This is the **marginal** likelihood — p ~ Beta(α,β) integrated out, not plugged in — which
        is what makes it the right evidential objective for COUNT data:

            −log P(w | n, α, β) = −[ log B(α+w, β+n−w) − log B(α, β) ]   (+ a constant)

        The dropped constant is ``log C(n, w)``, which does not depend on (α, β) and therefore
        contributes no gradient. Minimizing it does two things at once and that is the whole point:
        it pulls the MEAN α/(α+β) toward w/n, and it grows the PRECISION α+β only as far as the
        data across states actually supports — a head that inflates evidence on a state whose
        labels disagree pays for it here, where a plain BCE on the mean would not notice.
        """
        return -(cls._log_beta_fn(alpha + wins, beta + n - wins) - cls._log_beta_fn(alpha, beta))

    @classmethod
    def kl_to_uniform(cls, alpha: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        """PER-ROW ``KL( Beta(α,β) ‖ Beta(1,1) )`` — the evidential-overconfidence guard.

        Evidential heads have a standing failure mode: on data that is locally consistent, nothing
        in the likelihood stops α+β growing without bound, so the head reports certainty it has not
        earned (and the width — the entire product here — stops meaning anything). The standard
        remedy is a small pull back toward the uninformative prior, which for a Beta is the uniform
        Beta(1,1). Closed form, from the general KL between Betas with ``log B(1,1) = 0``:

            KL = −log B(α,β) + (α−1)ψ(α) + (β−1)ψ(β) + (2−α−β)ψ(α+β)

        It is exactly 0 at α = β = 1 (the reachable floor of the ``softplus+1`` parameterization),
        so the regularizer's fixed point is a representable state rather than an asymptote.
        """
        kl = (-cls._log_beta_fn(alpha, beta)
              + (alpha - 1.0) * torch.digamma(alpha)
              + (beta - 1.0) * torch.digamma(beta)
              + (2.0 - alpha - beta) * torch.digamma(alpha + beta))
        return kl  # type: ignore[no-any-return]

    @staticmethod
    def epistemic_std(alpha: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        """PER-ROW std of Beta(α, β) = sqrt(αβ / ((α+β)²(α+β+1))) — the CONFESSED width.

        The pre-registered read for the future A/B: this width, per stratum, should CORRELATE with
        the `cf_audit` bias map's measured `sd_true_excess` for that stratum. A head that is wide
        everywhere, or wide nowhere, has confessed nothing.
        """
        s = alpha + beta
        return torch.sqrt(alpha * beta / (s * s * (s + 1.0)))
