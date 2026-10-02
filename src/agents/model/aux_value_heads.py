"""Aux value readouts off value_pooled: WinProbHead, the EVIDENTIAL CfEvidentialHead, and the
passive ShadowValueHead.

Split out of `features_extractor.py` 2026-08-16 (one responsibility per file); that module
re-exports every name here, so historical import paths still resolve.
"""
from typing import Tuple

import torch
from agents.model.arch_constants import (D_MODEL,
)




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
    than `WinProbHead`, whose tri-state mode allows a shaping variant; here there is
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


class ShadowValueHead(torch.nn.Module):
    """The SHADOW CRITIC — a PASSIVE value twin trained on tight-MC ``mc_return`` labels.

    WHY IT EXISTS. The live critic V is trained on **bootstrapped GAE targets** built from single
    realized trajectories: one Monte-Carlo sample of the return per state, propagated through a
    self-referential bootstrap. The counterfactual label factory can measure something the on-policy
    stream structurally cannot — the *average* realized shaped return over R independent rollouts
    from the SAME board — which is a tight, ground-truth estimate of exactly the quantity V is
    supposed to be. This head is that estimate's own readout: same trunk summary, different target
    stream.

    WHAT IT IS NOT, and this is the whole safety argument. It **never computes an advantage**, it is
    **never consulted by GAE**, it feeds **nothing** forward (no pi, no vf, no other head), and its
    input is ``value_pooled.detach()`` **unconditionally** — there is no ``read_only``/``shaping``
    mode to change that, exactly as for :class:`CfEvidentialHead`. Swapping the live critic for an
    MC-grounded one is critic SURGERY, which owes the C4 offline gate; this head is the **staged
    promotion path** that earns (or refuses) that gate without risking a run: it accumulates the
    evidence — how far the live critic's V drifts from a tight-MC return on the same states — as a
    published number rather than an argument.

    THE FRAME. PopArt is deleted, so the head's output is in real return units and its loss is a raw
    MSE against ``mc_return``.

    THE UNITS OF THE LABEL are the run's own **shaped** return — Σ γᵏ r, with r produced by the
    run's `RewardConfig`. A shaped return is a fact about a board *under a reward composition*, so
    the producer stamps a reward digest on every ``mc_return`` row and the buffer REFUSES a row
    whose digest disagrees with this run's. Without that, a label produced under a different reward
    composition would be silently averaged into the target.

    Built LAST in ``Gen3FeaturesExtractor.__init__`` and never called by the forward, so OFF is
    byte-identical and ON-at-coefficient-0 is BIT-identical in pi/vf (the append-never-insert rule:
    a module inserted mid-constructor shifts the init RNG stream for everything after it).
    """

    def __init__(self) -> None:
        super().__init__()
        # The WinProbHead bottleneck, emitting one SCALAR VALUE rather than one logit. Same capacity
        # as the win-prob twins on purpose: every head in this family reads the same `value_pooled`
        # through the same shape, so a difference between two of them is a difference of TARGETS.
        self.net = torch.nn.Sequential(
            torch.nn.LayerNorm(D_MODEL),
            torch.nn.Linear(D_MODEL, D_MODEL),
            torch.nn.ReLU(),
            torch.nn.Linear(D_MODEL, 1),
        )

    def forward(self, value_pooled: torch.Tensor) -> torch.Tensor:
        """value_pooled [B, D_MODEL] → the predicted value [B, 1], in real return units."""
        return self.net(value_pooled)  # type: ignore[no-any-return]
