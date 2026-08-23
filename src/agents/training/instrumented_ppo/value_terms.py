"""The three CRITIC-side loss terms.

* `_win_prob_loss` — the auxiliary win-probability BCE, with the contested-band readout that is
  the term's actual information content (a blowout's P(win) is recoverable from material).
* `_value_dist_loss` — the distributional head's HL-Gauss cross-entropy.
* `_value_loss_from_se` — the tail-weighted (CVaR-blended) value loss, used at all THREE value
  sites in `train()`. At `value_tail_weight == 0` it is `se.mean()`, byte-identical to
  `F.mse_loss`.
"""
import torch as th
from torch.nn import functional as F

from agents.training.instrumented_ppo.constants import _VALUE_TAIL_FRAC, _WIN_CONTESTED_TAU


class ValueTerms:
    """The win-prob, value-distribution and tail-weighted value losses."""

    @staticmethod
    def _win_prob_loss(logits, target, mask, margin=None):
        """Supervised BCE loss for the auxiliary WIN-PROBABILITY head (``last_win_prob_logits`` [B,1]).

        ``target`` [B,1] = the Monte-Carlo episode OUTCOME (win=1 / loss=0) propagated to every step of
        the episode by the ``WinProbLabelCallback`` (it overwrites the obs-dict placeholder post-collection);
        ``mask`` [B,1] = 1 where that label is KNOWN (the step's episode finished within the rollout buffer)
        and 0 for the trailing in-progress episode (no outcome yet) — those transitions are excluded so the
        head is never trained toward a fabricated label. BCE-with-logits, masked-mean. Returns
        ``(loss, metrics)`` or ``None`` when nothing is scorable (head off / labels absent / a minibatch
        with zero known labels — the None guard keeps an empty minibatch from NaN-poisoning the loss). Pure
        + static so it unit-tests without a full PPO.

        When ``margin`` [B,1] (the normalized material margin ∈ [−1,1], from gen3_env's ``win_margin`` obs
        key) is given, ALSO reports the INFORMATION VALUE the aggregate Brier hides: the head's skill on
        CLOSE games (``|margin| < _WIN_CONTESTED_TAU`` — a blowout's P(win) is trivially recoverable from
        material), and a Brier SKILL SCORE vs a material-only baseline (``P_mat = clip(0.5 + 0.5·margin)``):
        ``skill_vs_material`` > 0 ⇒ the head beats 'just count the mons'."""
        if logits is None or target is None or mask is None:
            return None
        logits = logits.reshape(-1)
        target = target.to(logits.device).reshape(-1)
        mask = mask.to(logits.device).reshape(-1)
        n_known = mask.sum()
        if float(n_known) == 0.0:
            return None
        per = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        loss = (per * mask).sum() / n_known
        with th.no_grad():
            p = th.sigmoid(logits)
            sq = (p - target) ** 2
            correct = ((p > 0.5).float() == target).float()
            brier = (sq * mask).sum() / n_known                             # calibration (lower better)
            acc = (correct * mask).sum() / n_known
            pred_mean = (p * mask).sum() / n_known                          # mean predicted P(win)
            label_mean = (target * mask).sum() / n_known                    # actual win base rate
        metrics = {
            "loss": float(loss.item()),
            "acc": float(acc.item()),
            "brier": float(brier.item()),
            "pred_mean": float(pred_mean.item()),
            "label_mean": float(label_mean.item()),
            "coverage": float((n_known / mask.numel()).item()),             # fraction of minibatch labeled
        }
        # Information value the aggregate Brier hides (only when the material margin is available): the
        # head's skill on CLOSE games + a skill score beyond a material-only baseline.
        if margin is not None:
            with th.no_grad():
                margin = margin.to(logits.device).reshape(-1)
                close = (margin.abs() < _WIN_CONTESTED_TAU).float() * mask
                n_close = close.sum()
                metrics["contested_frac"] = float((n_close / n_known).item())
                if float(n_close) > 0.0:
                    # Brier/acc restricted to material-EVEN decisions — where a good P(win) is non-trivial
                    # (the aggregate is inflated by blowouts). Judge brier_contested vs a 50/50 game's
                    # ~0.25 no-skill floor; contested_label_mean ≈ 0.5 confirms these are genuinely even.
                    metrics["brier_contested"] = float((sq * close).sum() / n_close)
                    metrics["acc_contested"] = float((correct * close).sum() / n_close)
                    metrics["contested_label_mean"] = float((target * close).sum() / n_close)
                # Brier SKILL SCORE vs a material-only baseline P_mat = clip(0.5 + 0.5·margin) — the trivial
                # "predict win from the material lead" forecaster. >0 ⇒ the head adds info BEYOND material;
                # ≤0 ⇒ it's no better than counting mons. The headline "information value" number.
                p_mat = (0.5 + 0.5 * margin).clamp(1e-6, 1.0 - 1e-6)
                brier_mat = (((p_mat - target) ** 2) * mask).sum() / n_known
                metrics["brier_material"] = float(brier_mat.item())
                metrics["skill_vs_material"] = (
                    float((1.0 - brier / brier_mat).item()) if float(brier_mat) > 0.0 else 0.0)
        return loss, metrics

    @staticmethod
    def _value_dist_loss(logits, target, atoms):
        """HL-Gauss cross-entropy for the distributional VALUE head (Farebrother et al. 2024) + the
        interpretability diagnostics (v29; designs/ai_v6/design_distributional_value_critic.md).

        ``logits`` [B, N] are the head's per-atom logits; ``target`` [B] (or [B,1]) is the return in the
        SAME space as ``atoms`` [N] (the fixed support — the caller PopArt-normalizes the return when the
        scalar critic does, so the support lives in normalized units). Builds a Gaussian-smoothed soft
        target by integrating N(target, σ_g²) over each bin (σ_g = 0.75·Δ), with the two EDGE bins
        absorbing the outer tails (graceful out-of-support handling — an out-of-range return reads as
        "near the edge", not lost), then cross-entropy against ``log_softmax(logits)``. Returns
        (loss, metrics): ``entropy``/``std``/``pit_mean``/``mean_abs_err`` are the per-decision reads the
        prober renders, aggregated here for the launcher (``pit_mean`` ≈ 0.5 ⟺ calibrated). Pure + static
        → unit-testable without a full PPO. Returns None when nothing is scorable."""
        if logits is None or target is None or atoms is None:
            return None
        z = atoms.to(logits.device).reshape(-1)                      # [N] fixed support
        n = z.numel()
        if n < 2:
            return None
        t = target.to(logits.device).reshape(-1, 1)                  # [B, 1] return (already in z-space)
        delta = (z[-1] - z[0]) / (n - 1)                             # bin width (z is a linspace)
        sigma_g = 0.75 * delta                                       # HL-Gauss smoothing (σ/ς = 0.75)
        inv = 1.0 / (sigma_g * (2.0 ** 0.5))                        # 1/(σ_g·√2) for the erf-CDF
        # Standard-normal CDF Φ(u) = ½(1+erf(u/√2)), evaluated at each bin's upper / lower edge.
        cdf_hi = 0.5 * (1.0 + th.erf((z + 0.5 * delta - t) * inv))   # [B, N]
        cdf_lo = 0.5 * (1.0 + th.erf((z - 0.5 * delta - t) * inv))   # [B, N]
        p = cdf_hi - cdf_lo                                          # [B, N] interior bin masses
        # Edge-bin tail absorption: bin 0 = all mass below its upper edge; bin N-1 = all mass above its
        # lower edge. (Concatenation, not in-place, so it stays autograd-clean — p carries no grad anyway.)
        p = th.cat([cdf_hi[:, :1], p[:, 1:-1], 1.0 - cdf_lo[:, -1:]], dim=1)
        p = p / p.sum(-1, keepdim=True).clamp_min(1e-8)             # renormalize (numerical safety)
        logp = th.log_softmax(logits, dim=-1)                       # [B, N]
        loss = -(p * logp).sum(-1).mean()                           # masked-mean CE
        with th.no_grad():
            probs = th.softmax(logits, dim=-1)                      # [B, N]
            mean = (probs * z).sum(-1)                              # [B] E[Z]
            std = th.sqrt((probs * (z - mean.unsqueeze(-1)) ** 2).sum(-1).clamp_min(0.0))
            entropy = -(probs * logp).sum(-1)                      # [B] nats
            tt = t.reshape(-1)
            pit = (probs * (z.unsqueeze(0) <= tt.unsqueeze(-1)).float()).sum(-1)  # F_pred(target) ≈ PIT
            mean_abs_err = (mean - tt).abs()
        metrics = {
            "ce": float(loss.item()),
            "entropy": float(entropy.mean().item()),
            "std": float(std.mean().item()),
            "pit_mean": float(pit.mean().item()),                  # ≈0.5 ⟺ calibrated
            "mean_abs_err": float(mean_abs_err.mean().item()),
        }
        return loss, metrics

    def _value_loss_from_se(self, se: "th.Tensor") -> "th.Tensor":
        """Tail-weighted value loss from per-sample squared errors `se` (in whatever space the branch
        uses — NORMALIZED under PopArt, so the tail selection is on the same scale the loss trains in).

        value_tail_weight == 0 → plain `se.mean()`, byte-identical to `F.mse_loss`. >0 → blend
        `(1-w)·MSE + w·CVaR`, where CVaR = mean of the worst `_VALUE_TAIL_FRAC` squared errors — it
        upweights the big value misses (the V-tail craters a probe found the critic under-prices)
        WITHOUT biasing the mean (symmetric in error sign), so the de-normalized V the GAE advantages
        read stays unbiased. A scheduling/weighting change, not a new target."""
        mse = se.mean()
        w = self.value_tail_weight
        if w <= 0.0:
            return mse
        flat = se.reshape(-1)
        k = max(1, int(_VALUE_TAIL_FRAC * flat.numel()))
        tail = th.topk(flat, k).values.mean()   # mean of the worst-k squared errors (CVaR)
        return (1.0 - w) * mse + w * tail
