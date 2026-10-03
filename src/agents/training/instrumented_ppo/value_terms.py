"""The CRITIC-side loss terms.

* `_win_prob_loss` — the auxiliary win-probability BCE, with the contested-band readout that is
  the term's actual information content (a blowout's P(win) is recoverable from material).
"""
import torch as th
from torch.nn import functional as F

from agents.training.instrumented_ppo.constants import _WIN_CONTESTED_TAU


class ValueTerms:
    """The win-prob loss."""

    @staticmethod
    def _win_prob_loss(logits, target, mask, margin=None):
        """Supervised BCE loss for the auxiliary WIN-PROBABILITY head (``last_win_prob_logits`` [B,1]).

        ``target`` [B,1] = the Monte-Carlo episode OUTCOME (win=1 / loss=0) propagated to every step of
        the episode by the Rust collector's fill (`rust_rollout.store`, before ``train()``);
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
        ``skill_vs_material`` > 0 ⇒ the head beats 'just count the mons'. **A margin with no SPREAD
        is treated as absent** (`gen3_tb_relevance_v1`) — it cannot stratify, and the six tags it
        would produce are then copies of their pooled siblings plus two constants."""
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
        # gen3_tb_relevance_v1: a CONSTANT margin cannot stratify anything, and publishing the
        # split anyway is worse than publishing nothing. `win_margin` (`material_margin.py`) was once
        # a by-product of the material PBRS term, so a composition without that term left it
        # identically 0.0; it is computed unconditionally now, and this guard stays as the consumer
        # side of the same contract. The whole family then degenerates: `close` is all-ones so every
        # `*_contested` tag is a byte-identical copy of its pooled sibling, `contested_frac` is a
        # flat 1.0, the material baseline `p_mat` is a constant 0.5 so `brier_material` is a flat
        # 0.25, and `skill_vs_material` collapses to the affine transform `1 − 4·brier`. All six
        # read as measurements. Measured on ai_v12_01_winprob_critic: exactly that, for 35 rollouts.
        if margin is not None and float(margin.max() - margin.min()) > 0.0:
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
