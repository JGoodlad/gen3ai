"""The three DISTILLATION families' per-minibatch loss terms.

* `_searchteacher_loss` — advantage-weighted CE toward the search-verified better action (AWR).
* `_opd_loss` — on-policy self-distillation, KL toward the improved distribution pi'.
* `_distill_loss` / `_value_distill_mse` / `_value_feat_distill` — exploiter distillation: the
  masked policy KL, the scalar value MSE, and the FitNets cosine hint on `value_pooled`.

Every one is a pure `(...) -> (loss, metrics) | None` staticmethod-or-method; the fold that adds
them to `loss` lives in `ppo.py::train`, in the one place the ordering is readable.
"""
import torch as th
from torch.nn import functional as F


class DistillTerms:
    """The search-teacher, OPD and exploiter-distillation loss terms."""

    @staticmethod
    def _searchteacher_loss(logits, action_mask, better_action, advantage,
                            beta_awr: float = 1.0, w_clip: float = 20.0):
        """ADVANTAGE-WEIGHTED CE (AWR) toward the verified-better action A* (the search-teacher signal).

        ``logits`` [B, n_actions] = ``policy.get_distribution(obs_dict).distribution.logits`` on the
        CORRECTION obs (its own forward); ``better_action`` [B] = A*; ``advantage`` [B] = the CONFIRMED
        win-rate improvement of A* vs the EXACT opponent (> 0, already CI-gated) — NOT a critic
        advantage. Weight ``w = clamp(exp(adv/β), max=w_clip)`` up-weights high-margin corrections. The
        policy CE is in logit space (no PopArt). Returns ``(loss, metrics)`` or ``None`` on empty.
        """
        if logits is None or better_action is None or better_action.numel() == 0:
            return None
        masked = logits + (action_mask.to(logits.dtype) - 1.0) * 1e9   # illegal → −inf (A* is always legal)
        w = th.exp(advantage / beta_awr).clamp(max=w_clip)
        ce = F.cross_entropy(masked, better_action.long(), reduction="none")   # [B]
        loss = (w * ce).sum() / w.sum().clamp(min=1e-6)
        with th.no_grad():
            agree = (masked.argmax(-1) == better_action).float().mean()
            metrics = {"loss": float(loss), "agree_rate": float(agree),
                       "mean_adv": float(advantage.mean()), "mean_w": float(w.mean()),
                       "ce": float(ce.mean()), "n": int(better_action.numel())}
        return loss, metrics

    @staticmethod
    def _opd_loss(logits, action_mask, pi_target):
        """ON-POLICY SELF-DISTILLATION (OPD) — KL(π' ‖ π_student) toward the FULL improved distribution
        π' (the beam's per-action backed-up values, softmaxed over legal actions). The KL-form upgrade of
        the AWR :meth:`_searchteacher_loss` (which distils only the single action A*).

        ``logits`` [B, n_actions] = ``policy.get_distribution(obs_dict).distribution.logits`` on the
        CORRECTION obs (its own forward); ``action_mask`` [B, n_actions] = the legal mask; ``pi_target``
        [B, n_actions] = π' (already over LEGAL actions, 0 on illegal, L1-normed). Illegal logits are
        masked to −∞ so the student log-probs are over the legal set (matching π'). Forward KL
        ``Σ p_tgt·(log p_tgt − log p_student)``, mean over the batch. Returns ``(kl, metrics)`` or
        ``None`` on an empty / absent π' (the buffer had no OPD targets — an AWR-only run). Pure + static
        so it unit-tests without a full PPO."""
        if logits is None or pi_target is None or pi_target.numel() == 0:
            return None
        masked = logits + (action_mask.to(logits.dtype) - 1.0) * 1e9   # illegal → −inf (π' is 0 there)
        logp = F.log_softmax(masked, dim=-1)                           # student log-probs over legal
        p_tgt = pi_target                                              # already legal-only, illegal 0
        kl = (p_tgt * (th.log(p_tgt.clamp_min(1e-9)) - logp)).sum(-1).mean()
        with th.no_grad():
            ent = -(p_tgt * th.log(p_tgt.clamp_min(1e-9))).sum(-1).mean()   # π' sharpness (low = decisive)
            agree = (masked.argmax(-1) == p_tgt.argmax(-1)).float().mean()  # student ↔ π' mode agreement
            metrics = {"kl": float(kl), "pi_target_entropy": float(ent),
                       "agree_rate": float(agree), "n": int(pi_target.shape[0])}
        return kl, metrics

    @staticmethod
    def _distill_loss(student_logits, teacher_logits, action_mask, distill_mask):
        """EXPLOITER DISTILLATION — masked ON-POLICY KL(π_teacher ‖ π_student) over the rollout minibatch.

        ``student_logits`` / ``teacher_logits`` [B, n_actions] = raw ``get_distribution(...).distribution.
        logits`` (the teacher's ALREADY under no_grad — it is frozen). ``action_mask`` [B, n_actions] = the
        legal mask; ``distill_mask`` [B] or [B,1] = 1 on states where the trainee pilots the TEACHER's team
        (the only states where the teacher's advice is on-distribution — elsewhere it would corrupt the
        other teams, so those rows are excluded). Illegal logits → −∞ so both sides normalise over the legal
        set; forward KL ``Σ p_teacher·(log p_teacher − log p_student)`` per row, masked-mean over the
        teacher-team rows. Returns ``(kl, metrics)`` or ``None`` when the minibatch has no teacher-team rows
        (the None guard keeps an empty subset from NaN-poisoning the loss). Pure + static → unit-testable."""
        if student_logits is None or teacher_logits is None or distill_mask is None:
            return None
        m = distill_mask.reshape(-1).to(student_logits.dtype)              # [B] 1.0 on teacher-team states
        n_on = m.sum()
        if float(n_on) < 1.0:
            return None                                                   # no teacher-team states this batch
        neg = (action_mask.to(student_logits.dtype) - 1.0) * 1e9          # illegal → −inf (both sides)
        logp_s = F.log_softmax(student_logits + neg, dim=-1)              # student log-probs over legal
        p_t = F.softmax(teacher_logits + neg, dim=-1)                     # teacher probs over legal (detached)
        kl_row = (p_t * (th.log(p_t.clamp_min(1e-9)) - logp_s)).sum(-1)   # [B] forward KL per state
        loss = (kl_row * m).sum() / n_on.clamp(min=1e-6)                  # masked-mean over teacher-team rows
        with th.no_grad():
            agree_row = ((student_logits + neg).argmax(-1) == (teacher_logits + neg).argmax(-1)).float()
            metrics = {"kl": float(loss),
                       "agree_rate": float((agree_row * m).sum() / n_on.clamp(min=1e-6)),
                       "coverage": float(m.mean()),                       # fraction of minibatch on teacher team
                       "n": int(n_on.item())}
        return loss, metrics

    @staticmethod
    def _value_distill_mse(student_values, teacher_values, distill_mask, popart=None):
        """VALUE DISTILLATION — masked MSE(V_teacher ‖ V_student) over the teacher-team rows.

        ``student_values`` [B] carries grad; ``teacher_values`` [B] is the frozen teacher's (real-unit,
        already under no_grad). ``distill_mask`` [B]/[B,1] = 1 on teacher-team states. When a PopArt
        normalizer is given, both are mapped to the student's normalized frame first (so the coef is
        scale-comparable with the value loss); else a raw-unit MSE. Returns the masked-mean SE, or None
        when no teacher-team rows (the None guard, like _distill_loss). Pure + static → unit-testable."""
        if student_values is None or teacher_values is None or distill_mask is None:
            return None
        m = distill_mask.reshape(-1).to(student_values.dtype)
        n_on = m.sum()
        if float(n_on) < 1.0:
            return None
        sv, tv = student_values.reshape(-1), teacher_values.reshape(-1)
        if popart is not None:
            se = (popart.normalize(sv) - popart.normalize(tv)) ** 2
        else:
            se = (sv - tv) ** 2
        return (se * m).sum() / n_on.clamp(min=1e-6)

    @staticmethod
    def _value_feat_distill(student_feat, teacher_feat, distill_mask):
        """FITNETS VALUE-FEATURE DISTILLATION — masked COSINE distance between the value-CLS pools.

        The FitNets (Romero 2015) "hint" upgrade of scalar value distillation: instead of only matching the
        teacher's scalar V (which collapses to a ~4-dim critic — `_value_distill_mse` CRYSTALLIZES the head,
        value_cls rank DROPS), regress the student's INTERMEDIATE 128-dim `value_pooled` (the extractor's
        `last_value_pooled` HINT layer) toward the teacher's on the teacher-team states, so the trunk inherits
        the teacher's per-team value STRUCTURE, not just its output.

        ``student_feat`` [B,D] carries grad (the student's live `value_pooled`); ``teacher_feat`` [B,D] is the
        frozen teacher's (already under no_grad). ``distill_mask`` [B]/[B,1] = 1 on teacher-team states. COSINE
        (not MSE): the geometry analysis (`tmp/fitnet_analysis.py`) found the teachers' value subspaces are
        low-rank, COMPLEMENTARY (TSS orthogonal, collective effR ~12), and NON-competing (all pull-cosines
        positive) — so a scale-free directional pull transfers the correct structure without over-constraining
        a low-rank target the way an MSE on raw magnitudes would (the student/teacher live in separately-rotated
        128-dim bases, so absolute coordinates aren't comparable; direction is). Loss = ``1 − cos`` per row,
        masked-mean over the teacher-team rows. Returns the masked-mean cosine distance, or None when no
        teacher-team rows (the None guard, like `_value_distill_mse`). Pure + static → unit-testable."""
        if student_feat is None or teacher_feat is None or distill_mask is None:
            return None
        m = distill_mask.reshape(-1).to(student_feat.dtype)
        n_on = m.sum()
        if float(n_on) < 1.0:
            return None
        cos = F.cosine_similarity(student_feat, teacher_feat, dim=-1, eps=1e-6)   # [B] direction match per row
        dist = 1.0 - cos                                                          # [B] cosine DISTANCE (0=aligned)
        return (dist * m).sum() / n_on.clamp(min=1e-6)
