"""The BELIEF BANK as STATIC-SHAPE tensor programs — the twin the compiled learner micro-step folds
(M5 Lane K8, `gen3_static_belief_bank_v1`).

WHY A TWIN. `belief_bank`'s loss functions select the scored slots with boolean-mask indexing
(`x[mask]`), group samples by their believed-slot count with `nonzero`, return `None` on an empty
minibatch after a host read (`bool(mask.any())`), and report every metric with `.item()` — each a
dynamo graph break or a data-dependent shape (the K8 inventory: 58 break sites in this file alone).
The learner's micro-step can only be ONE `fullgraph=True` region when every term is a fixed-shape
program. This module is the same arithmetic with:

  * MASKED reductions instead of selection: per-element losses over EVERY slot, `torch.where(mask, x,
    0)` BEFORE the sum (so a masked-out slot cannot inject a NaN/Inf — Inf * 0 = NaN), divided by
    ``count.clamp(min=1)``; a supervised slot's NaN still reaches the loss (K9(c) fail-closed);
  * the Hungarian per-k groups computed for ALL rows at each k (believed positions sorted first by a
    stable argsort), each group's contribution masked by ``counts == k``;
  * ``present`` (a 0-d bool tensor) where the legacy function returned ``None`` — the term is then
    exactly 0.0 with no gradient, so ``loss + coef * term`` is unchanged;
  * every metric as a ``(value, weight)`` pair of 0-d float32 tensors, weight 1.0 exactly when the
    legacy function returned that key (the host averages present values over the update's
    micro-batches — the old ``list.append`` semantics);
  * the out-of-vocabulary label check (a data-dependent RAISE) moved to the BUFFER, once per update,
    on the host (`check_label_vocab`).

EQUIVALENCE, not identity. The legacy `belief_bank` functions stay as the REFERENCE implementation
(their unit tests and the eager callers keep them); `belief_bank_static_test.py` pins every static
term, metric and gradient to the legacy one — float64 to 1e-12, float32 to the float rounding of a
different summation order — and the learner golden's re-record carries that bar.
"""
from __future__ import annotations

import itertools
from typing import Any, Dict, Iterable, List, NamedTuple, Optional, Tuple

import numpy as np
import torch as th
import torch.nn.functional as F

from agents.training.belief_bank import (_EV_LOSS_SCALE, _EV_LOSS_WEIGHT, _LATENT_STD_TARGET,
                                         _LATENT_VICREG_WEIGHT, _NATURE_CE_WEIGHT,
                                         _SPREAD_LOSS_SCALE, ROWS, BeliefHeadRow)

Metric = Tuple[th.Tensor, th.Tensor]          # (0-d value, 0-d weight in {0., 1.})


class StaticTerm(NamedTuple):
    loss: th.Tensor                            # 0-d; exactly 0.0 when not present
    present: th.Tensor                         # 0-d bool: the legacy function returned a term
    metrics: Dict[str, Metric]


def _w(present: th.Tensor) -> th.Tensor:
    return present.to(th.float32)


def _f(x: Any, like: th.Tensor) -> th.Tensor:
    return x.to(th.float32) if th.is_tensor(x) else th.tensor(float(x), device=like.device)


def _absent(like: th.Tensor) -> StaticTerm:
    """A row whose inputs are missing (a static fact): no term, no metric."""
    z = like.new_zeros(())
    return StaticTerm(z, th.zeros((), dtype=th.bool, device=like.device), {})


#: The k! assignments for every believed-slot count k (TEAM_SIZE = 6 => at most 720), built ONCE at
#: import as Python constants: `itertools.permutations` is not traceable, a constant list is.
_PERM_LISTS: Dict[int, List[Tuple[int, ...]]] = {k: list(itertools.permutations(range(k)))
                                                  for k in range(1, 7)}


def _perms(k: int, device: Any) -> th.Tensor:
    return th.tensor(_PERM_LISTS[k], dtype=th.long, device=device)


def _believed_first(mask: th.Tensor) -> th.Tensor:
    """Slot indices with the masked (True) positions FIRST, each group in ascending index order —
    the order `mask[sel].nonzero()` lists a row's believed slots in."""
    return th.argsort((~mask).to(th.int8), dim=-1, stable=True)


# ------------------------------------------------------------------------------------ the heads
def spread_terms(sb: Optional[th.Tensor], belief_spread: Optional[th.Tensor],
                 belief_spread_mask: Optional[th.Tensor]) -> StaticTerm:
    """`belief_bank.spread_belief_loss`, static."""
    if sb is None or belief_spread is None or belief_spread_mask is None:
        return _absent(sb if sb is not None else th.zeros(()))
    target = belief_spread.to(sb.device).float()                          # [B,6,5]
    mask = belief_spread_mask.to(sb.device).float() > 0.5                  # [B,6]
    n = mask.sum().to(sb.dtype)
    present = n > 0
    d = n.clamp(min=1)
    m3 = mask.unsqueeze(-1)
    el = F.smooth_l1_loss(sb / _SPREAD_LOSS_SCALE, target / _SPREAD_LOSS_SCALE, reduction="none")
    loss = th.where(m3, el, el.new_zeros(())).sum() / (d * sb.shape[-1])
    with th.no_grad():
        err = sb - target
        mae = th.where(m3, err.abs(), err.new_zeros(())).sum() / (d * sb.shape[-1])
        amax = target.argmax(dim=-1, keepdim=True)                         # [B,6,1]
        lb = th.where(mask, err.gather(-1, amax).squeeze(-1), err.new_zeros(())).sum() / d
    w = _w(present)
    return StaticTerm(loss, present, {"mae": (mae.float(), w), "largest_bias": (lb.float(), w),
                                      "n_slots": (n.float(), w),
                                      "mask_rate": (mask.float().mean(), w)})


def nature_ev_terms(nat_logits: Optional[th.Tensor], ev_pred: Optional[th.Tensor],
                    belief_nature: Optional[th.Tensor], belief_nature_mask: Optional[th.Tensor],
                    belief_ev: Optional[th.Tensor],
                    belief_ev_mask: Optional[th.Tensor]) -> StaticTerm:
    """`belief_bank.nature_ev_belief_loss`, static."""
    if nat_logits is None or ev_pred is None or belief_nature is None or belief_ev is None \
            or belief_nature_mask is None:
        return _absent(nat_logits if nat_logits is not None else th.zeros(()))
    dev = nat_logits.device
    nmask = belief_nature_mask.to(dev).float() > 0.5                       # [B,6]
    n = nmask.sum().to(nat_logits.dtype)
    present = n > 0
    d = n.clamp(min=1)
    n_cls = nat_logits.shape[-1]
    nat_true = belief_nature.to(dev).long()
    safe = th.where(nmask, nat_true, th.zeros_like(nat_true)).clamp(0, n_cls - 1)
    per = F.cross_entropy(nat_logits.reshape(-1, n_cls), safe.reshape(-1),
                          reduction="none").reshape(nmask.shape)
    nat_ce = th.where(nmask, per, per.new_zeros(())).sum() / d
    ev_true = belief_ev.to(dev).float()
    evm = (belief_ev_mask.to(dev).float() > 0.5) if belief_ev_mask is not None else nmask
    n2 = evm.sum().to(ev_pred.dtype)
    d2 = n2.clamp(min=1) * ev_pred.shape[-1]
    em3 = evm.unsqueeze(-1)
    el = F.smooth_l1_loss(ev_pred / _EV_LOSS_SCALE, ev_true / _EV_LOSS_SCALE, reduction="none")
    ev_loss = th.where(em3, el, el.new_zeros(())).sum() / d2
    # gated: the legacy function returns None when no NATURE slot is scored, even if EV slots are
    loss = th.where(present, _NATURE_CE_WEIGHT * nat_ce + _EV_LOSS_WEIGHT * ev_loss,
                    nat_ce.new_zeros(()))
    with th.no_grad():
        ev_mae = th.where(em3, (ev_pred - ev_true).abs(), el.new_zeros(())).sum() / d2
        hit = (nat_logits.argmax(dim=-1) == safe).to(nat_logits.dtype)
        acc = th.where(nmask, hit, hit.new_zeros(())).sum() / d
    w = _w(present)
    return StaticTerm(loss, present, {"nature_acc": (acc.float(), w), "nature_ce": (nat_ce.detach().float(), w),
                                      "ev_mae": (ev_mae.float(), w), "n_slots": (n.float(), w),
                                      "mask_rate": (nmask.float().mean(), w)})


def _masked_ce_terms(logits: Optional[th.Tensor], label: Optional[th.Tensor],
                     mask_in: Optional[th.Tensor]) -> StaticTerm:
    """The hp-type / item shape: a CE over the masked slots, label PAD (-1) clamped to 0."""
    if logits is None or label is None or mask_in is None:
        return _absent(logits if logits is not None else th.zeros(()))
    dev = logits.device
    lab = label.to(dev).long()
    mask = mask_in.to(dev).float() > 0.5
    n = mask.sum().to(logits.dtype)
    present = n > 0
    d = n.clamp(min=1)
    n_cls = logits.shape[-1]
    safe = th.where(mask, lab.clamp(min=0), th.zeros_like(lab)).clamp(max=n_cls - 1)
    per = F.cross_entropy(logits.reshape(-1, n_cls), safe.reshape(-1),
                          reduction="none").reshape(mask.shape)
    loss = th.where(mask, per, per.new_zeros(())).sum() / d
    with th.no_grad():
        hit = (logits.argmax(dim=-1) == safe).to(logits.dtype)
        acc = th.where(mask, hit, hit.new_zeros(())).sum() / d
    w = _w(present)
    return StaticTerm(loss, present, {"acc": (acc.float(), w), "n_slots": (n.float(), w),
                                      "mask_rate": (mask.float().mean(), w)})


def hp_type_terms(logits: Optional[th.Tensor], hp_type_label: Optional[th.Tensor],
                  hp_type_mask: Optional[th.Tensor]) -> StaticTerm:
    """`belief_bank.hp_type_belief_loss`, static."""
    return _masked_ce_terms(logits, hp_type_label, hp_type_mask)


def item_terms(logits: Optional[th.Tensor], item_label: Optional[th.Tensor],
               item_mask: Optional[th.Tensor]) -> StaticTerm:
    """`belief_bank.item_belief_loss`, static."""
    return _masked_ce_terms(logits, item_label, item_mask)


def _multi_hot(ids: th.Tensor, n: int, like: th.Tensor) -> th.Tensor:
    """[..., K] move ids (-1 = PAD) -> [..., n] multi-hot (1.0 at every valid id; duplicates 1)."""
    valid = ids >= 0
    out = th.zeros(ids.shape[:-1] + (n,), dtype=like.dtype, device=like.device)
    out = out.scatter_add(-1, ids.clamp(min=0, max=n - 1), valid.to(like.dtype))
    return out.clamp(max=1.0)


def move_belief_terms(ml: Optional[th.Tensor], known_moves: Optional[th.Tensor],
                      belief_moves: Optional[th.Tensor], mode: str,
                      hypothesis: Any = None, belief_species: Optional[th.Tensor] = None) -> StaticTerm:
    """`belief_bank.move_belief_loss`, static (its vocab RAISE is `check_label_vocab`'s). Under
    fixed_mass (``hypothesis`` given) the unrevealed population is §3.4's iff-present rule."""
    if ml is None:
        return _absent(th.zeros(()))
    dev = ml.device
    B, n_slots, M = ml.shape
    total = ml.new_zeros(())
    count = ml.new_zeros(())
    tp = ml.new_zeros(())
    pred_pos = ml.new_zeros(())
    true_pos = ml.new_zeros(())
    n_rev = ml.new_zeros(())
    n_unrev = ml.new_zeros(())
    if mode in ("revealed", "both") and known_moves is not None:
        km = known_moves.long().to(dev)                                     # [B,6,4]
        slot_has = (km >= 0).any(-1)                                        # [B,6]
        mh = _multi_hot(km, M, ml)                                          # [B,6,M]
        per_slot = F.binary_cross_entropy_with_logits(ml, mh, reduction="none").mean(-1)
        total = total + th.where(slot_has, per_slot, per_slot.new_zeros(())).sum()
        cnt = slot_has.sum().to(ml.dtype)
        count = count + cnt
        n_rev = n_rev + cnt
        with th.no_grad():
            sel = slot_has.unsqueeze(-1)
            pp = (ml > 0.0) & sel
            mb = (mh > 0.5) & sel
            tp = tp + (pp & mb).sum().to(ml.dtype)
            pred_pos = pred_pos + pp.sum().to(ml.dtype)
            true_pos = true_pos + mb.sum().to(ml.dtype)
    if mode in ("unrevealed", "both") and belief_moves is not None and hypothesis is not None:
        # gen3_x5_belief_tokens_v1: hypothesis seats, supervised iff the species is present
        from agents.model.hypothesis_set import hypothesis_moves_targets
        if belief_species is None:
            raise ValueError("move_belief_terms under fixed_mass needs belief_species")
        mh, sup = hypothesis_moves_targets(hypothesis.slot_species, hypothesis.slot_is_hypothesis,
                                           belief_species, belief_moves, M, ml)
        per_slot = F.binary_cross_entropy_with_logits(ml, mh, reduction="none").mean(-1)
        total = total + th.where(sup, per_slot, per_slot.new_zeros(())).sum()
        cnt = sup.sum().to(ml.dtype)
        count = count + cnt
        n_unrev = n_unrev + cnt
        with th.no_grad():
            sel = sup.unsqueeze(-1)
            pp = (ml > 0.0) & sel
            mb = (mh > 0.5) & sel
            tp = tp + (pp & mb).sum().to(ml.dtype)
            pred_pos = pred_pos + pp.sum().to(ml.dtype)
            true_pos = true_pos + mb.sum().to(ml.dtype)
    elif mode in ("unrevealed", "both") and belief_moves is not None:
        bm = belief_moves.long().to(dev)                                    # [B,6,4]
        slot_has = (bm >= 0).any(-1)                                        # [B,6]
        counts = slot_has.sum(1)                                            # [B]
        order = _believed_first(slot_has)                                   # [B,6]
        for k in range(1, n_slots + 1):
            rows = counts == k                                              # [B]
            idx = order[:, :k]                                              # [B,k]
            preds = ml.gather(1, idx.unsqueeze(-1).expand(B, k, M))         # [B,k,M]
            ids = bm.gather(1, idx.unsqueeze(-1).expand(B, k, bm.shape[-1]))
            tgt = _multi_hot(ids, M, ml)                                    # [B,k,M]
            cost = -th.einsum("bkm,bjm->bkj", preds, tgt)                   # [B,k,k]
            perms = _perms(k, dev)
            ii = th.arange(k, device=dev).view(1, k).expand(perms.shape[0], k)
            best = perms[cost[:, ii, perms].sum(-1).argmin(1)]              # [B,k]
            matched = th.gather(tgt, 1, best.unsqueeze(-1).expand(B, k, M))
            per_slot = F.binary_cross_entropy_with_logits(preds, matched, reduction="none").mean(-1)
            r2 = rows.unsqueeze(-1)
            total = total + th.where(r2, per_slot, per_slot.new_zeros(())).sum()
            cnt = rows.sum().to(ml.dtype) * k
            count = count + cnt
            n_unrev = n_unrev + cnt
            with th.no_grad():
                r3 = r2.unsqueeze(-1)
                pp = (preds > 0.0) & r3
                mb = (matched > 0.5) & r3
                tp = tp + (pp & mb).sum().to(ml.dtype)
                pred_pos = pred_pos + pp.sum().to(ml.dtype)
                true_pos = true_pos + mb.sum().to(ml.dtype)
    present = count > 0
    loss = total / count.clamp(min=1)
    w = _w(present)
    with th.no_grad():
        prec = th.where(pred_pos > 0, tp / pred_pos.clamp(min=1), tp.new_zeros(()))
        rec = th.where(true_pos > 0, tp / true_pos.clamp(min=1), tp.new_zeros(()))
    return StaticTerm(loss, present, {"bce": (loss.detach().float(), w), "precision": (prec.float(), w),
                                      "recall": (rec.float(), w),
                                      "revealed_slots": (n_rev.float(), w),
                                      "unrevealed_slots": (n_unrev.float(), w)})


def move_latent_terms(ml: Optional[th.Tensor], latent_table: Optional[th.Tensor],
                      known_moves: Optional[th.Tensor]) -> StaticTerm:
    """`belief_bank.move_belief_latent_loss`, static."""
    if ml is None or latent_table is None or known_moves is None:
        return _absent(ml if ml is not None else th.zeros(()))
    dev = ml.device
    km = known_moves.long().to(dev)                                         # [B,6,4]
    valid = km >= 0
    slot_has = valid.any(-1)                                                # [B,6]
    n = slot_has.sum().to(ml.dtype)
    present = n > 0
    d = n.clamp(min=1)
    pred_latent = F.softmax(ml, dim=-1) @ latent_table                      # [B,6,D]
    move_lat = latent_table[km.clamp(min=0)] * valid.unsqueeze(-1).to(latent_table.dtype)
    tgt = (move_lat.sum(2) / valid.sum(-1, keepdim=True).clamp(min=1).to(latent_table.dtype)).detach()
    cos = F.cosine_similarity(pred_latent, tgt, dim=-1)                     # [B,6]
    cos_loss = th.where(slot_has, 1.0 - cos, cos.new_zeros(())).sum() / d
    m3 = slot_has.unsqueeze(-1)
    mean = th.where(m3, pred_latent, pred_latent.new_zeros(())).sum((0, 1)) / d
    var = th.where(m3, (pred_latent - mean) ** 2, pred_latent.new_zeros(())).sum((0, 1)) / d
    std = th.sqrt(var + 1e-4)                                               # [D]
    vicreg = F.relu(_LATENT_STD_TARGET - std).mean()
    # gated: with no scored slot the VICReg floor alone is non-zero (std of nothing = 0.01)
    loss = th.where(present, cos_loss + _LATENT_VICREG_WEIGHT * vicreg, cos_loss.new_zeros(()))
    w = _w(present)
    with th.no_grad():
        cmean = th.where(slot_has, cos, cos.new_zeros(())).sum() / d
    return StaticTerm(loss, present, {"cosine": (cmean.float(), w), "std": (std.detach().mean().float(), w),
                                      "slots": (n.float(), w)})


def belief_aux_terms(bl: Optional[Dict[str, th.Tensor]], sp_labels: Optional[th.Tensor],
                     mv_labels: Optional[th.Tensor], moves_weight: float = 1.0) -> StaticTerm:
    """`belief_bank.belief_aux_loss` (the hidden-team Hungarian aux), static (its vocab RAISE is
    `check_label_vocab`'s). ``moves_weight`` is a run constant (static)."""
    if bl is None or sp_labels is None or mv_labels is None:
        return _absent(th.zeros(()))
    sp_logits, mv_logits = bl["species"], bl["moves"]
    dev = sp_logits.device
    B, n_slots, S = sp_logits.shape
    M = mv_logits.shape[-1]
    sp = sp_labels.long().to(dev)
    mv = mv_labels.long().to(dev)
    believed = sp >= 0                                                      # [B,6]
    counts = believed.sum(1)                                                # [B]
    order = _believed_first(believed)
    do_moves = moves_weight != 0.0
    logp_all = th.log_softmax(sp_logits, dim=-1)                            # [B,6,S]
    ce_sum = sp_logits.new_zeros(())
    n_slots_t = sp_logits.new_zeros(())
    n_correct = sp_logits.new_zeros(())
    bce_sum = sp_logits.new_zeros(())
    bce_n = sp_logits.new_zeros(())
    tp = sp_logits.new_zeros(())
    pred_pos = sp_logits.new_zeros(())
    true_pos = sp_logits.new_zeros(())
    for k in range(1, n_slots + 1):
        rows = counts == k                                                  # [B]
        idx = order[:, :k]                                                  # [B,k]
        pred_logp = logp_all.gather(1, idx.unsqueeze(-1).expand(B, k, S))   # [B,k,S]
        tgt_sp = sp.gather(1, idx).clamp(0, S - 1)                          # [B,k]
        cost = -th.gather(pred_logp, 2, tgt_sp[:, None, :].expand(B, k, k))  # [B,k,k]
        perms = _perms(k, dev)
        ii = th.arange(k, device=dev).view(1, k).expand(perms.shape[0], k)
        best = perms[cost[:, ii, perms].sum(-1).argmin(1)]                  # [B,k]
        matched_sp = th.gather(tgt_sp, 1, best)                             # [B,k]
        ce = -th.gather(pred_logp, 2, matched_sp.unsqueeze(-1)).squeeze(-1)  # [B,k]
        r2 = rows.unsqueeze(-1)
        ce_sum = ce_sum + th.where(r2, ce, ce.new_zeros(())).sum()
        n_slots_t = n_slots_t + rows.sum().to(ce.dtype) * k
        with th.no_grad():
            hit = (pred_logp.argmax(-1) == matched_sp) & r2
            n_correct = n_correct + hit.sum().to(ce.dtype)
        if do_moves:
            label_slot = th.gather(idx, 1, best)                            # [B,k]
            mv_pred = mv_logits.gather(1, idx.unsqueeze(-1).expand(B, k, M))
            mv_ids = mv.gather(1, label_slot.unsqueeze(-1).expand(B, k, mv.shape[-1]))
            mh = _multi_hot(mv_ids, M, mv_pred)                             # [B,k,M]
            slot_has_moves = (mv_ids >= 0).any(-1) & r2                     # [B,k]
            per = F.binary_cross_entropy_with_logits(mv_pred, mh, reduction="none").mean(-1)
            bce_sum = bce_sum + th.where(slot_has_moves, per, per.new_zeros(())).sum()
            bce_n = bce_n + slot_has_moves.sum().to(per.dtype)
            with th.no_grad():
                r3 = r2.unsqueeze(-1)
                pp = (mv_pred > 0.0) & r3
                mb = (mh > 0.5) & r3
                tp = tp + (pp & mb).sum().to(per.dtype)
                pred_pos = pred_pos + pp.sum().to(per.dtype)
                true_pos = true_pos + mb.sum().to(per.dtype)
    present = n_slots_t > 0
    ce_mean = ce_sum / n_slots_t.clamp(min=1)
    bce = th.where(bce_n > 0, bce_sum / bce_n.clamp(min=1), bce_sum.new_zeros(()))
    aux = ce_mean + moves_weight * bce
    w = _w(present)
    with th.no_grad():
        n_samples = (counts > 0).sum().to(ce_mean.dtype)
        acc = n_correct / n_slots_t.clamp(min=1)
        metrics = {
            "species_ce": (ce_mean.detach().float(), w),
            "moves_bce": (bce.detach().float(), w),
            "species_acc": (acc.float(), w),
            "species_acc_above_chance": ((acc - 1.0 / S).float(), w),
            "moves_precision": (th.where(pred_pos > 0, tp / pred_pos.clamp(min=1),
                                         tp.new_zeros(())).float(), w),
            "moves_recall": (th.where(true_pos > 0, tp / true_pos.clamp(min=1),
                                      tp.new_zeros(())).float(), w),
            "k_mean": ((n_slots_t / n_samples.clamp(min=1)).float(), w),
            "coverage": ((n_samples / B).float(), w),
            "mask_rate": ((n_slots_t / believed.numel()).float(), w),
        }
    return StaticTerm(aux, present, metrics)


def hypothesis_set_terms(hs: Any, bl: Optional[Dict[str, th.Tensor]], sp_labels: Optional[th.Tensor],
                         mv_labels: Optional[th.Tensor], moves_weight: float = 1.0) -> StaticTerm:
    """X5's hidden-team supervision under ``--belief-tokens fixed_mass`` (`gen3_x5_hypothesis_set_v1`,
    design §3.2 "Supervision"; the `hidden_team` row's replacement — the two are gated exclusively):

      * the PRESENCE BCE — the set BCE on the T0 construction's exact logit ``a + τ`` (``hs.species``),
        the ONLY gradient δ_θ receives (M10);
      * BeliefHead RE-TARGETED to the same set BCE: its per-slot species logits reduced to one team
        score (`belief_head_team_scores`), through the same fixed-size construction over the same V;
      * ``moves_weight`` × BeliefHead's MOVES BCE on the hypothesis seats, supervised iff the seat's
        hypothesis species IS on the true unseen team (§3.4; the Hungarian matching is retired).

    No slot matching anywhere (unseen slots are exchangeable in gen 3). Static-shape: every row is
    scored by mask. ``present`` iff some row was scored by either set BCE."""
    from agents.model.hypothesis_set import (belief_head_team_scores, fixed_mass_presence,
                                             hypothesis_moves_bce, label_multi_hot, set_bce)
    if hs is None or bl is None or sp_labels is None or mv_labels is None:
        return _absent(th.zeros(()))
    pres = hs.species
    logits = pres.logits
    S = logits.shape[-1]
    y = label_multi_hot(sp_labels, S, logits)
    p_loss, p_n, p_bad = set_bce(logits, pres, y)
    believed = hs.slot_is_hypothesis
    team = belief_head_team_scores(bl["species"], believed)                       # [B,S] graph → BeliefHead
    bh = fixed_mass_presence(team, pres.cand, pres.k)
    b_loss, b_n, _ = set_bce(bh.logits, bh, y)
    m_loss, m_n = hypothesis_moves_bce(bl["moves"], hs.slot_species, believed, sp_labels, mv_labels)
    aux = p_loss + b_loss + moves_weight * m_loss
    present = (p_n > 0) | (b_n > 0)
    w = _w(present)
    with th.no_grad():
        B = logits.shape[0]
        live = pres.live
        nl = live.sum().clamp(min=1).to(th.float32)
        in_list = ((y > 0.5) & (hs.rank < pres.k.unsqueeze(-1)) & pres.cand).sum(-1).to(th.float32)
        recall = th.where(live, in_list / pres.k.clamp(min=1).to(th.float32), th.zeros_like(in_list)).sum() / nl
        k_f = pres.k.clamp(min=1).to(th.float32)
        other_share = th.where(live, hs.other_mass.float() / k_f, th.zeros_like(k_f)).sum() / nl
        metrics = {
            "presence_bce": (p_loss.detach().float(), w),
            "beliefhead_set_bce": (b_loss.detach().float(), w),
            "hyp_moves_bce": (m_loss.detach().float(), w),
            "hyp_recall": (recall, w),
            "other_share": (other_share, w),
            "presence_label_mismatch": ((p_bad / max(B, 1)).float(), w),
            "hyp_moves_supervised": (m_n.float(), w),
        }
    return StaticTerm(aux, present, metrics)


#: row name -> its static twin (the registry walk below dispatches on it).
_STATIC_FNS = {
    "hidden_team": belief_aux_terms,
    "hidden_team_set": hypothesis_set_terms,
    "move_belief": move_belief_terms,
    "move_latent": move_latent_terms,
    "spread": spread_terms,
    "nature_ev": nature_ev_terms,
    "hp_type": hp_type_terms,
    "item": item_terms,
}


def _resolve_arg(src: str, key: str, extractor: Any, observations: Dict[str, Any],
                 params: Dict[str, Any]) -> Any:
    if src == "stash":
        return extractor.belief_supervision(key)
    if src == "obs":
        return observations.get(key)
    if src == "attr":
        return getattr(extractor, key, None)
    if src == "param":
        return params[key]
    raise ValueError(f"unknown arg source {src!r}")


class StaticRowTerm(NamedTuple):
    row: BeliefHeadRow
    term: th.Tensor                            # coef x loss (0.0 when not present)
    present: th.Tensor
    metrics: Dict[str, Metric]                 # PREFIXED (row.prefix + key), incl. row.loss_key


def compute_static(extractor: Any, observations: Dict[str, Any], coefs: Dict[str, Any],
                   gates: Dict[str, bool], site: str, params: Optional[Dict[str, Any]] = None,
                   rows: Iterable[BeliefHeadRow] = ROWS) -> List[StaticRowTerm]:
    """`belief_bank.compute`, static: every gated row AT ``site`` in REGISTRY ORDER (the
    float-addition contract), each term already scaled by its coefficient (a tensor or a float).
    A row the legacy walk would have skipped for a STATIC reason (its inputs absent) is omitted; a
    row skipped for a DATA reason (nothing scored on this micro-batch) is returned with
    ``present`` False and a 0.0 term."""
    params = params or {}
    out: List[StaticRowTerm] = []
    for row in rows:
        if row.site != site or not gates.get(row.gate, False):
            continue
        argv = [_resolve_arg(src, k, extractor, observations, params) for src, k in row.args]
        res = _STATIC_FNS[row.name](*argv)
        if not res.metrics:                    # static absence (`_absent`): the legacy `None`
            continue
        mets = {row.prefix + k: v for k, v in res.metrics.items()}
        mets[row.prefix + row.loss_key] = (res.loss.detach().float(), _w(res.present))
        out.append(StaticRowTerm(row, coefs[row.coef] * res.loss, res.present, mets))
    return out


# ------------------------------------------------------------------------------- vocab, on host
def check_label_vocab(observations: Dict[str, Any], *, n_species: Optional[int],
                      n_moves: Optional[int]) -> None:
    """The legacy loss functions' FAIL-LOUD vocab checks, once per update on the BUFFER (host
    numpy, no device sync): a believed species / move label id at or above the head's vocabulary
    means the label↔embedding num space is corrupt (real Gen-3 nums are all < 400). Raises
    ValueError with the legacy messages' substance."""
    def _bad(key: str, n: Optional[int]) -> Optional[int]:
        a = observations.get(key) if isinstance(observations, dict) else None
        if a is None or n is None:
            return None
        arr = np.asarray(a)
        hi = int(arr.max()) if arr.size else -1
        return hi if hi >= int(n) else None

    sp = _bad("belief_species", n_species)
    if sp is not None:
        raise ValueError(f"belief label out of vocab: species max {sp} (n_species {n_species}) — "
                         "the embedding-num pipeline is corrupt (real Gen-3 nums are all < 400).")
    for key in ("belief_moves", "known_moves"):
        mv = _bad(key, n_moves)
        if mv is not None:
            raise ValueError(f"move-belief label out of vocab: {key} max {mv} (n_moves {n_moves}) — "
                             "the embedding-num pipeline is corrupt (real Gen-3 move nums are all "
                             "< 400).")
