"""The BELIEF BANK — one declarative home for the supervised belief-head losses.

`design_unified_belief.md` §4's code-shape half: today every hidden quantity costs a full
VERTICAL SLICE (a head + a label + a loss + a mask + call-site wiring + gates), and five of
those verticals live inline in `instrumented_ppo.train()`. This module is the fold: each head
is ONE ROW — where its predictions are stashed (`belief_supervision` keys), where its labels
ride (training-only obs keys), which coefficient scales it, which metric prefix it reports
under — and `compute()` walks the rows in REGISTRY ORDER, so the train loop replaces N inline
blocks with one loop and a sixth belief becomes a row, not a slice.

**Byte-identity contract.** The loss functions here are the exact bodies that lived as
`InstrumentedMaskablePPO` statics (moved, not rewritten; the statics remain as aliases so
every existing test and call site still resolves), and `compute()` yields terms in the same
order the inline blocks added them — so `loss = loss + term` accumulates bit-identically.
All SIX supervised heads are rows. They do NOT all run at one position: float addition order
is the byte-identity constraint, so each row carries the SITE of the inline block it replaced
(`hidden_move` — the hidden-team Hungarian aux + the move-belief BCE, adjacent, early in the
minibatch; `latent` — the move-latent grading, after the intent block; `revealed` — the
spread/nature-EV/hp-type trio) and `compute(site=…)` folds only that site's rows, in registry
order. One registry, three call sites, the exact historical addition sequence.

Masking conventions stay PER-HEAD ON PURPOSE (the revealed-slot heads mask REVEALED opp
slots; the hidden-team head masks HIDDEN ones — together they tile B×6); the uniform
`mask_rate` metric every row emits is what makes them comparable (gen3_belief_mask_rate_v1).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import itertools

import torch as th
import torch.nn.functional as F

# Spread-belief loss (gen3_unified_spread_belief_v1): the believed/true DERIVED stat VALUES are
# ~80-450, so normalise by this before smooth_l1 to keep the term O(1) (the coef then sets its
# true weight). The MAE metric is reported in RAW stat points (not normalised).
_SPREAD_LOSS_SCALE = 100.0

# Nature/EV-belief loss (gen3_nature_ev_belief_v1): the nature is a 25-way CE; the EV is a
# smooth_l1 over EV VALUES (0..252) normalised by this so the term stays O(1). The CE + EV
# sub-terms combine with these internal weights (the CE is the load-bearing decomposition
# signal; the EV-MAE metric is reported in RAW EV points).
_EV_LOSS_SCALE = 64.0
_NATURE_CE_WEIGHT = 1.0
_EV_LOSS_WEIGHT = 1.0

# Move-latent VICReg floor (gen3_unified_move_system_v1) — moved here with the latent loss.
_LATENT_STD_TARGET = 1.0
_LATENT_VICREG_WEIGHT = 1.0


def spread_belief_loss(sb, belief_spread, belief_spread_mask):
    """Supervised loss for the SPREAD belief (gen3_unified_spread_belief_v1).

    `sb` = the extractor's stashed `last_spread_belief` [B,6,5] — the believed DERIVED stats
    {atk,def,spa,spd,spe} the DamageOperator consumes. Supervise the REVEALED opp slots
    (`belief_spread_mask` [B,6]==1) toward their TRUE derived stats (`belief_spread` [B,6,5], a
    training-only privileged label) with a smooth-L1 in scale-normalised stat units, so the head
    learns the opponent's HIDDEN EV spread instead of sitting at the usage-mean prior (the
    "over-estimates the largest EV" miscalibration). The gradient flows believed → stat_head →
    opp tokens → trunk, so it joins the aux pull on the grad-balance probe.

    Returns (loss, metrics) or None (off / labels absent / no scored slot). LEAK-SAFE: the
    believed stats are a MODEL OUTPUT (the op's own input), not a label; the true-spread LABEL
    rides a training-only obs key read ONLY here, never in the pi/vf forward. Pure."""
    if sb is None or belief_spread is None or belief_spread_mask is None:
        return None
    device = sb.device
    target = belief_spread.to(device).float()                          # [B,6,5]
    mask = belief_spread_mask.to(device).float() > 0.5                 # [B,6]
    if not bool(mask.any()):
        return None
    sb_sel = sb[mask]                                                  # [N,5]
    tgt_sel = target[mask]                                             # [N,5]
    loss = F.smooth_l1_loss(sb_sel / _SPREAD_LOSS_SCALE, tgt_sel / _SPREAD_LOSS_SCALE)
    with th.no_grad():
        mae = (sb_sel - tgt_sel).abs().mean()
        # The "over-estimate the largest EV" diagnostic: signed error on each mon's LARGEST
        # true stat (>0 ⇒ over-estimating it). Trends toward 0 as the head learns off the prior.
        amax = tgt_sel.argmax(dim=1)
        rows = th.arange(tgt_sel.shape[0], device=device)
        largest_bias = (sb_sel[rows, amax] - tgt_sel[rows, amax]).mean()
    return loss, {"mae": float(mae.item()), "largest_bias": float(largest_bias.item()),
                  "n_slots": int(mask.sum().item()),
                  # UNIFORM across every belief head (belief/<head>_mask_rate): fraction of the
                  # B×6 slot grid this head scored (gen3_belief_mask_rate_v1).
                  "mask_rate": float(mask.float().mean().item())}


def nature_ev_belief_loss(nat_logits, ev_pred, belief_nature, belief_nature_mask,
                          belief_ev, belief_ev_mask):
    """Supervised loss for the NATURE/EV decomposition of the generative spread belief
    (`gen3_nature_ev_belief_v1`). `nat_logits` [B,6,25] + `ev_pred` [B,6,5] are the extractor's
    stashed `last_spread_nature_logits` / `last_spread_ev` (BOTH None when the additive head is
    used → returns None, so the term is auto-skipped unless --spread-belief-nature is on).
    Supervise the REVEALED opp slots toward the privileged INVERTED (nature, EVs) label: a
    25-way CE on the nature + a scale-normalised smooth_l1 on the EVs — this trains the
    decomposition DIRECTLY (the derived-stat smooth_l1 alone is many-to-one).
    Returns (loss, metrics) or None. LEAK-SAFE: labels are training-only, read ONLY here."""
    if nat_logits is None or ev_pred is None or belief_nature is None or belief_ev is None:
        return None
    device = nat_logits.device
    if belief_nature_mask is None:
        return None
    nmask = belief_nature_mask.to(device).float() > 0.5                 # [B,6]
    if not bool(nmask.any()):
        return None
    nat_true = belief_nature.to(device).long()                         # [B,6]
    nl = nat_logits[nmask]                                              # [N,25]
    nt = nat_true[nmask]                                               # [N]
    nat_ce = F.cross_entropy(nl, nt)
    ev_true = belief_ev.to(device).float()                             # [B,6,5]
    evm = (belief_ev_mask.to(device).float() > 0.5) if belief_ev_mask is not None else nmask
    if bool(evm.any()):
        ev_loss = F.smooth_l1_loss(ev_pred[evm] / _EV_LOSS_SCALE, ev_true[evm] / _EV_LOSS_SCALE)
        ev_mae = (ev_pred[evm] - ev_true[evm]).abs().mean().detach()
    else:
        ev_loss = nl.new_zeros(())
        ev_mae = nl.new_zeros(())
    loss = _NATURE_CE_WEIGHT * nat_ce + _EV_LOSS_WEIGHT * ev_loss
    with th.no_grad():
        acc = (nl.argmax(dim=-1) == nt).float().mean()
    return loss, {"nature_acc": float(acc.item()), "nature_ce": float(nat_ce.item()),
                  "ev_mae": float(ev_mae.item()), "n_slots": int(nmask.sum().item()),
                  "mask_rate": float(nmask.float().mean().item())}   # uniform: see spread


def hp_type_belief_loss(logits, hp_type_label, hp_type_mask):
    """Supervised CROSS-ENTROPY for the HP-TYPE belief (gen3_opp_hp_type_belief_v1).

    `logits` = the extractor's stashed `last_hp_type_logits` [B,6,16] — the HPTypeBelief head's
    per-opp-slot posterior (Smogon prior ⊕ learned delta) the DamageOperator consumes as its
    typed-HP candidate weights. Supervise the slots whose REVEALED species runs Hidden Power
    (`hp_type_mask` [B,6]==1) toward the TRUE HP type index (a training-only privileged label —
    Gen 3 never reveals it). Returns (loss, metrics) or None. LEAK-SAFE as above."""
    if logits is None or hp_type_label is None or hp_type_mask is None:
        return None
    device = logits.device
    label = hp_type_label.to(device).long()                            # [B,6]
    mask = hp_type_mask.to(device).float() > 0.5                       # [B,6]
    if not bool(mask.any()):
        return None
    sel_logits = logits[mask]                                          # [N,16]
    sel_label = label[mask].clamp(min=0)                               # [N] (PAD -1 masked out)
    loss = F.cross_entropy(sel_logits, sel_label)
    with th.no_grad():
        acc = (sel_logits.argmax(dim=-1) == sel_label).float().mean()
    return loss, {"acc": float(acc.item()), "n_slots": int(mask.sum().item()),
                  "mask_rate": float(mask.float().mean().item())}    # uniform: see spread


def move_belief_loss(ml, known_moves, belief_moves, mode: str):
    """Supervised loss for the MoveBelief REINJECTION head (``last_move_belief_logits`` [B,6,M]).

    Two DISJOINT slot populations, selected by ``mode``:
    - REVEALED slots (mode revealed|both): seen mons. ``known_moves`` [B,6,4] holds each revealed
      slot's FULL privileged moveset; those slots are supervised DIRECTLY (slot==species, no
      matching) by a multi-label BCE — the head learns the mon's as-yet-UNREVEALED moves
      (the surprise-OHKO new-move gap).
    - UNREVEALED slots (mode unrevealed|both): hidden mons. ``belief_moves`` [B,6,4] holds the hidden
      movesets at the believed (anonymous) slots; the k believed-slot predictions are
      order-invariantly matched to the k hidden movesets (per-sample min-cost assignment — the
      slots are interchangeable, so a fixed slot↔mon target would chase a reveal-shifting
      assignment, the same defect the species aux fixed). The matching cost is the
      assignment-relevant part of BCE, ``-(pred·target)`` (the per-slot constant terms drop out of
      the argmin), so it is a cheap einsum, not a full pairwise BCE.

    The two label tensors PAD each other's slots (known_moves PADs believed slots; belief_moves
    PADs revealed slots), so 'both' simply scores each population with its own rule. A slot whose
    moveset is all-PAD (unknown moves) is NOT supervised. Returns (loss, metrics) or None
    (off / labels absent / nothing scorable). FAILS LOUD on an out-of-vocab move id. Pure + static
    (unit-tests without a full PPO)."""
    if ml is None:
        return None
    device = ml.device
    n_moves = ml.shape[-1]
    terms = []
    mv_tp = mv_pred_pos = mv_true_pos = 0
    n_revealed = n_unrevealed = 0

    def _vocab_check(ids):
        if ids.numel() and bool((ids >= n_moves).any()):
            raise ValueError(
                f"move-belief label out of vocab: max {int(ids.max())} (n_moves {n_moves}) — "
                "the embedding-num pipeline is corrupt (real Gen-3 move nums are all < 400)."
            )

    # ---- REVEALED: direct multi-label BCE (slot identity == revealed species) ----
    if mode in ("revealed", "both") and known_moves is not None:
        km = known_moves.long().to(device)                                     # [B, 6, 4]
        valid = km >= 0
        _vocab_check(km[valid])
        slot_has = valid.any(-1)                                               # [B, 6]
        if bool(slot_has.any()):
            multi_hot = th.zeros_like(ml)                                      # [B, 6, M]
            bb, ss, _ = valid.nonzero(as_tuple=True)
            multi_hot[bb, ss, km[valid]] = 1.0
            per_slot = F.binary_cross_entropy_with_logits(
                ml, multi_hot, reduction="none").mean(-1)                      # [B, 6]
            terms.append(per_slot[slot_has].reshape(-1))
            with th.no_grad():
                sel = slot_has.unsqueeze(-1)
                pp = (ml > 0.0) & sel
                mh = multi_hot.bool() & sel
                mv_tp += int((pp & mh).sum()); mv_pred_pos += int(pp.sum()); mv_true_pos += int(mh.sum())
            n_revealed += int(slot_has.sum())

    # ---- UNREVEALED: order-invariant (Hungarian) multi-label BCE over the believed slots ----
    if mode in ("unrevealed", "both") and belief_moves is not None:
        bm = belief_moves.long().to(device)                                    # [B, 6, 4]
        valid = bm >= 0
        _vocab_check(bm[valid])
        slot_has = valid.any(-1)                                               # [B, 6] believed slots w/ a moveset
        counts = slot_has.sum(1)                                               # [B] k per sample
        for k in range(1, ml.shape[1] + 1):
            sel = (counts == k).nonzero(as_tuple=True)[0]
            if sel.numel() == 0:
                continue
            n = sel.numel()
            slot_idx = slot_has[sel].nonzero(as_tuple=False)[:, 1].view(n, k)  # [n, k] believed positions
            rows = sel.view(n, 1).expand(n, k)
            preds = ml[rows, slot_idx]                                         # [n, k, M] logits
            tgt = th.zeros((n, k, n_moves), device=device)                     # [n, k, M] target multi-hots
            ids = bm[rows, slot_idx]                                           # [n, k, 4]
            vmask = ids >= 0
            if bool(vmask.any()):
                aa, kk, _ = vmask.nonzero(as_tuple=True)
                tgt[aa, kk, ids[vmask]] = 1.0
            # cost[a,i,j] = assignment-relevant part of BCE(pred_i, tgt_j) = -(pred_i · tgt_j). The
            # per-slot constant BCE terms are independent of the assignment → drop from the argmin.
            cost = -th.einsum('akm,ajm->akj', preds, tgt)                      # [n, k, k]
            perms = th.tensor(list(itertools.permutations(range(k))), dtype=th.long, device=device)
            ii = th.arange(k, device=device).view(1, k).expand(perms.shape[0], k)
            best_perm = perms[cost[:, ii, perms].sum(-1).argmin(1)]            # [n, k] min-cost assignment
            matched = th.gather(tgt, 1, best_perm[:, :, None].expand(n, k, n_moves))  # [n, k, M]
            per_slot = F.binary_cross_entropy_with_logits(
                preds, matched, reduction="none").mean(-1)                     # [n, k]
            terms.append(per_slot.reshape(-1))
            with th.no_grad():
                pp = preds > 0.0
                mh = matched.bool()
                mv_tp += int((pp & mh).sum()); mv_pred_pos += int(pp.sum()); mv_true_pos += int(mh.sum())
            n_unrevealed += n * k

    if not terms:
        return None
    loss = th.cat(terms).mean()
    metrics = {
        "bce": float(loss.item()),
        "precision": (mv_tp / mv_pred_pos) if mv_pred_pos else 0.0,
        "recall": (mv_tp / mv_true_pos) if mv_true_pos else 0.0,
        "revealed_slots": float(n_revealed),
        "unrevealed_slots": float(n_unrevealed),
    }
    return loss, metrics


def move_belief_latent_loss(ml, latent_table, known_moves):
    """LATENT-space grading of the move belief (gen3_unified_move_system_v1) — the soft complement to
    the per-ID BCE so near-moves (Rock Slide ≈ Hidden Power Rock) grade as near.

    For each REVEALED scored slot (slot==species, like the BCE's revealed branch): the predicted
    distribution's EXPECTED move-latent ``softmax(ml) @ latent_table`` (softmax over the move axis →
    floor-robust, concentrates on the believed moves) is regressed by COSINE toward the slot's true
    moveset MEAN latent (stop-grad — the grading TARGET), plus a VICReg variance floor on the
    predictions. ``latent_table`` ``[M, D]`` is the context-free `MoveLatentEncoder.latent_table`;
    ``ml`` ``[B,6,M]``; ``known_moves``
    ``[B,6,4]`` (move nums, -1 PAD). Returns (loss, metrics) or None (off / labels absent / nothing
    scorable). Pure + static (unit-testable without a full PPO).

    COLLAPSE NOTE: the target here is derived from the SAME
    `latent_table` whose params the prediction gradient updates (the `.detach()` only severs the
    per-step path, not the shared table). So the cosine could in principle be gamed by collapsing the
    table rows to one DIRECTION. The VICReg term below is NOT the guard against that — it floors per-dim
    MAGNITUDE variance, which an angular collapse can satisfy by scaling magnitudes. The REAL
    anti-collapse pressure is the RL TASK-ANCHORING: the same per-move latent feeds the move network
    (`PokemonEncoder.forward`), so the table can't freely collapse without hurting the policy/value
    objective. (Empirically full direction-collapse is not a reachable attractor of this loss; the
    `movelatent_std` metric monitors magnitude, not angle — a future change that weakens the move-net
    anchoring would remove the only real guard, so add a row-decorrelation term then.)"""
    if ml is None or latent_table is None or known_moves is None:
        return None
    device = ml.device
    km = known_moves.long().to(device)                                # [B,6,4]
    valid = km >= 0
    slot_has = valid.any(-1)                                          # [B,6]
    if not bool(slot_has.any()):
        return None
    # predicted distribution's expected latent (softmax kills the ~0.02 prior floor → the latent is
    # the believed moves', not the global mean), [B,6,D].
    pred_latent = F.softmax(ml, dim=-1) @ latent_table               # [B,6,D]
    # target = mean of the slot's TRUE moves' latents (stop-grad). gather + zero pads + mean.
    move_lat = latent_table[km.clamp(min=0)] * valid.unsqueeze(-1).float()   # [B,6,4,D] (pads → 0)
    tgt = (move_lat.sum(2) / valid.sum(-1, keepdim=True).clamp(min=1).float()).detach()  # [B,6,D]
    pl = pred_latent[slot_has]                                        # [N,D]
    tl = tgt[slot_has]                                                # [N,D]
    cos = F.cosine_similarity(pl, tl, dim=-1)                         # [N]
    cos_loss = (1.0 - cos).mean()
    std = th.sqrt(pl.var(dim=0, unbiased=False) + 1e-4)              # [D] per-dim MAGNITUDE floor (see COLLAPSE NOTE)
    vicreg = F.relu(_LATENT_STD_TARGET - std).mean()
    loss = cos_loss + _LATENT_VICREG_WEIGHT * vicreg
    metrics = {
        "cosine": float(cos.mean().item()),
        "std": float(std.mean().item()),
        "slots": float(int(slot_has.sum())),
    }
    return loss, metrics


def belief_aux_loss(bl, sp_labels, mv_labels, moves_weight: float = 1.0):
    """Order-invariant (Hungarian / DETR-style) hidden-opponent belief aux loss.

    bl = {"species": [B,6,S], "moves": [B,6,M]} (the stashed BeliefHead logits); sp_labels [B,6]
    and mv_labels [B,6,4] are the privileged int labels (-1 = revealed/pad).
    The k believed-slot predictions of each sample are matched to its k hidden-mon targets by
    **per-sample min-cost assignment** (so the anonymous slot tokens collectively cover the hidden
    SET instead of each chasing a reveal-shifting fixed slot↔mon target), then species cross-entropy
    + moves multi-label BCE are taken over the matched pairs. The matching is exact: for k ≤ TEAM_SIZE
    the k! permutations are enumerated and the min-CE-cost one chosen (vectorised per distinct k — no
    per-sample Python loop, no scipy).

    Perf: the species log-softmax is taken on the GATHERED believed slots ([n,k,S]) not the full
    [B,6,S] (the non-believed slots are never read); the moves branch is skipped entirely when
    moves_weight==0; accuracy + moves P/R are diagnostics computed under no_grad.

    Returns (aux_tensor, metrics_dict) or None when nothing to score (belief
    off / labels absent / a minibatch with zero believed slots — the None guard keeps an empty
    minibatch from NaN-poisoning the loss). FAILS LOUD on an out-of-vocab label id (impossible on
    real data → a corrupt embedding-num pipeline), rather than silently dropping it. Pure + static so
    it unit-tests without a full PPO."""
    if bl is None or sp_labels is None or mv_labels is None:
        return None
    sp_logits = bl["species"]
    mv_logits = bl["moves"]
    device = sp_logits.device
    sp_labels = sp_labels.long().to(device)
    mv_labels = mv_labels.long().to(device)
    n_species = sp_logits.shape[-1]
    n_moves = mv_logits.shape[-1]
    believed = sp_labels >= 0                                                  # [B, 6] (-1 = not scored)
    counts = believed.sum(1)                                                   # [B] k per sample
    if int(counts.sum()) == 0:
        return None
    # FAIL LOUD: a believed label id must fit the vocab. Every real Gen-3 species/move num is well
    # inside max=400, so a violation means the label↔embedding num space is corrupt — crash, don't
    # silently filter and train on a hole. (-1 pads were already excluded by `believed`.) A SINGLE
    # host-sync on the happy path (the per-element max()/message only run on the failure path).
    sp_believed = sp_labels[believed]
    mv_believed = mv_labels[believed]
    sp_bad = (sp_believed >= n_species).any()
    mv_bad = (mv_believed >= n_moves).any() if mv_believed.numel() else sp_bad.new_zeros(())
    if bool(sp_bad | mv_bad):
        raise ValueError(
            f"belief label out of vocab: species max {int(sp_believed.max())} (n_species {n_species}) / "
            f"move max {int(mv_believed.max()) if mv_believed.numel() else -1} (n_moves {n_moves}) — "
            "the embedding-num pipeline is corrupt (real Gen-3 nums are all < 400)."
        )
    do_moves = moves_weight != 0.0
    ce_terms, bce_terms = [], []
    n_correct = th.zeros((), device=device)
    n_slots = 0
    mv_tp = mv_pred_pos = mv_true_pos = 0  # moves precision/recall accumulators (diagnostic)
    # Group samples by their believed-slot count k; within a group every cost matrix is k×k so the
    # whole group is matched with one vectorised permutation-enumeration.
    for k in range(1, sp_labels.shape[1] + 1):
        sel = (counts == k).nonzero(as_tuple=True)[0]                          # samples with k believed
        if sel.numel() == 0:
            continue
        n = sel.numel()
        slot_idx = believed[sel].nonzero(as_tuple=False)[:, 1].view(n, k)      # [n, k] believed positions
        rows = sel.view(n, 1).expand(n, k)                                     # [n, k] sample indices
        pred_logp = th.log_softmax(sp_logits[rows, slot_idx], dim=-1)          # [n, k, S] softmax on gathered
        tgt_sp = sp_labels[rows, slot_idx]                                     # [n, k] target species
        # cost[a,i,j] = CE of predicting believed-slot i's logits at target j = -logp[a,i,tgt[a,j]]
        cost = -th.gather(pred_logp, 2, tgt_sp[:, None, :].expand(n, k, k))    # [n, k, k]
        perms = th.tensor(list(itertools.permutations(range(k))), dtype=th.long, device=device)  # [P,k]
        ii = th.arange(k, device=device).view(1, k).expand(perms.shape[0], k)
        best_perm = perms[cost[:, ii, perms].sum(-1).argmin(1)]               # [n, k] min-cost assignment
        matched_sp = th.gather(tgt_sp, 1, best_perm)                           # [n, k] matched species target
        ce_terms.append((-th.gather(pred_logp, 2, matched_sp[:, :, None]).squeeze(-1)).reshape(-1))
        with th.no_grad():
            n_correct = n_correct + (pred_logp.argmax(-1) == matched_sp).sum()
        n_slots += n * k
        if do_moves:
            matched_label_slot = th.gather(slot_idx, 1, best_perm)             # [n, k] matched label slot
        if do_moves:
            mv_pred = mv_logits[rows, slot_idx]                                # [n, k, M] predictions
            mv_ids = mv_labels[rows, matched_label_slot]                      # [n, k, 4] matched move ids
            mvalid = mv_ids >= 0                                               # [n, k, 4] (pad excluded)
            multi_hot = th.zeros_like(mv_pred)                                 # [n, k, M]
            if bool(mvalid.any()):
                aa, kk, _ = mvalid.nonzero(as_tuple=True)
                multi_hot[aa, kk, mv_ids[mvalid]] = 1.0
            # per-slot BCE = mean over the M move classes (same per-slot scale as the per-slot CE),
            # but ONLY for slots with ≥1 labeled move — a slot whose moves are all-pad (unknown
            # moveset) must NOT be supervised toward "predict no moves" (all-negative).
            slot_has_moves = mvalid.any(-1)                                    # [n, k]
            per_slot_bce = F.binary_cross_entropy_with_logits(
                mv_pred, multi_hot, reduction="none").mean(-1)                 # [n, k]
            bce_terms.append(per_slot_bce[slot_has_moves].reshape(-1))
            with th.no_grad():
                pred_present = mv_pred > 0.0                                   # sigmoid>0.5 ⇒ predicted present
                mv_tp += int((pred_present & multi_hot.bool()).sum())
                mv_pred_pos += int(pred_present.sum())
                mv_true_pos += int(multi_hot.sum())
    ce = th.cat(ce_terms).mean()
    bce_cat = th.cat(bce_terms) if bce_terms else th.zeros(0, device=device)
    # numel guard: every believed slot could have an unknown moveset (all-pad) → no BCE terms → 0
    # (not NaN). In practice hidden mons have mapped moves so this is the degenerate edge.
    bce = bce_cat.mean() if bce_cat.numel() else th.zeros((), device=device)
    aux = ce + moves_weight * bce
    n_samples = int((counts > 0).sum())
    acc = float((n_correct.float() / max(1, n_slots)).item())
    metrics = {
        "species_ce": float(ce.item()),
        "moves_bce": float(bce.item()),
        "species_acc": acc,
        "species_acc_above_chance": acc - 1.0 / n_species,
        "moves_precision": (mv_tp / mv_pred_pos) if mv_pred_pos else 0.0,
        "moves_recall": (mv_tp / mv_true_pos) if mv_true_pos else 0.0,
        "k_mean": n_slots / max(1, n_samples),
        "coverage": n_samples / sp_labels.shape[0],
        # uniform per-head coverage (see _spread_belief_loss): here believed = HIDDEN slots,
        # so this is the flip side of the revealed-slot heads' rates — together they tile B×6.
        "mask_rate": n_slots / believed.numel(),
    }
    return aux, metrics


def item_belief_loss(logits, item_label, item_mask):
    """Supervised CROSS-ENTROPY for the ITEM belief (gen3_item_belief_v1) — the bank's seventh
    row, the exact hp_type shape over the item-num axis: `logits` = the extractor's stashed
    per-opp-slot item posterior logits [B,6,n_items] (Smogon prior ⊕ zero-init delta; the op
    consumes P(Choice Band) off the same posterior); supervise the REVEALED slots toward the
    TRUE item num (privileged, training-only — Gen 3 never reveals a Choice Band directly).
    Returns (loss, metrics) or None. LEAK-SAFE as the others."""
    if logits is None or item_label is None or item_mask is None:
        return None
    device = logits.device
    label = item_label.to(device).long()                               # [B,6]
    mask = item_mask.to(device).float() > 0.5                          # [B,6]
    if not bool(mask.any()):
        return None
    sel_logits = logits[mask]                                          # [N,n_items]
    sel_label = label[mask].clamp(min=0)                               # [N] (PAD -1 masked out)
    loss = F.cross_entropy(sel_logits, sel_label)
    with th.no_grad():
        acc = (sel_logits.argmax(dim=-1) == sel_label).float().mean()
    return loss, {"acc": float(acc.item()), "n_slots": int(mask.sum().item()),
                  "mask_rate": float(mask.float().mean().item())}    # uniform: see spread


@dataclass(frozen=True)
class BeliefHeadRow:
    """One supervised belief head: everything the train loop needs to fold it.

    ``args`` is the loss call's argument list in ORDER, each entry one of
    ``("stash", key)`` — an extractor ``belief_supervision`` key; ``("obs", key)`` — a
    training-only rollout obs key; ``("attr", name)`` — a plain extractor attribute (the
    hidden-team logits dict, the latent table, the move-belief mode); ``("param", name)`` — a
    value from `compute()`'s ``params`` dict (``moves_weight``). ``gate`` names the enable
    flag; ``coef`` the coefficient key; ``probe`` the `aux_probe_terms` key the grad-balance
    probe reports under; ``site`` the train() position whose inline block this row replaced;
    ``loss_key`` the metrics key carrying the unscaled loss (the hidden-team block's historic
    name is ``aux_loss``)."""
    name: str
    probe: str
    prefix: str
    gate: str
    coef: str
    loss_fn: Callable
    args: Tuple[Tuple[str, str], ...]
    site: str = "revealed"
    loss_key: str = "loss"


#: REGISTRY ORDER IS LOAD-BEARING: `compute()` yields terms in this order and the train loop
#: adds them in yield order, preserving the float-addition sequence of the inline blocks this
#: bank replaced (spread → nature/EV → hp-type, consecutive in train() since v40/v51).
ROWS: Tuple[BeliefHeadRow, ...] = (
    BeliefHeadRow(
        name="hidden_team", probe="species_belief", prefix="", gate="hidden_team",
        coef="opp_belief_aux_coef", loss_fn=belief_aux_loss, site="hidden_move",
        loss_key="aux_loss",
        args=(("attr", "last_belief_logits"), ("obs", "belief_species"),
              ("obs", "belief_moves"), ("param", "moves_weight"))),
    BeliefHeadRow(
        name="move_belief", probe="move_belief", prefix="move_", gate="move_belief",
        coef="move_belief_coef", loss_fn=move_belief_loss, site="hidden_move",
        args=(("stash", "move_belief_logits"), ("obs", "known_moves"),
              ("obs", "belief_moves"), ("attr", "move_belief_mode"))),
    BeliefHeadRow(
        name="move_latent", probe="move_latent", prefix="movelatent_", gate="move_latent",
        coef="move_belief_latent_coef", loss_fn=move_belief_latent_loss, site="latent",
        args=(("stash", "move_belief_logits"), ("attr", "last_move_latent_table"),
              ("obs", "known_moves"))),
    BeliefHeadRow(
        name="spread", probe="spread_belief", prefix="spread_", gate="spread",
        coef="spread_belief_coef", loss_fn=spread_belief_loss,
        args=(("stash", "spread_belief"), ("obs", "belief_spread"),
              ("obs", "belief_spread_mask"))),
    BeliefHeadRow(
        name="nature_ev", probe="nature_ev", prefix="natureev_", gate="spread",
        coef="spread_belief_coef", loss_fn=nature_ev_belief_loss,
        args=(("stash", "spread_nature_logits"), ("stash", "spread_ev"),
              ("obs", "belief_nature"), ("obs", "belief_nature_mask"),
              ("obs", "belief_ev"), ("obs", "belief_ev_mask"))),
    BeliefHeadRow(
        name="hp_type", probe="hp_type", prefix="hptype_", gate="hp_type",
        coef="hp_type_belief_coef", loss_fn=hp_type_belief_loss,
        args=(("stash", "hp_type_logits"), ("obs", "hp_type_label"),
              ("obs", "hp_type_mask"))),
    # gen3_item_belief_v1 — the SEVENTH head, and the bank's thesis made concrete: this row +
    # its loss fn above are the ENTIRE train()-side cost of the new belief (appended last at
    # the revealed site, so the pre-existing float-addition order is untouched).
    BeliefHeadRow(
        name="item", probe="item_belief", prefix="item_", gate="item",
        coef="item_belief_coef", loss_fn=item_belief_loss,
        args=(("stash", "item_logits"), ("obs", "item_label"),
              ("obs", "item_mask"))),
)


def _resolve_arg(src: str, key: str, extractor, observations, params: Dict) -> object:
    if src == "stash":
        return extractor.belief_supervision(key)
    if src == "obs":
        return observations.get(key)
    if src == "attr":
        return getattr(extractor, key, None)
    if src == "param":
        return params[key]
    raise ValueError(f"unknown arg source {src!r}")


def compute(extractor, observations, coefs: Dict[str, float], gates: Dict[str, bool],
            site: str, params: "Optional[Dict]" = None,
            rows: Iterable[BeliefHeadRow] = ROWS,
            ) -> List[Tuple[BeliefHeadRow, "th.Tensor", Dict[str, float]]]:
    """Fold every gated row AT ``site``: resolve its args, run its loss, scale by its coef.
    Returns ``[(row, term, metrics)]`` in registry order — the caller adds ``term`` to the
    loss in that order (the byte-identity contract) and prefixes ``metrics`` with
    ``row.prefix``. Each metrics dict carries the unscaled loss under ``row.loss_key`` (the
    inline blocks' historic names). A row whose loss returns None (labels absent /
    zero-scored minibatch) contributes nothing, exactly as the inline guards did."""
    params = params or {}
    out = []
    for row in rows:
        if row.site != site or not gates.get(row.gate, False):
            continue
        argv = [_resolve_arg(src, k, extractor, observations, params) for src, k in row.args]
        res = row.loss_fn(*argv)
        if res is None:
            continue
        raw, metrics = res
        metrics[row.loss_key] = float(raw.item())
        out.append((row, coefs[row.coef] * raw, metrics))
    return out
