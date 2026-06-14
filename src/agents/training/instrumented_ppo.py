"""InstrumentedMaskablePPO — MaskablePPO with `train/clip_fraction_vf` added.

sb3_contrib's MaskablePPO logs `train/clip_fraction` (policy clip fraction) but
not the equivalent metric for value-function clipping, even though VF clipping
is applied when `clip_range_vf` is non-None. This subclass copies the upstream
`train()` method verbatim and adds three lines: a per-batch fraction
computation, accumulation, and one final `self.logger.record(...)` call.

# Drift detection

`train()` is a vendored copy of upstream code. If sb3_contrib ever updates the
method, our copy will silently become stale and we'd run different training
logic than upstream. To prevent that, `_verify_upstream_unchanged()` runs at
import time and computes the SHA256 of `inspect.getsource(MaskablePPO.train)`.
A mismatch raises immediately with both hashes and instructions.

If upstream changes (e.g. after a `pip install -U sb3_contrib`):
1. Diff the new upstream `train()` against the one this subclass was vendored
   from (last known hash is in `_EXPECTED_UPSTREAM_TRAIN_HASH`).
2. Port any non-instrumentation changes into the `train()` override below.
3. Recompute the hash and update `_EXPECTED_UPSTREAM_TRAIN_HASH`.
"""

import hashlib
import inspect
import itertools

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.utils import explained_variance
from torch.nn import functional as F

from sb3_contrib import MaskablePPO

from agents.training.async_vec_env import AsyncSubprocVecEnv, collect_rollouts_async
from agents.training.grad_balance import (
    grad_balance_metrics,
    shared_trunk_parameters,
    value_scale_metrics,
)


# SHA256 of inspect.getsource(MaskablePPO.train) at the time this file was
# written. If sb3_contrib updates and this no longer matches, _verify_...
# raises at import time.
_EXPECTED_UPSTREAM_TRAIN_HASH = (
    "79500464b6a71d5adcfdf10028df56fbaf72b7754952e760f9e377610b9cf809"
)

# Fraction of each minibatch that forms the "tail" for the tail-weighted value loss — the worst
# _VALUE_TAIL_FRAC by squared value error (the V-tail craters the critic under-prices). 0.1 = worst
# 10%; loosely tracks the eval/td_resid_tail CVaR@5% diagnostic the loss is meant to pull down.
_VALUE_TAIL_FRAC = 0.1

# Latent-belief VICReg variance floor: a hinge `relu(_LATENT_STD_TARGET - std)` per latent dim pushes
# the predicted latents to stay spread (≈unit std), the belt-and-braces collapse guard on top of the
# stop-grad + task-anchored target. Weighted by _LATENT_VICREG_WEIGHT inside the latent loss. The
# `belief_latent_std` metric (mean per-dim std) is the NO-GO monitor: std→0 while cosine→1 is collapse.
_LATENT_STD_TARGET = 1.0
_LATENT_VICREG_WEIGHT = 1.0


def _verify_upstream_unchanged() -> None:
    """Fail-loud at import time if the upstream `MaskablePPO.train()` source
    has drifted from what this subclass was vendored against."""
    source = inspect.getsource(MaskablePPO.train)
    actual = hashlib.sha256(source.encode("utf-8")).hexdigest()
    if actual != _EXPECTED_UPSTREAM_TRAIN_HASH:
        upstream_file = inspect.getfile(MaskablePPO)
        raise RuntimeError(
            "[InstrumentedMaskablePPO] DRIFT DETECTED: upstream "
            "`sb3_contrib.MaskablePPO.train()` source has changed since this "
            "subclass was vendored.\n"
            f"  Expected SHA256: {_EXPECTED_UPSTREAM_TRAIN_HASH}\n"
            f"  Actual SHA256:   {actual}\n"
            f"  Upstream file:   {upstream_file}\n\n"
            "ACTION REQUIRED: diff the upstream train() against this subclass, "
            "port any non-instrumentation changes into the override in "
            f"{__file__}, then update _EXPECTED_UPSTREAM_TRAIN_HASH to silence "
            "this check."
        )


_verify_upstream_unchanged()


class InstrumentedMaskablePPO(MaskablePPO):
    """MaskablePPO with `train/clip_fraction_vf` instrumentation added.

    Behaviour-identical to `MaskablePPO` except for the additional TensorBoard
    metric. See module docstring for drift-detection details.

    Also dispatches rollout collection to the **non-barrier async collector** when
    ``self._async_rollout`` is set and the env is an ``AsyncSubprocVecEnv`` (``--async-rollout``);
    otherwise it is the unchanged stock ``MaskablePPO.collect_rollouts``.
    """

    # Set by train_rl_agent after construction; opt-in (default off → stock sync collection).
    _async_rollout: bool = False

    # Set by train_rl_agent after construction (like _async_rollout); resume-immutable (recorded +
    # version-checked). 0.0 = plain MSE value loss (byte-identical to upstream). >0 blends in the CVaR
    # of the worst value misses — see _value_loss_from_se.
    value_tail_weight: float = 0.0

    # Set by train_rl_agent after construction (like value_tail_weight). The hidden-opponent belief
    # aux-loss coefficient: opp_belief_aux_coef * (species_CE + moves_weight·moves_BCE) over the
    # believed opp slots is added to each minibatch loss. 0.0 = OFF (no aux term, byte-identical loss).
    # A TRAINING hparam (affects the loss only, never a forward pass) → NOT version-locked / NOT in
    # check_compatible (treat like ent_coef; a frozen eval/pool/distill opponent never runs train()).
    # Class default 0.0 so a smoke/test/frozen-opponent path that never sets it reads a safe no-op.
    opp_belief_aux_coef: float = 0.0
    # Relative weight of the moves multi-label BCE vs the species CE inside the aux term. Both are now
    # on a per-believed-slot scale (CE per slot ≈ log S; BCE = mean over the M move classes per slot),
    # so the species term dominates by default (moves is the weaker secondary signal); raise this to
    # up-weight move prediction. Training-only, like the coef.
    opp_belief_moves_weight: float = 1.0

    # Set by train_rl_agent (like opp_belief_aux_coef). The MOVE-belief reinjection-head loss weight:
    # move_belief_coef * (BCE over the scored opp slots) is added to each minibatch. 0.0 = OFF
    # (byte-identical). Training-only (scales the loss, never a forward pass) → NOT version-locked. The
    # MODE (which slots) lives on the extractor (move_belief_mode) — read from there, single source.
    move_belief_coef: float = 0.0

    # Set by train_rl_agent (like opp_belief_aux_coef). The LATENT-belief loss weight: opp_belief_latent_coef
    # * (cosine-to-encoder-role-token + VICReg variance floor) over the believed opp slots, matched on the
    # SAME Hungarian assignment as the species CE. 0.0 = OFF (no term; byte-identical loss). Training-only
    # (scales the loss, never a forward pass) → NOT version-locked. The latent PREDICTOR (a state_dict
    # change) is gated by the version-checked opp_belief_latent arch toggle, not this coef.
    opp_belief_latent_coef: float = 0.0

    @staticmethod
    def _move_belief_loss(ml, known_moves, belief_moves, mode: str):
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

    @staticmethod
    def _belief_aux_loss(bl, sp_labels, mv_labels, moves_weight: float = 1.0, latent_target=None):
        """Order-invariant (Hungarian / DETR-style) hidden-opponent belief aux loss.

        bl = {"species": [B,6,S], "moves": [B,6,M], ["latent": [B,6,D]]} (the stashed BeliefHead
        logits); sp_labels [B,6] and mv_labels [B,6,4] are the privileged int labels (-1 = revealed/pad).
        The k believed-slot predictions of each sample are matched to its k hidden-mon targets by
        **per-sample min-cost assignment** (so the anonymous slot tokens collectively cover the hidden
        SET instead of each chasing a reveal-shifting fixed slot↔mon target), then species cross-entropy
        + moves multi-label BCE are taken over the matched pairs. The matching is exact: for k ≤ TEAM_SIZE
        the k! permutations are enumerated and the min-CE-cost one chosen (vectorised per distinct k — no
        per-sample Python loop, no scipy).

        **Latent term (`latent_target` [B,6,D] given AND bl carries "latent").** On the SAME species-CE
        Hungarian assignment (one mon per slot across all heads — no conflicting pulls), a cosine loss
        regresses each believed slot's predicted latent toward the STOP-GRAD encoder role-token of its
        matched true hidden mon, plus a VICReg variance floor on the predictions (collapse guard). It is
        returned SEPARATELY (third tuple element) so the caller weights it by its own coef.

        Perf: the species log-softmax is taken on the GATHERED believed slots ([n,k,S]) not the full
        [B,6,S] (the non-believed slots are never read); the moves branch is skipped entirely when
        moves_weight==0; accuracy + moves P/R are diagnostics computed under no_grad.

        Returns (aux_tensor, metrics_dict, latent_loss_or_None) or None when nothing to score (belief
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
        latent_pred = bl.get("latent")
        do_latent = latent_pred is not None and latent_target is not None
        if do_latent:
            latent_target = latent_target.to(device)
        ce_terms, bce_terms = [], []
        cos_terms, lat_pred_terms = [], []  # latent: per-slot cosine distance + matched preds (VICReg)
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
            if do_moves or do_latent:
                matched_label_slot = th.gather(slot_idx, 1, best_perm)             # [n, k] matched label slot
            if do_latent:
                # Same assignment as species: pred at believed slot i ↔ target role-token at the
                # matched believed slot. Cosine distance (1 − cos) + collect preds for the VICReg floor.
                pred_l = latent_pred[rows, slot_idx]                               # [n, k, D] predictions
                tgt_l = latent_target[rows, matched_label_slot]                    # [n, k, D] matched targets
                cos = (F.normalize(pred_l, dim=-1) * F.normalize(tgt_l, dim=-1)).sum(-1)  # [n, k]
                cos_terms.append((1.0 - cos).reshape(-1))
                lat_pred_terms.append(pred_l.reshape(-1, pred_l.shape[-1]))        # [n*k, D]
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
        }
        # Latent-belief: mean cosine distance to the matched stop-grad role-token + a VICReg variance
        # floor on the predictions. Returned separately so the caller scales it by opp_belief_latent_coef.
        latent_loss = None
        if do_latent and cos_terms:
            cos_dist = th.cat(cos_terms).mean()
            lat_pred = th.cat(lat_pred_terms, dim=0)                               # [N, D] matched preds
            lat_std = th.sqrt(lat_pred.var(dim=0, unbiased=False) + 1e-4)          # [D] per-dim std
            vicreg = th.relu(_LATENT_STD_TARGET - lat_std).mean()
            latent_loss = cos_dist + _LATENT_VICREG_WEIGHT * vicreg
            metrics["latent_cosine"] = float(1.0 - cos_dist.item())               # similarity (higher better)
            metrics["latent_loss"] = float(latent_loss.item())
            metrics["latent_std"] = float(lat_std.mean().item())                  # collapse monitor (→0 = NO-GO)
            metrics["latent_vicreg"] = float(vicreg.item())
        return aux, metrics, latent_loss

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

    def collect_rollouts(self, env, callback, rollout_buffer, n_rollout_steps, use_masking=True):
        if self._async_rollout and isinstance(env, AsyncSubprocVecEnv):
            return collect_rollouts_async(
                self, env, callback, rollout_buffer, n_rollout_steps, use_masking)
        return super().collect_rollouts(env, callback, rollout_buffer, n_rollout_steps, use_masking)

    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.

        Vendored from `sb3_contrib.MaskablePPO.train` (hash pinned in
        `_EXPECTED_UPSTREAM_TRAIN_HASH`). The only deltas vs upstream are
        marked with `# +INSTRUMENTATION` comments.
        """
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)  # type: ignore[operator]
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)  # type: ignore[operator]

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []
        vf_clip_fractions: list[float] = []  # +INSTRUMENTATION
        belief_metrics: dict[str, list[float]] = {}  # +BELIEF: per-minibatch aux diagnostics (dict of lists)
        # Compute once: the aux path is fully skipped when off → loss stays byte-identical to upstream.
        belief_aux_on = self.opp_belief_aux_coef > 0.0
        move_belief_on = self.move_belief_coef > 0.0  # +MOVE-BELIEF reinjection-head supervised loss
        latent_belief_on = self.opp_belief_latent_coef > 0.0  # +LATENT belief (rides the species aux call)

        continue_training = True

        # +INSTRUMENTATION: gradient-balance + value-scale diagnostics (grad_balance.py).
        # The dual-head extractor shares one trunk; both losses' gradients compete there. We
        # sample that pull ONCE per train() call (first minibatch) so vf_coef / return
        # normalization (PopArt) can be tuned to a number rather than inferred from KL.
        shared_trunk = shared_trunk_parameters(self.policy.features_extractor)
        grad_balance: dict[str, float] = {}
        grad_norms: list[float] = []  # pre-clip total grad norm (shows grad-clip activity)

        # +PopArt: advance the value-target normalizer once per train() (before the epochs) from
        # this rollout's returns; update() also POP-rescales value_net so its de-normalized outputs
        # are preserved. The value loss below then trains in normalized space. No-op when disabled.
        popart = getattr(self.policy, "popart", None)
        if popart is not None:
            popart.update(
                th.as_tensor(self.rollout_buffer.returns, device=self.device), self.policy.value_net
            )

        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = rollout_data.actions.long().flatten()

                values, log_prob, entropy = self.policy.evaluate_actions(
                    rollout_data.observations,
                    actions,
                    action_masks=rollout_data.action_masks,
                )

                values = values.flatten()
                # Normalize advantage
                advantages = rollout_data.advantages
                if self.normalize_advantage:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # ratio between old and new policy, should be one at the first iteration
                ratio = th.exp(log_prob - rollout_data.old_log_prob)

                # clipped surrogate loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                # Logging
                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                if popart is not None:
                    # +PopArt: value loss in NORMALIZED space (both target and prediction scaled by
                    # the running mu/sigma) → O(1) gradient, no longer swamping the shared trunk.
                    # Mutually exclusive with vf-clipping (enforced at startup).
                    # +TAIL: per-sample SE in normalized space → _value_loss_from_se (w=0 ⇒ MSE).
                    value_loss = self._value_loss_from_se(
                        (popart.normalize(rollout_data.returns) - popart.normalize(values)) ** 2
                    )
                elif self.clip_range_vf is None:
                    # No clipping
                    value_loss = self._value_loss_from_se((rollout_data.returns - values) ** 2)
                else:
                    # Clip the different between old and new value
                    # NOTE: this depends on the reward scaling
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                    # +INSTRUMENTATION: fraction of value updates that hit the clip bound
                    vf_clip_fraction = th.mean(
                        (th.abs(values - rollout_data.old_values) > clip_range_vf).float()
                    ).item()
                    vf_clip_fractions.append(vf_clip_fraction)
                    # +TAIL: per-sample SE on the clipped prediction → _value_loss_from_se.
                    value_loss = self._value_loss_from_se((rollout_data.returns - values_pred) ** 2)
                value_losses.append(value_loss.item())

                # Entropy loss favor exploration
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)

                entropy_losses.append(entropy_loss.item())

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                # +BELIEF: hidden-opponent belief aux loss. evaluate_actions(rollout_data.observations,
                # …) ran the extractor forward just above, stashing per-slot logits for THIS minibatch;
                # the privileged labels ride the same obs dict (training-only keys). Masked to the
                # believed slots, folded in at opp_belief_aux_coef. OFF → skipped (loss byte-identical).
                belief_aux_term = None  # the WEIGHTED aux contribution, for the grad-balance probe
                latent_belief_term = None  # the WEIGHTED latent contribution (rides the same probe)
                if belief_aux_on:
                    aux_out = self._belief_aux_loss(
                        self.policy.features_extractor.last_belief_logits,
                        rollout_data.observations.get("belief_species"),
                        rollout_data.observations.get("belief_moves"),
                        moves_weight=self.opp_belief_moves_weight,
                        # The latent target (stop-grad encoder role-tokens) the extractor stashed this
                        # minibatch; passed only when the latent term is on (else no latent loss computed).
                        latent_target=(self.policy.features_extractor.last_belief_target_latent
                                       if latent_belief_on else None),
                    )
                    if aux_out is not None:
                        aux, belief_m, latent_loss = aux_out
                        belief_aux_term = self.opp_belief_aux_coef * aux
                        loss = loss + belief_aux_term
                        belief_m["aux_loss"] = float(aux.item())
                        if latent_loss is not None and latent_belief_on:
                            latent_belief_term = self.opp_belief_latent_coef * latent_loss
                            loss = loss + latent_belief_term
                        for _bk, _bv in belief_m.items():
                            belief_metrics.setdefault(_bk, []).append(float(_bv))

                # +MOVE-BELIEF: predict+reinject the opp moveset. The extractor forward (run by
                # evaluate_actions above) stashed last_move_belief_logits; known_moves/belief_moves ride
                # the same training-only obs dict. Folded at move_belief_coef; mode read off the extractor
                # (single source). OFF → skipped (byte-identical). Its gradient ALSO flows into the trunk
                # via the reinjection, so it joins the aux pull on the grad-balance probe.
                move_belief_term = None
                if move_belief_on:
                    mb_out = self._move_belief_loss(
                        self.policy.features_extractor.last_move_belief_logits,
                        rollout_data.observations.get("known_moves"),
                        rollout_data.observations.get("belief_moves"),
                        self.policy.features_extractor.move_belief_mode,
                    )
                    if mb_out is not None:
                        mb_loss, mb_m = mb_out
                        move_belief_term = self.move_belief_coef * mb_loss
                        loss = loss + move_belief_term
                        mb_m["loss"] = float(mb_loss.item())
                        for _mk, _mv in mb_m.items():
                            belief_metrics.setdefault("move_" + _mk, []).append(float(_mv))

                # Combined auxiliary pull on the shared trunk (species belief + move belief + latent
                # belief), for the grad-balance probe — all compete with policy/value there.
                aux_probe_term = belief_aux_term
                if move_belief_term is not None:
                    aux_probe_term = (
                        move_belief_term if aux_probe_term is None else aux_probe_term + move_belief_term
                    )
                if latent_belief_term is not None:
                    aux_probe_term = (
                        latent_belief_term if aux_probe_term is None else aux_probe_term + latent_belief_term
                    )
                aux_on = belief_aux_on or move_belief_on or latent_belief_on

                # +INSTRUMENTATION: sample the shared-trunk gradient balance on the first
                # minibatch (graph alive here; the probe uses read-only autograd.grad with
                # retain_graph, so loss.backward() below is unaffected). Skipped when the
                # extractor exposes no shared-trunk params (non-Gen3 policy).
                # Sample once per train(). When an aux is ON, wait for a minibatch that actually HAS
                # scored slots (aux_probe_term set) so grad/belief_share isn't silently dropped for
                # the call; when both are OFF, sample on the first minibatch as before.
                if shared_trunk and not grad_balance and (not aux_on or aux_probe_term is not None):
                    grad_balance = grad_balance_metrics(
                        policy_loss + self.ent_coef * entropy_loss,
                        self.vf_coef * value_loss,
                        shared_trunk,
                        aux_term=aux_probe_term,  # species+move belief grad-share on the trunk (None = off)
                    )

                # Calculate approximate form of reverse KL Divergence for early stopping
                # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
                # and Schulman blog: http://joschu.net/blog/kl-approx.html
                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

                # Optimization step
                self.policy.optimizer.zero_grad()
                loss.backward()
                # Clip grad norm
                grad_norms.append(float(  # +INSTRUMENTATION: pre-clip total grad norm
                    th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                ))
                self.policy.optimizer.step()

            self._n_updates += 1
            if not continue_training:
                break
        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        # Logs
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)
            # +INSTRUMENTATION: average fraction of value updates that hit the clip bound
            if vf_clip_fractions:
                self.logger.record("train/clip_fraction_vf", float(np.mean(vf_clip_fractions)))

        # +INSTRUMENTATION: gradient-balance + value-scale diagnostics. These prepare for
        # reducing vf_coef and adding return normalization (PopArt) — see grad_balance.py and
        # src/agents/training/CLAUDE.md. All ride the standard logger → TensorBoard + launcher TUI.
        for _key, _val in grad_balance.items():
            self.logger.record(_key, _val)
        for _key, _val in value_scale_metrics(
            self.rollout_buffer.returns, self.rollout_buffer.values
        ).items():
            self.logger.record(_key, _val)
        if grad_norms:
            self.logger.record("train/grad_norm", float(np.mean(grad_norms)))

        # +BELIEF: hidden-opponent belief-aux diagnostics (only when the aux is on AND some minibatch
        # had believed slots). belief_species_acc is the headline: top-1 accuracy of predicting a
        # hidden mon's species — rises as the model learns to anticipate the un-revealed party.
        if belief_metrics:
            for _bk, _bvals in belief_metrics.items():
                self.logger.record(f"train/belief_{_bk}", float(np.mean(_bvals)))

        # +PopArt diagnostics: mu/sigma should TRACK train/return_mean/return_std (the running
        # normalizer estimate); value_weight_norm watches the POP rescale stay bounded (an explosion
        # signals a degenerate sigma / broken preservation). With PopArt on, train/value_loss is the
        # NORMALIZED loss (≈O(1)) and grad/value_share should fall toward ~0.4.
        if popart is not None:
            self.logger.record("popart/mu", float(self.policy.popart.mu))
            self.logger.record("popart/sigma", float(self.policy.popart.sigma))
            self.logger.record("popart/value_weight_norm", float(self.policy.value_net.weight.norm()))
