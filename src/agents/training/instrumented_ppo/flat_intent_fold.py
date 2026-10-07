"""The FLAT opponent pointer's loss + metrics as ONE static, traceable fold (`gen3_x5_flat_pointer_v1`,
X5 build unit U4; design §3.7) — THE opponent-intent loss: `intent_fold` (`intent_fold.py`, the entry
point and the shared static primitives) dispatches here whenever the extractor published the flat
pointer.

Same static contract as `intent_fold` (the learner micro-step is compile region R1): every output
SHAPE depends only on input shapes, no host read, no boolean-mask indexing, no `nonzero`, no Python
branch on a tensor value; "emit only if non-empty" is a 0/1 metric WEIGHT.

THE LOSS. One cross-entropy over the flat list at the rows whose label names a live candidate
(`flat_intent_targets`): a move in the seats, OTHER_move, a switch target, OTHER_species. A belief
miss is an OTHER LABEL, so it is SUPERVISED (the OTHER_species label states "someone unseen" exactly).
`--intent-label-bot-weight` scales the bot-class rows before the mean, with the row-COUNT denominator.
A masked row's logits are replaced by zeros BEFORE the softmax (an exactly-zero gradient there; a
NaN in a SUPERVISED row still reaches the term — K9(c) fail-closed).

THE METRICS (`opp_intent/flat_*`, plus F-X5-8's isolated belief-miss share):
* ``other_label_rate`` — of the opponent's genuine CHOICES (kind MOVE or SWITCH), the share whose
  label is OTHER_move or OTHER_species: the belief-miss share, isolated from non-choices. Split ``other_move_label_rate`` (of MOVE choices) / ``other_species_label_rate`` (of
  SWITCH choices); ``flat_unmodeled_rate`` is the share of choices outside every candidate's support
  (masked, counted).
* the pointer's own reads, pooled and per opponent class: accuracy, info gain over the batch's
  empirical marginal, the KIND decision both ways (switch recall / precision), WHICH move given they
  moved, WHICH target given they switched.
"""
from __future__ import annotations

from typing import Any, Mapping, Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from agents.model.flat_intent import (LABEL_NONCHOICE, LABEL_OTHER_MOVE, LABEL_OTHER_SPECIES,
                                      LABEL_UNMODELED, FlatIntentInputs, flat_intent_targets)
from agents.model.opp_intent import INTENT_IGNORE, OPP_CLASS_BOT, OPP_CLASS_NAMES
from agents.training.instrumented_ppo.intent_fold import (Coef, IntentFoldOut, _rate, _scale, _Sink,
                                                          info_gain_nats_static)


def _flat_ce(logits: Tensor, tgt: Tensor, sup: Tensor, n_sup: Tensor, w_all: Optional[Tensor]) -> Tensor:
    """Masked-mean CE over `sup` (exactly 0.0 when empty); masked rows' logits zeroed before the softmax."""
    safe = torch.where(sup, tgt, torch.zeros_like(tgt))
    x = torch.where(sup[:, None], logits, torch.zeros_like(logits))
    per = -F.log_softmax(x, dim=-1).gather(1, safe[:, None]).squeeze(1)
    if w_all is not None:
        per = per * w_all
    return torch.where(sup, per, torch.zeros_like(per)).sum() / n_sup.clamp(min=1)


def _flat_subset_metrics(sink: _Sink, logits: Tensor, tgt: Tensor, mask: Tensor, gate: Tensor, k: int,
                         sfx: str = "") -> None:
    """The pointer's reads over ONE row subset (`mask`), each weighted by `gate` and its own emptiness."""
    pred = logits.argmax(dim=-1)
    true_sw = (tgt > k) & mask
    true_mv = (tgt <= k) & (tgt >= 0) & mask
    pred_sw = (pred > k) & mask
    v, _ = _rate(pred == tgt, mask)
    sink.put(f"flat_acc{sfx}", v, gate)
    v, c = _rate(pred > k, true_sw)
    sink.put(f"flat_switch_recall{sfx}", v, gate & (c > 0))
    v, c = _rate(tgt > k, pred_sw)
    sink.put(f"flat_switch_precision{sfx}", v, gate & (c > 0))
    v, _ = _rate(tgt > k, mask)
    sink.put(f"flat_switch_rate{sfx}", v, gate)
    neg = torch.full_like(logits, float("-inf"))
    col = torch.arange(logits.shape[-1], device=logits.device)
    mv_logits = torch.where((col <= k)[None, :], logits, neg)
    sw_logits = torch.where((col > k)[None, :], logits, neg)
    v, n_mv = _rate(mv_logits.argmax(dim=-1) == tgt, true_mv)
    sink.put(f"flat_move_recall_top1{sfx}", v, gate & (n_mv > 0))
    v, n_sw = _rate(sw_logits.argmax(dim=-1) == tgt, true_sw)
    sink.put(f"flat_switch_tgt_top1{sfx}", v, gate & (n_sw > 0))   # <= 33 chars with any class suffix (stdout table)
    sink.put(f"flat_info_gain_nats{sfx}", info_gain_nats_static(logits, tgt, mask, sink.odt), gate)


def _other_rates(sink: _Sink, cls: Tensor, kind: Tensor, rows: Tensor, gate: Tensor, sfx: str = "") -> None:
    """F-X5-8: the belief-miss share of the opponent's CHOICES, isolated from non-choices."""
    choice = (cls != LABEL_NONCHOICE) & rows
    other = ((cls == LABEL_OTHER_MOVE) | (cls == LABEL_OTHER_SPECIES)) & rows
    v, c = _rate(other, choice)
    sink.put(f"other_label_rate{sfx}", v, gate & (c > 0))
    v, c = _rate(cls == LABEL_OTHER_MOVE, choice & (kind == 0))
    sink.put(f"other_move_label_rate{sfx}", v, gate & (c > 0))
    v, c = _rate(cls == LABEL_OTHER_SPECIES, choice & (kind == 1))
    sink.put(f"other_species_label_rate{sfx}", v, gate & (c > 0))
    v, c = _rate(cls == LABEL_UNMODELED, choice)
    sink.put(f"flat_unmodeled_rate{sfx}", v, gate & (c > 0))


def flat_intent_fold_tensors(
    flat_logits: Tensor, fi: FlatIntentInputs, *, kind: Tensor, num: Tensor, switch_slot: Tensor,
    switch_species: Tensor, opp_class: Optional[Tensor] = None, intent_coef: Coef,
    bot_label_weight: float,
) -> IntentFoldOut:
    """The pure-tensor core. `flat_logits` is the LIVE supervision view; `fi` the forward's label-side
    stash; the label keys are the obs's (already aligned to the predictions)."""
    fl = flat_logits
    dev = fl.device
    odt = torch.promote_types(fl.dtype, torch.float32)
    sink = _Sink(odt, dev)
    B = kind.numel()
    k = fi.k
    kind = kind.long().reshape(-1)
    tgt, cls = flat_intent_targets(fi, kind, num, switch_slot, switch_species)
    oc: Optional[Tensor] = None if opp_class is None else opp_class.long().reshape(-1)
    w_all: Optional[Tensor] = None
    if oc is not None and bot_label_weight != 1.0:
        ones = torch.ones(oc.shape, dtype=fl.dtype, device=dev)
        w_all = torch.where(oc == OPP_CLASS_BOT, ones * float(bot_label_weight), ones)

    sup = tgt != INTENT_IGNORE
    n_sup = sup.sum()
    any_s = n_sup > 0
    lf = _flat_ce(fl, tgt, sup, n_sup, w_all)
    sink.put("flat_mask_rate", 1.0 - n_sup.to(odt) / max(B, 1))
    sink.put("flat_n_supervised", n_sup.to(odt))
    sink.put("flat_loss", lf.detach(), any_s)
    if oc is not None:
        v, _ = _rate(oc == OPP_CLASS_BOT, sup)
        sink.put("label_bot_frac", v, any_s)
    fl_d = fl.detach()
    all_rows = torch.ones_like(kind, dtype=torch.bool)
    true = torch.ones((), dtype=torch.bool, device=dev)
    _other_rates(sink, cls, kind, all_rows, true)
    _flat_subset_metrics(sink, fl_d, tgt, sup, any_s, k)
    if oc is not None:
        for code, name in OPP_CLASS_NAMES.items():
            rows = oc == code
            m = sup & rows
            n_m = m.sum()
            ok = any_s & (n_m >= 2)
            sink.put(f"flat_n_supervised_{name}", n_m.to(odt), ok)
            _flat_subset_metrics(sink, fl_d, tgt, m, ok, k, f"_{name}")
            _other_rates(sink, cls, kind, rows, rows.sum() >= 2, f"_{name}")
    return IntentFoldOut(intent_term=_scale(intent_coef, lf), intent_present=any_s, metrics=sink.out)


def flat_intent_fold(fe: Any, obs: Mapping[str, Tensor], flat_logits: Tensor, *, intent_coef: Coef,
                     bot_label_weight: float) -> Optional[IntentFoldOut]:
    """The reads, then `flat_intent_fold_tensors`. None ⇔ no label in the obs or no label-side stash."""
    fi = getattr(fe, "last_flat_intent", None)
    if fi is None or "opp_action_kind" not in obs:
        return None
    if "opp_switch_species" not in obs:
        raise RuntimeError(
            "the flat opponent pointer needs `opp_switch_species` in the obs: a hidden switch-in is "
            "labelled by its species (a hypothesis slot or OTHER_species) — without it every hidden "
            "switch would be silently unlabelled.")
    return flat_intent_fold_tensors(
        flat_logits, fi, kind=obs["opp_action_kind"], num=obs["opp_action_num"],
        switch_slot=obs["opp_switch_slot"], switch_species=obs["opp_switch_species"],
        opp_class=obs.get("opp_class"), intent_coef=intent_coef, bot_label_weight=bot_label_weight)
