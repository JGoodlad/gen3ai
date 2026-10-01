"""K8 — the OPPONENT-INTENT fold as ONE static, traceable function (`gen3_opp_intent_v1`).

`intent_fold` computes exactly what the inline opponent-intent block of `ppo.py::train()` computes
(the α/β cross-entropies, the set-valued partial credit and every `opp_intent/*` metric), but in a
form `torch.compile(..., fullgraph=True)` can trace: every output SHAPE depends only on input shapes,
there is no host read (`.item()`, `float(t)`, `int(t)`, `bool(t)`), no `nonzero`, no boolean-mask
indexing, no `bincount`, and no Python branch on a tensor VALUE. The only Python branches are on
STATIC facts — a stash or obs key being `None`, a tensor's shape, a Python flag, and
`bot_label_weight == 1.0` (which, as in the legacy code, selects the bit-identical unweighted path).

HOW THE LEGACY DATA-DEPENDENCE WAS MADE STATIC (one line each):

* a subset (`x[mask]`) -> the full batch with `torch.where(mask, per_row, 0)` BEFORE the reduction,
  divided by `mask.sum().clamp(min=1)` — a masked row can never inject NaN/Inf into a sum;
* "emit this metric only if the subset is non-empty / has >= 2 rows" -> the metric is ALWAYS
  computed and carries a 0/1 WEIGHT that is 1 exactly when the legacy code would have emitted it;
* "return None when no row qualifies" (`set_valued_switch_loss`, an all-unsupervised α/β) -> an
  exactly-0.0 term plus a 0-d bool `present`;
* `bincount` -> a one-hot sum into a fixed `[n_classes]` vector;
* the marginal entropy's sum over the NON-ZERO classes -> the same terms stably compacted to the
  front (`argsort(~nz, stable=True)`) and summed: bit-identical to the legacy float32 sum on CPU;
* the legacy `.float()` casts (the set-valued softmax, α's switch probability, the 0/1 rates, the
  class counts) are KEPT, so the float32 arithmetic is the legacy's own.

K9(c) FAIL-CLOSED is preserved: only rows the legacy code masks are excluded (the `isneginf` reach
test keeps a NaN logit SUPERVISED, so a NaN in a supervised row still reaches the term); and the
GRADIENT of a masked row is the legacy's too — the unweighted α/β CE reads the RAW logits, as
`cross_entropy(ignore_index=...)` does (a NaN in an IGNORED row therefore still yields a NaN
gradient there, exactly as before), while the weighted CE and the set-valued term substitute 0 for
masked rows' logits, which is what the legacy `logits[sup]` indexing does (a zero gradient there).

The caller folds, in the legacy order: `loss += setvalued_term` (only when `setvalued_present`
matters — it is exactly 0.0 otherwise), then `loss += intent_term`. `intent_fold` returning `None`
is the legacy block being SKIPPED entirely (`opp_intent_term = None`, no metrics). When it returns a
value, the legacy `opp_intent_term` was NOT None — even if no row was supervised (the legacy term
was then a grad-less 0.0; `intent_present` is False and `intent_term` is exactly 0.0).

Equivalence is pinned by `intent_fold_test.py` against a verbatim copy of the legacy block.
"""
from __future__ import annotations

from typing import Any, Dict, Mapping, NamedTuple, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor

from agents.model.opp_intent import (INTENT_IGNORE, OPP_CLASS_BOT, OPP_CLASS_NAMES,
                                     match_seats_to_move_num, resolve_believed_slot_by_content)

#: A metric: (value, weight). `weight` is exactly 1.0 when the legacy block would have emitted the
#: key on this micro-batch, else exactly 0.0 — and `value` is then exactly 0.0 too.
Metric = Tuple[Tensor, Tensor]
Coef = Union[float, Tensor]

_P = "opp_intent/"


class IntentFoldOut(NamedTuple):
    """Everything the legacy block produced, as tensors.

    `setvalued_term` — `intent_coef * setvalued_coef * sv`, the WEIGHTED set-valued contribution
    (what the legacy block added to `loss` first); exactly 0.0 when it would not have been added.
    `intent_term` — `intent_coef * (alpha CE + beta CE)`, the legacy `opp_intent_term`.
    `*_present` — 0-d bool: the term carries at least one graded row.
    `metrics` — every `opp_intent/*` key the legacy block can emit under these STATIC inputs, with
    the legacy name; keys whose static precondition fails (no `opp_class`, no beta head, ...) are
    absent, exactly as in the legacy dict.
    """
    setvalued_term: Tensor
    setvalued_present: Tensor
    intent_term: Tensor
    intent_present: Tensor
    metrics: Dict[str, Metric]


# ------------------------------------------------------------------------------------- primitives
def _rate(hit: Tensor, cond: Tensor) -> Tuple[Tensor, Tensor]:
    """`hit` over the rows of `cond`, in float32 — bit-identical to the legacy
    `float(hit[cond].float().mean())` whenever `cond` has a row. Returns (rate, float32 count)."""
    c = cond.float().sum()
    return (hit & cond).float().sum() / c.clamp(min=1.0), c


def _masked_ce(logits: Tensor, tgt: Tensor, mask: Tensor) -> Tensor:
    """Mean CE over the rows of `mask` (values only; the metrics' `cross_entropy(logits[m], t[m])`)."""
    lp = F.log_softmax(logits, dim=-1)
    safe = tgt.clamp(min=0, max=logits.shape[-1] - 1)
    per = -lp.gather(1, safe[:, None]).squeeze(1)
    return torch.where(mask, per, torch.zeros_like(per)).sum() / mask.sum().clamp(min=1)


def info_gain_nats_static(logits: Tensor, tgt: Tensor, mask: Tensor, odt: torch.dtype) -> Tensor:
    """`opp_intent.info_gain_nats(logits[mask], tgt[mask])`, statically, as a 0-d `odt` tensor."""
    n_classes = logits.shape[-1]
    cls = torch.arange(n_classes, device=tgt.device)
    counts = ((tgt[:, None] == cls) & mask[:, None]).float().sum(dim=0)        # == bincount().float()
    p = counts / counts.sum().clamp_min(1.0)
    nz = p > 0
    terms = p * torch.where(nz, p, torch.ones_like(p)).log()                   # 0 where p == 0
    # Compact the non-zero terms to the front, IN ORDER: the legacy summed `p[nz]`, and a CPU sum of
    # the compacted vector (trailing exact zeros) reproduces its float32 rounding bit for bit.
    order = torch.argsort((~nz).to(torch.int32), stable=True)
    h_marginal = -(terms.gather(0, order).sum())
    return h_marginal.to(odt) - _masked_ce(logits, tgt, mask).to(odt)


class _Sink:
    """The metric dict under construction — `(value, weight)` pairs, all in `odt`."""

    def __init__(self, odt: torch.dtype, device: torch.device):
        self.odt = odt
        self.one = torch.ones((), dtype=odt, device=device)
        self.out: Dict[str, Metric] = {}

    def put(self, key: str, value: Tensor, weight: Optional[Tensor] = None) -> None:
        v = value.to(self.odt)
        if weight is None:
            self.out[_P + key] = (v, self.one)
            return
        # A metric the legacy block would NOT have emitted reads exactly 0.0 (never NaN/Inf), so a
        # `sum(w * v)` downstream cannot be poisoned by a row set the legacy code never looked at.
        self.out[_P + key] = (torch.where(weight, v, torch.zeros_like(v)), weight.to(self.odt))


# ---------------------------------------------------------------------------------- subset metrics
def _alpha_subset_metrics(sink: _Sink, logits: Tensor, tgt: Tensor, mask: Tensor, gate: Tensor,
                          sfx: str = "") -> None:
    """`opp_intent._alpha_subset_metrics(logits[mask], tgt[mask], k, sfx)`, each key weighted by
    `gate` AND that key's own legacy emptiness condition."""
    k = logits.shape[-1] - 1
    pred = logits.argmax(dim=-1)
    is_sw = (tgt == k) & mask
    said_sw = (pred == k) & mask
    not_sw = mask & (tgt != k)
    not_said = mask & (pred != k)

    v, _ = _rate(pred == tgt, mask)
    sink.put(f"alpha_acc{sfx}", v, gate)
    v, c = _rate(pred == k, is_sw)
    sink.put(f"alpha_switch_recall{sfx}", v, gate & (c > 0))
    v, c = _rate(tgt == k, said_sw)
    sink.put(f"alpha_switch_precision{sfx}", v, gate & (c > 0))
    v, n_mv = _rate(pred != k, not_sw)
    sink.put(f"alpha_move_kind_recall{sfx}", v, gate & (n_mv > 0))
    v, c = _rate(tgt != k, not_said)
    sink.put(f"alpha_move_kind_precision{sfx}", v, gate & (c > 0))
    v, _ = _rate(pred == k, mask)
    sink.put(f"alpha_pred_switch_rate{sfx}", v, gate)

    has_mv = gate & (n_mv > 0)
    move_logits = logits[:, :k]
    v, _ = _rate(move_logits.argmax(dim=-1) == tgt, not_sw)
    sink.put(f"alpha_move_recall_top1{sfx}", v, has_mv)
    if k >= 2:                                                       # static: the seat count
        top2 = move_logits.topk(2, dim=-1).indices
        v, _ = _rate((top2 == tgt[:, None]).any(dim=-1), not_sw)
        sink.put(f"alpha_move_recall_top2{sfx}", v, has_mv)
    v, _ = _rate(tgt == 0, not_sw)
    sink.put(f"alpha_move_baseline_argmax_w{sfx}", v, has_mv)
    v, _ = _rate(tgt == k, mask)
    sink.put(f"alpha_switch_rate{sfx}", v, gate)

    sink.put(f"alpha_info_gain_nats{sfx}", info_gain_nats_static(logits, tgt, mask, sink.odt), gate)
    sink.put(f"alpha_info_gain_nats_move{sfx}",
             info_gain_nats_static(logits, tgt, not_sw, sink.odt), gate & (n_mv > 1))


def _beta_subset_metrics(sink: _Sink, logits: Tensor, tgt: Tensor, mask: Tensor, gate: Tensor,
                         sfx: str = "") -> None:
    """`opp_intent._beta_subset_metrics(logits[mask], tgt[mask], sfx)`, weighted by `gate`."""
    v, _ = _rate(logits.argmax(dim=-1) == tgt, mask)
    sink.put(f"beta_recall_top1{sfx}", v, gate)
    if logits.shape[-1] >= 2:                                        # static
        b2 = logits.topk(2, dim=-1).indices
        v, _ = _rate((b2 == tgt[:, None]).any(dim=-1), mask)
        sink.put(f"beta_recall_top2{sfx}", v, gate)
    sink.put(f"beta_info_gain_nats{sfx}", info_gain_nats_static(logits, tgt, mask, sink.odt), gate)


def _switch_coverage(sink: _Sink, kind: Tensor, need: Optional[Tensor], content: Optional[Tensor],
                     rows: Tensor, gate: Tensor, sfx: str = "") -> None:
    """`opp_intent.switch_coverage_metrics(kind, need, content, rows, sfx)`, weighted by `gate`."""
    odt = sink.odt
    n_sw = ((kind == 1) & rows).float().sum().to(odt)
    if need is None or content is None:                              # static: belief path off
        want = torch.zeros((), dtype=odt, device=kind.device)
        got = want
    else:
        want = (need & rows).float().sum().to(odt)
        got = (need & (content >= 0) & rows).float().sum().to(odt)
    present = gate & (n_sw > 0)
    den = n_sw.clamp(min=1.0)
    sink.put(f"beta_switch_n{sfx}", n_sw, present)
    sink.put(f"beta_switch_to_revealed{sfx}", (n_sw - want) / den, present)
    sink.put(f"beta_switch_to_hidden_found{sfx}", got / den, present)
    sink.put(f"beta_switch_to_hidden_missed{sfx}", (want - got) / den, present)
    sink.put(f"beta_belief_miss_rate{sfx}", 1.0 - got / want.clamp(min=1.0), present & (want > 0))


# ------------------------------------------------------------------------------------------ losses
def _ce_term(logits: Tensor, tgt: Tensor, sup: Tensor, n_sup: Tensor,
             w_all: Optional[Tensor]) -> Tensor:
    """`intent_losses`' per-axis CE: the masked mean over `sup` (exactly 0.0 when `sup` is empty)."""
    safe = torch.where(sup, tgt, torch.zeros_like(tgt))
    if w_all is None:
        # `cross_entropy(logits, tgt, ignore_index=...)`: the RAW logits, so an IGNORED row's
        # gradient is whatever the legacy call gives it (0 for a finite row).
        lp = F.log_softmax(logits, dim=-1)
        per = -lp.gather(1, safe[:, None]).squeeze(1)
    else:
        # `cross_entropy(logits[sup], tgt[sup], reduction="none") * w[sup]`: masked rows never
        # enter, so their logits are replaced BEFORE the softmax (an exactly-zero gradient).
        x = torch.where(sup[:, None], logits, torch.zeros_like(logits))
        per = -F.log_softmax(x, dim=-1).gather(1, safe[:, None]).squeeze(1) * w_all
    return torch.where(sup, per, torch.zeros_like(per)).sum() / n_sup.clamp(min=1)


def _set_valued(beta_logits: Tensor, believed: Tensor, miss: Tensor) -> Tuple[Tensor, Tensor]:
    """`opp_intent.set_valued_switch_loss` statically: (sv, present). The legacy `.float()` is kept."""
    finite = ~torch.isneginf(beta_logits)               # K9(c): only the deliberate -inf is illegal
    avail = (believed > 0.5) & finite
    rows = miss & (avail.sum(dim=-1) > 0)
    x = beta_logits.float()
    x = torch.where(rows[:, None], x, torch.zeros_like(x))
    p = torch.softmax(x, dim=-1)
    mass = (p * avail.float()).sum(dim=-1)
    per = -(mass.clamp_min(1e-8).log())
    n = rows.sum()
    return torch.where(rows, per, torch.zeros_like(per)).sum() / n.clamp(min=1), n > 0


def _scale(coef: Coef, x: Tensor) -> Tensor:
    return coef.to(x.dtype) * x if isinstance(coef, Tensor) else coef * x


# -------------------------------------------------------------------------------------- the fold
def intent_fold_tensors(
    alpha_logits: Tensor, beta_logits: Optional[Tensor], seat_nums: Tensor, *,
    kind: Tensor, num: Tensor, switch_slot: Tensor,
    switch_species: Optional[Tensor] = None, believed_mask: Optional[Tensor] = None,
    species_logits: Optional[Tensor] = None, opp_class: Optional[Tensor] = None,
    intent_coef: Coef, setvalued_coef: Coef, setvalued_on: bool, bot_label_weight: float,
) -> IntentFoldOut:
    """The pure-tensor core (what a compiled region wraps). Inputs are the legacy block's locals:
    `alpha_logits`/`beta_logits` the LIVE supervision views, `seat_nums` the α seat move nums,
    `kind`/`num`/`switch_slot`/`switch_species`/`opp_class` the obs label keys, `believed_mask` /
    `species_logits` the belief stashes (`None` = that path is off, exactly as in the legacy code).
    `setvalued_on` is the legacy `beta_setvalued_coef > 0.0`; `bot_label_weight` is a Python float
    (its `== 1.0` selects the unweighted CE, as `intent_label_weights` does)."""
    al = alpha_logits
    dev = al.device
    odt = torch.promote_types(al.dtype, torch.float32)
    sink = _Sink(odt, dev)
    B = kind.numel()
    false = torch.zeros((), dtype=torch.bool, device=dev)

    kind = kind.long().flatten()
    num = num.long().flatten()
    atgt = match_seats_to_move_num(seat_nums, num, kind, seat_nums.shape[-1])
    btgt = switch_slot.long().flatten()
    btgt = torch.where(kind == 1, btgt, torch.full_like(btgt, INTENT_IGNORE))

    content_on = (switch_species is not None and believed_mask is not None
                  and species_logits is not None)
    need: Optional[Tensor] = None
    content: Optional[Tensor] = None
    if content_on:
        assert switch_species is not None and believed_mask is not None and species_logits is not None
        content = resolve_believed_slot_by_content(species_logits.detach(), believed_mask.float(),
                                                   switch_species.long().flatten())
        need = (kind == 1) & (btgt < 0)
        btgt = torch.where(need, content, btgt)
        believed_targets = (need & (content >= 0)).float().sum()
        wanted_content = need.float().sum()
    else:
        believed_targets = torch.zeros((), dtype=odt, device=dev)
        wanted_content = torch.zeros((), dtype=odt, device=dev)

    bl = beta_logits
    if bl is not None:
        safe = btgt.clamp(min=0, max=bl.shape[-1] - 1)
        reach = ~torch.isneginf(bl.detach().gather(1, safe[:, None]).squeeze(1))
        btgt = torch.where(reach, btgt, torch.full_like(btgt, INTENT_IGNORE))

    # SET-VALUED partial credit (computed before the CE, folded first — the legacy order).
    if setvalued_on and bl is not None and content_on:
        assert need is not None and content is not None and believed_mask is not None
        miss = need & (content < 0)
        sv, sv_present = _set_valued(bl, believed_mask.float(), miss)
        # Legacy `opp_intent_coef * beta_setvalued_coef * _sv`: the coefficient PRODUCT first.
        setvalued_term = _scale(intent_coef * setvalued_coef, sv)
        sv_rows = miss.float().sum()
    else:
        sv = None
        sv_present = false
        setvalued_term = torch.zeros((), dtype=odt, device=dev)

    # ---- intent_losses
    oc: Optional[Tensor] = None if opp_class is None else opp_class.long().reshape(-1)
    w_all: Optional[Tensor] = None
    if oc is not None and bot_label_weight != 1.0:
        ones = torch.ones(oc.shape, dtype=al.dtype, device=dev)
        w_all = torch.where(oc == OPP_CLASS_BOT, ones * float(bot_label_weight), ones)

    al_d = al.detach()
    sup = atgt != INTENT_IGNORE
    n_sup = sup.sum()
    any_a = n_sup > 0
    sink.put("alpha_mask_rate", 1.0 - n_sup.to(odt) / max(B, 1))
    sink.put("alpha_n_supervised", n_sup.to(odt))
    la = _ce_term(al, atgt, sup, n_sup, w_all)
    total = la
    sink.put("alpha_loss", la.detach(), any_a)
    if oc is not None:
        v, _ = _rate(oc == OPP_CLASS_BOT, sup)
        sink.put("label_bot_frac", v, any_a)
    _alpha_subset_metrics(sink, al_d, atgt, sup, any_a)
    if oc is not None:
        for code, name in OPP_CLASS_NAMES.items():
            m = sup & (oc == code)
            n_m = m.sum()
            ok = any_a & (n_m >= 2)
            sink.put(f"alpha_n_supervised_{name}", n_m.to(odt), ok)
            _alpha_subset_metrics(sink, al_d, atgt, m, ok, f"_{name}")

    any_b = false
    if bl is not None:
        bl_d = bl.detach()
        supb = btgt != INTENT_IGNORE
        n_supb = supb.sum()
        any_b = n_supb > 0
        sink.put("beta_mask_rate", 1.0 - n_supb.to(odt) / max(B, 1))
        sink.put("beta_n_supervised", n_supb.to(odt))
        lb = _ce_term(bl, btgt, supb, n_supb, w_all)
        total = total + lb
        sink.put("beta_loss", lb.detach(), any_b)
        _beta_subset_metrics(sink, bl_d, btgt, supb, any_b)
        if oc is not None:
            for code, name in OPP_CLASS_NAMES.items():
                m = supb & (oc == code)
                n_m = m.sum()
                ok = any_b & (n_m >= 2)
                sink.put(f"beta_n_supervised_{name}", n_m.to(odt), ok)
                _beta_subset_metrics(sink, bl_d, btgt, m, ok, f"_{name}")
        # THE FALSIFIER for the alpha/beta split: beta's recall bucketed by alpha's switch confidence.
        p_sw = torch.softmax(al_d.float(), dim=-1)[:, -1]
        b_ok = bl_d.argmax(dim=-1) == btgt
        conf = p_sw >= 0.5
        for name, cm in (("alpha_confident", conf), ("alpha_unsure", ~conf)):
            v, c = _rate(b_ok, supb & cm)
            sink.put(f"beta_recall_{name}", v, any_b & (c > 1))

    # ---- the block's own counters, the coverage matrix, the set-valued metrics
    sink.put("beta_believed_targets", believed_targets)
    sink.put("beta_wanted_content", wanted_content)
    all_rows = torch.ones_like(kind, dtype=torch.bool)
    true = torch.ones((), dtype=torch.bool, device=dev)
    _switch_coverage(sink, kind, need, content, all_rows, true)
    if oc is not None:
        for code, name in OPP_CLASS_NAMES.items():
            rows = oc == code
            _switch_coverage(sink, kind, need, content, rows, rows.sum() >= 2, f"_{name}")
    if sv is not None:
        sink.put("beta_setvalued_loss", sv.detach(), sv_present)
        sink.put("beta_setvalued_rows", sv_rows, sv_present)

    return IntentFoldOut(setvalued_term=setvalued_term, setvalued_present=sv_present,
                         intent_term=_scale(intent_coef, total), intent_present=any_a | any_b,
                         metrics=sink.out)


def intent_fold(fe: Any, obs: Mapping[str, Tensor], *, intent_coef: Coef, setvalued_coef: Coef,
                setvalued_on: bool, bot_label_weight: float) -> Optional[IntentFoldOut]:
    """The legacy block's reads, then `intent_fold_tensors`. `None` ⇔ the legacy block was SKIPPED
    (no α logits, no seat nums, or no `opp_action_kind` label in the obs) — a STATIC fact."""
    al = fe.belief_supervision("alpha_logits")
    bl = fe.belief_supervision("beta_logits")
    sn = fe.last_alpha_seat_nums
    if al is None or sn is None or "opp_action_kind" not in obs:
        return None
    blog = getattr(fe, "last_belief_logits", None)
    return intent_fold_tensors(
        al, bl, sn, kind=obs["opp_action_kind"], num=obs["opp_action_num"],
        switch_slot=obs["opp_switch_slot"], switch_species=obs.get("opp_switch_species"),
        believed_mask=getattr(fe, "last_opp_believed_mask", None),
        species_logits=(blog["species"] if blog is not None and "species" in blog else None),
        opp_class=obs.get("opp_class"), intent_coef=intent_coef, setvalued_coef=setvalued_coef,
        setvalued_on=setvalued_on, bot_label_weight=bot_label_weight)
