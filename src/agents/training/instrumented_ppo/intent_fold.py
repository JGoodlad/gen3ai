"""K8 — the OPPONENT-INTENT fold's ENTRY POINT and its shared static primitives (`gen3_opp_intent_v1`).

The opponent-intent loss is X5's FLAT pointer cross-entropy (`flat_intent_fold.py`): `intent_fold`
dispatches there whenever the extractor published `flat_intent_logits`, and returns `None` otherwise
(the block is SKIPPED — no term, no metric). The pointer is the only intent head the forward
publishes (the X5 version break retired the α / β heads' supervision with the blob belief).

THE STATIC CONTRACT (the learner micro-step is compile region R1, `torch.compile(..., fullgraph=True)`):
every output SHAPE depends only on input shapes, there is no host read (`.item()`, `float(t)`,
`int(t)`, `bool(t)`), no `nonzero`, no boolean-mask indexing, no `bincount`, and no Python branch on a
tensor VALUE. The primitives below are how a data-dependent metric is made static:

* a subset (`x[mask]`) -> the full batch with `torch.where(mask, per_row, 0)` BEFORE the reduction,
  divided by `mask.sum().clamp(min=1)` — a masked row can never inject NaN/Inf into a sum;
* "emit this metric only if the subset is non-empty" -> the metric is ALWAYS computed and carries a
  0/1 WEIGHT (`_Sink.put`);
* `bincount` -> a one-hot sum into a fixed `[n_classes]` vector;
* the marginal entropy's sum over the NON-ZERO classes -> the same terms stably compacted to the
  front (`argsort(~nz, stable=True)`) and summed.
"""
from __future__ import annotations

from typing import Any, Dict, Mapping, NamedTuple, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor

#: A metric: (value, weight). `weight` is exactly 1.0 when the metric is emitted on this
#: micro-batch, else exactly 0.0 — and `value` is then exactly 0.0 too.
Metric = Tuple[Tensor, Tensor]
Coef = Union[float, Tensor]

_P = "opp_intent/"


class IntentFoldOut(NamedTuple):
    """The fold's outputs, as tensors.

    `intent_term` — `intent_coef * CE`, the opponent-intent term the micro-step adds to the loss.
    `intent_present` — 0-d bool: the term carries at least one graded row.
    `metrics` — every `opp_intent/*` key, each a `(value, weight)` pair.
    """
    intent_term: Tensor
    intent_present: Tensor
    metrics: Dict[str, Metric]


# ------------------------------------------------------------------------------------- primitives
def _rate(hit: Tensor, cond: Tensor) -> Tuple[Tensor, Tensor]:
    """`hit` over the rows of `cond`, in float32 — `float(hit[cond].float().mean())` whenever `cond`
    has a row. Returns (rate, float32 count)."""
    c = cond.float().sum()
    return (hit & cond).float().sum() / c.clamp(min=1.0), c


def _masked_ce(logits: Tensor, tgt: Tensor, mask: Tensor) -> Tensor:
    """Mean CE over the rows of `mask` (values only; `cross_entropy(logits[m], t[m])`)."""
    lp = F.log_softmax(logits, dim=-1)
    safe = tgt.clamp(min=0, max=logits.shape[-1] - 1)
    per = -lp.gather(1, safe[:, None]).squeeze(1)
    return torch.where(mask, per, torch.zeros_like(per)).sum() / mask.sum().clamp(min=1)


def info_gain_nats_static(logits: Tensor, tgt: Tensor, mask: Tensor, odt: torch.dtype) -> Tensor:
    """INFORMATION GAIN over the batch's empirical marginal, in nats, over the rows of `mask`, as a
    0-d `odt` tensor: H(empirical label marginal) - CE(logits, label). 0 = no better than always
    predicting the base rate; the label marginal's entropy = a perfect predictor; negative = worse
    than the base rate. Invariant to a constant logit shift."""
    n_classes = logits.shape[-1]
    cls = torch.arange(n_classes, device=tgt.device)
    counts = ((tgt[:, None] == cls) & mask[:, None]).float().sum(dim=0)        # == bincount().float()
    p = counts / counts.sum().clamp_min(1.0)
    nz = p > 0
    terms = p * torch.where(nz, p, torch.ones_like(p)).log()                   # 0 where p == 0
    # Compact the non-zero terms to the front, IN ORDER: a CPU sum of the compacted vector (trailing
    # exact zeros) reproduces the float32 rounding of summing `p[nz]` bit for bit.
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
        # A metric that is not emitted reads exactly 0.0 (never NaN/Inf), so a `sum(w * v)`
        # downstream cannot be poisoned by a row set nobody looked at.
        self.out[_P + key] = (torch.where(weight, v, torch.zeros_like(v)), weight.to(self.odt))


def _scale(coef: Coef, x: Tensor) -> Tensor:
    return coef.to(x.dtype) * x if isinstance(coef, Tensor) else coef * x


# -------------------------------------------------------------------------------------- the fold
def intent_fold(fe: Any, obs: Mapping[str, Tensor], *, intent_coef: Coef,
                bot_label_weight: float) -> Optional[IntentFoldOut]:
    """The opponent-intent fold: the FLAT pointer's (`flat_intent_fold`). `None` ⇔ the block is
    SKIPPED — the extractor published no `flat_intent_logits` (opponent intent off), a STATIC fact,
    or the flat fold itself found no label (see `flat_intent_fold`)."""
    fl = fe.belief_supervision("flat_intent_logits")
    if fl is None:
        return None
    from agents.training.instrumented_ppo.flat_intent_fold import flat_intent_fold
    return flat_intent_fold(fe, obs, fl, intent_coef=intent_coef, bot_label_weight=bot_label_weight)
