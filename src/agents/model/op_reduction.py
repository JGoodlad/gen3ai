"""The op's PRINCIPLED REDUCTIONS over the opponent's believed moves (`--op-reduction principled`,
`gen3_op_reduction_principled_v1`; architecture audit F6b, owner 2026-10-08).

**The defect.** Under ``max`` (production) the damage operator collapses the opponent's believed-move axis
with a hard maximum taken SEPARATELY per channel: ``max_c(w_c · low_c)``, ``max_c(w_c · high_c)``, … per
physical / special channel and per defender, with ``acc`` and ``provenance`` read at the argmax of yet another
channel (`design_pair_reduction.md` §2, defects D1 / D2). Each channel may describe a DIFFERENT move, so the
row is an opponent who clicks several moves at once; and ``max_c(w_c · v_c)`` is neither the expectation nor
the worst case (D1).

**The replacement — compute the FACTS exactly; the trunk's attention does the judgment.** Two coherent facts,
each from ONE mixture over the believed moves, no channel from a different move than its neighbours:

* **the α-weighted EXPECTATION** ``Σ_c α_c · v_c`` — what their likely action does. ``α`` is ONE
  distribution over the move axis, shared by every channel and every defender (Contract W,
  `design_pair_reduction.md` §3.1). At the op's tier (T1, before the trunk) the only distribution over
  "which of its moves the attacker clicks" is its presence belief, so ``α = w / Σ_m w_m`` — R1
  ``belief_mean`` (`pair_reduce.alpha_belief_mean`), normalised by the attacker's TOTAL presence over the
  whole move axis (a candidate cut off a top-K is unpriced mass, never renormalised away — the
  ``pair_outcome`` rule for an unmodelled seat). The flat opponent pointer's α is a T2 quantity (it reads
  the trunk), so it cannot weight a T1 reduction the trunk itself reads (`tier_contract`); its α-weighted
  rows already reach the pointer cells downstream (`pair_outcome` / the move-resolution family).
* **the noisy-OR WORST CASE** ``1 − Π_c (1 − p_c)`` with ``p_c = w_c · P(X | c)`` — P(they hold at least
  one move that achieves X), for a BINARY event X (a KO; Pursuit presence), reading each move's presence
  ``w_c`` exactly as the op reads it (DETACHED under X5). Independence across moves is the approximation
  (X5's fixed-mass presence is a conditional Bernoulli with Σ π = k); it is exact for presence 0 / 1.

Both are smooth and symmetric in the candidate axis: no argmax, no tie, no selection — a permutation of
the moves, or a swap of two tied candidates, leaves both unchanged up to fp reassociation.

A LEAF module (torch + `index_max` only): every forward module imports it without a cycle. Its one
discrete op, the incoming row's "damaging candidate" test, is declared in `selection_sites`.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch

from agents.model.index_max import max_by_index

#: The `--op-reduction` modes. ``max`` = the legacy per-channel hard maximum (production, byte-identical);
#: ``principled`` = the α-weighted expectation + the noisy-OR worst case (this module).
OP_REDUCTION_MODES: Tuple[str, ...] = ("max", "principled")

#: The width of the incoming per-mon WORST-CASE row under ``principled``: ``[P(some physical move KOs),
#: P(some special move KOs)]`` per our mon — the noisy-OR facts the expectation row cannot carry. Delivered to
#: our mon tokens by the extractor's zero-init `op_worst_proj` (never in the flat op block, so the block's
#: layout, its `out_gain` and every slicer are unchanged).
OP_WORST_DIM = 2
OP_WORST_COORDS: Tuple[str, ...] = ("p_ko_any_phys", "p_ko_any_spec")
assert len(OP_WORST_COORDS) == OP_WORST_DIM

_TINY = torch.finfo(torch.float32).tiny


def presence_alpha(w: torch.Tensor, total: Optional[torch.Tensor] = None) -> torch.Tensor:
    """α over the candidate axis (the LAST dim): ``w / total``, ``total`` = Σ w over the last dim unless given
    (the attacker's presence over its WHOLE move axis when ``w`` is a top-K slice of it; broadcastable to
    ``w[..., :1]``). ``w ≥ 0``, so an all-zero row (no attacker) reads all-zero α — the "no threat ⇒ 0"
    convention — with no comparison: ``0 / tiny == 0``."""
    if total is None:
        total = w.sum(dim=-1, keepdim=True)
    return w / total.clamp_min(_TINY)


def expectation(alpha: torch.Tensor, value: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """THE expectation fact: ``Σ_c α_c · value_c`` over ``dim`` (α broadcast against ``value``)."""
    return (alpha * value).sum(dim=dim)


def noisy_or(p: torch.Tensor, dim: int = -1, keepdim: bool = False) -> torch.Tensor:
    """THE worst-case fact: ``1 − Π_c (1 − p_c)`` over ``dim`` — P(at least one of independent events with
    probabilities ``p_c`` happens). ``p`` in [0, 1]. Exact at the edges: all ``p_c = 0`` → 0, any ``p_c = 1``
    → 1. The product (not a log-sum) keeps ``p = 1`` exact with a finite gradient (a log1p would need a
    comparison guard against ``log 0``)."""
    out: torch.Tensor = 1.0 - torch.prod(1.0 - p, dim=dim, keepdim=keepdim)
    return out


def believed_reduce(wv: torch.Tensor, w_total: Optional[torch.Tensor]) -> torch.Tensor:
    """THE reduction over an attacker's believed candidates (the LAST dim) at the op's per-attacker kernels
    (C1b / C2 / C3 / D4): ``wv`` = presence × value per candidate.

    ``w_total`` None (``max``, production): ``max_by_index(wv)`` — the legacy presence-scaled hard maximum,
    the expression the kernels always took (bit-identical). ``w_total`` given (``principled``): the attacker's
    total presence, broadcastable to the reduced shape — ``Σ_c wv_c / w_total`` = ``Σ_c α_c · value_c``, the
    α-weighted EXPECTATION with ``α = w / w_total`` (top-K-cut mass unpriced, never renormalised away)."""
    if w_total is None:
        return max_by_index(wv)
    return wv.sum(dim=-1) / w_total.clamp_min(_TINY)


def incoming_principled(w: torch.Tensor, w_total: Optional[torch.Tensor], low: torch.Tensor, high: torch.Tensor,
                        crit: torch.Tensor, ko: torch.Tensor, acc: torch.Tensor, phys: torch.Tensor,
                        high_cb: torch.Tensor, ko_cb: torch.Tensor) -> Tuple[torch.Tensor, ...]:
    """The op's INCOMING per-mon row under ``principled`` — the replacement of its ten channel maxima, its two
    argmax-picked accuracies and its argmax-picked provenance (`DamageOperator.forward`).

    ``w`` `[B,C]` the opponent active's candidate presence (the weights the max read); ``w_total`` `[B,1]` its
    presence over the whole move axis (None → Σ ``w``); ``low`` / ``high`` / ``crit`` / ``ko`` / ``high_cb`` /
    ``ko_cb`` `[B,6,C]` the per-(our mon, candidate) rolls (damage IF it lands; ``ko`` accuracy-folded);
    ``acc`` / ``phys`` `[B,C]` the candidate tables.

    Returns ``(phys_low, phys_high, phys_crit, phys_pko, phys_acc, spec_low, spec_high, spec_crit, spec_pko,
    spec_acc, provenance, phys_high_cb, phys_pko_cb, worst)`` — each `[B,6]`, ``worst`` `[B,6,2]`:

    * channel ``x`` ∈ {low, high, crit, pko}: ``Σ_c α_c · x_jc · chan_c`` — the expected roll / KO chance their
      α-mixed click inflicts through that channel (the channel is a mask on the candidate, never a second α);
    * ``acc``: ``Σ_c α_c · acc_c · chan_c · 1[c damages j]`` — P(their α-mixed click is a damaging move of that
      channel that hits mon j) (max read the accuracy at the channel's dominant move);
    * ``provenance``: ``Σ_c α_c · w_c · 1[c damages j]`` — the α-weighted presence of the damaging threat, 1 for
      a fully revealed attacking set (max read the dominant move's presence);
    * the Choice-Band tail: the same expectation of the CB-conditional high roll / KO (physical only);
    * ``worst``: ``[noisy_or(w · ko · phys), noisy_or(w · ko · spec)]`` — P(some move of theirs KOs mon j).
    """
    alpha = presence_alpha(w, w_total)[:, None, :]                                    # [B,1,C]
    pm = phys[:, None, :]                                                             # [B,1,C]
    sm = 1.0 - pm
    dmg_on = (high > 0).to(high.dtype)                       # a damaging candidate on THIS mon (immune → 0)
    hit = acc[:, None, :] * dmg_on                                                    # [B,6,C]
    a_p, a_s = alpha * pm, alpha * sm                                                 # [B,1,C]
    out = []
    for a in (a_p, a_s):
        out += [expectation(a, low), expectation(a, high), expectation(a, crit), expectation(a, ko),
                expectation(a, hit)]
    provenance = expectation(alpha, w[:, None, :] * dmg_on)                           # [B,6]
    high_cb_e = expectation(a_p, high_cb)
    ko_cb_e = expectation(a_p, ko_cb)
    p_ko = w[:, None, :] * ko                                                         # [B,6,C]
    worst = torch.stack([noisy_or(p_ko * pm), noisy_or(p_ko * sm)], dim=-1)           # [B,6,2]
    return (*out, provenance, high_cb_e, ko_cb_e, worst)


__all__ = ["OP_REDUCTION_MODES", "OP_WORST_DIM", "OP_WORST_COORDS", "presence_alpha", "expectation",
           "noisy_or", "believed_reduce", "incoming_principled"]
