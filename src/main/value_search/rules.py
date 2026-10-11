"""The ACTING RULES and the POLICY-IMPROVEMENT TARGET of the one-ply value search (pure numpy).

Everything here is Gumbel MuZero's (Danihelka et al., ICLR 2022) final-selection machinery with NO
Gumbel noise (`designs/endstate/design_value_discrimination.md` §7):

* ``sigma(q̃) = (c_visit + N_max) · c_scale · q̃`` with q̃ the leaf values min-max rescaled to [0, 1]
  over the actions they cover; ``c_visit`` 50, ``c_scale`` **1.0** (the paper's board-game value). The
  ``mctx`` default 0.1 was the pilot's registered rule and it LOST 1–3 points of decision improvement
  against a plain argmax of the leaf (R-PRIOR, PRIOR HURTS on both checkpoints); the post-hoc scale read
  (`target_scale_read.json`) recovers the full argmax value at c_scale ≥ 1.
* the GUMBEL score of an action = its policy logit + sigma(q̃): prior-aware, value-driven.
* SEQUENTIAL HALVING (Karnin et al. 2013, Gumbel AlphaZero's allocation): a playout budget split over
  ⌈log₂ m⌉ rounds among the policy's top-m, halving by the GUMBEL score after each round.
* the COMPLETED-Q policy target: softmax(logits + sigma(completed q)), unvisited actions completed by
  the prior-weighted mixed value (``mctx.qtransform_completed_by_mix_value``).

Ties break toward the policy's own order (top-1 first), deterministically (rule 8: ``TIE`` = 1e-6).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

C_VISIT = 50.0
C_SCALE = 1.0
TIE = 1e-6


def policy_order(logits: np.ndarray, legal: Sequence[int]) -> List[int]:
    """The legal actions in the policy's order: logit descending, the lower index first on a tie."""
    return sorted((int(a) for a in legal), key=lambda a: (-float(logits[a]), a))


def rescale(q: Dict[int, float]) -> Dict[int, float]:
    """Min-max rescale to [0, 1] over the actions given (all equal ⇒ all 0)."""
    if not q:
        return {}
    lo, hi = min(q.values()), max(q.values())
    if hi - lo <= 1e-12:
        return {a: 0.0 for a in q}
    return {a: (v - lo) / (hi - lo) for a, v in q.items()}


def sigma(q_rescaled: float, n_max: float, c_visit: float = C_VISIT, c_scale: float = C_SCALE) -> float:
    return (c_visit + n_max) * c_scale * q_rescaled


def gumbel_scores(logits: np.ndarray, q: Dict[int, float], n_max: float) -> Dict[int, float]:
    """logit(a) + sigma(q̃(a)) for every action in ``q``."""
    qs = rescale(q)
    return {a: float(logits[a]) + sigma(qs[a], n_max) for a in q}


def pick(score: Dict[int, float], order: Sequence[int]) -> int:
    """The argmax of ``score`` over the actions it covers, ties to the earliest in ``order``."""
    cand = [a for a in order if a in score]
    m = max(score[a] for a in cand)
    return next(a for a in cand if score[a] >= m - TIE)


def softmax_masked(logits: np.ndarray, legal: Sequence[int]) -> np.ndarray:
    out = np.zeros(len(logits), dtype=np.float64)
    lg = np.array([float(logits[a]) for a in legal])
    z = np.exp(lg - lg.max())
    out[list(legal)] = z / z.sum()
    return out


def completed_q(logits: np.ndarray, legal: Sequence[int], q: Dict[int, float], visits: Dict[int, int],
                value: float) -> Dict[int, float]:
    """Every legal action's q, the unvisited ones completed by the MIXED value
    v_mix = (value + Σ_b N(b) · Σ_visited π q / Σ_visited π) / (1 + Σ_b N(b))  (mctx)."""
    pi = softmax_masked(logits, legal)
    vis = [a for a in legal if visits.get(a, 0) > 0]
    n_sum = float(sum(visits.get(a, 0) for a in legal))
    if vis:
        pm = sum(pi[a] for a in vis)
        wq = sum(pi[a] * q[a] for a in vis) / pm if pm > 0 else float(np.mean([q[a] for a in vis]))
        v_mix = (value + n_sum * wq) / (1.0 + n_sum)
    else:
        v_mix = value
    return {int(a): (q[a] if visits.get(a, 0) > 0 else v_mix) for a in legal}


def improved_policy(logits: np.ndarray, legal: Sequence[int], q: Dict[int, float], visits: Dict[int, int],
                    value: float) -> np.ndarray:
    """Gumbel MuZero's policy-improvement TARGET: softmax(logits + sigma(completed q̃)) over the legal
    actions, an 11-vector (0 on illegal actions)."""
    cq = completed_q(logits, legal, q, visits, value)
    n_max = max([visits.get(a, 0) for a in legal] + [0])
    g = gumbel_scores(logits, cq, n_max)
    z = np.array([g[a] for a in legal])
    z = np.exp(z - z.max())
    out = np.zeros(len(logits), dtype=np.float64)
    out[list(legal)] = z / z.sum()
    return out


@dataclass
class SequentialHalving:
    """The playout allocator. ``budget`` playouts over the policy's top-``m`` (``m=None`` = all legal),
    ⌈log₂ m⌉ rounds; in a round each survivor gets max(1, ⌊budget / (R·|S|)⌋) NEW samples, capped at
    ``cap`` per action; after a round the top ⌈|S|/2⌉ by the GUMBEL score survive.

    Use: ``while not sh.done: req = sh.next_round(); …play…; sh.observe(a, values)``; then ``sh.pick()``.
    The offline pilot's ``score.seq_halving`` is this rule over pre-played samples (pinned equal by
    ``rules_test``)."""

    logits: np.ndarray
    legal: Sequence[int]
    budget: int
    m: Optional[int] = 4
    cap: int = 16
    order: List[int] = field(init=False)
    survivors: List[int] = field(init=False)
    samples: Dict[int, List[float]] = field(init=False)
    rounds: int = field(init=False)
    spent: int = 0
    capped: bool = False
    done: bool = False

    def __post_init__(self):
        self.order = policy_order(self.logits, self.legal)
        k = len(self.order) if self.m is None else min(self.m, len(self.order))
        self.survivors = list(self.order[:k])
        self.rounds = max(1, math.ceil(math.log2(len(self.survivors)))) if len(self.survivors) > 1 else 1
        self.samples = {a: [] for a in self.order}
        self._pending: Dict[int, int] = {}

    def next_round(self) -> Dict[int, int]:
        """{action: number of NEW samples} for this round."""
        if self.done:
            return {}
        per = max(1, self.budget // (self.rounds * len(self.survivors)))
        req = {}
        for a in self.survivors:
            add = min(per, self.cap - len(self.samples[a]))
            self.capped |= add < per
            if add > 0:
                req[a] = add
        self._pending = dict(req)
        self.spent += sum(req.values())
        if not req:                          # every survivor at its cap: halve on what there is
            self._halve()
        return req

    def observe(self, action: int, values: Sequence[float]) -> None:
        self.samples[action].extend(float(v) for v in values)
        self._pending.pop(action, None)
        if not self._pending:
            self._halve()

    def means(self) -> Dict[int, float]:
        return {a: float(np.mean(v)) for a, v in self.samples.items() if v}

    def visits(self) -> Dict[int, int]:
        return {a: len(v) for a, v in self.samples.items()}

    def _halve(self) -> None:
        if len(self.survivors) == 1:
            self.done = True
            return
        mean = {a: (float(np.mean(self.samples[a])) if self.samples[a] else 0.0) for a in self.survivors}
        n_max = max(len(v) for v in self.samples.values())
        g = gumbel_scores(self.logits, mean, n_max)
        keep = math.ceil(len(self.survivors) / 2)
        self.survivors = sorted(self.survivors, key=lambda a: (-g[a], self.order.index(a)))[:keep]
        if keep == 1:
            self.done = True

    def pick(self) -> int:
        return self.survivors[0]
