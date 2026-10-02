"""SPRT PROMOTION — a sequential test per candidate on MIRRORED PAIRS (T6, ``gen3_sprt_promotion_v1``).

**What it replaces.** Without it a snapshot is promoted the first cycle its ``win_rate_vs_pool`` crosses
``--promote-threshold``. Peeking at a noisy rate every cycle and acting on the first crossing is optional
stopping: the promoted snapshots are, on average, weaker than their promotion score (a winner's curse).

**The rule (Stockfish / Fishtest).** Per CANDIDATE, a generalized sequential probability ratio test of

    H0: the candidate's mean per-game score vs the pool  ≤ P0 = 0.50
    H1: the candidate's mean per-game score vs the pool  ≥ P1 = 0.55

at α = β = 0.05 — LLR bounds ``ln(β/(1−α)) = −2.944`` and ``ln((1−β)/α) = +2.944`` — over MIRRORED
PAIRS. The likelihood is the PENTANOMIAL over pair outcomes (``mirrored_pairs``): each pair is one draw
from a five-category distribution, and the GSPRT compares the MAXIMUM-LIKELIHOOD pentanomials whose mean
is exactly P0 and exactly P1 given the observed one (Van den Bergh, "A practical introduction to the
GSPRT", the Fishtest form). The pair, never the game, is the unit — the two games of a pair share their
teams and their dice.

**Truncation.** The test plays batches until the LLR leaves ``(lower, upper)``. A DECLARED cap
(``max_pairs``) bounds its cost; reaching it without a decision is a REJECT (the candidate is not
promoted). The cap is chosen from the simulated p95 of pairs-to-decide
(``designs/research_state/measurements/sprt_promotion/``), so the truncation costs only a little power.

**The three discipline rules** (``designs/training/eval_and_rating.md`` → *SPRT promotion*):
selection games never pool into the decision (the test plays its OWN fresh pairs on a seed namespace
disjoint from the cycle's), a failed test is never re-run (one verdict per candidate, durable), and the
pool is FROZEN at the test's start.

numpy-free on purpose (the eval core and the offline simulator both call it, and the simulator vectorizes
:func:`llr_many` itself).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

from agents.training.mirrored_pairs import N_CATEGORIES, n_pairs

SCHEMA = "gen3_sprt_promotion_v1"

P0, P1 = 0.50, 0.55
ALPHA = BETA = 0.05

#: The per-pair normalized scores of the five categories (half-points / 4).
SCORES = tuple(i / 4.0 for i in range(N_CATEGORIES))

#: A zero count is replaced by this before the MLE (Fishtest's regularization): the constrained MLE then
#: always exists, and a handful of all-win pairs cannot produce an infinite LLR.
REGULARIZE = 1e-3

ACCEPT, REJECT, CONTINUE = "accept", "reject", "continue"

#: The DECLARED schedule, chosen by the pre-declared rule of the Monte Carlo
#: (`designs/research_state/measurements/sprt_promotion/results.{json,md}`, 20,000 runs per cell): a check
#: every 40 mirrored pairs; the first decision at 40 (a larger minimum changed neither error rate beyond
#: simulation noise and cost pairs); the cap = the untruncated test's worst-cell p95, 1,680 pairs (3,360
#: games); plain WALD bounds (Siegmund-corrected bounds saved ~21% of pairs but held the error rates at
#: nominal only within simulation noise — false promotion 5.07% at the first-batch minimum). Every one is
#: stamped on every verdict, and `sprt_test.py` pins them to the committed results.
BATCH_PAIRS_DEFAULT = 40
MIN_PAIRS_DEFAULT = 40
MAX_PAIRS_DEFAULT = 1680
BOUNDS_MODE_DEFAULT = "wald"
#: Why a REJECT was reached: the LLR crossed the lower bound, or the declared cap ran out.
BY_BOUND, BY_CAP = "bound", "cap"


def bounds(alpha: float = ALPHA, beta: float = BETA) -> tuple:
    """``(lower, upper)`` LLR bounds — Wald's ``ln(β/(1−α))`` and ``ln((1−β)/α)``."""
    return math.log(beta / (1.0 - alpha)), math.log((1.0 - beta) / alpha)


#: Bisection steps for λ. Fixed, so the result is a deterministic function of the counts; 200 halvings
#: take the bracket far below float64 resolution (the constraint residual is pinned by a test).
LAMBDA_ITERS = 200

#: Siegmund's overshoot constant ρ = −ζ(½)/√(2π) ≈ 0.5826: the expected overshoot of a Gaussian random
#: walk over a boundary, in units of its per-STEP standard deviation (Siegmund, *Sequential Analysis*,
#: 1985, Thm 10.13 — the correction Fishtest-style analytics use for a test checked in batches).
SIEGMUND_RHO = 0.5826


def mle_lambda(p: Sequence[float], mu: float) -> float:
    """The Lagrange multiplier of the pentanomial MLE constrained to mean ``mu`` (Van den Bergh): the
    root of ``f(λ) = Σ pᵢ (xᵢ − μ) / (1 + λ (xᵢ − μ)) = 0`` on the interval where every
    ``1 + λ (xᵢ − μ) > 0``. ``f`` is strictly decreasing there, so BISECTION finds it — deterministically,
    in exactly :data:`LAMBDA_ITERS` halvings."""
    xs = SCORES
    lo = -1.0 / (max(xs) - mu) + 1e-12
    hi = 1.0 / (mu - min(xs)) - 1e-12
    for _ in range(LAMBDA_ITERS):
        mid = 0.5 * (lo + hi)
        if constraint_residual(p, mu, mid) > 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def constraint_residual(p: Sequence[float], mu: float, lam: float) -> float:
    """``f(λ)`` above — zero exactly at the constrained MLE."""
    return sum(pi * (x - mu) / (1.0 + lam * (x - mu)) for pi, x in zip(p, SCORES))


def _empirical(counts: Sequence[int]) -> List[float]:
    reg = [float(c) if c > 0 else REGULARIZE for c in counts]
    tot = sum(reg)
    return [c / tot for c in reg]


def _increments(p: Sequence[float], p0: float, p1: float) -> List[float]:
    """The per-PAIR LLR increment of each category: ``ln(q1ᵢ / q0ᵢ)``."""
    l0, l1 = mle_lambda(p, p0), mle_lambda(p, p1)
    return [math.log(1.0 + l0 * (x - p0)) - math.log(1.0 + l1 * (x - p1)) for x in SCORES]


def llr(counts: Sequence[int], p0: float = P0, p1: float = P1) -> float:
    """The GSPRT log-likelihood ratio of H1 (mean ``p1``) over H0 (mean ``p0``) for pentanomial ``counts``.

    ``N · Σ p̂ᵢ ln(q1ᵢ / q0ᵢ)`` where ``qμ = p̂ / (1 + λμ (x − μ))`` is the maximum-likelihood pentanomial
    with mean ``μ``. Zero pairs give 0."""
    n = n_pairs(counts)
    if n <= 0:
        return 0.0
    p = _empirical(counts)
    return n * sum(pi * d for pi, d in zip(p, _increments(p, p0, p1)))


def llr_increment_var(counts: Sequence[int], p0: float = P0, p1: float = P1) -> float:
    """The variance of ONE pair's LLR increment under the observed pentanomial — what Siegmund's
    overshoot correction scales by. 0 with no pairs."""
    if n_pairs(counts) <= 0:
        return 0.0
    p = _empirical(counts)
    d = _increments(p, p0, p1)
    m = sum(pi * x for pi, x in zip(p, d))
    return max(0.0, sum(pi * (x - m) ** 2 for pi, x in zip(p, d)))


def decision_bounds(cfg: "SprtConfig", counts: Sequence[int]) -> tuple:
    """The bounds a CHECK compares the LLR with. ``"wald"``: Wald's ``(ln β/(1−α), ln (1−β)/α)``.
    ``"siegmund"``: both moved INWARD by the expected overshoot ``ρ · σ_check``, where ``σ_check`` is the
    standard deviation of the LLR's movement between two checks (``batch_pairs`` pairs) — the test is
    checked in batches, so it crosses a bound by a jump, and Wald's bounds then hold the error rates
    BELOW nominal while spending extra pairs."""
    lo, hi = bounds(cfg.alpha, cfg.beta)
    if cfg.bounds_mode == "siegmund":
        shift = SIEGMUND_RHO * math.sqrt(cfg.batch_pairs * llr_increment_var(counts, cfg.p0, cfg.p1))
        lo, hi = min(lo + shift, 0.0), max(hi - shift, 0.0)
    return lo, hi


@dataclass
class SprtConfig:
    """The test's DECLARED parameters — stamped on every verdict, so a verdict is never read without them."""

    p0: float = P0
    p1: float = P1
    alpha: float = ALPHA
    beta: float = BETA
    max_pairs: int = MAX_PAIRS_DEFAULT
    batch_pairs: int = BATCH_PAIRS_DEFAULT
    min_pairs: int = MIN_PAIRS_DEFAULT
    bounds_mode: str = BOUNDS_MODE_DEFAULT

    def __post_init__(self) -> None:
        if not 0.0 < self.p0 < self.p1 < 1.0:
            raise ValueError(f"SPRT needs 0 < p0 < p1 < 1, got {self.p0}, {self.p1}")
        if self.batch_pairs < 1 or self.max_pairs < self.batch_pairs or self.min_pairs < 0:
            raise ValueError("SPRT needs batch_pairs >= 1, max_pairs >= batch_pairs, min_pairs >= 0")
        if self.bounds_mode not in ("wald", "siegmund"):
            raise ValueError(f"unknown SPRT bounds mode {self.bounds_mode!r}")

    def to_json(self) -> dict:
        lo, hi = bounds(self.alpha, self.beta)
        return {"schema": SCHEMA, "p0": self.p0, "p1": self.p1, "alpha": self.alpha, "beta": self.beta,
                "llr_bounds": [round(lo, 6), round(hi, 6)], "bounds_mode": self.bounds_mode,
                "max_pairs": int(self.max_pairs), "batch_pairs": int(self.batch_pairs),
                "min_pairs": int(self.min_pairs), "score_space": "mean per-game score, a draw = 1/2",
                "likelihood": "pentanomial GSPRT over mirrored pairs (Van den Bergh constrained MLE)"}


@dataclass
class SprtState:
    """One candidate's test in flight: the pooled pentanomial, the batch trail, the verdict."""

    cfg: SprtConfig
    counts: List[int] = field(default_factory=lambda: [0] * N_CATEGORIES)
    trail: List[dict] = field(default_factory=list)
    verdict: str = CONTINUE
    reason: Optional[str] = None

    @property
    def n_pairs(self) -> int:
        return n_pairs(self.counts)

    @property
    def llr(self) -> float:
        return llr(self.counts, self.cfg.p0, self.cfg.p1)

    def add_batch(self, batch_counts: Sequence[int]) -> str:
        """Fold one batch of pairs in and decide. Checks happen ONLY at batch boundaries — the simulated
        operating characteristics are for exactly this schedule."""
        if self.verdict != CONTINUE:
            raise RuntimeError(f"the SPRT already decided ({self.verdict}); a decided test takes no more games")
        if len(batch_counts) != N_CATEGORIES:
            raise ValueError("a batch is a pentanomial vector")
        self.counts = [a + int(b) for a, b in zip(self.counts, batch_counts)]
        lo, hi = decision_bounds(self.cfg, self.counts)
        v = self.llr
        if self.n_pairs >= self.cfg.min_pairs and v >= hi:
            self.verdict, self.reason = ACCEPT, BY_BOUND
        elif self.n_pairs >= self.cfg.min_pairs and v <= lo:
            self.verdict, self.reason = REJECT, BY_BOUND
        elif self.n_pairs >= self.cfg.max_pairs:
            self.verdict, self.reason = REJECT, BY_CAP
        self.trail.append({"n_pairs": self.n_pairs, "llr": round(v, 6), "bounds": [round(lo, 6), round(hi, 6)],
                           "counts": list(self.counts)})
        return self.verdict

    def next_batch_pairs(self) -> int:
        """How many pairs the next batch plays — the declared batch, clipped at the cap."""
        return max(0, min(int(self.cfg.batch_pairs), int(self.cfg.max_pairs) - self.n_pairs))

    def to_json(self) -> dict:
        from agents.training.mirrored_pairs import pair_score_ci

        ci = pair_score_ci(self.counts)
        # 🚨 the TEST's own score is a SELECTED number (the sample that decided): provenance only — a
        # promoted snapshot's strength is read from later, unselected games (`ladder.json`).
        return {"config": self.cfg.to_json(), "verdict": self.verdict, "reason": self.reason,
                "n_pairs": self.n_pairs, "pair_counts": list(self.counts), "llr": round(self.llr, 6),
                "test_score_selected": None if ci is None else round(ci[0], 6),
                "test_score_ci95_selected": None if ci is None else [round(ci[1], 6), round(ci[2], 6)],
                "trail": list(self.trail)}
