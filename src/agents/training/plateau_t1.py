"""THE PLATEAU TEST'S TIER 1 — the owner-registered head-to-head GSPRT, as a pure function of a request's rows
(``designs/endstate/design_evaluation.md`` §8.1; unit U9a, 2026-10-06).

**The question.** Once per check (every Δ = 10M steps), the newest node plays the node from W = 50M steps back
(≈ 7 GPU-hours at N = 256; lag 5 of the cycle monitor) on MIRRORED pairs, and a pentanomial GSPRT asks

    H0: the newest node's mean per-game score vs the W-back node  ≤ 0.50
    H1:                                                            ≥ 0.52     (2 Elo / GPU-h × 7 h ≈ 14 Elo ≈ 2 pp)

at α = β = 0.05, Wald bounds, checked only at 40-pair batch boundaries, capped at 6,000 pairs. The likelihood is
``sprt.llr`` (the Fishtest pentanomial GSPRT the promotion test uses), so the operating characteristics are §8.1's
table, simulated with that code.

**Verdicts** (the closed vocabulary :data:`VERDICTS`):

* ``GAIN``  — H1 accepted: the run still pays; keep training;
* ``FLAT``  — H0 accepted;
* ``UNDECIDED`` — the cap ran out. Unlike promotion the cap does NOT default to H0, so an undecided check never
  feeds a plateau call;
* ``INCONCLUSIVE`` — more than 25 % of the cell's attempted games aborted (§9.1); never a verdict about strength;
* ``CONTINUE`` — the test is still running (never written as a decision).

**Determinism (standing rule 8).** A verdict is a pure function of (the rows, the rule). The rows are folded in
BATCH ORDER, whatever order the ledger holds them in; an LLR within :data:`ROUNDING_BAND` (1e-9) of a bound is NOT a
crossing (§9.1); the first crossing STOPS the test, and a batch recorded after it (a second driver's race) is
ignored, never folded. The rule's every constant is spelled out in the decision row's ``rule`` string
(:meth:`Tier1Rule.rule_string`), so ``python -m main.eval_ledger verify`` re-derives the verdict from the rows alone
(:func:`rederive`, registered in ``eval_ledger.audit.RULES``).

**The run-level read** (:func:`tier1_status`): ``TIER1_PLATEAU`` iff the newest finally-decided check and the check
one Δ before it are both FLAT (§8.3's two consecutive checks). This is TIER 1 ONLY: a PLATEAU in §8.3's sense also
needs the cycle monitor NONE and the panel FLAT (Tier 2, U5 / U9), which are not built — every reader prints that.

**The check plan** (:func:`plan_checks`): a NODE is the run's periodic checkpoint at a multiple of Δ — the first one
at or above it, within :attr:`CheckPlan.slack_steps` (checkpoints land a few rollouts past the boundary). A check at
grid step g compares node(g) with node(g − W); a missing node makes the check MISSING, never a substitute.

Pure stdlib (+ ``sprt`` and ``mirrored_pairs``): the ledger's audit imports it lazily."""
from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from agents.training import sprt as SP
from agents.training.mirrored_pairs import N_CATEGORIES

RULE_NAME = "plateau_t1_gsprt"
RULE_VERSION = "gen3_plateau_t1_v1"
#: The ledger's decision kind, request kind and purpose for a Tier-1 check.
DECISION_KIND = "plateau"
REQUEST_KIND = "plateau_t1"
PURPOSE = "plateau"

GAIN, FLAT, UNDECIDED, INCONCLUSIVE, CONTINUE = "GAIN", "FLAT", "UNDECIDED", "INCONCLUSIVE", "CONTINUE"
#: A check's FINAL verdicts (what a decision row may say).
VERDICTS = (GAIN, FLAT, UNDECIDED, INCONCLUSIVE)
#: Why a test stopped.
BY_UPPER, BY_LOWER, BY_CAP, BY_ABORTS = "upper_bound", "lower_bound", "cap", "aborts"

#: §9.1: an LLR within this distance of a bound is NOT a crossing.
ROUNDING_BAND = 1e-9

# the run-level Tier-1 status vocabulary (tier1_status)
NOT_YET = "NOT_YET"                 # no check has a final verdict
CLIMBING = "CLIMBING"               # the newest final verdict is GAIN
FLAT_ONCE = "FLAT_ONCE"             # the newest final verdict is FLAT, the check one Δ before it is not
TIER1_PLATEAU = "TIER1_PLATEAU"     # the newest final verdict is FLAT, and so is the check one Δ before it
CONTINUE_STATUS = "CONTINUE"        # the newest final verdict is UNDECIDED or INCONCLUSIVE: re-check next interval
STATUSES = (NOT_YET, CLIMBING, FLAT_ONCE, TIER1_PLATEAU, CONTINUE_STATUS)

#: Printed beside every status: what Tier 1 alone does not say.
TIER2_CAVEAT = ("Tier 1 ONLY: a PLATEAU (design_evaluation.md §8.3) also needs the cycle monitor NONE and the panel "
                "FLAT at the same two checks — Tier 2 (U5) and the panel are NOT BUILT, so this is not a plateau "
                "declaration")


class Tier1RowsError(ValueError):
    """Rows that are not ONE Tier-1 test's: two matchups, two regimes, a gap in the batch sequence, a batch of the
    wrong length, or a duplicated batch. A data error, never a verdict."""


# ------------------------------------------------------------------------------------------------ the rule
@dataclass(frozen=True)
class Tier1Rule:
    """The test's DECLARED constants (§8.1). The defaults ARE the registered rule; anything else is a different rule
    and says so in its ``rule`` string."""

    p0: float = 0.50
    p1: float = 0.52
    alpha: float = 0.05
    beta: float = 0.05
    batch_pairs: int = 40
    min_pairs: int = 40
    max_pairs: int = 6000
    abort_limit: float = 0.25

    def __post_init__(self) -> None:
        if not 0.0 < self.p0 < self.p1 < 1.0:
            raise ValueError(f"Tier 1 needs 0 < p0 < p1 < 1, got {self.p0}, {self.p1}")
        if not (0.0 < self.alpha < 0.5 and 0.0 < self.beta < 0.5):
            raise ValueError("Tier 1 needs 0 < alpha, beta < 0.5")
        if self.batch_pairs < 1 or self.min_pairs < 0 or self.max_pairs < self.batch_pairs:
            raise ValueError("Tier 1 needs batch_pairs >= 1, min_pairs >= 0, max_pairs >= batch_pairs")
        if self.max_pairs % self.batch_pairs:
            raise ValueError("Tier 1's cap must be a whole number of batches (every batch is full)")
        if not 0.0 < self.abort_limit < 1.0:
            raise ValueError("Tier 1's abort limit is a fraction in (0, 1)")

    @property
    def bounds(self) -> Tuple[float, float]:
        return SP.bounds(self.alpha, self.beta)

    @property
    def registered(self) -> bool:
        return self == REGISTERED

    def rule_string(self) -> str:
        """The decision row's ``rule``: the name and EVERY constant, so a verdict is re-derivable from it alone."""
        return (f"{RULE_NAME}(p0={self.p0!r},p1={self.p1!r},alpha={self.alpha!r},beta={self.beta!r},"
                f"batch_pairs={self.batch_pairs},min_pairs={self.min_pairs},max_pairs={self.max_pairs},"
                f"abort_limit={self.abort_limit!r},bounds=wald,band={ROUNDING_BAND!r})")

    @classmethod
    def parse(cls, s: str) -> "Tier1Rule":
        m = re.fullmatch(re.escape(RULE_NAME) + r"\((.*)\)", s)
        if m is None:
            raise ValueError(f"not a {RULE_NAME} rule: {s!r}")
        kv = dict(part.split("=", 1) for part in m.group(1).split(","))
        if kv.pop("bounds", None) != "wald" or float(kv.pop("band", "nan")) != ROUNDING_BAND:
            raise ValueError(f"{s!r}: only Wald bounds at band {ROUNDING_BAND} are implemented")
        ints = {"batch_pairs", "min_pairs", "max_pairs"}
        typed: Dict[str, Any] = {k: (int(v) if k in ints else float(v)) for k, v in kv.items()}
        out = cls(**typed)
        if out.rule_string() != s:
            raise ValueError(f"{s!r} does not round-trip ({out.rule_string()!r})")
        return out

    def to_json(self) -> Dict[str, Any]:
        lo, hi = self.bounds
        return {"rule": RULE_NAME, "rule_version": RULE_VERSION, **asdict(self),
                "llr_bounds": [round(lo, 9), round(hi, 9)], "bounds_mode": "wald", "rounding_band": ROUNDING_BAND,
                "registered": self.registered, "score_space": "the newest node's mean per-game score, a draw = 1/2",
                "likelihood": "pentanomial GSPRT over mirrored pairs (sprt.llr)"}


#: The registered rule (§8.1).
REGISTERED = Tier1Rule()


# ------------------------------------------------------------------------------------------------ the fold
@dataclass
class Tier1Result:
    """One Tier-1 test folded over its rows."""

    rule: Tier1Rule
    verdict: str = CONTINUE
    reason: Optional[str] = None
    n_pairs: int = 0
    pair_counts: List[int] = field(default_factory=lambda: [0] * N_CATEGORIES)
    llr: float = 0.0
    batches_used: int = 0
    stop_batch: Optional[int] = None
    surplus_batches: List[int] = field(default_factory=list)
    attempted_games: int = 0
    aborted_games: int = 0
    player: Optional[str] = None
    opponent: Optional[str] = None
    trail: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def final(self) -> bool:
        return self.verdict in VERDICTS

    def to_json(self) -> Dict[str, Any]:
        lo, hi = self.rule.bounds
        tot = sum(self.pair_counts)
        # the TEST's own score is a stopped (selected) number: provenance only, never a strength estimate
        score = (sum(c * s for c, s in zip(self.pair_counts, SP.SCORES)) / tot) if tot else None
        return {"verdict": self.verdict, "reason": self.reason, "n_pairs": self.n_pairs,
                "pair_counts": list(self.pair_counts), "llr": round(self.llr, 9),
                "llr_bounds": [round(lo, 9), round(hi, 9)], "batches_used": self.batches_used,
                "stop_batch": self.stop_batch, "surplus_batches": list(self.surplus_batches),
                "attempted_games": self.attempted_games, "aborted_games": self.aborted_games,
                "test_score_selected": None if score is None else round(score, 6),
                "player": self.player, "opponent": self.opponent, "rule": self.rule.rule_string()}


def _side(p: Mapping[str, Any]) -> str:
    return str(p.get("sha256") or f"id:{p.get('id')}")


def ordered_batches(rows: Sequence[Mapping[str, Any]], rule: Tier1Rule) -> List[Mapping[str, Any]]:
    """The rows as ONE test's batch sequence 0, 1, 2, … — or :class:`Tier1RowsError`."""
    if not rows:
        return []
    keys = {(_side(r["player"]), _side(r["opponent"]), r["regime"]["regime_id"], (r.get("request") or {}).get("id"))
            for r in rows}
    if len(keys) != 1:
        raise Tier1RowsError(f"the rows are {len(keys)} (player, opponent, regime, request) combinations, not one "
                             f"test: {sorted(map(str, keys))[:4]}")
    by: Dict[int, Mapping[str, Any]] = {}
    for r in rows:
        b = int((r.get("request") or {}).get("batch", -1))
        if b < 0:
            raise Tier1RowsError(f"row {r.get('row_id')} carries no request batch index")
        if b in by:
            raise Tier1RowsError(f"batch {b} is recorded twice ({by[b].get('row_id')}, {r.get('row_id')})")
        by[b] = r
    order = sorted(by)
    if order != list(range(len(order))):
        missing = sorted(set(range(order[-1] + 1)) - set(order))
        raise Tier1RowsError(f"the batch sequence has a gap: batches {missing[:6]} are missing (a resumed driver "
                             "plays them first)")
    for b in order:
        n = int(by[b]["pairs"]["n_pairs"]) + int(by[b]["pairs"].get("voided", 0))
        if n != rule.batch_pairs:
            raise Tier1RowsError(f"batch {b} holds {n} pairs; the rule's batch is {rule.batch_pairs}")
    return [by[b] for b in order]


def _crossing(llr: float, lo: float, hi: float) -> Optional[str]:
    if llr - hi > ROUNDING_BAND:
        return BY_UPPER
    if lo - llr > ROUNDING_BAND:
        return BY_LOWER
    return None


def evaluate(rows: Sequence[Mapping[str, Any]], rule: Tier1Rule = REGISTERED) -> Tier1Result:
    """Fold one test's rows in batch order and stop at the first decision (module docstring)."""
    batches = ordered_batches(rows, rule)
    res = Tier1Result(rule=rule)
    if batches:
        res.player, res.opponent = _side(batches[0]["player"]), _side(batches[0]["opponent"])
    lo, hi = rule.bounds
    for b, r in enumerate(batches):
        pc = [int(x) for x in r["pairs"]["counts"]]
        if len(pc) != N_CATEGORIES:
            raise Tier1RowsError(f"batch {b}: the pair counts are not a pentanomial ({pc})")
        res.pair_counts = [a + x for a, x in zip(res.pair_counts, pc)]
        res.n_pairs = sum(res.pair_counts)
        c = r["counts"]
        res.aborted_games += int(c.get("aborted", 0))
        res.attempted_games += int(c["w"]) + int(c["l"]) + int(c["d"]) + int(c.get("aborted", 0))
        res.llr = SP.llr(res.pair_counts, rule.p0, rule.p1)
        if not math.isfinite(res.llr):
            raise Tier1RowsError(f"batch {b}: a non-finite LLR {res.llr} (pair counts {res.pair_counts})")
        res.batches_used = b + 1
        res.trail.append({"batch": b, "n_pairs": res.n_pairs, "llr": round(res.llr, 9)})
        how = _crossing(res.llr, lo, hi) if res.n_pairs >= rule.min_pairs else None
        if how is None and (b + 1) * rule.batch_pairs >= rule.max_pairs:
            how = BY_CAP
        if how is None:
            continue
        res.stop_batch = b
        res.surplus_batches = list(range(b + 1, len(batches)))
        if res.attempted_games and res.aborted_games > rule.abort_limit * res.attempted_games:
            res.verdict, res.reason = INCONCLUSIVE, BY_ABORTS
        else:
            res.verdict = {BY_UPPER: GAIN, BY_LOWER: FLAT, BY_CAP: UNDECIDED}[how]
            res.reason = how
        break
    return res


def rederive(decision: Mapping[str, Any], rows: List[Dict[str, Any]]) -> Optional[str]:
    """``eval_ledger.audit.RULES["plateau"]``: the verdict a Tier-1 decision's rows give under its OWN ``rule``
    string, ``None`` for a plateau decision of another rule (a later two-tier decision), and a ``REFUSED(...)``
    string — which can never equal a verdict — when the rows are not one test's."""
    s = str(decision.get("rule", ""))
    if not s.startswith(RULE_NAME + "("):
        return None
    try:
        return evaluate(rows, Tier1Rule.parse(s)).verdict
    except (Tier1RowsError, ValueError, KeyError) as e:
        return f"REFUSED({type(e).__name__}: {e})"


# ------------------------------------------------------------------------------------------------ the plan
@dataclass(frozen=True)
class CheckPlan:
    """WHEN a run is checked and AGAINST WHAT (§8.1): a check at every multiple of ``delta_steps``, the newest
    node against the node ``lag`` checks back (W = lag × Δ)."""

    delta_steps: int = 10_000_000
    lag: int = 5
    slack_steps: int = 500_000

    def __post_init__(self) -> None:
        if self.delta_steps < 1 or self.lag < 1:
            raise ValueError("a check plan needs delta_steps >= 1 and lag >= 1")
        if not 0 < self.slack_steps < self.delta_steps:
            raise ValueError("a node's slack must be positive and below the check spacing (one node per grid step)")

    @property
    def w_steps(self) -> int:
        return self.lag * self.delta_steps

    def to_json(self) -> Dict[str, Any]:
        return {**asdict(self), "w_steps": self.w_steps,
                "node_rule": "the first periodic checkpoint at or above each multiple of delta_steps, below it + "
                             "slack_steps"}


#: The registered plan: Δ = 10M (§2.5's spacing, aligned with the check), W = 50M (≈ 7 GPU-h at N = 256).
DEFAULT_PLAN = CheckPlan()


@dataclass(frozen=True)
class Check:
    grid: int                         # the check's step on the Δ grid
    newest: Optional[int]             # node(grid)'s checkpoint step, or None (missing)
    wback: Optional[int]              # node(grid - W)'s checkpoint step, or None (missing)

    @property
    def playable(self) -> bool:
        return self.newest is not None and self.wback is not None


def node_at(grid: int, steps: Sequence[int], slack: int) -> Optional[int]:
    """The node of grid step ``grid``: the smallest checkpoint step in ``[grid, grid + slack)``, or ``None``."""
    cands = [s for s in steps if grid <= s < grid + slack]
    return min(cands) if cands else None


def plan_checks(steps: Iterable[int], plan: CheckPlan = DEFAULT_PLAN) -> List[Check]:
    """Every check a run's checkpoints reach, oldest first: grid steps g = k·Δ with g ≥ W, up to the newest
    checkpoint. A check whose node or W-back node is missing is listed (``playable`` False), never substituted."""
    ss = sorted(set(int(s) for s in steps))
    if not ss:
        return []
    out: List[Check] = []
    k = plan.lag
    while k * plan.delta_steps <= ss[-1]:
        g = k * plan.delta_steps
        out.append(Check(grid=g, newest=node_at(g, ss, plan.slack_steps),
                         wback=node_at(g - plan.w_steps, ss, plan.slack_steps)))
        k += 1
    return out


def request_id(run: str, grid: int, plan: CheckPlan = DEFAULT_PLAN) -> str:
    """ONE request per (run, check step, W): a re-run resumes it; another W is another question."""
    return f"{REQUEST_KIND}:{run}:{int(grid)}:w{plan.w_steps}"


# ------------------------------------------------------------------------------------------------ the status
def tier1_status(checks: Sequence[Tuple[int, str]], delta_steps: int) -> Dict[str, Any]:
    """The run's Tier-1 status from its checks' states ``[(grid, state)]`` (a final verdict, ``CONTINUE``, or any
    not-yet-played label): see the module docstring. ``consecutive`` means grid steps exactly Δ apart."""
    by = {int(g): s for g, s in checks}
    finals = sorted(g for g, s in by.items() if s in VERDICTS)
    later = sorted(g for g, s in by.items() if s not in VERDICTS and (not finals or g > finals[-1]))
    if not finals:
        return {"status": NOT_YET, "as_of_check": None, "flat_checks": [], "pending_after": later}
    g = finals[-1]
    v = by[g]
    flat: List[int] = []
    if v == GAIN:
        st = CLIMBING
    elif v == FLAT:
        prev = by.get(g - delta_steps)
        st = TIER1_PLATEAU if prev == FLAT else FLAT_ONCE
        flat = [g - delta_steps, g] if prev == FLAT else [g]
    else:
        st = CONTINUE_STATUS
    return {"status": st, "as_of_check": g, "verdict": v, "flat_checks": flat, "pending_after": later}

