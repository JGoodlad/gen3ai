"""X35's REGISTERED readout (``designs/endstate/design_compute_efficiency.md`` §5.6.4-§5.6.6): the population-score
estimator and the deterministic decision function. Pure functions; no game is played here.

THE POPULATION SCORE. S(final) is the equal-weight mean, over the frozen reference population R's members, of the
final's mirrored-pair score against that member (pp; a draw = 1/2), each cell >= 500 completed pairs on
``main.h2h play-many``, greedy vs greedy (§5.6.4). :func:`population_score` builds it from ``eval_ledger`` cells: a
missing cell, a cell listed twice, a non-OK cell, a cell under 500 pairs, or a final that IS a member, is an
INCONCLUSIVE reason, never a score.

R1, THE PRIMARY. Δ_pop = mean over b-ON seed finals of S − mean over b-OFF seed finals of S (pp). Seed finals are
the units; the interval is Welch's two-sample t with the Satterthwaite df, TWO-sided at 97.5 % per look (two looks,
Bonferroni). Classes at the ±2.0 pp bar: GAIN (the interval wholly above +2.0), NULL (wholly inside [−2.0, +2.0]),
LOSS (wholly below −2.0), else UNDECIDED. Look 1 is 3 seeds per side; an UNDECIDED with Δ̂ > 0 CONTINUES to look 2
(5 per side), which is final.

R2, THE CO-PRIMARY for A3 / A4 / A6. gap = P(a fresh best responder beats its target) − 0.5, so
gap(ON) − gap(OFF) = p_ON − p_OFF; "the gap FALLS" iff the Newcombe 95 % UPPER bound of that difference < 0.

THE GUARDS (each PASS / FAIL / INCONCLUSIVE, deterministically):
* G-F, the head-to-head frontier: ``main.h2h.cross.cross_stat`` on the square b-ON × b-OFF matrix (rows ON seeds,
  columns OFF seeds; the look's seed count); FAIL iff its two-sided 95 % upper bound < −2.0.
* G-E, exploitability: the R2 read; FAIL iff the Newcombe 95 % LOWER bound of gap(ON) − gap(OFF) > 0.
* G-C, the collapse battery (§5.6.5): B1 entropy, FAIL iff the battle-clustered 95 % interval of ON − OFF lies
  wholly below −0.05 nats (upper < −0.05); B2 the starved share, FAIL iff its lower bound > 0.
* G-K, the external anchor (§5.6.6): FAIL iff ON − OFF is detected below 0 (Newcombe 95 % upper < 0) on EITHER team
  set (home72, away), each pooled over the seed finals and each side >= :data:`G_K_MIN_GAMES` counted games.

THE VERDICT. ADOPT-CANDIDATE iff R1 is GAIN, or (A3 / A4 / A6) R1 is NULL and R2 says the gap FALLS -- and every
guard PASSes. TRADE: R1 qualifies but a guard FAILs. INCONCLUSIVE: an input is missing or invalid where the verdict
needs it. CONTINUE: look 1 sent R1 to look 2. Otherwise R1's class (NULL / LOSS / UNDECIDED). Nothing is adopted
by this rule alone: an ADOPT-CANDIDATE goes to the owner.

RULE 8 (standing rule 8). A statistic within :data:`RULE8_EPS` (1e-9) of a boundary it is compared against is NOT
a crossing of it, and the read records which boundary it touched (``near``). For R1 that makes the class UNDECIDED
(§5.6.4: "a bound within 1e-9 of a boundary counts as UNDECIDED"); for a guard it is not a FAIL; for R2 it is not a
fall; for the look-2 rule, Δ̂ within 1e-9 of 0 is not "> 0".

DECLARED HERE, BEFORE ANY REAL READ (choices §5.6.4 leaves open; each is a FINDING of T55):
* R1's seed sets must be EQUAL on the two sides (b-ON and b-OFF share seed i, §5.6.1) and of the look's size;
  anything else is INCONCLUSIVE. Welch does not pair them; the shared seeds only match the dice.
* Zero variance on BOTH sides leaves Welch undefined: INCONCLUSIVE, never a verdict.
* G-K's floor is :data:`G_K_MIN_GAMES` = 400 counted games per team set per side (§5.6.6 plans 420, "so away
  carries >= 400"); G-K's per-game success is a WIN (ties count as non-wins, the anchor read's ``win_rate``).
* A guard input that is absent is INCONCLUSIVE; it blocks only a verdict that needs it (R1 qualifying).
* The dose confound (§5.6.1, > 10 % between b-ON and b-OFF) is REPORTED beside the verdict, never decides it.
* Each Newcombe interval is the one its meter prints: R2 / G-E use ``best_response_gap.newcombe_diff_ci``
  (z = 1.96), G-K uses ``main.anchors.results.newcombe`` (z = 1.959963985). The two differ by ≈ 1e-5 at the
  bounds (the self-check proves that at one z they agree to 1e-12), so a bound inside that gap would read
  differently under the other meter's z -- the read quotes the interval of the meter that produced its counts.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from main.h2h.cross import CrossInputError, CrossStat, cross_stat

#: §5.1's bar, pp.
BAR_PP = 2.0
#: Standing rule 8.
RULE8_EPS = 1e-9
#: Two looks, Bonferroni over a 95 % family: a two-sided 97.5 % interval per look.
R1_CONFIDENCE = 0.975
#: Seeds per side at each look (§5.6.4).
LOOK_SEEDS: Dict[int, int] = {1: 3, 2: 5}
#: §5.6.4: 500 mirrored pairs (1,000 games) per (final, member) cell.
MIN_PAIRS = 500
#: The arms whose mechanism is exploitability (§5.6.4: R2 is their co-primary).
R2_ARMS = frozenset({"A3", "A4", "A6"})
#: Every arm the readout knows (C0 is the control, never read as an arm).
ARMS = frozenset({"A1", "A2", "A3", "A4", "A5", "A6"})
#: G-C's B1 bar, nats (§5.6.5).
G_C_B1_BAR_NATS = -0.05
#: G-K's floor, counted games per team set per side (module docstring).
G_K_MIN_GAMES = 400
G_K_TEAMSETS = ("home72", "away")
#: §5.6.1: a dose gap over 10 % between b-ON and b-OFF is a reported confound.
DOSE_CONFOUND_REL = 0.10

GAIN, NULL, LOSS, UNDECIDED = "GAIN", "NULL", "LOSS", "UNDECIDED"
CONTINUE, INCONCLUSIVE = "CONTINUE", "INCONCLUSIVE"
ADOPT, TRADE = "ADOPT-CANDIDATE", "TRADE"
PASS, FAIL = "PASS", "FAIL"
GUARDS = ("G-F", "G-E", "G-C", "G-K")


class ReadoutInputError(ValueError):
    """An input the registered statistic is not defined on (too few seeds, zero variance, a non-finite value)."""


def t_quantile(q: float, df: float) -> float:
    """Student t's ``q`` quantile on a REAL df (Satterthwaite's is not an integer)."""
    from scipy.stats import t as student_t

    return float(student_t.ppf(q, df))


def _clears(stat: float, boundary: float, *, above: bool) -> Tuple[bool, bool]:
    """``(crosses, near)``: ``above`` asks stat > boundary, else stat < boundary; within RULE8_EPS is ``near`` and
    never a crossing (the ``main.h2h.cross`` convention)."""
    if abs(stat - boundary) <= RULE8_EPS:
        return False, True
    return (stat > boundary if above else stat < boundary), False


def _mean(xs: Sequence[float]) -> float:
    return math.fsum(xs) / len(xs)


def _var(xs: Sequence[float]) -> float:
    m = _mean(xs)
    return math.fsum((x - m) ** 2 for x in xs) / (len(xs) - 1)


# ------------------------------------------------------------------------------------------- the population score
@dataclass(frozen=True)
class PopulationScore:
    final: str
    s_pp: Optional[float]
    per_member_pp: Tuple[Tuple[str, float], ...]
    inconclusive: Tuple[str, ...]

    def as_dict(self) -> Dict[str, Any]:
        return {"final": self.final, "s_pp": self.s_pp, "per_member_pp": dict(self.per_member_pp),
                "inconclusive": list(self.inconclusive)}


def population_score(cells: Sequence[Any], final_sha: str, members: Mapping[str, str], *,
                     min_pairs: int = MIN_PAIRS) -> PopulationScore:
    """S(final) from ``eval_ledger.cells`` Cells (player = the final, opponent = the member; sha256 keys).

    ``members`` is ``{label: sha256}`` from the frozen R. The cell's orientation is the declared one: a cell with
    the member as player is NOT flipped (a different request); it is simply not this read's cell."""
    reasons: List[str] = []
    if not members:
        reasons.append("R has no members")
    if final_sha in set(members.values()):
        reasons.append(f"the final {final_sha[:12]} is itself a member of R")
    by: Dict[str, Any] = {}
    for c in cells:
        if c.player != final_sha:
            continue
        if c.opponent in by:
            reasons.append(f"cell {c.player_id} vs {c.opponent_id} appears twice")
        by[c.opponent] = c
    per: List[Tuple[str, float]] = []
    for label, sha in sorted(members.items()):
        c = by.get(sha)
        if c is None:
            reasons.append(f"missing cell vs member {label} ({sha[:12]})")
            continue
        if c.verdict != "OK":
            reasons.extend(f"cell vs {label}: {x}" for x in (c.reasons or ("verdict " + str(c.verdict),)))
        if c.pairs is None:
            reasons.append(f"cell vs {label}: unmirrored (the read is mirrored pairs)")
        elif c.n_pairs < min_pairs:
            reasons.append(f"cell vs {label}: {c.n_pairs} completed pairs < {min_pairs}")
        s = c.score
        if s is None or not math.isfinite(s):
            reasons.append(f"cell vs {label}: no score")
            continue
        per.append((label, 100.0 * float(s)))
    s_pp = _mean([x for _, x in per]) if per and not reasons else None
    return PopulationScore(final=final_sha, s_pp=s_pp, per_member_pp=tuple(per), inconclusive=tuple(reasons))


# --------------------------------------------------------------------------------------------------------- R1
@dataclass(frozen=True)
class WelchStat:
    n_on: int
    n_off: int
    mean_on: float
    mean_off: float
    delta: float
    var_on: float
    var_off: float
    se: float
    df: float
    q: float
    confidence: float
    ci: Tuple[float, float]

    def as_dict(self) -> Dict[str, Any]:
        return {"n_on": self.n_on, "n_off": self.n_off, "mean_on_pp": self.mean_on, "mean_off_pp": self.mean_off,
                "delta_pp": self.delta, "var_on": self.var_on, "var_off": self.var_off, "se_pp": self.se,
                "df": self.df, "q": self.q, "confidence": self.confidence, "ci_pp": list(self.ci)}


def welch_interval(on: Sequence[float], off: Sequence[float], *, confidence: float = R1_CONFIDENCE) -> WelchStat:
    """mean(on) − mean(off) with Welch's two-sided ``confidence`` interval on the Satterthwaite df."""
    if len(on) < 2 or len(off) < 2:
        raise ReadoutInputError(f"Welch needs >= 2 seeds per side (got {len(on)} ON, {len(off)} OFF)")
    if not all(math.isfinite(x) for x in (*on, *off)):
        raise ReadoutInputError("a non-finite seed score")
    va, vb = _var(on), _var(off)
    a, b = va / len(on), vb / len(off)
    v = a + b
    if not v > 0.0:
        raise ReadoutInputError("zero variance on both sides: Welch's t is undefined")
    df = v * v / ((a * a) / (len(on) - 1) + (b * b) / (len(off) - 1))
    se = math.sqrt(v)
    q = t_quantile(1.0 - (1.0 - confidence) / 2.0, df)
    d = _mean(on) - _mean(off)
    return WelchStat(n_on=len(on), n_off=len(off), mean_on=_mean(on), mean_off=_mean(off), delta=d, var_on=va,
                     var_off=vb, se=se, df=df, q=q, confidence=confidence, ci=(d - q * se, d + q * se))


def classify(ci: Tuple[float, float], bar: float = BAR_PP) -> Tuple[str, Tuple[str, ...]]:
    """R1's class of an interval, and every boundary a bound touched (rule 8 ⇒ UNDECIDED)."""
    lo, hi = ci
    near = [f"{name} {x!r} within {RULE8_EPS} of {b:+}" for name, x in (("lower", lo), ("upper", hi))
            for b in (-bar, bar) if abs(x - b) <= RULE8_EPS]
    if near:
        return UNDECIDED, tuple(near)
    if lo > bar:
        return GAIN, ()
    if hi < -bar:
        return LOSS, ()
    if lo > -bar and hi < bar:
        return NULL, ()
    return UNDECIDED, ()


@dataclass(frozen=True)
class R1Decision:
    look: int
    cls: Optional[str]
    outcome: str
    final: bool
    stat: Optional[WelchStat]
    seeds: Tuple[str, ...]
    near: Tuple[str, ...]
    reasons: Tuple[str, ...]

    def as_dict(self) -> Dict[str, Any]:
        return {"look": self.look, "class": self.cls, "outcome": self.outcome, "final": self.final,
                "stat": self.stat.as_dict() if self.stat else None, "seeds": list(self.seeds),
                "near_boundary": list(self.near), "reasons": list(self.reasons)}


def decide_r1(look: int, on: Mapping[str, Optional[float]], off: Mapping[str, Optional[float]], *,
              inconclusive: Sequence[str] = ()) -> R1Decision:
    """R1 at ``look`` from ``{seed: S_pp}`` per side (``None`` = that final's S is INCONCLUSIVE)."""
    if look not in LOOK_SEEDS:
        raise ValueError(f"look {look} is not one of the registered looks {sorted(LOOK_SEEDS)}")
    reasons = list(inconclusive)
    seeds = tuple(sorted(on))
    if set(on) != set(off):
        reasons.append(f"b-ON seeds {sorted(on)} != b-OFF seeds {sorted(off)} (they share seed i, §5.6.1)")
    want = LOOK_SEEDS[look]
    if len(on) != want or len(off) != want:
        reasons.append(f"{len(on)} ON / {len(off)} OFF seed finals; look {look} registers {want} per side")
    reasons.extend(f"{side} seed {s}: no population score" for side, d in (("ON", on), ("OFF", off))
                   for s, v in sorted(d.items()) if v is None)
    stat: Optional[WelchStat] = None
    if not reasons:
        xs_on = [float(v) for s in seeds if (v := on[s]) is not None]
        xs_off = [float(v) for s in seeds if (v := off[s]) is not None]
        try:
            stat = welch_interval(xs_on, xs_off)
        except ReadoutInputError as exc:
            reasons.append(str(exc))
    if reasons or stat is None:
        return R1Decision(look, None, INCONCLUSIVE, True, None, seeds, (), tuple(reasons))
    cls, near = classify(stat.ci)
    if cls == UNDECIDED and look == 1:
        pos, n0 = _clears(stat.delta, 0.0, above=True)
        nn = near + ((f"Δ̂ {stat.delta!r} within {RULE8_EPS} of 0 (the look-2 rule)",) if n0 else ())
        if pos:
            return R1Decision(look, cls, CONTINUE, False, stat, seeds, nn, ())
        return R1Decision(look, cls, cls, True, stat, seeds, nn, ())
    return R1Decision(look, cls, cls, True, stat, seeds, near, ())


# --------------------------------------------------------------------------------------------------------- R2
@dataclass(frozen=True)
class GapDiff:
    """gap(ON) − gap(OFF) = p_ON − p_OFF with its Newcombe 95 % interval (``best_response_gap.newcombe_diff_ci``)."""

    w_on: int
    n_on: int
    w_off: int
    n_off: int
    diff: float
    lo: float
    hi: float

    def as_dict(self) -> Dict[str, Any]:
        return {"on": [self.w_on, self.n_on], "off": [self.w_off, self.n_off], "diff": self.diff,
                "ci95": [self.lo, self.hi], "gap_on": self.w_on / self.n_on - 0.5,
                "gap_off": self.w_off / self.n_off - 0.5}


def gap_diff(on: Sequence[int], off: Sequence[int]) -> GapDiff:
    """``on`` / ``off`` = (best-responder wins, games) against the b-ON / b-OFF seed-3501 final."""
    from agents.training.best_response_gap import newcombe_diff_ci

    (w1, n1), (w2, n2) = (int(on[0]), int(on[1])), (int(off[0]), int(off[1]))
    if n1 <= 0 or n2 <= 0 or not (0 <= w1 <= n1 and 0 <= w2 <= n2):
        raise ReadoutInputError(f"invalid best-responder counts ON {w1}/{n1}, OFF {w2}/{n2}")
    d, lo, hi = newcombe_diff_ci(w1, n1, w2, n2)
    return GapDiff(w1, n1, w2, n2, d, lo, hi)


@dataclass(frozen=True)
class R2Read:
    gap: Optional[GapDiff]
    falls: bool
    near: Tuple[str, ...]
    reasons: Tuple[str, ...]

    def as_dict(self) -> Dict[str, Any]:
        return {"gap_diff": self.gap.as_dict() if self.gap else None, "falls": self.falls,
                "near_boundary": list(self.near), "reasons": list(self.reasons)}


def decide_r2(gap: Optional[GapDiff], *, inconclusive: Sequence[str] = ()) -> R2Read:
    """"The gap FALLS" iff the Newcombe 95 % upper bound of gap(ON) − gap(OFF) < 0 (rule 8 at 0)."""
    if gap is None or inconclusive:
        return R2Read(gap, False, (), tuple(inconclusive) or ("no R2 read",))
    falls, near = _clears(gap.hi, 0.0, above=False)
    return R2Read(gap, falls, (f"upper {gap.hi!r} within {RULE8_EPS} of 0",) if near else (), ())


# ------------------------------------------------------------------------------------------------------ guards
@dataclass(frozen=True)
class GuardResult:
    name: str
    status: str
    detail: Dict[str, Any] = field(default_factory=dict)
    near: Tuple[str, ...] = ()
    reasons: Tuple[str, ...] = ()

    def as_dict(self) -> Dict[str, Any]:
        return {"guard": self.name, "status": self.status, "detail": self.detail, "near_boundary": list(self.near),
                "reasons": list(self.reasons)}


def guard_f(h: Optional[Sequence[Sequence[float]]], look: int) -> GuardResult:
    """G-F: ``cross_stat`` on the square ON × OFF matrix (pp, rows ON seeds); FAIL iff its 95 % upper < −2.0."""
    if h is None:
        return GuardResult("G-F", INCONCLUSIVE, reasons=("no head-to-head matrix",))
    if len(h) != LOOK_SEEDS[look]:
        return GuardResult("G-F", INCONCLUSIVE, reasons=(f"{len(h)} rows; look {look} registers "
                                                         f"{LOOK_SEEDS[look]} x {LOOK_SEEDS[look]}",))
    try:
        st: CrossStat = cross_stat(h)
    except CrossInputError as exc:
        return GuardResult("G-F", INCONCLUSIVE, reasons=(str(exc),))
    fail, near = _clears(st.ci95[1], -BAR_PP, above=False)
    return GuardResult("G-F", FAIL if fail else PASS, detail={"delta_hat_pp": st.delta_hat,
                                                              "ci95_pp": list(st.ci95), "df": st.df},
                       near=(f"upper {st.ci95[1]!r} within {RULE8_EPS} of {-BAR_PP}",) if near else ())


def guard_e(gap: Optional[GapDiff]) -> GuardResult:
    """G-E: FAIL iff the Newcombe 95 % LOWER bound of gap(ON) − gap(OFF) > 0 (detected MORE exploitable)."""
    if gap is None:
        return GuardResult("G-E", INCONCLUSIVE, reasons=("no R2 read",))
    fail, near = _clears(gap.lo, 0.0, above=True)
    return GuardResult("G-E", FAIL if fail else PASS, detail=gap.as_dict(),
                       near=(f"lower {gap.lo!r} within {RULE8_EPS} of 0",) if near else ())


def _interval(x: Any, name: str) -> Tuple[float, float, float]:
    if x is None or len(x) != 3 or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in x):
        raise ReadoutInputError(f"{name}: an (estimate, lower, upper) triple is required, got {x!r}")
    est, lo, hi = (float(v) for v in x)
    if not lo <= est <= hi:
        raise ReadoutInputError(f"{name}: estimate {est} outside its interval [{lo}, {hi}]")
    return est, lo, hi


def guard_c(b1: Any, b2: Any) -> GuardResult:
    """G-C: B1 (entropy, ON − OFF, nats) FAILs iff upper < −0.05; B2 (starved share, ON − OFF) FAILs iff lower > 0.
    Each input is (estimate, lower, upper) of the battle-clustered 95 % interval, pooled over seeds."""
    try:
        e1, l1, u1 = _interval(b1, "B1")
        e2, l2, u2 = _interval(b2, "B2")
    except ReadoutInputError as exc:
        return GuardResult("G-C", INCONCLUSIVE, reasons=(str(exc),))
    f1, n1 = _clears(u1, G_C_B1_BAR_NATS, above=False)
    f2, n2 = _clears(l2, 0.0, above=True)
    near = tuple(m for m, n in ((f"B1 upper {u1!r} within {RULE8_EPS} of {G_C_B1_BAR_NATS}", n1),
                                (f"B2 lower {l2!r} within {RULE8_EPS} of 0", n2)) if n)
    return GuardResult("G-C", FAIL if (f1 or f2) else PASS,
                       detail={"B1": [e1, l1, u1], "B1_fail": f1, "B2": [e2, l2, u2], "B2_fail": f2}, near=near)


def guard_k(cells: Optional[Mapping[str, Mapping[str, Sequence[int]]]]) -> GuardResult:
    """G-K: ``{teamset: {"on": (wins, n), "off": (wins, n)}}`` for home72 and away; FAIL iff ON − OFF's Newcombe
    95 % upper bound < 0 on EITHER set (``main.anchors.results.newcombe``, the SOP's interval)."""
    from main.anchors.results import newcombe

    if cells is None:
        return GuardResult("G-K", INCONCLUSIVE, reasons=("no external-anchor read",))
    reasons: List[str] = []
    detail: Dict[str, Any] = {}
    near: List[str] = []
    fail = False
    for ts in G_K_TEAMSETS:
        c = cells.get(ts)
        if c is None or "on" not in c or "off" not in c:
            reasons.append(f"{ts}: no ON / OFF counts")
            continue
        (w1, n1), (w2, n2) = (int(c["on"][0]), int(c["on"][1])), (int(c["off"][0]), int(c["off"][1]))
        if not (0 <= w1 <= n1 and 0 <= w2 <= n2):
            reasons.append(f"{ts}: invalid counts ON {w1}/{n1}, OFF {w2}/{n2}")
            continue
        if n1 < G_K_MIN_GAMES or n2 < G_K_MIN_GAMES:
            reasons.append(f"{ts}: {n1} ON / {n2} OFF counted games < {G_K_MIN_GAMES}")
            continue
        d, lo, hi = newcombe(w1, n1, w2, n2)
        f, nb = _clears(hi, 0.0, above=False)
        if nb:
            near.append(f"{ts} upper {hi!r} within {RULE8_EPS} of 0")
        fail = fail or f
        detail[ts] = {"on": [w1, n1], "off": [w2, n2], "diff": d, "ci95": [lo, hi], "detected_below": f}
    if reasons:
        return GuardResult("G-K", INCONCLUSIVE, detail=detail, near=tuple(near), reasons=tuple(reasons))
    return GuardResult("G-K", FAIL if fail else PASS, detail=detail, near=tuple(near))


# ------------------------------------------------------------------------------------------------------ verdict
@dataclass(frozen=True)
class Verdict:
    arm: str
    look: int
    verdict: str
    r1: R1Decision
    r2: R2Read
    guards: Tuple[GuardResult, ...]
    reasons: Tuple[str, ...]
    reported: Dict[str, Any]

    def as_dict(self) -> Dict[str, Any]:
        return {"arm": self.arm, "look": self.look, "verdict": self.verdict, "r1": self.r1.as_dict(),
                "r2": self.r2.as_dict(), "guards": {g.name: g.as_dict() for g in self.guards},
                "reasons": list(self.reasons), "reported": self.reported}


def dose_confound(on: Optional[Sequence[float]], off: Optional[Sequence[float]]) -> Dict[str, Any]:
    """§5.6.1: |mean dose ON − mean dose OFF| / mean dose OFF > 10 % is a REPORTED confound (rule 8 at the bar)."""
    if not on or not off:
        return {"read": False}
    a, b = _mean(on), _mean(off)
    if not b > 0:
        return {"read": False, "reason": "non-positive OFF dose"}
    rel = abs(a - b) / b
    conf, near = _clears(rel, DOSE_CONFOUND_REL, above=True)
    return {"read": True, "dose_on": a, "dose_off": b, "rel_gap": rel, "confound": conf, "near": near}


def decide(arm: str, r1: R1Decision, r2: R2Read, guards: Sequence[GuardResult], *,
           reported: Optional[Dict[str, Any]] = None) -> Verdict:
    """The pre-registered verdict (module docstring)."""
    if arm not in ARMS:
        raise ValueError(f"arm {arm!r} is not one of {sorted(ARMS)}")
    by = {g.name: g for g in guards}
    gs = tuple(by.get(n) or GuardResult(n, INCONCLUSIVE, reasons=("not read",)) for n in GUARDS)
    rep = dict(reported or {})

    def out(v: str, *why: str) -> Verdict:
        return Verdict(arm, r1.look, v, r1, r2, gs, tuple(why), rep)

    if r1.outcome == INCONCLUSIVE:
        return out(INCONCLUSIVE, "R1 is INCONCLUSIVE", *r1.reasons)
    if r1.outcome == CONTINUE:
        return out(CONTINUE, f"R1 UNDECIDED with Δ̂ > 0 at look {r1.look}: go to look 2")
    via_r2 = arm in R2_ARMS and r1.cls == NULL
    if via_r2 and r2.reasons:
        return out(INCONCLUSIVE, f"{arm}: R1 NULL needs R2 to decide", *r2.reasons)
    qualifies = r1.cls == GAIN or (via_r2 and r2.falls)
    if not qualifies:
        return out(str(r1.cls), "R1 does not qualify" + (" (R2: the gap does not fall)" if via_r2 else ""))
    why = "R1 GAIN" if r1.cls == GAIN else "R1 NULL and R2: the gap FALLS"
    open_ = [g.name for g in gs if g.status == INCONCLUSIVE]
    if open_:
        return out(INCONCLUSIVE, f"{why}, but guard(s) {open_} INCONCLUSIVE")
    failed = [g.name for g in gs if g.status == FAIL]
    if failed:
        return out(TRADE, f"{why}, but guard(s) {failed} FAIL")
    return out(ADOPT, f"{why}; every guard PASSES")


def read(doc: Mapping[str, Any]) -> Verdict:
    """The whole decision from one JSON-shaped read (the CLI's input; ``selfcheck`` shows every field):

    ``{"arm": "A1", "look": 1, "r1": {"on": {seed: S_pp}, "off": {...}}, "r2": {"on": [w, n], "off": [w, n]},
    "g_f": {"h_pp": [[...]]}, "g_c": {"b1": [e, lo, hi], "b2": [e, lo, hi]},
    "g_k": {"home72": {"on": [w, n], "off": [w, n]}, "away": {...}}, "dose": {"on": [...], "off": [...]}}``"""
    arm, look = str(doc["arm"]), int(doc["look"])
    r1d = doc.get("r1") or {}
    r1 = decide_r1(look, dict(r1d.get("on") or {}), dict(r1d.get("off") or {}),
                   inconclusive=tuple(r1d.get("inconclusive") or ()))
    gap: Optional[GapDiff] = None
    r2why: List[str] = []
    r2d = doc.get("r2")
    if r2d:
        try:
            gap = gap_diff(r2d["on"], r2d["off"])
        except (ReadoutInputError, KeyError, TypeError, IndexError) as exc:
            r2why.append(f"R2 input: {exc}")
    r2 = decide_r2(gap, inconclusive=r2why)
    gf = (doc.get("g_f") or {}).get("h_pp")
    gc = doc.get("g_c") or {}
    guards = [guard_f(gf, look), guard_e(gap if not r2why else None), guard_c(gc.get("b1"), gc.get("b2")),
              guard_k(doc.get("g_k"))]
    dose = doc.get("dose") or {}
    return decide(arm, r1, r2, guards, reported={"dose": dose_confound(dose.get("on"), dose.get("off"))})


def _main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse
    import json
    import sys

    p = argparse.ArgumentParser(prog="python -m main.x35.readout", description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("decide", help="the verdict of one read JSON (see main.x35.readout.read)")
    d.add_argument("input")
    d1 = sub.add_parser("decide-d1", help="D1's (X38's) verdict of one read JSON (see main.x35.d1.read_d1)")
    d1.add_argument("input")
    sc = sub.add_parser("self-check", help="the constructed-case self-check (main.x35.selfcheck); exit 1 on any FAIL")
    sc.add_argument("--suite", default="all", choices=("all", "x35", "d1"))
    a = p.parse_args(argv)
    if a.cmd == "self-check":
        from main.x35 import selfcheck

        return selfcheck.main(a.suite)
    with open(a.input) as f:
        doc = json.load(f)
    if a.cmd == "decide-d1":
        from main.x35 import d1 as D1

        json.dump(D1.read_d1(doc).as_dict(), sys.stdout, indent=1, default=str)
        print()
        return 0
    v = read(doc)
    json.dump(v.as_dict(), sys.stdout, indent=1, default=str)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
