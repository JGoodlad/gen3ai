"""The X5 A/B's REGISTERED cross statistic and decision rule (``designs/endstate/design_x5_belief_tokens.md`` §7.4),
the oracle reads (§7.6–§7.7) and the matched wall-time read (§7.9) — pure functions over a seed × seed matrix of
head-to-head scores, plus the adapter that builds that matrix from a declared eval-ledger read.

THE STATISTIC (§7.4, verbatim). ``h[i][j]`` = arm-X seed i's score (pp, a draw = ½) against arm-B seed j, one
mirrored cell of ≥ 1,000 pairs each. Δ̂ = mean_ij h_ij − 50; row means R_i, column means C_j;
V̂ = (s²_R + s²_C) / n with n seeds per arm (a SQUARE matrix: the registration has n seeds per arm); df = 2(n − 1);
t = (Δ̂ + δ) / √V̂ with δ = 3.5 pp. Under the additive model h_ij = Δ + a_i − b_j + noise this is the two-sample t on
the two arms' run strengths.

THE DECISION (§7.4's outcome table; the boundaries are O'Brien–Fleming on the t scale ×1.02):

* INCONCLUSIVE (any look) — an input is incomplete or invalid (a missing cell, a cell with < 1,000 completed pairs or
  aborted games > 25 % of attempted, a broken precondition). Never interpreted.
* NON-INFERIOR — t ≥ the look's boundary (5.761 / 2.683 / 1.874 at looks 1 / 2 / 3).
* FUTILITY STOP (looks 1–2, non-binding) — Δ̂ ≤ −δ. Verdict NOT DETECTED, labelled INFERIOR iff the upper
  one-sided 95 % bound Δ̂ + t_{0.95, df}·√V̂ < −δ.
* CONTINUE (looks 1–2) — neither.
* NOT DETECTED (look 3) — t below the boundary; the INFERIOR label as above.

RULE 8 (standing rule 8; §7.4 states it for t): a statistic within ``RULE8_EPS`` (1e-9) of ANY decision boundary
it is compared against (t vs the look's boundary, Δ̂ vs −δ, the upper bound vs −δ) is NOT a crossing of it, and the
read records which boundary it touched (``near_boundary``). Deterministic: no tolerance decides by chance.

DECLARED HERE, BEFORE ANY CELL IS PLAYED (choices the registration leaves open; each a FINDING in the look's
ledger entry):

* The reported 95 % interval of a cross Δ̂ is the TWO-sided Student-t interval Δ̂ ± t_{0.975, df}·√V̂.
* C (oracle − reference, §7.6) is the same statistic with the oracle seeds as rows; "C's lower 95 % bound" is the
  lower end of that TWO-sided 95 % interval (the more conservative reading of "lower 95 % bound").
* C_full − C_species: both crosses share the SAME reference seeds (columns), so the reference effects b_j cancel;
  the contrast is the difference of the two arms' grand means with V = s²_{R,full}/n_f + s²_{R,sp}/n_s on
  df = n_f + n_s − 2 (the two-sample t on the oracle seeds' row means).
* The REPLICATE FLOOR for H (§7.6, UNDERSTANDING rule 3: the MAX pairwise |Δ| over the replicates in hand) is
  F = max(P0_H2H_FLOOR_PP, the max pairwise |C_j − C_k| over the blob columns of the matched-steps cross). P0's
  4.57 pp is the largest max-pairwise |Δ| among P0's replicate-like h2h runs (S3 / S4, 8M, the sizing finals,
  ``measurements/x5_p0_h2h_2026-10-03/result.md``); the in-experiment term is the blob replicates' spread at THIS
  depth on THIS scale. H = Δ / C is computed only when C's lower bound > F (rule 8 applies to that comparison).
* The seat effect u (the player always keeps seat p1) is NOT subtracted: the registered statistic does not, and
  P0 measured u = −0.006 ± 0.083 pp on this scale.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from agents.training import eval_ledger as L

#: §7.4: the non-inferiority margin, pp.
DELTA_PP = 3.5
#: §7.4: NON-INFERIOR iff t ≥ the look's boundary.
LOOK_BOUNDARIES: Dict[int, float] = {1: 5.761, 2: 2.683, 3: 1.874}
#: §7.4: seeds per arm at each look.
LOOK_SEEDS: Dict[int, int] = {1: 3, 2: 5, 3: 8}
#: §7.4: completed mirrored pairs per cell.
MIN_PAIRS = 1000
#: Rule 8: a statistic this close to a boundary is not a crossing of it.
RULE8_EPS = 1e-9
#: §7.6's floor, P0's h2h term (pp): S3 / S4's max |pairwise| (module docstring).
P0_H2H_FLOOR_PP = 4.57

INCONCLUSIVE = "INCONCLUSIVE"
NON_INFERIOR = "NON-INFERIOR"
FUTILITY_STOP = "FUTILITY STOP"
CONTINUE = "CONTINUE"
NOT_DETECTED = "NOT DETECTED"
INFERIOR = "INFERIOR"
OUTCOMES = (INCONCLUSIVE, NON_INFERIOR, FUTILITY_STOP, CONTINUE, NOT_DETECTED)


class CrossInputError(ValueError):
    """A matrix the registered statistic is not defined on (not square, n < 2, a non-finite cell, zero variance)."""


def t_ppf(q: float, df: int) -> float:
    from scipy.stats import t as student_t

    return float(student_t.ppf(q, df))


def _mean(xs: Sequence[float]) -> float:
    return math.fsum(xs) / len(xs)


def _var(xs: Sequence[float]) -> float:
    """Unbiased sample variance (ddof = 1)."""
    m = _mean(xs)
    return math.fsum((x - m) ** 2 for x in xs) / (len(xs) - 1)


def _clears(stat: float, boundary: float, *, above: bool) -> Tuple[bool, bool]:
    """``(crosses, near)``: ``above`` asks stat ≥ boundary, else stat ≤ boundary; within RULE8_EPS is ``near`` and
    never a crossing."""
    if abs(stat - boundary) <= RULE8_EPS:
        return False, True
    return (stat > boundary if above else stat < boundary), False


@dataclass(frozen=True)
class CrossStat:
    """The registered statistic on one n × n matrix (pp)."""

    n: int
    h: Tuple[Tuple[float, ...], ...]
    delta_hat: float
    row_means: Tuple[float, ...]
    col_means: Tuple[float, ...]
    s2_rows: float
    s2_cols: float
    v_hat: float
    se: float
    df: int
    t: float
    margin: float
    ci95: Tuple[float, float]
    upper95_one_sided: float

    def as_dict(self) -> Dict[str, Any]:
        return {"n": self.n, "h_pp": [list(r) for r in self.h], "delta_hat_pp": self.delta_hat,
                "row_means_pp": list(self.row_means), "col_means_pp": list(self.col_means),
                "s2_rows": self.s2_rows, "s2_cols": self.s2_cols, "v_hat": self.v_hat, "se_pp": self.se,
                "df": self.df, "t": self.t, "margin_pp": self.margin, "ci95_pp": list(self.ci95),
                "upper95_one_sided_pp": self.upper95_one_sided}


def cross_stat(h: Sequence[Sequence[float]], *, margin: float = DELTA_PP) -> CrossStat:
    """§7.4's statistic on the matrix ``h`` (rows = the arm under test's seeds, columns = the reference's; pp)."""
    n = len(h)
    if n < 2 or any(len(r) != n for r in h):
        raise CrossInputError(f"the registered cross needs an n × n matrix with n ≥ 2 (got {n} rows of "
                              f"{[len(r) for r in h]})")
    if not all(math.isfinite(x) for r in h for x in r):
        raise CrossInputError("a non-finite cell")
    hh = tuple(tuple(float(x) for x in r) for r in h)
    rows = tuple(_mean(r) for r in hh)
    cols = tuple(_mean([hh[i][j] for i in range(n)]) for j in range(n))
    dhat = _mean([x for r in hh for x in r]) - 50.0
    s2r, s2c = _var(rows), _var(cols)
    v = (s2r + s2c) / n
    if not v > 0.0:
        raise CrossInputError("zero variance across both arms' seed means: t is undefined")
    se = math.sqrt(v)
    df = 2 * (n - 1)
    q975, q95 = t_ppf(0.975, df), t_ppf(0.95, df)
    return CrossStat(n=n, h=hh, delta_hat=dhat, row_means=rows, col_means=cols, s2_rows=s2r, s2_cols=s2c,
                     v_hat=v, se=se, df=df, t=(dhat + margin) / se, margin=margin,
                     ci95=(dhat - q975 * se, dhat + q975 * se), upper95_one_sided=dhat + q95 * se)


@dataclass(frozen=True)
class CrossDecision:
    look: int
    boundary: float
    outcome: str
    label: Optional[str]
    reasons: Tuple[str, ...]
    near_boundary: Tuple[str, ...]
    stat: Optional[CrossStat]

    def as_dict(self) -> Dict[str, Any]:
        return {"look": self.look, "boundary": self.boundary, "outcome": self.outcome, "label": self.label,
                "reasons": list(self.reasons), "near_boundary": list(self.near_boundary),
                "stat": self.stat.as_dict() if self.stat is not None else None}


def decide(look: int, stat: Optional[CrossStat], inconclusive: Sequence[str] = ()) -> CrossDecision:
    """§7.4's outcome table at ``look``. ``inconclusive`` lists every invalid-input reason (any ⇒ INCONCLUSIVE,
    whatever the statistic says; ``stat`` may then be ``None``)."""
    if look not in LOOK_BOUNDARIES:
        raise ValueError(f"look {look} is not one of the registered looks {sorted(LOOK_BOUNDARIES)}")
    b = LOOK_BOUNDARIES[look]
    reasons = list(inconclusive)
    if stat is None and not reasons:
        reasons.append("no statistic")
    if stat is not None and stat.n != LOOK_SEEDS[look]:
        reasons.append(f"{stat.n} seeds per arm, look {look} registers {LOOK_SEEDS[look]}")
    if reasons:
        return CrossDecision(look, b, INCONCLUSIVE, None, tuple(reasons), (), stat)
    assert stat is not None
    near: List[str] = []
    crosses, nb = _clears(stat.t, b, above=True)
    if nb:
        near.append(f"t {stat.t!r} within {RULE8_EPS} of the boundary {b}")
    inferior, ni = _clears(stat.upper95_one_sided, -stat.margin, above=False)
    if ni:
        near.append(f"the upper one-sided 95 % bound {stat.upper95_one_sided!r} within {RULE8_EPS} of −δ")
    if crosses:
        return CrossDecision(look, b, NON_INFERIOR, None, (), tuple(near), stat)
    label = INFERIOR if inferior else None
    if look == 3:
        return CrossDecision(look, b, NOT_DETECTED, label, (), tuple(near), stat)
    futile, nf = _clears(stat.delta_hat, -stat.margin, above=False)
    if nf:
        near.append(f"Δ̂ {stat.delta_hat!r} within {RULE8_EPS} of −δ")
    if futile:
        return CrossDecision(look, b, FUTILITY_STOP, label, (), tuple(near), stat)
    return CrossDecision(look, b, CONTINUE, None, (), tuple(near), stat)


def verdict_text(d: CrossDecision) -> str:
    """The registered vocabulary, verbatim: a futility stop's verdict is NOT DETECTED (+ INFERIOR if labelled)."""
    if d.outcome == FUTILITY_STOP:
        return f"FUTILITY STOP (verdict NOT DETECTED{', INFERIOR' if d.label else ''})"
    if d.outcome == NOT_DETECTED and d.label:
        return "NOT DETECTED, INFERIOR"
    return d.outcome


# --------------------------------------------------------------------------------------------- the oracle reads
@dataclass(frozen=True)
class Contrast:
    """A difference of two arms' grand means against SHARED reference columns (C_full − C_species)."""

    estimate: float
    v: float
    se: float
    df: int
    ci95: Tuple[float, float]

    def as_dict(self) -> Dict[str, Any]:
        return {"estimate_pp": self.estimate, "v": self.v, "se_pp": self.se, "df": self.df,
                "ci95_pp": list(self.ci95)}


def shared_column_contrast(a: CrossStat, b: CrossStat) -> Contrast:
    """``a.Δ̂ − b.Δ̂`` for two crosses on the SAME reference seeds (the caller's contract): the column effects cancel,
    so V = s²_{R,a}/n_a + s²_{R,b}/n_b on n_a + n_b − 2 df (module docstring)."""
    v = a.s2_rows / a.n + b.s2_rows / b.n
    if not v > 0.0:
        raise CrossInputError("zero variance across both arms' row means")
    se = math.sqrt(v)
    df = a.n + b.n - 2
    est = a.delta_hat - b.delta_hat
    q = t_ppf(0.975, df)
    return Contrast(estimate=est, v=v, se=se, df=df, ci95=(est - q * se, est + q * se))


def max_pairwise(xs: Sequence[float]) -> float:
    return max(abs(x - y) for i, x in enumerate(xs) for y in xs[i + 1:]) if len(xs) > 1 else 0.0


def replicate_floor(steps_cross: CrossStat, p0_floor: float = P0_H2H_FLOOR_PP) -> Dict[str, Any]:
    """§7.6's floor F (module docstring): the larger of P0's h2h replicate floor and the blob columns' max pairwise
    spread in the matched-steps cross."""
    inexp = max_pairwise(steps_cross.col_means)
    return {"floor_pp": max(p0_floor, inexp), "p0_floor_pp": p0_floor, "in_experiment_blob_spread_pp": inexp,
            "source": "max(P0 h2h max-pairwise |Δ| (S3/S4, 8M), max pairwise |C_j − C_k| over the matched-steps "
                      "cross's blob columns)"}


def headroom(delta: CrossStat, c: CrossStat, floor_pp: float) -> Dict[str, Any]:
    """H = Δ / C, reported ONLY when C's lower 95 % bound (two-sided interval) is above the floor (rule 8 applies);
    otherwise the registered sentence."""
    above, near = _clears(c.ci95[0], floor_pp, above=True)
    out: Dict[str, Any] = {"c_lower95_pp": c.ci95[0], "floor_pp": floor_pp, "c_clears_floor": above,
                           "near_floor": near}
    if above:
        out["H"] = delta.delta_hat / c.delta_hat
    else:
        out["H"] = None
        out["read"] = ("the ceiling is within the noise: belief representation has little strength leverage at "
                       "this budget")
    return out


def grade(point: float, lo: float, hi: float) -> str:
    """A late-registered prediction's grade: IN iff the point estimate lies in its closed 80 % interval."""
    return "IN" if lo <= point <= hi else "OUT"


# --------------------------------------------------------------------------------------- matrix from ledger cells
@dataclass(frozen=True)
class MatrixFromCells:
    h: Optional[Tuple[Tuple[float, ...], ...]]
    inconclusive: Tuple[str, ...]
    cells: Tuple[Dict[str, Any], ...] = field(default=())


def matrix_from_cells(cells: Sequence[Any], rows: Sequence[str], cols: Sequence[str],
                      min_pairs: int = MIN_PAIRS) -> MatrixFromCells:
    """``h[i][j]`` (pp) from ``eval_ledger.cells`` Cells keyed by (player sha256, opponent sha256). A missing cell,
    a cell listed twice, a non-OK cell or one below ``min_pairs`` completed pairs is an INCONCLUSIVE reason (the
    per-cell abort rule is the ledger's ``cells``)."""
    by: Dict[Tuple[str, str], Any] = {}
    reasons: List[str] = []
    for c in cells:
        k = (c.player, c.opponent)
        if k in by:
            reasons.append(f"cell {c.player_id} vs {c.opponent_id} appears twice (two requests?)")
        by[k] = c
    h: List[List[float]] = []
    info: List[Dict[str, Any]] = []
    for r in rows:
        line: List[float] = []
        for o in cols:
            c = by.get((r, o))
            if c is None:
                reasons.append(f"missing cell {r[:12]} vs {o[:12]}")
                line.append(float("nan"))
                continue
            if c.verdict != "OK":
                reasons.extend(f"cell {c.player_id} vs {c.opponent_id}: {x}" for x in c.reasons)
            if c.n_pairs < min_pairs:
                reasons.append(f"cell {c.player_id} vs {c.opponent_id}: {c.n_pairs} completed pairs < {min_pairs}")
            s = c.score
            line.append(100.0 * s if s is not None else float("nan"))
            info.append({"player": c.player_id, "opponent": c.opponent_id, "n_pairs": c.n_pairs, "w": c.w, "l": c.l,
                         "d": c.d, "aborted": c.aborted, "pairs": list(c.pairs or ()), "score_pp": line[-1],
                         "verdict": c.verdict})
        h.append(line)
    extra = set(by) - {(r, o) for r in rows for o in cols}
    if extra:
        reasons.append(f"{len(extra)} cell(s) outside the declared rows × columns")
    ok = not reasons and all(math.isfinite(x) for line in h for x in line)
    return MatrixFromCells(h=tuple(tuple(x) for x in h) if ok else None, inconclusive=tuple(reasons),
                           cells=tuple(info))


def summarize(look: int, mat: MatrixFromCells, preconditions: Sequence[str] = ()) -> CrossDecision:
    """The look's decision from a matrix read: every INCONCLUSIVE reason (cells + preconditions) or the statistic."""
    reasons = list(mat.inconclusive) + list(preconditions)
    stat = cross_stat(mat.h) if mat.h is not None and not reasons else None
    return decide(look, stat, reasons)


# ------------------------------------------------------------------------------------------- the declared reads
#: A look of an X5 A/B family, per protocol (a family pins ONE protocol, §0c rule 6): the cells of every look, read
#: ACROSS RUNS (this module's seed-level estimator, never a pooled pentanomial).
FAMILY_READ_OFF = L.ReaderDecl(
    name="main.h2h.cross.family_off", purposes=frozenset({"ab"}),
    regime=L.RegimeFilter(protocol="gen3_eval_protocol_v1_h2h", play="greedy", opponent_play="greedy", mirrored=True),
    requests="family", selection="include", flags_ok=frozenset(), inference="across_runs",
    decision_kind="ab_verdict")
FAMILY_READ_ONE_SIDED = L.ReaderDecl(
    name="main.h2h.cross.family_oracle_one_sided", purposes=frozenset({"ab"}),
    regime=L.RegimeFilter(protocol="gen3_eval_protocol_v1_h2h_oracle_one_sided", play="greedy",
                          opponent_play="greedy", mirrored=True),
    requests="family", selection="include", flags_ok=frozenset(), inference="across_runs",
    decision_kind="ab_verdict")
FAMILY_READ_BOTH_SIDED = L.ReaderDecl(
    name="main.h2h.cross.family_oracle_both_sided", purposes=frozenset({"ab"}),
    regime=L.RegimeFilter(protocol="gen3_eval_protocol_v1_h2h_oracle_both_sided", play="greedy",
                          opponent_play="greedy", mirrored=True),
    requests="family", selection="include", flags_ok=frozenset(), inference="across_runs",
    decision_kind="ab_verdict")


def look_cells(root: Any, family: str, request_id: str, protocol: str) -> List[Any]:
    """Every cell (``eval_ledger.cells``) of ONE look (a request) of a registered family, at ``protocol``."""
    if protocol == "gen3_eval_protocol_v1_h2h":
        got = L.read(FAMILY_READ_OFF, root=root, family=family)
    elif protocol == "gen3_eval_protocol_v1_h2h_oracle_one_sided":
        got = L.read(FAMILY_READ_ONE_SIDED, root=root, family=family)
    elif protocol == "gen3_eval_protocol_v1_h2h_oracle_both_sided":
        got = L.read(FAMILY_READ_BOTH_SIDED, root=root, family=family)
    else:
        raise ValueError(f"protocol {protocol!r} is not one main.h2h plays")
    looks = dict(L.looks(got))
    return list(looks.get(request_id, []))
