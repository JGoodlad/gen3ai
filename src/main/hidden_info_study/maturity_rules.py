"""THE MATURITY BATTERY's decision rules as code: R-MATURITY (`design_hidden_info_binding.md` §12.13.4, read over the
§12.13.7 trajectory) and the per-checkpoint gate it consumes (R-STAGE-SEARCH, re-applied to ONE checkpoint).

Conventions (Tier 1's, `hidden_info_tier1_2026-10-13/t1_stats.py`; `maturity_rules_test` checks the bootstrap draws
equal Tier 1's exactly):

* the bootstrap: 2,000 draws, ``numpy.random.default_rng(20261011)``, battles resampled with replacement, percentile
  intervals; every quantity of one read is evaluated on the SAME draws. The battery scores ONE frozen bank at every
  checkpoint, so one :class:`Boot` over the bank's battles makes every across-checkpoint contrast PAIRED;
* RULE 8: a bound within ±0.0005 of its boundary is a BOUNDARY result and counts as not clear. A boundary of exactly
  ±0.0005 already carries the margin (the registration writes the zero edge as "> +0.0005"); any other boundary c is
  clear only beyond c ± 0.0005; "inside ±m" means lower > −m + 0.0005 and upper < m − 0.0005.

R-MATURITY (first = the series' first anchor, last = its last; per-draw vectors paired):

* GAP-SHRINKS iff upper95 [G_F(last) − G_F(first)] < −0.0005; GAP-GROWS iff lower95 > +0.0005; else NOT DETECTED.
  Per cell: the all-state cell and every stage bucket alike.
* BELIEF-RISES iff lower95 [TODAY(last) − TODAY(first)] > +0.0005, else NOT DETECTED. TODAY per draw is the
  p-equivalent of FN on the M8 curve, CLAMPED to [0, 1] (FN at or below the p = 0 value reads 0; above the p = 1 value
  reads 1), so a difference is defined on every draw.
* FRONTIER (on each stage axis; "earlier" = lower k, lower turn, MORE material): E(cp) = the earliest-game GATE-OPEN
  bucket, or "none" (later than every bucket). FRONTIER-MOVES-EARLIER iff on some axis E(last) is strictly earlier
  than E(first); FRONTIER-RECEDES iff on some axis E(last) is strictly later; both (on different axes) = FRONTIER-MIXED;
  neither = FRONTIER-STATIC. GATE-WEAK is not open.
* CONVERGED iff the interval of G_F(last) − G_F(previous anchor) on the all-state cell lies inside ±0.01.
  AMENDMENT A1 (2026-10-19, registered before any plateau-step output): the ±0.01 interval is unreachable on one bank
  at the registered sizes (README FINDING 2: ±0.021 adjacent on 287 LEAN states, ≈ 1,450 common states needed), so a
  read made with ``converged=False`` reports CONVERGED as NOT READ, and CONVERGED is read only POOLED over both banks
  (:func:`converged` on one bootstrap over both banks' battles) when both banks hold the last two anchors.
* GAP_BONFERRONI (A1, README FINDING 1): every stage bucket's GAP verdict again at the 99.6 % level (Bonferroni over
  the 13 buckets); the 95 % bucket verdicts stay as registered, and only one that also holds at 99.6 % is a finding.
* STILL-MOVING (A1, every read): the change G_F(last) − G_F(previous anchor) on the all-state cell; STILL-SHRINKING
  iff upper95 < −0.0005, STILL-GROWING iff lower95 > +0.0005, else NOT DETECTED (which is NOT evidence of convergence).

R3 at one anchor (A1: carries E2's registered rule, `hidden_info_e2_competent_2026-10-16/read_e2.py` ``verdict`` /
``read_frame``, to the plateau battery; :func:`r3_read`): Δs = IMP(F+U) − IMP(F+M) (reveal-only), Δn = IMP(O−M) −
IMP(O−U) (hide-only); each side's direction is U iff lower95 > 0 (rule 8), M iff upper95 < 0; U-FIRST iff both U,
M-FIRST iff both M, else SHARED iff both classes are detected (σ_X = IMP(F+X) − IMP(F) or ν_X = IMP(O) − IMP(O−X)
with lower95 > 0), else NOT DETECTED.

BANK-DEPENDENT (PREREGISTRATION §2.5; :func:`bank_dependent`): the record (plateau bank) and its replication (the
interim bank) read opposite directions (SHRINKS vs GROWS; U-FIRST vs M-FIRST; MOVES-EARLIER vs RECEDES), or, for
BELIEF, one RISES where the other is not detected.

The per-checkpoint GATE (R-STAGE-SEARCH with ONE network, the registration's single-checkpoint reading): GATE-OPEN iff
n >= 150 AND lower [TRUNC-4-K8 − CRITIC] > +0.0005 AND lower [TRUNC-4-K8] > +0.0005 at the 99.6 % level (Bonferroni
over the 13 primary buckets); GATE-WEAK iff both hold at 95 % only; else GATE-CLOSED. ("Each checkpoint separately" and
"pooled" coincide for one network.)

PURE: numpy + stdlib, no ``main`` / ``agents`` import (the reader loads this file by path).
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

B = 2000
SEED = 20261011
EDGE = 0.0005
BONF_LEVEL = 99.6                        # Tier 1's §12.13.4 level (13 primary buckets; `unblind_t1.BONF_LEVEL`)
GATE_MIN_N = 150
CONVERGED_MARGIN = 0.01

#: the stage axes: (field, buckets in EARLIEST-GAME-FIRST order). k and turn ascend; material descends (12 alive is
#: the earliest).
AXES: dict[str, tuple[str, tuple[str, ...]]] = {
    "k": ("k_bin", ("1-2", "3", "4", "5", "6")),
    "turn": ("turn_bin", ("<=7", "8-14", "15-21", ">=22")),
    "material": ("mat_bin", ("12", "10-11", "8-9", "<=7")),
}


class Boot:
    """The battle-clustered bootstrap of Tier 1 (`t1_stats.Boot`, unstratified): ``C[r, b]`` = how often battle b is
    drawn in replicate r."""

    def __init__(self, battles: Sequence[str]):
        self.ub = sorted(set(battles))
        ix = {b: i for i, b in enumerate(self.ub)}
        self.bi = np.array([ix[b] for b in battles])
        nb = len(self.ub)
        rng = np.random.default_rng(SEED)
        self.C = np.zeros((B, nb))
        draws = rng.integers(0, nb, size=(B, nb))
        for r in range(B):
            np.add.at(self.C[r], draws[r], 1.0)
        self.nb = nb

    def mean(self, x: npt.ArrayLike, sel: npt.ArrayLike | None = None) -> tuple[float, np.ndarray]:
        """(estimate, per-draw vector) of the mean of x over sel (non-finite entries dropped)."""
        x = np.asarray(x, dtype=np.float64)
        m = np.ones(len(x), bool) if sel is None else np.asarray(sel, bool).copy()
        m &= np.isfinite(x)
        S = np.bincount(self.bi[m], weights=x[m], minlength=self.nb)
        N = np.bincount(self.bi[m], minlength=self.nb).astype(np.float64)
        with np.errstate(invalid="ignore", divide="ignore"):
            vec = (self.C @ S) / (self.C @ N)
        est = float(S.sum() / N.sum()) if N.sum() > 0 else float("nan")
        return est, vec

    def ratio(self, num: npt.ArrayLike, den: npt.ArrayLike, sel: npt.ArrayLike | None = None
              ) -> tuple[float, np.ndarray]:
        """(estimate, per-draw vector) of mean(num) / mean(den) over sel (a state with a non-finite input dropped)."""
        a, b = np.asarray(num, dtype=np.float64), np.asarray(den, dtype=np.float64)
        m = np.ones(len(a), bool) if sel is None else np.asarray(sel, bool).copy()
        m &= np.isfinite(a) & np.isfinite(b)
        Sa = np.bincount(self.bi[m], weights=a[m], minlength=self.nb)
        Sb = np.bincount(self.bi[m], weights=b[m], minlength=self.nb)
        with np.errstate(invalid="ignore", divide="ignore"):
            vec = (self.C @ Sa) / (self.C @ Sb)
            est = float(Sa.sum() / Sb.sum()) if m.any() else float("nan")
        return est, vec


def ci(vec: npt.ArrayLike, level: float = 95.0) -> list[float | None]:
    v = np.asarray(vec, dtype=np.float64)
    v = v[np.isfinite(v)]
    if len(v) < 20:
        return [None, None]
    a = (100.0 - level) / 2.0
    lo, hi = np.percentile(v, [a, 100.0 - a])
    return [float(lo), float(hi)]


def _margin(c: float) -> float:
    return 0.0 if abs(abs(c) - EDGE) < 1e-12 else EDGE


def gt(x: float | None, c: float) -> bool:
    """x > c, CLEAR of the boundary (rule 8)."""
    return x is not None and bool(np.isfinite(x)) and x > c + _margin(c)


def lt(x: float | None, c: float) -> bool:
    return x is not None and bool(np.isfinite(x)) and x < c - _margin(c)


def inside(lohi: Sequence[float | None], m: float) -> bool:
    lo, hi = lohi
    return lo is not None and hi is not None and lo > -m + EDGE and hi < m - EDGE


def summary(est: float, vec: npt.ArrayLike, n: int | None = None, level: float = 95.0) -> dict:
    lo, hi = ci(vec, level)
    out: dict = {"mean": None if not np.isfinite(est) else round(float(est), 5),
                 "ci": [None if lo is None else round(lo, 5), None if hi is None else round(hi, 5)]}
    if n is not None:
        out["n"] = int(n)
    return out


# ------------------------------------------------------------------------------------------------
# TODAY: the p-equivalent of FN on the M8 curve
# ------------------------------------------------------------------------------------------------

def eq_p(curve: npt.ArrayLike, target: float, grid: npt.ArrayLike) -> float | None:
    """The p at which the curve first reaches ``target`` (linear interpolation); 0 at or below the p = 0 value; None
    above the p = 1 value (Tier 1's `unblind_t1.eq_p`)."""
    c = np.asarray(curve, dtype=np.float64)
    g = np.asarray(grid, dtype=np.float64)
    if not np.isfinite(target) or not np.all(np.isfinite(c)):
        return None
    if target <= c[0]:
        return 0.0
    for i in range(1, len(c)):
        if c[i] >= target:
            a, b = c[i - 1], c[i]
            return float(g[i - 1] + (g[i] - g[i - 1]) * (target - a) / (b - a)) if b > a else float(g[i])
    return None


def today_clamped(curve_est: npt.ArrayLike, target_est: float, curve_draws: npt.ArrayLike, target_draws: npt.ArrayLike,
                  grid: npt.ArrayLike) -> tuple[float | None, np.ndarray]:
    """(TODAY's point, its per-draw vector) with every value clamped to [0, 1] (None above the curve -> 1.0; a draw
    with a non-finite input -> NaN, dropped by the interval)."""
    def one(c: npt.ArrayLike, t: float) -> float:
        if not np.isfinite(t) or not np.all(np.isfinite(c)):
            return float("nan")
        p = eq_p(c, t, grid)
        return 1.0 if p is None else min(1.0, max(0.0, p))

    pt = one(np.asarray(curve_est), target_est)
    vec = np.array([one(c, t) for c, t in zip(np.asarray(curve_draws), np.asarray(target_draws))])
    return (None if not np.isfinite(pt) else pt), vec


# ------------------------------------------------------------------------------------------------
# the per-checkpoint GATE (R-STAGE-SEARCH on one network)
# ------------------------------------------------------------------------------------------------

def gate_status(n: int, t4_minus_crit: np.ndarray, t4: np.ndarray, n_min: int = GATE_MIN_N) -> str:
    """GATE-OPEN / GATE-WEAK / GATE-CLOSED from the per-draw vectors of [TRUNC-4-K8 − CRITIC] and [TRUNC-4-K8]."""
    bonf = gt(ci(t4_minus_crit, BONF_LEVEL)[0], EDGE) and gt(ci(t4, BONF_LEVEL)[0], EDGE)
    p95 = gt(ci(t4_minus_crit)[0], EDGE) and gt(ci(t4)[0], EDGE)
    if n >= n_min and bonf:
        return "GATE-OPEN"
    return "GATE-WEAK" if p95 else "GATE-CLOSED"


def earliest_open(gates: Mapping[str, str], axis: str) -> int:
    """The index (earliest-game first) of the earliest GATE-OPEN bucket on ``axis``; len(buckets) = none open."""
    buckets = AXES[axis][1]
    for i, b in enumerate(buckets):
        if gates.get(b) == "GATE-OPEN":
            return i
    return len(buckets)


def _bucket_name(buckets: Sequence[str], i: int) -> str:
    return buckets[i] if i < len(buckets) else "none"


def frontier_change(first: Mapping[str, Mapping[str, str]], last: Mapping[str, Mapping[str, str]]) -> dict:
    """{axis: {first, last, moved}} and the verdict (see the module docstring)."""
    per, earlier, later = {}, [], []
    for ax, (_f, buckets) in AXES.items():
        e0, e1 = earliest_open(first.get(ax, {}), ax), earliest_open(last.get(ax, {}), ax)
        per[ax] = {"first": _bucket_name(buckets, e0), "last": _bucket_name(buckets, e1),
                   "moved": "earlier" if e1 < e0 else "later" if e1 > e0 else "static"}
        if e1 < e0:
            earlier.append(ax)
        elif e1 > e0:
            later.append(ax)
    v = ("FRONTIER-MIXED" if earlier and later else "FRONTIER-MOVES-EARLIER" if earlier
         else "FRONTIER-RECEDES" if later else "FRONTIER-STATIC")
    return {"axes": per, "verdict": v}


# ------------------------------------------------------------------------------------------------
# R-MATURITY over the series
# ------------------------------------------------------------------------------------------------

@dataclass
class CheckpointRead:
    """One anchor's read on the frozen bank. Every vector is a per-draw vector of the SAME :class:`Boot`.

    ``g``: {cell: (estimate, vector)} of G_F = IMP_O − IMP_F at LEAF-K8 ("all" and "<axis>:<bucket>");
    ``today``: (point, clamped vector) or None when FN is not read; ``gates``: {axis: {bucket: status}}."""

    label: str
    step: int
    g: dict[str, tuple[float, np.ndarray]]
    today: tuple[float | None, np.ndarray] | None = None
    gates: dict[str, dict[str, str]] = field(default_factory=dict)


def gap_change(first: tuple[float, np.ndarray], last: tuple[float, np.ndarray], level: float = 95.0) -> dict:
    d_est = last[0] - first[0]
    d_vec = np.asarray(last[1]) - np.asarray(first[1])
    lo, hi = ci(d_vec, level)
    v = "GAP-SHRINKS" if lt(hi, -EDGE) else "GAP-GROWS" if gt(lo, EDGE) else "NOT DETECTED"
    return {"change": summary(d_est, d_vec, level=level), "verdict": v}


def belief_change(first: tuple[float | None, np.ndarray], last: tuple[float | None, np.ndarray]) -> dict:
    d_vec = np.asarray(last[1]) - np.asarray(first[1])
    d_est = (last[0] - first[0]) if (last[0] is not None and first[0] is not None) else float("nan")
    lo, _hi = ci(d_vec)
    return {"change": summary(d_est, d_vec), "verdict": "BELIEF-RISES" if gt(lo, EDGE) else "NOT DETECTED"}


def converged(prev: tuple[float, np.ndarray], last: tuple[float, np.ndarray]) -> dict:
    d_vec = np.asarray(last[1]) - np.asarray(prev[1])
    iv = ci(d_vec)
    return {"change": summary(last[0] - prev[0], d_vec), "verdict": "CONVERGED" if inside(iv, CONVERGED_MARGIN)
            else "NOT CONVERGED"}


def still_moving(prev: tuple[float, np.ndarray], last: tuple[float, np.ndarray]) -> dict:
    """A1: is G_F still moving between the last two anchors (a detection, reachable at the registered sizes)?"""
    d_vec = np.asarray(last[1]) - np.asarray(prev[1])
    lo, hi = ci(d_vec)
    v = "STILL-SHRINKING" if lt(hi, -EDGE) else "STILL-GROWING" if gt(lo, EDGE) else "NOT DETECTED"
    return {"change": summary(last[0] - prev[0], d_vec), "verdict": v}


CONVERGED_A1_REASON = ("NOT READ (amendment A1: the ±0.01 interval is unreachable on one bank at the registered sizes; "
                       "CONVERGED is read only pooled over both banks when both hold the last two anchors)")


def r_maturity(reads: Sequence[CheckpointRead], converged_on_this_bank: bool = True) -> dict:
    """R-MATURITY over the anchors (sorted by step here; at least two). ``converged_on_this_bank=False`` is A1: the
    one-bank CONVERGED is NOT READ (STILL-MOVING is read either way)."""
    rs = sorted(reads, key=lambda r: r.step)
    if len(rs) < 2:
        raise ValueError("R-MATURITY needs at least two anchors")
    if len({r.step for r in rs}) != len(rs):
        raise ValueError("two reads at the same step")
    first, last, prev = rs[0], rs[-1], rs[-2]
    cells = sorted(set(first.g) & set(last.g))
    out: dict = {"first": first.label, "last": last.label, "anchors": [r.label for r in rs],
                 "GAP": {c: gap_change(first.g[c], last.g[c]) for c in cells},
                 # A1 (README FINDING 1): every BUCKET's GAP also at the Bonferroni level over the 13 buckets; only a
                 # bucket verdict that holds here is quoted as a finding
                 "GAP_BONFERRONI": {c: gap_change(first.g[c], last.g[c], BONF_LEVEL) for c in cells if c != "all"}}
    out["BELIEF"] = (belief_change(first.today, last.today) if first.today is not None and last.today is not None
                     else {"verdict": "NOT READ"})
    out["FRONTIER"] = (frontier_change(first.gates, last.gates) if first.gates and last.gates
                       else {"verdict": "NOT READ"})
    has_all = "all" in prev.g and "all" in last.g
    if not converged_on_this_bank:
        out["CONVERGED"] = {"verdict": CONVERGED_A1_REASON}
    else:
        out["CONVERGED"] = converged(prev.g["all"], last.g["all"]) if has_all else {"verdict": "NOT READ"}
    out["CONVERGED"]["between"] = [prev.label, last.label]
    out["STILL-MOVING"] = still_moving(prev.g["all"], last.g["all"]) if has_all else {"verdict": "NOT READ"}
    out["STILL-MOVING"]["between"] = [prev.label, last.label]
    return out


# ------------------------------------------------------------------------------------------------
# R3 at one anchor (A1) and BANK-DEPENDENT (§2.5)
# ------------------------------------------------------------------------------------------------

R3_MATERIAL = 0.005
R3_CLOSE = 0.01


def _direction(vec: np.ndarray) -> str:
    lo, hi = ci(vec)
    return "U" if gt(lo, 0.0) else "M" if lt(hi, 0.0) else "none"


def r3_read(bt: Boot, imp: Mapping[str, npt.ArrayLike], sel: npt.ArrayLike | None = None) -> dict:
    """E2's R3 on one anchor's states: ``imp`` = {O, F, F+M, F+U, O-M, O-U: per-state IMP at the primary scorer (NaN =
    not ok)}, ``sel`` restricts the states (a stratum). Every quantity on the SAME draws of ``bt``."""
    n = len(np.asarray(imp["O"]))
    base_sel = np.ones(n, bool) if sel is None else np.asarray(sel, bool)
    x = {k: np.asarray(v, dtype=np.float64) for k, v in imp.items()}

    def ok(*ks: str) -> np.ndarray:
        m = base_sel.copy()
        for k in ks:
            m &= np.isfinite(x[k])
        return m

    ms, mn = ok("F+U", "F+M"), ok("O-M", "O-U")
    ds = bt.mean(x["F+U"] - x["F+M"], ms)
    dn = bt.mean(x["O-M"] - x["O-U"], mn)
    det, cls = {}, {}
    for X in ("M", "U"):
        sig = bt.mean(x[f"F+{X}"] - x["F"], ok(f"F+{X}", "F"))
        nu = bt.mean(x["O"] - x[f"O-{X}"], ok("O", f"O-{X}"))
        cls[f"sigma_{X} = IMP(F+{X}) - IMP(F)"] = summary(*sig, n=int(ok(f"F+{X}", "F").sum()))
        cls[f"nu_{X} = IMP(O) - IMP(O-{X})"] = summary(*nu, n=int(ok("O", f"O-{X}").sum()))
        det[X] = bool(gt(ci(sig[1])[0], 0.0) or gt(ci(nu[1])[0], 0.0))
    d_s, d_n = _direction(ds[1]), _direction(dn[1])
    if d_s == "U" and d_n == "U":
        v = "U-FIRST"
    elif d_s == "M" and d_n == "M":
        v = "M-FIRST"
    elif det["M"] and det["U"]:
        v = "SHARED"
    else:
        v = "NOT DETECTED"
    ls, hs = ci(ds[1])
    ln, hn = ci(dn[1])
    q = {"direction_reveal_only": d_s, "direction_hide_only": d_n, "sides_disagree": {d_s, d_n} == {"U", "M"},
         "material_reveal_only": bool(gt(ls, R3_MATERIAL) or lt(hs, -R3_MATERIAL)),
         "material_hide_only": bool(gt(ln, R3_MATERIAL) or lt(hn, -R3_MATERIAL)),
         "close": bool(inside([ls, hs], R3_CLOSE) and inside([ln, hn], R3_CLOSE)),
         "classes_detected": [c for c in ("M", "U") if det[c]]}
    base = ok("O", "F")
    g = bt.mean(x["O"] - x["F"], base)
    shares = {}
    for X in ("M", "U"):
        shares[f"s_{X}"] = summary(*bt.ratio(x[f"F+{X}"] - x["F"], x["O"] - x["F"], base & ok(f"F+{X}")))
        shares[f"n_{X}"] = summary(*bt.ratio(x["O"] - x[f"O-{X}"], x["O"] - x["F"], base & ok(f"O-{X}")))
    return {"verdict": v, "label": v + (" (SIDES-DISAGREE)" if v == "SHARED" and q["sides_disagree"] else ""),
            "reveal_only Δs = IMP(F+U) - IMP(F+M)": summary(*ds, n=int(ms.sum())),
            "hide_only Δn = IMP(O-M) - IMP(O-U)": summary(*dn, n=int(mn.sum())),
            "qualifiers": q, "class_contrasts": cls, "G = IMP(O) - IMP(F)": summary(*g, n=int(base.sum())),
            "GAP": bool(gt(ci(g[1])[0], 0.0)), "shares (descriptive)": shares}


_OPPOSITE = {"GAP": ({"GAP-SHRINKS", "GAP-GROWS"},), "R3": ({"U-FIRST", "M-FIRST"},),
             "FRONTIER": ({"FRONTIER-MOVES-EARLIER", "FRONTIER-RECEDES"},)}


def bank_dependent(kind: str, record: str, replication: str) -> bool:
    """§2.5: True iff the plateau bank's verdict (``record``) and the interim bank's (``replication``) disagree in the
    registered sense: opposite directions for GAP / R3 / FRONTIER; for BELIEF, one RISES where the other does not.
    A verdict NOT READ on either bank is never BANK-DEPENDENT."""
    if "NOT READ" in record or "NOT READ" in replication:
        return False
    if kind == "BELIEF":
        return (record == "BELIEF-RISES") != (replication == "BELIEF-RISES")
    if kind not in _OPPOSITE:
        raise ValueError(f"no BANK-DEPENDENT rule for {kind}")
    pair = _OPPOSITE[kind][0]
    return record != replication and {record, replication} == pair
