"""X70, the ent-0 control of the trailing-magnet screen: the REGISTERED decision rule, its self-check, its design
simulation and its entropy labels.

Registration: ``PREREG.md`` in this directory (REGISTERED 2026-11-13, before any X70 step or game). The estimator is
CE-3d's (``../ce3d_magnet_screen_2026-11-11/screen_rule.py``, itself X67's) with the cell's pair count a parameter:
``rect_stat`` below is that file's with ``cell_var`` passed in, and the self-check reproduces CE-3d's two banked
results from its record to 1e-9. A measurement's script is a frozen record, so the rule is copied here and never
imported from CE-3d's directory.

THE DECIDING MATRIX. h[i][j] = swap fork i's score (pp, a draw = 1/2) against ent-0 fork j, one mirrored cell of
4,000 pairs, both sides greedy. Rows = the three banked swap forks ``rb_x35_ce3dswap_s7321..s7323`` (r = 3; P* + 20M,
``--ent-coef 0 --magnet-kl-coef 0.05``). Columns = the four X70 forks ``rb_x35_x70ent0_s7331..s7334`` (c = 4; P* +
20M, ``--ent-coef 0``, no magnet). Both arms trained at the pin 9e97cad5; the ONE lever between them is the anchor.

THE STATISTIC.
    D   = mean(h) - 50                                            (the swap arm's lead = the anchor's contribution, pp)
    s2R = max(var of the r row means,    CELL_VAR / c)            (ddof 1; the floor is a row mean's binomial part)
    s2C = max(var of the c column means, CELL_VAR / r)
    V   = s2R / r + s2C / c
    df  = V^2 / ((s2R / r)^2 / (r - 1) + (s2C / c)^2 / (c - 1))   (Welch-Satterthwaite)
    90 % interval = D +- t(0.95, df) * sqrt(V)
``CELL_VAR`` = 2500 / (2 * pairs) pp^2, the binomial variance of one cell at 50 %: 0.3125 at the deciding cross's
4,000 pairs, 1.25 at the reported crosses' 1,000. A mean of k cells cannot vary by less than CELL_VAR / k between
seeds, so a smaller sample variance is luck and is not believed.

THE CLASSES (one look, no look 2).
* ANCHOR-ADDS    iff the interval's LOWER end > 0   (the swap forks beat the ent-0 forks, one-sided 95 %).
* ANCHOR-COSTS   iff the interval's UPPER end < 0   (the ent-0 forks beat the swap forks).
* NOT-SEPARATED  otherwise; quoted with its upper end ("the anchor adds less than <hi> pp at this horizon").
* INCONCLUSIVE   when an input is incomplete or invalid (never interpreted; D is not computed).

RULE 8 (standing rule 8): an end within ``RULE8_EPS`` (1e-9 pp) of 0 satisfies neither ANCHOR-ADDS nor ANCHOR-COSTS
and is recorded in ``near_boundary``. A tie never classifies.

REPORTED beside the verdict, never deciding: the same interval with both variances floored at the FRESH-run seed
variance (1.76 pp)^2 (``../boundary1_scratch_2026-11-09/seed_spread.log``).

THE TWO REPORTED CROSSES (PREREG §5.4; they DECIDE NOTHING and carry NO class). Rows = the four X70 forks; columns =
the eight C0 finals ``rb_x35_c0_s3501..s3508`` (``c0``) or X67's three finals ``rb_x35_x67ent_s6701..s6703``
(``x67``); 1,000 pairs a cell. ``report`` applies the same ``rect_stat`` and returns one of three registered READINGS
per cross (lower end > 0 / upper end < 0 / neither). ``decide`` never sees these matrices and refuses their shape.

THE ENTROPY LABELS (PREREG §8; a label never changes the class). ``labels`` compares the four X70 forks' end
entropies (nats; the mean of each run's last 20 ``-train/entropy_loss`` points) with the three swap forks', read from
CE-3d's banked ``record/diag9_ce3d.json`` and never retyped:
* ENT-BELOW-SWAP iff the HIGHEST X70 end entropy < the LOWEST swap one;
* ENT-ABOVE-SWAP iff the LOWEST X70 end entropy > the HIGHEST swap one; neither when the ranges overlap;
* ENT-COLLAPSED  iff ANY X70 fork ends below half the lowest swap end entropy.
Rule 8: a difference within 1e-9 nats of a bound fires nothing and is recorded.

    python screen_rule.py self-check            # cross_stat on a banked 8 x 8, CE-3d's two banked results, hand cases
    python screen_rule.py simulate              # the registered design's operating characteristics (PREREG §6)
    python screen_rule.py simulate-designs      # the sizes that were weighed (PREREG §6.2), same model
    python screen_rule.py simulate-reported     # what the two reported crosses can see (PREREG §5.4)
    python screen_rule.py decide <h.json>       # h.json = {"h_pp": [[...], ...]} -> one JSON verdict
    python screen_rule.py report c0|x67 <h.json>            # a reported cross -> one JSON report, no class
    python screen_rule.py labels <e1> <e2> <e3> <e4>        # the four X70 end entropies (nats) -> the labels

Run with ``PYTHONPATH=<a checkout>/src`` (``main.h2h.cross`` supplies ``t_ppf`` and ``RULE8_EPS``).
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from main.h2h.cross import RULE8_EPS, cross_stat, t_ppf

HERE = Path(__file__).resolve().parent
BANKED_8X8 = HERE.parent / "endstate_closing_test_2026-10-09" / "cross" / "h.json"
CE3D_RECORD = HERE.parent / "ce3d_magnet_screen_2026-11-11" / "record"

ANCHOR_ADDS, ANCHOR_COSTS, NOT_SEPARATED, INCONCLUSIVE = "ANCHOR-ADDS", "ANCHOR-COSTS", "NOT-SEPARATED", "INCONCLUSIVE"
N_ROWS, N_COLS = 3, 4                        # the swap forks x the X70 (ent-0) forks
PAIRS = 4000
CI_LEVEL = 0.90
FRESH_SEED_SD_PP = 1.76                      # the fresh-run seed SD: the REPORTED sensitivity line only
#: the REPORTED crosses: name -> (rows = the X70 forks, columns, pairs a cell, the three readings: lower end > 0,
#: upper end < 0, neither). Readings, not classes.
REPORTED: Dict[str, Tuple[int, int, int, Tuple[str, str, str]]] = {
    "c0": (4, 8, 1000, (
        "removing the entropy bonus alone, with no anchor, beats the C0 finals at this horizon",
        "removing the entropy bonus alone, with no anchor, loses to the C0 finals at this horizon",
        "the ent-0 forks are not separated from the C0 finals at this horizon")),
    "x67": (4, 3, 1000, (
        "ent_coef 0 beats ent_coef 0.0125 at this horizon: the gain from a lower uniform push continues below 0.0125",
        "ent_coef 0 loses to ent_coef 0.0125 at this horizon: the gain from a lower uniform push turns over above 0",
        "ent_coef 0 is not separated from ent_coef 0.0125 at this horizon")),
}
ENT_BELOW, ENT_ABOVE, ENT_COLLAPSED = "ENT-BELOW-SWAP", "ENT-ABOVE-SWAP", "ENT-COLLAPSED"
ENT_EPS_NATS = 1e-9
COLLAPSE_FRACTION = 0.5                      # ENT-COLLAPSED: below this fraction of the lowest swap end entropy


def cell_var(pairs: int) -> float:
    """One cell's binomial variance at 50 %, pp^2: 2500 / games."""
    return 2500.0 / (2 * pairs)


CELL_VAR_PP2 = cell_var(PAIRS)               # 0.3125: the deciding cross's cell


def _mean(xs: Sequence[float]) -> float:
    return math.fsum(xs) / len(xs)


def _var(xs: Sequence[float]) -> float:
    m = _mean(xs)
    return math.fsum((x - m) ** 2 for x in xs) / (len(xs) - 1)


def rect_stat(h: Sequence[Sequence[float]], *, cv: float = CELL_VAR_PP2, floor_r: Optional[float] = None,
              floor_c: Optional[float] = None) -> Dict[str, Any]:
    """The statistic on an r x c matrix (pp). ``cv`` is the cell's binomial variance; the floors default to the
    registered binomial ones (``cv / c`` for a row mean, ``cv / r`` for a column mean)."""
    r, c = len(h), len(h[0])
    rows = [_mean(line) for line in h]
    cols = [_mean([h[i][j] for i in range(r)]) for j in range(c)]
    raw_r, raw_c = _var(rows), _var(cols)
    fr = cv / c if floor_r is None else floor_r
    fc = cv / r if floor_c is None else floor_c
    s2r, s2c = max(raw_r, fr), max(raw_c, fc)
    tr, tc = s2r / r, s2c / c
    v = tr + tc
    df = v * v / (tr * tr / (r - 1) + tc * tc / (c - 1))
    d = _mean([x for line in h for x in line]) - 50.0
    q = t_ppf(0.5 + CI_LEVEL / 2.0, df)
    se = math.sqrt(v)
    return {"delta_hat_pp": d, "se_pp": se, "df": df, "t_0.95": q, "interval90_pp": [d - q * se, d + q * se],
            "s2_rows_raw": raw_r, "s2_cols_raw": raw_c, "s2_rows": s2r, "s2_cols": s2c,
            "floor_rows_bound": raw_r < fr, "floor_cols_bound": raw_c < fc,
            "row_means_pp": rows, "col_means_pp": cols, "shape": [r, c], "cell_var_pp2": cv}


def _shape_problems(h: Sequence[Sequence[float]], r: int, c: int, what: str) -> List[str]:
    if len(h) != r or any(len(line) != c for line in h):
        return [f"{what} is {r} x {c}; got {[len(line) for line in h]}"]
    if not all(math.isfinite(x) and 0.0 <= x <= 100.0 for line in h for x in line):
        return ["a cell is not a finite score in [0, 100] pp"]
    return []


def _ends(lo: float, hi: float) -> Tuple[List[str], bool, bool]:
    """Rule 8 on the two ends: (near-boundary names, lower end strictly above 0, upper end strictly below 0)."""
    near = [name for name, end in (("lower end vs 0", lo), ("upper end vs 0", hi)) if abs(end) <= RULE8_EPS]
    return near, lo > 0.0 and "lower end vs 0" not in near, hi < 0.0 and "upper end vs 0" not in near


def decide(h: Sequence[Sequence[float]], inconclusive: Sequence[str] = ()) -> Dict[str, Any]:
    """The registered verdict on ``h`` (rows = the swap forks, columns = the X70 forks; pp; 4,000 pairs a cell)."""
    reasons: List[str] = list(inconclusive)
    if not reasons:
        reasons = _shape_problems(h, N_ROWS, N_COLS, "the registered read (the swap forks x the X70 forks)")
    if reasons:
        return {"outcome": INCONCLUSIVE, "reasons": reasons}
    st = rect_stat(h)
    lo, hi = st["interval90_pp"]
    near, above, below = _ends(lo, hi)
    outcome = ANCHOR_ADDS if above else ANCHOR_COSTS if below else NOT_SEPARATED
    fresh = rect_stat(h, floor_r=FRESH_SEED_SD_PP ** 2, floor_c=FRESH_SEED_SD_PP ** 2)
    return {"outcome": outcome, "near_boundary": near, **st,
            "reported": {"interval90_fresh_floor_pp": fresh["interval90_pp"], "fresh_floor_df": fresh["df"],
                         "fresh_floor_se_pp": fresh["se_pp"]}}


def report(h: Sequence[Sequence[float]], cross: str, inconclusive: Sequence[str] = ()) -> Dict[str, Any]:
    """A REPORTED cross (rows = the X70 forks; columns = the C0 finals or X67's finals; pp; 1,000 pairs a cell).
    It decides nothing: no class, only D, its interval and the registered reading for the interval's position."""
    r, c, pairs, readings = REPORTED[cross]
    reasons: List[str] = list(inconclusive)
    if not reasons:
        reasons = _shape_problems(h, r, c, f"the reported cross '{cross}' (the X70 forks x its columns)")
    if reasons:
        return {"decides": "nothing", "cross": cross, "read": False, "reasons": reasons}
    st = rect_stat(h, cv=cell_var(pairs))
    lo, hi = st["interval90_pp"]
    near, above, below = _ends(lo, hi)
    return {"decides": "nothing", "cross": cross, "read": True,
            "reading": readings[0] if above else readings[1] if below else readings[2],
            "lower_end_above_0": above, "upper_end_below_0": below, "near_boundary": near, **st}


def swap_end_entropies() -> List[float]:
    """The three swap forks' end entropies (nats), from CE-3d's banked diagnostics; never retyped."""
    rec = json.loads((CE3D_RECORD / "diag9_ce3d.json").read_text())["summary"]["swap_end_entropy"]
    return [float(rec[f"rb_x35_ce3dswap_s{s}"]) for s in (7321, 7322, 7323)]


def labels(x70_ends: Sequence[float], swap_ends: Optional[Sequence[float]] = None) -> Dict[str, Any]:
    """The entropy labels of PREREG §8 from the four X70 forks' end entropies (nats). A label never moves a class."""
    swap = list(swap_end_entropies() if swap_ends is None else swap_ends)
    xs = [float(x) for x in x70_ends]
    if len(xs) != N_COLS or len(swap) != N_ROWS or not all(math.isfinite(x) and x >= 0.0 for x in xs + swap):
        return {"labels": None, "reasons": [f"needs {N_COLS} X70 and {N_ROWS} swap end entropies, finite and >= 0; "
                                            f"got {xs} and {swap}"]}
    s_lo, s_hi = min(swap), max(swap)
    bar = COLLAPSE_FRACTION * s_lo
    near = [name for name, d in (("highest X70 vs lowest swap", max(xs) - s_lo), ("lowest X70 vs highest swap", min(xs) - s_hi))
            if abs(d) <= ENT_EPS_NATS]
    near += [f"X70 fork {i + 1} vs the collapse bar" for i, x in enumerate(xs) if abs(x - bar) <= ENT_EPS_NATS]
    below = max(xs) < s_lo - ENT_EPS_NATS
    above = min(xs) > s_hi + ENT_EPS_NATS
    collapsed = [i + 1 for i, x in enumerate(xs) if x < bar - ENT_EPS_NATS]
    got = ([ENT_BELOW] if below else []) + ([ENT_ABOVE] if above else []) + ([ENT_COLLAPSED] if collapsed else [])
    return {"labels": got, "range": ENT_BELOW if below else ENT_ABOVE if above else "neither (the ranges overlap)",
            "collapsed_forks": collapsed, "x70_end_entropy": xs, "swap_end_entropy": swap,
            "swap_lowest": s_lo, "swap_highest": s_hi, "collapse_bar_nats": bar, "within_1e-9_of_a_bound": near}


def self_check() -> Dict[str, Any]:
    """(1) On a banked 8 x 8 with no floor binding, D and V equal ``cross_stat``'s. (2) ``rect_stat`` at 1,000 pairs
    reproduces CE-3d's two banked results (its deciding 3 x 8 and its reported 3 x 3) to 1e-9. (3) Hand cases for
    each class at 3 x 4, the rule-8 tie, the 4,000-pair floors and the INCONCLUSIVE shapes. (4) ``report``: its three
    readings per cross, its refusals, never a class, and ``decide`` refuses both reported shapes. (5) The labels."""
    probs: List[str] = []
    h8 = json.loads(BANKED_8X8.read_text())["h_pp"]
    ours, ref = rect_stat(h8, cv=cell_var(1000)), cross_stat(h8)
    if ours["floor_rows_bound"] or ours["floor_cols_bound"]:
        probs.append("a floor bound on the banked 8 x 8 (the equality check needs none)")
    if abs(ours["delta_hat_pp"] - ref.delta_hat) > 1e-9 or abs(ours["se_pp"] - ref.se) > 1e-9:
        probs.append(f"D / SE {ours['delta_hat_pp']!r} / {ours['se_pp']!r} != cross_stat's {ref.delta_hat!r} / {ref.se!r}")

    banked: Dict[str, Any] = {}
    for name, sub, fname in (("ce3d_deciding", "deciding", "verdict.json"), ("ce3d_x67", "x67", "report_x67.json")):
        hh = json.loads((CE3D_RECORD / sub / "h.json").read_text())["h_pp"]
        want = json.loads((CE3D_RECORD / sub / fname).read_text())
        again = rect_stat(hh, cv=cell_var(1000))
        for k in ("delta_hat_pp", "se_pp", "df"):
            if abs(again[k] - want[k]) > 1e-9:
                probs.append(f"{name}: banked {k} {want[k]!r}; this rule reads {again[k]!r}")
        if any(abs(x - y) > 1e-9 for x, y in zip(again["interval90_pp"], want["interval90_pp"])):
            probs.append(f"{name}: banked interval {want['interval90_pp']}; this rule reads {again['interval90_pp']}")
        if (again["floor_rows_bound"], again["floor_cols_bound"]) != (want["floor_rows_bound"], want["floor_cols_bound"]):
            probs.append(f"{name}: the floors bind differently here")
        banked[name] = {"delta_hat_pp": again["delta_hat_pp"], "interval90_pp": again["interval90_pp"], "df": again["df"],
                        "row_means_var_pp2": [again["s2_rows_raw"], "floor", cell_var(1000) / again["shape"][1]],
                        "col_means_var_pp2": [again["s2_cols_raw"], "floor", cell_var(1000) / again["shape"][0]]}
    fresh = rect_stat(json.loads((CE3D_RECORD / "deciding" / "h.json").read_text())["h_pp"], cv=cell_var(1000),
                      floor_r=FRESH_SEED_SD_PP ** 2, floor_c=FRESH_SEED_SD_PP ** 2)["interval90_pp"]
    want_fresh = json.loads((CE3D_RECORD / "deciding" / "verdict.json").read_text())["reported"]["interval90_fresh_floor_pp"]
    if any(abs(x - y) > 1e-9 for x, y in zip(fresh, want_fresh)):
        probs.append(f"ce3d_deciding: banked fresh-floor interval {want_fresh}; this rule reads {fresh}")

    def grid(d: float, row_off: Sequence[float], col_off: Sequence[float]) -> List[List[float]]:
        return [[50.0 + d + a + b for b in col_off] for a in row_off]

    cols4 = (-0.6, -0.2, 0.2, 0.6)
    cases = {
        "ANCHOR-ADDS (+2, rows +-0.5)": (grid(2.0, (-0.5, 0.0, 0.5), cols4), ANCHOR_ADDS),
        "ANCHOR-COSTS (-2, rows +-0.5)": (grid(-2.0, (-0.5, 0.0, 0.5), cols4), ANCHOR_COSTS),
        "NOT-SEPARATED (+0.5, rows +-1)": (grid(0.5, (-1.0, 0.0, 1.0), cols4), NOT_SEPARATED),
        "NOT-SEPARATED (-0.5, rows +-1)": (grid(-0.5, (-1.0, 0.0, 1.0), cols4), NOT_SEPARATED),
        "NOT-SEPARATED (+2, rows +-3: a wide arm)": (grid(2.0, (-3.0, 0.0, 3.0), cols4), NOT_SEPARATED),
    }
    got: Dict[str, Any] = {}
    for name, (h, want_cls) in cases.items():
        d = decide(h)
        got[name] = [d["outcome"], [round(x, 3) for x in d["interval90_pp"]], round(d["df"], 2)]
        if d["outcome"] != want_cls:
            probs.append(f"{name}: {d['outcome']} != {want_cls}")
    # the floors at 4,000 pairs: identical rows and identical columns have zero sample variance; both floors bind
    flat = decide(grid(0.3, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0)))
    if not (flat["floor_rows_bound"] and flat["floor_cols_bound"]) or abs(flat["s2_rows"] - 0.3125 / 4) > 1e-12 \
            or abs(flat["s2_cols"] - 0.3125 / 3) > 1e-12 or flat["outcome"] != NOT_SEPARATED:
        probs.append(f"a flat matrix: the 4,000-pair floors did not bind as registered ({flat})")
    got["floors (a flat matrix at +0.3)"] = [flat["outcome"], [round(x, 3) for x in flat["interval90_pp"]],
                                             round(flat["se_pp"], 4)]
    # rule 8: shift a classified matrix so an end lands on 0 to the last bit we can reach; it must not classify
    base = grid(2.0, (-0.5, 0.0, 0.5), cols4)
    lo = decide(base)["interval90_pp"][0]
    tie = decide([[x - lo for x in line] for line in base])
    if abs(tie["interval90_pp"][0]) > RULE8_EPS or tie["outcome"] != NOT_SEPARATED or not tie["near_boundary"]:
        probs.append(f"rule 8: a lower end of {tie['interval90_pp'][0]!r} read {tie['outcome']}")
    got["rule 8 (lower end = 0)"] = [tie["outcome"], tie["near_boundary"]]
    neg = grid(-2.0, (-0.5, 0.0, 0.5), cols4)
    hi = decide(neg)["interval90_pp"][1]
    tie_hi = decide([[x - hi for x in line] for line in neg])
    if abs(tie_hi["interval90_pp"][1]) > RULE8_EPS or tie_hi["outcome"] != NOT_SEPARATED or not tie_hi["near_boundary"]:
        probs.append(f"rule 8: an upper end of {tie_hi['interval90_pp'][1]!r} read {tie_hi['outcome']}")
    got["rule 8 (upper end = 0)"] = [tie_hi["outcome"], tie_hi["near_boundary"]]
    for name, h in (("2 rows", base[:2]), ("3 columns", [line[:3] for line in base]),
                    ("5 columns", grid(0.0, (-0.5, 0.0, 0.5), cols4 + (1.0,))), ("ragged", [base[0], base[1], base[2][:3]]),
                    ("nan", [[float("nan")] * 4] * 3), ("101 pp", [[101.0] * 4] * 3)):
        if decide(h)["outcome"] != INCONCLUSIVE:
            probs.append(f"{name}: not INCONCLUSIVE")
    if decide(base, inconclusive=["missing cell"])["outcome"] != INCONCLUSIVE:
        probs.append("a missing cell: not INCONCLUSIVE")
    # (4) the REPORTED crosses: a reading, never a class
    got_rep: Dict[str, Any] = {}
    cols8 = (-1.4, -1.0, -0.6, -0.2, 0.2, 0.6, 1.0, 1.4)
    rows4 = (-0.6, -0.2, 0.2, 0.6)
    for cross, col_off in (("c0", cols8), ("x67", (-0.5, 0.0, 0.5))):
        readings = REPORTED[cross][3]
        for d_true, want_i in ((4.0, 0), (-4.0, 1), (0.0, 2)):
            rp = report(grid(d_true, rows4, col_off), cross)
            if not rp["read"] or rp["reading"] != readings[want_i] or "outcome" in rp \
                    or rp["lower_end_above_0"] != (want_i == 0) or rp["upper_end_below_0"] != (want_i == 1):
                probs.append(f"report {cross} at {d_true:+g}: {rp}")
            else:
                got_rep[f"{cross} at {d_true:+g}"] = [rp["reading"], [round(x, 3) for x in rp["interval90_pp"]], round(rp["df"], 2)]
        full = grid(4.0, rows4, col_off)
        for name, h in (("3 rows", full[:3]), ("a column short", [line[:-1] for line in full]),
                        ("nan", [[float("nan")] * len(col_off)] * 4)):
            if report(h, cross)["read"]:
                probs.append(f"report {cross} {name}: read")
        if report(full, cross, inconclusive=["missing cell"])["read"]:
            probs.append(f"report {cross} a missing cell: read")
        if decide(full)["outcome"] != INCONCLUSIVE:
            probs.append(f"decide read the reported {cross} matrix as a class")
        lo_r = report(full, cross)["interval90_pp"][0]
        tie_r = report([[x - lo_r for x in line] for line in full], cross)
        if tie_r["reading"] != readings[2] or not tie_r["near_boundary"]:
            probs.append(f"report {cross} rule 8: a lower end of {tie_r['interval90_pp'][0]!r} read {tie_r['reading']!r}")
    if report(base, "c0")["read"] or report(base, "x67")["read"]:
        probs.append("report read the deciding 3 x 4 shape")
    # (5) the entropy labels, on fixed swap ends 0.48 / 0.49 / 0.50 (the banked ones are read, and printed, below)
    sw = (0.48, 0.49, 0.50)
    lab_cases = {
        "all below": ((0.30, 0.35, 0.40, 0.47), [ENT_BELOW]),
        "all above": ((0.51, 0.55, 0.60, 0.70), [ENT_ABOVE]),
        "overlap (one inside the swap's range)": ((0.40, 0.45, 0.485, 0.47), []),
        "overlap (straddles)": ((0.40, 0.45, 0.47, 0.55), []),
        "below and one collapsed": ((0.10, 0.35, 0.40, 0.47), [ENT_BELOW, ENT_COLLAPSED]),
        "collapsed without ENT-BELOW (one fork above the swap's lowest)": ((0.10, 0.35, 0.40, 0.49), [ENT_COLLAPSED]),
        "rule 8: the highest X70 equals the lowest swap": ((0.30, 0.35, 0.40, 0.48), []),
        "rule 8: a fork exactly on the collapse bar": ((0.24, 0.35, 0.40, 0.47), [ENT_BELOW]),
    }
    got_lab: Dict[str, Any] = {}
    for name, (xs, want_l) in lab_cases.items():
        lb = labels(xs, sw)
        got_lab[name] = [lb["labels"], lb["within_1e-9_of_a_bound"]]
        if lb["labels"] != want_l or (name.startswith("rule 8") and not lb["within_1e-9_of_a_bound"]):
            probs.append(f"labels {name}: {lb}")
    if labels((0.3, 0.3, 0.3), sw)["labels"] is not None or labels((0.3, 0.3, 0.3, float("nan")), sw)["labels"] is not None:
        probs.append("labels read three forks or a nan")
    banked_swap = swap_end_entropies()
    return {"self_check": "OK" if not probs else "MISMATCH", "problems": probs, "banked_8x8": {
        "delta_hat_pp": ours["delta_hat_pp"], "se_pp": ours["se_pp"], "welch_df": ours["df"], "cross_stat_df": ref.df},
        "ce3d_banked": banked, "cases": got, "reported_cases": got_rep, "label_cases": got_lab,
        "swap_end_entropy_banked": banked_swap, "collapse_bar_nats": COLLAPSE_FRACTION * min(banked_swap)}


#: (sig_R, sig_C, tau) scenarios, pp: the seed SD of the row arm's forks, of the column arm's, and a fork-by-fork
#: interaction that does NOT shrink with pairs. See ``simulate``.
SCENARIOS = ((0.0, 0.0, 0.0), (0.3, 0.3, 0.0), (0.4, 0.4, 0.0), (0.5, 0.5, 0.0), (0.5, 0.5, 0.3), (0.5, 1.0, 0.0),
             (1.0, 0.5, 0.0), (1.0, 1.0, 0.0), (1.76, 1.76, 0.0))
DELTAS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0)
SIG_E_1000 = 1.05


def _sim_one(rng: Any, reps: int, r: int, c: int, pairs: int, sig_r: float, sig_c: float, tau: float) -> Dict[str, Any]:
    import numpy as np
    from scipy.stats import t as student_t

    cv = cell_var(pairs)
    sig_e = math.sqrt(SIG_E_1000 ** 2 * 1000.0 / pairs + tau ** 2)
    a = rng.normal(0, sig_r, (reps, r))
    b = rng.normal(0, sig_c, (reps, c))
    e = rng.normal(0, sig_e, (reps, r, c))
    base = a[:, :, None] - b[:, None, :] + e
    s2r = np.maximum(base.mean(axis=2).var(axis=1, ddof=1), cv / c)
    s2c = np.maximum(base.mean(axis=1).var(axis=1, ddof=1), cv / r)
    tr, tc = s2r / r, s2c / c
    v = tr + tc
    df = v * v / (tr * tr / (r - 1) + tc * tc / (c - 1))
    hw = student_t.ppf(0.95, df) * np.sqrt(v)
    m = base.mean(axis=(1, 2))
    return {"rows": r, "cols": c, "pairs": pairs, "sigma_rows": sig_r, "sigma_cols": sig_c, "tau": tau,
            "median_half_width_pp": round(float(np.median(hw)), 2),
            "p_lower_above_0": [round(float(((m + d - hw) > 0).mean()), 3) for d in DELTAS],
            "p_upper_below_0_at_0": round(float(((m + hw) < 0).mean()), 3),
            "p_upper_below_0_at_minus1": round(float(((m - 1.0 + hw) < 0).mean()), 3),
            "p_upper_below_0_at_minus2": round(float(((m - 2.0 + hw) < 0).mean()), 3)}


def simulate(reps: int = 50000, seed: int = 20261113,
             designs: Sequence[Tuple[int, int, int]] = ((N_ROWS, N_COLS, PAIRS),),
             scenarios: Sequence[Tuple[float, float, float]] = SCENARIOS) -> Dict[str, Any]:
    """P(lower end > 0) / P(upper end < 0) under h_ij = 50 + D + a_i - b_j + e_ij, a ~ N(0, sig_R^2) (a row fork's
    strength), b ~ N(0, sig_C^2) (a column fork's), e ~ N(0, 1.05^2 * 1000 / pairs + tau^2).

    1.05 pp is the cell noise the banked 1,000-pair crosses measured (0.80-1.10 pp; binomial 1.118). The model
    ASSUMES it shrinks as 1 / sqrt(pairs): a 4,000-pair cell is eight 500-pair batches on eight distinct schedule
    seeds, so the assumption is the binomial one, but no banked cell has more than 1,000 pairs to check it against.
    ``tau`` is the part that would NOT shrink (a fork-by-fork interaction); one scenario carries 0.3 pp of it.

    The seed SDs. The swap forks' row means varied by LESS than their binomial floor in both of CE-3d's crosses, and
    X67's forks likewise (the self-check prints the four variances and floors), so 0.3-0.4 pp is the central case;
    C0's three G1 finals read 0.48 pp on S (X48 PREREG §6), so 0.5 is the fork evidence's upper figure; an ent-0
    fork has never been run, so the column SD runs to 1.0, and 1.76 pp (the FRESH-run pooled figure,
    ``../boundary1_scratch_2026-11-09/seed_spread.log``) is the pessimistic case no fork has shown."""
    import numpy as np

    rng = np.random.default_rng(seed)
    out: Dict[str, Any] = {"reps": reps, "seed": seed, "sigma_e_at_1000_pairs": SIG_E_1000, "deltas_pp": list(DELTAS),
                           "scenarios": []}
    for r, c, pairs in designs:
        for sig_r, sig_c, tau in scenarios:
            out["scenarios"].append(_sim_one(rng, reps, r, c, pairs, sig_r, sig_c, tau))
    return out


#: PREREG §6.2: the sizes weighed. (rows, cols, pairs); rows = swap forks (three are banked; six would need three NEW
#: swap forks), cols = ent-0 forks.
DESIGNS_WEIGHED = ((3, 3, 1000), (3, 3, 4000), (3, 4, 1000), (3, 4, 2000), (3, 4, 4000), (3, 4, 8000), (3, 5, 4000),
                   (3, 6, 4000), (6, 6, 2000))
SCENARIOS_WEIGHED = ((0.3, 0.3, 0.0), (0.5, 0.5, 0.0))
#: PREREG §5.4: the two reported crosses, rows = the four X70 forks.
DESIGNS_REPORTED = ((4, 8, 1000), (4, 3, 1000))


def _print_sim(res: Dict[str, Any], title: str) -> None:
    print(f"{title}\nreps {res['reps']}, rng {res['seed']}, cell noise SD {res['sigma_e_at_1000_pairs']} pp at 1,000 pairs "
          f"(x sqrt(1000 / pairs)), plus tau")
    print("rows cols  pairs  sig_R  sig_C   tau  half-width  P(lower end > 0) at D = "
          + " / ".join(f"{d:+g}" for d in res["deltas_pp"]) + "   P(upper end < 0) at 0 / -1 / -2")
    for s in res["scenarios"]:
        print(f"{s['rows']:>4} {s['cols']:>4}  {s['pairs']:>5}  {s['sigma_rows']:>5.2f}  {s['sigma_cols']:>5.2f}  "
              f"{s['tau']:>4.2f}  {s['median_half_width_pp']:>10.2f}  "
              + " / ".join(f"{p:.3f}" for p in s["p_lower_above_0"])
              + f"   {s['p_upper_below_0_at_0']:.3f} / {s['p_upper_below_0_at_minus1']:.3f} / {s['p_upper_below_0_at_minus2']:.3f}")


def main(argv: Sequence[str]) -> int:
    cmd = argv[1] if len(argv) > 1 else "self-check"
    if cmd == "self-check":
        res = self_check()
        print(json.dumps(res, indent=1))
        return 0 if res["self_check"] == "OK" else 1
    if cmd == "simulate":
        _print_sim(simulate(), "THE REGISTERED DECIDING CROSS: 3 swap forks x 4 X70 forks, 4,000 pairs a cell. "
                               "P(lower end > 0) = P(ANCHOR-ADDS); P(upper end < 0) = P(ANCHOR-COSTS)")
        return 0
    if cmd == "simulate-designs":
        _print_sim(simulate(seed=20261114, designs=DESIGNS_WEIGHED, scenarios=SCENARIOS_WEIGHED),
                   "THE SIZES WEIGHED (PREREG §6.2): rows = swap forks, cols = ent-0 forks")
        return 0
    if cmd == "simulate-reported":
        _print_sim(simulate(seed=20261115, designs=DESIGNS_REPORTED),
                   "THE REPORTED CROSSES (no class): 4 X70 forks x 8 C0 finals, and x 3 X67 finals, 1,000 pairs a cell")
        return 0
    if cmd == "decide":
        print(json.dumps(decide(json.loads(Path(argv[2]).read_text())["h_pp"]), indent=1))
        return 0
    if cmd == "report" and len(argv) > 3 and argv[2] in REPORTED:
        print(json.dumps(report(json.loads(Path(argv[3]).read_text())["h_pp"], argv[2]), indent=1))
        return 0
    if cmd == "labels" and len(argv) == 2 + N_COLS:
        res = labels([float(x) for x in argv[2:]])
        print(json.dumps(res, indent=1))
        return 0 if res["labels"] is not None else 1
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
