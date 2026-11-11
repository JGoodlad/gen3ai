"""CE-3d, the trailing-magnet screen: the REGISTERED decision rule, its self-check, and its design simulation.

Registration: ``PREREG.md`` in this directory (REGISTERED 2026-11-11, before any CE-3d step).

THREE MATRICES, one estimator (X67's, ``../x67_entropy_screen_2026-11-09/screen_rule.py``, copied here with the
cell size as a parameter; the self-check reproduces X67's banked verdict from its record to 1e-9).

* DECIDING, ``decide``: h[i][j] = MAGNET fork i's score (pp, a draw = 1/2) against CONTROL fork j, one mirrored
  cell of 2,000 pairs, both sides greedy. Exactly 3 x 3.
* REPORTED, ``report``: the MAGNET forks against the C0 finals, and the CONTROL forks against the C0 finals, each
  3 x c (5 <= c <= 8), 1,000 pairs a cell: X67's own shape and cell size, so each D sits beside X67's +1.97 pp.
  ``report`` returns numbers and where the interval lies; it never returns a class.

THE STATISTIC (r rows, c columns, n mirrored pairs a cell).
    D   = mean(h) - 50                                            (the row arm's lead over the column arm, pp)
    s2R = max(var of the r row means,    CELL_VAR(n) / c)         (ddof 1; the floor is a row mean's binomial part)
    s2C = max(var of the c column means, CELL_VAR(n) / r)
    V   = s2R / r + s2C / c
    df  = V^2 / ((s2R / r)^2 / (r - 1) + (s2C / c)^2 / (c - 1))   (Welch-Satterthwaite)
    90 % interval = D +- t(0.95, df) * sqrt(V)
``CELL_VAR(n)`` = 2500 / (2 n) pp^2, the binomial variance of one 2n-game cell at 50 %: 0.625 at n = 2,000, 1.25 at
n = 1,000. A mean of k cells cannot vary by less than CELL_VAR / k between seeds, so a smaller sample variance is
luck and is not believed.

THE CLASSES of the deciding matrix (one look, no look 2).
* PASS       iff the interval's LOWER end > 0     (the magnet forks beat their entropy-matched controls).
* WORSE      iff the interval's UPPER end < 0.
* NOT-SHOWN  otherwise.
* INCONCLUSIVE when an input is incomplete or invalid (never interpreted).

RULE 8 (standing rule 8): an end within ``RULE8_EPS`` (1e-9 pp) of 0 satisfies neither PASS nor WORSE and is
recorded in ``near_boundary``. A tie never passes.

REPORTED beside the verdict, never deciding: the deciding interval with both variances floored at the FRESH-run seed
variance (1.76 pp)^2; the two matrices against C0; and their difference D(magnet vs C0) - D(control vs C0), the
INDIRECT estimate of the same contrast (a sign that disagrees with the direct D is reported as non-transitivity).

    python screen_rule.py self-check                 # cross_stat on a banked 8 x 8, X67's banked verdict, hand cases
    python screen_rule.py simulate                   # the operating characteristics (PREREG §6), deterministic seed
    python screen_rule.py decide <h.json>            # {"h_pp": [[...] x 3] x 3} -> the verdict JSON
    python screen_rule.py report <h.json>            # {"h_pp": 3 x c, "col_seeds": [...]} -> the reported numbers

Run with ``PYTHONPATH=<a checkout>/src`` (``main.h2h.cross`` supplies ``t_ppf`` and ``RULE8_EPS``).
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from main.h2h.cross import RULE8_EPS, cross_stat, t_ppf

HERE = Path(__file__).resolve().parent
BANKED_8X8 = HERE.parent / "endstate_closing_test_2026-10-09" / "cross" / "h.json"
X67_RECORD = HERE.parent / "x67_entropy_screen_2026-11-09" / "record"

PASS, WORSE, NOT_SHOWN, INCONCLUSIVE = "PASS", "WORSE", "NOT-SHOWN", "INCONCLUSIVE"
N_ROWS = 3
DECIDING_COLS = 3
DECIDING_PAIRS = 2000
REPORTED_MIN_COLS, REPORTED_MAX_COLS = 5, 8
REPORTED_PAIRS = 1000
CI_LEVEL = 0.90
FRESH_SEED_SD_PP = 1.76                      # the fresh-run seed SD: the REPORTED sensitivity line only
TOPUP_SEEDS = (3506, 3507, 3508)


def cell_var(pairs: int) -> float:
    """One cell's binomial variance at 50 %, pp^2: 2 * pairs games."""
    return 2500.0 / (2 * pairs)


def _mean(xs: Sequence[float]) -> float:
    return math.fsum(xs) / len(xs)


def _var(xs: Sequence[float]) -> float:
    m = _mean(xs)
    return math.fsum((x - m) ** 2 for x in xs) / (len(xs) - 1)


def rect_stat(h: Sequence[Sequence[float]], pairs: int, *, floor_r: Optional[float] = None,
              floor_c: Optional[float] = None) -> Dict[str, Any]:
    """The statistic on an r x c matrix (pp) of ``pairs``-pair cells. The floors default to the binomial ones."""
    r, c = len(h), len(h[0])
    rows = [_mean(line) for line in h]
    cols = [_mean([h[i][j] for i in range(r)]) for j in range(c)]
    raw_r, raw_c = _var(rows), _var(cols)
    fr = cell_var(pairs) / c if floor_r is None else floor_r
    fc = cell_var(pairs) / r if floor_c is None else floor_c
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
            "row_means_pp": rows, "col_means_pp": cols, "shape": [r, c], "pairs_per_cell": pairs}


def _shape_problems(h: Sequence[Sequence[float]], min_cols: int, max_cols: int, what: str) -> List[str]:
    if len(h) != N_ROWS:
        return [f"{what} has {N_ROWS} rows; got {len(h)}"]
    if any(len(line) != len(h[0]) for line in h) or not min_cols <= len(h[0]) <= max_cols:
        return [f"{what} has {min_cols}..{max_cols} columns, every row the same; got {[len(line) for line in h]}"]
    if not all(math.isfinite(x) and 0.0 <= x <= 100.0 for line in h for x in line):
        return ["a cell is not a finite score in [0, 100] pp"]
    return []


def _side(lo: float, hi: float) -> Dict[str, Any]:
    """Where a 90 % interval lies against 0, with rule 8: ``above`` / ``below`` / ``spans``."""
    near = []
    if abs(lo) <= RULE8_EPS:
        near.append("lower end vs 0")
    if abs(hi) <= RULE8_EPS:
        near.append("upper end vs 0")
    if lo > 0.0 and "lower end vs 0" not in near:
        side = "above"
    elif hi < 0.0 and "upper end vs 0" not in near:
        side = "below"
    else:
        side = "spans"
    return {"side": side, "near_boundary": near}


def decide(h: Sequence[Sequence[float]], inconclusive: Sequence[str] = ()) -> Dict[str, Any]:
    """The registered verdict on the DECIDING matrix (rows = the magnet forks, columns = the control forks; pp)."""
    reasons: List[str] = list(inconclusive)
    if not reasons:
        reasons = _shape_problems(h, DECIDING_COLS, DECIDING_COLS, "the deciding read (magnet forks x control forks)")
    if reasons:
        return {"outcome": INCONCLUSIVE, "reasons": reasons}
    st = rect_stat(h, DECIDING_PAIRS)
    where = _side(*st["interval90_pp"])
    outcome = {"above": PASS, "below": WORSE, "spans": NOT_SHOWN}[where["side"]]
    fresh = rect_stat(h, DECIDING_PAIRS, floor_r=FRESH_SEED_SD_PP ** 2, floor_c=FRESH_SEED_SD_PP ** 2)
    return {"outcome": outcome, "near_boundary": where["near_boundary"], **st,
            "reported": {"interval90_fresh_floor_pp": fresh["interval90_pp"], "fresh_floor_df": fresh["df"],
                         "fresh_floor_se_pp": fresh["se_pp"]}}


def report(h: Sequence[Sequence[float]], inconclusive: Sequence[str] = (),
           col_seeds: Optional[Sequence[int]] = None) -> Dict[str, Any]:
    """A REPORTED matrix against the C0 finals (3 x c, 1,000 pairs a cell): X67's statistic, and no class."""
    reasons: List[str] = list(inconclusive)
    if not reasons:
        reasons = _shape_problems(h, REPORTED_MIN_COLS, REPORTED_MAX_COLS, "a reported read (forks x C0 finals)")
    if reasons:
        return {"reported_only": True, "available": False, "reasons": reasons}
    st = rect_stat(h, REPORTED_PAIRS)
    out: Dict[str, Any] = {"reported_only": True, "available": True, **_side(*st["interval90_pp"]), **st}
    if col_seeds is not None and len(col_seeds) == len(h[0]):
        top = [j for j, s in enumerate(col_seeds) if int(s) in TOPUP_SEEDS]
        bank = [j for j in range(len(col_seeds)) if j not in top]
        for name, idx in (("delta_banked_cols_pp", bank), ("delta_topup_cols_pp", top)):
            out[name] = (_mean([h[i][j] for i in range(len(h)) for j in idx]) - 50.0) if idx else None
    return out


def indirect(mag_vs_c0: Dict[str, Any], ent_vs_c0: Dict[str, Any], deciding: Dict[str, Any]) -> Dict[str, Any]:
    """REPORTED: D(magnet vs C0) - D(control vs C0), beside the direct D. No interval is registered for it (the two
    matrices share their columns and no pairing of rows exists); a sign that differs from the direct D's is named."""
    if not (mag_vs_c0.get("available") and ent_vs_c0.get("available")):
        return {"reported_only": True, "available": False}
    ind = mag_vs_c0["delta_hat_pp"] - ent_vs_c0["delta_hat_pp"]
    out: Dict[str, Any] = {"reported_only": True, "available": True, "indirect_delta_pp": ind}
    if deciding.get("outcome") != INCONCLUSIVE:
        direct = deciding["delta_hat_pp"]
        out.update({"direct_delta_pp": direct, "direct_minus_indirect_pp": direct - ind,
                    "signs_agree": (direct > 0) == (ind > 0) if direct != 0 and ind != 0 else None})
    return out


def self_check() -> Dict[str, Any]:
    """(1) On a banked 8 x 8 with no floor binding, D and V equal ``cross_stat``'s. (2) ``report`` on X67's banked
    matrix reproduces X67's banked verdict (D, interval, df, the floors). (3) Hand cases for each class of the 3 x 3,
    the rule-8 tie, the floors and the INCONCLUSIVE shapes."""
    probs: List[str] = []
    h8 = json.loads(BANKED_8X8.read_text())["h_pp"]
    ours, ref = rect_stat(h8, 1000), cross_stat(h8)
    if ours["floor_rows_bound"] or ours["floor_cols_bound"]:
        probs.append("a floor bound on the banked 8 x 8 (the equality check needs none)")
    if abs(ours["delta_hat_pp"] - ref.delta_hat) > 1e-9 or abs(ours["se_pp"] - ref.se) > 1e-9:
        probs.append(f"D / SE {ours['delta_hat_pp']!r} / {ours['se_pp']!r} != cross_stat's {ref.delta_hat!r} / {ref.se!r}")

    x67_h = json.loads((X67_RECORD / "h.json").read_text())
    x67_v = json.loads((X67_RECORD / "verdict.json").read_text())
    rep = report(x67_h["h_pp"], col_seeds=x67_h["col_seeds"])
    for k in ("delta_hat_pp", "se_pp", "df"):
        if abs(rep[k] - x67_v[k]) > 1e-9:
            probs.append(f"X67's banked {k}: {rep[k]!r} != {x67_v[k]!r}")
    if any(abs(a - b) > 1e-9 for a, b in zip(rep["interval90_pp"], x67_v["interval90_pp"])):
        probs.append(f"X67's banked interval: {rep['interval90_pp']} != {x67_v['interval90_pp']}")
    if rep["side"] != "above" or x67_v["outcome"] != PASS:
        probs.append(f"X67's banked verdict is PASS; this estimator's interval lies {rep['side']}")
    if (rep["floor_rows_bound"], rep["floor_cols_bound"]) != (x67_v["floor_rows_bound"], x67_v["floor_cols_bound"]):
        probs.append("X67's banked floors bind differently here")

    def grid(d: float, row_off: Sequence[float], col_off: Sequence[float]) -> List[List[float]]:
        return [[50.0 + d + a + b for b in col_off] for a in row_off]

    off1, off4 = (-1.0, 0.0, 1.0), (-4.0, 0.0, 4.0)
    cases = {
        "PASS (+4, rows and columns +-1)": (grid(4.0, off1, off1), PASS),
        "WORSE (-4, rows and columns +-1)": (grid(-4.0, off1, off1), WORSE),
        "NOT-SHOWN (+1, rows and columns +-1)": (grid(1.0, off1, off1), NOT_SHOWN),
        "NOT-SHOWN (+4, rows +-4: a wide arm)": (grid(4.0, off4, off1), NOT_SHOWN),
    }
    got: Dict[str, Any] = {}
    for name, (h, want) in cases.items():
        d = decide(h)
        got[name] = [d["outcome"], [round(x, 3) for x in d["interval90_pp"]], round(d["df"], 2)]
        if d["outcome"] != want:
            probs.append(f"{name}: {d['outcome']} != {want}")
    # the floors: identical rows and columns have zero sample variance; both floors (0.625 / 3) must bind
    flat = decide(grid(0.5, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)))
    want_floor = cell_var(DECIDING_PAIRS) / 3
    if not (flat["floor_rows_bound"] and flat["floor_cols_bound"]) or abs(flat["s2_rows"] - want_floor) > 1e-12 \
            or abs(flat["s2_cols"] - want_floor) > 1e-12 or flat["outcome"] != NOT_SHOWN:
        probs.append(f"a flat matrix: the floors did not bind as registered ({flat})")
    got["floors (a flat matrix, +0.5)"] = [flat["outcome"], [round(x, 3) for x in flat["interval90_pp"]]]
    if abs(cell_var(2000) - 0.625) > 1e-12 or abs(cell_var(1000) - 1.25) > 1e-12:
        probs.append("cell_var is not 0.625 / 1.25 pp^2 at 2,000 / 1,000 pairs")
    # rule 8: shift a PASS matrix so its lower end lands on 0 to the last bit we can reach; it must not PASS
    base = grid(4.0, off1, off1)
    lo = decide(base)["interval90_pp"][0]
    tie = decide([[x - lo for x in line] for line in base])
    if abs(tie["interval90_pp"][0]) > RULE8_EPS or tie["outcome"] != NOT_SHOWN or not tie["near_boundary"]:
        probs.append(f"rule 8: a lower end of {tie['interval90_pp'][0]!r} read {tie['outcome']}")
    got["rule 8 (lower end = 0)"] = [tie["outcome"], tie["near_boundary"]]
    wide = grid(0.0, off1, (-1.5, -0.5, 0.5, 1.5))
    for name, h in (("2 rows", base[:2]), ("2 columns", [line[:2] for line in base]), ("4 columns", wide),
                    ("ragged", [base[0], base[1], base[2][:2]]), ("nan", [[float("nan")] * 3] * 3)):
        if decide(h)["outcome"] != INCONCLUSIVE:
            probs.append(f"{name}: not INCONCLUSIVE")
    if decide(base, inconclusive=["missing cell"])["outcome"] != INCONCLUSIVE:
        probs.append("a missing cell: not INCONCLUSIVE")
    if report(base)["available"] or report(wide, inconclusive=["missing cell"])["available"]:
        probs.append("a reported matrix of the wrong shape / with a missing cell reads as available")
    # the indirect line: +3 and +1 against the same columns -> +2; against a direct +2 it agrees
    cols8 = (-1.4, -1.0, -0.6, -0.2, 0.2, 0.6, 1.0, 1.4)
    ind = indirect(report(grid(3.0, off1, cols8)), report(grid(1.0, off1, cols8)), decide(grid(2.0, off1, off1)))
    if abs(ind["indirect_delta_pp"] - 2.0) > 1e-9 or ind["signs_agree"] is not True \
            or abs(ind["direct_minus_indirect_pp"]) > 1e-9:
        probs.append(f"the indirect line is wrong: {ind}")
    return {"self_check": "OK" if not probs else "MISMATCH", "problems": probs, "banked_8x8": {
        "delta_hat_pp": ours["delta_hat_pp"], "se_pp": ours["se_pp"], "welch_df": ours["df"], "cross_stat_df": ref.df},
        "x67_banked": {"delta_hat_pp": rep["delta_hat_pp"], "interval90_pp": rep["interval90_pp"], "df": rep["df"]},
        "cases": got}


def simulate(reps: int = 50000, seed: int = 20261111) -> Dict[str, Any]:
    """P(PASS) / P(WORSE) of the DECIDING 3 x 3 under h_ij = 50 + D + a_i - b_j + e_ij, a ~ N(0, sig^2) (a magnet
    fork's strength), b ~ N(0, sig^2) (a control fork's), e ~ N(0, sig_e^2) (one cell's noise).

    sig_e = 1.05 pp at 1,000 pairs is the figure the banked crosses measured (0.80-1.10 pp; X67's own 3 x 8 read
    0.85); it scales by 1 / sqrt(pairs / 1000). sig is the fork-to-fork seed SD of an arm: 0.5 pp is the fork
    evidence (C0's G1 finals 0.48 pp on S; X67's three forks 0.32 pp and the eight C0 finals 0.59 pp as cross row /
    column means), 1.0 a doubling, 1.76 the FRESH-run figure (the pessimistic case for a fork). Both cell sizes are
    printed so the choice of 2,000 pairs is visible."""
    import numpy as np
    from scipy.stats import t as student_t

    rng = np.random.default_rng(seed)
    r = c = N_ROWS
    deltas = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]
    out: Dict[str, Any] = {"reps": reps, "seed": seed, "sigma_e_at_1000_pairs": 1.05, "shape": [r, c],
                           "deltas_pp": deltas, "scenarios": []}
    for pairs in (1000, DECIDING_PAIRS):
        sig_e = 1.05 / math.sqrt(pairs / 1000.0)
        cv = cell_var(pairs)
        for sig in (0.0, 0.32, 0.5, 1.0, 1.76):
            a = rng.normal(0, sig, (reps, r))
            b = rng.normal(0, sig, (reps, c))
            e = rng.normal(0, sig_e, (reps, r, c))
            base = a[:, :, None] - b[:, None, :] + e
            s2r = np.maximum(base.mean(axis=2).var(axis=1, ddof=1), cv / c)
            s2c = np.maximum(base.mean(axis=1).var(axis=1, ddof=1), cv / r)
            tr, tc = s2r / r, s2c / c
            v = tr + tc
            df = v * v / (tr * tr / (r - 1) + tc * tc / (c - 1))
            hw = student_t.ppf(0.95, df) * np.sqrt(v)
            m = base.mean(axis=(1, 2))
            out["scenarios"].append({
                "pairs": pairs, "sigma_forks": sig, "median_half_width_pp": round(float(np.median(hw)), 2),
                "p_pass": [round(float(((m + d - hw) > 0).mean()), 3) for d in deltas],
                "p_worse_at_0": round(float(((m + hw) < 0).mean()), 3),
                "p_worse_at_minus2": round(float(((m - 2.0 + hw) < 0).mean()), 3)})
    return out


def main(argv: Sequence[str]) -> int:
    cmd = argv[1] if len(argv) > 1 else "self-check"
    if cmd == "self-check":
        res = self_check()
        print(json.dumps(res, indent=1))
        return 0 if res["self_check"] == "OK" else 1
    if cmd == "simulate":
        res = simulate()
        print(f"the DECIDING 3 x 3; reps {res['reps']}, rng {res['seed']}, cell noise SD "
              f"{res['sigma_e_at_1000_pairs']} pp at 1,000 pairs (x 1/sqrt(pairs/1000))")
        print("pairs  sig_forks  half-width  P(PASS) at D = " + " / ".join(f"{d:+g}" for d in res["deltas_pp"])
              + "   P(WORSE) at 0 / -2")
        for s in res["scenarios"]:
            print(f"{s['pairs']:>5}  {s['sigma_forks']:>9.2f}  {s['median_half_width_pp']:>10.2f}  "
                  + " / ".join(f"{p:.3f}" for p in s["p_pass"])
                  + f"   {s['p_worse_at_0']:.3f} / {s['p_worse_at_minus2']:.3f}")
        return 0
    if cmd in ("decide", "report"):
        doc = json.loads(Path(argv[2]).read_text())
        res = decide(doc["h_pp"]) if cmd == "decide" else report(doc["h_pp"], col_seeds=doc.get("col_seeds"))
        print(json.dumps(res, indent=1))
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
