"""CE-3d, the trailing-magnet screen (the SWAP): the REGISTERED decision rule, its self-check, and its design simulation.

Registration: ``PREREG.md`` in this directory (REGISTERED 2026-11-11 and AMENDED the same day, Amendment 1, before
any CE-3d step). The rule is X67's (``../x67_entropy_screen_2026-11-09/screen_rule.py``) with nothing changed:
``rect_stat`` and ``decide`` below are that file's (only the row arm's name in two strings differs), and the
self-check reproduces X67's banked verdict from its record to 1e-9. A measurement's script is a frozen record, so the rule is
copied here and never imported from X67's directory.

THE MATRIX. h[i][j] = swap fork i's score (pp, a draw = 1/2) against C0 final j, one mirrored cell of 1,000 pairs,
both sides greedy. Rows = the three swap forks ``rb_x35_ce3dswap_s7321..s7323`` (r = 3). Columns = the C0 finals
that the plan admits (5 <= c <= 8; PREREG §5.2). The matrix is RECTANGULAR, so ``main.h2h.cross.cross_stat``
(square only) cannot read it; this file is the same estimator with the two factors' own counts.

THE STATISTIC.
    D   = mean(h) - 50                                            (the swap arm's lead over C0, pp)
    s2R = max(var of the r row means,    CELL_VAR / c)            (ddof 1; the floor is a row mean's binomial part)
    s2C = max(var of the c column means, CELL_VAR / r)
    V   = s2R / r + s2C / c
    df  = V^2 / ((s2R / r)^2 / (r - 1) + (s2C / c)^2 / (c - 1))   (Welch-Satterthwaite)
    90 % interval = D +- t(0.95, df) * sqrt(V)
``CELL_VAR`` = 2500 / 2000 = 1.25 pp^2, the binomial variance of one 2,000-game cell at 50 %: a mean of k cells
cannot vary by less than CELL_VAR / k between seeds, so a smaller sample variance is luck and is not believed.
On a square matrix with neither floor binding, D and V are ``cross_stat``'s exactly (self-check); the df differs by
design (``cross_stat`` uses 2(n - 1); with three rows against five to eight columns the row term carries nearly all
of V on 2 df, and Welch's df says so).

THE CLASSES (one look, no look 2).
* PASS       iff the interval's LOWER end > 0     (the swap forks beat the C0 finals, one-sided 95 %).
* WORSE      iff the interval's UPPER end < 0.
* NOT-SHOWN  otherwise.
* INCONCLUSIVE when an input is incomplete or invalid (never interpreted).

RULE 8 (standing rule 8): an end within ``RULE8_EPS`` (1e-9 pp) of 0 satisfies neither PASS nor WORSE and is
recorded in ``near_boundary``. A tie never passes.

REPORTED beside the verdict, never deciding: the same interval with both variances floored at the FRESH-run seed
variance (1.76 pp)^2 (``../boundary1_scratch_2026-11-09/seed_spread.log``), and D over the banked columns
(seeds 3501-3505) and over the top-up columns (3506-3508) separately.

    python screen_rule.py self-check        # cross_stat on a banked 8 x 8, X67's banked verdict, + the hand cases
    python screen_rule.py simulate          # the operating characteristics (PREREG §6), deterministic seed
    python screen_rule.py decide <h.json>   # h.json = {"h_pp": [[...], ...], "col_seeds": [...]} -> one JSON verdict

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
MIN_COLS, MAX_COLS = 5, 8
PAIRS = 1000
CELL_VAR_PP2 = 2500.0 / (2 * PAIRS)          # 1.25: one cell's binomial variance at 50 %, pp^2
CI_LEVEL = 0.90
FRESH_SEED_SD_PP = 1.76                      # the fresh-run seed SD: the REPORTED sensitivity line only
TOPUP_SEEDS = (3506, 3507, 3508)


def _mean(xs: Sequence[float]) -> float:
    return math.fsum(xs) / len(xs)


def _var(xs: Sequence[float]) -> float:
    m = _mean(xs)
    return math.fsum((x - m) ** 2 for x in xs) / (len(xs) - 1)


def rect_stat(h: Sequence[Sequence[float]], *, floor_r: Optional[float] = None,
              floor_c: Optional[float] = None) -> Dict[str, Any]:
    """The statistic on an r x c matrix (pp). The floors default to the registered binomial ones."""
    r, c = len(h), len(h[0])
    rows = [_mean(line) for line in h]
    cols = [_mean([h[i][j] for i in range(r)]) for j in range(c)]
    raw_r, raw_c = _var(rows), _var(cols)
    fr = CELL_VAR_PP2 / c if floor_r is None else floor_r
    fc = CELL_VAR_PP2 / r if floor_c is None else floor_c
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
            "row_means_pp": rows, "col_means_pp": cols, "shape": [r, c]}


def decide(h: Sequence[Sequence[float]], inconclusive: Sequence[str] = (),
           col_seeds: Optional[Sequence[int]] = None) -> Dict[str, Any]:
    """The registered verdict on ``h`` (rows = the swap forks, columns = the C0 finals; pp). X67's ``decide``."""
    reasons: List[str] = list(inconclusive)
    if not reasons:
        if len(h) != N_ROWS:
            reasons.append(f"the registered read has {N_ROWS} rows (the swap forks); got {len(h)}")
        elif any(len(line) != len(h[0]) for line in h) or not MIN_COLS <= len(h[0]) <= MAX_COLS:
            reasons.append(f"the registered read has {MIN_COLS}..{MAX_COLS} columns (C0 finals), every row the same; "
                           f"got {[len(line) for line in h]}")
        elif not all(math.isfinite(x) and 0.0 <= x <= 100.0 for line in h for x in line):
            reasons.append("a cell is not a finite score in [0, 100] pp")
    if reasons:
        return {"outcome": INCONCLUSIVE, "reasons": reasons}
    st = rect_stat(h)
    lo, hi = st["interval90_pp"]
    near = []
    if abs(lo) <= RULE8_EPS:
        near.append("lower end vs 0")
    if abs(hi) <= RULE8_EPS:
        near.append("upper end vs 0")
    if lo > 0.0 and "lower end vs 0" not in near:
        outcome = PASS
    elif hi < 0.0 and "upper end vs 0" not in near:
        outcome = WORSE
    else:
        outcome = NOT_SHOWN
    fresh = rect_stat(h, floor_r=FRESH_SEED_SD_PP ** 2, floor_c=FRESH_SEED_SD_PP ** 2)
    out: Dict[str, Any] = {"outcome": outcome, "near_boundary": near, **st,
                           "reported": {"interval90_fresh_floor_pp": fresh["interval90_pp"],
                                        "fresh_floor_df": fresh["df"], "fresh_floor_se_pp": fresh["se_pp"]}}
    if col_seeds is not None and len(col_seeds) == len(h[0]):
        top = [j for j, s in enumerate(col_seeds) if int(s) in TOPUP_SEEDS]
        bank = [j for j in range(len(col_seeds)) if j not in top]
        for name, idx in (("delta_banked_cols_pp", bank), ("delta_topup_cols_pp", top)):
            out["reported"][name] = (_mean([h[i][j] for i in range(len(h)) for j in idx]) - 50.0) if idx else None
    return out


def self_check() -> Dict[str, Any]:
    """(1) On a banked 8 x 8 with no floor binding, D and V equal ``cross_stat``'s. (2) ``decide`` on X67's banked
    3 x 8 matrix reproduces X67's banked verdict (class, D, SE, df, interval, floors, the fresh-floor interval).
    (3) Hand cases for each class, the rule-8 tie, the floors and the INCONCLUSIVE shapes."""
    probs: List[str] = []
    h8 = json.loads(BANKED_8X8.read_text())["h_pp"]
    ours, ref = rect_stat(h8), cross_stat(h8)
    if ours["floor_rows_bound"] or ours["floor_cols_bound"]:
        probs.append("a floor bound on the banked 8 x 8 (the equality check needs none)")
    if abs(ours["delta_hat_pp"] - ref.delta_hat) > 1e-9 or abs(ours["se_pp"] - ref.se) > 1e-9:
        probs.append(f"D / SE {ours['delta_hat_pp']!r} / {ours['se_pp']!r} != cross_stat's {ref.delta_hat!r} / {ref.se!r}")

    x67_h = json.loads((X67_RECORD / "h.json").read_text())
    x67_v = json.loads((X67_RECORD / "verdict.json").read_text())
    again = decide(x67_h["h_pp"], col_seeds=x67_h["col_seeds"])
    if again["outcome"] != x67_v["outcome"] or x67_v["outcome"] != PASS:
        probs.append(f"X67's banked class {x67_v['outcome']!r}; this rule reads {again['outcome']!r}")
    for k in ("delta_hat_pp", "se_pp", "df"):
        if abs(again[k] - x67_v[k]) > 1e-9:
            probs.append(f"X67's banked {k}: {again[k]!r} != {x67_v[k]!r}")
    for k, a, b in (("interval90_pp", again["interval90_pp"], x67_v["interval90_pp"]),
                    ("fresh-floor interval", again["reported"]["interval90_fresh_floor_pp"],
                     x67_v["reported"]["interval90_fresh_floor_pp"])):
        if any(abs(x - y) > 1e-9 for x, y in zip(a, b)):
            probs.append(f"X67's banked {k}: {a} != {b}")
    if (again["floor_rows_bound"], again["floor_cols_bound"]) != (x67_v["floor_rows_bound"], x67_v["floor_cols_bound"]):
        probs.append("X67's banked floors bind differently here")

    def grid(d: float, row_off: Sequence[float], col_off: Sequence[float]) -> List[List[float]]:
        return [[50.0 + d + a + b for b in col_off] for a in row_off]

    cols5, cols8 = (-1.0, -0.5, 0.0, 0.5, 1.0), (-1.4, -1.0, -0.6, -0.2, 0.2, 0.6, 1.0, 1.4)
    cases = {
        "PASS (+4, rows +-1)": (grid(4.0, (-1.0, 0.0, 1.0), cols5), PASS),
        "WORSE (-4, rows +-1)": (grid(-4.0, (-1.0, 0.0, 1.0), cols8), WORSE),
        "NOT-SHOWN (+1, rows +-2)": (grid(1.0, (-2.0, 0.0, 2.0), cols5), NOT_SHOWN),
        "NOT-SHOWN (+4, rows +-4: a wide arm)": (grid(4.0, (-4.0, 0.0, 4.0), cols8), NOT_SHOWN),
    }
    got: Dict[str, Any] = {}
    for name, (h, want) in cases.items():
        d = decide(h)
        got[name] = [d["outcome"], [round(x, 3) for x in d["interval90_pp"]], round(d["df"], 2)]
        if d["outcome"] != want:
            probs.append(f"{name}: {d['outcome']} != {want}")
    # the floors: three identical rows have zero sample variance; the row floor (1.25 / c) must bind and widen
    flat = decide(grid(0.6, (0.0, 0.0, 0.0), cols5))
    if not flat["floor_rows_bound"] or abs(flat["s2_rows"] - CELL_VAR_PP2 / 5) > 1e-12 or flat["outcome"] != NOT_SHOWN:
        probs.append(f"three identical rows: the row floor did not bind as registered ({flat})")
    got["floor (identical rows, +0.6)"] = [flat["outcome"], [round(x, 3) for x in flat["interval90_pp"]]]
    # rule 8: shift a PASS matrix so its lower end lands on 0 to the last bit we can reach; it must not PASS
    base = grid(4.0, (-1.0, 0.0, 1.0), cols5)
    lo = decide(base)["interval90_pp"][0]
    tie = decide([[x - lo for x in line] for line in base])
    if abs(tie["interval90_pp"][0]) > RULE8_EPS or tie["outcome"] != NOT_SHOWN or not tie["near_boundary"]:
        probs.append(f"rule 8: a lower end of {tie['interval90_pp'][0]!r} read {tie['outcome']}")
    got["rule 8 (lower end = 0)"] = [tie["outcome"], tie["near_boundary"]]
    for name, h in (("2 rows", base[:2]), ("4 columns", [line[:4] for line in base]),
                    ("9 columns", grid(0.0, (-1.0, 0.0, 1.0), cols8 + (1.8,))), ("ragged", [base[0], base[1], base[2][:4]]),
                    ("nan", [[float("nan")] * 5] * 3)):
        if decide(h)["outcome"] != INCONCLUSIVE:
            probs.append(f"{name}: not INCONCLUSIVE")
    if decide(base, inconclusive=["missing cell"])["outcome"] != INCONCLUSIVE:
        probs.append("a missing cell: not INCONCLUSIVE")
    rep = decide(grid(2.0, (-1.0, 0.0, 1.0), cols8), col_seeds=list(range(3501, 3509)))["reported"]
    if rep["delta_banked_cols_pp"] is None or rep["delta_topup_cols_pp"] is None or \
            abs(rep["delta_banked_cols_pp"] - (2.0 + _mean(cols8[:5]))) > 1e-9:
        probs.append(f"the banked / top-up split is wrong: {rep}")
    return {"self_check": "OK" if not probs else "MISMATCH", "problems": probs, "banked_8x8": {
        "delta_hat_pp": ours["delta_hat_pp"], "se_pp": ours["se_pp"], "welch_df": ours["df"], "cross_stat_df": ref.df},
        "x67_banked": {"outcome": again["outcome"], "delta_hat_pp": again["delta_hat_pp"],
                       "interval90_pp": again["interval90_pp"], "df": again["df"],
                       "row_means_sd_pp": math.sqrt(again["s2_rows_raw"]), "col_means_sd_pp": math.sqrt(again["s2_cols_raw"])},
        "cases": got}


def simulate(reps: int = 50000, seed: int = 20261111) -> Dict[str, Any]:
    """P(PASS) / P(WORSE) under h_ij = 50 + D + a_i - b_j + e_ij, a ~ N(0, sig_R^2) (a swap fork's strength),
    b ~ N(0, sig_C^2) (a C0 final's), e ~ N(0, 1.05^2) (the cell noise the banked crosses measured: 0.80-1.10 pp;
    X67's own 3 x 8 is consistent with it).

    X67's simulation with this screen's seed and two more D values. sig_C = 0.5 pp is the FORK evidence (C0's three
    G1 finals read a seed SD of 0.48 pp on S, X48 PREREG §6); (0.32, 0.59) is what X67's banked cross measured as
    the SD of its three row means and of its eight column means (the self-check prints both). sig_R is unknown for a
    fork with a trailing magnet and no entropy bonus, so it runs up to 1.76 pp, the fresh-run pooled figure
    (``seed_spread.log``), the pessimistic case for a fork."""
    import numpy as np
    from scipy.stats import t as student_t

    rng = np.random.default_rng(seed)
    sig_e, r = 1.05, N_ROWS
    deltas = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]
    out: Dict[str, Any] = {"reps": reps, "seed": seed, "sigma_e": sig_e, "rows": r, "deltas_pp": deltas,
                           "scenarios": []}
    for c in (5, 8):
        for sig_r, sig_c in ((0.0, 0.0), (0.32, 0.59), (0.5, 0.5), (1.0, 0.5), (1.76, 0.5), (1.76, 1.76)):
            a = rng.normal(0, sig_r, (reps, r))
            b = rng.normal(0, sig_c, (reps, c))
            e = rng.normal(0, sig_e, (reps, r, c))
            base = a[:, :, None] - b[:, None, :] + e
            s2r = np.maximum(base.mean(axis=2).var(axis=1, ddof=1), CELL_VAR_PP2 / c)
            s2c = np.maximum(base.mean(axis=1).var(axis=1, ddof=1), CELL_VAR_PP2 / r)
            tr, tc = s2r / r, s2c / c
            v = tr + tc
            df = v * v / (tr * tr / (r - 1) + tc * tc / (c - 1))
            hw = student_t.ppf(0.95, df) * np.sqrt(v)
            m = base.mean(axis=(1, 2))
            row = {"cols": c, "sigma_rows": sig_r, "sigma_cols": sig_c,
                   "median_half_width_pp": round(float(np.median(hw)), 2),
                   "p_pass": [round(float(((m + d - hw) > 0).mean()), 3) for d in deltas],
                   "p_worse_at_0": round(float(((m + hw) < 0).mean()), 3),
                   "p_worse_at_minus2": round(float(((m - 2.0 + hw) < 0).mean()), 3)}
            out["scenarios"].append(row)
    return out


def main(argv: Sequence[str]) -> int:
    cmd = argv[1] if len(argv) > 1 else "self-check"
    if cmd == "self-check":
        res = self_check()
        print(json.dumps(res, indent=1))
        return 0 if res["self_check"] == "OK" else 1
    if cmd == "simulate":
        res = simulate()
        print(f"reps {res['reps']}, rng {res['seed']}, cell noise SD {res['sigma_e']} pp, rows {res['rows']}")
        print("cols  sig_R  sig_C  half-width  P(PASS) at D = " + " / ".join(f"{d:+g}" for d in res["deltas_pp"])
              + "   P(WORSE) at 0 / -2")
        for s in res["scenarios"]:
            print(f"{s['cols']:>4}  {s['sigma_rows']:>5.2f}  {s['sigma_cols']:>5.2f}  {s['median_half_width_pp']:>10.2f}  "
                  + " / ".join(f"{p:.3f}" for p in s["p_pass"])
                  + f"   {s['p_worse_at_0']:.3f} / {s['p_worse_at_minus2']:.3f}")
        return 0
    if cmd == "decide":
        doc = json.loads(Path(argv[2]).read_text())
        print(json.dumps(decide(doc["h_pp"], col_seeds=doc.get("col_seeds")), indent=1))
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
