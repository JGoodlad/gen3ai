"""The X35 readout's SELF-CHECK on CONSTRUCTED cases with KNOWN answers (T40's discipline, precedent
``designs/research_state/measurements/toy_crosses_2026-10-16/rect_stat.py self-check``): it runs, and its output is
committed to the staging directory, BEFORE the estimator reads any real matrix.

    python -m main.x35.readout self-check        (or: python -m main.x35.selfcheck)

Every check is deterministic (one private ``numpy.random.default_rng`` seed for the coverage draws; no global
stream). A check either reproduces an answer known independently of :mod:`main.x35.readout` -- a hand computation,
scipy's own Welch interval, Newcombe's published 1998 table, an exactly additive matrix -- or pins one row of the
registered decision table. Exit 1 if any check fails.
"""
from __future__ import annotations

import math
from dataclasses import replace
from typing import Any, Callable, Dict, List, Sequence, Tuple

from agents.training.eval_ledger.cells import Cell
from main.x35 import readout as R

#: The Monte-Carlo coverage floor of the 97.5 % Welch interval (declared before the draws were run).
COVERAGE_FLOOR = 0.96
COVERAGE_DRAWS = 4000
COVERAGE_SEED = 20261017

#: Newcombe (1998), Stat. Med. 17:873-890, Table II, method 10 (the hybrid score), as published (4 d.p.).
NEWCOMBE_1998_METHOD10: Tuple[Tuple[Tuple[int, int, int, int], Tuple[float, float]], ...] = (
    ((56, 70, 48, 80), (0.0524, 0.3339)),
    ((9, 10, 3, 10), (0.1705, 0.8090)),
    ((6, 7, 2, 7), (0.0582, 0.8062)),
    ((5, 56, 0, 29), (-0.0381, 0.1926)),
    ((0, 10, 0, 20), (-0.1611, 0.2775)),
    ((0, 10, 0, 10), (-0.2775, 0.2775)),
)

Check = Tuple[str, bool, str]


def _seeds(xs: Sequence[float], base: int = 3501) -> Dict[str, float]:
    return {str(base + i): float(x) for i, x in enumerate(xs)}


def _shift(xs: Sequence[float], c: float) -> List[float]:
    return [x + c for x in xs]


def check_welch_by_hand() -> Check:
    """ON = 60, 61, 59 / OFF = 50, 51, 49: Δ = 10, both variances 1, se = √(2/3), df = 4 EXACTLY (equal n and
    variance), q = t_{0.9875, 4}."""
    st = R.welch_interval([60, 61, 59], [50, 51, 49])
    from scipy.stats import t

    q = float(t.ppf(0.9875, 4))
    se = math.sqrt(2.0 / 3.0)
    ok = (abs(st.delta - 10.0) < 1e-12 and abs(st.df - 4.0) < 1e-12 and abs(st.se - se) < 1e-12
          and abs(st.ci[0] - (10 - q * se)) < 1e-12 and abs(st.ci[1] - (10 + q * se)) < 1e-12)
    return "Welch by hand (Δ 10, df 4, q t_.9875,4)", ok, f"ci {st.ci}, df {st.df}, q {st.q} vs {q}"


def check_welch_vs_scipy() -> Check:
    """Our interval equals scipy's ``ttest_ind(equal_var=False).confidence_interval(0.975)`` on unequal-variance,
    unequal-n inputs (an independent implementation)."""
    from scipy.stats import ttest_ind

    cases = ([[52.1, 55.3, 49.8], [50.2, 50.9, 49.7]], [[47.0, 58.0, 51.5, 53.2, 49.9], [50.1, 48.3, 52.2, 49.0,
                                                                                      51.7]],
             [[61.0, 44.0, 57.5], [50.0, 50.5, 49.5, 50.25, 49.75]])
    worst = 0.0
    for on, off in cases:
        mine = R.welch_interval(on, off).ci
        ref = ttest_ind(on, off, equal_var=False).confidence_interval(0.975)
        worst = max(worst, abs(mine[0] - ref.low), abs(mine[1] - ref.high))
    return "Welch = scipy's (3 unequal-variance cases)", worst < 1e-9, f"max |Δbound| {worst:.3g}"


def check_coverage() -> Check:
    """Monte-Carlo coverage of the 97.5 % Welch interval at the two looks' sizes (3 v 3, 5 v 5), equal SDs and a
    1 : 3 SD ratio, from one private generator. Floor :data:`COVERAGE_FLOOR`."""
    import numpy as np

    rng = np.random.default_rng(COVERAGE_SEED)
    out = []
    ok = True
    for n in (3, 5):
        for sd_on, sd_off in ((2.0, 2.0), (1.0, 3.0)):
            hit = 0
            for _ in range(COVERAGE_DRAWS):
                on = (52.0 + sd_on * rng.standard_normal(n)).tolist()
                off = (50.0 + sd_off * rng.standard_normal(n)).tolist()
                lo, hi = R.welch_interval(on, off).ci
                hit += lo <= 2.0 <= hi
            cov = hit / COVERAGE_DRAWS
            ok = ok and cov >= COVERAGE_FLOOR
            out.append(f"{n}v{n} sd {sd_on:g}:{sd_off:g} {cov:.4f}")
    return f"Welch 97.5 % coverage >= {COVERAGE_FLOOR} ({COVERAGE_DRAWS} draws each)", ok, "; ".join(out)


def _r1(look: int, on: Sequence[float], off: Sequence[float]) -> R.R1Decision:
    return R.decide_r1(look, _seeds(on), _seeds(off))


def check_r1_classes() -> Check:
    """One constructed case per row of R1's class table, at both looks."""
    rows = [
        ("GAIN look 1", _r1(1, [60, 61, 59], [50, 51, 49]), (R.GAIN, R.GAIN, True)),
        ("LOSS look 1", _r1(1, [40, 41, 39], [50, 51, 49]), (R.LOSS, R.LOSS, True)),
        ("NULL look 1", _r1(1, [50.1, 50.2, 50.0], [50.0, 50.1, 49.9]), (R.NULL, R.NULL, True)),
        ("UNDECIDED Δ̂ > 0 look 1 → CONTINUE", _r1(1, [53, 55, 51], [50, 51, 49]), (R.UNDECIDED, R.CONTINUE, False)),
        ("UNDECIDED Δ̂ < 0 look 1 → final", _r1(1, [47, 49, 45], [50, 51, 49]), (R.UNDECIDED, R.UNDECIDED, True)),
        ("UNDECIDED Δ̂ > 0 look 2 → final", _r1(2, [53, 55, 51, 54, 52], [50, 51, 49, 52, 48]),
         (R.UNDECIDED, R.UNDECIDED, True)),
        ("GAIN look 2", _r1(2, [60, 61, 59, 60.5, 59.5], [50, 51, 49, 50.5, 49.5]), (R.GAIN, R.GAIN, True)),
    ]
    bad = [f"{n}: got {(d.cls, d.outcome, d.final)} {d.stat.ci if d.stat else None}" for n, d, want in rows
           if (d.cls, d.outcome, d.final) != want]
    return "R1 class table (7 constructed rows)", not bad, "; ".join(bad) or "all rows as registered"


def check_r1_rule8() -> Check:
    """Shift a GAIN case so its LOWER bound sits on +2.0 (within float error): UNDECIDED, not GAIN; 1e-6 further
    is GAIN. The same at the upper bound for LOSS (−2.0) and NULL (+2.0), and Δ̂ = 0 at look 1 is not "> 0"."""
    base_on, off = [60.0, 61.0, 59.0], [50.0, 51.0, 49.0]
    lo = R.welch_interval(base_on, off).ci[0]
    on_at = _shift(base_on, 2.0 - lo)
    at, past = _r1(1, on_at, off), _r1(1, _shift(on_at, 1e-6), off)
    hi_l = R.welch_interval([40.0, 41.0, 39.0], off).ci[1]
    loss_at = _r1(1, _shift([40.0, 41.0, 39.0], -2.0 - hi_l), off)
    null_on = [50.1, 50.2, 50.0]
    hi_n = R.welch_interval(null_on, [50.0, 50.1, 49.9]).ci[1]
    null_at = _r1(2, _shift(null_on + [50.15, 50.05], 0.0), [50.0, 50.1, 49.9, 50.05, 49.95])
    hi_n2 = null_at.stat.ci[1] if null_at.stat else float("nan")
    null_at = _r1(2, _shift(null_on + [50.15, 50.05], 2.0 - hi_n2), [50.0, 50.1, 49.9, 50.05, 49.95])
    zero = _r1(1, [53.0, 55.0, 51.0], [55.0, 51.0, 53.0])
    ok = (at.cls == R.UNDECIDED and bool(at.near) and past.cls == R.GAIN and loss_at.cls == R.UNDECIDED
          and bool(loss_at.near) and null_at.cls == R.UNDECIDED and bool(null_at.near)
          and zero.outcome == R.UNDECIDED and zero.final and math.isfinite(hi_n))
    return ("R1 rule 8 at ±2.0 and the look-2 rule at 0", ok,
            f"at {at.cls} {at.near}; past {past.cls}; LOSS edge {loss_at.cls}; NULL edge {null_at.cls}; "
            f"Δ̂ 0 → {zero.outcome}")


def check_r1_inconclusive() -> Check:
    """A wrong seed count, unequal seed sets, a missing population score and zero variance are INCONCLUSIVE."""
    cases = [
        R.decide_r1(1, _seeds([60, 61]), _seeds([50, 51])),
        R.decide_r1(1, _seeds([60, 61, 59]), _seeds([50, 51, 49], base=3601)),
        R.decide_r1(1, {"3501": 60.0, "3502": None, "3503": 59.0}, _seeds([50, 51, 49])),
        R.decide_r1(1, _seeds([55, 55, 55]), _seeds([50, 50, 50])),
        R.decide_r1(2, _seeds([60, 61, 59]), _seeds([50, 51, 49])),
    ]
    ok = all(c.outcome == R.INCONCLUSIVE and c.reasons for c in cases)
    return "R1 INCONCLUSIVE inputs (5 cases)", ok, "; ".join(c.reasons[0] for c in cases)


def check_newcombe_published() -> Check:
    """R2 / G-E's interval (``best_response_gap.newcombe_diff_ci``, the R2 meter's own, z = 1.96) and G-K's
    (``main.anchors.results.newcombe``, z = 1.959963985) both reproduce Newcombe's published method-10 table; at
    the SAME z the two implementations agree to 1e-12, so their only difference is the z constant (a T55 FINDING:
    each guard uses the interval its meter prints)."""
    from agents.training.best_response_gap import newcombe_diff_ci
    from main.anchors.results import Z95, newcombe

    worst_r2 = worst_k = worst_same_z = 0.0
    for (w1, n1, w2, n2), (lo, hi) in NEWCOMBE_1998_METHOD10:
        g = R.gap_diff((w1, n1), (w2, n2))
        worst_r2 = max(worst_r2, abs(g.lo - lo), abs(g.hi - hi))
        _, lo2, hi2 = newcombe(w1, n1, w2, n2)
        worst_k = max(worst_k, abs(lo2 - lo), abs(hi2 - hi))
        _, lo3, hi3 = newcombe_diff_ci(w1, n1, w2, n2, z=Z95)
        worst_same_z = max(worst_same_z, abs(lo3 - lo2), abs(min(1.0, hi3) - hi2))
    ok = worst_r2 < 6e-5 and worst_k < 6e-5 and worst_same_z < 1e-12
    return ("Newcombe = the 1998 table (6 rows), both implementations", ok,
            f"max |Δ| vs published: R2 {worst_r2:.2g}, G-K {worst_k:.2g}; same-z cross {worst_same_z:.2g}")


def check_r2_and_g_e() -> Check:
    """R2 FALLS iff upper < 0; G-E FAILS iff lower > 0; and rule 8 at 0 on a synthetic interval."""
    falls = R.decide_r2(R.gap_diff((200, 400), (264, 400)))          # −16 pp: detected fall
    flat = R.decide_r2(R.gap_diff((250, 400), (264, 400)))           # −3.5 pp: covers 0
    worse = R.guard_e(R.gap_diff((300, 400), (250, 400)))            # +12.5 pp: more exploitable
    calm = R.guard_e(R.gap_diff((250, 400), (264, 400)))
    edge = R.GapDiff(200, 400, 260, 400, -0.15, -0.2, -5e-10)
    near = R.decide_r2(edge)
    edge_e = R.guard_e(replace(edge, diff=0.15, lo=5e-10, hi=0.2))
    ok = (falls.falls and not flat.falls and worse.status == R.FAIL and calm.status == R.PASS
          and not near.falls and bool(near.near) and edge_e.status == R.PASS and bool(edge_e.near))
    return "R2 falls / G-E fail, rule 8 at 0", ok, (f"falls {falls.gap.hi if falls.gap else None:.4f}, "
                                                    f"flat {flat.falls}, G-E {worse.status}/{calm.status}, "
                                                    f"edges {near.falls}/{edge_e.status}")


def _additive(delta: float, a: Sequence[float], b: Sequence[float]) -> List[List[float]]:
    return [[50.0 + delta + ai - bj for bj in b] for ai in a]


def check_g_f() -> Check:
    """G-F on exactly additive matrices: Δ̂ is exact, and FAIL iff the 95 % upper bound < −2.0 (cross_stat's)."""
    loss = R.guard_f(_additive(-10.0, [1.0, -1.0, 0.0], [0.5, -0.5, 0.0]), 1)
    level = R.guard_f(_additive(0.0, [1.0, -1.0, 0.0], [0.5, -0.5, 0.0]), 1)
    wrong = R.guard_f(_additive(0.0, [1.0, -1.0, 0.0], [0.5, -0.5, 0.0]), 2)
    from main.h2h.cross import cross_stat

    st = cross_stat(_additive(-10.0, [1.0, -1.0, 0.0], [0.5, -0.5, 0.0]))
    ok = (loss.status == R.FAIL and level.status == R.PASS and wrong.status == R.INCONCLUSIVE
          and abs(loss.detail["delta_hat_pp"] + 10.0) < 1e-12 and loss.detail["ci95_pp"] == list(st.ci95))
    return "G-F (additive 3 x 3: Δ −10 FAIL, Δ 0 PASS, wrong size INCONCLUSIVE)", ok, \
        f"{loss.detail.get('ci95_pp')} / {level.detail.get('ci95_pp')}"


def check_g_c() -> Check:
    """G-C: B1 upper < −0.05 FAILs, B2 lower > 0 FAILs, both inside PASS, a missing triple is INCONCLUSIVE,
    and a B1 upper of exactly −0.05 is not a FAIL (rule 8)."""
    rows = [
        (R.guard_c((-0.09, -0.12, -0.06), (0.0, -0.05, 0.05)), R.FAIL),
        (R.guard_c((0.01, -0.03, 0.05), (0.06, 0.01, 0.11)), R.FAIL),
        (R.guard_c((0.01, -0.03, 0.05), (0.0, -0.05, 0.05)), R.PASS),
        (R.guard_c(None, (0.0, -0.05, 0.05)), R.INCONCLUSIVE),
        (R.guard_c((-0.08, -0.11, -0.05), (0.0, -0.05, 0.05)), R.PASS),
    ]
    bad = [f"row {i}: {g.status} != {w}" for i, (g, w) in enumerate(rows) if g.status != w]
    ok = not bad and bool(rows[4][0].near)
    return "G-C (5 constructed rows)", ok, "; ".join(bad) or "all rows as registered"


def check_g_k() -> Check:
    """G-K: ON − OFF detected below 0 on EITHER set FAILs; both level PASS; under 400 games INCONCLUSIVE."""
    level = {"home72": {"on": [252, 420], "off": [250, 420]}, "away": {"on": [228, 400], "off": [226, 400]}}
    away_bad = {"home72": {"on": [252, 420], "off": [250, 420]}, "away": {"on": [160, 400], "off": [228, 400]}}
    short = {"home72": {"on": [252, 420], "off": [250, 420]}, "away": {"on": [190, 380], "off": [228, 400]}}
    rows = [(R.guard_k(level), R.PASS), (R.guard_k(away_bad), R.FAIL), (R.guard_k(short), R.INCONCLUSIVE),
            (R.guard_k(None), R.INCONCLUSIVE)]
    bad = [f"row {i}: {g.status} != {w}" for i, (g, w) in enumerate(rows) if g.status != w]
    return "G-K (4 constructed rows)", not bad, "; ".join(bad) or "all rows as registered"


def _pass_guards() -> List[R.GuardResult]:
    return [R.GuardResult(n, R.PASS) for n in R.GUARDS]


def check_verdict_table() -> Check:
    """Every row of the verdict table from constructed R1 / R2 / guard results."""
    gain = _r1(1, [60, 61, 59], [50, 51, 49])
    null = _r1(1, [50.1, 50.2, 50.0], [50.0, 50.1, 49.9])
    loss = _r1(1, [40, 41, 39], [50, 51, 49])
    cont = _r1(1, [53, 55, 51], [50, 51, 49])
    und = _r1(2, [53, 55, 51, 54, 52], [50, 51, 49, 52, 48])
    falls = R.decide_r2(R.gap_diff((200, 400), (264, 400)))
    flat = R.decide_r2(R.gap_diff((250, 400), (264, 400)))
    none = R.decide_r2(None)
    gfail = [R.GuardResult("G-F", R.FAIL)] + _pass_guards()[1:]
    gopen = _pass_guards()[:3]
    rows = [
        ("A1 GAIN, guards pass", R.decide("A1", gain, none, _pass_guards()), R.ADOPT),
        ("A1 GAIN, G-F fails", R.decide("A1", gain, none, gfail), R.TRADE),
        ("A1 GAIN, G-K not read", R.decide("A1", gain, none, gopen), R.INCONCLUSIVE),
        ("A1 NULL + a falling gap (R2 is not A1's)", R.decide("A1", null, falls, _pass_guards()), R.NULL),
        ("A3 NULL + the gap FALLS", R.decide("A3", null, falls, _pass_guards()), R.ADOPT),
        ("A3 NULL + flat gap", R.decide("A3", null, flat, _pass_guards()), R.NULL),
        ("A4 NULL + no R2", R.decide("A4", null, none, _pass_guards()), R.INCONCLUSIVE),
        ("A6 NULL + falls, G-E fails", R.decide("A6", null, falls,
                                                 [R.GuardResult("G-E", R.FAIL)] + _pass_guards()[:1]
                                                 + _pass_guards()[2:]), R.TRADE),
        ("A2 LOSS", R.decide("A2", loss, none, _pass_guards()), R.LOSS),
        ("A1 look-1 CONTINUE", R.decide("A1", cont, none, _pass_guards()), R.CONTINUE),
        ("A1 look-2 UNDECIDED", R.decide("A1", und, none, _pass_guards()), R.UNDECIDED),
        ("A1 R1 INCONCLUSIVE", R.decide("A1", R.decide_r1(1, {}, {}), none, _pass_guards()), R.INCONCLUSIVE),
    ]
    bad = [f"{n}: {v.verdict} != {w}" for n, v, w in rows if v.verdict != w]
    return f"verdict table ({len(rows)} rows)", not bad, "; ".join(bad) or "all rows as registered"


def _cell(p: str, o: str, k_up: int, n_pairs: int = 500, verdict: str = "OK") -> Cell:
    """A mirrored cell with ``k_up`` pairs scoring 3/4 and the rest 1/2: score = 0.5 + k_up / (4 n_pairs)."""
    counts = (0, 0, n_pairs - k_up, k_up, 0)
    return Cell(request_id="r", family="f", opened="t", player=p, opponent=o, player_id=p, opponent_id=o, rows=1,
                w=k_up, l=0, d=2 * n_pairs - k_up, aborted=0, pairs=counts, verdict=verdict,
                reasons=() if verdict == "OK" else ("aborted 30 % > 25 %",))


def check_population_score() -> Check:
    """S(final) is the equal-weight member mean of the pair scores (exact on constructed cells), and each invalid
    input is an INCONCLUSIVE reason."""
    members = {"m1": "s1", "m2": "s2", "m3": "s3"}
    cells = [_cell("F", "s1", 0), _cell("F", "s2", 100), _cell("F", "s3", 200), _cell("G", "s1", 500)]
    good = R.population_score(cells, "F", members)
    want = (50.0 + 55.0 + 60.0) / 3.0
    bad_cases = [
        R.population_score(cells[:2], "F", members),                                   # missing m3
        R.population_score(cells[:2] + [_cell("F", "s3", 10, n_pairs=499)], "F", members),
        R.population_score(cells[:2] + [_cell("F", "s3", 10, verdict="INCONCLUSIVE")], "F", members),
        R.population_score(cells + [_cell("F", "s1", 3)], "F", members),               # listed twice
        R.population_score(cells, "s2", members),                                      # the final is a member
    ]
    ok = (good.s_pp is not None and abs(good.s_pp - want) < 1e-12 and not good.inconclusive
          and all(b.s_pp is None and b.inconclusive for b in bad_cases))
    return "population score (exact mean; 5 INCONCLUSIVE inputs)", ok, f"S {good.s_pp} vs {want}"


def check_read_roundtrip() -> Check:
    """The JSON entry point (:func:`readout.read`) reaches the same verdict as the functions it composes."""
    doc: Dict[str, Any] = {
        "arm": "A3", "look": 1,
        "r1": {"on": _seeds([50.1, 50.2, 50.0]), "off": _seeds([50.0, 50.1, 49.9])},
        "r2": {"on": [200, 400], "off": [264, 400]},
        "g_f": {"h_pp": _additive(0.0, [1.0, -1.0, 0.0], [0.5, -0.5, 0.0])},
        "g_c": {"b1": [0.01, -0.03, 0.05], "b2": [0.0, -0.05, 0.05]},
        "g_k": {"home72": {"on": [252, 420], "off": [250, 420]}, "away": {"on": [228, 400], "off": [226, 400]}},
        "dose": {"on": [1.0e-8, 1.02e-8, 0.99e-8], "off": [1.0e-8, 1.0e-8, 1.01e-8]},
    }
    v = R.read(doc)
    doc2 = dict(doc, r2={"on": [200, 400], "off": [200, 400]})
    v2 = R.read(doc2)
    ok = v.verdict == R.ADOPT and v2.verdict == R.NULL and v.reported["dose"]["confound"] is False
    return "read() JSON round trip (A3 ADOPT via R2; flat gap NULL)", ok, f"{v.verdict} / {v2.verdict}"


CHECKS: Tuple[Callable[[], Check], ...] = (
    check_welch_by_hand, check_welch_vs_scipy, check_coverage, check_r1_classes, check_r1_rule8,
    check_r1_inconclusive, check_newcombe_published, check_r2_and_g_e, check_g_f, check_g_c, check_g_k,
    check_verdict_table, check_population_score, check_read_roundtrip,
)


def run() -> List[Check]:
    return [c() for c in CHECKS]


def main() -> int:
    results = run()
    for name, ok, detail in results:
        print(f"{'PASS' if ok else 'FAIL'}  {name}  -- {detail}")
    n_ok = sum(ok for _, ok, _ in results)
    print(f"self-check: {n_ok} / {len(results)} PASS")
    return 0 if n_ok == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
