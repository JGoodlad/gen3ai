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
from main.x35 import d1 as D1
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


# ================================================================================ D1 (X38): main.x35.d1
#: D1's H coverage floor (nominal 95 %), declared before the draws were run.
D1_COVERAGE_FLOOR = 0.94
D1_COVERAGE_DRAWS = 10000
D1_COVERAGE_SEED = 20261019


def _hc(per_seed: Sequence[Sequence[int]], n: int = 140, base: int = 3501) -> Dict[str, Dict[str, List[int]]]:
    """``{seed: {panel: [wins, n]}}`` from one ``(x_hpfix, x_forum, x_p72)`` triple per seed (``n`` games each)."""
    return {str(base + i): {k: [int(x), n] for k, x in zip(D1.PANELS, xs)} for i, xs in enumerate(per_seed)}


def _flat(x: int, seeds: int = 3, n: int = 140) -> Dict[str, Dict[str, List[int]]]:
    return _hc([(x, x, x)] * seeds, n)


def check_d1_k1_is_newcombe_published() -> Check:
    """With ONE panel the effective size is n itself, so the D1 interval IS Newcombe's method 10: the published 1998
    table's rows with no zero cell (a zero cell has no variance), and ``main.anchors.results.newcombe`` to 1e-12."""
    from main.anchors.results import newcombe

    worst_pub = worst_k = 0.0
    for (w1, n1, w2, n2), (lo, hi) in NEWCOMBE_1998_METHOD10[:3]:
        a = D1.strat_score({"a": (w1, n1)}, ("a",))
        b = D1.strat_score({"a": (w2, n2)}, ("a",))
        d, l1, h1 = D1.newcombe_strat(a, b)
        worst_pub = max(worst_pub, abs(l1 - lo), abs(h1 - hi))
        d2, l2, h2 = newcombe(w1, n1, w2, n2)
        worst_k = max(worst_k, abs(d - d2), abs(l1 - l2), abs(h1 - h2))
    ok = worst_pub < 6e-5 and worst_k < 1e-12
    return "D1 K = 1 is Newcombe (1998 table, 3 rows; anchors' newcombe)", ok, \
        f"max |Δ| published {worst_pub:.2g}, anchors {worst_k:.2g}"


def check_d1_equal_strata_is_pooled() -> Check:
    """Equal panel n AND equal panel p: p̄ = Σx / Σn, n_eff = Σn, so the interval equals Newcombe on the pooled
    counts (``main.anchors.results.newcombe``) to 1e-12."""
    from main.anchors.results import newcombe

    worst = 0.0
    for x_on, x_off, n in ((231, 168, 420), (70, 56, 140), (300, 310, 700)):
        a = D1.strat_score({k: (x_on, n) for k in D1.PANELS})
        b = D1.strat_score({k: (x_off, n) for k in D1.PANELS})
        d, lo, hi = D1.newcombe_strat(a, b)
        d2, lo2, hi2 = newcombe(3 * x_on, 3 * n, 3 * x_off, 3 * n)
        worst = max(worst, abs(d - d2), abs(lo - lo2), abs(hi - hi2), abs(a.n_eff - 3 * n) / (3 * n))
    return "D1 equal strata = pooled Newcombe (3 cases)", worst < 1e-12, f"max |Δ| {worst:.2g}"


def check_d1_variance_by_hand() -> Check:
    """p = .30 / .40 / .50 at n = 420 each: p̄ = .40, V = (.21 + .24 + .25) / (9 · 420), n_eff = .24 / V = 1296
    (> 1260: the strata are fixed, so the mean is MORE precise than a pooled proportion's)."""
    a = D1.strat_score({"heldout72_hpfix": (126, 420), "forum72": (168, 420), "p72": (210, 420)})
    v = 0.70 / (9 * 420)
    ok = abs(a.mean - 0.40) < 1e-15 and abs(a.var - v) < 1e-18 and abs(a.n_eff - 1296.0) < 1e-9
    return "D1 equal-weight mean, variance, n_eff by hand", ok, f"mean {a.mean}, var {a.var}, n_eff {a.n_eff}"


def check_d1_coverage() -> Check:
    """Monte-Carlo coverage of Δ_H's 95 % interval, heterogeneous panels, n = 420 per panel per arm (one at 400),
    from one private generator. Floor :data:`D1_COVERAGE_FLOOR`."""
    import numpy as np

    rng = np.random.default_rng(D1_COVERAGE_SEED)
    out, ok = [], True
    cases = ((0.33, 0.43, 0.53), (0.30, 0.40, 0.50)), ((0.25, 0.40, 0.60), (0.25, 0.40, 0.60)), \
        ((0.45, 0.20, 0.70), (0.40, 0.25, 0.60))
    ns = (420, 420, 400)
    for p_on, p_off in cases:
        true = (sum(p_on) - sum(p_off)) / 3.0
        hit = 0
        for _ in range(D1_COVERAGE_DRAWS):
            xo = rng.binomial(ns, p_on)
            xf = rng.binomial(ns, p_off)
            a = D1.strat_score({k: (int(x), n) for k, x, n in zip(D1.PANELS, xo, ns)})
            b = D1.strat_score({k: (int(x), n) for k, x, n in zip(D1.PANELS, xf, ns)})
            _, lo, hi = D1.newcombe_strat(a, b)
            hit += lo <= true <= hi
        cov = hit / D1_COVERAGE_DRAWS
        ok = ok and cov >= D1_COVERAGE_FLOOR
        out.append(f"Δ {100 * true:+.1f} pp {cov:.4f}")
    return f"D1 H 95 % coverage >= {D1_COVERAGE_FLOOR} ({D1_COVERAGE_DRAWS} draws each)", ok, "; ".join(out)


def check_d1_h_classes() -> Check:
    """One constructed case per row of H's class table, at both looks (140 games per seed final and panel)."""
    big = 1400
    rows = [
        ("GAIN look 1 (3 / 3 seeds up)", D1.d1_primary(1, _flat(70), _flat(56)), (D1.GAIN, D1.GAIN, True)),
        ("lower > 0 but 1 / 3 seeds up -> UNDECIDED, CONTINUE",
         D1.d1_primary(1, _hc([(140, 140, 140), (55, 55, 55), (55, 55, 55)]), _flat(56)),
         (D1.UNDECIDED, D1.CONTINUE, False)),
        ("LOSS look 1", D1.d1_primary(1, _flat(56), _flat(70)), (D1.LOSS, D1.LOSS, True)),
        ("NULL look 1 (large n)", D1.d1_primary(1, _flat(700, n=big), _flat(700, n=big)), (D1.NULL, D1.NULL, True)),
        ("UNDECIDED Δ̂ > 0 look 1 -> CONTINUE", D1.d1_primary(1, _flat(58), _flat(56)),
         (D1.UNDECIDED, D1.CONTINUE, False)),
        ("UNDECIDED Δ̂ < 0 look 1 -> final", D1.d1_primary(1, _flat(54), _flat(56)),
         (D1.UNDECIDED, D1.UNDECIDED, True)),
        ("UNDECIDED Δ̂ > 0 look 2 -> final", D1.d1_primary(2, _flat(58, 5), _flat(56, 5)),
         (D1.UNDECIDED, D1.UNDECIDED, True)),
        ("GAIN look 2 (3 / 5 seeds up)",
         D1.d1_primary(2, _hc([(80, 80, 80)] * 3 + [(55, 55, 55)] * 2), _flat(56, 5)), (D1.GAIN, D1.GAIN, True)),
        ("lower > 0, 2 / 5 seeds up at look 2 -> UNDECIDED final",
         D1.d1_primary(2, _hc([(110, 110, 110)] * 2 + [(55, 55, 55)] * 3), _flat(56, 5)),
         (D1.UNDECIDED, D1.UNDECIDED, True)),
    ]
    bad = [f"{n}: got {(d.cls, d.outcome, d.final)} {d.ci_pp} +{d.positive_seeds}" for n, d, want in rows
           if (d.cls, d.outcome, d.final) != want]
    # The sign-rule rows were built with lower > 0. Under AM-5 (2026-10-19) their seeds disagree, so the WIDENED
    # lower bound falls below 0 too: the case the sign rule was registered for is checked on the UNWIDENED interval
    # (recorded beside the widened one), and the sign rule itself directly on an interval with lower > 0.
    lo_pos = all(d.ci_unwidened_pp is not None and d.ci_unwidened_pp[0] > 0 for d in (rows[1][1], rows[8][1]))
    sign = D1.classify_h((1.0, 9.0), False)[0] == D1.UNDECIDED and D1.classify_h((1.0, 9.0), True)[0] == D1.GAIN
    return f"D1 H class table ({len(rows)} constructed rows)", not bad and lo_pos and sign, \
        "; ".join(bad) or "all rows as registered (the sign-rule rows have unwidened lower > 0; the rule blocks)"


def check_d1_overdispersion() -> Check:
    """AM-5 (2026-10-19): (i) seeds that agree (identical per-seed cells) give φ̂ = 0 <= 1, and the interval is the
    unwidened one EXACTLY (φ̂ never narrows it); (ii) φ̂ by hand: ON seeds at p = .5 / .4 / .6 (140 games per panel),
    OFF flat at .4: between-seed variance .01, binomial expectation .73 / (3 · 420), so ratio_ON = 12.6 / .73,
    ratio_OFF = 0 and φ̂ = 6.3 / .73 (to 1e-12 relative), df 4; (iii) that case is GAIN on the unwidened interval
    (Δ̂ = +10 pp, 2 of 3 seeds up) and UNDECIDED -> CONTINUE on the widened one."""
    agree = D1.d1_primary(1, _flat(70), _flat(56))
    i_ok = agree.od is not None and agree.od.phi <= 1.0 and agree.od.factor == 1.0 and agree.ci_pp == agree.ci_unwidened_pp
    spread = D1.d1_primary(1, _hc([(70, 70, 70), (56, 56, 56), (84, 84, 84)]), _flat(56))
    od = spread.od
    ii_ok = (od is not None and abs(od.phi - 6.3 / 0.73) <= 1e-12 * (6.3 / 0.73)
             and abs(od.ratios[0] - 12.6 / 0.73) <= 1e-12 * (12.6 / 0.73) and od.ratios[1] == 0.0 and od.df == 4)
    unw = spread.ci_unwidened_pp
    iii_ok = (unw is not None and D1.classify_h(unw, spread.sign_ok)[0] == D1.GAIN
              and (spread.cls, spread.outcome) == (D1.UNDECIDED, D1.CONTINUE) and spread.positive_seeds == 2)
    return "D1 AM-5 overdispersion widening (agree / by hand / GAIN -> CONTINUE)", i_ok and ii_ok and iii_ok, \
        f"agree φ̂ {agree.od.phi if agree.od else None}; spread φ̂ {od.phi if od else None} (want {6.3 / 0.73}); " \
        f"unwidened {unw} widened {spread.ci_pp} {spread.outcome}"


def check_d1_rule8() -> Check:
    """A bound within 1e-9 of 0, −3.0 or +3.0 is UNDECIDED (1e-6 past it is the class); Δ_i or Δ̂ within 1e-9 of 0
    is not "> 0"; equal seed counts give Δ_i = 0 exactly, which is not a positive seed."""
    rows = [((5e-10, 9.0), D1.UNDECIDED), ((1e-6, 9.0), D1.GAIN), ((-9.0, -3e-10), D1.UNDECIDED),
            ((-9.0, -1e-6), D1.LOSS), ((-2.0, 3.0 - 1e-10), D1.UNDECIDED), ((-2.0, 3.0 - 1e-6), D1.NULL),
            ((-3.0 + 1e-10, 2.0), D1.UNDECIDED), ((-3.0 + 1e-6, 2.0), D1.NULL), ((-9.0, 3.0 + 5e-10), D1.UNDECIDED)]
    bad = [f"{ci}: {D1.classify_h(ci, True)[0]} != {w}" for ci, w in rows if D1.classify_h(ci, True)[0] != w]
    near_ok = all(bool(D1.classify_h(ci, True)[1]) == (w == D1.UNDECIDED) for ci, w in rows)
    signs = (not D1.positive(5e-10)) and D1.positive(2e-9) and not D1.positive(0.0)
    tie = D1.d1_primary(1, _hc([(70, 70, 70), (70, 70, 70), (56, 56, 56)]), _flat(56))
    ok = not bad and near_ok and signs and tie.positive_seeds == 2 and dict(tie.seed_deltas_pp)["3503"] == 0.0
    return "D1 rule 8 at 0 / ±3.0, and the sign rule at 0", ok, "; ".join(bad) or f"tie seeds +{tie.positive_seeds}"


def check_d1_h_inconclusive() -> Check:
    """Wrong seed count, unequal seed sets, a missing panel, a stray panel, invalid counts, under the floor and zero
    variance are INCONCLUSIVE; exactly the floor (one uncounted unit) is read."""
    short = _flat(56)
    short["3501"]["p72"] = [48, 120]
    short["3502"]["p72"] = [48, 120]                       # 380 < 400: two units missing on p72
    floor = _flat(56)
    floor["3501"]["p72"] = [48, 120]                       # 400: one unit missing -- read
    missing = _flat(56)
    del missing["3502"]["forum72"]
    stray = _flat(56)
    stray["3501"]["home72"] = [70, 140]
    cases = [
        D1.d1_primary(1, _flat(70, 2), _flat(56, 2)),
        D1.d1_primary(1, _flat(70), _hc([(56, 56, 56)] * 3, base=3601)),
        D1.d1_primary(1, missing, _flat(56)),
        D1.d1_primary(1, stray, _flat(56)),
        D1.d1_primary(1, _hc([(141, 70, 70)] * 3), _flat(56)),
        D1.d1_primary(1, short, _flat(56)),
        D1.d1_primary(1, _hc([(0, 140, 0)] * 3), _hc([(140, 0, 140)] * 3)),
        D1.d1_primary(2, _flat(70), _flat(56)),
    ]
    read = D1.d1_primary(1, floor, _flat(56))
    ok = all(c.outcome == D1.INCONCLUSIVE and c.reasons for c in cases) and read.outcome != D1.INCONCLUSIVE \
        and D1.min_games(3) == 400 and D1.min_games(5) == 680
    return f"D1 H INCONCLUSIVE inputs ({len(cases)} cases) and the floor", ok, \
        "; ".join(c.reasons[0] if c.reasons else "READ" for c in cases)


def check_d1_p() -> Check:
    """P: NON-INFERIOR iff Welch 97.5 % lower > −2.0; a lower bound on −2.0 (float error) is NOT non-inferior and is
    recorded; 1e-6 above is; a missing S or a wrong seed count is INCONCLUSIVE."""
    off = [50.0, 50.1, 49.9]
    ni = D1.d1_coprimary(1, _seeds([50.1, 50.2, 50.0]), _seeds(off))
    wide = D1.d1_coprimary(1, _seeds([44.0, 52.0, 48.0]), _seeds(off))
    lo = R.welch_interval([50.1, 50.2, 50.0], off).ci[0]
    at_on = _shift([50.1, 50.2, 50.0], -2.0 - lo)
    at = D1.d1_coprimary(1, _seeds(at_on), _seeds(off))
    past = D1.d1_coprimary(1, _seeds(_shift(at_on, 1e-6)), _seeds(off))
    miss = D1.d1_coprimary(1, {"3501": 50.0, "3502": None, "3503": 49.0}, _seeds(off))
    count = D1.d1_coprimary(2, _seeds([50.1, 50.2, 50.0]), _seeds(off))
    ok = (ni.cls == D1.NON_INFERIOR and wide.cls == D1.NOT_EXCLUDED and at.cls == D1.NOT_EXCLUDED and bool(at.near)
          and past.cls == D1.NON_INFERIOR and miss.cls is None and count.cls is None)
    return "D1 P non-inferiority at −2.0 (rule 8) and its INCONCLUSIVE inputs", ok, \
        f"{ni.cls} / {wide.cls} / at {at.cls} {at.near} / past {past.cls} / {miss.reasons} / {count.reasons}"


def _d1_guards(**over: str) -> List[R.GuardResult]:
    return [R.GuardResult(n, over.get(n.replace("-", "_"), R.PASS)) for n in D1.D1_GUARDS]


def check_d1_verdict_table() -> Check:
    """Every row of D1's verdict table, and TRADE's exchange rate exactly (Δ̂_H / −Δ̂_P)."""
    gain = D1.d1_primary(1, _flat(70), _flat(56))
    null = D1.d1_primary(1, _flat(700, n=1400), _flat(700, n=1400))
    loss = D1.d1_primary(1, _flat(56), _flat(70))
    cont = D1.d1_primary(1, _flat(58), _flat(56))
    und = D1.d1_primary(2, _flat(58, 5), _flat(56, 5))
    bad_h = D1.d1_primary(1, _flat(70, 2), _flat(56, 2))
    off = [50.0, 50.1, 49.9]
    p_ni = D1.d1_coprimary(1, _seeds([50.1, 50.2, 50.0]), _seeds(off))
    p_loss = D1.d1_coprimary(1, _seeds([47.0, 47.1, 46.9]), _seeds(off))          # Δ̂_P = −3 exactly
    p_wide = D1.d1_coprimary(1, _seeds([44.0, 58.0, 49.0]), _seeds(off))         # Δ̂_P > 0, lower < −2
    p_none = D1.d1_coprimary(1, {}, {})
    rows = [
        ("H GAIN, P NI, guards pass", D1.d1_verdict("X38", gain, p_ni, _d1_guards()), D1.ADOPT),
        ("H GAIN, P not excluded", D1.d1_verdict("X38", gain, p_loss, _d1_guards()), D1.TRADE),
        ("H GAIN, P NI, G-F fails", D1.d1_verdict("X38", gain, p_ni, _d1_guards(G_F=R.FAIL)), D1.TRADE),
        ("H GAIN, P wide (no point loss)", D1.d1_verdict("X38", gain, p_wide, _d1_guards()), D1.TRADE),
        ("H GAIN, P INCONCLUSIVE", D1.d1_verdict("X38", gain, p_none, _d1_guards()), D1.INCONCLUSIVE),
        ("H GAIN, G-C not read", D1.d1_verdict("X38", gain, p_ni, _d1_guards()[:2]), D1.INCONCLUSIVE),
        ("H NULL", D1.d1_verdict("X38", null, p_ni, _d1_guards()), D1.NULL),
        ("H LOSS (P NI does not rescue it)", D1.d1_verdict("X38", loss, p_ni, _d1_guards()), D1.LOSS),
        ("H look-1 CONTINUE", D1.d1_verdict("X38", cont, p_ni, _d1_guards()), D1.CONTINUE),
        ("H look-2 UNDECIDED", D1.d1_verdict("X38", und, D1.d1_coprimary(2, _seeds([50.0] * 4 + [50.5]),
                                                                       _seeds([50.0] * 4 + [49.5])),
                                           _d1_guards()), D1.UNDECIDED),
        ("H INCONCLUSIVE", D1.d1_verdict("X38", bad_h, p_ni, _d1_guards()), D1.INCONCLUSIVE),
    ]
    bad = [f"{n}: {v.verdict} != {w}" for n, v, w in rows if v.verdict != w]
    rate = rows[1][1].reported.get("exchange_rate", {})
    rate_ok = gain.delta_pp is not None and abs(rate.get("rate", 0.0) - gain.delta_pp / 3.0) < 1e-9
    wide_ok = rows[3][1].reported.get("exchange_rate", {}).get("rate", "x") is None
    return f"D1 verdict table ({len(rows)} rows) and the exchange rate", not bad and rate_ok and wide_ok, \
        "; ".join(bad) or f"rate {rate.get('rate')} = Δ̂_H {gain.delta_pp} / 3"


def check_d1_read_roundtrip() -> Check:
    """``read_d1`` (the ``decide-d1`` JSON) reaches the same verdict as the functions it composes; P's seeds must be
    H's."""
    doc: Dict[str, Any] = {
        "arm": "X38", "look": 1,
        "h": {"on": _flat(70), "off": _flat(56)},
        "p": {"on": _seeds([50.1, 50.2, 50.0]), "off": _seeds([50.0, 50.1, 49.9])},
        "r2": {"on": [250, 400], "off": [264, 400]},
        "g_f": {"h_pp": _additive(0.0, [1.0, -1.0, 0.0], [0.5, -0.5, 0.0])},
        "g_c": {"b1": [0.01, -0.03, 0.05], "b2": [0.0, -0.05, 0.05]},
        "kakuna_in_pool": {"home72": {"on": [252, 420], "off": [250, 420]},
                           "away": {"on": [228, 400], "off": [226, 400]}},
    }
    v = D1.read_d1(doc)
    v2 = D1.read_d1(dict(doc, p={"on": _seeds([50.1, 50.2, 50.0], base=3601), "off": _seeds([50.0, 50.1, 49.9],
                                                                                          base=3601)}))
    ok = v.verdict == D1.ADOPT and v2.verdict == D1.INCONCLUSIVE and "kakuna_in_pool" in v.reported
    return "D1 read_d1 JSON round trip (ADOPT; P seeds not H's INCONCLUSIVE)", ok, f"{v.verdict} / {v2.verdict}"


D1_CHECKS: Tuple[Callable[[], Check], ...] = (
    check_d1_k1_is_newcombe_published, check_d1_equal_strata_is_pooled, check_d1_variance_by_hand,
    check_d1_coverage, check_d1_h_classes, check_d1_rule8, check_d1_h_inconclusive, check_d1_p,
    check_d1_verdict_table, check_d1_read_roundtrip, check_d1_overdispersion,
)
SUITES: Dict[str, Tuple[Callable[[], Check], ...]] = {"x35": CHECKS, "d1": D1_CHECKS}


def run(suite: str = "x35") -> List[Check]:
    return [c() for c in SUITES[suite]]


def main(suite: str = "all") -> int:
    names = list(SUITES) if suite == "all" else [suite]
    total = n_ok = 0
    for name in names:
        results = run(name)
        for title, ok, detail in results:
            print(f"{'PASS' if ok else 'FAIL'}  [{name}] {title}  -- {detail}")
        k = sum(ok for _, ok, _ in results)
        print(f"self-check {name}: {k} / {len(results)} PASS")
        total, n_ok = total + len(results), n_ok + k
    print(f"self-check: {n_ok} / {total} PASS")
    return 0 if n_ok == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
