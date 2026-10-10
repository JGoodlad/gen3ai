"""The END-STATE vs PRODUCTION closing test's REGISTERED decision rule, its self-check, and its design simulation.

Registration: ``designs/endstate/design_endstate_closing_test.md`` §3 (REGISTERED 2026-10-09, before any seed).

THE RULE (owner, 2026-10-09). h[i][j] = end-state seed i's score (pp, a draw = ½) against production seed j, one
mirrored cell of 1,000 pairs. The estimator is X5 §7.4's, unchanged (``main.h2h.cross.cross_stat``): Δ̂ = mean(h) − 50,
V̂ = (s²_R + s²_C) / n, df = 2(n − 1). The two-sided 90 % interval is Δ̂ ± t_{0.95, df} · √V̂.

* PASS iff (a) the interval includes 0 or lies above it (upper end ≥ 0) AND (b) its lower end > −2.0 pp.
* FAIL otherwise (detectably worse, or too imprecise to rule out a 2-pp deficit).
* INCONCLUSIVE when an input is incomplete or invalid (never interpreted).

RULE 8 (standing rule 8): an interval end within ``RULE8_EPS`` (1e-9 pp) of its threshold (0 for (a), −2.0 for (b))
does NOT satisfy that clause, and the read records it in ``near_boundary``. Deterministic, and conservative: a tie
never passes.

Fixed n = 8 seeds per arm, ONE look (§3.4 explains why the optional 4-seed look is NOT taken).

    python closing_rule.py self-check     # the rule on the static screen's look-3 matrix (must read FAIL, both clauses)
    python closing_rule.py simulate       # the design's operating characteristics (§3.5), deterministic seed
    python closing_rule.py decide <h.json> # h.json = {"h_pp": [[...], ...]} → one JSON verdict

Run from a checkout's ``src`` (``main.h2h.cross`` is imported from it).
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Sequence

from main.h2h.cross import RULE8_EPS, cross_stat, t_ppf

HERE = Path(__file__).resolve().parent
LOOK3 = HERE.parent / "static_screen_look3_2026-10-09" / "result.json"

PASS, FAIL, INCONCLUSIVE = "PASS", "FAIL", "INCONCLUSIVE"
N_SEEDS = 8
LOWER_FLOOR_PP = -2.0
CI_LEVEL = 0.90


def decide(h: Sequence[Sequence[float]], inconclusive: Sequence[str] = ()) -> Dict[str, Any]:
    """The registered verdict on the n × n matrix ``h`` (rows = end state, columns = production; pp)."""
    if inconclusive:
        return {"outcome": INCONCLUSIVE, "reasons": list(inconclusive)}
    if len(h) != N_SEEDS:
        return {"outcome": INCONCLUSIVE, "reasons": [f"the registered read is {N_SEEDS} x {N_SEEDS}; got {len(h)} rows"]}
    st = cross_stat(h)
    q = t_ppf(0.5 + CI_LEVEL / 2.0, st.df)
    lo, hi = st.delta_hat - q * st.se, st.delta_hat + q * st.se
    near = []
    if abs(hi - 0.0) <= RULE8_EPS:
        near.append("upper end vs 0")
    if abs(lo - LOWER_FLOOR_PP) <= RULE8_EPS:
        near.append("lower end vs -2.0")
    clause_a = hi > 0.0 and "upper end vs 0" not in near
    clause_b = lo > LOWER_FLOOR_PP and "lower end vs -2.0" not in near
    return {"outcome": PASS if (clause_a and clause_b) else FAIL,
            "clause_a_includes_zero_or_above": clause_a, "clause_b_lower_above_minus_2": clause_b,
            "interval90_pp": [lo, hi], "t_0.95": q, "delta_hat_pp": st.delta_hat, "se_pp": st.se, "df": st.df,
            "s2_rows": st.s2_rows, "s2_cols": st.s2_cols, "t_sup": st.delta_hat / st.se,
            "near_boundary": near, "stat": st.as_dict()}


def self_check() -> Dict[str, Any]:
    """The rule on the static screen's look-3 cross: its README already reported (descriptively) FAIL on both clauses."""
    r = json.loads(LOOK3.read_text())
    d = decide(r["stat"]["h_pp"])
    want = r["closing_rule_descriptive"]
    ok = (d["outcome"] == FAIL and not d["clause_a_includes_zero_or_above"] and not d["clause_b_lower_above_minus_2"]
          and all(abs(a - b) < 1e-9 for a, b in zip(d["interval90_pp"], want["interval90"])))
    return {"self_check": "OK" if ok else "MISMATCH", "verdict": {k: d[k] for k in
            ("outcome", "interval90_pp", "clause_a_includes_zero_or_above", "clause_b_lower_above_minus_2")},
            "look3_descriptive": want["interval90"]}


def simulate(reps: int = 20000, seed: int = 20261009) -> Dict[str, Any]:
    """Operating characteristics under the additive model h_ij = 50 + Δ + a_i − b_j + e_ij.

    a ~ N(0, σ_E²) (end-state run strength), b ~ N(0, σ_P²) (production's), e ~ N(0, σ_e²) the meter's per-cell noise
    (1,000 mirrored pairs ⇒ σ_e ≈ 1.1 pp, §8.1 of the static design). Scenarios: σ_E / σ_P from the static screen's
    look 3 (s²_R 2.877, s²_C 1.364 at n = 8 ⇒ σ² ≈ s² − σ_e²/8: 1.65 / 1.10), both at the larger 1.65, and both at
    2.5 (a more variable end state AND production). Also the optional 4-seed look's futility stop (§3.4)."""
    import numpy as np
    from scipy.stats import norm

    rng = np.random.default_rng(seed)
    sig_e = 1.1
    q8 = t_ppf(0.95, 14)
    # The optional look: two-look one-sided O'Brien–Fleming (α = 0.05) for "Δ < 0", information fraction 1/2;
    # z1 = C·√2, z2 = C. Solve C by bisection on the bivariate normal (corr √0.5), then map z1 to t on 6 df at the
    # same nominal p (X5 §7.4's mapping).
    from scipy.stats import multivariate_normal

    def alpha_total(c: float) -> float:
        z1, z2 = c * math.sqrt(2.0), c
        rho = math.sqrt(0.5)
        p_no = multivariate_normal(mean=[0, 0], cov=[[1, rho], [rho, 1]]).cdf([z1, z2])
        return 1.0 - p_no

    lo_c, hi_c = 1.0, 3.0
    for _ in range(80):
        mid = (lo_c + hi_c) / 2
        if alpha_total(mid) > 0.05:
            lo_c = mid
        else:
            hi_c = mid
    c = (lo_c + hi_c) / 2
    z1 = c * math.sqrt(2.0)
    p1 = float(norm.sf(z1))
    from scipy.stats import t as student_t
    t1 = float(student_t.isf(p1, 6))

    scenarios = {"look3 (1.65 / 1.10)": (1.65, 1.10), "both 1.65": (1.65, 1.65), "both 2.5": (2.5, 2.5)}
    deltas = [1.0, 0.0, -0.5, -1.0, -1.5, -2.0, -2.5, -3.5]
    out: Dict[str, Any] = {"reps": reps, "seed": seed, "sigma_e": sig_e, "obf_C": c, "look1_z": z1,
                           "look1_nominal_p": p1, "look1_t_df6": t1, "scenarios": {}}
    for name, (se_, sp_) in scenarios.items():
        rows = []
        for dl in deltas:
            a = rng.normal(0, se_, (reps, 8))
            b = rng.normal(0, sp_, (reps, 8))
            e = rng.normal(0, sig_e, (reps, 8, 8))
            h = 50 + dl + a[:, :, None] - b[:, None, :] + e

            def stat(m):
                n = m.shape[1]
                R, C = m.mean(axis=2), m.mean(axis=1)
                d = m.mean(axis=(1, 2)) - 50
                v = (R.var(axis=1, ddof=1) + C.var(axis=1, ddof=1)) / n
                return d, np.sqrt(v)

            d8, s8 = stat(h)
            passed = ((d8 + q8 * s8) > 0) & ((d8 - q8 * s8) > LOWER_FLOOR_PP)
            d4, s4 = stat(h[:, :4, :4])
            stop = (d4 / s4) < -t1
            rows.append({"delta": dl, "p_pass_fixed_n": float(passed.mean()),
                         "p_early_stop": float(stop.mean()),
                         "p_pass_with_early_look": float((passed & ~stop).mean())})
        out["scenarios"][name] = rows
    return out


def main(argv: Sequence[str]) -> int:
    cmd = argv[1] if len(argv) > 1 else "self-check"
    if cmd == "self-check":
        r = self_check()
        print(json.dumps(r, indent=1))
        return 0 if r["self_check"] == "OK" else 1
    if cmd == "simulate":
        print(json.dumps(simulate(), indent=1))
        return 0
    if cmd == "decide":
        h = json.loads(Path(argv[2]).read_text())["h_pp"]
        print(json.dumps(decide(h), indent=1))
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
