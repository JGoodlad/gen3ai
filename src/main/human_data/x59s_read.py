"""X59-S's registered statistics (``x59s_matched_students_2026-10-25/PREREG.md`` §4–§5), as code BEFORE any data.

Input: per arm (``S-H``, ``S-P``, ``S-S``), per student (one training seed), per U panel, ``(wins, games)`` vs Kakuna
(ties are non-wins). A student's U rate is the equal-weight mean of its panels; an arm's rate is the mean of its
students. The interval of a contrast A − B is

    Δ̂ ± 1.96 sqrt(V),   V = [p̂_A (1 − p̂_A) + p̂_B (1 − p̂_B)] / (m k) + 2 max(SIGMA_MAX, σ̂)² / k,

``m`` the U games a student, ``k`` the students an arm, p̂ an arm's pooled U rate; σ̂² = W − b with W the mean of the
arms' sample variances of their students' U rates (df = arms × (k − 1)) and b the mean per-student binomial variance
``p (1 − p) / m``. GUARD: if σ̂'s one-sided 95 % lower bound ``W df / χ²(df, 0.95) − b`` exceeds SIGMA_MAX², the bound
is refuted and the contrast is read by a Welch t interval across its 2k students. Classes: LB > 0 / UB < 0 / else;
a bound within 1e-9 of 0 takes the undecided class (root rule 8).

``python -m main.human_data.x59s_read <counts.json>`` prints the three contrasts. The in-house fallback
(:func:`inhouse`) is a t interval across the k pairings' S-H win rates minus 0.50.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from typing import Dict, List, Optional, Sequence, Tuple

Z = 1.959963985
SIGMA_MAX = 0.015
TIE = 1e-9
ARMS = ("S-H", "S-P", "S-S")
CONTRASTS = {"delta_data": ("S-H", "S-S", ("HUMAN-DATA-BETTER", "SELF-PLAY-DATA-BETTER", "NOT-RESOLVED")),
             "delta_labels": ("S-H", "S-P", ("HUMAN-LABELS-BETTER", "OWN-LABELS-BETTER", "UNCLEAR")),
             "delta_states": ("S-P", "S-S", ("STATES-HELP", "STATES-HURT", "UNCLEAR"))}

Student = Dict[str, Tuple[int, int]]          # panel -> (wins, games)


def student_rate(s: Student) -> float:
    return sum(w / g for w, g in s.values()) / len(s)


def student_binom(s: Student) -> float:
    """The per-student variance of its equal-weight U rate: Σ_panels p (1 − p) / g / P²."""
    P = len(s)
    return sum((w / g) * (1 - w / g) / g for w, g in s.values()) / P ** 2


def _var(xs: Sequence[float]) -> float:
    m = sum(xs) / len(xs)
    return sum((x - m) ** 2 for x in xs) / (len(xs) - 1)


def seed_sd(arms: Dict[str, List[Student]]) -> Dict[str, float]:
    """σ̂² (floored at 0), its one-sided 95 % lower bound, and df, pooled over every arm."""
    from scipy.stats import chi2

    k = {len(v) for v in arms.values()}
    if len(k) != 1 or k.pop() < 2:
        raise ValueError("every arm needs the same number (>= 2) of students")
    W = sum(_var([student_rate(s) for s in v]) for v in arms.values()) / len(arms)
    b = sum(student_binom(s) for v in arms.values() for s in v) / sum(len(v) for v in arms.values())
    df = sum(len(v) - 1 for v in arms.values())
    return {"sigma2_hat": max(0.0, W - b), "sigma2_lb95": max(0.0, W * df / chi2.ppf(0.95, df) - b), "df": df,
            "within": W, "binom": b}


def classify(lo: float, hi: float, names: Tuple[str, str, str]) -> str:
    if lo > TIE:
        return names[0]
    if hi < -TIE:
        return names[1]
    return names[2]


def contrast(a: List[Student], b: List[Student], sd: Dict[str, float], names: Tuple[str, str, str]) -> Dict:
    from scipy.stats import t as tdist

    k = len(a)
    ra, rb = [student_rate(s) for s in a], [student_rate(s) for s in b]
    pa, pb = sum(ra) / k, sum(rb) / k
    d = pa - pb
    guard = sd["sigma2_lb95"] > SIGMA_MAX ** 2
    if guard:
        va, vb = _var(ra) / k, _var(rb) / k
        df = (va + vb) ** 2 / (va ** 2 / (k - 1) + vb ** 2 / (k - 1)) if va + vb > 0 else float(2 * k - 2)
        hw = tdist.ppf(0.975, df) * math.sqrt(va + vb)
        rule = f"Welch t (df {df:.1f}): the 1.5 pp seed bound was refuted"
    else:
        vbin = (sum(student_binom(s) for s in a) + sum(student_binom(s) for s in b)) / k ** 2
        hw = Z * math.sqrt(vbin + 2 * max(SIGMA_MAX ** 2, sd["sigma2_hat"]) / k)
        rule = "z with the registered seed term"
    lo, hi = d - hw, d + hw
    return {"delta": d, "lo": lo, "hi": hi, "halfwidth": hw, "class": classify(lo, hi, names), "rule": rule,
            "arm_means": [pa, pb]}


def read(arms: Dict[str, List[Student]]) -> Dict:
    """The registered read: σ̂ and the three contrasts (Δ_states descriptive)."""
    if set(arms) != set(ARMS):
        raise ValueError(f"arms {sorted(arms)} != {ARMS}")
    sd = seed_sd(arms)
    out = {"seed_sd": {"sigma_hat_pp": 100 * math.sqrt(sd["sigma2_hat"]),
                       "sigma_lb95_pp": 100 * math.sqrt(sd["sigma2_lb95"]), "df": sd["df"],
                       "guard_fires": sd["sigma2_lb95"] > SIGMA_MAX ** 2}}
    for name, (a, b, names) in CONTRASTS.items():
        out[name] = contrast(arms[a], arms[b], sd, names)
    out["floor"] = all(sum(student_rate(s) for s in v) / len(v) < 0.15 for v in arms.values())
    return out


def inhouse(rates: Sequence[float], names: Tuple[str, str, str]) -> Dict:
    """The in-house fallback: the k pairings' S-H win rates, mean − 0.50, a t interval (df k − 1)."""
    from scipy.stats import t as tdist

    k = len(rates)
    m = sum(rates) / k
    hw = tdist.ppf(0.975, k - 1) * math.sqrt(_var(rates) / k)
    lo, hi = m - 0.5 - hw, m - 0.5 + hw
    return {"delta": m - 0.5, "lo": lo, "hi": hi, "class": classify(lo, hi, names)}


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m main.human_data.x59s_read", description=__doc__.split("\n\n")[0])
    ap.add_argument("counts", help='{"S-H": [{"H": [w, g], "F": [w, g], "P72": [w, g]}, ...], "S-P": ..., "S-S": ...}')
    a = ap.parse_args(argv)
    raw = json.loads(open(a.counts).read())
    arms = {arm: [{p: (int(v[0]), int(v[1])) for p, v in s.items()} for s in raw[arm]] for arm in ARMS}
    print(json.dumps(read(arms), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
