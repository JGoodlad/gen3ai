r"""X59-S's registered statistics (``x59s_matched_students_2026-10-25/PREREG.md`` §4–§5), as code BEFORE any data.

Input: per arm (``S-H``, ``S-P``, ``S-S``), per student (one training seed), per U panel, ``(wins, games)`` vs Kakuna
(ties are non-wins). A student's U rate is the equal-weight mean of its panels; an arm's rate is the mean of its
students. The interval of a contrast A − B is

    Δ̂ ± 1.96 sqrt(V),   V = [p̂_A (1 − p̂_A) + p̂_B (1 − p̂_B)] / (m k) + 2 max(SIGMA_MAX, σ̂)² / k,

``m`` the U games a student, ``k`` the students an arm, p̂ an arm's pooled U rate; σ̂² = W − b with W the mean of the
arms' sample variances of their students' U rates (df = arms × (k − 1)) and b the mean per-student binomial variance
``p (1 − p) / m``. GUARD: if σ̂'s one-sided 95 % lower bound ``W df / χ²(df, 0.95) − b`` exceeds SIGMA_MAX², the bound
is refuted and the contrast is read by a Welch t interval across its 2k students; and in every case the interval is
WIDENED to the Welch interval on the contrast's own students when that is wider (Amendment 5 A5.2). Classes: LB > 0 /
UB < 0 / **NO-DIFFERENCE-AT-4PP** (Amendment 2, REVIEW E M6: the whole 95 % interval inside ±:data:`EQUIV`, the 80 %
MDE) / else undecided; a bound within 1e-9 of 0 or of ±EQUIV takes the undecided class (root rule 8).

The overrides :func:`verdict` applies after the contrasts (Amendment 2, REVIEW E; Amendment 5, REVIEW H):

* **UNINFORMATIVE, per contrast** (A5.1, :func:`gate`): both of its arms' PILOT means (units 0–3) are <= the better
  floor reference + :data:`FLOOR_BAND`. Reported; routes nothing beyond "H-TEAMS first".
* **NO-DIFFERENCE-AT-4PP** needs the matching in-house interval inside ±EQUIV too (A5.3).
* **KAKUNA-SPECIFIC** (A2.10, A5.5): a "better" class whose in-house point estimate is <= 0, or a "worse" class whose
  in-house point estimate is >= 0, opens, closes, routes and drops nothing.
* **POPULATION-SPECIFIC** (A5.4): a "better" class whose forum72-only point estimate is <= 0 opens no lever.
* **The H-BC gate** (A5.8): HUMAN-LABELS-BETTER routes H-BC only if the S-P students' mean yardstick on the S-S
  validation rows' P\* labels is closer to the S-S students' mean than to the S-H students' (:func:`midpoint`).
* **The closure** (A5.6, A5.7, A6.1, A6.2): SELF-PLAY-DATA-BETTER closes :data:`CLOSURE_TEXT` only if Δ_restriction
  (S-P0 − S-P) has been read and its LB <= 0 (a bound within 1e-9 of 0 blocks), Δ_labels is not HUMAN-LABELS-BETTER,
  Δ_states' LB <= 0, S-H is fitted (:data:`FIT_DROP`) and the re-encode share is <= :data:`REENCODE_SHARE`. FR-2 is
  reported (its class must exist) and gates nothing (A6.1).

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
EQUIV = 0.04          # the NO-DIFFERENCE bar: the registered 80 % MDE (PREREG §4)
FLOOR_BAND = 0.05     # UNINFORMATIVE: every arm within 5 pp of the better floor reference
NO_DIFF = "NO-DIFFERENCE-AT-4PP"
FR2_CLASSES = ("ARTEFACT", "NO-ARTEFACT", "UNCLEAR")
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
    if lo > -EQUIV + TIE and hi < EQUIV - TIE:
        return NO_DIFF
    return names[2]


def welch(ra: Sequence[float], rb: Sequence[float]) -> Tuple[float, float]:
    """(half-width, df) of the Welch 95 % interval of mean(ra) − mean(rb) on the contrast's own students."""
    from scipy.stats import t as tdist

    va, vb = _var(ra) / len(ra), _var(rb) / len(rb)
    if va + vb <= 0:
        return 0.0, float(len(ra) + len(rb) - 2)
    df = (va + vb) ** 2 / (va ** 2 / (len(ra) - 1) + vb ** 2 / (len(rb) - 1))
    return tdist.ppf(0.975, df) * math.sqrt(va + vb), df


def contrast(a: List[Student], b: List[Student], sd: Dict[str, float], names: Tuple[str, str, str]) -> Dict:
    """The registered interval (§4: z with the pooled seed term, Welch when the guard fires), WIDENED to the Welch
    interval on the contrast's own students when that is wider (Amendment 5 A5.2, REVIEW H M1)."""
    k = len(a)
    ra, rb = [student_rate(s) for s in a], [student_rate(s) for s in b]
    pa, pb = sum(ra) / k, sum(rb) / k
    d = pa - pb
    w_hw, w_df = welch(ra, rb)
    if sd["sigma2_lb95"] > SIGMA_MAX ** 2:
        reg_hw, rule = w_hw, f"Welch t (df {w_df:.1f}): the 1.5 pp seed bound was refuted"
    else:
        vbin = (sum(student_binom(s) for s in a) + sum(student_binom(s) for s in b)) / k ** 2
        reg_hw = Z * math.sqrt(vbin + 2 * max(SIGMA_MAX ** 2, sd["sigma2_hat"]) / k)
        rule = "z with the registered seed term"
    hw = max(reg_hw, w_hw)
    if w_hw > reg_hw:
        rule += f"; widened to its own Welch (df {w_df:.1f}, A5.2)"
    lo, hi = d - hw, d + hw
    return {"delta": d, "lo": lo, "hi": hi, "halfwidth": hw, "halfwidth_registered": reg_hw, "halfwidth_welch": w_hw,
            "class": classify(lo, hi, names), "rule": rule, "arm_means": [pa, pb],
            "arm_sds": [math.sqrt(_var(ra)), math.sqrt(_var(rb))]}


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
    return out


#: Amendment 5 A5.6: S-H is FITTED iff its students' mean last-quarter-epoch drop in validation NLL is below this
FIT_DROP = 0.01
#: Amendment 5 A5.7: more than this share of re-encoded rows differing blocks the closure and STATES-HURT
REENCODE_SHARE = 0.01
POP_SPECIFIC = "POPULATION-SPECIFIC"
KAKUNA_SPECIFIC = "KAKUNA-SPECIFIC"
UNINF = "UNINFORMATIVE"


def gate(pilot: Dict[str, List[Student]], floor_rates: Dict[str, float], band: float = FLOOR_BAND) -> Dict:
    """Amendment 5 A5.1: a contrast is UNINFORMATIVE iff BOTH its arms' pilot means are <= the better floor reference +
    ``band`` (a mean within 1e-9 counts as within); the rest of the read plays iff Δ_data or Δ_labels is informative."""
    bar = max(floor_rates.values()) + band
    mean = {a: sum(student_rate(s) for s in v) / len(v) for a, v in pilot.items()}
    out = {name: all(mean[x] <= bar + TIE for x in CONTRASTS[name][:2]) for name in CONTRASTS}
    return {"uninformative": out, "continue": not (out["delta_data"] and out["delta_labels"]), "bar": bar}


def midpoint(sp: Sequence[float], ss: Sequence[float], sh: Sequence[float]) -> bool:
    """A5.8: S-P learned its teacher iff its mean yardstick is closer to S-S's mean than to S-H's (a tie fails)."""
    m = lambda xs: sum(xs) / len(xs)  # noqa: E731
    return abs(m(sp) - m(ss)) < abs(m(sp) - m(sh)) - TIE


def _inside(lo: float, hi: float) -> bool:
    return lo > -EQUIV + TIE and hi < EQUIV - TIE


CLOSURE_TEXT = ("closes 1300+ ladder data in the panel-distant, forfeit-heavy slice (50.6 % of its rows from forfeits; "
                "median turn 15 against S-S's 19), at this mix and volume, as a behaviour-cloning target for a fresh "
                "network at 3 epochs (7,041 steps)")


def verdict(read_: Dict, inhouse_: Dict[str, Dict], uninf: Dict[str, bool], f_delta: Dict[str, float],
            yard: Dict[str, Sequence[float]], fr2: str, sh_fit_drop: float, reencode_share: float,
            restriction_lb: Optional[float]) -> Dict:
    r"""Amendment 5's final classes and actions (A5.1–A5.8 over A2's). ``inhouse_``: :func:`inhouse` reads keyed
    ``delta_data`` / ``delta_labels``; ``uninf``: :func:`gate`'s per-contrast booleans (from the pilot); ``f_delta``:
    each contrast's forum72-only point estimate; ``yard``: ``{"S-H", "S-P", "S-S"}`` -> the students' NLL on the S-S
    validation rows' P\* labels; ``fr2``: FR-2's class (read BEFORE Δ_data is reported); ``sh_fit_drop``: the S-H
    students' mean last-quarter-epoch drop in validation NLL; ``reencode_share``: A5.7's share of re-encoded rows that
    differ; ``restriction_lb``: Δ_restriction's (S-P0 − S-P) lower bound, None until it is read (A6.1: the closure's
    guard; FR-2 is reported and gates nothing)."""
    if fr2 not in FR2_CLASSES:
        raise ValueError(f"FR-2 must be read before Delta_data is reported (got {fr2!r}; one of {FR2_CLASSES})")
    out: Dict = {"classes": {}, "kakuna_classes": {}, "actions": [], "uninformative": dict(uninf)}
    for name in ("delta_data", "delta_labels"):
        better, worse, undecided = CONTRASTS[name][2]
        c = read_[name]["class"]
        out["kakuna_classes"][name] = c
        ih = inhouse_[name]
        if uninf[name]:
            c = UNINF
        elif c == NO_DIFF and not _inside(ih["lo"], ih["hi"]):
            c = undecided                                                # A5.3
        elif c == better and ih["delta"] <= TIE:
            c = KAKUNA_SPECIFIC                                          # A2.10
        elif c == worse and ih["delta"] >= -TIE:
            c = KAKUNA_SPECIFIC                                          # A5.5
        elif c == better and f_delta[name] <= TIE:
            c = POP_SPECIFIC                                             # A5.4
        out["classes"][name] = c
    d, lab = out["classes"]["delta_data"], out["classes"]["delta_labels"]
    learned = midpoint(yard["S-P"], yard["S-S"], yard["S-H"])
    out["sp_learned_its_teacher"] = learned
    fitted = sh_fit_drop < FIT_DROP - TIE
    out["sh_fitted"] = fitted
    out["reencode_blocks"] = reencode_share > REENCODE_SHARE + TIE
    if lab == "HUMAN-LABELS-BETTER":
        out["actions"].append("H-BC first (against P*-restricted-to-human-support)" if learned else
                              "HUMAN-LABELS-BETTER reported; H-BC NOT routed (S-P did not learn its teacher)")
    elif lab == "OWN-LABELS-BETTER":
        out["actions"].append("H-BC dropped (conservative: against a handicapped P*)")
    else:
        out["actions"].append(f"{lab}: H-TEAMS first, with the D1 contrast")
    if d == "SELF-PLAY-DATA-BETTER":
        pending = restriction_lb is None
        why = [w for w, bad in (("Delta_restriction LB > 0 (A6.1)", not pending and restriction_lb > -TIE),
                                ("Delta_labels HUMAN-LABELS-BETTER", lab == "HUMAN-LABELS-BETTER"),
                                ("Delta_states' LB > 0", read_["delta_states"]["lo"] > TIE),
                                ("S-H not fitted (at this budget)", not fitted),
                                ("the re-encode check blocks it (A5.7)", out["reencode_blocks"])) if bad]
        if why:
            out["actions"].append(f"SELF-PLAY-DATA-BETTER reported; closes nothing ({'; '.join(why)})")
        elif pending:
            out["actions"].append("SELF-PLAY-DATA-BETTER reported; closure pending Delta_restriction (A6.1)")
        else:
            out["actions"].append(CLOSURE_TEXT)
    if d not in ("HUMAN-DATA-BETTER", UNINF):
        out["actions"].append("the conditional >= 1500 follow-up fires (S-H>=1500 vs S-H-mix vs S-S at matched rows)")
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
