"""The KNOCK-IN FORKS' registered read (``designs/endstate/design_hidden_info_binding.md`` §8.4–§8.5), CODED BEFORE ANY
CELL IS PLAYED, with its rehearsal on synthetic matrices.

THE DATA (§8.4, as played — decisions in the Training session's DECISIONS.md, summarised here):

* 15 forks of ``rb_es_end2_s2001/final_model.zip`` (15,047,835) to 18,047,835 at P_hib ``64a695dc``: arms ``moves``
  (``ki_moves`` = ``moves:seen@0.95``), ``items`` (``ki_items``), ``spreads`` (``ki_spreads``), ``species``
  (``species:unseen@0.95``) and the NO-CHANGE control ``ctrl``, seeds 3101–3103; ``--trunk-precision bf16`` on every
  arm; ``--behaviour-check warn`` (Amendment 2, uniform) with |d log π| < 1e-4 held BY READ at each fork's bank.
* PRIMARY cross per lever L: L's 3 seeds (rows) × ctrl's 3 seeds (columns), mirrored, 1,000 pairs per cell,
  ``main.h2h`` ``--oracle-reveal-mode one_sided`` (L told at its recorded level, ctrl as trained; protocol
  ``gen3_eval_protocol_v1_h2h_oracle_one_sided``): h[i][j] = L seed i's score (pp) against ctrl seed j.
* v8 read: ctrl's 3 seeds × the parent (mode ``off``): what +3M of plain training buys.
* HONEST cross (secondary): L with its reveal WITHHELD at eval vs ctrl — played when the ``withheld`` mode lands; read
  with the same statistic, descriptively (§8.4 assigns it no rule).

THE STATISTIC: X5 §7.4's, unchanged (``main.h2h.cross.cross_stat``): Δ̂ = mean(h) − 50, V̂ = (s²_R + s²_C)/n, df =
2(n − 1) = 4 at n = 3. §8.4 names "the X5 A/B's cross statistic and interval", so the interval every rule below reads
is X5's REPORTED one: the TWO-SIDED 95 % Student-t interval Δ̂ ± t_{0.975, 4}·√V̂ (``CrossStat.ci95``).

THE RULES (§8.5; rule 8: a bound within ±0.05 pp of a boundary is NOT clear — :data:`RULE8_PP`):

* K-VALID: every fork VALID (its bank: pin, steps, strict-by-read) — an input (:func:`decide`'s ``invalid``); any
  invalid fork ⇒ INCONCLUSIVE. The placebo clause ("Δ_WRONG lower > +0.05 ⇒ every lever verdict VOID") is
  UNREADABLE: no wrong-moves arm exists (decision (a): the training observation refuses a placebo source).
* Per lever L, in this order:
  - **K-L-WORTH** requires Δ_L lower > +0.05 AND (Δ_L − Δ_WRONG) lower > +0.05. The second clause is UNREADABLE
    (decision (a)), so K-L-WORTH is NEVER declared in this batch: a lever whose Δ_L lower clears +0.05 reads
    ``WORTH_FIRST_CLAUSE_ONLY`` (the told network beats the control; that it is the CONTENT, not any confident told
    input, is not shown).
  - **K-L-NULL**: Δ_L's interval inside ±2.0 pp (lower > −2.0 AND upper < +2.0, both clear of rule 8) ⇒ ``NULL``.
  - otherwise **K-L-NOT-DETECTED** ⇒ ``NOT_DETECTED``.
  - A lever's own verdict never depends on another lever's.
* **The moves EXTENSION** (registered): Δ_MOVES ``NOT_DETECTED`` with an interval WIDER than ±2.5 pp — read as its
  half-width t·√V̂ > 2.5 pp, clear of rule 8 — ⇒ ``EXTEND``: moves + ctrl +3M more (to +6M), same seeds, BEFORE the
  verdict is read. A half-width within ±0.05 of 2.5 does NOT trigger it (rule 8: not clear ⇒ the clause fails).
* **K-RANK** (§8.5: L ∈ {items, spreads}; ``species`` added as a DECLARED descriptive extension, the arm postdates
  the registration): the PAIRED contrast d[i][j] = h_moves[i][j] − h_L[i][j] (same seed i, same control column j),
  read with the same statistic on 50 + d; ``MOVES_FIRST`` iff every contrast's lower > +0.05 pp (each clear of
  rule 8), else ``NOT_SHOWN``. Declared: "paired cells" = identical (seed, control seed) cells.
* **v8** (DESCRIPTIVE; decision (c): CONFOUNDED by the learnset data change + 3M steps): the 3 cells' scores and
  their mean − 50; no rule.

    python read_hib.py rehearse                   # every verdict branch + rule 8 on constructed matrices (exit 0 = OK)
    python read_hib.py decide <matrices.json>     # {"moves": h, "items": h, ..., "invalid": [...]} → the verdict JSON

Run from a checkout's ``src`` (``main.h2h.cross`` is imported from it).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from main.h2h.cross import CrossInputError, cross_stat, t_ppf

LEVERS = ("moves", "items", "spreads", "species")
RANK_REGISTERED = ("items", "spreads")
RANK_EXTENSION = ("species",)
RULE8_PP = 0.05
WORTH_BAR_PP = 0.05
NULL_BAR_PP = 2.0
EXTEND_HALF_WIDTH_PP = 2.5
CI_Q = 0.975  # the two-sided 95 % interval X5 §7.4 reports

UNREADABLE_A = ("UNREADABLE (decision (a)): no wrong-moves arm — the training observation refuses a placebo source; "
                "K-VALID's placebo clause and K-L-WORTH's Δ_L − Δ_WRONG cannot be read in this batch")
CONFOUNDED_C = ("CONFOUNDED (decision (c)): the control trained on the learnset-fixed data/ (64a695dc) and its parent "
                "did not, so control-vs-parent mixes +3M steps with the data change")


def _clear_above(x: float, bar: float) -> Optional[bool]:
    """True / False when x is clear of ``bar`` by more than rule 8's ±0.05, None when within it (not clear)."""
    if abs(x - bar) <= RULE8_PP:
        return None
    return x > bar


def interval(h: Sequence[Sequence[float]]) -> Dict[str, Any]:
    st = cross_stat(h)
    q = t_ppf(CI_Q, st.df)
    lo, hi = st.delta_hat - q * st.se, st.delta_hat + q * st.se
    return {"delta_hat_pp": st.delta_hat, "se_pp": st.se, "df": st.df, "t_0.975": q, "ci95_pp": [lo, hi],
            "half_width_pp": q * st.se, "row_means_pp": list(st.row_means), "col_means_pp": list(st.col_means)}


def lever_verdict(h: Sequence[Sequence[float]]) -> Dict[str, Any]:
    """One lever's §8.5 verdict on its n × n matrix (rows = L seeds, columns = control seeds; pp)."""
    iv = interval(h)
    lo, hi = iv["ci95_pp"]
    near: List[str] = []
    worth = _clear_above(lo, WORTH_BAR_PP)
    if worth is None:
        near.append(f"lower {lo:.4f} vs +{WORTH_BAR_PP}")
    null_lo = _clear_above(lo, -NULL_BAR_PP)
    null_hi = _clear_above(hi, NULL_BAR_PP)
    if null_lo is None:
        near.append(f"lower {lo:.4f} vs -{NULL_BAR_PP}")
    if null_hi is None:
        near.append(f"upper {hi:.4f} vs +{NULL_BAR_PP}")
    if worth is True:
        outcome = "WORTH_FIRST_CLAUSE_ONLY"
    elif null_lo is True and null_hi is False:
        outcome = "NULL"
    else:
        outcome = "NOT_DETECTED"
    return {"outcome": outcome, "k_l_worth": "NOT DECLARABLE — " + UNREADABLE_A, "near_boundary": near, **iv}


def extension(moves: Dict[str, Any]) -> Dict[str, Any]:
    """The registered moves-extension trigger from the moves lever's verdict."""
    hw = moves["half_width_pp"]
    wide = _clear_above(hw, EXTEND_HALF_WIDTH_PP)
    trig = moves["outcome"] == "NOT_DETECTED" and wide is True
    return {"extend": trig, "half_width_pp": hw, "near_boundary": [] if wide is not None else
            [f"half-width {hw:.4f} vs {EXTEND_HALF_WIDTH_PP}"]}


def rank(h_moves: Sequence[Sequence[float]], h_l: Sequence[Sequence[float]]) -> Dict[str, Any]:
    """K-RANK's paired contrast moves − L (same seed row i, same control column j)."""
    d = [[50.0 + a - b for a, b in zip(ra, rb)] for ra, rb in zip(h_moves, h_l)]
    try:
        iv = interval(d)
    except CrossInputError as e:  # a contrast with zero variance (identical seed means): t undefined ⇒ not shown
        return {"moves_first": False, "degenerate": str(e), "near_boundary": []}
    lo = iv["ci95_pp"][0]
    c = _clear_above(lo, WORTH_BAR_PP)
    return {"moves_first": c is True, "near_boundary": [] if c is not None else [f"lower {lo:.4f} vs +{WORTH_BAR_PP}"],
            "contrast_hat_pp": iv["delta_hat_pp"], **{k: v for k, v in iv.items() if k != "delta_hat_pp"}}


def v8(scores_pp: Sequence[float]) -> Dict[str, Any]:
    m = sum(scores_pp) / len(scores_pp)
    return {"descriptive": True, "stamp": CONFOUNDED_C, "cells_pp": list(scores_pp), "mean_minus_50_pp": m - 50.0}


def decide(mats: Dict[str, Any]) -> Dict[str, Any]:
    """The batch's verdict. ``mats``: {lever: n × n matrix (pp) or None, ..., "invalid": [reasons], "v8": [pp, ...],
    "inconclusive": {lever: [reasons]}}. Any K-VALID reason ⇒ the batch is INCONCLUSIVE (never interpreted)."""
    invalid = list(mats.get("invalid") or [])
    out: Dict[str, Any] = {"k_valid": {"invalid": invalid, "placebo_clause": UNREADABLE_A},
                           "rule8_pp": RULE8_PP, "interval": "two-sided 95 % Student-t (X5 §7.4's reported interval)"}
    if invalid:
        out["outcome"] = "INCONCLUSIVE"
        return out
    levers: Dict[str, Any] = {}
    for lv in LEVERS:
        reasons = (mats.get("inconclusive") or {}).get(lv) or []
        h = mats.get(lv)
        if reasons or h is None:
            levers[lv] = {"outcome": "INCONCLUSIVE", "reasons": list(reasons) or ["no matrix"]}
            continue
        try:
            levers[lv] = lever_verdict(h)
        except CrossInputError as e:  # zero variance across both arms' seed means: never interpreted
            levers[lv] = {"outcome": "INCONCLUSIVE", "reasons": [f"statistic undefined: {e}"]}
    out["levers"] = levers
    if levers["moves"].get("outcome") not in (None, "INCONCLUSIVE"):
        out["moves_extension"] = extension(levers["moves"])
    rk: Dict[str, Any] = {}
    for lv in RANK_REGISTERED + RANK_EXTENSION:
        if all(levers[x].get("outcome") not in (None, "INCONCLUSIVE") for x in ("moves", lv)):
            rk[lv] = {**rank(mats["moves"], mats[lv]),
                      "registered": lv in RANK_REGISTERED}
    out["k_rank"] = {"pairs": rk, "MOVES_FIRST": (all(rk[x]["moves_first"] for x in RANK_REGISTERED)
                                                    if all(x in rk for x in RANK_REGISTERED) else None)}
    if mats.get("v8"):
        out["v8"] = v8(mats["v8"])
    out["outcome"] = "READ"
    return out


def _m(rows: Sequence[float], cols: Sequence[float], c: float, eps: Sequence[float] = (0.3, -0.2, 0.1)) -> List[List[float]]:
    """A 3 × 3 matrix 50 + c + a_i − b_j + e_ij (pp) with fixed small noise (deterministic)."""
    return [[50.0 + c + a - b + eps[(i + j) % 3] for j, b in enumerate(cols)] for i, a in enumerate(rows)]


def _shift_to(h: List[List[float]], which: str, target: float) -> List[List[float]]:
    """Shift every cell so the interval's ``which`` end ("lo" / "hi") sits EXACTLY on ``target`` (float)."""
    iv = interval(h)
    end = iv["ci95_pp"][0 if which == "lo" else 1]
    return [[x + (target - end) for x in r] for r in h]


def rehearse() -> Dict[str, Any]:
    tight_r, tight_c = (0.3, -0.2, -0.1), (0.2, -0.1, -0.1)
    wide_r, wide_c = (4.0, -3.0, -1.0), (3.0, -2.0, -1.0)
    base = _m(tight_r, tight_c, 0.0)
    other = _m((0.1, 0.25, -0.35), (-0.2, 0.3, -0.1), 0.0, eps=(-0.1, 0.25, -0.15))  # a second NULL-sized lever
    cases: Dict[str, Any] = {}
    cases["worth"] = lever_verdict(_shift_to(base, "lo", 1.0))
    cases["null"] = lever_verdict(base)
    cases["null_negative_inside"] = lever_verdict(_shift_to(base, "hi", -0.5))
    cases["not_detected_low"] = lever_verdict(_shift_to(base, "lo", -2.5))
    cases["not_detected_wide"] = lever_verdict(_m(wide_r, wide_c, 0.0))
    cases["rule8_worth_tie"] = lever_verdict(_shift_to(base, "lo", WORTH_BAR_PP))
    cases["rule8_worth_inside"] = lever_verdict(_shift_to(base, "lo", WORTH_BAR_PP + 0.04))
    cases["rule8_null_lower_tie"] = lever_verdict(_shift_to(base, "lo", -NULL_BAR_PP))
    cases["worth_over_null"] = lever_verdict(_shift_to(base, "hi", NULL_BAR_PP))
    wide = _m(wide_r, wide_c, 0.0)
    cases["extend"] = extension(lever_verdict(wide))
    cases["no_extend_narrow"] = extension(lever_verdict(_shift_to(base, "hi", -0.5)))
    hw_target = EXTEND_HALF_WIDTH_PP
    s = hw_target / interval(wide)["half_width_pp"]
    on_edge = [[50.0 + (x - 50.0) * s for x in r] for r in wide]
    cases["rule8_extend_tie"] = extension(lever_verdict(on_edge))
    moves_hi = _shift_to(base, "lo", 3.0)
    cases["rank_first"] = rank(moves_hi, other)
    cases["rank_not_shown"] = rank(base, other)
    full = decide({"moves": moves_hi, "items": other, "spreads": other, "species": _m(wide_r, wide_c, 0.0),
                   "v8": [51.0, 49.5, 52.0]})
    cases["batch"] = {k: full[k] for k in ("outcome", "moves_extension")} | {
        "levers": {k: v["outcome"] for k, v in full["levers"].items()}, "MOVES_FIRST": full["k_rank"]["MOVES_FIRST"],
        "v8_mean_minus_50": full["v8"]["mean_minus_50_pp"]}
    cases["rank_degenerate"] = rank(base, base)
    cases["batch_invalid"] = decide({"invalid": ["rb_hib_items_s3102: a live |d log pi| >= 1e-4"]})["outcome"]
    cases["batch_missing"] = decide({"moves": base, "items": None, "spreads": base, "species": base,
                                     "inconclusive": {"items": ["cell (s3102, ctrl s3101) missing"]}})["levers"]["items"]["outcome"]
    want = {
        "worth": "WORTH_FIRST_CLAUSE_ONLY", "null": "NULL", "null_negative_inside": "NULL",
        "not_detected_low": "NOT_DETECTED", "not_detected_wide": "NOT_DETECTED", "rule8_worth_tie": "NULL",
        "rule8_worth_inside": "NULL", "rule8_null_lower_tie": "NOT_DETECTED", "worth_over_null": "WORTH_FIRST_CLAUSE_ONLY",
    }
    ok = all(cases[k]["outcome"] == v for k, v in want.items())
    ok &= bool(cases["rule8_worth_tie"]["near_boundary"]) and bool(cases["rule8_worth_inside"]["near_boundary"])
    ok &= bool(cases["rule8_null_lower_tie"]["near_boundary"]) and bool(cases["worth_over_null"]["near_boundary"])
    ok &= cases["extend"]["extend"] is True and cases["no_extend_narrow"]["extend"] is False
    ok &= cases["rule8_extend_tie"]["extend"] is False and bool(cases["rule8_extend_tie"]["near_boundary"])
    ok &= cases["rank_first"]["moves_first"] is True and cases["rank_not_shown"]["moves_first"] is False
    ok &= cases["batch"]["levers"] == {"moves": "WORTH_FIRST_CLAUSE_ONLY", "items": "NULL", "spreads": "NULL",
                                       "species": "NOT_DETECTED"}
    ok &= cases["batch"]["MOVES_FIRST"] is True and cases["batch"]["moves_extension"]["extend"] is False
    ok &= abs(cases["batch"]["v8_mean_minus_50"] - 0.8333333333) < 1e-6
    ok &= cases["batch_invalid"] == "INCONCLUSIVE" and cases["batch_missing"] == "INCONCLUSIVE"
    ok &= cases["rank_degenerate"]["moves_first"] is False and "degenerate" in cases["rank_degenerate"]
    return {"rehearsal": "OK" if ok else "MISMATCH", "cases": cases}


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "rehearse"
    if cmd == "rehearse":
        res = rehearse()
        print(json.dumps(res, indent=1, default=str))
        sys.exit(0 if res["rehearsal"] == "OK" else 1)
    if cmd == "decide":
        print(json.dumps(decide(json.loads(Path(sys.argv[2]).read_text())), indent=1, default=str))
        sys.exit(0)
    raise SystemExit(f"unknown command {cmd!r}")
