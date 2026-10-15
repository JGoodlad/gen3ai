"""The X26 exit check's SANITY SCREEN (E7) decision rule, coded before any seed or cell exists, and its self-check.

Registration: ``designs/endstate/design_x26_readiness.md`` §4 row E7 (written 2026-10-15, before P_X26 and before any
``rb_xs_*`` seed). It is a BREAKAGE screen after the legacy deletion's init-stream change (U3, D-LD7), never an
equivalence or adoption test.

THE RULE. h[i][j] = sanity seed i's score (pp, a draw = ½) against EB seed j's 5M model, one mirrored cell of 500
pairs; rows = the 2 fresh production seeds at P_X26 (``rb_xs_s2701``, ``rb_xs_s2702``), columns = the 8 EB seeds
(``rb_bf_eb_s2001``–``s2008``; s2001–s2004 ``final_model_stage1_5M.zip``, s2005–s2008 their 5M checkpoint
``checkpoints/checkpoint_5000*_steps.zip`` — those four ran fresh to 15M and have no stage-1 file).

* Pooled Δ̂ = mean(h) − 50; per-seed Δ̂_i = mean(h[i]) − 50.
* PASS iff pooled Δ̂ ≥ −3.0 pp AND every per-seed Δ̂_i ≥ −6.0 pp AND every run gate is clean (each seed reached 5M,
  no crash, no NaN, no K9(b) stop that is not a tie-share false alarm — Amendment 2 §A2.7's definition).
* FAIL otherwise ⇒ STOP and bisect the deletion units (U0, U1, U3).
* INCONCLUSIVE when an input is incomplete or invalid (never interpreted).

RULE 8 (standing rule 8): a value within ``RULE8_EPS`` (1e-9 pp) of a bound is NOT a crossing (it does not fail), and
the read records it in ``near_boundary``.

    python sanity_rule.py self-check        # constructed matrices: PASS, pooled FAIL, per-seed FAIL, the two rule-8 ties
    python sanity_rule.py decide <in.json>  # {"h_pp": [[8 floats], [8 floats]], "run_gates": {...}} → one JSON verdict

``run_gates`` = {"rb_xs_s2701": {"reached_5M": true, "crash": false, "nan": false, "k9b_real_stop": false}, ...}.
Run from a checkout's ``src`` (``main.h2h.cross`` is imported from it).
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

from main.h2h.cross import RULE8_EPS

PASS, FAIL, INCONCLUSIVE = "PASS", "FAIL", "INCONCLUSIVE"
ROWS, COLS = 2, 8
POOLED_FLOOR_PP = -3.0
SEED_FLOOR_PP = -6.0
SEEDS = ("rb_xs_s2701", "rb_xs_s2702")
GATE_KEYS = {"reached_5M": True, "crash": False, "nan": False, "k9b_real_stop": False}


def _gate_problems(run_gates: Mapping[str, Mapping[str, Any]]) -> List[str]:
    out = []
    for s in SEEDS:
        g = run_gates.get(s)
        if g is None:
            out.append(f"{s}: no run-gate record")
            continue
        for k, good in GATE_KEYS.items():
            if k not in g:
                out.append(f"{s}: run gate {k!r} not recorded")
            elif bool(g[k]) is not good:
                out.append(f"{s}: {k} = {g[k]!r}")
    return out


def decide(h: Sequence[Sequence[float]], run_gates: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    """The registered E7 verdict on the 2 × 8 matrix ``h`` (rows = sanity seeds, columns = EB seeds; pp)."""
    if len(h) != ROWS or any(len(r) != COLS for r in h):
        return {"outcome": INCONCLUSIVE, "reasons": [f"the registered E7 read is {ROWS} x {COLS}; got "
                                                     f"{[len(r) for r in h]}"]}
    if not all(isinstance(x, (int, float)) and math.isfinite(x) and 0.0 <= x <= 100.0 for r in h for x in r):
        return {"outcome": INCONCLUSIVE, "reasons": ["a cell is missing, non-finite or outside [0, 100] pp"]}
    missing = [p for p in _gate_problems(run_gates) if "not recorded" in p or "no run-gate" in p]
    if missing:
        return {"outcome": INCONCLUSIVE, "reasons": missing}
    pooled = sum(x for r in h for x in r) / (ROWS * COLS) - 50.0
    per_seed = [sum(r) / COLS - 50.0 for r in h]
    near = []
    if abs(pooled - POOLED_FLOOR_PP) <= RULE8_EPS:
        near.append("pooled vs -3.0")
    near += [f"{s} vs -6.0" for s, d in zip(SEEDS, per_seed) if abs(d - SEED_FLOOR_PP) <= RULE8_EPS]
    pooled_fail = pooled < POOLED_FLOOR_PP - RULE8_EPS
    seed_fail = [s for s, d in zip(SEEDS, per_seed) if d < SEED_FLOOR_PP - RULE8_EPS]
    gates = _gate_problems(run_gates)
    ok = not pooled_fail and not seed_fail and not gates
    return {"outcome": PASS if ok else FAIL, "pooled_delta_pp": pooled,
            "per_seed_delta_pp": dict(zip(SEEDS, per_seed)), "pooled_below_minus_3": pooled_fail,
            "seeds_below_minus_6": seed_fail, "run_gate_failures": gates, "near_boundary": near,
            "action": None if ok else "STOP: bisect the deletion units (U0, U1, U3) before X26"}


def _clean() -> Dict[str, Dict[str, Any]]:
    return {s: dict(GATE_KEYS) for s in SEEDS}


def self_check() -> Dict[str, Any]:
    """Constructed matrices: a centred PASS; a pooled FAIL (−3.5); a per-seed FAIL (row 2 at −7, pooled −2.5);
    the two rule-8 ties (pooled ON −3.0, a row ON −6.0: neither fails); a dirty run gate; a missing gate."""
    base = [[50.0 + e for e in (0.5, -0.5, 1.0, -1.0, 0.25, -0.25, 0.0, 0.0)]] * 2
    shift = lambda rows: [[x + c for x in r] for r, c in zip(base, rows)]  # noqa: E731
    c = {
        "centred": decide(base, _clean()),
        "pooled_fail": decide(shift((-3.5, -3.5)), _clean()),
        "seed_fail": decide(shift((2.0, -7.0)), _clean()),
        "rule8_pooled_tie": decide(shift((-3.0, -3.0)), _clean()),
        "rule8_seed_tie": decide(shift((1.0, -6.0)), _clean()),
        "dirty_gate": decide(base, {**_clean(), SEEDS[1]: {**GATE_KEYS, "k9b_real_stop": True}}),
        "missing_gate": decide(base, {SEEDS[0]: dict(GATE_KEYS)}),
    }
    ok = (c["centred"]["outcome"] == PASS and not c["centred"]["near_boundary"]
          and c["pooled_fail"]["outcome"] == FAIL and c["pooled_fail"]["pooled_below_minus_3"]
          and c["seed_fail"]["outcome"] == FAIL and c["seed_fail"]["seeds_below_minus_6"] == [SEEDS[1]]
          and not c["seed_fail"]["pooled_below_minus_3"]
          and c["rule8_pooled_tie"]["outcome"] == PASS and c["rule8_pooled_tie"]["near_boundary"] == ["pooled vs -3.0"]
          and c["rule8_seed_tie"]["outcome"] == PASS and c["rule8_seed_tie"]["near_boundary"] == [f"{SEEDS[1]} vs -6.0"]
          and c["dirty_gate"]["outcome"] == FAIL and c["missing_gate"]["outcome"] == INCONCLUSIVE)
    return {"self_check": "OK" if ok else "MISMATCH",
            "cases": {k: {kk: d.get(kk) for kk in ("outcome", "pooled_delta_pp", "per_seed_delta_pp", "near_boundary",
                                                    "run_gate_failures", "reasons")} for k, d in c.items()}}


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "self-check"
    if cmd == "self-check":
        out = self_check()
    elif cmd == "decide":
        doc = json.loads(Path(sys.argv[2]).read_text())
        out = decide(doc["h_pp"], doc.get("run_gates", {}))
    else:
        raise SystemExit(f"unknown command {cmd!r}")
    print(json.dumps(out, indent=1, default=str))
    sys.exit(0 if out.get("self_check", "OK") == "OK" else 1)
