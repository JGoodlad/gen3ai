"""The bf16 STRENGTH A/B's STAGE-1 decision rule, coded before any cell is played, and its self-check.

Registration: ``designs/endstate/design_bf16_strength_ab.md`` §4 (REGISTERED 2026-10-12, 981013f4, before any seed).

THE RULE. h[i][j] = EB (``--trunk-precision bf16``) seed i's score (pp, a draw = ½) against E (fp32) seed j at the
matched 5M finals, one mirrored cell of 1,000 pairs; n = 4 seeds per arm. The estimator is X5 §7.4's, unchanged
(``main.h2h.cross.cross_stat``): Δ̂ = mean(h) − 50, V̂ = (s²_R + s²_C) / n, df = 2(n − 1) = 6. The two-sided 90 %
interval is Δ̂ ± t_{0.95, 6} · √V̂ (t_{0.95, 6} = 1.9432).

* STOP (futility) iff the interval's UPPER end < −2.0 pp.
* CONTINUE (to stage 2) otherwise. No other stage-1 outcome exists (no early PASS).
* INCONCLUSIVE when an input is incomplete or invalid (never interpreted).

RULE 8 (standing rule 8): an upper end within ``RULE8_EPS`` (1e-9 pp) of −2.0 does NOT stop, and the read records it
in ``near_boundary``.

    python futility_rule.py self-check      # three constructed matrices: STOP, CONTINUE, and the rule-8 tie
    python futility_rule.py decide <h.json> # h.json = {"h_pp": [[...], ...]} → one JSON verdict

Run from a checkout's ``src`` (``main.h2h.cross`` is imported from it).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

from main.h2h.cross import RULE8_EPS, cross_stat, t_ppf

STOP, CONTINUE, INCONCLUSIVE = "STOP_FUTILITY", "CONTINUE", "INCONCLUSIVE"
N_SEEDS = 4
FLOOR_PP = -2.0
CI_LEVEL = 0.90


def decide(h: Sequence[Sequence[float]], inconclusive: Sequence[str] = ()) -> Dict[str, Any]:
    """The registered stage-1 verdict on the n × n matrix ``h`` (rows = EB, columns = E; pp)."""
    if inconclusive:
        return {"outcome": INCONCLUSIVE, "reasons": list(inconclusive)}
    if len(h) != N_SEEDS:
        return {"outcome": INCONCLUSIVE, "reasons": [f"the registered stage-1 read is {N_SEEDS} x {N_SEEDS}; got {len(h)} rows"]}
    st = cross_stat(h)
    q = t_ppf(0.5 + CI_LEVEL / 2.0, st.df)
    lo, hi = st.delta_hat - q * st.se, st.delta_hat + q * st.se
    near = ["upper end vs -2.0"] if abs(hi - FLOOR_PP) <= RULE8_EPS else []
    stop = hi < FLOOR_PP and not near
    return {"outcome": STOP if stop else CONTINUE, "upper_below_minus_2": stop, "interval90_pp": [lo, hi],
            "t_0.95": q, "delta_hat_pp": st.delta_hat, "se_pp": st.se, "df": st.df, "s2_rows": st.s2_rows,
            "s2_cols": st.s2_cols, "near_boundary": near, "stat": st.as_dict()}


def _base() -> List[List[float]]:
    return [[50.0 + a - b + e for b, e in zip((0.0, 1.0, -1.0, 0.5), es)]
            for a, es in zip((1.5, -0.5, 0.0, -1.0), ((0.3, -0.2, 0.1, 0.0), (-0.1, 0.2, 0.0, -0.3),
                                                     (0.2, 0.0, -0.2, 0.1), (0.0, 0.1, 0.2, -0.1)))]


def self_check() -> Dict[str, Any]:
    """Three constructed 4 × 4 matrices. A constant shift moves Δ̂ and both ends by exactly that shift (V̂ unchanged),
    so the same base placed at three offsets must read CONTINUE (centred on 0), STOP (upper end at −3) and CONTINUE
    with ``near_boundary`` (upper end placed ON −2.0, to float precision)."""
    base = _base()
    hi0 = decide(base)["interval90_pp"][1]
    shift = lambda c: [[x + c for x in r] for r in base]  # noqa: E731
    d_mid, d_stop, d_tie = decide(base), decide(shift(-3.0 - hi0)), decide(shift(FLOOR_PP - hi0))
    ok = (d_mid["outcome"] == CONTINUE and not d_mid["near_boundary"]
          and d_stop["outcome"] == STOP and abs(d_stop["interval90_pp"][1] + 3.0) < 1e-9
          and d_tie["outcome"] == CONTINUE and d_tie["near_boundary"] == ["upper end vs -2.0"]
          and abs(d_mid["t_0.95"] - 1.9431802805153042) < 1e-12 and d_mid["df"] == 6)
    return {"self_check": "OK" if ok else "MISMATCH",
            "cases": {k: {"outcome": d["outcome"], "interval90_pp": d["interval90_pp"], "near_boundary": d["near_boundary"]}
                      for k, d in (("centred", d_mid), ("futile", d_stop), ("rule8_tie", d_tie))}}


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "self-check"
    if cmd == "self-check":
        out = self_check()
    elif cmd == "decide":
        out = decide(json.loads(Path(sys.argv[2]).read_text())["h_pp"])
    else:
        raise SystemExit(f"unknown command {cmd!r}")
    print(json.dumps(out, indent=1, default=str))
    sys.exit(0 if out.get("self_check", "OK") == "OK" else 1)
