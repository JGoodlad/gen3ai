"""How far outside the FITTED rounding tail are the two events? (the d29f781c forecast model, refitted BEFORE them)

The tail refresh (`../x26_k9b_tail_refresh_2026-10-17/forecast.py`) models each probe's judged max as
log x = T(s) + r: a trend T (linear / exponential / power in steps) and a stationary residual r (empirical body,
generalised-Pareto tail above its 95th percentile). This refits that model on the probes from 100M up to the
LAST probe before u32180 (the event itself and everything after it are held out), then asks, per form:

* P(x >= the event's value) at the event's probe, and the expected number of probes at or above it over
  [300M, the stop] under the fitted model;
* where the event's residual sits: its excess over the fitted GPD's upper endpoint (a negative shape xi
  bounds the tail at u + sigma / |xi|), and over the largest residual seen before it.

    python3 surprise.py --series tb_series.json --violations behaviour_violations.jsonl --out surprise.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from typing import Any, Dict

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "k9b_forecast_d29f781c", os.path.join(_HERE, "..", "x26_k9b_tail_refresh_2026-10-17", "forecast.py"))
F = importlib.util.module_from_spec(_spec)  # type: ignore[arg-type]
_spec.loader.exec_module(F)  # type: ignore[union-attr]

EVENTS = {"u32180": 2.973616123199463e-03, "u32320": 8.115768432617188e-04}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--series", required=True)
    ap.add_argument("--violations", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fit-from", type=float, default=100e6)
    a = ap.parse_args()
    d = F.load(a.series, a.violations)
    s_all, x_all = d["s"], d["max"]
    ev = {k: int(s_all[np.flatnonzero(np.isclose(x_all, v, rtol=1e-6, atol=0))[0]]) for k, v in EVENTS.items()}
    s_ev = ev["u32180"]
    fit = (s_all >= a.fit_from) & (s_all < s_ev)
    s, lx = s_all[fit], np.log(x_all[fit])
    window = (s_all >= 300e6)
    out: Dict[str, Any] = {"event_steps": ev, "n_fit": int(fit.sum()), "fit_range": [float(s[0]), float(s[-1])],
                           "n_probes_300M_to_stop": int(window.sum()), "forms": {}}
    for form in F.FORMS:
        f = F.FORMS[form][0]
        p = F.fit_trend(s, lx, form)
        r = lx - f(s, *p)
        tail = F.Tail(r)
        endpoint = tail.u + tail.sigma / abs(tail.xi) if tail.xi < 0 else float("inf")
        per: Dict[str, Any] = {"aic": float(s.size * np.log(np.sum(r ** 2) / s.size) + 4),
                               "tail": {"u": tail.u, "xi": float(tail.xi), "sigma": float(tail.sigma),
                                        "endpoint": endpoint, "r_max_before": float(r.max())}}
        for name, v in EVENTS.items():
            se = ev[name]
            T_e = float(f(np.array([float(se)]), *p)[0])
            r_e = float(np.log(v) - T_e)
            Tw = f(s_all[window].astype(np.float64), *p)
            pw = F.crossing(s_all[window], Tw, tail, v)
            per[name] = {"trend_median_at_event": float(np.exp(T_e + np.median(r))), "residual": r_e,
                         "residual_minus_endpoint": r_e - endpoint, "residual_minus_r_max_before": r_e - float(r.max()),
                         "P_at_event_probe": float(F.crossing(np.array([se]), np.array([T_e]), tail, v)[0]),
                         "expected_probes_at_or_above_300M_to_stop": float(pw.sum()),
                         "ratio_to_largest_judged_max_before": float(v / x_all[(s_all < s_ev)].max())}
        out["forms"][form] = per
        print(form, json.dumps(per), flush=True)
    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
