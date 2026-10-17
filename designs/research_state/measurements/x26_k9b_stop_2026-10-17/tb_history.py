"""The K9(b) probe's history in rb_x26_s2601's TensorBoard: per update, the max and p99 |d log pi| over the
probe's current rows (`behaviour/max_abs_dlogp_current`, `behaviour/p99_abs_dlogp_current`), and the
policy-sharpness context (entropy). Fits the trend of the per-update max and forecasts the rate of a
max >= 1e-4 recurrence.

    python3 tb_history.py --tb models/rb_x26_s2601/tb --out tb_history.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Dict, List, Tuple

import numpy as np

TAGS = ("behaviour/max_abs_dlogp_current", "behaviour/p99_abs_dlogp_current", "behaviour/rows_current",
        "behaviour/excluded_frac", "train/entropy_loss", "rollout/policy_entropy", "train/approx_kl")


def read(tb: str) -> Dict[str, List[Tuple[int, float]]]:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    series: Dict[str, Dict[int, float]] = {}
    for f in sorted(glob.glob(os.path.join(tb, "events.out.tfevents.*"))):
        ea = EventAccumulator(f, size_guidance={"scalars": 0})
        ea.Reload()
        for tag in ea.Tags().get("scalars", []):
            if tag in TAGS or tag.startswith("behaviour/"):
                d = series.setdefault(tag, {})
                for ev in ea.Scalars(tag):
                    d[int(ev.step)] = float(ev.value)
    return {k: sorted(v.items()) for k, v in series.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tb", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    s = read(a.tb)
    # the GATE's statistic is the max over the JUDGED rows (tie-excluded rows are not judged); the
    # all-current max (`max_abs_dlogp_current`) includes selection flips at ties and is reported beside it
    mx = np.array(s.get("behaviour/max_abs_dlogp_judged", []))
    allc = np.array(s.get("behaviour/max_abs_dlogp_current", []))
    out_all: Dict[str, object] = {}
    if len(allc):
        out_all = {"n": int(len(allc)), "over_1e-4": int((allc[:, 1] >= 1e-4).sum()),
                   "top5": [{"step": int(allc[i, 0]), "max": float(allc[i, 1])} for i in np.argsort(-allc[:, 1])[:5]]}
    p99 = np.array(s.get("behaviour/p99_abs_dlogp_current", []))
    out: Dict[str, object] = {"tags": sorted(s), "n_updates_logged": int(len(mx)),
                              "max_all_current_incl_ties": out_all}
    if len(mx):
        steps, v = mx[:, 0], mx[:, 1]
        out["first_step"], out["last_step"] = int(steps[0]), int(steps[-1])
        # windows of ~10M steps: the max's quantiles and the share of updates over fractions of the bar
        edges = np.arange(np.floor(steps[0] / 1e7) * 1e7, steps[-1] + 1e7, 1e7)
        win = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (steps >= lo) & (steps < hi)
            if not m.any():
                continue
            w = v[m]
            pw = p99[(p99[:, 0] >= lo) & (p99[:, 0] < hi), 1] if len(p99) else np.array([])
            win.append({"lo_M": lo / 1e6, "hi_M": hi / 1e6, "n": int(m.sum()), "max_median": float(np.median(w)),
                        "max_p90": float(np.quantile(w, 0.9)), "max_max": float(w.max()),
                        "frac_over_5e-5": float((w >= 5e-5).mean()), "frac_over_1e-4": float((w >= 1e-4).mean()),
                        "p99_median": float(np.median(pw)) if pw.size else None})
        out["windows_10M"] = win
        # log-linear trend of the per-update max over the last 40M steps
        m = steps >= steps[-1] - 4e7
        if m.sum() > 10:
            k, c = np.polyfit(steps[m] / 1e6, np.log10(v[m]), 1)
            out["trend_last40M"] = {"log10_max_per_M_steps": float(k), "x10_every_M_steps": float(1 / k) if k > 0 else None,
                                    "median_now": float(10 ** (c + k * steps[-1] / 1e6))}
        # FORECAST: the per-update exceedance probability of the judged max in the last 30M steps (the FATAL
        # update itself is not in TB -- it raised before the record was written -- and is added by hand), a
        # log-linear tail fit over thresholds 3e-5..9e-5, extrapolated to the bar and to warn ceilings
        m30 = steps >= steps[-1] - 3e7
        tail = np.concatenate([v[m30], [1.010894775390625e-4]])
        thr = np.array([3e-5, 4e-5, 5e-5, 6e-5, 7e-5, 8e-5, 9e-5])
        p = np.array([(tail >= x).mean() for x in thr])
        ok = p * tail.size >= 2
        b, c = np.polyfit(thr[ok], np.log10(p[ok]), 1)
        fc = {f"{x:g}": float(10 ** (c + b * x)) for x in (1e-4, 1.5e-4, 2e-4, 5e-4, 1e-3)}
        out["forecast_last30M"] = {
            "n_updates": int(tail.size), "empirical_exceedance": {f"{x:g}": float(q) for x, q in zip(thr, p)},
            "empirical_over_1e-4": float((tail >= 1e-4).mean()),
            "fit_log10p_per_unit": float(b), "fit_intercept": float(c),
            "fit_thresholds_used": [float(x) for x in thr[ok]],
            "p_per_update": fc,
            "updates_between_events": {k: (1.0 / q if q > 0 else None) for k, q in fc.items()},
            "steps_per_update": float(np.median(np.diff(steps[m30]))),
        }
        out["top10"] = [{"step": int(steps[i]), "max": float(v[i])} for i in np.argsort(-v)[:10]]
    out["series_tail"] = {k: v[-5:] for k, v in s.items()}
    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "series_tail"}, indent=1))


if __name__ == "__main__":
    main()
