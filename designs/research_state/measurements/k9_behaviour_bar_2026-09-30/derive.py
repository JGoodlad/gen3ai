"""Derive K9(b)'s precision-keyed bar and the fault margins from ``result.json`` (``measure.py``).

For each precision, POOLED over its eager and compiled arms (the bar must hold for both):

* the healthy per-row p99.9 and the healthy per-MICRO-BATCH max (the gate reads the max over a micro);
* the bar = ``MULTIPLE[precision]`` x the healthy per-row p99.9 (fp32 keeps the fixed 1e-4, reported
  as a multiple for comparison);
* per fault, the SMALLEST micro-batch max (the least a fault can show the gate) / the bar = the margin,
  and the share of fault micro-batches the bar catches;
* the same read for two robust statistics (the micro's p99 and mean |Δ|), for comparison;
* ``--localized``: the LOCALIZED faults from ``corrupt.py``'s durable parts — per fault kind, which TF32
  condition catches it (the p99 should NOT, the max should), and the corrupted rows' own |Δ|.

    python designs/research_state/measurements/k9_behaviour_bar_2026-09-30/derive.py [--write]
    python designs/research_state/measurements/k9_behaviour_bar_2026-09-30/derive.py --localized
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
#: The DECLARED gates (``consistency.BEHAVIOUR_GATES``): fp32 = max < 1e-4 (fixed); TF32 = the
#: micro-batch's p99 < MULTIPLE x the healthy per-row p99.9, rounded up to 2 significant figures.
#: ``MAX_OPTION_MULTIPLE`` is the rejected alternative (a max bar clear of TF32's healthy max, reported).
MULTIPLE = {"high": 3.0}
STATISTIC = {"highest": "max", "high": "p99"}
MAX_OPTION_MULTIPLE = 60.0
FP32_BAR = 1e-4
COLS = {"max": 0, "p99": 1, "mean": 2}


def _round_up(x: float) -> float:
    """Round up to 2 significant figures (a declared bar reads as a round number)."""
    e = math.floor(math.log10(x)) - 1
    return math.ceil(x / 10 ** e) * 10 ** e


def derive(res: dict) -> dict:
    out = {}
    for prec in ("highest", "high"):
        arms = [v for k, v in res["arms"].items() if k.startswith(prec + "/")]
        p999 = max(a["healthy_per_row"]["p99_9"] for a in arms)
        row_max = max(a["healthy_per_row"]["max"] for a in arms)
        hm = {c: np.array([r[i] for a in arms for r in a["healthy_micro_stats"]]) for c, i in COLS.items()}
        stat = STATISTIC[prec]
        col = COLS[stat]
        bar = FP32_BAR if prec == "highest" else _round_up(MULTIPLE[prec] * p999)
        faults = {}
        for a in arms:
            for f, rows in a["faults_micro_stats"].items():
                faults.setdefault(f, []).extend(rows)
        d = {"statistic": stat, "healthy_per_row_p99_9": p999, "healthy_per_row_max": row_max,
             "n_healthy_rows": int(sum(a["healthy_per_row"]["n"] for a in arms)),
             "n_healthy_micro": int(hm["max"].size), "bar": bar, "bar_over_p99_9": bar / p999,
             "healthy_statistic_max": float(hm[stat].max()),
             "healthy_micros_over_bar": int((hm[stat] >= bar).sum()),
             "headroom_bar_over_healthy_statistic_max": bar / float(hm[stat].max()), "faults": {},
             "per_statistic": {}}
        for f, rows in faults.items():
            r = np.array(rows)
            d["faults"][f] = {"n_micro": int(len(r)), "min": float(r[:, col].min()),
                              "median": float(np.median(r[:, col])),
                              "margin_min_over_bar": float(r[:, col].min() / bar),
                              "caught_share": float((r[:, col] >= bar).mean())}
        for c, i in COLS.items():
            hmax = float(hm[c].max())
            d["per_statistic"][c] = {"healthy_micro_max": hmax,
                                     **{f: {"fault_min": float(np.array(rows)[:, i].min()),
                                            "separation_fault_min_over_healthy_max":
                                                float(np.array(rows)[:, i].min() / hmax) if hmax > 0 else None}
                                        for f, rows in faults.items()}}
        if prec == "high":
            mbar = _round_up(MAX_OPTION_MULTIPLE * p999)
            d["rejected_max_option"] = {
                "bar": mbar, "healthy_max_headroom": mbar / float(hm["max"].max()),
                **{f: {"margin_min_over_bar": float(np.array(rows)[:, 0].min() / mbar),
                       "caught_share": float((np.array(rows)[:, 0] >= mbar).mean())} for f, rows in faults.items()}}
        out[prec] = d
    return out


def main() -> int:
    path = HERE / "result.json"
    res = json.loads(path.read_text())
    res["derived"] = derive(res)
    print(json.dumps(res["derived"], indent=1))
    if "--write" in sys.argv:
        path.write_text(json.dumps(res, indent=1) + "\n")
    return 0




# ------------------------------------------------------------------ the LOCALIZED faults (corrupt.py)
GATES = {"highest": (("max", 1e-4),), "high": (("p99", 3.6e-3), ("max", 0.071))}


def derive_localized(parts_dir: Path = HERE / "parts") -> dict:
    """Per source (``high`` / ``highest`` on CUDA, ``cpu`` fp32) and fault kind: the micro-batch p99 and max,
    which condition catches it, and the corrupted rows' own |Δ|. Reads whatever durable parts exist."""
    import glob

    out = {}
    for source, pattern, gate in (("cuda_high", "corrupt_high_s*.json", GATES["high"]),
                                  ("cuda_highest", "corrupt_highest_s*.json", GATES["highest"]),
                                  ("cpu_fp32", "corrupt_cpu_s*.json", GATES["highest"])):
        files = sorted(glob.glob(str(parts_dir / pattern)))
        if not files:
            continue
        parts = [json.loads(Path(f).read_text()) for f in files]
        d = {"seeds": len(files)}
        h = np.array([r for p in parts for r in p["healthy_micro"]])
        d["healthy"] = {"n_micro": int(len(h)), "max_max": float(h[:, 0].max()), "p99_max": float(h[:, 1].max())}
        for kind in ("action_index", "obs_swap", "mask_mismatch"):
            m = np.array([r for p in parts for r in p[kind]["micro"]])
            rows = np.array([r for p in parts for r in p[kind]["rows"]])
            caught = {stat: (m[:, COLS[stat]] >= bar) for stat, bar in gate}
            d[kind] = {"n_micro": int(len(m)), "micro_max_min": float(m[:, 0].min()),
                       "micro_max_median": float(np.median(m[:, 0])), "micro_p99_max": float(m[:, 1].max()),
                       **{f"caught_by_{s}": float(c.mean()) for s, c in caught.items()},
                       "caught_by_gate": float(np.any(np.stack(list(caught.values())), axis=0).mean()),
                       "rows_n": int(rows.size), "rows_min": float(rows.min()),
                       "rows_median": float(np.median(rows)),
                       # a single corrupted row the TF32 max cannot see (a micro-batch holding only such
                       # rows passes) — the residual blind spot, measured
                       "rows_share_under_tf32_max": float((rows < GATES["high"][1][1]).mean()),
                       "rows_illegal": int(sum(p[kind]["illegal"] for p in parts))}
        out[source] = d
    return out


if __name__ == "__main__":
    if "--localized" in sys.argv:
        print(json.dumps(derive_localized(), indent=1))
    else:
        raise SystemExit(main())
