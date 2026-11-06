"""X65-F1 contention read (2026-11-05): the Kakuna-lane search workload with and without a pinned production trainer.

    python analyze.py [record dir] > summary.json

Two meters, both from the files in this directory:

1. RATE: scored decisions per minute over a span of a window's counts_<phase>.tsv (cumulative rows, one line a minute),
   beside the foreign load on the agent cores (foreign.tsv: CPU-seconds spent on CPUs 5-7, 13-15 by processes outside
   the workload's x65f1 scopes) over the sampler rows whose interval ends inside the span.
2. PAIRED CPU TIME: the same state (row key) searched in both conditions; the ratio of its mean cpu_s with the trainer
   to its mean cpu_s without, averaged over the states seen in both, with a 95 % bootstrap interval over states (seed
   fixed). A row's place in time is approximated by its worker's cumulative cpu_s (each worker is single-threaded on a
   dedicated CPU), and a span is selected by ROW COUNT: the rows ranked by that elapsed order, skipping the first
   `skip` and taking up to `take`, where skip / take are the counts.tsv totals at the span's wall-clock ends. (Selecting
   by a cpu_s threshold instead overshoots the wall clock — cpu_s omits each pass's process start-up — and would pull
   A's builder-confounded tail into its clean span.)
"""
import gzip
import json
import random
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

D = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent


def counts(phase):
    out = []
    for line in (D / f"counts_{phase}.tsv").read_text().splitlines():
        p = line.split("\t")
        out.append((int(p[0]), int(p[1]), float(p[2].split()[0])))
    return out


def nearest(rows, t):
    best = min(rows, key=lambda r: abs(r[0] - t))
    if abs(best[0] - t) > 45:
        raise SystemExit(f"no counts row within 45 s of {t}")
    return best


FOREIGN = []
for line in (D / "foreign.tsv").read_text().splitlines()[1:]:
    p = line.split("\t")
    FOREIGN.append({"t": int(p[0]), "dt": float(p[1]), "cpu_s": float(p[3]), "busy": float(p[4]), "frac6": float(p[5])})


def rate_window(name, phase, t0, t1, note):
    rows = counts(phase)
    a, b = nearest(rows, t0), nearest(rows, t1)
    minutes = (b[0] - a[0]) / 60
    fs = [f for f in FOREIGN if a[0] + 30 <= f["t"] <= b[0] + 30]
    load = [r[2] for r in rows if a[0] <= r[0] <= b[0]]
    return {
        "window": name, "phase": phase, "t0": a[0], "t1": b[0], "minutes": round(minutes, 2),
        "rows": b[1] - a[1], "rows_per_min": round((b[1] - a[1]) / minutes, 2),
        "foreign_minutes_covered": round(sum(f["dt"] for f in fs) / 60, 1),
        "foreign_cpu_s_total": round(sum(f["cpu_s"] for f in fs), 1),
        "foreign_frac_of_6_mean": round(statistics.mean(f["frac6"] for f in fs), 4) if fs else None,
        "foreign_frac_of_6_max": round(max(f["frac6"] for f in fs), 4) if fs else None,
        "loadavg1_median": round(statistics.median(load), 2) if load else None,
        "note": note,
    }


def rows_in(phase, skip, take):
    """The phase's rows ranked by their worker-elapsed cpu_s at the row's start; rows [skip, take) of that ranking."""
    by_worker = defaultdict(list)
    with gzip.open(D / f"rows_{phase}.jsonl.gz", "rt") as f:
        for line in f:
            r = json.loads(line)
            m = re.match(r"w(\d+)_p(\d+)\.", r["pass_file"])
            by_worker[int(m.group(1))].append((int(m.group(2)), r))
    ranked = []
    for lst in by_worker.values():
        lst.sort(key=lambda x: x[0])   # stable: keeps each pass's row order
        el = 0.0
        for _, r in lst:
            ranked.append((el, r))
            el += r["cpu_s"]
    ranked.sort(key=lambda x: x[0])
    return [r for _, r in ranked[skip:take]]


def paired(base_rows, test_rows, n_boot=10_000, seed=6599):
    b, t = defaultdict(list), defaultdict(list)
    for r in base_rows:
        b[r["key"]].append(r["cpu_s"])
    for r in test_rows:
        t[r["key"]].append(r["cpu_s"])
    keys = sorted(set(b) & set(t))
    ratios = [statistics.mean(t[k]) / statistics.mean(b[k]) for k in keys]
    rng = random.Random(seed)
    boots = sorted(statistics.mean(rng.choices(ratios, k=len(ratios))) for _ in range(n_boot))
    return {"states": len(keys), "mean_ratio": round(statistics.mean(ratios), 4),
            "ci95": [round(boots[int(0.025 * n_boot)], 4), round(boots[int(0.975 * n_boot) - 1], 4)],
            "median_ratio": round(statistics.median(ratios), 4)}


def main():
    spans = json.loads((D / "spans.json").read_text())
    rate = [rate_window(**s) for s in spans]
    by = {w["window"]: w for w in rate}

    def span_rows(w):
        start = nearest(counts(w["phase"]), w["t0"])[1]
        return rows_in(w["phase"], start, start + w["rows"])

    a_clean, b_train, a2 = span_rows(by["A_clean"]), span_rows(by["B2_training"]), span_rows(by["A2_clean"])
    mism = sum(len(r["mismatches"]) for r in a_clean + b_train + a2)
    base_rate = (by["A_clean"]["rows"] + by["A2_clean"]["rows"]) / (by["A_clean"]["minutes"] + by["A2_clean"]["minutes"])
    out = {
        "rate_windows": rate,
        "rate_ratio": {
            "B2_training_vs_A_clean": round(by["B2_training"]["rows_per_min"] / by["A_clean"]["rows_per_min"], 4),
            "B2_training_vs_A2_clean": round(by["B2_training"]["rows_per_min"] / by["A2_clean"]["rows_per_min"], 4),
            "B2_training_vs_pooled_clean": round(by["B2_training"]["rows_per_min"] / base_rate, 4),
            "A_clean_vs_A2_clean": round(by["A_clean"]["rows_per_min"] / by["A2_clean"]["rows_per_min"], 4),
        },
        "paired_cpu_s": {
            "rows": {"A_clean": len(a_clean), "B2_training": len(b_train), "A2_clean": len(a2)},
            "B2_training_vs_A_clean_plus_A2": paired(a_clean + a2, b_train),
            "B2_training_vs_A2": paired(a2, b_train),
            "A2_vs_A_clean (no-trainer null)": paired(a_clean, a2),
        },
        "search_mismatches_total": mism,
    }
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
