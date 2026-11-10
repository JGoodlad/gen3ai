"""The figures of README.md that the tool's two summaries do not hold. A frozen record: run it from the repo root.

    PYTHONPATH=src python3 designs/research_state/measurements/core_split_trainer_cpu_2026-11-09/analyze.py \
        [--log <run>/launcher_child.full.log] [--slot-events <heavy_slots>/events.jsonl ...]

With no option it reads only the files beside it (the agent cores' occupancy, the phase sequence, and the CPU time of
every finished Kakuna unit scope from ``kakuna_unit_scopes.tsv.gz``: systemd's "Consumed ... CPU time" journal line of
each ``gen3ai-capped-x35kakuna-*`` scope, 2026-11-04 to 11-09 18:10). ``--log`` adds the run's per-iteration
``collect_ms`` / ``update_wall_s`` spread; ``--slot-events`` the anchor units' slot wait and run time. Both were read on
2026-11-09 and the output of all of it is ``supplement.json``.
"""
from __future__ import annotations

import argparse
import collections
import gzip
import json
import re
import statistics as st
import time
from pathlib import Path
from typing import Any, Dict, List

from main.ops import trainer_cpu_profile as T

HERE = Path(__file__).resolve().parent
AGENT = (5, 6, 7, 13, 14, 15)


def rows_of(name: str) -> List[Any]:
    with gzip.open(HERE / name, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def q(xs: List[float], p: float) -> float:
    return round(T.percentile(xs, p), 3)


def occupancy() -> Dict[str, Any]:
    data = [r for r in rows_of("profile_rows.jsonl.gz") if "groups" in r]
    a = [sum(r["cpu"][str(c)] for c in AGENT) for r in data]
    seq: List[List[Any]] = []
    for r in data:
        p = T.phase_of(r)[0]
        if seq and seq[-1][0] == p:
            seq[-1][1] += 1
        else:
            seq.append([p, 1])
    return {"agent_cpus": list(AGENT), "intervals": len(a), "busy_mean": round(st.mean(a), 2), "busy_p5": q(a, 5),
            "busy_p50": q(a, 50), "busy_p95": q(a, 95), "share_of_intervals_at_or_over_5.7": round(sum(x >= 5.7 for x in a) / len(a), 3),
            "phase_sequence": " ".join(f"{p}{n}" for p, n in seq)}


def iterations(log: Path) -> Dict[str, Any]:
    c: List[float] = []
    u: List[float] = []
    for line in log.read_text(errors="replace").splitlines():
        m = re.match(r"\|\s+(?:\w+/)?(collect_ms|update_wall_s)\s+\|\s+([-\d.e+]+)\s+\|", line)
        if m:
            (c if m.group(1) == "collect_ms" else u).append(float(m.group(2)))
    med = st.median(c)
    plain = [x for x in c if x < 1.25 * med]          # an eval iteration's collect is ~2.4x the median
    return {"log": str(log), "iterations": len(c), "collect_ms_median": med, "eval_iterations": len(c) - len(plain),
            "eval_collect_ms_mean": round(st.mean(x for x in c if x >= 1.25 * med)) if len(c) > len(plain) else None,
            "plain_collect_ms_mean": round(st.mean(plain)), "plain_collect_ms_sd": round(st.pstdev(plain)),
            "plain_collect_cv_pct": round(100 * st.pstdev(plain) / st.mean(plain), 2),
            "update_wall_s_mean": round(st.mean(u), 2), "update_wall_cv_pct": round(100 * st.pstdev(u) / st.mean(u), 2)}


def units(paths: List[Path]) -> Dict[str, Any]:
    rows = []
    for p in paths:
        for line in p.read_text(errors="replace").splitlines():
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if r.get("kind") == "slot" and "held_s" in r and "main.anchors" in r.get("cmd", ""):
                rows.append(r)
    by: Dict[int, List[Dict[str, Any]]] = collections.defaultdict(list)
    for r in rows:
        by[int(r["ts"] // 21600)].append(r)
    out = []
    for k in sorted(by):
        v = by[k]
        out.append({"six_hours_from": time.strftime("%Y-%m-%d %H:%M", time.localtime(k * 21600)), "units": len(v),
                    "run_s_median": round(st.median(x["held_s"] for x in v)), "run_s_p90": round(q([x["held_s"] for x in v], 90)),
                    "slot_wait_s_median": round(st.median(x["waited_s"] for x in v)),
                    "slot_wait_s_mean": round(st.mean(x["waited_s"] for x in v))})
    return {"anchor_units": len(rows), "by_six_hours": out}


def unit_scopes() -> Dict[str, Any]:
    by: Dict[str, List[List[float]]] = collections.defaultdict(list)
    with gzip.open(HERE / "kakuna_unit_scopes.tsv.gz", "rt", encoding="utf-8") as f:
        next(f)
        for line in f:
            day, cpu, wall, mem = line.rstrip("\n").split("\t")
            by[day].append([float(cpu), float(wall), float(mem)])
    out = []
    for day in sorted(by):
        v = by[day]
        cpu, wall = [x[0] for x in v], [x[1] for x in v]
        out.append({"day": day, "units": len(v), "cpu_s_median": round(st.median(cpu)), "cpu_s_mean": round(st.mean(cpu)),
                    "cpu_s_p90": round(q(cpu, 90)), "wall_s_median": round(st.median(wall)),
                    "cpu_per_wall": round(sum(cpu) / sum(wall), 2), "cpu_hours": round(sum(cpu) / 3600, 1),
                    "mem_peak_gib_max": max(x[2] for x in v)})
    return {"scope": "gen3ai-capped-x35kakuna-*", "by_day": out}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", type=Path)
    ap.add_argument("--slot-events", type=Path, nargs="*", default=[])
    a = ap.parse_args()
    out: Dict[str, Any] = {"agent_cores": occupancy(), "kakuna_unit_scopes": unit_scopes()}
    if a.log:
        out["iterations"] = iterations(a.log)
    if a.slot_events:
        out["kakuna_units"] = units(a.slot_events)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
