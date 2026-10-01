"""K6 memory calibration — read every unit's rows, describe the healthy learner, replay the detector,
estimate the false-trip rate, and write the compact committed copies.

    export PYTHONPATH=$PYTHONPATH:src
    python3 designs/research_state/measurements/k6_k8/memory/analyze.py [--root ~/gen3ai_archive/k6_k8_memory]

Writes `result.json` (everything below) and `healthy_trace.json` (the per-sample rows of the GPU
units, the routine test's fixture) beside this file.
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path
from typing import Any, Dict, List

from agents.training import cuda_memory_trend as M

HERE = Path(__file__).resolve().parent
MiB = float(M.MiB)
FIELDS = list(M.MemorySample.__dataclass_fields__)


def _rows(p: Path) -> List[Dict[str, Any]]:
    out = []
    for line in p.read_text().splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            pass                         # a unit killed mid-write leaves one torn line
    return out


def _q(xs: List[float], q: float) -> float:
    xs = sorted(xs)
    if not xs:
        return float("nan")
    k = (len(xs) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def _desc(xs: List[float], scale: float = 1.0) -> Dict[str, Any]:
    if not xs:
        return {"n": 0}
    ys = [x / scale for x in xs]
    return {"n": len(ys), "min": min(ys), "p50": _q(ys, 0.5), "max": max(ys),
            "spread": max(ys) - min(ys), "sd": statistics.pstdev(ys) if len(ys) > 1 else 0.0}


def describe_unit(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    head = next((r for r in rows if r.get("kind") == "header"), {})
    foot = next((r for r in rows if r.get("kind") == "footer"), None)
    ss = [r for r in rows if r.get("kind") == "sample"]
    startup = next((r for r in ss if r["phase"] == "post_startup"), None)
    steady = [r for r in ss if r["phase"] in ("post_update", "post_rollout") and r["update"] >= M.WARMUP_UPDATES]
    ups = [r for r in ss if r["phase"] == "post_update"]
    st_ups = [r for r in ups if r["update"] >= M.WARMUP_UPDATES]
    out: Dict[str, Any] = {
        "torch": head.get("torch"), "alloc_conf": head.get("alloc_conf"),
        "n_epochs": head.get("cfg", {}).get("n_epochs"), "load_at_start": head.get("load1"),
        "load_at_end": (foot or {}).get("load1"), "complete": foot is not None,
        "updates": max((r["update"] for r in ups), default=0),
        "gate_s": (startup or {}).get("gate_s"), "prewarm_s": (startup or {}).get("prewarm_s"),
        "device_total_mib": (startup or {}).get("device_total", 0) / MiB,
    }
    if startup:
        out["at_startup_mib"] = {k: startup[k] / MiB for k in ("allocated", "reserved")}
        out["at_startup_segments"] = startup["segments"]
    if ups:
        out["first_update_mib"] = {k: ups[0][k] / MiB for k in ("allocated", "reserved", "peak_reserved")}
    for phase in ("post_update", "post_rollout"):
        xs = [r for r in steady if r["phase"] == phase]
        out[phase] = {k: _desc([r[k] for r in xs], MiB) for k in
                      ("allocated", "reserved", "active", "inactive_split", "peak_allocated", "peak_reserved")}
        out[phase]["segments"] = _desc([r["segments"] for r in xs])
    out["segments_after_warmup"] = (max((r["segments"] for r in steady), default=0)
                                    - min((r["segments"] for r in steady), default=0))
    out["segments_allocated_total_after_warmup"] = (max((r["segments_allocated_total"] for r in steady), default=0)
                                                    - min((r["segments_allocated_total"] for r in steady), default=0))
    out["alloc_retries"] = max((r["alloc_retries"] for r in ss), default=0)
    out["ooms"] = max((r["ooms"] for r in ss), default=0)
    out["reserved_steps_after_warmup"] = [[b["update"], (b["reserved"] - a["reserved"]) / MiB]
                                          for a, b in zip(steady, steady[1:]) if a["reserved"] != b["reserved"]]
    out["reserved_steps_before_warmup"] = [[b["update"], (b["reserved"] - a["reserved"]) / MiB]
                                           for a, b in zip(ss, ss[1:]) if a["reserved"] != b["reserved"]
                                           and b["update"] < M.WARMUP_UPDATES]
    out["floor_range_mib"] = {ph: ((max(r["allocated"] for r in xs) - min(r["allocated"] for r in xs)) / MiB
                                   if xs else None)
                              for ph, xs in (("post_update", [r for r in steady if r["phase"] == "post_update"]),
                                             ("post_rollout", [r for r in steady if r["phase"] == "post_rollout"]))}
    out["diag_updates"] = [r["update"] for r in st_ups if r.get("diag")]
    diag = [r["train_s"] for r in st_ups if r.get("diag")]
    nodiag = [r["train_s"] for r in st_ups if not r.get("diag")]
    out["train_s"] = {"diag": _desc(diag), "no_diag": _desc(nodiag)}
    out["rollout_s"] = _desc([r["rollout_s"] for r in ss if r["phase"] == "post_rollout" and r["update"] >= M.WARMUP_UPDATES])
    out["behaviour_max"] = _desc([r["behaviour_max"] for r in st_ups if r.get("behaviour_max") is not None])
    # the detector, replayed
    t = M.MemoryTrend()
    levels: Dict[str, int] = {}
    sustained = 0
    for r in ss:
        if r["phase"] not in ("post_update", "post_rollout"):
            continue
        v = t.observe(M.MemorySample(**{k: r[k] for k in FIELDS}))
        if v.window_closed:
            levels[v.level] = levels.get(v.level, 0) + 1
            sustained += int(v.sustained)
    floors = [w.floor for w in t.windows]
    out["detector"] = {"window_levels": levels, "sustained_closes": sustained,
                       "floors_mib": [f / MiB for f in floors],
                       "floor_deltas_mib": [(b - a) / MiB for a, b in zip(floors, floors[1:])],
                       "segments_after_freeze": t.segments_after_freeze()}
    return out


def residuals(rows: List[Dict[str, Any]]) -> List[float]:
    """Steady window floors minus their unit median (bytes)."""
    t = M.MemoryTrend()
    for r in rows:
        if r.get("kind") == "sample" and r["phase"] in ("post_update", "post_rollout"):
            t.observe(M.MemorySample(**{k: r[k] for k in FIELDS}))
    fl = [w.floor for w in t.windows]
    if not fl:
        return []
    med = statistics.median(fl)
    return [f - med for f in fl]


def false_trip(res: List[float], *, n_windows: int, trials: int, magnify: float, seed: int = 0,
               headroom_bytes: float = 1.0 * M.GiB) -> Dict[str, Any]:
    """Synthetic healthy processes: window floors = a constant + residuals resampled i.i.d. from the
    measured pool x ``magnify``, on a card with only ``headroom_bytes`` between demand and ceiling (a
    tight card, so any sustained read would project inside the horizon). Counts processes that ever
    read SUSTAINED, and ever STOP."""
    rng = random.Random(seed)
    any_sus = any_stop = 0
    base = 3 * M.GiB
    for _ in range(trials):
        t = M.MemoryTrend()
        hit_s = hit_stop = False
        for w in range(n_windows):
            fl = base + magnify * rng.choice(res)
            for k in range(M.WINDOW_UPDATES):
                u = M.WARMUP_UPDATES + w * M.WINDOW_UPDATES + k
                res_b = int(base + 2 * M.GiB)
                s = M.MemorySample(update=u, phase="post_update", allocated=int(fl), reserved=res_b,
                                   active=int(fl), inactive_split=0, segments=40, segments_allocated_total=40,
                                   peak_allocated=res_b, peak_reserved=res_b, alloc_retries=0, ooms=0,
                                   device_free=int(headroom_bytes + M.CEILING_MARGIN_BYTES),
                                   device_total=12 * M.GiB)
                v = t.observe(s)
                hit_s |= v.sustained
                hit_stop |= v.level == "STOP"
        any_sus += hit_s
        any_stop += hit_stop
    return {"trials": trials, "n_windows": n_windows, "magnify": magnify,
            "p_process_sustained": any_sus / trials, "p_process_stop": any_stop / trials}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(Path.home() / "gen3ai_archive/k6_k8_memory"))
    ap.add_argument("--trials", type=int, default=2000)
    a = ap.parse_args()
    root = Path(a.root).expanduser()
    units = {}
    raw = {}
    for d in sorted((root / "units").glob("*")):
        p = d / "samples.jsonl"
        if not p.exists() or "cpusmoke" in d.name or "t2probe" in d.name:
            continue
        rows = _rows(p)
        if not any(r.get("kind") == "sample" and r["phase"] == "post_update" for r in rows):
            continue
        raw[d.name] = rows
        units[d.name] = describe_unit(rows)
    probes = {}
    for d in sorted((root / "units").glob("*t2probe*")):
        p = d / "samples.jsonl"
        if p.exists():
            probes[d.name] = [{k: v for k, v in r.items() if k != "traceback"} for r in _rows(p)]
    pool: Dict[str, List[float]] = {}
    for name, rows in raw.items():
        arm = "expandable" if units[name]["alloc_conf"] else "default"
        pool.setdefault(arm, []).extend(residuals(rows))
    ft: Dict[str, Any] = {"per_arm_residual_mib": {arm: _desc(res, MiB) for arm, res in pool.items() if res}}
    allres = [x for res in pool.values() for x in res]
    if allres:
        rng_mib = (max(allres) - min(allres)) / MiB
        ft["pooled_residual_range_mib"] = rng_mib
        ft["noise_bar_over_range"] = (M.NOISE_BAR_BYTES / MiB) / rng_mib if rng_mib else None
        # the measured wobble, magnified until the rule starts to read it: where a false trip begins
        ft["sweep"] = [false_trip(allres, n_windows=60, trials=a.trials, magnify=m)
                       for m in (1.0, 4.0, 8.0, 16.0, 32.0, 64.0)]
    result = {"schema": "k6_memory_calibration_v1", "constants": {
        k: getattr(M, k) for k in ("WINDOW_UPDATES", "FIT_WINDOWS", "SEGMENTS", "WARMUP_UPDATES",
                                   "NOISE_BAR_BYTES", "MIN_SLOPE_BYTES_PER_UPDATE", "STEP_WARN_BYTES", "RESERVED_STEP_WARN_BYTES",
                                   "HORIZON_UPDATES", "CEILING_MARGIN_BYTES", "PERSIST_WINDOWS")},
        "units": units, "false_trip": ft, "t2probe": probes}
    (HERE / "result.json").write_text(json.dumps(result, indent=1, default=str))
    fixture = {"schema": "k6_memory_healthy_trace_v1",
               "note": "post_update / post_rollout samples of every GPU unit, as MemorySample rows",
               "units": [{"unit": n, "alloc_conf": units[n]["alloc_conf"], "torch": units[n]["torch"],
                          "samples": [{k: r[k] for k in FIELDS} for r in rows
                                      if r.get("kind") == "sample" and r["phase"] in ("post_update", "post_rollout")]}
                         for n, rows in raw.items()]}
    (HERE / "healthy_trace.json").write_text(json.dumps(fixture, separators=(",", ":")))
    for n, u in units.items():
        pu = u["post_update"]
        print(f"{n}: torch {u['torch']} conf {u['alloc_conf']} upd {u['updates']} load {u['load_at_start']:.1f}"
              f" | alloc floor {pu['allocated'].get('min', 0):.1f}..{pu['allocated'].get('max', 0):.1f} MiB"
              f" reserved {pu['reserved'].get('min', 0):.0f}..{pu['reserved'].get('max', 0):.0f}"
              f" peak_res {pu['peak_reserved'].get('max', 0):.0f} seg {pu['segments'].get('min')}..{pu['segments'].get('max')}"
              f" | train_s nodiag {u['train_s']['no_diag'].get('p50', float('nan')):.2f}"
              f" diag {u['train_s']['diag'].get('p50', float('nan')):.2f} | det {u['detector']['window_levels']}")
    print(json.dumps(ft, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
