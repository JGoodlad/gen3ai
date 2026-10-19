"""Read the GPU-window battery: ONE table (every cell's status, wall, peak memory and headline number) and the T62
gate decision, computed deterministically from the cells (README.md "Decision rules").

    python3 read_battery.py <out-dir> [--json]

THE T62 GATE (adopt the float64 tau solve in X35's pins iff it costs <= 3 % of a training cycle):
* learner: the two ABBA pairs' deltas of the 10-epoch update estimate, d_a = new_a - old_a and d_b = new_b - old_b,
  each as a FRACTION of the mean old estimate, scaled onto X26's live update time (LIVE_CYCLE["train_s"]);
* rollout: per host step, the trainee's bucket-256 forward plus OPP_SLOTS opponent forwards at bucket 64, each a
  per-block delta (new - old) from `t62_dec`, times X26's host steps per cycle. Two bounds: SERIAL (every opponent
  forward's delta adds: the lanes hide nothing) and TRAINEE-ONLY (the opponent lanes hide all of theirs);
* cost_hi = max learner delta + SERIAL rollout at the max per-block deltas; cost_lo = min learner delta +
  TRAINEE-ONLY rollout at the min per-block delta; both over X26's live cycle (train + collect + other).
* ADOPT iff cost_hi <= 3 %; FP32_ALT (the complementary-constraint alternative, design_x5_belief_tokens.md) iff
  cost_lo > 3 %; else INCONCLUSIVE -- the band straddles the bar (standing rule 8: never decided by chance; the
  orchestrator decides, e.g. with a collect-phase A/B). Any of the five T62 cells missing or not ok -> INCOMPLETE.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

GATE_PCT = 3.0
#: X26's opponent fan-out: 20 active pool snapshots (gpu_bench_suite FULL["pool_snapshots"]).
OPP_SLOTS = 20
#: X26's live cycle (gpu_bench_suite.LIVE_CYCLE, 2026-10-18 tb_read at 258.6M): copied so the reader runs anywhere.
LIVE_CYCLE = {"train_s": 42.24, "collect_s": 26.97, "other_s": 5.6, "host_steps": 416.5}
T62_UPD = ("t62_u_old_a", "t62_u_new_a", "t62_u_new_b", "t62_u_old_b")
X45 = tuple(f"x45_p{p}_b{b}" for p in (20, 5) for b in (3, 5))
ORDER = ("t62_dec", "frozen_guard") + T62_UPD + ("x45_cuda_test",) + X45 + ("x14_ab",)


def _load(p: Path) -> dict[str, Any] | None:
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return None


def t62_decision(cells: Path) -> dict[str, Any]:
    dec = _load(cells / "t62_dec.json")
    upd = {n: _load(cells / f"{n}.json") for n in T62_UPD}
    missing = [n for n, d in [("t62_dec", dec)] + list(upd.items()) if not d or d.get("status") != "ok"]
    if missing:
        return {"verdict": "INCOMPLETE", "missing": missing}
    est = {n: float(d["train"]["update_s_est"]) for n, d in upd.items()}  # type: ignore[index]
    base = statistics.mean([est["t62_u_old_a"], est["t62_u_old_b"]])
    d_a, d_b = est["t62_u_new_a"] - est["t62_u_old_a"], est["t62_u_new_b"] - est["t62_u_old_b"]
    f_lo, f_hi = min(d_a, d_b) / base, max(d_a, d_b) / base
    tr_lo, tr_hi = f_lo * LIVE_CYCLE["train_s"], f_hi * LIVE_CYCLE["train_s"]
    s = dec["summary"]  # type: ignore[index]
    d256, d64 = s["256"]["delta_ms"], s["64"]["delta_ms"]
    steps = LIVE_CYCLE["host_steps"]
    ro_hi = steps * (d256["max"] + OPP_SLOTS * d64["max"]) / 1e3
    ro_lo = steps * d256["min"] / 1e3
    cycle = LIVE_CYCLE["train_s"] + LIVE_CYCLE["collect_s"] + LIVE_CYCLE["other_s"]
    hi, lo = 100 * (tr_hi + ro_hi) / cycle, 100 * (tr_lo + ro_lo) / cycle
    verdict = "ADOPT" if hi <= GATE_PCT else "FP32_ALT" if lo > GATE_PCT else "INCONCLUSIVE"
    gates = {n: (d.get("gate") or {}).get("verdict") for n, d in upd.items()}  # type: ignore[union-attr]
    return {"verdict": verdict, "cost_pct_lo": lo, "cost_pct_hi": hi, "gate_pct": GATE_PCT,
            "learner": {"update_s_est": est, "delta_s": [d_a, d_b], "frac_lo": f_lo, "frac_hi": f_hi,
                        "live_s_lo": tr_lo, "live_s_hi": tr_hi, "compile_gate": gates},
            "rollout": {"delta_ms_b256": d256, "delta_ms_b64": d64, "s_per_cycle_lo": ro_lo, "s_per_cycle_hi": ro_hi,
                        "agreement": dec.get("agreement")},  # type: ignore[union-attr]
            "cycle_s": cycle}


def _headline(name: str, d: dict[str, Any] | None, cells: Path) -> str:
    if d is None:
        return ""
    if name == "t62_dec" and "summary" in d:
        return "; ".join(f"B{b}: {v['old_ms']:.2f} -> {v['new_ms']:.2f} ms (d {v['delta_ms']['median']:+.2f})"
                         for b, v in d["summary"].items())
    if name.startswith("t62_u_") and "train" in d:
        return f"update_s_est {d['train']['update_s_est']:.2f} s; gate {(d.get('gate') or {}).get('verdict')}"
    if name.startswith("x45_"):
        x = x45_row(d)
        return (f"per-slot {x['off_ms']} ms/step, grouped {x['on_ms']} ms/step, ratio {x['ratio']}"
                if x else "see the cell json")
    return ""


def x45_row(d: dict[str, Any]) -> dict[str, Any] | None:
    """One X45 cell: per-slot (OFF) vs grouped (ON) median ms/step, ON/OFF ratio, peak allocated MiB."""
    try:
        arm = d["per_arm_ms"]
        return {"off_ms": arm["off"]["step_median"], "on_ms": arm["on"]["step_median"],
                "ratio": d["on_over_off_ratio_median"], "peak_mib": d.get("max_memory_allocated_mib")}
    except (KeyError, TypeError):
        return None


def table(out: Path) -> list[dict[str, Any]]:
    cells = out / "cells"
    rows = []
    for n in ORDER:
        st = _load(cells / f"{n}.state.json") or {}
        res = _load(cells / f"{n}.json")
        peak = (res or {}).get("peak_reserved_mib", (res or {}).get("max_memory_allocated_mib"))
        rows.append({"cell": n, "status": st.get("status", "not run"), "wall_min": st.get("wall_min"),
                     "peak_gpu_mib": round(peak) if isinstance(peak, (int, float)) else None,
                     "headline": _headline(n, res, cells) or st.get("reason") or st.get("detail") or ""})
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("out")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    out = Path(a.out)
    rows, dec = table(out), t62_decision(out / "cells")
    if a.json:
        print(json.dumps({"cells": rows, "t62": dec}, indent=1))
        return 0
    status = (_load(out / "battery_status.json") or {}).get("status", "?")
    print(f"GPU-window battery {out}  (battery status: {status})\n")
    print(f"{'cell':14s} {'status':8s} {'wall min':>8s} {'GPU MiB':>8s}  headline")
    for r in rows:
        w = "" if r["wall_min"] is None else f"{r['wall_min']:.1f}"
        g = "" if r["peak_gpu_mib"] is None else str(r["peak_gpu_mib"])
        print(f"{r['cell']:14s} {r['status']:8s} {w:>8s} {g:>8s}  {r['headline']}")
    print("\nX45 grouped vs per-slot opponent forward (ms per host step; ratio = grouped / per-slot, <1 is a win):")
    print(f"{'cell':14s} {'per-slot':>9s} {'grouped':>9s} {'ratio':>7s} {'peak MiB':>9s}")
    for n in X45:
        x = x45_row(_load(out / "cells" / f"{n}.json") or {})
        print(f"{n:14s} " + (f"{x['off_ms']:9.3f} {x['on_ms']:9.3f} {x['ratio']:7.3f} {x['peak_mib'] or '':>9}" if x else "not run / no result"))
    print()
    if dec["verdict"] == "INCOMPLETE":
        print(f"T62 GATE: INCOMPLETE -- cells not ok: {', '.join(dec['missing'])}")
    else:
        print(f"T62 GATE: {dec['verdict']} -- per-cycle cost {dec['cost_pct_lo']:.2f} % .. {dec['cost_pct_hi']:.2f} % "
              f"(bar {GATE_PCT} %; learner {dec['learner']['live_s_lo']:+.2f} .. {dec['learner']['live_s_hi']:+.2f} s, "
              f"rollout {dec['rollout']['s_per_cycle_lo']:+.2f} .. {dec['rollout']['s_per_cycle_hi']:+.2f} s of a "
              f"{dec['cycle_s']:.1f} s cycle)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
