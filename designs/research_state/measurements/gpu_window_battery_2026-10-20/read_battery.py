"""Read the FP32_ALT GPU-window battery: ONE table (every cell's status, wall, peak memory and headline number) and
the FP32_ALT gate decision, computed deterministically from the cells (README.md "The FP32_ALT gate").

    python3 read_battery.py <out-dir> [--json]

THE FP32_ALT GATE: the 2026-10-19 battery's T62 cost computation (`../gpu_window_battery_2026-10-19/read_battery.py`)
with ``fp32_naive`` as old and ``fp32_alt`` as new.
* learner: the two ABBA pairs' deltas of the 10-epoch update estimate, d_a = alt_a - naive_a and d_b = alt_b - naive_b,
  each as a FRACTION of the mean naive estimate, scaled onto X26's live update time (LIVE_CYCLE["train_s"]);
* rollout: per host step, the trainee's bucket-256 forward plus OPP_SLOTS opponent forwards at bucket 64, each a
  per-block delta (alt - naive) from `tau_dec`, times X26's host steps per cycle. Two bounds: SERIAL (every opponent
  forward's delta adds) and TRAINEE-ONLY (the opponent lanes hide all of theirs);
* cost_hi = max learner delta + SERIAL rollout at the max per-block deltas; cost_lo = min learner delta +
  TRAINEE-ONLY rollout at the min per-block delta; both over X26's live cycle (train + collect + other).
* CONFIRMED iff cost_hi <= 3 %; OVER iff cost_lo > 3 %; else INCONCLUSIVE (the band straddles the bar: standing rule 8,
  never decided by chance -- the orchestrator decides). Any of the five cells missing, not ok, or without a clean
  ``tau_force`` record for its arm -> INCOMPLETE.
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
#: X26's live cycle (gpu_bench_suite.LIVE_CYCLE, 2026-10-18 tb_read at 258.6M): copied so the reader runs anywhere,
#: and the SAME numbers the 2026-10-19 T62 gate used, so the two reads compare.
LIVE_CYCLE = {"train_s": 42.24, "collect_s": 26.97, "other_s": 5.6, "host_steps": 416.5}
UPD_ARM = {"tau_u_naive_a": "fp32_naive", "tau_u_alt_a": "fp32_alt", "tau_u_alt_b": "fp32_alt",
           "tau_u_naive_b": "fp32_naive"}
DEC_ARMS = ("fp32_naive", "fp32_alt")
ORDER = ("tau_dec",) + tuple(UPD_ARM)


def _load(p: Path) -> dict[str, Any] | None:
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return None


def _forcing_bad(d: dict[str, Any], arms: tuple[str, ...]) -> str | None:
    """Why a cell's ``tau_force`` record does not prove its arm(s) ran (None = it does)."""
    tf = d.get("tau_force")
    if not isinstance(tf, dict):
        return "no tau_force record"
    if tf.get("violations"):
        return "; ".join(tf["violations"])
    if tuple(tf.get("arms") or ()) != arms or not tf.get("intact"):
        return f"tau_force arms {tf.get('arms')} intact {tf.get('intact')}, want {list(arms)} intact"
    calls = {a: (tf.get("eager_calls") or {}).get(a, 0) + (tf.get("trace_calls") or {}).get(a, 0) for a in arms}
    if any(n <= 0 for n in calls.values()):
        return f"tau_force calls {calls}: an arm never reached the wrapper"
    return None


def fp32_alt_decision(cells: Path) -> dict[str, Any]:
    dec = _load(cells / "tau_dec.json")
    upd = {n: _load(cells / f"{n}.json") for n in UPD_ARM}
    missing = []
    for n, d, arms in [("tau_dec", dec, DEC_ARMS)] + [(n, d, (UPD_ARM[n],)) for n, d in upd.items()]:
        if not d or d.get("status") != "ok":
            missing.append(n)
        else:
            why = _forcing_bad(d, arms)
            if why:
                missing.append(f"{n} ({why})")
    if missing:
        return {"verdict": "INCOMPLETE", "missing": missing}
    est = {n: float(d["train"]["update_s_est"]) for n, d in upd.items()}  # type: ignore[index]
    base = statistics.mean([est["tau_u_naive_a"], est["tau_u_naive_b"]])
    d_a, d_b = est["tau_u_alt_a"] - est["tau_u_naive_a"], est["tau_u_alt_b"] - est["tau_u_naive_b"]
    f_lo, f_hi = min(d_a, d_b) / base, max(d_a, d_b) / base
    tr_lo, tr_hi = f_lo * LIVE_CYCLE["train_s"], f_hi * LIVE_CYCLE["train_s"]
    s = dec["summary"]  # type: ignore[index]
    d256, d64 = s["256"]["delta_ms"], s["64"]["delta_ms"]
    steps = LIVE_CYCLE["host_steps"]
    ro_hi = steps * (d256["max"] + OPP_SLOTS * d64["max"]) / 1e3
    ro_lo = steps * d256["min"] / 1e3
    cycle = LIVE_CYCLE["train_s"] + LIVE_CYCLE["collect_s"] + LIVE_CYCLE["other_s"]
    hi, lo = 100 * (tr_hi + ro_hi) / cycle, 100 * (tr_lo + ro_lo) / cycle
    verdict = "CONFIRMED" if hi <= GATE_PCT else "OVER" if lo > GATE_PCT else "INCONCLUSIVE"
    gates = {n: (d.get("gate") or {}).get("verdict") for n, d in upd.items()}  # type: ignore[union-attr]
    return {"verdict": verdict, "cost_pct_lo": lo, "cost_pct_hi": hi, "gate_pct": GATE_PCT,
            "learner": {"update_s_est": est, "delta_s": [d_a, d_b], "frac_lo": f_lo, "frac_hi": f_hi,
                        "live_s_lo": tr_lo, "live_s_hi": tr_hi, "compile_gate": gates},
            "rollout": {"delta_ms_b256": d256, "delta_ms_b64": d64, "s_per_cycle_lo": ro_lo, "s_per_cycle_hi": ro_hi,
                        "agreement": dec.get("agreement")},  # type: ignore[union-attr]
            "cycle_s": cycle}


def _headline(name: str, d: dict[str, Any] | None) -> str:
    if d is None:
        return ""
    if name == "tau_dec" and "summary" in d:
        return "; ".join(f"B{b}: naive {v['old_ms']:.2f} -> alt {v['new_ms']:.2f} ms (d {v['delta_ms']['median']:+.2f})"
                         for b, v in d["summary"].items())
    if name.startswith("tau_u_") and "train" in d:
        return f"update_s_est {d['train']['update_s_est']:.2f} s; gate {(d.get('gate') or {}).get('verdict')}"
    return ""


def table(out: Path) -> list[dict[str, Any]]:
    cells = out / "cells"
    rows = []
    for n in ORDER:
        st = _load(cells / f"{n}.state.json") or {}
        res = _load(cells / f"{n}.json")
        peak = (res or {}).get("peak_reserved_mib")
        rows.append({"cell": n, "status": st.get("status", "not run"), "wall_min": st.get("wall_min"),
                     "peak_gpu_mib": round(peak) if isinstance(peak, (int, float)) else None,
                     "headline": _headline(n, res) or st.get("reason") or st.get("detail") or ""})
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("out")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    out = Path(a.out)
    rows, dec = table(out), fp32_alt_decision(out / "cells")
    if a.json:
        print(json.dumps({"cells": rows, "fp32_alt": dec}, indent=1))
        return 0
    status = (_load(out / "battery_status.json") or {}).get("status", "?")
    print(f"FP32_ALT GPU-window battery {out}  (battery status: {status})\n")
    print(f"{'cell':14s} {'status':8s} {'wall min':>8s} {'GPU MiB':>8s}  headline")
    for r in rows:
        w = "" if r["wall_min"] is None else f"{r['wall_min']:.1f}"
        g = "" if r["peak_gpu_mib"] is None else str(r["peak_gpu_mib"])
        print(f"{r['cell']:14s} {r['status']:8s} {w:>8s} {g:>8s}  {r['headline']}")
    print()
    if dec["verdict"] == "INCOMPLETE":
        print(f"FP32_ALT GATE: INCOMPLETE -- cells not ok: {', '.join(dec['missing'])}")
    else:
        print(f"FP32_ALT GATE: {dec['verdict']} -- per-cycle cost {dec['cost_pct_lo']:.2f} % .. "
              f"{dec['cost_pct_hi']:.2f} % (bar {GATE_PCT} %; learner {dec['learner']['live_s_lo']:+.2f} .. "
              f"{dec['learner']['live_s_hi']:+.2f} s, rollout {dec['rollout']['s_per_cycle_lo']:+.2f} .. "
              f"{dec['rollout']['s_per_cycle_hi']:+.2f} s of a {dec['cycle_s']:.1f} s cycle)")
        print(f"  agreement alt vs naive: {json.dumps(dec['rollout']['agreement'])}")
        print(f"  compile sentinel gate per learner cell: {dec['learner']['compile_gate']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
