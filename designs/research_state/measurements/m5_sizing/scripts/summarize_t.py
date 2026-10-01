#!/usr/bin/env python3
"""Part T's table from results/ (REGISTRATION §2.2-2.3): per N the buckets, serial and overlapped
decisions/s with CIs, overlap/serial, the step's components, the opponent share (real-flush split), rows
per slot, and — given U_E10 (s) and the game-length moments — E2E(N), predicted staleness and N*.

    summarize_t.py [--u-e10 SECONDS] [--el MEAN_LEN --el2 MEAN_SQ_LEN] [--json OUT]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

RES = Path(__file__).resolve().parents[1] / "results"
D = 98_304
NS = (48, 128, 256, 512, 1024, 2048)


def row(n: int) -> dict | None:
    t, s, b = RES / f"throughput_n{n}.json", RES / f"split_n{n}.json", RES / f"buckets_n{n}.json"
    if not t.exists():
        return None
    td = json.loads(t.read_text())
    out: dict = {"n": n, "buckets": json.loads(b.read_text())["buckets"] if b.exists() else None,
                 "warnings": td.get("warnings")}
    for arm, key in (("rust_serial_keyed", "serial"), ("rust_overlap_keyed", "overlap")):
        a = td["arms"].get(arm)
        if a:
            ci = a["decisions_per_s_ci95"]
            out[key] = {"dps": a["decisions_per_s_mean"], "lo": ci["lo"], "hi": ci["hi"],
                        "ms_step": a["ms_per_vec_step_mean"], "gpu_util": a.get("gpu_util_mean"),
                        "cpu_us_dec": a.get("cpu_us_per_decision_mean"),
                        "components": a["components_ms_per_step_mean"]}
    r = td.get("ratios", {}).get("decisions_per_s_rust_overlap_keyed_over_rust_serial_keyed")
    if r:
        out["overlap_over_serial"] = {"point": r["point"], "lo": r["lo"], "hi": r["hi"]}
    if s.exists():
        sd = json.loads(s.read_text())
        res = next(iter(sd["per_lanes"].values()))
        rs = res.get("real_split") or {}
        out["split"] = {k: rs.get(k) for k in ("full_ms", "trainee_only_ms", "opponent_ms",
                                               "opponent_share_of_flush", "opponent_slots_mean",
                                               "opponent_rows_mean", "trainee_rows_mean")}
        if "serial" in out and rs.get("opponent_ms"):
            out["opponent_share_of_step"] = rs["opponent_ms"]["mean"] / out["serial"]["ms_step"]
            out["opponent_share_of_step_ci"] = [rs["opponent_ms"]["lo"] / out["serial"]["ms_step"],
                                                rs["opponent_ms"]["hi"] / out["serial"]["ms_step"]]
        out["slots_per_step"] = res.get("slots_per_step")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--u-e10", type=float, default=None, help="median update wall at E10 (s), from arm A")
    ap.add_argument("--el", type=float, default=None, help="E[L], trainee decisions per game")
    ap.add_argument("--el2", type=float, default=None, help="E[L^2]")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    rows = [r for r in (row(n) for n in NS) if r]
    for r in rows:
        r["r"] = max(r.get("serial", {}).get("dps", 0.0), r.get("overlap", {}).get("dps", 0.0))
        if a.u_e10:
            r["e2e"] = D / (D / r["r"] + a.u_e10)
        if a.el and a.el2:
            r["staleness_pred"] = r["n"] * a.el2 / (2 * a.el * D)
    if a.u_e10:
        best = max(r["e2e"] for r in rows)
        ok = [r for r in rows if r["e2e"] >= 0.95 * best and r.get("staleness_pred", 0.0) <= 0.25]
        nstar = min(r["n"] for r in ok) if ok else None
        print(f"E2E max {best:.0f}/s; N* = {nstar}")
    hdr = f"{'N':>5} {'buckets':>18} {'serial d/s [CI]':>24} {'overlap d/s':>12} {'ov/ser [CI]':>20} {'ms/step':>8} {'core':>6} {'gpu_wait':>8} {'opp%step':>9} {'slots':>6}"
    print(hdr)
    for r in rows:
        s, o = r.get("serial", {}), r.get("overlap", {})
        ov = r.get("overlap_over_serial", {})
        print(f"{r['n']:>5} {str(r['buckets']):>18} {s.get('dps', 0):>8.0f} [{s.get('lo') or 0:>6.0f},{s.get('hi') or 0:>6.0f}] "
              f"{o.get('dps', 0):>12.0f} {ov.get('point', 0):>6.3f} [{ov.get('lo', 0):.3f},{ov.get('hi', 0):.3f}] "
              f"{s.get('ms_step', 0):>8.2f} {s.get('components', {}).get('core', 0):>6.2f} "
              f"{s.get('components', {}).get('gpu_wait', 0):>8.2f} {100 * r.get('opponent_share_of_step', 0):>8.1f}% "
              f"{(r.get('split') or {}).get('opponent_slots_mean') or 0:>6.1f}"
              + (f"  e2e {r['e2e']:.0f}/s" if 'e2e' in r else "")
              + (f"  stale {100 * r['staleness_pred']:.1f}%" if 'staleness_pred' in r else ""))
    if a.json:
        Path(a.json).write_text(json.dumps(rows, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
