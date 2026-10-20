"""Read the FP32_ALT GPU-window battery: ONE table (every cell's status, wall, peak memory and headline number), the
FP32_ALT gate decision and the X14 overlap A/B read, computed deterministically from the cells (README.md "The FP32_ALT
gate", "The X14 A/B").

    python3 read_battery.py <out-dir> [--json]
    python3 read_battery.py --selftest            # CPU: the fairness rule, the gate and the X14 read on synthetic cells

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
* FAIRNESS (`learner_path.py`): the four learner cells must have run the IDENTICAL compiled path (same ``--learner-gate``
  mode, every path marker equal and violation-free: a PASS startup gate and the production sentinel's lines under
  ``on``). A pair that did not is INCOMPLETE ("arms ran different compiled paths"), never a cost.

THE X14 A/B (``x14_off`` / ``x14_on``, `x14_read.py`): the ON/OFF ratio of the median WALL fps of the cycles after the 3rd
dump of each process, the bar >= x1.10, the exposed collection (``overlap/join_wait_ms``, ``rollout/collect_ms``), the
worker's play time, T2's pi_prox pass and the K9(b) section-7 ceiling read of each arm (a FLAG, never a HOLD).
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import learner_path as LP  # noqa: E402
import x14_read as XR  # noqa: E402

GATE_PCT = 3.0
#: X26's opponent fan-out: 20 active pool snapshots (gpu_bench_suite FULL["pool_snapshots"]).
OPP_SLOTS = 20
#: X26's live cycle (gpu_bench_suite.LIVE_CYCLE, 2026-10-18 tb_read at 258.6M): copied so the reader runs anywhere,
#: and the SAME numbers the 2026-10-19 T62 gate used, so the two reads compare.
LIVE_CYCLE = {"train_s": 42.24, "collect_s": 26.97, "other_s": 5.6, "host_steps": 416.5}
UPD_ARM = {"tau_u_naive_a": "fp32_naive", "tau_u_alt_a": "fp32_alt", "tau_u_alt_b": "fp32_alt",
           "tau_u_naive_b": "fp32_naive"}
DEC_ARMS = ("fp32_naive", "fp32_alt")
X14_CELLS = ("x14_off", "x14_on")
ORDER = ("tau_dec",) + tuple(UPD_ARM) + X14_CELLS


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
    paths = {n: (d or {}).get("path") for n, d in upd.items()}
    if any(not isinstance(v, dict) or "markers" not in v for v in paths.values()):
        return {"verdict": "INCOMPLETE", "missing": [f"{n} (no `path` record: the cell predates the fairness rule)"
                                                     for n, v in paths.items() if not isinstance(v, dict)]}
    unfair = [f"{n}: {'; '.join(v['violations'])}" for n, v in paths.items() if v.get("violations")]
    unfair += LP.fair_pair([v["markers"] for v in paths.values()])    # type: ignore[index, union-attr]
    if unfair:
        return {"verdict": "INCOMPLETE", "missing": [f"the learner arms ran different compiled paths: {u}" for u in unfair]}
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
                        "live_s_lo": tr_lo, "live_s_hi": tr_hi, "compile_gate": gates,
                        "path": {"mode": next(iter(paths.values()))["mode"],     # type: ignore[index, union-attr]
                                 "markers": next(iter(paths.values()))["markers"]}},    # type: ignore[index, union-attr]
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
        return (f"update_s_est {d['train']['update_s_est']:.2f} s; gate {(d.get('gate') or {}).get('verdict')}; path "
                f"{'clean' if not (d.get('path') or {}).get('violations') and d.get('path') else 'VIOLATED / absent'}")
    if name in X14_CELLS and isinstance(d.get("summary"), dict):
        s = d["summary"]
        f = (s.get("fps_wall") or {}).get("median")
        return (f"{s['n_cycles']} kept cycles over {s['n_segments']} process(es); wall fps "
                f"{'n/a' if f is None else f'{f:.0f}'}; K9(b) {(d.get('k9b') or {}).get('verdict')}")
    return ""


def table(out: Path) -> list[dict[str, Any]]:
    cells = out / "cells"
    rows = []
    for n in ORDER:
        st = _load(cells / f"{n}.state.json") or {}
        res = _load(cells / f"{n}.json")
        peak = (res or {}).get("peak_reserved_mib")
        if peak is None and n in X14_CELLS:
            peak = (((res or {}).get("summary") or {}).get("tags") or {}).get("lifecycle/cuda_update_peak_reserved_mib")
            peak = (peak or {}).get("max")
        rows.append({"cell": n, "status": st.get("status", "not run"), "wall_min": st.get("wall_min"),
                     "peak_gpu_mib": round(peak) if isinstance(peak, (int, float)) else None,
                     "headline": _headline(n, res) or st.get("reason") or st.get("detail") or ""})
    return rows


def x14_decision(cells: Path) -> dict[str, Any]:
    """The X14 A/B verdict (`x14_read.decide`) from the two cells' stored summaries; a cell whose own post-run read
    failed is re-read here from its run dir (read-only)."""
    summ: dict[str, Any] = {}
    for arm in ("off", "on"):
        d = _load(cells / f"x14_{arm}.json")
        s = None
        if d and d.get("status") == "ok":
            s = d.get("summary") if isinstance(d.get("summary"), dict) else None
            if s is None and d.get("run_dir"):
                try:
                    s = XR.summarise(d["run_dir"], arm)
                except Exception:  # noqa: BLE001 -- INCOMPLETE below names it
                    s = None
        summ[arm] = s
    dec = XR.decide(summ["off"], summ["on"])
    dec["off"], dec["on"] = summ["off"], summ["on"]
    return dec


def _med(s: dict[str, Any] | None, tag: str) -> float | None:
    return (((s or {}).get("tags") or {}).get(tag) or {}).get("median")


def _fmt(x: float | None, unit: str = "", scale: float = 1.0, nd: int = 1) -> str:
    return "n/a" if x is None else f"{x / scale:.{nd}f}{unit}"


def _g(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.3g}"


def x14_lines(dec: dict[str, Any]) -> list[str]:
    if dec["verdict"] == "INCOMPLETE":
        return [f"X14 A/B: INCOMPLETE -- {'; '.join(dec['missing'])}"]
    off, on = dec["off"], dec["on"]
    lo, hi = dec["ratio_band"]
    lg = dec["ratio_time_fps_logged"]
    logged = f"x{lg:.3f}" if lg else "n/a"
    out = [f"X14 A/B: {dec['verdict']} -- fps(ON) / fps(OFF) = x{dec['ratio']:.3f} (25-75 % band x{lo:.2f} .. x{hi:.2f}"
           f"{', STRADDLES the bar' if dec['band_straddles_bar'] else ''}; bar x{dec['bar']:.2f} = worth a learning A/B); "
           f"wall fps OFF {off['fps_wall']['median']:.0f} -> ON {on['fps_wall']['median']:.0f} on "
           f"{dec['cycles']['off']} / {dec['cycles']['on']} kept cycles (the README's logged time/fps, cumulative per "
           f"process: {logged})"]
    hs = dec["off_max_hideable_share"]
    out.append(f"  OFF: update {_fmt(_med(off, 'train/train_ms'), ' s', 1e3)}, collect {_fmt(_med(off, 'rollout/collect_ms'), ' s', 1e3)}"
               f" (max hideable share of the cycle {_fmt(None if hs is None else 100 * hs, ' %', 1.0, 0)})")
    out.append(f"  ON : update {_fmt(_med(on, 'train/train_ms'), ' s', 1e3)}, EXPOSED collect "
               f"{_fmt(_med(on, 'rollout/collect_ms'), ' s', 1e3)}, join wait {_fmt(_med(on, 'overlap/join_wait_ms'), ' ms', 1.0, 0)}, "
               f"worker play {_fmt(_med(on, 'overlap/play_ms'), ' s', 1e3)}, pi_prox pass "
               f"{_fmt(_med(on, 'overlap/prox_forward_ms'), ' s', 1e3)}, IS ESS {_fmt(_med(on, 'overlap/is_ess'), '', 1.0, 3)}, "
               f"truncated {_fmt(_med(on, 'overlap/is_weight_truncated_frac'), '', 1.0, 3)}")
    for arm, s in (("OFF", off), ("ON ", on)):
        k = s["k9b"]
        if k.get("verdict") == "UNREAD":
            out.append(f"  K9(b) section 7, {arm}: UNREAD ({k.get('error')}) -- FLAG")
            continue
        fam = k["families"][k["required_family"]]
        mx = fam["max_after_fork"]
        out.append(f"  K9(b) section 7, {arm}: {k['verdict']}{' -- FLAG' if k['flag'] else ''} (family {k['required_family']}: "
                   f"judged max {_g(mx['judged_max'])}, p99 max {_g(mx['p99'])}; ceilings {k['ceilings']['judged_max']:g} / "
                   f"{k['ceilings']['p99']:g}; rows {fam['rows_after_fork']['judged_max']}; fired {k['fired_families'] or 'none'}; "
                   f"judged rows at or over the 1e-4 fatal bar: {k['n_judged_at_or_over_1e-4'].get(k['required_family'])})")
    out.append(f"  T2 reserved-growth warnings (counted, not fatal): OFF {off['t2_growth_warnings']}, ON {on['t2_growth_warnings']};"
               f" update-peak reserved MiB OFF {_fmt((off['tags'].get('lifecycle/cuda_update_peak_reserved_mib') or {}).get('max'), '', 1.0, 0)}"
               f" / ON {_fmt((on['tags'].get('lifecycle/cuda_update_peak_reserved_mib') or {}).get('max'), '', 1.0, 0)}")
    if dec["k9b_flags"]:
        out.append(f"  FLAGGED: {'; '.join(dec['k9b_flags'])} -- the throughput stands, the arm is suspect (warn mode: reported, not aborted)")
    return out


def _commits_line(out: Path) -> str | None:
    seen = {}
    for n in ORDER:
        c = (_load(out / "cells" / f"{n}.state.json") or {}).get("commit")
        if c:
            seen[n] = c
    if len(set(seen.values())) > 1:
        return ("WARNING: the cells ran at different checkout commits "
                + ", ".join(f"{n} {c}" for n, c in seen.items())
                + " -- main moved during the window; the X14 arms ran at the battery's pin regardless")
    return None


def selftest() -> str:
    """CPU: the fairness rules, the X14 reader, and the FP32_ALT gate's use of both, on synthetic cells."""
    import tempfile

    a = LP.selftest()
    b = XR.selftest()
    lines = ["[CompileRegions] installing ['R1_learner_micro_step'] (fullgraph=True)", "[CompileRegions] parity PASS (x)",
             "[CompileRegions] inventory == declaration: R1 1 graph", "[CompileControl] prewarmed 1 production signature(s)",
             "[CompileCanary] armed: R1"]

    def upd(arm: str, est: float, verdict: str = "PASS") -> dict[str, Any]:
        res = {"status": "ok", "train": {"update_s_est": est}, "gate": {"verdict": verdict, "lines": lines},
               "tau_force": {"arms": [arm], "intact": True, "eager_calls": {arm: 1}, "trace_calls": {arm: 2},
                             "violations": []}}
        m = LP.path_markers(res, "on", arm)
        res["path"] = {"mode": "on", "arm": arm, "markers": m, "violations": LP.path_violations(m)}
        return res

    dec = {"status": "ok", "agreement": {},
           "tau_force": {"arms": ["fp32_naive", "fp32_alt"], "intact": True, "eager_calls": {"fp32_naive": 1, "fp32_alt": 1},
                         "trace_calls": {"fp32_naive": 1, "fp32_alt": 1}, "violations": []},
           "summary": {b_: {"old_ms": 5.0, "new_ms": 5.1, "delta_ms": {"min": 0.05, "max": 0.15, "median": 0.1}}
                       for b_ in ("64", "256")}}
    with tempfile.TemporaryDirectory(prefix="battery_reader_selftest_") as td:
        cells = Path(td)
        (cells / "tau_dec.json").write_text(json.dumps(dec))
        for n, arm in UPD_ARM.items():
            (cells / f"{n}.json").write_text(json.dumps(upd(arm, 42.0 if arm == "fp32_naive" else 42.2)))
        good = fp32_alt_decision(cells)
        assert good["verdict"] in ("CONFIRMED", "OVER", "INCONCLUSIVE"), good
        # the unfair pair: the naive arm's gate was REFUSED while the alt arm's PASSED => INCOMPLETE, never a cost
        (cells / "tau_u_naive_b.json").write_text(json.dumps(upd("fp32_naive", 41.9, verdict="REFUSED")))
        bad = fp32_alt_decision(cells)
        assert bad["verdict"] == "INCOMPLETE" and any("different compiled paths" in m or "not PASS" in m
                                                      for m in bad["missing"]), bad
        # a cell that predates the rule (no `path`) is INCOMPLETE too
        legacy = upd("fp32_naive", 42.0)
        legacy.pop("path")
        (cells / "tau_u_naive_b.json").write_text(json.dumps(legacy))
        assert fp32_alt_decision(cells)["verdict"] == "INCOMPLETE"
        assert x14_decision(cells)["verdict"] == "INCOMPLETE"
        # the X14 section end to end on synthetic runs: read, decide, print
        runs = Path(td) / "runs"
        runs.mkdir()
        nan_free = {"overlap/join_wait_ms": 1.0, "overlap/play_ms": 50000.0, "overlap/prox_forward_ms": 2500.0}

        def dumps(n: int, cyc: float, extra: dict[str, float], snap: bool) -> list[tuple[int, float, dict[str, float]]]:
            o, t = [], 1000.0
            for k in range(1, n + 1):
                t += 300.0 if k <= 3 else cyc
                tags = {"rollout/collect_ms": 30000.0, "train/train_ms": 40000.0, "time/fps": 400.0 + k,
                        "lifecycle/cuda_update_peak_reserved_mib": 8300.0, **extra,
                        "behaviour/max_abs_dlogp_judged": 5e-5, "behaviour/p99_abs_dlogp_current": 1e-6}
                if snap:
                    tags.update({"behaviour_snapshot/max_abs_dlogp_judged": 6e-5,
                                 "behaviour_snapshot/p99_abs_dlogp_current": 2e-6})
                o.append((XR.PARENT_STEP + k * XR.ROWS_PER_UPDATE, t, tags))
            return o
        pin = "a" * 40
        for arm, cyc, extra, snap in (("off", 100.0, {}, False), ("on", 85.0, nan_free, True)):
            rd = XR._write_run(runs, arm, [dumps(15, cyc, extra, snap)], pin=pin)
            (cells / f"x14_{arm}.json").write_text(json.dumps({"status": "ok", "run_dir": rd,
                                                               "summary": XR.summarise(rd, arm)}, default=str))
        x = x14_decision(cells)
        txt = "\n".join(x14_lines(x))
        assert x["verdict"] == "CLEARS" and "x1.176" in txt and "K9(b) section 7, ON" in txt, txt
        # a cell json WITHOUT a stored summary is re-read from its run dir (the cell's own read failed)
        d_on = json.loads((cells / "x14_on.json").read_text())
        d_on.pop("summary")
        (cells / "x14_on.json").write_text(json.dumps(d_on))
        assert x14_decision(cells)["verdict"] == "CLEARS"
    return (f"{a}; {b}; reader: the gate is {good['verdict']} on a fair set and INCOMPLETE on an unfair or path-less one; "
            "the X14 section reads, decides (CLEARS at x1.176) and prints from synthetic runs, re-reading a run dir when "
            "a cell stored no summary")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("out", nargs="?")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        print("ok: " + selftest())
        return 0
    if not a.out:
        ap.error("out is required (or --selftest)")
    out = Path(a.out)
    rows, dec, x14 = table(out), fp32_alt_decision(out / "cells"), x14_decision(out / "cells")
    if a.json:
        print(json.dumps({"cells": rows, "fp32_alt": dec, "x14": x14}, indent=1, default=str))
        return 0
    st = _load(out / "battery_status.json") or {}
    print(f"FP32_ALT GPU-window battery {out}  (battery status: {st.get('status', '?')}; pin {str(st.get('pin'))[:12]}; "
          f"learner gate {st.get('learner_gate')})\n")
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
        print(f"  learner path (identical in all four cells): gate {dec['learner']['path']['mode']}, markers "
              f"{json.dumps(dec['learner']['path']['markers'])}")
    print()
    for line in x14_lines(x14):
        print(line)
    w = _commits_line(out)
    if w:
        print("\n" + w)
    return 0


if __name__ == "__main__":
    sys.exit(main())
