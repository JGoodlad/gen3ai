"""Read one measured launch (`drive.sh`): startup / compile figures from the child log, the steady
updates from the phase markers + the run's own TensorBoard, memory from the lifecycle scalars and
nvidia-smi, contention from the windowed CPU sampler. (X5 U2 GPU checks, 2026-10-04.)

    python read_run.py --tag x5fm_a --dir <scratch> --out results/x5fm_a.json

Rules: an update whose window read CONTENDED (any sampler row with windowed factor >= 1.05 inside it)
is EXCLUDED from the steady statistics and listed, never tolerated (standing rule 8); the startup dry
update and the first real update are warm-up; the compile-canary update (the 11th train call) is
excluded. TB scalars are matched to updates by their order (one row per update, `train/train_ms`).
"""
from __future__ import annotations

import argparse
import json
import re
import statistics as st
from pathlib import Path


def tb_scalars(tb_dir: Path):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    ea = EventAccumulator(str(tb_dir), size_guidance={"scalars": 0})
    ea.Reload()
    out = {}
    for t in ea.Tags()["scalars"]:
        out[t] = [(e.step, e.wall_time, e.value) for e in ea.Scalars(t)]
    return out


def first(pat: str, text: str, grp: int = 1):
    m = re.search(pat, text)
    return m.group(grp) if m else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    d = Path(a.dir)
    tag = a.tag
    log = (d / f"{tag}.log").read_text(errors="replace")
    ph = [json.loads(line) for line in (d / f"{tag}.phases.jsonl").read_text().splitlines()]
    res = {"tag": tag}

    # ---- startup / compile figures straight from the child's own lines
    start = next(r for r in ph if r["phase"] == "start")
    res["argv"] = start["argv"]
    upd = [r for r in ph if r["phase"] == "update"]
    col = [r for r in ph if r["phase"] == "collect"]
    res["startup_wall_s_start_to_first_collect"] = round(col[0]["t0"] - start["t0"], 1) if col else None
    res["startup_wall_s_start_to_dry_update_end"] = round(upd[0]["t1"] - start["t0"], 1) if upd else None
    res["T2_up_s"] = first(r"T2 (?:SHARED|up) in ([0-9.]+)s", log)
    res["T2_line"] = first(r"(🦀 \[RUST ENV\] T2 .*)", log)
    res["R1_prewarm_s"] = first(r"reset \+ prewarm took ([0-9.]+)s", log)
    res["R1_prewarm_line"] = first(r"(\[CompileControl\] prewarmed .*)", log)
    res["compile_lock_line"] = first(r"(🧊 \[COMPILE LOCK\].*)", log)
    res["compile_regions_parity"] = first(r"(\[CompileRegions\] parity .{0,400})", log)
    res["dry_update_line"] = first(r"(🧮 \[UpdateFit\].*)", log)
    res["ledger"] = re.findall(r"^\s+(weights on the card.*|rust env core.*|compiled regions.*|optimizer state declared.*)$", log, re.M)
    res["t2_gate_lines"] = re.findall(r"^(?:.*)(\[T2[^\n]{0,300}parity[^\n]{0,200})$", log, re.M)[:6]
    res["arch_source"] = first(r"(arch_source[^\n]{0,200})", log)
    res["fatal_or_error_lines"] = [l[:300] for l in log.splitlines()
                                   if re.search(r"FATAL|Traceback|RuntimeError|Error:", l)][:10]

    # ---- contention windows
    cpu = []
    p = d / f"{tag}.cpu.jsonl"
    if p.exists():
        cpu = [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
        cpu = [s for s in cpu if "t" in s]

    def factor_in(t0, t1):
        f = [s["factor"] for s in cpu if t0 <= s["t"] - s["dt"] and s["t"] <= t1]
        return max(f) if f else None

    # startup contention (compile is CPU-bound; report the worst window, never hide it)
    if col and cpu:
        res["startup_max_contention_factor"] = factor_in(start["t0"], col[0]["t0"])
        fs = [s["factor"] for s in cpu if start["t0"] <= s["t"] - s["dt"] and s["t"] <= col[0]["t0"]]
        res["startup_windows_ge_1.05"] = sum(1 for f in fs if f >= 1.05)
        res["startup_windows"] = len(fs)

    # ---- steady updates
    tb = tb_scalars(d / "models" / tag / "tb") if (d / "models" / tag / "tb").exists() else {}
    res["tb_tags"] = len(tb)
    rows = []
    for i, r in enumerate(upd):
        dur = r["t1"] - r["t0"]
        f = factor_in(r["t0"], r["t1"])
        why = None
        if i < 2:
            why = "warm-up (startup dry update / first real update)"
        elif i == 10:
            why = "compile-canary update"
        elif f is None or f >= 1.05:
            why = f"contended (max factor {f})"
        rows.append({"i": i, "dur_s": round(dur, 2), "max_factor": f, "excluded": why})
    res["updates"] = rows
    kept = [r["dur_s"] for r in rows if not r["excluded"]]
    res["update_wall_s_steady"] = {"n": len(kept), "values": kept,
                                   "median": round(st.median(kept), 3) if kept else None,
                                   "mean": round(st.mean(kept), 3) if kept else None}

    def series(tagname):
        return [v for (_s, _w, v) in tb.get(tagname, [])]

    # TB rows are one per update after the dry update (the dry update logs nothing); align by count
    n_tb = len(series("train/train_ms"))
    res["tb_update_rows"] = n_tb
    keep_idx = []
    for k in range(n_tb):
        i = k + 1                              # TB row k <-> phases update row k+1 (row 0 = startup dry)
        if i < len(rows) and not rows[i]["excluded"]:
            keep_idx.append(k)
    want = ["train/train_ms", "lifecycle/update_wall_s", "rust_env/flush_ms_per_host_step",
            "rust_env/gpu_wait_ms_per_host_step", "rust_env/core_ms_per_host_step",
            "rust_env/host_steps", "rust_env/trainee_rows_per_host_step", "rust_env/p2_policy_decisions",
            "rust_env/submit_ms_per_host_step", "rust_env/post_ms_per_host_step", "rust_env/fill_ms",
            "lifecycle/cuda_device_free_mib", "lifecycle/cuda_update_peak_reserved_mib",
            "lifecycle/cuda_update_peak_alloc_mib", "lifecycle/cuda_rollout_peak_reserved_mib",
            "lifecycle/cuda_reserved_mib", "lifecycle/compiled_region_calls", "compile/graphs_total",
            "compile/recompiles_after_lock", "compile/cache_limit_hits", "compile/locked",
            "compile/train_ms_vs_lock_baseline", "lifecycle/eager_fallback_calls", "lifecycle/eager_share",
            "time/fps"]
    stats = {}
    for t in want:
        s = series(t)
        sel = [s[k] for k in keep_idx if k < len(s)]
        stats[t] = {"all": [round(x, 4) for x in s], "steady_kept": [round(x, 4) for x in sel],
                    "median_steady": round(st.median(sel), 4) if sel else None}
    res["tb"] = stats

    # ---- memory from nvidia-smi
    p = d / f"{tag}.smi.csv"
    if p.exists():
        used = []
        for line in p.read_text().splitlines():
            parts = [x.strip() for x in line.split(",")]
            if len(parts) >= 3 and parts[2].endswith("MiB"):
                used.append(float(parts[2].split()[0]))
        res["smi_memory_used_mib_max"] = max(used) if used else None
        res["smi_memory_used_mib_median"] = st.median(used) if used else None
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in res if k not in ("tb", "updates", "ledger", "argv")}, indent=1)[:3500])


if __name__ == "__main__":
    main()
