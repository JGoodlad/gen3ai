"""Read one measured launch (`drive.sh`): the startup / compile / parity lines from the run's child log, the
per-update cost from the run's own TensorBoard, contention + GPU memory from `sampler.py`. (gpu_checks_endstate,
2026-10-09; adapted from x5_u2_gpu_checks_2026-10-04/scripts/read_run.py.)

    python read_run.py --tag <run name> --run-dir <models/<run>> --scratch <dir> --out results/<tag>.json

Rules (standing rule 8): TB row 0 (the first real update, which carries the warm-up) and the compile-canary update
(update 10) are excluded from the steady statistics; an update whose window [wall_time - update_wall_s, wall_time]
holds any sampler window with factor >= 1.05 is CONTENDED and excluded, and listed, never tolerated. When fewer than
3 updates survive, the contended updates are reported SEPARATELY as a descriptive median (marked so), never mixed.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics as st
from pathlib import Path

CONTENDED = 1.05


def tb_scalars(tb_dir: Path):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    out = {}
    for ev in sorted(tb_dir.rglob("events.out.tfevents.*")):
        ea = EventAccumulator(str(ev), size_guidance={"scalars": 0})
        ea.Reload()
        for t in ea.Tags()["scalars"]:
            out.setdefault(t, []).extend((e.step, e.wall_time, e.value) for e in ea.Scalars(t))
    return out


def first(pat: str, text: str, grp: int = 1):
    m = re.search(pat, text, re.M)
    return m.group(grp) if m else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--scratch", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rd, sc = Path(a.run_dir), Path(a.scratch)
    log = (rd / "launcher_child.full.log").read_text(errors="replace")
    res: dict = {"tag": a.tag}
    res["child_attached"] = first(r"^===== child attached (.*) =====", log)
    res["arch_lines"] = re.findall(r"^(\[Arch\].{0,300})$", log, re.M)[:4]
    res["env_core_line"] = first(r"^(🦀 \[ENV CORE\].{0,400})$", log)
    res["critic_line"] = first(r"^(🎯 \[CRITIC\].{0,200})", log)
    res["model_version_line"] = first(r"^(.{0,40}\[ModelVersion\].{0,300})$", log)
    res["compile_trainer_line"] = first(r"^(⚡ \[CompileTrainer\].{0,300})$", log)
    res["dynamo_reset_line"] = first(r"^(\[CompileControl\] dynamo reset.{0,300})$", log)
    res["regions_parity"] = first(r"^(\[CompileRegions\] parity .{0,1600})$", log)
    res["regions_inventory"] = first(r"^(\[CompileRegions\] inventory.{0,300})$", log)
    res["prewarm_line"] = first(r"^(\[CompileControl\] prewarmed .{0,400})$", log)
    res["prewarm_s"] = first(r"reset \+ prewarm took ([0-9.]+)s", log)
    res["compile_lock_line"] = first(r"^(🧊 \[COMPILE LOCK\] after.{0,600})$", log)
    res["canary_armed"] = first(r"^(🐤 \[CompileCanary\] armed.{0,300})", log)
    res["canary_lines"] = re.findall(r"^(.{0,10}\[CompileCanary\](?! armed).{0,600})$", log, re.M)[:6]
    res["cuda_ledger"] = re.findall(r"^\s+((?:weights on the card|rust env core|compiled regions|optimizer state).{0,200})$",
                                    log, re.M)
    res["update_fit_line"] = first(r"^(🧮 \[UpdateFit\].{0,600})$", log)
    res["learner_freeze_line"] = first(r"^(🧊 \[LEARNER FREEZE\] the first rollout.{0,300})", log)
    res["t2_lines"] = [l[:400] for l in log.splitlines() if re.search(r"\bT2\b|InferenceService|\[T2", l)][:20]
    # the env core / T2 lines go through the launcher's ipc event channel (headless: echoed to its stdout)
    p = sc / f"{a.tag}.launcher.log"
    llog = p.read_text(errors="replace") if p.exists() else ""
    res["launcher_event_lines"] = [l[:500] for l in llog.splitlines()
                                   if re.search(r"RUST ENV|ENV CORE|\bT2\b|ModelVersion|CRITIC|FATAL|Crash|crash", l)][:30]
    res["t2_up_s"] = first(r"T2 (?:SHARED|up) in ([0-9.]+)s", llog)
    res["fatal_or_error_lines"] = [l[:300] for l in log.splitlines()
                                   if re.search(r"FATAL|Traceback|RuntimeError|Error:|recompil", l)][:20]
    p = rd / "canary_verdicts.jsonl"
    res["canary_verdicts"] = [json.loads(x) for x in p.read_text().splitlines()] if p.exists() else []
    p = rd / "update_fit.json"
    if p.exists():
        uf = json.loads(p.read_text())
        res["update_fit"] = {k: uf.get(k) for k in ("demand_mib", "peak_alloc_mib", "reserved_mib", "device_free_mib",
                                                     "card_mib", "headroom_mib", "headroom_required_mib", "seconds",
                                                     "oom")}
    p = rd / "model_config.json"
    if p.exists():
        mc = json.loads(p.read_text())
        res["model_config"] = {k: mc.get(k) for k in ("config_version", "arch_signature", "arch_source", "token_encoding",
                                                       "trunk_layers", "mon_hazard_cost", "move_actor_state",
                                                       "switch_hazard_cost", "eot_residual", "move_resolution",
                                                       "speed_physics", "op_reduction", "obs_facts",
                                                       "value_threat_inject")}

    cpu = []
    p = sc / f"{a.tag}.cpu.jsonl"
    if p.exists():
        cpu = [json.loads(x) for x in p.read_text().splitlines() if x.strip()]

    def window(t0, t1):
        return [s for s in cpu if t0 <= s["t"] - s["dt"] and s["t"] <= t1]

    t_start = None
    m = re.search(r"start ([0-9.]+)", (sc / f"{a.tag}.drive.log").read_text())
    if m:
        t_start = float(m.group(1))
    tb = tb_scalars(rd / "tb") if (rd / "tb").exists() else {}
    res["tb_tags"] = len(tb)
    tm = tb.get("train/train_ms", [])
    uw = {s: v for (s, _w, v) in tb.get("lifecycle/update_wall_s", [])}
    rows = []
    for k, (step, wall, v) in enumerate(tm):
        dur = uw.get(step, v / 1000.0)
        win = window(wall - dur - 30.0, wall)       # the update's train phase plus the rollout before it
        f = max((s["factor"] for s in win), default=None)
        why = None
        if k == 0:
            why = "warm-up (first real update)"
        elif k == 9:
            why = "compile-canary update (update 10)"
        elif k > 0 and step == tm[k - 1][0]:
            why = "the stop's interrupt-time row (same step as the previous update)"
        elif f is None or f >= CONTENDED:
            why = f"contended (max factor {f})"
        rows.append({"k": k, "step": step, "train_ms": round(v, 1), "max_factor": f, "excluded": why})
    res["updates"] = rows
    if cpu and t_start and tm:
        first_wall = tm[0][1]
        fs = [s["factor"] for s in window(t_start, first_wall)]
        res["startup_windows"] = len(fs)
        res["startup_windows_ge_1.05"] = sum(1 for f in fs if f >= CONTENDED)
        res["startup_max_factor"] = max(fs) if fs else None
        res["start_to_first_update_s"] = round(first_wall - t_start, 1)
        res["gpu_mem_used_mib_max"] = max((s.get("gpu_mem_used_mib", 0) for s in cpu), default=None)
    kept = [r["k"] for r in rows if not r["excluded"]]
    cont = [r["k"] for r in rows if r["excluded"] and r["excluded"].startswith("contended")]
    res["kept_updates"] = kept
    res["contended_updates"] = cont
    want = ["train/train_ms", "lifecycle/update_wall_s", "time/fps", "rust_env/flush_ms_per_host_step",
            "rust_env/gpu_wait_ms_per_host_step", "rust_env/core_ms_per_host_step", "rust_env/host_steps",
            "rust_env/fill_ms", "lifecycle/cuda_device_free_mib", "lifecycle/cuda_update_peak_reserved_mib",
            "lifecycle/cuda_update_peak_alloc_mib", "lifecycle/cuda_rollout_peak_reserved_mib",
            "compile/graphs_total", "compile/recompiles_after_lock", "compile/cache_limit_hits", "compile/locked",
            "compile/canary_unconfirmed_disagreements", "lifecycle/eager_fallback_calls"]
    stats = {}
    for t in want:
        s = [v for (_s, _w, v) in tb.get(t, [])]
        sel = [s[k] for k in kept if k < len(s)]
        csel = [s[k] for k in cont if k < len(s)]
        stats[t] = {"all": [round(x, 4) for x in s],
                    "median_steady": round(st.median(sel), 4) if sel else None, "n_steady": len(sel),
                    "median_contended_DESCRIPTIVE": round(st.median(csel), 4) if csel else None}
    # `time/fps` is SB3's CUMULATIVE rate since learn() began (startup included), not a per-update figure: the
    # per-update CYCLE rate is (steps between consecutive update rows) / (wall between them) = rollout + train.
    cyc = []
    for k in range(1, len(tm)):
        ds, dw = tm[k][0] - tm[k - 1][0], tm[k][1] - tm[k - 1][1]
        cyc.append({"k": k, "cycle_s": round(dw, 2), "steps": ds, "fps": round(ds / dw, 1) if dw > 0 else None,
                    "rollout_s": round(dw - tm[k][2] / 1000.0, 2)})
    res["cycles"] = cyc
    ck = [c for c in cyc if c["k"] in kept]
    stats["cycle_fps"] = {"all": [c["fps"] for c in cyc], "n_steady": len(ck),
                          "median_steady": round(st.median(c["fps"] for c in ck), 1) if ck else None}
    stats["cycle_rollout_s"] = {"all": [c["rollout_s"] for c in cyc], "n_steady": len(ck),
                                "median_steady": round(st.median(c["rollout_s"] for c in ck), 2) if ck else None}
    res["tb"] = stats
    res["tb_tags_rust_env"] = sorted(t for t in tb if t.startswith(("rust_env/", "compile/", "lifecycle/cuda")))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1, ensure_ascii=False))
    short = {k: res[k] for k in res if k not in ("tb", "updates", "cycles", "tb_tags_rust_env", "launcher_event_lines",
                                                    "cuda_ledger", "arch_lines")}
    print(json.dumps(short, indent=1, ensure_ascii=False)[:6000])
    for t in ("train/train_ms", "cycle_fps", "cycle_rollout_s","rust_env/gpu_wait_ms_per_host_step", "rust_env/flush_ms_per_host_step",
              "lifecycle/cuda_device_free_mib", "compile/graphs_total", "compile/recompiles_after_lock"):
        print(t, stats[t])


if __name__ == "__main__":
    main()
