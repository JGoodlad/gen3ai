"""Static-token screen look 1: the registered IN-ARM speed s (design_static_tokens.md §8.1 "Speed rule"), REPORTED.

    python speed_read.py [--models M] [--cpu-dir ~/.claude/jobs/st_screen] [--out speed.json]

s = (pooled median update-cycle wall of `static`) / (pooled median of `legacy`) − 1, over QUIET cycles only,
EXCLUDING every resume window (amendment 1). ``scripts/ops/s_read.py`` reads the PAIRED benchmark's blocks (its own
phase markers, one trainer per block), not a resumed training run, so this applies its cycle rule to the six runs:

* A CYCLE is the wall between two consecutive ``train/train_ms`` TensorBoard records WITHIN ONE CHILD's events file
  (one file per launcher child; never across a restart, so the gap a stop + resume leaves is never a cycle).
* EXCLUDED, in this order (each cycle carries its first reason):
  1. the START WINDOW of EVERY child — its first 11 update records (s_read's rule: the compile-canary update is the
     10th train call, and the record after it reads short), so a resumed child's warm-up, compile and canary never
     count. Applied to the fresh launch too, so both arms lose the same cycles;
  2. a record whose step does not advance (the final save's / an abort's dump);
  3. a compile-canary verdict (``canary_verdicts.jsonl`` ``time``) inside the window;
  4. an in-loop eval cycle overlapping the window (``eval/wall_sec`` record at t covers [t − wall_sec, t]);
  5. the QUIET rule (standing rule 8): no CPU-sampler row of the run's ``<run>*.cpu.jsonl`` lies inside the window,
     or any such row has a windowed contention factor ≥ 1.05.
* The speed rule (§8.1): s ≤ 5 % adopts on the strength rule; 5 % < s ≤ 15 % also needs non-inferiority at matched
  wall-clock; s > 15 % stops the screen as an optimisation unit. Rule 8: s within 1e-9 of a boundary is not past it.

Also reported (never the read): per-seed medians and kept counts, the median ``lifecycle/update_wall_s`` per arm on
the kept cycles, and each run's END-TO-END wall (first child attach → last record, downtime between children
subtracted)."""
from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path
from typing import Any, Dict, List, Tuple

ARCHIVE_MODELS = Path("/home/goodlad/dev/gen3ai/models")
CPU_DIR = Path.home() / ".claude/jobs/st_screen"
SEEDS = (1001, 1002, 1003)
START_WINDOW = 11
QUIET = 1.05
RULE8_EPS = 1e-9
S_ADOPT, S_STOP = 0.05, 0.15


def scalars(path: Path, tag: str) -> List[Tuple[float, int, float]]:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    ea = EventAccumulator(str(path), size_guidance={"scalars": 0})
    ea.Reload()
    if tag not in ea.Tags()["scalars"]:
        return []
    return [(e.wall_time, e.step, e.value) for e in ea.Scalars(tag)]


def cpu_rows(cpu_dir: Path, run: str) -> List[dict]:
    rows: List[dict] = []
    for f in sorted(cpu_dir.glob(f"{run}*.cpu.jsonl")):
        for x in f.read_text().splitlines():
            if not x.strip():
                continue
            d = json.loads(x)
            if "t" in d and "factor" in d:
                rows.append(d)
    rows.sort(key=lambda d: d["t"])
    return rows


def read_run(models: Path, cpu_dir: Path, run: str) -> Dict[str, Any]:
    rd = models / run
    cpu = cpu_rows(cpu_dir, run)
    canary = [json.loads(x)["time"] for x in (rd / "canary_verdicts.jsonl").read_text().splitlines() if x.strip()] \
        if (rd / "canary_verdicts.jsonl").exists() else []
    files = sorted((rd / "tb").glob("events.out.tfevents.*"), key=lambda p: int(p.name.split(".")[3]))
    cycles: List[dict] = []
    children = []
    e2e = 0.0
    for f in files:
        recs = scalars(f, "train/train_ms")
        upd = {stp: v for _, stp, v in scalars(f, "lifecycle/update_wall_s")}
        evals = [(w - v, w) for w, _, v in scalars(f, "eval/wall_sec")]
        start = int(f.name.split(".")[3])
        children.append({"file": f.name, "tb_records": len(recs)})
        if recs:
            e2e += recs[-1][0] - start
        for k in range(1, len(recs)):
            w0, w1 = recs[k - 1][0], recs[k][0]
            why = None
            if k + 1 <= START_WINDOW:
                why = "start window (child's first 11 updates)"
            elif recs[k][1] <= recs[k - 1][1]:
                why = "step does not advance (a save / abort dump)"
            elif any(w0 < t <= w1 for t in canary):
                why = "compile canary in window"
            elif any(a < w1 and b > w0 for a, b in evals):
                why = "eval cycle in window"
            else:
                fs = [r["factor"] for r in cpu if w0 <= r["t"] - r["dt"] and r["t"] <= w1]
                if not fs:
                    why = "no sampler rows"
                elif max(fs) >= QUIET:
                    why = f"contended (max factor {max(fs):.3f})"
            cycles.append({"file": f.name, "k": k, "step": recs[k][1], "wall_s": round(w1 - w0, 3),
                           "update_wall_s": upd.get(recs[k][1]), "excluded": why})
    kept = [c["wall_s"] for c in cycles if c["excluded"] is None]
    reasons: Dict[str, int] = {}
    for c in cycles:
        key = (c["excluded"] or "kept").split(" (max")[0]
        reasons[key] = reasons.get(key, 0) + 1
    return {"run": run, "children": children, "cpu_rows": len(cpu), "canary_verdicts": len(canary),
            "cycles_total": len(cycles), "kept_n": len(kept), "median_s": st.median(kept) if kept else None,
            "kept_update_wall_s_median": st.median([c["update_wall_s"] for c in cycles
                                                    if c["excluded"] is None and c["update_wall_s"] is not None])
            if any(c["excluded"] is None and c["update_wall_s"] is not None for c in cycles) else None,
            "end_to_end_wall_h": round(e2e / 3600, 3), "exclusions": reasons, "kept": kept}


def speed_rule(s: float) -> Tuple[str, List[str]]:
    near = [f"s within {RULE8_EPS} of {b}" for b in (S_ADOPT, S_STOP) if abs(s - b) <= RULE8_EPS]
    if s > S_STOP and not near:
        return "s > 15 %: STOP before any meter read (an optimisation unit)", near
    if s > S_ADOPT and not any(str(S_ADOPT) in n for n in near):
        return "5 % < s <= 15 %: adoption ALSO needs non-inferiority at MATCHED WALL-CLOCK", near
    return "s <= 5 %: the strength rule alone decides", near


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=str(ARCHIVE_MODELS))
    ap.add_argument("--cpu-dir", default=str(CPU_DIR))
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "speed.json"))
    a = ap.parse_args()
    out: Dict[str, Any] = {"rule": __doc__.split("\n\n")[1].strip(), "arms": {}}
    pooled: Dict[str, List[float]] = {}
    for arm in ("legacy", "static"):
        runs = [read_run(Path(a.models), Path(a.cpu_dir), f"rb_st_{arm}_s{s}") for s in SEEDS]
        pooled[arm] = [w for r in runs for w in r["kept"]]
        out["arms"][arm] = {"runs": [{k: v for k, v in r.items() if k != "kept"} for r in runs],
                            "kept_n": len(pooled[arm]),
                            "pooled_median_s": st.median(pooled[arm]) if pooled[arm] else None,
                            "end_to_end_wall_h_mean": round(st.mean(r["end_to_end_wall_h"] for r in runs), 3)}
    ml, ms = out["arms"]["legacy"]["pooled_median_s"], out["arms"]["static"]["pooled_median_s"]
    if ml and ms:
        s = ms / ml - 1
        out["s"] = s
        out["speed_rule"], out["near_boundary"] = speed_rule(s)
        el, es = out["arms"]["legacy"]["end_to_end_wall_h_mean"], out["arms"]["static"]["end_to_end_wall_h_mean"]
        out["end_to_end_ratio_minus_1"] = es / el - 1
        out["matched_wall_checkpoint_floor_1M"] = int(15_000_000 / (1 + s) // 1_000_000) * 1_000_000
    else:
        out["s"] = None
        out["speed_rule"] = "INSUFFICIENT quiet cycles in an arm: s not computed"
    Path(a.out).write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: out[k] for k in ("s", "speed_rule") if k in out}, indent=1))
    for arm in ("legacy", "static"):
        A = out["arms"][arm]
        print(arm, "kept", A["kept_n"], "median", A["pooled_median_s"], "e2e mean h", A["end_to_end_wall_h_mean"])
        for r in A["runs"]:
            print("  ", r["run"], "cycles", r["cycles_total"], "kept", r["kept_n"], "median", r["median_s"],
                  "e2e h", r["end_to_end_wall_h"], r["exclusions"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
