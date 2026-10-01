#!/usr/bin/env python3
"""REGISTRATION §5.2 meters for ONE arm's final model — durable, resumable, detached-friendly.

    meters.py <label> <final_model.zip> [--workers 6] [--only u|ga]

U  = untaught-8: 8 teams x 600 battles (24 units of 25), --seed 0, opponent N0 @24M (its FILE), through
     the battery's own unit driver (n0_endofrun_2026-09-27/scripts/gu_unit.py, fixed bfb8e7e2). A unit
     resumes at its first missing battle; rows are fsynced JSONL under ROWS/u_rows/<label>/.
G-A = SmallRL guard: main.anchors, greedy, --server rust, teamset away + home x seeds {0,10,...,50},
     100 games each (1,200). A unit is DONE iff its summary.json says status OK and n == 100; a partial
     unit dir is moved aside (<dir>.partial.<epoch>) and re-run. G-A units run ONE at a time (the
     anchors tool starts its own server on a 95xx port), after U.

Every process: nice 15, CUDA hidden, one BLAS thread, no argv carrying the trainer's script name.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

WT = Path("/home/goodlad/dev/gen3ai-wt/m5-sizing")
GU = WT / "designs/research_state/measurements/n0_endofrun_2026-09-27/scripts/gu_unit.py"
OPP = "/home/goodlad/dev/gen3ai/models/ai_v14_01_base/snapshots/snapshot_000024000000.zip"
ROWS = Path("/home/goodlad/dev/gen3ai-reads/m5_sizing")
PY = "/home/goodlad/miniconda3/envs/gen3ai_torch28/bin/python3"
TEAMS, CHUNKS, PER_CHUNK = 8, 24, 25
GA_SEEDS = (0, 10, 20, 30, 40, 50)


def env() -> dict:
    e = dict(os.environ)
    e.update({"PYTHONPATH": str(WT / "src"), "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1",
              "MKL_NUM_THREADS": "1"})
    e["N0Q_TREE_COMMIT"] = subprocess.run(["git", "-C", str(WT), "rev-parse", "--short", "HEAD"],
                                          capture_output=True, text=True).stdout.strip()
    return e


def log(msg: str) -> None:
    line = f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(ROWS / "meters.log", "a") as fh:
        fh.write(line + "\n")


def u_unit_done(label: str, t: int, c: int) -> bool:
    p = ROWS / "u_rows" / label / f"t{t}" / f"c{c:02d}.jsonl"
    if not p.exists():
        return False
    js = {json.loads(x)["j"] for x in p.read_text().splitlines() if x.strip().endswith("}")}
    return len(js) == PER_CHUNK


def run_u(label: str, ref: str, t: int, c: int, e: dict) -> int:
    if u_unit_done(label, t, c):
        return 0
    lg = ROWS / "logs" / f"u_{label}_t{t}.log"
    with open(lg, "a") as fh:
        r = subprocess.run(["nice", "-n", "15", "timeout", "3600", PY, str(GU), "--label", label, "--ref", ref,
                            "--opponent", OPP, "--team-index", str(t), "--chunk", str(c),
                            "--rows", str(ROWS / "u_rows")], stdout=fh, stderr=subprocess.STDOUT, env=e, cwd=WT)
    ok = u_unit_done(label, t, c)
    log(f"U {label} t{t} c{c:02d} rc={r.returncode} complete={ok}")
    return 0 if ok else 1


def ga_done(d: Path) -> bool:
    s = d / "summary.json"
    if not s.exists():
        return False
    try:
        j = json.loads(s.read_text())
    except ValueError:
        return False
    return j.get("status") == "OK" and int(j.get("n", 0)) == 100


def run_ga(label: str, ref: str, ts: str, seed: int, e: dict) -> int:
    d = ROWS / "ga" / label / f"{ts}_s{seed}"
    if ga_done(d):
        return 0
    if d.exists():
        d.rename(d.with_name(d.name + f".partial.{int(time.time())}"))
    d.parent.mkdir(parents=True, exist_ok=True)
    with open(ROWS / "logs" / f"ga_{label}_{ts}_s{seed}.log", "a") as fh:
        r = subprocess.run(["nice", "-n", "15", "timeout", "7200", PY, "-m", "main.anchors", "--model", ref,
                            "--opponent", "metamon:SmallRL", "--server", "rust", "--regime", "greedy",
                            "--teamset", ts, "--team-seed", str(seed), "--seed-base", str(seed),
                            "--games", "100", "--device", "cpu", "--out", str(d)],
                           stdout=fh, stderr=subprocess.STDOUT, env=e, cwd=WT)
    ok = ga_done(d)
    log(f"GA {label} {ts} s{seed} rc={r.returncode} ok={ok}")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("label")
    ap.add_argument("ref")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--only", choices=("u", "ga"), default=None)
    a = ap.parse_args()
    if not Path(a.ref).is_file():
        print(f"no such model: {a.ref}", file=sys.stderr)
        return 2
    for sub in ("u_rows", "ga", "logs"):
        (ROWS / sub).mkdir(parents=True, exist_ok=True)
    e = env()
    fails = 0
    if a.only in (None, "u"):
        log(f"U start {a.label} {a.ref} workers={a.workers}")
        # team-major interleave so a partial read covers every team evenly
        jobs = [(t, c) for c in range(CHUNKS) for t in range(TEAMS)]
        with ThreadPoolExecutor(max_workers=a.workers) as ex:
            fails += sum(ex.map(lambda tc: run_u(a.label, a.ref, tc[0], tc[1], e), jobs))
        log(f"U end {a.label} failed_units={fails}")
    if a.only in (None, "ga"):
        for ts in ("away", "home"):
            for s in GA_SEEDS:
                fails += run_ga(a.label, a.ref, ts, s, e)
        log(f"GA end {a.label}")
    log(f"METERS_DONE {a.label} failed_units={fails}")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
