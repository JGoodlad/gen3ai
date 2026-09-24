"""Interleaved A/B driver for the M1 transport-throughput measurement.

ONE binary per process: every sample is its own subprocess with `$POKESIM_SIM_BRIDGE_BIN` naming
exactly one side's build. Pair k runs A-then-B for odd k and
B-then-A for even k, so a monotone drift in the box's load cancels across pairs instead of
biasing every ratio the same way. Every row records the 1-min load average at its start and end.

    python run_pairs.py --bench env    --pairs 12 --out env_rows.jsonl
    python run_pairs.py --bench thru   --pairs 8  --out thru_rows.jsonl
    python run_pairs.py --bench startup --pairs 8 --out startup_rows.jsonl
    python run_pairs.py --bench replay --pairs 30 --reps 3 --out replay_rows.jsonl

All subprocesses run at `nice -n 10`. Python comes from the B worktree for BOTH arms (the
training-path Python is identical at A and B — see README); only the Rust binary varies.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import resource
import subprocess
import sys
import time

PY = "/home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3"
WT_A = "/home/goodlad/dev/gen3ai/.claude/worktrees/m1bench-A"
WT_B = "/home/goodlad/dev/gen3ai/.claude/worktrees/agent-a35e14fb9d6f01186"
BIN = {
    "A": f"{WT_A}/src/rust_sim/target/release",
    "B": f"{WT_B}/src/rust_sim/target/release",
}
HERE = os.path.dirname(os.path.abspath(__file__))


def _env(arm):
    e = dict(os.environ)
    e["PYTHONPATH"] = f"{WT_B}/src"
    e["POKESIM_SIM_BRIDGE_BIN"] = f"{BIN[arm]}/sim_bridge"
    e.pop("CARGO_TARGET_DIR", None)
    return e


def run_env(arm, pair, out, decisions):
    cmd = ["nice", "-n", "10", PY, f"{HERE}/env_step_rust.py", "--decisions", str(decisions),
           "--arm", arm, "--pair", str(pair), "--out", out]
    p = subprocess.run(cmd, env=_env(arm), cwd=WT_B, capture_output=True, text=True, timeout=1800)
    if p.returncode != 0:
        print(p.stdout[-2000:], p.stderr[-3000:], file=sys.stderr)
        raise SystemExit(f"env run failed: arm={arm} pair={pair}")
    print(p.stdout.strip().splitlines()[-1], flush=True)


def run_thru(arm, pair, out, workers, seconds):
    load0 = os.getloadavg()[0]
    cmd = ["nice", "-n", "10", PY, f"{WT_B}/src/utils/bridge/bridge_impl_throughput_benchmark.py",
           "--impl", "rust", "--workers", str(workers), "--seconds", str(seconds),
           "--warmup-steps", "200"]
    p = subprocess.run(cmd, env=_env(arm), cwd=WT_B, capture_output=True, text=True, timeout=3600)
    load1 = os.getloadavg()[0]
    m = re.search(r"rust:\s+([\d.]+) steps/s aggregate.*?\|\s+(\d+) battles.*?\|\s+(\d+) failed",
                  p.stdout)
    if p.returncode != 0 or not m:
        print(p.stdout[-3000:], p.stderr[-3000:], file=sys.stderr)
        raise SystemExit(f"thru run failed: arm={arm} pair={pair}")
    row = {"arm": arm, "pair": pair, "binary": f"{BIN[arm]}/sim_bridge", "workers": workers,
           "seconds": seconds, "fps": float(m.group(1)), "battles": int(m.group(2)),
           "failed_workers": int(m.group(3)), "load1_start": load0, "load1_end": load1}
    with open(out, "a") as f:
        f.write(json.dumps(row) + "\n")
    print(json.dumps(row), flush=True)


def run_startup(arm, pair, out, reps):
    """`reps` bare `sim_bridge` lifetimes (spawn, eager `Dex::for_gen(3)`, EOF on stdin, exit).
    The env arm spawns one child per battle, so this is the fixed per-battle term to subtract
    from its `child_cpu_s` to leave the per-decision Rust work."""
    exe = f"{BIN[arm]}/sim_bridge"
    load0 = os.getloadavg()[0]
    r0 = resource.getrusage(resource.RUSAGE_CHILDREN)
    t0 = time.perf_counter()
    for _ in range(reps):
        subprocess.run(["nice", "-n", "10", exe], stdin=subprocess.DEVNULL,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    wall = time.perf_counter() - t0
    r1 = resource.getrusage(resource.RUSAGE_CHILDREN)
    cpu = (r1.ru_utime - r0.ru_utime) + (r1.ru_stime - r0.ru_stime)
    row = {"arm": arm, "pair": pair, "binary": exe, "reps": reps, "cpu_per_spawn_ms":
           1000 * cpu / reps, "wall_per_spawn_ms": 1000 * wall / reps,
           "load1_start": load0, "load1_end": os.getloadavg()[0]}
    with open(out, "a") as f:
        f.write(json.dumps(row) + "\n")
    print(json.dumps(row), flush=True)


TRANSCRIPT = os.path.join(HERE, "transcript_env_seed0.txt")


def run_replay(arm, pair, out, reps):
    """`reps` x (ONE persistent `sim_bridge` fed the recorded transcript, stdout discarded).
    Pure Rust: no Python in the process, one spawn per rep. CPU = the child's user+sys."""
    exe = f"{BIN[arm]}/sim_bridge"
    load0 = os.getloadavg()[0]
    r0 = resource.getrusage(resource.RUSAGE_CHILDREN)
    t0 = time.perf_counter()
    for _ in range(reps):
        with open(TRANSCRIPT, "rb") as fin:
            subprocess.run(["nice", "-n", "10", exe], stdin=fin, stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL, check=True)
    wall = time.perf_counter() - t0
    r1 = resource.getrusage(resource.RUSAGE_CHILDREN)
    cpu = (r1.ru_utime - r0.ru_utime) + (r1.ru_stime - r0.ru_stime)
    row = {"arm": arm, "pair": pair, "binary": exe, "reps": reps,
           "replay_cpu_s": cpu / reps, "replay_wall_s": wall / reps,
           "load1_start": load0, "load1_end": os.getloadavg()[0]}
    with open(out, "a") as f:
        f.write(json.dumps(row) + "\n")
    print(json.dumps(row), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", choices=["env", "thru", "startup", "replay"], required=True)
    ap.add_argument("--pairs", type=int, required=True)
    ap.add_argument("--first-pair", type=int, default=1)
    ap.add_argument("--out", required=True)
    ap.add_argument("--decisions", type=int, default=3000)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seconds", type=float, default=45.0)
    ap.add_argument("--reps", type=int, default=100)
    a = ap.parse_args()
    for k in range(a.first_pair, a.first_pair + a.pairs):
        order = ["A", "B"] if k % 2 == 1 else ["B", "A"]
        for arm in order:
            if a.bench == "env":
                run_env(arm, k, a.out, a.decisions)
            elif a.bench == "thru":
                run_thru(arm, k, a.out, a.workers, a.seconds)
            elif a.bench == "startup":
                run_startup(arm, k, a.out, a.reps)
            else:
                run_replay(arm, k, a.out, a.reps)


if __name__ == "__main__":
    main()
