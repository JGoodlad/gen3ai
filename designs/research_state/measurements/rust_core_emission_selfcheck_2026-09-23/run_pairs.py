"""Interleaved A/B: does the EMISSION SELF-CHECK cost the PRODUCTION release build anything?

A = `origin/main` `c49ef704` (before the check), B = this change's `cargo build --release` (the
check compiled OUT), C = this change's `cargo build --profile selfcheck --features
emission-selfcheck` (the check ON — what the fuzzers pay; informational). ONE persistent
`sim_bridge` per sample fed the recorded 41-battle / 6,078-`CHOOSE` training transcript of the M1
record (`../m1_transport_throughput_2026-09-23/transcript_env_seed0.txt`), stdout discarded, `reps`
replays per sample; CPU = the reaped children's user+sys. Pair k runs the arms in a rotated order
so a monotone load drift cancels. Every sample runs at `nice -n 10`.

    python run_pairs.py --pairs 30 --reps 5 --out replay_rows.jsonl
    python ../m1_transport_throughput_2026-09-23/analyze.py replay_rows.jsonl   # B vs A
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import subprocess
import time

WT = "/home/goodlad/dev/gen3ai-wt"
BIN = {
    "A": f"{WT}/esc-base/src/rust_sim/target/release/sim_bridge",
    "B": f"{WT}/emission-selfcheck/src/rust_sim/target/release/sim_bridge",
    "C": f"{WT}/emission-selfcheck/src/rust_sim/target/selfcheck/sim_bridge",
}
HERE = os.path.dirname(os.path.abspath(__file__))
TRANSCRIPT = os.path.join(HERE, "..", "m1_transport_throughput_2026-09-23", "transcript_env_seed0.txt")


def run_replay(arm, pair, out, reps):
    exe = BIN[arm]
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
    row = {"arm": arm, "pair": pair, "binary": exe, "reps": reps, "replay_cpu_s": cpu / reps,
           "replay_wall_s": wall / reps, "load1_start": load0, "load1_end": os.getloadavg()[0]}
    with open(out, "a") as f:
        f.write(json.dumps(row) + "\n")
    print(json.dumps(row), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=int, required=True)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    orders = (["A", "B", "C"], ["B", "C", "A"], ["C", "A", "B"], ["B", "A", "C"], ["A", "C", "B"], ["C", "B", "A"])
    for k in range(1, a.pairs + 1):
        for arm in orders[k % len(orders)]:
            run_replay(arm, k, a.out, a.reps)


if __name__ == "__main__":
    main()
