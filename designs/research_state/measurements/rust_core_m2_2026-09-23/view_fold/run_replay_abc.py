"""Interleaved three-arm rerun of the M1 transport record's REPLAY bench, for M2's view-fold gate.

The bench is `m1_transport_throughput_2026-09-23/run_pairs.py --bench replay` unchanged in what it
measures: ONE persistent `sim_bridge` fed the recorded 41-battle transcript
(`transcript_env_seed0.txt`, 6,078 `CHOOSE`s), stdout discarded, CPU = the reaped child's
user+sys. Only the arm set differs:

    A  dfab2558  — before M1 (the record's A)
    B  7d71711c  — main before this fix (M1 + the truth-audit fold, the fold ON in sim_bridge)
    C  this tree — the fold opt-in (`gen3_view_fold_opt_in_v1`): sim_bridge folds nothing

ONE binary per process. Pair k runs the arms in the k-th of the six orders of (A, B, C), so every
arm sits in every position equally often and a monotone load drift cancels. The 1-min load average
is recorded at each sample's start and end. Every subprocess runs at `nice -n 10`.

    python run_replay_abc.py --bins A=/tmp/m2bench/A/src/rust_sim/target/release/sim_bridge \\
        B=... C=... --pairs 30 --reps 5 --out replay_abc_rows.jsonl
    python run_replay_abc.py --bins ... --identity   # byte identity of the three stdouts first
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import resource
import subprocess
import time

HERE = os.path.dirname(os.path.abspath(__file__))
TRANSCRIPT = os.path.join(HERE, "..", "..", "m1_transport_throughput_2026-09-23", "transcript_env_seed0.txt")
ORDERS = list(itertools.permutations(["A", "B", "C"]))


def sample(arm, exe, pair, reps, out):
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


def identity(bins):
    digests = {}
    for arm, exe in sorted(bins.items()):
        with open(TRANSCRIPT, "rb") as fin:
            p = subprocess.run(["nice", "-n", "10", exe], stdin=fin, capture_output=True, check=True)
        text = p.stdout.decode()
        digests[arm] = hashlib.sha256(p.stdout).hexdigest()
        print(json.dumps({"arm": arm, "bytes": len(p.stdout), "sha256": digests[arm],
                          "end": text.count("__END__"), "err": text.count("__ERR__")}))
    if len(set(digests.values())) != 1:
        raise SystemExit("STDOUT DIFFERS between arms")
    print("byte-identical stdout across", ", ".join(sorted(bins)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bins", nargs=3, required=True, help="A=path B=path C=path")
    ap.add_argument("--pairs", type=int, default=30)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--out", default=os.path.join(HERE, "replay_abc_rows.jsonl"))
    ap.add_argument("--identity", action="store_true")
    a = ap.parse_args()
    bins = dict(b.split("=", 1) for b in a.bins)
    assert sorted(bins) == ["A", "B", "C"], bins
    if a.identity:
        identity(bins)
        return
    for k in range(a.pairs):
        for arm in ORDERS[k % len(ORDERS)]:
            sample(arm, bins[arm], k + 1, a.reps, a.out)


if __name__ == "__main__":
    main()
