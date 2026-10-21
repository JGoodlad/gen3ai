"""X47 look 1's cells (BATCH1_LAUNCH_SHEET §4.5), played on CPU in <= 45-min chunks.

The four registered requests (`x35_x47_look1_{a,gf,b,c}`) with the registered settings (500 pairs, purpose ab,
family x35_x47, label x35_x47_look1, greedy vs greedy, schedule seed 0, the default 500-pair batch). Each cell is
its own `play-many` invocation under its request (a re-run of a request resumes it; the schedule key is the two
checkpoints' digest, so the rows are the same as one call's), which keeps every invocation under the 45-min
chunk bound. Lanes pull cells from one queue, primary cells first. Ledger root: OUTSIDE models/ (this unit is
read-only on models/).

    python play_cells.py <seq_look1.json> <ledger root> <out dir> [--lanes 3]
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import subprocess
import sys
import threading
import time

M = "/home/goodlad/dev/gen3ai/models"
REFPOP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "reference_population_toy.json")
LINEAGE = ("toy_pstar", "toy_280m", "toy_220m", "toy_150m", "toy_080m")
X_MEMBER = "pstar_x26"
REQ = "x35_x47_look1_{}"


def plan(seq: dict) -> list:
    members = {m["label"]: m["path"] for m in json.load(open(REFPOP))["members"]}
    assert set(members) == set(LINEAGE) | {X_MEMBER}, members
    s_eq = [os.path.join(M, run, v["file"]) for run, v in sorted(seq["s_eq"].items())]
    finals = [os.path.join(M, run, "final_model.zip") for run in sorted(seq["s_eq"])]
    assert not any(v["is_final"] for v in seq["s_eq"].values())          # else a node IS its final: played once
    c0 = [os.path.join(M, run, v["file"]) for run, v in sorted(seq["c0_final"].items())]
    items = []
    # primary first (s_eq x R, C0 finals x R), slow P* cells first within each tier
    items += [("b", x, members[X_MEMBER]) for x in s_eq]
    items += [("c", c, members[X_MEMBER]) for c in c0]
    items += [("c", c, members[m]) for c in c0 for m in LINEAGE]
    items += [("a", x, members[m]) for x in s_eq for m in LINEAGE]
    # the secondary (X47 finals x R)
    items += [("b", x, members[X_MEMBER]) for x in finals]
    items += [("a", x, members[m]) for x in finals for m in LINEAGE]
    # G-F (s_eq rows x C0 finals columns)
    items += [("gf", x, c) for x in s_eq for c in c0]
    assert len(items) == 63, len(items)
    for _, p, o in items:
        assert os.path.isfile(p) and os.path.isfile(o), (p, o)
    return items


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("seq")
    ap.add_argument("ledger")
    ap.add_argument("out")
    ap.add_argument("--lanes", type=int, default=3)
    a = ap.parse_args()
    items = plan(json.load(open(a.seq)))
    os.makedirs(a.out, exist_ok=True)
    json.dump([{"request": REQ.format(k), "player": p, "opponent": o} for k, p, o in items],
              open(os.path.join(a.out, "cells_plan.json"), "w"), indent=1)
    py = os.environ.get("GEN3AI_PYTHON", "/home/goodlad/miniconda3/envs/gen3ai_torch28/bin/python3")
    memcap = os.path.join(os.environ["X47_REPO"], "scripts/ops/mem_cap.sh")
    q: "queue.Queue" = queue.Queue()
    for i, it in enumerate(items):
        q.put((i, it, 0))
    lock = threading.Lock()
    status = {}

    def log(msg: str) -> None:
        with lock:
            with open(os.path.join(a.out, "driver.log"), "a") as f:
                f.write(time.strftime("%H:%M:%S ") + msg + "\n")

    def lane(n: int) -> None:
        while True:
            try:
                i, (k, p, o), tries = q.get_nowait()
            except queue.Empty:
                return
            cf = os.path.join(a.out, f"cell_{i:02d}.json")
            json.dump([[p, o]], open(cf, "w"))
            cmd = [memcap, "--name", f"x47l1-{i:02d}", "4", "timeout", "2700", py, "-m", "main.h2h", "play-many",
                   "--cells", cf, "--pairs", "500", "--purpose", "ab", "--family", "x35_x47",
                   "--request", REQ.format(k), "--label", "x35_x47_look1", "--seed", "0", "--out", a.ledger,
                   "--threads", "2", "--torch-threads", "2"]
            t0 = time.time()
            log(f"lane {n} START cell {i:02d} [{k}] {os.path.relpath(p, M)} vs {os.path.relpath(o, M)} (try {tries})")
            with open(os.path.join(a.out, f"cell_{i:02d}.log"), "a") as lf:
                rc = subprocess.call(["taskset", "-c", "5-7,13-15", *cmd], stdout=lf, stderr=subprocess.STDOUT)
            log(f"lane {n} END cell {i:02d} rc {rc} in {time.time() - t0:.0f}s")
            status[i] = rc
            if rc != 0 and tries < 1:
                q.put((i, (k, p, o), tries + 1))

    ts = [threading.Thread(target=lane, args=(n,)) for n in range(a.lanes)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    bad = {i: rc for i, rc in status.items() if rc != 0}
    log(f"DONE: {len(status)} cells, nonzero {bad}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
