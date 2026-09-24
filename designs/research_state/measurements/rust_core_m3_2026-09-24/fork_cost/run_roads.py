"""Interleaved, ONE ROAD PER PROCESS, runs of `search_decision_benchmark.py` for M3's tracker
fork-cost record (copied from M2's `search/run_roads.py`; resumable).

Roads: `core` (materializer=core, typed — today's production road: the successor's trackers are
the PYTHON ones, forked by the pinned-pickle thaw) and `core-trk` (the same road with the Rust
core's per-decision TRACKERS also folded on every version, `SearchConfig.core_trackers`). Pair k
runs the two in alternating order. The Python thaw + record_context + advance_window phases come
out of the `core` rows; the Rust trackers' cost is `core-trk`'s Rust `core_us` minus `core`'s.

    python run_roads.py --bin-dir <worktree>/src/rust_sim/target/release --label wide \\
        --pairs 6 -- --decisions 10 --m-opp 3 --arm honest --k-worlds 4
    python run_roads.py ... --label b1 --pairs 6 -- --decisions 10 --n-actions 1 --arm base

Each process appends one JSON row (tagged with `label` and `pair`) to `rows.jsonl` here.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PY = "/home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3"
ROADS = ("core", "core-trk")
ORDERS = list(itertools.permutations(ROADS))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin-dir", required=True)
    ap.add_argument("--src", required=True, help="the tree's src/ (PYTHONPATH)")
    ap.add_argument("--traces", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--pairs", type=int, default=6)
    ap.add_argument("--out", default=os.path.join(HERE, "rows.jsonl"))
    ap.add_argument("rest", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    rest = [x for x in a.rest if x != "--"]
    env = dict(os.environ)
    env["PYTHONPATH"] = a.src
    env["POKESIM_SEARCH_DRIVER_BIN"] = os.path.join(a.bin_dir, "search_driver")
    env["POKESIM_SIM_BRIDGE_BIN"] = os.path.join(a.bin_dir, "sim_bridge")
    env["POKESIM_CORE_EVENTS_BIN"] = os.path.join(a.bin_dir, "core_events")
    tmp = a.out + ".tmp"
    done = set()
    if os.path.exists(a.out):
        for line in open(a.out):
            if line.strip():
                r = json.loads(line)
                done.add((r.get("label"), r.get("pair"), r.get("road")))
    for k in range(a.pairs):
        for road in ORDERS[k % len(ORDERS)]:
            if (a.label, k + 1, road) in done:
                continue
            if os.path.exists(tmp):
                os.remove(tmp)
            cmd = ["nice", "-n", "10", PY, os.path.join(a.src, "main", "search_dividend",
                                                         "search_decision_benchmark.py"),
                   "--traces", a.traces, "--roads", road, "--rust-timing", "--json-out", tmp, *rest]
            load0 = os.getloadavg()[0]
            p = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=3600)
            if p.returncode != 0:
                print(p.stdout[-2000:], p.stderr[-3000:], file=sys.stderr)
                raise SystemExit(f"road {road} pair {k + 1} failed")
            (row,) = [json.loads(line) for line in open(tmp)]
            row.update(label=a.label, pair=k + 1, load1_start=load0, load1_end=os.getloadavg()[0],
                       argv=rest)
            with open(a.out, "a") as fh:
                fh.write(json.dumps(row) + "\n")
            print(json.dumps({x: row[x] for x in ("label", "pair", "road", "arms",
                                                  "ms_per_decision_median", "load1_start")}),
                  flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
