"""`trainer_turn_benchmark --pin-battles`, on the RUST bridge — the end-to-end env-step arm.

`trainer_turn_benchmark.main` drives its battles through `run_local_battles(...)` with the
runner's default `impl="node"`, so run as shipped it never touches `sim_bridge` and cannot see
a Rust change. This wrapper rebinds the ONE name the benchmark reads at call time
(`agents.training.trainer_turn_benchmark.run_local_battles`) to the same function with
`impl="rust"`, and changes nothing else: same pinned battle sequence (`_battle_seed`), same seeded
team draws and action picks, same per-stage timers. The binary is whatever
`$POKESIM_SIM_BRIDGE_BIN` names — ONE binary per process.

It also records the WALL of the whole measured loop and appends one JSON row to `--out`.

    POKESIM_SIM_BRIDGE_BIN=<abs sim_bridge> python env_step_rust.py --decisions 1500 \
        --seed 0 --arm A --pair 1 --out rows.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import functools
import io
import json
import os
import re
import resource
import sys
import time


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--decisions", type=int, default=1500)
    ap.add_argument("--battle-cap", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--pair", type=int, required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    binary = os.environ.get("POKESIM_SIM_BRIDGE_BIN")
    if not binary or not os.path.isfile(binary):
        print("POKESIM_SIM_BRIDGE_BIN must name the sim_bridge under test", file=sys.stderr)
        return 2

    import agents.training.trainer_turn_benchmark as ttb
    from utils.bridge.local_battle_runner import run_local_battles

    ttb.run_local_battles = functools.partial(run_local_battles, impl="rust")

    load0 = os.getloadavg()
    buf = io.StringIO()
    # `run_local_battles` spawns (and reaps) one sim_bridge child per battle, so the children's
    # rusage delta is the Rust side's CPU for the WHOLE pinned sequence — the transport's own
    # cost, with the Python parent's CPU excluded.
    ch0 = resource.getrusage(resource.RUSAGE_CHILDREN)
    self0 = resource.getrusage(resource.RUSAGE_SELF)
    t0 = time.perf_counter()
    with contextlib.redirect_stdout(buf):
        rc = asyncio.run(ttb.main(a.decisions, a.battle_cap, 3, a.seed,
                                  use_assembler=True, reward_argv=None, pin=True))
    wall = time.perf_counter() - t0
    ch1 = resource.getrusage(resource.RUSAGE_CHILDREN)
    self1 = resource.getrusage(resource.RUSAGE_SELF)
    load1 = os.getloadavg()
    text = buf.getvalue()

    def grab(pat):
        m = re.search(pat, text)
        return float(m.group(1)) if m else None

    m = re.search(r"TRAINER-TURN PROFILE\s+\((\d+) measured decisions, (\d+) battles\)", text)
    row = {
        "arm": a.arm, "pair": a.pair, "binary": binary, "rc": rc, "wall_s": wall,
        "measured": int(m.group(1)) if m else None, "battles": int(m.group(2)) if m else None,
        "cycle_med_ms": grab(r"full decision cycle\s+:\s+([\d.]+) ms"),
        "our_cpu_med_ms": grab(r"our controllable CPU\s+:\s+([\d.]+) ms"),
        "wait_med_ms": grab(r"sim \+ opp \+ framework\s+:\s+([\d.]+) ms"),
        "child_cpu_s": (ch1.ru_utime - ch0.ru_utime) + (ch1.ru_stime - ch0.ru_stime),
        "self_cpu_s": (self1.ru_utime - self0.ru_utime) + (self1.ru_stime - self0.ru_stime),
        "load1_start": load0[0], "load1_end": load1[0],
    }
    with open(a.out, "a") as f:
        f.write(json.dumps(row) + "\n")
    print(json.dumps(row))
    if rc != 0:
        sys.stderr.write(text[-3000:])
    return rc


if __name__ == "__main__":
    sys.exit(main())
