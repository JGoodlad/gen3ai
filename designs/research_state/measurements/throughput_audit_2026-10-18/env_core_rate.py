"""The Rust env core's rate PER CORE: rows/s, battles/s and rows per battle (a DESCRIPTOR, not a gate).

    export PYTHONPATH=$PWD/src
    CUDA_VISIBLE_DEVICES="" nice -n 19 taskset -c 5-7,13-15 scripts/ops/mem_cap.sh 8 \
        python designs/research_state/measurements/throughput_audit_2026-10-18/env_core_rate.py \
        --threads 1 --n 48 --blocks 6 --steps 300 --out <file.json>

The release cdylib of THIS checkout (build it first: `CARGO_TARGET_DIR=$PWD/src/rust_env/target cargo build
--release --lib --manifest-path src/rust_env/Cargo.toml`). The shape is `utils.rust_env.ffi_benchmark`'s (the bridge
corpus teams, BOTH sides' rows out, a seeded uniform-random legal policy, blocks of STEPs, block 0 = warm-up), plus
what the throughput audit needs and that benchmark does not report:

* ``battles_per_s``  — finished battles (``done`` flags) per second of ``Core::dispatch`` time; at T = 1 this is the
  per-core battle rate other projects quote;
* ``rows_per_battle`` — decision rows written per finished battle (BOTH sides; a random policy plays shorter or
  longer games than a trained one, so convert with a trained run's own episode length, never this one's);
* ``core_us_per_row`` / ``wall_us_per_row`` — inside ``Core::dispatch`` vs as Python sees the call (the gap is the
  FFI's transport); ``host_us_per_row`` — the Python side of THIS loop (random actions, staging), for scale only.

Every row is the full production work per decision: the Showdown-port step, its protocol emission, both sides'
parse chains and the 2,845-float observation encode. The box's load is PRINTED and recorded — WARN, never stretch.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from utils.contention import describe_contention
from utils.rust_env import ffi
from utils.rust_env import protocol as P
from utils.rust_env.ffi_benchmark import corpus_teams


def load1() -> float:
    return float(open("/proc/loadavg").read().split()[0])


def run(lib, teams, n: int, threads: int, blocks: int, steps: int, seed: int) -> dict:
    spec = P.spec_json(n=n, threads=threads, teams=teams, names=("bench1", "bench2"), turn_limit=300,
                       refusal_budget=64, bank_dir=None)
    rng = np.random.default_rng(seed)
    nt = len(teams)
    out = []
    with ffi.FfiCore(spec, lib=lib) as core:
        c = core.cols
        ki = P.counter_index()["CORE_NS_LAST"]

        def stage(idx):
            for i in idx:
                a = int(rng.integers(nt))
                b = int(rng.integers(nt - 1))
                c["ep_team"][i] = [a, b + (b >= a)]
                c["ep_seed"][i] = rng.integers(0, 65536, 4)

        stage(range(n))
        core.reset()
        for b in range(blocks + 1):
            wall = core_ns = rows = done_n = 0
            t_block = time.perf_counter_ns()
            for _ in range(steps):
                m = c["mask"]
                pick = np.where(m == 1, rng.random(m.shape), -1.0).argmax(-1)
                c["action"][...] = np.where(c["need"] == 1, pick, -1)
                t0 = time.perf_counter_ns()
                core.step()
                wall += time.perf_counter_ns() - t0
                core_ns += int(c["counters"][ki])
                rows += int(c["need"].sum())
                done = np.flatnonzero(c["done"])
                done_n += len(done)
                if len(done):
                    stage(done)
            block_ns = time.perf_counter_ns() - t_block
            if b > 0:   # block 0 is warm-up
                out.append({"rows": rows, "battles": done_n, "core_s": core_ns / 1e9, "wall_s": wall / 1e9,
                            "block_s": block_ns / 1e9, "load1": load1()})
        assert all(v == 0 for v in core.after_freeze().values()), core.after_freeze()

    def med(f):
        return float(np.median([f(r) for r in out]))
    rows = sum(r["rows"] for r in out)
    battles = sum(r["battles"] for r in out)
    return {"n": n, "threads": threads, "blocks": out,
            "core_us_per_row": med(lambda r: 1e6 * r["core_s"] / r["rows"]),
            "wall_us_per_row": med(lambda r: 1e6 * r["wall_s"] / r["rows"]),
            "host_us_per_row": med(lambda r: 1e6 * (r["block_s"] - r["wall_s"]) / r["rows"]),
            "rows_per_s_core": med(lambda r: r["rows"] / r["core_s"]),
            "battles_per_s_core": battles / sum(r["core_s"] for r in out),
            "rows_per_battle": rows / max(1, battles), "battles": battles, "rows": rows}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--threads", type=int, nargs="+", default=[1])
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--blocks", type=int, default=6)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)
    lib = ffi.load(ffi.default_path("release"), nan_poison=False)
    teams = corpus_teams()
    res = {"box": describe_contention(), "load1_start": load1(), "cpu_affinity": sorted(os.sched_getaffinity(0)),
           "teams": len(teams), "shapes": []}
    for t in a.threads:
        r = run(lib, teams, a.n, t, a.blocks, a.steps, a.seed)
        res["shapes"].append(r)
        print(f"N = {a.n}, T = {t}: core {r['core_us_per_row']:.1f} µs/row ({r['rows_per_s_core']:,.0f} rows/s), "
              f"wall {r['wall_us_per_row']:.1f} µs/row, {r['battles_per_s_core']:,.1f} battles/s of core time, "
              f"{r['rows_per_battle']:.1f} rows/battle (random policy), load1 {load1():.1f}", flush=True)
    res["load1_end"] = load1()
    if a.out:
        with open(a.out, "w") as f:
            json.dump(res, f, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
