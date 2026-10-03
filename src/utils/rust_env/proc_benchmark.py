"""The process front end's cost vs the FFI's — a DESCRIPTOR, not a gate (M5 Lane B, gate ⑥).

    export PYTHONPATH=$PYTHONPATH:src
    python3 src/utils/rust_env/proc_benchmark.py [--blocks 5] [--steps 200] [--rounds 3] [--out f.json]

Release builds of THIS checkout (the cdylib and ``rust_env_proc``, into ``src/rust_env/target``; both
stamps checked with ``nan_poison=False``). The shape is Lane A's ``ffi_benchmark.py``: the bridge
corpus, N = 48, both sides' rows out, a seeded random policy, blocks of STEPs. The two front ends
play the SAME battles (the same spec, the same staging + policy seed per round; the core is
deterministic), so a block of one is PAIRED with the same block of the other — and a CRC over every
obs column after every STEP proves it. Rounds are interleaved FFI PROC PROC FFI. Per shape, over
blocks (median):

* ``wall``  — µs per written row, timing ONLY the ``step()`` call as Python sees it;
* ``core``  — µs per row inside ``Core::dispatch`` (``CORE_NS_LAST``);
* ``xport`` — µs per DISPATCH of wall minus core: the front end's own cost (FFI: the ctypes call and
  GIL; process: a pipe round trip + two context switches).

``proc / ffi`` is the ratio of paired block walls, with a 95 % bootstrap CI over the pairs. The box's
load is PRINTED beside every figure — warn, never stretch.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shutil
import subprocess
import time
import zlib

import numpy as np

from utils.contention import describe_contention
from utils.paths import src_path
from utils.rust_env import ffi, proc
from utils.rust_env import protocol as P
from utils.rust_env.ffi_benchmark import corpus_teams


def load1() -> float:
    return float(open("/proc/loadavg").read().split()[0])


def build_release() -> None:
    cargo = shutil.which("cargo") or os.path.expanduser("~/.cargo/bin/cargo")
    crate = src_path("rust_env")
    env = dict(os.environ, CARGO_TARGET_DIR=str(crate / "target"))
    subprocess.run([cargo, "build", "--release", "--lib", "--bin", proc.BIN_NAME, "--manifest-path",
                    str(crate / "Cargo.toml")], env=env, check=True)


def run_shape(make, teams, n, threads, blocks, steps, seed):
    """One front end on one shape: per-block (wall µs/row, core µs/row, median xport µs/dispatch),
    and the CRC chain of every obs column after every STEP (the battles' identity)."""
    spec = P.spec_json(n=n, threads=threads, teams=teams, names=("bench1", "bench2"), turn_limit=300, refusal_budget=64, bank_dir=None)
    rng = np.random.default_rng(seed)
    nt = len(teams)
    crc = 0
    core = make(spec)
    try:
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
        out = []
        for b in range(blocks + 1):
            wall = core_ns = rows = 0
            xs = []
            for _ in range(steps):
                m = c["mask"]
                pick = np.where(m == 1, rng.random(m.shape), -1.0).argmax(-1)
                c["action"][...] = np.where(c["need"] == 1, pick, -1)
                t0 = time.perf_counter_ns()
                core.step()
                dt_ = time.perf_counter_ns() - t0
                k = int(c["counters"][ki])
                wall += dt_
                core_ns += k
                xs.append(dt_ - k)
                rows += int(c["need"].sum())
                crc = zlib.crc32(c["obs"].tobytes(), crc)
                done = np.flatnonzero(c["done"])
                if len(done):
                    stage(done)
            if b > 0:  # block 0 is warm-up
                out.append((wall / 1e3 / rows, core_ns / 1e3 / rows, float(np.median(xs)) / 1e3))
        af = core.after_freeze()
        assert all(v == 0 for v in af.values()), af
    finally:
        core.close()
    return out, crc


def bootstrap_ratio(num, den, reps=10000, seed=0):
    num, den = np.asarray(num), np.asarray(den)
    r = num / den
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(r), (reps, len(r)))
    boots = num[idx].sum(1) / den[idx].sum(1)
    return float(num.sum() / den.sum()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)), r.tolist()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--blocks", type=int, default=5)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--rounds", type=int, default=3, help="FFI PROC PROC FFI quartets")
    ap.add_argument("--threads", type=int, nargs="+", default=[8, 1])
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--out", help="write the raw result as JSON here")
    a = ap.parse_args(argv)
    build_release()
    lib = ffi.load(ffi.default_path("release"), nan_poison=False)
    binary = proc.default_path("release")
    makers = {
        "ffi": lambda spec: ffi.FfiCore(spec, lib=lib),
        "proc": lambda spec: proc.ProcCore(spec, binary=binary, nan_poison=False),
    }
    teams = corpus_teams()
    box = {"start": dt.datetime.now().astimezone().isoformat(timespec="seconds"), "contention": describe_contention(),
           "cpus": os.cpu_count(), "load1": [load1()]}
    print(f"box: {box['contention']}; load1 {load1():.1f} on {os.cpu_count()} cores")
    res = {"box": box, "n": a.n, "blocks": a.blocks, "steps": a.steps, "shapes": {}}
    for t in a.threads:
        per = {"ffi": [], "proc": []}
        for r in range(a.rounds):
            crcs = {}
            for which in ("ffi", "proc", "proc", "ffi"):
                blocks, crc = run_shape(makers[which], teams, a.n, t, a.blocks, a.steps, seed=1000 + r)
                crcs.setdefault(which, set()).add(crc)
                per[which].append(blocks)
                box["load1"].append(load1())
                w = np.median([x[0] for x in blocks])
                print(f"[T = {t}, round {r}] {which:4s}: wall {w:.2f} µs/row, core "
                      f"{np.median([x[1] for x in blocks]):.2f} µs/row, xport "
                      f"{np.median([x[2] for x in blocks]):.1f} µs/dispatch, load1 {load1():.1f}")
            assert len(crcs["ffi"] | crcs["proc"]) == 1, f"the two front ends played DIFFERENT battles: {crcs}"
        # Pair block k of the i-th run of each front end (the same battles).
        pw = [x[0] for run in per["proc"] for x in run]
        fw = [x[0] for run in per["ffi"] for x in run]
        ratio, lo, hi, _ = bootstrap_ratio(pw, fw)
        shape = {
            "ffi": {k: float(np.median([x[i] for run in per["ffi"] for x in run])) for i, k in enumerate(("wall", "core", "xport"))},
            "proc": {k: float(np.median([x[i] for run in per["proc"] for x in run])) for i, k in enumerate(("wall", "core", "xport"))},
            "proc_over_ffi_wall": [ratio, lo, hi], "pairs": len(pw),
        }
        res["shapes"][f"T{t}"] = shape
        print(f"N = {a.n}, T = {t}: FFI wall {shape['ffi']['wall']:.2f} / PROC {shape['proc']['wall']:.2f} µs/row; "
              f"xport FFI {shape['ffi']['xport']:.1f} / PROC {shape['proc']['xport']:.1f} µs/dispatch; "
              f"proc/FFI {ratio:.3f} [{lo:.3f}, {hi:.3f}] over {len(pw)} paired blocks")
    box["end"] = dt.datetime.now().astimezone().isoformat(timespec="seconds")
    print(f"load1 over the run: {min(box['load1']):.1f}–{max(box['load1']):.1f}")
    if a.out:
        with open(a.out, "w") as f:
            json.dump(res, f, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
