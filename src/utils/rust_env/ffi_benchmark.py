"""The FFI front end's cost vs the core in Rust — a DESCRIPTOR, not a gate (M5 Lane A, gate ④).

    export PYTHONPATH=$PYTHONPATH:src
    python3 src/utils/rust_env/ffi_benchmark.py [--blocks 5] [--steps 200] [--rounds 2]

Release build of THIS checkout (``cargo build --release --lib`` into ``src/rust_env/target``; the
stamp is checked with ``nan_poison=False``). The shape is Lane 0's ``tests/bench_test.rs``: the
bridge corpus, N = 48, both sides' rows out, T = 1 and 8, a seeded random policy, blocks of STEPs.
Per shape it reports, over blocks (median):

* ``wall``  — µs per written row, timing ONLY the ``dispatch`` call as Python sees it;
* ``core``  — µs per row inside ``Core::dispatch`` (the ``CORE_NS_LAST`` counter);
* ``xport`` — µs per DISPATCH of wall minus core: the FFI's own cost (ctypes call, GIL release and
  re-take, the handle lock, the address copy).

Rounds are interleaved A B B A with Lane 0's in-Rust bench (``bench_test`` — its own random
policy, so the battles differ: compare µs per row, never per battle). The box's load is PRINTED
beside every figure — the rule is WARN, never stretch.
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import shutil
import subprocess
import time

import numpy as np

from utils.contention import describe_contention
from utils.paths import src_path
from utils.rust_env import ffi
from utils.rust_env import protocol as P


def load1() -> float:
    return float(open("/proc/loadavg").read().split()[0])


def corpus_teams():
    out = []
    for f in sorted(glob.glob(str(src_path("rust_sim", "tests", "vectors", "bridge_corpus", "*.txt")))):
        for line in open(f):
            if line.startswith("TEAM\t"):
                t = line.rstrip("\n").split("\t")[3]
                if t and t not in out:
                    out.append(t)
    return out


def build_release() -> None:
    cargo = shutil.which("cargo") or os.path.expanduser("~/.cargo/bin/cargo")
    crate = src_path("rust_env")
    env = dict(os.environ, CARGO_TARGET_DIR=str(crate / "target"))
    subprocess.run([cargo, "build", "--release", "--lib", "--manifest-path", str(crate / "Cargo.toml")], env=env, check=True)
    subprocess.run([cargo, "test", "--release", "--no-run", "--test", "bench_test", "--manifest-path",
                    str(crate / "Cargo.toml")], env=env, check=True, capture_output=True)


def ffi_shape(lib, teams, n, threads, blocks, steps, seed=1):
    spec = P.spec_json(n=n, threads=threads, teams=teams, names=("bench1", "bench2"), turn_limit=300, refusal_budget=64, bank_dir=None)
    rng = np.random.default_rng(seed)
    nt = len(teams)
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
        per_wall, per_core, per_x = [], [], []
        for b in range(blocks + 1):
            wall = core_ns = rows = 0
            xs = []
            for _ in range(steps):
                m = c["mask"]
                pick = np.where(m == 1, rng.random(m.shape), -1.0).argmax(-1)
                c["action"][...] = np.where(c["need"] == 1, pick, -1)
                t0 = time.perf_counter_ns()
                core.step()
                dt = time.perf_counter_ns() - t0
                k = int(c["counters"][ki])
                wall += dt
                core_ns += k
                xs.append(dt - k)
                rows += int(c["need"].sum())
                done = np.flatnonzero(c["done"])
                if len(done):
                    stage(done)
            if b > 0:  # block 0 is warm-up
                per_wall.append(wall / 1e3 / rows)
                per_core.append(core_ns / 1e3 / rows)
                per_x.append(float(np.median(xs)) / 1e3)
        assert all(v == 0 for v in core.after_freeze().values()), core.after_freeze()
    return float(np.median(per_wall)), float(np.median(per_core)), float(np.median(per_x))


def rust_bench() -> str:
    cargo = shutil.which("cargo") or os.path.expanduser("~/.cargo/bin/cargo")
    crate = src_path("rust_env")
    r = subprocess.run([cargo, "test", "--release", "--test", "bench_test", "--manifest-path", str(crate / "Cargo.toml"),
                        "--", "--ignored", "--nocapture"],
                       env=dict(os.environ, CARGO_TARGET_DIR=str(crate / "target")), capture_output=True, text=True,
                       check=True)
    return "\n".join(ln for ln in r.stderr.splitlines() if re.match(r"N = \d+, T = \d+:", ln))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--blocks", type=int, default=5)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--rounds", type=int, default=2, help="A B B A pairs (A = in-Rust, B = FFI)")
    a = ap.parse_args(argv)
    build_release()
    lib = ffi.load(ffi.default_path("release"), nan_poison=False)
    teams = corpus_teams()
    print(f"box: {describe_contention()}; load1 {load1():.1f} on {os.cpu_count()} cores")
    for r in range(a.rounds):
        order = ("rust", "ffi", "ffi", "rust")
        for which in order:
            if which == "rust":
                print(f"[round {r}] IN-RUST (bench_test), load1 {load1():.1f}:\n  " + rust_bench().replace("\n", "\n  "))
            else:
                for t in (1, 8):
                    w, k, x = ffi_shape(lib, teams, 48, t, a.blocks, a.steps, seed=r * 10 + t)
                    print(f"[round {r}] FFI N = 48, T = {t}: wall {w:.1f} µs/row, core {k:.1f} µs/row, "
                          f"xport {x:.1f} µs/dispatch, load1 {load1():.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
