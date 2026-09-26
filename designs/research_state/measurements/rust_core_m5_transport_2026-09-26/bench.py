"""The M5 transport benchmark — the two front ends over the ONE env core, per CONSUMER SHAPE.

    nice -n 10 python3 bench.py <shape> [--pairs P] [--steps S] [--out results/<shape>.json]

shapes:
  train    N = 48, T = 8, opponent rows to the caller (the T2 shape) — the rollout
  trainbot N = 48, T = 8, opponent random INSIDE Rust (a scripted-bot shape)
  eval     N = 16, T = 4, opponent rows to the caller
  threads  N = 48, the FFI front end at T = 1, 2, 4, 8 (the core's scaling; a descriptor)
  search   successors(k = 64) from a live root, n = 1 (the fork burst)
  oneoff   start-up to the first row (fresh interpreter per rep) + N = 1 per-decision latency
  gil      a Python thread's progress while the main thread steps N = 48 (GIL release)

Method (the repo's benchmark rule: WARN, never stretch): both front ends are built ONCE and stay
alive; blocks alternate A B / B A (interleaved pairs), so slow drift of the shared box's load hits
both arms alike; the per-pair ratio of block medians is bootstrapped (95% CI, 10,000 resamples).
Every block records the 1-min load average; the contention factor (`utils.contention`) is read
before and after and printed with a WARNING above 1.25. The timed quantity is the `step()` call
alone (transport + core); the random policy's own NumPy cost is reported separately.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str((HERE / "../../../../src").resolve()))
import m5_loader as m5  # noqa: E402

try:
    from utils.contention import cpu_contention_factor  # noqa: E402
except Exception:  # pragma: no cover
    cpu_contention_factor = None


def contention() -> float | None:
    if cpu_contention_factor is None:
        return None
    try:
        return float(cpu_contention_factor(refresh=True))
    except Exception:
        return None


def boot_ci(x: np.ndarray, stat=np.median, reps: int = 10000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(reps, len(x)))
    s = np.array([stat(x[i]) for i in idx])
    return float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))


def run_block(pool, rng, steps: int) -> dict:
    """`steps` batches; per batch: the step() wall, the core's own wall, the decisions it answered."""
    wall, core, dec, pol = [], [], [], []
    for _ in range(steps):
        t0 = time.perf_counter_ns()
        m5.random_actions(pool, rng)
        t1 = time.perf_counter_ns()
        d = int((pool.need == 1).sum())
        pool.step()
        t2 = time.perf_counter_ns()
        wall.append(t2 - t1)
        core.append(pool.core_ns)
        dec.append(d)
        pol.append(t1 - t0)
    wall, core, dec = np.array(wall, float), np.array(core, float), np.array(dec, float)
    return {
        "batches": steps,
        "decisions": int(dec.sum()),
        "wall_per_batch_us": float(np.median(wall) / 1e3),
        "core_per_batch_us": float(np.median(core) / 1e3),
        "transport_per_batch_us": float(np.median(wall - core) / 1e3),
        "us_per_decision": float(wall.sum() / dec.sum() / 1e3),
        "decisions_per_s": float(dec.sum() / (wall.sum() / 1e9)),
        "policy_numpy_us_per_batch": float(np.median(pol) / 1e3),
        "load1": os.getloadavg()[0],
        "refusals_total": pool.refusals,
    }


def paired(n: int, threads: int, opp_external: bool, pairs: int, steps: int, warm: int = 100) -> dict:
    pools = {k: m5.make(k, n, threads, 1234, opp_external) for k in ("ffi", "proc")}
    rngs = {k: np.random.default_rng(99) for k in pools}
    for k, p in pools.items():
        p.reset()
        run_block(p, rngs[k], warm)
    c0 = contention()
    blocks = []
    try:
        for i in range(pairs):
            order = ("ffi", "proc") if i % 2 == 0 else ("proc", "ffi")
            row = {}
            for k in order:
                row[k] = run_block(pools[k], rngs[k], steps)
            blocks.append(row)
            ref = row['ffi']['refusals_total']
            print(f"  [refusals so far {ref}]" if ref else "", end="")
            print(f"  pair {i}: ffi {row['ffi']['us_per_decision']:.1f} us/dec  proc {row['proc']['us_per_decision']:.1f} "
                  f"us/dec  (transport ffi {row['ffi']['transport_per_batch_us']:.1f} / proc "
                  f"{row['proc']['transport_per_batch_us']:.1f} us/batch)  load1 {row['ffi']['load1']:.1f}", flush=True)
    finally:
        for p in pools.values():
            p.close()
    c1 = contention()
    out = {"n": n, "threads": threads, "opp_external": opp_external, "pairs": pairs, "steps_per_block": steps,
           "contention_before": c0, "contention_after": c1, "blocks": blocks}
    for metric in ("us_per_decision", "wall_per_batch_us", "transport_per_batch_us", "core_per_batch_us", "decisions_per_s"):
        for k in ("ffi", "proc"):
            x = np.array([b[k][metric] for b in blocks])
            lo, hi = boot_ci(x)
            out[f"{k}_{metric}"] = {"median": float(np.median(x)), "ci95": [lo, hi]}
        r = np.array([b["proc"][metric] / b["ffi"][metric] for b in blocks])
        lo, hi = boot_ci(r)
        out[f"ratio_proc_over_ffi_{metric}"] = {"median": float(np.median(r)), "ci95": [lo, hi]}
    return out


def shape_threads(steps: int, pairs: int) -> dict:
    res = {}
    for t in (1, 2, 4, 8):
        p = m5.make("ffi", 48, t, 1234, True)
        rng = np.random.default_rng(99)
        p.reset()
        run_block(p, rng, 50)
        xs = [run_block(p, rng, steps) for _ in range(pairs)]
        p.close()
        x = np.array([b["us_per_decision"] for b in xs])
        res[str(t)] = {"us_per_decision_median": float(np.median(x)), "ci95": list(boot_ci(x)),
                       "decisions_per_s_median": float(np.median([b["decisions_per_s"] for b in xs])),
                       "load1": [b["load1"] for b in xs]}
        print(f"  T={t}: {res[str(t)]['us_per_decision_median']:.1f} us/decision, "
              f"{res[str(t)]['decisions_per_s_median']:.0f} decisions/s", flush=True)
    return res


def shape_search(pairs: int, k: int = 64, roots: int = 40) -> dict:
    pools = {kind: m5.make(kind, 1, 1, 4321, True) for kind in ("ffi", "proc")}
    rngs = {kind: np.random.default_rng(5) for kind in pools}
    per = {"ffi": [], "proc": []}
    core = {"ffi": [], "proc": []}
    try:
        for p in pools.values():
            p.reset()
        for i in range(roots * pairs):
            if pools["ffi"].need[0, 0]:
                order = ("ffi", "proc") if i % 2 == 0 else ("proc", "ffi")
                for kind in order:
                    t0 = time.perf_counter_ns()
                    pools[kind].successors(k)
                    per[kind].append(time.perf_counter_ns() - t0)
                    core[kind].append(pools[kind].core_ns)
            for kind, p in pools.items():
                m5.random_actions(p, rngs[kind])
                p.step()
    finally:
        for p in pools.values():
            p.close()
    f, pr = np.array(per["ffi"], float), np.array(per["proc"], float)
    r = pr / f
    return {"k": k, "calls": len(f),
            "ffi_us_per_call": {"median": float(np.median(f) / 1e3), "ci95": [v / 1e3 for v in boot_ci(f)]},
            "proc_us_per_call": {"median": float(np.median(pr) / 1e3), "ci95": [v / 1e3 for v in boot_ci(pr)]},
            "ffi_us_per_successor": float(np.median(f) / 1e3 / k),
            "transport_us_per_call": {"ffi": float(np.median(f - np.array(core['ffi'])) / 1e3),
                                      "proc": float(np.median(pr - np.array(core['proc'])) / 1e3)},
            "ratio_proc_over_ffi": {"median": float(np.median(r)), "ci95": list(boot_ci(r))},
            "load1": os.getloadavg()[0]}


STARTUP = r"""
import sys, time
t0 = time.perf_counter()
sys.path.insert(0, {here!r})
import numpy as np
import m5_loader as m5
t1 = time.perf_counter()
p = m5.make({kind!r}, 1, 1, 77, True)
p.reset()
t2 = time.perf_counter()
p.close()
print(t1 - t0, t2 - t1)
"""


def shape_oneoff(pairs: int, steps: int) -> dict:
    exe = sys.executable
    st = {"ffi": [], "proc": []}
    for i in range(pairs):
        for kind in (("ffi", "proc") if i % 2 == 0 else ("proc", "ffi")):
            out = subprocess.run([exe, "-c", STARTUP.format(here=str(HERE), kind=kind)], capture_output=True, text=True, check=True)
            imp, first = map(float, out.stdout.split())
            st[kind].append(first)
    t0 = time.perf_counter()
    m5.source_hash()
    stamp_ms = (time.perf_counter() - t0) * 1e3
    lat = paired(1, 1, True, pairs, steps, warm=50)
    return {"startup_to_first_row_ms": {k: {"median": float(np.median(v) * 1e3), "ci95": [x * 1e3 for x in boot_ci(np.array(v))]}
                                        for k, v in st.items()},
            "stamp_check_ms": stamp_ms, "n1_latency": lat}


def shape_gil(seconds: float = 3.0) -> dict:
    res = {}
    for kind in ("none", "ffi", "proc"):
        count = [0]
        stop = threading.Event()

        def spin():
            while not stop.is_set():
                count[0] += 1

        p = None if kind == "none" else m5.make(kind, 48, 8, 1234, True)
        if p:
            p.reset()
        rng = np.random.default_rng(1)
        th = threading.Thread(target=spin)
        th.start()
        t0 = time.perf_counter()
        steps = 0
        while time.perf_counter() - t0 < seconds:
            if p:
                m5.random_actions(p, rng)
                p.step()
            else:
                time.sleep(0.001)
            steps += 1
        stop.set()
        th.join()
        if p:
            p.close()
        res[kind] = {"spin_iters_per_s": count[0] / seconds, "main_steps": steps}
        print(f"  {kind}: other-thread {count[0] / seconds:,.0f} it/s while the main thread did {steps} steps", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("shape")
    ap.add_argument("--pairs", type=int, default=8)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--out")
    a = ap.parse_args()
    c = contention()
    print(f"[{a.shape}] load1={os.getloadavg()[0]:.2f} contention={c} nice={os.nice(0)}", flush=True)
    if c is not None and c > 1.25:
        print(f"WARNING: the box is CONTENDED (factor {c:.2f}); the numbers are measured as-is, never stretched", flush=True)
    if a.shape == "train":
        res = paired(48, 8, True, a.pairs, a.steps)
    elif a.shape == "trainbot":
        res = paired(48, 8, False, a.pairs, a.steps)
    elif a.shape == "eval":
        res = paired(16, 4, True, a.pairs, a.steps)
    elif a.shape == "threads":
        res = shape_threads(a.steps, a.pairs)
    elif a.shape == "search":
        res = shape_search(a.pairs)
    elif a.shape == "oneoff":
        res = shape_oneoff(a.pairs, a.steps)
    elif a.shape == "gil":
        res = shape_gil()
    else:
        raise SystemExit(f"unknown shape {a.shape}")
    stamp = m5.load_ffi().m5_stamp().decode()
    rec = {"shape": a.shape, "stamp": stamp, "argv": sys.argv, "when": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
           "load1_start": os.getloadavg()[0], "contention_start": c, "contention_end": contention(),
           "cpu_count": os.cpu_count(), "result": res}
    out = Path(a.out or HERE / "results" / f"{a.shape}.json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(rec, indent=1))
    keys = [k for k in res if k.startswith("ratio_") or k.endswith("us_per_decision")] if isinstance(res, dict) else []
    for k in keys:
        print(f"  {k}: {res[k]}")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
