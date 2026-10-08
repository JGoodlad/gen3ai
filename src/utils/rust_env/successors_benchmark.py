"""Search on ``successors()`` in process vs the JSON road — a DESCRIPTOR, not a gate (M5 Lane I, unit 4).

    export PYTHONPATH=$PYTHONPATH:src
    python3 src/utils/rust_env/successors_benchmark.py [--battles 4] [--reps 4]

RELEASE builds of THIS checkout on both roads (the ``search_driver`` binary through
``SearchSession(impl="rust")``, the env core's cdylib through :class:`successors.Successors`), one
thread each, the same requests:

* **successors/s** — search's own shape: ``open_root(core="text", side, trackers)`` then ONE
  ``expand_many(rows=True, side)`` of every legal action x 3 opponent choices x 2 shared dice seeds
  (what ``SearchEngine._expand_ply`` sends at depth 1). Per root, the two roads run in alternating
  order (A B, then B A) ``--reps`` times; the ratio in-process / JSON per root is bootstrapped (95 %).
  The JSON road's wall includes what search pays for it: the request dump, the pipe, the reply parse
  and the base64 row decode (``wrap_row``); the in-process road's includes the arms' JSON (small),
  the rows landing in NumPy directly.
* **playouts/s** — Lane S's shape: every legal action of the side to move x ``--seeds`` dice seeds
  from a mid-battle decision, each played to the end under a batched policy (a cheap deterministic
  scorer over the rows — a stand-in for a forward, so the number is the CORE's cost). There is no
  JSON-road equivalent (the driver has no playout verb: a JSON playout is one ``expand_many`` round
  trip per turn per branch), so the descriptor reports it alone, plus decisions/s.

The box's load is PRINTED beside every figure — WARN, never stretch.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import time
from typing import List

import numpy as np

from utils.contention import describe_contention
from utils.paths import src_path
from utils.rust_env import ffi
from utils.rust_env import successors as S
from utils.rust_env import successors_parity as SP


def load1() -> float:
    return float(open("/proc/loadavg").read().split()[0])


def _build_release() -> None:
    cargo = shutil.which("cargo") or os.path.expanduser("~/.cargo/bin/cargo")
    crate = src_path("rust_env")
    subprocess.run([cargo, "build", "--release", "--lib", "--manifest-path", str(crate / "Cargo.toml")],
                   env=dict(os.environ, CARGO_TARGET_DIR=str(crate / "target")), check=True)


def _arms(root, side: str, rng) -> List[dict]:
    other = "p2" if side == "p1" else "p1"
    ours = [c["token"] for c in SP.legal_choices_from_request(root.requests[side])]
    opp = [c["token"] for c in SP.legal_choices_from_request(root.requests[other])][:3] or ["random"]
    seeds = [f"sodium,{int(x):032x}" for x in rng.integers(0, 2**63, 2)]
    return [{"node_id": root.node_id, f"{side}_action": a, f"{other}_action": o, "seed": s, "label": i}
            for i, (a, o, s) in enumerate([(a, o, s) for a in ours for o in opp for s in seeds])]


def bench_successors(logs, reps: int, lib) -> dict:
    from agents.battle.core_obs import wrap_row
    from utils.bridge.search_session import SearchSession

    per_root = []  # (json_us_per_succ, inproc_us_per_succ)
    rng = np.random.default_rng(0)
    n_succ = 0
    with SearchSession(impl="rust", timeout=600) as js, S.Successors(lib=lib, chunks=False) as ip:
        for bi, log in enumerate(logs):
            rec = S.log_to_record(log, battle_tag=f"bench{bi}")
            last = SP._last_turn(rec)
            for turn in (2, max(3, last // 3), max(4, (2 * last) // 3)):
                for side in ("p1", "p2"):
                    try:
                        root = ip.open_root(turn, record=rec, side=side)
                    except Exception:  # noqa: BLE001 — past the end
                        continue
                    arms = _arms(root, side, rng)
                    if not arms:
                        continue

                    def run_json():
                        r = js.open_root(turn, record=rec, core="text", side=side, trackers=True)
                        a = [dict(x, node_id=r.node_id) for x in arms]
                        out = js.expand_many(a, side=side, rows=True)
                        for e in out:
                            c = e.core_p1 if side == "p1" else e.core_p2
                            if c and c.get("row") is not None:
                                wrap_row(c["row"])
                        return len(out)

                    def run_ip():
                        r = ip.open_root(turn, record=rec, side=side)
                        a = [dict(x, node_id=r.node_id) for x in arms]
                        return len(ip.expand_many(a, side=side))

                    tj, ti = [], []
                    for k in range(reps):
                        order = (run_json, run_ip) if k % 2 == 0 else (run_ip, run_json)
                        for f in order:
                            t0 = time.perf_counter()
                            n = f()
                            dt = time.perf_counter() - t0
                            (tj if f is run_json else ti).append(dt / n * 1e6)
                    per_root.append((float(np.median(tj)), float(np.median(ti))))
                    n_succ += len(arms)
    a = np.array(per_root)
    ratio = a[:, 1] / a[:, 0]
    boot = np.random.default_rng(1)
    bs = [np.median(ratio[boot.integers(0, len(ratio), len(ratio))]) for _ in range(2000)]
    return {"roots": len(per_root), "successors_per_rep": n_succ,
            "json_us": float(np.median(a[:, 0])), "inproc_us": float(np.median(a[:, 1])),
            "ratio": float(np.median(ratio)), "ci": (float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)))}


def bench_playouts(logs, seeds: int, lib) -> dict:
    w = np.random.default_rng(3).standard_normal((1, 2761)).astype(np.float32)

    def scorer(rows, masks):  # a stand-in forward: one projection + a fixed per-action bias
        return (rows @ w.T) + np.arange(11)[None, :] * 1e-3

    pol = S.greedy(scorer)
    tot = {"branches": 0, "decisions": 0, "wall": 0.0, "batches": 0}
    with S.SearchCore(lib=lib) as core:
        for log in logs:
            at = next((k for k in range(len(log["cmds"]) // 3, len(log["cmds"])) if log["cmds"][k].startswith("CHOOSE p1")), None)
            if at is None:
                continue
            r = S.play_out(log, at, "p1", policy=pol, seeds=[f"{i + 1},7,7,7" for i in range(seeds)], core=core)
            tot["branches"] += len(r.branches)
            tot["decisions"] += r.answered
            tot["wall"] += r.wall_s
            tot["batches"] += r.batches
    return {**tot, "playouts_per_s": tot["branches"] / tot["wall"], "decisions_per_s": tot["decisions"] / tot["wall"],
            "us_per_decision": tot["wall"] / tot["decisions"] * 1e6}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--battles", type=int, default=4)
    ap.add_argument("--reps", type=int, default=4)
    ap.add_argument("--seeds", type=int, default=4)
    a = ap.parse_args()
    _build_release()
    os.environ["POKESIM_EMISSION_SELFCHECK"] = "0"  # both roads on their RELEASE builds
    lib = ffi.load(ffi.default_path("release"), nan_poison=False)
    print(f"[successors_benchmark] {describe_contention()}  load1={load1():.1f}")
    core = S.SearchCore(lib=lib)
    logs = SP.make_logs(SP._teams("ladder_milestone"), a.battles, 23, core=core)
    core.close()
    t = time.time()
    s = bench_successors(logs, a.reps, lib)
    print(f"successors: {s['roots']} roots, {s['successors_per_rep']} successors/rep  "
          f"JSON {s['json_us']:.0f} us/successor ({1e6 / s['json_us']:.0f}/s)  "
          f"in-process {s['inproc_us']:.0f} us/successor ({1e6 / s['inproc_us']:.0f}/s)  "
          f"ratio {s['ratio']:.3f} [{s['ci'][0]:.3f}, {s['ci'][1]:.3f}]  ({time.time() - t:.0f}s, load1={load1():.1f})")
    t = time.time()
    p = bench_playouts(logs, a.seeds, lib)
    print(f"playouts: {p['branches']} branches, {p['decisions']} policy decisions in {p['batches']} batches, "
          f"{p['playouts_per_s']:.1f} playouts/s, {p['decisions_per_s']:.0f} decisions/s "
          f"({p['us_per_decision']:.0f} us/decision incl. the stand-in scorer)  ({time.time() - t:.0f}s, load1={load1():.1f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
