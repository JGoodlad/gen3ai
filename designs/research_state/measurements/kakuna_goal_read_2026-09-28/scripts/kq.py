#!/usr/bin/env python3
"""X22 (b)-(f) — the Kakuna goal read's CPU work queue. Incremental, resumable, detached.

    kq.py run [--stop-at ISO8601]    # supervisor: 2 lanes, one unit per lane at a time
    kq.py status                     # per-cell pooled W/L/T + Wilson, units done/pending

A UNIT is one `python -m main.anchors` invocation of a few minutes, with its own --out, its own
--team-seed == --seed-base, and a fixed lane port. A unit is DONE iff its summary.json says
status OK with n == its games; a partial unit dir is moved aside to <dir>.partial.<epoch> and
re-run. Every start/end is one fsynced line in units.jsonl (rc, wall, load at start and end).

Seeds are PAIRED across Kakuna's temperatures: greedy / T=0.5 / T=1.0 chunk i share seed i, so the
three cells see the same team draws (and the top-up reuses greedy's seeds 5-9).

--stop-at: no unit STARTS unless its estimated wall ends before the stop; at stop+120 s a unit
still running is killed by its own process group (the child is started with start_new_session,
so its pgid is the PID this supervisor holds). Nothing is left running at the stop.
"""
import datetime as dt
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

STATE = Path(os.environ.get("KQ_STATE", "/home/goodlad/dev/gen3ai-reads/kakuna_goal_read_2026-09-28"))
TREE = Path(__file__).resolve().parents[5]
PY = "/home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3"
MODEL = "/home/goodlad/dev/gen3ai/models/ai_v14_01_base/final_model.zip"  # step 75,005,952
CONFIG = STATE / "configs" / "anchors_home72_kakuna.json"
# Rust binaries: the N0 read's pinned tree (dcddac0b); src/rust_sim + data/ are identical to HEAD.
REL = Path("/home/goodlad/dev/gen3ai-wt/pin-dcddac0b-n0reads/src/rust_sim/target/release")
PORTS = (9561, 9562)
FP_MS = 1000


def units():
    """Priority order. (name, cell, games, argv-extra, est_s)."""
    out = []
    common = ["--model", MODEL, "--opponent", "metamon:Kakuna", "--regime", "greedy"]
    cells = {"b_greedy": [], "b_t05": ["--opponent-temperature", "0.5"],
             "b_t10": ["--opponent-temperature", "1.0"]}
    for i in range(5):                       # 100 each, interleaved, paired seeds
        for c, extra in cells.items():
            out.append((f"{c}_{i:02d}", c, 20, i, common + extra, 330))
    for i in range(5, 10):                   # greedy to 200
        out.append((f"b_greedy_{i:02d}", "b_greedy", 20, i, common, 330))
    out.append(("TOPUP", None, None, None, None, None))   # resolved at run time
    for i in range(4):                       # (d) Kakuna vs SyntheticRLV2, greedy, home72
        out.append((f"d_{i:02d}", "d_kakuna_vs_synthv2", 50, 100 + i,
                    ["--opponent-a", "metamon:Kakuna", "--opponent-b", "metamon:SyntheticRLV2",
                     "--regime", "greedy", "--progress-timeout", "3600"], 900))
    for i in range(20):                      # (e) Kakuna (greedy) vs Foul Play @1000 ms
        out.append((f"e_{i:02d}", "e_kakuna_vs_fp1000", 10, 200 + i,
                    ["--opponent-a", "metamon:Kakuna", "--opponent-b", "foulplay",
                     "--regime", "greedy", "--search-time-ms", str(FP_MS),
                     "--search-parallelism", "1", "--progress-timeout", "3600"], 1100))
    # (f) owner 2026-09-28: does N0 play better greedy? N0 SAMPLING at T=1.0 (`--our-temperature`).
    # Seeds PAIRED with the banked cells: f1 with (c) b_t10 seeds 0-9, f2 with (b) b_greedy 0-4.
    # Our sampling generator is seeded per unit (GEN3AI_POLICY_SEED = unit seed), see run_unit.
    for i in range(10):                      # f1: N0 T=1.0 vs Kakuna T=1.0, 200 games
        out.append((f"f1_{i:02d}", "f1_n0t1_vs_kakuna_t1", 20, i,
                    common + ["--our-temperature", "1.0", "--opponent-temperature", "1.0"], 400))
    for i in range(5):                       # f2: N0 T=1.0 vs Kakuna greedy, 100 games
        out.append((f"f2_{i:02d}", "f2_n0t1_vs_kakuna_greedy", 20, i,
                    common + ["--our-temperature", "1.0"], 400))
    return out


def jl_append(path, obj):
    with open(path, "a") as fh:
        fh.write(json.dumps(obj) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def jl(path):
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def summary(name):
    p = STATE / "cells" / name / "summary.json"
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def is_done(name, games):
    s = summary(name)
    return bool(s and s.get("status") == "OK" and s.get("n") == games)


def wilson(w, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = w / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def pooled(cell):
    w = l = t = n = 0
    for (name, c, games, *_rest) in all_units():
        if c == cell and is_done(name, games):
            s = summary(name)
            w += s["wins"]; l += s["losses"]; t += s["ties"]; n += s["n"]
    return w, l, t, n


def topup_units():
    """Top the STRONGEST Kakuna temperature (lowest N0 win rate) up to 200 — decided once, from
    the 100-game reads, and recorded in topup.json so a resume never re-decides it."""
    f = STATE / "topup.json"
    if f.exists():
        cell = json.loads(f.read_text())["cell"]
    else:
        return None
    if cell == "b_greedy":
        return []
    extra = {"b_t05": ["--opponent-temperature", "0.5"],
             "b_t10": ["--opponent-temperature", "1.0"]}[cell]
    common = ["--model", MODEL, "--opponent", "metamon:Kakuna", "--regime", "greedy"]
    return [(f"{cell}_{i:02d}", cell, 20, i, common + extra, 330) for i in range(5, 10)]


def all_units():
    out = []
    for u in units():
        if u[0] == "TOPUP":
            out += topup_units() or []
        else:
            out.append(u)
    return out


def decide_topup():
    f = STATE / "topup.json"
    if f.exists():
        return True
    rates = {}
    for c in ("b_greedy", "b_t05", "b_t10"):
        done = [u for u in units() if u[1] == c and int(u[0][-2:]) < 5]
        if not all(is_done(u[0], u[2]) for u in done):
            return False
        w = l = t = n = 0
        for u in done:
            s = summary(u[0]); w += s["wins"]; n += s["n"]
        rates[c] = w / n
    cell = min(rates, key=lambda c: rates[c])
    f.write_text(json.dumps({"cell": cell, "n0_win_rates_at_100": rates,
                             "rule": "lowest N0 win rate over paired seeds 0-4 (100 games each)",
                             "decided_at": dt.datetime.now().isoformat()}, indent=1))
    return True


def child_env():
    env = dict(os.environ)
    env.update({
        "PYTHONPATH": str(TREE / "src"),
        "GEN3AI_ANCHORS_CONFIG": str(CONFIG),
        "POKESIM_SIM_BRIDGE_BIN": str(REL / "sim_bridge"),
        "POKESIM_SEARCH_DRIVER_BIN": str(REL / "search_driver"),
        "POKESIM_CORE_EVENTS_BIN": str(REL / "core_events"),
        "CUDA_VISIBLE_DEVICES": "",
        "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1", "PYTHONUNBUFFERED": "1",
    })
    return env


LOCK = threading.Lock()
CLAIMED = set()


def est_wall(cell, default):
    walls = [e["wall_s"] for e in jl(STATE / "units.jsonl")
             if e.get("event") == "end" and e.get("cell") == cell and e.get("done")]
    return (sum(walls) / len(walls)) * 1.15 if walls else default


def next_unit(stop_at):
    with LOCK:
        if (STATE / "STOP").exists():
            return None
        for u in units():
            if u[0] == "TOPUP":
                if not decide_topup():
                    # the top-up is not decidable yet; later units may still run
                    continue
                cands = topup_units()
            else:
                cands = [u]
            for (name, cell, games, seed, extra, est) in cands:
                if name in CLAIMED or is_done(name, games):
                    continue
                if stop_at and time.time() + est_wall(cell, est) > stop_at:
                    continue
                CLAIMED.add(name)
                return (name, cell, games, seed, extra)
        return None


def run_unit(u, port, lane, stop_at):
    name, cell, games, seed, extra = u
    out = STATE / "cells" / name
    if out.exists():
        out.rename(out.with_name(f"{name}.partial.{int(time.time())}"))
    argv = [PY, "-m", "main.anchors", *extra, "--teamset", "home", "--games", str(games),
            "--team-seed", str(seed), "--seed-base", str(seed), "--device", "cpu",
            "--port", str(port), "--nice", "19", "--out", str(out)]
    t0 = time.time()
    meta = {"unit": name, "cell": cell, "lane": lane, "port": port, "games": games,
            "seed": seed, "argv": argv, "load_start": os.getloadavg(), "t": t0,
            "fp_search_time_ms": FP_MS if cell.startswith("e_") else None}
    jl_append(STATE / "units.jsonl", {"event": "start", **meta})
    log = open(STATE / "logs" / f"{name}.log", "ab")
    env = child_env()
    if cell.startswith("f"):
        env["GEN3AI_POLICY_SEED"] = str(seed)   # our T=1 sampling, reproducible per unit
    meta_seed = env.get("GEN3AI_POLICY_SEED")
    jl_append(STATE / "units.jsonl", {"event": "policy_seed", "unit": name, "seed": meta_seed})
    proc = subprocess.Popen(argv, cwd=str(TREE), env=env, stdout=log,
                            stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                            start_new_session=True)
    jl_append(STATE / "units.jsonl", {"event": "pid", "unit": name, "pid": proc.pid})
    loads = []
    killed = False
    while proc.poll() is None:
        time.sleep(5)
        loads.append(os.getloadavg()[0])
        if stop_at and time.time() > stop_at + 120 and not killed:
            os.killpg(proc.pid, signal.SIGTERM)       # this unit's own group, by its PID
            killed = True
            try:
                proc.wait(30)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
    rc = proc.wait()
    wall = time.time() - t0
    done = is_done(name, games)
    jl_append(STATE / "units.jsonl", {
        "event": "end", "unit": name, "cell": cell, "lane": lane, "rc": rc, "done": done,
        "killed_at_stop": killed, "wall_s": round(wall, 1), "load_end": os.getloadavg(),
        "load1_mean": round(sum(loads) / len(loads), 2) if loads else None,
        "load1_max": round(max(loads), 2) if loads else None, "t": time.time()})
    with LOCK:
        CLAIMED.discard(name) if not done else None


def lane_loop(lane, port, stop_at):
    fails = 0
    while True:
        u = next_unit(stop_at)
        if u is None:
            return
        run_unit(u, port, lane, stop_at)
        if not is_done(u[0], u[2]):
            fails += 1
            if fails >= 3:
                jl_append(STATE / "units.jsonl", {"event": "lane_abort", "lane": lane,
                                                  "reason": "3 failed units"})
                return


def cmd_run(argv):
    stop_at = None
    if "--stop-at" in argv:
        stop_at = dt.datetime.fromisoformat(argv[argv.index("--stop-at") + 1]).timestamp()
    for d in ("cells", "logs"):
        (STATE / d).mkdir(parents=True, exist_ok=True)
    (STATE / "STOP").unlink(missing_ok=True)
    (STATE / "supervisor.pid").write_text(str(os.getpid()))
    jl_append(STATE / "units.jsonl", {"event": "supervisor_start", "pid": os.getpid(),
                                      "stop_at": stop_at, "tree": str(TREE), "t": time.time()})
    ths = [threading.Thread(target=lane_loop, args=(i, PORTS[i], stop_at)) for i in range(2)]
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    jl_append(STATE / "units.jsonl", {"event": "supervisor_exit", "t": time.time()})
    (STATE / "supervisor.pid").unlink(missing_ok=True)


def cmd_status():
    for cell in ("b_greedy", "b_t05", "b_t10", "d_kakuna_vs_synthv2", "e_kakuna_vs_fp1000",
                 "f1_n0t1_vs_kakuna_t1", "f2_n0t1_vs_kakuna_greedy"):
        w, l, t, n = pooled(cell)
        lo, hi = wilson(w, n)
        tot = sum(1 for u in all_units() if u[1] == cell)
        nd = sum(1 for u in all_units() if u[1] == cell and is_done(u[0], u[2]))
        print(f"{cell:22s} units {nd}/{tot}  n={n:4d}  W{w} L{l} T{t}  "
              f"wr={w / n if n else float('nan'):.3f}  [{lo:.3f}, {hi:.3f}]")
    f = STATE / "topup.json"
    print("topup:", f.read_text() if f.exists() else "undecided")


if __name__ == "__main__":
    {"run": lambda: cmd_run(sys.argv[2:]), "status": cmd_status}[sys.argv[1]]()
