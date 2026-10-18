"""K2 + X26 vs Metamon Kakuna on both team sets — the detached, resumable, ONE-lane unit queue
(2026-10-17). The pattern of x26_metamon_anchor_2026-10-17/scripts/xq.py.

A UNIT is one `python -m main.anchors` invocation (20 games, role-balanced halves,
--team-seed == --seed-base == the unit seed) run FROM ITS CELL'S OWN PINNED CHECKOUT ROOT with
that checkout's interpreter and PYTHONPATH:

  * K2 (`ai_v14_07_g0p_k2/final_model.zip` @91,127,808, arch gen3_event_record_v2, config v123,
    saved under torch 2.5.1+cu121) is played by ITS OWN TRAINING PIN a5b2fdba's code on
    `gen3ai_stable` (torch 2.5.1) -- the commit every one of its steps 83.07M-91.13M ran on. It does
    not load at HEAD (arch gen3_mon_tied_gain_v1); nothing is edited to make it load.
  * X26 (`rb_x26_s2601` checkpoint_161000169) is played by 63f97a57, the commit the 2026-10-17 X26
    read ran, on gen3ai_torch28, so a top-up row's stamp is that read's stamp byte for byte.

Rule 5b (X26 is live): every unit runs under `nice -n 19 taskset -c 5-7,13-15 mem_cap.sh 12`, one
at a time. A unit is DONE iff summary.json says status OK and n == 20; a partial dir is moved aside
and re-run (at most 2 tries).

    python kq2.py run [--stop-at ISO]   # work the queue (no unit STARTS after --stop-at)
    python kq2.py status
    python kq2.py one <unit> [--dry-run]
"""
import datetime as dt
import json
import os
import subprocess
import sys
import time
from pathlib import Path

STATE = Path(os.environ.get("KQ2_STATE", "/home/goodlad/dev/gen3ai-reads/k2_x26_kakuna_both_teamsets_2026-10-17"))
MAIN = Path("/home/goodlad/dev/gen3ai")
PIN_K2 = Path("/home/goodlad/.cache/gen3ai/k2_pin/pin_a5b2fdba")     # K2's training pin
PIN_X26 = Path("/home/goodlad/.cache/gen3ai/k2_pin/pin_63f97a57")    # the X26 read's commit
PY_STABLE = "/home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3"   # torch 2.5.1+cu121
PY_T28 = "/home/goodlad/miniconda3/envs/gen3ai_torch28/bin/python3"
K2 = str(STATE / "model_k2" / "final_model.zip")        # sha256 cc5d5502...07c4 (copied read-only)
X26 = "/home/goodlad/dev/gen3ai-reads/x26_metamon_anchor_2026-10-17/model/checkpoint_161000169_steps.zip"
SHA = {K2: "cc5d5502ca23b0879c531b515993073d83b42613b5a63f19f63e48bbf58207c4",
       X26: "bc6050d1ca254abd9866945f3dde74a2189514a551e83c29abb79b2a20e52992"}
HOME72 = str(STATE / "configs" / "anchors_home72_kakuna.json")   # sha256 e8a064db... (T55 manifest)
PORT = 9573
CORES = "5-7,13-15"
MEM_GB = "12"
GAMES = 20

# cell -> (tree, python, model, teamset, config or None, seeds)
CELLS = {
    "K2_greedy_h72": (PIN_K2, PY_STABLE, K2, "home", HOME72, list(range(0, 20))),   # 0-11 = X26's banked seeds
    "K2_greedy_away": (PIN_K2, PY_STABLE, K2, "away", None, list(range(0, 20))),
    # X26 away already holds seeds 0-4 (K_greedy_away_00..04); the top-up takes 5-19, disjoint.
    "X26_greedy_away": (PIN_X26, PY_T28, X26, "away", None, list(range(5, 20))),
    # FOLLOW-UP (coordinator-approved, 2026-10-17 23:15): both home72 cells to n = 400 on paired
    # seeds 12-19. X26 home72 already holds seeds 0-11 (K_greedy_h72_00..11 of the X26 read).
    "X26_greedy_h72": (PIN_X26, PY_T28, X26, "home", HOME72, list(range(12, 20))),
}


def units():
    """Round-robin by seed so every cell's n grows together (a time-boxed stop leaves each cell
    with whole units and no cell starved)."""
    out = []
    for i in range(20):
        for name, c in CELLS.items():
            if i in c[5]:
                out.append((f"{name}_{i:02d}", name, i))
    return out


def is_done(unit):
    p = STATE / "cells" / unit / "summary.json"
    if not p.exists():
        return False
    try:
        s = json.loads(p.read_text())
    except ValueError:
        return False
    return s.get("status") == "OK" and s.get("n") == GAMES


def jl(obj):
    with open(STATE / "units.jsonl", "a") as f:
        f.write(json.dumps({"t": time.time(), **obj}) + "\n")
        f.flush()
        os.fsync(f.fileno())


def sha256(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def preflight(cell):
    tree, _py, model, *_ = CELLS[cell]
    if sha256(model) != SHA[model]:
        raise SystemExit(f"model sha256 mismatch: {model}")
    r = subprocess.run([PY_T28, "-m", "main.live.halt", "status"], cwd=str(MAIN),
                       env={**os.environ, "PYTHONPATH": str(MAIN / "src"), "CUDA_VISIBLE_DEVICES": ""},
                       capture_output=True, text=True)
    if r.returncode != 0 or "clear" not in r.stdout:
        raise SystemExit(f"T28 halt marker is not clear: {r.stdout} {r.stderr}")


def argv_for(unit, cell, seed, dry=False):
    tree, py, model, teamset, _cfg, _ = CELLS[cell]
    a = [py, "-m", "main.anchors", "--model", model, "--opponent", "metamon:Kakuna", "--regime", "greedy",
         "--teamset", teamset, "--games", str(GAMES), "--team-seed", str(seed), "--seed-base", str(seed),
         "--device", "cpu", "--port", str(PORT), "--nice", "19", "--out", str(STATE / "cells" / unit)]
    if dry:
        a.append("--dry-run")
    return ["nice", "-n", "19", "taskset", "-c", CORES,
            str(MAIN / "scripts" / "ops" / "mem_cap.sh"), "--name", "k2x26kakuna", MEM_GB, *a]


def env_for(cell):
    tree, _py, _m, _ts, cfg, _ = CELLS[cell]
    env = dict(os.environ)
    for k in ("POKESIM_SIM_BRIDGE_BIN", "POKESIM_SEARCH_DRIVER_BIN", "POKESIM_CORE_EVENTS_BIN",
              "CARGO_TARGET_DIR", "GEN3AI_ANCHORS_CONFIG"):
        env.pop(k, None)
    env.update({
        "PYTHONPATH": str(tree / "src"),
        "CUDA_VISIBLE_DEVICES": "",
        "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1", "PYTHONUNBUFFERED": "1",
    })
    if cfg:
        env["GEN3AI_ANCHORS_CONFIG"] = cfg
    return env


def run_unit(unit, cell, seed, dry=False):
    preflight(cell)
    out = STATE / "cells" / unit
    if out.exists() and not dry:
        out.rename(out.with_name(f"{unit}.partial.{int(time.time())}"))
    argv = argv_for(unit, cell, seed, dry)
    tree = CELLS[cell][0]
    jl({"event": "start", "unit": unit, "cell": cell, "seed": seed, "tree": str(tree), "argv": argv,
        "load": os.getloadavg()})
    log = open(STATE / "logs" / f"{unit}{'.dry' if dry else ''}.log", "ab")
    t0 = time.time()
    # cwd = the pinned checkout ROOT (the pool source resolves data/teams against the CWD; X26 FINDING 1)
    proc = subprocess.Popen(argv, cwd=str(tree), env=env_for(cell), stdout=log,
                            stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
    jl({"event": "pid", "unit": unit, "pid": proc.pid})
    loads = []
    while proc.poll() is None:
        time.sleep(5)
        loads.append(os.getloadavg()[0])
    rc = proc.wait()
    jl({"event": "end", "unit": unit, "cell": cell, "rc": rc, "done": is_done(unit),
        "wall_s": round(time.time() - t0, 1),
        "load1_mean": round(sum(loads) / len(loads), 2) if loads else None,
        "load1_max": round(max(loads), 2) if loads else None})
    return rc


def cmd_run(args):
    stop_at = None
    if "--stop-at" in args:
        stop_at = dt.datetime.fromisoformat(args[args.index("--stop-at") + 1]).timestamp()
    jl({"event": "supervisor_start", "pid": os.getpid(), "stop_at": stop_at})
    fails = {}
    for unit, cell, seed in units():
        while not is_done(unit) and fails.get(unit, 0) < 2:
            if stop_at and time.time() > stop_at:
                jl({"event": "supervisor_stop_at"})
                (STATE / "queue.done").write_text("stop_at\n")
                return
            run_unit(unit, cell, seed)
            if not is_done(unit):
                fails[unit] = fails.get(unit, 0) + 1
    jl({"event": "supervisor_done", "fails": fails})
    (STATE / "queue.done").write_text("done\n")


def cmd_status(_args):
    agg = {}
    for unit, cell, _ in units():
        d = agg.setdefault(cell, [0, 0])
        d[1] += 1
        d[0] += is_done(unit)
    for k, (d, n) in agg.items():
        print(f"{k:18s} {d}/{n} units done")


def cmd_one(args):
    by = {u[0]: u for u in units()}
    unit, cell, seed = by[args[0]]
    sys.exit(run_unit(unit, cell, seed, dry="--dry-run" in args))


if __name__ == "__main__":
    for d in ("cells", "logs"):
        (STATE / d).mkdir(parents=True, exist_ok=True)
    {"run": cmd_run, "status": cmd_status, "one": cmd_one}[sys.argv[1]](sys.argv[2:])
