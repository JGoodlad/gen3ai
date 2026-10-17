"""X26 vs Metamon — the detached, resumable unit queue (2026-10-17).

ONE lane (root CLAUDE.md rule 5b: X26 is live on the GPU, so every unit runs under
`nice -n 19 taskset -c 5-7,13-15` inside `scripts/ops/mem_cap.sh`, and only one heavy job at a
time). A UNIT is one `python -m main.anchors` invocation with its own --out and
--team-seed == --seed-base == the unit seed. A unit is DONE iff its summary.json says
status OK and n == the unit's games. A partial unit is moved aside and re-run.

    python xq.py run [--stop-at ISO]     # work the queue until empty (or --stop-at)
    python xq.py status                  # done / pending per cell
    python xq.py one <unit> [--dry-run]  # run one named unit in the foreground

The child's PYTHONPATH is set HERE (the worktree's src/), so the tool runs this checkout's code.
"""
import datetime as dt
import json
import os
import subprocess
import sys
import time
from pathlib import Path

STATE = Path(os.environ.get("XQ_STATE", "/home/goodlad/dev/gen3ai-reads/x26_metamon_anchor_2026-10-17"))
TREE = Path(__file__).resolve().parents[5]
PY = "/home/goodlad/miniconda3/envs/gen3ai_torch28/bin/python3"
# X26 = rb_x26_s2601; the latest periodic checkpoint at the read's start, COPIED (sha256-checked)
# so a checkpoint rotation in the live run cannot pull it from under the read.
MODEL = str(STATE / "model" / "checkpoint_161000169_steps.zip")   # step 161,000,169
HOME72 = str(STATE / "configs" / "anchors_home72_kakuna.json")    # X22's home72 convention
PORT = 9571
CORES = "5-7,13-15"
MEM_GB = "12"

# (cell, opponent, extra argv, teamset, config or None, unit games, n units, seed base)
CELLS = [
    ("K_greedy_h72", "metamon:Kakuna", [], "home", HOME72, 20, 20, 0),
    ("K_t10_h72", "metamon:Kakuna", ["--opponent-temperature", "1.0"], "home", HOME72, 20, 20, 0),
    ("S_greedy_away", "metamon:SmallRL", [], "away", None, 20, 10, 0),
    ("S_greedy_home", "metamon:SmallRL", [], "home", None, 20, 10, 0),
    ("K_greedy_away", "metamon:Kakuna", [], "away", None, 20, 5, 0),
    # the SOP tier-B distribution check (both sides t1; a later --regime wins in argparse)
    ("S_t1_away", "metamon:SmallRL", ["--regime", "t1"], "away", None, 20, 5, 0),
]
# Sizing (unit K_greedy_h72_00, 2026-10-17, one lane on rule 5b's three physical cores, load ~7):
# 20 Kakuna games in 165 s = 8.3 s/game. The first queue (10 units per Kakuna cell, 5 per SmallRL)
# was extended to the sizes above after that read, before any other unit finished.


def units():
    """Kakuna goal cells on paired seeds 0-9, the SmallRL cells and K_greedy_away, then the Kakuna
    top-up to seeds 10-19 (moved last on the second queue start, 2026-10-17, so the cheap
    continuity cells finish first)."""
    out = []
    by = {c[0]: c for c in CELLS}
    for i in range(10):
        for name in ("K_greedy_h72", "K_t10_h72"):
            out.append((f"{name}_{i:02d}", by[name], i))
    for name in ("S_greedy_away", "S_greedy_home", "K_greedy_away", "S_t1_away"):
        c = by[name]
        for i in range(c[6]):
            out.append((f"{name}_{i:02d}", c, c[7] + i))
    for i in range(10, 20):
        for name in ("K_greedy_h72", "K_t10_h72"):
            out.append((f"{name}_{i:02d}", by[name], i))
    return out


def is_done(name, games):
    p = STATE / "cells" / name / "summary.json"
    if not p.exists():
        return False
    try:
        s = json.loads(p.read_text())
    except ValueError:
        return False
    return s.get("status") == "OK" and s.get("n") == games


def jl(obj):
    with open(STATE / "units.jsonl", "a") as f:
        f.write(json.dumps({"t": time.time(), **obj}) + "\n")


def argv_for(name, cell, seed, dry=False):
    _, opp, extra, teamset, _cfg, games, _, _ = cell
    out = STATE / "cells" / name
    a = [PY, "-m", "main.anchors", "--model", MODEL, "--opponent", opp, "--regime", "greedy",
         *extra, "--teamset", teamset, "--games", str(games), "--team-seed", str(seed),
         "--seed-base", str(seed), "--device", "cpu", "--port", str(PORT), "--nice", "19",
         "--out", str(out)]
    if dry:
        a.append("--dry-run")
    return ["nice", "-n", "19", "taskset", "-c", CORES,
            str(TREE / "scripts" / "ops" / "mem_cap.sh"), "--name", "x26anchor", MEM_GB, *a]


def env_for(cell):
    env = dict(os.environ)
    env.update({
        "PYTHONPATH": str(TREE / "src"),
        "CUDA_VISIBLE_DEVICES": "",
        "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1", "PYTHONUNBUFFERED": "1",
    })
    if cell[4]:
        env["GEN3AI_ANCHORS_CONFIG"] = cell[4]
    else:
        env.pop("GEN3AI_ANCHORS_CONFIG", None)
    return env


def run_unit(name, cell, seed, dry=False):
    out = STATE / "cells" / name
    if out.exists() and not dry:
        out.rename(out.with_name(f"{name}.partial.{int(time.time())}"))
    argv = argv_for(name, cell, seed, dry)
    jl({"event": "start", "unit": name, "cell": cell[0], "seed": seed, "argv": argv,
        "load": os.getloadavg()})
    log = open(STATE / "logs" / f"{name}{'.dry' if dry else ''}.log", "ab")
    t0 = time.time()
    # cwd = the checkout ROOT: the pool team source (`utils.team_loader.TeamLoader`) resolves
    # its default base dir "data/teams" against the CWD; from src/ it finds 0 teams and the
    # S_greedy_home units of the first queue died at once (IndexError, 2026-10-17 FINDING).
    proc = subprocess.Popen(argv, cwd=str(TREE), env=env_for(cell), stdout=log,
                            stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                            start_new_session=True)
    jl({"event": "pid", "unit": name, "pid": proc.pid})
    loads = []
    while proc.poll() is None:
        time.sleep(5)
        loads.append(os.getloadavg()[0])
    rc = proc.wait()
    jl({"event": "end", "unit": name, "cell": cell[0], "rc": rc,
        "done": is_done(name, cell[5]), "wall_s": round(time.time() - t0, 1),
        "load1_mean": round(sum(loads) / len(loads), 2) if loads else None,
        "load1_max": round(max(loads), 2) if loads else None})
    return rc


def cmd_run(args):
    stop_at = None
    if "--stop-at" in args:
        stop_at = dt.datetime.fromisoformat(args[args.index("--stop-at") + 1]).timestamp()
    jl({"event": "supervisor_start", "pid": os.getpid()})
    fails = {}
    for name, cell, seed in units():
        if is_done(name, cell[5]):
            continue
        while not is_done(name, cell[5]) and fails.get(name, 0) < 2:
            if stop_at and time.time() > stop_at:
                jl({"event": "supervisor_stop_at"})
                return
            run_unit(name, cell, seed)
            if not is_done(name, cell[5]):
                fails[name] = fails.get(name, 0) + 1
    jl({"event": "supervisor_done"})


def cmd_status(_args):
    agg = {}
    for name, cell, _ in units():
        d = agg.setdefault(cell[0], [0, 0])
        d[1] += 1
        d[0] += is_done(name, cell[5])
    for k, (d, n) in agg.items():
        print(f"{k:16s} {d}/{n} units done")


def cmd_one(args):
    by = {u[0]: u for u in units()}
    name, cell, seed = by[args[0]]
    sys.exit(run_unit(name, cell, seed, dry="--dry-run" in args))


if __name__ == "__main__":
    for d in ("cells", "logs"):
        (STATE / d).mkdir(parents=True, exist_ok=True)
    {"run": cmd_run, "status": cmd_status, "one": cmd_one}[sys.argv[1]](sys.argv[2:])
