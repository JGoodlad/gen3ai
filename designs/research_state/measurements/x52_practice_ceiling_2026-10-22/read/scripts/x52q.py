"""X52's REGISTERED read (AM-X52-e): the detached, resumable, ONE-lane unit queue (2026-10-22).

X51's `x51q.py` (`x51_plateau_vs_kakuna_2026-10-20/scripts/`) with these changes:
  * REVERSE cells: our model pilots one X52 half (36 files), Kakuna ckpt 34 pilots home72 (`make_configs.py`);
  * every cell runs from ONE fresh detached checkout of the X35 production pin `f4b25238` (the pin the X52 and C0
    runs trained at; config v161; P* trained at v158 and is the fork's start), so every model difference is a model
    difference only;
  * the models are read-only, sha-checked copies (`MODELS`), the five registered cells' models;
  * unit seeds 0-29 for every primary (model, half) cell, so every model on a half plays the same deals; the
    descriptive cells take the first 10 (s3501 @1M, practised) and 20 (C0 s3501 final, unpractised) of them.

A UNIT is one `python -m main.anchors` invocation (20 games, role-balanced, --team-seed == --seed-base == the unit
seed) run from the pin's ROOT. DONE iff summary.json says OK and n == 20 (whether it COUNTS is `aggregate.py`'s call);
a partial dir is moved aside and re-run (at most 2 tries per unit, across chunks). HEAVY SLOTS (rule 5b): ONE slot
per <= 45-min chunk (`mem_cap.sh ... timeout 2700 x52q.py chunk`).

    python x52q.py run --stop-at ISO                 # the supervisor (every cell)
    python x52q.py status
    python x52q.py one <unit> --games 2              # a smoke, outside the chunk loop (the caller holds a slot)
"""
import datetime as dt
import hashlib
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

STATE = Path(os.environ.get("X52_STATE", "/home/goodlad/gen3ai_archive/x52_read"))
MAIN = Path("/home/goodlad/dev/gen3ai")
MEAS = MAIN / "designs" / "research_state" / "measurements"
CONFIGS = STATE / "configs"
PIN = Path("/home/goodlad/.cache/gen3ai/x52/pin_f4b25238")
PY_T28 = "/home/goodlad/miniconda3/envs/gen3ai_torch28/bin/python3"
M = STATE / "models"
MODELS = {
    "pstar": (M / "pstar" / "checkpoint_280000170_steps.zip",
              "814823c514fb791e3c6e671ba126f7d0fd90733098b0539bb85babe7c5ab91ed"),
    "x52a": (M / "x52s3501_5m" / "final_model.zip",
             "58f92e7fbd5935ea75748e226680743a09c78f60a990e0832932582de09c9d2a"),
    "x52b": (M / "x52s3502_5m" / "final_model.zip",
             "c0058823e686742ecf00af9af1dbf2fadfc13d74b6cdbf9480f8e979b440ece0"),
    "x52a1m": (M / "x52s3501_1m" / "checkpoint_281000024_steps.zip",
               "07bc9af0126ef379700a476707097130bc11236aa59994d550ddde767b4509fe"),
    "c0a": (M / "c0s3501_final" / "final_model.zip",
            "09a6a2df48509a59fd9aefe6d20d9722aad6717f1730c6dfd75274446b283303"),
}
CONFIG_SHA = {
    "practised": "b067a143907628d7046268c237116d5ac8c9ba72370e5068195ff65034ff17c7",
    "unpractised": "7abf8001b5c12f4e25772549c89dc254ac5fbd21c355725b5970b21b599246af",
}
METAMON_HOME72 = Path("/home/goodlad/dev/metamon/cache/teams/gen3ai_home72/gen3ou")
MEMCAP = MAIN / "scripts" / "ops" / "mem_cap.sh"
PORT = 9552
CORES = "5-7,13-15"
MEM_GB = "12"
GAMES = 20
CHUNK_TIMEOUT_S = 2700
DEFAULT_BUDGET_S = 2400
DEFAULT_UNIT_S = 260.0
HALF_KEY = {"pr": "practised", "un": "unpractised"}


def _manifests():
    """half -> {stem: sha256} for OUR side (the X52 manifest), and home72 -> {stem: sha256} for Kakuna's."""
    x52 = json.loads((MEAS / "x52_practice_ceiling_2026-10-22" / "x52_teams_manifest.json").read_text())["teams"]
    out = {h: {i: t["sha256"] for i, t in x52.items() if t["half"] == h} for h in ("practised", "unpractised")}
    out["home72"] = json.loads((MEAS / "x35_breakers_stage_2026-10-17" / "home72_manifest.json").read_text())[
        "metamon_teams"]
    return out


def _cell(model, half, n_units):
    h = HALF_KEY[half]
    return dict(model=model, config=h, seeds=list(range(n_units)),
                sides=[(STATE / f"teams_{h}", ".txt", h), (METAMON_HOME72, ".gen3ou_team", "home72")])


# PRIMARY (priority 0): P*, s3501 @5M, s3502 @5M on each half, 30 units (600 games) each.
# DESCRIPTIVE (priority 1): s3501 @1M on the practised half (10 units), C0 s3501's final on the unpractised (20).
CELLS = {f"{m}_{h}": _cell(m, h, 30) for h in ("pr", "un") for m in ("pstar", "x52a", "x52b")}
CELLS["x52a1m_pr"] = _cell("x52a1m", "pr", 10)
CELLS["c0a_un"] = _cell("c0a", "un", 20)
PRIORITY = {c: (1 if c in ("x52a1m_pr", "c0a_un") else 0) for c in CELLS}


def config_path(cell):
    return CONFIGS / f"anchors_x52_{CELLS[cell]['config']}.json"


def units(cells):
    """Priority group first; inside a group interleaved by progress fraction (a time-boxed stop leaves every cell of
    the group with whole units in proportion)."""
    out = []
    for name in cells:
        seeds = CELLS[name]["seeds"]
        for i, s in enumerate(seeds):
            out.append(((PRIORITY[name], (i + 0.5) / len(seeds), name), name, f"{name}_{s:02d}", s))
    return [(u, c, s) for _, c, u, s in sorted(out)]


def is_done(unit):
    p = STATE / "cells" / unit / "summary.json"
    try:
        s = json.loads(p.read_text())
    except (OSError, ValueError):
        return False
    return s.get("status") == "OK" and s.get("n") == GAMES


def jl(obj):
    with open(STATE / "units.jsonl", "a") as f:
        f.write(json.dumps({"t": time.time(), **obj}) + "\n")
        f.flush()
        os.fsync(f.fileno())


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _fails():
    p = STATE / "fails.json"
    return json.loads(p.read_text()) if p.exists() else {}


def _bump_fail(unit):
    f = _fails()
    f[unit] = f.get(unit, 0) + 1
    (STATE / "fails.json").write_text(json.dumps(f, indent=1) + "\n")


def _port_free(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)   # X51 FINDING 6: TIME_WAIT is not a listener
        try:
            s.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


def preflight(cell):
    c = CELLS[cell]
    path, want = MODELS[c["model"]]
    if not want or sha256(path) != want:
        raise SystemExit(f"model sha256 mismatch (or unset): {path}")
    if sha256(config_path(cell)) != CONFIG_SHA[c["config"]]:
        raise SystemExit(f"config sha256 mismatch: {config_path(cell)}")
    man = _manifests()
    for d, ext, key in c["sides"]:
        teams = man[key]
        present = sorted(p.name for p in d.iterdir())
        if present != sorted(stem + ext for stem in teams):
            raise SystemExit(f"{key}: file set differs from the manifest in {d}")
        for stem, want_t in teams.items():
            if sha256(d / (stem + ext)) != want_t:
                raise SystemExit(f"{key}: sha256 mismatch {d / (stem + ext)}")
    if not (PIN / "src" / "main" / "anchors" / "cli.py").exists():
        raise SystemExit(f"tree missing: {PIN}")
    if not _port_free(PORT):
        raise SystemExit(f"port {PORT} is held by someone else: not ours to touch")
    r = subprocess.run([PY_T28, "-m", "main.live.halt", "status"], cwd=str(MAIN),
                       env={**os.environ, "PYTHONPATH": str(MAIN / "src"), "CUDA_VISIBLE_DEVICES": ""},
                       capture_output=True, text=True)
    if r.returncode != 0 or "clear" not in r.stdout:
        raise SystemExit(f"halt marker is not clear: {r.stdout} {r.stderr}")


def argv_for(unit, cell, seed, games=GAMES):
    out = STATE / "cells" / unit
    a = [PY_T28, "-m", "main.anchors", "--model", str(MODELS[CELLS[cell]["model"]][0]),
         "--opponent", "metamon:Kakuna", "--teamset", "home", "--regime", "greedy", "--games", str(games),
         "--team-seed", str(seed), "--seed-base", str(seed), "--device", "cpu", "--port", str(PORT),
         "--nice", "19", "--out", str(out), "--capture-dir", str(out / "capture")]
    return ["nice", "-n", "19", "taskset", "-c", CORES, *a]


def env_for(cell):
    env = dict(os.environ)
    for k in ("POKESIM_SIM_BRIDGE_BIN", "POKESIM_SEARCH_DRIVER_BIN", "POKESIM_CORE_EVENTS_BIN",
              "CARGO_TARGET_DIR", "GEN3AI_ANCHORS_CONFIG"):
        env.pop(k, None)
    env.update({
        "PYTHONPATH": str(PIN / "src"), "CUDA_VISIBLE_DEVICES": "",
        "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1", "TORCH_NUM_THREADS": "1", "PYTHONUNBUFFERED": "1",
        "GEN3AI_ANCHORS_CONFIG": str(config_path(cell)),
    })
    return env


def run_unit(unit, cell, seed, games=GAMES, tag=""):
    preflight(cell)
    out = STATE / "cells" / unit
    if out.exists():
        out.rename(out.with_name(f"{unit}.partial.{int(time.time())}"))
    argv = argv_for(unit, cell, seed, games)
    jl({"event": "start", "unit": unit, "cell": cell, "seed": seed, "tree": str(PIN), "argv": argv,
        "load": os.getloadavg(), "tag": tag})
    t0 = time.time()
    with open(STATE / "logs" / f"{unit}.log", "ab") as log:
        proc = subprocess.Popen(argv, cwd=str(PIN), env=env_for(cell), stdout=log, stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL, start_new_session=True)
        jl({"event": "pid", "unit": unit, "pid": proc.pid})
        loads = []
        while proc.poll() is None:
            time.sleep(5)
            loads.append(os.getloadavg()[0])
        rc = proc.wait()
    wall = round(time.time() - t0, 1)
    jl({"event": "end", "unit": unit, "cell": cell, "rc": rc, "done": is_done(unit), "wall_s": wall,
        "load1_mean": round(sum(loads) / len(loads), 2) if loads else None,
        "load1_max": round(max(loads), 2) if loads else None, "tag": tag})
    return rc, wall


def _cells_arg(args):
    return args[args.index("--cells") + 1].split(",") if "--cells" in args else list(CELLS)


def _pending(cells):
    fails = _fails()
    return [(u, c, s) for u, c, s in units(cells) if not is_done(u) and fails.get(u, 0) < 2]


def cmd_chunk(args):
    cells = _cells_arg(args)
    budget = float(args[args.index("--budget-s") + 1]) if "--budget-s" in args else DEFAULT_BUDGET_S
    t0 = time.time()
    walls = []
    n = 0
    jl({"event": "chunk_start", "pid": os.getpid(), "budget_s": budget,
        "slot": os.environ.get("GEN3AI_HEAVY_SLOT_HELD")})
    for unit, cell, seed in _pending(cells):
        est = max(walls[-3:] or [DEFAULT_UNIT_S]) * 1.15
        if time.time() - t0 + est > budget:
            break
        try:
            rc, wall = run_unit(unit, cell, seed, tag="chunk")
        except SystemExit as e:         # a preflight refusal ends the chunk, recorded; never a different unit
            jl({"event": "preflight_refused", "unit": unit, "why": str(e)})
            break
        n += 1
        if is_done(unit):
            walls.append(wall)
        else:
            _bump_fail(unit)
    jl({"event": "chunk_end", "pid": os.getpid(), "units_run": n, "wall_s": round(time.time() - t0, 1)})


def cmd_run(args):
    cells = _cells_arg(args)
    stop_at = dt.datetime.fromisoformat(args[args.index("--stop-at") + 1]).timestamp()
    jl({"event": "supervisor_start", "pid": os.getpid(), "stop_at": stop_at, "cells": cells})
    me = str(Path(__file__).resolve())
    idle = 0
    while True:
        pending = _pending(cells)
        if not pending:
            jl({"event": "supervisor_done", "fails": _fails()})
            (STATE / "queue.done").write_text(json.dumps({"done": True, "fails": _fails()}) + "\n")
            return
        left = stop_at - time.time()
        if left < 300:
            jl({"event": "supervisor_stop_at", "pending": len(pending)})
            (STATE / "queue.done").write_text("stop_at\n")
            return
        budget = min(DEFAULT_BUDGET_S, left - 120)
        argv = ["timeout", str(int(left)), "nice", "-n", "19", "taskset", "-c", CORES, str(MEMCAP),
                "--name", "x52read", MEM_GB, "timeout", str(CHUNK_TIMEOUT_S), PY_T28, me, "chunk",
                "--cells", ",".join(cells), "--budget-s", str(int(budget))]
        jl({"event": "chunk_launch", "argv": argv, "pending": len(pending)})
        before = len(pending)
        with open(STATE / "logs" / "chunks.log", "ab") as log:
            rc = subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                env={**os.environ, "CUDA_VISIBLE_DEVICES": ""}).returncode
        after = len(_pending(cells))
        jl({"event": "chunk_exit", "rc": rc, "pending_before": before, "pending_after": after})
        idle = idle + 1 if after == before else 0
        if idle >= 3:
            jl({"event": "supervisor_stuck", "rc": rc})
            (STATE / "queue.done").write_text(f"stuck rc={rc}\n")
            return
        if after == before:
            time.sleep(60)


def cmd_status(args):
    tot = 0
    for cell in _cells_arg(args):
        us = units([cell])
        d = sum(is_done(u) for u, _, _ in us)
        tot += d
        print(f"{cell:12s} {d}/{len(us)} units done")
    print(f"total {tot} units, fails {_fails()}")


def cmd_one(args):
    by = {u[0]: u for u in units(list(CELLS))}
    unit, cell, seed = by[args[0]]
    games = int(args[args.index("--games") + 1]) if "--games" in args else GAMES
    name = unit if games == GAMES else f"smoke_{unit}_g{games}"
    rc, _ = run_unit(name, cell, seed, games=games, tag="one")
    sys.exit(rc)


if __name__ == "__main__":
    for d in ("cells", "logs"):
        (STATE / d).mkdir(parents=True, exist_ok=True)
    {"run": cmd_run, "chunk": cmd_chunk, "status": cmd_status, "one": cmd_one}[sys.argv[1]](sys.argv[2:])
