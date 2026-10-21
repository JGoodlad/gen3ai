"""X51 — P* (and T*, and Kakuna-28) vs Metamon's Kakuna: the detached, resumable, ONE-lane unit queue (2026-10-20).

X37a's `xq.py` (`x37a_eval_matrix_2026-10-18/scripts/xq.py`) with three changes:
  * the MODEL is X26's plateau checkpoint P* (`rb_x26_s2601` @280,000,170) — or the toy's T* (`rb_toy_ent_s4002_deep`
    @290,000,087) — from read-only, sha-checked copies; every model cell runs from a fresh detached `63f97a57`, the
    checkout every X26 @161M Kakuna read used, so P* − X26@161M is a model difference only;
  * the Kakuna-28 vs Kakuna-34 PAIR cell runs from the tree of the commit that lands the agent `policy` key
    (`$X51_PAIR_TREE`), as X37a's `kcal` ran from its build commit;
  * HEAVY SLOTS (rule 5b, `utils.heavy_slots`): the supervisor takes ONE slot per <= 45-min CHUNK of units
    (`mem_cap.sh … timeout 2700 x51q.py chunk`), never per run; a chunk starts a unit only while its estimated end
    stays inside the chunk budget. Units inside a chunk run the anchors tool directly (the chunk's capped scope
    covers them).

A UNIT is one `python -m main.anchors` invocation (20 games, role-balanced halves, --team-seed == --seed-base == the
unit seed, seeds 0–19 so every P* unit twins an X26 @161M unit) run FROM ITS CELL'S PINNED CHECKOUT ROOT. DONE iff
summary.json says status OK and n == 20 (whether it COUNTS is the aggregate's call); a partial dir is moved aside and
re-run (at most 2 tries per unit, across chunks).

    python x51q.py run --cells pstar_home72,pstar_hpfix,pstar_forum72,pstar_p72 --stop-at ISO   # the supervisor
    python x51q.py chunk --cells … --budget-s 2400                                                 # one slot's work
    python x51q.py status
    python x51q.py one <unit> [--dry-run]                                                          # outside a slot: smoke only
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

STATE = Path(os.environ.get("X51_STATE", "/home/goodlad/dev/gen3ai-reads/x51_2026-10-20"))
MAIN = Path("/home/goodlad/dev/gen3ai")
MEAS = MAIN / "designs" / "research_state" / "measurements"
CONFIGS = STATE / "configs"
PIN = Path("/home/goodlad/.cache/gen3ai/x51/pin_63f97a57")
PAIR_TREE = Path(os.environ.get("X51_PAIR_TREE", "/home/goodlad/.cache/gen3ai/x51/pin_pair"))
PY_T28 = "/home/goodlad/miniconda3/envs/gen3ai_torch28/bin/python3"
MODELS = {
    "pstar": (STATE / "model" / "checkpoint_280000170_steps.zip",
              "814823c514fb791e3c6e671ba126f7d0fd90733098b0539bb85babe7c5ab91ed"),
    "toy": (STATE / "model_toy" / "checkpoint_290000087_steps.zip",
            "cfa7f7bee7b7a43959f496ce6d5a34556bae914de9c80f90f9342a05a22965e7"),
}
CONFIG_SHA = {
    "sym_home72": "f99f2360fe45e5261c53c7682324117d1e930cf8b4c24f24d7f4aa06724bb304",
    "sym_heldout72_hpfix": "2392bc9ef4c276ecbaed05e15fe75b41c0b65012c32c9a8c22fab6a18377eb02",
    "sym_forum72": "749a399e1a0718c4cd27b10208a188cfb8e6081dc3527ba7ca735f3ef79acf63",
    "sym_p72": "d4c4c0f6aa0488439974cb193172662f4673ee0eabcd3cf8f05e86a53aa8c289",
    "pair_home72": "d3fd2a2289f4a2679a30f1e42e14ef74d7eec8cb0c459d52fffa12fef5eba97a",
}
METAMON_TEAMS = Path("/home/goodlad/dev/metamon/cache/teams")
MEMCAP = MAIN / "scripts" / "ops" / "mem_cap.sh"
PORT = 9551
CORES = "5-7,13-15"
MEM_GB = "12"
GAMES = 20
CHUNK_TIMEOUT_S = 2700          # the slot policy's 45-min chunk, enforced inside the slot
DEFAULT_BUDGET_S = 2400         # a chunk STARTS no unit whose estimated end passes this
DEFAULT_UNIT_S = {"model": 260.0, "pair": 240.0}


def _manifests():
    """set -> {stem: sha256}: the committed manifests every team file on both sides must match."""
    def teams(rel):
        return {s: r["sha256"] for s, r in json.loads((MEAS / rel).read_text())["teams"].items()}
    return {
        "home72": json.loads((MEAS / "x35_breakers_stage_2026-10-17" / "home72_manifest.json").read_text())["our_teams"],
        "heldout72_hpfix": teams("heldout72_hpfix_kakuna_2026-10-18/heldout72_hpfix_manifest.json"),
        "forum72": teams("x37a_eval_matrix_2026-10-18/forum72_manifest.json"),
        "p72": teams("x37a_eval_matrix_2026-10-18/p72_manifest.json"),
    }


def _ours(s):
    return (STATE / f"teams_{s}", ".txt", s)


def _mm(s):
    return (METAMON_TEAMS / f"gen3ai_{s}" / "gen3ou", ".gen3ou_team", s)


def _model_cell(model, team_set):
    return dict(tree=PIN, kind="model", model=model, config=f"sym_{team_set}", seeds=list(range(20)),
                sides=[_ours(team_set), _mm(team_set)])


# cell -> dict(tree, kind, model, config, seeds, sides); ORDER = the queue's priority (P* first, then the pair, then T*)
CELLS = {
    "pstar_home72": _model_cell("pstar", "home72"),
    "pstar_hpfix": _model_cell("pstar", "heldout72_hpfix"),
    "pstar_forum72": _model_cell("pstar", "forum72"),
    "pstar_p72": _model_cell("pstar", "p72"),
    "k28v34_home72": dict(tree=PAIR_TREE, kind="pair", model=None, config="pair_home72", seeds=list(range(20)),
                          sides=[_mm("home72")]),
    "toy_home72": _model_cell("toy", "home72"),
    "toy_forum72": _model_cell("toy", "forum72"),
}
PRIORITY = {"pstar": 0, "k28v34": 1, "toy": 2}


def config_path(cell):
    return CONFIGS / f"anchors_x51_{CELLS[cell]['config']}.json"


def units(cells):
    """Priority group first; inside a group interleaved by progress fraction, so a time-boxed stop leaves every
    cell of the group with whole units in proportion."""
    out = []
    for name in cells:
        seeds = CELLS[name]["seeds"]
        for i, s in enumerate(seeds):
            out.append(((PRIORITY[name.split("_")[0]], (i + 0.5) / len(seeds)), name, f"{name}_{s:02d}", s))
    return [(u, c, s) for _, c, u, s in sorted(out)]


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
        # SO_REUSEADDR: the previous unit's own server leaves TIME_WAIT connections on the port for ~60 s, which
        # the front end's own bind (reuse on) ignores; only a LISTENING socket is someone else's server.
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


def preflight(cell):
    c = CELLS[cell]
    if c["model"] is not None:
        path, want = MODELS[c["model"]]
        if sha256(path) != want:
            raise SystemExit(f"model sha256 mismatch: {path}")
    if sha256(config_path(cell)) != CONFIG_SHA[c["config"]]:
        raise SystemExit(f"config sha256 mismatch: {config_path(cell)}")
    man = _manifests()
    for d, ext, key in c["sides"]:
        teams = man[key]
        present = sorted(p.name for p in d.iterdir())
        if present != sorted(stem + ext for stem in teams):
            raise SystemExit(f"{key}: file set differs from the manifest in {d}")
        for stem, want in teams.items():
            if sha256(d / (stem + ext)) != want:
                raise SystemExit(f"{key}: sha256 mismatch {d / (stem + ext)}")
    if not (c["tree"] / "src" / "main" / "anchors" / "cli.py").exists():
        raise SystemExit(f"tree missing: {c['tree']}")
    if not _port_free(PORT):
        raise SystemExit(f"port {PORT} is held by someone else: not ours to touch")
    r = subprocess.run([PY_T28, "-m", "main.live.halt", "status"], cwd=str(MAIN),
                       env={**os.environ, "PYTHONPATH": str(MAIN / "src"), "CUDA_VISIBLE_DEVICES": ""},
                       capture_output=True, text=True)
    if r.returncode != 0 or "clear" not in r.stdout:
        raise SystemExit(f"T28 halt marker is not clear: {r.stdout} {r.stderr}")


def argv_for(unit, cell, seed, dry=False, games=GAMES):
    c = CELLS[cell]
    out = STATE / "cells" / unit
    common = ["--regime", "greedy", "--games", str(games), "--team-seed", str(seed), "--seed-base", str(seed),
              "--device", "cpu", "--port", str(PORT), "--nice", "19", "--out", str(out),
              "--capture-dir", str(out / "capture")]
    if c["kind"] == "model":
        a = [PY_T28, "-m", "main.anchors", "--model", str(MODELS[c["model"]][0]), "--opponent", "metamon:Kakuna",
             "--teamset", "home", *common]
    else:   # the pair cell: side A (reported) = Kakuna ckpt 28, side B = Kakuna ckpt 34, both on home72
        a = [PY_T28, "-m", "main.anchors", "--opponent-a", "metamon:Kakuna28", "--opponent-b", "metamon:Kakuna",
             "--teamset", "home", "--progress-timeout", "1800", *common]
    if dry:
        a.append("--dry-run")
    return ["nice", "-n", "19", "taskset", "-c", CORES, *a]


def env_for(cell):
    c = CELLS[cell]
    env = dict(os.environ)
    for k in ("POKESIM_SIM_BRIDGE_BIN", "POKESIM_SEARCH_DRIVER_BIN", "POKESIM_CORE_EVENTS_BIN",
              "CARGO_TARGET_DIR", "GEN3AI_ANCHORS_CONFIG"):
        env.pop(k, None)
    env.update({
        "PYTHONPATH": str(c["tree"] / "src"),
        "CUDA_VISIBLE_DEVICES": "",
        "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1", "TORCH_NUM_THREADS": "1", "PYTHONUNBUFFERED": "1",
        "GEN3AI_ANCHORS_CONFIG": str(config_path(cell)),
    })
    return env


def run_unit(unit, cell, seed, dry=False, games=GAMES, tag=""):
    preflight(cell)
    out = STATE / "cells" / unit
    if out.exists() and not dry:
        out.rename(out.with_name(f"{unit}.partial.{int(time.time())}"))
    argv = argv_for(unit, cell, seed, dry, games)
    tree = CELLS[cell]["tree"]
    jl({"event": "start", "unit": unit, "cell": cell, "seed": seed, "tree": str(tree), "argv": argv,
        "load": os.getloadavg(), "tag": tag})
    log = open(STATE / "logs" / f"{unit}{'.dry' if dry else ''}.log", "ab")
    t0 = time.time()
    # cwd = the pinned checkout ROOT (the pool source resolves data/teams against the CWD)
    proc = subprocess.Popen(argv, cwd=str(tree), env=env_for(cell), stdout=log,
                            stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
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


#: the 2-game smokes a cell needs before its first unit (the setup chunk's T* and pair smokes hit the port check's
#: TIME_WAIT false positive, 2026-10-20 10:13): each runs once per chunk until it passes, at most twice in all.
SMOKES = (("k28v34_home72", "k28v34_home72_00"), ("toy_home72", "toy_home72_00"))


def smoke_ok(cell):
    unit = dict(SMOKES)[cell]
    p = STATE / "cells" / f"smoke_{unit}_g2" / "summary.json"
    try:
        s = json.loads(p.read_text())
    except (OSError, ValueError):
        return False
    return s.get("status") == "OK" and s.get("n") == 2 and bool(s.get("regime_verified_decisions"))


def run_smokes():
    for cell, unit in SMOKES:
        tries = STATE / f"smoke_{unit}.tries"
        n = int(tries.read_text()) if tries.exists() else 0
        if smoke_ok(cell) or n >= 2:
            continue
        tries.write_text(f"{n + 1}\n")
        try:
            run_unit(f"smoke_{unit}_g2", cell, CELLS[cell]["seeds"][0], games=2, tag="smoke")
        except SystemExit as e:          # a preflight refusal is recorded, never fatal to the chunk
            jl({"event": "smoke_refused", "unit": unit, "why": str(e)})


def cmd_chunk(args):
    """ONE heavy slot's work: units in queue order until the next one's estimated end passes the budget."""
    cells = _cells_arg(args)
    budget = float(args[args.index("--budget-s") + 1]) if "--budget-s" in args else DEFAULT_BUDGET_S
    t0 = time.time()
    run_smokes()
    walls = {}
    n = 0
    jl({"event": "chunk_start", "pid": os.getpid(), "cells": cells, "budget_s": budget,
        "slot": os.environ.get("GEN3AI_HEAVY_SLOT_HELD")})
    for unit, cell, seed in _pending(cells):
        kind = CELLS[cell]["kind"]
        est = max(walls.get(kind, [DEFAULT_UNIT_S[kind]])[-3:]) * 1.15
        if time.time() - t0 + est > budget:
            break
        rc, wall = run_unit(unit, cell, seed, tag="chunk")
        n += 1
        if is_done(unit):
            walls.setdefault(kind, []).append(wall)
        else:
            _bump_fail(unit)
    jl({"event": "chunk_end", "pid": os.getpid(), "units_run": n, "wall_s": round(time.time() - t0, 1)})


def cmd_run(args):
    cells = _cells_arg(args)
    tag = "_".join(cells)
    stop_at = dt.datetime.fromisoformat(args[args.index("--stop-at") + 1]).timestamp()
    jl({"event": "supervisor_start", "pid": os.getpid(), "stop_at": stop_at, "cells": cells})
    me = str(Path(__file__).resolve())
    idle = 0
    while True:
        pending = _pending(cells)
        if not pending:
            jl({"event": "supervisor_done", "fails": _fails(), "cells": cells})
            (STATE / f"queue_{tag}.done").write_text(json.dumps({"done": True, "fails": _fails()}) + "\n")
            return
        left = stop_at - time.time()
        if left < 300:
            jl({"event": "supervisor_stop_at", "pending": len(pending)})
            (STATE / f"queue_{tag}.done").write_text("stop_at\n")
            return
        budget = min(DEFAULT_BUDGET_S, left - 120)
        # the OUTER timeout bounds the slot wait (FIFO) by the registered stop; the INNER one is the 45-min chunk
        argv = ["timeout", str(int(left)), "nice", "-n", "19", "taskset", "-c", CORES, str(MEMCAP),
                "--name", "x51", MEM_GB, "timeout", str(CHUNK_TIMEOUT_S), PY_T28, me, "chunk",
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
            (STATE / f"queue_{tag}.done").write_text(f"stuck rc={rc}\n")
            return
        if after == before:
            time.sleep(60)


def cmd_status(args):
    for cell in _cells_arg(args):
        us = units([cell])
        print(f"{cell:16s} {sum(is_done(u) for u, _, _ in us)}/{len(us)} units done")


def cmd_one(args):
    """A smoke: one unit (or `--games N` games of it) outside the chunk loop; the caller supplies the slot."""
    by = {u[0]: u for u in units(list(CELLS))}
    unit, cell, seed = by[args[0]]
    games = int(args[args.index("--games") + 1]) if "--games" in args else GAMES
    name = unit if games == GAMES else f"smoke_{unit}_g{games}"
    rc, _ = run_unit(name, cell, seed, dry="--dry-run" in args, games=games, tag="one")
    sys.exit(rc)


if __name__ == "__main__":
    for d in ("cells", "logs"):
        (STATE / d).mkdir(parents=True, exist_ok=True)
    {"run": cmd_run, "chunk": cmd_chunk, "status": cmd_status, "one": cmd_one}[sys.argv[1]](sys.argv[2:])
