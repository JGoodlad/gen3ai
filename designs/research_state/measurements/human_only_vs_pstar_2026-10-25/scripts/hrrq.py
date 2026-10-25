"""FR-1 / FR-3: P* vs Metamon's human-only models, the detached, resumable unit queue (2026-10-25).

`../../metamon_strongest_2026-10-25/scripts/mmq.py` (itself X51's `x51q.py`) with these changes:
  * the opponents are `metamon:SmallRL` (FR-1; X51's committed symmetric configs, unchanged: they already pin it)
    and `metamon:LargeRL` / `metamon:LargeIL` (FR-3; X51's configs plus the two agents, `make_configs.py`);
  * every unit runs from ONE detached checkout of `d32ffcee` (`$HRRQ_TREE`), the commit the metamon_strongest read
    plays at, whose identity unit replayed X51's `pstar_home72_00` 20 / 20 (no identity unit here);
  * `HF_HUB_OFFLINE=1`: the peers resolve their weights from the cached hub revision `ac8e88e7` and never ask the hub;
  * TWO lanes, a per-unit claim lock (`claims/<unit>.lock`, an exclusive flock held for the unit's life); lane B
    takes FR-3 units only, lane A takes FR-1 first;
  * each unit is its own `mem_cap.sh` job: the short lane (`--max-s 600`) when the cell's measured unit fits it, else
    a long capped job (`timeout 2700`); a short unit hard-killed at 630 s (exit 88) is re-run once as a long job.

A UNIT is one `python -m main.anchors` call: 20 games, role-balanced halves, --team-seed == --seed-base == the unit
seed (0-19), so every game twins X51's P* vs Kakuna game of the same panel, seed, half and index. DONE iff
summary.json says status OK and n == 20 (whether it COUNTS is `aggregate.py`'s call); a partial dir is moved aside
and re-run (at most 2 tries).

    python hrrq.py run A|B   # one lane's supervisor (background); resumable: it skips every DONE unit
    python hrrq.py status
"""
import fcntl
import hashlib
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

STATE = Path(os.environ.get("HRRQ_STATE", "/home/goodlad/dev/gen3ai-reads/hr_reads_2026-10-25"))
TREE = Path(os.environ.get("HRRQ_TREE", "/home/goodlad/.cache/gen3ai/hr_reads/pin"))
MAIN = Path("/home/goodlad/dev/gen3ai")
X51_STATE = Path("/home/goodlad/dev/gen3ai-reads/x51_2026-10-20")
MEAS = MAIN / "designs" / "research_state" / "measurements"
X51_MEAS = MEAS / "x51_plateau_vs_kakuna_2026-10-20"
PY_T28 = "/home/goodlad/miniconda3/envs/gen3ai_torch28/bin/python3"
MODEL = (X51_STATE / "model" / "checkpoint_280000170_steps.zip",
         "814823c514fb791e3c6e671ba126f7d0fd90733098b0539bb85babe7c5ab91ed")
PANELS = ("home72", "heldout72_hpfix", "forum72", "p72")
SHORT = {"heldout72_hpfix": "hpfix"}
METAMON_TEAMS = Path("/home/goodlad/dev/metamon/cache/teams")
CORES = "5-7,13-15"
MEM_GB = "12"
GAMES = 20
SHORT_S = 600
LONG_S = 2700
#: per opponent: (cell prefix, config family, short lane?) — FR-3's lane choice is the feasibility smoke's (PREREG §4)
OPP = {"metamon:SmallRL": ("sr", "x51_sym", True),
       "metamon:LargeRL": ("lrl", "hr", True),
       "metamon:LargeIL": ("lil", "hr", True)}
FR1 = ("metamon:SmallRL",)
FR3 = ("metamon:LargeRL", "metamon:LargeIL")


def config_path(family, panel):
    if family == "x51_sym":
        return X51_MEAS / "configs" / f"anchors_x51_sym_{panel}.json"
    return STATE / "configs" / f"anchors_hr_{panel}.json"


def config_sha():
    """The registered sha256 of every config the queue may pass (PREREG §6; `CONFIG_SHA.json` beside this file)."""
    return json.loads((Path(__file__).resolve().parent / "CONFIG_SHA.json").read_text())


def _manifests():
    def teams(rel):
        return {s: r["sha256"] for s, r in json.loads((MEAS / rel).read_text())["teams"].items()}
    return {
        "home72": json.loads((MEAS / "x35_breakers_stage_2026-10-17" / "home72_manifest.json").read_text())["our_teams"],
        "heldout72_hpfix": teams("heldout72_hpfix_kakuna_2026-10-18/heldout72_hpfix_manifest.json"),
        "forum72": teams("x37a_eval_matrix_2026-10-18/forum72_manifest.json"),
        "p72": teams("x37a_eval_matrix_2026-10-18/p72_manifest.json"),
    }


def sides(panel):
    return [(X51_STATE / f"teams_{panel}", ".txt", panel),
            (METAMON_TEAMS / f"gen3ai_{panel}" / "gen3ou", ".gen3ou_team", panel)]


CELLS = {}
for _opp, (_pre, _fam, _short) in OPP.items():
    for _p in PANELS:
        CELLS[f"{_pre}_{SHORT.get(_p, _p)}"] = dict(opponent=_opp, panel=_p, family=_fam, short=_short,
                                                   seeds=list(range(20)))


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


def is_done(unit):
    try:
        s = json.loads((STATE / "cells" / unit / "summary.json").read_text())
    except (OSError, ValueError):
        return False
    return s.get("status") == "OK" and s.get("n") == GAMES


def _interleave(opps):
    cells = [c for c, v in CELLS.items() if v["opponent"] in opps]
    order = sorted(((i + 0.5) / 20, cells.index(c), c, s) for c in cells for i, s in enumerate(CELLS[c]["seeds"]))
    return [(f"{c}_{s:02d}", c, s) for _, _, c, s in order]


def units(lane="A"):
    """Lane A: FR-1 (four panels interleaved by progress), then FR-3. Lane B: FR-3 only (LargeRL and LargeIL
    interleaved, so a stop leaves every cell with whole units in proportion)."""
    return (_interleave(FR1) if lane == "A" else []) + _interleave(FR3)


def _fails():
    p = STATE / "fails.json"
    return json.loads(p.read_text()) if p.exists() else {}


def _bump_fail(unit):
    with open(STATE / "fails.lock", "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        f = _fails()
        f[unit] = f.get(unit, 0) + 1
        (STATE / "fails.json").write_text(json.dumps(f, indent=1) + "\n")


def preflight(cell):
    c = CELLS[cell]
    if sha256(MODEL[0]) != MODEL[1]:
        raise SystemExit(f"model sha256 mismatch: {MODEL[0]}")
    path = config_path(c["family"], c["panel"])
    want = config_sha()[f"{c['family']}_{c['panel']}"]
    if sha256(path) != want:
        raise SystemExit(f"config sha256 mismatch: {path}")
    man = _manifests()
    for d, ext, key in sides(c["panel"]):
        teams = man[key]
        if sorted(p.name for p in d.iterdir()) != sorted(stem + ext for stem in teams):
            raise SystemExit(f"{key}: file set differs from the manifest in {d}")
        for stem, w in teams.items():
            if sha256(d / (stem + ext)) != w:
                raise SystemExit(f"{key}: sha256 mismatch {d / (stem + ext)}")
    weights = json.loads((Path(__file__).resolve().parent / "WEIGHTS.json").read_text())
    agent = c["opponent"].split(":", 1)[1]
    if agent in weights and sha256(weights[agent]["path"]) != weights[agent]["sha256"]:
        raise SystemExit(f"{agent}: weights sha256 mismatch {weights[agent]['path']}")
    if not (TREE / "src" / "main" / "anchors" / "cli.py").exists():
        raise SystemExit(f"tree missing: {TREE}")
    r = subprocess.run([PY_T28, "-m", "main.live.halt", "status"], cwd=str(TREE),
                       env={**os.environ, "PYTHONPATH": str(TREE / "src"), "CUDA_VISIBLE_DEVICES": ""},
                       capture_output=True, text=True)
    if r.returncode != 0 or "clear" not in r.stdout:
        raise SystemExit(f"T28 halt marker is not clear: {r.stdout} {r.stderr}")


def claim_port(lo=9560, hi=9599):
    """``main.x35.kakuna.claim_port``'s rule, with its lock dir: an exclusive flock held for the unit's life."""
    lock_dir = Path.home() / ".claude" / "jobs" / "x35_kakuna_ports"
    lock_dir.mkdir(parents=True, exist_ok=True)
    for p in range(lo, hi + 1):
        fh = open(lock_dir / f"{p}.lock", "w")  # noqa: SIM115 - held by the caller
        try:
            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            fh.close()
            continue
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind(("127.0.0.1", p))
            except OSError:
                fh.close()
                continue
        return p, fh
    raise SystemExit("no free, unclaimed 95xx port")


def claim_unit(unit):
    (STATE / "claims").mkdir(exist_ok=True)
    fh = open(STATE / "claims" / f"{unit}.lock", "w")  # noqa: SIM115 - held by the caller
    try:
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        fh.close()
        return None
    return fh


def argv_for(unit, cell, seed, port, short):
    c = CELLS[cell]
    out = STATE / "cells" / unit
    anchors = [PY_T28, "-m", "main.anchors", "--model", str(MODEL[0]), "--opponent", c["opponent"],
               "--teamset", "home", "--regime", "greedy", "--games", str(GAMES), "--team-seed", str(seed),
               "--seed-base", str(seed), "--device", "cpu", "--port", str(port), "--nice", "19",
               "--out", str(out), "--capture-dir", str(out / "capture")]
    memcap = [str(TREE / "scripts" / "ops" / "mem_cap.sh"), "--name", "hrrq"]
    memcap += ["--max-s", str(SHORT_S), MEM_GB] if short else [MEM_GB, "timeout", str(LONG_S)]
    return [*memcap, "nice", "-n", "19", "taskset", "-c", CORES, *anchors]


def env_for(cell):
    env = dict(os.environ)
    for k in ("POKESIM_SIM_BRIDGE_BIN", "POKESIM_SEARCH_DRIVER_BIN", "POKESIM_CORE_EVENTS_BIN",
              "CARGO_TARGET_DIR", "GEN3AI_ANCHORS_CONFIG", "GEN3AI_BLOCKING"):
        env.pop(k, None)
    c = CELLS[cell]
    env.update({
        "PYTHONPATH": str(TREE / "src"), "CUDA_VISIBLE_DEVICES": "", "HF_HUB_OFFLINE": "1",
        "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1", "TORCH_NUM_THREADS": "1", "PYTHONUNBUFFERED": "1",
        "GEN3AI_ANCHORS_CONFIG": str(config_path(c["family"], c["panel"])),
    })
    return env


def run_unit(unit, cell, seed, short):
    preflight(cell)
    out = STATE / "cells" / unit
    if out.exists():
        out.rename(out.with_name(f"{unit}.partial.{int(time.time())}"))
    port, lock = claim_port()
    argv = argv_for(unit, cell, seed, port, short)
    jl({"event": "start", "unit": unit, "cell": cell, "seed": seed, "port": port, "short": short, "argv": argv,
        "load": os.getloadavg()})
    t0 = time.time()
    with open(STATE / "logs" / f"{unit}.log", "ab") as log:
        proc = subprocess.Popen(argv, cwd=str(TREE), env=env_for(cell), stdout=log, stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL, start_new_session=True)
        jl({"event": "pid", "unit": unit, "pid": proc.pid})
        rc = proc.wait()
    lock.close()
    jl({"event": "end", "unit": unit, "rc": rc, "done": is_done(unit), "wall_s": round(time.time() - t0, 1),
        "short": short})
    return rc


def cmd_run(args):
    lane = args[0] if args else "A"
    (STATE / f"supervisor_{lane}.pid").write_text(f"{os.getpid()}\n")
    jl({"event": "supervisor_start", "lane": lane, "pid": os.getpid(), "tree": str(TREE)})
    while True:
        fails = _fails()
        pending = [u for u in units(lane) if not is_done(u[0]) and fails.get(u[0], 0) < 2]
        if not pending:
            jl({"event": "supervisor_done", "lane": lane, "fails": fails})
            (STATE / f"queue_{lane}.done").write_text(json.dumps({"done": True, "fails": fails}) + "\n")
            return
        claimed = None
        for unit, cell, seed in pending:
            fh = claim_unit(unit)
            if fh is not None:
                if is_done(unit):
                    fh.close()
                    continue
                claimed = (unit, cell, seed, fh)
                break
        if claimed is None:                          # every pending unit is the other lane's: wait for it
            time.sleep(60)
            continue
        unit, cell, seed, fh = claimed
        try:
            short = CELLS[cell]["short"]
            rc = run_unit(unit, cell, seed, short=short)
            if short and rc == 88 and not is_done(unit):   # past the short lane's deadline: once more as a long job
                jl({"event": "short_lane_overrun", "unit": unit})
                rc = run_unit(unit, cell, seed, short=False)
        except SystemExit as e:                      # a preflight refusal stops this lane: never play around it
            jl({"event": "preflight_refused", "lane": lane, "unit": unit, "why": str(e)})
            (STATE / f"queue_{lane}.done").write_text(f"refused: {e}\n")
            return
        finally:
            fh.close()
        if not is_done(unit):
            _bump_fail(unit)


def cmd_status(_args):
    us = {u for lane in "AB" for u in units(lane)}
    by = {}
    for u, c, _ in us:
        by.setdefault(c, [0, 0])
        by[c][1] += 1
        by[c][0] += is_done(u)
    print(json.dumps({c: f"{d}/{n}" for c, (d, n) in sorted(by.items())}), "fails:", _fails())


if __name__ == "__main__":
    for d in ("cells", "logs", "claims"):
        (STATE / d).mkdir(parents=True, exist_ok=True)
    {"run": cmd_run, "status": cmd_status}[sys.argv[1]](sys.argv[2:])
