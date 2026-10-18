"""X37a — the detached, resumable, ONE-lane unit queue for the evaluation matrix's cells (2026-10-18).

`heldout72_hpfix_kakuna_2026-10-18/scripts/hq_hpfix.py` generalised to X37a's cells (PREREG.md §2). Changes:
  * per-cell configs and per-cell team sets on EACH side (the deployment, reverse and calibration cells are
    DECLARED asymmetric); every team file on BOTH sides is sha256-checked against its committed manifest, and
    the config against the registration, before every unit;
  * every unit passes `--capture-dir <unit>/capture` (the front end's per-battle repro record, whose START
    command carries BOTH packed teams): the ground truth of which team Kakuna piloted in each game (F-DV-6);
    it changes no stamp and no play (the seed comes from `--seed-base`);
  * the K(P, H) calibration cell (Kakuna vs Kakuna) runs from the tree of the commit that lands B-D9 / B-D6
    (`$X37A_BUILD_TREE`), every X26 cell from a detached `63f97a57` (the earlier reads' commit, so stamps match).

A UNIT is one `python -m main.anchors` invocation (20 games, role-balanced halves, --team-seed == --seed-base ==
the unit seed) run FROM ITS CELL'S PINNED CHECKOUT ROOT under `nice -n 19 taskset -c 5-7,13-15 mem_cap.sh 12`,
one at a time (rule 5b; X26 is live). DONE iff summary.json says status OK and n == 20 (whether it COUNTS is
the aggregate's call, `unit_counts`); a partial dir is moved aside and re-run (at most 2 tries).

    python xq.py run --cells deploy_core,deploy_near,deploy_far,reverse [--stop-at ISO]
    python xq.py status
    python xq.py one <unit> [--dry-run]
"""
import datetime as dt
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]                      # the measurement dir (in the worktree)
REG = json.loads((HERE / "x37a_stage1_registration.json").read_text())
STATE = Path(os.environ.get("X37A_STATE", "/home/goodlad/dev/gen3ai-reads/x37a_2026-10-18"))
MAIN = Path("/home/goodlad/dev/gen3ai")
MEAS = MAIN / "designs" / "research_state" / "measurements"
PIN_X26 = Path("/home/goodlad/.cache/gen3ai/x37a/pin_63f97a57")
BUILD_TREE = Path(os.environ.get("X37A_BUILD_TREE", "/home/goodlad/.cache/gen3ai/x37a/pin_build"))
PY_T28 = "/home/goodlad/miniconda3/envs/gen3ai_torch28/bin/python3"
X26 = "/home/goodlad/dev/gen3ai-reads/x26_metamon_anchor_2026-10-17/model/checkpoint_161000169_steps.zip"
X26_SHA = "bc6050d1ca254abd9866945f3dde74a2189514a551e83c29abb79b2a20e52992"
METAMON_TEAMS = Path("/home/goodlad/dev/metamon/cache/teams")
PORT = 9537
CORES = "5-7,13-15"
MEM_GB = "12"
GAMES = 20


def _manifests():
    """stem -> sha256 for every team set a cell reads (committed manifests; byte copies on both sides)."""
    hp = {s: r["sha256"] for s, r in json.loads(
        (HERE.parent / "heldout72_hpfix_kakuna_2026-10-18" / "heldout72_hpfix_manifest.json").read_text())["teams"].items()}
    h72 = json.loads((HERE.parent / "x35_breakers_stage_2026-10-17" / "home72_manifest.json").read_text())["our_teams"]
    out = {"home72": h72, "heldout72_hpfix": hp}
    for s, sub in REG["metamon_sub_sets"].items():
        out[sub["metamon_set"]] = {m: hp[m] for m in sub["members"]}
    for panel in ("forum72", "p72"):        # stage 2 (their manifests land before any stage-2 game)
        p = HERE / f"{panel}_manifest.json"
        if p.exists():
            out[panel] = {s: r["sha256"] for s, r in json.loads(p.read_text())["teams"].items()}
    return out


# side spec: (directory, extension, manifest key)
OURS_H72 = (STATE / "teams_home72", ".txt", "home72")
OURS_HP = (STATE / "teams_heldout72_hpfix", ".txt", "heldout72_hpfix")
MM_H72 = (METAMON_TEAMS / "gen3ai_home72" / "gen3ou", ".gen3ou_team", "home72")
MM_HP = (METAMON_TEAMS / "gen3ai_heldout72_hpfix" / "gen3ou", ".gen3ou_team", "heldout72_hpfix")


def _mm_sub(s):
    name = REG["metamon_sub_sets"][s]["metamon_set"]
    return (METAMON_TEAMS / name / "gen3ou", ".gen3ou_team", name)


# cell -> dict(tree, kind, config, seeds, sides)
CELLS = {
    "deploy_core": dict(tree=PIN_X26, kind="x26", config="deploy_core", seeds=list(range(20, 25)),
                        sides=[OURS_H72, _mm_sub("CORE")]),
    "deploy_near": dict(tree=PIN_X26, kind="x26", config="deploy_near", seeds=list(range(0, 20)),
                        sides=[OURS_H72, _mm_sub("NEAR")]),
    "deploy_far": dict(tree=PIN_X26, kind="x26", config="deploy_far", seeds=list(range(25, 30)),
                       sides=[OURS_H72, _mm_sub("FAR")]),
    "reverse": dict(tree=PIN_X26, kind="x26", config="reverse", seeds=list(range(0, 20)),
                    sides=[OURS_HP, MM_H72]),
    "kcal": dict(tree=BUILD_TREE, kind="kcal", config="kcal", seeds=list(range(0, 20)),
                 sides=[MM_H72, MM_HP]),
    # stage 2 (symmetric; PREREG.md §4): our copy + Metamon set, each sha-checked against the panel manifest
    "forum72": dict(tree=PIN_X26, kind="x26", config="forum72", seeds=list(range(0, 20)),
                    sides=[(STATE / "teams_forum72", ".txt", "forum72"),
                           (METAMON_TEAMS / "gen3ai_forum72" / "gen3ou", ".gen3ou_team", "forum72")]),
    "p72": dict(tree=PIN_X26, kind="x26", config="p72", seeds=list(range(0, 20)),
                sides=[(STATE / "teams_p72", ".txt", "p72"),
                       (METAMON_TEAMS / "gen3ai_p72" / "gen3ou", ".gen3ou_team", "p72")]),
}


def config_path(cell):
    return HERE / "configs" / f"anchors_x37a_{CELLS[cell]['config']}.json"


def config_sha(cell):
    name = CELLS[cell]["config"]
    if name in REG["configs"]:
        return REG["configs"][name]["sha256"]
    reg2 = HERE / "x37a_stage2_registration.json"
    return json.loads(reg2.read_text())["configs"][name]["sha256"]


def units(cells):
    """Interleaved by progress fraction, so a time-boxed stop leaves every cell with whole units in proportion."""
    out = []
    for name in cells:
        seeds = CELLS[name]["seeds"]
        for i, s in enumerate(seeds):
            out.append(((i + 0.5) / len(seeds), name, f"{name}_{s:02d}", s))
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


def preflight(cell):
    c = CELLS[cell]
    if c["kind"] == "x26" and sha256(X26) != X26_SHA:
        raise SystemExit(f"model sha256 mismatch: {X26}")
    if sha256(config_path(cell)) != config_sha(cell):
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
    r = subprocess.run([PY_T28, "-m", "main.live.halt", "status"], cwd=str(MAIN),
                       env={**os.environ, "PYTHONPATH": str(MAIN / "src"), "CUDA_VISIBLE_DEVICES": ""},
                       capture_output=True, text=True)
    if r.returncode != 0 or "clear" not in r.stdout:
        raise SystemExit(f"T28 halt marker is not clear: {r.stdout} {r.stderr}")


def argv_for(unit, cell, seed, dry=False):
    c = CELLS[cell]
    out = STATE / "cells" / unit
    common = ["--regime", "greedy", "--games", str(GAMES), "--team-seed", str(seed), "--seed-base", str(seed),
              "--device", "cpu", "--port", str(PORT), "--nice", "19", "--out", str(out),
              "--capture-dir", str(out / "capture")]
    if c["kind"] == "x26":
        a = [PY_T28, "-m", "main.anchors", "--model", X26, "--opponent", "metamon:Kakuna", "--teamset", "home",
             *common]
    else:   # K(P, H): side A (reported) = Kakuna piloting home72 (--our-teamset away), side B = Kakuna on H
        a = [PY_T28, "-m", "main.anchors", "--opponent-a", "metamon:Kakuna", "--opponent-b", "metamon:Kakuna",
             "--teamset", "home", "--our-teamset", "away", "--declare-asymmetric-teams",
             "--progress-timeout", "1800", *common]
    if dry:
        a.append("--dry-run")
    return ["nice", "-n", "19", "taskset", "-c", CORES,
            str(MAIN / "scripts" / "ops" / "mem_cap.sh"), "--name", "x37a", MEM_GB, *a]


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
        "NUMEXPR_NUM_THREADS": "1", "PYTHONUNBUFFERED": "1",
        "GEN3AI_ANCHORS_CONFIG": str(config_path(cell)),
    })
    return env


def run_unit(unit, cell, seed, dry=False):
    preflight(cell)
    out = STATE / "cells" / unit
    if out.exists() and not dry:
        out.rename(out.with_name(f"{unit}.partial.{int(time.time())}"))
    argv = argv_for(unit, cell, seed, dry)
    tree = CELLS[cell]["tree"]
    jl({"event": "start", "unit": unit, "cell": cell, "seed": seed, "tree": str(tree), "argv": argv,
        "load": os.getloadavg()})
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
    jl({"event": "end", "unit": unit, "cell": cell, "rc": rc, "done": is_done(unit),
        "wall_s": round(time.time() - t0, 1),
        "load1_mean": round(sum(loads) / len(loads), 2) if loads else None,
        "load1_max": round(max(loads), 2) if loads else None})
    return rc


def _cells_arg(args):
    return args[args.index("--cells") + 1].split(",") if "--cells" in args else list(CELLS)


def cmd_run(args):
    cells = _cells_arg(args)
    tag = "_".join(cells)
    stop_at = None
    if "--stop-at" in args:
        stop_at = dt.datetime.fromisoformat(args[args.index("--stop-at") + 1]).timestamp()
    jl({"event": "supervisor_start", "pid": os.getpid(), "stop_at": stop_at, "cells": cells})
    fails = {}
    for unit, cell, seed in units(cells):
        while not is_done(unit) and fails.get(unit, 0) < 2:
            if stop_at and time.time() > stop_at:
                jl({"event": "supervisor_stop_at"})
                (STATE / f"queue_{tag}.done").write_text("stop_at\n")
                return
            run_unit(unit, cell, seed)
            if not is_done(unit):
                fails[unit] = fails.get(unit, 0) + 1
    jl({"event": "supervisor_done", "fails": fails, "cells": cells})
    (STATE / f"queue_{tag}.done").write_text(json.dumps({"done": True, "fails": fails}) + "\n")


def cmd_status(args):
    for cell in _cells_arg(args):
        us = units([cell])
        print(f"{cell:12s} {sum(is_done(u) for u, _, _ in us)}/{len(us)} units done")


def cmd_one(args):
    by = {u[0]: u for u in units(list(CELLS))}
    unit, cell, seed = by[args[0]]
    sys.exit(run_unit(unit, cell, seed, dry="--dry-run" in args))


if __name__ == "__main__":
    for d in ("cells", "logs"):
        (STATE / d).mkdir(parents=True, exist_ok=True)
    {"run": cmd_run, "status": cmd_status, "one": cmd_one}[sys.argv[1]](sys.argv[2:])
