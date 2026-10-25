"""metamon_strongest: P* vs `PastaMittens`, the detached, resumable, ONE-lane unit queue (2026-10-25).

X51's `x51q.py` (`../../x51_plateau_vs_kakuna_2026-10-20/scripts/x51q.py`) with four changes:
  * the opponent is `metamon:PastaMittens` (the Kakuna router) on four panels, from configs `make_configs.py` wrote;
  * every unit runs from ONE detached checkout of the commit that lands the router instrument (`$MMQ_TREE`);
  * the SHORT LANE, per unit: each unit is its own `mem_cap.sh --max-s 600` job (rule 5b; it jumps queued 45-min
    chunks), on a 95xx port claimed with a held flock (`main.x35.kakuna.claim_port`). A unit hard-killed at
    630 s (exit 88) is re-run once as a normal capped job (`timeout 2700`), never shortened;
  * PREREG precondition 4 FIRST: the identity unit `ident_home72_00` (P* vs Kakuna ckpt 34, home72, seed 0, X51's
    own config) must replay X51's `pstar_home72_00` game for game (winner and turns, 20 / 20). If it does not, the
    registered fallback cell `kakctl_home72` (P* vs Kakuna, home72, seeds 0-19) joins the queue.

A UNIT is one `python -m main.anchors` call: 20 games, role-balanced halves, --team-seed == --seed-base == the
unit seed (0-19, so every unit twins X51's P* unit of the same seed). DONE iff summary.json says status OK and
n == 20 (whether it COUNTS is `aggregate.py`'s call); a partial dir is moved aside and re-run (at most 2 tries).

    python mmq.py run        # the supervisor (background); resumable: it skips every DONE unit
    python mmq.py status
"""
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

STATE = Path(os.environ.get("MMQ_STATE", "/home/goodlad/dev/gen3ai-reads/mm_strongest_2026-10-25"))
TREE = Path(os.environ.get("MMQ_TREE", "/home/goodlad/.cache/gen3ai/mm_strongest/pin"))
MAIN = Path("/home/goodlad/dev/gen3ai")
X51_STATE = Path("/home/goodlad/dev/gen3ai-reads/x51_2026-10-20")
X51_MEAS = MAIN / "designs" / "research_state" / "measurements" / "x51_plateau_vs_kakuna_2026-10-20"
MEAS = MAIN / "designs" / "research_state" / "measurements"
PY_T28 = "/home/goodlad/miniconda3/envs/gen3ai_torch28/bin/python3"
MODEL = (X51_STATE / "model" / "checkpoint_280000170_steps.zip",
         "814823c514fb791e3c6e671ba126f7d0fd90733098b0539bb85babe7c5ab91ed")
CONFIGS = {   # name -> (path, sha256)
    "x51_sym_home72": (X51_MEAS / "configs" / "anchors_x51_sym_home72.json",
                       "f99f2360fe45e5261c53c7682324117d1e930cf8b4c24f24d7f4aa06724bb304"),
    "mm_home72": (STATE / "configs" / "anchors_mm_home72.json",
                  "2c69c2ddd900b629e5c7796816807bee2b02c79280db34f1c3b9dfe47f1e6f33"),
    "mm_heldout72_hpfix": (STATE / "configs" / "anchors_mm_heldout72_hpfix.json",
                           "650bd037dc1dd4c12cbbdc3aaa8273c560e35487cf4803c6920c8360db4bb954"),
    "mm_forum72": (STATE / "configs" / "anchors_mm_forum72.json",
                   "bb442f3b606c874425c5be3cf1c98df67067c48dc29b915987e788a002e0899a"),
    "mm_p72": (STATE / "configs" / "anchors_mm_p72.json",
               "7397ceffcd9d2f0a10057589b911a4ce6642e81dfcb4dbc6b1f420bb21efb5fe"),
}
METAMON_TEAMS = Path("/home/goodlad/dev/metamon/cache/teams")
CORES = "5-7,13-15"
MEM_GB = "12"
GAMES = 20
SHORT_S = 600
LONG_S = 2700


def _manifests():
    def teams(rel):
        return {s: r["sha256"] for s, r in json.loads((MEAS / rel).read_text())["teams"].items()}
    return {
        "home72": json.loads((MEAS / "x35_breakers_stage_2026-10-17" / "home72_manifest.json").read_text())["our_teams"],
        "heldout72_hpfix": teams("heldout72_hpfix_kakuna_2026-10-18/heldout72_hpfix_manifest.json"),
        "forum72": teams("x37a_eval_matrix_2026-10-18/forum72_manifest.json"),
        "p72": teams("x37a_eval_matrix_2026-10-18/p72_manifest.json"),
    }


def _sides(s):
    return [(X51_STATE / f"teams_{s}", ".txt", s), (METAMON_TEAMS / f"gen3ai_{s}" / "gen3ou", ".gen3ou_team", s)]


def _cell(opponent, panel, config, seeds):
    return dict(opponent=opponent, panel=panel, config=config, seeds=seeds, sides=_sides(panel))


CELLS = {
    "ident_home72": _cell("metamon:Kakuna", "home72", "x51_sym_home72", [0]),
    "pm_home72": _cell("metamon:PastaMittens", "home72", "mm_home72", list(range(20))),
    "pm_hpfix": _cell("metamon:PastaMittens", "heldout72_hpfix", "mm_heldout72_hpfix", list(range(20))),
    "pm_forum72": _cell("metamon:PastaMittens", "forum72", "mm_forum72", list(range(20))),
    "pm_p72": _cell("metamon:PastaMittens", "p72", "mm_p72", list(range(20))),
    # PREREG precondition 4's fallback: queued only if the identity unit does not replay X51.
    "kakctl_home72": _cell("metamon:Kakuna", "home72", "x51_sym_home72", list(range(20))),
}
PM = ("pm_home72", "pm_hpfix", "pm_forum72", "pm_p72")


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


def _games(path):
    return {(r["half"], r["index"]): (r["result"], r["turns"])
            for r in map(json.loads, Path(path).read_text().splitlines())}


def identity():
    """PREREG precondition 4: ``None`` until the identity unit is done; else its verdict (written to ident.json)."""
    if not is_done("ident_home72_00"):
        return None
    ours = _games(STATE / "cells" / "ident_home72_00" / "games.jsonl")
    x51 = _games(X51_STATE / "cells" / "pstar_home72_00" / "games.jsonl")
    same = sorted(k for k in x51 if ours.get(k) == x51[k])
    out = {"identical": len(same) == len(x51) == len(ours) == GAMES, "n_same": len(same), "n": len(x51),
           "differ": {f"{h}/{i}": {"x51": x51[(h, i)], "here": ours.get((h, i))}
                      for (h, i) in sorted(x51) if ours.get((h, i)) != x51[(h, i)]}}
    (STATE / "ident.json").write_text(json.dumps(out, indent=1) + "\n")
    return out


def units():
    """The identity unit, then the four panels interleaved by progress (a stop leaves each with whole units in
    proportion), then (if precondition 4 failed) the Kakuna control."""
    out = [("ident_home72_00", "ident_home72", 0)]
    ident = identity()
    if ident is None:
        return out
    order = sorted(((i + 0.5) / 20, k, c, s) for c in PM for i, s in enumerate(CELLS[c]["seeds"])
                   for k in [PM.index(c)])
    out += [(f"{c}_{s:02d}", c, s) for _, _, c, s in order]
    if not ident["identical"]:
        out += [(f"kakctl_home72_{s:02d}", "kakctl_home72", s) for s in range(20)]
    return out


def _fails():
    p = STATE / "fails.json"
    return json.loads(p.read_text()) if p.exists() else {}


def _bump_fail(unit):
    f = _fails()
    f[unit] = f.get(unit, 0) + 1
    (STATE / "fails.json").write_text(json.dumps(f, indent=1) + "\n")


def preflight(cell):
    c = CELLS[cell]
    if sha256(MODEL[0]) != MODEL[1]:
        raise SystemExit(f"model sha256 mismatch: {MODEL[0]}")
    path, want = CONFIGS[c["config"]]
    if sha256(path) != want:
        raise SystemExit(f"config sha256 mismatch: {path}")
    man = _manifests()
    for d, ext, key in c["sides"]:
        teams = man[key]
        if sorted(p.name for p in d.iterdir()) != sorted(stem + ext for stem in teams):
            raise SystemExit(f"{key}: file set differs from the manifest in {d}")
        for stem, w in teams.items():
            if sha256(d / (stem + ext)) != w:
                raise SystemExit(f"{key}: sha256 mismatch {d / (stem + ext)}")
    if not (TREE / "src" / "main" / "anchors" / "cli.py").exists():
        raise SystemExit(f"tree missing: {TREE}")
    r = subprocess.run([PY_T28, "-m", "main.live.halt", "status"], cwd=str(TREE),
                       env={**os.environ, "PYTHONPATH": str(TREE / "src"), "CUDA_VISIBLE_DEVICES": ""},
                       capture_output=True, text=True)
    if r.returncode != 0 or "clear" not in r.stdout:
        raise SystemExit(f"T28 halt marker is not clear: {r.stdout} {r.stderr}")


def claim_port(lo=9540, hi=9599):
    """``main.x35.kakuna.claim_port``'s rule, with its lock dir: an exclusive flock held for the unit's life."""
    import socket
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


def argv_for(unit, cell, seed, port, short):
    c = CELLS[cell]
    out = STATE / "cells" / unit
    anchors = [PY_T28, "-m", "main.anchors", "--model", str(MODEL[0]), "--opponent", c["opponent"],
               "--teamset", "home", "--regime", "greedy", "--games", str(GAMES), "--team-seed", str(seed),
               "--seed-base", str(seed), "--device", "cpu", "--port", str(port), "--nice", "19",
               "--out", str(out), "--capture-dir", str(out / "capture")]
    memcap = [str(TREE / "scripts" / "ops" / "mem_cap.sh"), "--name", "mmq"]
    memcap += ["--max-s", str(SHORT_S), MEM_GB] if short else [MEM_GB, "timeout", str(LONG_S)]
    return [*memcap, "nice", "-n", "19", "taskset", "-c", CORES, *anchors]


def env_for(cell):
    env = dict(os.environ)
    for k in ("POKESIM_SIM_BRIDGE_BIN", "POKESIM_SEARCH_DRIVER_BIN", "POKESIM_CORE_EVENTS_BIN",
              "CARGO_TARGET_DIR", "GEN3AI_ANCHORS_CONFIG", "GEN3AI_BLOCKING"):
        env.pop(k, None)
    env.update({
        "PYTHONPATH": str(TREE / "src"), "CUDA_VISIBLE_DEVICES": "", "METAMON_ENSEMBLE_MEMBER_DEVICES": "cpu",
        "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1", "TORCH_NUM_THREADS": "1", "PYTHONUNBUFFERED": "1",
        "GEN3AI_ANCHORS_CONFIG": str(CONFIGS[CELLS[cell]["config"]][0]),
    })
    return env


def run_unit(unit, cell, seed, short=True):
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


def cmd_run(_args):
    (STATE / "supervisor.pid").write_text(f"{os.getpid()}\n")
    jl({"event": "supervisor_start", "pid": os.getpid(), "tree": str(TREE)})
    while True:
        fails = _fails()
        pending = [u for u in units() if not is_done(u[0]) and fails.get(u[0], 0) < 2]
        if not pending:
            ident = identity()
            jl({"event": "supervisor_done", "fails": fails, "ident": ident})
            (STATE / "queue.done").write_text(json.dumps({"done": True, "fails": fails, "ident": ident}) + "\n")
            return
        unit, cell, seed = pending[0]
        try:
            rc = run_unit(unit, cell, seed, short=True)
            if rc == 88 and not is_done(unit):        # past the short lane's deadline: once more as a long job
                jl({"event": "short_lane_overrun", "unit": unit})
                rc = run_unit(unit, cell, seed, short=False)
        except SystemExit as e:                      # a preflight refusal stops the queue: never play around it
            jl({"event": "preflight_refused", "unit": unit, "why": str(e)})
            (STATE / "queue.done").write_text(f"refused: {e}\n")
            return
        if not is_done(unit):
            _bump_fail(unit)


def cmd_status(_args):
    us = units()
    done = sum(is_done(u) for u, _, _ in us)
    print(f"{done}/{len(us)} units done; ident={identity()}; fails={_fails()}")


if __name__ == "__main__":
    for d in ("cells", "logs"):
        (STATE / d).mkdir(parents=True, exist_ok=True)
    {"run": cmd_run, "status": cmd_status}[sys.argv[1]](sys.argv[2:])
