r"""X59-S's READ OPERATIONS (``x59s_matched_students_2026-10-25/PREREG.md`` Amendment 4): the steps around the
``main.reads`` Kakuna lane, each resumable and each writing one artefact the registered rule reads
(``main.reads.rules_x59s``).

    python -m main.human_data.x59s_ops export   --students <trained dir> --out <read>/students \
        --human-rows <r2>/rows --selfplay-rows <ss_rows_v2>          # -> <read>/students/students.json
    python -m main.human_data.x59s_ops identity --state <kakuna state> --x51-rows <X51 rows.jsonl> --out <json>
    python -m main.human_data.x59s_ops floor    --state <dir> [--play] --out <json>   # bot:random / bot:aggressive
    python -m main.human_data.x59s_ops inhouse  --state <dir> --students <students.json> [--play] --out <json>

* ``export`` exports every trained student through ``main.human_data.student export`` into its OWN directory
  (``state.pt`` symlinked, the training archive untouched) with both yardsticks, then checks VOID 3 (one
  ``init_sha256`` per seed across the arms; every student at the registered step budget) and Amendment 2's
  learned-nothing guard, and reports m5 (a validation NLL that rose over the last epoch).
* ``identity`` compares a played unit with X51's banked unit game for game (``main.reads.assemble.GAME_KEYS``):
  VOID 5.
* ``floor`` plays the A2.6 floor references: ``main.x35.kakuna``'s unit, exactly, with ``--model <zip>`` replaced by
  ``--our-side bot:<name>`` (:func:`bot_argv`), units 0-3 of H, F and P72 against Kakuna; counted by the one rule
  (``main.anchors.results.unit_counts``).
* ``inhouse`` plays §4's head-to-heads (``main.h2h play --pairs 200 --seed <k> --panel <P>``, S-H_k as the player)
  and reads each pairing's rate: the equal-weight mean over the three panels of h2h's mirrored-pair score (a draw is
  1/2).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

SEEDS = (5901, 5902, 5903, 5904, 5905)
ARMS = {"S-H": "s_h", "S-P": "s_p", "S-S": "s_s", "S-P0": "s_p0"}
PRIMARY = ("S-H", "S-P", "S-S")
STEPS = 7041                                   # PREREG Amendment 2b: 3 x sum ceil(buffer rows / 512) on r2
PANELS = ("heldout72_hpfix", "forum72", "p72")
FLOOR_BOTS = ("random", "aggressive")
FLOOR_SEEDS = (0, 1, 2, 3)
INHOUSE = {"delta_data": ("S-H", "S-S"), "delta_labels": ("S-H", "S-P")}
PAIRS = 200


class OpsError(RuntimeError):
    """A precondition of a read step that does not hold (named)."""


def label(arm: str, seed: int) -> str:
    return f"{ARMS[arm]}_{seed}"


# ---------------------------------------------------------------------------------------------------- export
def last_epoch_rose(val_hist: Sequence[Sequence[float]], evals_per_epoch: int = 4) -> bool:
    """m5: the validation NLL at the end is above the one an epoch earlier (``val_hist`` = ``[[step, nll], ...]``)."""
    if len(val_hist) <= evals_per_epoch:
        return False
    return float(val_hist[-1][1]) > float(val_hist[-1 - evals_per_epoch][1])


def void3(sides: Mapping[str, Mapping[str, Any]], steps: int = STEPS) -> List[str]:
    """VOID 3 and Amendment 2's learned-nothing guard over the exported sidecars (``{label: sidecar}``): every
    reason a student or a seed is VOID (empty = none)."""
    bad: List[str] = []
    by_seed: Dict[int, Dict[str, str]] = {}
    for lab, s in sorted(sides.items()):
        if int(s["steps"]) != steps:
            bad.append(f"{lab}: {s['steps']} steps, registered {steps}")
        if not s["learned_guard"]["pass"]:
            bad.append(f"{lab}: the learned-nothing guard fails ({s['learned_guard']})")
        by_seed.setdefault(int(s["seed"]), {})[lab] = str(s["init_sha256"])
    for seed, inits in sorted(by_seed.items()):
        if len(set(inits.values())) != 1:
            bad.append(f"seed {seed}: init_sha256 differs across arms {inits}")
    return bad


def export_all(students: Path, out: Path, human_rows: Path, selfplay_rows: Path,
               arms: Sequence[str] = tuple(ARMS), only: Optional[str] = None) -> Dict[str, Any]:
    """Export each trained student (module docstring) and write ``<out>/students.json``. Resumable: a student whose
    sidecar exists is read, not re-exported."""
    from main.human_data import student as ST

    out.mkdir(parents=True, exist_ok=True)
    sides: Dict[str, Dict[str, Any]] = {}
    for arm in arms:
        for seed in SEEDS:
            lab = label(arm, seed)
            src, dst = students / lab, out / lab
            side_p = dst / "model" / "student_sidecar.json"
            if only is not None and lab not in only.split(","):
                if not side_p.exists():
                    continue
            elif not side_p.exists():
                dst.mkdir(parents=True, exist_ok=True)
                link = dst / "state.pt"
                if not link.is_symlink():
                    link.symlink_to((src / "state.pt").resolve())
                (dst / "rows_path.txt").write_text((src / "rows_path.txt").read_text())
                t0 = time.time()
                ST.export(dst, {"human_val_human_labels": (human_rows, "human"),
                                "selfplay_val_pstar_labels": (selfplay_rows, "selfplay")})
                print(f"[x59s_ops] exported {lab} ({time.time() - t0:.0f}s)", flush=True)
            s = json.loads(side_p.read_text())
            if s["arm"] != {"S-H": "human", "S-P": "pstar", "S-S": "selfplay", "S-P0": "pstar0"}[arm] or \
                    int(s["seed"]) != seed:
                raise OpsError(f"{side_p}: arm {s['arm']} seed {s['seed']} is not {arm} {seed}")
            s["zip_path"] = str(dst / "model" / Path(s["zip"]).name)
            s["val_rose_last_epoch"] = last_epoch_rose(s["val_hist"])
            sides[lab] = s
    if only is not None:
        return {"students": {k: sides[k] for k in only.split(",")}, "void3": [], "void3_sp0": []}
    doc = {"schema": "gen3_x59s_students_v1", "students": sides,
           "void3": void3({k: v for k, v in sides.items() if k.split("_")[1] != "p0"}),
           "void3_sp0": void3({k: v for k, v in sides.items() if k.split("_")[1] == "p0"})}
    (out / "students.json").write_text(json.dumps(doc, indent=1, sort_keys=True))
    return doc


# -------------------------------------------------------------------------------------------------- identity
def identity(state: Path, unit: str, x51_rows: Path, read_cell: str, unit_seed: int) -> Dict[str, Any]:
    """VOID 5: the play tree's unit ``unit`` against X51's banked unit, game for game."""
    from main.reads.assemble import GAME_KEYS, unit_games

    got = unit_games(state, unit)
    want = sorted(([r.get(k) for k in GAME_KEYS] for r in map(json.loads, x51_rows.read_text().splitlines())
                   if r.get("read_cell") == read_cell and int(r["unit_seed"]) == unit_seed),
                  key=lambda g: (str(g[0]), int(g[1])))
    eq = got is not None and len(want) == 20 and [list(g) for g in got] == want
    diff = [] if got is None else [{"x51": w, "here": g} for w, g in zip(want, got) if list(g) != w]
    return {"unit": unit, "x51_cell": read_cell, "unit_seed": unit_seed, "games_here": None if got is None else
            len(got), "games_x51": len(want), "game_keys": list(GAME_KEYS), "pass": bool(eq), "differing": diff}


# ----------------------------------------------------------------------------------------------------- floor
def bot_argv(argv: Sequence[str], bot: str) -> List[str]:
    """``main.x35.kakuna.unit_argv``'s command with ``--model <zip>`` replaced by ``--our-side bot:<bot>``; nothing
    else changes (the cell, the seeds, the port, the cap, the capture)."""
    a = list(argv)
    i = a.index("--model")
    return a[:i] + ["--our-side", f"bot:{bot}"] + a[i + 2:]


def floor_units() -> List[Any]:
    from main.x35 import kakuna as KA

    return [KA.Unit(name=f"bot_{b}__{ts}__{s:02d}", label=f"bot_{b}", arm=f"bot:{b}", seed=0, teamset=ts,
                    unit_seed=s, model="", model_sha256="")
            for s in FLOOR_SEEDS for ts in PANELS for b in FLOOR_BOTS]


def _play_bot(state: Path, u: Any, short: bool = True) -> int:
    """One attempt of a floor unit: ``kakuna._play`` with the bot's argv (its preflight, port claim and log)."""
    from main.x35 import kakuna as KA
    from utils.paths import repo_root

    KA.preflight(u)
    out = state / "cells" / u.name
    if out.exists():
        out.rename(out.with_name(f"{u.name}.partial.{int(time.time())}"))
    port, claim = KA.claim_port()
    try:
        argv = bot_argv(KA.unit_argv(u, out, port, short=short), u.arm.split(":", 1)[1])
        (state / "logs").mkdir(parents=True, exist_ok=True)
        with open(state / "units.jsonl", "a") as f:
            f.write(json.dumps({"t": time.time(), "event": "start", "unit": u.name, "argv": argv}) + "\n")
        t0 = time.time()
        with open(state / "logs" / f"{u.name}.log", "ab") as log:
            proc = subprocess.Popen(argv, cwd=str(repo_root()), env=KA.unit_env(config=KA.config_of(u.teamset)),
                                    stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                    start_new_session=True)
            rc = proc.wait()
    finally:
        claim.close()
    with open(state / "units.jsonl", "a") as f:
        f.write(json.dumps({"t": time.time(), "event": "end", "unit": u.name, "rc": rc,
                            "counted": KA.is_done(state, u), "wall_s": round(time.time() - t0, 1)}) + "\n")
    return rc


def floor_read(state: Path) -> Dict[str, Any]:
    """``{bot: {panel: [wins, games]}, rates: {bot: equal-weight U rate}}`` over COUNTED units (ties are non-wins)."""
    from main.x35 import kakuna as KA

    cells: Dict[str, Dict[str, List[int]]] = {}
    uncounted: List[str] = []
    engines = set()
    for u in floor_units():
        s = KA.read_summary(state, u.name)
        c = cells.setdefault(u.arm.split(":", 1)[1], {}).setdefault(u.teamset, [0, 0])
        if s is None or not KA.counts(s, u.games):
            uncounted.append(u.name)
            continue
        engines.add(str((s.get("cell") or {}).get("engine_stamp")))
        c[0] += int(s["wins"])
        c[1] += int(s["n"])
    rates = {b: (sum(w / g for w, g in p.values()) / len(p) if all(g for _, g in p.values()) else None)
             for b, p in cells.items()}
    return {"schema": "gen3_x59s_floor_v1", "cells": cells, "rates": rates, "uncounted": uncounted, "engines": sorted(engines),
            "units": [u.name for u in floor_units()]}


def floor_play(state: Path, attempts: int = 3) -> None:
    from main.x35 import kakuna as KA

    state.mkdir(parents=True, exist_ok=True)
    for u in floor_units():
        k = 0
        while not KA.is_done(state, u) and k < attempts:
            _play_bot(state, u, short=(k == 0))          # the short lane first; a retry is a long heavy-slot job
            k += 1


# --------------------------------------------------------------------------------------------------- inhouse
def inhouse_cells(students: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Every registered head-to-head cell: (contrast, seed, panel, player zip, opponent zip)."""
    side = students["students"]
    return [{"contrast": c, "seed": k, "panel": p, "player": side[label(a, k)]["zip_path"],
             "opponent": side[label(b, k)]["zip_path"], "name": f"{c}_{k}_{p}"}
            for c, (a, b) in INHOUSE.items() for k in SEEDS for p in PANELS]


def inhouse_argv(cell: Mapping[str, Any], ledger: Path, py: str = sys.executable) -> List[str]:
    from utils.paths import repo_path

    return ["nice", "-n", "19", "taskset", "-c", "5-7,13-15", str(repo_path("scripts/ops/mem_cap.sh")), "--name",
            "x59s_h2h", "12", "timeout", "2700", py, "-m", "main.h2h", "play", "--player", cell["player"],
            "--opponent", cell["opponent"], "--panel", cell["panel"], "--pairs", str(PAIRS), "--seed",
            str(cell["seed"]), "--out", str(ledger), "--label", "x59s_inhouse", "--device", "cpu", "--threads", "2",
            "--torch-threads", "2"]


def _summary_of(log: Path) -> Optional[Dict[str, Any]]:
    """The last JSON line ``main.h2h play`` printed (its edge summary), if the cell finished."""
    try:
        lines = log.read_text().splitlines()
    except OSError:
        return None
    for ln in reversed(lines):
        ln = ln.strip()
        if ln.startswith("{") and '"score"' in ln:
            try:
                return dict(json.loads(ln))
            except ValueError:
                return None
    return None


def inhouse_play(state: Path, students: Mapping[str, Any], attempts: int = 2) -> None:
    from utils.paths import repo_root

    state.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONPATH=str(repo_root() / "src"), CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="2")
    for cell in inhouse_cells(students):
        log = state / f"{cell['name']}.log"
        k = 0
        while (_summary_of(log) or {}).get("pairs") != PAIRS and k < attempts:
            with open(log, "a") as f:
                subprocess.run(inhouse_argv(cell, state / "ledger"), cwd=str(repo_root()), env=env, stdout=f,
                               stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, check=False)
            k += 1


def inhouse_read(state: Path, students: Mapping[str, Any]) -> Dict[str, Any]:
    """Each pairing's rate (module docstring) and the cells it came from; a cell short of :data:`PAIRS` pairs is
    listed and its pairing has no rate."""
    out: Dict[str, Any] = {"schema": "gen3_x59s_inhouse_v1", "pairs": PAIRS, "cells": {}, "rates": {}, "short": []}
    for cell in inhouse_cells(students):
        s = _summary_of(state / f"{cell['name']}.log")
        ok = s is not None and s.get("pairs") == PAIRS and s.get("score") is not None
        out["cells"][cell["name"]] = {"player": cell["player"], "opponent": cell["opponent"],
                                      "score": s.get("score") if ok else None, "pairs": (s or {}).get("pairs"),
                                      "regime_id": (s or {}).get("regime_id")}
        if not ok:
            out["short"].append(cell["name"])
    for c in INHOUSE:
        rates = []
        for k in SEEDS:
            sc = [out["cells"][f"{c}_{k}_{p}"]["score"] for p in PANELS]
            rates.append(None if any(x is None for x in sc) else sum(sc) / len(sc))
        out["rates"][c] = rates
    return out


# ------------------------------------------------------------------------------------------------------- cli
def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m main.human_data.x59s_ops", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    x = sub.add_parser("export")
    x.add_argument("--students", required=True)
    x.add_argument("--out", required=True)
    x.add_argument("--human-rows", required=True)
    x.add_argument("--selfplay-rows", required=True)
    x.add_argument("--only", default=None, help="export these students (a comma list: one short-lane job) and write no table")
    i = sub.add_parser("identity")
    i.add_argument("--state", required=True)
    i.add_argument("--unit", default="pstar__home72__00")
    i.add_argument("--x51-rows", required=True)
    i.add_argument("--read-cell", default="pstar_home72")
    i.add_argument("--unit-seed", type=int, default=0)
    i.add_argument("--out", required=True)
    f = sub.add_parser("floor")
    h = sub.add_parser("inhouse")
    h.add_argument("--students", required=True)
    for p in (f, h):
        p.add_argument("--state", required=True)
        p.add_argument("--play", action="store_true")
        p.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    try:
        if a.cmd == "export":
            from main.human_data.rowset import _guard_out

            _guard_out(Path(a.out))
            doc = export_all(Path(a.students), Path(a.out), Path(a.human_rows), Path(a.selfplay_rows),
                             only=a.only)
            print(json.dumps({"students": len(doc["students"]), "void3": doc["void3"], "void3_sp0": doc["void3_sp0"]}))
            return 0 if not doc["void3"] else 1
        if a.cmd == "identity":
            res = identity(Path(a.state), a.unit, Path(a.x51_rows), a.read_cell, a.unit_seed)
            Path(a.out).write_text(json.dumps(res, indent=1))
            print(json.dumps({k: res[k] for k in ("unit", "pass", "games_here", "games_x51")}))
            return 0 if res["pass"] else 1
        if a.cmd == "floor":
            if a.play:
                floor_play(Path(a.state))
            res = floor_read(Path(a.state))
        else:
            st = json.loads(Path(a.students).read_text())
            if a.play:
                inhouse_play(Path(a.state), st)
            res = inhouse_read(Path(a.state), st)
        Path(a.out).write_text(json.dumps(res, indent=1, sort_keys=True))
        print(json.dumps({"out": a.out, "uncounted_or_short": res.get("uncounted", res.get("short"))}))
        return 0
    except OpsError as exc:
        print(f"[x59s_ops] REFUSED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
