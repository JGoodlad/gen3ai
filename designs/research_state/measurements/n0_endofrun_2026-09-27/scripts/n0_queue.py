#!/usr/bin/env python3
"""The CPU WORK QUEUE for the N0 end-of-run reads and the learner battery's strength reads.

Registration: ``../README.md`` §5. One DETACHED supervisor runs minutes-long UNITS in a registered
priority order; every unit's output is durable and self-describing, so the queue is resumable by
simply starting it again (a unit is DONE iff its own output says so).

    n0_queue.py run      # the supervisor — start it DETACHED (see ``start_queue.sh``)
    n0_queue.py status   # progress from the outputs on disk; safe at any time
    n0_queue.py plan     # every unit in order, with its readiness and state; starts nothing
    n0_queue.py stop     # ask the supervisor to exit after its running units finish

Rules the code enforces:
* battery units (C → E5 → T32 → L95, each once its ``final_model.zip`` is complete and stable) come
  first; then N0's reads (leaf → anchors → the pair cell); then N0's descriptors;
* while ANY GPU compute process exists, exactly ONE unit runs; when the GPU is idle, untaught (U)
  units — a pure function of their inputs — may run ``idle_workers`` at once; a leaf or anchors
  unit ALWAYS runs alone (a wall-clock search budget buys a width set by the box, rule 23);
* ``control.json`` in the state dir overrides ``live_workers`` / ``idle_workers`` and holds
  ``pause`` / ``stop``; it is re-read before every launch;
* children run at nice 15, with the GPU hidden and one BLAS thread, each in its own session so a
  hung unit is stopped by its explicit process-group id and nothing else.
"""
from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

HERE = Path(__file__).resolve().parent
READ_DIR = HERE.parent
TREE = READ_DIR.parents[3]
STATE = Path(os.environ.get("N0Q_STATE", "/home/goodlad/dev/gen3ai-reads/n0_endofrun_2026-09-27"))
MODELS = Path(os.environ.get("N0Q_MODELS", "/home/goodlad/dev/gen3ai/models"))
PY = os.environ.get("N0Q_PYTHON", sys.executable)
METAMON_TEAMS = Path(os.environ.get("N0Q_METAMON_TEAMS", "/home/goodlad/dev/metamon/cache/teams"))
HOME72_SET = "gen3ai_home72"
NICE = 15
UNIT_WALL_CAP_S = 4 * 3600
MAX_ATTEMPTS = 3
STABLE_S = 300

N0 = "ai_v14_01_base"
U_OPPONENT = MODELS / N0 / "snapshots" / "snapshot_000024000000.zip"
ARMS = [("C", "ai_v14_02_lbat_ctrl"), ("E5", "ai_v14_03_lbat_e5"),
        ("T32", "ai_v14_04_lbat_t32"), ("L95", "ai_v14_05_lbat_l95")]
GA_SEEDS = (0, 10, 20, 30, 40, 50)
U_TEAMS, U_CHUNKS, U_CHUNK = 8, 24, 25          # 600 games per team

LEAF_COMMON = ["--arm", "honest", "--budget", "1", "--opponents", "self", "--games-seed", "7",
               "--device", "cpu", "--search-impl", "rust", "--battle-timeout-s", "1800",
               "--battle-idle-s", "120"]
DEFB = ["--root-strategy", "defensive", "--defensive-leaf", "winprob", "--defensive-wp-margin",
        "0.15", "--defensive-confirm", "0", "--defensive-contested-deadline-s", "3.0"]
LEAF_WINDOW = 10


def final_zip(run: str) -> Path:
    return MODELS / run / "final_model.zip"


def complete(path: Path) -> bool:
    """A final checkpoint is usable once it exists, is a valid zip, and has not changed for
    ``STABLE_S`` seconds (the launcher writes it in place at the end of a run)."""
    try:
        st = path.stat()
    except FileNotFoundError:
        return False
    return time.time() - st.st_mtime > STABLE_S and zipfile.is_zipfile(path)


# ------------------------------------------------------------------------------------ units
@dataclass
class Unit:
    name: str
    argv: List[str]
    parallel: bool
    ready: Callable[[], bool]
    done: Callable[[], bool]
    out: Path
    env: Dict[str, str] = field(default_factory=dict)
    retire: Optional[Callable[[], None]] = None      # move a partial output aside before a run


def _jsonl(path: Path) -> List[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return rows


def _aside(path: Path) -> Callable[[], None]:
    def go() -> None:
        if path.exists():
            path.rename(path.with_name(f"{path.name}.partial.{int(time.time())}"))
    return go


def u_unit(label: str, ref: Path, ti: int, chunk: int, ready: Callable[[], bool]) -> Unit:
    out = STATE / "gu_rows" / label / f"t{ti}" / f"c{chunk:02d}.jsonl"

    def done() -> bool:
        js = {r.get("j") for r in _jsonl(out)}
        return all(j in js for j in range(chunk * U_CHUNK, (chunk + 1) * U_CHUNK))
    return Unit(f"U_{label}_t{ti}_c{chunk:02d}",
                [PY, str(HERE / "gu_unit.py"), "--label", label, "--ref", str(ref),
                 "--opponent", str(U_OPPONENT), "--team-index", str(ti), "--chunk", str(chunk),
                 "--rows", str(STATE / "gu_rows")],
                True, lambda: ready() and U_OPPONENT.exists(),
                done, out)


def anchors_ok(d: Path, n: int) -> bool:
    try:
        s = json.loads((d / "summary.json").read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return False
    return s.get("status") == "OK" and s.get("n") == n


def anchors_unit(name: str, argv: List[str], games: int, ready: Callable[[], bool],
                 env: Optional[Dict[str, str]] = None) -> Unit:
    d = STATE / "anchors" / name
    return Unit(name, [PY, "-m", "main.anchors", *argv, "--games", str(games), "--device", "cpu",
                       "--nice", str(NICE), "--out", str(d)],
                False, ready, lambda: anchors_ok(d, games), d, env or {}, _aside(d))


def ga_units(label: str, zip_path: Path, ready: Callable[[], bool]) -> List[Unit]:
    return [anchors_unit(f"GA_{label}_{ts}_s{s}",
                         ["--model", str(zip_path), "--opponent", "metamon:SmallRL",
                          "--server", "rust", "--regime", "greedy", "--teamset", ts,
                          "--team-seed", str(s), "--seed-base", str(s)], 100, ready)
            for s in GA_SEEDS for ts in ("away", "home")]


def u_units(label: str, zip_path: Path, ready: Callable[[], bool]) -> List[Unit]:
    return [u_unit(label, zip_path, ti, c, ready)
            for c in range(U_CHUNKS) for ti in range(U_TEAMS)]


def leaf_unit(cell: str, lo: int, ready: Callable[[], bool]) -> Unit:
    out = STATE / "leaf" / cell / f"g{lo:04d}.jsonl"
    extra = {"defB": DEFB, "grid": ["--root-strategy", "grid"], "base": []}[cell]
    common = list(LEAF_COMMON)
    if cell == "base":
        common[common.index("honest")] = "base"

    def done() -> bool:
        seen = {(r.get("game"), r.get("orientation")) for r in _jsonl(out)}
        return len({g for g, _ in seen if g is not None and lo <= g < lo + LEAF_WINDOW}) == \
            LEAF_WINDOW and len(seen) >= 2 * LEAF_WINDOW
    return Unit(f"leaf_{cell}_g{lo:04d}",
                [PY, "-m", "main.search_dividend", str(final_zip(N0)), *common, *extra,
                 "--games-start", str(lo), "--games", str(LEAF_WINDOW), "--out", str(out)],
                False, ready, done, out, {}, _aside(out))


def leaf_block(defb_lo: int, defb_hi: int, base_lo: int, ready) -> List[Unit]:
    """``defB`` windows [defb_lo, defb_hi) with one 10-pair ``base`` window after every four."""
    units: List[Unit] = []
    base = base_lo
    for k, lo in enumerate(range(defb_lo, defb_hi, LEAF_WINDOW)):
        units.append(leaf_unit("defB", lo, ready))
        if k % 4 == 3:
            units.append(leaf_unit("base", base, ready))
            base += LEAF_WINDOW
    return units


def leaf_rows(cell: str) -> List[dict]:
    rows: List[dict] = []
    for p in sorted((STATE / "leaf" / cell).glob("g*.jsonl")):
        rows.extend(_jsonl(p))
    return rows


def extension_fires() -> bool:
    """The 09-11 rule, verbatim: pairs 0-399 read a paired point >= 0.51 with a CI straddling 0.50."""
    from main.search_dividend.summary import mirror_report

    rows = [r for r in leaf_rows("defB") if r.get("game") is not None and r["game"] < 400]
    if len({r["game"] for r in rows}) < 400:
        return False                 # the rule reads the REGISTERED 400 pairs, never fewer
    rep = mirror_report(rows)
    cells = rep.get("cells") or []
    if not cells:
        return False
    c = cells[0]
    p, ci = c.get("paired_win_rate"), c.get("paired_ci95")
    return p is not None and ci is not None and p >= 0.51 and ci[0] <= 0.50 <= ci[1]


def home72_config() -> Path:
    """The campaign config: the stock ``designs/ops/anchors.json`` with ``home`` pointed at the
    sanitized 72 for ALL THREE clients (README §2.1, hazard H19)."""
    teams = READ_DIR / "teams_home72"
    cfg = json.loads((TREE / "designs" / "ops" / "anchors.json").read_text())
    cfg["our_team_sources"]["home"] = {"kind": "dir", "path": str(teams)}
    cfg["opponents"]["foulplay"]["team_dirs"]["home"] = str(teams)
    cfg["opponents"]["metamon"]["team_sets"]["home"] = HOME72_SET
    cfg["_README"] = ["n0_endofrun_2026-09-27 campaign config: `home` = the sanitized 72 "
                      "(teams_home72/) for ours, Foul Play and Metamon alike; everything else stock."]
    out = STATE / "configs" / "anchors_home72.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(cfg, indent=1))
    mm = METAMON_TEAMS / HOME72_SET / "gen3ou"
    mm.mkdir(parents=True, exist_ok=True)
    for src in sorted(teams.glob("*.txt")):
        dst = mm / f"{src.stem}.gen3ou_team"
        if dst.exists():
            if dst.read_bytes() != src.read_bytes():
                raise SystemExit(f"REFUSING: {dst} differs from {src} — the shared set is not shared")
        else:
            shutil.copyfile(src, dst)
    assert len(list(mm.glob("*.gen3ou_team"))) == 72, mm
    return out


def all_units() -> List[Unit]:
    units: List[Unit] = []
    n0_ready = lambda: complete(final_zip(N0))                              # noqa: E731
    # ---- P0: the learner battery (registered in learner_battery_2026-09-26.md §4.2)
    for label, run in ARMS:
        z = final_zip(run)
        ready = (lambda z=z: complete(z))
        units += u_units(label, z, ready) + ga_units(label, z, ready)
    # ---- P1: N0's reads, README §1 -> §2 -> §3
    units += leaf_block(0, 400, 0, n0_ready)
    units += [leaf_unit("grid", lo, n0_ready) for lo in range(0, 100, LEAF_WINDOW)]
    ext_ready = lambda: n0_ready() and extension_fires()                      # noqa: E731
    units += leaf_block(400, 800, 100, ext_ready)
    h72 = {"GEN3AI_ANCHORS_CONFIG": str(STATE / "configs" / "anchors_home72.json")}
    n0 = ["--model", str(final_zip(N0)), "--server", "rust", "--regime", "greedy"]
    for s in (0, 10, 20, 30):
        units.append(anchors_unit(f"A1_synthv2_away_s{s}", n0 + [
            "--opponent", "metamon:SyntheticRLV2", "--teamset", "away",
            "--team-seed", str(s), "--seed-base", str(s)], 25, n0_ready))
    for s in (0, 10, 20, 30):
        units.append(anchors_unit(f"A2_synthv2_home_s{s}", n0 + [
            "--opponent", "metamon:SyntheticRLV2", "--teamset", "home",
            "--team-seed", str(s), "--seed-base", str(s)], 25, n0_ready))
    for s in range(0, 100, 10):
        units.append(anchors_unit(f"A3_foulplay1000_home72_s{s}", n0 + [
            "--opponent", "foulplay", "--teamset", "home", "--search-time-ms", "1000",
            "--search-parallelism", "1", "--team-seed", str(s), "--seed-base", str(s)],
            10, n0_ready, h72))
    for s in (0, 10, 20, 30):
        units.append(anchors_unit(f"A4_synthv2_home72_s{s}", n0 + [
            "--opponent", "metamon:SyntheticRLV2", "--teamset", "home",
            "--team-seed", str(s), "--seed-base", str(s)], 25, n0_ready, h72))
    for s in range(0, 100, 10):
        units.append(anchors_unit(f"P3_fp1000_vs_synthv2_home72_s{s}", [
            "--opponent-a", "foulplay", "--opponent-b", "metamon:SyntheticRLV2",
            "--server", "rust", "--regime", "greedy", "--teamset", "home",
            "--search-time-ms", "1000", "--search-parallelism", "1", "--progress-timeout", "3600",
            "--team-seed", str(s), "--seed-base", str(s)], 10, n0_ready, h72))
    # ---- P2: N0's own descriptors (README §4)
    units += u_units("N0", final_zip(N0), n0_ready) + ga_units("N0", final_zip(N0), n0_ready)
    return units


# ------------------------------------------------------------------------------- supervisor
def gpu_live() -> bool:
    try:
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=30).stdout
    except (OSError, subprocess.SubprocessError):
        return True                  # cannot tell ⇒ assume a training arm owns the box
    return bool(out.strip())


def control() -> dict:
    c = {"pause": False, "stop": False, "live_workers": 1, "idle_workers": 8}
    try:
        c.update(json.loads((STATE / "control.json").read_text()))
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    return c


def log_event(**kw) -> None:
    kw["t"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    with open(STATE / "units.jsonl", "a") as fh:
        fh.write(json.dumps(kw) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def attempts() -> Dict[str, int]:
    n: Dict[str, int] = {}
    for e in _jsonl(STATE / "units.jsonl"):
        if e.get("event") == "end" and e.get("rc") != 0:
            n[e["unit"]] = n.get(e["unit"], 0) + 1
        if e.get("event") == "end" and e.get("rc") == 0 and not e.get("done", True):
            n[e["unit"]] = n.get(e["unit"], 0) + 1
    return n


def child_env(extra: Dict[str, str]) -> Dict[str, str]:
    env = dict(os.environ)
    rel = TREE / "src" / "rust_sim" / "target" / "release"
    env.update({
        "PYTHONPATH": str(TREE / "src"),
        "POKESIM_SIM_BRIDGE_BIN": str(rel / "sim_bridge"),
        "POKESIM_SEARCH_DRIVER_BIN": str(rel / "search_driver"),
        "POKESIM_CORE_EVENTS_BIN": str(rel / "core_events"),
        "CUDA_VISIBLE_DEVICES": "",
        "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1", "PYTHONUNBUFFERED": "1",
        "N0Q_TREE_COMMIT": subprocess.run(["git", "-C", str(TREE), "rev-parse", "HEAD"],
                                          capture_output=True, text=True).stdout.strip(),
    })
    env.update(extra)
    return env


def launch(u: Unit) -> subprocess.Popen:
    if u.retire is not None and u.out.exists():
        u.retire()
    u.out.parent.mkdir(parents=True, exist_ok=True)
    logs = STATE / "logs"
    logs.mkdir(exist_ok=True)
    fh = open(logs / f"{u.name}.log", "a")
    fh.write(f"\n=== {time.strftime('%F %T')} {' '.join(u.argv)}\n")
    fh.flush()
    p = subprocess.Popen(["nice", "-n", str(NICE), *u.argv], cwd=str(TREE),
                         env=child_env(u.env), stdout=fh, stderr=subprocess.STDOUT,
                         stdin=subprocess.DEVNULL, start_new_session=True)
    log_event(event="start", unit=u.name, pid=p.pid, load=os.getloadavg()[0], gpu_live=gpu_live())
    return p


def run() -> int:
    STATE.mkdir(parents=True, exist_ok=True)
    (STATE / "supervisor.pid").write_text(f"{os.getpid()}\n")
    home72_config()
    stopping = {"flag": False}
    signal.signal(signal.SIGTERM, lambda *_: stopping.__setitem__("flag", True))
    running: Dict[str, tuple] = {}
    log_event(event="supervisor_start", pid=os.getpid(), tree=str(TREE))
    while True:
        for name, (p, u, t0) in list(running.items()):
            rc = p.poll()
            if rc is None and time.time() - t0 > UNIT_WALL_CAP_S:
                os.killpg(p.pid, signal.SIGTERM)
                time.sleep(20)
                if p.poll() is None:
                    os.killpg(p.pid, signal.SIGKILL)
                rc = p.wait()
                log_event(event="wall_cap", unit=name)
            if rc is not None:
                log_event(event="end", unit=name, rc=rc, wall_s=round(time.time() - t0, 1),
                          done=u.done(), load=os.getloadavg()[0])
                del running[name]
        c = control()
        if stopping["flag"] or c["stop"]:
            if not running:
                log_event(event="supervisor_stop")
                return 0
            time.sleep(10)
            continue
        if c["pause"]:
            time.sleep(30)
            continue
        tries = attempts()
        todo = [u for u in all_units() if u.name not in running
                and tries.get(u.name, 0) < MAX_ATTEMPTS and not u.done()]
        nxt = next((u for u in todo if u.ready()), None)
        if nxt is None:
            if not running and not any(u.ready() for u in todo) and not todo:
                log_event(event="all_done")
                return 0
            time.sleep(60)
            continue
        live = gpu_live()
        cap = int(c["live_workers"] if live else c["idle_workers"]) if nxt.parallel else 1
        busy_exclusive = any(not u.parallel for _, u, _ in running.values())
        if (nxt.parallel and not busy_exclusive and len(running) < cap) or \
                (not nxt.parallel and not running):
            running[nxt.name] = (launch(nxt), nxt, time.time())
            time.sleep(2)
            continue
        time.sleep(15)


def status() -> int:
    units = all_units()
    tries = attempts()
    groups: Dict[str, List[int]] = {}
    for u in units:
        key = u.name.split("_")[0] + "_" + (u.name.split("_")[1] if u.name[:2] in ("U_", "GA")
                                             else "")
        g = groups.setdefault(key.rstrip("_"), [0, 0, 0])
        g[0] += 1
        g[1] += int(u.done())
        g[2] += int(tries.get(u.name, 0) >= MAX_ATTEMPTS and not u.done())
    for k, (n, d, f) in groups.items():
        print(f"  {k:28s} {d:4d}/{n:<4d} done" + (f"   {f} GAVE UP" if f else ""))
    pidf = STATE / "supervisor.pid"
    if pidf.exists():
        pid = int(pidf.read_text().split()[0])
        alive = Path(f"/proc/{pid}").exists()
        print(f"  supervisor pid {pid}: {'RUNNING' if alive else 'not running'}")
    ev = _jsonl(STATE / "units.jsonl")
    live = {e["unit"]: e for e in ev if e.get("event") == "start"}
    for e in ev:
        if e.get("event") == "end":
            live.pop(e["unit"], None)
    for name, e in live.items():
        if not Path(f"/proc/{e['pid']}").exists():
            continue                 # a start whose supervisor was killed before its end event
        print(f"  running: {name} pid {e['pid']} since {e['t']}")
    return 0


def plan() -> int:
    for i, u in enumerate(all_units()):
        print(f"{i:4d} {'P' if u.parallel else 'X'} {'done' if u.done() else ('ready' if u.ready() else 'wait'):5s} {u.name}")
    return 0


def main() -> int:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    if cmd == "run":
        return run()
    if cmd == "plan":
        return plan()
    if cmd == "stop":
        c = control()
        c["stop"] = True
        (STATE / "control.json").write_text(json.dumps(c))
        print("stop requested: the supervisor exits after its running units finish")
        return 0
    return status()


if __name__ == "__main__":
    sys.exit(main())
