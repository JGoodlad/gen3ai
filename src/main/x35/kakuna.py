"""X35's EXTERNAL-ANCHOR unit runner, G-K's instrument (``design_compute_efficiency.md`` §5.6.6;
``designs/ops/EXTERNAL_ANCHORS_SOP.md``). The X26 read's ``xq.py`` pattern
(``measurements/x26_metamon_anchor_2026-10-17/scripts/xq.py``), made a tested module.

THE CELL. ``metamon:Kakuna`` (ckpt 34), GREEDY vs GREEDY (SOP rule 2), our side greedy on CPU, the in-process Rust
front end on a 95xx port this tool picks itself. BOTH team sets: ``home72`` and ``away``, through ONE COMMITTED
config, ``<stage>/anchors_home72_kakuna.json`` -- byte-identical to the X26 read's (sha256 ``e8a064db…``, F21) --
whose every team file is pinned by ``<stage>/home72_manifest.json`` (our 72 committed pastes under
``measurements/n0_endofrun_2026-09-27/teams_home72/`` AND Metamon's copy of them in its cache). The manifest is
re-checked before EVERY unit: a moved team is a refusal, never a different read.

A UNIT is one ``python -m main.anchors`` invocation of 20 role-balanced games with ``--team-seed = --seed-base =``
the unit seed; the seeds 0-6 are SHARED across arms. It runs from the checkout ROOT (the pool source's CWD hazard,
X26 read FINDING 1), under ``nice -n 19 taskset -c 5-7,13-15`` inside ``scripts/ops/mem_cap.sh`` (rule 5b), one lane.
A unit COUNTS iff ``status == OK``, ``n == 20``, ``regime_verified_decisions``, no ``team_source_asymmetry`` and
``peer_clean`` (§5.6.6). A T28 halt marker refuses every unit (``main.live.halt``).

    python -m main.x35.kakuna plan   --state DIR --finals finals.json   # [{label, arm, seed, path}]: the unit queue
    python -m main.x35.kakuna run    --state DIR [--stop-at-epoch E]    # work the queue (resumable, detached-safe)
    python -m main.x35.kakuna status --state DIR
    python -m main.x35.kakuna gk     --state DIR --on A1 --off C0       # G-K's input for main.x35.readout
    python -m main.x35.kakuna smoke  --state DIR --model ZIP [--games 2] # ONE unit vs metamon:SmallRL, home72
    python -m main.x35.kakuna manifest --check | --write
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from utils.paths import repo_path, repo_root

STAGE = repo_path("designs/research_state/measurements/x35_breakers_stage_2026-10-17")
CONFIG = STAGE / "anchors_home72_kakuna.json"
MANIFEST = STAGE / "home72_manifest.json"
CONFIG_SHA256 = "e8a064db845b058dee5ab2512e7f8d0ba636f10df58510e30b8cac9b30422fac"
OUR_TEAMS = repo_path("designs/research_state/measurements/n0_endofrun_2026-09-27/teams_home72")
OPPONENT = "metamon:Kakuna"
SMOKE_OPPONENT = "metamon:SmallRL"
TEAMSETS = {"home72": "home", "away": "away"}
UNIT_SEEDS = tuple(range(7))
UNIT_GAMES = 20
CORES = "5-7,13-15"
MEM_GB = "12"
PORT_RANGE = (9540, 9599)
MANIFEST_SCHEMA = "x35_home72_manifest_v1"
MAX_ATTEMPTS = 2


class KakunaError(RuntimeError):
    """A precondition of a unit does not hold (named; the unit does not run)."""


def _sha(path: "str | os.PathLike[str]") -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------------------------- the manifest
def _team_hashes(d: Path, suffix: str) -> Dict[str, str]:
    return {p.name[: -len(suffix)]: _sha(p) for p in sorted(d.iterdir()) if p.name.endswith(suffix)}


def build_manifest(config: Path = CONFIG) -> Dict[str, Any]:
    cfg = json.loads(config.read_text())
    ours = Path(cfg["our_team_sources"]["home"]["path"])
    mm = cfg["opponents"]["metamon"]
    theirs = Path(mm["cache_dir"]) / "teams" / mm["team_sets"]["home"] / "gen3ou"
    return {"schema": MANIFEST_SCHEMA, "config": str(config.relative_to(repo_root())), "config_sha256": _sha(config),
            "our_team_dir": str(ours), "our_teams": _team_hashes(ours, ".txt"),
            "metamon_team_dir": str(theirs), "metamon_teams": _team_hashes(theirs, ".gen3ou_team")}


def check_manifest(manifest: Path = MANIFEST, config: Path = CONFIG, *,
                   our_dir: Optional[Path] = None, their_dir: Optional[Path] = None) -> List[str]:
    """Every way the home72 set on disk differs from the committed manifest (empty = it is the X26 read's)."""
    m = json.loads(manifest.read_text())
    probs: List[str] = []
    if m.get("schema") != MANIFEST_SCHEMA:
        probs.append(f"manifest schema {m.get('schema')!r}")
    if _sha(config) != m["config_sha256"] or m["config_sha256"] != CONFIG_SHA256:
        probs.append(f"{config}: sha256 is not the X26 read's {CONFIG_SHA256[:12]}")
    ours = our_dir or Path(m["our_team_dir"])
    theirs = their_dir or Path(m["metamon_team_dir"])
    for name, d, suffix, want in (("our", ours, ".txt", m["our_teams"]),
                                  ("Metamon's", theirs, ".gen3ou_team", m["metamon_teams"])):
        if not d.is_dir():
            probs.append(f"{name} team dir {d} is missing")
            continue
        got = _team_hashes(d, suffix)
        if got != want:
            diff = sorted(set(got) ^ set(want)) + sorted(k for k in set(got) & set(want) if got[k] != want[k])
            probs.append(f"{name} home72 teams differ from the manifest: {diff[:5]}{' …' if len(diff) > 5 else ''}")
    if m["our_teams"] != m["metamon_teams"]:
        probs.append("the manifest's two copies of home72 are not byte-identical")
    return probs


# ------------------------------------------------------------------------------------------------- the units
@dataclass(frozen=True)
class Unit:
    name: str
    label: str
    arm: str
    seed: int          # the final's training seed
    teamset: str       # home72 | away
    unit_seed: int
    model: str
    model_sha256: str
    opponent: str = OPPONENT
    games: int = UNIT_GAMES


def plan_units(finals: Sequence[Mapping[str, Any]], *, hash_models: bool = True) -> List[Unit]:
    """Every (final × team set × unit seed) unit, unit seeds outermost so a stopped queue is role- and seed-balanced
    across finals."""
    seen = set()
    for f in finals:
        if f["label"] in seen:
            raise KakunaError(f"final label {f['label']!r} appears twice")
        seen.add(f["label"])
    shas = {f["label"]: (_sha(f["path"]) if hash_models else "") for f in finals}
    return [Unit(name=f"{f['label']}__{ts}__{s:02d}", label=str(f["label"]), arm=str(f["arm"]), seed=int(f["seed"]),
                 teamset=ts, unit_seed=s, model=str(f["path"]), model_sha256=shas[f["label"]])
            for s in UNIT_SEEDS for ts in TEAMSETS for f in finals]


def counts(summary: Mapping[str, Any], games: int) -> bool:
    """§5.6.6's rule: a unit COUNTS iff OK, complete, regime-verified, symmetric-sourced and peer-clean — THE ONE
    rule, :func:`main.anchors.results.unit_counts`, undeclared (these cells are symmetric home72 / away reads; a
    summary whose two sides' content digests differ no longer counts on equal file counts alone, F-DV-5)."""
    from main.anchors.results import unit_counts

    return unit_counts(summary, games=games).counts


def read_summary(state: Path, name: str) -> Optional[Dict[str, Any]]:
    p = state / "cells" / name / "summary.json"
    try:
        return dict(json.loads(p.read_text()))
    except (OSError, ValueError):
        return None


def is_done(state: Path, u: Unit) -> bool:
    s = read_summary(state, u.name)
    return s is not None and counts(s, u.games)


def choose_port(lo: int = PORT_RANGE[0], hi: int = PORT_RANGE[1]) -> int:
    """The first 95xx port nobody holds (bound and released at once; 8000 / 8001 are never in range)."""
    for p in range(lo, hi + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("0.0.0.0", p))
            except OSError:
                continue
            return p
    raise KakunaError(f"no free port in {lo}-{hi}")


def unit_argv(u: Unit, out: Path, port: int, py: str = sys.executable) -> List[str]:
    a = [py, "-m", "main.anchors", "--model", u.model, "--opponent", u.opponent, "--regime", "greedy",
         "--teamset", TEAMSETS[u.teamset], "--games", str(u.games), "--team-seed", str(u.unit_seed),
         "--seed-base", str(u.unit_seed), "--device", "cpu", "--port", str(port), "--nice", "19", "--out", str(out)]
    return ["nice", "-n", "19", "taskset", "-c", CORES, str(repo_path("scripts/ops/mem_cap.sh")), "--name",
            "x35kakuna", MEM_GB, *a]


def unit_env(base: Optional[Mapping[str, str]] = None) -> Dict[str, str]:
    env = dict(os.environ if base is None else base)
    env.update({"PYTHONPATH": str(repo_path("src")), "CUDA_VISIBLE_DEVICES": "", "GEN3AI_ANCHORS_CONFIG": str(CONFIG),
                "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                "NUMEXPR_NUM_THREADS": "1", "PYTHONUNBUFFERED": "1"})
    return env


def preflight(u: Unit) -> None:
    """Refuse a unit on a T28 halt, a moved home72 set, or a model whose bytes changed since the plan."""
    from main.live.halt import read_halt

    if read_halt() is not None:
        raise KakunaError("a T28 live-parse HALT marker is set (python -m main.live.halt status): no play")
    probs = check_manifest()
    if probs:
        raise KakunaError(f"the home72 manifest does not hold: {probs}")
    if u.model_sha256 and _sha(u.model) != u.model_sha256:
        raise KakunaError(f"{u.model}: sha256 changed since the plan")


def _jl(state: Path, obj: Mapping[str, Any]) -> None:
    with open(state / "units.jsonl", "a") as f:
        f.write(json.dumps({"t": time.time(), **obj}) + "\n")


def run_unit(state: Path, u: Unit) -> int:
    preflight(u)
    out = state / "cells" / u.name
    if out.exists():
        out.rename(out.with_name(f"{u.name}.partial.{int(time.time())}"))
    port = choose_port()
    argv = unit_argv(u, out, port)
    _jl(state, {"event": "start", "unit": u.name, "port": port, "argv": argv, "load": os.getloadavg()})
    (state / "logs").mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with open(state / "logs" / f"{u.name}.log", "ab") as log:
        proc = subprocess.Popen(argv, cwd=str(repo_root()), env=unit_env(), stdout=log, stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL, start_new_session=True)
        _jl(state, {"event": "pid", "unit": u.name, "pid": proc.pid})
        rc = proc.wait()
    _jl(state, {"event": "end", "unit": u.name, "rc": rc, "counted": is_done(state, u),
                "wall_s": round(time.time() - t0, 1)})
    return rc


def load_plan(state: Path) -> List[Unit]:
    return [Unit(**d) for d in json.loads((state / "plan.json").read_text())["units"]]


def check_stop_epoch(e: Optional[int], now: Optional[float] = None, horizon_s: int = 7 * 86400) -> None:
    """An ABSOLUTE epoch with a sanity bound (TRAINING_RUN_SOP process rule 1), never a clock string."""
    if e is None:
        return
    t = time.time() if now is None else now
    if not t < e <= t + horizon_s:
        raise KakunaError(f"--stop-at-epoch {e} is not in (now, now + {horizon_s // 86400} d] (now {int(t)})")


def cmd_run(state: Path, stop_at: Optional[int]) -> int:
    check_stop_epoch(stop_at)
    lock = open(state / "supervisor.lock", "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        raise KakunaError(f"another supervisor holds {state / 'supervisor.lock'}") from None
    _jl(state, {"event": "supervisor_start", "pid": os.getpid()})
    fails: Dict[str, int] = {}
    for u in load_plan(state):
        while not is_done(state, u) and fails.get(u.name, 0) < MAX_ATTEMPTS:
            if stop_at is not None and time.time() >= stop_at:
                _jl(state, {"event": "supervisor_stop_at"})
                return 0
            run_unit(state, u)
            if not is_done(state, u):
                fails[u.name] = fails.get(u.name, 0) + 1
    _jl(state, {"event": "supervisor_done", "not_counted": sorted(n for n, k in fails.items() if k >= MAX_ATTEMPTS)})
    return 0


def pool(state: Path, units: Sequence[Unit]) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """``{label: {teamset: {wins, ties, n, units, skipped}}}`` over COUNTED units only."""
    out: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for u in units:
        c = out.setdefault(u.label, {}).setdefault(u.teamset, {"arm": u.arm, "wins": 0, "ties": 0, "n": 0,
                                                               "units": 0, "skipped": []})
        s = read_summary(state, u.name)
        if s is None:
            continue
        if not counts(s, u.games):
            c["skipped"].append(u.name)
            continue
        c["units"] += 1
        c["wins"] += int(s["wins"])
        c["ties"] += int(s.get("ties", 0))
        c["n"] += int(s["n"])
    return out


def gk_input(pooled: Mapping[str, Mapping[str, Mapping[str, Any]]], on: str, off: str) -> Dict[str, Any]:
    """G-K's ``{teamset: {"on": [wins, n], "off": [wins, n]}}``, each side pooled over its arm's seed finals."""
    out: Dict[str, Any] = {}
    for ts in TEAMSETS:
        side: Dict[str, List[int]] = {"on": [0, 0], "off": [0, 0]}
        for cells in pooled.values():
            c = cells.get(ts)
            if c is None:
                continue
            key = "on" if c["arm"] == on else "off" if c["arm"] == off else None
            if key:
                side[key][0] += int(c["wins"])
                side[key][1] += int(c["n"])
        out[ts] = side
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m main.x35.kakuna", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for n in ("plan", "run", "status", "gk", "smoke"):
        sp = sub.add_parser(n)
        sp.add_argument("--state", required=True, help="the read's state dir (outside the repo and models/)")
        if n == "plan":
            sp.add_argument("--finals", required=True, help="JSON [{label, arm, seed, path}]")
        if n == "run":
            sp.add_argument("--stop-at-epoch", type=int, default=None)
        if n == "gk":
            sp.add_argument("--on", required=True)
            sp.add_argument("--off", required=True)
        if n == "smoke":
            sp.add_argument("--model", required=True)
            sp.add_argument("--games", type=int, default=2)
    m = sub.add_parser("manifest")
    m.add_argument("--write", action="store_true", help="(re)write the committed manifest from the files on disk")
    a = ap.parse_args(argv)
    try:
        if a.cmd == "manifest":
            if a.write:
                MANIFEST.write_text(json.dumps(build_manifest(), indent=1, sort_keys=True) + "\n")
            probs = check_manifest()
            print(f"[x35 kakuna] manifest: {'OK' if not probs else probs}")
            return 0 if not probs else 1
        state = Path(a.state)
        state.mkdir(parents=True, exist_ok=True)
        if a.cmd == "plan":
            if (state / "plan.json").exists():
                raise KakunaError(f"{state / 'plan.json'} exists: a plan is written once per read")
            units = plan_units(json.loads(Path(a.finals).read_text()))
            (state / "plan.json").write_text(json.dumps({"units": [u.__dict__ for u in units]}, indent=1) + "\n")
            print(f"[x35 kakuna] planned {len(units)} units")
            return 0
        if a.cmd == "run":
            return cmd_run(state, a.stop_at_epoch)
        if a.cmd == "smoke":
            u = Unit(name="smoke__home72__00", label="smoke", arm="smoke", seed=0, teamset="home72", unit_seed=0,
                     model=a.model, model_sha256=_sha(a.model), opponent=SMOKE_OPPONENT, games=a.games)
            rc = run_unit(state, u)
            s = read_summary(state, u.name) or {}
            keys = ("status", "n", "wins", "losses", "ties", "regime_verified_decisions", "team_source_asymmetry",
                    "peer_clean")
            print(json.dumps({"rc": rc, "counted": counts(s, u.games), **{k: s.get(k) for k in keys}}))
            return 0 if counts(s, u.games) else 1
        units = load_plan(state)
        pooled = pool(state, units)
        if a.cmd == "status":
            for lab, cells in pooled.items():
                for ts, c in cells.items():
                    print(f"{lab:24s} {ts:7s} {c['units']}/{len(UNIT_SEEDS)} units  {c['wins']}/{c['n']} wins  "
                          f"skipped {len(c['skipped'])}")
            return 0
        print(json.dumps(gk_input(pooled, a.on, a.off)))
        return 0
    except KakunaError as exc:
        print(f"[x35 kakuna] REFUSED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
