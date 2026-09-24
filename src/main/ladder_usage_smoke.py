"""The FULL Metamon ladder-usage smoke — every public-ladder team, played and ENCODED
(`gen3_ladder_usage_smoke_v1`).

The Heal Bell fix (``40330daa``) ran a scaled-down smoke: 710 battles on Metamon's ``hl_05_26``
gen3ou teams. This is the full one: all 22,862 teams, paired by a seeded permutation into
**11,431 battles** (the REGISTERED n), each played on the NODE bridge — the reference Showdown,
so a team the Rust engine cannot play (Metronome, Snore, Shell Bell, …) is played too — by two
seeded-random players that run the full training observation encode (``embed_battle``) at EVERY
decision, both sides. A battle is ``ok`` or it carries the exception that ended it: a poke-env
parse crash, an encoder crash, a hang. That is the class of failure a live ladder game dies of.

🚨 **A long measurement is INCREMENTAL** (``designs/ops/ORCHESTRATOR_SOP.md`` §2): the battles are
split into UNITS of ``--unit-size`` keys (minutes each); a unit's rows are written DURABLY and
atomically to ``<out>/units/unit_NNNNN.jsonl`` the moment it finishes; a restarted driver SKIPS
every unit already on disk (resume is pinned by ``ladder_usage_smoke_test.py`` and was proven by
a real kill + resume); the driver runs DETACHED (``--detach``); ``status`` reads progress from the
rows at any time; and the verdict (``report``) is read ONLY at the registered n — an interim read
is progress and health, never a result.

    python -m main.ladder_usage_smoke run --out DIR [--workers 4] [--detach]
    python -m main.ladder_usage_smoke status --out DIR
    python -m main.ladder_usage_smoke report --out DIR           # refuses before the registered n

Durable rows belong outside the repo and outside ``/tmp`` — e.g.
``~/gen3ai_archive/ladder_usage_smoke_2026-09-24`` — and the committed record is the report.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import random
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

DEFAULT_SRC = Path.home() / "gen3ai_archive" / "metamon_cache_2026-09-24" / "teams" / "hl_05_26" / "gen3ou"
SEED = 20260924
UNIT_SIZE = 50
SCHEMA = "gen3_ladder_usage_smoke_v1"

#: Called-move / redirect sources worth COUNTING in the protocol (`gen3_called_move_reading_v1`).
_CALLER_RE = re.compile(r"\|\[from\] ?(Metronome|Assist|Nature Power|Mirror Move|Sleep Talk|Magic Coat|Snatch)\b")


# ---------------------------------------------------------------------------------------------
# the registered battle list
# ---------------------------------------------------------------------------------------------

def team_files(src: Path) -> List[Path]:
    return sorted(p for p in src.iterdir() if p.name.endswith("_team"))


def pairing(n_teams: int, seed: int = SEED) -> List[Tuple[int, int]]:
    """Battle ``k`` plays teams ``perm[2k]`` and ``perm[2k+1]`` of one seeded permutation of the
    sorted team files; an odd team out is dropped. Every team plays exactly once."""
    perm = list(range(n_teams))
    random.Random(seed).shuffle(perm)
    return [(perm[2 * k], perm[2 * k + 1]) for k in range(n_teams // 2)]


def unit_keys(unit: int, n_battles: int, unit_size: int) -> range:
    return range(unit * unit_size, min((unit + 1) * unit_size, n_battles))


def n_units(n_battles: int, unit_size: int) -> int:
    return (n_battles + unit_size - 1) // unit_size


def unit_path(out: Path, unit: int) -> Path:
    return out / "units" / f"unit_{unit:05d}.jsonl"


def done_units(out: Path) -> set:
    d = out / "units"
    if not d.is_dir():
        return set()
    return {int(p.stem.split("_")[1]) for p in d.glob("unit_*.jsonl")}


# ---------------------------------------------------------------------------------------------
# one battle
# ---------------------------------------------------------------------------------------------

def classify(exc: BaseException) -> str:
    """A stable class key for an exception: its type + the message with battle tags, numbers and
    quoted protocol stripped, so 2,000 instances of one bug read as one row."""
    msg = str(exc).splitlines()[0] if str(exc) else ""
    msg = re.sub(r"battle-[\w-]+", "<battle>", msg)
    msg = re.sub(r"\[.*\]", "[…]", msg)
    msg = re.sub(r"\d+", "N", msg)
    return f"{type(exc).__name__}: {msg[:120]}"


def play_battle(key: int, t1: str, t2: str, idle_budget_s: float, total_budget_s: float) -> dict:
    """Play battle ``key`` on the node bridge; both players ENCODE at every decision."""
    import asyncio

    import numpy as np
    from poke_env import AccountConfiguration
    from poke_env.ps_client.server_configuration import LocalhostServerConfiguration

    from agents.inference.player import Gen3Player
    from utils.bridge.local_battle_runner import run_local_battles

    class _EncodingRandomPlayer(Gen3Player):
        def __init__(self, rng_seed: int, **kw):
            super().__init__(**kw)
            self._rng = random.Random(rng_seed)
            self.decisions = 0
            self.encode_error: Optional[BaseException] = None

        def choose_move(self, battle):
            try:
                obs = self.embed_battle(battle)
                if not np.all(np.isfinite(obs["observation"])):
                    raise FloatingPointError("non-finite observation")
            except Exception as e:  # recorded, and the battle goes on
                if self.encode_error is None:
                    self.encode_error = e
                return self.choose_random_move(battle)
            self.decisions += 1
            legal = [int(i) for i in np.flatnonzero(obs["action_mask"])]
            if not legal:
                return self.choose_default_move()
            idx = legal[self._rng.randrange(len(legal))]
            self._get_tracker(battle).advance(idx)
            return self.action_to_order(idx, battle)

    common = dict(battle_format="gen3ou", server_configuration=LocalhostServerConfiguration,
                  start_listening=False, max_concurrent_battles=1)
    p1 = _EncodingRandomPlayer(1000 + key, team=t1, account_configuration=AccountConfiguration(f"Lu{key}a", "pw"), **common)
    p2 = _EncodingRandomPlayer(2000 + key, team=t2, account_configuration=AccountConfiguration(f"Lu{key}b", "pw"), **common)
    sink: list = []
    row: dict = {"key": key}
    t0 = time.time()
    try:
        asyncio.run(run_local_battles(p1, p2, 1, seed=[11 + key, 22 + key, 33 + key, 44 + key],
                                      impl="node", chunk_sink=sink, idle_budget_s=idle_budget_s,
                                      total_budget_s=total_budget_s))
        err = p1.encode_error or p2.encode_error
        if err is not None:
            row.update(ok=False, stage="encode", error_class=classify(err), error=str(err)[:500])
        else:
            row.update(ok=True)
    except BaseException as e:  # noqa: BLE001 — a crash of the battle IS the measurement
        if isinstance(e, KeyboardInterrupt):
            raise
        row.update(ok=False, stage="battle", error_class=classify(e), error=str(e)[:500])
    text = "".join(c for _, c in sink)
    battles = list(p1.battles.values())
    b = battles[0] if battles else None
    row.update(
        wall_s=round(time.time() - t0, 3), decisions=p1.decisions + p2.decisions,
        turns=(b.turn if b is not None else None), finished=bool(b is not None and b.finished),
        won=(None if b is None or not b.finished else bool(b.won)),
        called=dict(collections.Counter(m.group(1) for m in _CALLER_RE.finditer(text))),
        heal_bell="|move: Heal Bell" in text)
    return row


# ---------------------------------------------------------------------------------------------
# a unit (one process), and the driver
# ---------------------------------------------------------------------------------------------

def run_unit(out: Path, unit: int, src: Path, unit_size: int, idle_budget_s: float,
             total_budget_s: float) -> int:
    """Play every battle of ``unit`` and write its rows ATOMICALLY (tmp + rename): a unit file on
    disk is always complete, so resume can trust its presence."""
    import logging

    logging.getLogger("poke-env").setLevel(logging.CRITICAL)
    files = team_files(src)
    pairs = pairing(len(files))
    rows = []
    for key in unit_keys(unit, len(pairs), unit_size):
        a, b = pairs[key]
        row = play_battle(key, files[a].read_text(), files[b].read_text(), idle_budget_s,
                          total_budget_s)
        row.update(p1_file=files[a].name, p2_file=files[b].name, unit=unit)
        rows.append(row)
    path = unit_path(out, unit)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".jsonl.tmp")
    tmp.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))
    os.replace(tmp, path)
    return len(rows)


def write_registration(out: Path, src: Path, unit_size: int) -> dict:
    """The registered n + recipe, written ONCE; a resume with a different recipe REFUSES."""
    files = team_files(src)
    reg = {"schema": SCHEMA, "src": str(src), "teams": len(files), "seed": SEED,
           "registered_n": len(files) // 2, "unit_size": unit_size,
           "units": n_units(len(files) // 2, unit_size), "bridge": "node",
           "players": "seeded-random (1000+key / 2000+key), full embed_battle every decision",
           "sim_seed": "[11+key, 22+key, 33+key, 44+key]"}
    path = out / "registration.json"
    if path.exists():
        have = json.loads(path.read_text())
        if have != reg:
            raise SystemExit(f"{path} records a different recipe — refusing to mix: {have} != {reg}")
        return have
    out.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(reg, indent=1, sort_keys=True) + "\n")
    return reg


def drive(out: Path, src: Path, workers: int, unit_size: int, units: Optional[Sequence[int]],
          idle_budget_s: float, total_budget_s: float) -> None:
    """Run every not-yet-done unit, ``workers`` unit processes at a time. Each unit is its own
    process (``unit`` subcommand), so a wedged battle costs one unit, not the run."""
    reg = write_registration(out, src, unit_size)
    todo = [u for u in (units if units is not None else range(reg["units"])) if u not in done_units(out)]
    log = open(out / "driver.log", "a")
    print(f"{time.strftime('%F %T')} driver pid={os.getpid()} todo={len(todo)} units "
          f"(done {len(done_units(out))}/{reg['units']})", file=log, flush=True)
    running: Dict[int, subprocess.Popen] = {}
    while todo or running:
        while todo and len(running) < workers:
            u = todo.pop(0)
            argv = [sys.executable, "-m", "main.ladder_usage_smoke", "unit", "--out", str(out),
                    "--src", str(src), "--unit", str(u), "--unit-size", str(unit_size),
                    "--idle-budget", str(idle_budget_s), "--total-budget", str(total_budget_s)]
            running[u] = subprocess.Popen(argv, stdout=log, stderr=log, stdin=subprocess.DEVNULL)
        time.sleep(2)
        for u, p in list(running.items()):
            if p.poll() is not None:
                del running[u]
                print(f"{time.strftime('%F %T')} unit {u} exit {p.returncode} "
                      f"(done {len(done_units(out))}/{reg['units']})", file=log, flush=True)
    print(f"{time.strftime('%F %T')} driver finished", file=log, flush=True)


# ---------------------------------------------------------------------------------------------
# reading the rows
# ---------------------------------------------------------------------------------------------

def load_rows(out: Path) -> List[dict]:
    rows = []
    for p in sorted((out / "units").glob("unit_*.jsonl")):
        rows += [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
    return rows


def summarize(rows: Iterable[dict]) -> dict:
    rows = list(rows)
    classes: collections.Counter = collections.Counter()
    examples: Dict[str, dict] = {}
    called: collections.Counter = collections.Counter()
    for r in rows:
        if not r["ok"]:
            k = f"[{r['stage']}] {r['error_class']}"
            classes[k] += 1
            examples.setdefault(k, {"key": r["key"], "p1_file": r["p1_file"], "p2_file": r["p2_file"],
                                    "error": r["error"]})
        called.update(r.get("called") or {})
    return {
        "battles": len(rows), "ok": sum(r["ok"] for r in rows),
        "failed": sum(not r["ok"] for r in rows),
        "finished": sum(bool(r.get("finished")) for r in rows),
        "decisions": sum(r.get("decisions") or 0 for r in rows),
        "heal_bell_battles": sum(bool(r.get("heal_bell")) for r in rows),
        "battles_with_a_called_move": sum(bool(r.get("called")) for r in rows),
        "called_move_lines": dict(called.most_common()),
        "failure_classes": dict(classes.most_common()), "failure_examples": examples,
        "wall_s_total": round(sum(r.get("wall_s") or 0 for r in rows), 1),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("run", "unit", "status", "report"):
        p = sub.add_parser(name)
        p.add_argument("--out", type=Path, required=True)
        p.add_argument("--src", type=Path, default=DEFAULT_SRC)
        p.add_argument("--unit-size", type=int, default=UNIT_SIZE)
        p.add_argument("--idle-budget", type=float, default=60.0)
        p.add_argument("--total-budget", type=float, default=600.0)
        if name == "run":
            p.add_argument("--workers", type=int, default=4)
            p.add_argument("--units", default=None, help="a subset, e.g. 0-3 (a small slice)")
            p.add_argument("--detach", action="store_true",
                           help="re-exec the driver in its own session, detached from this shell")
        if name == "unit":
            p.add_argument("--unit", type=int, required=True)
    a = ap.parse_args(argv)
    out = a.out.expanduser()
    if a.cmd == "unit":
        run_unit(out, a.unit, a.src, a.unit_size, a.idle_budget, a.total_budget)
        return 0
    if a.cmd == "run":
        units = None
        if a.units:
            lo, _, hi = a.units.partition("-")
            units = list(range(int(lo), int(hi or lo) + 1))
        if a.detach:
            out.mkdir(parents=True, exist_ok=True)
            argv2 = [sys.executable, "-m", "main.ladder_usage_smoke", "run", "--out", str(out),
                     "--src", str(a.src), "--unit-size", str(a.unit_size), "--workers", str(a.workers),
                     "--idle-budget", str(a.idle_budget), "--total-budget", str(a.total_budget)]
            if a.units:
                argv2 += ["--units", a.units]
            p = subprocess.Popen(argv2, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                 stderr=subprocess.DEVNULL, start_new_session=True)
            (out / "driver.pid").write_text(f"{p.pid}\n")
            print(f"detached driver pid {p.pid}; progress: python -m main.ladder_usage_smoke status --out {out}")
            return 0
        drive(out, a.src, a.workers, a.unit_size, units, a.idle_budget, a.total_budget)
        return 0
    reg = json.loads((out / "registration.json").read_text())
    rows = load_rows(out)
    s = summarize(rows)
    if a.cmd == "status":
        print(f"PROGRESS (not a result): units {len(done_units(out))}/{reg['units']}, battles "
              f"{s['battles']}/{reg['registered_n']}, failed so far {s['failed']}")
        return 0
    if s["battles"] != reg["registered_n"]:
        print(f"REFUSED: {s['battles']} of the registered {reg['registered_n']} battles are on disk — "
              "the verdict is read at the registered n", file=sys.stderr)
        return 2
    keys = sorted(r["key"] for r in rows)
    if keys != list(range(reg["registered_n"])):
        print("REFUSED: the rows do not cover each registered key exactly once", file=sys.stderr)
        return 2
    print(json.dumps({"registration": reg, "summary": s}, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
