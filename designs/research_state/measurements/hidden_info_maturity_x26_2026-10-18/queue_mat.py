"""THE BATTERY's RUNNER: one long-lived, detached, resumable job that keeps at most ``--slots`` capped worker processes
busy (root rule 5b: <= 2 heavy jobs beside the live run; each worker one thread, its own `mem_cap` scope via
`run_mat.sh`) until every registered (anchor, bank) pair (``<archive>/ckpts.json``; `trigger_mat.py register` /
`set-mode` write it) has every row its mode needs. It reads NO outcome: it counts ids.

    python -I queue_mat.py [--slots 2]           (from the DRIVER checkout; detached: `launch_queue.sh`)
    python -I queue_mat.py --status              (progress: per bank, anchor and unit, done / wanted; no outcome)

Units (each a resumable script call, minutes to an hour; every ``--limit`` caps NEW work; every unit names its bank):
  build(b, base)       build_mat_worlds.py --shared F[,F+U,F+M] --bank b   (HEAD = the builder; network-free; once)
  build(b, hide)       build_mat_worlds.py --shared O-M,O-U --bank b      (A1: the hide-only arms, after every base unit)
  vbuild               build_mat_worlds.py --validation oracle            (HEAD; the 12 identity states, interim)
  roots(b, a)          roots_mat.py                                       (pin)
  vplay(interim, a)    play_mat_worlds.py --arm V:oracle                  (pin; the cross-commit identity, per anchor)
  truth(b, a)          phase1_mat.py, <= 20 states per unit                (pin)
  belief(b, a)         fn_belief_mat.py, <= 100 per unit                   (pin)
  fnbuild(b, a)        build_mat_worlds.py --fn a, <= 100                  (HEAD)
  play(b, a, arm)      play_mat_worlds.py, <= 60 per unit                  (pin)
Priority (:func:`units`): per bank (interim first) its base build, then its (anchor) jobs by the pair's registered
priority (roots, vplay, truth, belief, play F, fnbuild, play FN, F+M, F+U); then every bank's hide-only build and
O−M / O−U plays. At most ONE HEAD builder at a time; a slot never takes a unit another slot is running. A unit that
fails 3 times in a row is PARKED and logged (a FINDING); the rest continue. Writes ``<archive>/QUEUE_DONE`` when
nothing is pending or running (a previous run's marker is renamed aside at start). Log:
``<archive>/logs/queue_mat.log`` (+ one log per unit under ``<archive>/logs/``).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mat_common as E  # noqa: E402

LOG = E.ARCH / "logs"
RUN = E.HERE / "run_mat.sh"
MIN_BATCH = 20
BUILDERS = ("build", "vbuild", "fnbuild")
_CACHE: dict = {}


def ts() -> str:
    return time.strftime("%F %T")


def ids_in(d: Path, pattern: str) -> set:
    """The ids with a row in ``d`` (any status), re-parsing only files whose size changed since the last call."""
    out: set = set()
    if not d.exists():
        return out
    for p in sorted(d.glob(pattern)):
        size = p.stat().st_size
        hit = _CACHE.get(p)
        if hit is None or hit[0] != size:
            hit = (size, {r["id"] for r in E.read_rows(p)})
            _CACHE[p] = hit
        out |= hit[1]
    return out


def anchors() -> list:
    reg = E.registered_ckpts()
    return sorted(reg, key=lambda x: (reg[x]["priority"], reg[x]["step"]))


def jobs() -> list:
    """[(bank, anchor)] of every registered (anchor, bank) pair, by the pair's priority, then the step (A1)."""
    reg = E.registered_ckpts()
    out = [(int(e["priority"]), reg[a]["step"], b, a) for a in reg for b, e in E.anchor_banks(a).items()]
    return [(b, a) for _p, _s, b, a in sorted(out)]


def banks() -> list:
    """The banks with at least one job, interim first."""
    used = {b for b, _a in jobs()}
    return [b for b in E.BANKS if b in used]


def mode(b: str, a: str) -> str:
    return E.anchor_mode(a, b)


def bank_modes(b: str) -> set:
    return {mode(b, a) for bb, a in jobs() if bb == b}


def base_arms(b: str) -> tuple:
    return ("F", "F+U", "F+M") if bank_modes(b) & {"full", "full_hide"} else ("F",)


def want_ids(b: str, m: str) -> set:
    with E.using_bank(b):
        return set(E.mode_ids(m))


def all_ids(b: str) -> set:
    out: set = set()
    for bb, a in jobs():
        if bb == b:
            out |= want_ids(b, mode(b, a))
    return out


def _arms_of(m: str) -> tuple:
    return E.MODE_ARMS[m]


def _shared_need(b: str, arms_fn) -> dict:
    """{id: arms it needs} over every job of bank b (``arms_fn(mode)`` = the shared arms a mode needs)."""
    need: dict = {}
    for bb, a in jobs():
        if bb != b:
            continue
        m = mode(b, a)
        for x in want_ids(b, m):
            need.setdefault(x, set()).update(arms_fn(m))
    return need


def _base_of(m: str) -> set:
    return {x for x in _arms_of(m) if x in ("F", "F+U", "F+M")}


def _hide_of(m: str) -> set:
    return {x for x in _arms_of(m) if x in E.HIDE_ARMS}


def pending(u: tuple) -> int:
    kind, b = u[0], u[1]
    with E.using_bank(b):
        if kind == "build":
            need = _shared_need(b, _base_of if u[2] == "base" else _hide_of)
            arms = sorted({x for v in need.values() for x in v})
            built = {arm: ids_in(E.worlds_dir(None, arm), "worlds_*.jsonl") for arm in arms}
            return sum(1 for x, v in need.items() if any(x not in built[arm] for arm in v))
        if kind == "vbuild":
            return len(set(E.identity_ids()) - ids_in(E.worlds_dir(None, "V:oracle"), "worlds_*.jsonl"))
        a = u[2]
        ids = want_ids(b, mode(b, a))
        if kind == "roots":
            return 0 if (E.d("roots") / f"{a}.jsonl").exists() and ids <= ids_in(E.d("roots"), f"{a}.jsonl") else 1
        if kind == "truth":
            return len(ids - ids_in(E.rows_dir(a, "O"), "shard_*.jsonl"))
        if kind == "belief":
            return len(ids - ids_in(E.d("fn_belief"), f"{a}.jsonl"))
        if kind == "fnbuild":
            have = ids_in(E.d("fn_belief"), f"{a}.jsonl") & ids
            return len(have - ids_in(E.worlds_dir(a, "FN"), "worlds_*.jsonl"))
        if kind == "vplay":
            w = ids_in(E.worlds_dir(None, "V:oracle"), "worlds_*.jsonl") & set(E.identity_ids())
            return len(w - ids_in(E.rows_dir(a, "V:oracle"), "rows_*.jsonl"))
        if kind == "play":
            w = ids_in(E.worlds_dir(a, u[3]), "worlds_*.jsonl") & ids
            return len(w - ids_in(E.rows_dir(a, u[3]), "rows_*.jsonl"))
    raise ValueError(u)


def upstream(u: tuple):
    """The unit whose output this one consumes while it is still being produced (None = none)."""
    if u[0] == "play":
        if u[3] == "FN":
            return ("fnbuild", u[1], u[2])
        return ("build", u[1], "hide" if u[3] in E.HIDE_ARMS else "base")
    if u[0] == "vplay":
        return ("vbuild", "interim")
    if u[0] == "fnbuild":
        return ("belief", u[1], u[2])
    return None


def command(u: tuple) -> list:
    kind, b = u[0], u[1]
    bank = ["--bank", b]
    if kind == "build":
        need = _shared_need(b, _base_of if u[2] == "base" else _hide_of)
        arms = [x for x in E.SHARED_BUILD_ORDER if any(x in v for v in need.values())]
        m = "lean" if not bank_modes(b) & {"full", "full_hide"} else "full_hide" if u[2] == "hide" else "full"
        return ["bash", str(RUN), "head", "1", "8", "12h", "build_mat_worlds.py", "--shared", ",".join(arms),
                "--mode", m, "--limit", "100"] + bank
    if kind == "vbuild":
        return ["bash", str(RUN), "head", "1", "8", "2h", "build_mat_worlds.py", "--validation", "oracle"]
    a = u[2]
    m = mode(b, a)
    if kind == "roots":
        return ["bash", str(RUN), "pin", "1", "8", "2h", "roots_mat.py", "--cp", a] + bank
    if kind == "truth":
        return ["bash", str(RUN), "pin", "1", "6", "6h", "phase1_mat.py", "--cp", a, "--mode", m, "--limit", "20"] + bank
    if kind == "belief":
        return ["bash", str(RUN), "pin", "1", "8", "2h", "fn_belief_mat.py", "--cp", a, "--mode", m,
                "--limit", "100"] + bank
    if kind == "fnbuild":
        return ["bash", str(RUN), "head", "1", "8", "6h", "build_mat_worlds.py", "--fn", a, "--mode", m,
                "--limit", "100"] + bank
    if kind == "vplay":
        return ["bash", str(RUN), "pin", "1", "8", "2h", "play_mat_worlds.py", "--cp", a, "--arm", "V:oracle",
                "--mode", m]
    return ["bash", str(RUN), "pin", "1", "8", "6h", "play_mat_worlds.py", "--cp", a, "--arm", u[3], "--mode", m,
            "--limit", "60"] + bank


def units() -> list:
    """Every unit in PRIORITY order (A1): per bank (interim first) its base world build, then its jobs by priority
    (roots, [the identity on interim], truth, belief, F, FN worlds, FN, F+M, F+U); after every bank's base units, the
    hide-only builds and the O−M / O−U plays (the plateau step's order 1 → 2 → 3 → 5)."""
    out: list = []
    for b in banks():
        out.append(("build", b, "base"))
        if b == "interim":
            out.append(("vbuild", "interim"))
        for bb, a in jobs():
            if bb != b:
                continue
            arms = E.MODE_ARMS[mode(b, a)]
            out += [("roots", b, a)] + ([("vplay", b, a)] if b == "interim" else []) + [
                ("truth", b, a), ("belief", b, a), ("play", b, a, "F"), ("fnbuild", b, a), ("play", b, a, "FN")]
            out += [("play", b, a, x) for x in arms if x not in ("F", "FN") and x not in E.HIDE_ARMS]
    for b in banks():
        if "full_hide" not in bank_modes(b):
            continue
        out.append(("build", b, "hide"))
        for bb, a in jobs():
            if bb == b and mode(b, a) == "full_hide":
                out += [("play", b, a, x) for x in E.HIDE_ARMS]
    return out


def wanted(u: tuple) -> int:
    kind, b = u[0], u[1]
    if kind == "build":
        return len(_shared_need(b, _base_of if u[2] == "base" else _hide_of))
    if kind in ("vbuild", "vplay"):
        return len(E.identity_ids())
    if kind == "roots":
        return 1
    return len(want_ids(b, mode(b, u[2])))


def done_count(u: tuple) -> int:
    """How many of the unit's wanted ids have their output row (any status)."""
    kind, b = u[0], u[1]
    with E.using_bank(b):
        if kind == "build":
            need = _shared_need(b, _base_of if u[2] == "base" else _hide_of)
            arms = sorted({x for v in need.values() for x in v})
            built = {arm: ids_in(E.worlds_dir(None, arm), "worlds_*.jsonl") for arm in arms}
            return sum(1 for x, v in need.items() if all(x in built[arm] for arm in v))
        if kind == "vbuild":
            return len(set(E.identity_ids()) & ids_in(E.worlds_dir(None, "V:oracle"), "worlds_*.jsonl"))
        a = u[2]
        ids = set(E.identity_ids()) if kind == "vplay" else want_ids(b, mode(b, a))
        if kind == "roots":
            return int(ids <= ids_in(E.d("roots"), f"{a}.jsonl"))
        src = {"truth": (E.rows_dir(a, "O"), "shard_*.jsonl"), "belief": (E.d("fn_belief"), f"{a}.jsonl"),
               "fnbuild": (E.worlds_dir(a, "FN"), "worlds_*.jsonl"),
               "vplay": (E.rows_dir(a, "V:oracle"), "rows_*.jsonl")}.get(kind)
        if src is None:
            src = (E.rows_dir(a, u[3]), "rows_*.jsonl")
        return len(ids & ids_in(*src))


def unit_tag(u: tuple) -> str:
    return "_".join(u).replace("+", "p").replace("-", "m")


def status() -> dict:
    out = {"/".join(u): f"{done_count(u)}/{wanted(u)}" for u in units()}
    done = E.ARCH / "QUEUE_DONE"
    out["QUEUE_DONE"] = done.read_text().strip() if done.exists() else "no"
    pid = E.ARCH / "queue.pid"
    out["runner_pid"] = pid.read_text().strip() if pid.exists() else None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slots", type=int, default=2)
    ap.add_argument("--poll", type=int, default=60)
    ap.add_argument("--status", action="store_true")
    a = ap.parse_args()
    if a.status:
        print(json.dumps(status(), indent=1))
        return
    if E.PILOT_MODE:
        raise SystemExit("the queue never runs in pilot mode")
    if not anchors():
        raise SystemExit("no registered anchor (ckpts.json)")
    for b in banks():
        with E.using_bank(b):
            if not E.bank_states():
                raise SystemExit(f"the {b} bank has a job but no states file ({E.states_path()})")
    LOG.mkdir(parents=True, exist_ok=True)
    qlog = open(LOG / "queue_mat.log", "a")
    done_marker = E.ARCH / "QUEUE_DONE"
    if done_marker.exists():                      # a previous run's marker: kept, renamed aside
        done_marker.rename(E.ARCH / f"QUEUE_DONE.{time.strftime('%Y%m%dT%H%M%S')}")

    def say(msg):
        qlog.write(f"{ts()} {msg}\n")
        qlog.flush()

    running: dict = {}
    fails: dict = {}
    parked: set = set()
    say(f"queue up: {a.slots} slots, jobs {jobs()}, HERE {E.HERE}, driver {os.environ.get('HIB_MAT_DRIVER_SHA')}, "
        f"builder {os.environ.get('HIB_MAT_HEAD')} @ {os.environ.get('HIB_MAT_HEAD_SHA')}")
    while True:
        for s, (u, p, fh) in list(running.items()):
            rc = p.poll()
            if rc is None:
                continue
            fh.close()
            del running[s]
            if rc == 0:
                fails[u] = 0
                say(f"slot {s} done {u}")
            else:
                fails[u] = fails.get(u, 0) + 1
                say(f"slot {s} FAILED {u} rc={rc} ({fails[u]} in a row)")
                if fails[u] >= 3:
                    parked.add(u)
                    say(f"PARKED {u} after 3 failures in a row (a FINDING; see its log)")
        busy = {u for u, _, _ in running.values()}
        build_busy = any(u[0] in BUILDERS for u in busy)
        for s in range(a.slots):
            if s in running:
                continue
            pick = None
            for u in units():
                if u in busy or u in parked or (u[0] in BUILDERS and build_busy):
                    continue
                n = pending(u)
                up = upstream(u)
                if n > 0 and up is not None and up not in parked and n < MIN_BATCH and pending(up) > 0:
                    continue                     # wait for a batch while the upstream is still producing
                if n > 0:
                    pick = u
                    break
            if pick is None:
                continue
            fh = open(LOG / f"{unit_tag(pick)}.log", "a")
            p = subprocess.Popen(command(pick), stdout=fh, stderr=subprocess.STDOUT, cwd=str(E.HERE))
            running[s] = (pick, p, fh)
            busy.add(pick)
            build_busy = build_busy or pick[0] in BUILDERS
            say(f"slot {s} start {pick} ({pending(pick)} pending) pid {p.pid}")
        if not running:
            left = {u: pending(u) for u in units()}
            left = {u: n for u, n in left.items() if n > 0}
            if not left or set(left) <= parked:
                (E.ARCH / "QUEUE_DONE").write_text(f"{ts()} parked={sorted(parked)}\n")
                say(f"QUEUE DONE (parked: {sorted(parked)})")
                return
        time.sleep(a.poll)


if __name__ == "__main__":
    main()
