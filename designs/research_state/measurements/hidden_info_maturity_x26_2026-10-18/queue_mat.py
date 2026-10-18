"""THE BATTERY's RUNNER: one long-lived, detached, resumable job that keeps at most ``--slots`` capped worker processes
busy (root rule 5b: <= 2 heavy jobs beside the live run; each worker one thread, its own `mem_cap` scope via
`run_mat.sh`) until every registered anchor (``<archive>/ckpts.json``; `trigger_mat.py register` appends) has every row
its mode needs. It reads NO outcome: it counts ids.

    python -I queue_mat.py [--slots 2]           (from the HEAD checkout; detached: `launch_queue.sh`)
    python -I queue_mat.py --status              (progress: per anchor and unit, done / wanted; no outcome)

Units (each a resumable script call, minutes to an hour; every ``--limit`` caps NEW work):
  build(shared)        build_mat_worlds.py --shared F[,F+M,F+U]  (HEAD; network-free; once for every anchor)
  vbuild               build_mat_worlds.py --validation oracle   (HEAD; the 12 identity states)
  roots(a)             roots_mat.py                              (pin)
  vplay(a)             play_mat_worlds.py --arm V:oracle         (pin; the cross-commit identity)
  truth(a)             phase1_mat.py, <= 20 states per unit       (pin)
  belief(a)            fn_belief_mat.py, <= 100 per unit          (pin)
  fnbuild(a)           build_mat_worlds.py --fn a, <= 100         (HEAD)
  play(a, arm)         play_mat_worlds.py, <= 60 per unit         (pin)
Priority: anchors by their registered ``priority`` (x26: 25M, 225M, 100M, 175M), and within an anchor roots, vplay,
truth, belief, play F, fnbuild, play FN (+ FULL's F+M, F+U). At most ONE HEAD builder at a time; a slot never takes a
unit another slot is running. A unit that fails 3 times in a row is PARKED and logged (a FINDING); the rest continue.
Writes ``<archive>/QUEUE_DONE`` when nothing is pending or running (a later `trigger_mat.py register` + relaunch
resumes). Log: ``<archive>/logs/queue_mat.log`` (+ one log per unit under ``<archive>/logs/``).
"""
from __future__ import annotations

import argparse
import json
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


def mode(a: str) -> str:
    return E.registered_ckpts()[a]["mode"]


def shared_arms() -> tuple:
    return ("F", "F+U", "F+M") if any(mode(a) == "full" for a in anchors()) else ("F",)


def want_ids(m: str) -> set:
    return set(E.mode_ids(m))


def all_ids() -> set:
    out: set = set()
    for a in anchors():
        out |= want_ids(mode(a))
    return out


def pending(u: tuple) -> int:
    kind = u[0]
    if kind == "build":
        built = {arm: ids_in(E.worlds_dir(None, arm), "worlds_*.jsonl") for arm in shared_arms()}
        full = set(E.mode_ids("full")) if any(mode(a) == "full" for a in anchors()) else set()
        return sum(1 for x in all_ids()
                   if x not in built["F"] or (x in full and any(x not in built[arm] for arm in ("F+U", "F+M"))))
    if kind == "vbuild":
        return len(set(E.identity_ids()) - ids_in(E.worlds_dir(None, "V:oracle"), "worlds_*.jsonl"))
    a = u[1]
    ids = want_ids(mode(a))
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
        w = ids_in(E.worlds_dir(a, u[2]), "worlds_*.jsonl") & ids
        return len(w - ids_in(E.rows_dir(a, u[2]), "rows_*.jsonl"))
    raise ValueError(u)


def upstream(u: tuple):
    """The unit whose output this one consumes while it is still being produced (None = none)."""
    if u[0] == "play":
        return ("fnbuild", u[1]) if u[2] == "FN" else ("build", "shared")
    if u[0] == "vplay":
        return ("vbuild", "shared")
    if u[0] == "fnbuild":
        return ("belief", u[1])
    return None


def command(u: tuple) -> list:
    kind = u[0]
    if kind == "build":
        return ["bash", str(RUN), "head", "1", "8", "12h", "build_mat_worlds.py", "--shared", ",".join(shared_arms()),
                "--mode", "full" if any(mode(a) == "full" for a in anchors()) else "lean", "--limit", "100"]
    if kind == "vbuild":
        return ["bash", str(RUN), "head", "1", "8", "2h", "build_mat_worlds.py", "--validation", "oracle"]
    a = u[1]
    m = mode(a)
    if kind == "roots":
        return ["bash", str(RUN), "pin", "1", "8", "2h", "roots_mat.py", "--cp", a]
    if kind == "truth":
        return ["bash", str(RUN), "pin", "1", "6", "6h", "phase1_mat.py", "--cp", a, "--mode", m, "--limit", "20"]
    if kind == "belief":
        return ["bash", str(RUN), "pin", "1", "8", "2h", "fn_belief_mat.py", "--cp", a, "--mode", m, "--limit", "100"]
    if kind == "fnbuild":
        return ["bash", str(RUN), "head", "1", "8", "6h", "build_mat_worlds.py", "--fn", a, "--mode", m,
                "--limit", "100"]
    if kind == "vplay":
        return ["bash", str(RUN), "pin", "1", "8", "2h", "play_mat_worlds.py", "--cp", a, "--arm", "V:oracle",
                "--mode", m]
    return ["bash", str(RUN), "pin", "1", "8", "6h", "play_mat_worlds.py", "--cp", a, "--arm", u[2], "--mode", m,
            "--limit", "60"]


def units() -> list:
    out = [("build", "shared"), ("vbuild", "shared")]
    for a in anchors():
        arms = E.MODE_ARMS[mode(a)]
        out += [("roots", a), ("vplay", a), ("truth", a), ("belief", a), ("play", a, "F"), ("fnbuild", a),
                ("play", a, "FN")] + [("play", a, x) for x in arms if x not in ("F", "FN")]
    return out


def wanted(u: tuple) -> int:
    kind = u[0]
    if kind in ("build",):
        return len(all_ids())
    if kind in ("vbuild", "vplay"):
        return len(E.identity_ids())
    if kind == "roots":
        return 1
    return len(want_ids(mode(u[1])))


def done_count(u: tuple) -> int:
    """How many of the unit's wanted ids have their output row (any status)."""
    kind = u[0]
    if kind == "build":
        return len(all_ids() & ids_in(E.worlds_dir(None, "F"), "worlds_*.jsonl"))
    if kind == "vbuild":
        return len(set(E.identity_ids()) & ids_in(E.worlds_dir(None, "V:oracle"), "worlds_*.jsonl"))
    a = u[1]
    ids = set(E.identity_ids()) if kind == "vplay" else want_ids(mode(a))
    if kind == "roots":
        return int(ids <= ids_in(E.d("roots"), f"{a}.jsonl"))
    src = {"truth": (E.rows_dir(a, "O"), "shard_*.jsonl"), "belief": (E.d("fn_belief"), f"{a}.jsonl"),
           "fnbuild": (E.worlds_dir(a, "FN"), "worlds_*.jsonl"),
           "vplay": (E.rows_dir(a, "V:oracle"), "rows_*.jsonl")}.get(kind)
    if src is None:
        src = (E.rows_dir(a, u[2]), "rows_*.jsonl")
    return len(ids & ids_in(*src))


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
    if not E.bank_states() or not anchors():
        raise SystemExit("no bank (states_mat_x26.json) or no registered anchor (ckpts.json)")
    LOG.mkdir(parents=True, exist_ok=True)
    qlog = open(LOG / "queue_mat.log", "a")

    def say(msg):
        qlog.write(f"{ts()} {msg}\n")
        qlog.flush()

    running: dict = {}
    fails: dict = {}
    parked: set = set()
    say(f"queue up: {a.slots} slots, anchors {anchors()}, HERE {E.HERE}")
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
            tag = "_".join(pick).replace("+", "p").replace("-", "m")
            fh = open(LOG / f"{tag}.log", "a")
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
