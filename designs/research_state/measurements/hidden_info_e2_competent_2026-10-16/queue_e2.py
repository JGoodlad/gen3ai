"""E2's RUNNER: one long-lived, detached, resumable job that keeps at most ``--slots`` capped worker processes busy (root
rule 5b: ≤ 2 heavy jobs beside the live run; each worker one thread, its own `mem_cap` scope via `run_e2.sh`) until every
(state, source) of the E2 bank has a row. It reads NO outcome: it counts ids.

    python -I queue_e2.py [--slots 2]           (from the E2 HEAD checkout; detached: `launch_queue.sh`)

Units (each a resumable script call; a unit is minutes to an hour):
  build(cp)            build_e2_worlds.py, every arm, every state (ONE builder at a time, at the E2 HEAD)
  vbuild(cp) / vplay   the cross-commit identity check (§5): `oracle` worlds on 12 fixed states, played at the pad pin
  truth(cp)            phase1_e2.py, ≤ 30 states per unit (the plain pin)
  play(cp, arm)        play_e2_worlds.py, ≤ 100 states per unit (the pad pin)
  regate(cp, arm)      pin_regate_e2.py, every accepted world (the pad pin; descriptive; once, last)
Priority: end2 before prod; within a checkpoint, the identity check, then the truth, then the arms in F, F+U, F+M, O−M,
O−U order. A slot never takes a unit another slot is running. A unit that fails 3 times in a row is parked and logged
(a FINDING for the reader); the rest continue. Writes ``<archive>/QUEUE_DONE`` when nothing is pending or running.
Log: ``<archive>/logs/queue_e2.log`` (+ one log per unit kind under ``<archive>/logs/``).
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e2_common as E  # noqa: E402

LOG = E.ARCH / "logs"
RUN = E.HERE / "run_e2.sh"
N_IDENTITY = 12


def ts() -> str:
    return time.strftime("%F %T")


def identity_ids(cp: str) -> Path:
    """The 12 identity states: the first 12 E2 ids in sha256('hib_e2_identity_v1:' + id) order (fixed, no outcome)."""
    import hashlib

    p = E.cp_dir(cp) / "identity_ids.txt"
    if not p.exists():
        ids = sorted(E.e2_states(cp), key=lambda x: hashlib.sha256(("hib_e2_identity_v1:" + x).encode()).hexdigest())
        p.write_text("\n".join(ids[:N_IDENTITY]) + "\n")
    return p


_CACHE: dict = {}
MIN_BATCH = 20


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


def pending(unit: tuple) -> int:
    kind, cp = unit[0], unit[1]
    st = set(E.e2_states(cp))
    if not st:
        return 0
    if kind == "build":
        built = [ids_in(E.cp_dir(cp) / "worlds" / E.arm_dir(a), "worlds_*.jsonl") for a in E.ARMS]
        return len({x for x in st if any(x not in b for b in built)})
    if kind == "vbuild":
        want = set(identity_ids(cp).read_text().split())
        return len(want - ids_in(E.cp_dir(cp) / "worlds" / "V_oracle", "worlds_*.jsonl"))
    if kind == "truth":
        return len(st - ids_in(E.rows_dir(cp, "O"), "shard_*.jsonl"))
    if kind == "regate":                         # once, after that checkpoint's worlds are all built
        done = (E.HERE / f"pin_regate_e2_{cp}_{E.arm_dir(unit[2])}.json").exists()
        return 0 if done or pending(("build", cp)) > 0 else 1
    sub = "V_oracle" if kind == "vplay" else E.arm_dir(unit[2])
    w = ids_in(E.cp_dir(cp) / "worlds" / sub, "worlds_*.jsonl")
    return len(w - ids_in(E.cp_dir(cp) / "rows" / sub, "rows_*.jsonl"))


def command(unit: tuple) -> list:
    kind, cp = unit[0], unit[1]
    if kind == "build":
        return ["bash", str(RUN), "head", "1", "8", "12h", "build_e2_worlds.py", "--cp", cp]
    if kind == "vbuild":
        return ["bash", str(RUN), "head", "1", "8", "2h", "build_e2_worlds.py", "--cp", cp, "--validation", "oracle",
                "--ids", str(identity_ids(cp))]
    if kind == "truth":
        return ["bash", str(RUN), cp, "1", "6", "6h", "phase1_e2.py", "--limit", "30"]
    if kind == "vplay":
        return ["bash", str(RUN), f"{cp}_pad", "1", "8", "4h", "play_e2_worlds.py", "--arm", "V:oracle"]
    if kind == "regate":
        return ["bash", str(RUN), f"{cp}_pad", "1", "8", "6h", "pin_regate_e2.py", "--arm", unit[2]]
    return ["bash", str(RUN), f"{cp}_pad", "1", "8", "6h", "play_e2_worlds.py", "--arm", unit[2], "--limit", "100"]


def units() -> list:
    out = []
    for cp in E.ORDER:
        out += [("vbuild", cp), ("vplay", cp), ("truth", cp)] + [("play", cp, a) for a in E.BUILD_ORDER]
    for cp in E.ORDER:                           # the descriptive pin re-gate (§5), after every playout unit
        out += [("regate", cp, a) for a in E.BUILD_ORDER]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slots", type=int, default=2)
    ap.add_argument("--poll", type=int, default=60)
    a = ap.parse_args()
    LOG.mkdir(parents=True, exist_ok=True)
    qlog = open(LOG / "queue_e2.log", "a")

    def say(msg):
        qlog.write(f"{ts()} {msg}\n")
        qlog.flush()

    missing = [cp for cp in E.ORDER if not E.e2_states(cp)]
    if missing:
        say(f"no states_e2 file for {missing}: those checkpoints wait (the bank is built and committed first)")
    running: dict = {}                       # slot -> (unit, Popen, log handle)
    fails: dict = {}
    parked: set = set()
    say(f"queue up: {a.slots} slots, HERE {E.HERE}")
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
        build_busy = any(u[0] in ("build", "vbuild") for u in busy)
        for s in range(a.slots):
            if s in running:
                continue
            cand = []
            if not build_busy:
                cand += [("build", cp) for cp in E.ORDER]
            cand += units()
            pick = None
            for u in cand:
                if u in busy or u in parked or (u[0] in ("build", "vbuild") and build_busy):
                    continue
                n = pending(u)
                if u[0] in ("play", "vplay"):
                    upstream = ("build", u[1]) if u[0] == "play" else ("vbuild", u[1])
                    if n < MIN_BATCH and pending(upstream) > 0:
                        continue                 # wait for a batch while that checkpoint's worlds are still building
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
            build_busy = build_busy or pick[0] in ("build", "vbuild")
            say(f"slot {s} start {pick} ({pending(pick)} pending) pid {p.pid}")
        if not running:
            left = {u: pending(u) for u in [("build", cp) for cp in E.ORDER] + units()}
            left = {u: n for u, n in left.items() if n > 0}
            if not left or set(left) <= parked:
                (E.ARCH / "QUEUE_DONE").write_text(f"{ts()} parked={sorted(parked)}\n")
                say(f"QUEUE DONE (parked: {sorted(parked)})")
                return
            if all(not E.e2_states(cp) for cp in E.ORDER):
                say("nothing to do: no bank")
                return
        time.sleep(a.poll)


if __name__ == "__main__":
    main()
