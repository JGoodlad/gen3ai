"""STAGE 1's RUNNER (``PREREGISTRATION.md`` §5): ONE heavy job at a time (root rule 5b; the brief). The states go in the
REGISTERED order (``s1_common.ordered_ids``) in chunks: the BUILDER builds every arm of a chunk, then the PIN plays every
arm of it, then the next chunk; so the complete states are always a prefix of the order (a uniform random sample of the
287). Detached (``launch_s1.sh``), resumable (every driver skips what it holds), and bounded:

  * THE REGISTERED STOP: no chunk STARTS after ``<archive>/STOP_AT`` (written once, at the first launch); a chunk in
    flight finishes (each driver runs under its own timeout). ``<archive>/STOPPED`` then records the time and the count.
  * ``QUEUE_DONE`` when every registered state is complete in every arm.
  * A chunk whose drivers fail twice without progress PARKS its first incomplete state (``<archive>/PARKED``, with the
    reason): a parked state is excluded from the read and is a FINDING.
  * ``--one-chunk`` (2026-10-20, the orchestrator's heavy-slot policy): ONE chunk, then exit; ``loop_s1.sh`` runs it
    in a loop, each call inside its OWN heavy slot (``mem_cap.sh --slot``), released between chunks.

    python -I queue_s1.py [--chunk 6]          (launched by launch_s1.sh; never by hand beside a live runner)
    python -I queue_s1.py --status             (counts only: complete / built / played per arm; no outcome)
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import s1_common as E  # noqa: E402

BUILD_TIMEOUT = "6h"
PLAY_TIMEOUT = "6h"


def log(msg: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    with open(E.d("logs") / "queue.log", "a") as f:
        f.write(line + "\n")
    print(line, flush=True)


def parked() -> dict:
    p = E.ARCH / "PARKED"
    return json.loads(p.read_text()) if p.exists() else {}


def state_of():
    ids = [i for i in E.ordered_ids() if i not in parked()]
    worlds = {a: E.read_by_id(E.worlds_path(a)) for a in E.ARMS}
    rows = {a: E.read_by_id(E.rows_path(a)) for a in E.ARMS}
    return ids, worlds, rows


def status() -> dict:
    ids, worlds, rows = state_of()
    complete = [i for i in ids if all(i in rows[a] for a in E.ARMS)]
    return {"registered": E.N_STATES, "parked": len(parked()), "complete": len(complete),
            "built": {a: sum(1 for i in ids if i in worlds[a]) for a in E.ARMS},
            "played": {a: sum(1 for i in ids if i in rows[a]) for a in E.ARMS},
            "stop_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(E.stop_at())) if E.stop_at() else None,
            "done": (E.ARCH / "QUEUE_DONE").exists(), "stopped": (E.ARCH / "STOPPED").exists()}


def run(where: str, script: str, ids, tag: str) -> int:
    """One driver call through ``run_s1.sh`` (``$WS1_RUN_S1``: the BUILDER checkout's copy, so the rows' drivers stay
    the launch commit's whichever checkout this runner runs from)."""
    lf = E.d("logs") / f"{tag}.log"
    run_s1 = os.environ.get("WS1_RUN_S1") or str(HERE / "run_s1.sh")
    cmd = ["bash", run_s1, where, "1", "8", BUILD_TIMEOUT if where == "builder" else PLAY_TIMEOUT,
           script, "--ids", ",".join(ids)]
    with open(lf, "a") as f:
        f.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} {' '.join(cmd[:7])} ({len(ids)} ids)\n")
        f.flush()
        return subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=os.environ.copy()).returncode


def _fails() -> dict:
    p = E.ARCH / "fails.json"
    return json.loads(p.read_text()) if p.exists() else {}


def step(chunk_size: int) -> str:
    """ONE chunk (module docstring), or the end: ``"stopped"`` (the registered stop), ``"done"`` (QUEUE_DONE),
    ``"chunk"``. The no-progress counts persist in ``<archive>/fails.json`` (a per-chunk runner is a new process)."""
    if time.time() >= (E.stop_at() or 0):
        st = status()
        (E.ARCH / "STOPPED").write_text(json.dumps({"at": time.strftime("%Y-%m-%d %H:%M:%S"), **st}) + "\n")
        log(f"THE REGISTERED STOP: {st['complete']} states complete; no new chunk")
        return "stopped"
    ids, worlds, rows = state_of()
    pending = [i for i in ids if not all(i in rows[x] for x in E.ARMS)]
    if not pending:
        (E.ARCH / "QUEUE_DONE").write_text(json.dumps({"at": time.strftime("%Y-%m-%d %H:%M:%S"),
                                                       "parked": parked()}) + "\n")
        log("QUEUE_DONE")
        return "done"
    chunk = pending[:chunk_size]
    tag = time.strftime("%Y%m%d-%H%M%S")
    before = sum(1 for i in chunk for x in E.ARMS if i in rows[x]) + sum(1 for i in chunk for x in E.ARMS
                                                                        if i in worlds[x])
    need_build = [i for i in chunk if not all(i in worlds[x] for x in E.ARMS)]
    rc_b = run("builder", "build_s1.py", need_build, f"build_{tag}") if need_build else 0
    rc_p = run("pin", "play_s1.py", chunk, f"play_{tag}")
    ids2, worlds2, rows2 = state_of()
    after = sum(1 for i in chunk for x in E.ARMS if i in rows2[x]) + sum(1 for i in chunk for x in E.ARMS
                                                                        if i in worlds2[x])
    done = sum(1 for i in ids2 if all(i in rows2[x] for x in E.ARMS))
    log(f"chunk {tag} ({len(chunk)} states): build rc {rc_b}, play rc {rc_p}, rows {before} -> {after}; "
        f"complete {done}")
    if after == before:
        fails = _fails()
        k = chunk[0]
        fails[k] = fails.get(k, 0) + 1
        (E.ARCH / "fails.json").write_text(json.dumps(fails, indent=1) + "\n")
        if fails[k] >= 2:
            p = parked()
            p[k] = f"no progress in two chunks (build rc {rc_b}, play rc {rc_p})"
            (E.ARCH / "PARKED").write_text(json.dumps(p, indent=1) + "\n")
            log(f"PARKED {k}: {p[k]}")
    return "chunk"


#: ``--one-chunk`` exit codes (``loop_s1.sh`` reads them)
EXIT_CHUNK, EXIT_DONE, EXIT_STOPPED = 0, 3, 4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk", type=int, default=6)
    ap.add_argument("--one-chunk", action="store_true",
                    help="run ONE chunk and exit (0 a chunk ran, 3 QUEUE_DONE, 4 the stop): loop_s1.sh wraps each "
                         "call in its own heavy slot (the orchestrator's policy, 2026-10-20)")
    ap.add_argument("--status", action="store_true")
    a = ap.parse_args()
    if a.status:
        print(json.dumps(status(), indent=1))
        return
    if E.stop_at() is None:
        raise SystemExit("no STOP_AT: launch through launch_s1.sh")
    if a.one_chunk:
        r = step(a.chunk)
        sys.exit({"chunk": EXIT_CHUNK, "done": EXIT_DONE, "stopped": EXIT_STOPPED}[r])
    log(f"runner up: pid {os.getpid()}, driver {os.environ.get('WS1_DRIVER_SHA', '?')[:8]}, chunk {a.chunk}")
    while step(a.chunk) == "chunk":
        pass


if __name__ == "__main__":
    main()
