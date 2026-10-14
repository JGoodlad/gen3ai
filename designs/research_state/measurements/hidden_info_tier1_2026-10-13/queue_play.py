"""THE PLAYOUT QUEUE: one long-lived job that keeps ONE playout worker busy (2 threads) in the registered priority order,
re-scanning as the world builders land new worlds, until every applicable (state, arm) has a row.

    python -I queue_play.py [--threads 2] [--shards 1] [--labels F,F+M,...]  (detached; each unit runs via run_pin.sh)

Priority (the decisive contrasts first): F, F+M, O-M, F+U, O-U on the registered 600 (R-MOVES B2-B4, B8, R1, R3),
the supplement's TRUTH (Phase 1), D90, W90, D50 (B6, B7, R2), FN (TODAY), the supplement arms (the stage rules),
MAP100, POOLSETS, POOL. A unit = one `play_worlds.py` call on whatever worlds exist (resumable); each unit is minutes to
an hour. With ``--shards N`` a unit runs N `play_worlds` shards at once (only when the CPU allowance permits).
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import t1_common as T  # noqa: E402

LOG = T.ARCH / "logs"
ORDER = [("reg", "F"), ("reg", "F+M"), ("reg", "O-M"), ("reg", "F+U"), ("reg", "O-U"), ("truth", None),
         ("reg", "D90"), ("reg", "W90"), ("reg", "D50"), ("reg", "FN"), ("supp", "F"), ("supp", "F+M"),
         ("supp", "O-M"), ("supp", "F+U"), ("supp", "O-U"), ("supp", "FN"), ("reg", "MAP100"), ("reg", "POOLSETS"),
         ("reg", "POOL")]


def pending(cp: str, which: str, label, key) -> int:
    if which == "truth":
        st = T.supplement_states(cp)
        done = set(T.read_dir(T.cp_dir(cp) / "rows_supp", "shard_*.jsonl"))
        return len(set(st) - done)
    states = "registered" if which == "reg" else "supp"
    w = T.read_dir(T.worlds_dir(cp, key[label]), f"worlds_{states}_*.jsonl")
    r = T.read_dir(T.rows_dir(cp, key[label]), "rows_*.jsonl")
    return len(set(w) - set(r))


def total_missing(cp: str, key) -> int:
    n = pending(cp, "truth", None, key)
    for which, lab in ORDER:
        if lab is None:
            continue
        states = "registered" if which == "reg" else "supp"
        if states == "supp" and T.ARMS[lab]["states"] != "all":
            continue
        ids = set(T.registered_states(cp)) if states == "registered" else set(T.supplement_states(cp))
        r = T.read_dir(T.rows_dir(cp, key[lab]), "rows_*.jsonl")
        n += len(ids - set(r))
    return n


def run_unit(cp: str, which: str, label, key, threads: int, shards: int) -> None:
    here = T.HERE
    procs = []
    for i in range(shards):
        if which == "truth":
            args = ["bash", str(here / "run_pin.sh"), cp, str(threads), "8", "12h", "phase1_supp.py", "--shard", str(i),
                    "--of", str(shards)]
            tag = f"phase1_supp_{cp}"
        else:
            states = "registered" if which == "reg" else "supp"
            args = ["bash", str(here / "run_pin.sh"), f"{cp}_pad", str(threads), "8", "12h", "play_worlds.py", "--code",
                    key[label], "--states", states, "--shard", str(i), "--of", str(shards)]
            tag = f"play_{cp}_{key[label]}_{states}"
        f = open(LOG / f"{tag}_{i}_of_{shards}.log", "a")
        procs.append((subprocess.Popen(args, stdout=f, stderr=subprocess.STDOUT), f))
    for p, f in procs:
        p.wait()
        f.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--min-batch", type=int, default=20, help="run a unit only when this many states are pending "
                                                                "(or the builders are done)")
    a = ap.parse_args()
    key = T.load_key()
    LOG.mkdir(parents=True, exist_ok=True)
    while True:
        did = False
        for which, lab in ORDER:
            for cp in T.CPS:
                if which == "supp" and T.ARMS[lab]["states"] != "all":
                    continue
                n = pending(cp, which, lab, key)
                if n >= a.min_batch or (n > 0 and (T.ARCH / "BUILDS_DONE").exists()):
                    print(time.strftime("%H:%M:%S"), f"unit {cp} {which} {lab}: {n} pending", flush=True)
                    run_unit(cp, which, lab, key, a.threads, a.shards)
                    did = True
                    break
            if did:
                break
        if not did:
            miss = {cp: total_missing(cp, key) for cp in T.CPS}
            if all(v == 0 for v in miss.values()):
                print(time.strftime("%H:%M:%S"), "queue complete", flush=True)
                return
            print(time.strftime("%H:%M:%S"), f"waiting for worlds: {miss}", flush=True)
            time.sleep(600)


if __name__ == "__main__":
    main()
