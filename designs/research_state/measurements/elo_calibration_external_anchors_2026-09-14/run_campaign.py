#!/usr/bin/env python3
"""Drive the whole external-anchor campaign — one `python -m main.anchors` process per cell.

    python3 run_campaign.py --out <dir> --lanes 3 [--games 100] [--only bot|snapshot] [--dry-run]

**Resumable and idempotent.** A cell whose `summary.json` already reads `status: "OK"` with the
full game count is SKIPPED, so an interrupted campaign resumes where it stopped and a re-run costs
nothing. A FAILED cell is retried (its old directory is moved aside with a `.failed.<n>` suffix, so
the evidence of the failure survives — a failure that gets overwritten by its own retry is a
failure nobody can diagnose).

**One lane = one Showdown server on its own 95XX port.** Lanes are fully independent processes, so
usernames cannot collide across lanes even though they are identical: each lane's clients speak
only to that lane's server. Everything runs `nice` and CPU-only — a training arm owns the GPU and
other agents own their own ports.

🚨 This script NEVER touches :8000 or :8001, never writes under `models/`, and kills only the PIDs
it started (each `main.anchors` process stops its own server by PID on exit).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import cells as cells_mod  # noqa: E402

#: The worktree whose `src/` the campaign imports. A long campaign that imports the MAIN checkout
#: is not pinned, and `main` is a moving target on a multi-session box — the 2026-09-14 Foul Play
#: session died silently 40 games in when an unrelated commit landed mid-run.
REPO = str(HERE.parents[3])
PYTHON = "/home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3"


def cell_dir(out: Path, cell: dict) -> Path:
    return out / "cells" / cell["id"]


def games_for(cell: dict, default: int) -> int:
    """A cell may name its own count. The head-to-head is 200 because it is the transitivity
    check on the whole fit and a near-even 100-game cell resolves +/-10 pp at best."""
    return int(cell.get("games") or default)


def cell_is_done(out: Path, cell: dict, games: int) -> bool:
    path = cell_dir(out, cell) / "summary.json"
    if not path.exists():
        return False
    try:
        blob = json.loads(path.read_text())
    except (OSError, ValueError):
        return False
    return blob.get("status") == "OK" and int(blob.get("n") or 0) >= games


def retire_failed(d: Path) -> None:
    """Move a failed attempt aside rather than overwriting it — the log IS the diagnosis."""
    if not d.exists():
        return
    n = 1
    while (alt := d.with_suffix(f".failed.{n}")).exists():
        n += 1
    d.rename(alt)


def argv_for(cell: dict, out: Path, port: int, games: int) -> "list[str]":
    return [PYTHON, "-m", "main.anchors",
            "--regime", "greedy", "--teamset", "home",
            "--games", str(games), "--port", str(port), "--device", "cpu",
            "--out", str(cell_dir(out, cell)), *cell["argv"]]


def run_cell(cell: dict, out: Path, port: int, games: int, nice: int) -> int:
    d = cell_dir(out, cell)
    d.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["PYTHONPATH"] = os.path.join(REPO, "src") + ":" + env.get("PYTHONPATH", "")
    env["CUDA_VISIBLE_DEVICES"] = ""      # a training arm owns the GPU
    # ONE THREAD for OUR side too, for the reason spelled out in `main.anchors.peers`: this is
    # B=1 CPU inference and the parallelism belongs across lanes, not inside a forward pass.
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "TORCH_NUM_THREADS"):
        env[k] = "1"
    argv = ["nice", "-n", str(nice), *argv_for(cell, out, port, games)]
    with open(d / "cell.log", "w", buffering=1) as fh:
        proc = subprocess.Popen(argv, cwd=REPO, env=env, stdout=fh,
                                stderr=subprocess.STDOUT)
        return proc.wait()


def lane(cells: "list[dict]", out: Path, base_port: int, games: int, nice: int,
         results: dict, lock: threading.Lock) -> None:
    for i, cell in enumerate(cells):
        # Rotate the port within the lane's own block of 10. The previous cell's Showdown was
        # stopped by PID a moment ago and the socket can still be in TIME_WAIT; reusing the same
        # number immediately turns a clean sequential campaign into a bind failure every few
        # cells, which is a whole cell lost to bookkeeping.
        port = base_port + (i % 8)
        n = games_for(cell, games)
        if cell_is_done(out, cell, n):
            with lock:
                print(f"[lane {base_port}] SKIP {cell['id']} (already OK)", flush=True)
            continue
        retire_failed(cell_dir(out, cell))
        t0 = time.time()
        rc = run_cell(cell, out, port, n, nice)
        ok = cell_is_done(out, cell, n)
        with lock:
            results[cell["id"]] = {"rc": rc, "ok": ok, "s": round(time.time() - t0, 1)}
            print(f"[lane {base_port}] {'OK  ' if ok else 'FAIL'} {cell['id']} "
                  f"rc={rc} in {time.time() - t0:.0f}s", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lanes", type=int, default=3)
    ap.add_argument("--games", type=int, default=cells_mod.GAMES_PER_CELL)
    ap.add_argument("--base-port", type=int, default=9500)
    ap.add_argument("--nice", type=int, default=15)
    ap.add_argument("--only", choices=("bot", "snapshot", "h2h"), default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    todo = [c for c in cells_mod.cells() if args.only is None or c["kind"] == args.only]
    pending = [c for c in todo if not cell_is_done(out, c, games_for(c, args.games))]
    print(f"{len(todo)} cells, {len(pending)} still to run, {args.games} games each "
          f"({sum(games_for(c, args.games) for c in pending)} games), {args.lanes} lanes "
          f"from port {args.base_port}")
    if args.dry_run:
        for c in pending:
            print("  ", c["id"], " ".join(c["argv"]))
        return 0

    out.mkdir(parents=True, exist_ok=True)
    # Round-robin rather than contiguous blocks: the two anchors cost very differently
    # (SyntheticRLV2 is 200M params and pays a multi-minute build per half), so a contiguous
    # split would leave one lane idle for hours.
    lanes = [pending[i::args.lanes] for i in range(args.lanes)]
    results: dict = {}
    lock = threading.Lock()
    threads = [threading.Thread(target=lane, args=(
        lanes[i], out, args.base_port + 10 * i, args.games, args.nice, results, lock))
        for i in range(args.lanes)]
    t0 = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    ok = sum(1 for v in results.values() if v["ok"])
    print(f"\n{ok}/{len(pending)} cells OK in {(time.time() - t0) / 60:.0f} min")
    (out / "campaign_status.json").write_text(json.dumps(results, indent=1, sort_keys=True))
    return 0 if ok == len(pending) else 1


if __name__ == "__main__":
    sys.exit(main())
