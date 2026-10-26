r"""X59-S's GPU WINDOW: every registered student trained under the lease, each under ``mem_cap.sh`` (PREREG Amendment 2,
REVIEW E M7).

    GEN3AI_GPU_LEASE_TOKEN=<token> <PY> -m main.human_data.window --dir <students dir> --registration <json> \
        --human-rows <r2>/rows --labels <r2>/labels_pstar --labels0 <r2>/labels_pstar0 --selfplay-rows <ss>/ \
        [--concurrency 3] [--dry-run]

* The order is registered: the 15 primary students (seeds 5901–5905 × S-H, S-P, S-S), then the 5 descriptive S-P0
  students, so S-P0 never delays a primary one (:func:`jobs`).
* Each student's chunk is ``scripts/ops/gpu_lock.sh scripts/ops/mem_cap.sh <GB> <PY> -m main.human_data.student
  train … --device cuda --budget-min 45``. **Exit 3 (the budget was reached) is the ONLY exit that continues a
  student**; exit 0 finishes it; ANY other exit (a refusal, an OOM kill at the cap, a lock refusal) stops the window
  and reports, with no new chunk started (:func:`next_action`). A student that is still at exit 3 after
  :data:`MAX_CHUNKS` chunks stops the window too.
* The CUDA pre-flight (``main.human_data.cuda_preflight``) must have PASSED (its JSON, ``--preflight``) before any
  chunk starts.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

SEEDS = (5901, 5902, 5903, 5904, 5905)
PRIMARY = ("human", "pstar", "selfplay")
DESCRIPTIVE = ("pstar0",)
ARM_DIR = {"human": "s_h", "pstar": "s_p", "selfplay": "s_s", "pstar0": "s_p0"}
MEM_GB = 10           # PREREG Amendment 3: the pre-flight's peak host RSS was 7.27 GiB, 0.73 under the old 8
BUDGET_MIN = 45
MAX_CHUNKS = 30
CONTINUE, DONE, STOP = "continue", "done", "stop"


def jobs() -> List[Tuple[str, int]]:
    """(arm, seed) in the registered order (module docstring)."""
    return [(a, s) for s in SEEDS for a in PRIMARY] + [(a, s) for a in DESCRIPTIVE for s in SEEDS]


def next_action(rc: int, chunks: int) -> str:
    """Exit 3 continues (until :data:`MAX_CHUNKS`), exit 0 is done, everything else stops the window."""
    if rc == 0:
        return DONE
    if rc == 3 and chunks < MAX_CHUNKS:
        return CONTINUE
    return STOP


def command(arm: str, seed: int, a: argparse.Namespace) -> List[str]:
    from utils.paths import repo_path

    rows = a.selfplay_rows if arm == "selfplay" else a.human_rows
    cmd = [str(repo_path("scripts", "ops", "gpu_lock.sh")), str(repo_path("scripts", "ops", "mem_cap.sh")),
           str(MEM_GB), a.python, "-m", "main.human_data.student", "train", "--arm", arm, "--rows", rows,
           "--seed", str(seed), "--epochs", "3", "--device", "cuda", "--budget-min", str(BUDGET_MIN),
           "--registration", a.registration, "--out", str(Path(a.dir) / f"{ARM_DIR[arm]}_{seed}")]
    if arm == "pstar":
        cmd[cmd.index("--seed"):cmd.index("--seed")] = ["--labels", a.labels]
    if arm == "pstar0":
        cmd[cmd.index("--seed"):cmd.index("--seed")] = ["--labels", a.labels0]
    return cmd


def run_window(todo: Sequence[Tuple[str, int]], start: Callable[[str, int], "subprocess.Popen"],
               concurrency: int, log: Callable[[str], None] = print, poll_s: float = 15.0) -> Dict[str, object]:
    """Up to ``concurrency`` students at once; each restarted only on exit 3 (:func:`next_action`). On a STOP no new
    chunk starts; the chunks already running finish theirs (their students resume later)."""
    queue = list(todo)
    running: Dict[Tuple[str, int], Tuple["subprocess.Popen", int]] = {}
    done: List[Tuple[str, int]] = []
    stopped: Optional[Dict[str, object]] = None
    while queue or running:
        while stopped is None and queue and len(running) < concurrency:
            job = queue.pop(0)
            running[job] = (start(*job), 1)
        for job, (proc, chunks) in list(running.items()):
            rc = proc.poll()
            if rc is None:
                continue
            act = next_action(rc, chunks)
            log(f"[window] {job[0]} {job[1]} chunk {chunks}: exit {rc} -> {act}")
            del running[job]
            if act == DONE:
                done.append(job)
            elif act == CONTINUE and stopped is None:
                running[job] = (start(*job), chunks + 1)
            elif act == STOP and stopped is None:
                stopped = {"job": list(job), "exit": rc, "chunks": chunks}
        if stopped is not None and not running:
            break
        if running:
            time.sleep(poll_s)
    return {"done": [list(j) for j in done], "stopped": stopped, "pending": [list(j) for j in queue],
            "pass": stopped is None and not queue}


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m main.human_data.window", description=__doc__.split("\n\n")[0])
    ap.add_argument("--dir", required=True)
    ap.add_argument("--registration", required=True)
    ap.add_argument("--preflight", required=True, help="the PASSED cuda_preflight JSON")
    ap.add_argument("--human-rows", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--labels0", required=True)
    ap.add_argument("--selfplay-rows", required=True)
    ap.add_argument("--concurrency", type=int, default=3)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--only-primary", action="store_true", help="the 15 primary students only")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    todo = [j for j in jobs() if not a.only_primary or j[0] in PRIMARY]
    if a.dry_run:
        for j in todo:
            print(" ".join(command(*j, a)))
        return 0
    pf = json.loads(Path(a.preflight).read_text())
    if not pf.get("pass"):
        print("[window] REFUSED: the CUDA pre-flight did not pass", file=sys.stderr, flush=True)
        return 1
    Path(a.dir).mkdir(parents=True, exist_ok=True)

    def start(arm: str, seed: int) -> "subprocess.Popen":
        lp = Path(a.dir) / f"{ARM_DIR[arm]}_{seed}.log"
        return subprocess.Popen(command(arm, seed, a), stdout=open(lp, "a"), stderr=subprocess.STDOUT)

    res = run_window(todo, start, a.concurrency)
    (Path(a.dir) / "window.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res, indent=1), flush=True)
    return 0 if res["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
