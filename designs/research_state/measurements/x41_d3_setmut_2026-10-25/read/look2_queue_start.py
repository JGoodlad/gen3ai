"""D3 look-2 QUEUE STARTER v2 (detached, bounded; PREREG AM-D3-L2 addendum, the orchestrator's 2026-10-27 schedule).

Two lanes, two triggers:
* the S lane (one job) starts when the H-CTX rc72b read's anchor supervisor has exited (pid + start ticks);
* the Kakuna ANCHOR lane (``--concurrency 1``) starts when N1-P look 1's Kakuna lane has finished: its supervisor pid
  has exited AND either every planned anchor unit of that state is counted, or no anchor supervisor of that state has
  been alive for ANCHOR_QUIET_S (a lane stopped with units left and not resumed: logged as such).
Before each lane starts, the external state is snapshotted (REVIEW m5: ``snapshot_ext.py``, label ``start_<lane>``).
Overrides by file in the D3 state: ``QUEUE_HOLD`` waits; ``QUEUE_GO_S`` / ``QUEUE_GO_ANCHOR`` start that lane now.
Preconditions are checked first and before each start; a failure HOLDs (logs, exits 3). Ends at the horizon.

usage: python queue_start2.py <pin worktree> <d3 state> <hctx pid> <hctx ticks> <n1p state> <n1p pid> <n1p ticks> <horizon>
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

POLL_S = 60
ANCHOR_QUIET_S = 1800
PY = "/home/goodlad/miniconda3/envs/gen3ai_torch28/bin/python3"
PIN = "c4a5c37b6b3749b93157048169e9c2fb348c995a"
CORES = "5-7,13-15"
HERE = Path(__file__).resolve().parent
LANE_ARGS = {"anchor": ["--hours", "60", "--concurrency", "1"], "s": ["--hours", "30"]}


def log(d3: Path, **ev):
    with open(d3 / "queue_starter.jsonl", "a") as f:
        f.write(json.dumps({"t": round(time.time(), 1), **ev}) + "\n")


def env(wt: Path):
    e = dict(os.environ)
    e.update({"PYTHONPATH": str(wt / "src"), "CUDA_VISIBLE_DEVICES": "", "HF_HUB_OFFLINE": "1"})
    return e


def proc_gone(pid: int, ticks: str) -> bool:
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return True
    return stat.rsplit(")", 1)[1].split()[19] != ticks


def n1p_status(wt: Path, state: Path):
    r = subprocess.run(["nice", "-n", "19", "taskset", "-c", CORES, PY, "-m", "main.reads", "status", "--state",
                        str(state)], cwd=str(wt), env=env(wt), capture_output=True, text=True, timeout=600)
    if r.returncode != 0:
        raise RuntimeError(f"status rc {r.returncode}: {r.stderr.strip()[-300:]}")
    st = json.loads(r.stdout)
    left = sum(c["planned"] - c["counted"] for by in st["anchor"].values() for c in by.values())
    live = list((st.get("supervisors") or {}).get("anchor") or [])
    return left, live


def preconditions(wt: Path, d3: Path):
    why = []
    head = subprocess.run(["git", "-C", str(wt), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    if head != PIN:
        why.append(f"pin worktree HEAD {head[:12]} != {PIN[:12]}")
    dirty = subprocess.run(["git", "-C", str(wt), "status", "--porcelain", "--untracked-files=no"],
                           capture_output=True, text=True).stdout.strip()
    if dirty:
        why.append(f"pin worktree is dirty: {dirty[:200]}")
    code = d3 / "code.json"
    if not code.exists() or json.loads(code.read_text()).get("commit") != PIN:
        why.append(f"{code}: missing or not the pin")
    return why


def start(wt: Path, d3: Path, lane: str):
    subprocess.run([PY, str(HERE / "snapshot_ext.py"), str(d3), f"start_{lane}"], capture_output=True, timeout=900)
    argv = ["setsid", "nohup", "nice", "-n", "19", "taskset", "-c", CORES, PY, "-m", "main.reads", "run",
            "--state", str(d3), "--lane", lane, *LANE_ARGS[lane]]
    with open(d3 / f"{lane}.log", "ab") as fh:
        p = subprocess.Popen(argv, cwd=str(wt), env=env(wt), stdout=fh, stderr=subprocess.STDOUT,
                             stdin=subprocess.DEVNULL, start_new_session=True)
    return {"wrapper_pid": p.pid, "argv": argv}


def main(argv):
    wt, d3 = Path(argv[0]), Path(argv[1])
    h_pid, h_ticks, n1p, n_pid, n_ticks, horizon = int(argv[2]), argv[3], Path(argv[4]), int(argv[5]), argv[6], int(argv[7])
    log(d3, event="queue_start_v2", pid=os.getpid(), horizon=horizon, hctx=[h_pid, h_ticks], n1p=[str(n1p), n_pid, n_ticks])
    why = preconditions(wt, d3)
    if why:
        log(d3, event="HOLD", why=why)
        return 3
    started = {}
    quiet_since = None
    last = None
    while time.time() < horizon and len(started) < 2:
        info = {}
        hold = (d3 / "QUEUE_HOLD").exists()
        go = {"s": False, "anchor": False}
        if not hold:
            go["s"] = (d3 / "QUEUE_GO_S").exists() or proc_gone(h_pid, h_ticks)
            if (d3 / "QUEUE_GO_ANCHOR").exists():
                go["anchor"] = True
                info["anchor_why"] = "go file"
            elif proc_gone(n_pid, n_ticks) and "anchor" not in started:
                try:
                    left, live = n1p_status(wt, n1p)
                    info.update(n1p_left=left, n1p_live=live)
                    if live:
                        quiet_since = None
                    else:
                        quiet_since = quiet_since or time.time()
                    if not live and left == 0:
                        go["anchor"], info["anchor_why"] = True, "N1-P's anchor lane counted every unit"
                    elif not live and time.time() - quiet_since >= ANCHOR_QUIET_S:
                        go["anchor"] = True
                        info["anchor_why"] = f"no N1-P anchor supervisor for {ANCHOR_QUIET_S} s, {left} unit(s) left"
                except Exception as exc:                         # noqa: BLE001 - logged, retried next poll
                    info["status_error"] = str(exc)[:300]
        state = {"hold": hold, "go": go, **info}
        if state != last:
            log(d3, event="poll", **state)
            last = state
        for lane in ("s", "anchor"):
            if go[lane] and lane not in started:
                why = preconditions(wt, d3)
                if why:
                    log(d3, event="HOLD", why=why)
                    return 3
                started[lane] = start(wt, d3, lane)
                log(d3, event="started", lane=lane, **started[lane])
        if len(started) < 2:
            time.sleep(POLL_S)
    if len(started) < 2:
        log(d3, event="horizon", started=sorted(started), why="not every lane's trigger fired by the horizon")
        return 4
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
