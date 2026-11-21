"""Operator's scheduler for V-RANK look 2's 4 branch shards (look 1's branch_driver.py with the plan root branches2 and the log name branch2_), 3 supervisors at a time (orchestrator's concurrency choice).
Scheduling only: each run is `main.x35.kakuna run --state sK --stop-at-epoch E` (a clean stop between units; resumable). Rerun to resume."""
import json, os, subprocess, sys, threading, time
from pathlib import Path
from main.x35 import kakuna as K

A = Path(sys.argv[1]); LOW = sys.argv[2].split(); PY = sys.argv[3]; WORKERS = int(sys.argv[4]); SLICE = int(sys.argv[5])
states = [A / "branches2" / f"s{k}" for k in range(4)]
lock = threading.Lock(); running = set(); finished = set()

def remaining(st):
    return sum(1 for u in K.load_plan(st) if not K.is_done(st, u))

def log(msg):
    print(time.strftime("[%F %T]"), msg, flush=True)

def pick():
    best, bn = None, 0
    for st in states:
        if st in running or st in finished:
            continue
        n = remaining(st)
        if n == 0:
            finished.add(st); continue
        if n > bn:
            best, bn = st, n
    return best, bn

def worker(i):
    while True:
        with lock:
            if os.path.exists(A / "STOP_DRIVER"):
                return
            st, n = pick()
            if st is None:
                if not running:
                    return
            else:
                running.add(st)
        if st is None:
            time.sleep(30); continue
        log(f"w{i} start {st.name} remaining {n}")
        rc = subprocess.call(LOW + [PY, "-m", "main.x35.kakuna", "run", "--state", str(st), "--stop-at-epoch", str(int(time.time()) + SLICE)],
                             stdout=open(A / f"branch2_{st.name}.log", "a"), stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        last = ""
        for l in open(st / "units.jsonl"):
            last = l
        with lock:
            running.discard(st)
            if '"supervisor_done"' in last or rc != 0:
                finished.add(st)
        log(f"w{i} end {st.name} rc={rc} remaining {remaining(st)} last_event={json.loads(last).get('event')}")

ts = [threading.Thread(target=worker, args=(i,)) for i in range(WORKERS)]
for t in ts: t.start()
for t in ts: t.join()
log("driver done: " + " ".join(f"{s.name}:{remaining(s)}" for s in states))
