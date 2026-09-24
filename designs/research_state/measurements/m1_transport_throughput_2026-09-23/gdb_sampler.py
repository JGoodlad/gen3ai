"""gdb stack sampler — the profiler of last resort (`perf_event_paranoid=4`, no valgrind).

gdb runs `sim_bridge` as its OWN child (legal at `ptrace_scope=1`), an EXTERNAL ticker process
SIGINTs the inferior every ~INTERVAL seconds (a Python thread inside gdb never runs while
`continue` holds the GIL), and each stop records the full backtrace (function names only,
inlined frames included — build with `CARGO_PROFILE_RELEASE_DEBUG=true`). One JSON list of frame
names per sample, innermost first, goes to $M1_SAMPLES_OUT.

    M1_SAMPLES_OUT=/tmp/x.jsonl M1_STDIN=/tmp/long.txt \
      gdb -q -batch -x gdb_sampler.py --args /tmp/m1bench_prof_B/release/sim_bridge

Run under `nice -n 10`. Sampling perturbs wall time; only the frame SHARES are read from it.
"""

import json
import os
import subprocess

import gdb  # type: ignore  # noqa: F401  (only importable inside gdb)

INTERVAL = float(os.environ.get("M1_INTERVAL", "0.01"))
OUT = os.environ["M1_SAMPLES_OUT"]
STDIN = os.environ["M1_STDIN"]

gdb.execute("set pagination off")
gdb.execute("set confirm off")
gdb.execute("handle SIGINT stop print nopass")  # NOT noprint: noprint implies nostop
gdb.execute("handle SIGPIPE nostop noprint pass")


def frames():
    names = []
    f = gdb.newest_frame()
    while f is not None:
        names.append(f.name() or "??")
        f = f.older()
    return names


gdb.execute("starti < %s > /dev/null" % STDIN)
pid = gdb.selected_inferior().pid
ticker = subprocess.Popen(["bash", "-c", f"while kill -INT {pid} 2>/dev/null; do sleep {INTERVAL}; done"])
n = 0
with open(OUT, "w") as out:
    while True:
        try:
            gdb.execute("continue", to_string=True)
        except gdb.error:
            break
        inf = gdb.selected_inferior()
        if inf.pid == 0 or not inf.threads():
            break
        try:
            out.write(json.dumps(frames()) + "\n")
            n += 1
        except gdb.error:
            break
ticker.kill()
print(f"samples: {n}")
