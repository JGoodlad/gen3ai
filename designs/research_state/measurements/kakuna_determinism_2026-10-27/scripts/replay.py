"""Replay one N1-P look-1 Kakuna unit from the E3 pin worktree, under a claimed 95xx port. Read-only on archives."""
import json, os, subprocess, sys, time
PIN = "/home/goodlad/dev/gen3ai-reads/n1p_play_sched_de15dfc7"
sys.path.insert(0, PIN + "/src")
from main.x35 import kakuna as KA  # noqa: E402
units_jsonl, unit, out = sys.argv[1], sys.argv[2], sys.argv[3]
argv = None
for line in open(units_jsonl):
    d = json.loads(line)
    if d.get("event") == "start" and d.get("unit") == unit:
        argv = d["argv"]
assert argv, unit
port, lock = KA.claim_port()
a = list(argv)
a[a.index("--port") + 1] = str(port)
a[a.index("--out") + 1] = out
if "--capture-dir" in a:
    a[a.index("--capture-dir") + 1] = out + "/capture"
teamset = unit.split("__")[1]
env = KA.unit_env(config=KA.config_of(teamset))
assert env["PYTHONPATH"] == PIN + "/src", env["PYTHONPATH"]
cmd = ["chrt", "-i", "0"] + a
print(json.dumps({"t": time.time(), "port": port, "cmd": cmd}), flush=True)
rc = subprocess.call(cmd, env=env, cwd=PIN)
print(json.dumps({"t": time.time(), "rc": rc}), flush=True)
lock.close()
sys.exit(rc)
