"""Join the 0.5 s CPU sampler with the nvidia-smi stream; classify each interval by the rust_env process's CPU rate
(collect = the env core busy) and report GPU util + per-CPU-set utilization per phase."""
import sys
from datetime import datetime

cpu_f, gpu_f = sys.argv[1], sys.argv[2]
HZ = 100
TRAINER = [0, 1, 2, 3, 4, 8, 9, 10, 11, 12]
AGENT = [5, 6, 7, 13, 14, 15]
rows = []
for line in open(cpu_f):
    t, p, c = line.strip().split("|")
    pj = list(map(int, p.split()))
    cc = {}
    for x in c.split():
        k, b, i = x.split(":")
        cc[int(k[3:])] = (int(b), int(i))
    rows.append((float(t), pj, cc))
gpu = []
for line in open(gpu_f):
    try:
        ts, u = line.strip().split(", ")
        gpu.append((datetime.strptime(ts, "%Y/%m/%d %H:%M:%S.%f").timestamp(), float(u.split()[0])))
    except ValueError:
        pass
out = []
for (t0, p0, c0), (t1, p1, c1) in zip(rows, rows[1:]):
    dt = t1 - t0
    rates = [(b - a) / HZ / dt for a, b in zip(p0, p1)]  # cores used by py trainer, env proc A, env proc B

    def util(cs):
        busy = sum(c1[k][0] - c0[k][0] for k in cs)
        idle = sum(c1[k][1] - c0[k][1] for k in cs)
        return busy / max(1, busy + idle)
    g = [u for ts, u in gpu if t0 <= ts < t1]
    out.append((t0, rates, util(TRAINER), util(AGENT), sum(g) / len(g) if g else None))
for name, sel in (("collect (env proc > 1 core)", lambda r: r[1][1] > 1.0),
                  ("train (env proc < 0.3 core)", lambda r: r[1][1] < 0.3),
                  ("all", lambda r: True)):
    s = [r for r in out if sel(r)]
    gs = [r[4] for r in s if r[4] is not None]
    n = len(s)
    if not n:
        continue
    print(f"{name:30s} n={n:4d} ({n * 0.5:.0f}s)  py_trainer {sum(r[1][0] for r in s) / n:.2f} cores  "
          f"env {sum(r[1][1] for r in s) / n:.2f} cores  env2 {sum(r[1][2] for r in s) / n:.2f}  "
          f"trainer-CPUs util {sum(r[2] for r in s) / n:.2f} (of 10 logical)  agent-CPUs util {sum(r[3] for r in s) / n:.2f}"
          f"  GPU util {sum(gs) / max(1, len(gs)):.1f}%")
