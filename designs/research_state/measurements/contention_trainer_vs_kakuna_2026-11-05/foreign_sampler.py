"""X65-F1 foreign-load sampler (orchestrator 2026-11-05): every PERIOD s, the CPU-seconds spent ON THE AGENT CORES
(5-7, 13-15) by (a) the contention workload (cgroup scope gen3ai-capped-x65f1_*), (b) everything else ("foreign"), plus
the per-CPU busy fraction of those six CPUs from /proc/stat, and the top foreign processes. A process is attributed to
the agent cores by the CPU it last ran on (/proc/<pid>/stat field 39) at the end of the interval: an approximation,
honest at a 60 s grain. Writes <out>.tsv (one row per interval) and <out>.top.jsonl.

    python foreign_sampler.py <out prefix> <seconds total> [period]
"""
import json
import os
import sys
import time

AGENT = {5, 6, 7, 13, 14, 15}
HZ = os.sysconf("SC_CLK_TCK")


def procs():
    out = {}
    for d in os.listdir("/proc"):
        if not d.isdigit():
            continue
        try:
            with open(f"/proc/{d}/stat") as f:
                s = f.read()
            rest = s[s.rindex(")") + 2:].split()
            ticks = int(rest[11]) + int(rest[12])
            cpu = int(rest[36])
            comm = s[s.index("(") + 1:s.rindex(")")]
            with open(f"/proc/{d}/cgroup") as f:
                cg = f.read()
            out[int(d)] = (ticks, cpu, comm, "x65f1" in cg)
        except (OSError, ValueError, IndexError):
            continue
    return out


def cpustat():
    res = {}
    with open("/proc/stat") as f:
        for line in f:
            if line.startswith("cpu") and line[3].isdigit():
                p = line.split()
                n = int(p[0][3:])
                v = list(map(int, p[1:]))
                res[n] = (sum(v), v[3] + v[4])
    return res


def main():
    pre, total = sys.argv[1], int(sys.argv[2])
    period = int(sys.argv[3]) if len(sys.argv) > 3 else 60
    end = time.time() + total
    p0, c0, t0 = procs(), cpustat(), time.time()
    with open(pre + ".tsv", "a") as tsv, open(pre + ".top.jsonl", "a") as top:
        if tsv.tell() == 0:
            tsv.write("t\tdt\tmine_cpu_s\tforeign_cpu_s\tagent_busy_frac\tforeign_frac_of_6\n")
        while time.time() < end:
            time.sleep(period)
            p1, c1, t1 = procs(), cpustat(), time.time()
            dt = t1 - t0
            mine = foreign = 0.0
            fl = []
            for pid, (tk, cpu, comm, isme) in p1.items():
                d = (tk - p0[pid][0]) / HZ if pid in p0 else 0.0
                if d <= 0 or cpu not in AGENT:
                    continue
                if isme:
                    mine += d
                else:
                    foreign += d
                    fl.append((round(d, 2), pid, comm))
            busy = []
            for n in AGENT:
                tot = c1[n][0] - c0[n][0]
                idle = c1[n][1] - c0[n][1]
                busy.append((tot - idle) / tot if tot else 0.0)
            tsv.write(f"{int(t1)}\t{dt:.1f}\t{mine:.1f}\t{foreign:.1f}\t{sum(busy)/len(busy):.3f}\t{foreign/(6*dt):.3f}\n")
            tsv.flush()
            fl.sort(reverse=True)
            top.write(json.dumps({"t": int(t1), "top_foreign": fl[:6]}) + "\n")
            top.flush()
            p0, c0, t0 = p1, c1, t1


if __name__ == "__main__":
    main()
