"""Read-only 0.5 s sampler of per-CPU busy jiffies and the trainer's process jiffies; GPU util comes from a
separate `nvidia-smi -lms` stream. Usage: sampler.py OUT SECONDS PID [PID ...]"""
import sys
import time

out, secs, pids = sys.argv[1], float(sys.argv[2]), sys.argv[3:]


def cpus():
    r = {}
    with open("/proc/stat") as f:
        for line in f:
            if line.startswith("cpu") and line[3].isdigit():
                x = line.split()
                v = list(map(int, x[1:]))
                r[x[0]] = (v[0] + v[1] + v[2] + v[5] + v[6], v[3] + v[4])
    return r


def pj(pid):
    try:
        with open(f"/proc/{pid}/stat") as f:
            x = f.read().rsplit(")", 1)[1].split()
        return int(x[11]) + int(x[12])
    except OSError:
        return -1


t_end = time.time() + secs
with open(out, "w") as fo:
    while time.time() < t_end:
        c = cpus()
        fo.write(f"{time.time():.3f}|{' '.join(str(pj(p)) for p in pids)}|"
                 + " ".join(f"{k}:{b}:{i}" for k, (b, i) in sorted(c.items(), key=lambda kv: int(kv[0][3:]))) + "\n")
        fo.flush()
        time.sleep(0.5)
