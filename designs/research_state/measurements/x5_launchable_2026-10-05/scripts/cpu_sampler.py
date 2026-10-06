"""Per-thread CPU occupancy of a process TREE + the box's contention factor, every --interval s.

    python cpu_sampler.py --root-cmd-match phase_probe_run.py --out cpu.jsonl [--interval 2] [--max-hours 3]

Finds the root process by a substring of its cmdline (waits for it, bounded), then every interval writes
one JSON row: wall time; for every live process in the root's subtree, CPU-seconds per THREAD NAME (comm)
over the interval (utime + stime from /proc/<pid>/task/<tid>/stat, so a thread pool's busy share is the
sum); and the WINDOWED contention reading between the previous sample and this one
(`utils.cpu_meter.reading_between`, self_root = the root: the profiled tree is SELF, so the factor is what
OTHER work on the box cost it). Exits when the root dies or at --max-hours. Pure /proc; no ptrace.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict

from utils.cpu_meter import reading_between, take_sample, topology

TCK = os.sysconf("SC_CLK_TCK")


def _children_map():
    kids = defaultdict(list)
    for d in os.listdir("/proc"):
        if not d.isdigit():
            continue
        try:
            with open(f"/proc/{d}/stat") as f:
                s = f.read()
            ppid = int(s[s.rfind(")") + 2:].split()[1])
            kids[ppid].append(int(d))
        except (OSError, ValueError, IndexError):
            continue
    return kids


def _subtree(root):
    kids = _children_map()
    out, stack = [], [root]
    while stack:
        p = stack.pop()
        out.append(p)
        stack.extend(kids.get(p, []))
    return out


def _proc_name(pid):
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            parts = f.read().split(b"\0")
        exe = os.path.basename(parts[0].decode(errors="replace")) if parts and parts[0] else "?"
        return exe
    except OSError:
        return "?"


def _thread_cpu(pid):
    out = {}
    try:
        tids = os.listdir(f"/proc/{pid}/task")
    except OSError:
        return out
    for tid in tids:
        try:
            with open(f"/proc/{pid}/task/{tid}/stat") as f:
                s = f.read()
            comm = s[s.find("(") + 1:s.rfind(")")]
            rest = s[s.rfind(")") + 2:].split()
            out[int(tid)] = (comm, (int(rest[11]) + int(rest[12])) / TCK)
        except (OSError, ValueError, IndexError):
            continue
    return out


def _find_root(match, deadline):
    me = os.getpid()
    while time.time() < deadline:
        for d in os.listdir("/proc"):
            if not d.isdigit() or int(d) == me:
                continue
            try:
                with open(f"/proc/{d}/cmdline", "rb") as f:
                    cl = f.read().replace(b"\0", b" ").decode(errors="replace")
            except OSError:
                continue
            if match in cl and "cpu_sampler" not in cl and "nsys" not in cl.split()[0]:
                return int(d)
        time.sleep(1.0)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root-cmd-match", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--interval", type=float, default=2.0)
    ap.add_argument("--max-hours", type=float, default=3.0)
    ap.add_argument("--wait-root-s", type=float, default=600)
    a = ap.parse_args()
    root = _find_root(a.root_cmd_match, time.time() + a.wait_root_s)
    if root is None:
        raise SystemExit("root process never appeared")
    topo = topology()
    end = time.time() + a.max_hours * 3600
    prev_threads = {}
    prev_sample = take_sample([root], topo)
    with open(a.out, "a", buffering=1) as fh:
        fh.write(json.dumps({"root": root, "logical": len(topo.logical), "physical": topo.physical}) + "\n")
        while time.time() < end and os.path.exists(f"/proc/{root}"):
            time.sleep(a.interval)
            pids = _subtree(root)
            cur = {}
            for p in pids:
                for tid, (comm, cpu) in _thread_cpu(p).items():
                    cur[(p, tid)] = (comm, cpu)
            smp = take_sample([root], topo)
            rd = reading_between(prev_sample, smp, self_root=root, topo=topo)
            dt = smp.mono - prev_sample.mono
            by = defaultdict(float)
            for key, (comm, cpu) in cur.items():
                if key in prev_threads:
                    d = cpu - prev_threads[key][1]
                    if d > 0:
                        by[f"{_proc_name(key[0])}:{comm}"] += d
            fh.write(json.dumps({
                "t": smp.wall, "dt": dt, "cpu_by_thread": {k: round(v / dt, 4) for k, v in by.items()},
                "self_cpus": rd.self_cpus, "busy_cpus": rd.busy_cpus, "factor": rd.factor,
                "smt": rd.smt, "queue": rd.queue, "psi": rd.psi_some, "load1": smp.load1,
                "n_pids": len(pids)}) + "\n")
            prev_threads, prev_sample = cur, smp


if __name__ == "__main__":
    main()
