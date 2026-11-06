"""S'-HONEST design input 4 (2026-11-05, OUTCOME-BLIND): a short TIMING RUN of today's ``r0s4_rolls`` world builder
(``WorldSampler(reveal="off", max_tries=60, dice="rolls")``, 8 worlds) on S'-LIVE's kept honest logs: CPU (this
process and its reaped ``core_events`` children), screening replays and tries per decision, beside the live record's own
``world_cpu_s``. One torch-free process, one thread.

    PYTHONPATH=src OMP_NUM_THREADS=1 python3 world_cost.py <S'-LIVE state>/anchor/u/pstar_s/cells 16 world_cost.json
"""
import glob
import json
import resource
import sys
import time

import numpy as np

from main.value_search.worlds import WorldError, WorldSampler


def cpu():
    a, c = resource.getrusage(resource.RUSAGE_SELF), resource.getrusage(resource.RUSAGE_CHILDREN)
    return a.ru_utime + a.ru_stime + c.ru_utime + c.ru_stime


def main(cells, nmax, out):
    rng = np.random.default_rng(20261105)
    files = sorted(f for f in glob.glob(cells + "/*/search_decisions.jsonl") if ".partial." not in f)
    picked = []
    for f in rng.permutation(files):
        for line in open(f):
            if '"rescore_sample": true' in line:
                picked.append(json.loads(line))
                break
        if len(picked) >= nmax:
            break
    sampler = WorldSampler(reveal="off", max_tries=60, dice="rolls")
    rows = []
    for d in picked:
        c0, w0 = cpu(), time.monotonic()
        try:
            w = sampler.sample(d["log"], d["at"], d["side"], 8, d["key"], root_choice_in_log=False)
            rec = {"ok": True, "screens": w.screens, "tries": w.tries}
        except WorldError as e:
            rec = {"ok": False, "error": str(e)[:200]}
        rec.update(key=d["key"], turn=d["turn"], at=d["at"], cpu_s=round(cpu() - c0, 2),
                   wall_s=round(time.monotonic() - w0, 2), live_world_cpu_s=d.get("world_cpu_s"),
                   live_searched=d.get("searched"))
        rows.append(rec)
        print(json.dumps(rec), flush=True)
    ok = [r for r in rows if r["ok"]]
    summary = {"decisions": len(rows), "built": len(ok), "screens_median": float(np.median([r["screens"] for r in ok])),
               "screens_range": [min(r["screens"] for r in ok), max(r["screens"] for r in ok)],
               "cpu_s_median_built": float(np.median([r["cpu_s"] for r in ok])),
               "cpu_ms_per_screen_median": round(float(np.median([1000 * r["cpu_s"] / r["screens"] for r in ok])), 2)}
    json.dump({"summary": summary, "decisions": rows}, open(out, "w"), indent=1, sort_keys=True)
    print(json.dumps(summary, indent=1, sort_keys=True))


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), sys.argv[3])
