"""Join the phase markers, the per-thread CPU sampler, nvidia-smi and the run's TensorBoard into one read.

    python analyze_phases.py --phases phases.jsonl --cpu cpu.jsonl --smi gpu_smi.csv --tb <run>/tb \
        --skip-updates 2 --out phase_read.json

* CYCLE = from one `collect` start to the next. Within it: `collect` minus any `eval` nested in it = PLAY,
  `eval`, `update`, and the remainder (dump, callbacks, checkpoint/snapshot saves) = OTHER.
* CPU: every sampler row (2 s window) is assigned to the phase that covers >= 80 % of it; per phase the
  mean CPUs busy per thread GROUP (python main / torch & T2 threads / rust env core / other) and the
  windowed contention factor (rows with factor >= 1.05 are reported separately and excluded from the
  quiet read).
* GPU: nvidia-smi utilization.gpu (the % of time ANY kernel ran) per phase.
* TB: the collector's own per-host-step timers (rust_env/*_ms_per_host_step) and train/train_ms.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics as st
from collections import defaultdict
from datetime import datetime


def _load_phases(path):
    rows = [json.loads(l) for l in open(path)]
    return [r for r in rows if r.get("phase") in ("collect", "update", "eval", "eval_play", "dump")]


def _cycles(rows, skip_updates):
    col = sorted([r for r in rows if r["phase"] == "collect"], key=lambda r: r["t0"])
    upd = sorted([r for r in rows if r["phase"] == "update"], key=lambda r: r["t0"])
    ev = [r for r in rows if r["phase"] == "eval"]
    out = []
    for i in range(len(col) - 1):
        c, nxt = col[i], col[i + 1]
        u = [x for x in upd if c["t1"] <= x["t0"] < nxt["t0"]]
        e = [x for x in ev if c["t0"] <= x["t0"] < c["t1"]]
        cyc = nxt["t0"] - c["t0"]
        coll = c["t1"] - c["t0"]
        evs = sum(x["t1"] - x["t0"] for x in e)
        us = sum(x["t1"] - x["t0"] for x in u)
        out.append({"i": i, "t0": c["t0"], "t1": nxt["t0"], "cycle_s": cyc, "play_s": coll - evs, "eval_s": evs,
                    "update_s": us, "other_s": cyc - coll - us, "has_eval": bool(e),
                    "upd": [(x["t0"], x["t1"]) for x in u], "ev": [(x["t0"], x["t1"]) for x in e],
                    "col": (c["t0"], c["t1"])})
    return out[skip_updates:]


def _phase_at(cycles, t0, t1):
    """The phase covering >= 80 % of [t0, t1], or None."""
    span = t1 - t0
    for c in cycles:
        if t1 < c["t0"] or t0 > c["t1"]:
            continue
        cover = defaultdict(float)
        for a, b in c["ev"]:
            cover["eval"] += max(0.0, min(t1, b) - max(t0, a))
        for a, b in c["upd"]:
            cover["update"] += max(0.0, min(t1, b) - max(t0, a))
        a, b = c["col"]
        cover["play"] += max(0.0, min(t1, b) - max(t0, a)) - cover["eval"]
        best = max(cover.items(), key=lambda kv: kv[1]) if cover else None
        if best and best[1] >= 0.8 * span:
            return best[0], c["has_eval"]
    return None, False


def _group(name):
    proc, _, th = name.partition(":")
    if proc.startswith("rust_env_proc"):
        return "rust_env_core"
    if proc.startswith("sim_bridge"):
        return "sim_bridge"
    if proc.startswith("python"):
        if th.startswith("python") or th.startswith("pt_main_thread"):
            return "py_main"
        return "py_other_threads"
    return "other:" + proc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phases", required=True)
    ap.add_argument("--cpu", required=True)
    ap.add_argument("--smi", required=True)
    ap.add_argument("--tb", default=None)
    ap.add_argument("--skip-updates", type=int, default=2)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cyc = _cycles(_load_phases(a.phases), a.skip_updates)
    res = {"n_cycles": len(cyc), "cycles": [{k: v for k, v in c.items() if k not in ("upd", "ev", "col")}
                                            for c in cyc]}
    noev = [c for c in cyc if not c["has_eval"]]
    if noev:
        res["median_no_eval"] = {k: st.median(c[k] for c in noev) for k in ("cycle_s", "play_s", "update_s", "other_s")}
    evc = [c for c in cyc if c["has_eval"]]
    if evc:
        res["eval_cycles"] = [{k: c[k] for k in ("cycle_s", "play_s", "eval_s", "update_s", "other_s")} for c in evc]

    # CPU per phase
    acc = defaultdict(lambda: defaultdict(list))
    contam = defaultdict(int)
    names = defaultdict(lambda: defaultdict(list))
    for line in open(a.cpu):
        r = json.loads(line)
        if "t" not in r:
            continue
        ph, _ = _phase_at(cyc, r["t"] - r["dt"], r["t"])
        if ph is None:
            continue
        if r["factor"] >= 1.05:
            contam[ph] += 1
            continue
        g = defaultdict(float)
        for k, v in r["cpu_by_thread"].items():
            g[_group(k)] += v
            names[ph][k].append(v)
        for k in ("py_main", "py_other_threads", "rust_env_core", "sim_bridge"):
            acc[ph][k].append(g.get(k, 0.0))
        acc[ph]["total"].append(sum(g.values()))
        acc[ph]["factor"].append(r["factor"])
        acc[ph]["busy_cpus_box"].append(r["busy_cpus"])
    res["cpu_by_phase"] = {ph: {k: {"mean": st.mean(v), "n": len(v)} for k, v in d.items()} for ph, d in acc.items()}
    res["cpu_rows_excluded_contended"] = dict(contam)
    res["cpu_top_threads_by_phase"] = {
        ph: sorted(((k, st.mean(v) * len(v) / max(1, len(acc[ph]["total"]))) for k, v in d.items()),
                   key=lambda kv: -kv[1])[:12] for ph, d in names.items()}

    # nvidia-smi per phase
    smi = defaultdict(lambda: defaultdict(list))
    for row in csv.reader(open(a.smi)):
        try:
            t = datetime.strptime(row[0].strip(), "%Y/%m/%d %H:%M:%S.%f").timestamp()
            util, memu, sm_clk, pw = float(row[1].split()[0]), float(row[2].split()[0]), float(row[3].split()[0]), float(row[5].split()[0])
        except (ValueError, IndexError):
            continue
        ph, _ = _phase_at(cyc, t - 0.25, t + 0.25)
        if ph is None:
            continue
        smi[ph]["util_gpu"].append(util)
        smi[ph]["util_mem"].append(memu)
        smi[ph]["sm_clock_mhz"].append(sm_clk)
        smi[ph]["power_w"].append(pw)
    res["smi_by_phase"] = {ph: {k: {"mean": st.mean(v), "n": len(v)} for k, v in d.items()} for ph, d in smi.items()}

    if a.tb:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        ea = EventAccumulator(a.tb, size_guidance={"scalars": 0})
        ea.Reload()
        tb = {}
        for tag in ea.Tags()["scalars"]:
            if tag.startswith("rust_env/") or tag in ("train/train_ms", "time/fps") or "duration" in tag or "wall" in tag:
                vals = [e.value for e in ea.Scalars(tag)]
                tb[tag] = {"median_after_skip": st.median(vals[a.skip_updates:]) if len(vals) > a.skip_updates else None,
                           "all": vals}
        res["tb"] = tb
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps({k: res[k] for k in res if k not in ("cycles", "tb")}, indent=1)[:6000])


if __name__ == "__main__":
    main()
