"""Per-PHASE GPU read of an nsys SQLite export (bottleneck profile, 2026-10-03).

    python nsys_phase_read.py --db win1.sqlite --out nsys_read.json

Phases are the NVTX ranges `phase_probe_run.py` pushes (`update`, `collect`, `eval`, and inside collect
`c_prepare` / `c_core` / `c_post` / `t2_flush`). For each top-level range (and its sub-ranges, aggregated):

* kernels: count, summed duration, GPU BUSY = the union of kernel intervals over all streams / range wall;
* the host: CUDA runtime/driver API calls by name on the range's thread (count + time), so launch rate,
  syncs and their blocking time are visible;
* memcpy: count and bytes by kind (H2D / D2H / D2D);
* gaps between consecutive kernels on the busiest stream (launch-latency signature: p50 / p90 of the idle
  gap, and the share of the stream's span that is idle);
* the top kernels by summed time (name, count, total, mean), with a coarse CLASS
  (gemm / attention / triton-fused / optimizer / reduction / elementwise / copy / other).
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from collections import defaultdict

import numpy as np


def klass(name: str) -> str:
    n = name.lower()
    if "gemm" in n or "cutlass" in n or "sgemm" in n or "gemv" in n or "matmul" in n or "xmma" in n:
        return "gemm"
    if "attention" in n or "fmha" in n or "flash" in n or "sdpa" in n:
        return "attention"
    if n.startswith("triton_per") or n.startswith("triton_red"):
        return "triton_reduction"
    if n.startswith("triton_poi"):
        return "triton_pointwise"
    if n.startswith("triton"):
        return "triton_other"
    if "multi_tensor" in n or "adam" in n or "foreach" in n:
        return "optimizer"
    if "reduce" in n or "softmax" in n or "norm" in n or "scan" in n or "sort" in n:
        return "reduction"
    if "elementwise" in n or "vectorized" in n or "unrolled" in n:
        return "elementwise"
    if "copy" in n or "memcpy" in n or "cat" in n or "index" in n or "gather" in n or "scatter" in n:
        return "copy_index"
    return "other"


def union_len(starts, ends):
    if len(starts) == 0:
        return 0
    o = np.argsort(starts)
    s, e = starts[o], ends[o]
    tot, cs, ce = 0, s[0], e[0]
    # vectorised merge
    run_max = np.maximum.accumulate(e)
    breaks = np.nonzero(s[1:] > run_max[:-1])[0]
    seg_start = np.concatenate(([0], breaks + 1))
    seg_end = np.concatenate((breaks, [len(s) - 1]))
    tot = int(np.sum(run_max[seg_end] - s[seg_start]))
    del cs, ce
    return tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--top", type=int, default=25)
    a = ap.parse_args()
    c = sqlite3.connect(a.db)
    strings = dict(c.execute("select id, value from StringIds"))
    nv = c.execute("select start, end, text, textId, globalTid from NVTX_EVENTS where end is not null").fetchall()
    ranges = defaultdict(list)
    for s, e, t, tid_, gt in nv:
        name = t if t is not None else strings.get(tid_)
        ranges[name].append((s, e, gt))
    K = np.array(c.execute("select start, end, streamId, shortName, coalesce(graphId, 0) from CUPTI_ACTIVITY_KIND_KERNEL").fetchall(),
                 dtype=np.int64)
    order = np.argsort(K[:, 0])
    K = K[order]
    R = np.array(c.execute("select start, end, nameId, globalTid from CUPTI_ACTIVITY_KIND_RUNTIME").fetchall(), dtype=np.int64)
    R = R[np.argsort(R[:, 0])]
    M = np.array(c.execute("select start, end, bytes, copyKind from CUPTI_ACTIVITY_KIND_MEMCPY").fetchall(), dtype=np.int64)
    M = M[np.argsort(M[:, 0])] if len(M) else np.zeros((0, 4), dtype=np.int64)
    copy_kind = {1: "H2D", 2: "D2H", 8: "D2D", 10: "P2P"}
    out = {"n_kernels_total": int(len(K)), "phases": {}}

    def window_stats(wins, label):
        if not wins:
            return None
        tot_wall = sum(e - s for s, e, _ in wins)
        kc = 0
        kt = 0
        busy = 0
        per_name = defaultdict(lambda: [0, 0])
        api = defaultdict(lambda: [0, 0])
        mem = defaultdict(lambda: [0, 0, 0])
        gaps_all = []
        stream_idle = []
        graph_k = 0
        for s, e, gt in wins:
            i0, i1 = np.searchsorted(K[:, 0], [s, e])
            k = K[i0:i1]
            k = k[k[:, 1] <= e + 5_000_000]
            kc += len(k)
            kt += int(np.sum(k[:, 1] - k[:, 0]))
            graph_k += int(np.sum(k[:, 4] != 0))
            busy += union_len(k[:, 0], np.minimum(k[:, 1], e))
            for nm, d in zip(k[:, 3], k[:, 1] - k[:, 0]):
                pn = per_name[int(nm)]
                pn[0] += 1
                pn[1] += int(d)
            if len(k):
                sid, cnt = np.unique(k[:, 2], return_counts=True)
                top = sid[np.argmax(cnt)]
                ks = k[k[:, 2] == top]
                if len(ks) > 1:
                    g = ks[1:, 0] - ks[:-1, 1]
                    g = g[g > 0]
                    gaps_all.append(g)
                    span = ks[-1, 1] - ks[0, 0]
                    stream_idle.append((int(np.sum(g)), int(span)))
            j0, j1 = np.searchsorted(R[:, 0], [s, e])
            r = R[j0:j1]
            r = r[r[:, 3] == gt] if gt else r
            for nm, d in zip(r[:, 2], r[:, 1] - r[:, 0]):
                ap_ = api[strings.get(int(nm), str(nm))]
                ap_[0] += 1
                ap_[1] += int(d)
            m0, m1 = np.searchsorted(M[:, 0], [s, e])
            for st_, en_, b, ck in M[m0:m1]:
                mm = mem[copy_kind.get(int(ck), str(int(ck)))]
                mm[0] += 1
                mm[1] += int(b)
                mm[2] += int(en_ - st_)
        g = np.concatenate(gaps_all) if gaps_all else np.zeros(0)
        top = sorted(per_name.items(), key=lambda kv: -kv[1][1])[:a.top]
        by_class = defaultdict(lambda: [0, 0])
        for nm, (cnt, d) in per_name.items():
            cl = klass(strings.get(nm, ""))
            by_class[cl][0] += cnt
            by_class[cl][1] += d
        idle = sum(x for x, _ in stream_idle)
        span = sum(y for _, y in stream_idle)
        return {
            "n_windows": len(wins), "wall_s": tot_wall / 1e9, "kernels": kc, "kernels_in_graphs": graph_k,
            "kernel_sum_s": kt / 1e9, "gpu_busy_union_s": busy / 1e9, "gpu_busy_frac": busy / max(1, tot_wall),
            "kernels_per_s_of_wall": kc / max(1e-9, tot_wall / 1e9),
            "mean_kernel_us": kt / max(1, kc) / 1e3,
            "busiest_stream_gap_us": {"n": int(len(g)), "p50": float(np.percentile(g, 50) / 1e3) if len(g) else None,
                                      "p90": float(np.percentile(g, 90) / 1e3) if len(g) else None,
                                      "p99": float(np.percentile(g, 99) / 1e3) if len(g) else None,
                                      "idle_frac_of_stream_span": idle / max(1, span)},
            "api_top": sorted(([k, v[0], v[1] / 1e9] for k, v in api.items()), key=lambda x: -x[2])[:15],
            "memcpy": {k: {"n": v[0], "MB": v[1] / 1e6, "s": v[2] / 1e9} for k, v in mem.items()},
            "by_class": {k: {"n": v[0], "s": v[1] / 1e9, "share_of_kernel_time": v[1] / max(1, kt)}
                         for k, v in sorted(by_class.items(), key=lambda kv: -kv[1][1])},
            "top_kernels": [{"name": strings.get(nm, str(nm))[:160], "class": klass(strings.get(nm, "")),
                             "n": cnt, "total_s": d / 1e9, "mean_us": d / cnt / 1e3,
                             "share_of_kernel_time": d / max(1, kt)} for nm, (cnt, d) in top],
        }

    def trim(name):
        return [(s, e, gt) for s, e, gt in ranges.get(name, [])]

    upd = trim("update")
    col = trim("collect")
    ev = trim("eval")
    # play = collect minus nested eval, approximated by c_prepare+c_core+c_post sub-ranges
    out["phases"]["update"] = window_stats(upd, "update")
    out["phases"]["eval"] = window_stats(ev, "eval")
    play = []
    for s, e, gt in col:
        inner = [(a_, b_) for a_, b_, _ in ev if s <= a_ <= e]
        cur = s
        for a_, b_ in sorted(inner):
            play.append((cur, a_, gt))
            cur = b_
        play.append((cur, e, gt))
    out["phases"]["play"] = window_stats(play, "play")
    for sub in ("c_prepare", "c_core", "c_post", "t2_flush"):
        w = trim(sub)
        if w:
            out["phases"][sub] = {"n": len(w), "total_s": sum(e - s for s, e, _ in w) / 1e9,
                                  "mean_ms": sum(e - s for s, e, _ in w) / len(w) / 1e6}
    out["phases"]["c_prepare_gpu"] = window_stats(trim("c_prepare"), "c_prepare")
    json.dump(out, open(a.out, "w"), indent=1)
    for k, v in out["phases"].items():
        if v and "wall_s" in v:
            print(k, {kk: v[kk] for kk in ("n_windows", "wall_s", "kernels", "kernel_sum_s", "gpu_busy_frac",
                                             "kernels_per_s_of_wall", "mean_kernel_us")})
        else:
            print(k, v)


if __name__ == "__main__":
    main()
