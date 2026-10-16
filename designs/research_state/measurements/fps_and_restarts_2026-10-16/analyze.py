"""T24: throughput over a run, and what a launcher restart buys or costs (read-only).

    python designs/research_state/measurements/fps_and_restarts_2026-10-16/extract.py <models> <cache.json>
    python designs/research_state/measurements/fps_and_restarts_2026-10-16/analyze.py <models> <cache.json> <out.json>

INPUTS: ``extract.py``'s per-process TensorBoard series (one event file == one child process) and
each run's ``launcher_child.full.log`` (the startup lines of every child: ``reset + prewarm took``,
the DiskGuard compile-cache line, how the child ended).

THE CYCLE. A rollout's ``rollout/collect_ms`` is logged at the step the rollout ends; the update
that follows is logged (``train/train_ms``) with the NEXT rollout. So between two consecutive
``collect_ms`` events the wall is ``collect_ms[i] + train_ms[i-1] + other``, where ``other`` is
everything that is neither (checkpoint saves, logging). THE IN-LOOP EVAL RUNS INSIDE THE COLLECT
WINDOW (measured: the eval cycle's ``collect_ms`` is ~20-40 s above its neighbours and ``other`` stays
~0), so a cycle is an EVAL cycle when an ``eval/wall_sec`` event's step lies in its step range, and its
eval wall is that scalar; steady statistics exclude eval cycles. The throughput of
a cycle is ``(step[i] - step[i-1]) / (wall[i] - wall[i-1])``. ``time/fps`` (SB3's) is CUMULATIVE
since the process started and is not used.

A PROCESS'S STARTUP COST, for a child that follows another in the same run: the dead wall from
the previous child's last rollout event to the start of this child's first rollout
(``wall[0] - collect_ms[0]``), plus the first cycles' excess over the steady cycle. Gaps longer
than MAX_LAUNCHER_GAP_S are a HUMAN gap (a second leg launched later), so only the in-log startup
lines (prewarm) are used for them.

BEFORE / AFTER a restart: the steady cycles in the hour before the previous child's end vs the
hour after this child's first WARMUP_CYCLES cycles, eval cycles excluded,
paired per restart. Each window also records how many OTHER runs were writing events in it
(GPU / CPU co-tenancy), because a box shared with another run is a different machine.
"""
from __future__ import annotations

import datetime as dt
import json
import re
import statistics as st
import sys
from pathlib import Path

WARMUP_CYCLES = 3
WINDOW_S = 3600.0
MAX_LAUNCHER_GAP_S = 1800.0  # a gap longer than this between children is a human relaunch
LEARN_TAGS = ("train/approx_kl", "train/entropy_loss", "train/clip_fraction", "train/value_loss",
              "behaviour/excluded_frac", "rollout/ep_len_mean")

_ATTACH = re.compile(r"===== child attached (\S+ \S+) \(pid (\d+)\)")
_PREWARM = re.compile(r"reset \+ prewarm took ([0-9.]+)s")
_DRY = re.compile(r"dry first update \([^)]*?([0-9.]+)s\)")
_CACHE = re.compile(r"\[DiskGuard\]\s+compile cache\s+.*?- ([0-9.]+) (GiB|MiB) already on disk")


def _med(xs):
    xs = [x for x in xs if x is not None]
    return st.median(xs) if xs else None


def _r(x, n=1):
    return None if x is None else round(x, n)


def log_segments(log: Path) -> list[dict]:
    """One dict per child that reached its TB writer (``Logging to``), in order."""
    if not log.exists():
        return []
    segs, cur = [], None
    for line in log.read_text(errors="replace").splitlines():
        if "[DesktopGpu] desktop GPU" in line:
            if cur is not None:
                segs.append(cur)
            cur = {"prewarm_s": None, "dry_update_s": None, "cache_on_disk_gib": None,
                   "logged": False, "end": "running/unknown"}
            continue
        if cur is None:
            continue
        if (m := _PREWARM.search(line)):
            cur["prewarm_s"] = float(m.group(1))
        elif (m := _DRY.search(line)):
            cur["dry_update_s"] = float(m.group(1))
        elif (m := _CACHE.search(line)):
            v = float(m.group(1))
            cur["cache_on_disk_gib"] = v if m.group(2) == "GiB" else v / 1024
        elif line.startswith("Logging to "):
            cur["logged"] = True
        elif "Training complete." in line:
            cur["end"] = "complete"
        elif "[ABORT] Checkpoint saved" in line:
            cur["end"] = "sigterm_saved"
        elif ("TRAINING CRASHED" in line or "FATAL ERROR DETECTED" in line
              or "[CompileCanary] FATAL" in line) and cur["end"] == "running/unknown":
            cur["end"] = "crash"
    if cur is not None:
        segs.append(cur)
    return [s for s in segs if s["logged"]]


def cycles(proc: dict) -> list[dict]:
    s = proc["series"]
    cm, tm = s.get("rollout/collect_ms", []), s.get("train/train_ms", [])
    evals = s.get("eval/wall_sec", [])
    out = []
    for i in range(1, len(cm)):
        (s0, w0, _), (s1, w1, c1) = cm[i - 1], cm[i]
        t = tm[i - 1][2] if i - 1 < len(tm) else None
        wall = w1 - w0
        other = None if t is None else wall - c1 / 1000 - t / 1000
        ev = [e[2] for e in evals if s0 < e[0] <= s1]
        out.append({"step": s1, "wall": w1, "dt": wall, "ds": s1 - s0, "fps": (s1 - s0) / wall,
                    "collect_s": c1 / 1000, "train_s": None if t is None else t / 1000,
                    "other_s": other, "eval": bool(ev), "eval_s": sum(ev)})
    return out


def _series_at(proc, tag):
    return proc["series"].get(tag, [])


def window(cyc, lo, hi):
    return [c for c in cyc if lo <= c["wall"] <= hi]


def summarize(cs: list[dict], proc: dict | None = None) -> dict:
    steady = [c for c in cs if not c["eval"]]
    out = {"n_cycles": len(cs), "n_steady": len(steady)}
    if not cs:
        return out
    wall = sum(c["dt"] for c in cs)
    out["fps_all"] = _r(sum(c["ds"] for c in cs) / wall, 0)
    if steady:
        out["fps_steady_median"] = _r(_med([c["fps"] for c in steady]), 0)
        out["collect_s_per_100k_median"] = _r(_med([c["collect_s"] / c["ds"] * 1e5 for c in steady]), 2)
        out["train_s_median"] = _r(_med([c["train_s"] for c in steady]), 2)
        out["other_s_median"] = _r(_med([c["other_s"] for c in steady]), 2)
    out["eval_share_of_wall"] = _r(sum(c["eval_s"] for c in cs) / wall, 4)
    out["collect_share_of_wall"] = _r(sum(c["collect_s"] for c in cs) / wall, 4)
    out["train_share_of_wall"] = _r(sum(c["train_s"] or 0 for c in cs) / wall, 4)
    if proc is not None:
        lo, hi = cs[0]["wall"] - cs[0]["dt"], cs[-1]["wall"]
        for tag, key in (("rollout/ep_len_mean", "ep_len_mean"),
                         ("rust_env/core_ms_per_host_step", "core_ms_per_host_step"),
                         ("rust_env/gpu_wait_ms_per_host_step", "gpu_wait_ms_per_host_step"),
                         ("rust_env/trainee_rows_per_host_step", "trainee_rows_per_host_step"),
                         ("rust_env/opp_draw_ms_per_host_step", "opp_draw_ms_per_host_step"),
                         ("train/selfplay_fraction", "selfplay_fraction"),
                         ("compile/recompiles_after_lock", "recompiles_after_lock"),
                         ("lifecycle/cuda_reserved_mib", "cuda_reserved_mib")):
            v = [x[2] for x in _series_at(proc, tag) if lo <= x[1] <= hi]
            out[key] = _r(_med(v), 3)
    return out


def activity_spans(cache: dict) -> dict[str, list[tuple[float, float]]]:
    spans = {}
    for run, r in cache.items():
        spans[run] = []
        for p in r["procs"]:
            cm = p["series"].get("rollout/collect_ms", [])
            if cm and "inherited" not in p["file"]:
                spans[run].append((cm[0][1] - cm[0][2] / 1000, cm[-1][1]))
    return spans


def cotenants(spans, run, lo, hi) -> list[str]:
    return sorted(o for o, ss in spans.items() if o != run
                  and any(a < hi and b > lo for a, b in ss))


def hourly(cyc: list[dict], proc: dict, t0: float) -> list[dict]:
    rows = []
    if not cyc:
        return rows
    h = 0
    while t0 + h * 3600 < cyc[-1]["wall"]:
        cs = window(cyc, t0 + h * 3600, t0 + (h + 1) * 3600)
        if cs:
            rows.append({"hour": h, **summarize(cs, proc)})
        h += 1
    return rows


def learning_jump(prev: dict, nxt: dict, k: int = 5) -> dict:
    """The jump across the seam vs the within-process update-to-update spread, per tag."""
    out = {}
    for tag in LEARN_TAGS:
        a, b = [x[2] for x in _series_at(prev, tag)], [x[2] for x in _series_at(nxt, tag)]
        if len(a) < 2 * k or len(b) < k:
            continue
        diffs = [abs(a[i] - a[i - 1]) for i in range(1, len(a))]
        spread = _med(diffs)
        jump = abs(st.mean(b[:k]) - st.mean(a[-k:]))
        base = st.pstdev(a[-4 * k:])
        out[tag] = {"before_mean": _r(st.mean(a[-k:]), 5), "after_mean": _r(st.mean(b[:k]), 5),
                    "jump": _r(jump, 5), "median_abs_step_diff": _r(spread, 5),
                    "before_sd_last20": _r(base, 5),
                    "jump_over_sd": _r(jump / base, 2) if base else None}
    return out


def analyze_run(run: str, r: dict, models: Path, spans) -> dict:
    procs = [p for p in r["procs"] if "inherited" not in p["file"]
             and p["series"].get("rollout/collect_ms")]
    segs = log_segments(models / run / "launcher_child.full.log")
    if len(segs) < len([p for p in r["procs"] if "inherited" not in p["file"]]):
        segs = segs + [{}] * 10
    own = [p for p in r["procs"] if "inherited" not in p["file"]]
    seg_by_file = {p["file"]: (segs[i] if i < len(segs) else {}) for i, p in enumerate(own)}
    out = {"run": run, "procs": [], "seams": []}
    prev = None
    for p in procs:
        cyc = cycles(p)
        cm = p["series"]["rollout/collect_ms"]
        t_first_rollout_start = cm[0][1] - cm[0][2] / 1000
        rec = {"file": p["file"], "log": seg_by_file.get(p["file"], {}),
               "start_step": cm[0][0], "end_step": cm[-1][0],
               "hours": _r((cm[-1][1] - t_first_rollout_start) / 3600, 2),
               "whole": summarize(cyc, p),
               "first_cycles": [{k: _r(c[k], 1) for k in ("dt", "collect_s", "train_s", "other_s")}
                                for c in cyc[:WARMUP_CYCLES + 2]],
               "hourly": hourly(cyc, p, t_first_rollout_start),
               "cotenants_whole": cotenants(spans, run, t_first_rollout_start, cm[-1][1])}
        if prev is not None:
            pcyc, pcm = cycles(prev), prev["series"]["rollout/collect_ms"]
            gap = t_first_rollout_start - pcm[-1][1]
            steady_after = cyc[WARMUP_CYCLES:]
            before = window(pcyc, pcm[-1][1] - WINDOW_S, pcm[-1][1])
            after = window(steady_after, steady_after[0]["wall"] - steady_after[0]["dt"],
                           steady_after[0]["wall"] + WINDOW_S) if steady_after else []
            ref = _med([c["dt"] for c in after if not c["eval"]])
            excess = (sum(c["dt"] for c in cyc[:WARMUP_CYCLES] if not c["eval"])
                      - ref * len([c for c in cyc[:WARMUP_CYCLES] if not c["eval"]])) if ref else None
            seam = {"from": prev["file"], "to": p["file"],
                    "prev_end": seg_by_file.get(prev["file"], {}).get("end"),
                    "gap_s": _r(gap, 1), "human_gap": gap > MAX_LAUNCHER_GAP_S,
                    "steps_prev_end": pcm[-1][0], "steps_resume_first": cm[0][0],
                    "prewarm_s": rec["log"].get("prewarm_s"),
                    "dry_update_s": rec["log"].get("dry_update_s"),
                    "cache_on_disk_gib": rec["log"].get("cache_on_disk_gib"),
                    "first_cycles_excess_s": _r(excess, 1), "steady_cycle_s": _r(ref, 1),
                    "before_1h": summarize(before, prev), "after_1h": summarize(after, p),
                    "cotenants_before": cotenants(spans, run, pcm[-1][1] - WINDOW_S, pcm[-1][1]),
                    "cotenants_after": cotenants(spans, run, after[0]["wall"] - after[0]["dt"],
                                                 after[-1]["wall"]) if after else None,
                    "learning": learning_jump(prev, p)}
            out["seams"].append(seam)
        out["procs"].append(rec)
        prev = p
    return out


def main(argv: list[str]) -> int:
    models, cache_p, out_p = Path(argv[0]), Path(argv[1]), Path(argv[2])
    cache = json.loads(cache_p.read_text())
    spans = activity_spans(cache)
    res = {"generated": dt.datetime.now().isoformat(timespec="seconds"),
           "params": {"WARMUP_CYCLES": WARMUP_CYCLES, "WINDOW_S": WINDOW_S,
                      "MAX_LAUNCHER_GAP_S": MAX_LAUNCHER_GAP_S},
           "runs": {run: analyze_run(run, r, models, spans) for run, r in sorted(cache.items())}}
    out_p.write_text(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
