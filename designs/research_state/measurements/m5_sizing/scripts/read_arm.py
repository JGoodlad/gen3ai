#!/usr/bin/env python3
"""REGISTRATION §5.2 DESCRIPTORS + the infrastructure checks of one learning arm, from its TensorBoard
events and run dir (read-only). Never a verdict: U and G-A are `read_meters.py`'s.

    read_arm.py models/<run> [--json OUT]

C-1 (PROGRESS): an eval cycle's mid-rollout logger.dump stamps the straddling update's train/* at the eval
step, so every train/* point whose step coincides with an eval step is DROPPED before any statistic.
Updates 1-2 (compile prewarm, Adam's lazy state) are dropped from the timing statistics.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def scalars(run: Path) -> Dict[str, List[Tuple[int, float, float]]]:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    out: Dict[str, List[Tuple[int, float, float]]] = {}
    for f in sorted(glob.glob(str(run / "tb" / "**" / "events.out*"), recursive=True)):
        ea = EventAccumulator(f, size_guidance={"scalars": 0})
        ea.Reload()
        for t in ea.Tags()["scalars"]:
            out.setdefault(t, []).extend((e.step, e.value, e.wall_time) for e in ea.Scalars(t))
    for v in out.values():
        v.sort()
    return out


def q(xs, ps=(25, 50, 75)):
    xs = np.asarray(xs, float)
    return [float(np.percentile(xs, p)) for p in ps] if xs.size else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    run = Path(a.run)
    s = scalars(run)
    eval_steps = {st for st, _v, _w in s.get("eval/duration_sec", [])}

    def series(tag, drop_eval=True, skip_first=0):
        rows = [(st, v, w) for st, v, w in s.get(tag, []) if not (drop_eval and st in eval_steps)]
        return rows[skip_first:]

    out: Dict[str, object] = {"run": str(run), "eval_steps": sorted(eval_steps)}
    tm = series("train/train_ms", skip_first=2)
    out["train_ms_q25_50_75"] = q([v for _s, v, _w in tm])
    out["updates_seen"] = len(s.get("train/train_ms", []))
    wall = [w for _s, _v, w in s.get("train/train_ms", [])]
    if len(wall) > 3:
        gaps = np.diff(wall[2:])
        out["update_cycle_wall_s_q25_50_75"] = q(gaps)
    for tag in ("train/approx_kl", "train/clip_fraction", "train/noise_scale_policy",
                "train/noise_scale_ratio_policy", "rollout/ep_len_mean",
                "rust_env/trainee_decisions_per_s", "rust_env/core_ms_per_host_step",
                "rust_env/gpu_wait_ms_per_host_step", "staleness/current_share", "staleness/age_mean",
                "staleness/probe_age_1_ratio_absdev", "staleness/probe_age_1_clip_frac",
                "staleness/probe_age_1_approx_kl", "staleness/games_split", "staleness/carry_rows"):
        r = series(tag)
        if r:
            v = [x for _s, x, _w in r]
            out[tag] = {"mean": float(np.mean(v)), "q25_50_75": q(v), "first": v[0], "last": v[-1], "n": len(v)}
    lr = s.get("train/learning_rate", [])
    if lr:
        out["lr_first_last_min_max"] = [lr[0][1], lr[-1][1], min(v for _s, v, _w in lr), max(v for _s, v, _w in lr)]
    ns = s.get("train/noise_scale_policy", [])
    if ns:
        out["noise_scale_policy_by_M"] = {f"{st / 1e6:.1f}M": round(v) for st, v, _w in ns[:: max(1, len(ns) // 8)]}
    life = {}
    for tag in ("cuda_demand_mib", "cuda_reserved_mib", "cuda_device_free_mib", "cuda_floor_mib", "cuda_ceiling_mib",
                "cuda_level", "cuda_ooms", "cuda_alloc_retries", "cuda_segments_after_freeze", "cuda_sustained"):
        r = s.get("lifecycle/" + tag, [])
        if r:
            v = [x for _s, x, _w in r]
            life[tag] = {"min": min(v), "max": max(v), "last": v[-1], "n": len(v),
                         "at_first_eval": next((x for st, x, _w in r if eval_steps and st >= min(eval_steps)), None)}
    out["lifecycle"] = life
    ev = s.get("eval/duration_sec", [])
    if ev:
        out["eval_cycle_s"] = [v for _s, v, _w in ev]
    cv = run / "canary_verdicts.jsonl"
    if cv.exists():
        rows = [json.loads(x) for x in cv.read_text().splitlines() if x.strip()]
        out["canary"] = {"n": len(rows), "verdicts": sorted({str(r.get("verdict", r.get("status"))) for r in rows})}
    else:
        out["canary"] = None
    md = run / "metadata.json"
    if md.exists():
        m = json.loads(md.read_text())
        out["pin_history"] = m.get("pin_history")
        out["git_hash"] = m.get("git_hash")
    for k in ("final_model.zip",):
        out[k] = (run / k).exists()
    txt = json.dumps(out, indent=1, default=str)
    if a.json:
        Path(a.json).write_text(txt)
    print(txt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
