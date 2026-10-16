"""Extract the per-update throughput series of every Rustboro run from its TensorBoard events.

    python designs/research_state/measurements/fps_and_restarts_2026-10-16/extract.py \
        <models-dir> <out-cache.json> [run-glob ...]

READ-ONLY on ``models/``. One TB event file == one child PROCESS (a fresh launch, a launcher
restart, or a manual second leg); a fork inherits its parent's event files (``tb-inherit``),
so a file whose first step lies at or below the fork point is the PARENT's and is tagged
``inherited`` by step, never counted as one of this run's restarts.

Per process it keeps, for every scalar in TAGS, the (step, wall_time, value) triples.
"""
from __future__ import annotations

import fnmatch
import json
import sys
from pathlib import Path

from tensorboard.backend.event_processing.event_file_loader import EventFileLoader
from tensorboard.util import tensor_util

TAGS = (
    "rollout/collect_ms", "train/train_ms", "lifecycle/update_wall_s", "rollout/ep_len_mean",
    "rollout/collect_decisions", "rust_env/host_steps", "rust_env/core_ms_per_host_step",
    "rust_env/gpu_wait_ms_per_host_step", "rust_env/trainee_rows_per_host_step",
    "rust_env/opp_draw_ms_per_host_step", "rust_env/flush_ms_per_host_step",
    "rust_env/submit_ms_per_host_step", "rust_env/fill_ms",
    "train/selfplay_fraction", "eval/duration_sec", "eval/wall_sec", "rust_eval/cycle_wall_s",
    "eval/win_rate_mean", "eval/elo", "train/approx_kl", "train/entropy_loss",
    "train/clip_fraction", "train/value_loss", "behaviour/excluded_frac",
    "compile/recompiles_after_lock", "compile/train_ms_vs_lock_baseline", "compile/graphs_total",
    "lifecycle/cuda_reserved_mib", "lifecycle/cuda_update_peak_alloc_mib",
    "lifecycle/cuda_rollout_peak_alloc_mib", "time/fps", "eval/pool_snapshot_count",
    "rust_env/p2_policy_decisions", "rust_env/arena_live_rows", "rust_env/games_ended",
    "rust_env/draw_ms_per_host_step", "rust_env/write_ms_per_host_step", "rust_env/post_ms_per_host_step",
)


def _value(ev_value) -> float | None:
    if ev_value.HasField("simple_value"):
        return float(ev_value.simple_value)
    if ev_value.HasField("tensor"):
        try:
            return float(tensor_util.make_ndarray(ev_value.tensor).item())
        except Exception:  # noqa: BLE001 - a non-scalar tensor is not ours
            return None
    return None


def read_file(path: Path) -> dict:
    series: dict[str, list] = {t: [] for t in TAGS}
    for ev in EventFileLoader(str(path)).Load():
        if not ev.HasField("summary"):
            continue
        for v in ev.summary.value:
            if v.tag in series:
                x = _value(v)
                if x is not None:
                    series[v.tag].append((int(ev.step), float(ev.wall_time), x))
    return {k: s for k, s in series.items() if s}


def main(argv: list[str]) -> int:
    models, out = Path(argv[0]), Path(argv[1])
    globs = argv[2:] or ["rb_*"]
    cache = json.loads(out.read_text()) if out.exists() else {}
    for run in sorted(models.iterdir()):
        if not any(fnmatch.fnmatch(run.name, g) for g in globs) or not (run / "tb").is_dir():
            continue
        files = sorted((run / "tb").glob("events.out.tfevents.*"))
        sig = [[f.name, f.stat().st_size] for f in files]
        if run.name in cache and cache[run.name]["sig"] == sig:
            continue
        procs = []
        for f in files:
            s = read_file(f)
            procs.append({"file": f.name, "series": s})
        cache[run.name] = {"sig": sig, "procs": procs}
        print(run.name, len(files), flush=True)
        out.write_text(json.dumps(cache))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
