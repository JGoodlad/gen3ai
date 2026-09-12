#!/usr/bin/env python3
"""THE STEP CURVE — what the 75M run's TRAINING MIX did across the three curve intervals.

🚨 THIS IS A HYPOTHESIS GENERATOR, NOT A TEST. The representation's class information is flat over
10M->20M, falls past the bar on BOTH draws over 20M->40M, and is flat again over 40M->73M. A step
change in ONE interval invites a curriculum explanation, so this reads the run's own TensorBoard
and asks a single narrow question: **does anything change over 20M-40M that does not change over
10M-20M or 40M-73M?**

One run and one interval cannot separate a curriculum change from any other thing that moved in
the same window. Whatever this finds is reported WITH that sentence attached, never as a cause.

Read-only: it opens the run's event files and writes nothing anywhere near `models/`.

Usage:  curriculum_read.py <run-dir> [--json OUT.json]
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

#: the curve's three intervals, plus the run-in below the first checkpoint for context.
WINDOWS = [("0-10M", 0, 9_969_408), ("10M-20M", 9_969_408, 18_660_864),
           ("20M-40M", 18_660_864, 40_935_168), ("40M-73M", 40_935_168, 73_121_280)]

TAGS = [
    "train/selfplay_fraction",     # share of rollout episodes played against a pool snapshot
    "train/nonbot_fraction",       # the complement of the bot share
    "train/stable_fraction",
    "signal/outcome_n_bots",       # episodes scored vs BOTS in the buffer
    "signal/outcome_n_pool",       # episodes scored vs POOL in the buffer
    "signal/outcome_win_rate_bots",
    "signal/outcome_win_rate_pool",
    "opp_intent/label_bot_frac",
    "eval/pool_snapshot_count",
    "eval/win_rate_vs_bots",
    "eval/win_rate_vs_pool",
]


def series(run_dir: Path, tag: str) -> tuple[np.ndarray, np.ndarray]:
    steps: list[int] = []
    vals: list[float] = []
    for f in sorted(glob.glob(str(run_dir / "tb" / "events.out.tfevents.*"))):
        ea = EventAccumulator(f, size_guidance={"scalars": 0})
        ea.Reload()
        if tag not in ea.Tags()["scalars"]:
            continue
        for e in ea.Scalars(tag):
            steps.append(e.step)
            vals.append(e.value)
    if not steps:
        return np.array([]), np.array([])
    o = np.argsort(steps)
    return np.array(steps)[o], np.array(vals)[o]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    run = Path(a.run_dir)
    out: dict = {"windows": [w[0] for w in WINDOWS], "tags": {}}

    header = "| tag | " + " | ".join(w[0] for w in WINDOWS) + " | changes in 20M-40M only? |"
    print(header)
    print("|---" * (len(WINDOWS) + 2) + "|")
    derived: dict = {}
    for tag in TAGS:
        s, v = series(run, tag)
        if s.size == 0:
            print(f"| `{tag}` | " + " | ".join("NO SCALAR" for _ in WINDOWS) + " | — |")
            out["tags"][tag] = None
            continue
        means = []
        for _name, lo, hi in WINDOWS:
            m = (s > lo) & (s <= hi)
            means.append(float(np.mean(v[m])) if m.any() else None)
        derived[tag] = means
        out["tags"][tag] = {"windows": means, "n_points": int(s.size),
                            "first_step": int(s[0]), "last_step": int(s[-1])}
        # "changes in the middle interval only" = the 10M->20M->40M move is large relative to
        # the 40M->73M move. Reported as the two consecutive deltas, never as a verdict.
        def d(i, j):
            if means[i] is None or means[j] is None:
                return None
            return means[j] - means[i]
        d12, d23, d34 = d(1, 2), d(2, 3), d(1, 3)
        note = "—"
        if None not in (d12, d23):
            note = f"Δ(10-20→20-40)={d12:+.4g}, Δ(20-40→40-73)={d23:+.4g}"
        print(f"| `{tag}` | " + " | ".join(
            "—" if m is None else f"{m:,.4g}" for m in means) + f" | {note} |")
    # 🚨 `signal/outcome_n_bots` / `_pool` are a FIXED DIAGNOSTIC SAMPLE CAP, not buffer counts:
    # both read exactly 200 at every one of their ~750 points, so their ratio is 0.5 BY
    # CONSTRUCTION and is not the training mix. The mix is `train/nonbot_fraction`'s complement.
    for tag in ("signal/outcome_n_bots", "signal/outcome_n_pool"):
        rec = out["tags"].get(tag)
        if rec is not None:
            vals = derived.get(tag) or []
            if vals and len({round(v, 6) for v in vals if v is not None}) == 1:
                rec["CAVEAT"] = ("constant at a fixed sample cap across the whole run — a "
                                 "diagnostic sample size, NOT the buffer's opponent mix; its "
                                 "ratio is 0.5 by construction and must not be read as a share")
    nonbot = derived.get("train/nonbot_fraction")
    if nonbot:
        print("\n**Bot share of TRAINING episodes** = `1 - train/nonbot_fraction` "
              "(the real mix; see the caveat on `signal/outcome_n_*`):\n")
        print("| " + " | ".join(w[0] for w in WINDOWS) + " |")
        print("|---" * len(WINDOWS) + "|")
        shares = [None if m is None else 1.0 - m for m in nonbot]
        print("| " + " | ".join("—" if v is None else f"{v:.4f}" for v in shares) + " |")
        out["bot_share_of_training_episodes"] = shares

    # when did the self-play POOL hit its cap (i.e. when did snapshot EVICTION begin)?
    s_pc, v_pc = series(run, "eval/pool_snapshot_count")
    if s_pc.size:
        cap = float(np.max(v_pc))
        at = int(s_pc[v_pc >= cap][0])
        print(f"\n**Pool cap reached** at `eval/pool_snapshot_count` = {cap:g}, "
              f"first at step **{at:,}** — i.e. snapshot EVICTION begins there. "
              f"(The run's retained `snapshots/` start at 36,000,000, consistent with a cap "
              f"of {cap:g} at a 2M promotion cadence.)")
        out["pool_cap"] = {"cap": cap, "first_step_at_cap": at}
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
