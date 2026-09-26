"""The LEARNER BATTERY's speed endpoint, the per-epoch KL / clip descriptor and the stall rate — read
from each arm's own TensorBoard events and child log. Model-free, torch-free, writes nothing unless
--json is given. Registration: designs/research_state/learner_battery_2026-09-26.md §4.1, §4.4, §5.

    # the registered read (after every arm has finished); arms as LABEL=RUN, C first
    python speed_read.py --fork-step 75005952 C=ai_v14_02_lbat_ctrl E5=ai_v14_03_lbat_e5 \
        T32=ai_v14_04_lbat_t32 L95=ai_v14_05_lbat_l95 --json OUT.json
    # the T32 FUTILITY look (§4.3): after T32's 11th post-fork rollout
    python speed_read.py --fork-step 75005952 --futility T32 C=ai_v14_02_lbat_ctrl T32=ai_v14_04_lbat_t32

ONE ROLLOUT = one `train/train_ms` event (logged once per train() call; 48 x 2048 = 98,304 steps).
Only events at step > --fork-step are read: a fork's tb/ also holds its PARENT's curves as a
truncated prefix (tb_inherit), and those are the parent's rollouts, not the arm's.

  iter_s   = wall-time gap between consecutive rollouts' train_ms events (step gap exactly 98,304),
             dropping the first WARMUP (3) rollouts of every events file (a process start: compile) and
             any gap across a restart (a step gap != 98,304). The median is taken, so the eval
             rollouts (every 2M) and one-off stalls do not move it.
  train_s  = train_ms / 1000 (the PPO update alone: GPU-bound, CPU-contention-robust).

THE SPEED ENDPOINT (registered, §4.1). Rollout collection is the same code on C, E5 and L95, and on
T32 differs only in the trainee's GPU forward (not credited: conservative). So the gain is PROJECTED
from the update time, holding the rollout at C's:
    t_X = med(iter_C) - (med(train_C) - med(train_X));   S_X = 1 - t_X / med(iter_C)
S_X is the fraction of GPU-time per step saved (1/(1-S) - 1 more steps per GPU-hour). CI: 95 %
percentile bootstrap over each arm's rollouts (the three medians resampled independently, 4,000
draws, seed 20260926). The MEASURED iteration ratio med(iter_X)/med(iter_C) is a DESCRIPTOR beside
it: it carries the box's contention on each arm's own days, which the projection removes.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import statistics as st
import sys
from pathlib import Path

ROLLOUT = 48 * 2048
WARMUP = 3
DRAWS = 4000
SEED = 20260926
MODELS = Path("/home/goodlad/dev/gen3ai/models")
#: logged sparsely (`train/n_epochs` only on some rollouts of an unfrozen KL controller; every rollout
#: under --fork-lr-freeze) or once at launch (`hparams/gae_lambda`): read as the latest value.
STATE_TAGS = ("train/n_epochs", "hparams/gae_lambda")
STATE: dict = {}


def run_dir(ref: str) -> Path:
    p = Path(ref)
    if not p.is_absolute():
        p = MODELS / ref
    if not (p / "tb").is_dir():
        sys.exit(f"REFUSED: {p}/tb does not exist")
    return p


def load_events(rd: Path, fork_step: int) -> list:
    """[(file_idx, step, wall_time, {tag: value})] per rollout (train_ms event), step > fork_step."""
    from tensorboard.backend.event_processing.event_file_loader import EventFileLoader
    from tensorboard.util import tensor_util

    files = sorted((rd / "tb").rglob("events.out.tfevents.*"))
    by_step: dict = {}
    for fi, f in enumerate(files):
        for ev in EventFileLoader(str(f)).Load():
            if not ev.HasField("summary") or ev.step < fork_step:
                continue
            for v in ev.summary.value:
                t = v.tag
                if t in STATE_TAGS:   # sparse / one-shot: the LATEST value at or after the fork step
                    val = (float(tensor_util.make_ndarray(v.tensor)) if v.HasField("tensor")
                           else float(v.simple_value))
                    if ev.step >= STATE.get(t, (-1, None))[0]:
                        STATE[t] = (ev.step, val)
                    continue
                if ev.step == fork_step:
                    continue
                if not (t == "train/train_ms" or t.startswith("train/approx_kl_epoch_")
                        or t.startswith("train/clip_fraction_epoch_") or t in (
                            "train/learning_rate", "train/dose_rate",
                            "train/approx_kl", "train/clip_fraction", "train/selfplay_fraction",
                            "rollout/ep_len_mean")):
                    continue
                val = (float(tensor_util.make_ndarray(v.tensor)) if v.HasField("tensor")
                       else float(v.simple_value))
                row = by_step.setdefault(ev.step, {"file": fi, "wall": None, "tags": {}})
                row["tags"][t] = val
                if t == "train/train_ms":
                    row["wall"], row["file"] = ev.wall_time, fi
    return [(r["file"], s, r["wall"], r["tags"]) for s, r in sorted(by_step.items())
            if r["wall"] is not None]


def arm_series(rows: list) -> dict:
    seen: dict = {}
    iters, trains = [], []
    prev = None
    for fi, step, wall, tags in rows:
        seen[fi] = seen.get(fi, 0) + 1
        warm = seen[fi] > WARMUP
        if warm:
            trains.append(tags["train/train_ms"] / 1000.0)
        if prev is not None and warm and prev[0] == fi and step - prev[1] == ROLLOUT:
            iters.append(wall - prev[2])
        prev = (fi, step, wall)
    return {"iter": iters, "train": trains}


def boot_median(xs: list, rng: random.Random) -> float:
    return st.median(rng.choices(xs, k=len(xs)))


def speed(c: dict, x: dict) -> dict:
    mi, mc, mx = st.median(c["iter"]), st.median(c["train"]), st.median(x["train"])
    s = (mc - mx) / mi
    rng = random.Random(SEED)
    draws = sorted((boot_median(c["train"], rng) - boot_median(x["train"], rng))
                   / boot_median(c["iter"], rng) for _ in range(DRAWS))
    lo, hi = draws[int(0.025 * DRAWS)], draws[int(0.975 * DRAWS) - 1]
    out = {"S": s, "S_lo": lo, "S_hi": hi, "steps_per_gpu_hour_gain": 1 / (1 - s) - 1,
           "train_s_C": mc, "train_s_X": mx, "iter_s_C": mi}
    if x["iter"]:
        out["measured_iter_ratio"] = st.median(x["iter"]) / mi
    return out


def per_epoch(rows: list) -> dict:
    out = {}
    for kind in ("approx_kl", "clip_fraction"):
        ks = sorted({int(m.group(1)) for _, _, _, t in rows for k in t
                     if (m := re.fullmatch(rf"train/{kind}_epoch_(\d+)", k))})
        out[kind] = {k: st.mean([t[f"train/{kind}_epoch_{k}"] for _, _, _, t in rows
                                 if f"train/{kind}_epoch_{k}" in t]) for k in ks}
    return out


def stalls_per_1m(rd: Path, rows: list) -> dict:
    """Every `[STALL LOGGED]` line in the run's child log over the read's steps. Valid ONLY when the
    log covers exactly those steps — a FORK's own log does; a read of a window inside a longer run
    (the stand-in) over-counts, and says so in the registration's validation note."""
    log = rd / "launcher_child.full.log"
    if not log.is_file() or not rows:
        return {"stalls": None}
    n = sum(1 for line in log.open("rb") if b"[STALL LOGGED]" in line)
    span = rows[-1][1] - rows[0][1] + ROLLOUT
    return {"stalls": n, "steps": span, "per_1m": n / (span / 1e6)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("arms", nargs="+", help="LABEL=RUN, the control labelled C")
    ap.add_argument("--fork-step", type=int, required=True)
    ap.add_argument("--futility", default=None, help="the §4.3 look for this arm label")
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)
    arms = dict(x.split("=", 1) for x in a.arms)
    if "C" not in arms:
        sys.exit("REFUSED: no C=<control run>")
    out: dict = {"fork_step": a.fork_step, "arms": {}}
    series = {}
    for lab, ref in arms.items():
        rd = run_dir(ref)
        STATE.clear()
        rows = load_events(rd, a.fork_step)
        if len(rows) <= WARMUP + 1:
            sys.exit(f"REFUSED: {lab} has {len(rows)} post-fork rollouts — nothing to read")
        series[lab] = arm_series(rows)
        last = rows[-1][3]
        out["arms"][lab] = {
            "run": str(rd), "rollouts": len(rows), "first_step": rows[0][1], "last_step": rows[-1][1],
            "median_iter_s": st.median(series[lab]["iter"]) if series[lab]["iter"] else None,
            "median_train_s": st.median(series[lab]["train"]),
            "update_share": (st.median(series[lab]["train"]) / st.median(series[lab]["iter"])
                             if series[lab]["iter"] else None),
            "n_epochs": STATE.get("train/n_epochs", (None, None))[1],
            "gae_lambda": STATE.get("hparams/gae_lambda", (None, None))[1],
            "lr_last": last.get("train/learning_rate"),
            "dose_rate_last": last.get("train/dose_rate"),
            "selfplay_fraction_last": last.get("train/selfplay_fraction"),
            "per_epoch": per_epoch(rows), "stall": stalls_per_1m(rd, rows)}
    for lab in arms:
        if lab != "C":
            out["arms"][lab]["speed_vs_C"] = speed(series["C"], series[lab])
    if a.futility:
        sp = out["arms"][a.futility]["speed_vs_C"]
        n = out["arms"][a.futility]["rollouts"]
        stop = sp["S"] < 0.025
        out["futility"] = {"arm": a.futility, "rollouts": n, "S": sp["S"],
                           "verdict": ("NOT YET (needs >= 11 post-fork rollouts)" if n < 11 else
                                       "STOP — projected S < 2.5 %, the 5 % bar is out of reach"
                                       if stop else "CONTINUE")}
    for lab, r in out["arms"].items():
        print(f"== {lab}  {Path(r['run']).name}  rollouts {r['rollouts']} "
              f"({r['first_step']:,} .. {r['last_step']:,})  n_epochs {r['n_epochs']}  "
              f"gae_lambda {r['gae_lambda']}  lr {r['lr_last']}")
        mi = r["median_iter_s"]
        print(f"   median iteration {mi:.1f} s   update {r['median_train_s']:.1f} s   update share "
              f"{r['update_share']:.3f}" if mi else f"   update {r['median_train_s']:.1f} s")
        for kind, d in r["per_epoch"].items():
            if d:
                print(f"   {kind:13s} by epoch: " + " ".join(f"{d[k]:.4f}" for k in sorted(d)))
            else:
                print(f"   {kind:13s} by epoch: NO SCALAR (pre-2cc83080 tree) — absence is not a zero")
        s = r["stall"]
        if s.get("stalls") is not None:
            print(f"   [STALL LOGGED] {s['stalls']} over {s['steps']:,} steps = {s['per_1m']:.1f} per 1M")
        if "speed_vs_C" in r:
            v = r["speed_vs_C"]
            print(f"   SPEED vs C: S = {100 * v['S']:+.2f} % [{100 * v['S_lo']:+.2f}, {100 * v['S_hi']:+.2f}]"
                  f"  (= {100 * v['steps_per_gpu_hour_gain']:+.1f} % steps per GPU-hour)"
                  + (f"; measured iteration ratio {v['measured_iter_ratio']:.3f} (descriptor)"
                     if "measured_iter_ratio" in v else ""))
    if a.futility:
        print(f"== FUTILITY ({a.futility}): {out['futility']['verdict']}")
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
