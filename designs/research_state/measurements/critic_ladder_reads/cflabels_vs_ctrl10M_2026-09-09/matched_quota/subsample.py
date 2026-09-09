"""Matched-quota re-read of the cf-label arm's conditioning rows.

THE QUESTION. `cond.own_team_r2.t1` read +0.060 on `ai_v12_12_ladder_cflabels` against -0.024 on
`ai_v12_11_ladder_ctrl10M` (Δ +0.0841 [+0.0323, +0.1825] DETECTED vs zero) — the first
conditioning row any ladder arm has moved. But the two sides were traced under DIFFERENT outcome
quotas (arm 40/40/10, control 5/10/5), so the arm's read frame is 627 battles / 183 teams against
198 / 76, and its own-team DECODER is fit on 429 battles against 104. Horvitz-Thompson
reweighting handles the loss-enrichment SELECTION; it does not handle DECODER POWER. This script
subsamples the arm's trace tree down to the control's realized per-opponent capture profile,
recomputes the capture rates for the subsample so the HT weights stay right, and re-reads all ten
conditioning rows.

NOTHING IS WRITTEN UNDER models/. The subsampled view is a tree of SYMLINKS into the archive plus
a rewritten `eval_manifest.json`, materialized under the job tmp dir.

🚨 THE CAPS ARE THE CONTROL'S REALIZED ONES, NOT ITS NOMINAL QUOTA. Under battle-level
work-stealing each shard unit carries `max(1, ceil(quota / n_shards))`, so the control's nominal
5/10/5 landed as 8 traced wins and up to 12 traced losses per opponent. Matching the NOMINAL 5/10
would over-shrink the arm below the control's own frame; the matched row therefore uses (8, 12).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

STEP = 10000032
ARM = "ai_v12_12_ladder_cflabels"
CTL = "ai_v12_11_ladder_ctrl10M"
ARCHIVE = "/home/goodlad/dev/gen3ai/models"

#: (cap_win, cap_loss, cap_draw) rungs of the frame-size curve. `matched` is the control's own
#: realized profile (matched on BATTLES); `decoder_matched` is the cap at which the arm's own-team
#: DECODER frame — the battles whose team clears `MIN_TEAM_BATTLES` — matches the control's 104,
#: which the battle-matched rung does NOT (the arm carries more distinct teams per battle, so
#: fewer of its teams reach the 4-battle threshold); `full` is the arm as traced. The intermediate rungs double each time, which is
#: the nominal 5/10 -> 10/20 -> 20/40 ladder expressed in realized units.
RUNGS: Tuple[Tuple[str, Optional[int], Optional[int], Optional[int]], ...] = (
    ("matched", 8, 12, 5),
    ("decoder_matched", 11, 16, 5),
    ("x2", 16, 24, 10),
    ("x4", 32, 48, 10),
    ("full", None, None, None),
)


def load_json(path: str) -> Any:
    with open(path) as fh:
        return json.load(fh)


def classify(trace_dir: str, opp: str) -> Dict[str, List[str]]:
    """The opponent's traced battle basenames, split WIN / LOSS / DRAW by the summary's result."""
    out: Dict[str, List[str]] = {"WIN": [], "LOSS": [], "DRAW": []}
    odir = os.path.join(trace_dir, opp)
    if not os.path.isdir(odir):
        return out
    for fn in sorted(os.listdir(odir)):
        if not fn.endswith("_states.npz"):
            continue
        base = fn[: -len("_states.npz")]
        spath = os.path.join(odir, base + "_summary.json")
        if not os.path.exists(spath):
            continue
        res = (load_json(spath).get("meta") or {}).get("result")
        out[res if res in ("WIN", "LOSS") else "DRAW"].append(base)
    return out


def plan(trace_dir: str, cap_w: Optional[int], cap_l: Optional[int], cap_d: Optional[int],
         seed: int) -> Tuple[Dict[str, List[str]], Dict[str, Any], Dict[str, int]]:
    """Per-opponent kept basenames + a REWRITTEN selection block whose capture rates describe the
    SUBSAMPLE.

    `capture_rate_win = kept_wins / battles_won` (and likewise for losses and draws), with
    `battles_lost = battles_played - battles_won - battles_drawn` — exactly the denominators the
    trainer's own manifest uses. A class with zero kept battles gets `None`, which
    `extract_cycle` turns into a DROPPED row rather than a weight of 1.0.
    """
    man = load_json(os.path.join(trace_dir, "eval_manifest.json"))
    sel = man["selection"]["opponents"]
    kept: Dict[str, List[str]] = {}
    new_sel: Dict[str, Any] = {}
    tally = {"WIN": 0, "LOSS": 0, "DRAW": 0}
    for i, opp in enumerate(sorted(sel)):
        rec = dict(sel[opp])
        cls = classify(trace_dir, opp)
        rng = np.random.default_rng([seed, i, 0xC0FFEE])
        pick: Dict[str, List[str]] = {}
        for name, cap in (("WIN", cap_w), ("LOSS", cap_l), ("DRAW", cap_d)):
            pool = sorted(cls[name])
            if cap is None or len(pool) <= cap:
                pick[name] = pool
            else:
                pick[name] = sorted(rng.choice(np.array(pool), size=cap, replace=False).tolist())
            tally[name] += len(pick[name])
        kept[opp] = pick["WIN"] + pick["LOSS"] + pick["DRAW"]
        played = int(rec.get("battles_played", 0))
        won = int(rec.get("battles_won", 0))
        drawn = int(rec.get("battles_drawn", 0))
        lost = played - won - drawn
        rec["traces_written"] = len(kept[opp])
        rec["traces_won"] = len(pick["WIN"])
        rec["traces_drawn"] = len(pick["DRAW"])
        rec["capture_rate_win"] = (len(pick["WIN"]) / won) if won > 0 and pick["WIN"] else None
        rec["capture_rate_loss"] = (len(pick["LOSS"]) / lost) if lost > 0 and pick["LOSS"] else None
        rec["capture_rate_draw"] = (len(pick["DRAW"]) / drawn) if drawn > 0 and pick["DRAW"] \
            else None
        new_sel[opp] = rec
    man["selection"]["opponents"] = new_sel
    man["selection"]["subsample"] = {"cap_win": cap_w, "cap_loss": cap_l, "cap_draw": cap_d,
                                     "seed": seed,
                                     "source": os.path.abspath(trace_dir)}
    return kept, man, tally


def materialize(trace_dir: str, kept: Dict[str, List[str]], man: Dict[str, Any],
                dest_run: str) -> str:
    """A SYMLINK view of the subsample: `<dest_run>/eval_traces/step_<STEP>/<opp>/<base>_*`."""
    dest = os.path.join(dest_run, "eval_traces", f"step_{STEP}")
    if os.path.exists(dest_run):
        shutil.rmtree(dest_run)
    for opp, bases in kept.items():
        odir = os.path.join(dest, opp)
        os.makedirs(odir, exist_ok=True)
        srcdir = os.path.join(os.path.abspath(trace_dir), opp)
        for base in bases:
            # EVERY sibling of the battle, not just the npz + summary the conditioning meters
            # read: `cf_audit.build_frame` refuses a battle with no `_reconstruction.json`, and a
            # view missing them yields an EMPTY frame and a silent all-zero `pop` weight.
            for fn in os.listdir(srcdir):
                if fn.startswith(base):
                    os.symlink(os.path.join(srcdir, fn), os.path.join(odir, fn))
    with open(os.path.join(dest, "eval_manifest.json"), "w") as fh:
        json.dump(man, fh)
    return dest


def read_block(run_dir_for_traces: str, strength_from: str, boot: int, seed: int) -> Dict[str, Any]:
    """`conditioning_block` on the subsampled tree, with the STRENGTH AXIS taken from the REAL run.

    The axis is a function of the run's own snapshot ladder and the manifest's true win rates —
    neither of which a subsample touches — so it is computed against the real run directory and
    injected. Computing it against the symlink tree would just fail (no `eval_results.jsonl`).
    """
    from main.ops import conditioning_meters as CM

    real = CM.strength_axis
    cache: Dict[str, Any] = {}

    def patched(run_dir: str, step: int, per_opp, *, say=lambda _m: None):
        if "v" not in cache:
            cache["v"] = real(strength_from, step, per_opp, say=say)
        return cache["v"]

    CM.strength_axis = patched                                    # type: ignore[assignment]
    try:
        return CM.conditioning_block(run_dir_for_traces, STEP, boot=boot, seed=seed)
    finally:
        CM.strength_axis = real                                   # type: ignore[assignment]


def summarise(block: Dict[str, Any]) -> Dict[str, Any]:
    from main.ops import critic_readouts as R

    return {"points": block["points"],
            "ci": {k: R.ci_of(block["points"][k], block["_draws"].get(k, np.empty(0)))
                   for k in block["points"]},
            "draws": {k: v.tolist() for k, v in block["_draws"].items()},
            "frame": block["frame"], "omitted": block["omitted"]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tmp", required=True, help="where the symlink views are materialized")
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--block-seed", type=int, default=0,
                    help="the registered read's seed (0) — folds, state cap and bootstrap")
    ap.add_argument("--curve-seeds", type=int, default=10,
                    help="subsample seeds for the intermediate frame-size rungs")
    args = ap.parse_args(argv)

    os.makedirs(args.out, exist_ok=True)
    os.makedirs(args.tmp, exist_ok=True)
    arm_traces = os.path.join(ARCHIVE, ARM, "eval_traces", f"step_{STEP}")
    arm_run = os.path.join(ARCHIVE, ARM)
    ctl_run = os.path.join(ARCHIVE, CTL)

    out: Dict[str, Any] = {"step": STEP, "arm": ARM, "control": CTL,
                           "block_seed": args.block_seed, "boot": args.boot,
                           "rungs": [list(r) for r in RUNGS]}

    print("control (as traced) ...", flush=True)
    out["control_block"] = summarise(read_block(ctl_run, ctl_run, args.boot, args.block_seed))
    print("arm FULL (as traced) ...", flush=True)
    out["arm_full"] = summarise(read_block(arm_run, arm_run, args.boot, args.block_seed))

    out["rung_reads"] = {}
    for name, cw, cl, cd in RUNGS:
        if name == "full":
            out["rung_reads"][name] = [{"seed": None, **out["arm_full"]}]
            continue
        n_seeds = (args.seeds if name in ("matched", "decoder_matched")
                   else args.curve_seeds)
        rows = []
        for s in range(n_seeds):
            kept, man, tally = plan(arm_traces, cw, cl, cd, s)
            dest_run = os.path.join(args.tmp, f"view_{name}_s{s}")
            materialize(arm_traces, kept, man, dest_run)
            r = summarise(read_block(dest_run, arm_run, args.boot, args.block_seed))
            r["seed"] = s
            r["tally"] = tally
            rows.append(r)
            shutil.rmtree(dest_run)
            print(f"  {name} seed {s}: battles={r['frame']['n_battles']} "
                  f"teams={r['frame']['n_teams']} "
                  f"own_team_t1={r['points'].get('cond.own_team_r2.t1', float('nan')):+.4f}",
                  flush=True)
        out["rung_reads"][name] = rows

    with open(os.path.join(args.out, "matched_quota.json"), "w") as fh:
        json.dump(out, fh)
    print("wrote", os.path.join(args.out, "matched_quota.json"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
