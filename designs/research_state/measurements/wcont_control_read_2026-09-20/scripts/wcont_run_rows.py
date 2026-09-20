#!/usr/bin/env python3
"""ROWS 4 AND 5 — THE CONTROL's OWN RUN ROWS, read from the TENSORBOARD EVENTS, POST-FORK ONLY.

🚨 **A FORK'S TB DIRECTORY CARRIES ITS PARENT'S WHOLE HISTORY.** ``ai_v13_09_wcont`` is a fork of
arm W, so its ``tb/`` carries arm W's events from 196,608 onward. Every series here is sliced at the
fork step (75,005,952). Reading an unsliced series is the trap that made ``g7_ladder`` structurally
wrong on a fork (ledger ``36f8f7eb``).

This script plays no battles and loads no model. It produces:

* **row 4 (DESCRIPTOR, already banked)** — the ENTROPY row: ``H_end`` = the median of the last 20
  ``train/entropy_loss`` points, beside the fold path's banked 0.7465 and the 75M run-level entropy
  floor of 0.019. 🚨 Banked in ledger ``e77963c4``; reproduced here as an INSTRUMENT CHECK, carrying
  **no bar**, and the only licensed statement about it is a conditional one (PREDICTION.md sec 3.4).
* **row 5 (STRUCTURAL DESCRIPTOR, disclosed pre-look)** — the PROMOTION ASYMMETRY: the control's own
  post-fork promotions and its ``snapshot_ladder/`` against the fold path's ZERO and none, with
  ``eval/pool_snapshot_count`` beside it. **No Elo is quoted for either arm** (n = 5 is far below
  the n >= 12 report floor).
* the frozen/moving-pool ``win_rate_vs_pool`` and ``win_rate_vs_bots`` descriptors.
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

MAIN = Path("/home/goodlad/dev/gen3ai")
CTRL = MAIN / "models/ai_v13_09_wcont"
FOLD = MAIN / "models/ai_v13_07_fold1"
CONT = MAIN / "models/ai_v13_08_fold1_cont"
FORK_STEP = 75_005_952
FOLD_H_END = 0.7465          # banked, ledger e77963c4
CTRL_H_END_BANKED = 0.9447   # banked, ledger e77963c4
ENTROPY_FLOOR_75M = 0.019    # arm W vs W_b, the 75M run-level entropy spread

TAGS = ("eval/win_rate_vs_pool", "eval/win_rate_vs_bots", "eval/mean_ep_len_vs_bots",
        "train/learning_rate", "train/entropy_loss", "eval/pool_snapshot_count",
        "train/selfplay_fraction", "distill/stop_signal", "distill/n_teachers_active")


def series(run_dir: Path, tags=TAGS) -> dict:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    out: dict = {}
    for d in sorted(p for p in (run_dir / "tb").glob("**/") if list(p.glob("events.out.*"))):
        ea = EventAccumulator(str(d), size_guidance={"scalars": 0})
        ea.Reload()
        have = set(ea.Tags().get("scalars", ()))
        for t in tags:
            if t in have:
                out.setdefault(t, []).extend((int(e.step), float(e.value)) for e in ea.Scalars(t))
    for t in out:
        out[t] = sorted(set(out[t]))
    return out


def post(sr: dict, tag: str):
    return [(s, v) for s, v in sr.get(tag, []) if s > FORK_STEP]


def promotions(run: Path) -> dict:
    snaps = sorted(p.name for p in (run / "snapshots").glob("snapshot_*.zip"))
    steps = [int(n.split("_")[1].split(".")[0]) for n in snaps]
    own = [s for s in steps if s > FORK_STEP]
    lad = run / "snapshot_ladder" / "ladder.json"
    rec = {"n_snapshots_in_pool": len(steps),
           "own_post_fork_promotions": len(own),
           "own_promotion_steps": own,
           "oldest_in_pool": min(steps) if steps else None,
           "newest_in_pool": max(steps) if steps else None,
           "snapshot_ladder_present": lad.exists()}
    if lad.exists():
        d = json.loads(lad.read_text())
        nodes = d.get("ratings", d.get("nodes")) or []
        rec["ladder_n_nodes"] = len(nodes)
        rec["ladder_recipe_stamp_eval_sentinel_edges_dropped"] = d.get("eval_sentinel_edges_dropped")
        rec["ladder_NOT_QUOTED"] = (
            "rule 5 / the n >= 12 report floor: a Bradley-Terry fit over this run's OWN post-fork "
            "nodes has n = %d. NO rating is quoted for this arm in either direction, and the two "
            "arms' pools are not the same object by the end." % len(own))
    return rec


def main() -> int:
    dest = Path(sys.argv[1])
    sr = series(CTRL)
    rec: dict = {
        "what": ("the CONTINUATION CONTROL's own run rows, POST-FORK ONLY: the entropy descriptor "
                 "(row 4, banked) and the promotion asymmetry (row 5, structural)"),
        "fork_step_the_cut": FORK_STEP,
        "fork_hazard": ("a fork INHERITS its parent's TB events, so models/ai_v13_09_wcont/tb "
                        "carries arm W's history. Every row below is sliced at 75,005,952. The "
                        "unsliced series is arm W's -- the trap that made g7_ladder structurally "
                        "wrong on a fork (36f8f7eb)."),
        "series_post_fork": {}, "row4_entropy": {}, "row5_promotions": {},
    }
    for t in TAGS:
        pts = post(sr, t)
        if pts:
            rec["series_post_fork"][t] = [[s, round(v, 6)] for s, v in pts]

    ent = [v for _s, v in post(sr, "train/entropy_loss")]
    if ent:
        # train/entropy_loss is the NEGATIVE entropy term SB3 logs; H = -entropy_loss
        h = [-v for v in ent]
        h_end = statistics.median(h[-20:]) if len(h) >= 20 else statistics.median(h)
        rec["row4_entropy"] = {
            "n_post_fork_points": len(h),
            "H_end_here_median_last_20": round(h_end, 4),
            "H_end_banked_ledger_e77963c4": CTRL_H_END_BANKED,
            "reproduces_banked": bool(abs(h_end - CTRL_H_END_BANKED) < 0.01),
            "fold_path_H_end_banked": FOLD_H_END,
            "gap_control_minus_fold_path": round(h_end - FOLD_H_END, 4),
            "run_level_entropy_floor_75M_armW_vs_Wb": ENTROPY_FLOOR_75M,
            "multiple_of_the_floor": round(abs(h_end - FOLD_H_END) / ENTROPY_FLOOR_75M, 1),
            "H_first_post_fork": round(h[0], 4), "H_min": round(min(h), 4),
            "H_max": round(max(h), 4),
            "status": ("DESCRIPTOR, already banked in the ledger, carrying NO BAR. Reproduced here "
                       "as an instrument check only. Two arms cannot support an "
                       "entropy-to-piloting claim and none is made (PREDICTION.md sec 3.4)."),
        }

    rec["row5_promotions"] = {
        "control_ai_v13_09_wcont": promotions(CTRL),
        "fold_ai_v13_07_fold1": promotions(FOLD),
        "fold_cont_ai_v13_08_fold1_cont": promotions(CONT),
        "pool_snapshot_count_post_fork_control": rec["series_post_fork"].get(
            "eval/pool_snapshot_count"),
        "finding": ("🚨 STRUCTURAL, disclosed pre-look in PREDICTION.md sec 0.3: over the same "
                    "12.09M steps, against the same auto-seeded arm-W pool at the same 0.55 gate, "
                    "the PLAIN CONTINUATION cleared the gate and promoted, and the FOLD PATH never "
                    "once did. NO BAR is attached and no Elo is quoted from either side."),
    }
    dest.write_text(json.dumps(rec, indent=1))
    print(json.dumps({k: v for k, v in rec.items() if k != "series_post_fork"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
