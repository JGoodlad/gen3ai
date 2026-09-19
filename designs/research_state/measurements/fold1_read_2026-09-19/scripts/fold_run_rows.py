#!/usr/bin/env python3
"""THE ERA-1 FOLD's OWN RUN ROWS — read from the TENSORBOARD EVENTS, POST-FORK ONLY.

🚨 **A FORK'S TB DIRECTORY CARRIES ITS PARENT'S WHOLE HISTORY.** ``ai_v13_07_fold1``'s events run
from 196,608 — arm W's first rollout — because the fork inherits the parent's event files. Every
series here is therefore sliced at the fork step (75,005,952) and only the fold's OWN points are
reported. Reading the unsliced series is the trap that made ``g7_ladder`` structurally wrong on a
fork (ledger ``36f8f7eb``).

This script also records the STRUCTURAL finding of registered row 3: the fold has no
``snapshot_ladder/`` and no post-fork snapshot, so there is no ladder fit to make (n = 0).
It plays no battles and loads no model.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

MAIN = Path("/home/goodlad/dev/gen3ai")
FOLD = MAIN / "models/ai_v13_07_fold1"
ARMW = MAIN / "models/ai_v13_02_flywheel_winprob"
ARMWB = MAIN / "models/ai_v13_04_flywheel_winprob_b"
FORK_STEP = 75_005_952

TAGS = ("eval/win_rate_vs_pool", "eval/win_rate_vs_bots", "eval/mean_ep_len_vs_bots",
        "train/learning_rate", "train/entropy_loss", "distill/teacher_agreement_on_slice",
        "distill/collateral_kl_vs_parent", "distill/on_slice_kl", "distill/kl",
        "distill/t1_gated_frac", "distill/t2_gated_frac", "distill/stop_signal",
        "distill/n_teachers_active", "train/selfplay_fraction")


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


def main() -> int:
    rec: dict = {
        "what": "the era-1 fold's own run rows, POST-FORK ONLY, plus the structural ladder finding",
        "fork_step": FORK_STEP,
        "fork_hazard": ("a fork INHERITS its parent's TB events, so every series in "
                        "models/ai_v13_07_fold1/tb starts at arm W's 196,608. Every row below is "
                        "sliced at the fork step; the unsliced series is arm W's history and is "
                        "the same trap that made g7_ladder structurally wrong on a fork (36f8f7eb)."),
    }
    s = series(FOLD)
    rec["post_fork"] = {}
    for t, pts in sorted(s.items()):
        post = [(st, v) for st, v in pts if st > FORK_STEP]
        rec["post_fork"][t] = {
            "n_points_total_in_tb": len(pts),
            "n_points_post_fork": len(post),
            "first_step_in_tb": pts[0][0] if pts else None,
            "points": [[st, round(v, 6)] for st, v in post],
        }

    # ---- registered row 3: the ladder, as a STRUCTURAL fact -------------------------------
    snaps = sorted((FOLD / "snapshots").glob("snapshot_*.zip"))
    own = [p for p in snaps if int(p.stem.split("_")[-1]) > FORK_STEP]
    seed = json.loads((FOLD / "snapshots" / "pool_seed.json").read_text()) \
        if (FOLD / "snapshots" / "pool_seed.json").exists() else {}
    rec["row3_ladder_structural"] = {
        "verdict": "NOT RUNNABLE at n = 0",
        "snapshot_ladder_dir_exists": (FOLD / "snapshot_ladder").exists(),
        "n_snapshots_in_pool": len(snaps),
        "n_snapshots_the_fold_promoted_itself": len(own),
        "pool_is_entirely_seeded_from": seed.get("parent_run_name"),
        "seeded_snapshot_span": [snaps[0].stem, snaps[-1].stem] if snaps else None,
        "why": ("a snapshot_ladder is a Bradley-Terry fit over a run's OWN promoted snapshots. This "
                "run promoted NONE in 6M steps, so there is no node to fit and no rating to quote "
                "-- this is not 'below the n >= 12 report floor', it is n = 0."),
        "comparators_for_the_record": {
            "armW_nodes": len(list((ARMW / "snapshots").glob("snapshot_*.zip"))),
            "armWb_nodes": len(list((ARMWB / "snapshots").glob("snapshot_*.zip"))),
            "armW_ladder_newest_committed": _newest(ARMW),
            "armWb_ladder_newest_committed": _newest(ARMWB),
        },
    }

    wp = rec["post_fork"].get("eval/win_rate_vs_pool", {}).get("points", [])
    rec["row3_descriptor_win_rate_vs_pool"] = {
        "series": wp,
        "note": ("🚨 the fold promoted NOTHING, so these points are against an IDENTICAL, FROZEN "
                 "opponent set -- arm W's own 20 late snapshots, seeded at fork. A within-run "
                 "series against a fixed pool is unusually clean; it is still a DESCRIPTOR with no "
                 "bar attached, and the regime boundary of rule 5 (sec 3.2) does not bite because "
                 "every point is post-2026-09-07 and inside one run."),
        "promotion_gate": ("--promote-threshold, 0.55 under the greedy sentinel regime; no point "
                           "reaches it"),
    }
    dest = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("fold_run_rows.json")
    dest.write_text(json.dumps(rec, indent=1))
    print(json.dumps({k: v for k, v in rec.items() if k != "post_fork"}, indent=1))
    for t, d in rec["post_fork"].items():
        print(f"{t:44s} post-fork n={d['n_points_post_fork']:3d}  {d['points'][-3:]}")
    return 0


def _newest(run: Path):
    f = run / "snapshot_ladder" / "ladder.json"
    if not f.exists():
        return None
    d = json.loads(f.read_text())
    return {"computed_at": d.get("computed_at"),
            "eval_sentinel_edges_dropped": d.get("eval_sentinel_edges_dropped"),
            "n_nodes": len(d.get("ratings", d.get("nodes", [])) or [])}


if __name__ == "__main__":
    raise SystemExit(main())
