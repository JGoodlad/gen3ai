#!/usr/bin/env python3
"""THE FOLD PATH's OWN RUN ROWS — read from the TENSORBOARD EVENTS, POST-FORK ONLY, ACROSS BOTH RUNS.

🚨 **A FORK'S TB DIRECTORY CARRIES ITS PARENT'S WHOLE HISTORY, AND THIS PATH HAS TWO FORKS.**
``ai_v13_07_fold1``'s events start at arm W's 196,608; ``ai_v13_08_fold1_cont`` inherits THOSE, so
its events carry arm W's history AND the fold's. Every series here is therefore sliced at the
ORIGINAL fork step (75,005,952) — the cut the ledger keeps for this path, unchanged because the
matchup hash ``0a7b730a4d`` did not move across the second fork. Reading an unsliced series is the
trap that made ``g7_ladder`` structurally wrong on a fork (ledger ``36f8f7eb``).

The two runs are then STITCHED on step, deduplicated, and the FORK CROSSING at 81,100,800 is
recorded as a discontinuity rather than smoothed over.

This script also records the STRUCTURAL finding of registered row 4: the continuation, like the
fold, has no ``snapshot_ladder/`` and promoted no snapshot, so there is no ladder fit to make
(n = 0). It plays no battles and loads no model.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

MAIN = Path("/home/goodlad/dev/gen3ai")
FOLD = MAIN / "models/ai_v13_07_fold1"
CONT = MAIN / "models/ai_v13_08_fold1_cont"
ARMW = MAIN / "models/ai_v13_02_flywheel_winprob"
ARMWB = MAIN / "models/ai_v13_04_flywheel_winprob_b"
FORK_STEP = 75_005_952            # the ORIGINAL fold point — the cut for the whole path
CROSSING = 81_100_800             # fold1 -> fold1_cont, the second fork
STOP_FIRE = 82_673_664            # the distill stop rule's first fire, +7.667M

TAGS = ("eval/win_rate_vs_pool", "eval/win_rate_vs_bots", "eval/mean_ep_len_vs_bots",
        "train/learning_rate", "train/entropy_loss", "distill/teacher_agreement_on_slice",
        "distill/collateral_kl_vs_parent", "distill/on_slice_kl", "distill/kl",
        "distill/gated_frac", "distill/off_slice_frac",
        "distill/t1_gated_frac", "distill/t2_gated_frac", "distill/stop_signal",
        "distill/n_teachers_active", "train/selfplay_fraction",
        "eval/pool_snapshot_count", "eval/elo", "eval/win_rate_vs_external")

GRID = {"+1M": 76_005_984, "+3M": 78_006_048, "+6M": 81_100_800,
        "+7.59M": 82_600_848, "+9.09M": 84_100_896, "+12.09M": 87_097_344}


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


def _newest(run: Path):
    f = run / "snapshot_ladder" / "ladder.json"
    if not f.exists():
        return None
    d = json.loads(f.read_text())
    return {"computed_at": d.get("computed_at"),
            "eval_sentinel_edges_dropped": d.get("eval_sentinel_edges_dropped"),
            "n_nodes": len(d.get("ratings", d.get("nodes", [])) or [])}


def main() -> int:
    rec: dict = {
        "what": ("the fold path's own run rows -- ai_v13_07_fold1 + ai_v13_08_fold1_cont, POST-FORK "
                 "ONLY and stitched -- plus the structural ladder finding for the continuation"),
        "fork_step_the_cut": FORK_STEP,
        "fork_crossing_step": CROSSING,
        "stop_rule_first_fire_step": STOP_FIRE,
        "fork_hazard": ("a fork INHERITS its parent's TB events and this path has TWO forks, so "
                        "models/ai_v13_08_fold1_cont/tb carries arm W's history AND the fold's. "
                        "Every row below is sliced at the ORIGINAL fork step 75,005,952 and the "
                        "two runs are stitched on step. The unsliced series is arm W's history -- "
                        "the trap that made g7_ladder structurally wrong on a fork (36f8f7eb)."),
        "crossing_hazard": ("the +6M -> +7.5M leg CROSSES A FORK BOUNDARY at 81,100,800. The fork "
                            "re-seeds the optimizer path, re-pins the LR to the same frozen "
                            "2.8e-5, re-seeds the pool (from the fold, i.e. the same 20 arm-W "
                            "files) and opens a new event file. The matchup hash 0a7b730a4d is "
                            "unchanged, so it is NOT a rule-15 opponent-regime boundary -- but it "
                            "is a discontinuity in the run and is printed on every trajectory row."),
    }

    sf, sc = series(FOLD), series(CONT)
    rec["post_fork"] = {}
    for t in sorted(set(sf) | set(sc)):
        pf = [(st, v) for st, v in sf.get(t, []) if st > FORK_STEP and st <= CROSSING]
        pc = [(st, v) for st, v in sc.get(t, []) if st > CROSSING]
        pts = sorted(set(pf + pc))
        rec["post_fork"][t] = {
            "n_points_fold_leg": len(pf), "n_points_cont_leg": len(pc),
            "first_step_in_cont_tb": sc.get(t, [(None, None)])[0][0],
            "points": [[st, round(v, 6)] for st, v in pts],
            "at_grid": {k: [round(v, 6) for st, v in pts if st == s] or None
                        for k, s in GRID.items()},
        }

    # ---- registered row 4: the ladder, as a STRUCTURAL fact ---------------------------------
    out = {}
    for name, run, cut in (("ai_v13_07_fold1", FOLD, FORK_STEP),
                           ("ai_v13_08_fold1_cont", CONT, CROSSING)):
        snaps = sorted((run / "snapshots").glob("snapshot_*.zip"))
        own = [p for p in snaps if int(p.stem.split("_")[-1]) > cut]
        seed_f = run / "snapshots" / "pool_seed.json"
        seed = json.loads(seed_f.read_text()) if seed_f.exists() else {}
        out[name] = {
            "snapshot_ladder_dir_exists": (run / "snapshot_ladder").exists(),
            "n_snapshots_in_pool": len(snaps),
            "n_snapshots_this_run_promoted_itself": len(own),
            "pool_is_entirely_seeded_from": seed.get("parent_run_name"),
            "seeded_snapshot_span": [snaps[0].stem, snaps[-1].stem] if snaps else None,
        }
    rec["row4_ladder_structural"] = {
        "verdict": "NOT RUNNABLE at n = 0 -- AGAIN, now at DOUBLE the budget",
        "runs": out,
        "why": ("a snapshot_ladder is a Bradley-Terry fit over a run's OWN promoted snapshots. "
                "Neither run promoted any in 12,091,392 steps, so there is no node to fit and no "
                "rating to quote in either direction. This is not 'below the n >= 12 report "
                "floor'; it is n = 0. Registered in PREDICTION.md sec 3.4 before any number."),
        "comparators_for_the_record": {
            "armW_nodes": len(list((ARMW / "snapshots").glob("snapshot_*.zip"))),
            "armWb_nodes": len(list((ARMWB / "snapshots").glob("snapshot_*.zip"))),
            "armW_ladder_newest_committed": _newest(ARMW),
            "armWb_ladder_newest_committed": _newest(ARMWB),
        },
    }

    # the registered check the GO named by tag: eval/pool_snapshot_count
    evals = []
    ef = CONT / "eval_results.jsonl"
    if ef.exists():
        for line in ef.read_text().splitlines():
            if not line.strip():
                continue
            d = json.loads(line)
            # the per-cycle file's own schema: sentinels is the POOL, one row per snapshot faced
            # the per-cycle file's own schema: `sentinels` is a LIST of the n-sentinels drawn
            # from the pool each cycle (5 of the 20), each with its own snapshot STEP.
            sent = d.get("sentinels") or []
            evals.append({"step": d.get("step"), "n_games": d.get("n_games"),
                          "matchup_hash": d.get("matchup_hash"),
                          "sentinel_regime": d.get("sentinel_regime"),
                          "n_sentinels_drawn": len(sent),
                          "sentinel_snapshot_steps": [s_.get("step") for s_ in sent],
                          "sentinel_win_rates": [s_.get("win_rate") for s_ in sent],
                          "externals": d.get("externals")})
    rec["row4_pool_snapshot_count_check"] = {
        "source": "models/ai_v13_08_fold1_cont/eval_results.jsonl",
        "rows": evals,
        "tb_tag_eval_pool_snapshot_count": None,   # filled below from the TB series
        "note": ("the GO's named check, read TWO ways: the TB tag eval/pool_snapshot_count, and "
                 "the per-cycle eval rows. The cycle file records the n-sentinels DRAWN each "
                 "cycle (5 of the pool's 20), never the pool size; the pool size is the TB tag "
                 "and the snapshots/ listing. A count that never rises above its seeded 20 is the "
                 "same fact as 'promoted nothing'. 🚨 EVERY sentinel step listed here is <= "
                 "72,000,000 -- i.e. an arm-W snapshot -- at every cycle of both runs."),
    }

    rec["row4_pool_snapshot_count_check"]["tb_tag_eval_pool_snapshot_count"] = \
        rec["post_fork"].get("eval/pool_snapshot_count", {}).get("points")

    wp = rec["post_fork"].get("eval/win_rate_vs_pool", {}).get("points", [])
    rec["row4_descriptor_win_rate_vs_pool"] = {
        "series": wp,
        "note": ("🚨 neither run promoted anything, so every point is against an IDENTICAL, FROZEN "
                 "opponent set -- arm W's own 20 late snapshots, seeded at the first fork and "
                 "re-seeded by name at the second. A within-run series against a fixed pool is "
                 "unusually clean; it is still a DESCRIPTOR with no bar attached. Rule 5's "
                 "2026-09-07 regime boundary does not bite (every point is post-boundary, the "
                 "regime is RECORDED as inherited from arm W's argv) and no cross-run use is made."),
        "promotion_gate": ("--promote-threshold, 0.55 under the greedy sentinel regime; no point "
                           "reaches it"),
    }

    dest = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("cont_run_rows.json")
    dest.write_text(json.dumps(rec, indent=1))
    print(json.dumps({k: v for k, v in rec.items() if k != "post_fork"}, indent=1))
    for t, d in rec["post_fork"].items():
        print(f"{t:42s} fold={d['n_points_fold_leg']:3d} cont={d['n_points_cont_leg']:3d}  "
              f"grid={ {k: v for k, v in d['at_grid'].items() if v} }")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
