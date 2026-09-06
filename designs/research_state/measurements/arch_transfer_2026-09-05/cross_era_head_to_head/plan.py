"""Generate the paired, side-swapped game plan for the cross-era head-to-head.

The plan is generated ONCE, written to JSON, and READ by both sides. Neither side
re-derives it, so the two eras cannot consume a shared random stream in a
scheduling-dependent order (the project's determinism rule: seeds alone are not enough
when two consumers interleave).

A pair is an ordered draw of two DISTINCT teams (T_a, T_b) from the verified
intersection of the two eras' team pools. Each pair is played twice:

    orientation 0:  side_a plays T_a,  side_b plays T_b
    orientation 1:  side_a plays T_b,  side_b plays T_a

so the team draw differences out within a pair.

Run:
    python designs/research_state/measurements/arch_transfer_2026-09-05/cross_era_head_to_head/plan.py \
        --pairs 280 --out <dir>/plan.json
    (in a linked worktree, first: export PYTHONPATH=$PYTHONPATH:src)
"""

import argparse
import glob
import hashlib
import json
import os
import random

# The two pools were verified byte-identical over their intersection; the era's
# sample/ set is a strict subset of the current tree's, so the ERA's directory IS
# the intersection. Reading it from the era side guarantees we never draw a team the
# era cannot load.
ERA_TEAM_DIR = "/tmp/v8rep_era/data/teams/sample"

TEAM_SEQ_SEED = 20260905


def load_teams(team_dir: str):
    """Every team in `team_dir`, as (name, sha256, text), sorted by filename.

    Sorted so the pool ORDER is a function of the directory contents alone — the
    'same pool SIZE is not the same pool ORDER' trap the teambuilder's own docstring
    warns about.
    """
    out = []
    for path in sorted(glob.glob(os.path.join(team_dir, "*.txt"))):
        text = open(path).read()
        out.append({
            "name": os.path.basename(path),
            "sha256": hashlib.sha256(text.encode()).hexdigest(),
            "text": text,
        })
    if not out:
        raise SystemExit(f"no teams found in {team_dir}")
    return out


def build_plan(teams, n_pairs: int, seed: int):
    rng = random.Random(seed)
    n = len(teams)
    games = []
    pairs = []
    for p in range(n_pairs):
        i = rng.randrange(n)
        j = rng.randrange(n - 1)
        if j >= i:            # draw a DISTINCT second team without rejection looping
            j += 1
        pairs.append((i, j))
        for orientation in (0, 1):
            a_idx, b_idx = (i, j) if orientation == 0 else (j, i)
            games.append({
                "game_index": len(games),
                "pair_index": p,
                "orientation": orientation,
                "side_a_team": teams[a_idx]["name"],
                "side_a_sha": teams[a_idx]["sha256"],
                "side_b_team": teams[b_idx]["name"],
                "side_b_sha": teams[b_idx]["sha256"],
            })
    return pairs, games


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--team-dir", default=ERA_TEAM_DIR)
    ap.add_argument("--pairs", type=int, default=280)
    ap.add_argument("--seed", type=int, default=TEAM_SEQ_SEED)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    teams = load_teams(args.team_dir)
    pairs, games = build_plan(teams, args.pairs, args.seed)

    plan = {
        "schema": 1,
        "team_dir": args.team_dir,
        "team_seq_seed": args.seed,
        "n_teams": len(teams),
        "n_pairs": args.pairs,
        "n_games": len(games),
        # side_a / side_b are ROLES in the plan; which era takes which is a runtime
        # argument, so the same plan file can be reused for a bot-calibration arm.
        "teams": [{"name": t["name"], "sha256": t["sha256"]} for t in teams],
        "games": games,
    }
    with open(args.out, "w") as f:
        json.dump(plan, f, indent=1)
    print(f"[plan] {len(teams)} teams, {args.pairs} pairs, {len(games)} games -> {args.out}")
    print(f"[plan] seed={args.seed}")
    counts = {}
    for g in games:
        counts[g["side_a_team"]] = counts.get(g["side_a_team"], 0) + 1
    print(f"[plan] distinct teams used as side_a: {len(counts)}; "
          f"min/max appearances {min(counts.values())}/{max(counts.values())}")


if __name__ == "__main__":
    main()
