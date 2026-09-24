"""Build the paired battle manifest for the belief win-rate A/B (PREDICTION.md §2).

Run with the PINNED tree (cwd = the pin worktree, PYTHONPATH=<pin>/src):

    python prep_manifest.py <cal_work> <n> <work> <out_dir>

<cal_work> is the calibration read's prep dir (`/tmp/belief_cal/`: pool.json, ladder_candidates.json,
filter_stats.json — the calibration's registered Metamon filter, already applied). This script adds
the A/B's own filters — Heal Bell (the obs encoder crashes on the 'healbell' volatile) and Sleep Talk —
to BOTH team sources, packs through the pinned `Gen3Teambuilder`, and draws per battle index i:
our (trainee) team from the filtered pool, a POOL opponent from the filtered pool, and a LADDER
opponent (distinct Metamon teams, seeded permutation). Every draw is shared by all four cells of
index i. Writes <work>/manifest_full.json (packed teams — stays OUTSIDE the repo) and
<out_dir>/manifest.json (shas + seeds + filter counts — the committed record).
"""
import hashlib
import json
import os
import re
import sys

import numpy as np

from utils.teambuilder import Gen3Teambuilder

SEED = 20260925
EXCLUDE = ("Heal Bell", "Sleep Talk")


def sha(text: str) -> str:
    return hashlib.sha1(text.strip().encode()).hexdigest()[:10]


def has_move(text: str, move: str) -> bool:
    return re.search(r"^-\s*" + re.escape(move) + r"\s*$", text, re.M | re.I) is not None


def excluded_by(text: str):
    return [m for m in EXCLUDE if has_move(text, m)]


def packed_by_sha(texts):
    tb = Gen3Teambuilder(texts)
    return dict(zip(tb._pool_keys, tb.packed_teams))


def main(cal, n, work, out_dir):
    pool_texts = json.load(open(os.path.join(cal, "pool.json")))
    ladder = json.load(open(os.path.join(cal, "ladder_candidates.json")))
    cal_stats = json.load(open(os.path.join(cal, "filter_stats.json")))
    rng = np.random.default_rng(SEED)

    pool_counts = {m: sum(has_move(t, m) for t in pool_texts) for m in EXCLUDE}
    lad_counts = {m: sum(has_move(c["text"], m) for c in ladder) for m in EXCLUDE}
    pool_keep = [t for t in pool_texts if not excluded_by(t)]
    lad_keep = [c for c in ladder if not excluded_by(c["text"])]

    pool_packed = packed_by_sha(pool_keep)
    pool_shas = [sha(t) for t in pool_keep if sha(t) in pool_packed]

    perm = rng.permutation(len(lad_keep))
    cand = [lad_keep[j] for j in perm[: n + 300]]
    lad_ok = packed_by_sha([c["text"] for c in cand])
    lad_sel, lad_tb_reject = [], 0
    for c in cand:
        s = sha(c["text"])
        if s in lad_ok and s not in {x[0] for x in lad_sel}:
            lad_sel.append((s, c["file"]))
        else:
            lad_tb_reject += 1
        if len(lad_sel) == n:
            break
    assert len(lad_sel) == n, len(lad_sel)

    trainee = [pool_shas[int(j)] for j in rng.integers(0, len(pool_shas), size=n)]
    pool_opp = [pool_shas[int(j)] for j in rng.integers(0, len(pool_shas), size=n)]
    packed = dict(pool_packed)
    packed.update(lad_ok)
    battles = [{"i": i, "seed_base": SEED * 1000 + i, "trainee": trainee[i],
                "opp": {"pool": pool_opp[i], "ladder": lad_sel[i][0]}, "ladder_file": lad_sel[i][1]}
               for i in range(n)]
    used = {b["trainee"] for b in battles} | {s for b in battles for s in b["opp"].values()}
    os.makedirs(work, exist_ok=True)
    json.dump({"battles": battles, "packed": {k: packed[k] for k in used}},
              open(os.path.join(work, "manifest_full.json"), "w"))
    rec = {
        "seed": SEED, "n_indices": n,
        "pool": {"n_teams": len(pool_texts), "excluded_counts": pool_counts,
                 "kept_after_exclusion": len(pool_keep), "packed_ok": len(pool_packed),
                 "used_as": "trainee team AND pool-column opponent (uniform, with replacement)"},
        "ladder": {"source": "metamon hl_05_26/gen3ou (teams revision v5), the calibration read's filter",
                   "calibration_filter": cal_stats["ladder"],
                   "candidates_after_calibration_filter": len(ladder),
                   "excluded_counts": lad_counts, "kept_after_exclusion": len(lad_keep),
                   "share_of_download_kept": len(lad_keep) / cal_stats["ladder"]["total"],
                   "gen3teambuilder_rejects_or_dups_in_sample_walk": lad_tb_reject,
                   "used_as": "ladder-column opponent (distinct teams, seeded permutation)"},
        "battles": [{"i": b["i"], "seed_base": b["seed_base"], "trainee": b["trainee"], **b["opp"],
                     "ladder_file": b["ladder_file"]} for b in battles],
    }
    os.makedirs(out_dir, exist_ok=True)
    json.dump(rec, open(os.path.join(out_dir, "manifest.json"), "w"), indent=0)
    print(json.dumps({k: v for k, v in rec.items() if k != "battles"}, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4])
