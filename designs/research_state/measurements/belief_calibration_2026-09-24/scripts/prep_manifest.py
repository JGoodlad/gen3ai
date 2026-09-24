"""Build the paired battle manifest for the belief-calibration read (PREDICTION.md §2).

Run with the PINNED tree on PYTHONPATH and cwd = the pin worktree (its data/ == main's):

    python prep_manifest.py <work_dir> <n_battles> <out_dir>

<work_dir> holds prep_teams.js's outputs (ladder_candidates.json, procedural.json, filter_stats.json)
and pool.json. Every team is packed by the pinned `Gen3Teambuilder` (the same validation + HP-IV fix
training uses). Writes <work_dir>/manifest_full.json (packed teams — stays OUTSIDE the repo) and
<out_dir>/manifest.json (shas, counts, seeds — the committed record).
"""
import hashlib
import json
import os
import sys

import numpy as np

from utils.teambuilder import Gen3Teambuilder

SEED = 20260924


def sha(text: str) -> str:
    return hashlib.sha1(text.strip().encode()).hexdigest()[:10]


def packed_by_sha(texts):
    """Pack via the pinned Gen3Teambuilder; returns {sha: packed} for the teams it ACCEPTED."""
    tb = Gen3Teambuilder(texts)
    return dict(zip(tb._pool_keys, tb.packed_teams))


def main(work, n, out_dir):
    pool_texts = json.load(open(os.path.join(work, "pool.json")))
    ladder = json.load(open(os.path.join(work, "ladder_candidates.json")))
    proc = json.load(open(os.path.join(work, "procedural.json")))
    rng = np.random.default_rng(SEED)

    pool_packed = packed_by_sha(pool_texts)
    pool_shas = [sha(t) for t in pool_texts if sha(t) in pool_packed]

    # LADDER: seeded permutation of the filtered candidates; take the first n the pinned
    # teambuilder accepts (rejections counted).
    perm = rng.permutation(len(ladder))
    cand = [ladder[j] for j in perm[: n + 200]]
    lad_ok = packed_by_sha([c["text"] for c in cand])
    lad_sel, lad_tb_reject = [], 0
    for c in cand:
        s = sha(c["text"])
        if s in lad_ok:
            lad_sel.append((s, c["file"]))
        else:
            lad_tb_reject += 1
        if len(lad_sel) == n:
            break
    # PROCEDURAL: generator order, first n accepted.
    proc_ok = packed_by_sha([p["text"] for p in proc])
    proc_sel, proc_tb_reject = [], 0
    for p in proc:
        s = sha(p["text"])
        if s in proc_ok:
            proc_sel.append(s)
        else:
            proc_tb_reject += 1
        if len(proc_sel) == n:
            break
    assert len(lad_sel) == n and len(proc_sel) == n, (len(lad_sel), len(proc_sel))

    trainee = [pool_shas[int(j)] for j in rng.integers(0, len(pool_shas), size=n)]
    pool_opp = [pool_shas[int(j)] for j in rng.integers(0, len(pool_shas), size=n)]

    packed = dict(pool_packed)
    packed.update(lad_ok)
    packed.update(proc_ok)
    battles = []
    for i in range(n):
        battles.append({
            "i": i, "seed_base": SEED * 1000 + i, "trainee": trainee[i],
            "opp": {"pool": pool_opp[i], "ladder": lad_sel[i][0], "procedural": proc_sel[i]},
            "ladder_file": lad_sel[i][1],
        })
    full = {"battles": battles, "packed": {k: packed[k] for k in
                                          {b["trainee"] for b in battles}
                                          | {s for b in battles for s in b["opp"].values()}}}
    json.dump(full, open(os.path.join(work, "manifest_full.json"), "w"))
    stats = json.load(open(os.path.join(work, "filter_stats.json")))
    rec = {
        "seed": SEED, "n_battles_per_arm": n,
        "pool": {"n_teams": len(pool_texts), "packed_ok": len(pool_packed),
                 "coverage_predicates_info": stats["pool"]},
        "ladder": {"source": "metamon hl_05_26/gen3ou (teams revision v5)", "filter": stats["ladder"],
                   "candidates_after_filter": len(ladder),
                   "gen3teambuilder_rejects_in_sample_walk": lad_tb_reject},
        "procedural": {"generator": stats["procedural"], "gen3teambuilder_rejects": proc_tb_reject},
        "battles": [{"i": b["i"], "seed_base": b["seed_base"], "trainee": b["trainee"], **b["opp"],
                     "ladder_file": b["ladder_file"]} for b in battles],
    }
    os.makedirs(out_dir, exist_ok=True)
    json.dump(rec, open(os.path.join(out_dir, "manifest.json"), "w"), indent=0)
    print(json.dumps({k: v for k, v in rec.items() if k != "battles"}, indent=1)[:3000])


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), sys.argv[3])
