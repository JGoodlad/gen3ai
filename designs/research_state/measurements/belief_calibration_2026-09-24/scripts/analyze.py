"""Analyse the belief-calibration rows (PREDICTION.md §4): per-arm, per-stratum endpoints for head and
Smogon prior, the memorisation gaps Δ, the head-over-prior advantage A, and the non-transferring
advantage D, each with a 95% PAIRED bootstrap CI over battle indices (2,000 resamples).

    python analyze.py <rows_dir[,rows_dir2]> <out_dir> [arm1,arm2,...]   (contrasts vs the FIRST arm)

Every metric is a ratio of sums over the resampled battles' slots (ECE via per-bin sums), so no
cross-battle correlation is ever computed. Battles are kept only if they finished OK in ALL arms.
"""
import glob
import json
import os
import sys

import numpy as np

ARMS = ("pool", "ladder", "procedural")
N_BOOT = 2000
BOOT_SEED = 20260924
N_BINS = 15
CW = 0.8

# family → list of (metric, head_num_col, prior_num_col, den_col or None (=count), lower_is_better)
# row layouts (run_arm.py):
#  species  [k, h_nll, h_corr, h_conf, h_cwset, h_cwm, p_nll, p_corr, p_conf, p_cwset, p_cwm]
#  moves_rev/moves_hid [k, n_true, h_bce, h_hits, p_bce, p_hits]
#  item/hp  [k, h_nll, h_acc, p_nll, p_acc]
#  spread   [k, h_natce, h_natacc, h_evmae, h_dermae, p_natce, p_natacc, p_evmae, p_dermae]
METRICS = {
    "species": [("nll", 1, 6, None, True), ("acc", 2, 7, None, False),
                ("cw_rate", 4, 9, None, True), ("cw_matched", 5, 10, None, True)],
    "moves_rev": [("bce", 2, 4, None, True), ("recall_hidden", 3, 5, 1, False)],
    "moves_hid": [("bce", 2, 4, None, True), ("recall4", 3, 5, 1, False)],
    "item": [("nll", 1, 3, None, True), ("acc", 2, 4, None, False)],
    "hp": [("nll", 1, 3, None, True), ("acc", 2, 4, None, False)],
    "spread": [("nature_ce", 1, 5, None, True), ("nature_acc", 2, 6, None, False),
               ("ev_mae", 3, 7, None, True), ("derived_mae", 4, 8, None, True)],
}
WIDTH = {"species": 11, "moves_rev": 6, "moves_hid": 6, "item": 5, "hp": 5, "spread": 9}
STRATA = {"species": [1, 2, 3, 4, 5], "moves_hid": [1, 2, 3, 4, 5]}
DEFAULT_STRATA = [1, 2, 3, 4, 5, 6]


def load(rows_dirs):
    recs = {}
    files = [f for d in rows_dirs.split(",") for f in sorted(glob.glob(os.path.join(d, "rows_w*.jsonl")))]
    for f in files:
        for ln in open(f):
            r = json.loads(ln)
            recs[(r["arm"], r["i"])] = r
    return recs


def main(rows_dir, out_dir, arms=None):
    global ARMS
    if arms:
        ARMS = tuple(arms.split(","))
    recs = {k: v for k, v in load(rows_dir).items() if k[0] in ARMS}
    idx_all = sorted({i for (_, i) in recs})
    status = {a: {s: 0 for s in ("ok", "timeout", "error", "unfinished", "missing")} for a in ARMS}
    keep = []
    for i in idx_all:
        ok = True
        for a in ARMS:
            r = recs.get((a, i))
            st = r["status"] if r else "missing"
            status[a][st] = status[a].get(st, 0) + 1
            ok &= (st == "ok")
        if ok:
            keep.append(i)
    nb = len(keep)
    rng = np.random.default_rng(BOOT_SEED)
    W = rng.multinomial(nb, np.full(nb, 1.0 / nb), size=N_BOOT).astype(np.float64)   # [B, nb]
    W = np.vstack([np.ones((1, nb)), W])                                               # row 0 = point est.

    def ci(x):
        return [float(x[0]), float(np.percentile(x[1:], 2.5)), float(np.percentile(x[1:], 97.5))]

    counts = {a: {"battles": nb, "decisions": int(sum(recs[(a, i)]["counters"]["decisions"] for i in keep)),
                  "turns_mean": float(np.mean([recs[(a, i)]["turns"] for i in keep])),
                  "trainee_win_rate": float(np.mean([recs[(a, i)]["winner"] == 1 for i in keep])),
                  "label_race_skipped": int(sum(recs[(a, i)]["counters"]["label_race_skipped"] for i in keep)),
                  "species_labelset_mismatch": int(sum(recs[(a, i)]["counters"]["species_labelset_mismatch"]
                                                       for i in keep)),
                  "hidden_move_maxdev": float(max(recs[(a, i)]["hidden_move_maxdev"] for i in keep))}
              for a in ARMS}
    out = {"arms": list(ARMS), "status": status, "n_paired_battles": nb, "counts": counts, "families": {}}

    for fam, mets in METRICS.items():
        strata = STRATA.get(fam, DEFAULT_STRATA)
        fam_out = {}
        # per arm: list over battles of np arrays of rows
        width = WIDTH[fam]
        rows = {a: [np.asarray(recs[(a, i)]["rows"][fam], dtype=np.float64).reshape(-1, width)
                    for i in keep] for a in ARMS}
        for st in ["all"] + strata:
            sel = (lambda R: R if st == "all" else R[R[:, 0] == st]) if True else None
            st_out = {"n_slots": {}}
            per = {}   # (arm, who, metric) -> bootstrap vector
            for a in ARMS:
                subs = [sel(R) if R.size else R for R in rows[a]]
                st_out["n_slots"][a] = int(sum(len(s) for s in subs))
                for (m, hc, pc, dc, _) in mets:
                    for who, col in (("head", hc), ("prior", pc)):
                        num = np.array([s[:, col].sum() if len(s) else 0.0 for s in subs])
                        den = np.array([(s[:, dc].sum() if dc is not None else len(s)) if len(s) else 0.0
                                        for s in subs])
                        with np.errstate(invalid="ignore", divide="ignore"):
                            per[(a, who, m)] = (W @ num) / (W @ den)
                if fam == "species":   # ECE from per-bin sums
                    for who, cc, kc in (("head", 3, 2), ("prior", 8, 7)):
                        bins_conf = np.zeros((nb, N_BINS))
                        bins_corr = np.zeros((nb, N_BINS))
                        for bi, s in enumerate(subs):
                            if not len(s):
                                continue
                            b = np.clip((s[:, cc] * N_BINS).astype(int), 0, N_BINS - 1)
                            np.add.at(bins_conf[bi], b, s[:, cc])
                            np.add.at(bins_corr[bi], b, s[:, kc])
                        tot = W @ np.array([len(s) for s in subs], dtype=np.float64)
                        per[(a, who, "ece")] = np.abs(W @ bins_conf - W @ bins_corr).sum(1) / tot
            mlist = list(mets) + ([("ece", None, None, None, True)] if fam == "species" else [])
            for (m, _, _, _, lower) in mlist:
                sgn = 1.0 if lower else -1.0          # A > 0 ⇔ head better than prior
                mo = {}
                for a in ARMS:
                    h, p = per[(a, "head", m)], per[(a, "prior", m)]
                    mo[a] = {"head": ci(h), "prior": ci(p), "A": ci(sgn * (p - h))}
                for a in ARMS[1:]:
                    h0, p0 = per[(ARMS[0], "head", m)], per[(ARMS[0], "prior", m)]
                    h1, p1 = per[(a, "head", m)], per[(a, "prior", m)]
                    A0, A1 = sgn * (p0 - h0), sgn * (p1 - h1)
                    mo[f"delta_{a}"] = ci(h1 - h0)                  # raw head gap vs pool
                    mo[f"D_{a}"] = ci(A0 - A1)                      # non-transferring advantage
                    with np.errstate(invalid="ignore", divide="ignore"):
                        mo[f"D_share_{a}"] = ci((A0 - A1) / A0)
                st_out[m] = mo
            fam_out[str(st)] = st_out
        out["families"][fam] = fam_out
    os.makedirs(out_dir, exist_ok=True)
    json.dump(out, open(os.path.join(out_dir, "results.json"), "w"), indent=1)
    print(json.dumps({"status": status, "n_paired_battles": nb, "counts": counts}, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
