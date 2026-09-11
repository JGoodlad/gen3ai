"""THE FRAME CHECK — is the opponent OBSERVABLE in the window a meter is read in?

This exists because of what it found. The probe read established that on turn-1 states
`value_pooled` decodes the opponent's CLASS at AUC 0.846 (arm A) / 0.861 (CTRL) while `V` sits at
chance, and concluded that "the head is handed the answer and does not use it". On the stationary
frame this measurement builds, the SAME decode from the SAME tensor reads **0.506** — chance.

One line explains the whole difference, and it is not about the head:

  in the probe read's frames the TRAINEE'S OWN TEAM predicts the opponent's class at AUC
  0.856 (CTRL) / 0.877 (A), because SENTINEL battles there drew only 37 of 180 (CTRL) and
  47 of 216 (A) distinct trainee teams while BOT battles drew 169 / 202.

The trainee's own team is in the observation verbatim at turn 1. So a decoder asked "is this a
sentinel?" at turn 1 could answer it by reading OUR team — an artefact of how that eval assigned
teams, not information about the opponent. On the offline `eval_trace_gen` frames every one of the
602 teams is played against every opponent, the own-team channel carries nothing about the class
(AUC 0.508), and `value_pooled` carries nothing either.

WHAT THIS DOES AND DOES NOT OVERTURN. It does NOT touch the head refit's central result, which is
a statement about two TARGETS scored on the same held-out outcome. It DOES change what the turn-1
opponent meters mean: on a matched-team frame the opponent is genuinely unobservable at turn 1
(Gen 3 has no team preview), so a critic that emits the marginal there is BAYES-OPTIMAL and a
turn-1 spread ratio of 0 is not a defect. The opponent becomes observable as it plays —
`value_pooled` decodes its class at 0.675 by turns 1-3 and 0.819 by turns 4-10 — and those are the
windows in which "does the head condition on the opponent" is a question with an answer.

Run over any pair of extraction dirs; writes one JSON. CPU only, seconds.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "winprob_probe_read_2026-09-09"))
import decode as DEC        # noqa: E402


def loo_group_rate(code, y, n_groups):
    """Leave-one-battle-out rate of the battle's group — the honest 'can this column alone
    predict the label' statistic (an in-sample group mean would score ~1.0 by construction)."""
    s = np.bincount(code, weights=y, minlength=n_groups)
    n = np.bincount(code, minlength=n_groups).astype(float)
    den = n[code] - 1.0
    return np.where(den > 0, (s[code] - y) / np.where(den > 0, den, 1.0), y.mean())


def check(d, n_battles_cap, seed):
    meta = np.load(f"{d}/meta.npy")
    P = np.load(f"{d}/pooled.npy")
    out = {"dir": d, "n_states": int(len(meta))}

    # (1) the OWN-TEAM -> OPPONENT-CLASS leak, at battle level
    t1 = np.where(meta["turn"] == 1)[0]
    _b, first = np.unique(meta["battle"][t1], return_index=True)
    sel = t1[first]
    cls = (meta["opp_class"][sel] == "sentinel").astype(float)
    team = meta["team"][sel]
    tu, ti = np.unique(team, return_inverse=True)
    auc = DEC._w_auc(cls, loo_group_rate(ti, cls, len(tu)), np.ones(len(sel)))
    out["own_team_leak"] = {
        "battles": int(len(sel)), "teams": int(len(tu)),
        "median_battles_per_team": float(np.median(np.bincount(ti))),
        "sentinel_fraction": round(float(cls.mean()), 4),
        "teams_seen_vs_bots": int(len(np.unique(team[cls == 0]))),
        "teams_seen_vs_sentinels": int(len(np.unique(team[cls == 1]))),
        "own_team_LOO_to_opp_class_AUC": round(float(auc), 4)}

    # (2) is the opponent OBSERVABLE, per turn bucket, from the tensor the head reads?
    rng = np.random.default_rng(seed)
    ub = np.unique(meta["battle"])
    keep = set(rng.permutation(ub)[:n_battles_cap].tolist())
    inb = np.isin(meta["battle"], np.array(sorted(keep)))
    y_all = (meta["opp_class"] == "sentinel").astype(float)
    tg, _x, _y, _z = DEC.build_targets(meta)
    out["observability"] = {}
    for bname, (lo, hi) in DEC.BUCKETS.items():
        m = inb & (meta["turn"] >= lo) & (meta["turn"] <= hi)
        idx = DEC.cap_per_battle(meta, m, 2, seed)
        if len(idx) < 200 or y_all[idx].min() == y_all[idx].max():
            continue
        w = meta["w"][idx].astype(float)
        g = meta["battle"][idx]
        y = y_all[idx]
        a_p = DEC._w_auc(y, DEC.GroupedRidgeCV(P[idx].astype(np.float64), w, g,
                                               seed=seed).oof(y, w, "auc")[0], w)
        V = meta["V_fwd"][idx].astype(float)
        a_v = DEC._w_auc(y, DEC.GroupedRidgeCV(V[:, None], w, g,
                                               seed=seed).oof(y, w, "auc")[0], w)
        t = tg["own_team_wr"]
        mm = m & t["mask"]
        it = DEC.cap_per_battle(meta, mm, 2, seed)
        yt = t["y"][it]
        wt = meta["w"][it].astype(float)
        gt = meta["battle"][it]
        r_p = DEC._w_r2(yt, DEC.GroupedRidgeCV(P[it].astype(np.float64), wt, gt,
                                               seed=seed).oof(yt, wt, "r2")[0], wt)
        r_v = DEC._w_r2(yt, DEC.GroupedRidgeCV(meta["V_fwd"][it].astype(float)[:, None], wt, gt,
                                               seed=seed).oof(yt, wt, "r2")[0], wt)
        out["observability"][bname] = {
            "n_states": int(len(idx)),
            "pooled_to_opp_class_AUC": round(float(a_p), 4),
            "V_to_opp_class_AUC": round(float(a_v), 4),
            "pooled_to_own_team_wr_R2": round(float(r_p), 4),
            "V_to_own_team_wr_R2": round(float(r_v), 4)}
        print(f"  [{os.path.basename(d)}] {bname}: pooled->class {a_p:.3f} V->class {a_v:.3f} "
              f"pooled->teamWR {r_p:.3f} V->teamWR {r_v:.3f}", flush=True)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", action="append", required=True, help="name=/path (repeat)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cap", type=int, default=4000, help="battles sampled for the decode rows")
    ap.add_argument("--seed", type=int, default=20260910)
    a = ap.parse_args()
    res = {}
    for spec in a.dir:
        name, d = spec.split("=", 1)
        print(f"== {name}", flush=True)
        res[name] = check(d, a.cap, a.seed)
    json.dump(res, open(a.out, "w"), indent=1)
    print("wrote", a.out)
    for n, v in res.items():
        L = v["own_team_leak"]
        print(f"{n:22s} own-team -> opp_class AUC {L['own_team_LOO_to_opp_class_AUC']:.3f} "
              f"({L['teams_seen_vs_sentinels']}/{L['teams']} teams ever face a sentinel)")
