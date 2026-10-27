"""V-RANK power probe (design_v_rank_on_u.md §5.1): the within-node signal of the search's Q-hat on P*'s plateau bank.

SELF-REFEREE PROXY, used only to size the read. The plateau bank (`~/gen3ai_archive/hidden_info_maturity/plateau/`,
read only, through L1's registered loader `../../search_l1_plateau_2026-10-25/l1_plateau.py`) holds, for 719 states of
P*'s own pool-team games, every legal action played to the END in the TRUE world by P* on both sides, 40 seeds
(seeds 16..39 are the referee, common dice across the actions of a seed), and the honest F worlds' TRUNC-4-K8 leaf
values (the search's Q-hat). One referee seed per action, the same seed index across a state's actions, is the shape
of one live branch per action (common dice: the branches share the battle's seed).

    PYTHONPATH=<main>/src python power_probe.py   # pure numpy over the archive, < 2 min, < 2 GB

Output: `../power_probe.json`.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
MEAS = HERE.parent.parent
sys.path.insert(0, str(MEAS / "search_l1_plateau_2026-10-25"))
import l1_plateau as P  # noqa: E402

PSTAR = "x26_P280M"
REF = range(16, 40)          # the referee seeds (t1_data: truth seeds 16..39)
B = 2000
RNG = 20261027


def w(x):
    return (np.asarray(x, dtype=np.float64) + 1.0) / 2.0


def within(x):
    return x - x.mean()


def slope_parts(q, y):
    qc = within(q)
    return float(qc @ within(y)), float(qc @ qc)


def partial_parts(q, lg, y):
    """Within-node partial slope of y on q given the logits: the residual of q on lg (within node), against y."""
    qc, lc = within(q), within(lg)
    if lc @ lc < 1e-12:
        r = qc
    else:
        r = qc - (qc @ lc) / (lc @ lc) * lc
    return float(r @ within(y)), float(r @ r)


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra, rb = within(ra), within(rb)
    d = np.sqrt((ra @ ra) * (rb @ rb))
    return float(ra @ rb / d) if d > 0 else np.nan


def main() -> None:
    log: dict = {}
    recs, _ = P.load(log)
    recs = [r for r in recs if r["cp"] == PSTAR]
    rows = []
    for r in recs:
        q = np.asarray(r["F"]["X4"], dtype=np.float64).mean(1)        # the search's Q-hat (TRUNC-4-K8, F worlds)
        lg = np.asarray(r["logit"], dtype=np.float64)
        Ob = w(r["Ob40"])                                                 # [A, 40] in {0, .5, 1}
        Y = Ob[:, list(REF)]                                              # [A, 24]
        tp = np.asarray(r["Tp"], dtype=np.float64)
        rows.append({"q": q, "lg": lg, "Y": Y, "tp": tp, "turn": r["turn"], "k": len(q), "battle": r["battle"]})
    n = len(rows)
    k = np.array([x["k"] for x in rows])
    # spreads
    sd_q = np.array([within(x["q"]).std() for x in rows])
    sd_tp = np.array([within(x["tp"]).std() for x in rows])
    # noise of Tp per action (24 seeds) and the within-node residual of ONE seed (common dice)
    noise_tp = np.array([float(np.mean(x["Y"].var(1, ddof=1)) / 24.0) for x in rows])
    true_sd2 = np.maximum(sd_tp ** 2 - noise_tp * (k - 1) / k, 0.0)
    # slopes: Tp on q (pooled within node), single-seed y on q, partial on q given logits
    num_t = np.array([slope_parts(x["q"], x["tp"])[0] for x in rows])
    den = np.array([slope_parts(x["q"], x["tp"])[1] for x in rows])
    pnum_t = np.array([partial_parts(x["q"], x["lg"], x["tp"])[0] for x in rows])
    pden = np.array([partial_parts(x["q"], x["lg"], x["tp"])[1] for x in rows])
    lnum_t = np.array([slope_parts(x["lg"], x["tp"])[0] for x in rows])
    lden = np.array([slope_parts(x["lg"], x["tp"])[1] for x in rows])
    rng = np.random.default_rng(RNG)
    seeds = rng.integers(0, 24, size=n)                                   # ONE referee seed per state
    num_1 = np.array([slope_parts(x["q"], x["Y"][:, s])[0] for x, s in zip(rows, seeds)])
    pnum_1 = np.array([partial_parts(x["q"], x["lg"], x["Y"][:, s])[0] for x, s in zip(rows, seeds)])
    sp = np.array([spearman(x["q"], x["tp"]) for x in rows])
    sp_l = np.array([spearman(x["lg"], x["tp"]) for x in rows])
    # the piKL pick's IMP with one seed (the decision-level alternative), for comparison
    imp_1 = []
    for x, s in zip(rows, seeds):
        lsm = x["lg"] - np.log(np.exp(x["lg"] - x["lg"].max()).sum()) - x["lg"].max()
        pick = int(np.argmax(x["q"] + 0.015 * lsm))
        top = int(np.argmax(x["lg"]))
        imp_1.append(float(x["Y"][pick, s] - x["Y"][top, s]))
    imp_1 = np.array(imp_1)

    def boot(num, den_):
        m = len(num)
        idx = rng.integers(0, m, size=(B, m))
        b = num[idx].sum(1) / den_[idx].sum(1)
        return [float(num.sum() / den_.sum()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5)),
                float(b.std())]

    t10 = np.array([x["turn"] >= 10 for x in rows])
    t1030 = np.array([10 <= x["turn"] <= 30 for x in rows])
    out = {
        "source": "plateau bank, P* anchor, SELF-REFEREE (P* both sides in the TRUE world), L1's loader",
        "n_states": n, "n_turn_ge_10": int(t10.sum()), "k_mean": float(k.mean()),
        "k_hist": {int(a): int((k == a).sum()) for a in np.unique(k)},
        "within_sd_qhat_mean": float(sd_q.mean()), "within_sd_qhat_median": float(np.median(sd_q)),
        "within_sd_true_noise_corrected_rms": float(np.sqrt(true_sd2.mean())),
        "within_sd_tp24_raw_rms": float(np.sqrt((sd_tp ** 2).mean())),
        "sum_qc2_per_state_mean": float(den.mean()),
        "turn_10_30": {"n": int(t1030.sum()), "sum_qc2_mean": float(den[t1030].mean()),
                       "beta_y1_on_q": boot(num_1[t1030], den[t1030]), "beta_tp_on_q": boot(num_t[t1030], den[t1030])},
        "beta_tp_on_q [est, lo, hi, se]": boot(num_t, den),
        "beta_y1_on_q [est, lo, hi, se] (ONE seed per action)": boot(num_1, den),
        "beta_partial_tp_on_q_given_logits": boot(pnum_t, pden),
        "beta_partial_y1_on_q_given_logits": boot(pnum_1, pden),
        "beta_tp_on_logits": boot(lnum_t, lden),
        "spearman_q_tp_mean": float(np.nanmean(sp)), "spearman_logits_tp_mean": float(np.nanmean(sp_l)),
        "imp_pikl_one_seed_mean": float(imp_1.mean()), "imp_pikl_one_seed_sd": float(imp_1.std()),
        "imp_pikl_deviation_share": float(np.mean(imp_1 != 0)),
        "turn_ge_10": {"beta_y1_on_q": boot(num_1[t10], den[t10]) if t10.sum() > 50 else None,
                       "beta_tp_on_q": boot(num_t[t10], den[t10]) if t10.sum() > 50 else None},
        "loader_log": log,
    }
    # THE CONTENDER SET C (design §4.4): pi's top 3 by logit, plus Q-hat's top 2 (the actions a one-ply search can
    # pick at piKL's small lambda); the within-node slope inside C, and the top-anchored contrast
    def contenders(x):
        o = list(np.argsort(-x["lg"], kind="stable")[:3]) + list(np.argsort(-x["q"], kind="stable")[:2])
        return sorted(set(int(i) for i in o))

    cn_t, cn_1, cd, ck = [], [], [], []
    ta_d, ta_t, ta_1, ta_state = [], [], [], []
    for i, (x, s) in enumerate(zip(rows, seeds)):
        c = contenders(x)
        ck.append(len(c))
        q, tp, y1 = x["q"][c], x["tp"][c], x["Y"][c, s]
        a, b = slope_parts(q, tp)
        cn_t.append(a); cd.append(b); cn_1.append(slope_parts(q, y1)[0])
        top = int(np.argmax(x["lg"]))
        for j in range(x["k"]):
            if j != top:
                ta_d.append(x["q"][j] - x["q"][top]); ta_t.append(x["tp"][j] - x["tp"][top])
                ta_1.append(x["Y"][j, s] - x["Y"][top, s]); ta_state.append(i)
    cn_t, cn_1, cd, ck = map(np.array, (cn_t, cn_1, cd, ck))
    out["contender_k_mean"] = float(ck.mean())
    out["contender_sum_qc2_mean"] = float(cd.mean())
    out["beta_C_tp_on_q"] = boot(cn_t, cd)
    out["beta_C_y1_on_q (ONE seed)"] = boot(cn_1, cd)
    ta_d, ta_t, ta_1, ta_state = map(np.array, (ta_d, ta_t, ta_1, ta_state))
    X = np.c_[np.ones_like(ta_d), ta_d]
    out["top_anchored_tp [alpha, beta]"] = [float(v) for v in np.linalg.lstsq(X, ta_t, rcond=None)[0]]
    out["top_anchored_y1 [alpha, beta]"] = [float(v) for v in np.linalg.lstsq(X, ta_1, rcond=None)[0]]
    pos = ta_d > 0
    out["top_anchored_tp_mean_true_diff_where_qhat_prefers_alt"] = float(ta_t[pos].mean())
    out["top_anchored_share_alt_preferred"] = float(pos.mean())
    for lo, hi in ((0, 0.02), (0.02, 0.05), (0.05, 1.0)):
        m = (ta_d > lo) & (ta_d <= hi)
        out[f"top_anchored_tp_mean_true_diff_qhat_gap_{lo}_{hi}"] = [float(ta_t[m].mean()) if m.any() else None,
                                                                     int(m.sum())]
    # H: per state, the mean TRUE advantage over pi's top of the contenders Q-hat prefers to it (one seed and Tp);
    # the search-facing estimand (design §4.4)
    h1, ht, hs = [], [], []
    for i, (x, s) in enumerate(zip(rows, seeds)):
        top = int(np.argmax(x["lg"]))
        pref = [j for j in contenders(x) if j != top and x["q"][j] > x["q"][top]]
        if pref:
            h1.append(float(np.mean([x["Y"][j, s] - x["Y"][top, s] for j in pref])))
            ht.append(float(np.mean([x["tp"][j] - x["tp"][top] for j in pref])))
            hs.append(i)
    h1, ht = np.array(h1), np.array(ht)
    out["H_states_with_a_preferred_contender"] = [len(h1), float(len(h1) / n)]
    out["H_tp_mean"] = boot(ht, np.ones_like(ht))
    out["H_y1_mean"] = boot(h1, np.ones_like(h1))
    out["H_y1_sd_per_state"] = float(h1.std())
    out["se_H_one_seed_at_n_states"] = {str(m): float(h1.std() / np.sqrt(m * len(h1) / n))
                                         for m in (500, 1000, 2000, 4000)}
    # clustering: the same one-seed slope bootstrapped over BATTLES (several states of one game), the inflation
    bat = np.array([x["battle"] for x in rows])
    ub, inv = np.unique(bat, return_inverse=True)
    cnum, cden = np.bincount(inv, num_1), np.bincount(inv, den)
    cb = rng.integers(0, len(ub), size=(B, len(ub)))
    se_cl = float((cnum[cb].sum(1) / cden[cb].sum(1)).std())
    out["battles"] = int(len(ub))
    out["beta_y1_on_q_se_battle_cluster"] = se_cl
    out["cluster_inflation"] = se_cl / out["beta_y1_on_q [est, lo, hi, se] (ONE seed per action)"][3]
    # the piKL deviations' Q-hat gap over pi's top (where a slope bar bites, design §4.2)
    gaps = []
    for x in rows:
        lsm = x["lg"] - np.log(np.exp(x["lg"] - x["lg"].max()).sum()) - x["lg"].max()
        pick, top = int(np.argmax(x["q"] + 0.015 * lsm)), int(np.argmax(x["lg"]))
        if pick != top:
            gaps.append(float(x["q"][pick] - x["q"][top]))
    gaps = np.array(gaps)
    out["pikl_deviation_qhat_gap"] = {"n": int(len(gaps)), "median": float(np.median(gaps)),
                                      "p25": float(np.percentile(gaps, 25)), "p75": float(np.percentile(gaps, 75)),
                                      "share_gt_0.064": float(np.mean(gaps > 0.064))}
    # the power table (normal approximation; design §5.2): SE(beta) at n states a set, scaled from this bank's
    # one-seed SE with the battle-cluster inflation; bars as in the design
    from math import erf, sqrt

    def Phi(z):
        return 0.5 * (1 + erf(z / sqrt(2)))

    se0 = se_cl
    pw = {}
    for m in (600, 800, 1000, 1200, 1500, 2000):
        se = se0 * np.sqrt(n / m)
        sd = np.sqrt(2) * se
        pw[str(m)] = {"se_beta": round(float(se), 4), "se_delta": round(float(sd), 4),
                      "P(NULL | beta_U=0)": round(Phi(0.25 / se - 1.96), 3),
                      "P(NULL | beta_U=0.10)": round(Phi(0.15 / se - 1.96), 3),
                      "P(NULL | beta_U=0.54) false kill": round(Phi(-0.29 / se - 1.96), 4),
                      "P(WORSE | delta=-0.27)": round(Phi(0.27 / sd - 1.96), 3),
                      "P(WORSE | delta=0) false": 0.025,
                      "P(HOLDS | delta=0)": round(Phi(0.25 / sd - 1.96), 3),
                      "P(HOLDS | delta=-0.27) false": round(Phi(-0.02 / sd - 1.96), 3)}
    out["power"] = pw
    # ACTION-SUBSET EFFICIENCY (design §4.1): the one-seed slope's SE per unit of CPU when a state branches ALL its
    # legal actions, against THREE (pi's top, Q-hat's top or its 2nd when equal, one other drawn by a fixed rng);
    # a state's cost in branch-game units = its forced branches + OVERHEAD (shadow search + oracle + raw share)
    OVERHEAD = 2.7
    sub_num, sub_den, nb_all, nb_sub = [], [], [], []
    for x, s_ in zip(rows, seeds):
        top = int(np.argmax(x["lg"]))
        oq = [int(i) for i in np.argsort(-x["q"], kind="stable")]
        qt = oq[0] if oq[0] != top else oq[1]
        rest = [j for j in range(x["k"]) if j not in (top, qt)]
        pick = [top, qt] + ([int(rest[rng.integers(0, len(rest))])] if rest else [])
        a, b = slope_parts(x["q"][pick], x["Y"][pick, s_])
        sub_num.append(a); sub_den.append(b)
        nb_all.append(x["k"] - 1 + OVERHEAD); nb_sub.append(len(pick) - 1 + OVERHEAD)
    sub = boot(np.array(sub_num), np.array(sub_den))
    se_all = out["beta_y1_on_q [est, lo, hi, se] (ONE seed per action)"][3]
    out["subset3_beta_y1"] = sub
    out["se_x_sqrt_cost_all"] = float(se_all * np.sqrt(np.mean(nb_all)))
    out["se_x_sqrt_cost_subset3"] = float(sub[3] * np.sqrt(np.mean(nb_sub)))
    # SE of the ONE-seed slope scaled to n states (SE ~ 1/sqrt(n)); at n states a set, the U - home difference
    se1 = out["beta_y1_on_q [est, lo, hi, se] (ONE seed per action)"][3]
    out["se_beta_one_seed_at_n"] = {str(m): float(se1 * np.sqrt(n / m)) for m in (300, 500, 800, 1000, 1500, 2000)}
    out["se_delta_beta_at_n_per_set"] = {m: float(np.sqrt(2) * v) for m, v in out["se_beta_one_seed_at_n"].items()}
    (HERE.parent / "power_probe.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps({k_: v for k_, v in out.items() if k_ != "loader_log"}, indent=1, default=float))


if __name__ == "__main__":
    main()
