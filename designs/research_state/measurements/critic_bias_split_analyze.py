"""PROBE G — decompose the critic's leaf error into SHARED (per-decision offset) and
DIFFERENTIAL (ranking) components, with an empirically-measured label noise floor.

Per decision d over its A_d legal actions:
    e_d[a]      = C_d[a] - L_d[a]                  (critic - Monte-Carlo label, win-prob units)
    offset_d    = mean_a e_d[a]                    the SHARED component -- paired evaluation
                                                   cancels this exactly
    resid_d[a]  = e_d[a] - offset_d                the DIFFERENTIAL component -- pairing does
                                                   NOT cancel this; it is what mis-ranks actions
and the identity  mean_a e^2 == offset^2 + mean_a resid^2  holds exactly, per decision.

THE NOISE FLOOR IS MEASURED, NOT ASSUMED. Every label was rolled as two independent CRN blocks
A and B of R/2 dice each (L = (L_A+L_B)/2). With D = L_A - L_B:
    E[D[a]^2] / 4                      = the sampling variance of the full-R label L[a]
    E[mean_a (D - mean_a D)^2] / 4     = that variance IN RESIDUAL SPACE -- i.e. after whatever
                                         variance the common-random-numbers coupling already
                                         removed, which no closed-form binomial floor can know
    E[(mean_a D)^2] / 4                = that variance in OFFSET space
Subtracting these is what turns "MSE we measured" into "MSE the critic actually has".

Run:
  python tmp/probe_g_analyze.py tmp/probe_g_out/*.jsonl --out designs/research_state/measurements
  (in a linked worktree, first: export PYTHONPATH=$PYTHONPATH:src)
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import math
import os
import sys

import numpy as np

Z = 1.96


# ---------------------------------------------------------------------------
# Loading — one decision becomes one aligned (C, L_A, L_B) triple over legal actions
# ---------------------------------------------------------------------------

def load(paths, *, head="win_prob"):
    rows, dropped = [], collections.Counter()
    for p in paths:
        for ln in open(p):
            try:
                r = json.loads(ln)
            except ValueError:
                dropped["bad_json"] += 1
                continue
            if "error" in r:
                dropped[r["error"].split(":")[0]] += 1
                continue
            crit = {c["action"]: c for c in r["critic"]}
            lab = {l["action"]: l for l in r["labels"]}
            acts, C, LA, LB, term = [], [], [], [], []
            for a in sorted(set(crit) & set(lab)):
                c, l = crit[a], lab[a]
                if "error" in l:
                    dropped["label_error"] += 1
                    continue
                cv = c[head]
                if cv is None:
                    # A candidate whose CRN turn ENDED the battle has no successor state to read:
                    # a one-ply search sees the terminal and scores it exactly, so that is what we
                    # give the critic here. Anything else is a genuinely missing cell.
                    if c.get("terminal") == "win":
                        cv = 1.0
                    elif c.get("terminal") == "loss":
                        cv = 0.0
                    else:
                        dropped["critic_no_value"] += 1
                        continue
                if l.get("n_A", 0) < 1 or l.get("n_B", 0) < 1:
                    dropped["no_split_half"] += 1
                    continue
                acts.append(a); C.append(float(cv)); term.append(c.get("terminal"))
                LA.append(l["wins_A"] / l["n_A"]); LB.append(l["wins_B"] / l["n_B"])
            if len(acts) < 3:
                dropped["lt3_usable_actions"] += 1
                continue
            rows.append({
                "battle": r["battle"], "short": r["short"], "inv": r["inv"], "turn": r["turn"],
                "opponent": r["opponent"], "opp_class": r["opp_class"], "outcome": r["outcome"],
                "stratum": r["stratum"], "n_legal": r["n_legal"], "n_used": len(acts),
                "td_delta": r.get("td_delta"), "rec_win_prob": r.get("win_prob"),
                "chosen": r["action"], "R": r["rollouts"],
                "actions": acts, "C": np.array(C), "LA": np.array(LA), "LB": np.array(LB),
                "terminal": term,
            })
    return rows, dropped


# ---------------------------------------------------------------------------
# Cluster bootstrap over BATTLES (decisions inside one battle are not independent)
# ---------------------------------------------------------------------------

def boot_ci(per_decision_values, battles, *, draws=4000, seed=11, stat=np.mean):
    v = np.asarray(per_decision_values, dtype=float)
    ok = np.isfinite(v)
    v, b = v[ok], np.asarray(battles)[ok]
    if len(v) < 3:
        return None, None, None
    uniq = sorted(set(b.tolist()))
    idx = {u: np.flatnonzero(b == u) for u in uniq}
    rng = np.random.default_rng(seed)
    out = np.empty(draws)
    for k in range(draws):
        pick = rng.integers(0, len(uniq), len(uniq))
        sel = np.concatenate([idx[uniq[i]] for i in pick])
        out[k] = stat(v[sel])
    return float(stat(v)), float(np.quantile(out, 0.025)), float(np.quantile(out, 0.975))


def _spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3:
        return None
    def rk(a):
        o = a.argsort(kind="mergesort"); r = np.empty(len(a)); r[o] = np.arange(len(a), dtype=float)
        # average ties
        s = np.sort(a); i = 0
        while i < len(s):
            j = i
            while j + 1 < len(s) and s[j + 1] == s[i]:
                j += 1
            if j > i:
                r[np.isin(a, s[i])] = (i + j) / 2.0
            i = j + 1
        return r
    rx, ry = rk(x), rk(y)
    if rx.std() == 0 or ry.std() == 0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


# ---------------------------------------------------------------------------
# Per-decision statistics
# ---------------------------------------------------------------------------

def per_decision(rows):
    for r in rows:
        C, LA, LB = r["C"], r["LA"], r["LB"]
        L = 0.5 * (LA + LB)
        D = LA - LB
        r["L"] = L
        e = C - L
        off = float(e.mean())
        resid = e - off
        r["offset"] = off
        r["mse_e"] = float((e ** 2).mean())
        r["mse_offset"] = off ** 2
        r["mse_resid"] = float((resid ** 2).mean())
        # measured noise floors (variance of the FULL-R label, = D-variance / 4)
        r["noise_mse_label"] = float((D ** 2).mean()) / 4.0
        r["noise_mse_offset"] = float(D.mean() ** 2) / 4.0
        r["noise_mse_resid"] = float(((D - D.mean()) ** 2).mean()) / 4.0
        # ranking
        r["spearman"] = _spearman(C, L)
        r["spearman_halves"] = _spearman(LA, LB)          # the label's own reliability
        aC, aL = int(np.argmax(C)), int(np.argmax(L))
        r["argmax_C"], r["argmax_L"] = aC, aL
        r["regret_naive"] = float(L.max() - L[aC])        # upward-biased (winner's curse on L)
        r["flip_naive"] = float(L[aC] < L.max() - 1e-12)
        # CROSS-FITTED: select on one half, score on the OTHER. Immune to the winner's curse;
        # conservative because the selector is itself an R/2 estimate.
        rAB = float(LB[int(np.argmax(LA))] - LB[aC])
        rBA = float(LA[int(np.argmax(LB))] - LA[aC])
        r["regret_cf"] = 0.5 * (rAB + rBA)
        r["flip_cf"] = float(0.5 * ((LB[int(np.argmax(LA))] > LB[aC] + 1e-12)
                                    + (LA[int(np.argmax(LB))] > LA[aC] + 1e-12)))
        # NOISE REFERENCE: what an R/2 MC oracle -- a scorer with NO bias, only sampling noise --
        # scores on held-out dice, against that half's own best. The magnitude scale for "a flip
        # within label noise is not a flip".
        r["regret_noise_ref"] = 0.5 * (float(LB.max() - LB[int(np.argmax(LA))])
                                       + float(LA.max() - LA[int(np.argmax(LB))]))
        r["flip_noise_ref"] = float(int(np.argmax(LA)) != int(np.argmax(LB)))
        r["spread_L"] = float(L.max() - L.min())
        # NO-INFORMATION CONTROL: the regret of picking uniformly at random. Cross-fitted the same
        # way so it sits on the same scale as the rows above. This is what turns "regret 0.0x" into
        # a statement -- without it, a small number could mean a good critic or an easy decision.
        r["regret_random"] = 0.5 * (float(LB[int(np.argmax(LA))] - LB.mean())
                                    + float(LA[int(np.argmax(LB))] - LA.mean()))
        # THE 1-PLY SEARCH DIVIDEND. Both selectors are noise-free (the critic's argmax, and the
        # action the policy actually played), so the paired difference is UNBIASED at full R -- no
        # cross-fitting needed, and CRN has already removed the shared dice variance.
        try:
            pos = r["actions"].index(int(r["chosen"]))
        except ValueError:
            pos = None
        if pos is None:
            r["search_dividend"] = np.nan
            r["regret_policy_cf"] = np.nan
            r["policy_is_critic_argmax"] = np.nan
        else:
            r["search_dividend"] = float(L[aC] - L[pos])
            r["regret_policy_cf"] = 0.5 * (float(LB[int(np.argmax(LA))] - LB[pos])
                                           + float(LA[int(np.argmax(LB))] - LA[pos]))
            r["policy_is_critic_argmax"] = float(pos == aC)
    return rows


def agg(rows, key, *, draws=3000):
    b = [r["battle"] for r in rows]
    m, lo, hi = boot_ci([r[key] for r in rows], b, draws=draws)
    return {"mean": m, "ci": [lo, hi], "n": sum(1 for r in rows if np.isfinite(float(r[key] if r[key] is not None else np.nan)))}


MIN_CELL = 5


def block(rows, *, draws=3000):
    if len(rows) < MIN_CELL:
        return {"n_decisions": len(rows),
                "MISSING": f"fewer than {MIN_CELL} decisions in this cell"}
    b = [r["battle"] for r in rows]
    tot = float(np.mean([r["mse_e"] for r in rows]))
    off = float(np.mean([r["mse_offset"] for r in rows]))
    res = float(np.mean([r["mse_resid"] for r in rows]))
    n_off = float(np.mean([r["noise_mse_offset"] for r in rows]))
    n_res = float(np.mean([r["noise_mse_resid"] for r in rows]))
    off_t, res_t = max(0.0, off - n_off), max(0.0, res - n_res)
    tot_t = off_t + res_t
    # the offset share of the decomposition, bootstrapped as a RATIO over battles
    uniq = sorted(set(b)); idx = {u: [i for i, x in enumerate(b) if x == u] for u in uniq}
    rng = np.random.default_rng(23)
    o = np.array([r["mse_offset"] for r in rows]); s = np.array([r["mse_resid"] for r in rows])
    no = np.array([r["noise_mse_offset"] for r in rows]); ns = np.array([r["noise_mse_resid"] for r in rows])
    sh = np.empty(draws)
    for k in range(draws):
        pick = rng.integers(0, len(uniq), len(uniq))
        sel = np.concatenate([idx[uniq[i]] for i in pick])
        a1 = max(0.0, o[sel].mean() - no[sel].mean()); a2 = max(0.0, s[sel].mean() - ns[sel].mean())
        sh[k] = a1 / (a1 + a2) if (a1 + a2) > 0 else np.nan
    sh = sh[np.isfinite(sh)]
    # THREE-WAY split. The offset itself has two parts with different consequences:
    #   b            = mean_d offset_d           a GLOBAL calibration bias -- a constant shift, so
    #                                            it cancels in ANY comparison, at any depth
    #   offset_d - b = the per-decision part     cancels WITHIN a decision (paired 1-ply search),
    #                                            NOT across decisions (a deeper tree compares
    #                                            states at different nodes)
    #   resid        = differential              never cancels
    offs = np.array([r["offset"] for r in rows])
    n_off_arr = np.array([r["noise_mse_offset"] for r in rows])
    bglob = float(offs.mean())
    # noise in the mean of A independent-ish offsets shrinks as 1/n; the per-decision spread's
    # floor is the same offset-space floor measured above.
    off_spread_true = max(0.0, float(((offs - bglob) ** 2).mean()) - float(n_off_arr.mean()))
    out = {
        "n_decisions": len(rows), "n_battles": len(set(b)),
        "n_action_cells": int(sum(r["n_used"] for r in rows)),
        "mean_actions_per_decision": round(float(np.mean([r["n_used"] for r in rows])), 2),
        "mse": {
            "total_raw": tot, "offset_raw": off, "residual_raw": res,
            "noise_floor_offset": n_off, "noise_floor_residual": n_res,
            "offset_true": off_t, "residual_true": res_t, "total_true": tot_t,
            "offset_share_of_true": (off_t / tot_t) if tot_t > 0 else None,
            "residual_share_of_true": (res_t / tot_t) if tot_t > 0 else None,
            "offset_share_ci": [float(np.quantile(sh, .025)), float(np.quantile(sh, .975))] if len(sh) else None,
            "rms_offset_true": math.sqrt(off_t), "rms_residual_true": math.sqrt(res_t),
            "rms_label_noise": math.sqrt(float(np.mean([r["noise_mse_label"] for r in rows]))),
        },
        "three_way": {
            "global_bias": bglob,
            "global_bias_mse": bglob ** 2,
            "per_decision_offset_spread_mse_true": off_spread_true,
            "rms_per_decision_offset": math.sqrt(off_spread_true),
            "differential_mse_true": res_t,
            "shares_of_true_total": {
                "global_bias": (bglob ** 2) / (bglob ** 2 + off_spread_true + res_t)
                if (bglob ** 2 + off_spread_true + res_t) > 0 else None,
                "per_decision_offset": off_spread_true / (bglob ** 2 + off_spread_true + res_t)
                if (bglob ** 2 + off_spread_true + res_t) > 0 else None,
                "differential": res_t / (bglob ** 2 + off_spread_true + res_t)
                if (bglob ** 2 + off_spread_true + res_t) > 0 else None,
            },
        },
        "mean_offset_signed": agg(rows, "offset", draws=draws),
        "spearman_C_vs_L": agg(rows, "spearman", draws=draws),
        "spearman_halves_reliability": agg(rows, "spearman_halves", draws=draws),
        "flip_naive": agg(rows, "flip_naive", draws=draws),
        "flip_cross_fitted": agg(rows, "flip_cf", draws=draws),
        "flip_noise_reference": agg(rows, "flip_noise_ref", draws=draws),
        "regret_naive": agg(rows, "regret_naive", draws=draws),
        "regret_cross_fitted": agg(rows, "regret_cf", draws=draws),
        "regret_noise_reference": agg(rows, "regret_noise_ref", draws=draws),
        "regret_random_no_information": agg(rows, "regret_random", draws=draws),
        "regret_policy_cross_fitted": agg(rows, "regret_policy_cf", draws=draws),
        "search_dividend_1ply": agg(rows, "search_dividend", draws=draws),
        "policy_agrees_with_critic_argmax": agg(rows, "policy_is_critic_argmax", draws=draws),
        "regret_cf_median": float(np.median([r["regret_cf"] for r in rows])),
        "regret_cf_p90": float(np.quantile([r["regret_cf"] for r in rows], 0.90)),
        "regret_cf_p95": float(np.quantile([r["regret_cf"] for r in rows], 0.95)),
        "regret_naive_median": float(np.median([r["regret_naive"] for r in rows])),
        "regret_naive_p90": float(np.quantile([r["regret_naive"] for r in rows], 0.90)),
        "label_spread_mean": float(np.mean([r["spread_L"] for r in rows])),
    }
    # the decision-relevant excess: critic regret MINUS what pure label noise already costs
    m_c, lo_c, hi_c = boot_ci([r["regret_cf"] - r["regret_noise_ref"] for r in rows], b, draws=draws)
    out["regret_cf_minus_noise_reference"] = {"mean": m_c, "ci": [lo_c, hi_c]}
    # How much of the ACHIEVABLE ranking gain the critic captures. 1.0 = ranks as well as a
    # tight-MC oracle measured at this R; 0.0 = no better than choosing at random. The
    # denominator is (random - noise_floor) because an oracle at finite R cannot reach 0 regret.
    rr = float(np.mean([r["regret_random"] for r in rows]))
    rn = float(np.mean([r["regret_noise_ref"] for r in rows]))
    rc = float(np.mean([r["regret_cf"] for r in rows]))
    # GUARD: when the no-information baseline is barely worse than the noise floor there is
    # almost no ranking gain available at all, and the ratio is a division by nearly nothing --
    # it reads as a huge "capture" that means only that the cell's decisions were easy. Report
    # MISSING with the reason rather than a number that will be quoted.
    denom = rr - rn
    if denom > 0.02:
        out["capture_fraction_of_achievable_ranking_gain"] = (rr - rc) / denom
        out["capture_fraction_MISSING"] = None
    else:
        out["capture_fraction_of_achievable_ranking_gain"] = None
        out["capture_fraction_MISSING"] = (
            f"denominator (random {rr:.4f} - noise floor {rn:.4f} = {denom:.4f}) is below 0.02: "
            "this cell has almost no achievable ranking gain to capture, so the ratio is unstable")
    out["regret_random_minus_noise_floor"] = denom
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--head", default="win_prob", choices=["win_prob", "value"])
    ap.add_argument("--json-out", default=None)
    ap.add_argument("--min-cell", type=int, default=5)
    a = ap.parse_args(argv)
    globals()["MIN_CELL"] = a.min_cell
    paths = [p for g in a.inputs for p in sorted(glob.glob(g))]
    rows, dropped = load(paths, head=a.head)
    if not rows:
        print(json.dumps({"MISSING": "no usable decisions", "dropped": dict(dropped)}, indent=2))
        return 2
    rows = per_decision(rows)
    turns = np.array([r["turn"] for r in rows])
    t1, t2 = np.quantile(turns, 1 / 3), np.quantile(turns, 2 / 3)

    def nl_bin(r):
        return "3-4" if r["n_legal"] <= 4 else ("5-6" if r["n_legal"] <= 6 else "7-9")

    cells = {"ALL": rows}
    for k in ("pivotal", "ordinary"):
        cells[f"stratum={k}"] = [r for r in rows if r["stratum"] == k]
    for k in ("bot", "sentinel"):
        cells[f"opp_class={k}"] = [r for r in rows if r["opp_class"] == k]
    for k in ("3-4", "5-6", "7-9"):
        cells[f"n_legal={k}"] = [r for r in rows if nl_bin(r) == k]
    cells["turn_tercile=early"] = [r for r in rows if r["turn"] <= t1]
    cells["turn_tercile=mid"] = [r for r in rows if t1 < r["turn"] <= t2]
    cells["turn_tercile=late"] = [r for r in rows if r["turn"] > t2]
    for k in ("win", "loss"):
        cells[f"outcome={k}"] = [r for r in rows if r["outcome"] == k]

    out = {
        "head": a.head,
        "n_decisions": len(rows), "n_battles": len(set(r["battle"] for r in rows)),
        "n_action_cells": int(sum(r["n_used"] for r in rows)),
        "R_per_action": int(np.median([r["R"] for r in rows])),
        "dropped": dict(dropped),
        "turn_tercile_edges": [float(t1), float(t2)],
        "cells": {k: block(v) for k, v in cells.items()},
        # EVERY NUMBER. One row per decision: the critic vector and BOTH label half-blocks, so any
        # statistic in this file can be recomputed from here without re-running the rollouts.
        "decisions": [
            {"battle": os.path.basename(r["battle"]), "short": r["short"], "inv": r["inv"],
             "turn": r["turn"], "opponent": r["opponent"], "opp_class": r["opp_class"],
             "outcome": r["outcome"], "stratum": r["stratum"], "n_legal": r["n_legal"],
             "chosen_action": r["chosen"], "td_delta": r["td_delta"],
             "recorded_win_prob": r["rec_win_prob"], "actions": r["actions"],
             "C": [round(float(x), 6) for x in r["C"]],
             "L_A": [round(float(x), 6) for x in r["LA"]],
             "L_B": [round(float(x), 6) for x in r["LB"]],
             "offset": r["offset"], "mse_resid": r["mse_resid"],
             "noise_mse_resid": r["noise_mse_resid"],
             "regret_cf": r["regret_cf"], "regret_naive": r["regret_naive"],
             "regret_noise_ref": r["regret_noise_ref"],
             "search_dividend": (None if not np.isfinite(r["search_dividend"])
                                 else r["search_dividend"]),
             "spearman": r["spearman"], "spearman_halves": r["spearman_halves"]}
            for r in rows
        ],
    }
    if a.json_out:
        os.makedirs(os.path.dirname(a.json_out) or ".", exist_ok=True)
        with open(a.json_out, "w") as f:
            json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
