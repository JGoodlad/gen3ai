"""PROBE L, stage 3 — the cross-tab.

Reads the sweep shards and answers, with matched controls and a MEASURED dice floor:

  1. Does the one-ply WIN-PROB head rank an alternative above the whiff AT DECISION TIME?
     Reported against the `hit_pivot` / `no_pivot` base rates from the same battles, because the
     head beats the played action GENERALLY (probe G: 35% agreement) and a raw fraction is
     therefore not evidence of whiff-specific knowledge.
  2. Is the margin real? The CRN margin is quoted against the per-decision spread of the SAME
     margin over R independent dice streams (the leaf noise floor), and against sign stability
     across those streams.
  3. Repeat offenders: does the disagreement GROW with the click ordinal inside a loop?
  4. alpha elevation: P(SWITCH) at whiff decisions vs the same controls.

Bootstrap CIs are over BATTLES (the cluster), never over decisions — decisions within a battle
share a game, an opponent and a team draw.

Run:  python tmp/probe_l_analyze.py --out designs/research_state/measurements/whiff_head_knowledge_2026-08-29.json
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

MIN_LEGAL = 3


# --------------------------------------------------------------------------- statistics
def boot_ci(values, clusters, *, n=4000, seed=3, stat=np.mean):
    """Cluster bootstrap over BATTLES. Returns (point, lo, hi) or (None, None, None)."""
    v = np.asarray(values, dtype=float)
    if v.size == 0:
        return None, None, None
    cl = np.asarray(clusters)
    uniq = np.unique(cl)
    idx = {c: np.where(cl == c)[0] for c in uniq}
    rng = np.random.default_rng(seed)
    out = np.empty(n)
    for b in range(n):
        pick = rng.choice(uniq, size=uniq.size, replace=True)
        sel = np.concatenate([idx[c] for c in pick])
        out[b] = stat(v[sel])
    return float(stat(v)), float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def diff_ci(a_vals, a_cl, b_vals, b_cl, *, n=4000, seed=5):
    """Cluster bootstrap on the DIFFERENCE of two means, resampling each arm's battles."""
    a, b = np.asarray(a_vals, float), np.asarray(b_vals, float)
    if a.size == 0 or b.size == 0:
        return None, None, None
    acl, bcl = np.asarray(a_cl), np.asarray(b_cl)
    au, bu = np.unique(acl), np.unique(bcl)
    ai = {c: np.where(acl == c)[0] for c in au}
    bi = {c: np.where(bcl == c)[0] for c in bu}
    rng = np.random.default_rng(seed)
    out = np.empty(n)
    for k in range(n):
        pa = rng.choice(au, size=au.size, replace=True)
        pb = rng.choice(bu, size=bu.size, replace=True)
        out[k] = (a[np.concatenate([ai[c] for c in pa])].mean()
                  - b[np.concatenate([bi[c] for c in pb])].mean())
    return float(a.mean() - b.mean()), float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def med(xs):
    xs = [x for x in xs if x is not None]
    return float(np.median(xs)) if xs else None


# --------------------------------------------------------------------------- per-decision read
def read_decision(r):
    """Collapse one swept decision into the quantities the cross-tab needs.

    A terminal arm is scored 1.0 / 0.0 win-prob (a one-ply search sees the terminal exactly), the
    same convention probe G used, so a KO-ing alternative is not silently dropped."""
    cands = r.get("candidates") or []
    if not cands:
        return None

    def wp(c):
        if c["terminal"] == "win":
            return 1.0
        if c["terminal"] == "loss":
            return 0.0
        return c["win_prob_crn"]

    def vv(c):
        if c["terminal"] == "win":
            return float("inf")
        if c["terminal"] == "loss":
            return float("-inf")
        return c["value_crn"]

    chosen = next((c for c in cands if c["is_chosen"]), None)
    if chosen is None or wp(chosen) is None:
        return None
    alts = [c for c in cands if not c["is_chosen"] and wp(c) is not None]
    if not alts:
        return None
    best = max(alts, key=lambda c: wp(c))
    margin = wp(best) - wp(chosen)
    valts = [c for c in cands if not c["is_chosen"] and vv(c) is not None]
    vbest = max(valts, key=vv) if valts else None
    vmargin = (vv(vbest) - vv(chosen)) if (vbest is not None and vv(chosen) is not None
                                           and math.isfinite(vv(vbest))
                                           and math.isfinite(vv(chosen))) else None

    # --- dice floor: the SAME paired margin recomputed on each independent seed ---------------
    seed_margins = None
    if r.get("wp_map_verified"):
        n_seed = min((len(c["wp_seeds"]) for c in cands if c.get("wp_seeds")), default=0)
        # the LAST evaluated seed is the CRN "original" line (verified); the earlier ones are the
        # independent draws that measure the floor
        if n_seed >= 3 and all(c.get("wp_seeds") and len(c["wp_seeds"]) == n_seed for c in cands):
            ms = []
            for j in range(n_seed - 1):
                ch = chosen["wp_seeds"][j]
                al = [c["wp_seeds"][j] for c in cands if not c["is_chosen"]]
                if ch is None or any(x is None for x in al) or not al:
                    continue
                ms.append(max(al) - ch)
            seed_margins = ms or None

    # How much probability mass the POLICY put on the action the head prefers. This is the size of
    # the distillation gap AND the exploration-starvation reading: an alternative the head ranks
    # first but the policy samples at p≈0.01 realizes its advantage essentially never.
    pol = {k: (float(str(v).rstrip("%")) / 100.0 if v is not None else None)
           for k, v in (r.get("policy_probs") or {}).items()}
    p_best = pol.get(best["label"])

    alpha = r.get("alpha") or []
    a_sw = next((float(x["p"]) for x in alpha if x.get("name") == "SWITCH"), None)
    return {
        "battle": r["base"], "step": r["step"], "opponent": r["opponent"],
        "outcome": r["outcome"], "tag": r["tag"], "kind": r.get("kind"),
        "inv": r["inv"], "turn": r["turn"], "n_legal": r.get("n_legal"),
        "loop_step": bool(r.get("loop_step")), "reclick": bool(r.get("reclick")),
        "click_ordinal": r.get("click_ordinal"),
        "chosen_prob": r.get("chosen_prob"),
        "wp_chosen": wp(chosen), "wp_best_alt": wp(best), "wp_margin": margin,
        "best_alt_label": best["label"], "best_alt_is_switch": best["choice"].startswith("switch"),
        "chosen_label": r.get("chosen_label"),
        "v_margin": vmargin, "v_best_alt_label": (vbest["label"] if vbest else None),
        "policy_p_on_head_best": p_best,
        "prefers_alt": margin > 0.0,
        "v_prefers_alt": (vmargin > 0.0) if vmargin is not None else None,
        "seed_margins": seed_margins,
        "alpha_switch_p": a_sw,
        "alpha_top_is_switch": (bool(alpha[0].get("name") == "SWITCH") if alpha else None),
        "wp_s": r.get("wp_s"), "v_s": r.get("v_s"),
        "delta_win_prob_realized": r.get("delta_win_prob"),
        "delta_v_realized": r.get("delta_v"),
        "wp_map_verified": bool(r.get("wp_map_verified")),
    }


def arm(rows, pred):
    return [d for d in rows if pred(d)]


def frac_block(ds, key, name):
    vals = [1.0 if d[key] else 0.0 for d in ds if d[key] is not None]
    cl = [d["battle"] for d in ds if d[key] is not None]
    p, lo, hi = boot_ci(vals, cl)
    return {"metric": name, "n": len(vals), "n_battles": len(set(cl)),
            "frac": p, "ci": [lo, hi]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="tmp/probe_l_out/shard*.jsonl")
    ap.add_argument("--out", default="tmp/probe_l_result.json")
    ap.add_argument("--min-legal", type=int, default=MIN_LEGAL)
    ap.add_argument("--census", default="tmp/whiff_census_all.jsonl")
    a = ap.parse_args(argv)

    # (base, arrival) -> the set of OUR moves observed to whiff against that arrival in that
    # battle. Used to check that the alternative the head prefers is not itself a known whiff —
    # "prefers an alternative" and "prefers a NON-WHIFF action" are different claims.
    known_whiff: "dict[tuple, set]" = collections.defaultdict(set)
    if os.path.exists(a.census):
        for ln in open(a.census):
            try:
                r = json.loads(ln)
            except ValueError:
                continue
            for b in r["baits"]:
                if b["whiff"]:
                    known_whiff[(r["base"], b["arrival"])].add(b["move"])

    raw = []
    for f in sorted(glob.glob(a.glob)):
        for ln in open(f):
            try:
                raw.append(json.loads(ln))
            except ValueError:
                pass
    # DEDUP by (base, inv). The sweep was restarted mid-flight with a different population order,
    # and a shard's resume set only knows its own file — so a decision reassigned across shards can
    # be swept twice. Two identical reads of one decision are not two decisions; keeping both would
    # silently double its weight in every mean below.
    seen: set = set()
    dedup = []
    for r in raw:
        k = (r.get("base"), r.get("inv"))
        if k in seen:
            continue
        seen.add(k)
        dedup.append(r)
    n_dupes = len(raw) - len(dedup)
    raw = dedup
    errs = [r for r in raw if "error" in r]
    ok = [r for r in raw if "error" not in r]
    ds = [d for d in (read_decision(r) for r in ok) if d is not None]
    dropped = len(ok) - len(ds)
    ds = [d for d in ds if (d["n_legal"] or 0) >= a.min_legal]
    for d in ds:
        ks = known_whiff.get((d["battle"], d.get("arrival")), set()) if d.get("arrival") else set()
        d["best_alt_is_known_whiff"] = (d["best_alt_label"] in ks) if ks else False
        d["prefers_nonwhiff_alt"] = bool(d["prefers_alt"] and not d["best_alt_is_known_whiff"])

    W = arm(ds, lambda d: d["tag"] == "whiff")
    WI = arm(ds, lambda d: d["tag"] == "whiff" and d["kind"] == "immune")
    H = arm(ds, lambda d: d["tag"] == "hit_pivot")
    N = arm(ds, lambda d: d["tag"] == "no_pivot")

    res: dict = {
        "what": "PROBE L — the whiff x head-knowledge cross-tab",
        "date": "2026-08-29",
        "run": "ai_v9_29_rev1_0823",
        "scope": "sentinel_* eval traces, all 11 steps with traces; one snapshot per step "
                 "(the EXACT model that wrote those traces)",
        "coverage": {
            "swept": len(raw), "duplicate_rows_dropped": n_dupes, "errors": len(errs),
            "error_kinds": dict(collections.Counter(
                r["error"].split(":")[0] for r in errs)),
            "unreadable_decisions": dropped,
            "after_min_legal": len(ds), "min_legal": a.min_legal,
            "wp_map_verified": int(sum(d["wp_map_verified"] for d in ds)),
            "by_tag": dict(collections.Counter(d["tag"] for d in ds)),
            "whiff_kinds": dict(collections.Counter(d["kind"] for d in W if d["kind"])),
            "battles": len({d["battle"] for d in ds}),
        },
    }

    # ---------------------------------------------------------------- 1. THE CROSS-TAB
    cross = {}
    for name, sub in (("whiff_all", W), ("whiff_immune", WI), ("hit_pivot", H), ("no_pivot", N)):
        margins = [d["wp_margin"] for d in sub]
        cl = [d["battle"] for d in sub]
        mp, mlo, mhi = boot_ci(margins, cl)
        vm = [d["v_margin"] for d in sub if d["v_margin"] is not None]
        vcl = [d["battle"] for d in sub if d["v_margin"] is not None]
        vp, vlo, vhi = boot_ci(vm, vcl)
        cross[name] = {
            "n": len(sub), "n_battles": len(set(cl)),
            "head_prefers_alt": frac_block(sub, "prefers_alt", "win-prob head ranks an "
                                           "alternative above the played action"),
            "head_prefers_NONWHIFF_alt": frac_block(
                sub, "prefers_nonwhiff_alt",
                "…and that alternative is not itself a move seen to whiff against this arrival "
                "in this battle"),
            "wp_margin_mean": mp, "wp_margin_ci": [mlo, mhi],
            "wp_margin_median": med(margins),
            "wp_margin_p75": (float(np.percentile(margins, 75)) if margins else None),
            "wp_margin_p90": (float(np.percentile(margins, 90)) if margins else None),
            "v_head_prefers_alt": frac_block(sub, "v_prefers_alt", "scalar V ranks an "
                                             "alternative above the played action"),
            "v_margin_mean": vp, "v_margin_ci": [vlo, vhi], "v_margin_median": med(vm),
            "median_chosen_prob": med([d["chosen_prob"] for d in sub]),
            "best_alt_is_switch_frac": (float(np.mean([d["best_alt_is_switch"] for d in sub]))
                                        if sub else None),
            "median_wp_s": med([d["wp_s"] for d in sub]),
            # the starvation reading: what the policy actually sampled the preferred action at
            "median_policy_p_on_head_best": med([d["policy_p_on_head_best"] for d in sub
                                                 if d["prefers_alt"]]),
            "frac_head_best_under_p05": (
                float(np.mean([1.0 if (d["policy_p_on_head_best"] or 0.0) < 0.05 else 0.0
                               for d in sub if d["prefers_alt"]
                               and d["policy_p_on_head_best"] is not None]))
                if any(d["prefers_alt"] and d["policy_p_on_head_best"] is not None
                       for d in sub) else None),
        }
    res["cross_tab"] = cross

    # whiff MINUS control, one CI on the DIFFERENCE (the only honest comparison)
    res["whiff_minus_control"] = {}
    for cname, C in (("hit_pivot", H), ("no_pivot", N)):
        for qname, key in (("head_prefers_alt", "prefers_alt"), ):
            av = [1.0 if d[key] else 0.0 for d in WI]
            bv = [1.0 if d[key] else 0.0 for d in C]
            p, lo, hi = diff_ci(av, [d["battle"] for d in WI], bv, [d["battle"] for d in C])
            res["whiff_minus_control"][f"{qname}_vs_{cname}"] = {"diff": p, "ci": [lo, hi]}
        p, lo, hi = diff_ci([d["wp_margin"] for d in WI], [d["battle"] for d in WI],
                            [d["wp_margin"] for d in C], [d["battle"] for d in C])
        res["whiff_minus_control"][f"wp_margin_vs_{cname}"] = {"diff": p, "ci": [lo, hi]}

    # ---------------------------------------------------------------- 2. THE DICE FLOOR
    floor_rows = [d for d in ds if d["seed_margins"]]
    def floor_block(sub):
        sub = [d for d in sub if d["seed_margins"]]
        if not sub:
            return {"n": 0}
        sds = [float(np.std(d["seed_margins"], ddof=1)) for d in sub if len(d["seed_margins"]) > 1]
        # sign stability of the preference across INDEPENDENT dice
        stab = [float(np.mean([m > 0 for m in d["seed_margins"]])) for d in sub]
        # CRN margin measured against that decision's own dice spread
        clears2 = []
        for d in sub:
            s = float(np.std(d["seed_margins"], ddof=1)) if len(d["seed_margins"]) > 1 else 0.0
            clears2.append(1.0 if d["wp_margin"] > 2.0 * s else 0.0)
        cl = [d["battle"] for d in sub]
        p2, lo2, hi2 = boot_ci(clears2, cl)
        allpos = [1.0 if all(m > 0 for m in d["seed_margins"]) and d["wp_margin"] > 0 else 0.0
                  for d in sub]
        pa, loa, hia = boot_ci(allpos, cl)
        return {
            "n": len(sub), "n_battles": len(set(cl)),
            "median_within_decision_sd_of_margin": med(sds),
            "mean_within_decision_sd_of_margin": (float(np.mean(sds)) if sds else None),
            "p90_within_decision_sd": (float(np.percentile(sds, 90)) if sds else None),
            "mean_sign_stability_across_dice": float(np.mean(stab)),
            "frac_crn_margin_gt_2sd": p2, "frac_crn_margin_gt_2sd_ci": [lo2, hi2],
            "frac_preference_holds_on_every_dice_stream": pa,
            "frac_preference_holds_on_every_dice_stream_ci": [loa, hia],
        }
    res["dice_floor"] = {
        "note": "R independent dice streams per action beside the CRN line; the paired margin "
                "max_alt wp - chosen wp is recomputed on each. sd across those streams IS the "
                "leaf noise floor for this comparison.",
        "n_with_seeds": len(floor_rows),
        "whiff_immune": floor_block(WI), "hit_pivot": floor_block(H), "no_pivot": floor_block(N),
    }

    # ---------------------------------------------------------------- 3. REPEAT OFFENDERS
    def ord_bucket(d):
        o = d["click_ordinal"]
        if o is None:
            return "not_in_loop"
        return "click_1" if o == 1 else ("click_2" if o == 2 else "click_3plus")
    rep = {}
    for b in ("not_in_loop", "click_1", "click_2", "click_3plus"):
        sub = [d for d in WI if ord_bucket(d) == b]
        if not sub:
            rep[b] = {"n": 0}
            continue
        cl = [d["battle"] for d in sub]
        p, lo, hi = boot_ci([1.0 if d["prefers_alt"] else 0.0 for d in sub], cl)
        mp, mlo, mhi = boot_ci([d["wp_margin"] for d in sub], cl)
        rep[b] = {"n": len(sub), "n_battles": len(set(cl)),
                  "head_prefers_alt": p, "ci": [lo, hi],
                  "wp_margin_mean": mp, "wp_margin_ci": [mlo, mhi],
                  "wp_margin_median": med([d["wp_margin"] for d in sub]),
                  "median_chosen_prob": med([d["chosen_prob"] for d in sub]),
                  "median_alpha_switch_p": med([d["alpha_switch_p"] for d in sub])}
    # first vs later clicks, one CI on the difference
    first = [d for d in WI if d["click_ordinal"] == 1]
    later = [d for d in WI if (d["click_ordinal"] or 0) >= 2]
    if first and later:
        p, lo, hi = diff_ci([d["wp_margin"] for d in later], [d["battle"] for d in later],
                            [d["wp_margin"] for d in first], [d["battle"] for d in first])
        rep["later_minus_first_margin"] = {"diff": p, "ci": [lo, hi],
                                           "n_later": len(later), "n_first": len(first)}
        p, lo, hi = diff_ci([1.0 if d["prefers_alt"] else 0.0 for d in later],
                            [d["battle"] for d in later],
                            [1.0 if d["prefers_alt"] else 0.0 for d in first],
                            [d["battle"] for d in first])
        rep["later_minus_first_prefers"] = {"diff": p, "ci": [lo, hi]}
    res["repeat_offenders"] = rep

    # ---------------------------------------------------------------- 4. ALPHA ELEVATION
    alpha = {}
    for name, sub in (("whiff_immune", WI), ("whiff_all", W), ("hit_pivot", H), ("no_pivot", N)):
        vals = [d["alpha_switch_p"] for d in sub if d["alpha_switch_p"] is not None]
        cl = [d["battle"] for d in sub if d["alpha_switch_p"] is not None]
        p, lo, hi = boot_ci(vals, cl)
        tops = [1.0 if d["alpha_top_is_switch"] else 0.0 for d in sub
                if d["alpha_top_is_switch"] is not None]
        tcl = [d["battle"] for d in sub if d["alpha_top_is_switch"] is not None]
        tp, tlo, thi = boot_ci(tops, tcl)
        alpha[name] = {"n": len(vals), "mean_alpha_switch_p": p, "ci": [lo, hi],
                       "median_alpha_switch_p": med(vals),
                       "alpha_top1_is_switch": tp, "top1_ci": [tlo, thi]}
    for cname, C in (("hit_pivot", H), ("no_pivot", N)):
        p, lo, hi = diff_ci([d["alpha_switch_p"] for d in WI if d["alpha_switch_p"] is not None],
                            [d["battle"] for d in WI if d["alpha_switch_p"] is not None],
                            [d["alpha_switch_p"] for d in C if d["alpha_switch_p"] is not None],
                            [d["battle"] for d in C if d["alpha_switch_p"] is not None])
        alpha[f"whiff_immune_minus_{cname}"] = {"diff": p, "ci": [lo, hi]}
    # loop steps specifically
    ls = [d for d in WI if d["loop_step"]]
    if ls:
        vals = [d["alpha_switch_p"] for d in ls if d["alpha_switch_p"] is not None]
        alpha["loop_steps"] = {
            "n": len(ls), "median_alpha_switch_p": med(vals),
            "alpha_top1_is_switch": (float(np.mean([bool(d["alpha_top_is_switch"]) for d in ls
                                                    if d["alpha_top_is_switch"] is not None]))
                                     if any(d["alpha_top_is_switch"] is not None for d in ls)
                                     else None)}
    res["alpha_elevation"] = alpha

    # ---------------------------------------------------------------- 5. by step / outcome
    res["by_step"] = {}
    for s in sorted({d["step"] for d in WI}):
        sub = [d for d in WI if d["step"] == s]
        res["by_step"][s] = {"n": len(sub),
                             "head_prefers_alt": float(np.mean([d["prefers_alt"] for d in sub])),
                             "wp_margin_median": med([d["wp_margin"] for d in sub])}
    res["by_outcome"] = {}
    for o in sorted({d["outcome"] for d in WI}):
        sub = [d for d in WI if d["outcome"] == o]
        res["by_outcome"][o] = {"n": len(sub),
                                "head_prefers_alt": float(np.mean([d["prefers_alt"] for d in sub])),
                                "wp_margin_median": med([d["wp_margin"] for d in sub])}

    # ---------------------------------------------------------------- 6. realized post-whiff drop
    # what the whiff actually COST in win-prob units, for the shaping accounting
    dwp = [d["delta_win_prob_realized"] for d in WI if d["delta_win_prob_realized"] is not None]
    res["realized_cost"] = {
        "n": len(dwp),
        "median_delta_win_prob_on_whiff_turn": med(dwp),
        "mean_delta_win_prob": (float(np.mean(dwp)) if dwp else None),
        "p10": (float(np.percentile(dwp, 10)) if dwp else None),
        "loop_step_median": med([d["delta_win_prob_realized"] for d in WI
                                 if d["loop_step"] and d["delta_win_prob_realized"] is not None]),
        "control_no_pivot_median": med([d["delta_win_prob_realized"] for d in N
                                        if d["delta_win_prob_realized"] is not None]),
        "note": "realized per-turn change in the recorded win-prob series, from the census "
                "(worst decision on the turn); the input to the shaping-dose accounting.",
    }

    # ---------------------------------------------------------------- 7. the MODEL-FREE census
    # The population this probe drew from, in the bait hunt's own units, so the cross-tab is read
    # beside the pathology it is about rather than in isolation.
    cen = [json.loads(ln) for ln in open(a.census)] if os.path.exists(a.census) else []

    def cen_agg(rs):
        if not rs:
            return {}
        piv = sum(r["moved_into_pivots"] for r in rs)
        wh = sum(r["whiffs"] for r in rs)
        dec = sum(r["n_decisions"] for r in rs)
        rc = sum(r["reclicks"] for r in rs)
        lb = sum(r["loop_battle"] for r in rs)
        ls = [b["chosen_prob"] for r in rs for b in r["baits"]
              if b["loop_step"] and b["chosen_prob"] is not None]
        return {
            "battles": len(rs), "moved_into_pivots": piv, "whiffs": wh, "decisions": dec,
            "whiff_rate_per_pivot": wh / max(1, piv),
            "whiff_rate_per_decision": wh / max(1, dec),
            "B1_within_battle_reclick_rate": rc / max(1, wh), "reclicks": rc,
            "B2_loop_battle_rate": lb / max(1, len(rs)), "loop_battles": lb,
            "B3_median_chosen_prob_on_loop_steps": med(ls), "n_loop_steps": len(ls),
        }
    res["census"] = {
        "all_steps": cen_agg(cen),
        "step_24000000": cen_agg([r for r in cen if r["step"] == "step_24000000"]),
        "by_outcome": {o: cen_agg([r for r in cen if r["outcome"] == o])
                       for o in ("WIN", "LOSS")},
        "gen15_baseline_for_reference": {
            "whiff_rate_per_pivot": 0.167, "B1": 0.322, "B2": 0.139, "B3": 0.963,
            "source": "loops.LOOP_BASELINES / bait_loop_hunt.md; a reference, not a target"},
    }

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)
    # a per-decision dump for any follow-up
    with open(os.path.splitext(a.out)[0] + "_decisions.json", "w") as f:
        json.dump([{k: v for k, v in d.items() if k != "seed_margins"} for d in ds], f)
    print(json.dumps({k: res[k] for k in
                      ("coverage", "cross_tab", "whiff_minus_control", "dice_floor",
                       "repeat_offenders", "alpha_elevation", "realized_cost")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
