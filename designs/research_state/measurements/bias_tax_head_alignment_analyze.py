#!/usr/bin/env python3
"""PROBE M — analysis: does the WIN-PROB head price what `no_progress_tax` taxes?

THE DEFINITIONS, STATED BEFORE COMPUTING
========================================
phi(s_t) = the win-prob head's P(win) recorded AT DECISION TIME for decision t.

  d_out(t) = phi(s_{t+1}) - phi(s_t)   the head's verdict on the window the tax CHARGES.
                                        The charge for action a_t lands in reward r_t and is
                                        decided by the fold that produces obs_{t+1}, so d_out is
                                        the head's own pricing of exactly that window.
  d_in(t)  = phi(s_t) - phi(s_{t-1})    was the head ALREADY declining coming in (anticipation).

  EPS = 0.0025. Not arbitrary: it is the TAX'S OWN SIZE expressed in win-prob units.
        `no_progress_penalty` = 0.15 and the terminal reward spans -30..+30 (VICTORY_VALUE = 30),
        so a span of 60 reward units == 1.0 of win probability, and 0.15 / 60 = 0.0025. EPS
        therefore asks the exact question the mission asks: "did the head price this window at
        least as costly as the hand-coded rule charges for it?" Sensitivity is reported at
        EPS in {0, 0.0025, 0.01}.

  ALIGNED-consequence(t)  :=  d_out(t) < -EPS
  ALIGNED-anticipation(t) :=  d_in(t)  < -EPS     <- the REGISTERED prediction's literal reading
                                                     ("the head's signal is already depressed /
                                                      declining BEFORE the tax fires")
  OVER-TAXED(t)           :=  taxed AND d_out(t) >= 0   (flat or rising: no outcome cost at all)

CONTROL. phi is approximately a martingale, so P(d_out < 0) is near 1/2 on ANY decision and a raw
"X% of taxed decisions show a decline" is not evidence of anything. Every arm is therefore reported
beside its base rate, and every claim is carried by a CONTRAST with one CI.

STRATIFY BY ACTION KIND — MANDATORY, not optional. The census shows the taxed and untaxed sets have
wildly different action composition (73% of voluntary switches are taxed vs 6.7% of moves), and a
switch and a move have different phi dynamics for reasons that have nothing to do with the tax. An
unstratified contrast measures the composition, not the tax.

CIs are cluster-bootstrapped over BATTLES, never over decisions.

Run:
  nice -n 15 python bias_tax_head_alignment_analyze.py --census '/tmp/probeM/census_*.jsonl' \
      --out bias_tax_head_alignment_2026-08-29.json
(in a linked worktree, first: export PYTHONPATH=$PYTHONPATH:src)
"""
from __future__ import annotations

import argparse
import glob
import json
from collections import defaultdict

import numpy as np

EPS = 0.0025
VICTORY_SPAN = 60.0          # -VICTORY_VALUE .. +VICTORY_VALUE
NO_PROGRESS_PENALTY = 0.15
TAX_IN_PHI = NO_PROGRESS_PENALTY / VICTORY_SPAN
N_BOOT = 4000
RNG = np.random.default_rng(20260829)


def load(patterns) -> list:
    rows = []
    for pat in patterns:
        for f in sorted(glob.glob(pat)):
            with open(f) as fh:
                for line in fh:
                    rows.append(json.loads(line))
    return rows


def kind_of(r: dict) -> str:
    if r["self_forced"]:
        return "forced_switch"
    if r["action"] == 10:
        return "struggle"
    return "switch" if r["action"] < 6 else "move"


# ---------------------------------------------------------------- cluster bootstrap
# EVERY statistic here is the MEAN of a per-decision scalar, so a cluster bootstrap reduces to
# resampling per-battle (sum, count) pairs: mean = sum(sums[idx]) / sum(counts[idx]). That makes
# the whole battery O(n_boot x n_battles) instead of O(n_boot x n_decisions) with Python-level
# list building, and it is EXACT — not an approximation of the naive resample.
def _clusters(rows):
    by = defaultdict(list)
    for r in rows:
        by[(r["run"], r["step"], r["opponent"], r["battle"])].append(r)
    return list(by.values())


def _cluster_index(rows):
    ids, out = {}, np.empty(len(rows), dtype=np.int64)
    for i, r in enumerate(rows):
        k = (r["run"], r["step"], r["opponent"], r["battle"])
        out[i] = ids.setdefault(k, len(ids))
    return out, len(ids)


def _agg(rows, valfn):
    """Per-battle (sum, count) of `valfn`, skipping rows where it is None."""
    ci, n_cl = _cluster_index(rows)
    v = np.array([np.nan if (x := valfn(r)) is None else float(x) for r in rows])
    keep = ~np.isnan(v)
    s = np.bincount(ci[keep], weights=v[keep], minlength=n_cl)
    c = np.bincount(ci[keep], minlength=n_cl).astype(float)
    return s, c


def _boot_means(s, c, n_boot):
    if s.size < 3 or c.sum() == 0:
        return None
    idx = RNG.integers(0, s.size, size=(n_boot, s.size))
    num, den = s[idx].sum(axis=1), c[idx].sum(axis=1)
    ok = den > 0
    return num[ok] / den[ok]


def boot_stat(rows, valfn, n_boot=None):
    """Point estimate + percentile CI of mean(valfn), resampling BATTLES with replacement."""
    n_boot = n_boot or N_BOOT
    if not rows:
        return {"n": 0}
    s, c = _agg(rows, valfn)
    if c.sum() == 0:
        return {"n": len(rows), "battles": int(s.size), "point": None}
    point = float(s.sum() / c.sum())
    draws = _boot_means(s, c, n_boot)
    if draws is None or draws.size == 0:
        return {"n": len(rows), "battles": int(s.size), "point": point}
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return {"n": int(c.sum()), "battles": int(s.size), "point": point,
            "ci": [float(lo), float(hi)]}


def boot_contrast(a, b, valfn, n_boot=None):
    """One CI on the DIFFERENCE mean_a - mean_b, resampling battles INDEPENDENTLY in each arm."""
    n_boot = n_boot or N_BOOT
    if not a or not b:
        return {"n_a": len(a), "n_b": len(b)}
    sa, ca = _agg(a, valfn)
    sb, cb = _agg(b, valfn)
    if ca.sum() == 0 or cb.sum() == 0:
        return {"n_a": len(a), "n_b": len(b)}
    point = float(sa.sum() / ca.sum() - sb.sum() / cb.sum())
    da, db = _boot_means(sa, ca, n_boot), _boot_means(sb, cb, n_boot)
    if da is None or db is None:
        return {"n_a": len(a), "n_b": len(b), "point": point}
    m = min(da.size, db.size)
    diff = da[:m] - db[:m]
    lo, hi = np.percentile(diff, [2.5, 97.5])
    return {"n_a": int(ca.sum()), "n_b": int(cb.sum()),
            "battles_a": int(sa.size), "battles_b": int(sb.size),
            "point": point, "ci": [float(lo), float(hi)],
            "excludes_zero": bool(lo > 0 or hi < 0)}


# ---------------------------------------------------------------- statistics
def d_out(r):
    return r["phi_next"] - r["phi"]


def d_in(r):
    return None if r["phi_prev"] is None else r["phi"] - r["phi_prev"]


f_mean_out = d_out                                     # per-row value functions
f_mean_in = d_in


def mk_align_out(eps):
    return lambda r: float(d_out(r) < -eps)


def mk_align_in(eps):
    return lambda r: (None if r["phi_prev"] is None else float(d_in(r) < -eps))


def f_over(r):
    return float(d_out(r) >= 0.0)


def arm_summary(rows, label):
    if not rows:
        return {"arm": label, "n": 0}
    do = np.array([d_out(r) for r in rows])
    di = np.array([d_in(r) for r in rows if r["phi_prev"] is not None])
    out = {
        "arm": label, "n": len(rows), "battles": len(_clusters(rows)),
        "mean_d_out": boot_stat(rows, f_mean_out),
        "median_d_out": float(np.median(do)),
        "mean_d_out_reward_units": float(np.mean(do) * VICTORY_SPAN),
        "aligned_consequence": {f"eps_{e}": boot_stat(rows, mk_align_out(e))
                                for e in (0.0, EPS, 0.01)},
        "aligned_anticipation": {f"eps_{e}": boot_stat(rows, mk_align_in(e))
                                 for e in (0.0, EPS, 0.01)},
        "over_taxed_frac": boot_stat(rows, f_over),
        "mean_d_in": boot_stat(rows, f_mean_in) if len(di) else {"n": 0},
        "median_d_in": float(np.median(di)) if len(di) else None,
    }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--census", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--boot", type=int, default=N_BOOT)
    args = ap.parse_args()
    globals()["N_BOOT"] = args.boot

    rows = load(args.census)
    for r in rows:
        r["kind"] = kind_of(r)
    print(f"loaded {len(rows)} decisions, {len(_clusters(rows))} battles")

    # `CAP` is excluded from every arm: at n == 10 an increment is invisible, so those rows cannot
    # be classified taxed-vs-frozen at all. Their count is reported as a caveat with a sensitivity.
    usable = [r for r in rows if r["cls"] != "CAP"]
    cap = [r for r in rows if r["cls"] == "CAP"]

    taxed = [r for r in usable if r["cls"] == "TAXED"]
    untaxed = [r for r in usable if r["cls"] != "TAXED"]

    res = {
        "definitions": {
            "eps": EPS, "eps_rationale": "no_progress_penalty / (2*VICTORY_VALUE) = 0.15/60",
            "tax_in_phi_units": TAX_IN_PHI,
            "aligned_consequence": "d_out(t) = phi(s_{t+1}) - phi(s_t) < -eps",
            "aligned_anticipation": "d_in(t) = phi(s_t) - phi(s_{t-1}) < -eps  [REGISTERED]",
            "over_taxed": "taxed AND d_out(t) >= 0",
            "bootstrap": "percentile, clustered over battles", "n_boot": N_BOOT,
        },
        "corpus": {
            "decisions": len(rows), "battles": len(_clusters(rows)),
            "usable": len(usable), "cap_excluded": len(cap),
            "runs": sorted({r["run"] for r in rows}),
        },
    }

    # ---- 1. the tax census ------------------------------------------------------------------
    census = {}
    for c in ("TAXED", "TRAPPED", "FROZEN", "PROGRESS", "NEUTRAL", "SITOUT", "CAP", "ANOMALY"):
        sub = [r for r in rows if r["cls"] == c]
        if sub:
            census[c] = {"n": len(sub), "frac": len(sub) / len(rows)}
    res["census_by_class"] = census

    kinds = ("move", "switch", "forced_switch", "struggle")
    res["census_by_action_kind"] = {}
    for k in kinds:
        sub = [r for r in rows if r["kind"] == k]
        if not sub:
            continue
        res["census_by_action_kind"][k] = {
            "n": len(sub),
            "taxed_frac": float(np.mean([r["cls"] == "TAXED" for r in sub])),
            "share_of_all_taxed": float(sum(r["cls"] == "TAXED" for r in sub) / max(len(taxed), 1)),
        }
    res["tax_magnitude"] = {
        "per_charge": -NO_PROGRESS_PENALTY,
        "charges_per_battle": len(taxed) / max(len(_clusters(rows)), 1),
        "reward_per_battle": -NO_PROGRESS_PENALTY * len(taxed) / max(len(_clusters(rows)), 1),
        "as_frac_of_a_win": NO_PROGRESS_PENALTY * len(taxed)
                            / max(len(_clusters(rows)), 1) / 30.0,
        "taxed_frac_of_all_decisions": len(taxed) / len(rows),
    }
    # phase distribution (game turn) of the charge
    tt = np.array([r["turn"] for r in taxed if r["turn"]], dtype=float)
    at = np.array([r["turn"] for r in rows if r["turn"]], dtype=float)
    res["tax_magnitude"]["turn_percentiles_taxed"] = \
        {f"p{p}": float(np.percentile(tt, p)) for p in (10, 25, 50, 75, 90)}
    res["tax_magnitude"]["turn_percentiles_all"] = \
        {f"p{p}": float(np.percentile(at, p)) for p in (10, 25, 50, 75, 90)}

    # ---- 2. alignment, stratified by action kind --------------------------------------------
    res["arms_overall"] = {
        "TAXED": arm_summary(taxed, "TAXED"),
        "UNTAXED_all": arm_summary(untaxed, "UNTAXED_all"),
    }
    res["arms_by_kind"] = {}
    res["contrasts_by_kind"] = {}
    for k in kinds:
        tk = [r for r in taxed if r["kind"] == k]
        uk = [r for r in untaxed if r["kind"] == k]
        if len(tk) < 50 or len(uk) < 50:
            continue
        res["arms_by_kind"][k] = {"TAXED": arm_summary(tk, f"TAXED/{k}"),
                                  "UNTAXED": arm_summary(uk, f"UNTAXED/{k}")}
        res["contrasts_by_kind"][k] = {
            "mean_d_out": boot_contrast(tk, uk, f_mean_out),
            "aligned_consequence": boot_contrast(tk, uk, mk_align_out(EPS)),
            "aligned_anticipation": boot_contrast(tk, uk, mk_align_in(EPS)),
        }

    # ---- 3. the matched control: same battle, within +-3 game turns --------------------------
    by_batt = defaultdict(list)
    for r in usable:
        by_batt[(r["run"], r["step"], r["opponent"], r["battle"])].append(r)
    paired = []
    for key, rs in by_batt.items():
        for r in rs:
            if r["cls"] != "TAXED" or r["turn"] is None:
                continue
            ctrl = [q for q in rs if q["cls"] != "TAXED" and q["kind"] == r["kind"]
                    and q["turn"] is not None and abs(q["turn"] - r["turn"]) <= 3]
            if ctrl:
                ca = [d_in(q) < -EPS for q in ctrl if q["phi_prev"] is not None]
                paired.append({
                    **r,
                    "_pair_delta": d_out(r) - float(np.mean([d_out(q) for q in ctrl])),
                    "_pair_align": float(d_out(r) < -EPS)
                    - float(np.mean([d_out(q) < -EPS for q in ctrl])),
                    "_pair_antic": (None if (r["phi_prev"] is None or not ca)
                                    else float(d_in(r) < -EPS) - float(np.mean(ca))),
                })
    res["matched_control_same_battle_pm3turns_same_kind"] = {
        "n_pairs": len(paired),
        "mean_paired_d_out_diff": boot_stat(paired, lambda r: r["_pair_delta"]),
        "mean_paired_alignment_diff": boot_stat(paired, lambda r: r["_pair_align"]),
        "mean_paired_anticipation_diff": boot_stat(
            paired, lambda r: r.get("_pair_antic")),
    }

    # ---- 3b. what the tax IMPLIES about switching, vs what the head says ---------------------
    # At a switch NO available action can produce our-attributed damage / status / hazards, so the
    # clock's escape routes are opponent-determined. The charge is therefore near-constant across
    # the whole switch branch: it functions as a FLAT tax on switching rather than a discriminator
    # between switches. This cell prices that against the head's own switch-vs-attack verdict.
    vol_sw = [r for r in usable if r["kind"] == "switch"]
    mv = [r for r in usable if r["kind"] == "move"]
    p_tax_sw = float(np.mean([r["cls"] == "TAXED" for r in vol_sw]))
    p_tax_mv = float(np.mean([r["cls"] == "TAXED" for r in mv]))
    res["implied_switch_tax"] = {
        "p_taxed_given_voluntary_switch": p_tax_sw,
        "p_taxed_given_move": p_tax_mv,
        "expected_charge_switch": -NO_PROGRESS_PENALTY * p_tax_sw,
        "expected_charge_move": -NO_PROGRESS_PENALTY * p_tax_mv,
        "implied_reward_differential_against_switching":
            -NO_PROGRESS_PENALTY * (p_tax_sw - p_tax_mv),
        "implied_in_phi_units": -NO_PROGRESS_PENALTY * (p_tax_sw - p_tax_mv) / VICTORY_SPAN,
        "head_mean_d_out_switch_minus_move": boot_contrast(vol_sw, mv, f_mean_out),
    }

    # ---- 3c. FROZEN split: the HEAL-WAR grace vs an exogenous RNG denial ---------------------
    # `HEAL_FREEZE_GRACE` forgives the first 2 consecutive productive heals; `_denial_kind`
    # freezes misses / cants / blocked attacks. Only the first is the mission's "no-progress heal
    # loop"; the second is an RNG denial the rule is RIGHT to exempt. Split on the chosen move.
    HEALS = {"milkdrink", "moonlight", "morningsun", "recover", "rest", "slackoff",
             "softboiled", "swallow", "synthesis", "wish"}
    frozen = [r for r in usable if r["cls"] == "FROZEN"]
    fz_heal = [r for r in frozen if (r["chosen"] or "") in HEALS]
    fz_other = [r for r in frozen if (r["chosen"] or "") not in HEALS]
    res["frozen_split"] = {
        "heal_grace": arm_summary(fz_heal, "FROZEN/heal-in-grace"),
        "other_denial": arm_summary(fz_other, "FROZEN/exogenous-denial"),
        "heal_grace_vs_TAXED_mean_d_out": boot_contrast(fz_heal, taxed, f_mean_out),
    }

    # ---- 3d. PROTECT, the single most over-taxed move ----------------------------------------
    pr_tax = [r for r in usable if r["chosen"] == "protect" and r["cls"] == "TAXED"]
    pr_un = [r for r in usable if r["chosen"] == "protect" and r["cls"] != "TAXED"]
    if len(pr_tax) >= 50 and len(pr_un) >= 50:
        res["protect_cell"] = {
            "TAXED": arm_summary(pr_tax, "protect/TAXED"),
            "UNTAXED": arm_summary(pr_un, "protect/UNTAXED"),
            "contrast_mean_d_out": boot_contrast(pr_tax, pr_un, f_mean_out),
        }

    # ---- 3e. battle level: does the taxed RATE predict losing? -------------------------------
    # If the hand-coded rule targets losing behaviour, battles carrying more charges should be
    # lost more. One row per battle, so the bootstrap here is an ordinary one over battles.
    batt = []
    for key, rs in by_batt.items():
        batt.append({"run": rs[0]["run"], "step": rs[0]["step"], "opponent": rs[0]["opponent"],
                     "battle": rs[0]["battle"], "result": rs[0]["result"],
                     "rate": float(np.mean([q["cls"] == "TAXED" for q in rs])),
                     "count": float(sum(q["cls"] == "TAXED" for q in rs)),
                     "n": len(rs)})
    wins = [b for b in batt if b["result"] == "WIN"]
    loss = [b for b in batt if b["result"] == "LOSS"]
    res["battle_level_outcome"] = {
        "n_battles": len(batt), "wins": len(wins), "losses": len(loss),
        "taxed_rate_in_wins": boot_stat(wins, lambda b: b["rate"]),
        "taxed_rate_in_losses": boot_stat(loss, lambda b: b["rate"]),
        "rate_loss_minus_win": boot_contrast(loss, wins, lambda b: b["rate"]),
        "count_in_wins": boot_stat(wins, lambda b: b["count"]),
        "count_in_losses": boot_stat(loss, lambda b: b["count"]),
        "count_loss_minus_win": boot_contrast(loss, wins, lambda b: b["count"]),
    }

    # ---- 4. the UNDER-tax hunt: stall-shaped windows the rule EXEMPTS ------------------------
    res["under_tax_arms"] = {}
    for c in ("FROZEN", "TRAPPED", "SITOUT"):
        sub = [r for r in usable if r["cls"] == c]
        if len(sub) < 50:
            continue
        res["under_tax_arms"][c] = arm_summary(sub, c)
        res["under_tax_arms"][c]["vs_TAXED_mean_d_out"] = boot_contrast(sub, taxed, f_mean_out)
    # the sharpest under-tax cell: an in-grace / denied window on a MOVE (a heal-war turn the
    # HEAL_FREEZE_GRACE forgives) against a taxed move.
    fz_move = [r for r in usable if r["cls"] == "FROZEN" and r["kind"] == "move"]
    tx_move = [r for r in taxed if r["kind"] == "move"]
    if len(fz_move) >= 50:
        res["under_tax_arms"]["FROZEN_move_vs_TAXED_move"] = boot_contrast(
            fz_move, tx_move, f_mean_out)

    # ---- 5. splits ---------------------------------------------------------------------------
    def split(rowset, keyfn, label):
        out = {}
        for k in sorted({keyfn(r) for r in rowset}):
            sub = [r for r in rowset if keyfn(r) == k]
            if len(sub) < 100:
                continue
            out[str(k)] = {"n": len(sub),
                           "mean_d_out": boot_stat(sub, f_mean_out, n_boot=1200),
                           "aligned_consequence": boot_stat(sub, mk_align_out(EPS), n_boot=1200),
                           "aligned_anticipation": boot_stat(sub, mk_align_in(EPS), n_boot=1200),
                           "over_taxed_frac": boot_stat(sub, f_over, n_boot=1200)}
        return out

    res["splits_taxed"] = {
        "by_result": split(taxed, lambda r: r["result"], "result"),
        "by_opponent_class": split(taxed, lambda r: r["opponent"].split("_")[0], "opp"),
        "by_run": split(taxed, lambda r: r["run"], "run"),
    }
    # the same splits on the switch cell only, where the mass is
    tx_sw = [r for r in taxed if r["kind"] in ("switch", "forced_switch")]
    res["splits_taxed_switch_only"] = {
        "by_result": split(tx_sw, lambda r: r["result"], "result"),
        "by_opponent_class": split(tx_sw, lambda r: r["opponent"].split("_")[0], "opp"),
    }

    # ---- 6. the over-tax set, characterized --------------------------------------------------
    over = [r for r in taxed if d_out(r) >= 0.0]
    comp = defaultdict(int)
    for r in over:
        comp[r["kind"]] += 1
    top_moves = defaultdict(int)
    for r in over:
        if r["kind"] == "move":
            top_moves[r["chosen"]] += 1
    res["over_tax_set"] = {
        "n": len(over), "frac_of_taxed": len(over) / max(len(taxed), 1),
        "by_action_kind": dict(comp),
        "mean_d_out": float(np.mean([d_out(r) for r in over])),
        "top_moves": sorted(top_moves.items(), key=lambda kv: -kv[1])[:20],
        "strictly_rising_frac": float(np.mean([d_out(r) > EPS for r in taxed])),
    }
    strong = [r for r in taxed if d_out(r) > EPS]
    res["over_tax_set"]["strictly_rising"] = {
        "n": len(strong), "mean_d_out": float(np.mean([d_out(r) for r in strong])) if strong else None}

    # ---- 7. CAP sensitivity ------------------------------------------------------------------
    if cap:
        res["cap_sensitivity"] = {
            "n": len(cap), "frac_of_all": len(cap) / len(rows),
            "mean_d_out": float(np.mean([d_out(r) for r in cap])),
            "aligned_consequence_eps": float(np.mean([d_out(r) < -EPS for r in cap])),
            "note": "at n==10 an increment is invisible, so these rows are taxed-or-frozen and "
                    "cannot be told apart from the obs; excluded from every arm.",
        }

    with open(args.out, "w") as fh:
        json.dump(res, fh, indent=1)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
