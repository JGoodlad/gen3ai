"""M4 analysis — turn the dual-scored decision rows into the two AXIS TABLES and the verdict.

Input : the ``*.jsonl.gz`` decision rows written by ``v8_fold_behavioral_fingerprint_probe.py``.
Output: ``v8_fold_behavioral_fingerprint_2026-08-31.{json,md}``.

THE UNIT OF MEASUREMENT is a decision on an IDENTICAL BOARD. Every row carries the acting arm's
argmax and the OTHER arm's argmax on the same (obs, mask), so an axis effect is a PAIRED
difference over the same rows — never a comparison of two different games.

    delta_j = mean_rows[ ind_j(fold_action) ] - mean_rows[ ind_j(parent_action) ]

Rows come from BOTH arms' trajectories, so the pooled delta is measured over a mixture of the
two state distributions; the per-arm split is reported beside it, because an axis whose sign
flips between the two is measuring the state shift rather than the policy change. The REALIZED
delta (each arm's rate on its OWN trajectories) is reported too: realized − identical-board is
the state-distribution contribution, and realized is the behaviour that actually produced the
win-rate change.

CIs are a cluster bootstrap over TEAMS — the unit probe P's claim generalises over, and the unit
this one must match to be comparable to it.

Run:
  python designs/research_state/measurements/v8_fold_behavioral_fingerprint_analyze.py \
      --rows '/tmp/m4/run_v8_*.jsonl.gz' --slice-name v8
  (in a linked worktree, first: export PYTHONPATH=$PYTHONPATH:src)
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import math
import sys
from collections import Counter, defaultdict

import numpy as np

CLASSES = ("SWITCH", "ATTACK", "SETUP", "RECOVER", "STATUS", "PROTECT", "PHAZE", "HAZARD",
           "OTHER_STATUS", "STRUGGLE")


# ---------------------------------------------------------------------------------------------
# Axis definitions. Each axis = (filter over the row, indicator over a chosen action index).
# The filter must NOT depend on which arm chose — that is what keeps the difference paired.
# ---------------------------------------------------------------------------------------------
def _free(r) -> bool:
    """A genuine move-or-switch choice exists: not a forced switch, not trapped, and both
    families are available. Forced switches and trapped turns are scored by their own axes."""
    return (not r["force_switch"]) and (not r["trapped"]) and r["n_sw"] > 0 and r["n_mv"] > 0


def _cls_ind(name):
    return lambda r, i: 1.0 if r["cls"][i] == name else 0.0


def _has_se_attack(r) -> bool:
    return any(c == "ATTACK" and e >= 2.0 for c, e in zip(r["cls"], r["eff"]))


def _attacks(r):
    return [i for i in range(11) if r["cls"][i] == "ATTACK"]


def _best_dmg_set(r):
    """Indices of the attacks maximising base_power * type effectiveness — the model-free
    'hit hardest' proxy. Ties all count as correct."""
    at = _attacks(r)
    if not at:
        return set()
    sc = {i: r["bp"][i] * max(r["eff"][i], 1e-9) for i in at}
    top = max(sc.values())
    return {i for i in at if sc[i] >= top - 1e-9}


def _has_resist_switch(r) -> bool:
    return any(r["cls"][i] == "SWITCH" and r["sw_resist"][i] > 0 for i in range(6))


AXES = [
    # --- A. action-class shares on a free choice --------------------------------------------
    ("switch_rate", _free, _cls_ind("SWITCH"), "A"),
    ("attack_rate", _free, _cls_ind("ATTACK"), "A"),
    ("setup_rate", _free, _cls_ind("SETUP"), "A"),
    ("recover_rate", _free, _cls_ind("RECOVER"), "A"),
    ("status_rate", _free, _cls_ind("STATUS"), "A"),
    ("protect_rate", _free, _cls_ind("PROTECT"), "A"),
    ("phaze_rate", _free, _cls_ind("PHAZE"), "A"),
    ("hazard_rate", _free, _cls_ind("HAZARD"), "A"),
    ("other_status_rate", _free, _cls_ind("OTHER_STATUS"), "A"),
    # --- B. conditional switching -------------------------------------------------------------
    ("switch|losing_matchup", lambda r: _free(r) and r["matchup"] == "losing",
     _cls_ind("SWITCH"), "B"),
    ("switch|even_matchup", lambda r: _free(r) and r["matchup"] == "even",
     _cls_ind("SWITCH"), "B"),
    ("switch|winning_matchup", lambda r: _free(r) and r["matchup"] == "winning",
     _cls_ind("SWITCH"), "B"),
    ("switch|low_hp(<1/3)", lambda r: _free(r) and r["our_hp"] < 1 / 3, _cls_ind("SWITCH"), "B"),
    ("switch|high_hp(>2/3)", lambda r: _free(r) and r["our_hp"] > 2 / 3, _cls_ind("SWITCH"), "B"),
    ("switch|early(turn<=8)", lambda r: _free(r) and r["turn"] <= 8, _cls_ind("SWITCH"), "B"),
    ("switch|late(turn>=20)", lambda r: _free(r) and r["turn"] >= 20, _cls_ind("SWITCH"), "B"),
    ("switch|opp_low_hp(<1/3)", lambda r: _free(r) and r["opp_hp"] < 1 / 3,
     _cls_ind("SWITCH"), "B"),
    ("switch|we_are_boosted", lambda r: _free(r) and r["our_boosted"], _cls_ind("SWITCH"), "B"),
    ("switch|behind_on_mons", lambda r: _free(r) and r["our_alive"] < r["opp_alive"],
     _cls_ind("SWITCH"), "B"),
    ("switch|ahead_on_mons", lambda r: _free(r) and r["our_alive"] > r["opp_alive"],
     _cls_ind("SWITCH"), "B"),
    # --- C. move quality, conditional on the option existing ---------------------------------
    ("take_SE_attack|SE_available", lambda r: _free(r) and _has_se_attack(r),
     lambda r, i: 1.0 if (r["cls"][i] == "ATTACK" and r["eff"][i] >= 2.0) else 0.0, "C"),
    ("take_best_damage|>=2_attacks", lambda r: _free(r) and len(_attacks(r)) >= 2,
     lambda r, i: 1.0 if i in _best_dmg_set(r) else 0.0, "C"),
    ("attack_at_all|>=2_attacks", lambda r: _free(r) and len(_attacks(r)) >= 2,
     _cls_ind("ATTACK"), "C"),
    # --- D. switch-target quality -------------------------------------------------------------
    ("switch_to_resist|resist_avail", lambda r: _free(r) and _has_resist_switch(r),
     lambda r, i: 1.0 if (r["cls"][i] == "SWITCH" and r["sw_resist"][i] > 0) else 0.0, "D"),
    # --- E. the FORCED replacement (a different decision problem: who comes in after a KO) ----
    ("forced_repl_resists|resist_avail",
     lambda r: r["force_switch"] and r["n_sw"] > 0 and _has_resist_switch(r),
     lambda r, i: 1.0 if (r["cls"][i] == "SWITCH" and r["sw_resist"][i] > 0) else 0.0, "E"),
]

AXIS_NAMES = [a[0] for a in AXES]


def load_rows(patterns: list[str]) -> list[dict]:
    """Read the decision rows, TOLERATING a truncated tail.

    The producer flushes with ``Z_SYNC_FLUSH`` after every cell, so a file being written (or
    left by a killed shard) is readable up to the last flush but carries no end-of-stream
    marker. Refusing such a file would make the analysis unavailable exactly while the probe is
    running; dropping the incomplete tail loses at most one partial line."""
    rows = []
    for pat in patterns:
        for path in sorted(glob.glob(pat)):
            n0 = len(rows)
            try:
                with gzip.open(path, "rt") as f:
                    for ln in f:
                        ln = ln.strip()
                        if ln:
                            try:
                                rows.append(json.loads(ln))
                            except json.JSONDecodeError:
                                pass  # a half-written final line
            except (EOFError, OSError) as e:
                print(f"[m4] {path}: truncated stream ({type(e).__name__}) — kept "
                      f"{len(rows) - n0} rows", flush=True)
    return rows


def fold_parent(r) -> tuple[int, int]:
    """(fold_action, parent_action) for this row, whichever arm was acting."""
    if r["arm"] == "fold":
        return r["act_idx"], r["oth_idx"]
    return r["oth_idx"], r["act_idx"]


def axis_table(rows: list[dict], reps: int = 4000, seed: int = 20260831) -> list[dict]:
    """Per-axis paired delta with a cluster bootstrap over TEAMS."""
    by_team: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_team[r["team"]].append(r)
    teams = sorted(by_team)
    rng = np.random.default_rng(seed)
    boot_idx = [rng.integers(0, len(teams), len(teams)) for _ in range(reps)]

    out = []
    for name, filt, ind, group in AXES:
        # per-team (sum_fold, sum_parent, n) — the bootstrap resamples teams, so keep per-team
        # partials rather than per-row values.
        per_team = {}
        for t in teams:
            sf = sp = 0.0
            n = 0
            sf_arm = {"parent": 0.0, "fold": 0.0}
            sp_arm = {"parent": 0.0, "fold": 0.0}
            n_arm = {"parent": 0, "fold": 0}
            for r in by_team[t]:
                if not filt(r):
                    continue
                fi, pi = fold_parent(r)
                vf, vp = ind(r, fi), ind(r, pi)
                sf += vf
                sp += vp
                n += 1
                sf_arm[r["arm"]] += vf
                sp_arm[r["arm"]] += vp
                n_arm[r["arm"]] += 1
            per_team[t] = (sf, sp, n, sf_arm, sp_arm, n_arm)

        n_tot = sum(v[2] for v in per_team.values())
        if n_tot < 30:
            out.append({"axis": name, "group": group, "n": n_tot, "skipped": "n<30"})
            continue
        # equal-weight-over-teams point estimate (matches probe P's per-team mean convention)
        live = [t for t in teams if per_team[t][2] > 0]
        point = float(np.mean([(per_team[t][0] - per_team[t][1]) / per_team[t][2] for t in live]))
        base = float(np.mean([per_team[t][1] / per_team[t][2] for t in live]))
        draws = np.empty(reps)
        for k, idx in enumerate(boot_idx):
            num = 0.0
            m = 0
            for j in idx:
                t = teams[j]
                if per_team[t][2] == 0:
                    continue
                num += (per_team[t][0] - per_team[t][1]) / per_team[t][2]
                m += 1
            draws[k] = num / m if m else np.nan
        draws = draws[~np.isnan(draws)]
        lo, hi = float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))
        sd = float(draws.std(ddof=1)) if len(draws) > 2 else float("nan")
        # per-arm (state-distribution) split, pooled over rows
        arm_split = {}
        for arm in ("parent", "fold"):
            na = sum(per_team[t][5][arm] for t in teams)
            if na >= 20:
                fa = sum(per_team[t][3][arm] for t in teams) / na
                pa = sum(per_team[t][4][arm] for t in teams) / na
                arm_split[arm] = {"n": na, "delta": round(fa - pa, 5)}
        # REALIZED: each arm's own rate on its OWN trajectories (pooled over rows)
        rf = rp = 0.0
        nf = np_ = 0
        for r in rows:
            if not filt(r):
                continue
            if r["arm"] == "fold":
                rf += ind(r, r["act_idx"])
                nf += 1
            else:
                rp += ind(r, r["act_idx"])
                np_ += 1
        realized = (rf / nf - rp / np_) if (nf > 20 and np_ > 20) else float("nan")
        out.append({
            "axis": name, "group": group, "n": n_tot, "n_teams": len(live),
            "parent_rate": round(base, 5), "delta": round(point, 5),
            "ci": [round(lo, 5), round(hi, 5)],
            "z": round(point / sd, 3) if sd and sd > 0 else None,
            "arm_split": arm_split,
            "realized_delta": None if math.isnan(realized) else round(realized, 5),
        })
    return out


def divergence(rows: list[dict]) -> dict:
    """How much the two policies differ at all, and where the flip mass sits."""
    dis = 0
    n = 0
    kl = []
    dv = []
    flips: Counter = Counter()
    flip_by_group: Counter = Counter()
    for r in rows:
        fi, pi = fold_parent(r)
        n += 1
        if fi != pi:
            dis += 1
            flips[(r["cls"][pi] or "?", r["cls"][fi] or "?")] += 1
        p = np.asarray(r["act_p"] if r["arm"] == "fold" else r["oth_p"])
        q = np.asarray(r["act_p"] if r["arm"] == "parent" else r["oth_p"])
        m = (p > 0) & (q > 0)
        if m.any():
            kl.append(float((p[m] * np.log(p[m] / q[m])).sum()))
        dv.append(abs(r["act_v"] - r["oth_v"]))
    tot_flip = sum(flips.values()) or 1
    sw_involved = sum(v for (a, b), v in flips.items() if "SWITCH" in (a, b))
    for (a, b), v in flips.items():
        flip_by_group[f"{a}->{b}"] = v

    # WITHIN-FAMILY target agreement. A `SWITCH->SWITCH` flip is invisible to every rate axis
    # (the class share is unchanged) yet it is the single largest flip bucket, so it gets its
    # own reading: given that BOTH arms want to switch, do they want the same mon?
    both_sw = same_sw = both_mv = same_mv = 0
    for r in rows:
        fi, pi = fold_parent(r)
        if fi < 6 and pi < 6:
            both_sw += 1
            same_sw += int(fi == pi)
        elif 6 <= fi < 10 and 6 <= pi < 10:
            both_mv += 1
            same_mv += int(fi == pi)

    # Per-class FLOW: how much mass each class gains or loses, and how leaky each class is.
    per_class = {}
    for c in CLASSES:
        npar = sum(1 for r in rows if r["cls"][fold_parent(r)[1]] == c)
        nfol = sum(1 for r in rows if r["cls"][fold_parent(r)[0]] == c)
        if npar == 0 and nfol == 0:
            continue
        out_flips = sum(v for (a, _b), v in flips.items() if a == c)
        per_class[c] = {"parent_n": npar, "fold_n": nfol, "net": nfol - npar,
                        "flip_out_rate": round(out_flips / npar, 4) if npar else None}
    # GAME LENGTH — a realized (not identical-board) signature, and the cheapest read on whether
    # the fold shifted the tempo/attrition balance at all.
    per_tag: Counter = Counter()
    tag_arm: dict[str, str] = {}
    for r in rows:
        per_tag[r["tag"]] += 1
        tag_arm[r["tag"]] = r["arm"]
    length = {}
    for arm in ("parent", "fold"):
        v = [c for t, c in per_tag.items() if tag_arm[t] == arm]
        if v:
            length[arm] = {"battles": len(v), "mean_decisions": round(float(np.mean(v)), 2)}
    return {
        "n_decisions": n,
        "argmax_disagreement": round(dis / max(n, 1), 5),
        "battle_length": length,
        "mean_KL_fold_parent": round(float(np.mean(kl)), 5) if kl else None,
        "mean_abs_dV": round(float(np.mean(dv)), 5) if dv else None,
        "flip_mass_involving_SWITCH": round(sw_involved / tot_flip, 5),
        "both_switch_n": both_sw,
        "same_switch_target_rate": round(same_sw / both_sw, 4) if both_sw else None,
        "both_move_n": both_mv,
        "same_move_slot_rate": round(same_mv / both_mv, 4) if both_mv else None,
        "per_class_flow": per_class,
        "top_flip_transitions": [
            {"transition": k, "n": v, "share": round(v / tot_flip, 4)}
            for k, v in flip_by_group.most_common(12)
        ],
    }


def team_axis_matrix(rows: list[dict], names: list[str]) -> tuple[list[str], np.ndarray,
                                                                  np.ndarray]:
    """Per-TEAM axis deltas, computed in ONE pass over the rows.

    Returns ``(teams, delta[team, axis], n[team, axis])`` with NaN where a team has no rows on
    that axis. A per-axis re-scan would be 25 passes; the split-half reliability calls this
    dozens of times, so the one-pass form is what makes the control affordable."""
    idx = {a[0]: a for a in AXES}
    specs = [idx[nm] for nm in names]
    acc: dict[str, np.ndarray] = {}
    for r in rows:
        t = r["team"]
        a = acc.get(t)
        if a is None:
            a = acc[t] = np.zeros((len(specs), 3))
        fi, pi = fold_parent(r)
        for j, (_, filt, ind, _g) in enumerate(specs):
            if not filt(r):
                continue
            a[j, 0] += ind(r, fi)
            a[j, 1] += ind(r, pi)
            a[j, 2] += 1
    teams = sorted(acc)
    n = np.array([acc[t][:, 2] for t in teams])
    with np.errstate(invalid="ignore", divide="ignore"):
        d = np.array([(acc[t][:, 0] - acc[t][:, 1]) for t in teams]) / np.where(n > 0, n, np.nan)
    return teams, d, n


def axis_vector(rows: list[dict], names: list[str]) -> np.ndarray:
    """The equal-weight-over-teams axis vector (point estimate, no bootstrap)."""
    _, d, _n = team_axis_matrix(rows, names)
    if d.size == 0:
        return np.zeros(len(names))
    with np.errstate(invalid="ignore"):
        v = np.nanmean(d, axis=0)
    return np.nan_to_num(v)


def split_half_reliability(rows: list[dict], names: list[str], draws: int = 200,
                           seed: int = 5) -> float:
    """How well does this slice's axis vector correlate WITH ITSELF across a team split?

    The ceiling any cross-slice cosine can reach. Two noisy vectors agree less than one, so a
    cosine(untaught, taught) of 0.6 means something quite different when the vectors are
    individually reliable at 0.95 than at 0.65 — the matched-noise control this programme has
    already been bitten by once (a −0.020 that became +0.206)."""
    teams = sorted({r["team"] for r in rows})
    if len(teams) < 4:
        return float("nan")
    tlist, d, _n = team_axis_matrix(rows, names)
    if len(tlist) < 4:
        return float("nan")
    rng = np.random.default_rng(seed)
    cos = []
    for _ in range(draws):
        perm = rng.permutation(len(tlist))
        h = len(perm) // 2
        with np.errstate(invalid="ignore"):
            va = np.nan_to_num(np.nanmean(d[perm[:h]], axis=0))
            vb = np.nan_to_num(np.nanmean(d[perm[h:]], axis=0))
        na, nb = np.linalg.norm(va), np.linalg.norm(vb)
        if na > 0 and nb > 0:
            cos.append(float(va @ vb / (na * nb)))
    return float(np.mean(cos)) if cos else float("nan")


def shape_compare(u: list[dict], t: list[dict], reps: int = 4000, seed: int = 7,
                  u_rows: list[dict] | None = None, t_rows: list[dict] | None = None) -> dict:
    """Is the untaught fingerprint a WEAKER VERSION of the taught one, or a DIFFERENT one?

    Three readings of one question, because no single statistic settles it:
      - cosine(u, t): SHAPE agreement, scale-free. ~1 = same axes in the same proportions.
      - k = <u,t>/<t,t>: the through-origin slope. 'weaker version' predicts 0 < k < 1.
      - sign agreement: the cheapest, most robust reading, and the one a single dominant axis
        cannot manufacture.
    Reported for the full axis set and for the class-share block alone (group A), which is a
    proper compositional vector and cannot be dominated by a conditional axis's base rate.
    """
    ud = {r["axis"]: r for r in u if "delta" in r}
    td = {r["axis"]: r for r in t if "delta" in r}
    # A DEGENERATE axis (both slices exactly 0 — e.g. a move class no probe team carries) is
    # dropped: it contributes nothing to the cosine but would pad the sign-agreement denominator
    # with free agreements, which is exactly how a shape statistic gets talked up.
    shared = [a for a in AXIS_NAMES if a in ud and a in td
              and not (abs(ud[a]["delta"]) < 1e-9 and abs(td[a]["delta"]) < 1e-9)]

    def block(names):
        uu = np.array([ud[a]["delta"] for a in names])
        tt = np.array([td[a]["delta"] for a in names])
        nu, nt = np.linalg.norm(uu), np.linalg.norm(tt)
        cos = float(uu @ tt / (nu * nt)) if nu > 0 and nt > 0 else float("nan")
        k = float(uu @ tt / (tt @ tt)) if nt > 0 else float("nan")
        sign = int(sum(1 for x, y in zip(uu, tt) if x * y > 0))
        # a permutation null for cosine: shuffle the untaught vector's axis assignment
        rng = np.random.default_rng(seed)
        null = np.array([float(rng.permutation(uu) @ tt / (nu * nt)) for _ in range(reps)])
        resid = uu - k * tt
        r2 = float(1 - (resid @ resid) / (uu @ uu)) if nu > 0 else float("nan")
        ru = split_half_reliability(u_rows, names) if u_rows else float("nan")
        rt = split_half_reliability(t_rows, names) if t_rows else float("nan")
        dis = (cos / math.sqrt(ru * rt)) if (ru == ru and rt == rt and ru > 0 and rt > 0) \
            else float("nan")
        return {"axes": names, "n_axes": len(names), "cosine": round(cos, 4),
                "slope_through_origin": round(k, 4),
                "r2_through_origin": round(r2, 4),
                "sign_agreement": f"{sign}/{len(names)}",
                "norm_untaught": round(float(nu), 5), "norm_taught": round(float(nt), 5),
                "norm_ratio_u_over_t": round(float(nu / nt), 4) if nt > 0 else None,
                "cosine_perm_null_p": round(float((null >= cos).mean()), 4),
                "cosine_perm_null_mean": round(float(null.mean()), 4),
                "split_half_reliability_untaught": round(ru, 4) if ru == ru else None,
                "split_half_reliability_taught": round(rt, 4) if rt == rt else None,
                "disattenuated_cosine": round(dis, 4) if dis == dis else None}

    a_only = [a for a in shared if ud[a]["group"] == "A"]
    return {"all_axes": block(shared), "class_share_block_A": block(a_only),
            "per_axis": [{"axis": a, "untaught": ud[a]["delta"], "taught": td[a]["delta"],
                          "ratio_u_over_t": (round(ud[a]["delta"] / td[a]["delta"], 3)
                                             if abs(td[a]["delta"]) > 1e-6 else None)}
                         for a in shared]}


def per_team_attribution(rows: list[dict], probe_p_json: str) -> dict:
    """Do the per-team behavioural deltas track the per-team WIN-RATE deltas probe P measured?

    CO-OCCURRENCE, never causation — 16 teams is 16 points, and the two quantities are measured
    on the same battles. Reported with a team bootstrap so the width is visible.
    """
    try:
        pp = json.load(open(probe_p_json))["per_team"]
    except Exception as e:
        return {"unavailable": f"{type(e).__name__}: {e}"}
    wr = {k: v["fold_wr"] - v["parent_wr"] for k, v in pp.items() if v["kind"] == "untaught"}
    by_team: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_team[r["team"]].append(r)
    teams = [t for t in sorted(by_team) if t in wr]
    if len(teams) < 5:
        return {"unavailable": f"only {len(teams)} teams overlap probe P's per-team table"}

    out = {"n_teams": len(teams), "axes": []}
    rng = np.random.default_rng(11)
    y = np.array([wr[t] for t in teams])
    for name, filt, ind, group in AXES:
        x = []
        for t in teams:
            sf = sp = 0.0
            n = 0
            for r in by_team[t]:
                if not filt(r):
                    continue
                fi, pi = fold_parent(r)
                sf += ind(r, fi)
                sp += ind(r, pi)
                n += 1
            x.append((sf - sp) / n if n >= 20 else np.nan)
        x = np.array(x)
        m = ~np.isnan(x)
        if m.sum() < 6 or np.std(x[m]) < 1e-9:
            continue
        xr = np.argsort(np.argsort(x[m])).astype(float)
        yr = np.argsort(np.argsort(y[m])).astype(float)
        rho = float(np.corrcoef(xr, yr)[0, 1])
        draws = []
        idx = np.arange(m.sum())
        for _ in range(4000):
            s = rng.integers(0, len(idx), len(idx))
            if np.std(xr[s]) < 1e-9 or np.std(yr[s]) < 1e-9:
                continue
            draws.append(float(np.corrcoef(xr[s], yr[s])[0, 1]))
        lo, hi = (float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))) \
            if draws else (float("nan"), float("nan"))
        out["axes"].append({"axis": name, "spearman_rho": round(rho, 3),
                            "ci": [round(lo, 3), round(hi, 3)], "n": int(m.sum())})
    out["axes"].sort(key=lambda d: -abs(d["spearman_rho"]))
    # the divergence magnitude itself
    dis = []
    for t in teams:
        rs = by_team[t]
        d = sum(1 for r in rs if fold_parent(r)[0] != fold_parent(r)[1]) / max(len(rs), 1)
        dis.append(d)
    dis = np.array(dis)
    xr = np.argsort(np.argsort(dis)).astype(float)
    yr = np.argsort(np.argsort(y)).astype(float)
    out["disagreement_vs_wr_gain"] = {
        "spearman_rho": round(float(np.corrcoef(xr, yr)[0, 1]), 3), "n": len(teams)}
    return out


def _read_cells(cell_globs: list[str]) -> tuple[list[dict], dict]:
    """Cell records, DEDUPED by (kind, team, opp, arm), preferring the record that carries the
    per-game outcome vector.

    A cell can appear twice because an ``--outcomes-only`` pass replays the same cells to supply
    the per-game vectors a dual-scored pass predates. Greedy play on a fixed seed is
    deterministic, so the two passes MUST report the same win count — and that agreement is
    reported as a first-class determinism check, because it is exactly the assumption the
    battle-level join rests on. A disagreement invalidates the join, not merely the check."""
    raw = []
    for pat in cell_globs:
        for path in sorted(glob.glob(pat)):
            for ln in open(path):
                try:
                    raw.append(json.loads(ln))
                except json.JSONDecodeError:
                    pass
    best: dict[tuple, dict] = {}
    agree = disagree = 0
    for r in raw:
        k = (r["kind"], r["team"], r["opp"], r["arm"])
        prev = best.get(k)
        if prev is None:
            best[k] = r
            continue
        if prev["wins"] == r["wins"] and prev["finished"] == r["finished"]:
            agree += 1
        else:
            disagree += 1
        if r.get("per_game") and not prev.get("per_game"):
            best[k] = r
    check = {"duplicate_cells": agree + disagree, "replays_agreeing": agree,
             "replays_disagreeing": disagree}
    return list(best.values()), check


def cell_attribution(rows: list[dict], cells: list[dict], kind: str) -> dict:
    """Does a CELL where the fold behaved most differently also gain the most win rate?

    A cell is one (probe team, opponent team) pair — 128 of them for the untaught slice, so this
    has an order of magnitude more points than the 16-team correlation, at the price of a unit
    the claim does not directly generalise over (hence the team-clustered bootstrap).

    CO-OCCURRENCE. A cell's behavioural delta and its win-rate delta are computed on the SAME
    battles, so a common cause (a matchup that simply admits more switching AND more winning)
    reproduces any correlation here without any of the behaviour causing any of the winning."""
    wr: dict[tuple, dict] = defaultdict(dict)
    for c in cells:
        if c["kind"] != kind or c["finished"] == 0:
            continue
        wr[(c["team"], c["opp"])][c["arm"]] = c["wins"] / c["finished"]
    keys = [k for k, v in wr.items() if "parent" in v and "fold" in v]
    if len(keys) < 20:
        return {"unavailable": f"only {len(keys)} complete cells"}
    y = np.array([wr[k]["fold"] - wr[k]["parent"] for k in keys])
    by_cell: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        by_cell[(r["team"], r["opp"])].append(r)
    teams = [k[0] for k in keys]
    rng = np.random.default_rng(23)
    uniq = sorted(set(teams))

    out = {"n_cells": len(keys), "n_teams": len(uniq),
           "net_wr_delta": round(float(y.mean()), 4), "axes": []}
    for name, filt, ind, group in AXES:
        x = []
        for k in keys:
            sf = sp = 0.0
            n = 0
            for r in by_cell.get(k, ()):
                if not filt(r):
                    continue
                fi, pi = fold_parent(r)
                sf += ind(r, fi)
                sp += ind(r, pi)
                n += 1
            x.append((sf - sp) / n if n >= 15 else np.nan)
        x = np.array(x)
        m = ~np.isnan(x)
        if m.sum() < 20 or np.std(x[m]) < 1e-9:
            continue
        xr, yr = x[m], y[m]
        rho = float(np.corrcoef(np.argsort(np.argsort(xr)), np.argsort(np.argsort(yr)))[0, 1])
        # cluster bootstrap over TEAMS — cells inside a team share the pilot and are correlated
        tk = np.array(teams)[m]
        draws = []
        for _ in range(3000):
            pick = rng.choice(uniq, len(uniq), replace=True)
            sel = np.concatenate([np.where(tk == t)[0] for t in pick])
            if len(sel) < 10 or np.std(xr[sel]) < 1e-9 or np.std(yr[sel]) < 1e-9:
                continue
            draws.append(float(np.corrcoef(np.argsort(np.argsort(xr[sel])),
                                           np.argsort(np.argsort(yr[sel])))[0, 1]))
        lo, hi = (round(float(np.percentile(draws, 2.5)), 3),
                  round(float(np.percentile(draws, 97.5)), 3)) if draws else (None, None)
        out["axes"].append({"axis": name, "spearman_rho": round(rho, 3), "ci": [lo, hi],
                            "n_cells": int(m.sum())})
    out["axes"].sort(key=lambda d: -abs(d["spearman_rho"]))
    return out


def battle_attribution(rows: list[dict], cells: list[dict], kind: str) -> dict:
    """The sharpest attribution the design admits: the arms play the SAME seed at the same game
    index, so every game is a paired trial with four outcomes — FLIP_WIN (fold won where the
    parent lost), FLIP_LOSS, BOTH_WIN, BOTH_LOSS. The win-rate gain IS
    (FLIP_WIN − FLIP_LOSS)/N by identity, so asking how the fold behaved differently *inside the
    FLIP_WIN battles* relative to the BOTH_LOSS ones asks which behavioural change travelled with
    the gain rather than merely with the fold.

    Needs the per-game outcome vectors; older cell files without them return ``unavailable``."""
    by_game: dict[tuple, dict] = defaultdict(dict)
    for c in cells:
        if c["kind"] != kind or not c.get("per_game") or not c.get("tags"):
            continue
        for i, (o, t) in enumerate(zip(c["per_game"], c["tags"])):
            by_game[(c["team"], c["opp"], i)][c["arm"]] = (o, t)
    keys = [k for k, v in by_game.items()
            if "parent" in v and "fold" in v and v["parent"][0] >= 0 and v["fold"][0] >= 0]
    if len(keys) < 40:
        return {"unavailable": "per-game outcomes not recorded for this slice "
                               f"({len(keys)} paired games)"}
    tag_class: dict[str, str] = {}
    counts: Counter = Counter()
    for k in keys:
        p, f = by_game[k]["parent"][0], by_game[k]["fold"][0]
        cl = ("BOTH_WIN" if (p and f) else "BOTH_LOSS" if (not p and not f)
              else "FLIP_WIN" if f else "FLIP_LOSS")
        counts[cl] += 1
        for arm in ("parent", "fold"):
            t = by_game[k][arm][1]
            if t:
                tag_class[t] = cl
    n = len(keys)
    out = {"n_paired_games": n,
           "counts": dict(counts),
           "net_wr_delta_identity": round((counts["FLIP_WIN"] - counts["FLIP_LOSS"]) / n, 4),
           "axes": []}
    groups = {c: [r for r in rows if tag_class.get(r.get("tag")) == c]
              for c in ("FLIP_WIN", "FLIP_LOSS", "BOTH_WIN", "BOTH_LOSS")}
    for name, filt, ind, group in AXES:
        rec = {"axis": name}
        ok = True
        for c, rs in groups.items():
            sf = sp = 0.0
            m = 0
            for r in rs:
                if not filt(r):
                    continue
                fi, pi = fold_parent(r)
                sf += ind(r, fi)
                sp += ind(r, pi)
                m += 1
            if m < 40:
                ok = False
                break
            rec[c] = round((sf - sp) / m, 4)
            rec[c + "_n"] = m
        if ok:
            rec["flipwin_minus_bothloss"] = round(rec["FLIP_WIN"] - rec["BOTH_LOSS"], 4)
            out["axes"].append(rec)
    out["axes"].sort(key=lambda d: -abs(d.get("flipwin_minus_bothloss", 0)))
    return out


def winrate_check(cells: list[dict]) -> dict:
    """The IN-SITU consistency check: this probe's cells are a CRN subsample of probe P's, so
    its win-rate delta must land near probe P's +5.42pp (untaught) / +26.18pp (taught).

    It is not a re-measurement — the subsample is 4 of 30 games per cell and carries its own
    sampling error — but a delta of the wrong SIGN would mean the arms, the seeds or the cell
    identity drifted, and the behavioural tables would be describing something else."""
    out = {}
    for kind in ("untaught", "taught"):
        agg = defaultdict(lambda: [0, 0])
        defaults = redecides = decisions = 0
        for r in cells:
            if r["kind"] != kind:
                continue
            agg[r["arm"]][0] += r["wins"]
            agg[r["arm"]][1] += r["finished"]
            defaults += r.get("n_defaults", 0)
            redecides += r.get("n_redecides", 0)
            decisions += r.get("n_decisions", 0)
        if "parent" not in agg or "fold" not in agg:
            continue
        pw, pn = agg["parent"]
        fw, fn = agg["fold"]
        if pn == 0 or fn == 0:
            continue
        out[kind] = {"parent_wr": round(pw / pn, 4), "fold_wr": round(fw / fn, 4),
                     "delta_pp": round(100 * (fw / fn - pw / pn), 2),
                     "n_parent": pn, "n_fold": fn,
                     "n_defaults": defaults, "n_redecides": redecides,
                     "n_decisions": decisions}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", nargs="+", required=True, help="glob(s) of decision-row jsonl.gz")
    ap.add_argument("--cells", nargs="*", default=[], help="glob(s) of per-cell jsonl")
    ap.add_argument("--probe-p", default="designs/research_state/measurements/"
                                         "v8_redistribution_pfsp_2026-08-30.json")
    ap.add_argument("--out", default="designs/research_state/measurements/"
                                     "v8_fold_behavioral_fingerprint_2026-08-31")
    ap.add_argument("--label", default="v8")
    ap.add_argument("--merge-into", default="", help="existing json to add this family into")
    a = ap.parse_args(argv)

    rows = load_rows(a.rows)
    if not rows:
        raise SystemExit(f"[m4] no rows matched {a.rows}")
    un = [r for r in rows if r["kind"] == "untaught"]
    tg = [r for r in rows if r["kind"] == "taught"]
    print(f"[m4] {len(rows)} rows  untaught={len(un)}  taught={len(tg)}  "
          f"battles={len({r['tag'] for r in rows})}", flush=True)

    res = {"family": a.label, "n_rows": len(rows),
           "n_battles": len({r["tag"] for r in rows}),
           "n_rows_untaught": len(un), "n_rows_taught": len(tg),
           "untaught": {"axes": axis_table(un), "divergence": divergence(un),
                        "n_teams": len({r["team"] for r in un})},
           }
    # The ARCHETYPE-MATCHED control. v8's taught set is entirely stall / semi_stall, while the
    # untaught probe set is archetype-stratified — so a shape difference between them could be
    # an archetype effect wearing a taught/untaught label. This slice holds archetype fixed.
    taught_arch = {r.get("arch") for r in tg if r.get("arch")}
    und = [r for r in un if r.get("arch") in taught_arch] if taught_arch else []
    res["n_rows_untaught_defensive"] = len(und)
    if len(und) > 500:
        res["untaught_defensive"] = {
            "axes": axis_table(und), "divergence": divergence(und),
            "n_teams": len({r["team"] for r in und}),
            "matched_archetypes": sorted(taught_arch),
            "note": "untaught teams whose archetype MATCHES the taught set's own archetypes"}
    if tg:
        res["taught"] = {"axes": axis_table(tg), "divergence": divergence(tg),
                         "n_teams": len({r["team"] for r in tg})}
        res["shape"] = shape_compare(res["untaught"]["axes"], res["taught"]["axes"],
                                     u_rows=un, t_rows=tg)
        if "untaught_defensive" in res:
            res["shape_archetype_matched"] = shape_compare(
                res["untaught_defensive"]["axes"], res["taught"]["axes"],
                u_rows=und, t_rows=tg)
    res["attribution"] = per_team_attribution(un, a.probe_p)
    if a.cells:
        cells, determinism = _read_cells(a.cells)
        res["replay_determinism"] = determinism
        res["winrate_check"] = winrate_check(cells)
        print("[m4] winrate check " + json.dumps(res["winrate_check"]), flush=True)
        print("[m4] replay determinism " + json.dumps(determinism), flush=True)
        res["cell_attribution_untaught"] = cell_attribution(un, cells, "untaught")
        res["battle_attribution_untaught"] = battle_attribution(un, cells, "untaught")
        if tg:
            res["battle_attribution_taught"] = battle_attribution(tg, cells, "taught")

    payload = {}
    if a.merge_into:
        try:
            payload = json.load(open(a.merge_into))
        except Exception:
            payload = {}
    payload[a.label] = res
    with open(a.out + ".json", "w") as f:
        json.dump(payload, f, indent=1)
    print(f"[m4] wrote {a.out}.json", flush=True)

    # ---- console table (markdown, so the report can quote it verbatim) ----
    for slc in ("untaught", "untaught_defensive", "taught"):
        if slc not in res:
            continue
        print(f"\n### {a.label} · {slc.upper()} ({res[slc]['n_teams']} teams, "
              f"{res['n_rows_' + slc]} decisions)\n")
        print("| axis | n | parent rate | Δ (fold−parent) | 95% CI | z | realized Δ |")
        print("|---|---:|---:|---:|---|---:|---:|")
        tab = [x for x in res[slc]["axes"] if "delta" in x]
        for x in sorted(tab, key=lambda d: -abs(d["z"] or 0)):
            rd = x["realized_delta"]
            print(f"| `{x['axis']}` | {x['n']} | {x['parent_rate']:.3f} | "
                  f"**{x['delta']:+.4f}** | [{x['ci'][0]:+.4f}, {x['ci'][1]:+.4f}] | "
                  f"{(x['z'] or 0):+.2f} | {('%+.4f' % rd) if rd is not None else '—'} |")
        d = res[slc]["divergence"]
        print(f"\nargmax disagreement **{d['argmax_disagreement']:.3f}** · "
              f"mean KL(fold‖parent) {d['mean_KL_fold_parent']} · "
              f"flip mass involving SWITCH {d['flip_mass_involving_SWITCH']:.3f} · "
              f"same switch target when both switch {d['same_switch_target_rate']} "
              f"(n={d['both_switch_n']}) · same move slot when both move "
              f"{d['same_move_slot_rate']} (n={d['both_move_n']})")
        print("\n| parent class | parent n | fold n | net | flip-out rate |")
        print("|---|---:|---:|---:|---:|")
        for c, v in d["per_class_flow"].items():
            print(f"| {c} | {v['parent_n']} | {v['fold_n']} | {v['net']:+d} | "
                  f"{v['flip_out_rate']} |")
    for key, title in (("shape", "SHAPE — untaught vs taught"),
                       ("shape_archetype_matched",
                        "SHAPE — untaught DEFENSIVE (archetype-matched) vs taught")):
        if key in res:
            print(f"\n=== {title} ===")
            print(json.dumps({k: v for k, v in res[key].items() if k != "per_axis"}, indent=1))
    if "shape" in res:
        print("\n| axis | untaught Δ | taught Δ | u/t |")
        print("|---|---:|---:|---:|")
        for p in sorted(res["shape"]["per_axis"], key=lambda d: -abs(d["taught"])):
            print(f"| `{p['axis']}` | {p['untaught']:+.4f} | {p['taught']:+.4f} | "
                  f"{p['ratio_u_over_t'] if p['ratio_u_over_t'] is not None else '—'} |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
