"""FROZEN record (the Kakuna read of the banked entropy finals, X67 PREREG §13; DESCRIPTIVE, deciding nothing).

(1) THE HAND CHECK of rule `b1_kakuna`'s seven contrasts, from `kakuna_read.json`'s own rows and WITHOUT the rule's
    code: psi, the pooled within-arm seed variance, the team sandwich by our team, and the dyadic one as the literal
    pair sum (every pair of deal rows sharing a team in either role, the pair matrix built outright), the composition
    V = max(V_S, V_TS) + max(0, V_T - V_TS), the Satterthwaite df, the Student-t interval at the registered level and
    the class with the 0.005 pp boundary. Each is compared with the record's `decision`; a difference above 1e-9 pp,
    or another class, exits 1.
(2) THE ARM MEANS against Kakuna ("where we are"; NOT a registered statistic): each arm's mean over its runs on F,
    H and U, with (a) the pooled count, (b) the seed-only Student-t interval over the run rates, and (c) the same
    two-way composition as (1) applied to the arm mean (coefficient +1 on one arm), the wider scheme, at the
    registered level. (c) treats the panel's 72 teams as a sample of teams; (b) holds the panel's teams fixed.
(3) The games at our turn cap (booked losses) by arm, and each arm's rate with those games left out of both the
    wins and the count, as a description of how little they move.

Reads this folder's kakuna_read.json only; prints JSON (kakuna_read_by_hand.json). Not a tool: never re-run as one.
"""
import json
import math
import os
import sys

import numpy as np
from scipy.stats import t as student_t

HERE = os.path.dirname(os.path.abspath(__file__))
rec = json.load(open(os.path.join(HERE, "kakuna_read.json")))
inp, dec = rec["inputs"], rec["decision"]
ARMS = inp["arms"]
LEVEL = float(dec["level"])
EPS_PP = 0.005
F, H = "forum72", "heldout72_hpfix"
ROWS = {lab: {p: np.array(c["rows"], dtype=float).reshape(-1, 6) for p, c in by.items()}
        for lab, by in inp["cells"].items()}


def rate(lab, panel):
    r = ROWS[lab][panel]
    return r[:, 3].sum() / r[:, 2].sum()


def sandwich_our_team(psi, ours):
    """Sum over our-team clusters of (cluster sum)^2, times G / (G - 1)."""
    ids = sorted(set(ours.tolist()))
    sums = [psi[ours == i].sum() for i in ids]
    g = len(ids)
    return sum(s * s for s in sums) * g / (g - 1), g


def sandwich_dyadic(psi, ours, theirs):
    """The literal pair sum: psi_g psi_h over every pair of rows (g = h included) that share a team in any role."""
    share = ((ours[:, None] == ours[None, :]) | (ours[:, None] == theirs[None, :])
             | (theirs[:, None] == ours[None, :]) | (theirs[:, None] == theirs[None, :]))
    k = len(set(ours.tolist()) | set(theirs.tolist()))
    return float(psi @ (share @ psi)) * k / (k - 1), k


def two_way(coef, panels):
    """psi and both schemes' intervals for arm coefficients `coef` on `panels` at equal weight (proportions)."""
    pw = 1.0 / len(panels)
    w = {lab: coef[a] / len(ARMS[a]) for a in coef for lab in ARMS[a]}
    score = {lab: sum(pw * rate(lab, p) for p in panels) for lab in w}
    psi_hat = sum(w[lab] * score[lab] for lab in w)
    df_s = sum(len(ARMS[a]) - 1 for a in coef)
    ss = 0.0
    for a in coef:
        m = sum(score[lab] for lab in ARMS[a]) / len(ARMS[a])
        ss += sum((score[lab] - m) ** 2 for lab in ARMS[a])
    v_s = ss / df_s * sum(x * x for x in w.values())
    out = {"psi": psi_hat, "v_s": v_s, "df_s": df_s, "schemes": {}}
    for scheme in ("our_team", "dyadic"):
        v_t = v_ts = 0.0
        df_t = 0
        for p in panels:
            parts = []
            for lab in w:
                r = ROWS[lab][p]
                lin = w[lab] * pw * (r[:, 3] - rate(lab, p) * r[:, 2]) / r[:, 2].sum()
                parts.append((lin, r[:, 0].astype(int), r[:, 1].astype(int)))
                v_ts += (sandwich_our_team(lin, parts[-1][1]) if scheme == "our_team"
                         else sandwich_dyadic(*parts[-1]))[0]
            lin, ours, theirs = (np.concatenate([x[i] for x in parts]) for i in range(3))
            v, g = sandwich_our_team(lin, ours) if scheme == "our_team" else sandwich_dyadic(lin, ours, theirs)
            v_t += v
            df_t += g - 1
        a, b = max(v_s, v_ts), max(0.0, v_t - v_ts)
        v = a + b
        df = min(max(v * v / (a * a / df_s + b * b / df_t), df_s), df_s + df_t)
        half = float(student_t.ppf(0.5 + LEVEL / 2.0, df)) * math.sqrt(v)
        out["schemes"][scheme] = {"v_t": v_t, "v_ts": v_ts, "v": v, "df": df, "half": half}
    wide = max(("our_team", "dyadic"), key=lambda s: out["schemes"][s]["half"])
    out.update(deciding=wide, lo=psi_hat - out["schemes"][wide]["half"], hi=psi_hat + out["schemes"][wide]["half"])
    return out


def klass(lo_pp, hi_pp):
    return "POSITIVE" if lo_pp > EPS_PP else "NEGATIVE" if hi_pp < -EPS_PP else "SPANS-ZERO"


out = {"record": {"read": rec["read"], "spec_sha256": rec["spec_sha256"], "verdict": rec["verdict"]}, "level": LEVEL}

# ---------------------------------------------------------------- (1) the hand check of every contrast
worst = 0.0
agree = True
out["hand_check"] = {}
for name, c in dec["contrasts"].items():
    st = two_way(c["arms"], c["panels"])
    mine = {"psi_pp": 100 * st["psi"], "lo_pp": 100 * st["lo"], "hi_pp": 100 * st["hi"]}
    diff = max(abs(mine[k] - c[k]) for k in mine)
    for s in ("our_team", "dyadic"):
        diff = max(diff, abs(1e4 * st["schemes"][s]["v"] - c["schemes"][s]["v_pp2"]),
                   abs(st["schemes"][s]["df"] - c["schemes"][s]["df"]))
    worst = max(worst, diff)
    same = klass(mine["lo_pp"], mine["hi_pp"]) == c["class"] and st["deciding"] == c["deciding_scheme"]
    agree = agree and same
    out["hand_check"][name] = dict(mine, deciding_scheme=st["deciding"], df=st["schemes"][st["deciding"]]["df"],
                                   se_pp=100 * math.sqrt(st["schemes"][st["deciding"]]["v"]),
                                   klass=klass(mine["lo_pp"], mine["hi_pp"]), tool_class=c["class"],
                                   max_abs_diff_vs_tool=diff, same_class_and_scheme=same)
out["hand_check_summary"] = {"contrasts": len(dec["contrasts"]), "max_abs_diff_vs_tool": worst, "all_agree": agree}

# ---------------------------------------------------------------- (2) the arm means against Kakuna
out["arm_means"] = {}
for arm, labs in ARMS.items():
    out["arm_means"][arm] = {}
    for tag, panels in (("F", [F]), ("H", [H]), ("U", [F, H])):
        st = two_way({arm: 1.0}, panels)
        scores = [sum(rate(lab, p) for p in panels) / len(panels) for lab in labs]
        sd = math.sqrt(sum((x - st["psi"]) ** 2 for x in scores) / (len(labs) - 1))
        hw = float(student_t.ppf(0.5 + LEVEL / 2.0, len(labs) - 1)) * sd / math.sqrt(len(labs))
        wins = sum(int(inp["cells"][lab][p]["wins"]) for lab in labs for p in panels)
        n = sum(int(inp["cells"][lab][p]["n"]) for lab in labs for p in panels)
        out["arm_means"][arm][tag] = {
            "mean_of_runs": st["psi"], "wins": wins, "games": n, "runs": len(labs), "run_sd_pp": 100 * sd,
            "run_min": min(scores), "run_max": max(scores),
            "seed_only_interval": [st["psi"] - hw, st["psi"] + hw],
            "two_way_interval": [st["lo"], st["hi"]], "two_way_scheme": st["deciding"],
            "two_way_se_pp": 100 * math.sqrt(st["schemes"][st["deciding"]]["v"]),
            "two_way_df": st["schemes"][st["deciding"]]["df"]}

# ---------------------------------------------------------------- (3) the games at our turn cap
out["turn_cap"] = {}
for arm, labs in ARMS.items():
    caps = sum(int(inp["cells"][lab][p]["caps"]) for lab in labs for p in (F, H))
    n = sum(int(inp["cells"][lab][p]["n"]) for lab in labs for p in (F, H))
    with_caps = sum(rate(lab, p) for lab in labs for p in (F, H)) / (2 * len(labs))
    without = sum(inp["cells"][lab][p]["wins"] / (inp["cells"][lab][p]["n"] - inp["cells"][lab][p]["caps"])
                  for lab in labs for p in (F, H)) / (2 * len(labs))
    out["turn_cap"][arm] = {"caps": caps, "games": n, "share": caps / n, "u_rate": with_caps,
                            "u_rate_caps_left_out": without}
tc = out["turn_cap"]
out["turn_cap"]["contrasts_caps_left_out_pp"] = {
    "x67_U": 100 * (tc["x67"]["u_rate_caps_left_out"] - tc["c0"]["u_rate_caps_left_out"]),
    "ce3d_U": 100 * (tc["ce3d"]["u_rate_caps_left_out"] - tc["c0"]["u_rate_caps_left_out"])}

json.dump(out, sys.stdout, indent=1, sort_keys=True)
sys.stdout.write("\n")
if not agree or worst > 1e-9:
    sys.exit(1)
