"""S'-HONEST clustering addendum, step 2: the read's scores and paired gain under six dependence models.

REPORTED, never decided: the registered verdict (GO-HONEST, GOAL-B-REACHED; PREREG §5) is unchanged. This script asks
how wide the same statistics' intervals are when the 72 teams of each panel are treated as a SAMPLE (UNDERSTANDING §7
rule 10, "the team is the unit"), not as fixed. Frozen record script, never a tool.

    python3 clustering.py deals.csv clustering.json tables.md

Input: `deals.csv` (`extract_deals.py`), one row per registered deal with both sides' teams.

The statistics, each per panel and pooled over U at equal panel weight (panels are fixed strata, never resampled):
  searched_goal_b   searched win share over all 5,100 deals, a missing searched game a loss, a tie a non-win (GOAL-B's)
  raw_paired        raw win share over the 5,097 paired deals, a tie a non-win (the record's `score_u.raw`)
  searched_paired   searched win share over the same paired deals (the record's `score_u.searched`)
  G_go              the paired gain with unpaired deals imputed at the worst case for GO (decides GO-HONEST)
  G_paired          the paired gain over paired deals only (descriptive)
  G_null            the paired gain with unpaired deals imputed at the worst case for NULL
A panel statistic is sum(num) / sum(den) over its deals. In a gain a tie is half a win (PREREG §5's y).

The dependence models (README.md argues each):
  iid         every deal independent
  matchup     cluster = one (unit seed, game mod 10) team draw: the SAME two teams play games g and g + 10
  unit_seed   cluster = (panel, unit seed), 85 a panel: the registered unit
  our_team    cluster = the team WE pilot, 72 a panel
  their_team  cluster = the team Kakuna pilots, 72 a panel
  two_way     ours x theirs crossed, the two roles independent (Cameron, Gelbach & Miller 2011): V_ours + V_theirs - V_cell
  dyadic      a team is ONE cluster in either role (Aronow, Samii & Assenova 2015): two deals covary iff they share a team

One-way models: a percentile cluster bootstrap (B replicates, clusters resampled within panel) AND the analytic
cluster-robust sandwich on the ratio's influence values; the two must agree, which licenses the sandwich for the two
crossed models, where no bootstrap is unbiased (McCullagh 2000). The pigeonhole bootstrap (Owen 2007: resample both
roles independently, weight a deal by c_ours * c_theirs) is REPORTED as the known-conservative bracket: its variance
is about V_ours + V_theirs + 3 V_noise, so `pigeonhole_minus_2iid` should sit near `two_way`.
"""
import csv
import json
import math
import sys

import numpy as np

PANELS = ("heldout72_hpfix", "forum72", "p72")          # registered order: H, F, P72
SHORT = {"heldout72_hpfix": "H", "forum72": "F", "p72": "P72"}
REG_SEED, REG_B = 20261106, 2000                         # PREREG §5, reproduced as the self-check
SEED, B = 20261109, 20000                                # this addendum's
Z = 1.959963984540054
STATS = ("searched_goal_b", "raw_paired", "searched_paired", "G_go", "G_paired", "G_null")
ONE_WAY = ("iid", "matchup", "unit_seed", "our_team", "their_team")
BARS = {"searched_goal_b": 0.5, "raw_paired": 0.5, "searched_paired": 0.5, "G_go": 0.0, "G_paired": 0.0, "G_null": 0.03}
REGISTERED = {  # record/decision.json, to 1e-9: the self-check refuses before any new number is printed
    "G_paired": (0.02000515995872033, 0.003449045407636738, 0.035067466460268315),
    "G_go_lb": 0.0032328431372549006, "G_null_ub": 0.03529411764705883,
    "goal_b": (0.5145098039215686, 0.5007893328985898, 0.528208439413879),
}


def phi(z):
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


def wilson_pn(p, n, z=Z):
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return mid - half, mid + half


def load(path):
    """Per panel: cluster ids for each model and, per statistic, a (num, den) pair of per-deal arrays."""
    rows = list(csv.DictReader(open(path)))
    assert len(rows) == 5100
    out = {}
    for p in PANELS:
        rs = [r for r in rows if r["panel"] == p]
        assert len(rs) == 1700
        teams = sorted({r["our_team"].rsplit(".", 1)[0] for r in rs} | {r["their_team"].rsplit(".", 1)[0] for r in rs})
        tid = {t: i for i, t in enumerate(teams)}
        a = np.array([tid[r["our_team"].rsplit(".", 1)[0]] for r in rs])
        b = np.array([tid[r["their_team"].rsplit(".", 1)[0]] for r in rs])
        seed = np.array([int(r["unit_seed"]) for r in rs])
        game = np.array([int(r["game"]) for r in rs])
        half = {"win": 1.0, "tie": 0.5, "loss": 0.0}
        raw_ok = np.array([r["raw"] != "missing" for r in rs])
        s_ok = np.array([r["searched"] != "missing" for r in rs])
        pair = raw_ok & s_ok
        yr = np.array([half.get(r["raw"], np.nan) for r in rs])
        ys = np.array([half.get(r["searched"], np.nan) for r in rs])
        d_pair = np.where(pair, ys - yr, 0.0)
        # PREREG §5's imputation table: searched missing -> (-y_raw, 1 - y_raw); raw missing -> (y_s - 1, y_s)
        d_go = np.where(pair, ys - yr, np.where(raw_ok, -yr, ys - 1.0))
        d_null = np.where(pair, ys - yr, np.where(raw_ok, 1.0 - yr, ys))
        assert not (~raw_ok & ~s_ok).any() and not np.isnan(d_go).any() and not np.isnan(d_null).any()
        ones = np.ones(len(rs))
        pf = pair.astype(float)
        stats = {
            "searched_goal_b": (np.array([float(r["searched"] == "win") for r in rs]), ones),
            "raw_paired": (np.array([float(r["raw"] == "win") for r in rs]) * pf, pf),
            "searched_paired": (np.array([float(r["searched"] == "win") for r in rs]) * pf, pf),
            "G_go": (d_go, ones), "G_paired": (d_pair, pf), "G_null": (d_null, ones),
        }
        # games g and g + 10 of a unit seed play the SAME two teams (the two challenge halves): refuse otherwise
        key = {}
        for i in range(len(rs)):
            k = (int(seed[i]), int(game[i]) % 10)
            assert key.setdefault(k, (a[i], b[i])) == (a[i], b[i]), (p, k)
        assert len(key) == 850
        cl = {
            "iid": np.arange(len(rs)),
            "matchup": np.unique(seed * 10 + game % 10, return_inverse=True)[1],
            "unit_seed": np.unique(seed, return_inverse=True)[1],
            "our_team": a, "their_team": b,
            "cell": np.unique(a * len(teams) + b, return_inverse=True)[1],
        }
        out[p] = {"n_teams": len(teams), "a": a, "b": b, "seed": seed, "game": game, "cl": cl, "stats": stats,
                  "teams_ours": int(len(set(a))), "teams_theirs": int(len(set(b))),
                  "cells_distinct": int(cl["cell"].max() + 1), "mirror_deals": int((a == b).sum())}
    return out


def registered_check(data):
    """PREREG §5 verbatim (unit statistics, mean of unit means, `default_rng(20261106)`, B = 2,000) against the record."""
    out = {}
    for kind in ("G_paired", "G_go", "G_null"):
        rng = np.random.default_rng(REG_SEED)
        reps, G = [], []
        for p in PANELS:
            num, den = data[p]["stats"][kind]
            seed = data[p]["seed"]
            x = np.array([num[seed == s].sum() / den[seed == s].sum() for s in sorted(set(seed.tolist()))])
            assert len(x) == 85
            G.append(x.mean())
            idx = rng.integers(0, len(x), size=(REG_B, len(x)))
            reps.append(x[idx].mean(axis=1))
        lo, hi = np.percentile(np.mean(reps, axis=0), [2.5, 97.5])
        out[kind] = {"point": float(np.mean(G)), "ci95": [float(lo), float(hi)]}
    counts = {p: (int(data[p]["stats"]["searched_goal_b"][0].sum()), 1700) for p in PANELS}
    ps = [w / n for w, n in counts.values()]
    mean = math.fsum(ps) / 3
    var = math.fsum(q * (1 - q) / 1700 for q in ps) / 9
    lo, hi = wilson_pn(mean, mean * (1 - mean) / var)
    out["goal_b"] = {"counts": counts, "point": mean, "ci95": [lo, hi]}
    close = lambda x, y: abs(x - y) < 1e-9
    reg = REGISTERED
    assert all(close(x, y) for x, y in zip((out["G_paired"]["point"], *out["G_paired"]["ci95"]), reg["G_paired"])), out
    assert close(out["G_go"]["ci95"][0], reg["G_go_lb"]) and close(out["G_null"]["ci95"][1], reg["G_null_ub"]), out
    assert all(close(x, y) for x, y in zip((mean, lo, hi), reg["goal_b"])), out
    out["matches_record"] = True
    return out


def influence(num, den):
    theta = num.sum() / den.sum()
    return theta, (num - theta * den) / den.sum()


def v_cluster(psi, ids):
    """The cluster-robust sandwich with the G / (G - 1) correction (for `iid`, G = n: the usual s^2 / n)."""
    sums = np.bincount(ids, weights=psi)
    g = int((np.bincount(ids) > 0).sum())
    return float((sums ** 2).sum() * g / (g - 1))


def v_dyadic(psi, a, b, n_teams):
    """sum of psi_g * psi_h over every pair of deals sharing a team in ANY role, corrected by G / (G - 1)."""
    m = np.zeros((len(psi), n_teams))
    m[np.arange(len(psi)), a] = 1.0
    m[np.arange(len(psi)), b] = 1.0
    share = (m @ m.T) > 0
    return float(psi @ share @ psi * n_teams / (n_teams - 1))


def boot_one_way(d, ids, rng):
    """Replicates (B,) of every statistic for one panel: clusters resampled with replacement, the same draw for all."""
    g = int(ids.max() + 1)
    idx = rng.integers(0, g, size=(B, g))
    out = {}
    for s in STATS:
        num, den = d["stats"][s]
        cn, cd = np.bincount(ids, weights=num, minlength=g), np.bincount(ids, weights=den, minlength=g)
        out[s] = cn[idx].sum(axis=1) / cd[idx].sum(axis=1)
    return out


def boot_pigeonhole(d, rng):
    g = d["n_teams"]
    out = {s: np.empty(B) for s in STATS}
    for lo in range(0, B, 1000):
        ca = np.stack([np.bincount(r, minlength=g) for r in rng.integers(0, g, size=(1000, g))]).astype(float)
        cb = np.stack([np.bincount(r, minlength=g) for r in rng.integers(0, g, size=(1000, g))]).astype(float)
        w = ca[:, d["a"]] * cb[:, d["b"]]
        for s in STATS:
            num, den = d["stats"][s]
            out[s][lo:lo + 1000] = (w @ num) / (w @ den)
    return out


def summarise(point, se, bar, reps=None):
    row = {"point": point, "se": se}
    if reps is None:
        row["ci95"] = [point - Z * se, point + Z * se]
        row["p_beyond_bar"] = phi((bar - point) / se)
        row["interval"] = "normal"
    else:
        row["ci95"] = [float(x) for x in np.percentile(reps, [2.5, 97.5])]
        row["p_beyond_bar"] = float((reps <= bar).mean())
        row["interval"] = "percentile"
    return row


def main(deals_path, json_path, md_path):
    data = load(deals_path)
    out = {"schema": "sprime_honest_clustering_addendum_v1", "status": "REPORTED, never decided",
           "bootstrap": {"B": B, "seed": SEED, "scheme": "SeedSequence([seed, model index, panel index]); clusters "
                         "resampled within panel; pooled replicate b = the mean of the panels' replicate b"},
           "bars": BARS, "p_beyond_bar": "one-sided: the share of replicates (or normal mass) at or below the bar",
           "registered_check": registered_check(data),
           "design": {SHORT[p]: {k: data[p][k] for k in ("n_teams", "teams_ours", "teams_theirs", "cells_distinct",
                                                         "mirror_deals")} for p in PANELS}}
    scopes = [SHORT[p] for p in PANELS] + ["U", "H+F"]
    point = {s: {} for s in STATS}
    var = {m: {s: {} for s in STATS} for m in ONE_WAY + ("cell", "two_way", "dyadic")}
    for s in STATS:
        for p in PANELS:
            num, den = data[p]["stats"][s]
            theta, psi = influence(num, den)
            point[s][SHORT[p]] = float(theta)
            for m in ONE_WAY + ("cell",):
                var[m][s][SHORT[p]] = v_cluster(psi, data[p]["cl"][m])
            two = var["our_team"][s][SHORT[p]] + var["their_team"][s][SHORT[p]] - var["cell"][s][SHORT[p]]
            dyad = v_dyadic(psi, data[p]["a"], data[p]["b"], data[p]["n_teams"])
            assert two > 0 and dyad > 0, (s, p, two, dyad)
            var["two_way"][s][SHORT[p]], var["dyadic"][s][SHORT[p]] = two, dyad
        point[s]["U"] = sum(point[s][SHORT[p]] for p in PANELS) / 3
        point[s]["H+F"] = (point[s]["H"] + point[s]["F"]) / 2
        for m in var:
            var[m][s]["U"] = sum(var[m][s][SHORT[p]] for p in PANELS) / 9
            var[m][s]["H+F"] = (var[m][s]["H"] + var[m][s]["F"]) / 4
    reps = {}
    for mi, m in enumerate(ONE_WAY + ("pigeonhole",)):
        reps[m] = {s: {} for s in STATS}
        for pi, p in enumerate(PANELS):
            rng = np.random.default_rng(np.random.SeedSequence([SEED, mi, pi]))
            r = boot_pigeonhole(data[p], rng) if m == "pigeonhole" else boot_one_way(data[p], data[p]["cl"][m], rng)
            for s in STATS:
                reps[m][s][SHORT[p]] = r[s]
        for s in STATS:
            reps[m][s]["U"] = np.mean([reps[m][s][SHORT[p]] for p in PANELS], axis=0)
            reps[m][s]["H+F"] = (reps[m][s]["H"] + reps[m][s]["F"]) / 2
    res = {s: {sc: {} for sc in scopes} for s in STATS}
    for s in STATS:
        for sc in scopes:
            pt, cell = point[s][sc], res[s][sc]
            for m in ONE_WAY:
                cell[m] = summarise(pt, float(np.std(reps[m][s][sc], ddof=1)), BARS[s], reps[m][s][sc])
                cell[m]["se_sandwich"] = math.sqrt(var[m][s][sc])
            for m in ("two_way", "dyadic"):
                cell[m] = summarise(pt, math.sqrt(var[m][s][sc]), BARS[s])
            ph = float(np.var(reps["pigeonhole"][s][sc], ddof=1))
            cell["pigeonhole"] = summarise(pt, math.sqrt(ph), BARS[s], reps["pigeonhole"][s][sc])
            cell["pigeonhole"]["se_minus_2iid"] = math.sqrt(max(ph - 2 * var["iid"][s][sc], 0.0))
            for m in cell:
                cell[m]["deff_vs_iid"] = cell[m]["se"] ** 2 / var["iid"][s][sc]
    out["results"] = res
    # the registered Wilson interval on GOAL-B's score, for the table's first row
    out["goal_b_wilson_registered"] = out["registered_check"]["goal_b"]
    # the panels against each other: POST HOC, three contrasts and one omnibus test, on the decided gain
    het = {}
    for m in ("iid", "unit_seed", "our_team", "their_team", "two_way", "dyadic"):
        g = np.array([point["G_go"][SHORT[p]] for p in PANELS])
        v = np.array([var[m]["G_go"][SHORT[p]] for p in PANELS])
        w = 1 / v
        q = float((w * (g - (w * g).sum() / w.sum()) ** 2).sum())
        row = {"Q": q, "df": 2, "p": math.exp(-q / 2)}                 # chi-square(2) survival = exp(-Q / 2)
        for name, (i, js) in {"P72-F": (2, (1,)), "P72-H": (2, (0,)), "P72-mean(H,F)": (2, (0, 1))}.items():
            d = g[i] - g[list(js)].mean()
            se = math.sqrt(v[i] + v[list(js)].sum() / len(js) ** 2)
            row[name] = {"point": float(d), "se": se, "ci95": [d - Z * se, d + Z * se], "p_two_sided": 2 * phi(-abs(d) / se)}
        het[m] = row
    out["panel_heterogeneity_G_go"] = {"note": "POST HOC: the PREREG predicted no panel ordering; sandwich variances, "
                                       "normal intervals, unadjusted for the three contrasts", "by_model": het}
    json.dump(out, open(json_path, "w"), indent=1)
    open(md_path, "w").write(tables(out))


def fmt(row, pp):
    k = 100.0 if pp else 1.0
    f = (lambda x: f"{x * k:+.2f}") if pp else (lambda x: f"{x:.4f}")
    return f"{f(row['point'])} [{f(row['ci95'][0])}, {f(row['ci95'][1])}]"


def tables(out):
    res = out["results"]
    names = {"iid": "i.i.d. deals", "matchup": "matchup draw (850 a panel)", "unit_seed": "unit seed (85; REGISTERED unit)",
             "our_team": "our team (72)", "their_team": "their team (72)", "two_way": "two-way, roles independent (CGM)",
             "dyadic": "two-way, a team one cluster in either role (dyadic)",
             "pigeonhole": "pigeonhole bootstrap (conservative bracket)"}
    lines = ["<!-- GENERATED by clustering.py from deals.csv; never edit -->", ""]
    lines += ["## Pooled U (H, F, P72 at equal weight)", "",
              "Scores as proportions; gains in pp. `P` = the one-sided mass at or below the bar (0.50 for a score, 0 for "
              "a gain). `deff` = variance over the i.i.d. variance.", "",
              "| dependence model | searched score (GOAL-B's) | P(≤ 0.50) | raw score | paired gain G_U (GO-imputed) | P(≤ 0) | "
              "deff score / gain |", "|---|---|---|---|---|---|---|"]
    w = out["goal_b_wilson_registered"]
    lines.append(f"| REGISTERED: Wilson at the effective n (score); unit-seed bootstrap, B = 2,000 (gain) | {w['point']:.4f} "
                 f"[{w['ci95'][0]:.4f}, {w['ci95'][1]:.4f}] | — | — | "
                 f"{out['registered_check']['G_go']['point'] * 100:+.2f} [{out['registered_check']['G_go']['ci95'][0] * 100:+.2f}, "
                 f"{out['registered_check']['G_go']['ci95'][1] * 100:+.2f}] | — | — |")
    for m, label in names.items():
        s, r, g = res["searched_goal_b"]["U"][m], res["raw_paired"]["U"][m], res["G_go"]["U"][m]
        lines.append(f"| {label} | {fmt(s, False)} | {s['p_beyond_bar']:.3f} | {fmt(r, False)} | {fmt(g, True)} | "
                     f"{g['p_beyond_bar']:.4f} | {s['deff_vs_iid']:.2f} / {g['deff_vs_iid']:.2f} |")
    for stat, title, pp in (("G_go", "The paired gain by panel (GO-imputed, pp)", True),
                            ("searched_goal_b", "The searched score by panel (GOAL-B's statistic)", False),
                            ("raw_paired", "The raw score by panel (paired deals)", False)):
        lines += ["", f"## {title}", "", "| dependence model | H | F | P72 | H + F (P72 left out) | U |", "|---|---|---|---|---|---|"]
        for m, label in names.items():
            lines.append(f"| {label} | " + " | ".join(fmt(res[stat][sc][m], pp) for sc in ("H", "F", "P72", "H+F", "U")) + " |")
    lines += ["", "## The other two gain intervals, pooled U (pp)", "", "| dependence model | paired only | NULL-imputed "
              "(NULL iff UB < +3) |", "|---|---|---|"]
    for m, label in names.items():
        lines.append(f"| {label} | {fmt(res['G_paired']['U'][m], True)} | {fmt(res['G_null']['U'][m], True)} |")
    lines += ["", "## Bootstrap against sandwich (the check that licenses the analytic crossed models): SE of pooled U, pp", "",
              "| one-way model | searched score: bootstrap / sandwich | gain: bootstrap / sandwich |", "|---|---|---|"]
    for m in ONE_WAY:
        s, g = res["searched_goal_b"]["U"][m], res["G_go"]["U"][m]
        lines.append(f"| {names[m]} | {s['se'] * 100:.3f} / {s['se_sandwich'] * 100:.3f} | {g['se'] * 100:.3f} / "
                     f"{g['se_sandwich'] * 100:.3f} |")
    s, g = res["searched_goal_b"]["U"], res["G_go"]["U"]
    lines += ["", f"Pigeonhole with 2 x the i.i.d. variance removed (Owen 2007), against the CGM two-way: searched score "
              f"{s['pigeonhole']['se_minus_2iid'] * 100:.3f} vs {s['two_way']['se'] * 100:.3f} pp; gain "
              f"{g['pigeonhole']['se_minus_2iid'] * 100:.3f} vs {g['two_way']['se'] * 100:.3f} pp."]
    lines += ["", "## The panels against each other (POST HOC; on the GO-imputed gain; pp)", "",
              "| dependence model | P72 − F | P72 − H | P72 − mean(H, F) | omnibus Q (χ², 2 df) | p |", "|---|---|---|---|---|---|"]
    for m, row in out["panel_heterogeneity_G_go"]["by_model"].items():
        c = lambda k: f"{row[k]['point'] * 100:+.2f} [{row[k]['ci95'][0] * 100:+.2f}, {row[k]['ci95'][1] * 100:+.2f}]"
        lines.append(f"| {names[m]} | {c('P72-F')} | {c('P72-H')} | {c('P72-mean(H,F)')} | {row['Q']:.2f} | {row['p']:.3f} |")
    lines += ["", "## The design the clusters come from", "", "| panel | teams | piloted by us | piloted by Kakuna | distinct "
              "(ours, theirs) cells in 1,700 deals | mirror deals (same team both sides) |", "|---|---|---|---|---|---|"]
    for p, d in out["design"].items():
        lines.append(f"| {p} | {d['n_teams']} | {d['teams_ours']} | {d['teams_theirs']} | {d['cells_distinct']} | {d['mirror_deals']} |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main(*sys.argv[1:4])
