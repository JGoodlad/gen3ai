"""TSR — DESCRIPTIVE supplements to the registered read (no verdicts; README §"Beyond the registration").

  PYTHONPATH=<checkout>/src python supplement.py      # writes supplement.json

(a) cross-pilot rho with a TEAMS-ONLY bootstrap on the point ratings (the registered CI resamples each pilot's REPLICATE
    ratings, which carry the game noise a second time and pull every replicate rho toward 0);
(b) D3 with the piloted team's strength REGRESSION-CALIBRATED: z_cal = mu_U + rel_U (z - mu_U), per familiarity group U
    (a noisy z attenuates the logistic slope, so per-team matching under-adjusts; the group means enter in full);
(c) the logistic coefficients of the registered D3 fits;
(d) the SET-LEVEL correction: the raw gap on all 800 games minus the strength term read from pilot B's set means;
(e) X26's relative familiarity premium: (X26's home72 - H) - (Kakuna's home72 - H) on the SAME subset teams;
(f) standing rule 8: pilot A refitted WITHOUT the pairs holding a game with a decision inside the CPU near-tie margin
    (``h2h.play.NEAR_TIE`` 2e-5: an outcome that could rest on a rounding error), set means and comparisons.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze  # noqa: E402

from main.team_strength import cli  # noqa: E402
from main.team_strength import fit as F  # noqa: E402
from main.team_strength import register as RG  # noqa: E402
from main.team_strength import report as RP  # noqa: E402

HERE = analyze.HERE


def teams_only_rho(ra, rb, common, B, seed):
    ia, ib = [ra.ids.index(t) for t in common], [rb.ids.index(t) for t in common]
    x, y = ra.r[ia], rb.r[ib]
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(B):
        p = rng.integers(0, len(common), len(common))
        d.append(F.spearman(x[p], y[p]))
    return {"rho": F.spearman(x, y), "ci_teams_only": [float(v) for v in np.percentile(d, [2.5, 97.5])]}


def calibrated(rt, groups):
    """A Rating whose r / reps are regression-calibrated within each group (module docs)."""
    r, reps = rt.r.copy(), rt.reps.copy()
    rel = {}
    for g, ids in groups.items():
        cols = np.array([rt.ids.index(t) for t in ids])
        obs = float(np.var(rt.r[cols], ddof=1))
        noise = float(np.mean(np.var(rt.reps[:, cols], axis=0, ddof=1)))
        rel[g] = max(0.0, 1.0 - noise / obs)
        mu = rt.r[cols].mean()
        r[cols] = mu + rel[g] * (rt.r[cols] - mu)
        mus = rt.reps[:, cols].mean(axis=1, keepdims=True)
        reps[:, cols] = mus + rel[g] * (rt.reps[:, cols] - mus)
    return F.Rating(ids=rt.ids, r=r, reps=reps, games=rt.games, score=rt.score, seed=rt.seed, prior_sd=rt.prior_sd,
                    anchor=rt.anchor), rel


def coefs(rows, rt):
    teams = sorted({x["team"] for x in rows})
    z = {t: rt.r[rt.ids.index(t)] for t in teams}
    X = np.column_stack([np.ones(len(rows)), [1.0 if x["unfamiliar"] else 0.0 for x in rows],
                         [z[x["team"]] for x in rows]])
    y = np.array([1.0 if x["win"] else 0.0 for x in rows])
    a, g, b = RP.logistic(X, y)
    return {"a": float(a), "g_unfamiliar": float(g), "b_strength_per_logit": float(b)}


def main() -> int:
    reg = RG.load(HERE / "tsr_registration.json")
    out, ratings = cli.compute(reg, analyze.STATE)
    ra, rb = ratings["x26"], ratings["kakuna"]
    refs = {m["team_id"] for m in reg["reference"]["members"]}
    common = sorted((set(rb.ids) - refs) & set(ra.ids))
    sup = {"a_rho_teams_only": teams_only_rho(ra, rb, common, 2000, 3830)}
    rows = analyze.piloted_rows(reg)
    ks = reg["pilots"]["kakuna"]["subsets"]
    sub = set(ks["home72"]) | set(ks["heldout72_hpfix"])
    rb_rows = [x for x in rows if x["team"] in sub]
    mg = reg["fit"]["matched_gap"]
    fam_b = sorted({x["team"] for x in rb_rows if not x["unfamiliar"]})
    unf_b = sorted({x["team"] for x in rb_rows if x["unfamiliar"]})
    cb, rel_b = calibrated(rb, {"familiar": fam_b, "unfamiliar": unf_b})
    fam_a = sorted({x["team"] for x in rows if not x["unfamiliar"]})
    unf_a = sorted({x["team"] for x in rows if x["unfamiliar"]})
    ca, rel_a = calibrated(ra, {"familiar": fam_a, "unfamiliar": unf_a})
    sup["b_calibrated"] = {
        "kakuna": {"reliability": rel_b, **RP.matched_gap(rb_rows, cb, B=int(mg["B"]), seed=int(mg["seed"]))},
        "x26": {"reliability": rel_a, **RP.matched_gap(rows, ca, B=int(mg["B"]), seed=int(mg["seed"]))}}
    sup["c_coefficients"] = {"kakuna_primary": coefs(rb_rows, rb), "x26_secondary": coefs(rows, ra)}
    raw_all = (np.mean([x["win"] for x in rows if not x["unfamiliar"]]) -
               np.mean([x["win"] for x in rows if x["unfamiliar"]]))
    kb = out["pilots"]["kakuna"]["sets"]
    d_elo = kb["home72"]["mean_elo"] - kb["heldout72_hpfix"]["mean_elo"]
    sup["d_set_level"] = {"raw_gap_all_800": float(raw_all), "kakuna_strength_term_elo": d_elo,
                          "kakuna_strength_term_p": F.elo_to_p(d_elo) - 0.5,
                          "corrected": float(raw_all - (F.elo_to_p(d_elo) - 0.5)),
                          "x37a_pilot_c": 0.123, "x37a_delta_hat": 0.0725}
    # (e) on the same subset teams, per replicate (independent pilots: replicate k of each)
    prem = []
    for k in range(min(ra.reps.shape[0], rb.reps.shape[0])):
        def m(rt, ids):
            return float(np.mean([rt.reps[k, rt.ids.index(t)] for t in ids])) * F.ELO
        prem.append((m(ra, ks["home72"]) - m(ra, ks["heldout72_hpfix"]))
                    - (m(rb, ks["home72"]) - m(rb, ks["heldout72_hpfix"])))
    est = ((np.mean([ra.elo(t) for t in ks["home72"]]) - np.mean([ra.elo(t) for t in ks["heldout72_hpfix"]]))
           - (np.mean([rb.elo(t) for t in ks["home72"]]) - np.mean([rb.elo(t) for t in ks["heldout72_hpfix"]])))
    sup["e_x26_familiarity_premium_elo"] = {"est": float(est),
                                            "ci_games_only": [float(v) for v in np.percentile(prem, [2.5, 97.5])]}
    # (f) rule 8
    from main.team_strength import x26_lane as XL

    near = set()
    for p in sorted((analyze.STATE / "x26" / "units").glob("x26_*.json")):
        rec = json.loads(p.read_text())
        for row in rec["pairs"]:
            if any(g.get("near_ties") for g in row["games"]):
                near.add(f"{rec['unit']}:{row['k']}")
    games = [g for g in XL.load_games(reg, analyze.STATE) if g.cluster not in near]
    fz = reg["fit"]
    refs_l = [m["team_id"] for m in reg["reference"]["members"]]
    rt = F.rate(games, prior_sd=float(fz["prior_sd"]), anchor=refs_l, B=int(fz["B"]), seed=int(fz["seed"]))
    members = {s2: sorted(set(b["team_ids"].values())) for s2, b in reg["sets"].items()}
    means = F.set_means(rt, members, seed=int(fz["seed"]) + 10)
    sup["f_rule8_x26_without_near_tie_pairs"] = {
        "pairs_dropped": len(near), "games_kept": len(games),
        "sets": {s2: {"mean_elo": means[s2]["mean_elo"], "ci": means[s2]["ci"]} for s2 in members},
        "comparisons": {f"{x}-{y}": {**F.diff(means, x, y), **RP.competitive_class(F.diff(means, x, y)["ci"],
                                                                                   float(fz["competitive_margin_elo"]))}
                        for x, y in fz["comparisons"]}}
    (HERE / "supplement.json").write_text(json.dumps(sup, indent=1, default=float) + "\n")
    print(json.dumps(sup, indent=1, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
