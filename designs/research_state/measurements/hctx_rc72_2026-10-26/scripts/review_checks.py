"""Adversarial review checks on the banked H-CTX Amendment-2 games (REVIEW.md). Plays nothing, loads no model.

    CUDA_VISIBLE_DEVICES="" PYTHONPATH=<worktree>/src nice -n 19 taskset -c 5-7,13-15 \
        python designs/research_state/measurements/hctx_rc72_2026-10-26/scripts/review_checks.py OUT.json

Inputs (read only): the read state ``~/gen3ai_archive/hctx_rc72_am2_read`` (games.jsonl + capture START lines of all
140 units), the three panels' team files, our 719-team pool (``main.anchors.session.pool_team_texts``) and Smogon's
teammate priors (``agents.gen3_data.priors``, the builder's lift).

Checks (numbered as REVIEW.md cites them):
- C1 the rule re-derived from the unit rows (premium, recovery, ratio, B = 10,000 seed 7273 unit bootstrap) plus the
  LINEAR contrasts R - 0.25 P and R - 0.75 P under the same bootstrap, and Fieller's interval for R / P;
- C2 per-arm game length, forfeits, ties, seat halves;
- C3 team composition per panel: pool repeats of each set, Smogon lift coherence, pool co-occurrence of the team's
  set pairs and species pairs, role counts (Spikes, Rapid Spin, Explosion users, Baton Pass, sleep moves);
- C4 per-team P* edge e(T) (score in games where T appears on either side: T's strength cancels) and holder
  advantage h(T) (T's strength), with the between-team excess variance of each per panel; within-panel slopes of
  e(T) on the C3 metrics;
- C5 species mix: a ridge logistic of P*'s result on species held by each side fitted on home72 alone, predicting
  rc72b's and F's games (what home72 play predicts for those compositions);
- C6 forum72's legacy-pack team (sft431, Kakuna's Frustration at base power 1): games where Kakuna held it.
"""
from __future__ import annotations

import collections
import itertools
import json
import math
import random
import sys
from pathlib import Path

import numpy as np

STATE = Path("/home/goodlad/gen3ai_archive/hctx_rc72_am2_read")
HERE = Path(__file__).resolve().parent.parent
PANEL_DIRS = {
    "rc72b": HERE / "teams_rc72b",
    "home72": Path("/home/goodlad/dev/gen3ai/designs/research_state/measurements/n0_endofrun_2026-09-27/teams_home72"),
    "forum72": Path("/home/goodlad/dev/gen3ai-reads/x37a_2026-10-18/teams_forum72"),
}
GROUPS = {"rc72b": ["rc72b_a", "rc72b_b", "rc72b_c"], "home72": ["home72_a", "home72_b"],
          "forum72": ["forum72_a", "forum72_b"]}
ROLE_MOVES = {"spikes": {"spikes"}, "spin": {"rapidspin"}, "boom": {"explosion", "selfdestruct"},
              "bpass": {"batonpass"}, "sleep": {"spore", "sleeppowder", "hypnosis", "lovelykiss", "sing", "yawn"},
              "heal": {"wish", "softboiled", "recover", "milkdrink", "moonlight", "morningsun", "synthesis", "slackoff"}}


def to_id(s: str) -> str:
    return "".join(c for c in s.lower() if c.isalnum())


def load_panels():
    from main.anchors.session import load_team_texts
    from main.anchors.team_overlap import parse_team

    panels = {}
    for name, d in PANEL_DIRS.items():
        teams = {}
        for label, text in load_team_texts(d):
            teams[label] = parse_team(text, label)
        by6 = collections.defaultdict(list)
        for label, t in teams.items():
            by6[t.species6].append(label)
        dup = {k: v for k, v in by6.items() if len(v) > 1}
        panels[name] = {"teams": teams, "by6": by6, "dup_species6": len(dup)}
    return panels


def packed_species(team: str):
    return frozenset(to_id(m.split("|")[1] or m.split("|")[0]) for m in team.split("]") if m)


def peer_teams(unit: Path, half: str):
    """Kakuna's team per game of one half, in play order: the peer report's ``team_draws`` (verified against every
    capture's START line below). NOT the peer CSV's "Team File" column: it names the NEXT draw (off by one, the last
    one repeated), checked on rc72b's captures (REVIEW.md, FINDING)."""
    sub = unit / half / ("peer_acceptor_report.json" if half == "ours_challenge" else "peer_challenger_report.json")
    draws = json.loads(sub.read_text())["team_draws"]
    return [(d.replace(".gen3ou_team", ".txt"), None) for d in draws]


def load_games(panels):
    """One row per game. Kakuna's team comes from the peer report's draws ( in play order within each half); where a capture exists, both sides' sixes are also checked against its START line."""
    rows, n_cap = [], 0
    for panel, groups in GROUPS.items():
        p = panels[panel]
        for g in groups:
            cells = STATE / "anchor" / g / "pstar" / "cells"
            for unit in sorted(cells.iterdir()):
                games = [json.loads(line) for line in (unit / "games.jsonl").read_text().splitlines()]
                for half in ("ours_challenge", "peer_challenge"):
                    gs = sorted((r for r in games if r["half"] == half), key=lambda r: r["index"])
                    peer = peer_teams(unit, half)
                    if len(peer) != len(gs):
                        raise SystemExit(f"{unit.name} {half}: {len(gs)} games vs {len(peer)} peer rows")
                    for r, (their_lab, peer_res) in zip(gs, peer):
                        our_lab = r["our_team"]
                        if our_lab not in p["teams"] or their_lab not in p["teams"]:
                            raise SystemExit(f"{unit.name} {r['battle_tag']}: team not in {panel}: {our_lab} / {their_lab}")
                        cap_path = unit / "capture" / f"{r['battle_tag']}.json"
                        if cap_path.exists():
                            n_cap += 1
                            cap = json.loads(cap_path.read_text())
                            start = json.loads(cap["commands"][0][len("START "):])
                            for side in ("p1", "p2"):
                                lab = our_lab if start[side]["name"].startswith("Gen3AI") else their_lab
                                if packed_species(start[side]["team"]) != p["teams"][lab].species6:
                                    raise SystemExit(f"{unit.name} {r['battle_tag']}: {side} START != {lab}")
                        rows.append({"panel": panel, "group": g, "unit": unit.name, "half": half,
                                     "won": bool(r["won"]), "tie": r["result"] == "tie", "turns": r["turns"],
                                     "forfeit": bool(r.get("hit_forfeit_limit")), "ours": our_lab, "theirs": their_lab,
                                     "our6": sorted(p["teams"][our_lab].species6),
                                     "their6": sorted(p["teams"][their_lab].species6)})
    return rows, n_cap


# ---------------------------------------------------------------------------------------------------------- C1
def unit_scores(rows, panel):
    u = collections.defaultdict(lambda: [0, 0])
    for r in rows:
        if r["panel"] == panel:
            u[r["unit"]][0] += r["won"]
            u[r["unit"]][1] += 1
    return [tuple(v) for _, v in sorted(u.items())]


def pooled(units):
    return sum(w for w, _ in units) / sum(n for _, n in units)


def c1(rows):
    U = {p: unit_scores(rows, p) for p in GROUPS}
    rc, fam, unf = (pooled(U[p]) for p in ("rc72b", "home72", "forum72"))
    P, R = fam - unf, rc - unf
    rng = random.Random(7273)
    draws = []
    for _ in range(10000):
        s = {}
        for p in ("rc72b", "home72", "forum72"):
            us = U[p]
            s[p] = pooled([us[rng.randrange(len(us))] for _ in us])
        draws.append((s["home72"] - s["forum72"], s["rc72b"] - s["forum72"]))
    Pd = np.array([d[0] for d in draws])
    Rd = np.array([d[1] for d in draws])

    def pct(a):
        return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]

    out = {"scores": {"rc72b": rc, "home72": fam, "forum72": unf}, "premium": P, "recovery": R, "ratio": R / P,
           "boot_premium": pct(Pd), "boot_recovery": pct(Rd), "boot_ratio_pct": pct(Rd / Pd),
           "frac_draws_premium_le_0": float((Pd <= 0).mean()),
           "R_minus_025P": R - 0.25 * P, "boot_R_minus_025P": pct(Rd - 0.25 * Pd),
           "R_minus_075P": R - 0.75 * P, "boot_R_minus_075P": pct(Rd - 0.75 * Pd),
           "P_boot_sd": float(Pd.std()), "R_boot_sd": float(Rd.std())}
    # Fieller 95 % for theta = R / P with the bootstrap covariance (normal approx)
    vP, vR, cRP = float(Pd.var()), float(Rd.var()), float(np.cov(Rd, Pd)[0, 1])
    z = 1.96
    a = P * P - z * z * vP
    b = -2 * (R * P - z * z * cRP)
    c = R * R - z * z * vR
    disc = b * b - 4 * a * c
    out["fieller95"] = sorted([(-b - math.sqrt(disc)) / (2 * a), (-b + math.sqrt(disc)) / (2 * a)]) if (
        a > 0 and disc >= 0) else "unbounded"
    # unit dispersion (phi) per arm
    phi = {}
    for p, us in U.items():
        k = len(us)
        pbar = pooled(us)
        n = us[0][1]
        var = sum((w / n - pbar) ** 2 for w, _ in us) / (k - 1)
        phi[p] = var / (pbar * (1 - pbar) / n)
    out["phi"] = phi
    # vs X51's banked U mean (cross-engine, descriptive only)
    out["cross_engine_vs_U"] = {"x51_home72": 0.5975, "x51_U_mean": 0.48666666666666664,
                                "rc72b_ratio_vs_U": (rc - 0.48666666666666664) / (0.5975 - 0.48666666666666664)}
    return out


# ---------------------------------------------------------------------------------------------------------- C2
def c2(rows):
    out = {}
    for p in GROUPS:
        rs = [r for r in rows if r["panel"] == p]
        t = np.array([r["turns"] for r in rs])
        won = np.array([r["won"] for r in rs])
        halves = {}
        for h in ("ours_challenge", "peer_challenge"):
            hs = [r for r in rs if r["half"] == h]
            halves[h] = sum(r["won"] for r in hs) / len(hs)
        out[p] = {"n": len(rs), "turns_mean": float(t.mean()), "turns_median": float(np.median(t)),
                  "turns_p90": float(np.percentile(t, 90)), "ties": sum(r["tie"] for r in rs),
                  "forfeits": sum(r["forfeit"] for r in rs), "halves": halves,
                  "score_turns_le_30": float(won[t <= 30].mean()), "share_turns_le_30": float((t <= 30).mean()),
                  "score_turns_gt_60": float(won[t > 60].mean()), "share_turns_gt_60": float((t > 60).mean()),
                  "their_team_matched": sum(r["theirs"] is not None for r in rs)}
    return out


# ---------------------------------------------------------------------------------------------------------- C3
def c3(panels):
    from main.anchors.session import pool_team_texts
    from main.anchors.team_overlap import Reference, parse_team

    pool = [parse_team(t) for t in pool_team_texts()]
    if len(pool) != 719:
        raise SystemExit(f"pool has {len(pool)} teams, expected 719")
    ref = Reference(pool)
    set_n = collections.Counter(k for t in pool for k in {m.key for m in t.mons})
    pair_n = collections.Counter(p for t in pool for p in itertools.combinations(sorted({m.key for m in t.mons}), 2))
    sp_pair_n = collections.Counter(p for t in pool for p in itertools.combinations(sorted(t.species6), 2))
    sp_n = collections.Counter(s for t in pool for s in t.species6)
    from agents.gen3_data import priors

    usage = priors.species_usage()
    tot = sum(usage.values())
    lift_cache = {}

    def lift(t, s):
        if (t, s) not in lift_cache:
            tm = priors.teammates(t)
            share = usage.get(s, 0.0) / tot
            raw = (tm.get(s, 0.0) / share) if (tm and share > 0) else 1.0
            lift_cache[(t, s)] = min(max(raw, 0.25), 4.0)
        return lift_cache[(t, s)]

    metrics, summary, species_mix = {}, {}, {}
    for name, p in panels.items():
        m = {}
        for label, t in p["teams"].items():
            keys = sorted({x.key for x in t.mons})
            sp = sorted(t.species6)
            moves = [set(x.moves) for x in t.mons]
            role = {r: sum(bool(mv & ids) for mv in moves) for r, ids in ROLE_MOVES.items()}
            m[label] = {
                "set_seen_frac": sum(set_n[k] > 0 for k in keys) / len(keys),
                "log_set_repeats": float(np.mean([math.log1p(set_n[k]) for k in keys])),
                "set_pairs_in_pool": sum(pair_n[q] > 0 for q in itertools.combinations(keys, 2)),
                "species_pair_cooc": float(np.mean([math.log1p(sp_pair_n[q]) for q in itertools.combinations(sp, 2)])),
                "species_pair_pmi": float(np.mean([math.log((sp_pair_n[q] + 0.5) * 719 / ((sp_n[q[0]] + 0.5) * (sp_n[q[1]] + 0.5)))
                                                    for q in itertools.combinations(sp, 2)])),
                "smogon_lift": float(np.mean([math.log(lift(a, b)) for a in sp for b in sp if a != b])),
                "max_shared_pool": ref.max_shared(t)[0],
                **{f"n_{r}": v for r, v in role.items()},
                "spikes_and_spin": int(role["spikes"] >= 1 and role["spin"] >= 1),
                "dup_spikes": int(role["spikes"] >= 2),
            }
        metrics[name] = m
        cols = list(next(iter(m.values())).keys())
        summary[name] = {c: float(np.mean([v[c] for v in m.values()])) for c in cols}
        species_mix[name] = collections.Counter(s for t in p["teams"].values() for s in t.species6)
    tv = {}
    for a, b in (("rc72b", "home72"), ("forum72", "home72"), ("rc72b", "forum72")):
        A, B = species_mix[a], species_mix[b]
        na, nb = sum(A.values()), sum(B.values())
        tv[f"{a}_vs_{b}"] = 0.5 * sum(abs(A[s] / na - B[s] / nb) for s in set(A) | set(B))
    diffs = sorted(((species_mix["rc72b"][s] - species_mix["home72"][s], s) for s in
                    set(species_mix["rc72b"]) | set(species_mix["home72"])))
    return metrics, {"summary": summary, "species_tv": tv, "rc_minus_home_species_most_under": diffs[:8],
                     "rc_minus_home_species_most_over": diffs[-8:][::-1],
                     "home72_species_absent_from_rc72b": sorted(set(species_mix["home72"]) - set(species_mix["rc72b"]))}


# ---------------------------------------------------------------------------------------------------------- C4
def c4(rows, metrics):
    out = {}
    for p in GROUPS:
        rs = [r for r in rows if r["panel"] == p]
        hold = collections.defaultdict(lambda: [0, 0])     # P* holds T: P* wins, n
        face = collections.defaultdict(lambda: [0, 0])     # Kakuna holds T: P* wins, n
        for r in rs:
            hold[r["ours"]][0] += r["won"]
            hold[r["ours"]][1] += 1
            if r["theirs"] is not None:
                face[r["theirs"]][0] += r["won"]
                face[r["theirs"]][1] += 1
        teams = sorted(set(hold) & set(face))
        e, h, nn = [], [], []
        for t in teams:
            a = hold[t][0] / hold[t][1]
            b = face[t][0] / face[t][1]
            e.append((a + b) / 2)
            h.append((a - b) / 2)   # holder advantage: mean holder score (a + (1 - b)) / 2, minus 0.5
            nn.append((hold[t][1], face[t][1]))
        e, h = np.array(e), np.array(h)
        pbar = sum(r["won"] for r in rs) / len(rs)
        # binomial noise of e and h per team: 1/4 (1/n1 + 1/n2) p(1-p)
        noise = np.array([0.25 * pbar * (1 - pbar) * (1 / a + 1 / b) for a, b in nn])
        res = {"teams": len(teams), "games_per_team_mean": float(np.mean([a + b for a, b in nn])),
               "edge_mean": float(e.mean()), "edge_var": float(e.var(ddof=1)), "noise_var": float(noise.mean()),
               "edge_excess_sd": float(math.sqrt(max(e.var(ddof=1) - noise.mean(), 0.0))),
               "holder_var": float(h.var(ddof=1)),
               "holder_excess_sd": float(math.sqrt(max(h.var(ddof=1) - noise.mean(), 0.0))),
               "corr_edge_holder": float(np.corrcoef(e, h)[0, 1])}
        m = metrics[p]
        slopes = {}
        for col in ("log_set_repeats", "set_pairs_in_pool", "species_pair_cooc", "species_pair_pmi", "smogon_lift",
                    "spikes_and_spin", "n_boom", "n_heal"):
            x = np.array([m[t][col] for t in teams], dtype=float)
            if x.std() == 0:
                slopes[col] = None
                continue
            r_e = float(np.corrcoef(x, e)[0, 1])
            r_h = float(np.corrcoef(x, h)[0, 1])
            slopes[col] = {"corr_edge": r_e, "corr_holder": r_h, "x_mean": float(x.mean()), "x_sd": float(x.std())}
        res["corr_with_metrics"] = slopes
        # terciles of coherence (within the panel): P*'s score in games where EITHER side holds a team of the tercile
        terc = {}
        for col in ("smogon_lift", "species_pair_pmi"):
            x = np.array([m[t][col] for t in teams], dtype=float)
            cuts = np.percentile(x, [100 / 3, 200 / 3])
            band = {t: int(np.searchsorted(cuts, m[t][col], side="right")) for t in teams}
            tb = {}
            for k in range(3):
                g_ours = [r["won"] for r in rs if band.get(r["ours"]) == k]
                g_theirs = [r["won"] for r in rs if band.get(r["theirs"]) == k]
                tb[k] = {"x_range": [float(x[[band[t] == k for t in teams]].min()), float(x[[band[t] == k for t in teams]].max())],
                         "pstar_holds_score": float(np.mean(g_ours)), "n_holds": len(g_ours),
                         "kakuna_holds_score": float(np.mean(g_theirs)), "n_kakuna_holds": len(g_theirs)}
            terc[col] = tb
        res["coherence_terciles"] = terc
        out[p] = res
    return out


# ---------------------------------------------------------------------------------------------------------- C5
def c5(rows):
    from sklearn.linear_model import LogisticRegression

    species = sorted({s for r in rows for s in r["our6"] + r["their6"]})
    ix = {s: i for i, s in enumerate(species)}

    def X(rs):
        a = np.zeros((len(rs), 2 * len(species)))
        for k, r in enumerate(rs):
            for s in r["our6"]:
                a[k, ix[s]] = 1
            for s in r["their6"]:
                a[k, len(species) + ix[s]] = 1
        return a

    home = [r for r in rows if r["panel"] == "home72"]
    out = {}
    for C in (0.1, 0.3, 1.0):
        lr = LogisticRegression(C=C, max_iter=2000)
        lr.fit(X(home), np.array([r["won"] for r in home]))
        out[f"C={C}"] = {p: {"predicted": float(lr.predict_proba(X([r for r in rows if r["panel"] == p]))[:, 1].mean()),
                             "observed": float(np.mean([r["won"] for r in rows if r["panel"] == p]))}
                         for p in GROUPS}
    # species absent from home72 carry no fitted coefficient (zero = average): count them per panel
    home_sp = {s for r in home for s in r["our6"] + r["their6"]}
    out["mons_outside_home72_species"] = {p: float(np.mean([sum(s not in home_sp for s in r["our6"] + r["their6"])
                                                           for r in rows if r["panel"] == p])) for p in GROUPS}
    return out


# ---------------------------------------------------------------------------------------------------------- C6
def c6(rows):
    rs = [r for r in rows if r["panel"] == "forum72"]
    k = [r for r in rs if r["theirs"] == "sft431.txt"]
    o = [r for r in rs if r["ours"] == "sft431.txt"]
    rest = [r for r in rs if r["theirs"] != "sft431.txt"]
    return {"kakuna_held_sft431": len(k), "pstar_score_when_kakuna_held": (sum(r["won"] for r in k) / len(k)) if k else None,
            "pstar_held_sft431": len(o), "pstar_score_when_pstar_held": (sum(r["won"] for r in o) / len(o)) if o else None,
            "forum72_score_without_kakuna_sft431": sum(r["won"] for r in rest) / len(rest),
            "max_bias_pp_if_all_those_games_flipped": 100 * sum(r["won"] for r in k) / len(rs)}


def main() -> int:
    out_path = Path(sys.argv[1])
    panels = load_panels()
    rows, n_cap = load_games(panels)
    metrics, comp = c3(panels)
    res = {"n_games": len(rows), "n_games_start_checked": n_cap, "dup_species6": {k: v["dup_species6"] for k, v in panels.items()},
           "C1": c1(rows), "C2": c2(rows), "C3": comp, "C4": c4(rows, metrics), "C5": c5(rows), "C6": c6(rows)}
    out_path.write_text(json.dumps(res, indent=1, default=str) + "\n")
    print(json.dumps(res, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
