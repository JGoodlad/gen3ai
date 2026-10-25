"""REVIEW D (re-review of 431b1fdf): the CPU checks behind REVIEW_D_rereview.md. Seeded, deterministic.

    PYTHONPATH=$PWD/src CUDA_VISIBLE_DEVICES="" nice -n 19 taskset -c 5-7,13-15 scripts/ops/mem_cap.sh 14 \
        $PY designs/research_state/measurements/human_replay_quality_2026-10-25/review_d_checks.py [power|data|all]

`power` (seconds, numpy + scipy) re-derives the plan's X60 and X59-S numbers under what the plan REGISTERS but
`plan_power.py` does not model: AM-5 with an upper-quantile phi (REVIEW B m6, adopted by the plan), the seed rule's
firing rate, a between-seed SD in the H-TEAMS - D1 contrast, and a training-seed SD between X59-S's two students.

`data` (minutes, ~6 GB) reads the revision's stamped hrq cache of `hr_spec_v1` (read-only, through
`hrq.load_snapshots`, which checks the stamp and the record count) and asks two questions of the new >= 3 move-revealed
fit exclusion (the plan §3.1): what the kept 29 % looks like (length, forfeits, reveals; and how much of the pool-
UNFAMILIAR region, the point of goal (b), survives), and whether a C-GAME-like fill of the KEPT sides re-creates panel
cores that the pre-fill rule could not see (moves drawn by `main.value_search.world_draws.draw_moves` from the Smogon
prior, as C-GAME draws them; legality redraws omitted). The sealed `forum216b` is never read.
"""
from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import chi2, norm
from scipy.stats import t as tdist

sys.path.insert(0, str(Path(__file__).resolve().parent))
import review_b_power as RB

SEED = 20261027
Z = RB.Z
OUT: dict = {}


def say(k, v):
    OUT[k] = v
    print(k, json.dumps(v, default=float), flush=True)


# ============================================================================================ power
def x60(rng, d_f, d_p, seed_sd=0.0, games=630, base=0.47, band=0.03, reps=100_000, phi_mode="point"):
    """plan_power.x60_fixed with AM-5's multiplier chosen by ``phi_mode``: 'point' (plan_power, main.x35.d1 today),
    't' (t at df instead of z), 'chi2' (phi's one-sided 95 % upper quantile, df / chi2_0.05(df) x phi-hat)."""
    K = 2
    d = np.array([d_f, d_p])

    def arm(eff):
        s = rng.normal(0.0, seed_sd, (reps, 5))
        p = np.clip(base + eff[None, None, :] + s[:, :, None], 0.01, 0.99)
        return rng.binomial(games, p) / games
    on, off = arm(d), arm(np.zeros(K))
    looks = []
    for ns in (3, 5):
        a, c = on[:, :ns, :], off[:, :ns, :]
        pa, pc = a.mean(1), c.mean(1)
        va = (pa * (1 - pa) / (games * ns)).sum(1) / K ** 2
        vc = (pc * (1 - pc) / (games * ns)).sum(1) / K ** 2

        def ratio(x):
            e = (x * (1 - x) / games).sum(2).mean(1) / K ** 2
            return x.mean(2).var(1, ddof=1) / e
        phi = (ratio(a) + ratio(c)) / 2
        df = 2 * (ns - 1)
        z = Z
        if phi_mode == "chi2":
            phi = phi * df / chi2.ppf(0.05, df)
        elif phi_mode == "t":
            z = tdist.ppf(0.975, df)
        hw = z * np.sqrt(va + vc) * np.sqrt(np.maximum(1.0, phi))
        dd = pa.mean(1) - pc.mean(1)
        sign_ok = ((a.mean(2) - c.mean(2)) > 0).sum(1) >= (2 if ns == 3 else 3)
        het_ok = ((pa - pc) > 0).all(1)
        lo, hi = dd - hw, dd + hw
        cls = np.where((lo > 0) & sign_ok & het_ok, "GAIN", np.where(hi < 0, "LOSS",
                       np.where((lo > -band) & (hi < band), "NULL", "UNDECIDED")))
        looks.append((cls, dd, hw))
    (c1, d1, _), (c2, _, h2) = looks
    final = np.where((c1 == "UNDECIDED") & (d1 > 0), c2, c1)
    out = {k: round(float((final == k).mean()), 3) for k in ("GAIN", "NULL", "LOSS", "UNDECIDED")}
    out["halfwidth_look2_pp_median"] = round(float(np.median(h2)) * 100, 2)
    return out


def seed_rule(rng, seed_sd, games=630, base=0.47, d1=0.021, reps=100_000, bar=0.015):
    """The plan's seed rule: the topped-up C0 + D1 finals (5 + 5, F + P72 at ``games`` each) give phi-hat (df 8); its
    one-sided 95 % upper bound implies a between-seed SD of the pooled mean; the rule FIRES above ``bar``."""
    K = 2

    def arm(eff):
        s = rng.normal(0.0, seed_sd, (reps, 5))
        p = np.clip(base + eff + s[:, :, None] + np.zeros(K), 0.01, 0.99)
        return rng.binomial(games, p) / games
    c0, dd = arm(0.0), arm(d1)

    def parts(x):
        e = (x * (1 - x) / games).sum(2).mean(1) / K ** 2
        return x.mean(2).var(1, ddof=1) / e, e
    (r0, e0), (r1, e1) = parts(c0), parts(dd)
    phi = (r0 + r1) / 2
    ub = phi * 8 / chi2.ppf(0.05, 8)
    sd_ub = np.sqrt(np.maximum(0.0, ub - 1.0) * (e0 + e1) / 2)
    return {"P(rule fires)": round(float((sd_ub > bar).mean()), 3),
            "median implied seed-SD UB (pp)": round(float(np.median(sd_ub)) * 100, 2)}


def contrast(delta, seed_sd, games=630, seeds=5, panels=2, base=0.47):
    """H-TEAMS - D1 (two arms, ``seeds`` finals each) with a between-seed SD of each final's pooled mean."""
    se = np.sqrt(2 * base * (1 - base) / (games * seeds) / panels + 2 * seed_sd ** 2 / seeds)
    hw_registered = Z * np.sqrt(2 * base * (1 - base) / (games * seeds) / panels)
    return {"true_se_pp": round(se * 100, 2), "registered_halfwidth_pp": round(hw_registered * 100, 2),
            "P(registered LB > 0)": round(float(1 - norm.cdf((hw_registered - delta) / se)), 3)}


def x59s(delta, student_sd, n_u=2400, base=0.30):
    """X59-S's registered interval covers game noise only; a training-seed SD ``student_sd`` per student's U mean adds
    2 x sd^2 to Delta_S. Returns P(HUMAN-LABELS-BETTER) for one pair and for the replicated pair."""
    k = n_u // 3
    se_games = np.sqrt(2 * 3 * base * (1 - base) / k / 9)
    se = np.sqrt(se_games ** 2 + 2 * student_sd ** 2)
    p1 = float(1 - norm.cdf((Z * se_games - delta) / se))
    return {"P(HLB) one pair": round(p1, 3), "P(HLB) replicated": round(p1 * p1, 3)}


def power():
    rng = np.random.default_rng(SEED)
    for mode in ("point", "t", "chi2"):
        for sd in (0.0, 0.015):
            for dv in (0.0, 0.025):
                say(f"x60 phi={mode} seed_sd={sd*100:.1f}pp delta={dv*100:.1f}pp", x60(rng, dv, dv, sd, phi_mode=mode))
    for g in (1000, 1260):          # what the registered phi rule needs (seed SD 0)
        for mode in ("t", "chi2"):
            for dv in (0.0, 0.025):
                say(f"x60 phi={mode} games={g} seed_sd=0.0pp delta={dv*100:.1f}pp", x60(rng, dv, dv, 0.0, games=g,
                                                                                         phi_mode=mode))
    for sd in (0.0, 0.0075, 0.015, 0.027):
        say(f"seed_rule true_seed_sd={sd*100:.2f}pp", seed_rule(rng, sd))
    for sd in (0.0, 0.015, 0.027):
        for dv in (0.0, 0.03):
            say(f"contrast H-TEAMS-D1 seed_sd={sd*100:.1f}pp delta={dv*100:.0f}pp", contrast(dv, sd))
    for sd in (0.0, 0.01, 0.02, 0.03):
        for dv in (0.0, 0.04):
            say(f"x59s student_sd={sd*100:.0f}pp delta={dv*100:.0f}pp", x59s(dv, sd))


# ============================================================================================ data
def data():
    import hrq

    from main import heldout_panels as HP
    from main.anchors.team_overlap import parse_team
    from main.value_search import world_draws as WD

    recs, _ = hrq.load_snapshots(Path.home() / "gen3ai_archive/human_replay_quality/cache_v2", ["hr_spec_v1"])
    spec = [r for r in recs if "err" not in r and r["pop"] == "spec"]
    del recs
    flagged = set(hrq.bot_flags(hrq.accounts(spec)))
    teams = hrq.Teams()
    # heldout72 (the pre-fix H) is in the plan's rule but not in hrq.Teams: index it the same way
    name = "heldout72"
    teams.ref[name] = [parse_team(txt, lab) for lab, txt in HP.panel_team_texts(name)]
    vec = {}
    for i, tm in enumerate(teams.ref[name]):
        for ms in tm.mons:
            vec.setdefault(ms.species, np.zeros(len(teams.ref[name]), np.int8))[i] = 1
    teams.vec[name] = vec
    teams.byteam[name] = [{ms.species: ms for ms in tm.mons} for tm in teams.ref[name]]
    PANELS = ("heldout72", "heldout72_hpfix", "forum72", "p72")

    def informative(m):
        return {k: v for k, v in m.items() if v[0]}

    def revealed_best(m):
        mi = informative(m)
        out = {}
        for p in PANELS:
            out[p] = 0
            if mi:
                _, fi = teams.overlaps(mi, p, min_sp=1)
                out[p] = int(fi.max()) if len(fi) else 0
        return out

    prior = WD.Prior()
    rng = np.random.default_rng(SEED)
    panel_sets = {p: [{ms.species: tuple(sorted(ms.moves)) for ms in tm.mons} for tm in teams.ref[p]] for p in PANELS}

    def filled(m):
        """One C-GAME-like move fill of a side: revealed moves kept, the rest drawn from the Smogon move prior."""
        out = {}
        for sp, (mv, _, _) in m.items():
            c = teams.canon(sp)
            rev = [x for x in mv if x != "struggle"][:4]
            keep, hp = {}, None
            for i, x in enumerate(rev):
                if x == "hiddenpower":
                    hp = WD.MX.draw(rng, prior.hidden_power(c)) or "dark"
                    keep[i] = f"hiddenpower{hp}"
                else:
                    keep[i] = x
            try:
                got = WD.draw_moves(rng, prior.moves(c), keep, 4, None, [], hp)
            except Exception:                                   # noqa: BLE001 - a table quirk: count as unfilled
                got = None
            out[c] = tuple(sorted(got[0])) if got else tuple(sorted(keep.values()))
        return out

    def postfill_best(fm):
        best = {}
        for p in PANELS:
            b = 0
            for tm in panel_sets[p]:
                b = max(b, sum(1 for c, mv in fm.items() if tm.get(c) == mv))
            best[p] = b
        return best

    agg = {k: collections.Counter() for k in ("kept", "excluded")}
    sides = {k: collections.Counter() for k in ("kept", "excluded")}
    for r in spec:
        if r["date"] > hrq.TRAIN_LAST or not r["rated"] or max(r.get("r1") or 0, r.get("r2") or 0) < 1500:
            continue
        bests = [revealed_best(r["m" + s]) for s in ("1", "2")]
        cls = "excluded" if any(b[p] >= 3 for b in bests for p in PANELS) else "kept"
        a = agg[cls]
        a["battles"] += 1
        a["turns"] += r["turns"]
        a["forfeit_or_timer"] += r["end"] in ("forfeit", "timer")
        a["turns<=10"] += r["turns"] <= 10
        for s in ("1", "2"):
            m = r["m" + s]
            a["mons_seen"] += len(m)
            a["moves_seen"] += sum(len(v[0]) for v in m.values())
            a["full6_sides"] += len(r["o" + s]) == 6
        for s in ("1", "2"):
            if (r.get("r" + s) or 0) < 1500 or len(r["o" + s]) != 6 or hrq.toid(r.get("n" + s) or "") in flagged:
                continue
            x = sides[cls]
            x["ge1500_full6_sides"] += 1
            pool_fits = teams.side(r["m" + s], "pool")[1]
            x["pool_unfamiliar(<=2 fits)"] += pool_fits <= 2
            if cls == "kept":
                pf = postfill_best(filled(r["m" + s]))
                for p in PANELS:
                    x[f"postfill_exact>=3_{p}"] += pf[p] >= 3
                x["postfill_exact>=3_any"] += any(pf[p] >= 3 for p in PANELS)
                if pool_fits <= 2:
                    x["unfamiliar_and_postfill_exact>=3_any"] += any(pf[p] >= 3 for p in PANELS)
    nb = agg["kept"]["battles"] + agg["excluded"]["battles"]
    say("exclusion_share_of_ge1500_train_battles (4 panels, forum216b not read)", agg["excluded"]["battles"] / nb)
    for cls in ("kept", "excluded"):
        a = agg[cls]
        say(f"{cls}_battles", {"battles": a["battles"], "mean_turns": a["turns"] / a["battles"],
                               "forfeit_or_timer": a["forfeit_or_timer"] / a["battles"],
                               "turns<=10": a["turns<=10"] / a["battles"],
                               "mons_seen_per_side": a["mons_seen"] / (2 * a["battles"]),
                               "moves_seen_per_mon": a["moves_seen"] / max(1, a["mons_seen"]),
                               "full6_side_share": a["full6_sides"] / (2 * a["battles"])})
        x = sides[cls]
        n = max(1, x["ge1500_full6_sides"])
        say(f"{cls}_ge1500_full6_nonbot_sides", {k: (v if k == "ge1500_full6_sides" else v / n) for k, v in x.items()}
            | {"pool_unfamiliar_count": x["pool_unfamiliar(<=2 fits)"]})
    tot_unf = sides["kept"]["pool_unfamiliar(<=2 fits)"] + sides["excluded"]["pool_unfamiliar(<=2 fits)"]
    say("pool_unfamiliar_sides_surviving_the_exclusion", sides["kept"]["pool_unfamiliar(<=2 fits)"] / max(1, tot_unf))


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("power", "all"):
        power()
    if what in ("data", "all"):
        data()
