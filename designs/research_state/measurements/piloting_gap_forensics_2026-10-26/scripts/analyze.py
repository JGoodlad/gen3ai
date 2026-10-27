"""Where in the game do P*'s extra unfamiliar-team losses against Kakuna come from? Descriptive forensics on banked
games (``parse_games.py``'s rows); no play, no model.

Usage: PYTHONPATH=<worktree>/src:<this dir> python analyze.py GAMES.jsonl.gz OUT.json

Groups. ``U`` is H, F and P72 with EQUAL WEIGHT (each game weighted 1 / (3 n_panel)), X51's definition. Every
interval is a 95 % percentile bootstrap over UNITS (20-game blocks, one seed of one cell), resampled within each cell,
B = 1000, seed 0. A contrast's interval resamples both of its groups.
"""
from __future__ import annotations

import gzip
import json
import math
import os
import sys
from collections import defaultdict

import numpy as np

from agents.gen3_data import moves as M
from teams import Familiarity

B = int(os.environ.get("PGF_B", "1000"))
RNG = np.random.default_rng(0)
EXPLODE = {"explosion", "selfdestruct"}
TS = [1, 3, 5, 8, 10, 15, 20, 25, 30, 40, 60]
D_EDGES = [-2.0, -1.0, -0.4, 0.4, 1.0, 2.0]


# ---------------------------------------------------------------- per-game features
def move_class(mid: str) -> str:
    if mid in EXPLODE:
        return "explode"
    md = M.get(mid)
    if md is None:
        return "unknown"
    if md.is_dex_damaging:
        return "attack"
    if md.is_boost or md.self_boosts or mid in ("curse", "bellydrum"):
        return "setup"
    if md.status_inflicted or mid in ("toxic", "leechseed", "yawn", "swagger"):
        return "status"
    if md.is_hazard:
        return "hazard"
    if md.is_heal or md.cures_team_status or md.cures_self_status:
        return "recover"
    if md.is_protect or mid == "substitute":
        return "protect_sub"
    if md.is_phaze or mid == "haze":
        return "phaze_haze"
    return "other"


CLASSES = ["attack", "switch", "setup", "status", "hazard", "recover", "protect_sub", "phaze_haze", "explode",
           "other"]


def d_bin(d: float) -> int:
    return int(np.searchsorted(D_EDGES, d, side="right"))


def state_at(g: dict, t: int) -> float:
    """Material lead (focal minus other, in mon-HP units over six mons) at the start of turn t; the final state
    for a game already over."""
    best = None
    for s in g["snaps"]:
        if s[0] <= t:
            best = s
        else:
            break
    if best is None or g["turns"] < t:
        best = g["final"]
    return best[1] - best[3]


def features(g: dict, fam: Familiarity) -> dict:
    f = {"won": g["result"], "turns": g["turns"]}
    for side in ("f", "o"):
        dec = g[side + "_dec"]
        normal = [d for d in dec if not d["forced"]]
        cnt = defaultdict(int)
        for d in normal:
            kind, _, arg = d["a"].partition(":")
            cnt["switch" if kind == "switch" else move_class(arg)] += 1
        n = max(len(normal), 1)
        for c in CLASSES:
            f[f"{side}_rate_{c}"] = cnt[c] / n
        f[f"{side}_n_normal"] = len(normal)
        f[f"{side}_n_forced"] = len(dec) - len(normal)
        # turn-1 action class
        t1 = next((d for d in normal if d["t"] == 1), None)
        if t1 is not None:
            kind, _, arg = t1["a"].partition(":")
            f[f"{side}_t1"] = "switch" if kind == "switch" else move_class(arg)
        # consecutive voluntary switches (a switch answered by another switch next turn)
        sw_turns = [d["t"] for d in normal if d["a"].startswith("switch")]
        f[f"{side}_double_switch"] = sum(1 for a, b in zip(sw_turns, sw_turns[1:]) if b == a + 1) / n
        # explosion use and outcome
        ex = [m for m in g["moves"] if m[1] == side and m[3] in EXPLODE]
        f[f"{side}_explode_used"] = 1.0 if ex else 0.0
        if ex:
            m = ex[0]
            f[f"{side}_explode_turn"] = m[0]
            f[f"{side}_explode_ko"] = 1.0 if m[5].get("ko") else 0.0
            f[f"{side}_explode_wasted"] = 1.0 if (m[5].get("immune") or m[5].get("protected") or m[5].get("miss")
                                                  or m[5].get("fail")) else 0.0
        mv = [m for m in g["moves"] if m[1] == side]
        att = [m for m in mv if move_class(m[3]) == "attack"]
        sta = [m for m in mv if move_class(m[3]) not in ("attack", "explode")]
        f[f"{side}_n_moves"] = len(mv)
        f[f"{side}_attack_immune"] = sum(1 for m in att if m[5].get("immune")) / max(len(att), 1)
        f[f"{side}_attack_se"] = sum(1 for m in att if m[5].get("supereffective")) / max(len(att), 1)
        f[f"{side}_attack_resisted"] = sum(1 for m in att if m[5].get("resisted")) / max(len(att), 1)
        f[f"{side}_status_failed"] = sum(1 for m in sta if m[5].get("fail") or m[5].get("immune")) / max(len(sta), 1)
        f[f"{side}_n_status_moves"] = len(sta)
        f[f"{side}_n_attacks"] = len(att)
        # switch-ins punished: the mon switched in voluntarily took a super-effective hit or was KO'd that turn
        other = "o" if side == "f" else "f"
        hits = defaultdict(list)
        for m in g["moves"]:
            if m[1] == other and m[4] is not None:
                hits[(m[0], m[4])].append(m[5])
        sw = [(d["t"], d["a"].split(":", 1)[1]) for d in normal if d["a"].startswith("switch")]
        pun = 0
        for t, sp in sw:
            fl = hits.get((t, sp), [])
            if any(x.get("supereffective") or x.get("ko") for x in fl):
                pun += 1
        f[f"{side}_switch_punished"] = pun / max(len(sw), 1)
        f[f"{side}_n_switches"] = len(sw)
        f[f"{side}_kills"] = sum(1 for m in mv if m[5].get("ko"))
    # who lost a mon first
    first = next((x for x in g["faints"]), None)
    f["first_faint"] = None if first is None else first["side"]
    f["first_faint_t"] = None if first is None else first["t"]
    for t in TS:
        f[f"D_{t}"] = state_at(g, t)
    # familiarity
    fm = fam.mons(g["our_panel"], g["f_team"])
    om = fam.mons(g["their_panel"], g["o_team"])
    f["f_sets_seen"] = sum(v[0] for v in fm.values())
    f["o_sets_seen"] = sum(v[0] for v in om.values())
    f["f_max_shared"] = fam.max_shared(g["our_panel"], g["f_team"])
    f["o_max_shared"] = fam.max_shared(g["their_panel"], g["o_team"])
    f["f_species_share"] = float(np.mean([v[1] for v in fm.values()]))
    f["o_species_share"] = float(np.mean([v[1] for v in om.values()]))
    # per-mon contribution: kills made by each mon, faints of each mon
    kills = defaultdict(int)
    for m in g["moves"]:
        if m[5].get("ko"):
            kills[(m[1], m[2])] += 1
    fainted = {(x["side"], x["sp"]) for x in g["faints"]}
    entered = {("f", d["me"]) for d in g["f_dec"] if d["me"]} | {("o", d["me"]) for d in g["o_dec"] if d["me"]}
    for m in g["moves"]:
        entered.add((m[1], m[2]))
    mons = []
    for side, mm in (("f", fm), ("o", om)):
        for sp, (seen, share) in mm.items():
            key = (side, sp)
            mons.append({"side": side, "sp": sp, "seen": seen, "share": share, "entered": key in entered,
                         "kills": kills.get(key, 0), "fainted": key in fainted})
    f["mons"] = mons
    f["f_lead"] = g["f_dec"][0]["me"] if g["f_dec"] and g["f_dec"][0]["me"] else None
    return f


# ---------------------------------------------------------------- weighting + bootstrap
class Group:
    """A set of games with weights (equal weight per panel) and unit ids for the cluster bootstrap."""

    def __init__(self, games: list):
        self.games = games
        by_cell = defaultdict(int)
        for g in games:
            by_cell[g["_panelkey"]] += 1
        k = len(by_cell)
        self.w = np.array([1.0 / (k * by_cell[g["_panelkey"]]) for g in games])
        units = defaultdict(list)
        for i, g in enumerate(games):
            units[(g["_panelkey"], g["_unit"])].append(i)
        self.cells = defaultdict(list)
        for (cell, _), idx in units.items():
            self.cells[cell].append(np.array(idx))

    def resample(self) -> np.ndarray:
        """Indices of one bootstrap replicate (units resampled within each cell)."""
        out = []
        for units in self.cells.values():
            pick = RNG.integers(0, len(units), len(units))
            out.extend(units[j] for j in pick)
        return np.concatenate(out)


def wmean(vals, w):
    vals = np.asarray(vals, dtype=float)
    ok = ~np.isnan(vals)
    if not ok.any():
        return float("nan")
    return float(np.sum(vals[ok] * w[ok]) / np.sum(w[ok]))


def stat_ci(group: Group, fn) -> list:
    """fn(idx, w) -> float; returns [point, lo, hi]."""
    idx0 = np.arange(len(group.games))
    pt = fn(idx0, group.w)
    reps = []
    for _ in range(B):
        idx = group.resample()
        reps.append(fn(idx, group.w[idx]))
    reps = np.array([r for r in reps if not math.isnan(r)])
    return [round(pt, 4), round(float(np.percentile(reps, 2.5)), 4), round(float(np.percentile(reps, 97.5)), 4)]


def contrast_ci(ga: Group, gb: Group, fn) -> dict:
    """fn evaluated on each group; returns a, b and a - b with bootstrap intervals."""
    ia, ib = np.arange(len(ga.games)), np.arange(len(gb.games))
    pa, pb = fn(ga, ia, ga.w), fn(gb, ib, gb.w)
    ra, rb, rd = [], [], []
    for _ in range(B):
        xa, xb = ga.resample(), gb.resample()
        va, vb = fn(ga, xa, ga.w[xa]), fn(gb, xb, gb.w[xb])
        ra.append(va), rb.append(vb), rd.append(va - vb)

    def ci(p, r):
        r = np.array([x for x in r if not math.isnan(x)])
        return [round(p, 4), round(float(np.percentile(r, 2.5)), 4), round(float(np.percentile(r, 97.5)), 4)]

    return {"a": ci(pa, ra), "b": ci(pb, rb), "a_minus_b": ci(pa - pb, rd)}


_COLS: dict = {}


def col(group: Group, key: str) -> np.ndarray:
    ck = (id(group), key)
    if ck not in _COLS:
        _COLS[ck] = np.array([np.nan if g["F"].get(key) is None else float(g["F"][key]) for g in group.games])
    return _COLS[ck]


def did_fn(key: str):
    """Within-cell focal minus other (both sides pilot teams from the same panel, so a team-distribution effect
    cancels); contrasted between groups this is a difference in differences."""
    def fn(group, idx, w):
        return wmean(col(group, "f_" + key)[idx], w) - wmean(col(group, "o_" + key)[idx], w)
    return fn


def mean_fn(key):
    cache = {}

    def fn(group, idx, w):
        if id(group) not in cache:
            cache[id(group)] = col(group, key)
        return wmean(cache[id(group)][idx], w)

    return fn


# ---------------------------------------------------------------- the phase decomposition
def decomposition(ga: Group, gb: Group, t: int) -> dict:
    """gap = W_a - W_b = composition (the material state at turn t) + conversion (what happens after t), with the
    symmetric (midpoint) weights, so the two parts add up exactly."""
    DA = np.array([d_bin(g["F"][f"D_{t}"]) for g in ga.games])
    DB = np.array([d_bin(g["F"][f"D_{t}"]) for g in gb.games])
    YA, YB = col(ga, "won"), col(gb, "won")

    def parts(ia, wa, ib, wb):
        da, db, ya, yb = DA[ia], DB[ib], YA[ia], YB[ib]
        comp = conv = 0.0
        for b in range(len(D_EDGES) + 1):
            ma, mb = da == b, db == b
            pa, pb = wa[ma].sum() / wa.sum(), wb[mb].sum() / wb.sum()
            wa_b = (ya[ma] * wa[ma]).sum() / wa[ma].sum() if ma.any() else None
            wb_b = (yb[mb] * wb[mb]).sum() / wb[mb].sum() if mb.any() else None
            if wa_b is None and wb_b is None:
                continue
            if wa_b is None:
                wa_b = wb_b
            if wb_b is None:
                wb_b = wa_b
            comp += (pa - pb) * (wa_b + wb_b) / 2
            conv += (pa + pb) / 2 * (wa_b - wb_b)
        return comp, conv

    ia, ib = np.arange(len(ga.games)), np.arange(len(gb.games))
    c0, v0 = parts(ia, ga.w, ib, gb.w)
    rc, rv, rs = [], [], []
    for _ in range(B):
        xa, xb = ga.resample(), gb.resample()
        c, v = parts(xa, ga.w[xa], xb, gb.w[xb])
        rc.append(c), rv.append(v)
    def ci(p, r):
        return [round(p, 4), round(float(np.percentile(r, 2.5)), 4), round(float(np.percentile(r, 97.5)), 4)]
    return {"t": t, "gap": ci(c0 + v0, np.array(rc) + np.array(rv)), "state_at_t": ci(c0, rc),
            "after_t": ci(v0, rv), "share_state": round(c0 / (c0 + v0), 3) if abs(c0 + v0) > 1e-9 else None}


# ---------------------------------------------------------------- logistic regression (IRLS)
def logit_fit(X, y, w):
    beta = np.zeros(X.shape[1])
    for _ in range(50):
        p = 1 / (1 + np.exp(-X @ beta))
        W = w * p * (1 - p) + 1e-9
        z = X @ beta + (y - p) / (p * (1 - p) + 1e-9)
        beta_new = np.linalg.solve(X.T @ (X * W[:, None]) + 1e-6 * np.eye(X.shape[1]), X.T @ (W * z))
        if np.max(np.abs(beta_new - beta)) < 1e-8:
            beta = beta_new
            break
        beta = beta_new
    return beta


def familiarity_regression(group: Group, panels: list, covs=("f_sets_seen", "o_sets_seen")) -> dict:
    """win ~ panel fixed effects + covariates (default: our sets seen in the pool, 0-6, and theirs), within U.
    Ties count 0.5 (fractional response). Reports the LAST TWO covariates."""
    FE = np.array([[1.0 if g["_panelkey"] == p else 0.0 for p in panels] for g in group.games])
    CV = np.array([[float(g["F"][c]) for c in covs] for g in group.games])
    XA, YA = np.hstack([FE, CV]), col(group, "won")

    def design(idx):
        return XA[idx], YA[idx]

    X, y = design(np.arange(len(group.games)))
    b0 = logit_fit(X, y, np.ones(len(y)))
    reps = []
    for _ in range(B):
        idx = group.resample()
        Xb, yb = design(idx)
        reps.append(logit_fit(Xb, yb, np.ones(len(yb)))[-2:])
    reps = np.array(reps)

    def ci(j, p):  # noqa: E306
        return [round(float(p), 4), round(float(np.percentile(reps[:, j], 2.5)), 4),
                round(float(np.percentile(reps[:, j], 97.5)), 4)]
    # marginal effect at the U mean (pp of win probability per mon-set seen)
    p = 1 / (1 + np.exp(-X @ b0))
    slope = float(np.mean(p * (1 - p)))
    return {"covariates": list(covs), "beta_second_last_logit": ci(0, b0[-2]), "beta_last_logit": ci(1, b0[-1]),
            "pp_per_unit_second_last": round(100 * slope * b0[-2], 2), "pp_per_unit_last": round(100 * slope * b0[-1], 2),
            "pp_per_unit_ci_second_last": [round(100 * slope * float(np.percentile(reps[:, 0], q)), 2) for q in (2.5, 97.5)],
            "pp_per_unit_ci_last": [round(100 * slope * float(np.percentile(reps[:, 1], q)), 2) for q in (2.5, 97.5)],
            "pp_sum": round(100 * slope * (b0[-2] + b0[-1]), 2),
            "pp_sum_ci": [round(100 * slope * float(np.percentile(reps[:, 0] + reps[:, 1], q)), 2) for q in (2.5, 97.5)],
            "n": int(len(y)), "mean_second_last": round(float(X[:, -2].mean()), 3),
            "mean_last": round(float(X[:, -1].mean()), 3)}


# ---------------------------------------------------------------- per-mon contribution
def ratio_ci(group: Group, pred, val: str) -> dict:
    """Sum over entered mons matching pred of m[val], over their count; equal panel weight; per-game aggregates."""
    num = np.array([sum(float(m[val]) for m in g["F"]["mons"] if m["entered"] and pred(m)) for g in group.games])
    den = np.array([sum(1.0 for m in g["F"]["mons"] if m["entered"] and pred(m)) for g in group.games])

    def fn(idx, w):
        d = np.sum(den[idx] * w)
        return float(np.sum(num[idx] * w) / d) if d else float("nan")
    n = int(den.sum())
    return {"value": stat_ci(group, fn), "n_mons": n} if n >= 30 else {"n_mons": n}


SHARE_BUCKETS = [("absent", lambda s: s == 0.0), ("(0,2%]", lambda s: 0.0 < s <= 0.02),
                 ("(2,10%]", lambda s: 0.02 < s <= 0.10), (">10%", lambda s: s > 0.10)]


def mon_table(group: Group, side: str) -> dict:
    """Per entered mon: kills made and faint rate, by whether its set is in the pool and by its species' pool share."""
    out = {}
    for seen in (True, False):
        tag = "seen" if seen else "unseen"
        pred = (lambda m, seen=seen: m["side"] == side and m["seen"] == seen)
        out[f"kills_{tag}"] = ratio_ci(group, pred, "kills")
        out[f"fainted_{tag}"] = ratio_ci(group, pred, "fainted")
    for name, ok in SHARE_BUCKETS:
        pred = (lambda m, ok=ok: m["side"] == side and ok(m["share"]))
        out[f"kills_species_{name}"] = ratio_ci(group, pred, "kills")
    return out


# ---------------------------------------------------------------- main
def main(src: str, dst: str) -> None:
    fam = Familiarity()
    games = [json.loads(line) for line in gzip.open(src, "rt")]
    for g in games:
        g["F"] = features(g, fam)
        g["_panelkey"] = g["cell"]
        g["_unit"] = g["unit"]
        g.pop("moves"), g.pop("snaps"), g.pop("f_dec"), g.pop("o_dec")

    def grp(read, cells):
        return Group([g for g in games if g["read"] == read and g["cell"] in cells])

    groups = {
        "pstar_home72": grp("x51", {"pstar_home72"}),
        "pstar_U": grp("x51", {"pstar_hpfix", "pstar_forum72", "pstar_p72"}),
        "pstar_H": grp("x51", {"pstar_hpfix"}), "pstar_F": grp("x51", {"pstar_forum72"}),
        "pstar_P72": grp("x51", {"pstar_p72"}),
        "pm_home72": grp("mm", {"pm_home72"}),
        "pm_U": grp("mm", {"pm_hpfix", "pm_forum72", "pm_p72"}),
        "x26_deploy": grp("x37a", {"deploy_core", "deploy_near", "deploy_far"}),
        "x26_reverse": grp("x37a", {"reverse"}),
        "kakuna_kcal": grp("x37a", {"kcal"}),
    }
    # deploy's three strata are one cell for weighting: X37a weights them by game (100 / 400 / 100)
    for g in groups["x26_deploy"].games:
        g["_panelkey"] = "deploy"
    groups["x26_deploy"] = Group(groups["x26_deploy"].games)

    res = {"n_games": {k: len(v.games) for k, v in groups.items()}}
    res["win"] = {k: stat_ci(v, lambda idx, w, v=v: wmean(col(v, "won")[idx], w)) for k, v in groups.items()}

    contrasts = {
        "pstar_home72_vs_U": ("pstar_home72", "pstar_U"),
        "pm_home72_vs_U": ("pm_home72", "pm_U"),
        "x26_deploy_vs_reverse": ("x26_deploy", "x26_reverse"),   # face-H vs pilot-H
        "kcal_vs_x26_reverse": ("kakuna_kcal", "x26_reverse"),    # Kakuna vs X26 piloting H, both facing Kakuna-home72
    }
    # 1. phase decomposition
    res["decomposition"] = {}
    for name, (a, b) in contrasts.items():
        res["decomposition"][name] = [decomposition(groups[a], groups[b], t) for t in TS]
    # 2. material at turn t, first faint
    res["material"] = {}
    for name, (a, b) in contrasts.items():
        ga, gb = groups[a], groups[b]
        r = {}
        for t in [1, 3, 5, 10, 20]:
            r[f"D_{t}"] = contrast_ci(ga, gb, mean_fn(f"D_{t}"))
        def ff(group, idx, w):
            v = np.array([1.0 if group.games[i]["F"]["first_faint"] == "o" else 0.0 for i in idx])
            return wmean(v, w)
        r["first_faint_is_theirs"] = contrast_ci(ga, gb, ff)
        r["turns"] = contrast_ci(ga, gb, mean_fn("turns"))
        res["material"][name] = r
    # conversion of an early state: P(win | first faint theirs) and P(win | first faint ours)
    res["conditional"] = {}
    for name, (a, b) in contrasts.items():
        r = {}
        for who in ("o", "f"):
            def fn(group, idx, w, who=who):
                m = np.array([group.games[i]["F"]["first_faint"] == who for i in idx])
                y = col(group, "won")[idx]
                return wmean(y[m], w[m]) if m.any() else float("nan")
            r[f"win_given_first_faint_{who}"] = contrast_ci(groups[a], groups[b], fn)
        for t in (10, 20):
            for lab, cond in (("ahead", lambda d: d >= 1.0), ("level", lambda d: -1.0 < d < 1.0),
                              ("behind", lambda d: d <= -1.0)):
                def fn(group, idx, w, t=t, cond=cond):
                    m = np.array([cond(group.games[i]["F"][f"D_{t}"]) for i in idx])
                    y = col(group, "won")[idx]
                    return wmean(y[m], w[m]) if m.any() else float("nan")
                def share(group, idx, w, t=t, cond=cond):
                    m = np.array([cond(group.games[i]["F"][f"D_{t}"]) for i in idx], dtype=float)
                    return wmean(m, w)
                r[f"win_given_{lab}_at_{t}"] = contrast_ci(groups[a], groups[b], fn)
                r[f"share_{lab}_at_{t}"] = contrast_ci(groups[a], groups[b], share)
        res["conditional"][name] = r
    # 3. decision mix and decision outcomes, both sides
    keys = ([f"{s}_rate_{c}" for s in ("f", "o") for c in CLASSES]
            + [f"{s}_{k}" for s in ("f", "o") for k in ("double_switch", "explode_used", "explode_ko",
                                                       "explode_wasted", "explode_turn", "attack_immune",
                                                       "attack_se", "attack_resisted", "status_failed",
                                                       "switch_punished", "kills", "n_normal", "n_forced")])
    res["decisions"] = {}
    for name, (a, b) in contrasts.items():
        res["decisions"][name] = {k: contrast_ci(groups[a], groups[b], mean_fn(k)) for k in keys}
    did_keys = [f"rate_{c}" for c in CLASSES] + ["double_switch", "explode_used", "explode_ko", "explode_wasted",
                                                 "attack_immune", "attack_se", "attack_resisted", "status_failed",
                                                 "switch_punished", "kills"]
    res["decisions_did"] = {}
    for name in ("pstar_home72_vs_U", "pm_home72_vs_U"):
        a, b = contrasts[name]
        res["decisions_did"][name] = {k: contrast_ci(groups[a], groups[b], did_fn(k)) for k in did_keys}
    # turn-1 action class mix
    res["turn1"] = {}
    for name, g in groups.items():
        cnt = defaultdict(float)
        for x, w in zip(g.games, g.w):
            cnt[x["F"].get("f_t1", "none")] += w
        tot = sum(cnt.values())
        res["turn1"][name] = {k: round(v / tot, 4) for k, v in sorted(cnt.items())}
    # 4. matchup knowledge: kills per entered mon by set familiarity and species pool share
    res["mons"] = {}
    for name in ("pstar_home72", "pstar_U", "pm_home72", "pm_U", "x26_deploy", "x26_reverse", "kakuna_kcal"):
        res["mons"][name] = {"ours": mon_table(groups[name], "f"), "theirs": mon_table(groups[name], "o")}
    # 4b. our mons minus their mons, kills per entered mon, by set familiarity (symmetric cells only)
    def rel_fn(our_seen, their_seen, share=None):
        def per(group):
            ck = (id(group), "rel", our_seen, their_seen, share)
            if ck not in _COLS:
                arr = []
                for g in group.games:
                    r = [0.0, 0.0, 0.0, 0.0]
                    for m in g["F"]["mons"]:
                        if not m["entered"]:
                            continue
                        if share is not None and not dict(SHARE_BUCKETS)[share](m["share"]):
                            continue
                        if m["side"] == "f" and m["seen"] in our_seen:
                            r[0] += m["kills"]; r[1] += 1
                        if m["side"] == "o" and m["seen"] in their_seen:
                            r[2] += m["kills"]; r[3] += 1
                    arr.append(r)
                _COLS[ck] = np.array(arr)
            return _COLS[ck]

        def fn(group, idx, w):
            a = per(group)[idx] * w[:, None]
            s_ = a.sum(0)
            return s_[0] / s_[1] - s_[2] / s_[3]
        return fn
    res["mons_relative"] = {}
    for name in ("pstar_home72_vs_U", "pm_home72_vs_U"):
        a, b = contrasts[name]
        res["mons_relative"][name] = {
            "all_sets": contrast_ci(groups[a], groups[b], rel_fn((True, False), (True, False))),
            "seen_sets_only": contrast_ci(groups[a], groups[b], rel_fn((True,), (True,))),
        }
        ub = groups[b]
        res["mons_relative"][name]["U_unseen_sets_only"] = stat_ci(
            ub, lambda idx, w, f=rel_fn((False,), (False,)), ub=ub: f(ub, idx, w))
        for bucket in ("(0,2%]", ">10%"):
            for gn in (a, b):
                gg = groups[gn]
                res["mons_relative"][name][f"{gn}_species_{bucket}"] = stat_ci(
                    gg, lambda idx, w, f=rel_fn((True, False), (True, False), bucket), gg=gg: f(gg, idx, w))
    # 5. our team's role vs theirs: familiarity regression within U
    pan = {"pstar_U": ["pstar_hpfix", "pstar_forum72", "pstar_p72"], "pm_U": ["pm_hpfix", "pm_forum72", "pm_p72"]}
    res["familiarity_regression"] = {}
    for gname, panels in pan.items():
        g = groups[gname]
        res["familiarity_regression"][gname] = {
            "sets_ours_theirs": familiarity_regression(g, panels),
            # team context: our sets seen anywhere vs the most of them on ONE pool team (their side as controls)
            "ours_sets_vs_context": familiarity_regression(
                g, panels, ("o_sets_seen", "o_max_shared", "f_sets_seen", "f_max_shared")),
            "theirs_sets_vs_context": familiarity_regression(
                g, panels, ("f_sets_seen", "f_max_shared", "o_sets_seen", "o_max_shared")),
        }
        mx = defaultdict(int)
        for x in g.games:
            mx[(x["F"]["f_max_shared"])] += 1
        res["familiarity_regression"][gname]["f_max_shared_hist"] = dict(sorted(mx.items()))
    json.dump(res, open(dst, "w"), indent=1)
    print(f"wrote {dst}", file=sys.stderr)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
