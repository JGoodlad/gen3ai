"""The record of ``hctx_kk_2026-10-27`` (PREREG.md §4-§5): Kakuna on BOTH sides, rc72b against the coherent reference.
Plays nothing and loads no model.

    CUDA_VISIBLE_DEVICES="" PYTHONPATH=<checkout>/src nice -n 19 taskset -c 5-7,13-15 \\
        python designs/research_state/measurements/hctx_kk_2026-10-27/scripts/record.py \\
        --state ~/gen3ai_archive/hctx_kk_read --out designs/research_state/measurements/hctx_kk_2026-10-27/record.json

Inputs (read only): the frozen registration (``tsr_registration.json``), the lane's state (every unit's
``summary.json``, ``games.jsonl``, captures and peer logs, read through ``main.team_strength.cli.load_kakuna``: team
identity from the capture START lines, the winner from the capture, cross-checked against ``games.jsonl``; the
re-derived guards in ``count`` mode), the H-CTX Amendment-2 read state for D5 (P*'s banked per-team edge on rc72b),
our pool and Smogon's teammate priors for D4 (the review's coherence formulas, ``hctx_rc72_2026-10-26/scripts/
review_checks.py`` C3, restated here: that script is a frozen record).
"""
from __future__ import annotations

import argparse
import collections
import itertools
import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np

HERE = Path(__file__).resolve().parent.parent
REG = HERE / "tsr_registration.json"
HCTX_STATE = Path("/home/goodlad/gen3ai_archive/hctx_rc72_am2_read")
HCTX_RC_GROUPS = ("rc72b_a", "rc72b_b", "rc72b_c")
BAR = 0.450
EDGE = 0.001          # a bound within this of the bar is UNDECIDED (rule 8)
UNITS_PLANNED = 60
UNITS_FLOOR = 54
B = 10_000
SEED = 7274


# ------------------------------------------------------------------------------------------------ games and units
def load(state: Path) -> Dict[str, Any]:
    from main.team_strength import cli
    from main.team_strength import register as RG
    from utils.paths import repo_path

    reg = RG.load(REG)
    kk = cli.load_kakuna(reg, state, repo_path("deps", "pokemon-showdown", "dist", "sim"), cluster="game")
    ref_set = {m["team_id"]: m["set"] for m in reg["reference"]["members"]}
    games = []
    for g in kk["games"]:
        unit = g.cluster.split(":", 1)[0]
        games.append({"unit": unit, "a": g.a, "b": g.b, "s": float(g.s), "control": ref_set[g.b]})
    return {"reg": reg, "games": games, "verdicts": kk["verdicts"], "self_games": kk["self_games"]}


def by_unit(games: Sequence[Mapping[str, Any]], keep=lambda g: True) -> Dict[str, Tuple[float, int]]:
    u: Dict[str, List[float]] = collections.defaultdict(lambda: [0.0, 0])
    for g in games:
        if keep(g):
            u[g["unit"]][0] += g["s"]
            u[g["unit"]][1] += 1
    return {k: (v[0], v[1]) for k, v in sorted(u.items())}


def unit_boot(units: Mapping[str, Tuple[float, int]], seed: int = SEED, b: int = B) -> Dict[str, Any]:
    """Point and 95 % percentile interval of the pooled score, resampling UNITS (PREREG §4)."""
    keys = sorted(units)
    s = np.array([units[k][0] for k in keys])
    n = np.array([units[k][1] for k in keys])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(keys), size=(b, len(keys)))
    reps = s[idx].sum(1) / n[idx].sum(1)
    return {"S": float(s.sum() / n.sum()), "ci": [float(np.percentile(reps, 2.5)), float(np.percentile(reps, 97.5))],
            "games": int(n.sum()), "units": len(keys), "unit_sd": float(np.std(s / n, ddof=1)),
            "binomial_unit_sd": float(math.sqrt((s.sum() / n.sum()) * (1 - s.sum() / n.sum()) / n.mean()))}


def diff_boot(u1: Mapping[str, Tuple[float, int]], u2: Mapping[str, Tuple[float, int]], seed: int) -> Dict[str, Any]:
    """S1 - S2 over the SAME units (each unit's games split by control), units resampled jointly."""
    keys = sorted(set(u1) & set(u2))
    a = np.array([[u1[k][0], u1[k][1], u2[k][0], u2[k][1]] for k in keys])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(keys), size=(B, len(keys)))
    r = a[idx].sum(1)
    reps = r[:, 0] / r[:, 1] - r[:, 2] / r[:, 3]
    t = a.sum(0)
    return {"diff": float(t[0] / t[1] - t[2] / t[3]),
            "ci": [float(np.percentile(reps, 2.5)), float(np.percentile(reps, 97.5))], "units": len(keys)}


def team_boot(games: Sequence[Mapping[str, Any]], seed: int) -> Dict[str, Any]:
    """D2: side A's teams resampled, then units (two-stage); each resampled team weighs its games by multiplicity."""
    teams = sorted({g["a"] for g in games})
    units = sorted({g["unit"] for g in games})
    ti = {t: i for i, t in enumerate(teams)}
    ui = {u: i for i, u in enumerate(units)}
    s = np.zeros((len(teams), len(units)))
    n = np.zeros((len(teams), len(units)))
    for g in games:
        s[ti[g["a"]], ui[g["unit"]]] += g["s"]
        n[ti[g["a"]], ui[g["unit"]]] += 1
    rng = np.random.default_rng(seed)
    reps = np.empty(B)
    for k in range(B):
        wt = np.bincount(rng.integers(0, len(teams), len(teams)), minlength=len(teams))
        wu = np.bincount(rng.integers(0, len(units), len(units)), minlength=len(units))
        w = np.outer(wt, wu)
        reps[k] = (w * s).sum() / max((w * n).sum(), 1)
    return {"ci": [float(np.percentile(reps, 2.5)), float(np.percentile(reps, 97.5))], "teams": len(teams)}


def classify(ci: Sequence[float]) -> Dict[str, Any]:
    lo, hi = ci
    near = abs(lo - BAR) < EDGE or abs(hi - BAR) < EDGE
    if near:
        cls = "UNDECIDED"
    elif hi < BAR:
        cls = "MATERIALLY WEAKER"
    elif lo > BAR:
        cls = "NOT MATERIALLY WEAKER"
    else:
        cls = "UNDECIDED"
    return {"class": cls, "bound_near_bar": near, "detected_weaker": bool(hi < 0.5)}


def elo(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return 400.0 * math.log10(p / (1 - p))


# ------------------------------------------------------------------------------------------------ D4: coherence
def coherence(reg: Mapping[str, Any]) -> Dict[str, Dict[str, float]]:
    """Per rc72b team id: Smogon teammate lift and pool species-pair PMI (the review's C3 formulas, as logs)."""
    from agents.gen3_data import priors
    from main.anchors.session import pool_team_texts
    from main.anchors.team_overlap import parse_team
    from main.team_strength import panel as PN

    pool = [parse_team(t) for t in pool_team_texts()]
    if len(pool) != 719:
        raise SystemExit(f"pool has {len(pool)} teams, expected 719")
    sp_pair_n = collections.Counter(p for t in pool for p in itertools.combinations(sorted(t.species6), 2))
    sp_n = collections.Counter(s for t in pool for s in t.species6)
    usage = priors.species_usage()
    tot = sum(usage.values())
    cache: Dict[Tuple[str, str], float] = {}

    def lift(t: str, s: str) -> float:
        if (t, s) not in cache:
            tm = priors.teammates(t)
            share = usage.get(s, 0.0) / tot
            raw = (tm.get(s, 0.0) / share) if (tm and share > 0) else 1.0
            cache[(t, s)] = min(max(raw, 0.25), 4.0)
        return cache[(t, s)]

    sets = PN.load_registered(reg, ["rc72b"])
    out = {}
    for m in sets["rc72b"].members:
        if m.team_id in out:
            continue
        sp = sorted(parse_team(m.text).species6)
        out[m.team_id] = {
            "smogon_lift": float(np.mean([math.log(lift(a, b)) for a in sp for b in sp if a != b])),
            "species_pair_pmi": float(np.mean([math.log((sp_pair_n[q] + 0.5) * 719 / ((sp_n[q[0]] + 0.5) *
                                                                                      (sp_n[q[1]] + 0.5)))
                                               for q in itertools.combinations(sp, 2)])),
        }
    return out


def per_team(games: Sequence[Mapping[str, Any]]) -> Dict[str, Tuple[float, int]]:
    t: Dict[str, List[float]] = collections.defaultdict(lambda: [0.0, 0])
    for g in games:
        t[g["a"]][0] += g["s"]
        t[g["a"]][1] += 1
    return {k: (v[0], v[1]) for k, v in t.items()}


def d4(games, coh) -> Dict[str, Any]:
    pt = per_team(games)
    teams = sorted(set(pt) & set(coh))
    out: Dict[str, Any] = {"teams": len(teams)}
    for col in ("smogon_lift", "species_pair_pmi"):
        x = np.array([coh[t][col] for t in teams])
        y = np.array([pt[t][0] / pt[t][1] for t in teams])
        cuts = np.percentile(x, [100 / 3, 200 / 3])
        band = {t: int(np.searchsorted(cuts, coh[t][col], side="right")) for t in teams}
        terc = {}
        for k in range(3):
            gs = [g for g in games if band.get(g["a"]) == k]
            xs = [coh[t][col] for t in teams if band[t] == k]
            terc[str(k)] = {"x_range": [float(min(xs)), float(max(xs))], "games": len(gs),
                            "S": float(np.mean([g["s"] for g in gs]))}
        out[col] = {"terciles": terc, "corr_team_score": float(np.corrcoef(x, y)[0, 1]),
                    "x_mean": float(x.mean())}
    return out


# ------------------------------------------------------------------------------------------------ D5: H-CTX bridge
def pstar_edges(reg: Mapping[str, Any]) -> Dict[str, Dict[str, float]]:
    """P*'s banked per-team edge e(T) on rc72b (REVIEW C4): the mean of P*'s score when P* holds T and when Kakuna
    holds T, so T's own strength cancels. P*'s team is its own row's ``our_team``; Kakuna's is the peer report's
    ``team_draws`` in play order (REVIEW F1: never the CSV's Team File column). Labels map to team ids by file."""
    from main.anchors.session import load_team_texts
    from main.team_strength import panel as PN

    d = Path(reg["sets"]["rc72b"]["dir"])
    by_label = {lab: PN.team_id(txt) for lab, txt in load_team_texts(d)}
    hold = collections.defaultdict(lambda: [0, 0])
    face = collections.defaultdict(lambda: [0, 0])
    for g in HCTX_RC_GROUPS:
        for unit in sorted((HCTX_STATE / "anchor" / g / "pstar" / "cells").iterdir()):
            rows = [json.loads(x) for x in (unit / "games.jsonl").read_text().splitlines() if x.strip()]
            for half in ("ours_challenge", "peer_challenge"):
                rep = unit / half / ("peer_acceptor_report.json" if half == "ours_challenge"
                                     else "peer_challenger_report.json")
                draws = [x.replace(".gen3ou_team", ".txt") for x in json.loads(rep.read_text())["team_draws"]]
                gs = sorted((r for r in rows if r["half"] == half), key=lambda r: r["index"])
                if len(gs) != len(draws):
                    raise SystemExit(f"{unit.name} {half}: {len(gs)} games, {len(draws)} draws")
                for r, theirs in zip(gs, draws):
                    w = 1.0 if r["result"] == "win" else (0.5 if r["result"] == "tie" else 0.0)
                    hold[by_label[r["our_team"]]][0] += w
                    hold[by_label[r["our_team"]]][1] += 1
                    face[by_label[theirs]][0] += w
                    face[by_label[theirs]][1] += 1
    return {t: {"edge": (hold[t][0] / hold[t][1] + face[t][0] / face[t][1]) / 2,
                "games": hold[t][1] + face[t][1]} for t in set(hold) & set(face)}


def d5(games, coh, edges) -> Dict[str, Any]:
    pt = per_team(games)
    teams = sorted(set(pt) & set(coh) & set(edges))
    k = np.array([pt[t][0] / pt[t][1] for t in teams])
    e = np.array([edges[t]["edge"] for t in teams])
    out: Dict[str, Any] = {"teams": len(teams), "corr_edge_kakuna_strength": float(np.corrcoef(e, k)[0, 1])}
    for col in ("smogon_lift", "species_pair_pmi"):
        x = np.array([coh[t][col] for t in teams])
        # partial correlation of e with x, controlling for the team's Kakuna-vs-Kakuna strength
        X = np.column_stack([np.ones_like(k), k])
        re = e - X @ np.linalg.lstsq(X, e, rcond=None)[0]
        rx = x - X @ np.linalg.lstsq(X, x, rcond=None)[0]
        out[col] = {"corr_edge": float(np.corrcoef(e, x)[0, 1]),
                    "partial_corr_edge_given_strength": float(np.corrcoef(re, rx)[0, 1])}
    lo = np.percentile(k, 100 / 3)
    hi = np.percentile(k, 200 / 3)
    out["edge_by_kakuna_strength_tercile"] = {
        str(i): float(np.mean(e[m])) for i, m in enumerate((k <= lo, (k > lo) & (k <= hi), k > hi))}
    return out


# ------------------------------------------------------------------------------------------------ D6 + main
def d6(state: Path, verdicts, games) -> Dict[str, Any]:
    stamps = collections.Counter(json.dumps({x: v.get(x) for x in ("engine_stamp", "peer_code", "peer_pack_changed",
                                                                   "server_version")}, sort_keys=True)
                                 for v in verdicts if v["counts"])
    turns, seats = [], collections.defaultdict(list)
    for v in verdicts:
        if not v["counts"]:
            continue
        gj = state / "kakuna" / "cells" / v["unit"] / "games.jsonl"
        for line in gj.read_text().splitlines():
            if line.strip():
                turns.append(int(json.loads(line).get("turns") or 0))
    return {"units_counted": sum(v["counts"] for v in verdicts), "units_seen": len(verdicts),
            "refused": [{"unit": v["unit"], "reasons": v["reasons"]} for v in verdicts if not v["counts"]],
            "rederived": collections.Counter(v["rederived"]["status"] for v in verdicts),
            "rederived_hits": [{"unit": v["unit"], "hits": v["rederived"]["hits"]} for v in verdicts
                               if v["rederived"]["hits"]],
            "stamps": {k: n for k, n in stamps.items()}, "ties": sum(g["s"] == 0.5 for g in games),
            "turns_mean": float(np.mean(turns)) if turns else None,
            "turns_median": float(np.median(turns)) if turns else None}


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-bridge", action="store_true", help="skip D5 (needs the H-CTX read state)")
    a = ap.parse_args(argv)
    state = Path(a.state).expanduser()
    L = load(state)
    games, verdicts = L["games"], L["verdicts"]
    integ = d6(state, verdicts, games)
    n_ok = integ["units_counted"]
    one_regime = len(integ["stamps"]) == 1
    rec: Dict[str, Any] = {"schema": "hctx_kk_record_v1", "read": "hctx_kk_2026-10-27",
                           "registration_digest": L["reg"]["digest"], "units_planned": UNITS_PLANNED,
                           "units_floor": UNITS_FLOOR, "bar": BAR, "integrity": integ,
                           "self_games_dropped": L["self_games"]}
    prim = unit_boot(by_unit(games))
    prim["elo"] = elo(prim["S"])
    prim["elo_ci"] = [elo(prim["ci"][0]), elo(prim["ci"][1])]
    if n_ok < UNITS_FLOOR:
        verdict = {"class": "INCONCLUSIVE", "why": f"{n_ok} of {UNITS_PLANNED} units counted (< {UNITS_FLOOR})"}
    elif not one_regime:
        verdict = {"class": "INCONCLUSIVE", "why": f"counted units span {len(integ['stamps'])} engine / peer stamps"}
    else:
        verdict = classify(prim["ci"])
    rec["primary"] = prim
    rec["verdict"] = verdict
    rec["verdict_line"] = (f"hctx_kk_2026-10-27: rc72b {verdict['class']} under Kakuna "
                           f"(S = {prim['S']:.3f} [{prim['ci'][0]:.3f}, {prim['ci'][1]:.3f}])")
    uh = by_unit(games, lambda g: g["control"] == "home72")
    uf = by_unit(games, lambda g: g["control"] == "forum72")
    rec["D1"] = {"vs_home72": unit_boot(uh, SEED + 1), "vs_forum72": unit_boot(uf, SEED + 2),
                 "home72_minus_forum72": diff_boot(uh, uf, SEED + 3)}
    rec["D2"] = team_boot(games, SEED + 4)
    coh = coherence(L["reg"])
    rec["D4"] = d4(games, coh)
    if not a.no_bridge:
        rec["D5"] = d5(games, coh, pstar_edges(L["reg"]))
    Path(a.out).write_text(json.dumps(rec, indent=1, sort_keys=True, default=float) + "\n")
    print(rec["verdict_line"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
