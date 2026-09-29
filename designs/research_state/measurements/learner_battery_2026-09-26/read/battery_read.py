#!/usr/bin/env python3
"""The learner battery's STRENGTH reads, the C_fix-vs-C bug-effect read and the G0' plateau read —
from the durable per-battle / per-unit rows of the N0 end-of-run CPU queue. Model-free, torch-free.

    PYTHONPATH=src python battery_read.py [--state DIR] [--json OUT.json]

Inputs (written by ``n0_endofrun_2026-09-27/scripts/n0_queue.py`` at the queue's pin):
  <state>/gu_rows/<LABEL>/t<ti>/c<cc>.jsonl  one row per untaught-8 battle (label, team_key, j, won…)
  <state>/anchors/GA_<LABEL>_<away|home>_s<S>/summary.json   one 100-game SmallRL greedy unit

UNTAUGHT-8 (registration §4.2): per-team win rate = wins / finished (a tie is not a win; an
unfinished battle is a TIMEOUT, never a loss); Δ = the equal-weight CLUSTER bootstrap over the 8
teams, using ``agents.training.untaught_meter``'s own ``bootstrap_index`` / ``cluster_ci`` with ONE
fixed index set shared by every contrast (paired), 20,000 draws, seed 20260915 (the registered seed;
the engine's default is 20260906, reported as a sensitivity row). A cell is a pure function of
(ref, team, battle j), so a contrast at a SUBSET of j is well defined: both refs are read on the
SAME j set (``--jmax`` / the common set).

G-A (§4.2): pool each model's 12 × 100-game units (6 team seeds × {away, home}) = 1,200 games;
arm − C by Newcombe 95 % (the anchors tool's own). A unit that is not status OK, n = 100,
regime-verified per decision and team-source symmetric makes the model's G-A VOID.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from agents.training.untaught_meter import (DEFAULT_BOOTSTRAP_SEED, bootstrap_index,
                                            cluster_ci)
from main.anchors.results import newcombe, wilson

STATE = Path("/home/goodlad/dev/gen3ai-reads/n0_endofrun_2026-09-27")
DRAWS = 20000
SEED_REG = 20260915
BAR_U = 3.69
FLOOR_GA = 11.0
GA_SEEDS = (0, 10, 20, 30, 40, 50)
LABELS = ("C", "E5", "L95", "Cfix", "K2", "K3", "N0", "T32b")


def load_u(state: Path, label: str) -> dict:
    """{team_key: {j: row}} with duplicate-j and cross-file checks."""
    out: dict = defaultdict(dict)
    trees, refs, opps = set(), set(), set()
    for f in sorted((state / "gu_rows" / label).glob("t*/c*.jsonl")):
        for line in f.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            assert r["label"] == label, (f, r["label"])
            j = int(r["j"])
            if j in out[r["team_key"]]:
                assert out[r["team_key"]][j]["won"] == r["won"], ("non-deterministic duplicate", f, j)
            out[r["team_key"]][j] = r
            trees.add(r["tree"]); refs.add(r["ref"]); opps.add(r["opponent"])
    return {"rows": dict(out), "trees": sorted(trees), "refs": sorted(refs), "opps": sorted(opps)}


def rates(u: dict, teams: list, js: dict) -> tuple:
    """per-team win rate over the given j set per team; plus counts."""
    r, fin, att, wins, ties = [], 0, 0, 0, 0
    for t in teams:
        rows = [u["rows"][t][j] for j in js[t]]
        f = sum(x["finished"] for x in rows)
        w = sum(x["won"] for x in rows)
        r.append(w / f if f else 0.0)
        fin += f; att += len(rows); wins += w; ties += sum(x["tied"] for x in rows)
    return np.array(r), {"attempted": att, "finished": fin, "timeouts": att - fin, "wins": wins,
                         "ties": ties}


def common_js(us: list, teams: list, jmax: int | None) -> dict:
    js = {}
    for t in teams:
        s = set.intersection(*[set(u["rows"].get(t, {})) for u in us])
        if jmax is not None:
            s = {j for j in s if j < jmax}
        js[t] = sorted(s)
    return js


def u_contrast(U: dict, a: str, b: str, teams: list, idx: np.ndarray, jmax: int | None,
               idx_alt: np.ndarray | None = None) -> dict:
    js = common_js([U[a], U[b]], teams, jmax)
    ra, ca = rates(U[a], teams, js)
    rb, cb = rates(U[b], teams, js)
    d, lo, hi = cluster_ci(ra - rb, idx)
    la, lb = cluster_ci(ra, idx), cluster_ci(rb, idx)
    out = {"a": a, "b": b, "games_per_team": sorted({len(v) for v in js.values()}),
           "level_a_pp": la, "level_b_pp": lb, "delta_pp": d, "ci95_pp": [lo, hi],
           "per_team_delta_pp": {t: round(100 * (x - y), 2) for t, x, y in zip(teams, ra, rb)},
           "teams_up": int(((ra - rb) > 0).sum()), "counts_a": ca, "counts_b": cb}
    if idx_alt is not None:
        _, alo, ahi = cluster_ci(ra - rb, idx_alt)
        out["ci95_pp_engine_default_seed"] = [alo, ahi]
    return out


def classify(d, lo, hi, bar=BAR_U):
    if abs(d) > bar and (hi < -bar if d < 0 else lo > bar):
        return "OUTSIDE, BELOW" if d < 0 else "OUTSIDE, ABOVE"
    if lo > -bar and hi < bar:
        return "EQUIVALENT"
    return "NOT DETECTED"


def plateau(d, lo, hi, bar=BAR_U):
    """new_lineage §3.1: PLATEAU iff the point is inside ±bar AND the CI does not exclude +bar from
    above (not OUTSIDE ABOVE)."""
    return abs(d) < bar and not (lo > bar)


def load_ga(state: Path, label: str) -> dict:
    units, problems, missing = {}, [], []
    for ts in ("away", "home"):
        for s in GA_SEEDS:
            name = f"GA_{label}_{ts}_s{s}"
            f = state / "anchors" / name / "summary.json"
            if not f.exists():
                missing.append(name)
                continue
            d = json.loads(f.read_text())
            pr = d.get("provenance", {}).get("peer_report", {})
            rec = {"n": d["n"], "wins": d["wins"], "losses": d["losses"], "ties": d["ties"],
                   "status": d["status"], "hit_forfeit_limit": d.get("hit_forfeit_limit"),
                   "regime_verified_decisions": pr.get("series_regime_verified_decisions",
                                                       pr.get("regime_verified_decisions")),
                   "argmax_match_rate": pr.get("argmax_match_rate"),
                   "team_source_asymmetry": d.get("team_source_asymmetry"),
                   "model_zip": d.get("cell", {}).get("model_zip"),
                   "model_step": d.get("cell", {}).get("model_step"),
                   "server_version": d.get("cell", {}).get("server_version")}
            units[name] = rec
            if rec["status"] != "OK" or rec["n"] != 100 or rec["regime_verified_decisions"] is not True \
                    or rec["team_source_asymmetry"]:
                problems.append(name)
    w = sum(u["wins"] for u in units.values())
    n = sum(u["n"] for u in units.values())
    by_ts = {ts: (sum(u["wins"] for k, u in units.items() if f"_{ts}_" in k),
                  sum(u["n"] for k, u in units.items() if f"_{ts}_" in k)) for ts in ("away", "home")}
    return {"units": units, "problems": problems, "missing": missing, "wins": w, "n": n,
            "rate": w / n if n else None, "wilson95": list(wilson(w, n)[1:]) if n else None,
            "by_teamset": by_ts, "hit_forfeit_limit": sum(u["hit_forfeit_limit"] or 0 for u in units.values()),
            "complete": not missing and not problems and n == 1200}


def ga_contrast(G: dict, a: str, b: str) -> dict:
    d, lo, hi = newcombe(G[a]["wins"], G[a]["n"], G[b]["wins"], G[b]["n"])
    return {"a": a, "b": b, "delta_pp": 100 * d, "ci95_pp": [100 * lo, 100 * hi],
            "n": [G[a]["n"], G[b]["n"]],
            "OUTSIDE_BELOW_floor": d * 100 < -FLOOR_GA and hi * 100 < -FLOOR_GA,
            "verdict_vs_0": "NOT DETECTED" if lo <= 0 <= hi else "DETECTED"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", default=str(STATE))
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    state = Path(a.state)

    U = {lab: load_u(state, lab) for lab in LABELS}
    teams = sorted(U["C"]["rows"])
    assert len(teams) == 8, teams
    idx = bootstrap_index(8, DRAWS, SEED_REG)
    idx_alt = bootstrap_index(8, DRAWS, DEFAULT_BOOTSTRAP_SEED)

    doc: dict = {"state": str(state), "teams": teams,
                 "bootstrap": {"draws": DRAWS, "seed": SEED_REG, "sensitivity_seed": DEFAULT_BOOTSTRAP_SEED,
                               "index_set": "ONE fixed set shared by every contrast (paired)"},
                 "provenance": {lab: {"trees": U[lab]["trees"], "refs": U[lab]["refs"], "opponents": U[lab]["opps"],
                                      "games_per_team": {t: len(U[lab]["rows"].get(t, {})) for t in teams}}
                                for lab in LABELS}}
    full = {lab: all(len(U[lab]["rows"].get(t, {})) == 600 for t in teams) for lab in LABELS}
    doc["complete_600"] = full

    # levels at 600/team where complete
    lv = {}
    for lab in LABELS:
        if full[lab]:
            js = {t: list(range(600)) for t in teams}
            r, c = rates(U[lab], teams, js)
            lv[lab] = {"level_pp": cluster_ci(r, idx), "counts": c,
                       "per_team_pp": {t: round(100 * x, 2) for t, x in zip(teams, r)}}
    doc["levels_600"] = lv

    # 1. battery (all on the buggy compiled learner: C, E5, L95)
    doc["battery"] = {arm: u_contrast(U, arm, "C", teams, idx, 600, idx_alt) for arm in ("E5", "L95")}
    for arm, row in doc["battery"].items():
        row["class"] = classify(row["delta_pp"], *row["ci95_pp"])
        row["non_inferior"] = row["ci95_pp"][0] > -BAR_U

    # 2. bug effect
    doc["bug_effect_U"] = u_contrast(U, "Cfix", "C", teams, idx, 600, idx_alt)
    doc["bug_effect_U"]["class"] = classify(doc["bug_effect_U"]["delta_pp"], *doc["bug_effect_U"]["ci95_pp"])
    doc["bug_effect_U_first200"] = u_contrast(U, "Cfix", "C", teams, idx, 200)

    # 3. plateau blocks: registered G-U n (200/team) + the 600/team descriptor (N0 on its common set)
    blocks = [("block1 (N0->Cfix)", "Cfix", "N0"), ("block2 (Cfix->K2)", "K2", "Cfix"),
              ("block3 (K2->K3)", "K3", "K2"), ("buggy block1 (N0->C)", "C", "N0")]
    doc["plateau"] = {}
    for name, end, start in blocks:
        rows = {}
        for tag, jmax in (("200", 200), ("600_or_common", 600)):
            r = u_contrast(U, end, start, teams, idx, jmax)
            r["plateau"] = plateau(r["delta_pp"], *r["ci95_pp"])
            r["class"] = classify(r["delta_pp"], *r["ci95_pp"])
            rows[tag] = r
        doc["plateau"][name] = rows
    # cumulative descriptors
    doc["cumulative"] = {f"{e}-{s}": u_contrast(U, e, s, teams, idx, 600)
                         for e, s in (("K3", "Cfix"), ("K3", "N0"), ("K2", "N0"), ("K3", "C"), ("K2", "C"),
                                     ("C", "N0"), ("E5", "N0"), ("L95", "N0"), ("Cfix", "N0"))}

    # G-A
    G = {lab: load_ga(state, lab) for lab in ("C", "E5", "L95", "Cfix", "K2", "K3", "N0")}
    doc["ga_levels"] = {k: {kk: vv for kk, vv in v.items() if kk != "units"} for k, v in G.items()}
    doc["ga_units"] = {k: v["units"] for k, v in G.items()}
    ok = [k for k, v in G.items() if v["complete"]]
    doc["ga_contrasts"] = {f"{x}-{y}": ga_contrast(G, x, y)
                           for x, y in (("E5", "C"), ("L95", "C"), ("Cfix", "C"), ("K2", "Cfix"),
                                        ("K3", "K2"), ("K3", "Cfix")) if x in ok and y in ok}
    doc["ga_contrasts_by_teamset"] = {}
    for x, y in (("E5", "C"), ("L95", "C"), ("Cfix", "C")):
        if x in ok and y in ok:
            doc["ga_contrasts_by_teamset"][f"{x}-{y}"] = {
                ts: dict(zip(("delta", "lo", "hi"), [100 * v for v in newcombe(*G[x]["by_teamset"][ts], *G[y]["by_teamset"][ts])]))
                for ts in ("away", "home")}

    # print
    def fmt(r):
        return f"{r['delta_pp']:+6.2f} [{r['ci95_pp'][0]:+6.2f}, {r['ci95_pp'][1]:+6.2f}]"
    print("teams:", teams)
    print("complete at 600/team:", full)
    for lab, v in lv.items():
        m, lo, hi = v["level_pp"]
        print(f"  U({lab:4s}) = {m:6.2f} [{lo:6.2f}, {hi:6.2f}]  timeouts {v['counts']['timeouts']}/{v['counts']['attempted']} ties {v['counts']['ties']}")
    print("\n== BATTERY (untaught Δ vs C, 600/team)")
    for arm, r in doc["battery"].items():
        print(f"  {arm:4s} − C = {fmt(r)}  {r['class']}  non-inferior {r['non_inferior']}  teams up {r['teams_up']}/8  (seed-default CI {r['ci95_pp_engine_default_seed']})")
    r = doc["bug_effect_U"]
    print(f"\n== BUG EFFECT  Cfix − C = {fmt(r)}  {r['class']}  teams up {r['teams_up']}/8  (seed-default CI {r['ci95_pp_engine_default_seed']})")
    print(f"   first 200/team: {fmt(doc['bug_effect_U_first200'])}")
    print("\n== PLATEAU blocks (PLATEAU iff |Δ| < 3.69 and not OUTSIDE ABOVE)")
    for name, rows in doc["plateau"].items():
        for tag, r in rows.items():
            print(f"  {name:22s} n/team={r['games_per_team']} {fmt(r)} {r['class']:14s} plateau={r['plateau']}  teams up {r['teams_up']}/8")
    print("\n== cumulative (600/team or common)")
    for k, r in doc["cumulative"].items():
        print(f"  {k:8s} {fmt(r)}  n/team={r['games_per_team']}")
    print("\n== G-A SmallRL greedy (pooled 1,200)")
    for k, v in G.items():
        print(f"  {k:4s} {v['wins']}/{v['n']} = {v['rate'] if v['rate'] is None else round(v['rate'], 4)} "
              f"wilson {v['wilson95']} problems {v['problems']} missing {len(v['missing'])} forfeit-limit {v['hit_forfeit_limit']}")
    for k, r in doc["ga_contrasts"].items():
        print(f"  {k:8s} {fmt(r)}  OUTSIDE BELOW −11.0: {r['OUTSIDE_BELOW_floor']}  vs 0: {r['verdict_vs_0']}")
    if a.json:
        Path(a.json).write_text(json.dumps(doc, indent=1, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
