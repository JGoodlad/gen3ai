#!/usr/bin/env python3
"""REGISTRATION §5.2–5.3: the U (untaught-8) and G-A (SmallRL guard) reads and the Part L decision rules,
from the durable rows of `meters.py` (/home/goodlad/dev/gen3ai-reads/m5_sizing). Model-free, torch-free.

    PYTHONPATH=src python read_meters.py [--json OUT] [--allow-partial]

U: per-team win rate = wins / finished; Δ by the meter's own cluster bootstrap over the 8 teams
(`untaught_meter.bootstrap_index` / `cluster_ci`), 20,000 draws, registered seed 20260915, ONE index set
shared by every contrast (paired by team), on the common battle set. The aggregation is the learner
battery's (`learner_battery_2026-09-26/read/battery_read.py`: `load_u`, `rates`, `common_js`,
`u_contrast`), imported, not re-implemented.
G-A: pool each model's 12 x 100-game units; Δ by the anchors tool's Newcombe 95 %. A unit that is not OK,
n = 100, regime-verified and team-source symmetric makes that model's G-A VOID.

Rules (§5.3; BAR_U 3.69 pp, G-A floor 11.0 pp):
  N VETO  iff  U(B)-U(A2) < -3.69 AND its CI hi < 0 AND |U(B)-U(A2)| > |U(A')-U(A2)|,
          OR   G-A(B)-G-A(A2) CI hi < 0 AND |Δ| > 11.0;  else "no loss detected" (NOT "equivalent").
  E5 ADOPT iff U(C)-U(B) CI lo > -3.69 AND (Δ >= 0 OR |Δ| <= |U(A')-U(A2)|) AND G-A Δ CI lo > -11.0
          AND C's update faster (from the descriptors); else E10 STAYS.
  |U(A')-U(A2)| > 3.69 => both verdicts flagged "run floor exceeds bar — n = 1 is not decisive".
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np

from agents.training.untaught_meter import DEFAULT_BOOTSTRAP_SEED, bootstrap_index
from main.anchors.results import newcombe, wilson

HERE = Path(__file__).resolve().parent
ROWS = Path("/home/goodlad/dev/gen3ai-reads/m5_sizing")
BR = HERE.parents[1] / "learner_battery_2026-09-26" / "read" / "battery_read.py"
DRAWS, SEED_REG, BAR_U, FLOOR_GA = 20000, 20260915, 3.69, 11.0
GA_SEEDS = (0, 10, 20, 30, 40, 50)
LABELS = ("A2", "B", "C", "Ap", "A")


def _battery():
    spec = importlib.util.spec_from_file_location("battery_read", BR)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def load_ga(label: str) -> dict:
    units, problems, missing = {}, [], []
    for ts in ("away", "home"):
        for s in GA_SEEDS:
            f = ROWS / "ga" / label / f"{ts}_s{s}" / "summary.json"
            if not f.exists():
                missing.append(f"{ts}_s{s}")
                continue
            d = json.loads(f.read_text())
            pr = d.get("provenance", {}).get("peer_report", {})
            rec = {"n": d["n"], "wins": d["wins"], "status": d["status"],
                   "regime_verified_decisions": pr.get("series_regime_verified_decisions",
                                                       pr.get("regime_verified_decisions", d.get("regime_verified_decisions"))),
                   "team_source_asymmetry": d.get("team_source_asymmetry")}
            units[f"{ts}_s{s}"] = rec
            if rec["status"] != "OK" or rec["n"] != 100 or rec["regime_verified_decisions"] is not True \
                    or rec["team_source_asymmetry"]:
                problems.append(f"{ts}_s{s}")
    w = sum(u["wins"] for u in units.values())
    n = sum(u["n"] for u in units.values())
    return {"wins": w, "n": n, "rate": w / n if n else None, "wilson95": list(wilson(w, n)[1:]) if n else None,
            "problems": problems, "missing": missing, "complete": not missing and not problems and n == 1200}


def ga_delta(a: dict, b: dict) -> dict | None:
    if not (a["n"] and b["n"]):
        return None
    d, lo, hi = newcombe(a["wins"], a["n"], b["wins"], b["n"])
    return {"delta_pp": 100 * d, "ci95_pp": [100 * lo, 100 * hi]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None)
    ap.add_argument("--allow-partial", action="store_true", help="read an incomplete set (PROGRESS only, never a verdict)")
    a = ap.parse_args()
    br = _battery()
    U = {}
    for lab in LABELS:
        if (ROWS / "u_rows" / lab).exists():
            U[lab] = _load_u_rows(lab)
    teams = sorted(next(iter(U.values()))["rows"]) if U else []
    idx = bootstrap_index(len(teams), DRAWS, seed=SEED_REG)
    idx_alt = bootstrap_index(len(teams), DRAWS, seed=DEFAULT_BOOTSTRAP_SEED)
    out: dict = {"levels": {}, "contrasts": {}, "ga": {}, "rules": {}}
    for lab, u in U.items():
        js = br.common_js([u], teams, None)
        r, c = br.rates(u, teams, js)
        out["levels"][lab] = {"games_per_team": sorted({len(v) for v in js.values()}), "counts": c,
                              "mean_pp": 100 * float(r.mean()), "per_team_pp": [round(100 * x, 1) for x in r]}
        out["ga"][lab] = load_ga(lab)
    for x, y in (("B", "A2"), ("Ap", "A2"), ("C", "B"), ("A", "A2"), ("C", "A2")):
        if x in U and y in U:
            c = br.u_contrast(U, x, y, teams, idx, None, idx_alt)     # already in pp (cluster_ci)
            out["contrasts"][f"{x}-{y}"] = c
            g = ga_delta(out["ga"][x], out["ga"][y])
            out["contrasts"][f"{x}-{y}"]["ga"] = g
    full = all(out["levels"].get(l, {}).get("games_per_team") == [600] for l in ("A2", "B", "C", "Ap")) and \
        all(out["ga"].get(l, {}).get("complete") for l in ("A2", "B", "C", "Ap"))
    out["complete"] = full
    cs = out["contrasts"]
    if "Ap-A2" in cs:
        floor = abs(cs["Ap-A2"]["delta_pp"])
        out["rules"]["replicate_abs_pp"] = floor
        out["rules"]["run_floor_exceeds_bar"] = floor > BAR_U
    if "B-A2" in cs and "replicate_abs_pp" in out["rules"]:
        d = cs["B-A2"]; dd = d["delta_pp"]; lo, hi = d["ci95_pp"]; g = d.get("ga")
        u_veto = dd < -BAR_U and hi < 0 and abs(dd) > out["rules"]["replicate_abs_pp"]
        g_veto = bool(g) and g["ci95_pp"][1] < 0 and abs(g["delta_pp"]) > FLOOR_GA
        out["rules"]["N_256"] = "VETO" if (u_veto or g_veto) else "NO LOSS DETECTED (not equivalence)"
    if "C-B" in cs and "replicate_abs_pp" in out["rules"]:
        d = cs["C-B"]; dd = d["delta_pp"]; lo, hi = d["ci95_pp"]; g = d.get("ga")
        ok = lo > -BAR_U and (dd >= 0 or abs(dd) <= out["rules"]["replicate_abs_pp"]) and bool(g) and g["ci95_pp"][0] > -FLOOR_GA
        out["rules"]["E5_on_U_and_GA"] = "PASS (speed from the descriptors decides the rest)" if ok else "FAIL — E10 STAYS"
    if not full and not a.allow_partial:
        out["rules"] = {"NOT READ": "the registered n is not complete; rerun with --allow-partial for PROGRESS only"}
    txt = json.dumps(out, indent=1, default=float)
    if a.json:
        Path(a.json).write_text(txt)
    print(txt)
    return 0


def _load_u_rows(label: str) -> dict:
    from collections import defaultdict

    out: dict = defaultdict(dict)
    trees, refs, opps = set(), set(), set()
    for f in sorted((ROWS / "u_rows" / label).glob("t*/c*.jsonl")):
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


if __name__ == "__main__":
    raise SystemExit(main())
