"""TSR — the read (PREREG.md §3-§4): both pilots' fits, D1-D3, D4, written to results.json + teams.tsv.

  PYTHONPATH=<a checkout at or after the TSR commit>/src python analyze.py [--state DIR] [--integrity-only]

`--integrity-only` prints the unit counts and the pilot-B counting verdicts and NO rating (used while a lane runs).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from main.team_strength import cli
from main.team_strength import register as RG
from main.team_strength import report as RP

HERE = Path(__file__).resolve().parents[1]
MEAS = HERE.parent
STATE = Path("/home/goodlad/dev/gen3ai-reads/tsr_2026-10-18")
K_PH = 0.5725                                   # X37a stage 1 K(P, H), 229-171-0 of 400


def piloted_rows(reg):
    """D3's games: (familiar) the banked S(P, P) rows, (unfamiliar) X37a's reverse cell; team id per row."""
    hid = reg["sets"]["home72"]["team_ids"]
    Hid = reg["sets"]["heldout72_hpfix"]["team_ids"]
    fam = [json.loads(x) for x in open(MEAS / "k2_x26_kakuna_both_teamsets_2026-10-17" / "rows.jsonl")]
    fam = [r for r in fam if r["read_cell"] == "X26_greedy_h72"]
    unf = [json.loads(x) for x in open(MEAS / "x37a_eval_matrix_2026-10-18" / "rows_stage1.jsonl")]
    unf = [r for r in unf if r["read_cell"] == "reverse"]
    assert len(fam) == 400 and len(unf) == 400, (len(fam), len(unf))
    rows = [{"team": hid[Path(r["our_team"]).stem], "unfamiliar": False, "win": r["result"] == "win"} for r in fam]
    rows += [{"team": Hid[r["our_team_stem"]], "unfamiliar": True, "win": r["result"] == "win"} for r in unf]
    return rows


def group_means(rt, groups):
    out = {}
    for g, ids in sorted(groups.items()):
        ids = [t for t in ids if t in rt.ids]
        if ids:
            out[g] = {"n_teams": len(ids), "mean_elo": float(np.mean([rt.elo(t) for t in ids]))}
    return out


def main(argv) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", default=str(STATE))
    ap.add_argument("--integrity-only", action="store_true")
    a = ap.parse_args(argv)
    reg = RG.load(HERE / "tsr_registration.json")
    state = Path(a.state)
    if a.integrity_only:
        xu = cli.XL.x26_units(reg)
        print("x26 units:", sum((state / "x26" / "units" / f"{u.name}.json").exists() for u in xu), "/", len(xu))
        from utils.paths import repo_path
        kk = cli.load_kakuna(reg, state, repo_path("deps", "pokemon-showdown", "dist", "sim"))
        bad = [v for v in kk["verdicts"] if not v["counts"]]
        print("kakuna units:", len(kk["verdicts"]), "not counting:", json.dumps(bad, indent=1)[:2000],
              "self games:", kk["self_games"])
        return 0
    out, ratings = cli.compute(reg, state)                     # pilot B clustered by MATCHUP (the README's FINDING)
    # the PREREG-as-written clustering (each pilot-B game its own cluster), for pilot B's intervals only
    asw, _r = cli.compute(reg, state, kakuna_cluster="game")
    if "kakuna" in asw["pilots"]:
        out["kakuna_prereg_game_clusters"] = {"sets": asw["pilots"]["kakuna"]["sets"],
                                              "comparisons": asw["pilots"]["kakuna"]["comparisons"],
                                              "agreement": asw.get("agreement")}
    mg = reg["fit"]["matched_gap"]
    rows = piloted_rows(reg)
    d3 = {}
    if "kakuna" in ratings:
        sub = set(reg["pilots"]["kakuna"]["subsets"]["home72"]) | set(reg["pilots"]["kakuna"]["subsets"]["heldout72_hpfix"])
        rb = [r for r in rows if r["team"] in sub]
        d3["primary_kakuna"] = RP.matched_gap(rb, ratings["kakuna"], B=int(mg["B"]), seed=int(mg["seed"]))
        kb = out["pilots"]["kakuna"]["sets"]
        d3["strength_term_check"] = {
            "kakuna_home72_minus_H_elo": kb["home72"]["mean_elo"] - kb["heldout72_hpfix"]["mean_elo"],
            "x37a_K_PH": K_PH, "x37a_K_PH_elo": 400 * math.log10(K_PH / (1 - K_PH)),
            "comparison": out["pilots"]["kakuna"]["comparisons"].get("heldout72_hpfix-home72")}
    if "x26" in ratings:
        d3["secondary_x26_circular"] = RP.matched_gap(rows, ratings["x26"], B=int(mg["B"]), seed=int(mg["seed"]))
    out["d3"] = d3
    # D4
    d4 = {}
    if "x26" in ratings and "kakuna" in ratings:
        d4["x26_minus_kakuna_by_set"] = {
            s: out["pilots"]["x26"]["sets"][s]["mean_elo"] - out["pilots"]["kakuna"]["sets"][s]["mean_elo"]
            for s in out["pilots"]["kakuna"]["sets"]}
        # the same on the SUBSET teams only (like for like)
        d4["x26_minus_kakuna_on_subsets"] = {
            s: float(np.mean([ratings["x26"].elo(t) for t in ids]) - np.mean([ratings["kakuna"].elo(t) for t in ids]))
            for s, ids in reg["pilots"]["kakuna"]["subsets"].items()}
    fm = json.loads((MEAS / "x37a_eval_matrix_2026-10-18" / "forum72_manifest.json").read_text())["teams"]
    fid = reg["sets"]["forum72"]["team_ids"]
    by_tier, by_cat = defaultdict(list), defaultdict(list)
    for stem, m in fm.items():
        by_tier[m["tier"]].append(fid[stem])
        by_cat[m["category"]].append(fid[stem])
    x37 = json.loads((MEAS / "x37a_eval_matrix_2026-10-18" / "x37a_stage1_registration.json").read_text())
    Hid = reg["sets"]["heldout72_hpfix"]["team_ids"]
    strata = {s: [Hid[m] for m in v["members"]] for s, v in x37["metamon_sub_sets"].items()}
    for pilot, rt in ratings.items():
        d4[pilot] = {"forum72_by_tier": group_means(rt, by_tier), "forum72_by_category": group_means(rt, by_cat),
                     "heldout_by_stratum": group_means(rt, strata)}
    out["d4"] = d4
    (HERE / "results.json").write_text(json.dumps(out, indent=1, sort_keys=True, default=float) + "\n")
    # the per-team table
    stems = defaultdict(list)
    for s, blk in reg["sets"].items():
        for stem, tid in blk["team_ids"].items():
            stems[tid].append(f"{s}/{stem}")
    refs = {m["team_id"] for m in reg["reference"]["members"]}
    lines = ["team_id\tfiles\treference\tx26_elo\tx26_lo\tx26_hi\tx26_games\tx26_score\tkakuna_elo\tkakuna_lo\t"
             "kakuna_hi\tkakuna_games\tkakuna_score"]
    for tid in sorted(stems, key=lambda t: -(ratings["x26"].elo(t) if "x26" in ratings and t in ratings["x26"].ids
                                              else 0)):
        cells = [tid, ",".join(stems[tid]), "ref" if tid in refs else ""]
        for p in ("x26", "kakuna"):
            rt = ratings.get(p)
            if rt is not None and tid in rt.ids:
                lo, hi = rt.ci(tid)
                cells += [f"{rt.elo(tid):.1f}", f"{lo:.1f}", f"{hi:.1f}", str(rt.games[tid]), f"{rt.score[tid]:.3f}"]
            else:
                cells += [""] * 5
        lines.append("\t".join(cells))
    (HERE / "teams.tsv").write_text("\n".join(lines) + "\n")
    # summary
    for p, blk in out["pilots"].items():
        print(f"== {p}: {blk['n_games']} games")
        for s, m in blk["sets"].items():
            print(f"  {s:16s} n={m['n_teams']:3d} mean {m['mean_elo']:+7.1f} [{m['ci'][0]:+.1f}, {m['ci'][1]:+.1f}] "
                  f"sd {m['sd_elo']:.1f}")
        for c, v in blk["comparisons"].items():
            print(f"  {c:28s} {v['est']:+7.1f} [{v['ci'][0]:+.1f}, {v['ci'][1]:+.1f}] {v['class']}"
                  f"{' (detected weaker)' if v['detected_weaker'] else ''}")
    if "agreement" in out:
        print("agreement", json.dumps(out["agreement"]))
    print("d3", json.dumps(d3, indent=1))
    print("d4", json.dumps({k: v for k, v in d4.items()}, indent=1)[:3000])
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
