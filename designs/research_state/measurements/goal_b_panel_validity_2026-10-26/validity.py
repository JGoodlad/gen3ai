"""Goal (b) panel validity: is each unfamiliar panel also unfamiliar to Kakuna? (2026-10-26, CPU, read-only)

    PYTHONPATH=src python designs/research_state/measurements/goal_b_panel_validity_2026-10-26/validity.py \
        --out designs/research_state/measurements/goal_b_panel_validity_2026-10-26/results.json

Reads (never writes) Metamon's archived `hl_05_26/gen3ou` copy, Metamon's `competitive/gen3ou`, the registered panel
directories (home72, away, heldout72, heldout72_hpfix, forum72, p72), forum216b's MANIFEST ids only (never its team
files: it is SEALED), the forum v2 manifest's per-team metadata, and banked Kakuna rows (X51, the X26 @161M twins,
X26c, D1 look 2). Plays nothing.

Blocks:
  A. overlap of each panel with the two Metamon team sets we hold (`gen3_team_overlap_v1`: a mon set = species +
     sorted moves, Hidden Power typed), at radii 6 / >= 4 / >= 3 shared sets (>= 3 = review A's "3-fit" radius), plus
     an HP-agnostic variant (hl_05_26 types 96 % of its Hidden Powers Dark by a fill artifact) and species sextets.
     For heldout72 against hl_05_26 the team's own source file is excluded.
  B. forum216b and forum72 from metadata: the v2 manifest's precomputed hl_05_26 / competitive overlap and post dates.
  C. a descriptive familiarity gradient: our win rate in banked symmetric cells, split by whether KAKUNA's team has a
     >= 3-set neighbour in hl_05_26 (its own source excluded). No rule reads it.
  D. goal (b) recomputed without H (F + P72 only), with the reads' own statistics (`main.x35.d1`).
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from main.anchors import team_overlap as TO
from main.anchors.results import newcombe
from main.anchors.session import load_team_texts
from main.x35 import d1 as D
from utils.paths import repo_path

MEAS = repo_path("designs", "research_state", "measurements")
HL = Path("/home/goodlad/gen3ai_archive/metamon_cache_2026-09-24/teams/hl_05_26/gen3ou")
COMP = Path("/home/goodlad/dev/metamon/cache/teams/competitive/gen3ou")
PANELS = {
    "home72": MEAS / "n0_endofrun_2026-09-27/teams_home72",
    "away": MEAS / "team_strength_2026-10-18/teams_away",
    "heldout72": MEAS / "heldout_teams_kakuna_2026-10-18/teams_heldout72",
    "heldout72_hpfix": MEAS / "heldout72_hpfix_kakuna_2026-10-18/teams_heldout72_hpfix",
    "forum72": MEAS / "x37a_eval_matrix_2026-10-18/teams_forum72",
    "p72": MEAS / "x37a_eval_matrix_2026-10-18/teams_p72",
}
Z = 1.959963985


def hp_agnostic(key):
    return frozenset((sp, tuple(sorted("hiddenpower" if m.startswith("hiddenpower") else m for m in mv)))
                     for sp, mv in key)


class Index:
    """Per-mon-set inverted index over a reference set; `shared(team)` = {ref index: shared set count}."""

    def __init__(self, teams, agnostic=False):
        self.teams = teams
        self.keys = [hp_agnostic(t.key) if agnostic else t.key for t in teams]
        self.agnostic = agnostic
        self.inv = defaultdict(list)
        for i, k in enumerate(self.keys):
            for m in k:
                self.inv[m].append(i)
        self.sp6 = Counter(t.species6 for t in teams)

    def shared(self, team, exclude=None):
        k = hp_agnostic(team.key) if self.agnostic else team.key
        c = Counter()
        for m in k:
            for i in self.inv.get(m, ()):
                if self.teams[i].label != exclude:
                    c[i] += 1
        return c


def q(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * (len(xs) - 1) + 0.5))]


def overlap_block(read, idx, exclude=None):
    best, n4, n3 = [], [], []
    for t in read:
        c = idx.shared(t, exclude=(exclude or {}).get(t.label))
        best.append(max(c.values(), default=0))
        n4.append(sum(1 for v in c.values() if v >= 4))
        n3.append(sum(1 for v in c.values() if v >= 3))
    hist = Counter(best)
    return {
        "n": len(read), "max_shared_hist": {str(k): hist.get(k, 0) for k in range(7)},
        "exact6": hist.get(6, 0), "ge4": sum(b >= 4 for b in best), "ge3": sum(b >= 3 for b in best),
        "neighbours_ge4": {"median": statistics.median(n4), "p90": q(n4, 0.9), "share_ge1": sum(x > 0 for x in n4) / len(n4)},
        "neighbours_ge3": {"median": statistics.median(n3), "p10": q(n3, 0.1), "p90": q(n3, 0.9),
                           "share_ge1": sum(x > 0 for x in n3) / len(n3)},
        "_per_team_ge3": {t.label: x for t, x in zip(read, n3)},
    }


def load(d):
    return TO.parse_items(load_team_texts(d))


def stem(s):
    return s.rsplit(".", 1)[0]


def block_a():
    hl = TO.parse_items(load_team_texts(HL))
    comp = load(COMP)
    out, per_team = {}, {}
    hl_strict, hl_agn = Index(hl), Index(hl, agnostic=True)
    comp_strict, comp_agn = Index(comp), Index(comp, agnostic=True)
    man = json.loads((MEAS / "heldout_teams_kakuna_2026-10-18/heldout72_manifest.json").read_text())["teams"]
    for name, d in PANELS.items():
        read = load(d)
        excl = None
        if name in ("heldout72", "heldout72_hpfix"):
            excl = {lab: man[stem(lab)]["source_file"] for lab in (t.label for t in read)}
        res = {}
        for rname, idx in (("hl_05_26", hl_strict), ("hl_05_26_hp_agnostic", hl_agn),
                           ("competitive", comp_strict), ("competitive_hp_agnostic", comp_agn)):
            b = overlap_block(read, idx, exclude=excl if rname.startswith("hl") else None)
            if rname == "hl_05_26_hp_agnostic":
                per_team[name] = {stem(k): v for k, v in b["_per_team_ge3"].items()}
            b.pop("_per_team_ge3")
            res[rname] = b
        # species sextets (excluding H's own source)
        sp_hl = Counter(t.species6 for t in hl)
        src_sp = {x.label: x.species6 for x in hl}
        res["same_species6_in_hl"] = sum(
            1 for t in read if sp_hl[t.species6] - (1 if excl and src_sp.get(excl[t.label]) == t.species6 else 0) > 0)
        res["same_species6_in_competitive"] = sum(1 for t in read if comp_strict.sp6[t.species6] > 0)
        out[name] = res
    # sanity: each heldout72 team is its own hl source (exact, strict key) -- the exclusion must matter
    h = load(PANELS["heldout72"])
    by_label = {x.label: x for x in hl}
    out["_check_heldout72_is_hl_copy"] = sum(1 for t in h if by_label[man[stem(t.label)]["source_file"]].key == t.key)
    # within-hl reuse (a random 2,000-team sample of hl itself, own file excluded): the ladder's team-reuse base rate
    import random
    smp = random.Random(26).sample(hl, 2000)
    b = overlap_block(smp, hl_agn, exclude={t.label: t.label for t in smp})
    b.pop("_per_team_ge3")
    out["_hl_self_sample2000"] = b
    return out, per_team


def block_b():
    v2 = {}
    for line in open(MEAS / "smogon_forum_teams_v2_2026-10-18/manifest_teams.jsonl"):
        r = json.loads(line)
        if "id" in r:
            v2[r["id"]] = r
    f216 = json.loads((MEAS / "forum216b_x54_2026-10-20/forum216b_manifest.json").read_text())["teams"]
    ids = list(f216)
    rows = [v2[i] for i in ids if i in v2]
    hist = Counter(r["overlap"]["hl_05_26"]["max_shared_mon_sets"] for r in rows)
    chist = Counter(r["overlap"]["metamon_competitive"]["max_shared_mon_sets"] for r in rows)
    years = Counter((r.get("post_date") or "unknown")[:4] for r in rows)
    f72 = json.loads((MEAS / "x37a_eval_matrix_2026-10-18/forum72_manifest.json").read_text())["teams"]
    y72 = Counter((v.get("post_date") or "unknown")[:4] for v in f72.values())
    late = sum(1 for r in rows if (r.get("post_date") or "") >= "2025-05")
    return {"forum216b": {"n_ids": len(ids), "n_joined": len(rows),
                          "hl_05_26_max_shared_hist": {str(k): hist.get(k, 0) for k in range(7)},
                          "competitive_max_shared_hist": {str(k): chist.get(k, 0) for k in range(7)},
                          "post_year": dict(sorted(years.items())), "posted_on_or_after_2025_05": late,
                          "note": "metadata only: the v2 manifest's precomputed overlap and post dates; no team file read"},
            "forum72": {"post_year": dict(sorted(y72.items()))}}


def won(r):
    return 1 if r["result"] == "win" else 0


def block_c(per_team):
    """Our win rate by whether Kakuna's (their) team has a >= 3-set hl_05_26 neighbour (HP-agnostic)."""
    x51 = [json.loads(line) for line in open(MEAS / "x51_plateau_vs_kakuna_2026-10-20/rows.jsonl")]
    cells = {
        ("pstar", "heldout72_hpfix"): [r for r in x51 if r["read_cell"] == "pstar_hpfix"],
        ("pstar", "forum72"): [r for r in x51 if r["read_cell"] == "pstar_forum72"],
        ("pstar", "p72"): [r for r in x51 if r["read_cell"] == "pstar_p72"],
    }
    for lab, (rd, fn, cell) in {
        ("x26_161M", "heldout72_hpfix"): ("heldout72_hpfix_kakuna_2026-10-18", "rows.jsonl", "X26_greedy_hl72hp"),
        ("x26_161M", "forum72"): ("x37a_eval_matrix_2026-10-18", "rows_stage2.jsonl", "forum72"),
        ("x26_161M", "p72"): ("x37a_eval_matrix_2026-10-18", "rows_stage2.jsonl", "p72"),
    }.items():
        cells[lab] = [r for r in (json.loads(x) for x in open(MEAS / rd / fn)) if r["read_cell"] == cell]
    out = {}
    pooled = {"near": [0, 0], "far": [0, 0]}
    for (model, panel), rs in cells.items():
        pt = per_team[panel]
        g = {"near": [0, 0], "far": [0, 0]}
        miss = 0
        for r in rs:
            t = r.get("their_team")
            if t is None or stem(t) not in pt:
                miss += 1
                continue
            k = "near" if pt[stem(t)] > 0 else "far"
            g[k][0] += won(r)
            g[k][1] += 1
        res = {"games": len(rs), "unmapped": miss, "kakuna_team_has_hl_ge3_neighbour": g["near"],
               "kakuna_team_has_none": g["far"]}
        if g["near"][1] and g["far"][1]:
            d, lo, hi = newcombe(g["near"][0], g["near"][1], g["far"][0], g["far"][1])
            res["near_minus_far"] = [d, lo, hi]
        if panel == "heldout72_hpfix":
            for k in pooled:
                pooled[k][0] += g[k][0]
                pooled[k][1] += g[k][1]
        out[f"{model}/{panel}"] = res
    d, lo, hi = newcombe(pooled["near"][0], pooled["near"][1], pooled["far"][0], pooled["far"][1])
    out["H_both_models_pooled"] = {"near": pooled["near"], "far": pooled["far"], "near_minus_far": [d, lo, hi],
                                   "note": "the two models' games share deals: descriptive, not independent"}
    return out


def ss(counts, panels):
    s = D.strat_score(counts, panels)
    return {"per_panel": {k: list(counts[k]) for k in panels}, "mean": s.mean, "n_eff": s.n_eff,
            "wilson95": list(s.ci), "lb_clears_0.50": s.ci[0] > 0.5 + 1e-9}, s


def paired_units(rows, on, off, panels):
    per = {(r["label"], r["panel"], r["unit_seed"]): r["wins"] / r["n"] for r in rows if r.get("counted")}
    by = {k: [per[(on, k, s)] - per[(off, k, s)] for s in range(20) if (on, k, s) in per and (off, k, s) in per]
          for k in panels}
    K = len(panels)
    m = sum(sum(v) / len(v) for v in by.values()) / K
    var = sum((sum((x - sum(v) / len(v)) ** 2 for x in v) / (len(v) - 1)) / len(v) for v in by.values()) / (K * K)
    return {"d": m, "lo": m - Z * math.sqrt(var), "hi": m + Z * math.sqrt(var), "pairs": {k: len(v) for k, v in by.items()}}


def block_d():
    out = {}
    X51 = {"pstar": {"heldout72_hpfix": (199, 400), "forum72": (212, 400), "p72": (173, 400)},
           "x26_161M": {"heldout72_hpfix": (179, 400), "forum72": (188, 400), "p72": (181, 400)}}
    # cross-check against the banked aggregate
    c = json.loads((MEAS / "x51_plateau_vs_kakuna_2026-10-20/cells.json").read_text())["analysis"]["goal_b"]
    assert c["pstar"]["per_panel"]["forum72"] == [212, 400] and c["x26_161M"]["per_panel"]["p72"] == [181, 400]
    x26c = json.loads((MEAS / "x26c_continuation_2026-10-25/read/verdict.json").read_text())["kakuna"]
    XC = {lab: {k: tuple(v) for k, v in x26c[lab]["per_panel"].items()} for lab in ("pstar", "x26c_480m")}
    units = [json.loads(x) for x in open(MEAS / "x26c_continuation_2026-10-25/read/kakuna_units.jsonl")]
    d1 = json.loads((MEAS / "x35_breakers_stage_2026-10-17/d1_read/read_look2_input.json").read_text())["h"]
    for name, panels in (("U_registered_HFP", D.PANELS), ("U_minus_H_FP", ("forum72", "p72")),
                         ("F_only", ("forum72",)), ("H_only", ("heldout72_hpfix",))):
        blk = {}
        for lab, cnt in (("x51_pstar", X51["pstar"]), ("x51_x26_161M", X51["x26_161M"]),
                         ("x26c_pstar_replay", XC["pstar"]), ("x26c_480m", XC["x26c_480m"])):
            blk[lab], _ = ss(cnt, panels)
        _, a = ss(X51["pstar"], panels)
        _, b = ss(X51["x26_161M"], panels)
        blk["x51_pstar_minus_161M_newcombe"] = list(D.newcombe_strat(a, b))
        _, a = ss(XC["x26c_480m"], panels)
        _, b = ss(XC["pstar"], panels)
        blk["x26c_480m_minus_pstar_newcombe"] = list(D.newcombe_strat(a, b))
        blk["x26c_480m_minus_pstar_paired_units"] = paired_units(units, "x26c_480m", "pstar", panels)
        # (P*'s X26c re-play is game-for-game identical to X51's -- deterministic deals -- so it is never pooled)
        # D1 look 2 with the registered rule's statistic on these panels
        on = {s: {k: tuple(v[k]) for k in panels} for s, v in d1["on"].items()}
        off = {s: {k: tuple(v[k]) for k in panels} for s, v in d1["off"].items()}
        h = D.d1_primary(2, on, off, panels=panels)
        blk["d1_look2"] = {"class": h.cls, "delta_pp": h.delta_pp, "ci_pp": list(h.ci_pp),
                           "ci_unwidened_pp": list(h.ci_unwidened_pp),
                           "d1_score": {"mean": h.on.mean, "wilson95": list(h.on.ci)},
                           "c0_score": {"mean": h.off.mean, "wilson95": list(h.off.ci)}}
        out[name] = blk
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    A, per_team = block_a()
    doc = {"schema": "goal_b_panel_validity_v1", "A_overlap": A, "B_forum_metadata": block_b(),
           "C_gradient": block_c(per_team), "D_goal_b_recomputed": block_d()}
    Path(a.out).write_text(json.dumps(doc, indent=1, sort_keys=True, default=float) + "\n")
    print(json.dumps(doc, indent=1, sort_keys=True, default=float)[:200])


if __name__ == "__main__":
    main()
