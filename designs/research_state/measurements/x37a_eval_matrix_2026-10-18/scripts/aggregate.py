"""X37a — pool the queue's units into rows.jsonl + cells.json, and apply PREREG.md's estimators and decision rules.

  PYTHONPATH=<checkout at or after the B-D6 commit>/src python aggregate.py [--stage 1|2|all] [--integrity-only]

A unit COUNTS iff `main.anchors.results.unit_counts` (B-D6) says so: declared cells against the registration's
`declared_digests`, with `verified_digests` recomputed here from the very files the queue sha-checked (the pinned
`63f97a57` summaries predate the digest); symmetric stage-2 cells undeclared, plus the content digests of both sides
equal. AND the CAPTURE check: per game, our side's packed team in the front end's START command is the row's
`our_team` file, and Kakuna's packed team matches exactly one team of ITS declared set (that team is the row's
`their_team`, the cluster of L's bootstrap). `--integrity-only` prints the counting and capture verdicts and NO
win rate (used while a cell is still running).
"""
from __future__ import annotations

import hashlib
import json
import math
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE.parent
sys.path.insert(0, str(HERE))
import xq  # noqa: E402

from main.anchors import team_overlap as TO  # noqa: E402
from main.anchors.results import newcombe, unit_counts, wilson  # noqa: E402
from utils.paths import repo_path  # noqa: E402

REG = json.loads((OUT / "x37a_stage1_registration.json").read_text())
REG2_P = OUT / "x37a_stage2_registration.json"
MEAS = OUT.parent
Z = 1.959964
B = 20000
GAMES = 20
STAGE1 = ("deploy_core", "deploy_near", "deploy_far", "reverse", "kcal")
STAGE2 = ("forum72", "p72")
STAMP = ("opponent", "opponent_version", "opponent_commit", "our_regime", "their_regime", "regime_matched",
         "teamset", "our_team_count", "their_team_count", "battle_format", "model_zip", "model_step",
         "model_loader", "forfeit_turn_limit", "showdown_pin", "server_impl", "server_version", "our_side",
         "our_transport", "mirrored_pairs")
SHOWDOWN = repo_path("deps", "pokemon-showdown", "dist", "sim")


def content_digest(d: Path) -> str:
    fs = sorted(f for f in d.iterdir() if f.suffix in (".txt", ".gen3ou_team"))
    hs = sorted(hashlib.sha256(f.read_text().strip().encode()).hexdigest() for f in fs)
    return hashlib.sha256("\n".join(hs).encode()).hexdigest()


def set_index(d: Path) -> dict:
    """team key (six mon sets, species + moves) -> [file stems]."""
    idx = defaultdict(list)
    for f in sorted(d.iterdir()):
        if f.suffix in (".txt", ".gen3ou_team"):
            idx[TO.parse_team(f.read_text(errors="ignore"), f.stem).key].append(f.stem)
    return idx


def export(packed: list) -> list:
    js = ("const {Teams}=require(process.argv[1]);const ls=JSON.parse(require('fs').readFileSync(0,'utf8'));"
          "process.stdout.write(JSON.stringify(ls.map(p=>Teams.export(Teams.unpack(p)))));")
    p = subprocess.run(["node", "-e", js, str(SHOWDOWN)], input=json.dumps(packed), capture_output=True, text=True,
                       check=True)
    return json.loads(p.stdout)


def score(r) -> float:
    return 1.0 if r["result"] == "win" else 0.0          # ties: denominator only (every read's convention)


def capture_check(cell: str, udir: Path, rows: list) -> tuple:
    """(reasons, their_team per battle_tag) — the ground truth of which team each side played."""
    ours_d, theirs_d = [Path(s[0]) for s in xq.CELLS[cell]["sides"]]
    caps = {}
    for f in sorted((udir / "capture").glob("*.json")):
        c = json.loads(f.read_text())
        start = json.loads(c["commands"][0].split(" ", 1)[1])
        caps[c["tag"]] = start
    reasons, their = [], {}
    if len(caps) != GAMES:
        reasons.append(f"capture holds {len(caps)} battles, not {GAMES}")
    packed, keys = [], []
    for tag, st in caps.items():
        for slot in ("p1", "p2"):
            packed.append(st[slot]["team"])
            keys.append((tag, st[slot]["name"]))
    texts = dict(zip(keys, export(packed))) if packed else {}
    iours, itheirs = set_index(ours_d), set_index(theirs_d)
    if cell == "kcal":
        # side A (reported, our_side = Kakuna on home72) vs side B (Kakuna on H): names differ by the B-D9 build
        for tag, st in caps.items():
            hits = []
            for slot in ("p1", "p2"):
                k = TO.parse_team(texts[(tag, st[slot]["name"])]).key
                hits.append((st[slot]["name"], bool(iours.get(k)), bool(itheirs.get(k))))
            if sorted((a, b) for _, a, b in hits) != [(False, True), (True, False)]:
                reasons.append(f"{tag}: teams not one from each declared set: {hits}")
        return reasons, their
    for r in rows:
        st = caps.get(r.get("battle_tag"))
        if st is None:
            reasons.append(f"no capture for {r.get('battle_tag')}")
            continue
        names = {st["p1"]["name"], st["p2"]["name"]}
        our_name = next((n for n in names if n.lower().startswith("gen3aianchor")), None)
        their_name = next((n for n in names if n != our_name), None)
        k_ours = TO.parse_team(texts[(r["battle_tag"], our_name)]).key
        k_their = TO.parse_team(texts[(r["battle_tag"], their_name)]).key
        if Path(r["our_team"]).stem not in iours.get(k_ours, []):
            reasons.append(f"{r['battle_tag']}: our packed team is not {r['our_team']}")
        m = itheirs.get(k_their, [])
        if not m:
            reasons.append(f"{r['battle_tag']}: Kakuna's team is not in its declared set")
        else:
            their[r["battle_tag"]] = m[0] if len(m) == 1 else "|".join(sorted(m))
    return reasons, their


def collect(cells) -> tuple:
    rows, units, skipped = [], [], []
    reg2 = json.loads(REG2_P.read_text()) if REG2_P.exists() else {}
    for unit, cell, seed in xq.units(list(cells)):
        udir = xq.STATE / "cells" / unit
        sp = udir / "summary.json"
        if not sp.exists():
            continue
        s = json.loads(sp.read_text())
        ours_d, theirs_d = [Path(x[0]) for x in xq.CELLS[cell]["sides"]]
        verified = (content_digest(ours_d), content_digest(theirs_d))
        declared = tuple(REG["declared_digests"][cell]) if cell in REG["declared_digests"] else None
        uc = unit_counts(s, games=GAMES, declared_digests=declared, verified_digests=verified)
        reasons = list(uc.reasons)
        if declared is None and verified[0] != verified[1]:
            reasons.append("symmetric cell: the two sides' content digests differ")
        if declared is None and cell in reg2.get("panels", {}) and verified[0] != reg2["panels"][cell]["content_digest"]:
            reasons.append("panel content digest differs from the stage-2 registration")
        urows = [json.loads(line) for line in open(udir / "games.jsonl")] if (udir / "games.jsonl").exists() else []
        cap_reasons, their = capture_check(cell, udir, urows)
        reasons += cap_reasons
        rec = {"unit": unit, "cell": cell, "seed": seed, "counts": not reasons, "reasons": reasons,
               "status": s.get("status"), "n": s.get("n"), "peer_exit_notes": s.get("peer_exit_notes")}
        units.append(rec)
        if reasons:
            skipped.append(rec)
            continue
        for r in urows:
            rows.append({**r, "read_cell": cell, "unit": unit, "unit_seed": seed,
                         "their_team": their.get(r.get("battle_tag")),
                         "our_team_stem": Path(r["our_team"]).stem if r.get("our_team") else None})
    return rows, units, skipped


def banked(read: str, cell: str) -> list:
    return [json.loads(line) for line in open(MEAS / read / "rows.jsonl") if json.loads(line)["read_cell"] == cell]


def cell_stats(rs: list) -> dict:
    n = len(rs)
    w = sum(r["result"] == "win" for r in rs)
    t = sum(r["result"] == "tie" for r in rs)
    p, lo, hi = wilson(w, n)
    by_half = {h: round(sum(score(r) for r in rs if r["half"] == h) / max(1, sum(r["half"] == h for r in rs)), 4)
               for h in ("ours_challenge", "peer_challenge")}
    turns = sorted(r["turns"] for r in rs if r.get("turns") is not None)
    return {"n": n, "wins": w, "losses": n - w - t, "ties": t, "win_rate": p, "wilson95": [lo, hi],
            "by_half": by_half, "hit_forfeit_limit": sum(bool(r.get("hit_forfeit_limit")) for r in rs),
            "median_turns": turns[len(turns) // 2] if turns else None,
            "distinct_our_teams": len({r.get("our_team") for r in rs}),
            "distinct_their_teams": len({r.get("their_team") for r in rs if r.get("their_team")}),
            "regime_stamps": sorted({json.dumps({k: r.get(k) for k in STAMP}, sort_keys=True) for r in rs})}


def wald(p, n):
    return p * (1 - p) / n


def ci(est, var):
    h = Z * math.sqrt(var)
    return [est - h, est + h]


def near(x, b=0.0):
    return abs(x - b) < 1e-9


def two_stage(groups: dict, rng) -> float:
    """Resample clusters (with replacement), then games within each resampled cluster; the mean score."""
    keys = list(groups)
    pick = rng.integers(0, len(keys), len(keys))
    tot, n = 0.0, 0
    for i in pick:
        g = groups[keys[i]]
        tot += g[rng.integers(0, len(g), len(g))].sum()
        n += len(g)
    return tot / n


def analysis_stage1(rows) -> dict:
    by = defaultdict(list)
    for r in rows:
        by[r["read_cell"]].append(r)
    SPP = banked("k2_x26_kakuna_both_teamsets_2026-10-17", "X26_greedy_h72")
    SHH = banked("heldout72_hpfix_kakuna_2026-10-18", "X26_greedy_hl72hp")
    lw = REG["ladder_weights"]
    w = {s: lw["counts"][s] / lw["n_ladder_teams"] for s in ("CORE", "NEAR", "FAR")}
    D = {s: by[f"deploy_{s.lower()}"] for s in w}
    p = {s: sum(score(r) for r in D[s]) / len(D[s]) for s in w}
    pPP = sum(score(r) for r in SPP) / len(SPP)
    pHH = sum(score(r) for r in SHH) / len(SHH)
    pHP = sum(score(r) for r in by["reverse"]) / len(by["reverse"])
    pK = sum(score(r) for r in by["kcal"]) / len(by["kcal"])
    vPP, vHH, vHP, vK = wald(pPP, len(SPP)), wald(pHH, len(SHH)), wald(pHP, len(by["reverse"])), wald(pK, len(by["kcal"]))
    vD = {s: wald(p[s], len(D[s])) for s in w}

    L = sum(w[s] * p[s] for s in w)
    Lc1 = w["CORE"] * pPP + w["NEAR"] * p["NEAR"] + w["FAR"] * p["FAR"]
    rng = np.random.default_rng(3700)
    cl = {s: {} for s in w}
    for s in w:
        g = defaultdict(list)
        for r in D[s]:
            g[r["their_team"]].append(score(r))
        cl[s] = {k: np.array(v) for k, v in g.items()}
    gpp = defaultdict(list)
    for r in SPP:
        gpp[r["our_team"]].append(score(r))
    gpp = {k: np.array(v) for k, v in gpp.items()}
    bl, blc = np.empty(B), np.empty(B)
    for b in range(B):
        d = {s: two_stage(cl[s], rng) for s in w}
        spp = two_stage(gpp, rng)
        bl[b] = sum(w[s] * d[s] for s in w)
        blc[b] = w["CORE"] * spp + w["NEAR"] * d["NEAR"] + w["FAR"] * d["FAR"]
    Lci = [float(np.percentile(bl, 2.5)), float(np.percentile(bl, 97.5))]
    Lc1ci = [float(np.percentile(blc, 2.5)), float(np.percentile(blc, 97.5))]
    L_wald = ci(L, sum(w[s] ** 2 * vD[s] for s in w))
    Lc1_wald = ci(Lc1, w["CORE"] ** 2 * vPP + w["NEAR"] ** 2 * vD["NEAR"] + w["FAR"] ** 2 * vD["FAR"])

    nH = REG["heldout72_hpfix"]["strata_counts"]
    sPH = sum(nH[s] / 72 * p[s] for s in w)
    vPH = sum((nH[s] / 72) ** 2 * vD[s] for s in w)
    dK = pK - 0.5
    PILOT, FACE = pPP - pHP, pPP - sPH
    PILOT_c, FACE_c = PILOT - dK, FACE + dK
    I = sPH + pHP - pPP - pHH
    ciP = ci(PILOT_c, vPP + vHP + vK)
    ciF = ci(FACE_c, vPP + vPH + vK)
    ciPF = ci(PILOT_c - FACE_c, vHP + vPH + 4 * vK)
    ciFP = [-ciPF[1], -ciPF[0]]
    ciI = ci(I, vPH + vHP + vPP + vHH)
    bounds = [ciI[0], ciI[1], ciP[0], ciPF[0]]
    i_det = ciI[0] > 0 or ciI[1] < 0
    if i_det or any(near(x) for x in bounds) or near(ciP[1], 0.030) or near(ciFP[0]):
        hp = "UNDECIDED" + (" (I detected)" if i_det else " (a bound within 1e-9 of its boundary)")
    elif ciP[0] > 0 and ciPF[0] > 0:
        hp = "SUPPORTED"
    elif ciP[1] < 0.030 or ciFP[0] > 0:
        hp = "KILLED"
    else:
        hp = "UNDECIDED"

    # H-PILOT-OFFMETA: reverse cell, our piloted team's frozen stratum
    stratum_of = {m: s for s, sub in REG["metamon_sub_sets"].items() for m in sub["members"]}
    g = {"CORE": defaultdict(list), "FAR": defaultdict(list), "NEAR": defaultdict(list)}
    for r in by["reverse"]:
        st = r["our_team_stem"]
        g[stratum_of[st]][st].append(score(r))
    g = {s: {k: np.array(v) for k, v in d.items()} for s, d in g.items()}
    rng2 = np.random.default_rng(3701)
    bd = np.array([two_stage(g["FAR"], rng2) - two_stage(g["CORE"], rng2) for _ in range(B)])
    est = (sum(v.sum() for v in g["FAR"].values()) / sum(len(v) for v in g["FAR"].values())
           - sum(v.sum() for v in g["CORE"].values()) / sum(len(v) for v in g["CORE"].values()))
    om_ci = [float(np.percentile(bd, 2.5)), float(np.percentile(bd, 97.5))]
    om = "UNDECIDED" if near(om_ci[1]) else ("SUPPORTED" if om_ci[1] < 0 else "NOT SUPPORTED")
    rev_by = {s: {"games": int(sum(len(v) for v in g[s].values())), "teams": len(g[s]),
                  "score": float(sum(v.sum() for v in g[s].values()) / max(1, sum(len(v) for v in g[s].values())))}
              for s in g}
    return {
        "weights": w, "D": {s: {"n": len(D[s]), "p": p[s], "wilson95": list(wilson(int(round(p[s] * len(D[s]))), len(D[s]))[1:])}
                            for s in w},
        "S_PP": {"n": len(SPP), "p": pPP}, "S_HH": {"n": len(SHH), "p": pHH},
        "S_HP_reverse": {"n": len(by["reverse"]), "p": pHP}, "K_PH": {"n": len(by["kcal"]), "p": pK},
        "L": {"est": L, "boot95": Lci, "wald95": L_wald}, "L_c1": {"est": Lc1, "boot95": Lc1ci, "wald95": Lc1_wald},
        "S_PH_reweighted": {"est": sPH, "wald95": ci(sPH, vPH)},
        "PILOT": PILOT, "FACE": FACE, "Delta_hat": dK, "PILOT_c": {"est": PILOT_c, "ci95": ciP},
        "FACE_c": {"est": FACE_c, "ci95": ciF}, "PILOT_c_minus_FACE_c": {"est": PILOT_c - FACE_c, "ci95": ciPF},
        "I": {"est": I, "ci95": ciI}, "H_PILOT": hp,
        "H_PILOT_OFFMETA": {"FAR_minus_CORE": est, "boot95": om_ci, "verdict": om, "reverse_by_our_stratum": rev_by},
        "bootstrap": {"B": B, "seeds": [3700, 3701], "clusters": "Kakuna's team (from the capture) within stratum; "
                      "S(P, P) by our piloted team"},
    }


def analysis_stage2(rows) -> dict:
    by = defaultdict(list)
    for r in rows:
        by[r["read_cell"]].append(r)
    H = banked("heldout72_hpfix_kakuna_2026-10-18", "X26_greedy_hl72hp")
    P = banked("k2_x26_kakuna_both_teamsets_2026-10-17", "X26_greedy_h72")
    kH, kP = sum(r["result"] == "win" for r in H), sum(r["result"] == "win" for r in P)
    out = {}
    for T in STAGE2:
        rs = by[T]
        k = sum(r["result"] == "win" for r in rs)
        d1 = newcombe(k, len(rs), kH, len(H))
        d2 = newcombe(k, len(rs), kP, len(P))
        det1 = d1[1] > 0 or d1[2] < 0
        det2 = d2[1] > 0 or d2[2] < 0
        if any(near(x) for x in (d1[1], d1[2], d2[1], d2[2])):
            cls = "UNDECIDED"
        elif not det1 and d2[2] < 0:
            cls = "UNSEEN-DRIVEN"
        elif d1[1] > 0 and not det2:
            cls = "POPULATION / KAKUNA-DRIVEN"
        elif d1[1] > 0 and d2[2] < 0:
            cls = "MIXED"
        else:
            cls = "UNDECIDED"
        out[T] = {"n": len(rs), "wins": k, "d1_vs_hpfix": list(d1), "d2_vs_home72": list(d2), "class": cls}
    gov = out["forum72"]["class"]
    out["governing"] = {"source": "forum72", "class": gov,
                        "agree": out["forum72"]["class"] == out["p72"]["class"]}
    return out


def overlap_block(cell) -> dict:
    ours_d, theirs_d = [Path(s[0]) for s in xq.CELLS[cell]["sides"]]
    from main.anchors.session import load_team_texts, pool_team_texts
    refs = {"training_pool": TO.Reference([TO.parse_team(t) for t in pool_team_texts()]),
            "competitive": TO.Reference(TO.parse_items(load_team_texts(Path(
                "/home/goodlad/dev/metamon/cache/teams/competitive/gen3ou")))),
            "hl_05_26": TO.Reference(TO.parse_items(load_team_texts(Path(
                "/home/goodlad/gen3ai_archive/metamon_cache_2026-09-24/teams/hl_05_26/gen3ou"))))}
    out = {"version": TO.OVERLAP_VERSION}
    for side, d in (("ours", ours_d), ("theirs", theirs_d)):
        read = TO.parse_items(load_team_texts(d))
        out[side] = {"dir": str(d), **{k: TO.overlap(read, r) for k, r in refs.items()}}
    return out


def main(argv) -> int:
    stage = argv[argv.index("--stage") + 1] if "--stage" in argv else "all"
    cells = STAGE1 if stage == "1" else (STAGE2 if stage == "2" else STAGE1 + STAGE2)
    rows, units, skipped = collect(cells)
    if "--integrity-only" in argv:
        c = Counter((u["cell"], u["counts"]) for u in units)
        print(json.dumps({f"{k[0]}:{'counts' if k[1] else 'dropped'}": v for k, v in sorted(c.items())}, indent=1))
        for s in skipped:
            print("DROPPED", s["unit"], s["reasons"][:3])
        return 0
    cellstats = {}
    for cell in cells:
        rs = [r for r in rows if r["read_cell"] == cell]
        if rs:
            cellstats[cell] = {**cell_stats(rs), "units": [u["unit"] for u in units if u["cell"] == cell and u["counts"]],
                               "team_overlap": overlap_block(cell)}
    doc = {"cells": cellstats, "units": units, "skipped_units": skipped}
    if stage in ("1", "all") and all(c in cellstats for c in STAGE1):
        doc["stage1"] = analysis_stage1(rows)
    if stage in ("2", "all") and all(c in cellstats for c in STAGE2):
        doc["stage2"] = analysis_stage2(rows)
    suffix = "" if stage == "all" else f"_stage{stage}"
    (OUT / f"cells{suffix}.json").write_text(json.dumps(doc, indent=1, default=float) + "\n")
    with open(OUT / f"rows{suffix}.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(json.dumps({k: v for k, v in doc.items() if k in ("stage1", "stage2")}, indent=1, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
