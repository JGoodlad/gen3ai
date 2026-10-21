"""X51 — pool the queue's units into rows.jsonl + cells.json; every comparison the README quotes.

  PYTHONPATH=<a checkout at or after b9c75884>/src python aggregate.py [--integrity-only]

COUNTING (X37a's rule, `main.anchors.results.unit_counts`, undeclared symmetric cells): status OK, n == 20,
`regime_verified_decisions`, `peer_clean`, no team-source asymmetry, the two sides' CONTENT digests equal (recomputed
here from the files the queue sha-checked; the pinned `63f97a57` summaries carry none) AND the CAPTURE check: per
game, our packed team in the front end's START command is the row's `our_team`, and Kakuna's packed team is exactly
one team of the same set (the row's `their_team`). For the pair cell both packed teams must be home72 teams.
`--integrity-only` prints the counting verdicts and NO win rate.

THE COMPARISONS (every one a descriptor or a stated rule; ties are non-wins, every read's convention):
  * per cell: Wilson 95 %;
  * P* − X26 @161M per set: Newcombe (the headline) + the PAIRED descriptor (twins = unit seed, half, index; our team
    checked equal per twin; normal and seed-bootstrap intervals, Random(0), 20,000 resamples) — the X26 rows are
    the banked cells of `k2_x26_kakuna_both_teamsets_2026-10-17` (home72), `heldout72_hpfix_kakuna_2026-10-18`
    (heldout72_hpfix) and `x37a_eval_matrix_2026-10-18` stage 2 (forum72, p72);
  * goal (a): P* home72 Wilson LB > 0.50; goal (b): the equal-weight mean over heldout72_hpfix / forum72 / p72 with
    its Kish effective n and Wilson interval (`main.x35.d1.strat_score`, D1's H statistic) LB > 0.50; the pooled
    P* − X26 by `main.x35.d1.newcombe_strat`;
  * the pair cell: Kakuna-28's win rate against Kakuna-34; the design's rule: ckpt 28 is the stronger iff LB > 0.50
    (and 34 iff UB < 0.50);
  * T* − P* per set (Newcombe + paired), when the toy cells ran.
  RULE 8: a bound within 1e-9 of its decision boundary decides nothing (UNDECIDED).
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import random
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE.parent
MEAS = OUT.parent
_spec = importlib.util.spec_from_file_location("x51q", HERE / "x51q.py")
q = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(q)

from main.anchors import team_overlap as TO  # noqa: E402
from main.anchors.results import newcombe, unit_counts, wilson  # noqa: E402
from main.x35.d1 import newcombe_strat, strat_score  # noqa: E402
from utils.paths import repo_path  # noqa: E402

GAMES = 20
EPS = 1e-9
STAMP = ("opponent", "opponent_version", "opponent_commit", "our_regime", "their_regime", "regime_matched",
         "teamset", "our_team_count", "their_team_count", "battle_format", "model_zip", "model_step",
         "model_loader", "forfeit_turn_limit", "showdown_pin", "server_impl", "server_version", "our_side",
         "our_transport", "mirrored_pairs")
SHOWDOWN = repo_path("deps", "pokemon-showdown", "dist", "sim")
SET_OF = {"home72": "home72", "hpfix": "heldout72_hpfix", "forum72": "forum72", "p72": "p72"}
UNFAMILIAR = ("hpfix", "forum72", "p72")
X26_BANKED = {   # set -> (read dir, rows file, read_cell)
    "home72": ("k2_x26_kakuna_both_teamsets_2026-10-17", "rows.jsonl", "X26_greedy_h72"),
    "hpfix": ("heldout72_hpfix_kakuna_2026-10-18", "rows.jsonl", "X26_greedy_hl72hp"),
    "forum72": ("x37a_eval_matrix_2026-10-18", "rows_stage2.jsonl", "forum72"),
    "p72": ("x37a_eval_matrix_2026-10-18", "rows_stage2.jsonl", "p72"),
}


def content_digest(d: Path) -> str:
    fs = sorted(f for f in d.iterdir() if f.suffix in (".txt", ".gen3ou_team"))
    hs = sorted(hashlib.sha256(f.read_text().strip().encode()).hexdigest() for f in fs)
    return hashlib.sha256("\n".join(hs).encode()).hexdigest()


def set_index(d: Path) -> dict:
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


def won(r) -> int:
    return 1 if r["result"] == "win" else 0


def capture_check(cell: str, udir: Path, rows: list) -> tuple:
    sides = [Path(s[0]) for s in q.CELLS[cell]["sides"]]
    caps = {}
    for f in sorted((udir / "capture").glob("*.json")):
        c = json.loads(f.read_text())
        caps[c["tag"]] = json.loads(c["commands"][0].split(" ", 1)[1])
    reasons, their = [], {}
    if len(caps) != GAMES:
        reasons.append(f"capture holds {len(caps)} battles, not {GAMES}")
    packed, keys = [], []
    for tag, st in caps.items():
        for slot in ("p1", "p2"):
            packed.append(st[slot]["team"])
            keys.append((tag, st[slot]["name"]))
    texts = dict(zip(keys, export(packed))) if packed else {}
    if q.CELLS[cell]["kind"] == "pair":
        idx = set_index(sides[0])
        for tag, st in caps.items():
            for slot in ("p1", "p2"):
                if not idx.get(TO.parse_team(texts[(tag, st[slot]["name"])]).key):
                    reasons.append(f"{tag}: {st[slot]['name']}'s team is not a home72 team")
        return reasons, their
    iours, itheirs = set_index(sides[0]), set_index(sides[1])
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
            reasons.append(f"{r['battle_tag']}: Kakuna's team is not in its set")
        else:
            their[r["battle_tag"]] = m[0] if len(m) == 1 else "|".join(sorted(m))
    return reasons, their


def collect(cells) -> tuple:
    rows, units, skipped = [], [], []
    for unit, cell, seed in q.units(list(cells)):
        udir = q.STATE / "cells" / unit
        sp = udir / "summary.json"
        if not sp.exists():
            continue
        s = json.loads(sp.read_text())
        sides = [Path(x[0]) for x in q.CELLS[cell]["sides"]]
        verified = (content_digest(sides[0]), content_digest(sides[-1]))
        uc = unit_counts(s, games=GAMES, declared_digests=None, verified_digests=verified)
        reasons = list(uc.reasons)
        if verified[0] != verified[1]:
            reasons.append("symmetric cell: the two sides' content digests differ")
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


def banked(s: str) -> list:
    read, fname, cell = X26_BANKED[s]
    return [r for r in (json.loads(line) for line in open(MEAS / read / fname)) if r["read_cell"] == cell]


def cell_stats(rs: list) -> dict:
    n = len(rs)
    w = sum(won(r) for r in rs)
    t = sum(r["result"] == "tie" for r in rs)
    p, lo, hi = wilson(w, n)
    by_half = {h: round(sum(won(r) for r in rs if r["half"] == h) / max(1, sum(r["half"] == h for r in rs)), 4)
               for h in ("ours_challenge", "peer_challenge")}
    turns = sorted(r["turns"] for r in rs if r.get("turns") is not None)
    return {"n": n, "wins": w, "losses": n - w - t, "ties": t, "win_rate": p, "wilson95": [lo, hi],
            "by_half": by_half, "hit_forfeit_limit": sum(bool(r.get("hit_forfeit_limit")) for r in rs),
            "median_turns": turns[len(turns) // 2] if turns else None,
            "distinct_our_teams": len({r.get("our_team") for r in rs}),
            "distinct_their_teams": len({r.get("their_team") for r in rs if r.get("their_team")}),
            "regime_stamps": dict(Counter(json.dumps({k: r.get(k) for k in STAMP}, sort_keys=True) for r in rs))}


def verdict_vs(lo: float, hi: float, bar: float) -> str:
    if abs(lo - bar) < EPS or abs(hi - bar) < EPS:
        return "UNDECIDED (rule 8)"
    return "BETTER" if lo > bar else ("WORSE" if hi < bar else "NOT DETECTED")


def paired(a: list, b: list) -> dict:
    """a − b over twins (unit seed, half, index); our team must be equal in each twin."""
    def key(r):
        return (r["unit_seed"], r["half"], r["index"])
    A = {key(r): (won(r), r["our_team"]) for r in a}
    Bm = {key(r): (won(r), r["our_team"]) for r in b}
    ks = sorted(set(A) & set(Bm))
    if not ks:
        return {"pairs": 0}
    bad = [k for k in ks if A[k][1] != Bm[k][1]]
    if bad:
        return {"pairs": len(ks), "error": f"{len(bad)} twins disagree on our team", "first": list(bad[0])}
    d = [A[k][0] - Bm[k][0] for k in ks]
    n = len(d)
    m = sum(d) / n
    se = math.sqrt(sum((v - m) ** 2 for v in d) / (n - 1)) / math.sqrt(n)
    seeds = sorted({k[0] for k in ks})
    by = {s: [A[k][0] - Bm[k][0] for k in ks if k[0] == s] for s in seeds}
    rng = random.Random(0)
    bs = []
    for _ in range(20000):
        smp = [v for s in (rng.choice(seeds) for _ in seeds) for v in by[s]]
        bs.append(sum(smp) / len(smp))
    bs.sort()
    return {"pairs": n, "seeds": seeds, "diff": m, "normal95": [m - 1.96 * se, m + 1.96 * se],
            "seed_bootstrap95": [bs[500], bs[19499]], "a_only_wins": d.count(1), "b_only_wins": d.count(-1)}


def diff_block(a: list, b: list) -> dict:
    ka, kb = sum(won(r) for r in a), sum(won(r) for r in b)
    d, lo, hi = newcombe(ka, len(a), kb, len(b))
    return {"newcombe95": [d, lo, hi], "verdict": verdict_vs(lo, hi, 0.0), "paired": paired(a, b)}


def overlap_block(cell) -> dict:
    from main.anchors.session import load_team_texts, pool_team_texts
    d = Path(q.CELLS[cell]["sides"][0][0])
    refs = {"training_pool": TO.Reference([TO.parse_team(t) for t in pool_team_texts()]),
            "competitive": TO.Reference(TO.parse_items(load_team_texts(Path(
                "/home/goodlad/dev/metamon/cache/teams/competitive/gen3ou"))))}
    read = TO.parse_items(load_team_texts(d))
    return {"version": TO.OVERLAP_VERSION, "dir": str(d), **{k: TO.overlap(read, r) for k, r in refs.items()}}


def analysis(by: dict) -> dict:
    out: dict = {}
    X = {s: banked(s) for s in SET_OF}
    out["x26_161M"] = {s: {k: v for k, v in cell_stats(X[s]).items() if k != "regime_stamps"} for s in SET_OF}
    out["pstar_minus_x26"] = {s: diff_block(by[f"pstar_{s}"], X[s]) for s in SET_OF if by.get(f"pstar_{s}")}
    if by.get("pstar_home72"):
        h = cell_stats(by["pstar_home72"])
        out["goal_a"] = {"cell": "pstar_home72", "win_rate": h["win_rate"], "wilson95": h["wilson95"],
                         "verdict_vs_0.50": verdict_vs(*h["wilson95"], 0.5),
                         "met": h["wilson95"][0] > 0.5 + EPS}
    if all(by.get(f"pstar_{s}") for s in UNFAMILIAR):
        def counts(src):
            return {s: (sum(won(r) for r in src[s]), len(src[s])) for s in UNFAMILIAR}
        ps = strat_score(counts({s: by[f"pstar_{s}"] for s in UNFAMILIAR}), UNFAMILIAR)
        xs = strat_score(counts(X), UNFAMILIAR)
        d, lo, hi = newcombe_strat(ps, xs)
        out["goal_b"] = {"pstar": ps.as_dict(), "x26_161M": xs.as_dict(),
                         "verdict_vs_0.50": verdict_vs(ps.ci[0], ps.ci[1], 0.5), "met": ps.ci[0] > 0.5 + EPS,
                         "gap_to_0.53_pp": 100 * (0.53 - ps.mean),
                         "pstar_minus_x26": {"newcombe_strat95": [d, lo, hi], "verdict": verdict_vs(lo, hi, 0.0)}}
        # the familiarity premium at the plateau, raw: home72 − the unfamiliar mean
        if by.get("pstar_home72"):
            hm = sum(won(r) for r in by["pstar_home72"]) / len(by["pstar_home72"])
            out["goal_b"]["premium_home72_minus_U_pp"] = 100 * (hm - ps.mean)
    if by.get("k28v34_home72"):
        k = cell_stats(by["k28v34_home72"])
        lo, hi = k["wilson95"]
        rule = ("UNDECIDED (rule 8)" if min(abs(lo - 0.5), abs(hi - 0.5)) < EPS else
                "CKPT 28 STRONGER" if lo > 0.5 else "CKPT 34 STRONGER" if hi < 0.5 else "NOT DETECTED")
        out["kakuna28_vs_34"] = {"ckpt28_win_rate": k["win_rate"], "wilson95": k["wilson95"], "rule": rule}
    out["toy_minus_pstar"] = {s: diff_block(by[f"toy_{s}"], by[f"pstar_{s}"]) for s in ("home72", "forum72")
                              if by.get(f"toy_{s}") and by.get(f"pstar_{s}")}
    return out


def main(argv) -> int:
    cells = list(q.CELLS)
    rows, units, skipped = collect(cells)
    if "--integrity-only" in argv:
        c = Counter((u["cell"], u["counts"]) for u in units)
        print(json.dumps({f"{k[0]}:{'counts' if k[1] else 'dropped'}": v for k, v in sorted(c.items())}, indent=1))
        for s in skipped:
            print("DROPPED", s["unit"], s["reasons"][:3])
        return 0
    by = defaultdict(list)
    for r in rows:
        by[r["read_cell"]].append(r)
    cellstats = {}
    for cell in cells:
        if by.get(cell):
            cellstats[cell] = {**cell_stats(by[cell]),
                               "units": [u["unit"] for u in units if u["cell"] == cell and u["counts"]]}
            if q.CELLS[cell]["kind"] == "model" and cell.startswith("pstar_"):
                cellstats[cell]["team_overlap"] = overlap_block(cell)
    doc = {"cells": cellstats, "analysis": analysis(by), "units": units, "skipped_units": skipped}
    (OUT / "cells.json").write_text(json.dumps(doc, indent=1, default=float) + "\n")
    with open(OUT / "rows.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(json.dumps(doc["analysis"], indent=1, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
