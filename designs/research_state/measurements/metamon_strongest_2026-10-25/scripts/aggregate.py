"""metamon_strongest: pool the queue's units into rows.jsonl + cells.json and read PREREG.md §5.

  PYTHONPATH=<the read's tree>/src python aggregate.py [--integrity-only]

COUNTING (PREREG §5 = X51's rule): `main.anchors.results.unit_counts` (undeclared symmetric), status OK, n == 20,
`regime_verified_decisions` (for the ensemble: the router's no-RNG rate exactly 1.0 and `sample == [False]` on both
halves, checked again here from the rows), `peer_clean`, the two sides' CONTENT digests equal (recomputed from the
files the queue sha-checked) AND the capture check (our packed team is the row's team; the peer's is one team of
the same set). `--integrity-only` prints the counting verdicts and NO win rate.

THE READ (PREREG §5; ties are non-wins): per panel Wilson 95 %; E-U = the equal-weight mean over H / F / P72
(`main.x35.d1.strat_score`, Kish n_eff) vs 0.50; Δ = E − X51's P* vs Kakuna (Newcombe per panel, `newcombe_strat`
on U), classes ENSEMBLE STRONGER (UB < 0) / KAKUNA STRONGER (LB > 0) / NOT DETECTED; the paired descriptor over
twins (unit seed, half, index; our team equal per twin). Δ-home reads against `kakctl_home72` instead when the
identity unit did not replay X51 (precondition 4). RULE 8: a bound within 1e-9 of its boundary is UNDECIDED.
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
_spec = importlib.util.spec_from_file_location("mmq", HERE / "mmq.py")
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
PANEL = {"home72": "pm_home72", "hpfix": "pm_hpfix", "forum72": "pm_forum72", "p72": "pm_p72"}
UNFAMILIAR = ("hpfix", "forum72", "p72")
X51_ROWS = q.X51_MEAS / "rows.jsonl"


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
            reasons.append(f"{r['battle_tag']}: the peer's team is not in its set")
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


def x51(s: str) -> list:
    """X51's counted P* vs Kakuna rows of panel ``s`` (the comparator every game twins)."""
    return [r for r in (json.loads(line) for line in open(X51_ROWS)) if r["read_cell"] == f"pstar_{s}"]


def router_ok(rows: list) -> list:
    bad = [r["battle_tag"] for r in rows if r.get("opponent") == "metamon:PastaMittens"
           and not (r.get("their_router_no_rng_rate") == 1.0 and r.get("their_sample_kwargs") == [False])]
    return [f"{len(bad)} rows without a 1.0 router no-RNG rate and sample == [False]"] if bad else []


def collect() -> tuple:
    rows, units, skipped = [], [], []
    for unit, cell, seed in q.units():
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
        reasons += router_ok(urows)
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


def classes(lo: float, hi: float) -> str:
    """Δ = E − Kakuna from P*'s side: a NEGATIVE Δ means the ensemble beats P* more often."""
    if abs(lo) < EPS or abs(hi) < EPS:
        return "UNDECIDED (rule 8)"
    return "ENSEMBLE STRONGER" if hi < 0 else ("KAKUNA STRONGER" if lo > 0 else "NOT DETECTED")


def analysis(by: dict, ident) -> dict:
    out: dict = {"identity_unit": ident}
    K = {s: x51(s) for s in PANEL}
    home_cmp = K["home72"] if (ident or {}).get("identical") else by.get("kakctl_home72", [])
    out["home_comparator"] = "x51_pstar_home72" if (ident or {}).get("identical") else "kakctl_home72"
    out["kakuna_x51"] = {s: {k: v for k, v in cell_stats(K[s]).items() if k != "regime_stamps"} for s in PANEL}
    out["per_panel"] = {}
    for s, c in PANEL.items():
        if not by.get(c):
            continue
        st = cell_stats(by[c])
        cmp_rows = home_cmp if s == "home72" else K[s]
        blk = diff_block(by[c], cmp_rows) if cmp_rows else None
        out["per_panel"][s] = {"win_rate": st["win_rate"], "wilson95": st["wilson95"], "n": st["n"],
                               "vs_0.50": verdict_vs(*st["wilson95"], 0.5),
                               "delta_vs_kakuna": blk, "delta_class": classes(*blk["newcombe95"][1:]) if blk else None}
    if all(by.get(PANEL[s]) for s in UNFAMILIAR):
        def counts(src):
            return {s: (sum(won(r) for r in src[s]), len(src[s])) for s in UNFAMILIAR}
        es = strat_score(counts({s: by[PANEL[s]] for s in UNFAMILIAR}), UNFAMILIAR)
        ks = strat_score(counts(K), UNFAMILIAR)
        d, lo, hi = newcombe_strat(es, ks)
        out["E_U"] = {"ensemble": es.as_dict(), "kakuna_x51": ks.as_dict(),
                      "vs_0.50": verdict_vs(es.ci[0], es.ci[1], 0.5),
                      "delta_U": {"newcombe_strat95": [d, lo, hi], "class": classes(lo, hi)}}
    return out


def main(argv) -> int:
    rows, units, skipped = collect()
    if "--integrity-only" in argv:
        c = Counter((u["cell"], u["counts"]) for u in units)
        print(json.dumps({f"{k[0]}:{'counts' if k[1] else 'dropped'}": v for k, v in sorted(c.items())}, indent=1))
        for s in skipped:
            print("DROPPED", s["unit"], s["reasons"][:3])
        return 0
    by = defaultdict(list)
    for r in rows:
        by[r["read_cell"]].append(r)
    cellstats = {c: {**cell_stats(by[c]), "units": [u["unit"] for u in units if u["cell"] == c and u["counts"]]}
                 for c in q.CELLS if by.get(c)}
    doc = {"cells": cellstats, "analysis": analysis(by, q.identity()), "units": units, "skipped_units": skipped}
    (OUT / "cells.json").write_text(json.dumps(doc, indent=1, default=float) + "\n")
    with open(OUT / "rows.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(json.dumps(doc["analysis"], indent=1, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
