"""FR-1 / FR-3: pool the queue's units into rows.jsonl + cells.json and read PREREG.md §4.

  PYTHONPATH=<the play tree>/src python aggregate.py [--integrity-only] [--opponent SmallRL|LargeRL|LargeIL ...]

COUNTING (PREREG §3 = X51's rule): `main.anchors.results.unit_counts` (undeclared symmetric), status OK, n == 20,
`regime_verified_decisions` (argmax match 1.0 and `sample == [False]` on every row, checked again here), the two
sides' CONTENT digests equal (recomputed from the files the queue sha-checked) AND the capture check (our packed team
is the row's team; the peer's is one team of the same set). `--integrity-only` prints the counting verdicts and NO
win rate.

THE READ (PREREG §4; ties are non-wins): per panel Wilson 95 %; W-U = the equal-weight mean over H / F / P72
(`main.x35.d1.strat_score`, Kish n_eff) and the rule HUMAN-ONLY-BEATS-OURS (UB < 0.50) / OURS-BEATS (LB > 0.50) /
UNDECIDED; Δ-K = W − X51's P* vs Kakuna on the same twins (Newcombe per panel, `newcombe_strat` on U); the paired
descriptor over twins (unit seed, half, index; our team equal per twin); FR-3's LargeIL − LargeRL. RULE 8: a bound
within 1e-9 of its boundary is UNDECIDED.
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
_spec = importlib.util.spec_from_file_location("hrrq", HERE / "hrrq.py")
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
PANELS = ("home72", "hpfix", "forum72", "p72")
UNFAMILIAR = ("hpfix", "forum72", "p72")
PREFIX = {"SmallRL": "sr", "LargeRL": "lrl", "LargeIL": "lil"}
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
    sides = [Path(s[0]) for s in q.sides(q.CELLS[cell]["panel"])]
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


def regime_ok(rows: list) -> list:
    bad = [r["battle_tag"] for r in rows
           if not (r.get("their_argmax_match_rate") == 1.0 and r.get("their_sample_kwargs") == [False]
                   and r.get("our_regime") == "greedy" and r.get("their_regime") == "greedy")]
    return [f"{len(bad)} rows without argmax match 1.0, sample == [False] and greedy / greedy"] if bad else []


def collect(opps) -> tuple:
    rows, units, skipped = [], [], []
    seen = set()
    for unit, cell, seed in q.units("A"):
        if unit in seen or q.CELLS[cell]["opponent"].split(":", 1)[1] not in opps:
            continue
        seen.add(unit)
        udir = q.STATE / "cells" / unit
        sp = udir / "summary.json"
        if not sp.exists():
            continue
        s = json.loads(sp.read_text())
        sides = [Path(x[0]) for x in q.sides(q.CELLS[cell]["panel"])]
        verified = (content_digest(sides[0]), content_digest(sides[-1]))
        uc = unit_counts(s, games=GAMES, declared_digests=None, verified_digests=verified)
        reasons = list(uc.reasons)
        if verified[0] != verified[1]:
            reasons.append("symmetric cell: the two sides' content digests differ")
        urows = [json.loads(line) for line in open(udir / "games.jsonl")] if (udir / "games.jsonl").exists() else []
        reasons += regime_ok(urows)
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


def near(x: float, bar: float) -> bool:
    return abs(x - bar) < EPS


def verdict_vs(lo: float, hi: float, bar: float) -> str:
    if near(lo, bar) or near(hi, bar):
        return "UNDECIDED (rule 8)"
    return "BETTER" if lo > bar else ("WORSE" if hi < bar else "NOT DETECTED")


def fr_rule(lo: float, hi: float) -> str:
    """PREREG §4: the per-opponent verdict on W-U."""
    if near(lo, 0.5) or near(hi, 0.5):
        return "UNDECIDED"
    return "HUMAN-ONLY-BEATS-OURS" if hi < 0.5 else ("OURS-BEATS" if lo > 0.5 else "UNDECIDED")


def dk_class(agent: str, lo: float, hi: float) -> str:
    if near(lo, 0.0) or near(hi, 0.0):
        return "UNDECIDED (rule 8)"
    return (f"{agent} WEAKER THAN KAKUNA" if lo > 0 else
            (f"{agent} STRONGER THAN KAKUNA" if hi < 0 else "NOT DETECTED"))


def paired(a: list, b: list) -> dict:
    """a − b over twins (unit seed, half, index); our team must be equal in each twin."""
    def key(r):
        return (r["unit_seed"], r["half"], r["index"])
    A = {key(r): (won(r), r["our_team"]) for r in a}
    Bm = {key(r): (won(r), r["our_team"]) for r in b}
    ks = sorted(set(A) & set(Bm))
    if len(ks) < 2:
        return {"pairs": len(ks)}
    bad = [k for k in ks if Path(A[k][1]).name != Path(Bm[k][1]).name]
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


def x51(s: str) -> list:
    """X51's counted P* vs Kakuna rows of panel ``s`` (the comparator every game twins)."""
    return [r for r in (json.loads(line) for line in open(X51_ROWS)) if r["read_cell"] == f"pstar_{s}"]


def u_score(src: dict):
    return strat_score({s: (sum(won(r) for r in src[s]), len(src[s])) for s in UNFAMILIAR}, UNFAMILIAR)


def analysis_for(agent: str, by: dict, K: dict) -> dict:
    pre = PREFIX[agent]
    mine = {s: by.get(f"{pre}_{s}", []) for s in PANELS}
    out: dict = {"per_panel": {}}
    for s in PANELS:
        if not mine[s]:
            continue
        st = cell_stats(mine[s])
        ka, kb = st["wins"], sum(won(r) for r in K[s])
        d, lo, hi = newcombe(ka, st["n"], kb, len(K[s]))
        out["per_panel"][s] = {"win_rate": st["win_rate"], "wilson95": st["wilson95"], "n": st["n"],
                               "vs_0.50": verdict_vs(*st["wilson95"], 0.5),
                               "delta_vs_kakuna": {"newcombe95": [d, lo, hi], "class": dk_class(agent, lo, hi),
                                                   "paired": paired(mine[s], K[s])}}
    if all(len(mine[s]) for s in UNFAMILIAR):
        ws, ks = u_score(mine), u_score(K)
        d, lo, hi = newcombe_strat(ws, ks)
        out["W_U"] = {"score": ws.as_dict(), "verdict": fr_rule(*ws.ci), "complete": all(
            len(mine[s]) == 400 for s in UNFAMILIAR),
            "delta_vs_kakuna": {"newcombe_strat95": [d, lo, hi], "class": dk_class(agent, lo, hi),
                                "kakuna_x51": ks.as_dict()}}
    return out


def main(argv) -> int:
    opps = [a for a in PREFIX if a in argv] or list(PREFIX)
    rows, units, skipped = collect(opps)
    if "--integrity-only" in argv:
        c = Counter((u["cell"], u["counts"]) for u in units)
        print(json.dumps({f"{k[0]}:{'counts' if k[1] else 'dropped'}": v for k, v in sorted(c.items())}, indent=1))
        for s in skipped:
            print("DROPPED", s["unit"], s["reasons"][:3])
        return 0
    by = defaultdict(list)
    for r in rows:
        by[r["read_cell"]].append(r)
    K = {s: x51(s) for s in PANELS}
    an = {"kakuna_x51": {s: {k: v for k, v in cell_stats(K[s]).items() if k != "regime_stamps"} for s in PANELS}}
    for a in opps:
        an[a] = analysis_for(a, by, K)
    if all(a in an and "W_U" in an[a] for a in ("LargeRL", "LargeIL")):
        il = {s: by[f"lil_{s}"] for s in PANELS}
        rl = {s: by[f"lrl_{s}"] for s in PANELS}
        d, lo, hi = newcombe_strat(u_score(il), u_score(rl))
        kh = newcombe(sum(won(r) for r in il["home72"]), len(il["home72"]),
                      sum(won(r) for r in rl["home72"]), len(rl["home72"])) if il["home72"] and rl["home72"] else None
        an["IL_minus_RL"] = {"U_newcombe_strat95": [d, lo, hi], "home72_newcombe95": kh,
                             "note": "P*'s win rate vs LargeIL minus vs LargeRL: positive = LargeIL the weaker opponent"}
    cellstats = {c: {**cell_stats(by[c]), "units": [u["unit"] for u in units if u["cell"] == c and u["counts"]]}
                 for c in q.CELLS if by.get(c)}
    doc = {"cells": cellstats, "analysis": an, "units": units, "skipped_units": skipped}
    (OUT / "cells.json").write_text(json.dumps(doc, indent=1, default=float) + "\n")
    with open(OUT / "rows.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(json.dumps(an, indent=1, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
