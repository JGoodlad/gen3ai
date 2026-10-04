"""U6 / X5 U0's STORAGE-AND-ENGINE-REUSE proof for ``main.h2h play-many``: many cells on ONE engine play the games
single-cell ``main.h2h play`` plays on the same seeds, on REAL production-architecture checkpoints (the P0 runs).

    PYTHONPATH=<tree>/src python multicell_proof.py games  --out games.json   # engine level, every game
    PYTHONPATH=<tree>/src python multicell_proof.py rows   --out rows.json    # the CLI paths, row level
    PYTHONPATH=<tree>/src python multicell_proof.py compare games.json rows.json > compare.json

``games``: for each cell, (a) a FRESH engine (the single-cell path) and (b) ONE engine swapped from cell to cell
(``H2HEngine.set_cell``) play the cell's batches at the CLI's own seeds; the game logs, the outcome digest over the
non-near-tie games (the row's ``compute.outcome_digest``) and ``digest_all`` (every game, near-ties included) are
dumped, with the timings (engine start, swap, play). ``rows``: ``play_edge`` per cell into one ledger root and
``play_cells`` over the same cells into another; their rows. ``compare``: (1) fresh == swapped game logs, byte for
byte, every cell and batch; (2) their digests and digests-all equal; (3) the single-cell and multi-cell ROWS carry
the same counts, pentanomial, team counters, seed block and outcome digest, and (4) those digests equal the
engine-level games' digests. Exit 0 iff all hold.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from typing import Any, Dict, List

M = "/home/goodlad/dev/gen3ai/models"
A2, AP, B = (f"{M}/sizing_A2_n48_e10_s1001/final_model.zip", f"{M}/sizing_Ap_n48_e10_s1002/final_model.zip",
             f"{M}/sizing_B_n256_e10_s1001/final_model.zip")
CELLS = [(A2, AP), (AP, B), (B, A2), (AP, A2)]
PAIRS, BATCH, SEED = 100, 50, 0
KEEP = ("game", "result", "swapped", "teams", "end_turn", "winner", "forfeit", "near_ties", "near_ties_wide")


def compute():
    from main.h2h import play as PL

    return PL.Compute(device="cpu", backend="eager", n_envs=64, threads=4, torch_threads=4)


def batch_dump(PL: Any, L: Any, eng: Any, p: Any, o: Any) -> List[Dict[str, Any]]:
    key = PL.schedule_key_of(p, o)
    out = []
    for b, n in enumerate(PL.batch_plan(PAIRS, BATCH)):
        cs = PL.cycle_seed(SEED, key, b)
        t0 = time.perf_counter()
        games, _st = eng.play_batch(n, cs)
        wall = time.perf_counter() - t0
        gl = sorted(({k: g.get(k) for k in KEEP} for g in games), key=lambda g: int(g["game"]))
        sc = PL.score_games(games, eng.team_packed, n)
        out.append({"batch": b, "cycle_seed": cs, "games": gl, "digest": sc.outcome_digest,
                    "digest_all": L.outcome_digest(PL.outcome_vector(games), []), "wall_s": round(wall, 2),
                    "near_wide": sc.near_tie_idx})
    return out


def cmd_games(a: argparse.Namespace) -> int:
    from agents.training import eval_ledger as L
    from main.h2h import play as PL

    cells = [(PL.resolve_player(p), PL.resolve_player(o)) for p, o in CELLS]
    res: Dict[str, Any] = {"fresh": [], "swapped": [], "timing": {"fresh_startup_s": [], "swap_s": []}}
    for p, o in cells:                                               # (a) the single-cell path: one engine per cell
        eng = PL.H2HEngine(p, o, compute())
        try:
            res["timing"]["fresh_startup_s"].append(round(eng.startup_s, 2))
            res["fresh"].append({"cell": [p.id, o.id], "batches": batch_dump(PL, L, eng, p, o)})
        finally:
            eng.close()
    eng = PL.H2HEngine(*cells[0], compute())                         # (b) ONE engine, swapped
    try:
        res["timing"]["swapped_startup_s"] = round(eng.startup_s, 2)
        for i, (p, o) in enumerate(cells):
            res["timing"]["swap_s"].append(round(eng.set_cell(p, o), 3) if i else 0.0)
            res["swapped"].append({"cell": [p.id, o.id], "batches": batch_dump(PL, L, eng, p, o)})
    finally:
        eng.close()
    with open(a.out, "w") as f:
        json.dump(res, f, sort_keys=True)
    return 0


def cmd_rows(a: argparse.Namespace) -> int:
    from agents.training import eval_ledger as L
    from main.h2h import many as MANY
    from main.h2h import play as PL

    decl = L.ReaderDecl(name="multicell_proof", purposes=L.ALL_PURPOSES, regime=L.RegimeFilter(), requests="any",
                        selection="include", flags_ok=frozenset(), inference="conditional")
    cells = MANY.resolve_cells(CELLS)
    with tempfile.TemporaryDirectory(prefix="h2h_multicell_") as tmp:
        t0 = time.perf_counter()
        for p, o in cells:
            PL.play_edge(f"{tmp}/single", p, o, pairs=PAIRS, batch_pairs=BATCH, schedule_seed=SEED,
                         run_label="h2h_multicell_proof", compute=compute())
        t_single = time.perf_counter() - t0
        t0 = time.perf_counter()
        many = MANY.play_cells(f"{tmp}/many", cells, pairs=PAIRS, batch_pairs=BATCH, schedule_seed=SEED,
                               run_label="h2h_multicell_proof", compute=compute())
        t_many = time.perf_counter() - t0
        out = {"single": list(L.read(decl, root=f"{tmp}/single").rows), "many": list(L.read(decl, root=f"{tmp}/many").rows),
               "engine": many["engine"], "wall_single_s": round(t_single, 1), "wall_many_s": round(t_many, 1)}
    with open(a.out, "w") as f:
        json.dump(out, f, sort_keys=True)
    return 0


def _load(path: str) -> Any:
    import gzip

    with (gzip.open(path, "rt") if path.endswith(".gz") else open(path)) as f:
        return json.load(f)


def cmd_compare(a: argparse.Namespace) -> int:
    games, rows = _load(a.games), _load(a.rows)
    problems: List[str] = []
    n_games = 0
    eng_digest = {}
    for fr, sw in zip(games["fresh"], games["swapped"]):
        for bf, bs in zip(fr["batches"], sw["batches"]):
            tag = f"{fr['cell']} batch {bf['batch']}"
            n_games += len(bf["games"])
            if bf["games"] != bs["games"]:
                problems.append(f"{tag}: the game logs differ")
            for k in ("digest", "digest_all", "cycle_seed"):
                if bf[k] != bs[k]:
                    problems.append(f"{tag}: {k} differs")
            # keyed by the CELL and batch: A-vs-B and B-vs-A share a schedule key, hence their cycle seeds
            eng_digest[(fr["cell"][0], fr["cell"][1], bf["batch"])] = bf["digest"]
    key = lambda r: (r["player"]["sha256"], r["opponent"]["sha256"], r["seed"]["batch"])   # noqa: E731
    single, many = {key(r): r for r in rows["single"]}, {key(r): r for r in rows["many"]}
    if set(single) != set(many) or not single:
        problems.append(f"row sets differ: {len(single)} single vs {len(many)} many")
    for k, s in single.items():
        m = many.get(k)
        if m is None:
            continue
        for f in ("counts", "pairs", "teams", "seed"):
            if s[f] != m[f]:
                problems.append(f"row {k}: {f} differs")
        if s["compute"]["outcome_digest"] != m["compute"]["outcome_digest"]:
            problems.append(f"row {k}: outcome digest differs")
        if eng_digest.get((s["player"]["id"], s["opponent"]["id"], s["seed"]["batch"])) != s["compute"]["outcome_digest"]:
            problems.append(f"row {k}: the row's digest is not the engine-level games' digest")
    near = sum(len(b["near_wide"]) for c in games["fresh"] for b in c["batches"])
    out = {"ok": not problems, "problems": problems, "cells": len(games["fresh"]), "games_compared": n_games,
           "near_tie_games_wide": near, "rows_compared": len(single), "timing": games["timing"],
           "rows_engine": rows["engine"], "wall_single_s": rows["wall_single_s"], "wall_many_s": rows["wall_many_s"]}
    print(json.dumps(out, indent=1, sort_keys=True))
    return 0 if not problems else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("games")
    g.add_argument("--out", required=True)
    r = sub.add_parser("rows")
    r.add_argument("--out", required=True)
    c = sub.add_parser("compare")
    c.add_argument("games")
    c.add_argument("rows")
    a = ap.parse_args()
    return {"games": cmd_games, "rows": cmd_rows, "compare": cmd_compare}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
