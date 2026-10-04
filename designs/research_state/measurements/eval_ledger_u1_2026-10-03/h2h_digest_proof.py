"""U1's STORAGE-ONLY proof for ``main.h2h``: the same seeded batches play IDENTICAL games before and after the
ledger-v2 migration (design_evaluation.md §0c rule 6; X5's registration pins the h2h protocol, review M3(c)).

    # 1. the per-game outcome vectors, in EACH tree (PYTHONPATH picks the tree):
    PYTHONPATH=<base>/src  python h2h_digest_proof.py games --out base_games.json
    PYTHONPATH=<u1>/src    python h2h_digest_proof.py games --out u1_games.json
    # 2. the CLI in each tree (base writes v1 rows to a dir, U1 writes v2 rows to a ledger root):
    PYTHONPATH=<base>/src  python -m main.h2h play ... --out <base_rows>
    PYTHONPATH=<u1>/src    python -m main.h2h play ... --out <u1_root>
    # 3. the comparison (U1 tree):
    PYTHONPATH=<u1>/src    python h2h_digest_proof.py compare base_games.json u1_games.json <base_rows> <u1_root>

``games`` builds ``main.h2h.play.H2HEngine`` on the two players and plays each batch at the CLI's own seed
(``schedule_key_of`` + ``cycle_seed``), dumping every game's ``(game, result, swapped, teams, end_turn, winner,
forfeit, near-tie counts)``. The outcome digest is computed HERE by a local copy of the v2 algorithm
(``eval_ledger.schema.outcome_digest``, tag ``gen3_eval_outcome_digest_v1``), so the base tree — which has no digest
— is hashed by exactly the same code. ``compare`` asserts: (a) the two trees' game vectors are byte-identical; (b)
their digests over the non-near-tie games are equal; (c) the U1 CLI rows' ``compute.outcome_digest`` equal the base
tree's games' digests batch for batch; (d) the base v1 rows and the U1 v2 rows carry the same counts, pentanomial,
team counters and cycle seeds. Exit 0 iff all hold.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from typing import Any, Dict, List

A = "/home/goodlad/dev/gen3ai/models/sizing_A_n48_e10_s1001/final_model.zip"
B = "/home/goodlad/dev/gen3ai/models/sizing_B_n256_e10_s1001/final_model.zip"
TAG = "gen3_eval_outcome_digest_v1"
NEAR_WIDE_KEY = "near_ties_wide"


def digest(games: List[Dict[str, Any]], skip_near: bool = True) -> str:
    letter = {"WIN": "W", "LOSS": "L", "DRAW": "D"}
    lines = [TAG]
    for g in sorted(games, key=lambda g: int(g["game"])):
        if skip_near and int(g.get(NEAR_WIDE_KEY, 0)):
            continue
        t = g.get("end_turn")
        lines.append(f"{int(g['game'])}:{letter[str(g['result'])]}:{'' if t is None else int(t)}")
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()


def cmd_games(a: argparse.Namespace) -> int:
    from main.h2h import play as PL

    pa, pb = PL.resolve_player(a.player), PL.resolve_player(a.opponent)
    comp = PL.Compute(device="cpu", backend="eager", n_envs=a.n_envs, threads=a.threads, torch_threads=a.threads)
    key = PL.schedule_key_of(pa, pb)
    eng = PL.H2HEngine(pa, pb, comp)
    out: Dict[str, Any] = {"player": pa.sha256, "opponent": pb.sha256, "schedule_key": key, "batches": []}
    try:
        for b in range(a.batches):
            cs = PL.cycle_seed(a.seed, key, b)
            games, st = eng.play_batch(a.pairs, cs)
            keep = ("game", "result", "swapped", "teams", "end_turn", "winner", "forfeit", "near_ties", NEAR_WIDE_KEY)
            gl = sorted(({k: g.get(k) for k in keep} for g in games), key=lambda g: int(g["game"]))
            sc = PL.score_games(games, eng.team_packed, a.pairs)
            out["batches"].append({"batch": b, "cycle_seed": cs, "games": gl, "pair_counts": sc.pair_counts,
                                   "wld": [sc.w, sc.l, sc.d], "teams": sc.teams,
                                   "near_wide": sorted(int(g["game"]) for g in gl if int(g.get(NEAR_WIDE_KEY, 0))),
                                   "digest": digest(gl), "digest_all_games": digest(gl, skip_near=False)})
            print(f"[proof] batch {b}: cycle seed {cs} W/L/D {sc.w}/{sc.l}/{sc.d} pairs {sc.pair_counts} "
                  f"near-wide {len(out['batches'][-1]['near_wide'])} digest {out['batches'][-1]['digest'][:12]}",
                  flush=True)
    finally:
        eng.close()
    with open(a.out, "w") as f:
        json.dump(out, f, sort_keys=True)
    return 0


def cmd_compare(a: argparse.Namespace) -> int:
    from agents.training import eval_ledger as L

    base, u1 = json.load(open(a.base_games)), json.load(open(a.u1_games))
    problems: List[str] = []
    if json.dumps(base, sort_keys=True) != json.dumps(u1, sort_keys=True):
        problems.append("(a) the two trees' per-game vectors differ")
    for bb, ub in zip(base["batches"], u1["batches"]):
        if bb["digest"] != ub["digest"]:
            problems.append(f"(b) batch {bb['batch']}: digests differ")
    decl = L.ReaderDecl(name="u1_digest_proof", purposes=L.ALL_PURPOSES, regime=L.RegimeFilter(), requests="any",
                        selection="include", flags_ok=frozenset({"digest_unrecorded"}), inference="conditional")
    v1 = sorted(L.read(decl, root=a.base_rows).rows, key=lambda r: r["seed"]["batch"])
    v2 = sorted(L.read(decl, root=a.u1_root).rows, key=lambda r: r["seed"]["batch"])
    if len(v1) != len(base["batches"]) or len(v2) != len(base["batches"]):
        problems.append(f"row counts: base {len(v1)}, U1 {len(v2)}, batches {len(base['batches'])}")
    for bb, r1, r2 in zip(base["batches"], v1, v2):
        if r2["compute"]["outcome_digest"] != bb["digest"]:
            problems.append(f"(c) batch {bb['batch']}: the U1 row's digest != the base tree's games' digest")
        if r2["compute"]["near_tie_games"] != bb["near_wide"]:
            problems.append(f"(c) batch {bb['batch']}: near-tie indices differ")
        for k in ("counts", "pairs", "teams"):
            x1 = {kk: vv for kk, vv in r1[k].items() if kk != "aborted"} if k == "counts" else r1[k]
            x2 = {kk: vv for kk, vv in r2[k].items() if kk != "aborted"} if k == "counts" else r2[k]
            if x1 != x2:
                problems.append(f"(d) batch {bb['batch']}: {k} differ")
        if r1["seed"]["cycle_seed"] != r2["seed"]["cycle_seed"] or r2["seed"]["cycle_seed"] != bb["cycle_seed"]:
            problems.append(f"(d) batch {bb['batch']}: cycle seeds differ")
    rep = {"batches": len(base["batches"]), "games": sum(len(b["games"]) for b in base["batches"]),
           "digests": [b["digest"] for b in base["batches"]],
           "near_tie_games": [len(b["near_wide"]) for b in base["batches"]],
           "wld": [b["wld"] for b in base["batches"]], "pair_counts": [b["pair_counts"] for b in base["batches"]],
           "base_schema": sorted({r.get("regime", {}).get("v1_id") is not None for r in v1}),
           "problems": problems, "verdict": "IDENTICAL" if not problems else "DIFFERENT"}
    print(json.dumps(rep, indent=1, sort_keys=True))
    return 0 if not problems else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("games")
    g.add_argument("--player", default=A)
    g.add_argument("--opponent", default=B)
    g.add_argument("--pairs", type=int, default=50)
    g.add_argument("--batches", type=int, default=2)
    g.add_argument("--seed", type=int, default=7)
    g.add_argument("--n-envs", type=int, default=16)
    g.add_argument("--threads", type=int, default=4)
    g.add_argument("--out", required=True)
    c = sub.add_parser("compare")
    c.add_argument("base_games")
    c.add_argument("u1_games")
    c.add_argument("base_rows")
    c.add_argument("u1_root")
    a = ap.parse_args()
    return cmd_games(a) if a.cmd == "games" else cmd_compare(a)


if __name__ == "__main__":
    sys.exit(main())
