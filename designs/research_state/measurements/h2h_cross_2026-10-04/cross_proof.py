"""F-U6-1's proof for the TWO-ARCHITECTURE head-to-head engine (``main.h2h.arch``): on REAL production (``blob``)
checkpoints — the X5 P0 sizing finals, read-only — beside an X5 ``fixed_mass`` checkpoint (seeded perturbed-fresh: no
trained ``fixed_mass`` run exists yet), CPU only.

    PYTHONPATH=<tree>/src python cross_proof.py play --out rows.json      # from the repo root (the team pool)
    PYTHONPATH=<tree>/src python cross_proof.py compare rows.json > compare.json

``play``: (1) ONE ``play_cells`` call over U6's four blob cells PLUS two cross cells (fixed_mass vs A2, fixed_mass vs
B) — a two-group engine (group 0 blob: a player and an opponent slot; group 1 fixed_mass: a player slot; two eval
cores); (2) the two cross cells again, alone, on a fresh ledger — an engine whose groups are declared the other way
round (group 0 fixed_mass: a player slot; group 1 blob: an opponent slot; one core). ``compare``: (a) every blob row
of (1) vs U6's single-cell ``play_edge`` rows on the same seeds (``../h2h_multicell_2026-10-04/rows.json.gz``):
counts, pentanomial, team counters, seed block, outcome digest; (b) its ``outcome_digest_all`` (EVERY game) vs U6's
engine-level ``digest_all`` of the same cell and batch (``games.json.gz``, a fresh single-group engine); (c) the cross
rows of (1) vs (2): every count, digest and decision count equal. Exit 0 iff all hold.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List

M = "/home/goodlad/dev/gen3ai/models"
A2, AP, B = (f"{M}/sizing_A2_n48_e10_s1001/final_model.zip", f"{M}/sizing_Ap_n48_e10_s1002/final_model.zip",
             f"{M}/sizing_B_n256_e10_s1001/final_model.zip")
BLOB_CELLS = [(A2, AP), (AP, B), (B, A2), (AP, A2)]            # U6's cells, in U6's order
PAIRS, BATCH, SEED = 100, 50, 0
U6 = Path(__file__).resolve().parent.parent / "h2h_multicell_2026-10-04"
SAME = ("counts", "pairs", "teams", "seed", "regime")
SAME_COMPUTE = ("outcome_digest", "outcome_digest_all", "near_tie_games", "trainee_decisions", "p2_policy_decisions",
                "near_tie_decisions", "near_tie_decisions_wide")


def compute():
    from main.h2h import play as PL

    return PL.Compute(device="cpu", backend="eager", n_envs=64, threads=4, torch_threads=4)


def build_fixed_mass(dst: Path) -> str:
    """The test fixture's recipe (``main/h2h/conftest.py`` ``foreign``): production surface + ``--belief-tokens
    fixed_mass``, seed 7, perturbed (seed 2700), built at one thread; ``model_config.json`` beside it."""
    from agents.model.parity_probe import PERTURB_SCALE, perturb_
    from agents.model.snapshot import arch_toggles_from_model, current_model_version
    from agents.observation.state_encoder import load_mappings
    from agents.training.rust_eval import parity as PAR
    from main.fresh_checkpoint import build_fresh_model
    from main.train.production_args import production_args

    dst.mkdir(parents=True)
    args = production_args()
    args.belief_tokens = "fixed_mass"
    with PAR.declared_torch_state(1):
        model, _, _ = build_fresh_model(7, args=args)
        perturb_(model.policy, seed=2700, scale=PERTURB_SCALE)
        path = dst / "snapshot_000000007000.zip"
        model.save(str(path))
    (dst / "model_config.json").write_text(
        current_model_version(load_mappings(), **arch_toggles_from_model(model)).to_json())
    return str(path)


def cmd_play(a: argparse.Namespace) -> int:
    from agents.training import eval_ledger as L
    from main.h2h import many as MANY
    from main.h2h import play as PL

    decl = L.ReaderDecl(name="cross_proof", purposes=L.ALL_PURPOSES, regime=L.RegimeFilter(), requests="any",
                        selection="include", flags_ok=frozenset(), inference="conditional")
    with tempfile.TemporaryDirectory(prefix="h2h_cross_") as tmp:
        fm = build_fixed_mass(Path(tmp) / "run_x5_fixed_mass_fresh")
        kw = dict(pairs=PAIRS, batch_pairs=BATCH, schedule_seed=SEED, run_label="h2h_cross_proof", compute=compute())
        t0 = time.perf_counter()
        look = MANY.play_cells(f"{tmp}/look", MANY.resolve_cells(BLOB_CELLS + [(fm, A2), (fm, B)]), **kw)
        t_look = time.perf_counter() - t0
        t0 = time.perf_counter()
        rerun = MANY.play_cells(f"{tmp}/rerun", MANY.resolve_cells([(fm, A2), (fm, B)]), **kw)
        t_rerun = time.perf_counter() - t0
        out = {"fixed_mass": {"path": "<tmp>/run_x5_fixed_mass_fresh/snapshot_000000007000.zip",
                              "sha256": PL.file_sha256(fm)},
               "look": list(L.read(decl, root=f"{tmp}/look").rows), "rerun": list(L.read(decl, root=f"{tmp}/rerun").rows),
               "look_engine": look["engine"], "rerun_engine": rerun["engine"],
               "wall_look_s": round(t_look, 1), "wall_rerun_s": round(t_rerun, 1), "pid": os.getpid()}
    with open(a.out, "w") as f:
        json.dump(out, f, sort_keys=True)
    return 0


def _load(path: Any) -> Any:
    path = str(path)
    with (gzip.open(path, "rt") if path.endswith(".gz") else open(path)) as f:
        return json.load(f)


def cmd_compare(a: argparse.Namespace) -> int:
    got = _load(a.rows)
    u6_rows, u6_games = _load(U6 / "rows.json.gz"), _load(U6 / "games.json.gz")
    key = lambda r: (r["player"]["sha256"], r["opponent"]["sha256"], r["seed"]["batch"])   # noqa: E731
    problems: List[str] = []
    single = {key(r): r for r in u6_rows["single"]}
    look = {key(r): r for r in got["look"]}
    blob_shas = {r["player"]["sha256"] for r in u6_rows["single"]} | {r["opponent"]["sha256"] for r in u6_rows["single"]}
    look_blob = {k: r for k, r in look.items() if k[0] in blob_shas}
    look_cross = {k: r for k, r in look.items() if k[0] not in blob_shas}
    # (a) blob rows on the two-group engine == U6's single-cell rows
    if set(look_blob) != set(single) or not single:
        problems.append(f"blob row sets differ: {len(look_blob)} two-group vs {len(single)} single-cell")
    for k, s in single.items():
        m = look_blob.get(k)
        if m is None:
            continue
        for f in ("counts", "pairs", "teams", "seed"):
            if s[f] != m[f]:
                problems.append(f"blob row {k}: {f} differs")
        if s["compute"]["outcome_digest"] != m["compute"]["outcome_digest"]:
            problems.append(f"blob row {k}: outcome digest differs")
    # (b) every game: the row's outcome_digest_all == U6's fresh single-group engine digest_all
    eng_all: Dict[Any, str] = {}
    for c in u6_games["fresh"]:
        for b in c["batches"]:
            eng_all[(c["cell"][0], c["cell"][1], b["batch"])] = b["digest_all"]
    n_all = 0
    for k, m in look_blob.items():
        want = eng_all.get((m["player"]["id"], m["opponent"]["id"], m["seed"]["batch"]))
        n_all += 1
        if want is None or want != m["compute"]["outcome_digest_all"]:
            problems.append(f"blob row {k}: outcome_digest_all {m['compute']['outcome_digest_all'][:12]} != the "
                            f"single-group engine's {str(want)[:12]}")
    # (c) the cross rows replay exactly on an engine declared the other way round
    rerun = {key(r): r for r in got["rerun"]}
    if set(rerun) != set(look_cross) or len(look_cross) != 4:
        problems.append(f"cross row sets differ: {len(look_cross)} in the look vs {len(rerun)} re-run (want 4)")
    for k, r in look_cross.items():
        q = rerun.get(k)
        if q is None:
            continue
        for f in SAME:
            if r[f] != q[f]:
                problems.append(f"cross row {k}: {f} differs")
        for f in SAME_COMPUTE:
            if r["compute"][f] != q["compute"][f]:
                problems.append(f"cross row {k}: compute.{f} differs")
    near_cross = sum(len(r["compute"]["near_tie_games"]) for r in look_cross.values())
    near_blob = sum(len(r["compute"]["near_tie_games"]) for r in look_blob.values())
    out = {"ok": not problems, "problems": problems, "blob_rows_compared": len(single),
           "blob_rows_digest_all_compared": n_all, "cross_rows_compared": len(look_cross),
           "games_compared": 2 * PAIRS * (len(single) // 2 + len(look_cross) // 2),
           "near_tie_games_wide": {"blob": near_blob, "cross": near_cross},
           "cross_win_rates": {f"{r['player']['id']} vs {r['opponent']['id']} b{r['seed']['batch']}":
                               r["counts"] for r in look_cross.values()},
           "look_engine": got["look_engine"], "rerun_engine": got["rerun_engine"],
           "wall_look_s": got["wall_look_s"], "wall_rerun_s": got["wall_rerun_s"]}
    print(json.dumps(out, indent=1, sort_keys=True))
    return 0 if not problems else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("play")
    p.add_argument("--out", required=True)
    c = sub.add_parser("compare")
    c.add_argument("rows")
    a = ap.parse_args()
    return {"play": cmd_play, "compare": cmd_compare}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
