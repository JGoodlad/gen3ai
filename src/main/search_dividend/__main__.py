"""``python -m main.search_dividend`` — the SEARCH-DIVIDEND probe's CLI.

    # play a cell
    python -m main.search_dividend <run_dir_or_ckpt> --arm oracle --budget 1 \
        --games 10 --opponents heuristic --out tmp/sd.jsonl

    # the MIRROR: our side searches, the opponent is the SAME network with search off
    python -m main.search_dividend <run_dir_or_ckpt> --arm oracle --budget 3 \
        --opponents self --games 30 --out tmp/sd_mirror.jsonl

    # read the file back (no model loaded, no battles)
    python -m main.search_dividend --summary tmp/sd.jsonl

Arms: ``base`` (policy alone — the control), ``honest`` (belief-determinized search) and
``oracle`` (search on the TRUE hidden state). Budgets are a per-decision WALL-CLOCK deadline in
seconds; width is bought in the registered order (opponent actions -> worlds -> dice), then
whatever the clock has left buys DEPTH by iterative deepening (``--max-depth``). The REALIZED
widths AND depth are written into every row, because what a budget actually bought is the finding
and a cap that was never reached is not one.

``--opponents self`` is the MIRROR mode: the searched side against the same network unsearched, so
the two differ in exactly one thing and the no-effect point is 0.50 by construction — unlike the
scripted roster, which saturates near 90% and can hide a real dividend in its ceiling. Mirror
cells default to ``--side-swap`` (every game played in both team orientations off one pinned seed)
so the report can difference out the team draw, and they are read against the null rather than
folded into the anchored-ELO fit, which has no anchor for ``self``.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from typing import List, Optional

from main.search_dividend.battery import MIRROR, Cell, ResultsFile, run_cell
from main.search_dividend.budget import WidthCaps
from main.search_dividend.search import ARMS, SearchConfig
from main.search_dividend.summary import format_report

DEFAULT_BUDGETS = (0.5, 1.0, 3.0, 8.0)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m main.search_dividend",
                                description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("model", nargs="?", help="run dir or checkpoint .zip")
    p.add_argument("--arm", action="append", choices=list(ARMS),
                   help="repeatable; default all three")
    p.add_argument("--budget", action="append", type=float,
                   help="per-decision wall-clock seconds; repeatable (default 0.5 1 3 8)")
    p.add_argument("--games", type=int, default=20, help="games per (arm, budget, opponent) cell")
    p.add_argument("--opponents", default="", help="comma-separated bot names, or \"self\" for the MIRROR (the same network, search off — the sensitive contrast); default the roster")
    p.add_argument("--out", default="tmp/search_dividend.jsonl", help="append-only results JSONL")
    p.add_argument("--summary", metavar="JSONL",
                   help="print the report for an existing results file and exit (no model load)")
    p.add_argument("--anchors", default=None, help="bot ELO anchors json")
    p.add_argument("--games-seed", type=int, default=12345,
                   help="salt for the per-game seed and team draw; MATCHES arms to each other")
    p.add_argument("--device", default="cpu")
    p.add_argument("--impl", default="node", choices=["node", "rust"],
                   help="live battle bridge child")
    p.add_argument("--search-impl", default="node", choices=["node", "rust"],
                   help="search-driver child (node is the validated default for open_root)")
    p.add_argument("--score", default="auto", choices=["auto", "value", "win_prob"])
    p.add_argument("--max-opp", type=int, default=6, help="cap on alpha-pruned opponent actions")
    p.add_argument("--max-worlds", type=int, default=8, help="cap on determinized worlds K")
    p.add_argument("--max-dice", type=int, default=8, help="cap on CRN dice resamples R")
    p.add_argument("--max-depth", type=int, default=3,
                   help="iterative-deepening CAP in plies (default 3). The wall-clock budget "
                        "governs the depth actually reached; this only stops it climbing "
                        "further. 1 = the width-only depth-1 reference. NOTE width is spent "
                        "FIRST (the registered order), so at the default caps it absorbs the "
                        "whole budget and every decision reports depth 1 — lower --max-opp / "
                        "--max-dice to free time for a ply. WARNING depth>=2 is built and gated "
                        "but its successor replay has an OPEN fidelity defect (see deepen.py); "
                        "it fails safe as a counted search_error, but do not publish a depth-2 "
                        "number yet.")
    p.add_argument("--side-swap", dest="side_swap", action="store_true", default=None,
                   help="play every game index in BOTH orientations (searched side gets team A "
                        "then team B, one pinned seed) so the summary can difference out the "
                        "team draw. DEFAULT ON for --opponents self, off otherwise.")
    p.add_argument("--no-side-swap", dest="side_swap", action="store_false",
                   help="disable side-swap pairing even in the mirror")
    p.add_argument("--honest-swap-moves", action="store_true",
                   help="axis M — also resample REVEALED mons' unused moves in the honest arm")
    p.add_argument("--pool-size", type=int, default=0,
                   help="subsample the team pool to N teams (smoke only; 0 = the full pool)")
    p.add_argument("--seed", type=int, default=0, help="engine RNG seed (world sampling)")
    p.add_argument("--compile-extractor", action="store_true",
                   help="torch.compile the extractor for the B=1 LIVE decision (5.45x measured; "
                        "the search's WIDE arm-scoring batch stays eager, where compiling "
                        "measures 0.15-0.43x). Buys games/hour, NOT search width — the live "
                        "forward is outside the per-decision budget. OFF by default because it "
                        "perturbs the forward at ~1e-6 and an argmax over near-tied actions can "
                        "flip on that, which would make rows incomparable across relaunches. "
                        "See perf.py.")
    return p


def resolve_side_swap(flag: Optional[bool], opponent: str) -> bool:
    """Side-swap pairing is ON by default in the MIRROR and off against the scripted bots.

    The asymmetry is the measurement, not a preference. In a mirror both sides are the same
    network, so the ONLY structural difference between them besides the search is which team each
    happens to draw — and at these n that dominates. Against a scripted bot the two sides are not
    interchangeable at all (a heuristic bot is not our network), so swapping the teams does not
    pair anything; it just plays each draw twice. An explicit flag always wins over both.
    """
    if flag is not None:
        return bool(flag)
    return opponent == MIRROR


def _pin_blas() -> None:
    """One BLAS thread per process. The search runs B=1..64 forwards inside a battle loop that
    already owns the box's spare capacity; unpinned, N workers x 16 threads thrashes it (the
    measured ~38x cliff `thread_pinning_test.py` defends)."""
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
              "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(v, "1")


def _load_model(path: str, device: str):
    from sb3_contrib import MaskablePPO
    from main.prober.model import sanitized_load_custom_objects

    ckpt = path
    if os.path.isdir(path):
        latest = os.path.join(path, "latest.txt")
        if os.path.exists(latest):
            ckpt = os.path.join(path, open(latest).read().strip())
        else:
            raise SystemExit(f"{path} has no latest.txt — pass a checkpoint .zip directly")
    custom_objects, dropped = sanitized_load_custom_objects(ckpt, device)
    model = MaskablePPO.load(ckpt, env=None, device=device, custom_objects=custom_objects)
    model.policy.set_training_mode(False)
    for mod in model.policy.modules():
        if hasattr(mod, "_debugger"):
            mod._debugger = None                 # a periodic-log checkpoint prints on every forward
    if dropped:
        print(f"[search_dividend] dropped saved extractor kwargs: {sorted(dropped)}",
              file=sys.stderr)
    return model, ckpt


def _pool(n: int) -> List[str]:
    from utils.team_loader import TeamLoader
    from utils.teambuilder import Gen3Teambuilder

    packed = list(Gen3Teambuilder(TeamLoader().get_all_teams()).packed_teams)
    if n and n < len(packed):
        import random as _r
        # Deterministic subsample: a smoke that draws a different pool every run cannot be
        # compared to itself.
        return _r.Random(0).sample(packed, n)
    return packed


def _pin_sys_path() -> None:
    """Absolutize every ``sys.path`` entry at startup.

    A relative ``PYTHONPATH=…:src`` entry is re-resolved against the CURRENT cwd at every import,
    and this driver imports lazily hours into a run (the ELO fit happens at report time). Measured
    failure, 2026-08-23: the worktree the battery ran from was pruned mid-run, and the final
    report died on ``No module named 'agents.training.elo'`` AFTER 30 games had played — the rows
    survived (append-only), but the report should not be hostage to where the process happens to
    be standing hours after launch."""
    sys.path[:] = [os.path.abspath(p) if p else os.getcwd() for p in sys.path]


def main(argv: Optional[List[str]] = None) -> int:
    _pin_sys_path()
    args = build_parser().parse_args(argv)
    if args.summary:
        rows = ResultsFile(args.summary).rows()
        print(format_report(rows, args.anchors))
        return 0
    if not args.model:
        build_parser().print_usage(sys.stderr)
        print("error: a model path is required unless --summary is given", file=sys.stderr)
        return 2

    _pin_blas()
    from agents.observation.state_encoder import load_mappings
    from agents.training.eval_callback import eval_opponent_names

    model, ckpt = _load_model(args.model, args.device)
    if args.compile_extractor:
        from main.search_dividend.perf import compile_b1_extractor
        ok = compile_b1_extractor(model, enabled=True)
        print(f"[search_dividend] compile-extractor (B=1 only): "
              f"{'ON' if ok else 'UNAVAILABLE — running eager'}", flush=True)
    mappings = load_mappings()
    pool = _pool(args.pool_size)
    opponents = ([o.strip() for o in args.opponents.split(",") if o.strip()]
                 or eval_opponent_names())
    arms = args.arm or list(ARMS)
    budgets = args.budget or list(DEFAULT_BUDGETS)
    results = ResultsFile(args.out)
    caps = WidthCaps(m_opp=args.max_opp, k_worlds=args.max_worlds, r_dice=args.max_dice)

    print(f"[search_dividend] ckpt={ckpt}\n"
          f"  arms={arms} budgets={budgets} opponents={opponents} games={args.games}\n"
          f"  max_depth={args.max_depth} side_swap={'auto' if args.side_swap is None else args.side_swap}\n"
          f"  pool={len(pool)} teams  impl={args.impl}/search:{args.search_impl}  out={args.out}",
          flush=True)

    def progress(row: dict) -> None:
        fb = ",".join(f"{k}:{v}" for k, v in sorted((row.get("fallbacks") or {}).items())) or "-"
        print(f"  [{row['arm']}@{row['budget']:g} {row['opponent']} "
              f"g{row['game']}/o{row.get('orientation', 0)}] "
              f"{row['result']} {row['wall_s']}s dec={row['n_decisions']} "
              f"searched={row['n_searched']} changed={row['n_changed']} "
              f"deep={row.get('n_deepened', 0)}@{row.get('max_depth_realized', 1)} fb={fb} "
              f"realized={row.get('realized_mean')}" + (f" ERR={row['error']}" if row.get("error") else ""),
              flush=True)

    total = 0
    for arm in arms:
        for budget in ([0.0] if arm == "base" else budgets):
            cfg = SearchConfig(arm=arm, budget_s=budget, caps=caps, score=args.score,
                               search_impl=args.search_impl,
                               honest_swap_moves=args.honest_swap_moves, seed=args.seed,
                               max_depth=args.max_depth)
            for opp in opponents:
                cell = Cell(arm=arm, budget=budget, opponent=opp)
                n = asyncio.run(run_cell(
                    cell, model=model, mappings=mappings, cfg=cfg, games=args.games,
                    results=results, salt=args.games_seed, impl=args.impl,
                    pool_packed=pool, progress=progress,
                    side_swap=resolve_side_swap(args.side_swap, opp)))
                total += n
    print(f"\n[search_dividend] played {total} new games -> {args.out}\n", flush=True)
    print(format_report(results.rows(), args.anchors))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
