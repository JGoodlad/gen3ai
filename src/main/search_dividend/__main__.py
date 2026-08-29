"""``python -m main.search_dividend`` — the SEARCH-DIVIDEND probe's CLI.

    # play a cell
    python -m main.search_dividend <run_dir_or_ckpt> --arm oracle --budget 1 \
        --games 10 --opponents heuristic --out tmp/sd.jsonl

    # the MIRROR: our side searches, the opponent is the SAME network with search off
    python -m main.search_dividend <run_dir_or_ckpt> --arm oracle --budget 3 \
        --opponents self --games 30 --out tmp/sd_mirror.jsonl

    # the TOP-2 PLAYOFF mirror cell — screen with the critic, decide with paired rollouts
    python -m main.search_dividend <ckpt> --arm playoff --budget 10 --opponents self \
        --games 40 --seed 11 --max-depth 1 --compile-extractor --out tmp/sd_playoff.jsonl

    # read the file back (no model loaded, no battles)
    python -m main.search_dividend --summary tmp/sd.jsonl

Arms: ``base`` (policy alone — the control), ``honest`` (belief-determinized search), ``oracle``
(search on the TRUE hidden state) and ``playoff`` (opt-in — the oracle sweep demoted to a SCREEN,
with the top-2 settled by paired ROLLOUTS to a terminal; see :mod:`playoff`). Budgets are a
per-decision WALL-CLOCK deadline in seconds; width is bought in the registered order (opponent actions -> worlds -> dice), then
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
from main.search_dividend.defensive import LEAVES, DefensiveConfig
from main.search_dividend.playoff import (DEFAULT_ROLLOUTS, DEFAULT_SCREEN_MARGIN, MIN_PAIRS,
                                          SE_MULTIPLE, PlayoffConfig)
from main.search_dividend.racing import RULES, RacingConfig
from main.search_dividend.search import ARMS, ROOT_STRATEGIES, SearchConfig
from main.search_dividend.summary import format_report

DEFAULT_BUDGETS = (0.5, 1.0, 3.0, 8.0)

#: The arms a flagless run plays. NOT ``ARMS``: ``playoff`` is a registered single-cell experiment
#: whose rollouts cost orders of magnitude more than a critic sweep, so it is opt-in by name. A
#: default that silently added it would turn every existing battery invocation into a different
#: (and much longer) experiment.
DEFAULT_ARMS = ("base", "honest", "oracle")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m main.search_dividend",
                                description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("model", nargs="?", help="run dir or checkpoint .zip")
    p.add_argument("--arm", action="append", choices=list(ARMS),
                   help="repeatable; default base/honest/oracle. `playoff` is OPT-IN: it screens "
                        "with the depth-1 critic sweep and settles the top-2 with paired "
                        "rollouts to a terminal (see playoff.py)")
    p.add_argument("--budget", action="append", type=float,
                   help="per-decision wall-clock seconds; repeatable (default 0.5 1 3 8)")
    p.add_argument("--games", type=int, default=20, help="games per (arm, budget, opponent) cell")
    p.add_argument("--games-start", type=int, default=0, metavar="I",
                   help="play game indices [I, I+--games) instead of [0, --games). SHARDING only: "
                        "the per-game seed and team draw are functions of the index, so two "
                        "processes over disjoint windows write rows that concatenate into exactly "
                        "the file one process would have. Use this rather than a second "
                        "--games-seed, which would re-use index 0 for a different battle and break "
                        "the summary's paired differencing.")
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
    p.add_argument("--root-strategy", default="grid", choices=list(ROOT_STRATEGIES),
                   help="how the budget is spread ACROSS OUR ROOT ACTIONS. `grid` (default) is "
                        "the registered fixed sweep — every action on every sample. `racing` "
                        "eliminates candidates whose CRN-paired difference CI separates below the "
                        "leader and spends the saved arm evaluations on MORE samples instead; the "
                        "width ORDER is unchanged, and a racing round is depth 1 (racing and "
                        "iterative deepening are not composed). See racing.py. `defensive` is the "
                        "same race wrapped in two REFUSALS — a triage gate that never searches a "
                        "decided position, and a futility stop that never overrules without "
                        "separation. See defensive.py.")
    p.add_argument("--defensive-leaf", default=DefensiveConfig.leaf, choices=list(LEAVES),
                   help="which critic readout the race scores on. `winprob` (default) is MEASURED, "
                        "not preferred: ranking by the one-ply win-prob head beat the played "
                        "action by +0.0219 [+0.0089,+0.0364] win probability, while the scalar "
                        "`value` head's +0.0135 [-0.0007,+0.0280] does not clear zero (probe G, "
                        "317 decisions / 142,208 terminal rollouts). Unlike `--score auto` this "
                        "never silently falls back — a checkpoint with no win-prob head raises. "
                        "--root-strategy defensive only.")
    p.add_argument("--defensive-wp-margin", type=float, default=DefensiveConfig.wp_margin,
                   help=f"the triage gate: play the policy immediately when n_legal<=1 or "
                        f"|P(win)-0.5| >= this (default {DefensiveConfig.wp_margin}). Probe H's "
                        "operating point — 82.5%% of decisions forced, 5.7x budget concentration, "
                        "31.0%% of the claimed dividend retained vs a random triage's 16.5%%. "
                        "--root-strategy defensive only.")
    p.add_argument("--defensive-confirm", type=int, default=DefensiveConfig.confirm_rollouts,
                   metavar="N",
                   help="OPT-IN fourth stage (default 0 = off): before acting on an overrule, "
                        "settle race-winner vs policy action with N PAIRED rollouts to a terminal "
                        "through the playoff mechanism, and keep the policy's action unless the "
                        "paired difference clears 2*SE. Off in the first registered cell — one "
                        "new mechanism at a time. Pair it with --defensive-confirm-deadline-s: "
                        "on the race's leftover clock the runner can afford ONE pair and "
                        "MIN_PAIRS=4 then declines every confirm. --root-strategy defensive only.")
    p.add_argument("--defensive-confirm-deadline-s", type=float, default=None, metavar="SEC",
                   help="the CONFIRM stage's own wall clock (default: unset = share whatever the "
                        "race left, which is the built behaviour and is ~1 s against a measured "
                        "~1.5 s per PAIR). Set it so N pairs, not the clock, is what binds — and "
                        "it doubles as the ADJUDICATION CAP that keeps a nested rollout family "
                        "inside the live battle's --battle-idle-s watchdog. "
                        "--defensive-confirm > 0 only.")
    p.add_argument("--defensive-contested-deadline-s", type=float, default=None, metavar="SEC",
                   help="THE TIME MANAGER (default: unset = a contested decision gets --budget, "
                        "i.e. the first registered cell's behaviour exactly). The triage gate "
                        "forces ~74%% of decisions and spends nothing on them, so the notional "
                        "budget they would have burned is real and BANKED — measured at 0.77 s of "
                        "every 1 s, 28.8 s per game. This grants a CONTESTED decision that clock "
                        "instead. It buys ROUNDS: the first cell's mean race ran 4.61 rounds "
                        "against the seq rule's elimination FLOOR of 5, and every one of its "
                        "futility stops was also deadline-truncated, so the strategy was "
                        "budget-limited at the floor rather than evidence-limited. Size it so "
                        "(contested decisions per game) x SEC stays inside the uniform notional "
                        "the gate hands back. --root-strategy defensive only.")
    p.add_argument("--racing-rule", default=RacingConfig.rule, choices=list(RULES),
                   help="`z` = a one-sided normal test per look (aggressive; the A/B ran on this); "
                        "`seq` inflates the radius by a union bound over rounds and comparisons so "
                        "the error is controlled ANYTIME. --root-strategy racing only.")
    p.add_argument("--racing-z", type=float, default=RacingConfig.z,
                   help=f"elimination threshold in SEs of the paired difference (default "
                        f"{RacingConfig.z}); --racing-rule z only")
    p.add_argument("--racing-delta", type=float, default=RacingConfig.delta,
                   help=f"family-wise error target for --racing-rule seq (default "
                        f"{RacingConfig.delta})")
    p.add_argument("--racing-min-samples", type=int, default=RacingConfig.min_samples,
                   help=f"rounds every action is scored on before ANY elimination (default "
                        f"{RacingConfig.min_samples}). Below this a paired sd is not an estimate "
                        "and an elimination made on it is a coin flip the race then treats as "
                        "settled forever.")
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
    p.add_argument("--playoff-rollouts", type=int, default=DEFAULT_ROLLOUTS,
                   help=f"R — paired rollouts per playoff (default {DEFAULT_ROLLOUTS}). A CAP, not "
                        "a target: the per-decision budget governs the REALIZED count and every "
                        "row records it. --arm playoff only.")
    p.add_argument("--playoff-screen-margin", type=float, default=DEFAULT_SCREEN_MARGIN,
                   help=f"skip the playoff when the screen's top1 IS the policy's action and its "
                        f"top1-top2 margin exceeds this (default {DEFAULT_SCREEN_MARGIN} = 2x the "
                        "measured per-leaf sd of 0.0115). Raising it spends more rollouts on "
                        "decisions the screen already agrees with; lowering it spends fewer.")
    p.add_argument("--playoff-se-k", type=float, default=SE_MULTIPLE,
                   help=f"how many SEs the paired difference must clear before the playoff may "
                        f"override (default {SE_MULTIPLE}). Below this it plays the POLICY's "
                        "action and counts playoff_inconclusive — the search never overrides on "
                        "noise.")
    p.add_argument("--playoff-min-pairs", type=int, default=MIN_PAIRS,
                   help=f"the fewest pairs that may produce a verdict (default {MIN_PAIRS}). A "
                        "sample too small to have a spread cannot certify one.")
    p.add_argument("--battle-timeout-s", type=float, default=None,
                   help="raise local_battle_runner's per-battle TOTAL livelock backstop (default "
                        "180 s, contention-scaled). A playoff cell legitimately spends tens of "
                        "seconds per decision, so a whole game overruns the default — and a "
                        "timed-out game does NOT merely cost one row, it POISONS THE REST OF THE "
                        "CELL (see _raise_battle_backstop). Set it on any arm whose per-decision "
                        "wall x decisions can approach 180 s.")
    p.add_argument("--battle-idle-s", type=float, default=None,
                   help="raise local_battle_runner's IDLE wedge detector (default 30 s between "
                        "protocol chunks). The playoff arm nests a whole rollout battle inside "
                        "one live decision, so the LIVE battle is legitimately silent for as long "
                        "as the per-decision budget — the bound's own premise ('the longest "
                        "plausible gap in a healthy battle') is what changes, not the safety "
                        "margin. Size it above --budget with headroom.")
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


def _raise_battle_backstop(total_s: Optional[float], idle_s: Optional[float]) -> None:
    """Raise the LIVE battle's two time bounds, loudly.

    ``local_battle_runner`` bounds a battle two ways: an IDLE gap between protocol chunks (30 s —
    the real wedge detector) and a TOTAL backstop against livelock (180 s). Both of them assume a
    battle whose decisions cost milliseconds, and the ``playoff`` arm breaks that assumption BY
    DESIGN: it nests whole rollout battles inside one live decision, so the live stream is silent
    for as long as the per-decision budget and a 25-decision game runs for many minutes.

    🚨 **A TIMED-OUT GAME DOES NOT COST ONE ROW — IT POISONS THE REST OF THE CELL.** Measured
    2026-08-24 while bringing this arm up: a first game killed by the 180 s backstop was followed
    by a second whose searches failed the world PREFIX GATE on **22 of 23** decisions, against
    **0 of 56** in a control run of the same battles that did not time out (and ~2.5% across the
    R-ladder's 3212 decisions). The mechanism is that ``_await_battle`` cancels the battle task,
    but the killed decision's search is running in a THREAD-POOL executor and cannot be cancelled:
    it keeps the warm ``SearchSession`` and goes on issuing ``open_root`` for the DEAD battle while
    the next game's searches issue theirs, and the two interleave on one stdin/stdout pair. The
    surviving symptom is a root whose replayed prefix belongs to another battle — counted, but
    counted as a determinization failure, which is the wrong diagnosis.

    So this is not a comfort setting: on a slow arm, an un-raised backstop silently converts most
    of the cell into policy fallbacks. It stays a CLI opt-in with a printed line rather than a new
    default, because the backstop is every other caller's safety net and quietly widening it is how
    a real livelock gets to run for an hour.
    """
    from utils.bridge import local_battle_runner as lbr

    if total_s:
        was = lbr._PER_BATTLE_TIMEOUT
        lbr._PER_BATTLE_TIMEOUT = float(total_s)
        print(f"[search_dividend] per-battle LIVELOCK backstop {was:g}s -> {total_s:g}s",
              flush=True)
    if idle_s:
        was_i = lbr._BATTLE_IDLE_BUDGET
        lbr._BATTLE_IDLE_BUDGET = float(idle_s)
        print(f"[search_dividend] per-battle IDLE wedge detector {was_i:g}s -> {idle_s:g}s "
              f"(a nested rollout silences the live stream for a whole decision)", flush=True)


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
    arms = args.arm or list(DEFAULT_ARMS)
    budgets = args.budget or list(DEFAULT_BUDGETS)
    results = ResultsFile(args.out)
    caps = WidthCaps(m_opp=args.max_opp, k_worlds=args.max_worlds, r_dice=args.max_dice)
    playoff_cfg = PlayoffConfig(rollouts=args.playoff_rollouts,
                                screen_margin=args.playoff_screen_margin,
                                se_multiple=args.playoff_se_k,
                                min_pairs=args.playoff_min_pairs,
                                impl=args.impl)
    if args.battle_timeout_s or args.battle_idle_s:
        _raise_battle_backstop(args.battle_timeout_s, args.battle_idle_s)

    print(f"[search_dividend] ckpt={ckpt}\n"
          f"  arms={arms} budgets={budgets} opponents={opponents} "
          f"games=[{args.games_start},{args.games_start + args.games})\n"
          f"  max_depth={args.max_depth} side_swap={'auto' if args.side_swap is None else args.side_swap}\n"
          f"  pool={len(pool)} teams  impl={args.impl}/search:{args.search_impl}  out={args.out}\n"
          f"  root_strategy={args.root_strategy}"
          + (f"  defensive: leaf={args.defensive_leaf} "
             f"wp_margin={args.defensive_wp_margin:g} confirm={args.defensive_confirm}"
             f"@{'race-residual' if args.defensive_confirm_deadline_s is None else f'{args.defensive_confirm_deadline_s:g}s'} "
             f"contested_deadline="
             f"{'budget' if args.defensive_contested_deadline_s is None else f'{args.defensive_contested_deadline_s:g}s'} "
             f"racing_rule={args.racing_rule} floor="
             f"{RacingConfig(rule=args.racing_rule, min_samples=args.racing_min_samples).effective_min_samples()}"
             if args.root_strategy == "defensive" else ""),
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
                               max_depth=args.max_depth,
                               root_strategy=args.root_strategy,
                               racing=RacingConfig(rule=args.racing_rule, z=args.racing_z,
                                                   delta=args.racing_delta,
                                                   min_samples=args.racing_min_samples),
                               defensive=DefensiveConfig(
                                   wp_margin=args.defensive_wp_margin,
                                   leaf=args.defensive_leaf,
                                   confirm_rollouts=args.defensive_confirm,
                                   confirm_deadline_s=args.defensive_confirm_deadline_s,
                                   contested_deadline_s=args.defensive_contested_deadline_s))
            for opp in opponents:
                cell = Cell(arm=arm, budget=budget, opponent=opp)
                n = asyncio.run(run_cell(
                    cell, model=model, mappings=mappings, cfg=cfg, games=args.games,
                    results=results, salt=args.games_seed, impl=args.impl,
                    pool_packed=pool, progress=progress,
                    side_swap=resolve_side_swap(args.side_swap, opp),
                    games_start=args.games_start,
                    playoff_cfg=playoff_cfg))
                total += n
    print(f"\n[search_dividend] played {total} new games -> {args.out}\n", flush=True)
    print(format_report(results.rows(), args.anchors))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
