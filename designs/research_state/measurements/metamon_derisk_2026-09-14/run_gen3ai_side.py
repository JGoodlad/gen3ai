#!/usr/bin/env python3
"""Run OUR checkpoint as the other client, through ``src/main/play.py``'s own code path.

This is ``play.py`` plus an OBSERVER: it calls ``main.play``'s parser and ``main()`` unchanged, so
the client, the teambuilder, the forfeit limit and the connect guard are all exactly what a ladder
session uses. The observer adds two things a measurement needs and ``play.py`` does not print:

* a per-battle JSONL row (winner, turn count, whether OUR forfeit limit fired, the battle tag,
  the team drawn), written as each battle finishes rather than at the end, so a crash mid-series
  still leaves the games that completed; and
* per-decision inference wall time, sampled around ``RLPlayer._predict_best_action``.

"Did the forfeit limit fire" is read from the TURN COUNT against ``--forfeit-turn-limit``, the same
way ``agents.training.trace_result`` classifies a training timeout: a cap forfeit is reported by
the server as an ordinary loss, so the flags alone cannot distinguish it from a decisive one.

Run from the MAIN checkout (it owns ``models/``):

    export PYTHONPATH=$PYTHONPATH:src
    python3 run_gen3ai_side.py --mode challenge --port 9217 --opponent MetaSmallRL \
        --username Gen3AIv12 --model models/<run>/final_model.zip --team-pool \
        --n-battles 60 --games-out games.jsonl
"""

import argparse
import asyncio
import json
import os
import sys
import time


def _install_observer(games_path: str, forfeit_limit: int, tag: str, timings: list):
    """Patch ``RLPlayer`` so every finished battle writes one JSONL row."""
    from agents.inference.player import RLPlayer

    seen = set()
    f = open(games_path, "a", buffering=1)

    inner_finish = RLPlayer._battle_finished_callback
    inner_predict = RLPlayer._predict_best_action

    def finished(self, battle):
        view = battle.strict_view()
        battle_tag = view.battle_tag
        turns = view.turn
        if battle_tag not in seen:
            seen.add(battle_tag)
            won = getattr(battle, "won", None)
            row = {
                "series": tag,
                "battle_tag": battle_tag,
                "t": time.time(),
                "turns": turns,
                # poke-env's own three flags. `won is None` with `finished` true is a TIE.
                "won": won,
                "lost": getattr(battle, "lost", None),
                "finished": getattr(battle, "finished", None),
                # The cap forfeit is reported as an ordinary loss; only the turn count separates
                # it from a decisive one (agents.training.trace_result, same rule).
                "hit_forfeit_limit": turns >= forfeit_limit,
                "forfeit_limit": forfeit_limit,
                "n_decisions": getattr(self, "_n_decisions", None),
                "n_defaults": getattr(self, "_n_defaults", None),
                "n_redecides": getattr(self, "_n_redecides", None),
            }
            f.write(json.dumps(row) + "\n")
            print(f"[gen3ai-side] GAME {len(seen)}: won={won} turns={turns} "
                  f"cap={row['hit_forfeit_limit']}", flush=True)
        return inner_finish(self, battle)

    def predict(self, *a, **k):
        t0 = time.perf_counter()
        try:
            return inner_predict(self, *a, **k)
        finally:
            timings.append(time.perf_counter() - t0)

    RLPlayer._battle_finished_callback = finished
    RLPlayer._predict_best_action = predict
    return f


def main(argv=None) -> int:
    import main.play as play

    parser = play.build_parser()
    parser.add_argument("--games-out", required=True, help="append one JSONL row per battle")
    parser.add_argument("--series", default="gen3ai", help="label written into every row")
    parser.add_argument("--timing-out", default=None, help="write inference timing JSON here")
    args = parser.parse_args(argv)

    timings: list = []
    handle = _install_observer(args.games_out, args.forfeit_turn_limit, args.series, timings)
    t0 = time.time()
    try:
        rc = asyncio.run(play.main(args))
    finally:
        handle.close()
        elapsed = time.time() - t0
        if timings:
            s = sorted(timings)
            n = len(s)
            timing = {"n_decisions": n, "mean_s": sum(s) / n, "median_s": s[n // 2],
                      "p90_s": s[int(0.9 * n)], "max_s": s[-1], "wall_s": elapsed,
                      "model": args.model, "device": args.device, "series": args.series}
            print("[gen3ai-side] TIMING " + json.dumps(timing), flush=True)
            if args.timing_out:
                with open(args.timing_out, "w") as tf:
                    json.dump(timing, tf, indent=1)
    return rc


if __name__ == "__main__":
    sys.exit(main())
