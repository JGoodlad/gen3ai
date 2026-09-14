#!/usr/bin/env python3
"""Run OUR checkpoint as the other client, through ``src/main/play.py``'s own code path.

The 2026-09-14 de-risk's observer, plus the two things a matched-regime 2x2 needs:

* ``--team-dir`` — draw OUR side from a DIRECTORY of Showdown-export team files (Metamon's own
  ``competitive`` gen3ou set, for the away cells) instead of our 719-team pool. The files still go
  through ``Gen3Teambuilder``, so local validation and ``fix_gen3_hp_ivs`` are applied exactly as
  they are to our own pool — the away cell changes the team DISTRIBUTION, not the pipeline.
* ``--team-seed`` — a private draw RNG (``gen3_team_draw_rng_v1``'s supported ``rng_seed=`` seam),
  with the team actually yielded for each game recorded in order.

It calls ``main.play``'s parser and ``main()`` unchanged, so the client, the forfeit limit and the
connect guard are exactly what a ladder session uses. The observer adds what a measurement needs
and ``play.py`` does not print:

* one JSONL row per battle (winner, turn count, whether OUR forfeit limit fired, the battle tag,
  the team drawn), written as each battle finishes, so a crash mid-cell keeps what completed;
* per-decision inference wall time around ``RLPlayer._predict_best_action``; and
* **the regime our player actually used** — the ``stochastic`` keyword each decision really
  received, not the flag we believe we passed.

"Did the forfeit limit fire" is read from the TURN COUNT against ``--forfeit-turn-limit``, the same
way ``agents.training.trace_result`` classifies a training timeout: a cap forfeit is reported by
the server as an ordinary loss, so the flags alone cannot distinguish it from a decisive one.

Run from the MAIN checkout (it owns ``models/``), with PYTHONPATH pointed at the worktree's src.
"""

import argparse
import asyncio
import glob
import json
import os
import sys
import time


def _install_team_source(team_dir, seed, draw_log):
    """Patch ``play.build_teambuilder`` so our side draws from ``team_dir`` and/or a seeded RNG.

    Returns nothing; the patch is installed on the module. When ``team_dir`` is None the pool /
    single-team behaviour of ``play.py`` is preserved and only the seed is added.
    """
    import main.play as play
    from utils.teambuilder import Gen3Teambuilder

    def build_teambuilder(team_file, pool):
        if team_dir:
            files = sorted(glob.glob(os.path.join(team_dir, "*_team")))
            if not files:
                raise SystemExit(f"--team-dir {team_dir} holds no *_team files")
            # 🚨 .strip() IS LOAD-BEARING, not tidiness. Our VENDORED poke-env fork's
            # `Teambuilder.parse_showdown_team` splits on "\n\n" and skips a chunk only when it
            # is EXACTLY "", so a file ending in a blank line (every Metamon `competitive` file
            # does) yields an EMPTY 7th Pokemon. Showdown then rejects the team with
            # `You may only bring up to 6 Pokemon (your team has 7)` and the challenge never
            # becomes a battle: the series HANGS, exactly as de-risk hazard H1 did. Upstream
            # poke-env 0.8.3.3 parses the same file as 6 — this is our fork's defect, and it is
            # invisible to `validate_teams_locally`, which validates the TEXT, not the pack.
            texts = [open(f).read().strip() for f in files]
            tb = Gen3Teambuilder(texts, rng_seed=seed)
            print(f"[gen3ai-side] team-dir {team_dir}: {len(files)} files, "
                  f"{len(tb.packed_teams)} locally valid, seed={seed}", flush=True)
            # Label each pool index by its source filename, so a draw is reportable as a NAME.
            # Gen3Teambuilder drops locally-invalid teams, so the labels must follow the
            # surviving indices rather than the original file order.
            labels, j = [], 0
            from utils.bridge.team_validator import validate_teams_locally
            for f, v in zip(files, validate_teams_locally("gen3ou", texts)):
                if v.get("valid"):
                    labels.append(os.path.basename(f))
                    j += 1
            tb._label_by_idx = labels
        elif team_file:
            with open(team_file) as f:
                tb = Gen3Teambuilder(f.read().strip(), rng_seed=seed)
            tb._label_by_idx = [os.path.basename(team_file)]
        elif pool:
            from utils.team_loader import TeamLoader
            tb = Gen3Teambuilder([t.strip() for t in TeamLoader().get_all_teams()],
                                 rng_seed=seed)
            # Our pool's own fingerprints ARE the de-risk export's filenames (`<sha>.gen3ou_team`),
            # so the two sides' team identifiers join without a translation table.
            tb._label_by_idx = list(tb.get_pool_team_keys())
            print(f"[gen3ai-side] pool: {len(tb.packed_teams)} teams, seed={seed}", flush=True)
        else:
            tb = Gen3Teambuilder(play.STAR_TSS_TEAM, rng_seed=seed)
            tb._label_by_idx = ["STAR_TSS_TEAM"]

        # A THROWING GUARD at the construction site, because the failure it catches is a HANG.
        # A packed team Showdown rejects never becomes a battle, so the series sits there with no
        # error to read — the de-risk's H1 shape. `validate_teams_locally` cannot catch it: it
        # validates the paste, and this defect is created by the PACK.
        bad = [i for i, packed in enumerate(tb.packed_teams) if len(packed.split("]")) != 6]
        if bad:
            names = getattr(tb, "_label_by_idx", [])
            raise SystemExit(
                f"{len(bad)} packed team(s) do not have exactly 6 Pokemon — Showdown will "
                f"REJECT them and the series will HANG, not error. First offenders: "
                + ", ".join(str(names[i]) if i < len(names) else str(i) for i in bad[:5]))

        inner_yield = tb.yield_team

        def yield_team():
            packed = inner_yield()
            idx = tb._last_pool_idx
            labels = getattr(tb, "_label_by_idx", [])
            draw_log.append(labels[idx] if (idx is not None and idx < len(labels)) else None)
            return packed

        tb.yield_team = yield_team
        return tb

    play.build_teambuilder = build_teambuilder


def _install_observer(games_path, forfeit_limit, tag, timings, regime_kwargs, draw_log):
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
                # The team OUR side drew for this battle, by the same name the other side's log
                # uses. Draws are appended at yield time and battles finish in order at
                # --concurrency 1, so the n-th draw belongs to the n-th finished battle.
                "our_team": (draw_log[len(seen) - 1] if len(seen) <= len(draw_log) else None),
                "our_team_draws_so_far": len(draw_log),
                "our_battles_so_far": len(seen),
                "our_stochastic": self._stochastic,
                "our_temperature": self._temperature,
            }
            f.write(json.dumps(row) + "\n")
            print(f"[gen3ai-side] GAME {len(seen)}: won={won} turns={turns} "
                  f"cap={row['hit_forfeit_limit']} team={row['our_team']}", flush=True)
        return inner_finish(self, battle)

    def predict(self, *a, **k):
        # The regime VERIFICATION for our half: record the `stochastic` keyword the decision
        # really received, rather than trusting the flag we think we passed.
        # Signature is `_predict_best_action(self, battle, stochastic=False, ...)`, so a
        # positional `stochastic` would be a[1]; RLPlayer.choose_move passes it by keyword.
        regime_kwargs.add(bool(k["stochastic"] if "stochastic" in k
                               else (a[1] if len(a) > 1 else False)))
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
    parser.add_argument("--team-dir", default=None,
                        help="draw from this directory of *_team files instead of --team-pool")
    parser.add_argument("--team-seed", type=int, default=None, help="seed the team draw")
    parser.add_argument("--draws-out", default=None, help="write the ordered team draws here")
    args = parser.parse_args(argv)

    draw_log: list = []
    _install_team_source(args.team_dir, args.team_seed, draw_log)

    timings: list = []
    regime_kwargs: set = set()
    handle = _install_observer(args.games_out, args.forfeit_turn_limit, args.series,
                               timings, regime_kwargs, draw_log)
    t0 = time.time()
    try:
        rc = asyncio.run(play.main(args))
    finally:
        handle.close()
        elapsed = time.time() - t0
        timing = {"n_decisions": len(timings), "wall_s": elapsed, "model": args.model,
                  "device": args.device, "series": args.series,
                  "stochastic_kwarg_values": sorted(regime_kwargs),
                  "temperature_flag": args.temperature,
                  "team_dir": args.team_dir, "team_seed": args.team_seed,
                  "mode": args.mode, "n_team_draws": len(draw_log)}
        if timings:
            s = sorted(timings)
            n = len(s)
            timing |= {"mean_s": sum(s) / n, "median_s": s[n // 2],
                       "p90_s": s[int(0.9 * n)], "max_s": s[-1]}
        expect = [args.temperature > 0.0]
        timing["regime_check_ok"] = timing["stochastic_kwarg_values"] == expect
        print("[gen3ai-side] TIMING " + json.dumps(timing), flush=True)
        print(f"[gen3ai-side] REGIME CHECK temperature={args.temperature} "
              f"stochastic_kwargs={timing['stochastic_kwarg_values']} "
              f"(ok={timing['regime_check_ok']})", flush=True)
        if args.timing_out:
            with open(args.timing_out, "w") as tf:
                json.dump(timing, tf, indent=1)
        if args.draws_out:
            with open(args.draws_out, "w") as df:
                json.dump(draw_log, df)
    return rc


if __name__ == "__main__":
    sys.exit(main())
