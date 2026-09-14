"""Drive one head-to-head SESSION: our checkpoint (via `main.play`'s own builders)
accepting challenges from a Foul Play client on a local Showdown server.

It is deliberately a THIN wrapper over `main.play`: the teambuilder, the server
resolution and the model player all come from `main.play`, so this measures the
same client code path a ladder game uses. What it adds is per-battle FORENSICS,
which `play.py` (a session runner, not a logger) does not emit:

  * winner / turns / our decision count
  * whether OUR 250-turn forfeit fired (the training deadline, `MAX_TURNS`)
  * our per-decision think time
  * any UnknownMessageType / UnsupportedMessageType raised by our parser

Writes one JSON line per battle to --out.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
import traceback

sys.path.insert(0, "/home/goodlad/dev/gen3ai/src")

from agents.observation.constants import MAX_TURNS  # noqa: E402
from agents.training.stall import StallConfig  # noqa: E402
from main.play import (  # noqa: E402
    build_account,
    build_teambuilder,
    resolve_server,
)


def build_args(ns):
    """The argparse-shaped object `main.play.build_model_player` expects."""
    class A:
        pass
    a = A()
    a.model = ns.model
    a.device = "cpu"
    a.debug_obs = False
    a.format = ns.format
    a.concurrency = 1
    a.temperature = 0.0
    a.avatar = None
    a.proxy = None
    a.connect_timeout = 60.0
    # `main.play` grew a named `--forfeit-turn-limit` (default DEFAULT_FORFEIT_TURN_LIMIT,
    # read from the trainer's StallConfig) on 2026-09-14 in commit 127cf199. We take the
    # DEFAULT, which is the whole point: the head-to-head must use the trainer's number.
    from main.play import DEFAULT_FORFEIT_TURN_LIMIT
    a.forfeit_turn_limit = DEFAULT_FORFEIT_TURN_LIMIT
    return a


async def main(ns) -> int:
    from main.play import build_model_player

    server_config = resolve_server("local", ns.port)
    teambuilder = build_teambuilder(ns.team, False)
    account = build_account(ns.username, None, "local")
    player = build_model_player(build_args(ns), teambuilder, server_config, account)

    # --- forensics hooks -------------------------------------------------------
    records = []
    state = {"forfeit_fired": False, "think": [], "t_last": None, "parse_errors": []}

    orig_handle_stall = player._handle_stall
    def handle_stall(battle, suffix):
        out = orig_handle_stall(battle, suffix)
        if out is not None:
            state["forfeit_fired"] = True
        return out
    player._handle_stall = handle_stall

    orig_choose = player.choose_move
    def choose_move(battle):
        t0 = time.perf_counter()
        try:
            return orig_choose(battle)
        finally:
            state["think"].append(time.perf_counter() - t0)
    player.choose_move = choose_move

    orig_finished = player._battle_finished_callback
    def finished(battle):
        view = battle.strict_view()
        rec = {
            "battle_tag": view.battle_tag,
            "turns": view.turn,
            "won": battle.won,
            "our_forfeit_fired": state["forfeit_fired"],
            "our_decisions": len(state["think"]),
            "our_think_ms_mean": (1000 * sum(state["think"]) / len(state["think"]))
            if state["think"] else None,
            "our_think_ms_max": (1000 * max(state["think"])) if state["think"] else None,
            "team_file": os.path.basename(ns.team),
            "ts": time.time(),
        }
        records.append(rec)
        with open(ns.out, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print("[derisk] " + json.dumps(rec), flush=True)
        state["forfeit_fired"] = False
        state["think"] = []
        return orig_finished(battle)
    player._battle_finished_callback = finished

    print(f"[derisk] our forfeit deadline = MAX_TURNS = {MAX_TURNS} "
          f"(StallConfig.threshold={StallConfig().threshold}) — the SAME constant the trainer uses",
          flush=True)
    print(f"[derisk] accepting {ns.n_battles} {ns.format} from {ns.opponent!r} as "
          f"{player.username} on {server_config.websocket_url}", flush=True)

    try:
        await asyncio.wait_for(
            player.accept_challenges(ns.opponent, ns.n_battles), timeout=ns.session_timeout
        )
    except asyncio.TimeoutError:
        print(f"[derisk] SESSION TIMEOUT after {ns.session_timeout}s "
              f"({len(records)}/{ns.n_battles} battles finished)", flush=True)
    except Exception:
        print("[derisk] SESSION EXCEPTION:\n" + traceback.format_exc(), flush=True)

    won = sum(1 for r in records if r["won"])
    print(f"[derisk] finished={len(records)} won={won}", flush=True)
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--username", required=True)
    p.add_argument("--opponent", required=True)
    p.add_argument("--team", required=True)
    p.add_argument("--n-battles", type=int, default=10)
    p.add_argument("--format", default="gen3ou")
    p.add_argument("--out", required=True)
    p.add_argument("--session-timeout", type=float, default=3600.0)
    logging.basicConfig(level=logging.WARNING)
    sys.exit(asyncio.run(main(p.parse_args())))
