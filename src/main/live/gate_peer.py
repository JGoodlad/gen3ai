"""The OTHER side of P4's gate (c): a scripted gen-3 BOT as a Rust-stack websocket client in its own process,
accepting our challenges (``main.live.master_series`` starts it and waits for ``[peer] READY``).

The bot is the RUST port of the roster bot (``src/rust_env/src/bots/``, gated action-equal to the Python bots per
decision) deciding on its own side's reading of the battle (:class:`main.live.bot_reader.BotReader` — the same
``pokesim::side_reader::SideReader`` chain the live client reads with), so nothing here imports poke-env. Its random
streams follow the env core's declared rule (``opponents::stream_seed(seed, env=0, k)``) and run across the peer's
battles (one bot per process, ``--seed``).

History (P6 of the poke-env retirement, 2026-10-08): this peer used to be a poke-env client — ``--role bot`` a Python
``agents.opponents`` player, ``--role shadow`` gate (d)'s legacy ``RLPlayer`` with the Rust reader shadowing it. Gate
(d) is BANKED (16,523 shadow decisions, 0 differences, ``measurements/pokeenv_p4_live_2026-10-07/``) and its legacy
client is deleted, so the shadow role is gone; the bot role moved onto the Rust port.
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from typing import List, Optional

#: The roster bots a peer can play, by the env core's ``bots::Kind`` name (``main.live.bot_reader``). The seven
#: P4's gate (c) ran (``Gen3HeuristicV2Player`` → ``heuristic2``, …; ``random`` and ``heuristic`` are playable too).
BOTS = ("heuristic2", "staller_v2", "aggressive_v2", "setup_sweep_v2", "staller", "aggressive", "setup_sweep")
#: Every bot the Rust core plays (``bots::Kind::name``).
ALL_BOTS = BOTS + ("random", "heuristic")


async def run(a: argparse.Namespace) -> int:
    from main.live.bot_reader import BotPolicy, BotReader
    from main.live.client import ClientConfig, LiveClient
    from main.play import STAR_TSS_TEAM, resolve_uri
    from utils.teambuilder import Gen3Teambuilder

    uri = resolve_uri("local", a.port)
    if a.team_pool:
        import random

        from utils.team_sources import packed_teams
        teams = packed_teams("pool")
        rng = random.Random(a.seed)

        def team_fn() -> Optional[str]:
            return rng.choice(teams)
    else:
        team_fn = Gen3Teambuilder(STAR_TSS_TEAM).yield_team
    # a bot never stall-forfeits (the Python bots carried no stall check; nor does the env core's bot route)
    cfg = ClientConfig(uri=uri, username=a.username, battle_format=a.format, auth="local",
                       forfeit_turn_limit=10 ** 9)
    client = LiveClient(cfg, policy=BotPolicy(), team_fn=team_fn,
                        reader_factory=lambda: BotReader(a.bot, seed=a.seed, env=0))
    try:
        await client.connect()
        print(f"[peer] READY {a.username}", flush=True)
        res = await client.accept(a.opponent, a.n)
    finally:
        await client.close()
    print(f"[peer] DONE finished={len(res)} won={sum(1 for r in res if r.won)}", flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m main.live.gate_peer")
    ap.add_argument("--bot", choices=ALL_BOTS, default="heuristic2")
    ap.add_argument("--seed", type=int, default=0, help="the bot's route seed (its streams: stream_seed(seed, 0, k))")
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--username", required=True)
    ap.add_argument("--opponent", required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--format", default="gen3ou")
    ap.add_argument("--team-pool", action="store_true")
    return ap


def main(argv: Optional[List[str]] = None) -> int:
    return asyncio.run(run(build_parser().parse_args(argv)))


if __name__ == "__main__":
    sys.exit(main())
