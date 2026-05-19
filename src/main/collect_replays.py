"""
Daemon script that spectates Pokémon Showdown battles and saves raw replay logs.

Runs until Ctrl+C. Each battle is saved immediately when it finishes.
Restarts are safe — already-saved files are not overwritten.

Usage:
    python src/main/collect_replays.py --format gen3ou --save-dir replays/gen3ou
    python src/main/collect_replays.py --format gen3ou --save-dir replays/gen3ou --local

The script runs the spectator on poke-env's background POKE_LOOP rather than
asyncio.run(), because PSClient lives on that loop and all callbacks execute there.
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from poke_env.concurrency import POKE_LOOP
from poke_env.ps_client.server_configuration import (
    LocalhostServerConfiguration,
    ShowdownServerConfiguration,
)
from poke_env.spectator import BattleSpectator


async def _run(format_id: str, save_dir: Path, server_config, max_concurrent: int) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    spectator = BattleSpectator(server_configuration=server_config, max_concurrent=max_concurrent)
    count = 0
    async for battle in spectator.watch(format_id):
        path = save_dir / f"{battle.battle_tag}.log"
        if not path.exists():
            path.write_text(battle.log_text, encoding="utf-8")
        count += 1
        print(f"[{count}] {battle.battle_tag}  winner={battle.winner or 'tie'}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect Showdown battle replays as a daemon."
    )
    parser.add_argument("--format", default="gen3ou", help="Showdown format ID")
    parser.add_argument("--save-dir", default="replays", help="Directory for .log files")
    parser.add_argument("--local", action="store_true", help="Use localhost:8000")
    parser.add_argument("--max-concurrent", type=int, default=10, help="Max simultaneous rooms")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)

    server = LocalhostServerConfiguration if args.local else ShowdownServerConfiguration
    save_dir = Path(args.save_dir)

    print(f"Collecting {args.format} replays → {save_dir}/  (max-concurrent={args.max_concurrent}, Ctrl+C to stop)")
    future = asyncio.run_coroutine_threadsafe(
        _run(args.format, save_dir, server, args.max_concurrent), POKE_LOOP
    )
    try:
        future.result()
    except KeyboardInterrupt:
        future.cancel()
        print(f"\nStopped. Replays saved to {save_dir}/", file=sys.stderr)


if __name__ == "__main__":
    main()
