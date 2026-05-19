"""
E2E test for BattleSpectator.

Requires a live Showdown server on localhost:8000 with active battles.
(The training run provides these — no need to spawn extra bots.)
Run directly:
    export PYTHONPATH=$PYTHONPATH:src
    python src/poke_env/spectator/spectator_e2e_test.py

Tests:
  Part 1 — spectator captures one completed battle from the server.
  Part 2 — round-trip: feed the saved log back into poke-env's Battle parser
            and assert the parsed state is valid.
"""

import asyncio
import logging
import sys

from poke_env.battle.battle import Battle
from poke_env.concurrency import POKE_LOOP
from poke_env.ps_client.server_configuration import LocalhostServerConfiguration
from poke_env.spectator import BattleSpectator
from poke_env.spectator.spectated_battle import SpectatedBattle


FORMAT = "gen3ou"
TIMEOUT = 120  # seconds to wait for one completed battle


async def _collect_one(spectator: BattleSpectator) -> SpectatedBattle:
    async for battle in spectator.watch(FORMAT):
        return battle


def _parse_log_with_poke_env(battle: SpectatedBattle) -> Battle:
    """Feed battle.log_text back through poke-env's Battle parser."""
    lines = battle.log_text.splitlines()

    p1_name = "unknown"
    for line in lines:
        parts = line.split("|")
        if len(parts) >= 4 and parts[1] == "player" and parts[2] == "p1":
            p1_name = parts[3]
            break

    parsed = Battle(
        battle_tag=battle.battle_tag,
        username=p1_name,
        logger=logging.getLogger("round_trip"),
        gen=3,
    )
    for line in lines:
        if not line:
            continue
        parts = line.split("|")
        if len(parts) < 2:
            continue
        # win/tie are dispatched by Player directly, not via parse_message
        if parts[1] == "win":
            parsed.won_by(parts[2] if len(parts) > 2 else "unknown")
        elif parts[1] == "tie":
            parsed.tied()
        else:
            parsed.parse_message(parts)

    return parsed


def main() -> None:
    print(f"=== BattleSpectator E2E Test ({FORMAT} on localhost:8000) ===")

    spectator = BattleSpectator(
        server_configuration=LocalhostServerConfiguration,
        join_interval=2.0,
        poll_interval=5.0,
    )

    # --- Part 1 ---
    print("Collecting one completed battle ...")
    battle = asyncio.run_coroutine_threadsafe(
        _collect_one(spectator), POKE_LOOP
    ).result(timeout=TIMEOUT)

    assert battle.finished, "battle.finished must be True"
    assert "|turn|" in battle.log_text, "log must contain turn markers"
    assert "|win|" in battle.log_text or "|tie" in battle.log_text, \
        "log must end with |win| or |tie"
    if battle.winner is not None:
        assert f"|win|{battle.winner}" in battle.log_text
    else:
        assert "|tie" in battle.log_text

    print(f"[PASS] Part 1 — {battle.battle_tag}, winner={battle.winner or 'tie'}, "
          f"{len(battle.log_text.splitlines())} log lines")

    # --- Part 2: round-trip ---
    print("Round-tripping log through poke-env parser ...")
    parsed = _parse_log_with_poke_env(battle)

    assert parsed.finished, "parsed.finished must be True"
    assert parsed.turn > 0, f"expected turn > 0, got {parsed.turn}"
    assert len(parsed.team) > 0, "expected our team to be populated"
    assert len(parsed.opponent_team) > 0, "expected opponent team to be populated"

    print(f"[PASS] Part 2 — {parsed.turn} turns, "
          f"{len(parsed.team)} our mons, {len(parsed.opponent_team)} opp mons")
    print("=== ALL TESTS PASSED ===")


if __name__ == "__main__":
    main()
