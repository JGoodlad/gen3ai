"""
Snatch parsing fuzz test (real battles via the local BattleStream bridge).

Snatch (Gen 3, priority +4) steals the target's self-targeting status move and
makes the *snatcher* execute it. Showdown emits that as

    |move|p2a: Blissey|Calm Mind|p2a: Blissey|[from] Snatch

i.e. the snatcher is shown "using" a move it does not own. poke-env did not
recognise the `[from] Snatch` tag, so it (a) logged a per-occurrence
"Unmanaged move message format received" warning and (b) added the stolen move
to the SNATCHER's revealed moveset (`reveal=True`) — which then leaked into the
observation's opponent move slots. The fix treats Snatch like Magic Coat /
Mirror Move (`use=False, reveal=False`, tag stripped) in
`poke_env/battle/abstract_battle.py`.

This test exercises that path end to end against the actual Showdown protocol
stream. Two forcing players play real `gen3ou` battles:

  * the SNATCHER team leads with Snatch users (Gengar / Blissey / Crobat / …)
    and presses **Snatch** on setup turns, attacking otherwise;
  * the VICTIM team presses a **snatchable setup move** (Calm Mind / Dragon
    Dance / Agility — none of which appear on any snatcher mon) on setup turns,
    attacking otherwise.

On every setup turn the snatcher's +4 Snatch resolves first, then the victim's
setup move is snatched → a `[from] Snatch` move line. Both sides attack on the
other turns so the battle actually terminates.

Validation, against the raw protocol:
  1. NO exception — battles complete. The bridge propagates a raise out of
     `parse_message`, so an unhandled `[from] Snatch` (or any newly-unhandled
     move message under the now-strict parser) would crash the run. A clean run
     proves Snatch is handled AND the strict "raise on unhandled" change does
     not mis-fire on normal play.
  2. CORRECT attribution — for each observed snatch, the stolen move is NOT in
     the snatcher's revealed moveset (`battle.get_pokemon(snatcher).moves`).
     Checked from the opponent's perspective (the realistic, reveal-gated view
     the obs encoder reads). With the old `reveal=True` bug this fails.
  3. COVERAGE — the run FAILS if it never observed a Snatch, a green run with
     zero coverage would validate nothing.

Run directly (no server needed; in-process via the local BattleStream bridge):
    python src/agents/training/poke_env_gaps/snatch_fuzz_test.py [n_battles]
    (in a linked worktree, first: export PYTHONPATH=$PYTHONPATH:src)
"""

import asyncio
import sys
import time
import traceback
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from poke_env import AccountConfiguration
from poke_env.data.normalize import to_id_str
from poke_env.player.player import Player
from poke_env.ps_client.server_configuration import LocalhostServerConfiguration

from utils.teambuilder import Gen3Teambuilder
from utils.bridge.local_battle_runner import run_local_battles

BATTLE_FORMAT = "gen3ou"

# Snatchable setup moves the victim presses — deliberately disjoint from every
# move on the SNATCHER team, so "stolen move appears on the snatcher" can only
# be the reveal bug, never a genuine shared move.
SNATCHABLE_SETUP = {
    "calmmind",
    "dragondance",
    "swordsdance",
    "agility",
    "bulkup",
    "amnesia",
    "barrier",
}

# ---------------------------------------------------------------------------
# Teams (validated gen3ou via utils.bridge.team_validator)
# ---------------------------------------------------------------------------

SNATCHER_TEAM = """\
Gengar @ Leftovers
Ability: Levitate
EVs: 4 HP / 252 SpA / 252 Spe
Timid Nature
- Snatch
- Thunderbolt
- Ice Punch
- Shadow Ball

Blissey @ Leftovers
Ability: Natural Cure
EVs: 252 HP / 4 Def / 252 SpD
Calm Nature
- Snatch
- Seismic Toss
- Ice Beam
- Thunderbolt

Crobat @ Leftovers
Ability: Inner Focus
EVs: 4 HP / 252 Atk / 252 Spe
Jolly Nature
- Snatch
- Sludge Bomb
- Aerial Ace
- Shadow Ball

Houndoom @ Leftovers
Ability: Flash Fire
EVs: 4 HP / 252 SpA / 252 Spe
Timid Nature
- Snatch
- Flamethrower
- Crunch
- Shadow Ball

Dusclops @ Leftovers
Ability: Pressure
EVs: 252 HP / 4 Atk / 252 SpD
Careful Nature
- Snatch
- Shadow Ball
- Ice Beam
- Seismic Toss

Sableye @ Leftovers
Ability: Keen Eye
EVs: 252 HP / 4 Atk / 252 SpD
Careful Nature
- Snatch
- Shadow Ball
- Knock Off
- Seismic Toss
"""

VICTIM_TEAM = """\
Suicune @ Leftovers
Ability: Pressure
EVs: 252 HP / 252 Def / 4 SpD
Bold Nature
- Calm Mind
- Surf
- Ice Beam
- Rest

Raikou @ Leftovers
Ability: Pressure
EVs: 4 HP / 252 SpA / 252 Spe
Timid Nature
- Calm Mind
- Thunderbolt
- Hidden Power [Ice]
- Rest

Jirachi @ Leftovers
Ability: Serene Grace
EVs: 252 HP / 4 SpA / 252 SpD
Careful Nature
- Calm Mind
- Psychic
- Thunderbolt
- Wish

Salamence @ Leftovers
Ability: Intimidate
EVs: 4 HP / 252 Atk / 252 Spe
Adamant Nature
- Dragon Dance
- Earthquake
- Rock Slide
- Dragon Claw

Tyranitar @ Leftovers
Ability: Sand Stream
EVs: 4 HP / 252 Atk / 252 Spe
Adamant Nature
- Dragon Dance
- Rock Slide
- Earthquake
- Crunch

Metagross @ Leftovers
Ability: Clear Body
EVs: 4 HP / 252 Atk / 252 Spe
Adamant Nature
- Agility
- Meteor Mash
- Earthquake
- Psychic
"""


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

@dataclass
class SnatchStats:
    snatch_seen: int = 0          # `[from] Snatch` move lines observed
    attribution_ok: int = 0       # snatch where the stolen move was NOT on the snatcher
    attribution_bad: int = 0      # snatch where the stolen move LEAKED onto the snatcher
    stolen_moves: set = field(default_factory=set)
    examples: List[dict] = field(default_factory=list)

    def record_bad(self, detail: dict):
        if len(self.examples) < 20:
            self.examples.append(detail)


def _is_snatch_line(msg: List[str]) -> Optional[Tuple[str, str]]:
    """If ``msg`` is a snatched-move line, return (snatcher_ident, stolen_move)."""
    if len(msg) >= 5 and msg[1] == "move" and msg[-1] in ("[from] Snatch", "[from]Snatch"):
        return msg[2], msg[3]
    return None


# ---------------------------------------------------------------------------
# Forcing player
# ---------------------------------------------------------------------------

class SnatchFuzzPlayer(Player):
    """Forces Snatch (snatcher) / a snatchable setup move (victim) on setup
    turns and attacks otherwise, while validating the parse of every observed
    `[from] Snatch` line against the raw protocol."""

    def __init__(self, role: str, stats: SnatchStats, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert role in ("snatcher", "victim")
        self.role = role
        self.stats = stats

    async def _handle_battle_message(self, split_messages):
        battle_tag = split_messages[0][0].lstrip(">")
        snatches = [
            hit for msg in split_messages[1:] if len(msg) >= 2
            for hit in (_is_snatch_line(msg),) if hit is not None
        ]

        # Let poke-env parse the chunk (applies the Snatch handler under test).
        await super()._handle_battle_message(split_messages)

        if not snatches:
            return
        battle = self._battles.get(battle_tag)
        if battle is None or battle.player_role is None:
            return
        for snatcher_ident, stolen_move in snatches:
            # Validate from the OPPONENT's perspective — the realistic,
            # reveal-gated view the obs encoder consumes. (Both players see the
            # line; we assert on whichever side observes the snatcher as the foe
            # so we never double-count the same physical event.)
            if snatcher_ident[:2] == battle.player_role:
                continue
            self.stats.snatch_seen += 1
            stolen_id = to_id_str(stolen_move)
            self.stats.stolen_moves.add(stolen_id)
            snatcher = battle.get_pokemon(snatcher_ident)
            if stolen_id in snatcher.moves:
                self.stats.attribution_bad += 1
                self.stats.record_bad({
                    "battle": battle_tag,
                    "turn": battle.turn,
                    "snatcher": snatcher_ident,
                    "stolen_move": stolen_id,
                    "revealed_moves": sorted(snatcher.moves.keys()),
                })
            else:
                self.stats.attribution_ok += 1

    def choose_move(self, battle):
        if battle.available_moves:
            setup = next((m for m in battle.available_moves
                          if m.id in SNATCHABLE_SETUP), None)
            snatch = next((m for m in battle.available_moves if m.id == "snatch"), None)
            attack = max((m for m in battle.available_moves if m.base_power > 0),
                         key=lambda m: m.base_power, default=None)
            setup_turn = battle.turn % 2 == 1  # alternate: snatch/setup vs attack
            if self.role == "snatcher":
                choice = snatch if (setup_turn and snatch) else attack
            else:
                choice = setup if (setup_turn and setup) else attack
            return self.create_order(choice or battle.available_moves[0])
        if battle.available_switches:
            return self.create_order(battle.available_switches[0])
        return self.choose_default_move()


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

async def main(n_battles: int = 20) -> None:
    ts = int(time.time()) % 100000
    print(f"Snatch Fuzz Test — gen3ou — {n_battles} battles", flush=True)

    stats = SnatchStats()
    snatcher = SnatchFuzzPlayer(
        role="snatcher",
        stats=stats,
        battle_format=BATTLE_FORMAT,
        team=Gen3Teambuilder(SNATCHER_TEAM),
        server_configuration=LocalhostServerConfiguration, start_listening=False,
        account_configuration=AccountConfiguration(f"Snf{ts}", "password"),
        max_concurrent_battles=5,
    )
    victim = SnatchFuzzPlayer(
        role="victim",
        stats=stats,
        battle_format=BATTLE_FORMAT,
        team=Gen3Teambuilder(VICTIM_TEAM),
        server_configuration=LocalhostServerConfiguration, start_listening=False,
        account_configuration=AccountConfiguration(f"Snv{ts}", "password"),
        max_concurrent_battles=5,
    )

    await run_local_battles(snatcher, victim, n_battles)

    print(f"\n{'=' * 65}")
    print(f"Snatch move lines observed : {stats.snatch_seen}")
    print(f"  correct attribution      : {stats.attribution_ok}")
    print(f"  LEAKED onto snatcher      : {stats.attribution_bad}")
    print(f"Distinct stolen moves      : {sorted(stats.stolen_moves)}")

    failures = []
    if stats.attribution_bad > 0:
        failures.append(f"{stats.attribution_bad} snatched move(s) leaked onto the "
                        f"snatcher's revealed moveset")
        for ex in stats.examples:
            print(f"  LEAK: {ex}")
    if stats.snatch_seen == 0:
        failures.append("no Snatch was ever observed — coverage gap, nothing validated")

    print("=" * 65)
    if failures:
        print("FAIL:")
        for f in failures:
            print(f"  {f}")
        sys.exit(1)
    print("PASS — Snatch handled: no parse crash, stolen move never attributed "
          "to the snatcher, coverage satisfied.")
    print("=" * 65)


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    try:
        asyncio.run(main(n))
    except Exception:
        traceback.print_exc()
        sys.exit(1)
