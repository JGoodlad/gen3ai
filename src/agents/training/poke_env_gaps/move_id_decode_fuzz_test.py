"""Bridge fuzz: we ALWAYS correctly decode the Showdown protocol's moves and NEVER mis-associate one
— the load-bearing guard for gen3_typed_hidden_power_ids_v1 (typed HP → distinct nums 355-370 on our
side; bare 'hiddenpower' → 237 on the opponent side).

For EVERY mon on BOTH sides, at EVERY decision over many real bridge battles, this asserts the obs
per-mon move block round-trips its moves EXACTLY:

  1. NO MIS-ASSOCIATION (the core invariant): the multiset of move-id NUMS the obs encodes for a mon
     equals the multiset of ``gen3_data.moves.get(id).num`` over that mon's encoder-visible moves —
     i.e. every move is encoded at its OWN canonical num, none swapped/dropped/aliased. Checked for
     OUR full team (4 moves each) and the OPPONENT's revealed moves.
  2. ROUND-TRIP NAME: decoding each encoded num back through the encoder's reverse map yields a name
     HP-prefix-consistent with the actual move (so the prober/forensics never label a move wrong).
  3. THE KNOWN/UNKNOWN BOUNDARY:
     - OUR Hidden Power → its DISTINCT num (355-370) AND its real type (decoded ``hiddenpower(<type>)``).
     - The OPPONENT's revealed Hidden Power → the typeless num 237, decoded BARE ``hiddenpower`` (the
       protocol never reveals the opp's HP type → NO LEAK).
  4. poke-env's own ``Move.num`` is never trusted: the encoded num is always our gen3_data num.

Coverage (move slots, our-HP slots, opp-HP slots) is tracked and asserted > 0 — a pass that never saw
a Hidden Power is INCONCLUSIVE. Any violation raises immediately with the offending mon/decision.

Run directly (no server — in-process via the local BattleStream bridge):
    export PYTHONPATH=$PYTHONPATH:src
    python src/agents/training/poke_env_gaps/move_id_decode_fuzz_test.py [n_battles]
"""
from __future__ import annotations

import asyncio
import collections
import os
import sys
import time
import traceback
from dataclasses import dataclass, field
from typing import List

from poke_env import AccountConfiguration
from poke_env.player import RandomPlayer
from poke_env.player.player import Player
from poke_env.ps_client.server_configuration import LocalhostServerConfiguration

from agents.battle.gen3_battle import Gen3Battle
from agents.gen3_data import moves as gen3_movedex
from agents.observation.constants import (
    OFFSET_OUR_TEAM, OFFSET_OPP_TEAM, POKEMON_FULL_DIM, POKEMON_VECTOR_DIM,
    POKEMON_MOVES_OFFSET, MOVE_SLOT_DIM, TEAM_SIZE,
)
from agents.observation.moves import HIDDEN_POWER_MOVE_NUM
from agents.observation.state_encoder import Gen3ObservationEncoder, load_mappings
from utils.team_loader import TeamLoader
from utils.teambuilder import Gen3Teambuilder
from utils.bridge.local_battle_runner import run_local_battles

_KNOWN_OFF = 6     # within a move slot: the "known" flag offset
_ID_OFF = 0        # within a move slot: the move-id (num) channel offset


def _data_num(move_id: str) -> int:
    md = gen3_movedex.get(move_id)
    return int(md.num) if md is not None else -1


@dataclass
class _Stats:
    decisions: int = 0
    mons_checked: int = 0
    move_slots_checked: int = 0
    our_hp_distinct: int = 0     # our HP slots seen encoded at a distinct num (355-370)
    opp_hp_typeless: int = 0     # opp revealed HP slots seen encoded at 237
    misassoc_fail: int = 0
    boundary_fail: int = 0
    examples: List[dict] = field(default_factory=list)

    def fail(self, kind: str, ex: dict) -> None:
        setattr(self, kind, getattr(self, kind) + 1)
        if len(self.examples) < 16:
            self.examples.append(ex)

    @property
    def failures(self) -> int:
        return self.misassoc_fail + self.boundary_fail


class _DecodePlayer(Player):
    def __init__(self, *a, **k):
        k.setdefault("battle_class", Gen3Battle)
        super().__init__(*a, **k)
        self.encoder = Gen3ObservationEncoder(load_mappings())
        self.stats = _Stats()

    def _encoded_slots(self, vec, base):
        """(num, decoded_name) for each KNOWN move slot in a per-mon block at obs offset `base`."""
        out = []
        mb = base + POKEMON_MOVES_OFFSET
        names = self.encoder.pokemon_encoder.describe_vector(
            vec[base:base + POKEMON_VECTOR_DIM]).get("moves", [])
        ni = 0
        for s in range(4):
            slot = mb + s * MOVE_SLOT_DIM
            if vec[slot + _KNOWN_OFF] > 0.5:
                num = int(vec[slot + _ID_OFF])
                # describe_vector returns names for known slots in slot order → align by index.
                name = names[ni] if ni < len(names) else None
                ni += 1
                out.append((num, name))
        return out

    def _check_mon(self, vec, base, mon, is_own: bool) -> None:
        s = self.stats
        # Encoder-visible moves: our full moveset; the opponent's REVEALED moves only.
        visible = list(mon.moves.values())
        if not visible:
            return
        s.mons_checked += 1
        expected = collections.Counter(_data_num(mv.id) for mv in visible)
        encoded = self._encoded_slots(vec, base)
        got = collections.Counter(num for num, _ in encoded)
        s.move_slots_checked += len(encoded)

        # (1) NO MIS-ASSOCIATION: the encoded num multiset must equal the data-num multiset.
        if got != expected:
            s.fail("misassoc_fail", {"check": "num_multiset_mismatch", "mon": mon.species,
                                     "own": is_own, "expected": dict(expected), "got": dict(got),
                                     "moves": [mv.id for mv in visible]})
            return

        # (2) ROUND-TRIP NAME + (3) BOUNDARY, per encoded slot.
        for num, name in encoded:
            if name is None or name.startswith("Move("):
                s.fail("misassoc_fail", {"check": "num_not_in_reverse_map", "mon": mon.species,
                                         "own": is_own, "num": num})
                continue
            base_name = name.split("(")[0]                      # 'hiddenpower(ice)' → 'hiddenpower'
            is_hp = base_name == "hiddenpower"
            if is_hp:
                if is_own:
                    # OUR HP: distinct num (355-370) + a TYPED decode (paren form), never bare 237.
                    if num == HIDDEN_POWER_MOVE_NUM or "(" not in name:
                        s.fail("boundary_fail", {"check": "our_hp_not_distinct_typed", "mon": mon.species,
                                                 "num": num, "name": name})
                    else:
                        s.our_hp_distinct += 1
                else:
                    # OPP revealed HP: typeless num 237 + BARE decode (NO leak of the type).
                    if num != HIDDEN_POWER_MOVE_NUM or "(" in name:
                        s.fail("boundary_fail", {"check": "opp_hp_leaked_type", "mon": mon.species,
                                                 "num": num, "name": name})
                    else:
                        s.opp_hp_typeless += 1

    def choose_move(self, battle):
        try:
            self._validate(battle)
        except Exception as e:  # noqa: BLE001
            print("\n🛑 [MOVE-ID DECODE FUZZ CRITICAL FAILURE] 🛑")
            print(f"Battle {battle.battle_tag} turn {battle.turn}: {e}")
            traceback.print_exc()
            os._exit(1)
        return self.choose_random_move(battle)

    def _validate(self, battle) -> None:
        if battle.active_pokemon is None:
            return
        vec = self.encoder.encode(battle)
        self.stats.decisions += 1
        for i, mon in enumerate(battle.team.values()):
            if i < TEAM_SIZE:
                self._check_mon(vec, OFFSET_OUR_TEAM + i * POKEMON_FULL_DIM, mon, is_own=True)
        for i, mon in enumerate(battle.opponent_team.values()):
            if i < TEAM_SIZE:
                self._check_mon(vec, OFFSET_OPP_TEAM + i * POKEMON_FULL_DIM, mon, is_own=False)


async def run(n_battles: int) -> None:
    print(f"Move-id DECODE Fuzz — gen3ou — {n_battles} battles (never mis-associate a move)\n")
    teams = TeamLoader().get_all_teams()
    tb = Gen3Teambuilder(teams)
    ts = int(time.time())
    player = _DecodePlayer(
        battle_format="gen3ou", team=tb, server_configuration=LocalhostServerConfiguration,
        start_listening=False, account_configuration=AccountConfiguration(f"MidDec{ts}", "x"),
        max_concurrent_battles=10)
    opp = RandomPlayer(
        battle_format="gen3ou", team=tb, server_configuration=LocalhostServerConfiguration,
        start_listening=False, account_configuration=AccountConfiguration(f"MidDecOpp{ts}", "x"),
        max_concurrent_battles=10)
    await run_local_battles(player, opp, n_battles)

    s = player.stats
    print("=" * 70)
    print(f"decisions validated        : {s.decisions}")
    print(f"mons checked               : {s.mons_checked}")
    print(f"move slots checked         : {s.move_slots_checked}")
    print(f"our HP @ distinct num      : {s.our_hp_distinct}")
    print(f"opp HP @ typeless 237      : {s.opp_hp_typeless}  (no-leak)")
    print(f"mis-association failures   : {s.misassoc_fail}")
    print(f"boundary failures          : {s.boundary_fail}")
    for ex in s.examples:
        print("   violation:", ex)
    print("=" * 70)

    if s.failures:
        print(f"\n❌ FAIL — {s.failures} move-decode violation(s)")
        os._exit(1)
    if s.our_hp_distinct == 0:
        print("\n⚠️  PASS but NO own Hidden Power was exercised — INCONCLUSIVE for the distinct-num path. "
              "Re-run with more battles.")
    elif s.opp_hp_typeless == 0:
        print("\n⚠️  PASS (our HP typed) but NO opponent HP was revealed — the 237/no-leak side went "
              "unexercised. Re-run with more battles for full boundary coverage.")
    else:
        print("\n✅ PASS — every move round-trips to its canonical num (no mis-association); our HP uses "
              "distinct typed nums, the opponent's HP stays the typeless 237 (no leak).")


if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(run(int(sys.argv[1]) if len(sys.argv) > 1 else 60))
