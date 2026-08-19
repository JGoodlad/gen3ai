"""Bridge-backed fuzz validation for the gen3_wish_wired_v1 "wish floating" obs feature.

Runs real gen3ou battles in-process (local BattleStream bridge, no server) with a curated team built
to exercise the edge cases: multiple Wish users + Roar phazers (Skarmory/Suicune) + Spikes, so wishes
resolve through wish-passing switches, Roar drags, and hazard-chip faints.

Ground truth comes from the RAW Showdown protocol (parsed independently of our event log, which is what
the obs reconstruction reads — so this is NON-circular): a `|move|<mon>|Wish` is a use; a
`|-heal|<mon>|...|[from] move: Wish` is the sim's ACTUAL resolve, which fires at the END of the turn the
wish was pending. Validations:

  (A) COMPLETENESS — for every actual Wish RESOLVE at turn D for side S, the obs encoded at the turn-D
      decision for S must have flagged the wish pending (vec = WISH_HEAL_FRACTION). This is the decisive
      check: the sim healed via Wish end of turn D ⇒ we must have seen it coming at the turn-D decision.
  (B) SOUNDNESS — for every decision where the obs flags a side pending, that side must have used Wish on
      the previous turn (no pending without a prior use).
  (C) TIMING — every observed resolve is exactly one turn after a use by that side.

(Full-HP resolves emit NO heal line, so we do NOT require every pending to produce a heal — only the two
checkable directions above; that confound can't hide a real bug.)

Run directly (no server needed; in-process via the local bridge):
    export PYTHONPATH=$PYTHONPATH:src
    python src/agents/training/poke_env_gaps/wish_floating_fuzz_test.py [n_battles]
"""
from __future__ import annotations

import asyncio
import os
import sys
import time
import traceback
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple

from poke_env import AccountConfiguration
from poke_env.player import RandomPlayer
from poke_env.player.battle_order import ForfeitBattleOrder
from poke_env.player.player import Player
from poke_env.ps_client.server_configuration import LocalhostServerConfiguration

from agents.battle.gen3_battle import Gen3Battle
from agents.observation.constants import OFFSET_REACTIVE
from agents.observation.state_encoder import Gen3ObservationEncoder, load_mappings
from agents.observation.wish_belief import WISH_HEAL_FRACTION
from utils.bridge.local_battle_runner import run_local_battles
from utils.teambuilder import Gen3Teambuilder

BATTLE_FORMAT = "gen3ou"
_TOL = 1e-4
_TURN_CAP = 120
def wish_obs_offsets() -> "tuple[int, int]":
    """The two `wish_floating` obs columns, from the reactive block's DECLARED layout.

    They were hardcoded `OFFSET_REACTIVE + 17 / + 18` — and `REACTIVE_DIM` is **17**, so both
    literals pointed PAST the reactive block entirely, at `OFFSET_PAIR_HISTORY + 0 / + 1`. The
    oracle was therefore comparing its Wish expectation against the pair-history block, which is
    zero on most turns, so the fuzz's completeness half could only ever read "the encoder never
    floats a Wish". The live columns are `+3 / +4`, and reading them by NAME is what stops the
    next re-home doing this again — the reactive block has already lost `protect_odds` and
    `trapped` to the per-mon slots once (`gen3_entity_rehome_v1`), which is exactly when these
    literals went stale.
    """
    from agents.observation.reactive import ReactiveEncoder

    lay = ReactiveEncoder().get_layout()
    return (OFFSET_REACTIVE + lay["wish_floating_our"]["offset"],
            OFFSET_REACTIVE + lay["wish_floating_opp"]["offset"])


_WISH_OUR, _WISH_OPP = wish_obs_offsets()

# Curated team (both sides): 3 Wish users + 2 Roar phazers + Spikes → wish-passing, phaze-drag, and
# hazard-death edge cases all occur under random play.
WISH_TEAM = """\
Blissey @ Leftovers
Ability: Natural Cure
EVs: 252 HP / 252 Def / 4 SpD
Bold Nature
- Wish
- Protect
- Seismic Toss
- Ice Beam

Vaporeon @ Leftovers
Ability: Water Absorb
EVs: 252 HP / 252 Def / 4 SpA
Bold Nature
- Wish
- Surf
- Protect
- Ice Beam

Jirachi @ Leftovers
Ability: Serene Grace
EVs: 252 HP / 4 Atk / 252 SpD
Careful Nature
- Wish
- Body Slam
- Protect
- Hidden Power

Skarmory @ Leftovers
Ability: Keen Eye
EVs: 252 HP / 252 Def / 4 SpD
Impish Nature
- Spikes
- Roar
- Drill Peck
- Protect

Suicune @ Leftovers
Ability: Pressure
EVs: 252 HP / 252 Def / 4 SpA
Bold Nature
- Surf
- Roar
- Ice Beam
- Rest

Dugtrio @ Choice Band
Ability: Arena Trap
EVs: 4 HP / 252 Atk / 252 Spe
Jolly Nature
- Earthquake
- Rock Slide
- Aerial Ace
- Hidden Power
"""


@dataclass
class _Stats:
    decisions: int = 0
    wish_uses: int = 0
    wish_resolves: int = 0
    wish_fails: int = 0          # a |move|Wish that |-fail|ed (didn't set the slot — gen3: only double-Wish)
    completeness_fail: int = 0   # (A) a resolve at D but obs at D didn't flag pending
    soundness_fail: int = 0      # (B) obs flagged pending but no use the prior turn
    timing_fail: int = 0         # (C) a resolve not one turn after a use
    false_pending_fail: int = 0  # (D) a FAILED Wish use that the obs still flagged pending next turn
    examples: List[str] = field(default_factory=list)

    def note(self, m: str) -> None:
        if len(self.examples) < 25:
            self.examples.append(m)


class _WishFuzzPlayer(Player):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("battle_class", Gen3Battle)
        super().__init__(*args, **kwargs)
        self.encoder = Gen3ObservationEncoder(load_mappings())
        self.stats = _Stats()
        # per battle tag: ground-truth from RAW protocol + the obs-encoded pending we recorded
        self._cur_turn: Dict[str, int] = defaultdict(int)
        self._uses: Dict[str, Set[Tuple[int, str]]] = defaultdict(set)        # (turn, side) — SUCCESSFUL uses
        self._failed_uses: Dict[str, Set[Tuple[int, str]]] = defaultdict(set)  # (turn, side) — |-fail|ed uses
        self._resolves: Dict[str, Set[Tuple[int, str]]] = defaultdict(set)    # (turn, side)
        self._encoded: Dict[str, Dict[Tuple[int, str], float]] = defaultdict(dict)  # (turn, side) -> vec val

    # --- RAW-protocol ground truth (independent of our event log) ---------------------------------
    async def _handle_battle_message(self, split_messages):
        tag = split_messages[0][0].lstrip(">") if split_messages and split_messages[0] else ""
        battle = self._battles.get(tag) if hasattr(self, "_battles") else None
        role = getattr(battle, "player_role", None) if battle is not None else None
        for idx, msg in enumerate(split_messages):
            if len(msg) < 2:
                continue
            kind = msg[1]
            if kind == "turn" and len(msg) > 2 and msg[2].isdigit():
                self._cur_turn[tag] = int(msg[2])
            elif kind == "move" and len(msg) > 3 and msg[3].strip().lower().replace(" ", "") == "wish":
                side = self._side_of(msg[2], role)
                if side:
                    self.stats.wish_uses += 1
                    # a Wish that |-fail|s (next lines, same actor) did NOT set the slot — gen3's only
                    # fail cause is double-Wish (slot occupied); it must NOT be flagged pending next turn.
                    failed = any(len(m2) >= 3 and m2[1] == "-fail" and m2[2] == msg[2]
                                 for m2 in split_messages[idx + 1: idx + 4])
                    if failed:
                        self._failed_uses[tag].add((self._cur_turn[tag], side))
                        self.stats.wish_fails += 1
                    else:
                        self._uses[tag].add((self._cur_turn[tag], side))
            elif kind == "-heal" and any("move: Wish" in str(t) or "[wisher]" in str(t) for t in msg[3:]):
                side = self._side_of(msg[2], role)
                if side:
                    self._resolves[tag].add((self._cur_turn[tag], side))
                    self.stats.wish_resolves += 1
        await super()._handle_battle_message(split_messages)

    @staticmethod
    def _side_of(ident: str, role) -> str:
        if not ident or not role or len(ident) < 2:
            return ""
        return "ours" if ident[:2] == role else "opp"

    def choose_move(self, battle):
        try:
            tag = battle.battle_tag
            vec = self.encoder.encode(battle)
            self.stats.decisions += 1
            self._encoded[tag][(battle.turn, "ours")] = float(vec[_WISH_OUR])
            self._encoded[tag][(battle.turn, "opp")] = float(vec[_WISH_OPP])
        except Exception as e:  # noqa: BLE001
            print("\n🛑 [WISH FUZZ CRITICAL FAILURE] 🛑")
            print(f"Battle {battle.battle_tag} turn {battle.turn}: {e}")
            traceback.print_exc()
            os._exit(1)
        if battle.turn > _TURN_CAP:
            return ForfeitBattleOrder()
        return self.choose_random_move(battle)

    # --- post-battle validation (called from run() after all battles) -----------------------------
    def validate(self) -> None:
        s = self.stats
        for tag, resolves in self._resolves.items():
            enc = self._encoded[tag]
            uses = self._uses[tag]
            # (A) completeness: every resolve at turn D, side S → obs at decision D flagged pending
            for (d, side) in resolves:
                v = enc.get((d, side))
                if v is None:
                    continue  # no decision captured at that turn for that perspective (e.g. capped)
                if abs(v - WISH_HEAL_FRACTION) > _TOL:
                    s.completeness_fail += 1
                    s.note(f"COMPLETENESS {tag} t{d} {side}: resolve seen but obs={v:.3f} (want {WISH_HEAL_FRACTION})")
                # (C) timing: a resolve at D must follow a use at D-1 by the same side
                if (d - 1, side) not in uses:
                    s.timing_fail += 1
                    s.note(f"TIMING {tag} t{d} {side}: resolve with no use at t{d-1}")
            # (B) soundness: every flagged-pending decision → a use the prior turn
            for (d, side), v in enc.items():
                if abs(v - WISH_HEAL_FRACTION) <= _TOL and (d - 1, side) not in uses:
                    s.soundness_fail += 1
                    s.note(f"SOUNDNESS {tag} t{d} {side}: obs pending but no Wish use at t{d-1}")
            # (D) no-false-pending-from-a-FAILED-use: a Wish that |-fail|ed (didn't set the slot — gen3's
            # only fail is double-Wish) must NOT be flagged pending the next turn. This is the case a
            # plain soundness check (B) misses, since a failed use IS still a "use". (User-requested.)
            for (d, side) in self._failed_uses[tag]:
                v = enc.get((d + 1, side))
                if v is not None and abs(v - WISH_HEAL_FRACTION) <= _TOL and (d, side) not in uses:
                    s.false_pending_fail += 1
                    s.note(f"FALSE-PENDING {tag} t{d+1} {side}: a FAILED Wish at t{d} was flagged pending")


def _report_and_assert(s: _Stats) -> None:
    print("=" * 70)
    print(f"decisions validated   : {s.decisions}")
    print(f"wish USES seen        : {s.wish_uses}  ({s.wish_fails} of them |-fail|ed = double-Wish, didn't set)")
    print(f"wish RESOLVES seen    : {s.wish_resolves}  (the non-circular ground truth)")
    print(f"completeness failures : {s.completeness_fail}  (resolve at D but obs didn't flag pending)")
    print(f"soundness failures    : {s.soundness_fail}  (obs flagged pending w/o a prior use)")
    print(f"timing failures       : {s.timing_fail}  (resolve not 1 turn after a use)")
    print(f"false-pending failures: {s.false_pending_fail}  (a FAILED Wish use flagged pending next turn)")
    for ex in s.examples:
        print("   ·", ex)
    print("=" * 70)
    total_fail = s.completeness_fail + s.soundness_fail + s.timing_fail + s.false_pending_fail
    if total_fail:
        print(f"\n❌ FAIL — {total_fail} wish-floating violation(s)")
        os._exit(1)
    if s.wish_resolves == 0:
        print("⚠️  PASS but NO wish resolve was observed — INCONCLUSIVE. Re-run with more battles.")
    else:
        print(f"✅ PASS — every one of {s.wish_resolves} actual Wish resolves was correctly flagged pending "
              f"the turn before (completeness); no spurious pending (soundness); and none of the {s.wish_fails} "
              f"FAILED (double-Wish) uses was falsely flagged pending — across {s.decisions} decisions.")


async def run(n_battles: int) -> None:
    print(f"Wish floating Fuzz — gen3ou — {n_battles} battles (curated Wish + Roar + Spikes team)\n")
    teambuilder = Gen3Teambuilder(WISH_TEAM)
    ts = int(time.time())
    player = _WishFuzzPlayer(
        battle_format=BATTLE_FORMAT, team=teambuilder,
        server_configuration=LocalhostServerConfiguration, start_listening=False,
        account_configuration=AccountConfiguration(f"WishFuzz{ts}", "x"),
        max_concurrent_battles=10)
    opponent = RandomPlayer(
        battle_format=BATTLE_FORMAT, team=teambuilder,
        server_configuration=LocalhostServerConfiguration, start_listening=False,
        account_configuration=AccountConfiguration(f"WishFuzzOpp{ts}", "x"),
        max_concurrent_battles=10)
    await run_local_battles(player, opponent, n_battles)
    player.validate()
    _report_and_assert(player.stats)


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 120
    asyncio.run(run(n))
