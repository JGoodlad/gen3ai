"""
E2E effectiveness + move-order fuzz test.

Validates the full pipeline correctness at five independent layers:

  Layer 1 — raw Showdown messages vs poke-env battle properties
              (battle.our/opp_last_effectiveness, battle.we_moved_first)
  Layer 2 — poke-env properties vs BattleContext snapshot
              (curr_ctx.our/opp_last_effectiveness, curr_ctx.we_moved_first)
  Layer 3 — BattleContext vs TurnDelta fields
              (delta.our/opp_effectiveness, delta.we_moved_first)
  Layer 4 — TurnDelta vs encoded observation vector
              (TurnDeltaEncoder output at dims 29..38)
  Layer 5 — battle.{our,opp}_last_damaging_move (DamagingMoveEvent) vs the
              raw |move| lines that produced it. Asserts user_species,
              target_species, move_id, and effectiveness all match. Added in
              ai_v4 step 3 alongside the foundational attribution primitive.

Runs three targeted scenarios:
  A — Immunity + SE: teams with Levitate/Volt Absorb/Water Absorb abilities and
      diverse attacking coverage, exercising all four effectiveness categories.
  B — Mixed type matchups: fire, water, electric, ground, normal attacks vs
      appropriate weaknesses and resistances.
  C — Priority + speed: Quick Attack users vs slower mons; verifies we_moved_first
      matches known priority, and that we_moved_first=None on switch turns.

Run directly (requires: npm run showdown):
    python src/agents/training/poke_env_gaps/effectiveness_fuzz_e2e_test.py [n_battles]
    (in a linked worktree, first: export PYTHONPATH=$PYTHONPATH:src)
"""

import asyncio
import os
import sys
import time
import traceback
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from poke_env import AccountConfiguration
from poke_env.player import RandomPlayer
from poke_env.player.player import Player
from poke_env.ps_client.server_configuration import LocalhostServerConfiguration

from agents.action.mapper import Gen3ActionMapper
from agents.action.mask_generator import Gen3ActionMasker
from agents.observation.state_encoder import load_mappings
from agents.observation.turn_delta_encoder import (
    TurnDeltaEncoder, EFF_DIM, ORDER_DIM,
    OFFSET_OUR_EFF, OFFSET_OPP_EFF, OFFSET_ORDER,
)
from agents.training.battle_snapshot import BattleContext
from agents.training.turn_delta import TurnDelta
from agents.training.turn_delta_legacy import build_legacy
from agents.training.slot_registry import SlotRegistry
from utils.teambuilder import Gen3Teambuilder

BATTLE_FORMAT = "gen3ou"

# ---------------------------------------------------------------------------
# Teams
# ---------------------------------------------------------------------------

# Scenario A — Immunity interactions
# Gengar/Flygon (Levitate → Ground immune), Jolteon (Volt Absorb → Electric immune),
# Vaporeon (Water Absorb → Water immune). Diverse attacking coverage ensures all four
# effectiveness categories fire from both sides.
IMMUNE_TEAM = """\
Gengar @ Leftovers
Ability: Levitate
EVs: 4 HP / 252 SpA / 252 Spe
Timid Nature
- Shadow Ball
- Thunderbolt
- Fire Punch
- Hypnosis

Flygon @ Leftovers
Ability: Levitate
EVs: 4 HP / 252 Atk / 252 Spe
Jolly Nature
- Earthquake
- Flamethrower
- Rock Slide
- Dragon Claw

Jolteon @ Leftovers
Ability: Volt Absorb
EVs: 4 HP / 252 SpA / 252 Spe
Timid Nature
- Thunderbolt
- Shadow Ball
- Hidden Power Ice
- Baton Pass

Vaporeon @ Leftovers
Ability: Water Absorb
EVs: 252 HP / 252 Def / 4 SpA
Bold Nature
- Surf
- Ice Beam
- Wish
- Protect

Metagross @ Leftovers
Ability: Clear Body
EVs: 252 HP / 228 Atk / 28 Spe
Adamant Nature
- Meteor Mash
- Earthquake
- Explosion
- Psychic

Tyranitar @ Leftovers
Ability: Sand Stream
EVs: 252 HP / 40 Atk / 216 SpD
Careful Nature
- Rock Slide
- Earthquake
- Crunch
- Fire Blast
"""

# Scenario B — SE and resisted matchups
# Fire/Water/Electric/Ground/Fighting/Rock type coverage ensures consistent
# SE and resisted interactions from both sides.
SE_RESISTED_TEAM = """\
Charizard @ Leftovers
Ability: Blaze
EVs: 4 HP / 252 SpA / 252 Spe
Timid Nature
- Flamethrower
- Dragon Claw
- Earthquake
- Aerial Ace

Blastoise @ Leftovers
Ability: Torrent
EVs: 252 HP / 252 SpA / 4 SpD
Modest Nature
- Surf
- Ice Beam
- Rapid Spin
- Toxic

Jolteon @ Lum Berry
Ability: Volt Absorb
EVs: 4 HP / 252 SpA / 252 Spe
Timid Nature
- Thunderbolt
- Shadow Ball
- Bite
- Agility

Steelix @ Leftovers
Ability: Rock Head
EVs: 252 HP / 252 Atk / 4 Def
Adamant Nature
- Earthquake
- Iron Tail
- Rock Slide
- Explosion

Heracross @ Choice Band
Ability: Guts
EVs: 4 HP / 252 Atk / 252 Spe
Adamant Nature
- Megahorn
- Brick Break
- Rock Slide
- Earthquake

Skarmory @ Leftovers
Ability: Keen Eye
EVs: 252 HP / 252 Def / 4 Spe
Impish Nature
- Drill Peck
- Spikes
- Roar
- Rest
"""

# Scenario C — Priority + speed tier verification
# Arcanine (ExtremeSpeed, +1 priority in Gen 3 — same tier as Quick Attack),
# Jolteon (Quick Attack, +1) vs slow tanky mons. Priority users go first even
# when their base speed would normally lose.
PRIORITY_TEAM = """\
Arcanine @ Lum Berry
Ability: Intimidate
EVs: 4 HP / 252 Atk / 252 Spe
Jolly Nature
- Extreme Speed
- Flamethrower
- Iron Tail
- Overheat

Jolteon @ Leftovers
Ability: Volt Absorb
EVs: 4 HP / 252 SpA / 252 Spe
Timid Nature
- Thunderbolt
- Shadow Ball
- Quick Attack
- Agility

Marowak @ Thick Club
Ability: Rock Head
EVs: 252 HP / 252 Atk / 4 SpD
Adamant Nature
- Earthquake
- Rock Slide
- Hidden Power
- Bonemerang

Blissey @ Leftovers
Ability: Natural Cure
EVs: 252 HP / 252 Def / 4 SpD
Bold Nature
- Softboiled
- Ice Beam
- Thunder Wave
- Seismic Toss

Snorlax @ Leftovers
Ability: Thick Fat
EVs: 252 HP / 16 Atk / 136 Def / 104 SpD
Careful Nature
- Body Slam
- Earthquake
- Rest
- Sleep Talk

Forretress @ Leftovers
Ability: Sturdy
EVs: 252 HP / 252 Def / 4 SpD
Relaxed Nature
- Spikes
- Rapid Spin
- Explosion
- Hidden Power
"""


# ---------------------------------------------------------------------------
# Raw message tracking: per battle, per turn
# ---------------------------------------------------------------------------

@dataclass
class _TurnRaw:
    """Raw Showdown protocol events for one turn (keyed by mover/defender prefix)."""
    # List of (defender_prefix, multiplier) for effectiveness events.
    # Stored in order; the LAST event for each side wins (mirrors poke-env's overwrite).
    eff_events: list = field(default_factory=list)  # (defender_prefix: str, mult: float)
    # Move event prefixes in order of appearance
    move_sides: list = field(default_factory=list)  # e.g. ["p1", "p2"]
    # Full move events from the |move| line. Each entry:
    #   (user_prefix, user_name, move_name, target_prefix, target_name)
    # The user/target names are the post-": " portion of the protocol ident
    # (e.g. for "p1a: Salamence" → "Salamence"); they are lowercased for
    # direct comparison against DamagingMoveEvent.user_species /
    # target_species. The move name is also lowercased+space-stripped via
    # to_id_str so it matches DamagingMoveEvent.move_id (e.g. "hiddenpower").
    move_events: list = field(default_factory=list)


def _opp_role(player_role: str) -> str:
    """Return the opposite side prefix (p1↔p2)."""
    return "p2" if player_role == "p1" else "p1"


def _expected_effectiveness_from_raw(
    raw: _TurnRaw,
    player_role: str,
) -> tuple[Optional[float], Optional[float]]:
    """
    Interpret raw events using player_role.
    Returns (our_eff, opp_eff): what poke-env SHOULD have recorded.

    our_eff  = effectiveness of OUR last damaging move (defender is opponent)
    opp_eff  = effectiveness of OPP's last damaging move (defender is us)
    """
    our_eff = None
    opp_eff = None
    for defender_prefix, mult in raw.eff_events:
        if defender_prefix == player_role:
            # Defender is US → OPP's move hit us
            opp_eff = mult
        else:
            # Defender is OPP → OUR move hit them
            our_eff = mult
    return our_eff, opp_eff


def _expected_we_moved_first_from_raw(
    raw: _TurnRaw,
    player_role: str,
) -> Optional[bool]:
    """True if we moved first, False if opp first, None if only one/neither moved.

    Sleep Talk (and similar) fires two |move| messages for the same side; we deduplicate
    to match poke-env's behaviour (which uses `if side not in seen` guard).
    """
    seen: list[str] = []
    for side in raw.move_sides:
        if side not in seen:
            seen.append(side)
    if len(seen) < 2:
        return None
    return seen[0] == player_role


# ---------------------------------------------------------------------------
# Stats dataclass
# ---------------------------------------------------------------------------

@dataclass
class ScenarioStats:
    name: str
    n_turns: int = 0

    # Layer mismatch counts (all must be 0 to PASS)
    layer1_mismatches: int = 0
    layer2_mismatches: int = 0
    layer3_mismatches: int = 0
    layer4_mismatches: int = 0
    layer5_mismatches: int = 0  # battle.{our,opp}_last_damaging_move vs raw |move| lines

    # Coverage counters (must be > 0 after run)
    immune_seen: int = 0
    resisted_seen: int = 0
    normal_seen: int = 0
    se_seen: int = 0
    we_first_seen: int = 0
    opp_first_seen: int = 0
    switched_no_order_seen: int = 0

    # Up to 10 mismatch examples for debugging
    examples: list = field(default_factory=list)

    def record_example(self, layer: int, detail: dict):
        if len(self.examples) < 10:
            self.examples.append({"layer": layer, **detail})


# ---------------------------------------------------------------------------
# Fuzz player
# ---------------------------------------------------------------------------

# Load shared mappings and encoder once at module level
_MAPPINGS = None
_ENCODER = None


def _get_encoder():
    global _MAPPINGS, _ENCODER
    if _ENCODER is None:
        _MAPPINGS = load_mappings()
        _ENCODER = TurnDeltaEncoder(
            _MAPPINGS.get("moves", {}),
            _MAPPINGS.get("species", {}),
        )
    return _ENCODER


def _eff_to_category(mult: Optional[float]) -> Optional[int]:
    """Map effectiveness multiplier to one-hot index (matches TurnDeltaEncoder)."""
    if mult is None:
        return None
    if mult == 0.0:
        return 0  # immune
    elif mult <= 0.5:
        return 1  # resisted
    elif mult == 1.0:
        return 2  # normal
    else:
        return 3  # super-effective


class EffectivenessFuzzPlayer(Player):
    def __init__(self, scenario_name: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stats = ScenarioStats(name=scenario_name)
        # Per-battle state, keyed by battle_tag (no leading ">")
        self._raw_prev: dict[str, Optional[_TurnRaw]] = {}  # prev-turn raw events
        self._raw_curr: dict[str, _TurnRaw] = {}            # current-turn raw events
        self._prev_ctx: dict[str, Optional[BattleContext]] = {}  # prev BattleContext
        self._last_action: dict[str, Optional[int]] = {}
        # Slot registries persist for the lifetime of each battle
        self._our_slots: dict[str, SlotRegistry] = {}
        self._opp_slots: dict[str, SlotRegistry] = {}

    async def _handle_battle_message(self, split_messages):
        """Override to intercept raw Showdown messages before poke-env parses them."""
        battle_tag = split_messages[0][0].lstrip(">")

        # Initialize per-battle raw tracking
        if battle_tag not in self._raw_curr:
            self._raw_curr[battle_tag] = _TurnRaw()
            self._raw_prev[battle_tag] = None

        curr = self._raw_curr[battle_tag]

        for msg in split_messages[1:]:
            if len(msg) < 2:
                continue
            mt = msg[1]

            if mt == "turn":
                # A new turn starts: archive current raw events as previous turn
                self._raw_prev[battle_tag] = curr
                curr = _TurnRaw()
                self._raw_curr[battle_tag] = curr

            elif mt == "-supereffective" and len(msg) >= 3:
                curr.eff_events.append((msg[2][:2], 2.0))

            elif mt == "-resisted" and len(msg) >= 3:
                curr.eff_events.append((msg[2][:2], 0.5))

            elif mt == "-immune" and len(msg) >= 3:
                curr.eff_events.append((msg[2][:2], 0.0))

            elif mt == "move" and len(msg) >= 3:
                # Track move order: extract "p1"/"p2" prefix from "p1a: MonName"
                curr.move_sides.append(msg[2][:2])
                # Track full move events for Layer 5 (DamagingMoveEvent) validation:
                # user/target identity and the move name. The target slot can be
                # missing for self-targeted moves; record it as None in that case.
                user_ident = msg[2]  # "p1a: Salamence"
                user_prefix = user_ident[:2]
                user_name = user_ident.split(": ", 1)[1].lower() if ": " in user_ident else ""
                move_name = msg[3] if len(msg) >= 4 else ""
                move_id = move_name.lower().replace(" ", "").replace("-", "")
                # Strip "[Bug]" / "[Ice]" etc. from Hidden Power so move_id matches
                # to_id_str (which strips brackets too).
                if "[" in move_id:
                    move_id = move_id.split("[", 1)[0]
                target_prefix = None
                target_name = None
                if len(msg) >= 5 and msg[4] and msg[4].startswith(("p1", "p2")):
                    target_ident = msg[4]
                    target_prefix = target_ident[:2]
                    target_name = (
                        target_ident.split(": ", 1)[1].lower()
                        if ": " in target_ident else ""
                    )
                curr.move_events.append(
                    (user_prefix, user_name, move_id, target_prefix, target_name)
                )

        # Let poke-env parse normally
        await super()._handle_battle_message(split_messages)

    def _validate_turn(self, battle, curr_ctx: BattleContext, delta: TurnDelta):
        """Run all four validation layers for the turn that just ended."""
        s = self.stats
        s.n_turns += 1
        battle_tag = battle.battle_tag

        player_role = battle.player_role
        if player_role is None:
            return  # player_role not yet set (very early in battle)

        prev_raw = self._raw_prev.get(battle_tag)
        if prev_raw is None:
            return  # first turn: no previous raw data

        # === Layer 1: raw messages vs poke-env properties ===
        expected_our, expected_opp = _expected_effectiveness_from_raw(prev_raw, player_role)
        expected_order = _expected_we_moved_first_from_raw(prev_raw, player_role)

        actual_our_eff = battle.our_last_effectiveness
        actual_opp_eff = battle.opp_last_effectiveness
        actual_order = battle.we_moved_first

        # Only flag a mismatch when the raw stream has an EXPLICIT effectiveness signal
        # (SE/resisted/immune message fired) and poke-env disagrees.
        # We skip turns where raw returns None (no explicit message) but poke-env returns
        # 1.0 — that is intentional tentative-1.0 logic for normal-effectiveness moves.
        # We also skip forced-switch decisions (battle.force_switch) since they fire
        # mid-turn before |turn|N+1, making _xxx_effectiveness unavailable via properties.
        if not battle.force_switch:
            if expected_our is not None and actual_our_eff != expected_our:
                s.layer1_mismatches += 1
                s.record_example(1, {
                    "check": "our_last_effectiveness",
                    "turn": battle.turn,
                    "expected": expected_our,
                    "actual": actual_our_eff,
                    "raw_eff_events": list(prev_raw.eff_events),
                    "player_role": player_role,
                })
            if expected_opp is not None and actual_opp_eff != expected_opp:
                s.layer1_mismatches += 1
                s.record_example(1, {
                    "check": "opp_last_effectiveness",
                    "turn": battle.turn,
                    "expected": expected_opp,
                    "actual": actual_opp_eff,
                    "raw_eff_events": list(prev_raw.eff_events),
                    "player_role": player_role,
                })
            if actual_order != expected_order:
                s.layer1_mismatches += 1
                s.record_example(1, {
                    "check": "we_moved_first",
                    "turn": battle.turn,
                    "expected": expected_order,
                    "actual": actual_order,
                    "raw_move_sides": list(prev_raw.move_sides),
                    "player_role": player_role,
                })

        # === Layer 2: poke-env properties vs BattleContext snapshot ===
        if curr_ctx.our_last_effectiveness != actual_our_eff:
            s.layer2_mismatches += 1
            s.record_example(2, {
                "check": "our_last_effectiveness",
                "turn": battle.turn,
                "battle_prop": actual_our_eff,
                "ctx_val": curr_ctx.our_last_effectiveness,
            })
        if curr_ctx.opp_last_effectiveness != actual_opp_eff:
            s.layer2_mismatches += 1
            s.record_example(2, {
                "check": "opp_last_effectiveness",
                "turn": battle.turn,
                "battle_prop": actual_opp_eff,
                "ctx_val": curr_ctx.opp_last_effectiveness,
            })
        if curr_ctx.we_moved_first != actual_order:
            s.layer2_mismatches += 1
            s.record_example(2, {
                "check": "we_moved_first",
                "turn": battle.turn,
                "battle_prop": actual_order,
                "ctx_val": curr_ctx.we_moved_first,
            })

        # === Layer 3: BattleContext vs TurnDelta ===
        if delta.our_effectiveness != curr_ctx.our_last_effectiveness:
            s.layer3_mismatches += 1
            s.record_example(3, {
                "check": "our_effectiveness",
                "turn": battle.turn,
                "ctx_val": curr_ctx.our_last_effectiveness,
                "delta_val": delta.our_effectiveness,
            })
        if delta.opp_effectiveness != curr_ctx.opp_last_effectiveness:
            s.layer3_mismatches += 1
            s.record_example(3, {
                "check": "opp_effectiveness",
                "turn": battle.turn,
                "ctx_val": curr_ctx.opp_last_effectiveness,
                "delta_val": delta.opp_effectiveness,
            })
        if delta.we_moved_first != curr_ctx.we_moved_first:
            s.layer3_mismatches += 1
            s.record_example(3, {
                "check": "we_moved_first",
                "turn": battle.turn,
                "ctx_val": curr_ctx.we_moved_first,
                "delta_val": delta.we_moved_first,
            })

        # === Layer 4: TurnDelta vs encoded vector ===
        encoder = _get_encoder()
        enc = encoder.encode(delta)

        # Read via named offsets — the eff/order block shifts when CANT_DIM or
        # any other base-block dim changes (e.g. gen3_move_outcome_v1).
        our_eff_vec = enc[OFFSET_OUR_EFF:OFFSET_OUR_EFF + EFF_DIM]
        opp_eff_vec = enc[OFFSET_OPP_EFF:OFFSET_OPP_EFF + EFF_DIM]
        order_vec = enc[OFFSET_ORDER:OFFSET_ORDER + ORDER_DIM]

        expected_our_idx = _eff_to_category(delta.our_effectiveness)
        expected_opp_idx = _eff_to_category(delta.opp_effectiveness)

        # our effectiveness: check one-hot matches expected category
        if expected_our_idx is None:
            if np.any(our_eff_vec > 0.5):
                s.layer4_mismatches += 1
                s.record_example(4, {
                    "check": "our_eff_onehot_should_be_zero",
                    "turn": battle.turn,
                    "delta_our_eff": delta.our_effectiveness,
                    "vec": our_eff_vec.tolist(),
                })
        elif our_eff_vec[expected_our_idx] < 0.5:
            s.layer4_mismatches += 1
            s.record_example(4, {
                "check": "our_eff_onehot",
                "turn": battle.turn,
                "expected_idx": expected_our_idx,
                "delta_our_eff": delta.our_effectiveness,
                "vec": our_eff_vec.tolist(),
            })

        if expected_opp_idx is None:
            if np.any(opp_eff_vec > 0.5):
                s.layer4_mismatches += 1
                s.record_example(4, {
                    "check": "opp_eff_onehot_should_be_zero",
                    "turn": battle.turn,
                    "delta_opp_eff": delta.opp_effectiveness,
                    "vec": opp_eff_vec.tolist(),
                })
        elif opp_eff_vec[expected_opp_idx] < 0.5:
            s.layer4_mismatches += 1
            s.record_example(4, {
                "check": "opp_eff_onehot",
                "turn": battle.turn,
                "expected_idx": expected_opp_idx,
                "delta_opp_eff": delta.opp_effectiveness,
                "vec": opp_eff_vec.tolist(),
            })

        # order one-hot
        if delta.we_moved_first is True:
            if order_vec[0] < 0.5:
                s.layer4_mismatches += 1
                s.record_example(4, {
                    "check": "order_onehot_we_first",
                    "turn": battle.turn,
                    "vec": order_vec.tolist(),
                })
        elif delta.we_moved_first is False:
            if order_vec[1] < 0.5:
                s.layer4_mismatches += 1
                s.record_example(4, {
                    "check": "order_onehot_opp_first",
                    "turn": battle.turn,
                    "vec": order_vec.tolist(),
                })
        else:
            if np.any(order_vec > 0.5):
                s.layer4_mismatches += 1
                s.record_example(4, {
                    "check": "order_onehot_should_be_zero",
                    "turn": battle.turn,
                    "vec": order_vec.tolist(),
                })

        # === Layer 5: DamagingMoveEvent vs raw |move| lines ===
        # The DamagingMoveEvent (battle.{our,opp}_last_damaging_move) is the
        # protocol-truth attribution primitive. Whenever it's set on the
        # just-ended turn, we expect:
        #   - move_id matches a |move| line where the *user* belongs to that
        #     side (us vs opp, decided by player_role)
        #   - target_species matches that line's target (post-": " name)
        #   - effectiveness matches what an explicit |-supereffective|/
        #     |-resisted|/|-immune|/|-start ability: Flash Fire| in the same
        #     turn yielded against the same target
        # This is the regression net for ai_v4 step 3's foundational primitive.
        for side_label, event, expected_eff in [
            ("our", battle.our_last_damaging_move, expected_our),
            ("opp", battle.opp_last_damaging_move, expected_opp),
        ]:
            if event is None:
                # No event set this turn — protocol either had no damaging
                # move or no explicit effectiveness emission fired. Either
                # way, nothing to assert (Layer 1 already validated the eff
                # scalars match the raw stream).
                continue
            # Find the |move| line for this side. The user side is "ours"
            # if `event` came from `our_last_damaging_move` (side_label "our").
            user_side = player_role if side_label == "our" else _opp_role(player_role)
            matching = [
                (u_pref, u_name, m_id, t_pref, t_name)
                for (u_pref, u_name, m_id, t_pref, t_name) in prev_raw.move_events
                if u_pref == user_side
            ]
            if not matching:
                s.layer5_mismatches += 1
                s.record_example(5, {
                    "check": f"{side_label}_event_without_raw_move",
                    "turn": battle.turn,
                    "event_move_id": event.move_id,
                    "event_user": event.user_species,
                    "event_target": event.target_species,
                    "raw_move_events": list(prev_raw.move_events),
                })
                continue
            # Take the LAST |move| from this side this turn (matches the
            # protocol-parser's overwrite semantics for pending events).
            _, last_user, last_move_id, _, last_target = matching[-1]
            if event.user_species != last_user:
                s.layer5_mismatches += 1
                s.record_example(5, {
                    "check": f"{side_label}_event_user_mismatch",
                    "turn": battle.turn,
                    "event_user": event.user_species,
                    "raw_user": last_user,
                })
            if event.move_id != last_move_id:
                s.layer5_mismatches += 1
                s.record_example(5, {
                    "check": f"{side_label}_event_move_id_mismatch",
                    "turn": battle.turn,
                    "event_move_id": event.move_id,
                    "raw_move_id": last_move_id,
                })
            # Target identity — only when the raw |move| line carried one.
            if last_target is not None and event.target_species != last_target:
                s.layer5_mismatches += 1
                s.record_example(5, {
                    "check": f"{side_label}_event_target_mismatch",
                    "turn": battle.turn,
                    "event_target": event.target_species,
                    "raw_target": last_target,
                })
            # Effectiveness — should match whatever Layer 1 expected.
            if expected_eff is not None and event.effectiveness != expected_eff:
                s.layer5_mismatches += 1
                s.record_example(5, {
                    "check": f"{side_label}_event_effectiveness_mismatch",
                    "turn": battle.turn,
                    "event_effectiveness": event.effectiveness,
                    "expected_effectiveness": expected_eff,
                })

        # === Coverage tracking ===
        for (val, label) in [(actual_our_eff, "our"), (actual_opp_eff, "opp")]:
            if val == 0.0:
                s.immune_seen += 1
            elif val == 0.5:
                s.resisted_seen += 1
            elif val == 1.0:
                s.normal_seen += 1
            elif val is not None and val > 1.0:
                s.se_seen += 1

        if actual_order is True:
            s.we_first_seen += 1
        elif actual_order is False:
            s.opp_first_seen += 1
        else:
            s.switched_no_order_seen += 1

    def choose_move(self, battle):
        try:
            mask = Gen3ActionMasker.get_mask(battle)
            battle_tag = battle.battle_tag

            # Initialize per-battle slot registries on first encounter
            if battle_tag not in self._our_slots:
                self._our_slots[battle_tag] = SlotRegistry()
                self._opp_slots[battle_tag] = SlotRegistry()

            # Build current BattleContext for effectiveness validation
            curr_ctx = BattleContext.from_battle(
                battle,
                mask=mask,
                our_slots=self._our_slots[battle_tag],
                opp_slots=self._opp_slots[battle_tag],
            )

            prev_ctx = self._prev_ctx.get(battle_tag)
            last_action = self._last_action.get(battle_tag)

            if prev_ctx is not None and last_action is not None:
                # Build TurnDelta for the turn that just completed
                delta = build_legacy(prev_ctx, curr_ctx, last_action)
                self._validate_turn(battle, curr_ctx, delta)

            if battle.finished:
                self._prev_ctx[battle_tag] = None
                self._last_action[battle_tag] = None
                self._our_slots.pop(battle_tag, None)
                self._opp_slots.pop(battle_tag, None)
                valid = np.where(mask == 1)[0]
                return Gen3ActionMapper.action_to_order(int(np.random.choice(valid)), battle)

            valid = np.where(mask == 1)[0]
            choice = int(np.random.choice(valid))
            self._prev_ctx[battle_tag] = curr_ctx
            self._last_action[battle_tag] = choice
            return Gen3ActionMapper.action_to_order(choice, battle)

        except Exception as e:
            print(f"\n[FUZZ FATAL] Battle {battle.battle_tag}, Turn {battle.turn}: {e}")
            traceback.print_exc()
            os._exit(1)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_report(s: ScenarioStats) -> None:
    total_mismatches = (
        s.layer1_mismatches + s.layer2_mismatches
        + s.layer3_mismatches + s.layer4_mismatches
        + s.layer5_mismatches
    )
    status = "PASS" if total_mismatches == 0 else "FAIL"

    print(f"\n{'=' * 65}")
    print(f"SCENARIO: {s.name}  [{status}]")
    print(f"{'=' * 65}")
    print(f"Total decision turns  : {s.n_turns}")
    print(f"Layer 1 mismatches    : {s.layer1_mismatches}  (raw messages vs poke-env properties)")
    print(f"Layer 2 mismatches    : {s.layer2_mismatches}  (poke-env properties vs BattleContext)")
    print(f"Layer 3 mismatches    : {s.layer3_mismatches}  (BattleContext vs TurnDelta)")
    print(f"Layer 4 mismatches    : {s.layer4_mismatches}  (TurnDelta vs encoded vector)")
    print(f"Layer 5 mismatches    : {s.layer5_mismatches}  (DamagingMoveEvent vs raw |move| lines)")
    print("Coverage:")
    print(f"  immune seen         : {s.immune_seen}")
    print(f"  resisted seen       : {s.resisted_seen}")
    print(f"  normal seen         : {s.normal_seen}")
    print(f"  super-effective seen: {s.se_seen}")
    print(f"  we_first=True seen  : {s.we_first_seen}")
    print(f"  we_first=False seen : {s.opp_first_seen}")
    print(f"  we_first=None seen  : {s.switched_no_order_seen}")

    if s.examples:
        print("Mismatch examples (up to 10):")
        for ex in s.examples:
            print(f"  {ex}")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

async def run_scenario(name: str, team_str: str, n_battles: int, ts: int) -> ScenarioStats:
    print(f"\n--- {name}: {n_battles} battles ---", flush=True)
    tb = Gen3Teambuilder(team_str)

    tag = name.replace("-", "").replace("/", "").replace(" ", "")[:6]
    fuzz = EffectivenessFuzzPlayer(
        scenario_name=name,
        battle_format=BATTLE_FORMAT,
        team=tb,
        server_configuration=LocalhostServerConfiguration,
        account_configuration=AccountConfiguration(f"EFz{ts}{tag}", "password"),
        max_concurrent_battles=5,
    )
    opp = RandomPlayer(
        battle_format=BATTLE_FORMAT,
        team=tb,
        server_configuration=LocalhostServerConfiguration,
        account_configuration=AccountConfiguration(f"EFo{ts}{tag}", "password"),
        max_concurrent_battles=5,
    )

    await fuzz.battle_against(opp, n_battles=n_battles)
    return fuzz.stats


async def main(n_battles: int = 50) -> None:
    ts = int(time.time()) % 100000
    print(f"Effectiveness Fuzz Test — gen3ou — {n_battles} battles per scenario")

    scenarios = [
        ("A-Immunity", IMMUNE_TEAM),
        ("B-SE-Resisted", SE_RESISTED_TEAM),
        ("C-Priority", PRIORITY_TEAM),
    ]

    all_stats = []
    for name, team in scenarios:
        stats = await run_scenario(name, team, n_battles, ts)
        all_stats.append(stats)

    for s in all_stats:
        print_report(s)

    # Final verdict
    total_mismatches = sum(
        s.layer1_mismatches + s.layer2_mismatches
        + s.layer3_mismatches + s.layer4_mismatches
        + s.layer5_mismatches
        for s in all_stats
    )

    # Coverage checks
    total_immune = sum(s.immune_seen for s in all_stats)
    total_normal = sum(s.normal_seen for s in all_stats)
    total_se = sum(s.se_seen for s in all_stats)
    total_we_first = sum(s.we_first_seen for s in all_stats)
    total_opp_first = sum(s.opp_first_seen for s in all_stats)
    total_switched = sum(s.switched_no_order_seen for s in all_stats)

    coverage_issues = []
    if total_immune == 0:
        coverage_issues.append("immune (0.0×) never seen — check team abilities")
    if total_normal == 0:
        coverage_issues.append("normal (1.0×) never seen")
    if total_se == 0:
        coverage_issues.append("super-effective (2.0×+) never seen")
    if total_we_first == 0:
        coverage_issues.append("we_moved_first=True never seen")
    if total_opp_first == 0:
        coverage_issues.append("we_moved_first=False never seen")
    if total_switched == 0:
        coverage_issues.append("switch turns (we_moved_first=None) never seen")

    print(f"\n{'=' * 65}")
    if total_mismatches == 0 and not coverage_issues:
        print("PASS — All five layers correct across all scenarios.")
        print(f"  Coverage confirmed: immune={total_immune} resisted={sum(s.resisted_seen for s in all_stats)}"
              f" normal={total_normal} SE={total_se}"
              f" we_first={total_we_first} opp_first={total_opp_first}"
              f" switches={total_switched}")
    else:
        print("FAIL:")
        if total_mismatches > 0:
            print(f"  Total mismatches: {total_mismatches}")
            for s in all_stats:
                sm = (
                    s.layer1_mismatches + s.layer2_mismatches
                    + s.layer3_mismatches + s.layer4_mismatches
                    + s.layer5_mismatches
                )
                if sm > 0:
                    print(f"    {s.name}: L1={s.layer1_mismatches} L2={s.layer2_mismatches}"
                          f" L3={s.layer3_mismatches} L4={s.layer4_mismatches}"
                          f" L5={s.layer5_mismatches}")
        for issue in coverage_issues:
            print(f"  Coverage gap: {issue}")
        sys.exit(1)
    print("=" * 65)


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    asyncio.run(main(n))
