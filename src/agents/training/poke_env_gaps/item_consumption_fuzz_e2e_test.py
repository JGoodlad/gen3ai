"""
E2E item-consumption fuzz test.

Validates the full item-consumption pipeline at two independent layers:

  Layer 1 — raw -enditem message vs poke-env Pokemon.consumed_item
             (confirms end_item() stores the item name we observed)
  Layer 2 — Pokemon state vs ItemsEncoder output
             (confirms consumed=1/known=1/id=correct for consumed items;
              known=1/consumed=0/id=correct for held items;
              all-zero for unrevealed items)

Also checks that the item_id in the encoding matches the known mapping, and
that held items never have their consumed bit accidentally set.

Runs three targeted scenarios:
  A — Lum Berry (status trigger): status-heavy teams ensure Lum Berry activates
      on both sides every few turns, giving dense -enditem coverage.
  B — HP Berries (Salac / Petaya / Sitrus): glass-cannon teams dip below the
      25%/50% thresholds frequently, exercising HP-triggered consumption.
  C — Held items (Leftovers / Choice Band): items that are never consumed;
      verifies held encoding never spuriously sets consumed=1.

Run directly (requires: npm run showdown):
    export PYTHONPATH=$PYTHONPATH:src
    python src/agents/training/poke_env_gaps/item_consumption_fuzz_e2e_test.py [n_battles]
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
from agents.observation.items import ItemsEncoder
from agents.observation.state_encoder import load_mappings
from agents.observation.constants import ITEM_ID_DIM, ITEM_KNOWN_DIM, ITEM_CONSUMED_DIM
from utils.teambuilder import Gen3Teambuilder

BATTLE_FORMAT = "gen3ou"

# Indices within the 3-dim item vector
_ID      = 0
_KNOWN   = ITEM_ID_DIM                      # 1
_CONSUMED = ITEM_ID_DIM + ITEM_KNOWN_DIM    # 2


# ---------------------------------------------------------------------------
# Teams
# ---------------------------------------------------------------------------

# Scenario A — Lum Berry (status cure, very reliable trigger)
# Both sides carry Lum Berry and have status-inflicting moves so berries fire
# on both sides every few turns.
LUM_BERRY_TEAM = """\
Gengar @ Lum Berry
Ability: Levitate
EVs: 4 HP / 252 SpA / 252 Spe
Timid Nature
- Shadow Ball
- Hypnosis
- Thunderbolt
- Fire Punch

Smeargle @ Lum Berry
Ability: Own Tempo
EVs: 252 HP / 4 Def / 252 Spe
Timid Nature
- Spore
- Will-O-Wisp
- Spikes
- Explosion

Jolteon @ Lum Berry
Ability: Volt Absorb
EVs: 4 HP / 252 SpA / 252 Spe
Timid Nature
- Thunderbolt
- Thunder Wave
- Shadow Ball
- Agility

Suicune @ Lum Berry
Ability: Pressure
EVs: 252 HP / 252 Def / 4 SpA
Bold Nature
- Surf
- Ice Beam
- Roar
- Calm Mind

Blissey @ Lum Berry
Ability: Natural Cure
EVs: 4 HP / 252 Def / 252 SpD
Bold Nature
- Thunder Wave
- Soft-Boiled
- Ice Beam
- Seismic Toss

Heracross @ Lum Berry
Ability: Guts
EVs: 4 HP / 252 Atk / 252 Spe
Adamant Nature
- Megahorn
- Brick Break
- Earthquake
- Rock Slide
"""

# Scenario B — HP Berries
# Salac Berry (≤25% HP → speed+1), Petaya Berry (≤25% HP → SpA+1),
# Sitrus Berry (≤50% HP → restore 25% HP). Hard-hitting teams on both sides
# push each other below thresholds frequently.
HP_BERRY_TEAM = """\
Salamence @ Salac Berry
Ability: Intimidate
EVs: 4 HP / 252 Atk / 252 Spe
Adamant Nature
- Dragon Claw
- Earthquake
- Rock Slide
- Dragon Dance

Starmie @ Petaya Berry
Ability: Natural Cure
EVs: 4 HP / 252 SpA / 252 Spe
Timid Nature
- Surf
- Ice Beam
- Thunderbolt
- Rapid Spin

Swampert @ Sitrus Berry
Ability: Torrent
EVs: 240 HP / 216 Atk / 48 SpD
Adamant Nature
- Earthquake
- Ice Beam
- Surf
- Protect

Gengar @ Petaya Berry
Ability: Levitate
EVs: 4 HP / 252 SpA / 252 Spe
Timid Nature
- Shadow Ball
- Thunderbolt
- Fire Punch
- Hypnosis

Tyranitar @ Salac Berry
Ability: Sand Stream
EVs: 4 HP / 252 Atk / 252 Spe
Adamant Nature
- Rock Slide
- Earthquake
- Crunch
- Dragon Dance

Heracross @ Sitrus Berry
Ability: Guts
EVs: 4 HP / 252 Atk / 252 Spe
Adamant Nature
- Megahorn
- Brick Break
- Earthquake
- Rock Slide
"""

# Scenario C — Held items that are never consumed
# Leftovers and Choice Band are passive; they never fire -enditem.
# Validates that consumed=0 stays clean for the full battle.
HELD_ITEM_TEAM = """\
Metagross @ Leftovers
Ability: Clear Body
EVs: 252 HP / 236 Atk / 20 Spe
Adamant Nature
- Meteor Mash
- Earthquake
- Explosion
- Brick Break

Tyranitar @ Choice Band
Ability: Sand Stream
EVs: 4 HP / 252 Atk / 252 Spe
Adamant Nature
- Rock Slide
- Earthquake
- Crunch
- Hyper Beam

Skarmory @ Leftovers
Ability: Keen Eye
EVs: 252 HP / 252 Def / 4 Spe
Impish Nature
- Drill Peck
- Spikes
- Roar
- Rest

Blissey @ Leftovers
Ability: Natural Cure
EVs: 4 HP / 252 Def / 252 SpD
Bold Nature
- Soft-Boiled
- Ice Beam
- Thunder Wave
- Seismic Toss

Swampert @ Leftovers
Ability: Torrent
EVs: 240 HP / 216 Atk / 48 SpD
Adamant Nature
- Earthquake
- Ice Beam
- Surf
- Protect

Salamence @ Choice Band
Ability: Intimidate
EVs: 4 HP / 252 Atk / 252 Spe
Adamant Nature
- Dragon Claw
- Earthquake
- Rock Slide
- Hidden Power
"""


# ---------------------------------------------------------------------------
# Raw event tracking
# ---------------------------------------------------------------------------

@dataclass
class _TurnItemEvents:
    """Item events observed in raw Showdown protocol messages for one turn."""
    enditem: list = field(default_factory=list)    # [(ident, item_raw)]
    item_gained: list = field(default_factory=list) # [(ident, item_raw)]


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

@dataclass
class ScenarioStats:
    name: str
    n_turns: int = 0

    # Enditem event accounting
    enditem_validated: int = 0       # events fully validated (both layers)
    enditem_swap_skipped: int = 0    # item replaced same turn (Trick) — skip consumed check
    enditem_mon_not_found: int = 0   # couldn't find the Pokemon object

    # Layer mismatch counts (all must be 0 to PASS)
    layer1_mismatches: int = 0       # raw -enditem vs Pokemon.consumed_item
    layer2_consumed_mismatches: int = 0  # consumed=1 encoding wrong
    layer2_held_mismatches: int = 0     # held item encoding wrong

    # Coverage
    consumed_our_side: int = 0
    consumed_opp_side: int = 0
    held_item_checks: int = 0        # mon-turns with a known held item verified

    examples: list = field(default_factory=list)

    def record_example(self, layer: int, detail: dict):
        if len(self.examples) < 10:
            self.examples.append({"layer": layer, **detail})


# ---------------------------------------------------------------------------
# Module-level encoder (initialised once)
# ---------------------------------------------------------------------------

_MAPPINGS = None
_ITEM_ENC = None

def _get_encoder() -> ItemsEncoder:
    global _MAPPINGS, _ITEM_ENC
    if _ITEM_ENC is None:
        _MAPPINGS = load_mappings()
        items = _MAPPINGS.get("items", {})
        rev = {v["num"]: k for k, v in items.items()}
        _ITEM_ENC = ItemsEncoder(items, reverse_mapping=rev)
    return _ITEM_ENC


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize(item_raw: str) -> str:
    return item_raw.lower().replace(" ", "").replace("_", "").replace("-", "")


def _find_mon(battle, ident: str):
    """
    Return the Pokemon object for a Showdown identifier like 'p1a: Gengar'.
    Matches by _name first (the display name poke-env stores), then by species.
    Returns None if the mon cannot be located.
    """
    role = battle.player_role
    if role is None or len(ident) < 5:
        return None
    ident_prefix = ident[:2]
    mon_name = ident[5:].strip()
    is_ours = (ident_prefix == role)
    team = battle.team if is_ours else battle.opponent_team

    for mon in team.values():
        name = getattr(mon, "_name", None) or ""
        if name.lower() == mon_name.lower():
            return mon
    # Fallback: species normalisation handles forme variants
    for mon in team.values():
        sp = getattr(mon, "species", "") or ""
        if sp.lower() == mon_name.lower():
            return mon
    return None


def _is_our_side(battle, ident: str) -> bool:
    role = battle.player_role or "p1"
    return ident[:2] == role


# ---------------------------------------------------------------------------
# Fuzz player
# ---------------------------------------------------------------------------

class ItemFuzzPlayer(Player):
    def __init__(self, scenario_name: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stats = ScenarioStats(name=scenario_name)
        self._curr: dict[str, _TurnItemEvents] = {}
        self._prev: dict[str, Optional[_TurnItemEvents]] = {}

    async def _handle_battle_message(self, split_messages):
        """Intercept raw Showdown messages before poke-env parses them."""
        battle_tag = split_messages[0][0].lstrip(">")

        if battle_tag not in self._curr:
            self._curr[battle_tag] = _TurnItemEvents()
            self._prev[battle_tag] = None

        curr = self._curr[battle_tag]

        for msg in split_messages[1:]:
            if len(msg) < 2:
                continue
            mt = msg[1]
            if mt == "turn":
                # Archive current turn's events; reset for incoming turn
                self._prev[battle_tag] = curr
                curr = _TurnItemEvents()
                self._curr[battle_tag] = curr
            elif mt == "-enditem" and len(msg) >= 4:
                # msg[2] = "p1a: Gengar", msg[3] = "Sitrus Berry"
                curr.enditem.append((msg[2], msg[3]))
            elif mt == "-item" and len(msg) >= 4:
                # Item revealed or restored (Trick, Recycle, Frisk, etc.)
                curr.item_gained.append((msg[2], msg[3]))

        # Let poke-env process normally (this sets mon.consumed_item etc.)
        await super()._handle_battle_message(split_messages)

    def _validate_turn(self, battle) -> None:
        """
        Validate item state for the turn that just completed.

        Called at the start of every choose_move() invocation so that the
        poke-env state has already been updated by the preceding message batch.
        """
        s = self.stats
        s.n_turns += 1

        battle_tag = battle.battle_tag
        prev = self._prev.get(battle_tag)
        if prev is None:
            return  # nothing to validate on the first turn

        enc = _get_encoder()

        # Idents that received a NEW item this turn (Trick / Switcheroo / Recycle).
        # For these, mon.consumed_item may have been cleared by item.setter when
        # the replacement item was assigned, so skip the consumed-state check.
        gained_idents = {ident for ident, _ in prev.item_gained}

        # ── Layer 1 + Layer 2 (consumed state) ─────────────────────────────
        for ident, item_raw in prev.enditem:
            mon = _find_mon(battle, ident)
            if mon is None:
                s.enditem_mon_not_found += 1
                continue

            if ident in gained_idents:
                s.enditem_swap_skipped += 1
                continue

            s.enditem_validated += 1
            if _is_our_side(battle, ident):
                s.consumed_our_side += 1
            else:
                s.consumed_opp_side += 1

            # Layer 1: poke-env must have stored consumed_item = raw item string
            actual = mon.consumed_item
            if actual != item_raw:
                s.layer1_mismatches += 1
                s.record_example(1, {
                    "check": "consumed_item_mismatch",
                    "ident": ident,
                    "expected": item_raw,
                    "actual": actual,
                    "mon_item_now": mon.item,
                })

            # Layer 2: encoder must produce [id, known=1, consumed=1]
            try:
                vec = enc.encode(mon, battle)
            except ValueError as e:
                # Item not in our mapping — flag it but don't abort
                s.record_example(2, {
                    "check": "item_not_in_mapping",
                    "ident": ident,
                    "item_raw": item_raw,
                    "error": str(e),
                })
                continue

            if vec[_CONSUMED] < 0.5:
                s.layer2_consumed_mismatches += 1
                s.record_example(2, {
                    "check": "consumed_bit_not_set",
                    "ident": ident,
                    "item_raw": item_raw,
                    "vec": vec.tolist(),
                    "mon_consumed_item": mon.consumed_item,
                })
            elif vec[_KNOWN] < 0.5:
                s.layer2_consumed_mismatches += 1
                s.record_example(2, {
                    "check": "known_bit_not_set_for_consumed",
                    "ident": ident,
                    "item_raw": item_raw,
                    "vec": vec.tolist(),
                })
            else:
                # Also verify the item ID is correct (if the item is in the mapping)
                item_key = _normalize(item_raw)
                if item_key in enc.item_to_id:
                    expected_id = float(enc.item_to_id[item_key].get("num", 0))
                    if vec[_ID] != expected_id:
                        s.layer2_consumed_mismatches += 1
                        s.record_example(2, {
                            "check": "wrong_id_for_consumed",
                            "ident": ident,
                            "item_raw": item_raw,
                            "expected_id": expected_id,
                            "actual_id": vec[_ID],
                        })

        # ── Layer 2 (held state) — check every observable mon ───────────────
        for team in (battle.team, battle.opponent_team):
            for mon in team.values():
                item = mon.item
                if not item:
                    continue
                item_key = _normalize(item)
                if item_key == "unknownitem":
                    continue
                if item_key not in enc.item_to_id:
                    continue  # unmapped held item — skip

                try:
                    vec = enc.encode(mon, battle)
                except ValueError:
                    continue

                s.held_item_checks += 1
                ok = vec[_KNOWN] >= 0.5 and vec[_CONSUMED] < 0.5
                if not ok:
                    s.layer2_held_mismatches += 1
                    s.record_example(2, {
                        "check": "held_item_encoding_wrong",
                        "species": mon.species,
                        "item": item,
                        "vec": vec.tolist(),
                        "consumed_item": mon.consumed_item,
                    })
                else:
                    # Verify item ID
                    expected_id = float(enc.item_to_id[item_key].get("num", 0))
                    if vec[_ID] != expected_id:
                        s.layer2_held_mismatches += 1
                        s.record_example(2, {
                            "check": "wrong_id_for_held",
                            "species": mon.species,
                            "item": item,
                            "expected_id": expected_id,
                            "actual_id": vec[_ID],
                        })

    def choose_move(self, battle):
        try:
            self._validate_turn(battle)

            mask = Gen3ActionMasker.get_mask(battle)
            if battle.finished:
                valid = np.where(mask == 1)[0]
                return Gen3ActionMapper.action_to_order(int(np.random.choice(valid)), battle)

            valid = np.where(mask == 1)[0]
            return Gen3ActionMapper.action_to_order(int(np.random.choice(valid)), battle)

        except Exception as e:
            print(f"\n[FUZZ FATAL] Battle {battle.battle_tag}, Turn {battle.turn}: {e}")
            traceback.print_exc()
            os._exit(1)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_report(s: ScenarioStats) -> None:
    total_mis = s.layer1_mismatches + s.layer2_consumed_mismatches + s.layer2_held_mismatches
    status = "PASS" if total_mis == 0 else "FAIL"

    print(f"\n{'=' * 65}")
    print(f"SCENARIO: {s.name}  [{status}]")
    print(f"{'=' * 65}")
    print(f"Total decision turns      : {s.n_turns}")
    print(f"Enditem events validated  : {s.enditem_validated}")
    print(f"  Our side                : {s.consumed_our_side}")
    print(f"  Opp side                : {s.consumed_opp_side}")
    print(f"  Swaps skipped (Trick)   : {s.enditem_swap_skipped}")
    print(f"  Mon not found           : {s.enditem_mon_not_found}")
    print(f"Held-item checks          : {s.held_item_checks}")
    print(f"Layer 1 mismatches        : {s.layer1_mismatches}  (raw -enditem vs consumed_item)")
    print(f"Layer 2 consumed mis.     : {s.layer2_consumed_mismatches}  (consumed encoding wrong)")
    print(f"Layer 2 held mis.         : {s.layer2_held_mismatches}  (held encoding wrong)")

    if s.examples:
        print(f"Mismatch examples (up to 10):")
        for ex in s.examples:
            print(f"  {ex}")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

async def run_scenario(name: str, team_str: str, n_battles: int, ts: int) -> ScenarioStats:
    print(f"\n--- {name}: {n_battles} battles ---", flush=True)
    tb = Gen3Teambuilder(team_str)
    tag = name.replace("-", "").replace("/", "").replace(" ", "")[:6]

    fuzz = ItemFuzzPlayer(
        scenario_name=name,
        battle_format=BATTLE_FORMAT,
        team=tb,
        server_configuration=LocalhostServerConfiguration,
        account_configuration=AccountConfiguration(f"IFz{ts}{tag}", "password"),
        max_concurrent_battles=5,
    )
    opp = RandomPlayer(
        battle_format=BATTLE_FORMAT,
        team=tb,
        server_configuration=LocalhostServerConfiguration,
        account_configuration=AccountConfiguration(f"IFo{ts}{tag}", "password"),
        max_concurrent_battles=5,
    )

    await fuzz.battle_against(opp, n_battles=n_battles)
    return fuzz.stats


async def main(n_battles: int = 50) -> None:
    ts = int(time.time()) % 100000
    print(f"Item Consumption Fuzz Test — gen3ou — {n_battles} battles per scenario")

    scenarios = [
        ("A-LumBerry", LUM_BERRY_TEAM),
        ("B-HP-Berries", HP_BERRY_TEAM),
        ("C-Held-Items", HELD_ITEM_TEAM),
    ]

    all_stats = []
    for name, team in scenarios:
        stats = await run_scenario(name, team, n_battles, ts)
        all_stats.append(stats)

    for s in all_stats:
        print_report(s)

    # Final verdict
    total_mis = sum(
        s.layer1_mismatches + s.layer2_consumed_mismatches + s.layer2_held_mismatches
        for s in all_stats
    )

    # Coverage: scenarios A and B must observe actual consumption events
    coverage_issues = []
    for s in all_stats:
        if s.name.startswith("A-") or s.name.startswith("B-"):
            if s.enditem_validated == 0:
                coverage_issues.append(
                    f"{s.name}: 0 consumption events validated — "
                    "berry items may be missing from gen3_items.json or never activated"
                )
        if s.name.startswith("C-"):
            if s.held_item_checks == 0:
                coverage_issues.append(
                    f"{s.name}: 0 held-item checks — "
                    "mons with known items not observed during battles"
                )

    print(f"\n{'=' * 65}")
    if total_mis == 0 and not coverage_issues:
        total_consumed = sum(s.enditem_validated for s in all_stats)
        total_held = sum(s.held_item_checks for s in all_stats)
        print("PASS — All layers correct across all scenarios.")
        print(f"  Consumption events validated : {total_consumed}")
        print(f"  Held-item checks             : {total_held}")
    else:
        print("FAIL:")
        if total_mis > 0:
            print(f"  Total mismatches: {total_mis}")
            for s in all_stats:
                sm = s.layer1_mismatches + s.layer2_consumed_mismatches + s.layer2_held_mismatches
                if sm > 0:
                    print(
                        f"    {s.name}: L1={s.layer1_mismatches} "
                        f"L2-consumed={s.layer2_consumed_mismatches} "
                        f"L2-held={s.layer2_held_mismatches}"
                    )
        for issue in coverage_issues:
            print(f"  Coverage gap: {issue}")
        sys.exit(1)
    print("=" * 65)


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    asyncio.run(main(n))
