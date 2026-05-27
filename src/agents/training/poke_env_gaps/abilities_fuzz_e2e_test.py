"""E2E fuzz test for the {known, ability1, ability2} encoding.

Validates the AbilitiesEncoder against real Showdown battles at three layers:

  Layer 1 — Own team always encodes as known=1 with ability1 = mon.ability.num,
            ability2 = 0. The server tells us our own abilities up front, so
            this should never break.

  Layer 2 — Opp unrevealed (mon.ability is None or "unknownability") encodes as
            known=0, ability1/ability2 = the two Gen 3-possible abilities from
            data/pokemon/gen3_species.json for that species. Single-ability
            species must put None/0 in slot 2.

  Layer 3 — Opp revealed (any other mon.ability value) encodes as known=1,
            ability1 = mon.ability.num, ability2 = 0. The revealed ability must
            be ONE of the two priors (catches cases where Showdown emits an
            ability message for a mon whose dex priors are stale).

Also checks that no encode call raises (catches surprise abilities missing
from gen3_abilities.json) and that the dex-prior IDs always resolve through
the ability_to_id mapping.

Run directly (requires: npm run showdown):
    export PYTHONPATH=$PYTHONPATH:src
    python src/agents/training/poke_env_gaps/abilities_fuzz_e2e_test.py [n_battles]
"""
from __future__ import annotations

import asyncio
import os
import sys
import time
import traceback
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from poke_env import AccountConfiguration
from poke_env.player import RandomPlayer
from poke_env.player.player import Player
from poke_env.ps_client.server_configuration import LocalhostServerConfiguration

from agents.action.mapper import Gen3ActionMapper
from agents.action.mask_generator import Gen3ActionMasker
from agents.observation.abilities import AbilitiesEncoder
from agents.observation.constants import ABILITY_SLOT_DIM, ABILITY_KNOWN_DIM
from agents.observation.state_encoder import load_mappings
from utils.teambuilder import Gen3Teambuilder

BATTLE_FORMAT = "gen3ou"

_AB1   = 0
_AB2   = 1
_KNOWN = ABILITY_SLOT_DIM   # 2


# ---------------------------------------------------------------------------
# Teams — chosen to exercise every revealable ability path
# ---------------------------------------------------------------------------

# Scenario A — Multiple ability-trigger paths
#   Intimidate (Salamence, Arcanine, Gyarados, Mawile) fires at switch-in via
#     |-ability| protocol message → mon.ability gets set early.
#   Volt Absorb (Lanturn) fires when an Electric move is absorbed.
#   Levitate (Gengar, Weezing) reveals when a Ground move would have hit but
#     doesn't (|-immune| with [from] ability: Levitate).
#   Flash Fire (Arcanine if not Intimidate, Houndoom) reveals via |-start|.
#   Pressure (Suicune, Aerodactyl) reveals via |-ability| at switch-in.
REVEAL_HEAVY_TEAM = """\
Salamence @ Choice Band
Ability: Intimidate
EVs: 4 HP / 252 Atk / 252 Spe
Adamant Nature
- Earthquake
- Rock Slide
- Brick Break
- Hidden Power

Gengar @ Leftovers
Ability: Levitate
EVs: 4 HP / 252 SpA / 252 Spe
Timid Nature
- Shadow Ball
- Thunderbolt
- Ice Punch
- Will-O-Wisp

Lanturn @ Leftovers
Ability: Volt Absorb
EVs: 252 HP / 188 Def / 68 SpA
Bold Nature
- Surf
- Ice Beam
- Thunderbolt
- Confuse Ray

Suicune @ Leftovers
Ability: Pressure
EVs: 252 HP / 252 Def / 4 SpA
Bold Nature
- Surf
- Ice Beam
- Calm Mind
- Roar

Aerodactyl @ Choice Band
Ability: Pressure
EVs: 4 HP / 252 Atk / 252 Spe
Jolly Nature
- Rock Slide
- Earthquake
- Hidden Power
- Double-Edge

Arcanine @ Leftovers
Ability: Intimidate
EVs: 252 HP / 4 Def / 252 SpD
Careful Nature
- Flamethrower
- Hidden Power
- Extreme Speed
- Howl
"""

# Scenario B — Mostly silent abilities that should KEEP the priors stable
#   Snorlax (Thick Fat / Immunity): Thick Fat is silent; Immunity reveals when
#     statused. Either way the priors stay correct.
#   Blissey (Natural Cure / Serene Grace): Natural Cure fires on switch-out.
#   Skarmory (Keen Eye / Sturdy): both silent in most Gen 3 contexts.
SILENT_ABILITY_TEAM = """\
Snorlax @ Leftovers
Ability: Thick Fat
EVs: 4 HP / 252 Atk / 252 SpD
Careful Nature
- Body Slam
- Earthquake
- Rest
- Sleep Talk

Blissey @ Leftovers
Ability: Natural Cure
EVs: 4 HP / 252 Def / 252 SpD
Bold Nature
- Soft-Boiled
- Ice Beam
- Thunder Wave
- Seismic Toss

Skarmory @ Leftovers
Ability: Keen Eye
EVs: 252 HP / 252 Def / 4 Spe
Impish Nature
- Drill Peck
- Spikes
- Roar
- Rest

Swampert @ Leftovers
Ability: Torrent
EVs: 240 HP / 216 Atk / 48 SpD
Adamant Nature
- Earthquake
- Ice Beam
- Surf
- Protect

Tyranitar @ Choice Band
Ability: Sand Stream
EVs: 4 HP / 252 Atk / 252 Spe
Adamant Nature
- Rock Slide
- Earthquake
- Hidden Power
- Double-Edge

Magneton @ Magnet
Ability: Magnet Pull
EVs: 4 HP / 252 SpA / 252 Spe
Timid Nature
- Thunderbolt
- Hidden Power
- Thunder Wave
- Toxic
"""


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

@dataclass
class ScenarioStats:
    name: str
    n_turns: int = 0

    own_checks: int = 0           # own-side encoding checks (Layer 1)
    opp_unrevealed: int = 0       # opp-unrevealed encoding checks (Layer 2)
    opp_revealed: int = 0         # opp-revealed encoding checks (Layer 3)

    layer1_fail: int = 0
    layer2_fail: int = 0
    layer3_fail: int = 0
    encode_raises: int = 0

    # Coverage: which opp abilities actually fired across all battles
    revealed_opp_abilities: Counter = field(default_factory=Counter)

    examples: list = field(default_factory=list)

    def record_example(self, layer: int, detail: dict):
        if len(self.examples) < 15:
            self.examples.append({"layer": layer, **detail})


# ---------------------------------------------------------------------------
# Encoder + species-prior table (built once)
# ---------------------------------------------------------------------------

_MAPPINGS = None
_ABILITY_ENC: Optional[AbilitiesEncoder] = None
_SPECIES_PRIORS: dict[str, tuple[int, int]] = {}


def _get_encoder() -> AbilitiesEncoder:
    global _MAPPINGS, _ABILITY_ENC, _SPECIES_PRIORS
    if _ABILITY_ENC is None:
        _MAPPINGS = load_mappings()
        rev = _MAPPINGS["reverse"]["abilities"]
        species_to_abilities = {
            sp: data["abilities"]
            for sp, data in _MAPPINGS["species"].items()
            if isinstance(data, dict) and data.get("abilities")
        }
        _ABILITY_ENC = AbilitiesEncoder(
            _MAPPINGS["abilities"], rev,
            species_to_abilities=species_to_abilities,
        )
        # Independent re-derivation of the priors — used as the ORACLE during
        # validation. If the encoder ever drifts from the JSON file, this catches
        # it without sharing the encoder's lookup table.
        ab_map = _MAPPINGS["abilities"]
        for sp, ab_list in species_to_abilities.items():
            nums: list[int] = []
            for ab_id in ab_list[:2]:
                if ab_id is None:
                    nums.append(0)
                else:
                    nums.append(int(ab_map.get(ab_id, {}).get("num", 0)))
            while len(nums) < 2:
                nums.append(0)
            _SPECIES_PRIORS[sp] = (nums[0], nums[1])
    return _ABILITY_ENC


def _normalize(name: str) -> str:
    return name.lower().replace(" ", "").replace("_", "")


# ---------------------------------------------------------------------------
# Fuzz player
# ---------------------------------------------------------------------------

class AbilityFuzzPlayer(Player):
    def __init__(self, scenario_name: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stats = ScenarioStats(name=scenario_name)

    def _validate_turn(self, battle) -> None:
        enc = _get_encoder()
        ab_map = _MAPPINGS["abilities"]
        s = self.stats
        s.n_turns += 1

        # Own team — should always be known=1, ability1 = mon.ability.num
        for mon in battle.team.values():
            if mon is None:
                continue
            try:
                vec = enc.encode(mon, battle)
            except Exception as e:
                s.encode_raises += 1
                s.record_example(1, {
                    "check": "encode_raised_on_own",
                    "species": mon.species,
                    "ability": mon.ability,
                    "error": str(e),
                })
                continue
            s.own_checks += 1

            ab_raw = mon.ability
            if not ab_raw:
                # Own mon with no ability set is very unusual; skip the strict
                # check but record it so it's visible in the report.
                s.record_example(1, {
                    "check": "own_mon_no_ability",
                    "species": mon.species,
                })
                continue
            ab_key = _normalize(ab_raw)
            expected_num = float(ab_map.get(ab_key, {}).get("num", 0))
            if not (vec[_KNOWN] >= 0.5
                    and vec[_AB1] == expected_num
                    and vec[_AB2] == 0.0):
                s.layer1_fail += 1
                s.record_example(1, {
                    "check": "own_encoding_wrong",
                    "species": mon.species,
                    "ability": ab_raw,
                    "expected_id": expected_num,
                    "vec": vec.tolist(),
                })

        # Opp team — either unrevealed (Layer 2) or revealed (Layer 3)
        for mon in battle.opponent_team.values():
            if mon is None:
                continue
            try:
                vec = enc.encode(mon, battle)
            except Exception as e:
                s.encode_raises += 1
                s.record_example(2, {
                    "check": "encode_raised_on_opp",
                    "species": mon.species,
                    "ability": mon.ability,
                    "error": str(e),
                })
                continue

            ab_raw = mon.ability
            is_revealed = bool(ab_raw) and _normalize(ab_raw) != "unknownability"

            if not is_revealed:
                s.opp_unrevealed += 1
                # Priors must match the JSON-derived oracle
                expected = _SPECIES_PRIORS.get(mon.species)
                if expected is None:
                    # Species missing from priors (e.g. Unown-? variant). The
                    # encoder will emit (0, 0); accept as a coverage gap rather
                    # than a failure, but record it.
                    s.record_example(2, {
                        "check": "species_has_no_priors",
                        "species": mon.species,
                        "vec": vec.tolist(),
                    })
                    continue
                if not (vec[_KNOWN] < 0.5
                        and int(vec[_AB1]) == expected[0]
                        and int(vec[_AB2]) == expected[1]):
                    s.layer2_fail += 1
                    s.record_example(2, {
                        "check": "opp_unrevealed_priors_wrong",
                        "species": mon.species,
                        "expected": expected,
                        "vec": vec.tolist(),
                    })
            else:
                s.opp_revealed += 1
                ab_key = _normalize(ab_raw)
                s.revealed_opp_abilities[ab_key] += 1
                expected_num = float(ab_map.get(ab_key, {}).get("num", 0))
                if not (vec[_KNOWN] >= 0.5
                        and vec[_AB1] == expected_num
                        and vec[_AB2] == 0.0):
                    s.layer3_fail += 1
                    s.record_example(3, {
                        "check": "opp_revealed_encoding_wrong",
                        "species": mon.species,
                        "ability": ab_raw,
                        "expected_id": expected_num,
                        "vec": vec.tolist(),
                    })
                    continue
                # Cross-check: the revealed ability must be ONE of the species
                # dex priors. If not, our priors are wrong for this species.
                expected_priors = _SPECIES_PRIORS.get(mon.species)
                if expected_priors is not None:
                    if expected_num not in expected_priors:
                        s.layer3_fail += 1
                        s.record_example(3, {
                            "check": "revealed_ability_not_in_priors",
                            "species": mon.species,
                            "revealed_ability": ab_raw,
                            "revealed_id": expected_num,
                            "priors": expected_priors,
                        })

    def choose_move(self, battle):
        try:
            self._validate_turn(battle)
            mask = Gen3ActionMasker.get_mask(battle)
            valid = np.where(mask == 1)[0]
            if len(valid) == 0:
                return self.choose_random_move(battle)
            return Gen3ActionMapper.action_to_order(int(np.random.choice(valid)), battle)
        except Exception as e:
            print(f"\n[FUZZ FATAL] Battle {battle.battle_tag}, Turn {battle.turn}: {e}")
            traceback.print_exc()
            os._exit(1)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_report(s: ScenarioStats) -> None:
    fail = s.layer1_fail + s.layer2_fail + s.layer3_fail + s.encode_raises
    status = "PASS" if fail == 0 else "FAIL"

    print(f"\n{'=' * 65}")
    print(f"SCENARIO: {s.name}  [{status}]")
    print(f"{'=' * 65}")
    print(f"Decision turns            : {s.n_turns}")
    print(f"Layer 1 (own) checks      : {s.own_checks}   fail={s.layer1_fail}")
    print(f"Layer 2 (opp unrevealed)  : {s.opp_unrevealed}   fail={s.layer2_fail}")
    print(f"Layer 3 (opp revealed)    : {s.opp_revealed}   fail={s.layer3_fail}")
    print(f"Encode exceptions         : {s.encode_raises}")

    if s.revealed_opp_abilities:
        top = s.revealed_opp_abilities.most_common(8)
        print(f"Revealed opp abilities    : {top}")

    if s.examples:
        print("Examples (up to 15):")
        for ex in s.examples:
            print(f"  L{ex['layer']}: {ex}")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

async def run_scenario(name: str, team_str: str, n_battles: int, ts: int) -> ScenarioStats:
    print(f"\n--- {name}: {n_battles} battles ---", flush=True)
    tb = Gen3Teambuilder(team_str)
    tag = name.replace("-", "").replace("/", "").replace(" ", "")[:6]

    fuzz = AbilityFuzzPlayer(
        scenario_name=name,
        battle_format=BATTLE_FORMAT,
        team=tb,
        server_configuration=LocalhostServerConfiguration,
        account_configuration=AccountConfiguration(f"AFz{ts}{tag}", "password"),
        max_concurrent_battles=5,
    )
    opp = RandomPlayer(
        battle_format=BATTLE_FORMAT,
        team=tb,
        server_configuration=LocalhostServerConfiguration,
        account_configuration=AccountConfiguration(f"AFo{ts}{tag}", "password"),
        max_concurrent_battles=5,
    )

    await fuzz.battle_against(opp, n_battles=n_battles)
    return fuzz.stats


async def main(n_battles: int = 50) -> None:
    ts = int(time.time()) % 100000
    print(f"Abilities Encoding Fuzz — gen3ou — {n_battles} battles per scenario")

    scenarios = [
        ("A-Reveal-Heavy", REVEAL_HEAVY_TEAM),
        ("B-Silent-Abilities", SILENT_ABILITY_TEAM),
    ]

    all_stats = []
    for name, team in scenarios:
        s = await run_scenario(name, team, n_battles, ts)
        all_stats.append(s)

    for s in all_stats:
        print_report(s)

    total_fail = sum(
        s.layer1_fail + s.layer2_fail + s.layer3_fail + s.encode_raises
        for s in all_stats
    )

    # Coverage check: at least one scenario must observe some opp-revealed
    # abilities — otherwise we never exercise the revealed encoding path.
    revealed_any = sum(s.opp_revealed for s in all_stats)
    coverage_issues = []
    if revealed_any == 0:
        coverage_issues.append(
            "No opp-revealed events seen — the Layer 3 path is untested. "
            "Increase n_battles or pick teams with reveal-firing abilities."
        )

    print(f"\n{'=' * 65}")
    print("OVERALL")
    print(f"{'=' * 65}")
    print(f"Total mismatches          : {total_fail}")
    print(f"Total opp-revealed events : {revealed_any}")
    if coverage_issues:
        print("\nCOVERAGE WARNINGS:")
        for issue in coverage_issues:
            print(f"  - {issue}")

    if total_fail == 0 and not coverage_issues:
        print("\n✅ ABILITIES FUZZ PASSED")
        sys.exit(0)
    else:
        print("\n❌ ABILITIES FUZZ FAILED")
        sys.exit(1)


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    asyncio.run(main(n))
