"""
E2E fuzz test for HiddenPowerTracker candidate elimination.

Two-layer validation:
  1. INVARIANT: after every observe() call, every surviving candidate type
     satisfies effective_multiplier(type, our_mon) == observed_effectiveness.
  2. GROUND TRUTH: at battle end, for each opponent species that used HP, the
     true HP type (known from the fixed opponent team spec) must still be a
     non-zero candidate.

Setup:
  - Our team: one mon per immune ability (Volt Absorb, Water Absorb, Levitate,
    Flash Fire) + two neutral mons for coverage.
  - Opponent team: fixed mons each running a known HP type. The opponent player
    (HiddenPowerSpammer) biases toward choosing HP whenever available.

Run (requires: npm run showdown):
    export PYTHONPATH=$PYTHONPATH:src
    python src/agents/training/hidden_power_tracker_fuzz_e2e_test.py [n_battles]
"""
import asyncio
import os
import sys
import time
import traceback
from collections import Counter
from dataclasses import dataclass, field

import numpy as np
from poke_env import AccountConfiguration
from poke_env.player.player import Player
from poke_env.ps_client.server_configuration import LocalhostServerConfiguration

from agents.gen3_mechanics import bucket_effectiveness, effective_multiplier
from agents.training.hidden_power_tracker import HiddenPowerTracker, HIDDEN_POWER_TYPE_ORDER
from utils.teambuilder import Gen3Teambuilder


@dataclass(frozen=True)
class _HpTargetMon:
    """Tracker-shaped target with status from the DamagingMoveEvent.

    Mirrors EpisodeTracker._HpTargetMon. Duplicated deliberately: this fuzz is
    an E2E validator and importing the production wrapper would couple the
    test to a code path it's meant to verify.
    """
    species: str
    type_1: object
    type_2: object
    ability: object
    status: object

BATTLE_FORMAT = "gen3ou"

# ---------------------------------------------------------------------------
# Our team: cover all four ability immunities + neutral bulk
# ---------------------------------------------------------------------------
OUR_TEAM = """\
Lanturn @ Leftovers
Ability: Volt Absorb
EVs: 252 HP / 68 Def / 188 SpD
Bold Nature
- Surf
- Thunderbolt
- Ice Beam
- Thunder Wave

Vaporeon @ Leftovers
Ability: Water Absorb
EVs: 204 HP / 252 Def / 52 SpD
Bold Nature
- Surf
- Ice Beam
- Acid Armor
- Baton Pass

Weezing @ Leftovers
Ability: Levitate
EVs: 252 HP / 252 Def / 4 SpD
Bold Nature
- Sludge Bomb
- Thunderbolt
- Fire Blast
- Rest

Arcanine @ Leftovers
Ability: Flash Fire
EVs: 4 HP / 252 Atk / 252 Spe
Adamant Nature
- Flamethrower
- Extreme Speed
- Body Slam
- Iron Tail

Blissey (F) @ Leftovers
Ability: Natural Cure
EVs: 4 HP / 252 Def / 252 SpD
Bold Nature
- Soft-Boiled
- Ice Beam
- Thunder Wave
- Toxic

Snorlax @ Leftovers
Ability: Thick Fat
EVs: 252 HP / 16 Atk / 136 Def / 104 SpD
Careful Nature
- Body Slam
- Earthquake
- Rest
- Sleep Talk
"""

# Second team variant: keeps the three ability quirks the resolver depends on
# (Volt Absorb, Flash Fire, Thick Fat) but swaps the rest in for 4×-weak mons.
# This drives opp HP into the bucketed-2.0 path that revealed the production
# crash — Salamence 4× to Ice (Jolteon, Gengar), Swampert 4× to Grass (Zapdos,
# Raikou), Forretress 4× to Fire (Starmie, Alakazam).
OUR_TEAM_QUADWEAK = """\
Salamence @ Leftovers
Ability: Intimidate
EVs: 252 HP / 4 Atk / 252 Spe
Jolly Nature
- Dragon Claw
- Earthquake
- Fire Blast
- Rock Slide

Swampert @ Leftovers
Ability: Torrent
EVs: 240 HP / 196 Atk / 72 Spe
Adamant Nature
- Earthquake
- Surf
- Ice Beam
- Roar

Forretress @ Leftovers
Ability: Sturdy
EVs: 252 HP / 4 Atk / 252 SpD
Sassy Nature
- Spikes
- Rapid Spin
- Earthquake
- Explosion

Lanturn @ Leftovers
Ability: Volt Absorb
EVs: 252 HP / 68 Def / 188 SpD
Bold Nature
- Surf
- Thunderbolt
- Ice Beam
- Thunder Wave

Arcanine @ Leftovers
Ability: Flash Fire
EVs: 4 HP / 252 Atk / 252 Spe
Adamant Nature
- Flamethrower
- Extreme Speed
- Body Slam
- Iron Tail

Snorlax @ Leftovers
Ability: Thick Fat
EVs: 252 HP / 16 Atk / 136 Def / 104 SpD
Careful Nature
- Body Slam
- Earthquake
- Rest
- Sleep Talk
"""

# Third our-team variant — exercises 0.25× → bucketed 0.5 (Skarmory's Steel/Flying
# is double-resisted by Grass, hit at 0.25× by opp HP Grass) and a different 4× case
# (Gyarados 4× to Electric, paired with Tentacruel's HP Electric in OPP_TEAM_VARIETY).
OUR_TEAM_VARIETY = """\
Gyarados @ Leftovers
Ability: Intimidate
EVs: 156 HP / 100 Atk / 252 Spe
Adamant Nature
- Dragon Dance
- Earthquake
- Return
- Roar

Skarmory @ Leftovers
Ability: Sturdy
EVs: 252 HP / 232 Def / 24 Spe
Impish Nature
- Spikes
- Drill Peck
- Roar
- Rest

Tyranitar @ Leftovers
Ability: Sand Stream
EVs: 252 HP / 80 Atk / 176 SpD
Adamant Nature
- Rock Slide
- Earthquake
- Pursuit
- Roar

Vaporeon @ Leftovers
Ability: Water Absorb
EVs: 204 HP / 252 Def / 52 SpD
Bold Nature
- Surf
- Ice Beam
- Acid Armor
- Rest

Weezing @ Leftovers
Ability: Levitate
EVs: 252 HP / 252 Def / 4 SpD
Bold Nature
- Sludge Bomb
- Thunderbolt
- Fire Blast
- Rest

Blissey @ Leftovers
Ability: Natural Cure
EVs: 4 HP / 252 Def / 252 SpD
Bold Nature
- Soft-Boiled
- Ice Beam
- Thunder Wave
- Toxic
"""

# ---------------------------------------------------------------------------
# Opponent team: every mon runs HP with a known type.
# IVs are auto-corrected by Gen3Teambuilder via fix_gen3_hp_ivs.
# Ground truth: OPP_HP_GROUND_TRUTH maps species → true HP type (lowercase).
# ---------------------------------------------------------------------------
OPP_TEAM = """\
Jolteon @ Leftovers
Ability: Volt Absorb
EVs: 4 HP / 252 SpA / 252 Spe
Timid Nature
- Thunderbolt
- Hidden Power [Ice]
- Shadow Ball
- Agility

Starmie @ Leftovers
Ability: Natural Cure
EVs: 4 HP / 252 SpA / 252 Spe
Timid Nature
- Surf
- Hidden Power [Fire]
- Ice Beam
- Rapid Spin

Zapdos @ Leftovers
Ability: Pressure
EVs: 248 HP / 216 Def / 44 Spe
Bold Nature
- Thunderbolt
- Hidden Power [Grass]
- Roar
- Rest

Alakazam @ Lum Berry
Ability: Synchronize
EVs: 4 HP / 252 SpA / 252 Spe
Timid Nature
- Psychic
- Hidden Power [Fire]
- Shadow Ball
- Recover

Gengar @ Leftovers
Ability: Levitate
EVs: 4 HP / 252 SpA / 252 Spe
Timid Nature
- Shadow Ball
- Hidden Power [Ice]
- Thunderbolt
- Taunt

Raikou @ Leftovers
Ability: Pressure
EVs: 4 HP / 252 SpA / 252 Spe
Timid Nature
- Thunderbolt
- Hidden Power [Grass]
- Sleep Talk
- Rest
"""

# Second opp team variant — different HP-type coverage. Tentacruel runs HP Electric
# (only competitive HP Electric user), Celebi runs HP Fire. Ground truth for these
# new species is added to OPP_HP_GROUND_TRUTH; species shared with OPP_TEAM use the
# same HP type (consistency is required because the truth dict is per-species, not
# per-team).
OPP_TEAM_VARIETY = """\
Tentacruel @ Leftovers
Ability: Liquid Ooze
EVs: 4 HP / 252 SpA / 252 Spe
Timid Nature
- Surf
- Hidden Power [Electric]
- Ice Beam
- Toxic

Celebi @ Leftovers
Ability: Natural Cure
EVs: 252 HP / 4 Def / 252 SpD
Calm Nature
- Recover
- Hidden Power [Fire]
- Psychic
- Calm Mind

Jolteon @ Leftovers
Ability: Volt Absorb
EVs: 4 HP / 252 SpA / 252 Spe
Timid Nature
- Thunderbolt
- Hidden Power [Ice]
- Shadow Ball
- Substitute

Zapdos @ Leftovers
Ability: Pressure
EVs: 248 HP / 216 Def / 44 Spe
Bold Nature
- Thunderbolt
- Hidden Power [Grass]
- Roar
- Rest

Starmie @ Leftovers
Ability: Natural Cure
EVs: 4 HP / 252 SpA / 252 Spe
Timid Nature
- Surf
- Hidden Power [Fire]
- Ice Beam
- Rapid Spin

Alakazam @ Lum Berry
Ability: Synchronize
EVs: 4 HP / 252 SpA / 252 Spe
Timid Nature
- Psychic
- Hidden Power [Fire]
- Shadow Ball
- Recover
"""

# Third opp team variant — Forretress as the HP user (HP Bug, its dominant
# competitive spread per Smogon usage). Reproduces the original production crash
# scenario: a bulky spiker that switches frequently, opening Case A
# (prev_active newly fainted) and Case B (voluntary switch) target-resolution
# paths. Paired with other Bug-bucketing targets in OUR_TEAM_VARIETY (Skarmory
# Steel/Flying → 0.25× → bucket 0.5; Gengar / Weezing → 0.5×) the resolver gets
# heavy stress on switch-attributed HP observations.
OPP_TEAM_FORRETRESS = """\
Forretress (M) @ Leftovers
Ability: Sturdy
EVs: 252 HP / 4 Def / 252 SpD
Sassy Nature
- Spikes
- Rapid Spin
- Hidden Power [Bug]
- Explosion

Gengar @ Leftovers
Ability: Levitate
EVs: 4 HP / 252 SpA / 252 Spe
Timid Nature
- Shadow Ball
- Hidden Power [Ice]
- Thunderbolt
- Taunt

Jolteon @ Leftovers
Ability: Volt Absorb
EVs: 4 HP / 252 SpA / 252 Spe
Timid Nature
- Thunderbolt
- Hidden Power [Ice]
- Shadow Ball
- Substitute

Zapdos @ Leftovers
Ability: Pressure
EVs: 248 HP / 216 Def / 44 Spe
Bold Nature
- Thunderbolt
- Hidden Power [Grass]
- Roar
- Rest

Starmie @ Leftovers
Ability: Natural Cure
EVs: 4 HP / 252 SpA / 252 Spe
Timid Nature
- Surf
- Hidden Power [Fire]
- Ice Beam
- Rapid Spin

Alakazam @ Lum Berry
Ability: Synchronize
EVs: 4 HP / 252 SpA / 252 Spe
Timid Nature
- Psychic
- Hidden Power [Fire]
- Shadow Ball
- Recover
"""

# True HP types for each opponent species (lowercase, matches HIDDEN_POWER_TYPE_ORDER names)
OPP_HP_GROUND_TRUTH: dict[str, str] = {
    "jolteon":    "ice",
    "starmie":    "fire",
    "zapdos":     "grass",
    "alakazam":   "fire",
    "gengar":     "ice",
    "raikou":     "grass",
    "tentacruel": "electric",
    "celebi":     "fire",
    "forretress": "bug",
}


# ---------------------------------------------------------------------------
# Opponent player: prefer HP, otherwise random
# ---------------------------------------------------------------------------

class HiddenPowerSpammer(Player):
    """Always picks Hidden Power if available; falls back to random."""
    def choose_move(self, battle):
        for move in battle.available_moves:
            if move.id == "hiddenpower":
                return self.create_order(move)
        return self.choose_random_move(battle)


# ---------------------------------------------------------------------------
# Our fuzz player: tracks HP observations and validates both invariant + ground truth
# ---------------------------------------------------------------------------

@dataclass
class FuzzStats:
    battles: int = 0
    hidden_power_observations: int = 0
    invariant_checks_passed: int = 0
    ground_truth_checks: int = 0
    ground_truth_passed: int = 0
    ground_truth_failed: int = 0
    ground_truth_failures: list = field(default_factory=list)
    immunity_hits: Counter = field(default_factory=Counter)  # ability → count
    species_observations: Counter = field(default_factory=Counter)


class HiddenPowerTrackerFuzzPlayer(Player):
    """End-to-end validator for HiddenPowerTracker.

    Reads opp's last damaging move directly from poke-env's protocol-derived
    `battle.opp_last_damaging_move` property — the SAME signal the production
    HP code consumes. The fuzz's value here is no longer "alternate target
    resolution" (the protocol carries user/target/move/eff explicitly, so
    resolution is unnecessary). What it validates is:

      1. INVARIANT — after every observe(), every surviving HP candidate type
         satisfies effective_multiplier(type, target) == observed effectiveness.
      2. GROUND TRUTH — at battle end, for each opp species that ever used HP,
         the true HP type (known from the fixed opp team spec) is still a
         non-zero candidate. A bug in the tracker's narrowing logic would
         eliminate it.
      3. PROTOCOL CROSS-CHECK — every observed event's user_species and
         target_species are corroborated by an archived `|move|...|hidden
         power|...|target|` line, catching any drift between poke-env's
         protocol parser and the DamagingMoveEvent it emits.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stats = FuzzStats()
        self._trackers: dict[str, HiddenPowerTracker] = {}
        # Per-(battle, opp_species) observation log for diagnostic dumps.
        self._obs_log: dict[tuple, list] = {}
        # Per-battle raw protocol lines — used for both the protocol cross-check
        # invariant and the diagnostic dump on failure.
        self._proto_log: dict[str, list] = {}
        # Per-battle set of (turn, user_species, target_species) tuples we've
        # already observed, so we don't double-count when choose_move is called
        # multiple times in one turn (e.g. force-switch sub-calls).
        self._observed_keys: dict[str, set] = {}

    async def _handle_battle_message(self, split_messages):
        # Archive raw protocol lines per battle for the protocol cross-check
        # and the diagnostic dump on failure.
        current_tag = None
        for sm in split_messages:
            if len(sm) > 0 and sm[0].startswith(">"):
                current_tag = sm[0][1:]
                continue
            if current_tag is not None and len(sm) > 1:
                self._proto_log.setdefault(current_tag, []).append(tuple(sm))
        return await super()._handle_battle_message(split_messages)

    def _tracker(self, tag: str) -> HiddenPowerTracker:
        if tag not in self._trackers:
            self._trackers[tag] = HiddenPowerTracker()
        return self._trackers[tag]

    def _protocol_says_hp(
        self, tag: str, turn: int, user_species: str, target_species: str,
    ) -> bool:
        """Cross-check the DamagingMoveEvent against the raw protocol log:
        confirm there's a `|move|<user-side>: <UserName>|Hidden Power|<targ-side>: <TargName>|`
        line attributable to this turn. Returns True if found.
        """
        # Walk back through proto_log to find the most recent |turn|<turn>|
        # marker, then scan forward up to the next |turn|N+1| or end.
        plog = self._proto_log.get(tag, [])
        turn_str = str(turn)
        # Find the marker for `turn` (the turn during which the HP fired —
        # which is the one before the snapshot's current turn).
        for i, sm in enumerate(plog):
            if len(sm) >= 3 and sm[1] == "turn" and sm[2] == turn_str:
                # Scan forward until the next |turn| marker
                for sm2 in plog[i + 1:]:
                    if len(sm2) >= 3 and sm2[1] == "turn":
                        return False
                    if len(sm2) >= 5 and sm2[1] == "move":
                        # sm2 = ('', 'move', '<side>: <name>', '<move>', '<side>: <target>', ...)
                        if sm2[3].lower().replace(" ", "") == "hiddenpower":
                            mover = sm2[2].split(": ", 1)[-1].lower()
                            target = sm2[4].split(": ", 1)[-1].lower()
                            if mover == user_species and target == target_species:
                                return True
                return False
        return False

    def choose_move(self, battle):
        try:
            tag = battle.battle_tag
            tracker = self._tracker(tag)
            event = battle.opp_last_damaging_move
            if event is not None and event.move_id == "hiddenpower":
                # Dedupe: the same turn-gated event can show up on multiple
                # consecutive choose_move calls (force-switch sub-calls).
                dedupe_key = (battle.turn, event.user_species, event.target_species)
                if dedupe_key not in self._observed_keys.setdefault(tag, set()):
                    self._observed_keys[tag].add(dedupe_key)
                    self._observe_event(battle, tag, tracker, event)
            return self.choose_random_move(battle)
        except AssertionError:
            raise
        except Exception as e:
            print(f"\n[FUZZ FATAL] {battle.battle_tag} turn {battle.turn}: {e}", flush=True)
            traceback.print_exc()
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(1)

    def _observe_event(self, battle, tag, tracker, event):
        """Record one HP observation, enforce invariants, dump diagnostics on
        failure. Pulled out so choose_move stays a thin event-router."""
        target_mon = next(
            (m for m in battle.team.values() if m.species == event.target_species),
            None,
        )
        if target_mon is None:
            return
        # Wrap with the event's snapshot status (Flash Fire-vs-frozen).
        wrapped = _HpTargetMon(
            species=target_mon.species,
            type_1=target_mon.type_1,
            type_2=target_mon.type_2,
            ability=target_mon.ability,
            status=event.target_status,
        )

        # Protocol cross-check: confirm the event corresponds to a real
        # `|move|<user>|Hidden Power|<target>|` line in the just-ended turn.
        fired_turn = battle.turn - 1
        if not self._protocol_says_hp(
            tag, fired_turn, event.user_species, event.target_species,
        ):
            print(
                f"\n[FUZZ FATAL] {tag} turn {battle.turn}: DamagingMoveEvent "
                f"user={event.user_species} target={event.target_species} "
                f"move_id={event.move_id} has no matching |move|...|Hidden Power|...| "
                f"line in turn {fired_turn}'s protocol.",
                flush=True,
            )
            plog = self._proto_log.get(tag, [])
            print("[FUZZ FATAL] last 60 protocol lines:", flush=True)
            for sm in plog[-60:]:
                print(f"  {sm}", flush=True)
            os._exit(1)

        self.stats.hidden_power_observations += 1
        self.stats.species_observations[event.user_species] += 1
        if event.effectiveness == 0.0:
            ability = (wrapped.ability or "").lower()
            self.stats.immunity_hits[ability] += 1

        log_key = (tag, event.user_species)
        self._obs_log.setdefault(log_key, []).append(
            (battle.turn, event.effectiveness, wrapped.species,
             str(wrapped.type_1),
             str(wrapped.type_2) if wrapped.type_2 else None,
             wrapped.ability,
             str(wrapped.status) if wrapped.status else None)
        )
        try:
            tracker.observe(event.user_species, event.effectiveness, wrapped)
        except ValueError as e:
            print(
                f"\n[FUZZ DBG] turn={battle.turn} event={event} target_wrapped="
                f"{wrapped}",
                flush=True,
            )
            print(f"[FUZZ DBG] observation log for {event.user_species}:", flush=True)
            for entry in self._obs_log.get(log_key, []):
                print(f"  turn={entry[0]} eff={entry[1]}× target={entry[2]} "
                      f"types=({entry[3]}/{entry[4]}) ability={entry[5]} status={entry[6] if len(entry) > 6 else None}",
                      flush=True)
            true_type = OPP_HP_GROUND_TRUTH.get(event.user_species, "?")
            print(f"[FUZZ DBG] true HP type for {event.user_species}: {true_type}",
                  flush=True)
            plog = self._proto_log.get(tag, [])
            print("[FUZZ DBG] last 60 protocol lines:", flush=True)
            for sm in plog[-60:]:
                print(f"  {sm}", flush=True)
            raise

        # Invariant: every surviving candidate must produce the observed
        # effectiveness against this target.
        probs = tracker.get_probs(event.user_species)
        for i, prob in enumerate(probs):
            if prob > 0.0:
                hp_type = HIDDEN_POWER_TYPE_ORDER[i]
                actual = bucket_effectiveness(
                    effective_multiplier(hp_type, wrapped)
                )
                if actual != event.effectiveness:
                    raise AssertionError(
                        f"INVARIANT VIOLATED: {event.user_species} used HP on "
                        f"{wrapped.species} ({wrapped.type_1}/{wrapped.type_2}/"
                        f"{wrapped.ability}) effectiveness={event.effectiveness}, "
                        f"but {hp_type.name} survives with mult={actual}"
                    )
        self.stats.invariant_checks_passed += 1

    def _battle_finished_callback(self, battle) -> None:
        self.stats.battles += 1
        tag = battle.battle_tag
        tracker = self._trackers.get(tag)

        if tracker is not None:
            # Ground truth check: true HP type must still be in the candidate set
            for species, true_type in OPP_HP_GROUND_TRUTH.items():
                probs = tracker.get_probs(species)
                if not np.any(probs > 0):
                    continue  # HP never observed for this species this battle

                self.stats.ground_truth_checks += 1
                true_idx = next(
                    i for i, t in enumerate(HIDDEN_POWER_TYPE_ORDER)
                    if t.name.lower() == true_type
                )
                if probs[true_idx] > 0.0:
                    self.stats.ground_truth_passed += 1
                else:
                    self.stats.ground_truth_failed += 1
                    surviving = [
                        HIDDEN_POWER_TYPE_ORDER[i].name.lower()
                        for i, p in enumerate(probs)
                        if p > 0
                    ]
                    obs_log = self._obs_log.get((tag, species), [])
                    proto_log = self._proto_log.get(tag, [])
                    self.stats.ground_truth_failures.append({
                        "species": species,
                        "true_type": true_type,
                        "surviving": surviving,
                        "battle": tag,
                        "obs_log": list(obs_log),
                        "proto_log": list(proto_log),
                    })

        # Clean up per-battle state
        self._trackers.pop(tag, None)
        for key in list(self._obs_log):
            if key[0] == tag:
                self._obs_log.pop(key, None)
        self._proto_log.pop(tag, None)
        self._observed_keys.pop(tag, None)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

async def main(n_battles: int = 500) -> None:
    ts = int(time.time()) % 100000
    print(f"HiddenPowerTracker fuzz — gen3ou — {n_battles} battles", flush=True)

    # Rotate three our-team variants and three opp-team variants — 9 combinations
    # in total, each sampled randomly per battle by Gen3Teambuilder.yield_team():
    #   A. OUR_TEAM             — ability-immunity coverage (Volt Absorb / Water Absorb /
    #                             Levitate / Flash Fire / Thick Fat / Natural Cure)
    #   B. OUR_TEAM_QUADWEAK    — Salamence / Swampert / Forretress for 4× hits from
    #                             opp HP Ice / Grass / Fire respectively
    #   C. OUR_TEAM_VARIETY     — Skarmory for 0.25× hits from opp HP Grass, Gyarados
    #                             for 4× from opp HP Electric (Tentacruel in OPP_VARIETY)
    #   1. OPP_TEAM             — HP Ice / Fire / Grass coverage
    #   2. OPP_TEAM_VARIETY     — adds HP Electric (Tentacruel) and a second HP Fire
    #                             user (Celebi)
    #   3. OPP_TEAM_FORRETRESS  — Forretress with HP Bug as a bulky switching pivot,
    #                             reproducing the original production crash scenario
    #                             (resolver target-misidentification on switch turns)
    our_tb = Gen3Teambuilder([OUR_TEAM, OUR_TEAM_QUADWEAK, OUR_TEAM_VARIETY])
    opp_tb = Gen3Teambuilder([OPP_TEAM, OPP_TEAM_VARIETY, OPP_TEAM_FORRETRESS])

    fuzz = HiddenPowerTrackerFuzzPlayer(
        battle_format=BATTLE_FORMAT,
        team=our_tb,
        server_configuration=LocalhostServerConfiguration,
        account_configuration=AccountConfiguration(f"HPFz{ts}", "password"),
        max_concurrent_battles=8,
    )
    opp = HiddenPowerSpammer(
        battle_format=BATTLE_FORMAT,
        team=opp_tb,
        server_configuration=LocalhostServerConfiguration,
        account_configuration=AccountConfiguration(f"HPFo{ts}", "password"),
        max_concurrent_battles=8,
    )

    await fuzz.battle_against(opp, n_battles=n_battles)

    s = fuzz.stats
    print(f"\n{'=' * 65}")
    print(f"Battles                       : {s.battles}")
    print(f"HP observations               : {s.hidden_power_observations}")
    print(f"Invariant checks passed       : {s.invariant_checks_passed}")
    print(f"Ground truth checks           : {s.ground_truth_checks}")
    print(f"  Passed                      : {s.ground_truth_passed}")
    print(f"  FAILED                      : {s.ground_truth_failed}")

    if s.species_observations:
        print(f"\nObservations per opponent species:")
        for sp, cnt in sorted(s.species_observations.items(), key=lambda x: -x[1]):
            true_type = OPP_HP_GROUND_TRUTH.get(sp, "?")
            print(f"  {sp:<16} {cnt:5d}  (true: {true_type})")

    if s.immunity_hits:
        print(f"\nImmunity (0×) triggers by ability:")
        for ability, count in sorted(s.immunity_hits.items(), key=lambda x: -x[1]):
            print(f"  {ability:<20} {count}")

    if s.ground_truth_failures:
        print(f"\nGROUND TRUTH FAILURES (first 3 with full log):")
        for f in s.ground_truth_failures[:3]:
            print(f"\n  ===== {f['species']}: true={f['true_type']}, survivors={f['surviving']} =====")
            print(f"    battle={f['battle']}")
            print(f"    Observations for {f['species']}:")
            for entry in f.get("obs_log", []):
                print(f"      turn={entry[0]} eff={entry[1]}× target={entry[2]} "
                      f"types=({entry[3]}/{entry[4]}) ability={entry[5]} status={entry[6] if len(entry) > 6 else None}")
            turns_of_interest = {e[0] for e in f.get("obs_log", [])}
            # Dump protocol log around the buggy observation turns. For each
            # buggy turn T, find the index of '|turn|T' / '|turn|T-1' and dump
            # the lines between them.
            plog = f.get("proto_log", [])
            for obs_turn in sorted(turns_of_interest):
                print(f"    Raw protocol around turn {obs_turn-1}→{obs_turn}:")
                start_idx = next(
                    (i for i, sm in enumerate(plog)
                     if len(sm) >= 3 and sm[1] == "turn" and sm[2] == str(obs_turn - 1)),
                    None,
                )
                end_idx = next(
                    (i for i, sm in enumerate(plog)
                     if len(sm) >= 3 and sm[1] == "turn" and sm[2] == str(obs_turn + 1)),
                    len(plog),
                )
                if start_idx is not None:
                    for sm in plog[start_idx:end_idx]:
                        # Skip 'request' lines (long JSON)
                        if len(sm) >= 2 and sm[1] != "request":
                            print(f"      {sm}")

    print(f"{'=' * 65}")

    if s.ground_truth_failed > 0:
        print("FAIL — true HP type was eliminated from candidate set.")
        sys.exit(1)
    elif s.invariant_checks_passed != s.hidden_power_observations:
        diff = s.hidden_power_observations - s.invariant_checks_passed
        print(f"FAIL — {diff} invariant check(s) did not pass.")
        sys.exit(1)
    elif s.hidden_power_observations == 0:
        print("WARNING — no HP observations recorded; check opponent team and spammer logic.")
        sys.exit(1)
    else:
        print("PASS — all observations consistent; ground truth always survived.")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    asyncio.run(main(n))
