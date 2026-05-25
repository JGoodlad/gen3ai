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
from poke_env.battle.pokemon import Pokemon
from poke_env.player import RandomPlayer
from poke_env.player.player import Player
from poke_env.ps_client.server_configuration import LocalhostServerConfiguration

from agents.gen3_mechanics import effective_multiplier
from agents.training.hidden_power_tracker import HiddenPowerTracker, HIDDEN_POWER_TYPE_ORDER
from utils.teambuilder import Gen3Teambuilder


@dataclass(frozen=True)
class _HpTargetMon:
    """Tracker-shaped target with status overridden to its value at HP-fire time.

    Mirrors EpisodeTracker._HpTargetMon. Duplicated deliberately: this fuzz is an
    *independent* validator of the tracker — sharing the resolver with production
    code would let a bug in either piece pass the test.
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

# True HP types for each opponent species (lowercase, matches HIDDEN_POWER_TYPE_ORDER names)
OPP_HP_GROUND_TRUTH: dict[str, str] = {
    "jolteon":   "ice",
    "starmie":   "fire",
    "zapdos":    "grass",
    "alakazam":  "fire",
    "gengar":    "ice",
    "raikou":    "grass",
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
    """Independent validator of HiddenPowerTracker.

    Reimplements HP-target resolution from raw battle state and asserts that
    every observation the tracker accepts is consistent with the live mon hit by
    the move (invariant) and that the true HP type for every opp species that
    used HP is still in the candidate set at battle end (ground truth).

    Resolution mirrors EpisodeTracker._resolve_hp_target by intent but takes a
    different path through the data — actions are detected from the chosen
    BattleOrder rather than from a TurnDelta. Sharing implementations would
    defeat the test (a bug in shared logic would slip through).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stats = FuzzStats()
        self._trackers: dict[str, HiddenPowerTracker] = {}
        self._prev_our_active: dict[str, str | None] = {}
        self._prev_opp_active: dict[str, str | None] = {}
        self._prev_our_fainted: dict[str, frozenset] = {}
        self._prev_action_was_switch: dict[str, bool] = {}
        # Per-mon status snapshot at the start of the most recent normal turn.
        # Used to evaluate Flash Fire-vs-frozen at the time the just-fired HP
        # actually resolved (the move thaws the target, so post-turn status lies).
        self._prev_our_team_status: dict[str, dict] = {}
        # Per-(battle, opp_species) list of (turn, effectiveness, target_species)
        # for diagnostic dump on observation failure.
        self._obs_log: dict[tuple, list] = {}
        # Per-battle list of (turn, force_sw, our_active_at_call, opp_active_at_call,
        # opp_last_move_id, opp_last_eff, action_was_switch, switch_to, we_first)
        self._turn_log: dict[str, list] = {}
        # Per-battle list of raw protocol lines, for protocol-level debugging.
        self._proto_log: dict[str, list] = {}

    async def _handle_battle_message(self, split_messages):
        # Intercept and archive raw protocol messages, then defer to base impl.
        for sm in split_messages:
            if len(sm) > 0 and sm[0].startswith(">"):
                tag = sm[0][1:]
            elif len(sm) > 1 and sm[1] in ("move", "switch", "-immune", "-supereffective",
                                            "-resisted", "-ability", "-activate",
                                            "faint", "turn"):
                # Use the most recent known tag (last battle we saw a header for)
                # The msg list usually starts with a tag header.
                pass
        # Walk through to find any battle tag header and accumulate
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

    @staticmethod
    def _resolve_target(battle, prev_active, prev_fainted, prev_was_switch,
                        prev_team_status):
        """Mirror of EpisodeTracker._resolve_hp_target using raw battle state.

        Driven by what *we* did, not visible side state — the active mon at
        turn N+1 start can be the same species as turn N start even after we
        switched out (switch-in died and forced-replace cycled the same mon back).

        Handles voluntary switches (priority +6, always before HP) and Baton Pass
        (move action that changes our active mid-turn; speed-based ordering vs HP).
        """
        curr_fainted = frozenset(
            m.species for m in battle.team.values() if m.fainted
        )
        newly_fainted = curr_fainted - prev_fainted

        # Case A: prev_active fainted → opp HP hit them on the field.
        if prev_active in newly_fainted:
            return next(
                (m for m in battle.team.values() if m.species == prev_active), None
            )

        curr_active_mon = battle.active_pokemon
        curr_active = (
            curr_active_mon.species
            if curr_active_mon and not curr_active_mon.fainted
            else None
        )
        visible_side_change = (prev_active != curr_active)
        # In Gen 3, only Baton Pass causes our side to change via a move action.
        is_baton_pass = (not prev_was_switch and visible_side_change)

        if prev_was_switch:
            switch_first = True   # voluntary switch — always priority +6
        elif is_baton_pass:
            switch_first = battle.we_moved_first is True   # speed decides
        else:
            switch_first = False

        if switch_first:
            # Switch-in (voluntary or via BP) took the HP. Either curr_active
            # (survived) or the newly-fainted mon != prev_active (KO'd by HP).
            switch_in_fainted = newly_fainted - {prev_active}
            if not switch_in_fainted:
                target_species = curr_active
            elif len(switch_in_fainted) == 1:
                target_species = next(iter(switch_in_fainted))
            else:
                raise RuntimeError(
                    f"HP target ambiguous: multiple newly-fainted mons "
                    f"{switch_in_fainted} (prev={prev_active}, curr={curr_active})."
                )
        else:
            # No switch action, or BP fired AFTER HP — prev_active took the hit.
            target_species = prev_active

        if target_species is None:
            return None
        live_mon = next(
            (m for m in battle.team.values() if m.species == target_species), None
        )
        if live_mon is None:
            return None
        # Override .status with the historical status at the start of the turn
        # HP fired in. See episode_tracker._resolve_hp_target for the reasoning.
        return _HpTargetMon(
            species=live_mon.species,
            type_1=live_mon.type_1,
            type_2=live_mon.type_2,
            ability=live_mon.ability,
            status=prev_team_status.get(target_species, live_mon.status),
        )

    def choose_move(self, battle):
        try:
            tag = battle.battle_tag
            tracker = self._tracker(tag)
            opp_mon = battle.opponent_active_pokemon
            opp_last_move = opp_mon.last_move if opp_mon else None
            opp_last_effectiveness = battle.opp_last_effectiveness

            prev_our = self._prev_our_active.get(tag)
            prev_opp = self._prev_opp_active.get(tag)
            prev_fainted = self._prev_our_fainted.get(tag, frozenset())
            prev_was_switch = self._prev_action_was_switch.get(tag, False)
            prev_team_status = self._prev_our_team_status.get(tag, {})

            # Skip HP processing on forced-switch calls. The opp_last_effectiveness
            # property gates on turn_set == self._turn - 1, so a forced switch mid-turn
            # would return None for the just-fired HP. But a forced switch triggered
            # by end-of-turn effects (poison etc.) AFTER battle.turn has ticked to N+1
            # WILL still return the value from turn N — and re-running our resolver at
            # that point would double-observe with stale prev state. The preceding
            # normal call already captured this HP correctly.
            if (not battle.force_switch
                    and prev_our is not None
                    and prev_opp is not None
                    and opp_last_move is not None
                    and opp_last_move.id == "hiddenpower"
                    and opp_last_effectiveness is not None):

                target_mon = self._resolve_target(
                    battle, prev_our, prev_fainted, prev_was_switch, prev_team_status
                )

                if target_mon is not None:
                    self.stats.hidden_power_observations += 1
                    self.stats.species_observations[prev_opp] += 1

                    if opp_last_effectiveness == 0.0:
                        ability = (getattr(target_mon, "ability", None) or "").lower()
                        self.stats.immunity_hits[ability] += 1

                    log_key = (tag, prev_opp)
                    self._obs_log.setdefault(log_key, []).append(
                        (battle.turn, opp_last_effectiveness, target_mon.species,
                         str(target_mon.type_1),
                         str(target_mon.type_2) if target_mon.type_2 else None,
                         target_mon.ability)
                    )
                    try:
                        tracker.observe(prev_opp, opp_last_effectiveness, target_mon)
                    except ValueError as e:
                        curr_active = (battle.active_pokemon.species
                                       if battle.active_pokemon else None)
                        force_sw = getattr(battle, "force_switch", None)
                        we_first = getattr(battle, "we_moved_first", None)
                        print(
                            f"\n[FUZZ DBG] turn={battle.turn} force_sw={force_sw} "
                            f"we_first={we_first} prev_our={prev_our} curr={curr_active} "
                            f"prev_was_switch={prev_was_switch} target={target_mon.species} "
                            f"eff={opp_last_effectiveness} opp={prev_opp} "
                            f"opp_move={opp_last_move.id}",
                            flush=True,
                        )
                        print(f"[FUZZ DBG] observation log for {prev_opp}:", flush=True)
                        for entry in self._obs_log.get(log_key, []):
                            print(f"  turn={entry[0]} eff={entry[1]}× target={entry[2]} "
                                  f"types=({entry[3]}/{entry[4]}) ability={entry[5]}",
                                  flush=True)
                        true_type = OPP_HP_GROUND_TRUTH.get(prev_opp, "?")
                        print(f"[FUZZ DBG] true HP type for {prev_opp}: {true_type}", flush=True)
                        # Print last 15 turn entries for context
                        recent = self._turn_log.get(tag, [])[-15:]
                        print(f"[FUZZ DBG] recent turns for {tag}:", flush=True)
                        for entry in recent:
                            print(f"  turn={entry[0]} fsw={entry[1]} our={entry[2]} "
                                  f"opp={entry[3]} opp_move={entry[4]} eff={entry[5]} "
                                  f"sw_action={entry[6]} sw_to={entry[7]} "
                                  f"we_first={entry[8]}", flush=True)
                        # Dump protocol around the crash turn
                        plog = self._proto_log.get(tag, [])
                        print(f"[FUZZ DBG] last 60 protocol lines:", flush=True)
                        for sm in plog[-60:]:
                            print(f"  {sm}", flush=True)
                        raise

                    # Invariant: every survivor is consistent with the observation.
                    probs = tracker.get_probs(prev_opp)
                    for i, prob in enumerate(probs):
                        if prob > 0.0:
                            hp_type = HIDDEN_POWER_TYPE_ORDER[i]
                            actual = effective_multiplier(hp_type, target_mon)
                            if actual != opp_last_effectiveness:
                                raise AssertionError(
                                    f"INVARIANT VIOLATED: {prev_opp} used HP on "
                                    f"{target_mon.species} "
                                    f"({target_mon.type_1}/{target_mon.type_2}/"
                                    f"{target_mon.ability}) "
                                    f"effectiveness={opp_last_effectiveness}, but "
                                    f"{hp_type.name} survives with mult={actual}"
                                )
                    self.stats.invariant_checks_passed += 1

            # Forced-switch calls happen mid-turn (after a mid-turn faint) and
            # don't represent the start of a new turn. We must NOT update prev_*
            # state here, or the next regular turn's HP resolution loses track of
            # who was active when opp HP fired.
            our_active_at_call = (
                battle.active_pokemon.species
                if battle.active_pokemon and not battle.active_pokemon.fainted
                else None
            )
            if not battle.force_switch:
                self._prev_our_active[tag] = our_active_at_call
                self._prev_opp_active[tag] = opp_mon.species if opp_mon else None
                self._prev_our_fainted[tag] = frozenset(
                    m.species for m in battle.team.values() if m.fainted
                )
                self._prev_our_team_status[tag] = {
                    m.species: m.status for m in battle.team.values()
                }
                order = self.choose_random_move(battle)
                action_is_switch = self._is_switch_order(order)
                self._prev_action_was_switch[tag] = action_is_switch
                switch_to = (order.order.species if action_is_switch
                             and hasattr(order, "order")
                             and hasattr(order.order, "species") else None)
                self._turn_log.setdefault(tag, []).append(
                    (battle.turn, False, our_active_at_call,
                     opp_mon.species if opp_mon else None,
                     opp_last_move.id if opp_last_move else None,
                     opp_last_effectiveness, action_is_switch, switch_to,
                     battle.we_moved_first)
                )
                return order
            else:
                order = self.choose_random_move(battle)
                switch_to = (order.order.species if hasattr(order, "order")
                             and hasattr(order.order, "species") else None)
                self._turn_log.setdefault(tag, []).append(
                    (battle.turn, True, our_active_at_call,
                     opp_mon.species if opp_mon else None,
                     opp_last_move.id if opp_last_move else None,
                     opp_last_effectiveness, True, switch_to,
                     battle.we_moved_first)
                )
                return order

        except AssertionError:
            raise
        except Exception as e:
            print(f"\n[FUZZ FATAL] {battle.battle_tag} turn {battle.turn}: {e}", flush=True)
            traceback.print_exc()
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(1)

    @staticmethod
    def _is_switch_order(order) -> bool:
        """True if the BattleOrder is a switch (sends in a Pokémon) rather than a move."""
        return isinstance(getattr(order, "order", None), Pokemon)

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
                    turn_log = self._turn_log.get(tag, [])
                    proto_log = self._proto_log.get(tag, [])
                    self.stats.ground_truth_failures.append({
                        "species": species,
                        "true_type": true_type,
                        "surviving": surviving,
                        "battle": tag,
                        "obs_log": list(obs_log),
                        "turn_log": list(turn_log),
                        "proto_log": list(proto_log),
                    })

        # Clean up per-battle state
        self._trackers.pop(tag, None)
        self._prev_our_active.pop(tag, None)
        self._prev_opp_active.pop(tag, None)
        self._prev_our_fainted.pop(tag, None)
        self._prev_action_was_switch.pop(tag, None)
        self._prev_our_team_status.pop(tag, None)
        for key in list(self._obs_log):
            if key[0] == tag:
                self._obs_log.pop(key, None)
        self._turn_log.pop(tag, None)
        self._proto_log.pop(tag, None)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

async def main(n_battles: int = 500) -> None:
    ts = int(time.time()) % 100000
    print(f"HiddenPowerTracker fuzz — gen3ou — {n_battles} battles", flush=True)

    our_tb = Gen3Teambuilder(OUR_TEAM)
    opp_tb = Gen3Teambuilder(OPP_TEAM)

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
                      f"types=({entry[3]}/{entry[4]}) ability={entry[5]}")
            # Dump turn log windows around each obs to show context
            turns_of_interest = {e[0] for e in f.get("obs_log", [])}
            tlog = f.get("turn_log", [])
            print(f"    Turn-by-turn context (turns appearing in obs):")
            for e in tlog:
                if e[0] in turns_of_interest or any(abs(e[0] - t) <= 1 for t in turns_of_interest):
                    print(f"      turn={e[0]} fsw={e[1]} our={e[2]} opp={e[3]} "
                          f"move={e[4]} eff={e[5]} sw_a={e[6]} sw_to={e[7]} "
                          f"we_first={e[8]}")
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
