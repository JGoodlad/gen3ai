"""
E2E transition-coverage fuzz test for BattleContext / TurnDelta in gen3ou.

Runs four targeted scenario batches to exercise known edge cases:
  A — Explosion        : fainted-attacker gap in opp_last_move_id
  B — Rest / Sleep Talk: last_move persistence during sleep turns + delegated-move tracking
  C — Hyper Beam       : last_move persistence across recharge turns
  D — Roar / Whirlwind : phaze-induced switch — opp_move_id must be recovered from
                         opp_all_last_move_ids (the active slot changes to the new mon
                         before the snapshot, so opp_last_move_id reads None otherwise)

Key findings from runtime observation (confirmed by running this test):
  - |cant| turns (par full-para, flinch, confusion self-hit, sleep no-sleep-talk):
      cant_move() does NOT clear _is_last_used. So if the mon has used a move before,
      last_move persists from the prior turn. But if |cant| fires on the FIRST active
      turn (before any move was ever used), last_move is None. This is correct behavior.
  - Sleep Talk:
      When successful delegation fires, poke-env calls moved(delegated_move) and
      last_move = delegated move (e.g., "surf"). However, when Sleep Talk fails to
      delegate (e.g., picks a 0-PP move, all moves exhausted), only the first
      |move|SleepTalk message fires and last_move = "sleeptalk". Observed ~80 times
      in scenario B.
  - Recharge turns (Hyper Beam): last_move persists as "hyperbeam" on the forced
      recharge turn (|cant|recharge — cant_move() does not clear _is_last_used).
  - Baton Pass: triggers opp_active change => TurnDelta classifies as switch (correct).
  - Protect/Detect: standard |move| processing => last_move = "protect".

Classification of opp_last_move_id=None cases (not bugs):
  1. Explosion gap     — attacker fainted before new active has any last_move
  2. cant-move (est.)  — first active turn + |cant| (par/flinch/frz/slp-no-talk)
                         detected heuristically: revealed_moves didn't grow this turn
  3. True anomaly      — revealed_moves GREW this turn but last_move is still None
                         (should be 0; indicates a poke-env parsing gap)

TODO (poke-env / Explosion gap): When the opponent uses Explosion and faints, by the time
  _get_observation() reads opponent_active_pokemon.last_move the new switch-in is active.
  last_move is None for the switch-in. Fix requires forking AbstractBattle._parse_message
  to snapshot last_move before the |faint| message clears the active slot.

Run directly (requires: npm run showdown):
    export PYTHONPATH=$PYTHONPATH:src
    python src/agents/training/poke_env_gaps/transition_fuzz_e2e_test.py [n_battles]
"""

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
from utils.teambuilder import Gen3Teambuilder

BATTLE_FORMAT = "gen3ou"

# ---------------------------------------------------------------------------
# Scenario A — Explosion
# Multiple Pokémon with Explosion so the fainted-attacker gap fires often.
# Gengar, Claydol, Metagross all confirmed valid with Explosion in sample pool.
# ---------------------------------------------------------------------------
EXPLOSION_TEAM = """\
Gengar @ Leftovers
Ability: Levitate
EVs: 4 HP / 252 SpA / 252 Spe
Timid Nature
- Shadow Ball
- Thunderbolt
- Fire Punch
- Explosion

Claydol @ Leftovers
Ability: Levitate
EVs: 244 HP / 204 Atk / 32 SpA / 20 SpD / 8 Spe
Adamant Nature
- Rapid Spin
- Earthquake
- Psychic
- Explosion

Metagross @ Leftovers
Ability: Clear Body
EVs: 252 HP / 236 Atk / 20 Spe
Adamant Nature
- Meteor Mash
- Earthquake
- Explosion
- Brick Break

Skarmory @ Leftovers
Ability: Keen Eye
EVs: 252 HP / 4 Def / 252 Spe
Jolly Nature
- Spikes
- Roar
- Drill Peck
- Rest

Blissey (F) @ Leftovers
Ability: Natural Cure
EVs: 4 HP / 252 Def / 252 SpD
Bold Nature
- Soft-Boiled
- Ice Beam
- Thunder Wave
- Aromatherapy

Tyranitar @ Leftovers
Ability: Sand Stream
EVs: 252 HP / 40 Atk / 216 SpD
Careful Nature
- Rock Slide
- Earthquake
- Crunch
- Fire Blast
"""

# ---------------------------------------------------------------------------
# Scenario B — Rest + Sleep Talk
# Tests:
#   1. last_move persists as "rest" during the 2 forced sleep turns
#   2. Sleep Talk returns the delegated move (e.g. "surf"), not "sleeptalk"
#   3. Spore/Hypnosis ensure sleep activations happen frequently
# Both Suicune and Snorlax carry Rest+Sleep Talk so the nuance fires on both sides.
# ---------------------------------------------------------------------------
REST_SLEEP_TEAM = """\
Suicune @ Leftovers
Ability: Pressure
EVs: 240 HP / 244 Def / 24 Spe
Bold Nature
- Surf
- Ice Beam
- Rest
- Sleep Talk

Snorlax @ Leftovers
Ability: Thick Fat
EVs: 252 HP / 16 Atk / 136 Def / 104 SpD
Careful Nature
- Body Slam
- Earthquake
- Rest
- Sleep Talk

Smeargle @ Salac Berry
Ability: Own Tempo
EVs: 60 HP / 196 SpD / 252 Spe
Timid Nature
- Spore
- Explosion
- Spikes
- Will-O-Wisp

Gengar @ Leftovers
Ability: Levitate
EVs: 4 HP / 252 SpA / 252 Spe
Timid Nature
- Shadow Ball
- Hypnosis
- Thunderbolt
- Fire Punch

Blissey (F) @ Leftovers
Ability: Natural Cure
EVs: 4 HP / 252 Def / 252 SpD
Bold Nature
- Soft-Boiled
- Ice Beam
- Thunder Wave
- Aromatherapy

Skarmory @ Leftovers
Ability: Keen Eye
EVs: 252 HP / 4 Def / 252 Spe
Jolly Nature
- Spikes
- Roar
- Drill Peck
- Rest
"""

# ---------------------------------------------------------------------------
# Scenario C — Hyper Beam (recharge turns)
# Tests:
#   Hyper Beam: last_move persists as "hyperbeam" on the forced recharge turn
#   (|cant|recharge is sent — cant_move() does not clear _is_last_used).
#   Tyranitar and Regice both carry Hyper Beam; Salamence rounds out the team.
#
# NOTE: Outrage is not available in Gen 3 for Salamence (was introduced Gen 4).
# ---------------------------------------------------------------------------
LOCKED_TEAM = """\
Salamence @ Lum Berry
Ability: Intimidate
EVs: 4 HP / 252 Atk / 252 Spe
Adamant Nature
- Dragon Claw
- Earthquake
- Rock Slide
- Dragon Dance

Tyranitar @ Choice Band
Ability: Sand Stream
EVs: 4 HP / 252 Atk / 252 Spe
Adamant Nature
- Rock Slide
- Earthquake
- Hyper Beam
- Crunch

Regice @ Leftovers
Ability: Clear Body
EVs: 252 HP / 136 SpA / 120 SpD
Modest Nature
- Ice Beam
- Thunderbolt
- Hyper Beam
- Rest

Metagross @ Leftovers
Ability: Clear Body
EVs: 252 HP / 236 Atk / 20 Spe
Adamant Nature
- Meteor Mash
- Earthquake
- Explosion
- Brick Break

Blissey (F) @ Leftovers
Ability: Natural Cure
EVs: 4 HP / 252 Def / 252 SpD
Bold Nature
- Soft-Boiled
- Ice Beam
- Thunder Wave
- Aromatherapy

Skarmory @ Leftovers
Ability: Keen Eye
EVs: 252 HP / 4 Def / 252 Spe
Jolly Nature
- Spikes
- Roar
- Drill Peck
- Rest
"""

# ---------------------------------------------------------------------------
# Scenario D — Roar / Whirlwind (phazing)
# Tests:
#   When we use Roar or Whirlwind, the opponent moves first (Gen 3 phazing moves
#   have -6 priority), then their mon is forced out. TurnDelta must recover the
#   phazed mon's last_move from opp_all_last_move_ids rather than reading from
#   the newly-active mon (which has never moved and returns None).
#
# Both sides carry Roar/Whirlwind and a full bench so phazing fires from both
# sides frequently.
# ---------------------------------------------------------------------------
ROAR_TEAM = """\
Skarmory @ Leftovers
Ability: Keen Eye
EVs: 252 HP / 252 Def / 4 Spe
Impish Nature
- Whirlwind
- Spikes
- Steel Wing
- Toxic

Suicune @ Leftovers
Ability: Pressure
EVs: 252 HP / 252 Def / 4 Spe
Bold Nature
- Roar
- Surf
- Ice Beam
- Calm Mind

Blissey (F) @ Leftovers
Ability: Natural Cure
EVs: 4 HP / 252 Def / 252 SpD
Bold Nature
- Soft-Boiled
- Ice Beam
- Thunder Wave
- Aromatherapy

Tyranitar @ Leftovers
Ability: Sand Stream
EVs: 252 HP / 40 Atk / 216 SpD
Careful Nature
- Rock Slide
- Earthquake
- Crunch
- Fire Blast

Salamence @ Leftovers
Ability: Intimidate
EVs: 4 HP / 252 Atk / 252 Spe
Adamant Nature
- Dragon Claw
- Earthquake
- Rock Slide
- Dragon Dance

Metagross @ Leftovers
Ability: Clear Body
EVs: 252 HP / 236 Atk / 20 Spe
Adamant Nature
- Meteor Mash
- Earthquake
- Psychic
- Brick Break
"""


# ---------------------------------------------------------------------------
# Snapshot and stats
# ---------------------------------------------------------------------------

@dataclass
class TurnSnapshot:
    turn: int
    force_switch: bool
    our_active: str
    opp_active: str
    opp_status: Optional[str]           # sleep/burn/etc — for detecting sleep-state turns
    our_hp: np.ndarray                  # (6,)
    opp_hp: np.ndarray                  # (6,)
    our_fainted_count: int
    opp_fainted_count: int
    active_move_ids: list               # list[str | None] len 4
    opp_last_move_id: Optional[str]
    opp_all_last_move_ids: dict         # dict[str, str | None] — all opp mons' last_move
    opp_active_revealed_moves: frozenset


@dataclass
class ScenarioStats:
    name: str
    our_move_known: int = 0
    our_switch_known: int = 0
    our_move_slot_unknown: int = 0      # should always be 0
    opp_switch_known: int = 0
    opp_move_known: int = 0
    opp_move_unknown: int = 0
    opp_explosion_gap: int = 0          # opp fainted AND last_move=None (expected)
    opp_cant_move_estimated: int = 0    # revealed_moves didn't grow: probably |cant| (expected)
    opp_true_anomaly: int = 0           # revealed_moves GREW but last_move=None (should be 0)
    two_turn_same_move: int = 0         # last_move same on consecutive non-switch turns
    sleep_state_move_known: int = 0     # opp was asleep AND we have a move ID
    sleep_state_move_unknown: int = 0   # opp was asleep AND move ID is None
    sleeptalk_as_last_move: int = 0     # last_move=="sleeptalk" (delegation failed)
    # Phaze tracking (Scenario D)
    phaze_fired: int = 0                # turns where we used Roar/Whirlwind and opp species changed
    phaze_opp_move_captured: int = 0    # phaze turn where opp move was recovered correctly
    phaze_opp_move_missing: int = 0     # phaze turn where opp move is None (should be 0 after fix)
    opp_move_id_counts: Counter = field(default_factory=Counter)
    true_anomaly_details: list = field(default_factory=list)
    _prev_opp_move_id: Optional[str] = field(default=None, repr=False)
    _prev_opp_status: Optional[str] = field(default=None, repr=False)

    @property
    def total(self) -> int:
        return self.our_move_known + self.our_switch_known + self.our_move_slot_unknown


# ---------------------------------------------------------------------------
# Fuzz player
# ---------------------------------------------------------------------------

class TransitionFuzzPlayer(Player):
    def __init__(self, scenario_name: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stats = ScenarioStats(name=scenario_name)
        self._prev_snapshot: Optional[TurnSnapshot] = None
        self._last_action: Optional[int] = None

    def _build_snapshot(self, battle) -> TurnSnapshot:
        dec_ctx = getattr(battle, "_gen3_decision_context", None)
        if dec_ctx and dec_ctx.get("turn") == battle.turn:
            raw_ids = dec_ctx.get("move_ids", [])
        elif battle.last_request:
            active_req = battle.last_request.get("active", [{}])[0]
            raw_ids = [m.get("id") for m in active_req.get("moves", [])]
        else:
            raw_ids = []
        active_move_ids = (list(raw_ids) + [None, None, None, None])[:4]

        opp_mon = battle.opponent_active_pokemon
        opp_last_move = opp_mon.last_move if opp_mon else None

        # Status as a plain string for easy comparison
        opp_status = None
        if opp_mon and opp_mon.status:
            opp_status = opp_mon.status.name.lower()  # e.g. "slp", "brn", "par"

        our_hp = np.zeros(6, dtype=np.float32)
        opp_hp = np.zeros(6, dtype=np.float32)
        for i, mon in enumerate(list(battle.team.values())[:6]):
            our_hp[i] = mon.current_hp_fraction
        for i, mon in enumerate(list(battle.opponent_team.values())[:6]):
            opp_hp[i] = mon.current_hp_fraction

        opp_all_last_move_ids: dict = {}
        for mon in battle.opponent_team.values():
            lm = mon.last_move
            opp_all_last_move_ids[mon.species] = lm.id if lm else None

        return TurnSnapshot(
            turn=battle.turn,
            force_switch=battle.force_switch,
            our_active=(
                battle.active_pokemon.species
                if battle.active_pokemon and not battle.active_pokemon.fainted
                else "NONE"
            ),
            opp_active=opp_mon.species if opp_mon else "NONE",
            opp_status=opp_status,
            our_hp=our_hp,
            opp_hp=opp_hp,
            our_fainted_count=sum(1 for m in battle.team.values() if m.fainted),
            opp_fainted_count=sum(1 for m in battle.opponent_team.values() if m.fainted),
            active_move_ids=active_move_ids,
            opp_last_move_id=opp_last_move.id if opp_last_move else None,
            opp_all_last_move_ids=opp_all_last_move_ids,
            opp_active_revealed_moves=frozenset(opp_mon.moves.keys() if opp_mon else []),
        )

    def _analyze_transition(self, prev: TurnSnapshot, curr: TurnSnapshot, action: int):
        s = self.stats

        # --- Our action ---
        if action < 6:
            s.our_switch_known += 1
        elif action < 10:
            slot = action - 6
            move_id = prev.active_move_ids[slot]
            if move_id is not None:
                s.our_move_known += 1
            else:
                s.our_move_slot_unknown += 1
                s.true_anomaly_details.append({
                    "type": "our_move_slot_unknown",
                    "turn": curr.turn, "action": action,
                    "active_move_ids": prev.active_move_ids,
                })
        else:
            s.our_move_known += 1  # struggle

        # --- Opponent action ---
        opp_switched = prev.opp_active != curr.opp_active
        if opp_switched:
            # Check if this was a phaze we induced (Roar/Whirlwind have -6 priority
            # so the opponent always moves first before being forced out).
            our_move_id = prev.active_move_ids[action - 6] if 6 <= action < 10 else None
            if our_move_id in {"roar", "whirlwind"} and prev.opp_active != "NONE":
                s.phaze_fired += 1
                # Recover opp move from the full-team snapshot (the phazed mon retains
                # last_move even after being swapped out by poke-env).
                recovered = curr.opp_all_last_move_ids.get(prev.opp_active)
                if recovered is not None:
                    s.phaze_opp_move_captured += 1
                else:
                    s.phaze_opp_move_missing += 1
            else:
                s.opp_switch_known += 1
            s._prev_opp_move_id = None
            s._prev_opp_status = None
            return

        opp_move_id = curr.opp_last_move_id
        was_asleep = prev.opp_status == "slp"

        if opp_move_id is not None:
            s.opp_move_known += 1
            s.opp_move_id_counts[opp_move_id] += 1
            if opp_move_id == "sleeptalk":
                s.sleeptalk_as_last_move += 1

            # Two-turn persistence: same move_id on consecutive non-switch turns.
            # Fires for: recharge turns (hyperbeam→hyperbeam), |cant| turns (par, frz,
            # slp with last_move persisting), and any other cant-move scenario.
            if opp_move_id == s._prev_opp_move_id:
                s.two_turn_same_move += 1

            if was_asleep:
                s.sleep_state_move_known += 1
        else:
            s.opp_move_unknown += 1
            if curr.opp_fainted_count > prev.opp_fainted_count:
                # Explosion/Self-Destruct: attacker fainted, switch-in has no last_move.
                s.opp_explosion_gap += 1
            else:
                # Distinguish cant-move from true anomaly using revealed_moves growth.
                # If revealed_moves grew this turn, the opponent DEFINITELY used a move —
                # so last_move=None here is a real parsing gap (should be 0).
                # If it didn't grow, they probably couldn't move (|cant|: par/flinch/frz/slp).
                new_reveals = curr.opp_active_revealed_moves - prev.opp_active_revealed_moves
                if new_reveals:
                    s.opp_true_anomaly += 1
                    s.true_anomaly_details.append({
                        "type": "opp_new_move_but_no_last_move",
                        "turn": curr.turn,
                        "opp_active": prev.opp_active,
                        "opp_status": prev.opp_status,
                        "new_reveals": sorted(new_reveals),
                        "revealed_moves": sorted(curr.opp_active_revealed_moves),
                    })
                else:
                    s.opp_cant_move_estimated += 1

            if was_asleep:
                s.sleep_state_move_unknown += 1

        s._prev_opp_move_id = opp_move_id
        s._prev_opp_status = curr.opp_status

    def choose_move(self, battle):
        try:
            mask = Gen3ActionMasker.get_mask(battle)
            curr = self._build_snapshot(battle)

            if self._prev_snapshot is not None and self._last_action is not None:
                self._analyze_transition(self._prev_snapshot, curr, self._last_action)

            if battle.finished:
                self._prev_snapshot = None
                self._last_action = None
                valid = np.where(mask == 1)[0]
                return Gen3ActionMapper.action_to_order(int(np.random.choice(valid)), battle)

            valid = np.where(mask == 1)[0]
            choice = int(np.random.choice(valid))
            self._prev_snapshot = curr
            self._last_action = choice
            return Gen3ActionMapper.action_to_order(choice, battle)

        except Exception as e:
            print(f"\n[FUZZ FATAL] Battle {battle.battle_tag}, Turn {battle.turn}: {e}")
            traceback.print_exc()
            os._exit(1)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _pct(n: int, total: int) -> str:
    if total == 0:
        return f"{n:4d} (  n/a)"
    return f"{n:4d} ({100 * n / total:5.1f}%)"


def print_report(s: ScenarioStats) -> None:
    t = s.total
    sleep_total = s.sleep_state_move_known + s.sleep_state_move_unknown

    print(f"\n{'=' * 65}")
    print(f"SCENARIO: {s.name}")
    print(f"{'=' * 65}")
    print(f"Total transitions     : {t}")
    print(f"  Our action:")
    print(f"    Known move        : {_pct(s.our_move_known, t)}")
    print(f"    Known switch      : {_pct(s.our_switch_known, t)}")
    print(f"    Unknown slot [!]  : {_pct(s.our_move_slot_unknown, t)}  <- should be 0")
    print(f"  Opp action:")
    print(f"    Switch known      : {_pct(s.opp_switch_known, t)}")
    print(f"    Move known        : {_pct(s.opp_move_known, t)}")
    print(f"    Move unknown      : {_pct(s.opp_move_unknown, t)}")
    print(f"      Explosion gap   :   {s.opp_explosion_gap}  (expected — attacker fainted)")
    print(f"      Cant-move est.  :   {s.opp_cant_move_estimated}  (par/flinch/frz/slp, expected)")
    print(f"      True anomaly[!] :   {s.opp_true_anomaly}  <- new move revealed but no last_move; should be 0")

    print(f"  Persistence / nuance:")
    print(f"    Two-turn same move: {s.two_turn_same_move}  (recharge/sleep rest/|cant| persist)")
    print(f"    Sleep Talk failed : {s.sleeptalk_as_last_move}  (last_move==\"sleeptalk\"; delegation failed)")
    if sleep_total > 0:
        print(f"    Sleep-state turns : {sleep_total}")
        print(f"      Move known      : {_pct(s.sleep_state_move_known, sleep_total)}"
              "  (Sleep Talk delegated OR rest/recharge persisting)")
        print(f"      Move unknown    : {_pct(s.sleep_state_move_unknown, sleep_total)}"
              "  (sleep-no-talk: last_move=None on first active turn)")

    # Top-10 opponent move IDs seen
    if s.opp_move_id_counts:
        print(f"  Top opp move IDs seen:")
        for move_id, count in s.opp_move_id_counts.most_common(10):
            marker = "  <- Sleep Talk delegation FAILED" if move_id == "sleeptalk" else ""
            print(f"    {move_id:<20} {count:4d}{marker}")

    if s.phaze_fired > 0:
        print(f"  Phaze (Roar/Whirlwind):")
        print(f"    Phaze turns         : {s.phaze_fired}")
        print(f"    Opp move captured   : {_pct(s.phaze_opp_move_captured, s.phaze_fired)}")
        print(f"    Opp move missing[!] :   {s.phaze_opp_move_missing}  <- should be 0")

    if s.opp_true_anomaly > 0:
        print(f"  TRUE ANOMALIES (new move revealed but last_move=None):")
        for a in s.true_anomaly_details[:5]:
            if a.get("type") == "opp_new_move_but_no_last_move":
                print(f"    {a}")
        if s.opp_true_anomaly > 5:
            print(f"    ... and {s.opp_true_anomaly - 5} more")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

async def run_scenario(name: str, team_str: str, n_battles: int, ts: int) -> ScenarioStats:
    print(f"\n--- {name}: {n_battles} battles ---", flush=True)
    tb = Gen3Teambuilder(team_str)

    tag = name.replace("-", "").replace("/", "")[:6]
    fuzz = TransitionFuzzPlayer(
        scenario_name=name,
        battle_format=BATTLE_FORMAT,
        team=tb,
        server_configuration=LocalhostServerConfiguration,
        account_configuration=AccountConfiguration(f"TFz{ts}{tag}", "password"),
        max_concurrent_battles=5,
    )
    opp = RandomPlayer(
        battle_format=BATTLE_FORMAT,
        team=tb,
        server_configuration=LocalhostServerConfiguration,
        account_configuration=AccountConfiguration(f"TFo{ts}{tag}", "password"),
        max_concurrent_battles=5,
    )

    await fuzz.battle_against(opp, n_battles=n_battles)
    return fuzz.stats


async def main(n_battles: int = 50) -> None:
    ts = int(time.time()) % 100000
    print(f"Transition Fuzz Test — gen3ou — {n_battles} battles per scenario")

    scenarios = [
        ("A-Explosion", EXPLOSION_TEAM),
        ("B-Rest/SleepTalk", REST_SLEEP_TEAM),
        ("C-Outrage/HyperBeam", LOCKED_TEAM),
        ("D-Roar/Whirlwind", ROAR_TEAM),
    ]

    all_stats = []
    for name, team in scenarios:
        stats = await run_scenario(name, team, n_battles, ts)
        all_stats.append(stats)

    for s in all_stats:
        print_report(s)

    # Final verdict
    total_our_unknown = sum(s.our_move_slot_unknown for s in all_stats)
    total_true_anomaly = sum(s.opp_true_anomaly for s in all_stats)
    total_phaze_missing = sum(s.phaze_opp_move_missing for s in all_stats)
    total_phaze_fired = sum(s.phaze_fired for s in all_stats)

    print(f"\n{'=' * 65}")
    issues = total_our_unknown > 0 or total_true_anomaly > 0 or total_phaze_missing > 0
    if not issues:
        print("PASS — All transitions representable.")
        print("  Explosion gaps and cant-move cases are expected and classified correctly.")
        if total_phaze_fired > 0:
            print(f"  Phaze coverage: {total_phaze_fired} phaze turns observed across all scenarios.")
    else:
        print("ISSUES FOUND:")
        if total_our_unknown:
            print(f"  our_move_slot_unknown : {total_our_unknown}  (should be 0)")
        if total_true_anomaly:
            print(f"  opp true anomalies    : {total_true_anomaly}  (should be 0)")
            print("  -> new move was revealed this turn but opp_last_move_id is None")
            print("  -> indicates a poke-env parsing gap for those move types")
        if total_phaze_missing:
            print(f"  phaze_opp_move_missing: {total_phaze_missing}  (should be 0)")
            print("  -> we used Roar/Whirlwind but opp move was not recovered from opp_all_last_move_ids")
    print("=" * 65)


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    asyncio.run(main(n))
