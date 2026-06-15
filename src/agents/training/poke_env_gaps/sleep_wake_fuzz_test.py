"""Bridge-backed fuzz validation for the gen3_sleep_wake_belief_v1 obs feature.

Runs real gen3ou battles in-process (local BattleStream bridge, no server) and validates the
sleep WAKE belief two ways — the second is the one that proves we got the gen3 RATES right:

1. **Per-decision obs wiring (exact).** For every currently-asleep mon, decode the 3-dim sleep
   block from the FULL obs vector and assert it equals an INDEPENDENT recompute:
   - ``sleep_is_deterministic`` == (the sleep's protocol ``[from]`` source is Rest),
   - ``sleep_counter_reliable`` == (no Sleep Talk / Snore seen this episode),
   - ``p_wake`` == ``sleep_wake_probability(poke_env_counter, is_rest, p_earlybird)``,
   and that poke-env's ``status_counter`` matches the protocol cant-count on clean episodes.

2. **Empirical calibration vs the real sim RNG (the rate check).** Reconstruct every CLEAN sleep
   episode (no sleep-usable move, no Early Bird) from the event log: each ``|cant|slp`` at cant-count
   K is an observed "did NOT wake at counter K", each ``|-curestatus|slp`` at count K is a "woke at
   counter K". Bucketed by (K, Rest-vs-opp), the empirical wake frequency must match the COMPUTED
   table within binomial tolerance — i.e. the gen3 sleep RNG (opp time∈{2,3,4,5}, Rest time=3) is
   priced correctly. This is independent of our code: the sim's PRNG decides the wake, our table
   predicts it.

Run directly (no server needed; in-process via the local bridge):
    export PYTHONPATH=$PYTHONPATH:src
    python src/agents/training/poke_env_gaps/sleep_wake_fuzz_test.py [n_battles]
"""
from __future__ import annotations

import asyncio
import math
import os
import sys
import time
import traceback
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from poke_env import AccountConfiguration
from poke_env.player import RandomPlayer
from poke_env.player.battle_order import ForfeitBattleOrder
from poke_env.player.player import Player
from poke_env.ps_client.server_configuration import LocalhostServerConfiguration

from agents.battle.battle_event import EventKind
from agents.battle.gen3_battle import Gen3Battle
from agents.observation.constants import (
    OFFSET_OUR_TEAM, OFFSET_OPP_TEAM, POKEMON_FULL_DIM,
    POKEMON_SLEEP_BELIEF_OFFSET, POKEMON_COUNTER_OFFSET,
)
from agents.observation.sleep_belief import (
    sleep_wake_probability, early_bird_probability, _reason_is_rest, _SLEEP_USABLE_MOVES,
)
from agents.observation.state_encoder import Gen3ObservationEncoder, load_mappings
from utils.bridge.local_battle_runner import run_local_battles
from utils.teambuilder import Gen3Teambuilder

BATTLE_FORMAT = "gen3ou"
_TOL = 1e-4
_TURN_CAP = 120   # forfeit past here so a sleep/Rest staller can't run to the 250-turn cap

# Curated SLEEP-HEAVY team (both sides) — to get statistical power on the OPPONENT-sleep tables
# fast. Four move-sleep inducers (Spore 100% / Hypnosis 60% / Sleep Powder 75% ×2) drive lots of
# random-duration opp sleeps; two clean Rest users (Suicune / Snorlax) drive deterministic ones.
# Deliberately NO Early Bird ability and NO Sleep Talk / Snore — so every episode is in the no-EB,
# clean-counter domain the verified tables cover (the calibration target).
SLEEP_TEAM = """\
Breloom @ Leftovers
Ability: Effect Spore
EVs: 4 HP / 252 Atk / 252 Spe
Adamant Nature
- Spore
- Sky Uppercut
- Mach Punch
- Hidden Power

Gengar @ Leftovers
Ability: Levitate
EVs: 4 HP / 252 SpA / 252 Spe
Timid Nature
- Hypnosis
- Shadow Ball
- Thunderbolt
- Ice Punch

Jumpluff @ Leftovers
Ability: Chlorophyll
EVs: 252 HP / 4 Def / 252 Spe
Timid Nature
- Sleep Powder
- Leech Seed
- Encore
- Hidden Power

Venusaur @ Leftovers
Ability: Overgrow
EVs: 252 HP / 4 SpA / 252 Spe
Timid Nature
- Sleep Powder
- Giga Drain
- Leech Seed
- Sludge Bomb

Suicune @ Leftovers
Ability: Pressure
EVs: 252 HP / 252 Def / 4 SpA
Bold Nature
- Surf
- Ice Beam
- Calm Mind
- Rest

Snorlax @ Leftovers
Ability: Thick Fat
EVs: 252 HP / 128 Atk / 128 SpD
Adamant Nature
- Body Slam
- Earthquake
- Shadow Ball
- Rest
"""
# Verified no-Early-Bird tables, recomputed here as the calibration target (independent of the impl
# via a fresh `sleep_wake_probability` call below — these are just the human-readable expectations).
_OPP_EXPECT = {0: 0.0, 1: 0.25, 2: 1.0 / 3.0, 3: 0.5, 4: 1.0}
_REST_EXPECT = {0: 0.0, 1: 0.0, 2: 1.0}


@dataclass
class _Episode:
    is_rest: bool
    cants: int = 0          # |cant|slp turns observed so far this episode
    sleep_usable: bool = False
    early_bird: bool = False


@dataclass
class _Stats:
    decisions: int = 0
    asleep_checks: int = 0
    wiring_fail: int = 0
    counter_mismatch: int = 0
    # calibration[(source, K)] -> [n_woke, n_total]
    calib: Dict[Tuple[str, int], List[int]] = field(default_factory=lambda: defaultdict(lambda: [0, 0]))
    examples: List[str] = field(default_factory=list)

    def note(self, msg: str) -> None:
        if len(self.examples) < 20:
            self.examples.append(msg)


class _SleepFuzzPlayer(Player):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("battle_class", Gen3Battle)   # event log + strict_view
        super().__init__(*args, **kwargs)
        self.encoder = Gen3ObservationEncoder(load_mappings())
        self.stats = _Stats()
        # per-battle-tag cursors + per (tag, side, species) live episode state
        self._cursor: Dict[str, int] = defaultdict(int)
        self._episodes: Dict[Tuple[str, str, str], _Episode] = {}

    # --- ground-truth from the event log (sim truth, independent of sleep_belief) -----------------
    def _process_events(self, battle) -> None:
        tag = battle.battle_tag
        events = battle.events
        cur = self._cursor[tag]
        for e in events[cur:]:
            if not e.side or not e.actor_species:
                continue
            key = (tag, e.side, e.actor_species)
            if e.kind is EventKind.STATUS and e.status == "slp":
                eb = self._is_early_bird(battle, e.side, e.actor_species)
                self._episodes[key] = _Episode(is_rest=_reason_is_rest(e.reason), early_bird=eb)
            elif e.kind is EventKind.MOVE and e.move_id in _SLEEP_USABLE_MOVES:
                ep = self._episodes.get(key)
                if ep is not None:
                    ep.sleep_usable = True
            elif e.kind is EventKind.CANT and e.reason == "slp":
                ep = self._episodes.get(key)
                if ep is not None:
                    self._record_calib(ep, ep.cants, woke=0)
                    ep.cants += 1
            elif e.kind is EventKind.CURESTATUS and e.status == "slp":
                ep = self._episodes.pop(key, None)
                if ep is not None:
                    self._record_calib(ep, ep.cants, woke=1)
        self._cursor[tag] = len(events)

    def _record_calib(self, ep: _Episode, k: int, woke: int) -> None:
        # CLEAN episodes only: a Sleep-Talk/Snore turn corrupts the counter (+3) and Early Bird
        # halves it — both are explicitly OUT of the no-EB table's domain.
        if ep.sleep_usable or ep.early_bird:
            return
        bucket = self.stats.calib[("rest" if ep.is_rest else "opp", k)]
        bucket[0] += woke
        bucket[1] += 1

    def _is_early_bird(self, battle, side: str, species: str) -> bool:
        team = battle.team if side == "ours" else battle.opponent_team
        for mon in team.values():
            if mon.species == species:
                return early_bird_probability(mon) > 0.0
        return False

    # --- per-decision obs wiring check ------------------------------------------------------------
    def _check_obs(self, battle, vec) -> None:
        s = self.stats
        for is_own, team, off in (
            (True, battle.team, OFFSET_OUR_TEAM),
            (False, battle.opponent_team, OFFSET_OPP_TEAM),
        ):
            side = "ours" if is_own else "opp"
            team_list = self.encoder.get_team_list(battle, is_opponent=not is_own)
            for i, mon in enumerate(team_list):
                if mon is None or i >= 6 or getattr(mon, "status", None) is None:
                    continue
                if str(getattr(mon.status, "name", mon.status)).lower() != "slp":
                    continue
                start = off + i * POKEMON_FULL_DIM
                det = vec[start + POKEMON_SLEEP_BELIEF_OFFSET]
                p_wake = vec[start + POKEMON_SLEEP_BELIEF_OFFSET + 1]
                reliable = vec[start + POKEMON_SLEEP_BELIEF_OFFSET + 2]

                ep = self._episodes.get((battle.battle_tag, side, mon.species))
                exp_rest = bool(ep.is_rest) if ep else False
                exp_usable = bool(ep.sleep_usable) if ep else False
                ctr = int(getattr(mon, "status_counter", 0) or 0)
                p_eb = early_bird_probability(mon)
                exp_p = sleep_wake_probability(ctr, exp_rest, p_eb)

                s.asleep_checks += 1
                ok = (
                    abs(det - (1.0 if exp_rest else 0.0)) < _TOL
                    and abs(reliable - (0.0 if exp_usable else 1.0)) < _TOL
                    and abs(p_wake - exp_p) < _TOL
                )
                if not ok:
                    s.wiring_fail += 1
                    s.note(f"WIRING {side}:{mon.species} ctr={ctr} rest={exp_rest} usable={exp_usable} "
                           f"obs=[det {det:.3f} p {p_wake:.3f} rel {reliable:.3f}] exp_p={exp_p:.3f}")
                # On a clean episode poke-env's counter must equal the protocol cant-count.
                if ep and not ep.sleep_usable and not ep.early_bird and ctr != ep.cants:
                    s.counter_mismatch += 1
                    s.note(f"COUNTER {side}:{mon.species} poke-env={ctr} protocol-cants={ep.cants}")

    def _validate(self, battle) -> None:
        self._process_events(battle)
        self.stats.decisions += 1
        vec = self.encoder.encode(battle)
        self._check_obs(battle, vec)

    def choose_move(self, battle):
        try:
            self._validate(battle)
        except Exception as e:  # noqa: BLE001
            print("\n🛑 [SLEEP-WAKE FUZZ CRITICAL FAILURE] 🛑")
            print(f"Battle {battle.battle_tag} turn {battle.turn}: {e}")
            traceback.print_exc()
            os._exit(1)
        # Cap battle length: a mutual sleep/Rest staller can run to 250 turns, whose huge
        # reconstruction command-log line overruns the bridge's asyncio readline buffer. Forfeiting
        # ends it well before that (we still validated every turn up to here).
        if battle.turn > _TURN_CAP:
            return ForfeitBattleOrder()
        return self.choose_random_move(battle)


def _report_and_assert(s: _Stats) -> None:
    print("=" * 70)
    print(f"decisions validated     : {s.decisions}")
    print(f"asleep-mon obs checks   : {s.asleep_checks}")
    print(f"obs wiring failures     : {s.wiring_fail}")
    print(f"counter mismatches      : {s.counter_mismatch}")
    print("\nCALIBRATION  (empirical wake freq vs the COMPUTED gen3 table):")
    print(f"  {'source':6} {'K':>2} {'n':>5} {'emp':>7} {'table':>7} {'|err|':>7}")
    calib_fail = []
    n_buckets_checked = 0
    for (source, k), (woke, total) in sorted(s.calib.items()):
        table = sleep_wake_probability(k, source == "rest", 0.0)
        emp = woke / total if total else float("nan")
        err = abs(emp - table) if total else float("nan")
        flag = ""
        # Assert only where we have enough samples; tolerance is a generous binomial band so a
        # passing run is meaningful but not flaky. K=0 (always 0) and Rest (deterministic) are exact.
        deterministic = table in (0.0, 1.0)
        min_n = 20 if deterministic else 60
        if total >= min_n:
            n_buckets_checked += 1
            tol = _TOL if deterministic else max(0.10, 2.5 * math.sqrt(max(table * (1 - table), 0.01) / total))
            if err > tol:
                flag = "  ❌"
                calib_fail.append(f"{source} K={k}: emp {emp:.3f} vs table {table:.3f} (n={total}, tol={tol:.3f})")
        print(f"  {source:6} {k:>2} {total:>5} {emp:>7.3f} {table:>7.3f} {err:>7.3f}{flag}")
    for ex in s.examples:
        print("   ·", ex)
    print("=" * 70)

    failed = False
    if s.wiring_fail:
        print(f"❌ FAIL — {s.wiring_fail} obs-wiring mismatch(es)"); failed = True
    if s.counter_mismatch:
        print(f"❌ FAIL — {s.counter_mismatch} poke-env-vs-protocol counter mismatch(es)"); failed = True
    if calib_fail:
        print(f"❌ FAIL — {len(calib_fail)} calibration bucket(s) off the gen3 table:")
        for c in calib_fail:
            print("    ", c)
        failed = True
    if failed:
        os._exit(1)
    if s.asleep_checks == 0:
        print("⚠️  PASS but NO asleep mon was observed — INCONCLUSIVE. Re-run with more battles "
              "(gen3ou Rest/Spore mons produce sleep).")
    elif n_buckets_checked < 3:
        print(f"⚠️  PASS (wiring clean over {s.asleep_checks} checks) but only {n_buckets_checked} "
              "calibration bucket(s) had enough samples — re-run with more battles for the rate check.")
    else:
        print(f"✅ PASS — {s.asleep_checks} asleep-mon obs checks exact; calibration matched the gen3 "
              f"table across {n_buckets_checked} well-sampled (K, source) buckets.")


async def run(n_battles: int) -> None:
    print(f"Sleep WAKE belief Fuzz — gen3ou — {n_battles} battles (curated sleep-heavy team)\n")
    teambuilder = Gen3Teambuilder(SLEEP_TEAM)
    ts = int(time.time())
    player = _SleepFuzzPlayer(
        battle_format=BATTLE_FORMAT, team=teambuilder,
        server_configuration=LocalhostServerConfiguration, start_listening=False,
        account_configuration=AccountConfiguration(f"SleepFuzz{ts}", "x"),
        max_concurrent_battles=10)
    opponent = RandomPlayer(
        battle_format=BATTLE_FORMAT, team=teambuilder,
        server_configuration=LocalhostServerConfiguration, start_listening=False,
        account_configuration=AccountConfiguration(f"SleepFuzzOpp{ts}", "x"),
        max_concurrent_battles=10)
    await run_local_battles(player, opponent, n_battles)
    _report_and_assert(player.stats)


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 150
    asyncio.run(run(n))
