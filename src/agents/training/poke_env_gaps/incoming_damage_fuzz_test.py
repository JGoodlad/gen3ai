"""Bridge fuzz test for the incoming-damage / OHKO belief block (incoming_damage_v1).

Unit tests pin the belief MATH on hand-built ``Defender``/``AttackerThreat`` objects; this test
exercises the BATTLE EXTRACTION (``incoming_damage_encoder.encode_block``) against thousands of real
Showdown states drawn from the full gen3ou team pool — the only way to catch a poke-env-shaped gap
the math unit tests can't see: a species/move with no usage prior, a ``stats``/``boosts``/``effects``
read that surprises the extractor, an Explosion user or sand-setter the candidate builder mishandles.

Per-decision invariants asserted against the live battle (a violation raises immediately):
  1. ``encode_block`` never raises on a real state (catches missing dex/prior entries, attr gaps).
  2. The block has the right width and every value is finite.
  3. Ranges: P(KO)/P(outspeed) ∈ [0,1]; expected-damage-fraction ∈ [0,1.5]; recovery scalars ∈ [0,1].
     Fields are read by NAMED index (incoming_damage.IDX_*), pinning p_outspeed at IDX_OUTSPEED — the
     exact slot the reward PBRS reads — so a layout drift fails here, not silently in the reward.
  4. No opponent active (forced switch / battle start) ⟹ the whole block is zero.
  5. Our active behind a Substitute ⟹ its P(KO) rows are zero (the hit eats the Sub).
  6. A fainted / absent team slot ⟹ its per-mon slot is zero (no belief about a dead mon).

Scenario invariant (over a whole run):
  7. With an opponent active present, the block is non-zero for the large majority of decisions —
     the belief actually FIRES (revealed ∪ prior candidates exist for any known species). Catches a
     silent all-zeros regression that the per-decision range checks would happily pass.

Run directly (no server needed; runs in-process via the local BattleStream bridge):
    python src/agents/training/poke_env_gaps/incoming_damage_fuzz_test.py [n_battles]
    (in a linked worktree, first: export PYTHONPATH=$PYTHONPATH:src)
"""
from __future__ import annotations

import asyncio
import os
import sys
import time
import traceback
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from poke_env import AccountConfiguration
from poke_env.player import RandomPlayer
from poke_env.player.player import Player
from poke_env.ps_client.server_configuration import LocalhostServerConfiguration

from agents.action.mapper import Gen3ActionMapper
from agents.action.mask_generator import Gen3ActionMasker
# gen3_cpu_damage_deleted_v1: the block is no longer part of the OBSERVATION, so its dims come from
# the module that owns them. This test targets `encode_block` (the REWARD PBRS's source) directly,
# so it keeps its full value after the obs deletion.
from agents.observation.incoming_damage import (
    PER_MON as INCOMING_PER_MON, RECOVERY as INCOMING_RECOVERY_DIM,
)
from agents.observation.constants import (
    TEAM_SIZE,
)
from agents.observation.incoming_damage_encoder import encode_block
from agents.observation.incoming_damage import IDX_OUTSPEED, IDX_PHYS_PKO, IDX_SPEC_PKO
from agents.battle.live_view import LiveView
from utils.team_loader import TeamLoader
from utils.teambuilder import Gen3Teambuilder
from utils.bridge.local_battle_runner import run_local_battles

BATTLE_FORMAT = "gen3ou"
_EXP_MAX = 1.5          # the encoder clamps expected-damage-fraction to this
_NONZERO_FLOOR = 0.5    # ≥ this fraction of opp-active decisions must produce a non-zero block


@dataclass
class Stats:
    name: str
    n_turns: int = 0
    n_blocks: int = 0
    opp_active_decisions: int = 0
    nonzero_blocks: int = 0
    encode_raises: int = 0
    shape_fail: int = 0
    finite_fail: int = 0
    range_fail: int = 0
    zero_no_opp_fail: int = 0
    sub_fail: int = 0
    fainted_slot_fail: int = 0
    examples: list = field(default_factory=list)
    species_seen: Counter = field(default_factory=Counter)

    def record(self, ex: dict) -> None:
        if len(self.examples) < 15:
            self.examples.append(ex)

    @property
    def failures(self) -> int:
        return (self.encode_raises + self.shape_fail + self.finite_fail + self.range_fail
                + self.zero_no_opp_fail + self.sub_fail + self.fainted_slot_fail)


def _has_sub(mon) -> bool:
    return mon is not None and any(
        getattr(e, "name", "") == "SUBSTITUTE" for e in (getattr(mon, "effects", {}) or {}))


class IncomingDmgFuzzPlayer(Player):
    def __init__(self, scenario_name: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stats = Stats(name=scenario_name)

    def _validate_turn(self, battle) -> None:
        s = self.stats
        s.n_turns += 1
        # The belief now reads the LiveView read-model (not the raw battle); build it from this
        # plain poke-env battle exactly as the env's strict_view does. our_team stays
        # battle.team.values() order, which is identical to live.ours.mons order (the block's slot
        # alignment), so the per-slot invariant checks below still line up with the block.
        our_team = list(battle.team.values())
        try:
            live = LiveView.from_battle(battle)
            block = encode_block(live)
        except Exception as e:  # invariant 1
            s.encode_raises += 1
            s.record({"check": "encode_raised", "turn": battle.turn, "error": str(e)})
            return
        s.n_blocks += 1

        # invariant 2 — width + finiteness
        if block.shape[0] != (TEAM_SIZE * INCOMING_PER_MON + INCOMING_RECOVERY_DIM):
            s.shape_fail += 1
            s.record({"check": "wrong_width", "got": int(block.shape[0]), "want": (TEAM_SIZE * INCOMING_PER_MON + INCOMING_RECOVERY_DIM)})
            return
        if not np.all(np.isfinite(block)):
            s.finite_fail += 1
            s.record({"check": "non_finite", "turn": battle.turn, "block": block.tolist()})
            return

        # invariant 3 — per-mon + recovery ranges. Read fields by NAMED index (the single source of
        # truth in incoming_damage.py) — and pin p_outspeed at IDX_OUTSPEED specifically, since the
        # reward PBRS reads `block[base+IDX_OUTSPEED]`: a future field insert that shifts outspeed
        # would fail here loudly instead of silently mis-shaping the reward (the gen3_incoming_crit_
        # split layout drift that read phys_crit_delta as outspeed).
        for i in range(TEAM_SIZE):
            b = i * INCOMING_PER_MON
            slot = block[b:b + INCOMING_PER_MON]
            phys_pko = slot[IDX_PHYS_PKO]
            spec_pko = slot[IDX_SPEC_PKO]
            outspeed = slot[IDX_OUTSPEED]
            phys_exp, spec_exp = slot[0], slot[1]
            ok = (0.0 <= phys_pko <= 1.0 and 0.0 <= spec_pko <= 1.0 and 0.0 <= outspeed <= 1.0
                  and 0.0 <= phys_exp <= _EXP_MAX and 0.0 <= spec_exp <= _EXP_MAX)
            if not ok:
                s.range_fail += 1
                s.record({"check": "out_of_range", "slot": i, "vals": slot.tolist()})
        rec = block[TEAM_SIZE * INCOMING_PER_MON:]
        if not np.all((rec >= 0.0) & (rec <= 1.0)):
            s.range_fail += 1
            s.record({"check": "recovery_out_of_range", "rec": rec.tolist()})

        opp_active = battle.opponent_active_pokemon
        if opp_active is None:                      # invariant 4
            if block.any():
                s.zero_no_opp_fail += 1
                s.record({"check": "nonzero_without_opp", "turn": battle.turn})
            return
        s.opp_active_decisions += 1
        if block.any():
            s.nonzero_blocks += 1
        if getattr(opp_active, "species", None):
            s.species_seen[opp_active.species] += 1

        # invariant 5 — active behind a Substitute → its KO rows are zero
        active = battle.active_pokemon
        active_idx = None
        if active is not None:
            try:
                active_idx = our_team.index(active)
            except ValueError:
                active_idx = None
        if active_idx is not None and _has_sub(active):
            b = active_idx * INCOMING_PER_MON
            if block[b + IDX_PHYS_PKO] != 0.0 or block[b + IDX_SPEC_PKO] != 0.0:
                s.sub_fail += 1
                s.record({"check": "sub_not_zeroed", "slot": active_idx,
                          "vals": block[b:b + INCOMING_PER_MON].tolist()})

        # invariant 6 — fainted / absent slot → all-zero per-mon slot (no belief about a dead mon)
        for i in range(TEAM_SIZE):
            mon = our_team[i] if i < len(our_team) else None
            if mon is None or getattr(mon, "fainted", False):
                b = i * INCOMING_PER_MON
                if block[b:b + INCOMING_PER_MON].any():
                    s.fainted_slot_fail += 1
                    s.record({"check": "fainted_slot_nonzero", "slot": i,
                              "vals": block[b:b + INCOMING_PER_MON].tolist()})

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


def print_report(s: Stats) -> bool:
    nonzero_frac = (s.nonzero_blocks / s.opp_active_decisions) if s.opp_active_decisions else 1.0
    nonzero_ok = nonzero_frac >= _NONZERO_FLOOR or s.opp_active_decisions == 0
    passed = s.failures == 0 and nonzero_ok
    print(f"\n{'=' * 65}")
    print(f"SCENARIO: {s.name}  [{'PASS' if passed else 'FAIL'}]")
    print(f"{'=' * 65}")
    print(f"Decision turns            : {s.n_turns}")
    print(f"Blocks built              : {s.n_blocks}")
    print(f"Opp-active decisions       : {s.opp_active_decisions}")
    print(f"Non-zero blocks            : {s.nonzero_blocks}  ({nonzero_frac:.1%}, floor {_NONZERO_FLOOR:.0%})"
          + ("" if nonzero_ok else "   <-- TOO LOW"))
    print(f"encode raised             : {s.encode_raises}")
    print(f"width/finite/range fails  : {s.shape_fail} / {s.finite_fail} / {s.range_fail}")
    print(f"no-opp-zero / sub / fainted: {s.zero_no_opp_fail} / {s.sub_fail} / {s.fainted_slot_fail}")
    print(f"distinct opp species seen : {len(s.species_seen)}")
    if s.examples:
        print("Examples (up to 15):")
        for ex in s.examples:
            print(f"  {ex}")
    return passed


async def run_scenario(name: str, tb: Gen3Teambuilder, n_battles: int, ts: int) -> Stats:
    print(f"\n--- {name}: {n_battles} battles ---", flush=True)
    fuzz = IncomingDmgFuzzPlayer(
        scenario_name=name, battle_format=BATTLE_FORMAT, team=tb,
        server_configuration=LocalhostServerConfiguration, start_listening=False,
        account_configuration=AccountConfiguration(f"IDz{ts}", "password"),
        max_concurrent_battles=5,
    )
    opp = RandomPlayer(
        battle_format=BATTLE_FORMAT, team=tb,
        server_configuration=LocalhostServerConfiguration, start_listening=False,
        account_configuration=AccountConfiguration(f"IDo{ts}", "password"),
        max_concurrent_battles=5,
    )
    await run_local_battles(fuzz, opp, n_battles)
    return fuzz.stats


async def main(n_battles: int = 60) -> None:
    ts = int(time.time()) % 100000
    print(f"Incoming-damage block Fuzz — gen3ou — {n_battles} battles")
    pool = TeamLoader().get_all_teams() or TeamLoader().get_sample_teams()
    if not pool:
        print("[FUZZ FATAL] no teams in data/teams/")
        sys.exit(1)
    tb = Gen3Teambuilder(pool)
    stats = await run_scenario("FullPool", tb, n_battles, ts)
    passed = print_report(stats)
    print(f"\n{'OVERALL: PASS' if passed else 'OVERALL: FAIL'}")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    asyncio.run(main(n))
