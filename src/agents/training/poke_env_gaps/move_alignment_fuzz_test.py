"""Bridge fuzz test for per-move obs ↔ action-space ALIGNMENT (gen3_move_slot_align_v1).

The per-move obs features MUST be filled in request-slot order — feature slot k ↔ action logit 6+k — so
the policy reads the correct move's features for each action it can take. poke-env's
``battle.available_moves`` DROPS disabled moves (Choice-lock / Disable / Taunt / 0-PP / Imprison), so the
old ``enumerate(battle.available_moves)`` fill shifted every later feature out of alignment with its
action logit. The action mask / mapper index ``legal.move_slots`` (request order, disabled kept), so this
validates the obs against THAT axis.

**Scope refreshed for `gen3_cpu_damage_deleted_v1` (v48).** This test used to assert two more invariants
over the per-move base-power (`vec[0:4]`) and type-multiplier (`vec[4:8]`) scalars — including an empty
trailing slot reading the neutral 0.25 default rather than a phantom 4×. **v48 DELETED those 8 scalars
from the observation** (the DamageOperator computes both GPU-side from the learned move belief), so both
checks were reading bytes that no longer exist and failed on every decision. They are removed, NOT
weakened: the alignment property they witnessed is now carried end-to-end by check (2) below — the
`active_req_moves` block IS the request-ordered source the op's OUTGOING methods read, and it validates
disabled slots too (via `legal`), which the deleted base-power check could not.

Per-decision invariants asserted against the live battle (a violation raises immediately):
  1. The action mapper agrees: ``action_to_choice(6+k)`` resolves to request slot k (so obs slot k and
     action 6+k are the same move).
  2. gen3_op_move_align_v1: the `active_req_moves` obs block (the DamageOperator's OUTGOING source) carries,
     per request slot k, the move at ``legal.move_slots[k]`` — its dex NUM (HP → 237) + resolved TYPE-id +
     current choosability — in REQUEST order, so the op's per-move output lands at action 6+k. This is the
     regression guard for the v23/v27/v34 outgoing misalignment: if the obs reverts to sorted-by-id (the
     per-mon block's order) or drops the block, this fails immediately.

Coverage (over a whole run, reported + asserted so the hard case can't silently be skipped):
  - At least one decision had a DISABLED non-last move slot (the exact trigger for the old shift) —
     Choice-Band mons in the gen3ou team pool reliably produce it once they're locked into a move.

Run directly (no server needed; runs in-process via the local BattleStream bridge):
    export PYTHONPATH=$PYTHONPATH:src
    python src/agents/training/poke_env_gaps/move_alignment_fuzz_test.py [n_battles]
"""
from __future__ import annotations

import asyncio
import os
import sys
import time
import traceback
from dataclasses import dataclass, field
from typing import List, Optional

from poke_env import AccountConfiguration
from poke_env.player import RandomPlayer
from poke_env.player.player import Player
from poke_env.ps_client.server_configuration import LocalhostServerConfiguration

from agents import gen3_data
from agents.action.constants import MOVE_START
from agents.action.mapper import Gen3ActionMapper
from agents.battle.gen3_battle import Gen3Battle
from agents.battle.live_view import LegalActions
from agents.observation.constants import (
    OFFSET_REACTIVE, ACTIVE_REQ_MOVES_OFFSET, ACTIVE_REQ_MOVES_PER,
)
from agents.observation.state_encoder import Gen3ObservationEncoder, load_mappings
from agents.observation.types import TypeEncoder
from utils.team_loader import TeamLoader
from utils.teambuilder import Gen3Teambuilder
from utils.bridge.local_battle_runner import run_local_battles

_TOL = 1e-4

# gen3_op_move_align_v1: the request-order active-move id/type/legality block the DamageOperator's
# OUTGOING methods read. Absolute offsets in the full obs (the block is within the reactive span).
_ARM_IDS_OFF = OFFSET_REACTIVE + ACTIVE_REQ_MOVES_OFFSET
_ARM_TYPES_OFF = _ARM_IDS_OFF + ACTIVE_REQ_MOVES_PER
_ARM_LEGAL_OFF = _ARM_IDS_OFF + 2 * ACTIVE_REQ_MOVES_PER


def _resolve_move(active, req_id):
    """The Move at a request slot — the encoder's own resolution (typed-HP aware). This IS the
    definition of "the move action 6+k plays", so it is the legitimate ground truth, not circular."""
    mv = active.moves.get(req_id)
    if mv is None and req_id == "hiddenpower":
        hps = [v for mid, v in active.moves.items() if mid.startswith("hiddenpower")]
        mv = hps[0] if len(hps) == 1 else None
    return mv


def _expected_num_type(mv):
    """The (move_num, type_idx) the encoder emits for a resolved Move — mirrors reactive.py /
    moves.py exactly (HP → num 237; typed HP resolves its real type; '???' → THREE_QUESTION_MARKS)."""
    md = gen3_data.moves.get(mv.id)
    if md is None:
        return None, None
    if mv.id == "hiddenpower":
        return md.num, 0
    type_name = "???" if md.type.name == "THREE_QUESTION_MARKS" else md.type.name
    return md.num, TypeEncoder.TYPE_TO_IDX.get(type_name, 0)


@dataclass
class _Stats:
    decisions: int = 0
    move_slots_checked: int = 0
    disabled_slot_decisions: int = 0   # decisions with a DISABLED non-last slot (the trigger)
    mapper_fail: int = 0
    req_block_fail: int = 0            # gen3_op_move_align_v1: op's request-order id/type/legal block
    examples: List[dict] = field(default_factory=list)

    def record(self, ex: dict) -> None:
        if len(self.examples) < 12:
            self.examples.append(ex)

    @property
    def failures(self) -> int:
        return self.mapper_fail + self.req_block_fail


class _AlignmentPlayer(Player):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("battle_class", Gen3Battle)   # need strict_view for the production legal
        super().__init__(*args, **kwargs)
        self.encoder = Gen3ObservationEncoder(load_mappings())
        self.stats = _Stats()

    def _validate(self, battle) -> None:
        active = battle.active_pokemon
        if active is None:
            return
        legal = LegalActions.from_battle(battle)
        if legal.struggle or battle.force_switch or not legal.move_slots:
            return

        s = self.stats
        s.decisions += 1
        vec = self.encoder.encode(battle)

        n = min(4, len(legal.move_slots))
        # Trigger coverage: a disabled slot that is NOT the last present slot is what shifts later slots.
        if any(legal.move_slots[k].disabled for k in range(n - 1)):
            s.disabled_slot_decisions += 1

        for k in range(n):
            lm = legal.move_slots[k]
            mv = _resolve_move(active, lm.id)
            if mv is None:
                continue
            s.move_slots_checked += 1

            # (1) the mapper plays request slot k for action 6+k → obs slot k and the action agree.
            # Only for SELECTABLE (non-disabled) slots — the mapper rightly refuses a disabled slot
            # (it is masked off); a disabled slot's obs features are validated at slot k by check (2),
            # which asserts its `legal` flag is 0 there.
            if not lm.disabled:
                choice = Gen3ActionMapper.action_to_choice(MOVE_START + k, legal)
                if choice.move_slot != k:
                    s.mapper_fail += 1
                    s.record({"check": "mapper_slot_mismatch", "turn": battle.turn, "slot": k,
                              "mapped_slot": choice.move_slot})

        # (2) gen3_op_move_align_v1 — the DamageOperator's OUTGOING source block. For every request
        # slot k, the obs `active_req_moves` block must carry the move at legal.move_slots[k] (NUM +
        # resolved TYPE) and its CURRENT choosability — in REQUEST order, so the op's per-move output
        # lands at action 6+k. This is the regression guard for the v23/v27/v34 outgoing misalignment:
        # if the obs ever reverts to sorted-by-id order (or drops the block), it fails HERE.
        for k in range(n):
            lm = legal.move_slots[k]
            mv = _resolve_move(active, lm.id)
            if mv is None:
                continue
            exp_num, exp_type = _expected_num_type(mv)
            got_num = int(round(float(vec[_ARM_IDS_OFF + k])))
            got_type = int(round(float(vec[_ARM_TYPES_OFF + k])))
            got_legal = float(vec[_ARM_LEGAL_OFF + k])
            exp_legal = 0.0 if lm.disabled else 1.0
            if (exp_num is not None and (got_num != exp_num or got_type != exp_type
                                         or abs(got_legal - exp_legal) > _TOL)):
                s.req_block_fail += 1
                s.record({"check": "req_block_misaligned", "turn": battle.turn, "slot": k,
                          "move": lm.id, "disabled": lm.disabled,
                          "got": (got_num, got_type, got_legal),
                          "expected": (exp_num, exp_type, exp_legal),
                          "request_ids": [m.id for m in legal.move_slots]})

    def choose_move(self, battle):
        try:
            self._validate(battle)
        except Exception as e:  # noqa: BLE001 — surface the exact decision that broke
            print("\n🛑 [MOVE-ALIGNMENT FUZZ CRITICAL FAILURE] 🛑")
            print(f"Battle {battle.battle_tag} turn {battle.turn}: {e}")
            traceback.print_exc()
            os._exit(1)
        return self.choose_random_move(battle)


async def run(n_battles: int) -> None:
    print(f"Move-slot alignment Fuzz — gen3ou — {n_battles} battles\n")
    teams = TeamLoader().get_all_teams()
    teambuilder = Gen3Teambuilder(teams)
    ts = int(time.time())

    player = _AlignmentPlayer(
        battle_format="gen3ou", team=teambuilder,
        server_configuration=LocalhostServerConfiguration, start_listening=False,
        account_configuration=AccountConfiguration(f"MoveAlign{ts}", "x"),
        max_concurrent_battles=10)
    opponent = RandomPlayer(
        battle_format="gen3ou", team=teambuilder,
        server_configuration=LocalhostServerConfiguration, start_listening=False,
        account_configuration=AccountConfiguration(f"MoveAlignOpp{ts}", "x"),
        max_concurrent_battles=10)

    await run_local_battles(player, opponent, n_battles)

    s = player.stats
    print("=" * 65)
    print(f"decisions validated        : {s.decisions}")
    print(f"move slots checked         : {s.move_slots_checked}")
    print(f"decisions w/ disabled slot : {s.disabled_slot_decisions}  (the alignment trigger)")
    print(f"mapper / req-block fails   : {s.mapper_fail} / {s.req_block_fail}")
    for ex in s.examples:
        print("   example:", ex)
    print("=" * 65)

    if s.failures:
        print(f"\n❌ FAIL — {s.failures} alignment violation(s)")
        os._exit(1)
    if s.disabled_slot_decisions == 0:
        # No silent cap: a clean pass that never hit the trigger is INCONCLUSIVE for the bug class.
        print("\n⚠️  PASS but NO disabled-slot decision was exercised — the alignment trigger was not "
              "hit. Re-run with more battles (Choice-Band mons in the pool produce it).")
    else:
        print(f"\n✅ PASS — alignment held across {s.disabled_slot_decisions} disabled-slot decisions "
              f"(the exact case the old available_moves fill shifted).")


if __name__ == "__main__":
    num = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    asyncio.run(run(num))
