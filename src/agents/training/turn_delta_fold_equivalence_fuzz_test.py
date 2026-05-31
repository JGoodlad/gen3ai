"""Fold-equivalence fuzz (Phase 5 — TurnDelta event-fold).

Proves that the PRODUCTION ``TurnDelta.build_from_events`` — which now folds the per-slot
HP deltas (and the target-HP attribution that rides on them) from the DAMAGE/HEAL/SETHP
event stream — is value-identical to the OLD snapshot-diff path, over real gen3ou battles
run in-process via the local BattleStream bridge (no server), for a sustained time budget.

At every decision the harness builds the SAME ``(prev_ctx, curr_ctx, action, events)`` window
two ways:

  * ``new``    — the production ``TurnDelta.build_from_events`` (HP/target folded from events).
  * ``frozen`` — ``_frozen_old_build_from_events`` below: a SELF-CONTAINED copy of the
                 pre-fold builder that snapshot-diffs ``curr_ctx.our_hp − prev_ctx.our_hp``.
                 Marked "frozen reference, NOT production"; it must survive any later edit to
                 the production fold, so it depends on nothing in ``turn_delta.py`` except the
                 ``TurnDelta`` dataclass + ``TurnView`` (the attribution fold, identical on
                 both paths).

Every ``TurnDelta`` field is compared exactly (``np.array_equal`` for the arrays). The only
field families that CAN differ are the re-sourced ones (per-slot HP deltas, HP-after, the two
target-HP scalars); everything else (moves / switches / cant / effectiveness / faints+causes /
status transitions / item-lost / outcome / crit / move-order / boosts) is computed identically
on both paths, so a diff there is a regression in the shared fold. HP magnitudes are
float-fraction subtractions — the harness flags a diff above ``_HP_FP_TOL`` as a REAL bug
(``os._exit(1)``) and, separately, counts any sub-tolerance float-rounding noise so a "0 real
diffs" run is trustworthy AND the float behaviour is visible.

Both deltas are ALSO run through ``Gen3RewardManager`` and every ``RewardBreakdown`` field is
diffed (reusing the comparison style of ``reward_resourcing_equivalence_fuzz_test.py``).

Coverage counters track the corner paths (boosts / CLEARBOOST / switch / Pain-Split-SETHP /
double-KO / multi-hit / cant) so a path that never fired is reported as a coverage hole.

Run directly (no server needed — local bridge):
    export PYTHONPATH=$PYTHONPATH:src
    python src/agents/training/turn_delta_fold_equivalence_fuzz_test.py [--minutes 15]
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import os
import sys
import time
import traceback
from collections import Counter
from typing import Optional

import numpy as np

from poke_env import AccountConfiguration
from poke_env.player import RandomPlayer
from poke_env.player.player import Player
from poke_env.ps_client.server_configuration import LocalhostServerConfiguration
from poke_env.battle.abstract_battle import DamagingMoveEvent

from agents.action.mapper import Gen3ActionMapper
from agents.action.mask_generator import Gen3ActionMasker
from agents.battle.gen3_battle import Gen3Battle
from agents.battle.battle_event import OURS, OPP, EventKind
from agents.training.battle_snapshot import BattleContext
from agents.training.turn_delta import TurnDelta
from agents.training.reward_manager import Gen3RewardManager
from agents.training.slot_registry import SlotRegistry
from agents.gen3_mechanics import BOOST_DIM
from utils.bridge.local_battle_runner import run_local_battles
from utils.team_loader import TeamLoader
from utils.teambuilder import Gen3Teambuilder

BATTLE_FORMAT = "gen3ou"

# A deliberately mechanic-dense (and Gen-3-legal) team so random play exercises every
# corner the fold must get right within one window: Pain Split (SETHP) + Haze (CLEARBOOST)
# on Weezing, Explosion (self-KO HP→0 with NO -damage line), Spikes/Roar (hazard + phaze),
# Calm Mind / Dragon Dance (boosts), Will-O-Wisp/Toxic/Thunder Wave (status → cant),
# Sand Stream (weather residual → multi-hit damage + weather faints). Both players run this
# team, so each mechanic fires on both sides. (Built on the validated reward-equivalence
# team with Metagross swapped for Weezing — the legal carrier of Haze + Pain Split.)
MIXED_TEAM = """\
Suicune @ Leftovers
Ability: Pressure
EVs: 240 HP / 244 Def / 24 Spe
Bold Nature
- Surf
- Ice Beam
- Calm Mind
- Rest

Tyranitar @ Leftovers
Ability: Sand Stream
EVs: 252 HP / 40 Atk / 216 SpD
Careful Nature
- Rock Slide
- Earthquake
- Crunch
- Dragon Dance

Gengar @ Leftovers
Ability: Levitate
EVs: 4 HP / 252 SpA / 252 Spe
Timid Nature
- Shadow Ball
- Thunderbolt
- Will-O-Wisp
- Explosion

Skarmory @ Leftovers
Ability: Keen Eye
EVs: 252 HP / 4 Def / 252 Spe
Jolly Nature
- Spikes
- Roar
- Protect
- Toxic

Blissey (F) @ Leftovers
Ability: Natural Cure
EVs: 4 HP / 252 Def / 252 SpD
Bold Nature
- Soft-Boiled
- Ice Beam
- Thunder Wave
- Aromatherapy

Weezing @ Leftovers
Ability: Levitate
EVs: 252 HP / 252 Def / 4 SpD
Bold Nature
- Sludge Bomb
- Haze
- Pain Split
- Toxic
"""

# HP deltas are differences of HP FRACTIONS (current_hp / max_hp). The event-sum fold and
# the endpoint snapshot diff are computed from the same fractions but in a different
# accumulation order, so a sub-ULP float32 mismatch is float noise, not a value bug; a
# larger gap is a genuine attribution/zeroing bug. Reward terms that ride HP inherit the
# same scale (×HP_VALUE=2), so the reward tolerance is set in step with it.
_HP_FP_TOL = 1e-5
_REWARD_FP_TOL = 1e-4

# Coverage paths that MUST fire for a "0 diffs" run to be trustworthy (the corners the
# fold has to get right). setboost (Belly Drum needs ≥50% HP) and item_used are reported
# but not required — they are bonus coverage, not load-bearing for the HP/attribution fold.
_REQUIRED_COVERAGE = (
    "boost", "clearboost", "switch", "pain_split_sethp", "double_ko", "multi_hit", "cant",
)


def _teams() -> list:
    """All Gen-3-legal sample teams (for species/move diversity) PLUS the mechanic-dense
    MIXED_TEAM (to guarantee the rare corners — Haze/CLEARBOOST, Pain Split/SETHP — fire).
    The curated team is weighted up a little so its corners are reached well inside the
    time budget even though the sample pool dominates the diversity."""
    loader = TeamLoader()
    pool = list(loader.get_sample_teams() or loader.get_all_teams())
    if not pool:
        raise RuntimeError("no gen3ou teams found under data/teams")
    # ~25% weight on the corner-carrier so CLEARBOOST/SETHP are hit early, the rest of the
    # pool covers diverse species/movesets the fold's attribution must handle.
    weight = max(1, len(pool) // 3)
    return pool + [MIXED_TEAM] * weight


# =====================================================================================
# FROZEN REFERENCE — a verbatim copy of the PRE-FOLD TurnDelta.build_from_events, with
# the per-slot HP deltas computed by SNAPSHOT DIFF (curr_ctx.*_hp − prev_ctx.*_hp + the
# newly-revealed-opp zeroing). This is the "old snapshot-diff builder". It is deliberately
# self-contained (only TurnDelta + TurnView) so it survives any edit to the production
# fold — do NOT refactor it to call production helpers.
# =====================================================================================
def _frozen_resolve_target_hp_delta(event, hp_delta, slot_map):
    if event is None:
        return None
    slot = slot_map.get(event.target_species)
    if slot is None:
        return None
    return float(hp_delta[slot])


def _frozen_old_build_from_events(prev_ctx, curr_ctx, action, events) -> TurnDelta:
    """FROZEN REFERENCE, NOT PRODUCTION — the pre-fold snapshot-diff HP path."""
    from agents.battle.turn_view import TurnView, FAINT_CAUSE_DIM, FAINT_CAUSE_VOCAB
    from agents.enums import Status
    _cause_idx = {c: i for i, c in enumerate(FAINT_CAUSE_VOCAB)}

    if action < 6:
        our_attempted_move_id = None
    elif action < 10:
        slot = action - 6
        ids = prev_ctx.active_move_ids
        our_attempted_move_id = ids[slot] if slot < len(ids) else None
    else:
        our_attempted_move_id = "struggle"

    # --- the OLD snapshot-diff HP path ---
    our_hp_delta = curr_ctx.our_hp - prev_ctx.our_hp
    opp_hp_delta = curr_ctx.opp_hp - prev_ctx.opp_hp
    for species, slot in curr_ctx.opp_slot_map.items():
        if species not in prev_ctx.opp_slot_map and opp_hp_delta[slot] > 0:
            opp_hp_delta[slot] = 0.0

    empty_faint_causes = np.zeros(FAINT_CAUSE_DIM, dtype=np.float32)

    if not events:
        we_fainted = curr_ctx.our_fainted_count > prev_ctx.our_fainted_count
        opp_fainted = curr_ctx.opp_fainted_count > prev_ctx.opp_fainted_count
        return TurnDelta(
            our_move_id=None, our_switch_to=None,
            our_prev_active=prev_ctx.our_active,
            opp_move_id=None, opp_switch_to=None,
            opp_prev_active=prev_ctx.opp_active,
            opp_move_known=False,
            our_hp_delta=our_hp_delta, opp_hp_delta=opp_hp_delta,
            we_fainted=we_fainted, opp_fainted=opp_fainted,
            our_failed_to_move=False, our_cant_reason=None,
            opp_failed_to_move=False, opp_cant_reason=None,
            our_boost_delta=np.zeros(BOOST_DIM, dtype=np.int8),
            opp_boost_delta=np.zeros(BOOST_DIM, dtype=np.int8),
            our_effectiveness=None, opp_effectiveness=None,
            we_moved_first=None,
            phase_is_forced_switch=(curr_ctx.phase == "forced_switch"),
            our_hp_after=curr_ctx.our_hp.copy(),
            opp_hp_after=curr_ctx.opp_hp.copy(),
            our_faint_count=int(we_fainted),
            opp_faint_count=int(opp_fainted),
            our_faint_causes=empty_faint_causes.copy(),
            opp_faint_causes=empty_faint_causes.copy(),
            our_attempted_move_id=our_attempted_move_id,
        )

    view = TurnView.from_events(events)
    our = view.ours
    opp = view.opp

    our_move_id = our.move_id
    our_switch_to = our.switched_to if our.switched else None
    opp_move_id = opp.move_id
    opp_switch_to = opp.switched_to if opp.switched else None
    opp_move_known = opp.moved or opp.switched

    our_cant_reason = our.cant_reason
    opp_cant_reason = opp.cant_reason
    our_failed_to_move = our_cant_reason is not None
    opp_failed_to_move = opp_cant_reason is not None

    def _status(s):
        return Status.__members__.get(s.upper()) if s else None
    our_status_applied = _status(our.status_applied)
    our_status_cured = _status(our.status_cured)
    opp_status_applied = _status(opp.status_applied)
    opp_status_cured = _status(opp.status_cured)

    our_item_lost = our.item_lost
    opp_item_lost = opp.item_lost

    our_boost_delta = (
        np.zeros(BOOST_DIM, dtype=np.int8) if our_switch_to is not None
        else (curr_ctx.our_boosts - prev_ctx.our_boosts).astype(np.int8)
    )
    opp_boost_delta = (
        np.zeros(BOOST_DIM, dtype=np.int8) if opp_switch_to is not None
        else (curr_ctx.opp_boosts - prev_ctx.opp_boosts).astype(np.int8)
    )

    def _to_dme(dm):
        if dm is None:
            return None
        ts_str = dm.target_status
        target_status = Status.__members__.get(ts_str) if ts_str else None
        return DamagingMoveEvent(
            user_species=dm.user_species or "",
            target_species=dm.target_species or "",
            target_status=target_status,
            move_id=dm.move_id or "",
            effectiveness=dm.effectiveness if dm.effectiveness is not None else 1.0,
        )

    our_damaging_event = _to_dme(our.damaging_move)
    opp_damaging_event = _to_dme(opp.damaging_move)

    faints = view.faint_details()
    our_faint_list = [f for f in faints if f.side == OURS]
    opp_faint_list = [f for f in faints if f.side == OPP]
    our_faint_count = len(our_faint_list)
    opp_faint_count = len(opp_faint_list)

    our_faint_causes_arr = np.zeros(FAINT_CAUSE_DIM, dtype=np.float32)
    for f in our_faint_list:
        our_faint_causes_arr[_cause_idx[f.cause]] = 1.0
    opp_faint_causes_arr = np.zeros(FAINT_CAUSE_DIM, dtype=np.float32)
    for f in opp_faint_list:
        opp_faint_causes_arr[_cause_idx[f.cause]] = 1.0

    our_target_hp_delta = _frozen_resolve_target_hp_delta(
        opp_damaging_event, our_hp_delta, curr_ctx.our_slot_map
    )
    opp_target_hp_delta = _frozen_resolve_target_hp_delta(
        our_damaging_event, opp_hp_delta, curr_ctx.opp_slot_map
    )

    return TurnDelta(
        our_move_id=our_move_id,
        our_switch_to=our_switch_to,
        our_prev_active=prev_ctx.our_active,
        opp_move_id=opp_move_id,
        opp_switch_to=opp_switch_to,
        opp_prev_active=prev_ctx.opp_active,
        opp_move_known=opp_move_known,
        our_hp_delta=our_hp_delta,
        opp_hp_delta=opp_hp_delta,
        we_fainted=our_faint_count > 0,
        opp_fainted=opp_faint_count > 0,
        our_failed_to_move=our_failed_to_move,
        our_cant_reason=our_cant_reason,
        opp_failed_to_move=opp_failed_to_move,
        opp_cant_reason=opp_cant_reason,
        our_boost_delta=our_boost_delta,
        opp_boost_delta=opp_boost_delta,
        our_effectiveness=our.effectiveness,
        opp_effectiveness=opp.effectiveness,
        we_moved_first=view.we_moved_first,
        our_damaging_event=our_damaging_event,
        opp_damaging_event=opp_damaging_event,
        phase_is_forced_switch=(curr_ctx.phase == "forced_switch"),
        our_hp_after=curr_ctx.our_hp.copy(),
        opp_hp_after=curr_ctx.opp_hp.copy(),
        our_target_hp_delta=our_target_hp_delta,
        opp_target_hp_delta=opp_target_hp_delta,
        our_move_outcome=our.outcome,
        opp_move_outcome=opp.outcome,
        our_move_crit=our.crit,
        opp_move_crit=opp.crit,
        our_faint_count=our_faint_count,
        opp_faint_count=opp_faint_count,
        our_faint_causes=our_faint_causes_arr,
        opp_faint_causes=opp_faint_causes_arr,
        our_attempted_move_id=our_attempted_move_id,
        our_status_applied=our_status_applied,
        our_status_cured=our_status_cured,
        opp_status_applied=opp_status_applied,
        opp_status_cured=opp_status_cured,
        our_item_lost=our_item_lost,
        opp_item_lost=opp_item_lost,
    )


# Fields compared as float arrays (HP family — FP-tolerant) vs everything else (exact).
_HP_ARRAY_FIELDS = ("our_hp_delta", "opp_hp_delta", "our_hp_after", "opp_hp_after")
_HP_SCALAR_FIELDS = ("our_target_hp_delta", "opp_target_hp_delta")
_EXACT_ARRAY_FIELDS = (
    "our_boost_delta", "opp_boost_delta", "our_faint_causes", "opp_faint_causes",
)
_EXACT_SCALAR_FIELDS = (
    "our_move_id", "opp_move_id", "our_switch_to", "opp_switch_to",
    "our_prev_active", "opp_prev_active", "opp_move_known",
    "we_fainted", "opp_fainted", "our_faint_count", "opp_faint_count",
    "our_failed_to_move", "our_cant_reason", "opp_failed_to_move", "opp_cant_reason",
    "our_effectiveness", "opp_effectiveness", "we_moved_first", "phase_is_forced_switch",
    "our_move_outcome", "opp_move_outcome", "our_move_crit", "opp_move_crit",
    "our_attempted_move_id",
    "our_status_applied", "our_status_cured", "opp_status_applied", "opp_status_cured",
    "our_item_lost", "opp_item_lost",
)
_DME_FIELDS = ("our_damaging_event", "opp_damaging_event")


def _dme_tuple(dme):
    if dme is None:
        return None
    return (dme.user_species, dme.target_species, dme.move_id,
            dme.effectiveness, dme.target_status)


class _BattleState:
    def __init__(self):
        self.prev_ctx: Optional[BattleContext] = None
        self.last_action: Optional[int] = None
        self.prev_cursor: int = 0
        self.our_slots = SlotRegistry()
        self.opp_slots = SlotRegistry()
        self.mgr_new = Gen3RewardManager()
        self.mgr_frozen = Gen3RewardManager()


class FoldEquivalencePlayer(Player):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.decisions = 0
        self.field_diffs: Counter = Counter()
        self.reward_diffs: Counter = Counter()
        self.violations: list[str] = []
        self.hp_fp_noise = 0          # sub-tolerance HP float rounding (not a bug)
        self.hp_fp_max = 0.0
        self.reward_fp_noise = 0
        self.reward_fp_max = 0.0
        self.cov: Counter = Counter()  # corner-path coverage
        self._per_battle: dict[str, _BattleState] = {}

    def _get_state(self, battle) -> _BattleState:
        tag = battle.battle_tag
        if tag not in self._per_battle:
            self._per_battle[tag] = _BattleState()
        return self._per_battle[tag]

    def _coverage(self, delta: TurnDelta, events) -> None:
        c = self.cov
        if any(e.kind in (EventKind.BOOST, EventKind.UNBOOST, EventKind.SETBOOST)
               for e in events):
            c["boost"] += 1
        if any(e.kind is EventKind.CLEARBOOST for e in events):
            c["clearboost"] += 1
        if any(e.kind is EventKind.SETBOOST for e in events):
            c["setboost"] += 1
        if any(e.kind is EventKind.SETHP for e in events):
            c["pain_split_sethp"] += 1
        if delta.our_switch_to is not None or delta.opp_switch_to is not None:
            c["switch"] += 1
        if (delta.our_faint_count + delta.opp_faint_count) >= 2:
            c["double_ko"] += 1
        if delta.our_cant_reason is not None or delta.opp_cant_reason is not None:
            c["cant"] += 1
        # multi-hit: >1 DAMAGE event naming the same actor on the same side this window
        dmg_by_actor: Counter = Counter()
        for e in events:
            if e.kind is EventKind.DAMAGE and e.actor_species:
                dmg_by_actor[(e.side, e.actor_species)] += 1
        if any(v > 1 for v in dmg_by_actor.values()):
            c["multi_hit"] += 1
        if delta.our_item_lost is not None or delta.opp_item_lost is not None:
            c["item_used"] += 1

    def _record_violation(self, battle, field, a, b, events, decision_idx) -> None:
        win = " | ".join(
            f"{e.kind.name}:{e.side or '-'}:{e.actor_species or '-'}:"
            f"{e.value.get('amount', e.value.get('move_id', e.value.get('op', '')))}"
            for e in events
        )
        msg = (
            f"[{battle.battle_tag}] turn={battle.turn} dec={decision_idx} "
            f"FIELD={field} new={a!r} frozen={b!r}\n    window: {win}"
        )
        self.field_diffs[field] += 1
        self.violations.append(msg)
        print(f"\n[FOLD DIFF] {msg}", flush=True)

    def _compare(self, battle, new: TurnDelta, frozen: TurnDelta, events, decision_idx) -> bool:
        """Return True if a REAL (non-FP) divergence was found."""
        real = False
        # HP arrays — FP-tolerant.
        for f in _HP_ARRAY_FIELDS:
            a = getattr(new, f); b = getattr(frozen, f)
            if np.array_equal(a, b):
                continue
            md = float(np.abs(a.astype(np.float64) - b.astype(np.float64)).max())
            if md > _HP_FP_TOL:
                self._record_violation(battle, f, a.tolist(), b.tolist(), events, decision_idx)
                real = True
            else:
                self.hp_fp_noise += 1
                self.hp_fp_max = max(self.hp_fp_max, md)
        # HP scalars (target-HP) — FP-tolerant, may be None.
        for f in _HP_SCALAR_FIELDS:
            a = getattr(new, f); b = getattr(frozen, f)
            if a is None and b is None:
                continue
            if (a is None) != (b is None):
                self._record_violation(battle, f, a, b, events, decision_idx)
                real = True
                continue
            md = abs(float(a) - float(b))
            if md > _HP_FP_TOL:
                self._record_violation(battle, f, a, b, events, decision_idx)
                real = True
            elif md > 0.0:
                self.hp_fp_noise += 1
                self.hp_fp_max = max(self.hp_fp_max, md)
        # Exact arrays.
        for f in _EXACT_ARRAY_FIELDS:
            if not np.array_equal(getattr(new, f), getattr(frozen, f)):
                self._record_violation(battle, f, getattr(new, f).tolist(),
                                       getattr(frozen, f).tolist(), events, decision_idx)
                real = True
        # Exact scalars.
        for f in _EXACT_SCALAR_FIELDS:
            a = getattr(new, f); b = getattr(frozen, f)
            if a != b:
                self._record_violation(battle, f, a, b, events, decision_idx)
                real = True
        # Damaging events (tuple compare).
        for f in _DME_FIELDS:
            a = _dme_tuple(getattr(new, f)); b = _dme_tuple(getattr(frozen, f))
            if a != b:
                self._record_violation(battle, f, a, b, events, decision_idx)
                real = True
        return real

    def _compare_reward(self, battle, state, new, frozen, decision_idx) -> bool:
        real = False
        state.mgr_new.process_turn_reward(battle, new)
        state.mgr_frozen.process_turn_reward(battle, frozen)
        bn = state.mgr_new._last_breakdown
        bf = state.mgr_frozen._last_breakdown
        for fld in dataclasses.fields(bn):
            a = float(getattr(bn, fld.name)); b = float(getattr(bf, fld.name))
            d = abs(a - b)
            if d <= 1e-12:
                continue
            if d > _REWARD_FP_TOL:
                self.reward_diffs[fld.name] += 1
                msg = (f"[{battle.battle_tag}] turn={battle.turn} dec={decision_idx} "
                       f"REWARD={fld.name} new={a:.6f} frozen={b:.6f}")
                self.violations.append(msg)
                print(f"\n[REWARD DIFF] {msg}", flush=True)
                real = True
            else:
                self.reward_fp_noise += 1
                self.reward_fp_max = max(self.reward_fp_max, d)
        return real

    def _settle(self, battle, state, curr_ctx, events) -> None:
        new = TurnDelta.build_from_events(
            state.prev_ctx, curr_ctx, state.last_action, events
        )
        frozen = _frozen_old_build_from_events(
            state.prev_ctx, curr_ctx, state.last_action, events
        )
        self.decisions += 1
        self._coverage(new, events)
        real = self._compare(battle, new, frozen, events, self.decisions)
        real |= self._compare_reward(battle, state, new, frozen, self.decisions)
        if real:
            print(f"\n[FOLD FATAL] real divergence at decision {self.decisions} "
                  f"({battle.battle_tag} turn {battle.turn})", flush=True)
            self._print_summary()
            os._exit(1)

    def choose_move(self, battle):
        try:
            state = self._get_state(battle)
            mask = Gen3ActionMasker.get_mask(battle)
            curr_ctx = BattleContext.from_battle(
                battle, mask, state.our_slots, state.opp_slots
            )
            if state.prev_ctx is not None and state.last_action is not None:
                events = battle.events_since(state.prev_cursor)
                self._settle(battle, state, curr_ctx, events)

            valid = np.where(mask == 1)[0]
            choice = int(np.random.choice(valid))

            if not battle.finished:
                # Drive BOTH managers' record_action identically so internal state
                # (switch subsidy, repetition counters, …) evolves in lockstep.
                state.mgr_new.record_action(curr_ctx, choice)
                state.mgr_frozen.record_action(curr_ctx, choice)
                state.prev_ctx = curr_ctx
                state.last_action = choice
                state.prev_cursor = battle.event_cursor

            return Gen3ActionMapper.action_to_order(choice, battle)
        except Exception as e:
            print(f"\n[FUZZ FATAL] {battle.battle_tag} turn {battle.turn}: {e}", flush=True)
            traceback.print_exc()
            os._exit(1)

    def _battle_finished_callback(self, battle) -> None:
        tag = battle.battle_tag
        state = self._per_battle.get(tag)
        if state is None or state.prev_ctx is None or state.last_action is None:
            self._per_battle.pop(tag, None)
            return
        try:
            mask = np.zeros(11, dtype=np.float32)
            terminal_ctx = BattleContext.from_battle(
                battle, mask, state.our_slots, state.opp_slots
            )
            events = battle.events_since(state.prev_cursor)
            self._settle(battle, state, terminal_ctx, events)
        except Exception as e:  # pragma: no cover - diagnostic only
            print(f"  [NOTICE] {tag}: final-turn compare skipped: {e}", flush=True)
        finally:
            self._per_battle.pop(tag, None)

    def _print_summary(self) -> None:
        print(f"\n{'=' * 70}", flush=True)
        print("TurnDelta Fold-Equivalence Results", flush=True)
        print(f"{'=' * 70}", flush=True)
        print(f"Decisions compared : {self.decisions}", flush=True)
        total_real = sum(self.field_diffs.values()) + sum(self.reward_diffs.values())
        print(f"REAL field diffs   : {sum(self.field_diffs.values())}", flush=True)
        print(f"REAL reward diffs  : {sum(self.reward_diffs.values())}", flush=True)
        print(f"HP float-noise     : {self.hp_fp_noise} (max abs {self.hp_fp_max:.2e}, "
              f"tol {_HP_FP_TOL:.0e})", flush=True)
        print(f"Reward float-noise : {self.reward_fp_noise} (max abs {self.reward_fp_max:.2e}, "
              f"tol {_REWARD_FP_TOL:.0e})", flush=True)
        if self.field_diffs:
            print("Per-field REAL diffs:", flush=True)
            for n, c in self.field_diffs.most_common():
                print(f"  {n:<24} {c}", flush=True)
        if self.reward_diffs:
            print("Per-field REAL reward diffs:", flush=True)
            for n, c in self.reward_diffs.most_common():
                print(f"  {n:<24} {c}", flush=True)
        print("Coverage (corner paths exercised):", flush=True)
        for path in ("boost", "clearboost", "setboost", "switch", "pain_split_sethp",
                     "double_ko", "multi_hit", "cant", "item_used"):
            print(f"  {path:<18} {self.cov.get(path, 0)}", flush=True)
        return total_real


async def main(minutes: float, chunk: int) -> None:
    ts = int(time.time()) % 100000
    print(f"TurnDelta Fold-Equivalence Fuzz — {BATTLE_FORMAT} — budget {minutes:.1f} min",
          flush=True)
    tb = Gen3Teambuilder(_teams())
    fuzz = FoldEquivalencePlayer(
        battle_format=BATTLE_FORMAT,
        team=tb,
        account_configuration=AccountConfiguration(f"TDf{ts}", "pw"),
        server_configuration=LocalhostServerConfiguration, start_listening=False,
        battle_class=Gen3Battle,
    )
    opp = RandomPlayer(
        battle_format=BATTLE_FORMAT,
        team=tb,
        account_configuration=AccountConfiguration(f"TDo{ts}", "pw"),
        server_configuration=LocalhostServerConfiguration, start_listening=False,
        battle_class=Gen3Battle,
    )

    t0 = time.time()
    deadline = t0 + minutes * 60.0
    battles = 0
    infra_skips = 0
    while time.time() < deadline:
        try:
            await run_local_battles(fuzz, opp, chunk)
            battles += chunk
        except Exception as e:
            infra_skips += 1
            if infra_skips <= 5:
                print(f"  [infra-skip #{infra_skips}] {type(e).__name__}: {str(e)[:140]}",
                      flush=True)
        # Drop finished battles between chunks (bounded memory over a long run).
        fuzz._battles.clear()
        opp._battles.clear()
        fuzz._per_battle.clear()
        elapsed = time.time() - t0
        cov = fuzz.cov
        print(
            f"  …{elapsed / 60:5.1f}m | battles≈{battles} decisions={fuzz.decisions} "
            f"| REALdiffs={sum(fuzz.field_diffs.values()) + sum(fuzz.reward_diffs.values())} "
            f"hp-fpnoise={fuzz.hp_fp_noise} "
            f"| cov: sw={cov.get('switch',0)} boost={cov.get('boost',0)} "
            f"clr={cov.get('clearboost',0)} set={cov.get('setboost',0)} "
            f"sethp={cov.get('pain_split_sethp',0)} dko={cov.get('double_ko',0)} "
            f"mh={cov.get('multi_hit',0)} cant={cov.get('cant',0)} "
            f"item={cov.get('item_used',0)}",
            flush=True,
        )

    total_real = fuzz._print_summary()
    print(f"Infra-skips        : {infra_skips}", flush=True)
    coverage_holes = [p for p in _REQUIRED_COVERAGE if fuzz.cov.get(p, 0) == 0]
    ok = (total_real == 0) and not coverage_holes and fuzz.decisions > 0
    if coverage_holes:
        print(f"COVERAGE HOLES (never fired): {coverage_holes}", flush=True)
    print("\nPASS — event-fold value-identical to the snapshot-diff path (no real diffs, "
          "all coverage paths fired)." if ok else "\nFAIL", flush=True)
    print("=" * 70, flush=True)
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=15.0)
    ap.add_argument("--chunk", type=int, default=20)
    args = ap.parse_args()
    asyncio.run(main(args.minutes, args.chunk))
