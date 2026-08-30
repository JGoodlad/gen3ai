"""Legacy snapshot-diff ``TurnDelta`` builder, retired from production (Phase-5 event-fold).

Retained for the poke-env-gap fuzz suite only — **not imported by any production module.**

``build_legacy`` below and the four helpers in this module are the old "detective": they
reconstructed what happened on a turn by DIFFING two
:class:`~agents.training.battle_snapshot.BattleContext` snapshots with a pile of heuristics
(KO-before-acting, phaze recovery via ``opp_all_last_move_ids``, effectiveness alignment,
move-outcome derivation). The event-fold ``TurnDelta.build_from_events`` replaced them on
EVERY production path (training env, reward tracker, episode tracker, forensic recorder).
They survive ONLY as test scaffolding: the poke-env-gap fuzz harnesses
(``poke_env_gaps/move_outcome_fuzz_test.py``, ``effectiveness_fuzz_e2e_test.py``) and a few
crafted-context unit tests validate BattleContext's snapshot-derived per-turn flags through
this path. Do not call them from production code.

The two helpers the production fold ALSO needs — ``_fold_hp_deltas`` and
``_resolve_target_hp_delta`` — stay in :mod:`agents.training.turn_delta`; this module imports
the latter by name.
"""
from __future__ import annotations
from typing import Optional, TYPE_CHECKING
import numpy as np

from agents.gen3_mechanics import PHAZING_MOVES, BOOST_DIM
from agents.training.turn_delta import (
    TurnDelta, SELF_KO_MOVES, _resolve_target_hp_delta,
)

if TYPE_CHECKING:
    from agents.training.battle_snapshot import BattleContext


def _moves_match(a: Optional[str], b: Optional[str]) -> bool:
    """True if two move ids refer to the same move. Treats all Hidden Power
    variants ('hiddenpower', 'hiddenpowerfire', …) as one move."""
    if a == b:
        return True
    return bool(a and b and a.startswith("hiddenpower") and b.startswith("hiddenpower"))


def _align_effectiveness(move_id, effectiveness, event):
    """Keep a side's (effectiveness, damaging_event) only when they describe the
    SAME move we recorded as having fired. They are turn-gated independently of
    move_id and on same-turn forced-switch / faint double-records can lag at a
    different move — feeding the wrong effectiveness/target would corrupt the obs
    one-hot, the actor/target attribution, and the immune reward term. On
    disagreement we drop to "unknown" (None) rather than serve wrong data.
    Applied identically for our and opp sides."""
    if (event is not None and move_id is not None
            and not _moves_match(event.move_id, move_id)):
        return None, None
    return effectiveness, event


def _ko_before_acting(*, fainted, switched_voluntarily, move_resolved,
                      other_side_moved_first, cant_reason) -> bool:
    """A side was KO'd BEFORE it could act: it fainted this turn, did not choose
    to switch, no move of its own resolved (no damaging event / miss / fail), and
    the OTHER side moved first (Gen 3: mover lands the KO before the slower mon's
    turn). When True the side did NOTHING — its move_id must be None and its
    "didn't move" reason is 'fainted'. Symmetric for both sides; only the
    per-side inputs differ (our faint splits across two turns so we never
    voluntarily switch on it; an opp faint folds the forced replacement into the
    same delta, so its other_side_moved_first is our we_moved_first)."""
    return bool(
        fainted and not switched_voluntarily and not move_resolved
        and other_side_moved_first is True and cant_reason is None
    )


def _derive_move_outcome(
    move_used: bool,
    missed: bool,
    failed: bool,
    suppressed: bool,
    connected: bool = False,
) -> Optional[str]:
    """Collapse the per-side protocol outcome flags into one category.

    Returns "miss" / "fail" / "hit", or None when no move resolved.

    `suppressed` is True when the side was prevented from acting by a |cant|
    (sleep/par/flinch/etc.) — no move resolved, so the outcome is None (the cant
    reason is carried separately).

    The outcome describes a MOVE, so it is None unless a move was actually used
    (`move_used`). The miss/fail flags are turn-gated and can leak onto a turn the
    move did NOT have them — a no-move turn (a switch on the same game turn as an
    earlier sub-turn move) or a self-faint move like Explosion. So:
      - gate on `move_used` (a switch is never "miss"/"fail"), and
      - `connected` (a damaging event resolved this turn) means the move DEALT
        damage, i.e. it landed — that overrides a stale miss/fail flag (Explosion
        can't both deal damage and "miss").
    Precedence among real outcomes is miss > fail > hit; the protocol never emits
    more than one for a single move.
    """
    if suppressed or not move_used:
        return None
    if connected:
        return "hit"
    if missed:
        return "miss"
    if failed:
        return "fail"
    return "hit"


def build_legacy(prev_ctx: "BattleContext", curr_ctx: "BattleContext", action: int) -> "TurnDelta":
    """LEGACY snapshot-diff detective — RETIRED FROM PRODUCTION; use ``TurnDelta.build_from_events``.

    Reconstructs the turn by diffing two ``BattleContext`` snapshots with heuristics
    (see the module docstring). Retained only for the poke-env-gap fuzz harnesses +
    crafted-context unit tests that validate BattleContext's snapshot-derived per-turn flags.
    Every production caller folds the event log via ``build_from_events`` instead.
    """
    our_hp_delta = curr_ctx.our_hp - prev_ctx.our_hp
    opp_hp_delta = curr_ctx.opp_hp - prev_ctx.opp_hp

    # When an opponent mon is revealed for the first time (not in prev slot_map),
    # its HP slot transitions from 0 (unrevealed default) to its actual value.
    # This looks like the opponent "gained" HP, but we didn't cause that — we just
    # learned about it. Zero out the delta for newly-revealed slots to prevent false
    # hp_opp penalties in compute_base_reward.
    for species, slot in curr_ctx.opp_slot_map.items():
        if species not in prev_ctx.opp_slot_map and opp_hp_delta[slot] > 0:
            opp_hp_delta[slot] = 0.0

    we_fainted = curr_ctx.our_fainted_count > prev_ctx.our_fainted_count
    opp_fainted = curr_ctx.opp_fainted_count > prev_ctx.opp_fainted_count

    # --- Our action ---
    if action < 6:
        # The action's INTENT — which team-slot species we sent in — is the
        # only correct answer here. Reading curr_ctx.our_active instead
        # gets bitten when the switch-in dies and forced-replacements cycle
        # more mons before the next snapshot (our_active ends up pointing
        # at the final replacement, not the mon we switched to).
        our_switch_to = (
            prev_ctx.our_team_order[action]
            if action < len(prev_ctx.our_team_order)
            else None
        )
        our_move_id = None
    elif action < 10:
        our_switch_to = None
        slot = action - 6
        ids = prev_ctx.active_move_ids
        our_move_id = ids[slot] if slot < len(ids) else None
    else:
        # action == 10: Struggle
        our_switch_to = None
        our_move_id = "struggle"

    # --- Protocol-truth override for our_move_id ---
    # The action-derived id above is vulnerable to the action-bookkeeping
    # desync (_last_action can be a different turn's action on faint /
    # forced-switch cadence — confirmed mislabeling moves in training). The
    # protocol is authoritative for the move OUR mon actually used; we take it
    # from two turn-gated sources depending on whether the mon survived:
    we_stayed_in = (
        prev_ctx.our_active == curr_ctx.our_active
        and curr_ctx.our_active != "NONE"
    )
    if curr_ctx.our_cant_reason is None:
        if (our_switch_to is None and we_stayed_in
                and curr_ctx.our_last_move_id is not None):
            # Survived: active_pokemon.last_move is the move we used — fresh
            # and DELEGATION-AWARE (Sleep Talk / Metronome store the CALLED
            # move, e.g. "surf"), so the delegated move becomes first-class.
            our_move_id = curr_ctx.our_last_move_id
        elif we_fainted and curr_ctx.our_last_damaging_event is not None:
            # Used a (damaging) move and fainted THIS turn — active_pokemon now
            # reads the replacement (last_move wrong) and the action is desynced.
            # The turn-gated DamagingMoveEvent names the move the fainted mon
            # actually used. This also corrects a move↔switch misclassification
            # when the desynced action looked like a switch.
            our_move_id = curr_ctx.our_last_damaging_event.move_id
            our_switch_to = None

    # --- Protocol-accuracy guards (shared with the opp side below) ---
    # The action-derived id would misrepresent a KO'd-before-acting turn as
    # "used the clicked move" (and the outcome as "hit"). _ko_before_acting
    # reports the truth: nothing fired (move_id None, reason "fainted").
    our_ko = _ko_before_acting(
        fainted=we_fainted,
        switched_voluntarily=our_switch_to is not None,
        move_resolved=(curr_ctx.our_last_damaging_event is not None
                       or curr_ctx.our_move_missed or curr_ctx.our_move_failed),
        other_side_moved_first=(curr_ctx.we_moved_first is not True),  # opp first
        cant_reason=curr_ctx.our_cant_reason,
    )
    if our_ko:
        our_move_id = None
    our_cant_reason = curr_ctx.our_cant_reason  # KO'd-before-acting → None, not "fainted"
    our_effectiveness, our_damaging_event = _align_effectiveness(
        our_move_id, curr_ctx.our_last_effectiveness, curr_ctx.our_last_damaging_event,
    )

    # --- Opponent action ---
    # opp_last_move_id in curr_ctx was read from battle.opponent_active_pokemon.last_move
    # AFTER the turn resolved. Guard against contamination from a newly switched-in
    # Pokémon's prior-appearance last_move by checking if the opponent switched.
    opp_switched = prev_ctx.opp_active != curr_ctx.opp_active
    opp_switch_to = curr_ctx.opp_active if opp_switched and curr_ctx.opp_active != "NONE" else None

    if opp_switched:
        if our_move_id in PHAZING_MOVES:
            # Phaze case (Roar/Whirlwind): the opponent moved first (Gen 3 phazing moves
            # have -6 priority), then was forced out. opp_last_move_id reads from the NEW
            # active mon (which hasn't moved), so we recover the phazed mon's last_move
            # from the full-team snapshot in opp_all_last_move_ids.
            opp_move_id = curr_ctx.opp_all_last_move_ids.get(prev_ctx.opp_active)
            opp_move_known = opp_move_id is not None
        elif opp_fainted:
            # Forced switch after their mon fainted: they may have moved before dying.
            # Recover from the full-team snapshot (opp_last_move_id reads the NEW active
            # mon, which hasn't moved yet, so we must use the per-species snapshot).
            opp_move_id = curr_ctx.opp_all_last_move_ids.get(prev_ctx.opp_active)
            opp_move_known = True   # switch was forced (faint), whether or not we know the move
        else:
            opp_move_id = None
            opp_move_known = True   # voluntary switch — no move was used
    else:
        opp_move_id = curr_ctx.opp_last_move_id
        opp_move_known = opp_move_id is not None

    # --- Same protocol-accuracy guards, opp side ---
    # The faint-recovery above may have pulled a STALE prior move from
    # opp_all_last_move_ids when the opp was KO'd before acting; correct it.
    # (Per-side inputs differ: an opp faint folds the forced replacement into
    # this delta, so its "switched_voluntarily" is only true off a faint, and
    # "other side moved first" is OUR we_moved_first.)
    opp_ko = _ko_before_acting(
        fainted=opp_fainted,
        switched_voluntarily=(opp_switch_to is not None and not opp_fainted),
        move_resolved=(curr_ctx.opp_last_damaging_event is not None
                       or curr_ctx.opp_move_missed or curr_ctx.opp_move_failed),
        other_side_moved_first=(curr_ctx.we_moved_first is True),  # we went first
        cant_reason=curr_ctx.opp_cant_reason,
    )
    if opp_ko:
        opp_move_id = None
        opp_move_known = True  # we positively know nothing fired
    opp_cant_reason = curr_ctx.opp_cant_reason  # KO'd-before-acting → None, not "fainted"
    opp_effectiveness, opp_damaging_event = _align_effectiveness(
        opp_move_id, curr_ctx.opp_last_effectiveness, curr_ctx.opp_last_damaging_event,
    )

    our_failed_to_move = our_cant_reason is not None
    opp_failed_to_move = opp_cant_reason is not None

    # Boost deltas: meaningful when the same mon stayed in; zeroed when switched
    # (the switch-in's boosts are its own baseline, not a change from the prev mon).
    our_boost_delta = (
        np.zeros(BOOST_DIM, dtype=np.int8) if our_switch_to is not None
        else (curr_ctx.our_boosts - prev_ctx.our_boosts).astype(np.int8)
    )
    opp_boost_delta = (
        np.zeros(BOOST_DIM, dtype=np.int8) if opp_switch_to is not None
        else (curr_ctx.opp_boosts - prev_ctx.opp_boosts).astype(np.int8)
    )

    # Target-HP-delta attribution: look the named target's species up in
    # the current slot map (where it actually lives now, post-turn) so
    # newly-revealed opp mons resolve correctly. Returns None when no
    # damaging move fired or the target species isn't in the slot map.
    our_target_hp_delta = _resolve_target_hp_delta(
        opp_damaging_event, our_hp_delta, curr_ctx.our_slot_map
    )
    opp_target_hp_delta = _resolve_target_hp_delta(
        our_damaging_event, opp_hp_delta, curr_ctx.opp_slot_map
    )

    # Per-side move outcome. `suppressed` covers the |cant| case where an
    # action was selected (our_move_id set) but no move resolved. For the
    # opponent, opp_move_id is already None on a voluntary switch and is
    # recovered on a faint-forced switch, so move_used alone is the right
    # signal — no extra switch suppression needed.
    our_move_outcome = _derive_move_outcome(
        move_used=our_move_id is not None,
        missed=curr_ctx.our_move_missed,
        failed=curr_ctx.our_move_failed,
        suppressed=our_failed_to_move,
        connected=our_damaging_event is not None or our_move_id in SELF_KO_MOVES,
    )
    opp_move_outcome = _derive_move_outcome(
        move_used=opp_move_id is not None,
        missed=curr_ctx.opp_move_missed,
        failed=curr_ctx.opp_move_failed,
        suppressed=opp_failed_to_move,
        connected=opp_damaging_event is not None or opp_move_id in SELF_KO_MOVES,
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
        we_fainted=we_fainted,
        opp_fainted=opp_fainted,
        our_failed_to_move=our_failed_to_move,
        our_cant_reason=our_cant_reason,
        opp_failed_to_move=opp_failed_to_move,
        opp_cant_reason=opp_cant_reason,
        our_boost_delta=our_boost_delta,
        opp_boost_delta=opp_boost_delta,
        our_effectiveness=our_effectiveness,
        opp_effectiveness=opp_effectiveness,
        we_moved_first=curr_ctx.we_moved_first,
        our_damaging_event=our_damaging_event,
        opp_damaging_event=opp_damaging_event,
        phase_is_forced_switch=(curr_ctx.phase == "forced_switch"),
        decision_was_forced_switch=(prev_ctx.phase == "forced_switch"),
        our_hp_after=curr_ctx.our_hp.copy(),
        opp_hp_after=curr_ctx.opp_hp.copy(),
        our_target_hp_delta=our_target_hp_delta,
        opp_target_hp_delta=opp_target_hp_delta,
        our_move_outcome=our_move_outcome,
        opp_move_outcome=opp_move_outcome,
        our_move_crit=curr_ctx.our_move_crit,
        opp_move_crit=curr_ctx.opp_move_crit,
    )
