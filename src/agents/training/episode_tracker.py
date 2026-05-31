from __future__ import annotations
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import numpy as np

from poke_env.battle.abstract_battle import DamagingMoveEvent

from agents.enums import PokemonType

from agents.action.ordering_integrity import (
    reorder_move_bits_to_sorted,
    assert_sorted_validity_correct,
)
from agents.training.battle_context import BattleContext, TurnDelta
from agents.training.hidden_power_tracker import HiddenPowerTracker
from agents.training.slot_registry import SlotRegistry

if TYPE_CHECKING:
    from agents.enums import Status

    from agents.battle.live_view import LiveView
    from agents.observation.turn_delta_encoder import TurnDeltaEncoder


@dataclass(frozen=True)
class _HpTargetMon:
    """Minimal mon-shaped object for HiddenPowerTracker.observe().

    Carries exactly the attributes effective_multiplier() reads. The status
    field is the target's status AT MOVE-FIRE TIME (sourced from the
    DamagingMoveEvent captured by the protocol parser), not the current live
    status — needed because Gen 3 Fire moves thaw their target in the same hit
    that resolves them.
    """
    species: str
    type_1: "PokemonType"
    type_2: "PokemonType | None"
    ability: "str | None"
    status: "Status | None"


def _wrap_hp_target(live: "LiveView", event: DamagingMoveEvent) -> Optional[_HpTargetMon]:
    """Look up the target's current type/ability from the LiveView and overlay the
    status captured at the moment HP fired.

    The target of an opponent's Hidden Power is always one of OUR mons, so resolve it
    against ``live.ours``. Returns None if the species can't be resolved (shouldn't
    happen — every HP hit names a real teammate).

    The defender's types are reconstructed from ``LivePokemon.types`` (current-board,
    lowercased names) into the ``PokemonType`` enums ``effective_multiplier`` reads. This
    is value-identical to the old ``mon.type_1``/``mon.type_2`` reads: poke-env defines all
    three properties off the same ``_temporary_types`` / ``_type_1`` / ``_type_2`` state, so
    ``types`` reconstructed to ``(type_1, type_2)`` equals the pair the properties return —
    type-change (Conversion / Camouflage) cases included.
    """
    live_mon = live.ours.get(event.target_species)
    if live_mon is None or not live_mon.types:
        return None
    type_1 = PokemonType[live_mon.types[0].upper()]
    type_2 = (
        PokemonType[live_mon.types[1].upper()] if len(live_mon.types) > 1 else None
    )
    return _HpTargetMon(
        species=live_mon.species,
        type_1=type_1,
        type_2=type_2,
        ability=live_mon.ability,
        status=event.target_status,
    )


class EpisodeTracker:
    """
    Tracks per-episode state needed to build observations and reward signals.

    Owns slot registries, a rolling history of BattleContexts, and a parallel
    list of actions taken at each context so historical TurnDeltas can be
    reconstructed for the N-turn history observation feature.
    """

    def __init__(self, history_cap: Optional[int] = None):
        """``history_cap`` = the N passed to :meth:`prev_N_delta_vecs` (i.e.
        ``N_HISTORY_TURNS``). When set, the rolling history is bounded so a long game /
        250-turn stall can't accumulate hundreds of ``BattleContext``s (each carrying
        ``obs``+``mask``). ``prev_N_delta_vecs`` reaches ``_history[-2-i]`` (→ -(N+1)) and
        ``_actions``/``_cursors[-1-i]`` (→ -N), so we keep N+1 contexts and N aux entries;
        the ``len(actions)==len(cursors)==len(history)-1`` invariant is preserved by those
        maxlens. ``None`` ⇒ unbounded (short-episode callers: inference, tests)."""
        self._our_slots = SlotRegistry()
        self._opp_slots = SlotRegistry()
        _hist_max = (history_cap + 1) if history_cap else None
        _aux_max = history_cap if history_cap else None
        self._history: "deque[BattleContext]" = deque(maxlen=_hist_max)
        self._actions: "deque[int]" = deque(maxlen=_aux_max)   # _actions[i]=action FROM _history[i]
        self._cursors: "deque[int]" = deque(maxlen=_aux_max)   # _cursors[i]=event_cursor at _history[i]
        self._last_action: int = -1
        self._last_cursor: int = 0      # event_cursor at the last record() call
        self._hidden_power_tracker = HiddenPowerTracker()
        # Memoized turn-history: encoded TurnDelta vectors, oldest-left/newest-right.
        # A past turn's window is bounded and immutable (see prev_N_delta_vecs), so its
        # encoded vector never changes — we encode only the NEWEST delta each step and
        # reuse the rest, instead of re-folding+re-encoding all N slots every step.
        self._hist_vec_cache: "deque[np.ndarray]" = deque(maxlen=_aux_max)
        self._n_cached_deltas: int = 0   # total completed deltas the cache has encoded
        # MONOTONIC count of completed turn transitions this episode. NOT len(_history)-1,
        # which caps once the (bounded) history deque starts dropping — using the capped
        # length would make the cache miss new turns and serve stale ones.
        self._n_transitions: int = 0

    @property
    def hidden_power_tracker(self) -> HiddenPowerTracker:
        return self._hidden_power_tracker

    @property
    def last_ctx(self) -> Optional[BattleContext]:
        return self._history[-1] if self._history else None

    @property
    def prev_mask(self) -> np.ndarray:
        """Previous turn's action mask, as an obs feature.

        The MOVE bits are reordered from action/request order into sorted-by-id
        order so they line up with the active mon's move slots in the feature
        extractor (which reads moves via ``get_sorted_moves``). Without this the
        validity bit for one move lands on a different move's embedding — silent
        when all moves are legal, wrong on disabled/zero-PP turns. All-ones if no
        previous turn recorded yet."""
        if len(self._history) >= 2:
            prev_ctx = self._history[-2]
            reordered = reorder_move_bits_to_sorted(
                prev_ctx.mask.astype(np.float32), prev_ctx.active_move_ids
            )
            assert_sorted_validity_correct(
                reordered, prev_ctx.mask, prev_ctx.active_move_ids
            )
            return reordered
        return np.ones(11, dtype=np.float32)

    def record(self, battle, mask: np.ndarray, legal=None) -> BattleContext:
        """Build and store a context snapshot for the current turn.

        Also commits the pending _last_action and event_cursor as the action
        and window-start taken FROM the previous context, so prev_N_delta_vecs()
        can reconstruct all N deltas. Updates the HiddenPowerTracker BEFORE the
        env encodes the observation, so the encoded obs includes the just-fired
        HP's narrowing.

        ``legal`` is the per-decision :class:`LegalActions` snapshot the masker built
        the mask from; threaded onto the stored context so the action mapper decodes
        against the same snapshot the model saw.
        """
        if self._history:
            self._actions.append(self._last_action)
            self._cursors.append(self._last_cursor)
            self._n_transitions += 1   # one more completed transition becomes available
        # Capture cursor NOW (before we build the context snapshot) so it marks
        # the start of the window for the NEXT decision — events emitted between
        # this record() and the next one are the delta for this turn.
        self._last_cursor = getattr(battle, "event_cursor", 0)
        ctx = BattleContext.from_battle(battle, mask, self._our_slots, self._opp_slots, legal)

        # Current-board reads (the HP-candidate moveset scan + the HP-target type/ability
        # lookup) go through our LiveView, never the raw poke-env Battle/Pokemon objects
        # (ai_v4 Phase 3 — encapsulation behind the strict boundary).
        live = battle.strict_view().live
        self._maybe_observe_hidden_power(live, ctx)
        self._scan_opp_movesets_for_no_hp(live)

        self._history.append(ctx)
        return ctx

    def _scan_opp_movesets_for_no_hp(self, live: "LiveView") -> None:
        """Mark any opponent species whose four moves are fully revealed and
        none is Hidden Power as definitively HP-less.

        This converts the previously ambiguous "all-zero HP probs" state into a
        positive signal (hp_revealed=1, probs all zero in the encoder).
        Idempotent and cheap (~12 lookups per turn).

        Reads the opponent's revealed movesets off the current-board ``LiveView``
        (``live.opp.mons`` / ``LivePokemon.move_ids``) rather than the raw
        ``battle.opponent_team`` — value-identical, since ``move_ids`` is exactly the
        revealed move-dict keys the old path iterated.
        """
        for mon in live.opp.mons:
            if not mon.species:
                continue
            move_ids = mon.move_ids
            if len(move_ids) >= 4 and not any(
                mid.startswith("hiddenpower") for mid in move_ids
            ):
                self._hidden_power_tracker.mark_no_hp(mon.species)

    def _maybe_observe_hidden_power(self, live: "LiveView", ctx: BattleContext) -> None:
        """Feed an HP observation to the tracker when opp's last damaging move
        was Hidden Power.

        Everything we need — firer species, target species, target status at
        move-fire time, effectiveness — comes from ctx.opp_last_damaging_event,
        which is set by poke-env's protocol parser at the |move| line and
        finalized by the matching |-supereffective|/|-resisted|/|-immune|
        event. The event is turn-gated to the just-ended turn, so a stale HP
        from earlier in the battle never leaks. The target's current type/ability
        (for the effectiveness filter) is resolved through the current-board
        ``live`` view. No before/after inference, no resolver, no edge cases for
        switches / Roar / phazing / faint chains — the protocol stated the facts
        and we just record them.
        """
        if ctx.phase != "move_selection":
            return
        event = ctx.opp_last_damaging_event
        if event is None or event.move_id != "hiddenpower":
            return
        target = _wrap_hp_target(live, event)
        if target is None:
            return
        self._hidden_power_tracker.observe(
            event.user_species, event.effectiveness, target
        )

    def advance(self, action: int) -> None:
        """Record the action chosen this turn, before the game steps forward."""
        self._last_action = action

    def _get_events_for_window(self, battle, cursor: int) -> list:
        """Return events from ``cursor`` to now using battle.events_since, or []."""
        if battle is None or not hasattr(battle, "events_since"):
            return []
        return battle.events_since(cursor)

    def build_delta(self, battle=None) -> TurnDelta:
        """Diff between the last two turns. Returns an empty delta at episode start.

        Pass ``battle`` (a Gen3Battle) to use the event-fold path (Step 4).
        Without it the old snapshot-diff path is used as a fallback.
        """
        if len(self._history) < 2:
            return TurnDelta.empty()
        prev_ctx = self._history[-2]
        curr_ctx = self._history[-1]
        if battle is not None and hasattr(battle, "events_since") and self._cursors:
            cursor = self._cursors[-1]
            events = self._get_events_for_window(battle, cursor)
            return TurnDelta.build_from_events(prev_ctx, curr_ctx, self._last_action, events)
        return TurnDelta.build(prev_ctx, curr_ctx, self._last_action)

    def _encode_delta_slot(self, i: int, encoder, battle, use_events) -> np.ndarray:
        """Encode the TurnDelta for history slot ``i`` (0 = most-recent), folded over its
        OWN bounded decision window: ``[cursors[-1-i] : cursors[-i])`` (``end=None`` ⇒ to
        "now" for the most-recent slot). The upper bound is what makes the slot represent
        its own turn and the per-step cost bounded; the most-recent slot's ``end=None``
        resolves to the same cursor that bounds it on the NEXT step, so a cached vector is
        bit-identical to recomputing it later — which is why the deque memoization is safe.
        """
        action = self._actions[-1 - i]
        ctx_prev = self._history[-2 - i]
        ctx_curr = self._history[-1 - i]
        if use_events:
            start = self._cursors[-1 - i]
            end = None if i == 0 else self._cursors[-i]
            events = battle.events_between(start, end)
            delta = TurnDelta.build_from_events(ctx_prev, ctx_curr, action, events)
        else:
            delta = TurnDelta.build(ctx_prev, ctx_curr, action)
        return encoder.encode(delta)

    def prev_N_delta_vecs(
        self, n: int, encoder: "TurnDeltaEncoder", battle=None
    ) -> np.ndarray:
        """Return (n, TURN_DELTA_DIM) array of encoded TurnDeltas, oldest-first.

        Index n-1 is the most recent delta (same data as build_delta()). Turns not yet
        played are zero-padded. Pass ``battle`` (Gen3Battle) to use the event-fold path.

        Each past turn's bounded window is immutable, so on the event path we **encode only
        the newly-completed delta(s) and reuse the cached rest** (deque memoization) — one
        fold+encode per step instead of N. The output is identical to recomputing every
        slot from scratch.
        """
        result = np.zeros((n, encoder.dimension), dtype=np.float32)
        use_events = battle is not None and hasattr(battle, "events_between")
        available = min(n, len(self._history) - 1, len(self._actions))
        if use_events:
            available = min(available, len(self._cursors))
        if available <= 0:
            return result

        if not use_events:
            # Fallback (no event log): recompute every slot, uncached.
            for i in range(available):
                result[n - 1 - i] = self._encode_delta_slot(i, encoder, battle, use_events)
            return result

        # Event path: extend the deque by the newly-completed delta(s) only. Use the
        # MONOTONIC transition count (not len(_history)-1, which caps with the deque).
        total_completed = self._n_transitions
        new = total_completed - self._n_cached_deltas
        if new == 1:
            # Common case: one turn finished since last call → encode just slot 0.
            self._hist_vec_cache.append(
                self._encode_delta_slot(0, encoder, battle, use_events)
            )
        elif new != 0:
            # Rare (calls skipped, or first call mid-episode): rebuild the kept tail.
            self._hist_vec_cache.clear()
            for i in range(available - 1, -1, -1):  # oldest-kept first → newest appended last
                self._hist_vec_cache.append(
                    self._encode_delta_slot(i, encoder, battle, use_events)
                )
        self._n_cached_deltas = total_completed

        for k, vec in enumerate(reversed(self._hist_vec_cache)):  # newest first
            if k >= n:
                break
            result[n - 1 - k] = vec
        return result

    def reset(self) -> None:
        self._our_slots.reset()
        self._opp_slots.reset()
        self._history.clear()
        self._actions.clear()
        self._cursors.clear()
        self._hist_vec_cache.clear()
        self._n_cached_deltas = 0
        self._n_transitions = 0
        self._last_action = -1
        self._last_cursor = 0
        self._hidden_power_tracker.reset()
