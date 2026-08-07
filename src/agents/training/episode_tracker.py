from __future__ import annotations
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import numpy as np

from poke_env.battle.abstract_battle import DamagingMoveEvent

from agents.enums import PokemonType

from agents.action.constants import MOVE_START, MOVE_END
from agents.action.ordering_integrity import (
    reorder_move_bits_to_sorted,
    assert_sorted_validity_correct,
)
from agents.training.battle_snapshot import BattleContext
from agents.training.turn_delta import TurnDelta
from agents.training.hidden_power_tracker import HiddenPowerTracker
from agents.training.progress_clock import ProgressClock
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


class RecencyTracker:
    """E9 step 1 (roadmap §3.9): per-(side, species) RECENCY counters — the first
    history-attaches-to-entities increment. Three turn-denominated counters per mon:

      * seen    — turns since last ON FIELD (reset by the live actives + SWITCH events)
      * acted   — turns since it last EXECUTED a move (reset by MOVE events)
      * was_hit — turns since it last TOOK damage (reset by DAMAGE events — the event's
                  (side, actor_species) attribution names the mon AFFECTED)

    Ticked by observed TURN-number deltas (a multi-decision turn ticks once, not per
    decision); resets consume the SAME per-decision event window the TurnDelta fold reads,
    so obs and history can never disagree about what happened. Both sides PUBLIC (every
    reset derives from observed protocol events). ``values()`` returns the obs form:
    log-saturated to [0, 1] over a 10-turn cap (the ``turns_since_progress`` convention);
    a never-tracked mon reads 1.0 — maximum staleness, the honest default for a mon that
    has not appeared this episode."""

    _SAT = 10

    def __init__(self):
        self._last_turn: Optional[int] = None
        self._seen: dict = {}
        self._acted: dict = {}
        self._hit: dict = {}

    def update(self, turn: int, events, our_active: Optional[str],
               opp_active: Optional[str]) -> None:
        from agents.battle.battle_event import OURS, OPP, EventKind
        if self._last_turn is None:
            self._last_turn = turn
        dt = max(0, int(turn) - self._last_turn)
        self._last_turn = int(turn)
        if dt:
            for d in (self._seen, self._acted, self._hit):
                for k in d:
                    d[k] = min(d[k] + dt, self._SAT)
        for e in events or []:
            if not getattr(e, "actor_species", None) or getattr(e, "side", None) is None:
                continue
            key = (e.side, e.actor_species)
            if e.kind is EventKind.MOVE:
                self._acted[key] = 0
                self._seen[key] = 0
            elif e.kind is EventKind.SWITCH:
                self._seen[key] = 0
            elif e.kind is EventKind.DAMAGE:
                self._hit[key] = 0
        for side, sp in ((OURS, our_active), (OPP, opp_active)):
            if sp:
                self._seen[(side, sp)] = 0

    def values(self, side: str, species: Optional[str]):
        """→ (seen, acted, was_hit), each log-saturated to [0, 1]."""
        import math
        out = []
        for d in (self._seen, self._acted, self._hit):
            n = self._SAT if species is None else d.get((side, species), self._SAT)
            out.append(math.log1p(min(n, self._SAT)) / math.log(11.0))
        return tuple(out)


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
        # Episode-scoped no-progress counter (design §5.1). Updated at record()/embed time so the
        # obs is fresh; read by BOTH the obs encoder (value()) and the reward (last_penalty).
        self._progress_clock = ProgressClock()
        # E9 step 1 (roadmap §3.9): per-entity recency, fed by the same decision window.
        self._recency = RecencyTracker()
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
    def progress_clock(self) -> ProgressClock:
        return self._progress_clock

    @property
    def recency(self) -> RecencyTracker:
        return self._recency

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
        previous turn recorded yet.

        If our active mon CHANGED since the previous decision (a switch / forced
        replacement), the previous mask's MOVE bits describe the PREVIOUS mon's moves
        (sorted by that mon's ids), so they don't correspond to THIS mon's sorted move
        slots — the validity bit lands on an unrelated move. In that case the move bits
        are reset to the no-prior-info default (all-ones, like the first-turn case); the
        SWITCH bits stay (team-ordered, mon-independent). ``active_move_ids`` is a safe
        discriminator: it differs whenever the moveset differs, and is equal only when
        the sorted order already aligns. (gen3_move_slot_align_v1 — same alignment class.)"""
        if len(self._history) >= 2:
            prev_ctx = self._history[-2]
            reordered = reorder_move_bits_to_sorted(
                prev_ctx.mask.astype(np.float32), prev_ctx.active_move_ids
            )
            assert_sorted_validity_correct(
                reordered, prev_ctx.mask, prev_ctx.active_move_ids
            )
            cur_ctx = self._history[-1]
            if cur_ctx.active_move_ids != prev_ctx.active_move_ids:
                reordered[MOVE_START:MOVE_END] = 1.0
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

    # ------------------------------------------------------------------ #
    # Rolling-history snapshot / restore — for the opponent's stale RE-DECIDE
    # ------------------------------------------------------------------ #
    def snapshot(self) -> tuple:
        """Cheap shallow snapshot of the rolling-history state, taken before a self-play
        opponent decides so a stale RE-DECIDE can roll the superseded attempt back out
        (``RLPlayer.choose_move``). Without it, each re-decide would leave a phantom turn —
        the record() of the decision that never happened — in the opponent's turn-history obs.

        The bounded deques are short (≈``N_HISTORY_TURNS``) and ``BattleContext``s are immutable,
        so snapshotting the deques as plain lists is cheap and *shares* (never clones) the
        contexts. The ``HiddenPowerTracker`` and slot registries are deliberately NOT snapshotted:
        their updates are of real protocol events and idempotent (probability narrowing /
        stable slot ids), so a rolled-back attempt leaves them correct — re-observing the same
        events on the retry is a no-op. The trainee never re-decides (a stale trainee decision
        crashes), so only the opponent path uses this."""
        return (
            list(self._history),
            list(self._actions),
            list(self._cursors),
            list(self._hist_vec_cache),
            self._n_transitions,
            self._last_cursor,
            self._last_action,
        )

    def restore(self, snap: tuple) -> None:
        """Restore the state captured by :meth:`snapshot`, undoing the ``record()`` (and any
        memoized delta) that a stale, re-decided decision applied — so the committed
        turn-history never keeps a phantom turn. Rebuilding each deque from its snapshot list
        re-establishes the *exact* pre-attempt contents, including an entry ``maxlen`` would
        have since dropped, so the rollback is exact even when a deque sat at its cap."""
        history, actions, cursors, hist_vec, n_transitions, last_cursor, last_action = snap
        self._history.clear()
        self._history.extend(history)
        self._actions.clear()
        self._actions.extend(actions)
        self._cursors.clear()
        self._cursors.extend(cursors)
        self._hist_vec_cache.clear()
        self._hist_vec_cache.extend(hist_vec)
        self._n_transitions = n_transitions
        self._last_cursor = last_cursor
        self._last_action = last_action

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
        # FEASIBILITY GUARD (default since gen3_typed_hp_belief_v1). Two different things can make an
        # observation eliminate every candidate, and they deserve opposite responses:
        #
        #   * NO Hidden Power type could produce this effectiveness against this target. Then the
        #     TARGET IDENTIFICATION is wrong (the classic case: a switch resolved on our side in the
        #     same window, so the mon we resolved isn't the mon that was hit) — the observation is
        #     junk about a mon that was never involved. DISCARD it: narrowing on it would zero
        #     perfectly possible types, and raising would crash a run over a misattribution.
        #   * Some type could have produced it, but none of THIS SPECIES' surviving candidates can.
        #     That is a real contradiction — a tracker bug or a gap in the HP-type priors — and
        #     `observe` still RAISES on it, with its full per-species observation-log dump.
        #
        # So the guard removes the crash for the misattribution class WITHOUT weakening the GIGO
        # detector for the genuine one. Discards are counted rather than silent (see the tracker's
        # `infeasible_observations`) — a rising count means the target resolution is drifting.
        if not self._hidden_power_tracker.is_feasible(event.effectiveness, target):
            self._hidden_power_tracker.note_infeasible()
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
        """Fold the most-recent turn's TurnDelta from the event log. Returns an empty delta
        at episode start.

        ``battle`` (a Gen3Battle) supplies the event window via ``events_since``; the single
        event-fold path (``TurnDelta.build_from_events``) is always used. A standalone caller
        without an event log passes ``battle=None`` ⇒ an empty window, which folds the
        current-board snapshot for HP (see ``build_from_events``).
        """
        if len(self._history) < 2:
            return TurnDelta.empty()
        prev_ctx = self._history[-2]
        curr_ctx = self._history[-1]
        cursor = self._cursors[-1] if self._cursors else 0
        events = self._get_events_for_window(battle, cursor)
        return TurnDelta.build_from_events(prev_ctx, curr_ctx, self._last_action, events)

    def update_progress_clock(self, battle, legal) -> TurnDelta:
        """Fold the just-completed window's delta and advance the shared ``ProgressClock``, so the obs
        scalar (``value()``) and the reward's no-progress penalty (``last_penalty``) key on ONE value.

        Call from ``embed_battle`` AFTER :meth:`record`, BEFORE ``encode`` — poke-env runs
        ``embed_battle`` before ``calc_reward``, so updating here (not at reward time) keeps the obs
        fresh. Returns the folded delta so the caller can reuse it (the env caches it for
        ``calc_reward``, avoiding a second fold). The penalty magnitude lives on the clock itself
        (set once from the reward config), so this stays an obs-side call with no reward param.
        Single home for the 3-step protocol the env + inference players both need (no copy-paste).
        """
        delta = self.build_delta(battle=battle)
        live = battle.strict_view().live
        self._progress_clock.update(delta, live, legal)
        # E9 recency: the SAME per-decision window the newest TurnDelta slot folds
        # ([cursors[-1], now)), plus the live actives for the seen reset.
        if self._cursors and hasattr(battle, "events_since"):
            _ev = battle.events_since(self._cursors[-1])
        else:
            _ev = []
        self._recency.update(
            live.turn, _ev,
            live.ours.active.species if live.ours.active else None,
            live.opp.active.species if live.opp.active else None)
        return delta

    def _encode_delta_slot(self, i: int, encoder, battle) -> np.ndarray:
        """Encode the TurnDelta for history slot ``i`` (0 = most-recent), folded over its
        OWN bounded decision window: ``[cursors[-1-i] : cursors[-i])`` (``end=None`` ⇒ to
        "now" for the most-recent slot). The upper bound is what makes the slot represent
        its own turn and the per-step cost bounded; the most-recent slot's ``end=None``
        resolves to the same cursor that bounds it on the NEXT step, so a cached vector is
        bit-identical to recomputing it later — which is why the deque memoization is safe.

        Always folds via ``build_from_events``. ``battle=None`` (or a battle without an event
        log) ⇒ an empty window per slot — the standalone/test path.
        """
        action = self._actions[-1 - i]
        ctx_prev = self._history[-2 - i]
        ctx_curr = self._history[-1 - i]
        if battle is not None and hasattr(battle, "events_between") and self._cursors:
            start = self._cursors[-1 - i]
            end = None if i == 0 else self._cursors[-i]
            events = battle.events_between(start, end)
        else:
            events = []
        delta = TurnDelta.build_from_events(ctx_prev, ctx_curr, action, events)
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
            # Standalone / no-event-log path: recompute every slot, uncached (each folds
            # over an empty window — the current-board snapshot for HP).
            for i in range(available):
                result[n - 1 - i] = self._encode_delta_slot(i, encoder, battle)
            return result

        # Event path: extend the deque by the newly-completed delta(s) only. Use the
        # MONOTONIC transition count (not len(_history)-1, which caps with the deque).
        total_completed = self._n_transitions
        new = total_completed - self._n_cached_deltas
        if new == 1:
            # Common case: one turn finished since last call → encode just slot 0.
            self._hist_vec_cache.append(
                self._encode_delta_slot(0, encoder, battle)
            )
        elif new != 0:
            # Rare (calls skipped, or first call mid-episode): rebuild the kept tail.
            self._hist_vec_cache.clear()
            for i in range(available - 1, -1, -1):  # oldest-kept first → newest appended last
                self._hist_vec_cache.append(
                    self._encode_delta_slot(i, encoder, battle)
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
        self._progress_clock.reset()
