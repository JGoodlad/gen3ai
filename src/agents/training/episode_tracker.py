from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import numpy as np

from agents.training.battle_context import BattleContext, TurnDelta
from agents.training.hidden_power_tracker import HiddenPowerTracker
from agents.training.slot_registry import SlotRegistry

if TYPE_CHECKING:
    from poke_env.battle.pokemon import Pokemon
    from poke_env.battle.pokemon_type import PokemonType
    from poke_env.battle.status import Status

    from agents.observation.turn_delta_encoder import TurnDeltaEncoder


# Move ID for Baton Pass — only Gen 3 move that changes our side via a move action.
BATON_PASS = "batonpass"


@dataclass(frozen=True)
class _HpTargetMon:
    """Minimal mon-shaped object for HiddenPowerTracker.observe().

    Carries exactly the attributes effective_multiplier() reads. We construct
    one when we resolve an HP target so we can override .status with the value
    from the start of the turn HP fired in (some Fire moves thaw a frozen target
    and clear .status before we get to observe).
    """
    species: str
    type_1: "PokemonType"
    type_2: "PokemonType | None"
    ability: "str | None"
    status: "Status | None"


def _find_mon(battle, species: str) -> "Pokemon | None":
    return next((m for m in battle.team.values() if m.species == species), None)


def _resolve_hp_target(
    battle, prev: BattleContext, curr: BattleContext, delta: TurnDelta
) -> "_HpTargetMon | None":
    """Identify which of our mons was actually hit by the opponent's Hidden Power.

    Driven by *what we did* (read from `delta`), not by visible side state — the
    active mon at turn N+1 can match turn N's active even after a real switch
    (switch-in died, forced replacement cycled the same mon back). The structured
    `TurnDelta` is the single source of truth for "did we switch / move / which
    move"; we never re-decode raw `last_action` here.

    Resolution cases:
      A) prev.our_active is in `newly_fainted`: opp HP killed them on the field.
      B) Voluntary switch (delta.our_switch_to set): the switch resolves at
         priority +6 before HP — switch-in is the target.
      C) Baton Pass (delta.our_move_id == "batonpass" AND active species visibly
         changed): timing is speed-based. If `we_moved_first`, BP fired before
         HP and the switch-in is hit; otherwise HP hit prev.our_active first.
      D) Otherwise: prev.our_active stayed on the field and took the hit.

    For cases B and C, when the switch-in died from the HP, curr.our_active is
    a forced replacement; the dead switch-in is the lone member of
    `newly_fainted - {prev.our_active}`.

    Status is sourced from `prev.our_team_status` (snapshot at start of HP turn)
    rather than the live mon's current status. Required because Gen 3 Fire moves
    thaw their target — frozen Flash Fire mons take resisted (0.5×) damage AND
    lose the freeze in the same turn, so reading status post-resolution
    misidentifies the multiplier.
    """
    newly_fainted = curr.our_fainted_species - prev.our_fainted_species

    # Case A
    if prev.our_active in newly_fainted:
        return _snapshot_target(battle, prev.our_active, prev)

    voluntary_switch = delta.our_switch_to is not None
    visible_side_change = prev.our_active != curr.our_active
    baton_pass = delta.our_move_id == BATON_PASS and visible_side_change

    if voluntary_switch:
        switch_first = True
    elif baton_pass:
        switch_first = delta.we_moved_first is True
    else:
        switch_first = False

    if switch_first:
        # Cases B / C with switch-in on field for HP.
        switch_in_fainted = newly_fainted - {prev.our_active}
        if not switch_in_fainted:
            target_species = curr.our_active
        elif len(switch_in_fainted) == 1:
            target_species = next(iter(switch_in_fainted))
        else:
            raise RuntimeError(
                f"HP target ambiguous: multiple newly-fainted mons "
                f"{switch_in_fainted} on a turn opp used Hidden Power "
                f"(prev_active={prev.our_active}, curr_active={curr.our_active})."
            )
    else:
        # Case D, or BP fired AFTER HP.
        target_species = prev.our_active

    return _snapshot_target(battle, target_species, prev)


def _snapshot_target(
    battle, species: str, prev: BattleContext
) -> "_HpTargetMon | None":
    """Wrap a live mon with its status from the start of the HP turn."""
    if species == "NONE":
        return None
    live_mon = _find_mon(battle, species)
    if live_mon is None:
        return None
    return _HpTargetMon(
        species=live_mon.species,
        type_1=live_mon.type_1,
        type_2=live_mon.type_2,
        ability=live_mon.ability,
        status=prev.our_team_status.get(species, live_mon.status),
    )


class EpisodeTracker:
    """
    Tracks per-episode state needed to build observations and reward signals.

    Owns slot registries, a rolling history of BattleContexts, and a parallel
    list of actions taken at each context so historical TurnDeltas can be
    reconstructed for the N-turn history observation feature.
    """

    def __init__(self):
        self._our_slots = SlotRegistry()
        self._opp_slots = SlotRegistry()
        self._history: list[BattleContext] = []
        self._actions: list[int] = []   # _actions[i] = action taken FROM _history[i]
        self._last_action: int = -1
        self._hidden_power_tracker = HiddenPowerTracker()

    @property
    def hidden_power_tracker(self) -> HiddenPowerTracker:
        return self._hidden_power_tracker

    @property
    def last_ctx(self) -> Optional[BattleContext]:
        return self._history[-1] if self._history else None

    @property
    def prev_mask(self) -> np.ndarray:
        """Action mask from the previous turn. All-ones if no previous turn recorded yet."""
        if len(self._history) >= 2:
            return self._history[-2].mask.astype(np.float32)
        return np.ones(11, dtype=np.float32)

    def record(self, battle, mask: np.ndarray) -> BattleContext:
        """Build and store a context snapshot for the current turn.

        Also commits the pending _last_action as the action taken FROM the
        previous context, so prev_N_delta_vecs() can reconstruct all N deltas.
        Updates the HiddenPowerTracker BEFORE the env encodes the observation,
        so the encoded obs includes the just-fired HP's narrowing.
        """
        if self._history:
            self._actions.append(self._last_action)
        ctx = BattleContext.from_battle(battle, mask, self._our_slots, self._opp_slots)

        self._maybe_observe_hidden_power(battle, ctx)

        self._history.append(ctx)
        return ctx

    def _maybe_observe_hidden_power(self, battle, ctx: BattleContext) -> None:
        """Feed an HP observation to the tracker when applicable.

        Only fires on move_selection contexts (a forced-switch context can carry
        a stale opp_last_effectiveness from the prior turn). The `prev` snapshot
        used here is the most recent *move_selection* context — intervening
        forced-switch contexts in history would otherwise corrupt `newly_fainted`
        (the fainted mon is already in the forced-switch snapshot, so the diff
        against the current turn returns empty and the resolver falls through
        to the switch-in path, picking the wrong target).
        """
        if ctx.phase != "move_selection":
            return
        if ctx.opp_last_move_id != "hiddenpower" or ctx.opp_last_effectiveness is None:
            return

        prev_idx = self._find_prev_move_selection_index()
        if prev_idx is None:
            return
        prev = self._history[prev_idx]
        if prev.our_active == "NONE" or prev.opp_active == "NONE":
            return

        # Build TurnDelta once and consume its already-decoded action / timing
        # fields — never re-decode raw last_action here.
        action = self._actions[prev_idx] if prev_idx < len(self._actions) else self._last_action
        delta = TurnDelta.build(prev, ctx, action)
        target = _resolve_hp_target(battle, prev, ctx, delta)
        if target is not None:
            self._hidden_power_tracker.observe(
                prev.opp_active, ctx.opp_last_effectiveness, target
            )

    def _find_prev_move_selection_index(self) -> Optional[int]:
        """Most recent move_selection ctx in `_history`, or None.

        Skips intervening forced-switch contexts so prev-vs-curr deltas span a
        full turn rather than a mid-turn faint snapshot.
        """
        # _history was just appended in record(); skip the just-added ctx.
        for i in range(len(self._history) - 1, -1, -1):
            if self._history[i].phase == "move_selection":
                return i
        return None

    def advance(self, action: int) -> None:
        """Record the action chosen this turn, before the game steps forward."""
        self._last_action = action

    def build_delta(self) -> TurnDelta:
        """Diff between the last two turns. Returns an empty delta at episode start."""
        if len(self._history) < 2:
            return TurnDelta.empty()
        return TurnDelta.build(self._history[-2], self._history[-1], self._last_action)

    def prev_N_delta_vecs(self, n: int, encoder: "TurnDeltaEncoder") -> np.ndarray:
        """Return (n, TURN_DELTA_DIM) array of encoded TurnDeltas, oldest-first.

        Index n-1 is the most recent delta (same data as build_delta()).
        Turns not yet played are zero-padded.
        """
        result = np.zeros((n, encoder.dimension), dtype=np.float32)
        available = min(n, len(self._history) - 1, len(self._actions))
        for i in range(available):
            action = self._actions[-1 - i]
            ctx_prev = self._history[-2 - i]
            ctx_curr = self._history[-1 - i]
            delta = TurnDelta.build(ctx_prev, ctx_curr, action)
            result[n - 1 - i] = encoder.encode(delta)
        return result

    def reset(self) -> None:
        self._our_slots.reset()
        self._opp_slots.reset()
        self._history.clear()
        self._actions.clear()
        self._last_action = -1
        self._hidden_power_tracker.reset()
