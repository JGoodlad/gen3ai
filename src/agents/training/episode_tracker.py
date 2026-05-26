from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import numpy as np

from poke_env.battle.abstract_battle import DamagingMoveEvent

from agents.training.battle_context import BattleContext, TurnDelta
from agents.training.hidden_power_tracker import HiddenPowerTracker
from agents.training.slot_registry import SlotRegistry

if TYPE_CHECKING:
    from poke_env.battle.pokemon import Pokemon
    from poke_env.battle.pokemon_type import PokemonType
    from poke_env.battle.status import Status

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


def _find_mon(battle, species: str) -> "Pokemon | None":
    return next((m for m in battle.team.values() if m.species == species), None)


def _wrap_hp_target(battle, event: DamagingMoveEvent) -> Optional[_HpTargetMon]:
    """Look up the target's current type/ability from battle.team and overlay
    the status captured at the moment HP fired.

    Returns None if the target species can't be resolved (shouldn't happen for
    our-side targets — every HP hit names a real teammate).
    """
    live_mon = _find_mon(battle, event.target_species)
    if live_mon is None:
        return None
    return _HpTargetMon(
        species=live_mon.species,
        type_1=live_mon.type_1,
        type_2=live_mon.type_2,
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
        self._scan_opp_movesets_for_no_hp(battle)

        self._history.append(ctx)
        return ctx

    def _scan_opp_movesets_for_no_hp(self, battle) -> None:
        """Mark any opponent species whose four moves are fully revealed and
        none is Hidden Power as definitively HP-less.

        This converts the previously ambiguous "all-zero HP probs" state into a
        positive signal (hp_revealed=1, probs all zero in the encoder).
        Idempotent and cheap (~12 dict lookups per turn).
        """
        for mon in battle.opponent_team.values():
            if mon is None or not mon.species:
                continue
            moves = mon.moves
            if len(moves) >= 4 and not any(
                k.startswith("hiddenpower") for k in moves
            ):
                self._hidden_power_tracker.mark_no_hp(mon.species)

    def _maybe_observe_hidden_power(self, battle, ctx: BattleContext) -> None:
        """Feed an HP observation to the tracker when opp's last damaging move
        was Hidden Power.

        Everything we need — firer species, target species, target status at
        move-fire time, effectiveness — comes from ctx.opp_last_damaging_event,
        which is set by poke-env's protocol parser at the |move| line and
        finalized by the matching |-supereffective|/|-resisted|/|-immune|
        event. The event is turn-gated to the just-ended turn, so a stale HP
        from earlier in the battle never leaks. No before/after inference,
        no resolver, no edge cases for switches / Roar / phazing / faint
        chains — the protocol stated the facts and we just record them.
        """
        if ctx.phase != "move_selection":
            return
        event = ctx.opp_last_damaging_event
        if event is None or event.move_id != "hiddenpower":
            return
        target = _wrap_hp_target(battle, event)
        if target is None:
            return
        self._hidden_power_tracker.observe(
            event.user_species, event.effectiveness, target
        )

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
