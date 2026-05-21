from __future__ import annotations
from typing import TYPE_CHECKING, Optional
import numpy as np

from agents.training.battle_context import BattleContext, TurnDelta
from agents.training.slot_registry import SlotRegistry

if TYPE_CHECKING:
    from agents.observation.turn_delta_encoder import TurnDeltaEncoder


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

    @property
    def last_ctx(self) -> Optional[BattleContext]:
        return self._history[-1] if self._history else None

    @property
    def prev_mask(self) -> np.ndarray:
        """Action mask from the previous turn. All-ones if no previous turn recorded yet."""
        if len(self._history) >= 2:
            return self._history[-2].mask.astype(np.float32)
        return np.ones(11, dtype=np.float32)

    def record(self, battle, mask: np.ndarray, obs: np.ndarray) -> BattleContext:
        """Build and store a context snapshot for the current turn.

        Also commits the pending _last_action as the action taken FROM the
        previous context, so prev_N_delta_vecs() can reconstruct all N deltas.
        """
        if self._history:
            self._actions.append(self._last_action)
        ctx = BattleContext.from_battle(battle, mask, obs, self._our_slots, self._opp_slots)
        self._history.append(ctx)
        return ctx

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
