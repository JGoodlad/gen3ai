from __future__ import annotations
from typing import Optional
import numpy as np

from agents.training.battle_context import BattleContext, TurnDelta
from agents.training.slot_registry import SlotRegistry


class EpisodeTracker:
    """
    Tracks per-episode state needed to build observations and reward signals.

    Owns slot registries and a rolling history of BattleContexts. Currently
    keeps the full episode history (used as a 2-frame window for TurnDelta).
    Growing to N-frame turn history is a one-line change: cap _history to a
    deque of maxlen N and update build_delta() to expose more frames.
    """

    def __init__(self):
        self._our_slots = SlotRegistry()
        self._opp_slots = SlotRegistry()
        self._history: list[BattleContext] = []
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
        """Build and store a context snapshot for the current turn."""
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

    def reset(self) -> None:
        self._our_slots.reset()
        self._opp_slots.reset()
        self._history.clear()
        self._last_action = -1
