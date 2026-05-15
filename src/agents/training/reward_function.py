from __future__ import annotations
from typing import Protocol, runtime_checkable

from agents.training.battle_context import BattleContext, TurnDelta


@runtime_checkable
class RewardFunction(Protocol):
    """
    Protocol for trainee reward functions.

    Injected into Gen3Env at construction time. Battles not belonging to the
    trainee skip this entirely and fall back to poke-env's default reward helper,
    so implementations only ever see the trainee's battle.

    Implement all four methods to satisfy the protocol:
      - record_action       — called once per step, before the turn is processed,
                              with the context the model saw when it chose the action
      - process_turn_reward — called once per step, after the turn resolves,
                              with the resulting state delta
      - reset               — called at the start of each episode
      - report_episode      — called at episode end for logging/metrics
    """

    def record_action(self, ctx: BattleContext, action: int) -> None: ...

    def process_turn_reward(self, battle, delta: TurnDelta) -> float: ...

    def reset(self) -> None: ...

    def report_episode(self, battle) -> None: ...
