"""``Choice`` — a poke-env-free, tagged representation of a chosen action.

The action pipeline has three stages, and this is the seam between the first two:

1. **decode** (pure): ``Gen3ActionMapper.action_to_choice(action_idx, legal) -> Choice``
   resolves a discrete action index against a captured :class:`LegalActions` snapshot
   into a tagged ``Choice``. No poke-env objects, fully testable with a stub.
2. **serialize** (the one poke-env touch): ``serialize.choice_to_order(choice, battle)``
   turns a ``Choice`` into the ``BattleOrder`` the client sends.

Keeping ``Choice`` free of any poke-env import is the point — it lets the decode step
(the part with all the index/legality logic) be exercised without a live battle, and it
confines every poke-env order type to the single serialization adapter.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class ChoiceKind(Enum):
    """Which of the three action families a :class:`Choice` is."""

    MOVE = "move"
    SWITCH = "switch"
    STRUGGLE = "struggle"


@dataclass(frozen=True)
class Choice:
    """An immutable, poke-env-free description of one chosen action.

    Construct via the classmethods rather than the raw constructor so the tag and its
    payload can't drift apart:

    * :meth:`move`     — use the move in request slot ``slot`` (0–3); ``move_id`` is the
      move's id as the server listed it (``hiddenpower`` stays bare — the serializer
      reconciles it to poke-env's typed id).
    * :meth:`switch`   — switch to the bench mon at team ``slot`` (0–5); ``species`` is
      carried for the serializer's lookup + diagnostics.
    * :meth:`struggle` — the active mon must Struggle (all PP gone).
    """

    kind: ChoiceKind
    move_id: Optional[str] = None       # MOVE only
    move_slot: Optional[int] = None     # MOVE only — request slot 0–3
    switch_species: Optional[str] = None  # SWITCH only
    switch_slot: Optional[int] = None     # SWITCH only — team slot 0–5

    @classmethod
    def move(cls, move_id: str, slot: int) -> "Choice":
        return cls(kind=ChoiceKind.MOVE, move_id=move_id, move_slot=slot)

    @classmethod
    def switch(cls, species: str, slot: int) -> "Choice":
        return cls(kind=ChoiceKind.SWITCH, switch_species=species, switch_slot=slot)

    @classmethod
    def struggle(cls) -> "Choice":
        return cls(kind=ChoiceKind.STRUGGLE)
