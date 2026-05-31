"""``StrictBattleView`` — the front door our code reads battle state through.

Why this exists
---------------
poke-env's ``Battle`` / ``Pokemon`` objects mix *current* facts with *temporal* ones
(``last_move``, ``last_cant_reason``, ``protect_counter``, mutable ``effects`` …). Any
consumer that reaches into the raw object can misread a past-turn field as current — the
Focus-Punch / Sleep-Talk class of bug. The fix is **encapsulation**: keep poke-env running
underneath as the state engine, but force *all of our code* to read through a small set of
vetted, extensively-fuzzed surfaces and never touch the raw battle.

:class:`StrictBattleView` is that boundary. It is constructed from a :class:`Gen3Battle`
(``battle.strict_view()``) and exposes **only**:

* :attr:`live`                  → :class:`~agents.battle.live_view.LiveView` (current board)
* :meth:`turn_view` / :attr:`history` → :class:`~agents.battle.turn_view.TurnView` (what happened)
* :attr:`legal`                 → :class:`~agents.battle.live_view.LegalActions` (server-truth legality)
* :meth:`events_since` / :attr:`event_cursor` → the per-decision event window
* meta passthroughs: :attr:`turn`, :attr:`battle_tag`, :attr:`finished`, :attr:`won`, :attr:`lost`

The raw ``Gen3Battle`` is held privately and **never returned**. Any other attribute access
raises via :meth:`__getattr__`, naming the right accessor — so a new raw read can't sneak in.

Note on ``turn`` (naming-collision resolution): the plan listed *both* a ``.turn(n)``
history accessor and a ``turn`` scalar meta field — but a method and a property can't share a
name. We resolve it the way that minimises surprise and naming divergence: ``turn`` is the
**scalar current-turn int**, identical in name and meaning to ``battle.turn`` and
``LiveView.turn`` (so migrating a consumer from ``battle.turn`` is a drop-in), and the
``TurnView`` accessor is :meth:`turn_view` — self-documenting (it returns a ``TurnView``,
mirroring :attr:`live` → ``LiveView`` and :attr:`history` → all ``TurnView``s).
"""

from __future__ import annotations

from typing import List

from agents.battle.battle_event import BattleEvent
from agents.battle.live_view import LegalActions, LiveView
from agents.battle.turn_view import TurnView


class StrictBattleView:
    """Encapsulating read-only view over a :class:`Gen3Battle`. The only object our
    non-``battle/`` code should hold; the raw battle stays unreachable behind it."""

    __slots__ = ("_battle",)

    def __init__(self, battle):
        self._battle = battle

    # ---- current board -------------------------------------------------- #
    @property
    def live(self) -> LiveView:
        """A fresh :class:`LiveView` snapshot of the current board."""
        return self._battle.live_view()

    # ---- legality (server-authoritative) -------------------------------- #
    @property
    def legal(self) -> LegalActions:
        """The current :class:`LegalActions` (parsed straight from the server request)."""
        return LegalActions.from_battle(self._battle)

    # ---- history -------------------------------------------------------- #
    def turn_view(self, n: int) -> TurnView:
        """The :class:`TurnView` for protocol turn ``n`` (what happened that turn).

        Named ``turn_view`` (not ``turn``) so the scalar :attr:`turn` can stay an int that
        matches ``battle.turn`` / ``LiveView.turn``; this returns a ``TurnView``, mirroring
        :attr:`live` → ``LiveView`` and :attr:`history` → all ``TurnView``s."""
        return TurnView.for_turn(self._battle, n)

    @property
    def history(self) -> List[TurnView]:
        """A :class:`TurnView` per turn played so far, in order (turn 1 … current)."""
        return [
            TurnView.for_turn(self._battle, t)
            for t in range(1, self._battle.turn + 1)
        ]

    # ---- per-decision event window -------------------------------------- #
    @property
    def event_cursor(self) -> int:
        """Seq of the next event — capture before acting, pass to :meth:`events_since`."""
        return self._battle.event_cursor

    def events_since(self, cursor: int) -> List[BattleEvent]:
        """Events revealed since ``cursor`` (the per-decision ``TurnDelta`` window)."""
        return self._battle.events_since(cursor)

    # ---- scalar meta passthroughs --------------------------------------- #
    @property
    def turn(self) -> int:
        """The current turn number — same name and meaning as ``battle.turn`` /
        ``LiveView.turn`` (use :meth:`turn_view` to fetch a turn's ``TurnView``)."""
        return self._battle.turn

    @property
    def battle_tag(self) -> str:
        return self._battle.battle_tag

    @property
    def finished(self) -> bool:
        return bool(self._battle.finished)

    @property
    def won(self):  # Optional[bool] — None until the battle ends
        return self._battle.won

    @property
    def lost(self):  # Optional[bool] — None until the battle ends
        return self._battle.lost

    # ---- the lock ------------------------------------------------------- #
    def __getattr__(self, name: str):
        """Called only when normal lookup fails. The raw battle is deliberately hidden,
        so guide the caller to the right vetted accessor rather than silently proxying."""
        raise AttributeError(
            f"{type(self).__name__!r} has no attribute {name!r}. The raw battle is "
            f"deliberately hidden behind this boundary — read current state via .live "
            f"(LiveView), legality via .legal (LegalActions), history via "
            f".turn_view(n)/.history (TurnView), or the event window via "
            f".events_since(cursor)/.event_cursor. The current turn number is .turn (an int, "
            f"like battle.turn). If you genuinely need a field none of these expose, widen "
            f"LiveView / LegalActions rather than reaching past the boundary."
        )

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        b = self._battle
        return (
            f"StrictBattleView(tag={b.battle_tag!r}, turn={b.turn}, "
            f"finished={bool(b.finished)})"
        )
