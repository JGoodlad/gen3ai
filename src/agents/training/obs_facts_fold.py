"""The OBS-FACTS fold (`gen3_obs_facts_v1`) — the two observation facts that need HISTORY.

Owned by :class:`~agents.training.event_window_tracker.EventWindowTracker` and fed the SAME
per-decision event window, each event once (the tracker's seq dedupe). It folds, per side:

* **the STINT** of that side's active mon — the moves it has used since it last entered the field
  (a ``|switch|`` / ``|drag|`` / faint ends a stint). It counts only a FREE selection: a called move
  (``from_move`` set — Sleep Talk's / Metronome's call) is not the mon's choice, and Struggle is
  what a locked mon falls back to, so neither is a new move. A ``lockedmove`` continuation (Outrage
  turn 2) and Pursuit's self tag carry no ``from_move`` (``Gen3Battle._delegated_from``) and count.
  The encoder reads the opponent's stint as CHOICE-LOCK EVIDENCE: the first move (the one a Choice
  Band would lock it into), whether a second DISTINCT move has been used (a proof it is NOT locked),
  and the trailing run of the same move.
* **the Encore / Disable start ADJUSTMENT** — Showdown adds one turn to an Encore (base
  ``encore`` ``onStart``) or a gen-3 Disable (gen4 mod ``disable.onStart``) when the TARGET will not
  move again this turn (``!this.queue.willMove(target)``), i.e. it already ACTED this turn. The fold
  records, at the ``-start`` line, whether the target's side had acted this turn (a ``|move|``, a
  ``|cant|``, or a confusion self-hit), so the encoder's min / max turns-left bounds are exact.

Mirrored line for line by the Rust core (``trackers::history::FactsFold``); the encoded block is
byte-gated by slice O. Pure state: no poke-env, no battle reference.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

from agents.battle.battle_event import OPP, OURS, EventKind

# The volatiles whose duration Showdown adjusts by the target's willMove at -start (gen 3).
ADJUSTED_EFFECTS = ("encore", "disable")


def effect_key(effect: Optional[str]) -> str:
    """``"move: Taunt"`` / ``"Encore"`` / ``"Disable"`` → ``"taunt"`` / ``"encore"`` /
    ``"disable"`` — lower-cased, a ``move:`` prefix dropped, alphanumerics only."""
    s = (effect or "").strip().lower()
    if s.startswith("move:"):
        s = s[5:]
    return "".join(c for c in s if c.isalnum())


class _Stint:
    __slots__ = ("first", "distinct2", "run_move", "run_len")

    def __init__(self) -> None:
        self.first: Optional[str] = None
        self.distinct2: bool = False
        self.run_move: Optional[str] = None
        self.run_len: int = 0

    def use(self, move_id: str) -> None:
        if self.first is None:
            self.first = move_id
        elif move_id != self.first:
            self.distinct2 = True
        if move_id == self.run_move:
            self.run_len += 1
        else:
            self.run_move = move_id
            self.run_len = 1


class ObsFactsFold:
    """See the module docstring."""

    def __init__(self) -> None:
        self._act_turn: Optional[int] = None
        self._acted: Dict[str, bool] = {OURS: False, OPP: False}
        self._stint: Dict[str, _Stint] = {OURS: _Stint(), OPP: _Stint()}
        self._adj: Dict[Tuple[str, str], bool] = {}

    def observe(self, e, et: int) -> None:
        """Fold ONE event (already seq-deduplicated by the caller) at event turn ``et``."""
        if et != self._act_turn:
            self._act_turn = et
            self._acted = {OURS: False, OPP: False}
        k = e.kind
        side = e.side
        if k is EventKind.MOVE:
            if side is None:
                return
            self._acted[side] = True
            mid = e.move_id
            if mid and not e.from_move and mid != "struggle":
                self._stint[side].use(mid)
        elif k is EventKind.CANT:
            s = e.blocked_side or side
            if s is not None:
                self._acted[s] = True
        elif k is EventKind.DAMAGE:
            if side is not None and (e.from_clause or "").strip().lower() == "confusion":
                self._acted[side] = True
        elif k in (EventKind.SWITCH, EventKind.DRAG, EventKind.FAINT):
            if side is not None:
                self._stint[side] = _Stint()
        elif k is EventKind.VOLATILE_START:
            if side is not None and e.value.get("op") == "start":
                key = effect_key(e.effect)
                if key in ADJUSTED_EFFECTS:
                    self._adj[(side, key)] = self._acted[side]

    # ------------------------------------------------------------------ reads
    def stint(self, side: str) -> Tuple[Optional[str], bool, int]:
        """``(first move id or None, a second distinct move was used, trailing run length)``."""
        st = self._stint[side]
        return st.first, st.distinct2, st.run_len

    def adjusted(self, side: str, effect: str) -> Optional[bool]:
        """Whether the side's latest ``effect`` start got Showdown's +1 (the target had already
        acted that turn); ``None`` when no start of it was folded."""
        return self._adj.get((side, effect))
