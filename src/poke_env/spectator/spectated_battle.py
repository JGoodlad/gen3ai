import time
from typing import Dict, List, Optional


class SpectatedBattle:
    """
    Pure data object that accumulates Showdown protocol lines for a single battle.
    No file I/O — the caller decides what to do with the completed log.
    """

    def __init__(self, battle_tag: str) -> None:
        self._battle_tag = battle_tag
        self._lines: List[str] = []
        self._finished = False
        self._winner: Optional[str] = None
        self._turn: int = 0
        self._players: Dict[str, str] = {}   # "p1"/"p2" → username
        self._joined_at: float = time.time()

    def add_lines(self, split_messages: List[List[str]]) -> None:
        """Append a batch of already-split Showdown message lines to the log."""
        if self._finished:
            return
        for parts in split_messages:
            if len(parts) <= 1:
                continue
            msg_type = parts[1]
            if msg_type == "turn" and len(parts) > 2:
                try:
                    self._turn = int(parts[2])
                except ValueError:
                    pass
            elif msg_type == "player" and len(parts) > 3:
                # |player|p1|username|avatar|rating
                self._players[parts[2]] = parts[3]
            self._lines.append("|".join(parts))

    def finish(self, winner: Optional[str]) -> None:
        """Mark the battle complete. Idempotent — safe to call more than once."""
        if self._finished:
            return
        self._winner = winner
        if winner is not None:
            self._lines.append(f"|win|{winner}")
        else:
            self._lines.append("|tie")
        self._finished = True

    @property
    def finished(self) -> bool:
        return self._finished

    @property
    def battle_tag(self) -> str:
        return self._battle_tag

    @property
    def winner(self) -> Optional[str]:
        """Player name of the winner, or None for a tie."""
        return self._winner

    @property
    def turn(self) -> int:
        """Most recent turn number seen in the battle log (0 before the first |turn| line)."""
        return self._turn

    @property
    def players(self) -> Dict[str, str]:
        """Mapping of side ("p1"/"p2") to player username."""
        return dict(self._players)

    @property
    def joined_at(self) -> float:
        """Unix timestamp when this SpectatedBattle was created (room joined)."""
        return self._joined_at

    @property
    def log_text(self) -> str:
        """Full Showdown-format log as a single string, one protocol line per line."""
        return "\n".join(self._lines)
