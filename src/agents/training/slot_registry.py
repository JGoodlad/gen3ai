from __future__ import annotations


class SlotRegistry:
    """
    Assigns stable integer slots (0-5) to Pokémon species in first-seen order.

    poke-env yields battle.team.values() in insertion order, which changes as
    mons are revealed mid-battle. This registry pins each species to the slot it
    was first seen in so the observation vector always places the same species at
    the same index, turn after turn.
    """
    MAX_SLOTS = 6

    def __init__(self):
        self._slots: dict[str, int] = {}
        self._next = 0

    def assign(self, species: str) -> int:
        """Return the slot for species, assigning a new one on first call."""
        if species not in self._slots:
            if self._next >= self.MAX_SLOTS:
                raise RuntimeError(
                    f"SlotRegistry overflow: '{species}' exceeds {self.MAX_SLOTS} slots. "
                    f"Current slots: {list(self._slots)}"
                )
            self._slots[species] = self._next
            self._next += 1
        return self._slots[species]

    def get(self, species: str) -> int | None:
        return self._slots.get(species)

    def snapshot(self) -> dict[str, int]:
        return dict(self._slots)

    def reset(self):
        self._slots = {}
        self._next = 0
