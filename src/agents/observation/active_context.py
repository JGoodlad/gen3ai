import numpy as np
from .base import ObservationEncoder
from .constants import ACTIVE_CONTEXT_DIM, BOOSTS_DIM, VOLATILES_DIM
from agents.observation.gen3_effects import (
    VOLATILE_DIM,
    encode_volatiles,
    describe_volatiles,
)
from typing import Any, Dict, Optional


class ActiveContextEncoder(ObservationEncoder):
    """Encodes the active mon's current-board context: stat boosts + the FULL gen3
    volatile set.

    Sourced from a :class:`~agents.battle.live_view.LivePokemon` (current state only —
    no history). Volatiles can only exist on the ACTIVE mon in gen3 (they clear on
    switch), so the full ``VOLATILE_DIM`` block lives here, on the two active slots,
    rather than wasting dims on the 12 team slots. The volatile vector is built by
    ``gen3_effects.encode_volatiles``, which **raises** on any volatile it can't
    classify (crash-don't-drop) — we never silently lose a battle fact.
    """

    def __init__(self, move_to_id: Optional[Dict[str, Any]] = None) -> None:
        self.move_to_id: Dict[str, Any] = move_to_id or {}

    @property
    def dimension(self) -> int:
        return ACTIVE_CONTEXT_DIM

    # Why the `type: ignore[override]` below — the ABC still declares the pre-ai_v4 `encode(item, battle)`.
    # The read-model migration moved this encoder (and GlobalEnv / Reactive / StateEncoder)
    # onto LiveView, so its SUBJECT is a LivePokemon, not a poke-env mon plus a battle.
    # Nothing calls these through the base, so the divergence is declared here rather than
    # papered over by widening the ABC to `*args` and losing the check for everyone else.
    def encode(self, live_mon: Optional[Any]) -> np.ndarray:  # type: ignore[override]
        """``live_mon`` is a ``LivePokemon`` (or None when there is no active mon)."""
        vec = np.zeros(self.dimension, dtype=np.float32)
        if live_mon is None:
            return vec

        cursor = 0
        # 1. Boosts (14) — 2 dims per stage [positive mag, negative mag], 7 stats.
        boosts = live_mon.boosts
        for stat in ("atk", "def", "spa", "spd", "spe", "accuracy", "evasion"):
            stage = boosts.get(stat, 0)
            vec[cursor] = max(0, stage) / 6.0
            vec[cursor + 1] = max(0, -stage) / 6.0
            cursor += 2

        # 2. Volatiles (VOLATILE_DIM) — full gen3 set, counters normalised. Raises
        #    UnknownVolatileError on an unclassified volatile (no silent drop).
        vec[cursor : cursor + VOLATILE_DIM] = encode_volatiles(live_mon.volatiles)
        cursor += VOLATILE_DIM

        return vec

    def get_layout(self) -> Dict[str, Any]:
        return {
            "boosts": {"offset": 0, "dim": BOOSTS_DIM},
            "volatiles": {"offset": BOOSTS_DIM, "dim": VOLATILES_DIM},
        }

    def describe_vector(self, vector: np.ndarray) -> Dict[str, Any]:
        stats = ["atk", "def", "spa", "spd", "spe", "acc", "eva"]
        boosts = {}
        for i, stat in enumerate(stats):
            val = int(vector[i * 2] * 6.0 - vector[i * 2 + 1] * 6.0)
            if val != 0:
                boosts[stat] = f"{val:+d}"

        vol_vec = vector[BOOSTS_DIM : BOOSTS_DIM + VOLATILE_DIM]
        return {
            "boosts": boosts,
            "volatiles": describe_volatiles(vol_vec),
        }
