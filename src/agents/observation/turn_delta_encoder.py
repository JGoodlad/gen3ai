"""
Encodes a TurnDelta into a fixed-dim float32 vector that is appended to the
per-turn observation, giving the feedforward model a one-turn memory without
requiring an RNN.

All values are normalized floats so they flow through the features_extractor's
projection layer without special embedding treatment.

Output layout (29 dims total):
  our_move  (5): id_norm, power_norm, has_secondary, has_recoil, type_id_norm
  opp_move  (5): same
  scalars  (19): our_switched, opp_switched, our_failed_to_move, opp_failed_to_move,
                 our_cant_onehot(5), opp_cant_onehot(5),
                 our_hp_delta, opp_hp_delta, we_fainted, opp_fainted, opp_move_known
"""
from __future__ import annotations
import numpy as np
from typing import Optional
from agents.training.battle_context import TurnDelta
from agents.observation.types import TypeEncoder

# Reason → one-hot index (unknown/other → all-zeros)
_CANT_REASONS = ["par", "slp", "frz", "flinch", "confusion"]

MOVE_FEAT_DIM = 5   # per-move feature block
CANT_DIM = 5        # one-hot over cant reasons
SCALAR_DIM = 4 + CANT_DIM * 2 + 5  # switched×2 + failed×2 + cant×2 + hp×2 + faint×2 + known
TURN_DELTA_DIM = MOVE_FEAT_DIM * 2 + SCALAR_DIM  # 5+5+19 = 29

_MAX_TYPE_ID = float(max(TypeEncoder.TYPE_TO_IDX.values(), default=1))


class TurnDeltaEncoder:
    """
    Stateless encoder: encodes a TurnDelta into a 27-dim float32 vector.
    Requires the gen3_moves mapping to look up move attributes by ID.
    """

    TURN_DELTA_DIM: int = TURN_DELTA_DIM

    def __init__(self, gen3_moves: dict):
        self._moves = gen3_moves
        nums = [v.get("num", 0) for v in gen3_moves.values()]
        self._max_num = float(max(nums, default=1))

    @property
    def dimension(self) -> int:
        return TURN_DELTA_DIM

    def _move_features(self, move_id: Optional[str]) -> np.ndarray:
        """5-dim move feature vector; zeros when move_id is None or unknown."""
        vec = np.zeros(MOVE_FEAT_DIM, dtype=np.float32)
        if move_id is None:
            return vec
        entry = self._moves.get(move_id)
        if entry is None:
            return vec
        vec[0] = entry.get("num", 0) / self._max_num
        vec[1] = min(entry.get("basePower", 0) / 200.0, 1.0)
        vec[2] = float(bool(entry.get("hasSecondary")))
        vec[3] = float(bool(entry.get("hasRecoil")))
        move_type = entry.get("type", "Normal").upper()
        type_id = TypeEncoder.TYPE_TO_IDX.get(move_type, 0)
        vec[4] = float(type_id) / _MAX_TYPE_ID
        return vec

    def _cant_onehot(self, reason: Optional[str]) -> np.ndarray:
        vec = np.zeros(CANT_DIM, dtype=np.float32)
        if reason is not None:
            try:
                vec[_CANT_REASONS.index(reason)] = 1.0
            except ValueError:
                pass
        return vec

    def encode(self, delta: TurnDelta) -> np.ndarray:
        our_hp_delta = float(delta.our_hp_delta.sum())
        opp_hp_delta = float(delta.opp_hp_delta.sum())
        scalars = np.array([
            float(delta.our_switch_to is not None),
            float(delta.opp_switch_to is not None),
            float(delta.our_failed_to_move),
            float(delta.opp_failed_to_move),
        ], dtype=np.float32)
        return np.concatenate([
            self._move_features(delta.our_move_id),        # 5
            self._move_features(delta.opp_move_id),        # 5
            scalars,                                        # 4
            self._cant_onehot(delta.our_cant_reason),      # 5
            self._cant_onehot(delta.opp_cant_reason),      # 5
            np.array([our_hp_delta, opp_hp_delta,          # 3
                      float(delta.we_fainted),
                      float(delta.opp_fainted),
                      float(delta.opp_move_known)],
                     dtype=np.float32),
        ])
