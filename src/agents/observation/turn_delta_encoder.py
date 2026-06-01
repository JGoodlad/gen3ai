"""
Encodes a TurnDelta into a fixed-dim float32 vector that is appended to the
per-turn observation, giving the feedforward model a one-turn memory.

Output layout (TURN_DELTA_DIM dims total; all offsets are computed from the
named OFFSET_* / *_DIM constants below — never hardcode indices):

  Base block (indices 0 .. TURN_DELTA_BASE_DIM-1):
    our_move  (5): move_id (raw int as float), power_norm, has_secondary, has_recoil, type_id (raw int as float)
    opp_move  (5): same
    scalars:       our_switched, opp_switched, our_failed_to_move, opp_failed_to_move,
                   our_cant_onehot(CANT_DIM), opp_cant_onehot(CANT_DIM),
                   our_hp_delta_sum, opp_hp_delta_sum, we_fainted, opp_fainted, opp_move_known,
                   our_effectiveness_onehot(4), opp_effectiveness_onehot(4),
                   move_order(2)
    The cant one-hot uses gen3_effects.CANT_DIM (12) and gen3_effects.encode_cant_reason.
    Unknown/None → zeros (no crash). Reasons: slp/frz/par/flinch/recharge/attract/
    disable/taunt/imprison/focuspunch/nopp/truant (source-derived from Showdown moves+abilities).

  Extended block (gen3_unified_v2 + gen3_move_outcome_v1 + gen3_turn_delta_v2):
    our_boost_delta (7), opp_boost_delta (7),
    phase_is_forced_switch (1),
    our_target_hp_delta (1), opp_target_hp_delta (1),
    our_hp_levels (6), opp_hp_levels (6),
    our_target_status_onehot (7), opp_target_status_onehot (7),
    our_move_outcome_onehot (3, [hit, miss, fail]),
    opp_move_outcome_onehot (3, [hit, miss, fail]),
    our_move_crit (1), opp_move_crit (1),
    --- gen3_turn_delta_v2 additions ---
    our_faint_causes (FAINT_CAUSE_DIM=8, multi-hot),
    opp_faint_causes (FAINT_CAUSE_DIM=8, multi-hot),
    our_attempted_move_id (1, raw int — embedded by extractor via move_embedding),
    --- species block (6 IDs, embedded by extractor per the manifest) ---
    our_actor_species_id (1, raw int as float),
    opp_actor_species_id (1, raw int as float),
    our_target_species_id (1, raw int as float),
    opp_target_species_id (1, raw int as float),
    our_switch_to_species_id (1, raw int as float),
    opp_switch_to_species_id (1, raw int as float),
    --- gen3_trapping_signals_v1 additions ---
    attempted_switch_rejected (1, bit): the server REFUSED a switch we chose this window
      (|error|[Unavailable choice] — we tried to pivot and got trapped). Folded from the
      out-of-band CHOICE_REJECTED event. Zero on every turn with no rejection.
    our_attempted_switch_to_species_id (1, raw int — embedded): the mon we PRESSED a switch
      to (intent). On a rejected pivot, our_switch_to is 0 (the switch never happened) while
      this names the mon we tried to bring in; on a successful switch it equals
      our_switch_to_species_id. 0 (unknown sentinel) when we didn't press a switch.

  Embedded-ID positions (move/type/species) are declared in TURN_DELTA_EMBEDDED_IDS
  (the manifest) — the extractor reads it directly, so the species block no longer
  needs to be a contiguous tail and there are no hardcoded positions in the extractor.
  Effectiveness one-hot (4): [immune, resisted, normal, super-effective]. Move-outcome
  one-hot (3): [hit, miss, fail]; all-zeros when the side switched / was prevented by
  |cant| / used no move. crit is orthogonal.

  Dropped vs the first gen3_turn_delta_v2 draft: faint_count scalars (redundant with
  we_fainted/opp_fainted + cause popcount) and attempted_switch_to (switches always
  execute, so it always equalled switch_to — only attempted_MOVE carries signal, since
  a move can be prevented by freeze/sleep/flinch/cant/KO-before-act).

FAINT_CAUSE_DIM=8 vocab (attack/hazard/weather/status/recoil/selfko/leechseed/other):
  Multi-hot — the rare 2-on-one-side window (Pursuit/Future-Sight into a low-HP mon on
  hazards) can set 2 bits. All-zeros when no faint. our_faint_causes tracks our-side
  faints; opp_faint_causes tracks theirs (the common Explosion double-KO is one bit each).

target_status_onehot: status of the named target AT MOVE-FIRE TIME. 7 states
  (BRN, FNT, FRZ, PAR, PSN, SLP, TOX); all-zeros when side didn't use a
  damaging move or target had no status.

move_id, type_id, attempted_move_id, and species_id positions are raw ints
(float32 exact for < 2^24). features_extractor routes them through the
appropriate embedding tables.

Move-order (2 dims): [we_first, opp_first]; all-zeros = na / unknown.

Actor species ID: prefers damaging_event.user_species (protocol-truth), falls
back to prev_active for switches and non-damaging moves. 0 = unknown.

Target species ID: from the OTHER side's damaging_event.target_species.
0 when the other side didn't use a confirmed damaging move.
"""
from __future__ import annotations
import numpy as np
from typing import Optional
from agents.training.turn_delta import TurnDelta
from agents.observation.types import TypeEncoder
from agents.observation.gen3_effects import (
    CANT_DIM,
    encode_cant_reason,
)
from agents.battle.turn_view import FAINT_CAUSE_DIM, FAINT_CAUSE_VOCAB
from agents.gen3_mechanics import BOOST_DIM
from agents.enums import Status


# Move-outcome one-hot: the resolved fate of a side's move this turn. None (the
# side switched, was prevented by |cant|, or used no identifiable move) → zeros.
_OUTCOME_ORDER = ["hit", "miss", "fail"]
_OUTCOME_TO_IDX = {o: i for i, o in enumerate(_OUTCOME_ORDER)}

# Status one-hot index. None / no status → all-zeros (the natural sentinel).
# Order is stable; do not reorder without bumping ARCH_SIGNATURE.
_STATUS_ORDER: tuple = (
    Status.BRN,
    Status.FNT,
    Status.FRZ,
    Status.PAR,
    Status.PSN,
    Status.SLP,
    Status.TOX,
)
_STATUS_TO_IDX = {s: i for i, s in enumerate(_STATUS_ORDER)}
_STATUS_LABELS = [s.name for s in _STATUS_ORDER]
STATUS_DIM = len(_STATUS_ORDER)  # 7

MOVE_FEAT_DIM = 5   # per-move feature block
# CANT_DIM is imported from gen3_effects (12 source-derived gen3 cant reasons).
OUTCOME_DIM = len(_OUTCOME_ORDER)  # one-hot over move outcomes (3: hit/miss/fail)
EFF_DIM = 4         # one-hot: immune | resisted | normal | super-effective
ORDER_DIM = 2       # binary: [we_first, opp_first]; all-zero = na / unknown
SCALAR_DIM = 4 + CANT_DIM * 2 + 5 + EFF_DIM * 2 + ORDER_DIM
# FAINT_CAUSE_DIM is imported from turn_view (8 cause labels).

# Base block (indices 0..TURN_DELTA_BASE_DIM-1).
TURN_DELTA_BASE_DIM = MOVE_FEAT_DIM * 2 + SCALAR_DIM

# Named offsets into the base block. All computed from the dims above so they
# stay correct when CANT_DIM changes (e.g. adding a cant reason). describe_vector
# and the e2e fuzz tests reference these instead of magic indices.
OFFSET_OUR_MOVE_BLOCK       = 0                                        # (5 dims)
OFFSET_OPP_MOVE_BLOCK       = MOVE_FEAT_DIM                            # (5 dims)
OFFSET_OUR_SWITCHED         = MOVE_FEAT_DIM * 2                        # +0
OFFSET_OPP_SWITCHED         = OFFSET_OUR_SWITCHED + 1
OFFSET_OUR_FAILED_TO_MOVE   = OFFSET_OPP_SWITCHED + 1
OFFSET_OPP_FAILED_TO_MOVE   = OFFSET_OUR_FAILED_TO_MOVE + 1
OFFSET_OUR_CANT             = OFFSET_OPP_FAILED_TO_MOVE + 1            # (CANT_DIM)
OFFSET_OPP_CANT             = OFFSET_OUR_CANT + CANT_DIM               # (CANT_DIM)
OFFSET_OUR_HP_DELTA_SUM     = OFFSET_OPP_CANT + CANT_DIM               # summed our hp delta scalar
OFFSET_OPP_HP_DELTA_SUM     = OFFSET_OUR_HP_DELTA_SUM + 1
OFFSET_WE_FAINTED           = OFFSET_OPP_HP_DELTA_SUM + 1
OFFSET_OPP_FAINTED          = OFFSET_WE_FAINTED + 1
OFFSET_OPP_MOVE_KNOWN       = OFFSET_OPP_FAINTED + 1
OFFSET_OUR_EFF              = OFFSET_OPP_MOVE_KNOWN + 1                # (EFF_DIM)
OFFSET_OPP_EFF              = OFFSET_OUR_EFF + EFF_DIM                 # (EFF_DIM)
OFFSET_ORDER                = OFFSET_OPP_EFF + EFF_DIM                 # (ORDER_DIM)
assert OFFSET_ORDER + ORDER_DIM == TURN_DELTA_BASE_DIM

# Extended block:
#   gen3_unified_v2: boost deltas, phase flag, target_hp_deltas, HP levels,
#   gen3_move_outcome_v1: target status onehots, move-outcome, crit,
#   gen3_turn_delta_v2: faint-cause multi-hots + attempted move id + status transitions.
# (faint_count scalars dropped — redundant with we_fainted/opp_fainted + cause popcount;
#  attempted_switch_to dropped — switches always execute, so it == switch_to always.)
HP_LEVEL_DIM = 6
SPECIES_ID_COUNT = 6  # our/opp × actor/target/switch_to
TURN_DELTA_EXT_DIM = (
    BOOST_DIM * 2          # 14 — our/opp boost deltas
    + 1                    # phase_is_forced_switch
    + 2                    # our/opp target_hp_delta
    + HP_LEVEL_DIM * 2     # 12 — per-slot HP levels for both sides
    + STATUS_DIM * 2       # 14 — target status onehots (at move-fire time)
    + OUTCOME_DIM * 2      # 6  — our/opp move-outcome onehots (hit/miss/fail)
    + 2                    # our/opp crit bit
    + FAINT_CAUSE_DIM * 2  # 16 — our/opp faint-cause multi-hots
    + STATUS_DIM * 4       # 28 — our/opp status APPLIED + CURED this window (transition)
    + 2                    # our/opp item_used bit (resource event; identity in item block)
    + 1                    # our_attempted_move_id (raw int — embedded by extractor)
    + SPECIES_ID_COUNT     # 6  — species IDs (embedded by extractor)
    # gen3_trapping_signals_v1: rejected-switch history.
    + 1                    # attempted_switch_rejected bit
    + 1                    # our_attempted_switch_to species id (embedded by extractor)
)

TURN_DELTA_DIM = TURN_DELTA_BASE_DIM + TURN_DELTA_EXT_DIM  # 53 + 106 = 159

# Offsets into the slot vector for the extended block. The embedded-ID positions
# (move/type/species) are NOT hardcoded in the extractor — they're declared in the
# TURN_DELTA_EMBEDDED_IDS manifest below, which both this encoder's layout and
# features_extractor.embed_delta_slot consume. So the species block no longer needs
# to be a contiguous tail; the manifest is the single source of truth.
OFFSET_OUR_BOOST_DELTA      = TURN_DELTA_BASE_DIM
OFFSET_OPP_BOOST_DELTA      = OFFSET_OUR_BOOST_DELTA + BOOST_DIM
OFFSET_PHASE_FORCED_SWITCH  = OFFSET_OPP_BOOST_DELTA + BOOST_DIM
OFFSET_OUR_TARGET_HP_DELTA  = OFFSET_PHASE_FORCED_SWITCH + 1
OFFSET_OPP_TARGET_HP_DELTA  = OFFSET_OUR_TARGET_HP_DELTA + 1
OFFSET_OUR_HP_LEVELS        = OFFSET_OPP_TARGET_HP_DELTA + 1
OFFSET_OPP_HP_LEVELS        = OFFSET_OUR_HP_LEVELS + HP_LEVEL_DIM
OFFSET_OUR_TARGET_STATUS    = OFFSET_OPP_HP_LEVELS + HP_LEVEL_DIM
OFFSET_OPP_TARGET_STATUS    = OFFSET_OUR_TARGET_STATUS + STATUS_DIM
OFFSET_OUR_MOVE_OUTCOME     = OFFSET_OPP_TARGET_STATUS + STATUS_DIM         # (OUTCOME_DIM)
OFFSET_OPP_MOVE_OUTCOME     = OFFSET_OUR_MOVE_OUTCOME + OUTCOME_DIM
OFFSET_OUR_CRIT             = OFFSET_OPP_MOVE_OUTCOME + OUTCOME_DIM
OFFSET_OPP_CRIT             = OFFSET_OUR_CRIT + 1
# gen3_turn_delta_v2 additions.
OFFSET_OUR_FAINT_CAUSES     = OFFSET_OPP_CRIT + 1                           # (FAINT_CAUSE_DIM)
OFFSET_OPP_FAINT_CAUSES     = OFFSET_OUR_FAINT_CAUSES + FAINT_CAUSE_DIM     # (FAINT_CAUSE_DIM)
# Status transitions this window (applied = gained, cured = lost), per side.
OFFSET_OUR_STATUS_APPLIED   = OFFSET_OPP_FAINT_CAUSES + FAINT_CAUSE_DIM     # (STATUS_DIM)
OFFSET_OUR_STATUS_CURED     = OFFSET_OUR_STATUS_APPLIED + STATUS_DIM        # (STATUS_DIM)
OFFSET_OPP_STATUS_APPLIED   = OFFSET_OUR_STATUS_CURED + STATUS_DIM          # (STATUS_DIM)
OFFSET_OPP_STATUS_CURED     = OFFSET_OPP_STATUS_APPLIED + STATUS_DIM        # (STATUS_DIM)
# Item-used bits (this side's active lost/spent its item this window). Just a BIT —
# the WHICH lives in the per-mon item block; this marks the resource event + timing.
OFFSET_OUR_ITEM_USED        = OFFSET_OPP_STATUS_CURED + STATUS_DIM          # 1
OFFSET_OPP_ITEM_USED        = OFFSET_OUR_ITEM_USED + 1                      # 1
OFFSET_OUR_ATTEMPTED_MOVE   = OFFSET_OPP_ITEM_USED + 1                      # 1 (embedded move id)
# Species block (6 IDs — embedded; positions declared in the manifest below).
OFFSET_OUR_ACTOR_SPECIES    = OFFSET_OUR_ATTEMPTED_MOVE + 1
OFFSET_OPP_ACTOR_SPECIES    = OFFSET_OUR_ACTOR_SPECIES + 1
OFFSET_OUR_TARGET_SPECIES   = OFFSET_OPP_ACTOR_SPECIES + 1
OFFSET_OPP_TARGET_SPECIES   = OFFSET_OUR_TARGET_SPECIES + 1
OFFSET_OUR_SWITCH_TO_SPEC   = OFFSET_OPP_TARGET_SPECIES + 1
OFFSET_OPP_SWITCH_TO_SPEC   = OFFSET_OUR_SWITCH_TO_SPEC + 1
# gen3_trapping_signals_v1: rejected-switch history (append-only).
OFFSET_ATTEMPTED_SWITCH_REJECTED = OFFSET_OPP_SWITCH_TO_SPEC + 1            # 1 scalar bit
OFFSET_OUR_ATTEMPTED_SWITCH_SPEC = OFFSET_ATTEMPTED_SWITCH_REJECTED + 1     # 1 embedded species id
assert OFFSET_OUR_ATTEMPTED_SWITCH_SPEC + 1 == TURN_DELTA_DIM, (
    f"Layout mismatch: last offset+1={OFFSET_OUR_ATTEMPTED_SWITCH_SPEC + 1} != TURN_DELTA_DIM={TURN_DELTA_DIM}"
)

# --------------------------------------------------------------------------- #
# Embedded-ID manifest — THE single source of truth for which slot positions
# carry raw embedding IDs and which table each routes to. features_extractor's
# embed_delta_slot consumes this list verbatim (gather → clamp → embed in this
# order); every position NOT listed here is a pass-through scalar. Adding an
# embedded ID is a ONE-LINE change here that both the encoder layout and the
# extractor pick up — no hardcoded positions, no hand-maintained width formula,
# and no silent "raw id flows through as a scalar" failure mode.
#
# Order matters: it fixes the concatenation order of the embedded outputs, so it
# is part of the arch layout (reordering = retrain). Append new entries.
# --------------------------------------------------------------------------- #
TURN_DELTA_EMBEDDED_IDS: tuple = (
    (OFFSET_OUR_MOVE_BLOCK + 0, "move"),   # our move id
    (OFFSET_OUR_MOVE_BLOCK + 4, "type"),   # our move type
    (OFFSET_OPP_MOVE_BLOCK + 0, "move"),   # opp move id
    (OFFSET_OPP_MOVE_BLOCK + 4, "type"),   # opp move type
    (OFFSET_OUR_ATTEMPTED_MOVE, "move"),   # our attempted move id
    (OFFSET_OUR_ACTOR_SPECIES,  "species"),
    (OFFSET_OPP_ACTOR_SPECIES,  "species"),
    (OFFSET_OUR_TARGET_SPECIES, "species"),
    (OFFSET_OPP_TARGET_SPECIES, "species"),
    (OFFSET_OUR_SWITCH_TO_SPEC, "species"),
    (OFFSET_OPP_SWITCH_TO_SPEC, "species"),
    (OFFSET_OUR_ATTEMPTED_SWITCH_SPEC, "species"),  # gen3_trapping_signals_v1
)
_EMBEDDED_POSITIONS = frozenset(pos for pos, _ in TURN_DELTA_EMBEDDED_IDS)
assert len(_EMBEDDED_POSITIONS) == len(TURN_DELTA_EMBEDDED_IDS), (
    "duplicate position in TURN_DELTA_EMBEDDED_IDS"
)
# Pass-through scalar positions: every slot index not in the embedded manifest,
# ascending. embed_delta_slot index_selects these after the embedded outputs.
TURN_DELTA_SCALAR_OFFSETS: tuple = tuple(
    i for i in range(TURN_DELTA_DIM) if i not in _EMBEDDED_POSITIONS
)


class TurnDeltaEncoder:
    """
    Stateless encoder: encodes a TurnDelta into a `TURN_DELTA_DIM`-dim
    (currently 88) float32 vector. Requires the gen3_moves and gen3_species
    mappings to look up move/species attributes by ID.
    """

    TURN_DELTA_DIM: int = TURN_DELTA_DIM

    def __init__(self, gen3_moves: dict, gen3_species: Optional[dict] = None):
        self._moves = gen3_moves
        self._species = gen3_species or {}
        nums = [v.get("num", 0) for v in gen3_moves.values()]
        self._max_num = float(max(nums, default=1))
        # Build reverse map preferring "hiddenpower" base over typed variants for num=237
        self._num_to_name = {}
        for k, v in gen3_moves.items():
            if "num" not in v:
                continue
            num = v["num"]
            if num not in self._num_to_name or k == "hiddenpower":
                self._num_to_name[num] = k
        self._idx_to_type = TypeEncoder.IDX_TO_TYPE
        # Species name → num lookup; reverse map for describe_vector.
        self._species_num = {
            name: int(entry.get("num", 0))
            for name, entry in self._species.items()
        }
        self._num_to_species = {
            num: name for name, num in self._species_num.items() if num > 0
        }

    @property
    def dimension(self) -> int:
        return TURN_DELTA_DIM

    def _species_id(self, species: Optional[str]) -> float:
        """Map a species name to its embedding-table index.

        Returns 0.0 for the "no species this turn / unrevealed" sentinels
        (None, "NONE", "NULL") — that's the legitimate empty case where there
        is genuinely no actor / target / switch_to to identify.

        Raises ValueError for a non-sentinel species name that isn't in the
        gen3_species mapping. That case means either poke-env handed us a
        species ID we don't know about or the JSON mapping is out of date —
        both are bugs we want to fail loudly on rather than silently train
        with the wrong embedding (id 0 collides with the unrevealed sentinel).
        Matches SpeciesEncoder.encode()'s behavior.

        Tests that exercise the encoder must pass a species mapping that
        covers every species name reachable through their delta fixtures.
        """
        if species is None or species == "NONE" or species == "NULL":
            return 0.0
        num = self._species_num.get(species)
        if num is None:
            raise ValueError(
                f"Unrecognized species in TurnDelta: {species!r}. "
                "Either poke-env produced an unknown species ID or the "
                "gen3_species.json mapping is out of date."
            )
        return float(num)

    def _move_features(self, move_id: Optional[str]) -> np.ndarray:
        """5-dim move feature vector; zeros when move_id is None or unknown.
        Positions 0 and 4 carry raw int IDs (as float32) for the extractor to embed."""
        vec = np.zeros(MOVE_FEAT_DIM, dtype=np.float32)
        if move_id is None:
            return vec
        entry = self._moves.get(move_id)
        if entry is None:
            return vec
        vec[0] = float(entry.get("num", 0))          # raw move num — embedded by extractor
        vec[2] = float(bool(entry.get("hasSecondary")))
        vec[3] = float(bool(entry.get("hasRecoil")))
        if move_id == "hiddenpower":
            # Gen 3 Showdown sends "Hidden Power" without type; type is unknowable from the
            # battle log. Use 70bp (competitive assumption) and type_id=0 (unknown sentinel,
            # distinct from Normal=1) so the model learns HP ≠ Normal-type move.
            vec[1] = 70.0 / 200.0
            # vec[4] stays 0 (unknown)
        else:
            vec[1] = min(entry.get("basePower", 0) / 200.0, 1.0)
            move_type = entry.get("type", "Normal").upper()
            vec[4] = float(TypeEncoder.TYPE_TO_IDX.get(move_type, 0))
        return vec

    def _move_id(self, move_id: Optional[str]) -> float:
        """Raw move num (0 = no move / unknown). Used for single-dim embedded IDs."""
        if move_id is None:
            return 0.0
        entry = self._moves.get(move_id)
        if entry is None:
            return 0.0
        return float(entry.get("num", 0))

    def _effectiveness_onehot(self, mult: Optional[float]) -> np.ndarray:
        """4-dim one-hot over [immune, resisted, normal, super-effective]; zeros when None."""
        vec = np.zeros(EFF_DIM, dtype=np.float32)
        if mult is None:
            return vec
        if mult == 0.0:
            vec[0] = 1.0
        elif mult <= 0.5:
            vec[1] = 1.0
        elif mult == 1.0:
            vec[2] = 1.0
        else:
            vec[3] = 1.0
        return vec

    def _order_onehot(self, we_first: Optional[bool]) -> np.ndarray:
        """2-dim: [we_first, opp_first]; all-zero when na or unknown."""
        vec = np.zeros(ORDER_DIM, dtype=np.float32)
        if we_first is True:
            vec[0] = 1.0
        elif we_first is False:
            vec[1] = 1.0
        return vec

    @staticmethod
    def _cant_onehot(reason: Optional[str]) -> np.ndarray:
        """One-hot a cant reason using the gen3_effects authoritative vocab
        (crash-don't-drop).

        ``reason=None`` (side moved normally) → all-zeros. A non-None reason
        outside the source-derived gen3 vocab **raises**
        :class:`UnknownCantReasonError` — this encoder is the FIRST and ONLY
        place a cant reason is validated against the vocab (``gen3_battle``
        records the ``|cant|`` reason verbatim, without normalising), so the
        tripwire must live here. The cant onehot has no "other" slot, so a
        silent all-zeros fallback would be indistinguishable from "moved
        normally" — exactly the silent drop the invariant forbids. An unknown
        gen3 cant reason means the vocab is stale: add it to
        ``gen3_effects.CANT_REASONS`` rather than swallowing it here."""
        return encode_cant_reason(reason)

    @staticmethod
    def _outcome_onehot(outcome: Optional[str]) -> np.ndarray:
        """3-dim one-hot over [hit, miss, fail]; all-zeros when outcome is None
        (the side switched, was prevented by |cant|, or used no move)."""
        vec = np.zeros(OUTCOME_DIM, dtype=np.float32)
        if outcome is not None:
            idx = _OUTCOME_TO_IDX.get(outcome)
            if idx is not None:
                vec[idx] = 1.0
        return vec

    @staticmethod
    def _status_onehot(status: Optional[Status]) -> np.ndarray:
        """7-dim onehot over Status enum; all-zeros when status is None.

        Status values are captured AT MOVE-FIRE TIME (per DamagingMoveEvent
        docstring), which matters for Gen 3 Flash Fire-vs-frozen: a frozen
        Flash Fire holder takes Fire-type 0.5× damage AND thaws in the same
        hit, so reading status from the next snapshot misses the FRZ.
        """
        vec = np.zeros(STATUS_DIM, dtype=np.float32)
        if status is None:
            return vec
        idx = _STATUS_TO_IDX.get(status)
        if idx is not None:
            vec[idx] = 1.0
        return vec

    @staticmethod
    def _actor_species(delta: TurnDelta, side: str) -> Optional[str]:
        """Pick the actor species for a side this turn.

        Prefers the protocol-truth damaging_event.user_species (handles mirror
        matches and post-switch attribution correctly). Falls back to the
        prev_active species, which is what the side acted from for switches
        and non-damaging moves. Returns None when prev_active is "NONE" or
        "NULL" so the encoded ID falls back to the unknown sentinel.
        """
        if side == "our":
            event = delta.our_damaging_event
            fallback = delta.our_prev_active
        else:
            event = delta.opp_damaging_event
            fallback = delta.opp_prev_active
        if event is not None:
            return event.user_species
        if fallback in (None, "NONE", "NULL"):
            return None
        return fallback

    def _faint_cause_multihot(self, causes: np.ndarray) -> np.ndarray:
        """Pass through the faint-cause multi-hot (already FAINT_CAUSE_DIM float32).

        Returns zeros if the array is missing/wrong-shape (shouldn't happen in
        normal operation but guards against empty() deltas in tests)."""
        if causes is None or causes.shape != (FAINT_CAUSE_DIM,):
            return np.zeros(FAINT_CAUSE_DIM, dtype=np.float32)
        return causes.astype(np.float32)

    @staticmethod
    def _target_species(delta: TurnDelta, side: str) -> Optional[str]:
        """Pick the target species ON THIS side (the mon that got hit on this
        side this turn).

        The slot's `our_*` fields follow the "this-side-this-turn" convention:
        `our_actor_species_id` is the mon ON our side that acted,
        `our_target_hp_delta` is the HP loss on the mon ON our side that got
        hit, and `our_target_species_id` is the species of that mon — sourced
        from the OPPONENT's damaging event (their move's target IS our mon).
        Returns None when the other side didn't use a damaging move this turn.
        """
        event = delta.opp_damaging_event if side == "our" else delta.our_damaging_event
        if event is None:
            return None
        return event.target_species

    def describe_vector(self, vec: np.ndarray) -> dict:
        """Decode a TURN_DELTA_DIM-dim encoded TurnDelta vector back to
        human-readable form. Base-block fields are read via the named OFFSET_*
        constants so this stays correct when CANT_DIM (or any base dim) changes."""
        from agents.observation.gen3_effects import CANT_REASONS as _CANT_REASONS_GE
        _CANT = list(_CANT_REASONS_GE)
        _EFF_LABELS = ["immune", "resisted", "normal", "super-effective"]

        def _cant_label(onehot):
            idx = int(np.argmax(onehot))
            return _CANT[idx] if onehot[idx] > 0.5 else None

        def _onehot_label(onehot, labels):
            idx = int(np.argmax(onehot))
            return labels[idx] if onehot[idx] > 0.5 else None

        def _move_dict(id_raw, pwr_raw, sec_raw, recoil_raw, type_raw):
            num = int(id_raw)
            type_id = int(type_raw)
            return {
                "move_id": num,
                "move_name": self._num_to_name.get(num),
                "power": round(float(pwr_raw) * 200),
                "secondary": bool(sec_raw > 0.5),
                "recoil": bool(recoil_raw > 0.5),
                "type_id": type_id,
                "move_type": self._idx_to_type.get(type_id),
            }

        def _species_name(raw):
            num = int(raw)
            return self._num_to_species.get(num) if num > 0 else None

        our_move = _move_dict(vec[OFFSET_OUR_MOVE_BLOCK + 0], vec[OFFSET_OUR_MOVE_BLOCK + 1],
                              vec[OFFSET_OUR_MOVE_BLOCK + 2], vec[OFFSET_OUR_MOVE_BLOCK + 3],
                              vec[OFFSET_OUR_MOVE_BLOCK + 4])
        opp_move = _move_dict(vec[OFFSET_OPP_MOVE_BLOCK + 0], vec[OFFSET_OPP_MOVE_BLOCK + 1],
                              vec[OFFSET_OPP_MOVE_BLOCK + 2], vec[OFFSET_OPP_MOVE_BLOCK + 3],
                              vec[OFFSET_OPP_MOVE_BLOCK + 4])
        order_label = None
        if vec[OFFSET_ORDER] > 0.5:
            order_label = "we_first"
        elif vec[OFFSET_ORDER + 1] > 0.5:
            order_label = "opp_first"
        return {
            "our_move": our_move if (our_move["move_id"] > 0 or our_move["power"] > 0) else None,
            "opp_move": opp_move if (opp_move["move_id"] > 0 or opp_move["power"] > 0) else None,
            "our_switched": bool(vec[OFFSET_OUR_SWITCHED] > 0.5),
            "opp_switched": bool(vec[OFFSET_OPP_SWITCHED] > 0.5),
            "our_failed": bool(vec[OFFSET_OUR_FAILED_TO_MOVE] > 0.5),
            "opp_failed": bool(vec[OFFSET_OPP_FAILED_TO_MOVE] > 0.5),
            "our_cant": _cant_label(vec[OFFSET_OUR_CANT:OFFSET_OUR_CANT + CANT_DIM]),
            "opp_cant": _cant_label(vec[OFFSET_OPP_CANT:OFFSET_OPP_CANT + CANT_DIM]),
            "our_hp_delta": float(vec[OFFSET_OUR_HP_DELTA_SUM]),
            "opp_hp_delta": float(vec[OFFSET_OPP_HP_DELTA_SUM]),
            "we_fainted": bool(vec[OFFSET_WE_FAINTED] > 0.5),
            "opp_fainted": bool(vec[OFFSET_OPP_FAINTED] > 0.5),
            "opp_move_known": bool(vec[OFFSET_OPP_MOVE_KNOWN] > 0.5),
            "our_effectiveness": _onehot_label(vec[OFFSET_OUR_EFF:OFFSET_OUR_EFF + EFF_DIM], _EFF_LABELS),
            "opp_effectiveness": _onehot_label(vec[OFFSET_OPP_EFF:OFFSET_OPP_EFF + EFF_DIM], _EFF_LABELS),
            "move_order": order_label,
            "our_boost_delta": vec[OFFSET_OUR_BOOST_DELTA:OFFSET_OUR_BOOST_DELTA + BOOST_DIM].tolist(),
            "opp_boost_delta": vec[OFFSET_OPP_BOOST_DELTA:OFFSET_OPP_BOOST_DELTA + BOOST_DIM].tolist(),
            "phase_is_forced_switch": bool(vec[OFFSET_PHASE_FORCED_SWITCH] > 0.5),
            "our_target_hp_delta": float(vec[OFFSET_OUR_TARGET_HP_DELTA]),
            "opp_target_hp_delta": float(vec[OFFSET_OPP_TARGET_HP_DELTA]),
            "our_hp_levels": vec[OFFSET_OUR_HP_LEVELS:OFFSET_OUR_HP_LEVELS + HP_LEVEL_DIM].tolist(),
            "opp_hp_levels": vec[OFFSET_OPP_HP_LEVELS:OFFSET_OPP_HP_LEVELS + HP_LEVEL_DIM].tolist(),
            "our_target_status": _onehot_label(
                vec[OFFSET_OUR_TARGET_STATUS:OFFSET_OUR_TARGET_STATUS + STATUS_DIM], _STATUS_LABELS,
            ),
            "opp_target_status": _onehot_label(
                vec[OFFSET_OPP_TARGET_STATUS:OFFSET_OPP_TARGET_STATUS + STATUS_DIM], _STATUS_LABELS,
            ),
            "our_move_outcome": _onehot_label(
                vec[OFFSET_OUR_MOVE_OUTCOME:OFFSET_OUR_MOVE_OUTCOME + OUTCOME_DIM], _OUTCOME_ORDER,
            ),
            "opp_move_outcome": _onehot_label(
                vec[OFFSET_OPP_MOVE_OUTCOME:OFFSET_OPP_MOVE_OUTCOME + OUTCOME_DIM], _OUTCOME_ORDER,
            ),
            "our_move_crit": bool(vec[OFFSET_OUR_CRIT] > 0.5),
            "opp_move_crit": bool(vec[OFFSET_OPP_CRIT] > 0.5),
            "our_actor_species": _species_name(vec[OFFSET_OUR_ACTOR_SPECIES]),
            "opp_actor_species": _species_name(vec[OFFSET_OPP_ACTOR_SPECIES]),
            "our_target_species": _species_name(vec[OFFSET_OUR_TARGET_SPECIES]),
            "opp_target_species": _species_name(vec[OFFSET_OPP_TARGET_SPECIES]),
            "our_switch_to_species": _species_name(vec[OFFSET_OUR_SWITCH_TO_SPEC]),
            "opp_switch_to_species": _species_name(vec[OFFSET_OPP_SWITCH_TO_SPEC]),
            "our_faint_causes": [
                FAINT_CAUSE_VOCAB[i]
                for i in range(FAINT_CAUSE_DIM)
                if vec[OFFSET_OUR_FAINT_CAUSES + i] > 0.5
            ],
            "opp_faint_causes": [
                FAINT_CAUSE_VOCAB[i]
                for i in range(FAINT_CAUSE_DIM)
                if vec[OFFSET_OPP_FAINT_CAUSES + i] > 0.5
            ],
            "our_status_applied": _onehot_label(
                vec[OFFSET_OUR_STATUS_APPLIED:OFFSET_OUR_STATUS_APPLIED + STATUS_DIM], _STATUS_LABELS,
            ),
            "our_status_cured": _onehot_label(
                vec[OFFSET_OUR_STATUS_CURED:OFFSET_OUR_STATUS_CURED + STATUS_DIM], _STATUS_LABELS,
            ),
            "opp_status_applied": _onehot_label(
                vec[OFFSET_OPP_STATUS_APPLIED:OFFSET_OPP_STATUS_APPLIED + STATUS_DIM], _STATUS_LABELS,
            ),
            "opp_status_cured": _onehot_label(
                vec[OFFSET_OPP_STATUS_CURED:OFFSET_OPP_STATUS_CURED + STATUS_DIM], _STATUS_LABELS,
            ),
            "our_item_used": bool(vec[OFFSET_OUR_ITEM_USED] > 0.5),
            "opp_item_used": bool(vec[OFFSET_OPP_ITEM_USED] > 0.5),
            "our_attempted_move": self._num_to_name.get(int(vec[OFFSET_OUR_ATTEMPTED_MOVE])),
            "attempted_switch_rejected": bool(vec[OFFSET_ATTEMPTED_SWITCH_REJECTED] > 0.5),
            "our_attempted_switch_to_species": _species_name(vec[OFFSET_OUR_ATTEMPTED_SWITCH_SPEC]),
        }

    def encode(self, delta: TurnDelta) -> np.ndarray:
        our_hp_delta_sum = float(delta.our_hp_delta.sum())
        opp_hp_delta_sum = float(delta.opp_hp_delta.sum())
        scalars = np.array([
            float(delta.our_switch_to is not None),
            float(delta.opp_switch_to is not None),
            float(delta.our_failed_to_move),
            float(delta.opp_failed_to_move),
        ], dtype=np.float32)
        # opp move feature uses the protocol-truth event when available
        # (falls back to delta.opp_move_id for non-damaging moves). Raw
        # delta.opp_move_id can be a stale last_move from a different opp
        # mon when opp switched between snapshots — feeding that into the
        # model's observation pollutes training with mis-attributed moves.
        base_block = np.concatenate([
            self._move_features(delta.our_move_id),                    # 5
            self._move_features(delta.opp_resolved_move_id),           # 5
            scalars,                                                    # 4
            self._cant_onehot(delta.our_cant_reason),                  # 5
            self._cant_onehot(delta.opp_cant_reason),                  # 5
            np.array([our_hp_delta_sum, opp_hp_delta_sum,              # 5
                      float(delta.we_fainted),
                      float(delta.opp_fainted),
                      float(delta.opp_move_known)],
                     dtype=np.float32),
            self._effectiveness_onehot(delta.our_effectiveness),       # 4
            self._effectiveness_onehot(delta.opp_effectiveness),       # 4
            self._order_onehot(delta.we_moved_first),                  # 2
        ])
        assert base_block.shape[0] == TURN_DELTA_BASE_DIM

        # Target status at move-fire time, paired with `our_target_species_id`
        # and `our_target_hp_delta` (all "this-side-this-turn" convention):
        # our_target_status = status of OUR mon when OPP hit it = read from
        # opp_damaging_event.target_status. None when the other side didn't
        # use a damaging move this turn.
        our_target_status = (
            delta.opp_damaging_event.target_status
            if delta.opp_damaging_event is not None else None
        )
        opp_target_status = (
            delta.our_damaging_event.target_status
            if delta.our_damaging_event is not None else None
        )

        # Extended block — order matches the OFFSET_* constants + the embedded-ID
        # manifest above. Move/species IDs are stored as raw ints (float32); the
        # extractor routes them through their embedding tables per the manifest.
        ext_block = np.concatenate([
            delta.our_boost_delta.astype(np.float32),                  # 7
            delta.opp_boost_delta.astype(np.float32),                  # 7
            np.array([float(delta.phase_is_forced_switch)], dtype=np.float32),  # 1
            np.array([
                float(delta.our_target_hp_delta) if delta.our_target_hp_delta is not None else 0.0,
                float(delta.opp_target_hp_delta) if delta.opp_target_hp_delta is not None else 0.0,
            ], dtype=np.float32),                                       # 2
            delta.our_hp_after.astype(np.float32),                     # 6
            delta.opp_hp_after.astype(np.float32),                     # 6
            self._status_onehot(our_target_status),                    # 7
            self._status_onehot(opp_target_status),                    # 7
            self._outcome_onehot(delta.our_move_outcome),              # 3
            self._outcome_onehot(delta.opp_move_outcome),              # 3
            np.array([
                float(delta.our_move_crit),
                float(delta.opp_move_crit),
            ], dtype=np.float32),                                       # 2
            # gen3_turn_delta_v2: faint-cause multi-hots + attempted move id.
            # (faint_count scalars dropped — redundant with we_fainted + cause popcount.)
            self._faint_cause_multihot(
                getattr(delta, "our_faint_causes", None)               # 8
            ),
            self._faint_cause_multihot(
                getattr(delta, "opp_faint_causes", None)               # 8
            ),
            # Status transitions this window (the EVENT — identity of the cause,
            # item vs ability, lives in the per-mon block, not here).
            self._status_onehot(getattr(delta, "our_status_applied", None)),  # 7
            self._status_onehot(getattr(delta, "our_status_cured", None)),    # 7
            self._status_onehot(getattr(delta, "opp_status_applied", None)),  # 7
            self._status_onehot(getattr(delta, "opp_status_cured", None)),    # 7
            # Item-used bits — resource event + timing; the WHICH is in the item block.
            np.array([
                float(getattr(delta, "our_item_lost", None) is not None),     # 1
                float(getattr(delta, "opp_item_lost", None) is not None),     # 1
            ], dtype=np.float32),
            np.array([
                self._move_id(getattr(delta, "our_attempted_move_id", None)),  # 1 embedded
            ], dtype=np.float32),
            # Species block (6 IDs — embedded by extractor via the manifest).
            np.array([
                self._species_id(self._actor_species(delta, "our")),
                self._species_id(self._actor_species(delta, "opp")),
                self._species_id(self._target_species(delta, "our")),
                self._species_id(self._target_species(delta, "opp")),
                self._species_id(delta.our_switch_to),
                self._species_id(delta.opp_switch_to),
            ], dtype=np.float32),                                       # 6
            # gen3_trapping_signals_v1: rejected-switch history (bit + attempted-switch
            # species id, embedded by the extractor per the manifest). On a rejected pivot
            # our_switch_to is None (the switch never happened) but attempted_switch_to names
            # the mon we tried to bring in.
            np.array([
                float(getattr(delta, "attempted_switch_rejected", False)),  # 1
            ], dtype=np.float32),
            np.array([
                self._species_id(getattr(delta, "attempted_switch_to", None)),  # 1 embedded
            ], dtype=np.float32),
        ])
        assert ext_block.shape[0] == TURN_DELTA_EXT_DIM, (
            f"ext_block shape {ext_block.shape[0]} != TURN_DELTA_EXT_DIM {TURN_DELTA_EXT_DIM}"
        )

        return np.concatenate([base_block, ext_block])
