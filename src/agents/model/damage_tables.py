"""Fixed lookup tensors for the differentiable GPU damage operator (`DamageOperator`).

These are pure *physics* — never learned — built ONCE from the `gen3_data` facade and the
`TypeEncoder` type axis, then registered as float32 **buffers** on the operator so they move
with `.to(device)` and never desync. They are indexed by national-dex ``num`` (the SAME axis the
move-belief logits + the embedding tables live on — ``species_id``/``move_id`` are fed to the
embeddings as their ``num``), so a buffer row ``m`` aligns with belief logit ``m``.

The TYPE axis is `TypeEncoder.TYPE_TO_IDX` (1-based: NORMAL=1..DARK=17, "???"=18, **idx 0 =
unknown sentinel**), the SAME axis the obs `type1_ids`/`type2_ids` ride — so the chart columns
gathered for a believed move line up with the defender types read straight from the obs. There
is no FAIRY in gen3 (the chart's dead FAIRY row/col is skipped).

Hidden Power is a special case: 17 move ids (bare + 16 typed) all share ``num=237``, so the
num-indexed move buffers cannot represent its type. The bare slot is left at BP 0; the operator
instead expands HP into 16 **typed candidates** (one per type, BP 70 — gen3's max HP power),
using `HP_TYPE_IDX` / `HP_IS_PHYS` here + the per-mon HP-type belief (`hp_probs`) from the obs.

Reference: `designs/ai_v6/design_differentiable_damage_op.md` (§3 the gen3 formula, §7 the
edge-case matrix). The math core mirrors `agents/observation/incoming_damage.py` (the live CPU
belief) and the proven torch port in the ai_v6 design.
"""
from __future__ import annotations

from typing import Dict

import torch

from agents import gen3_data
from agents.enums import MoveCategory, PokemonType
from agents.observation.types import TypeEncoder
from agents.training.hidden_power_tracker import HIDDEN_POWER_TYPE_ORDER

# Type-index axis (shared with the obs). Index 0 = unknown sentinel, 18 = "???"; size 19 so
# every TypeEncoder index (0..18) is addressable. Rows/cols 0 and 18 stay NEUTRAL (×1.0): an
# unknown / typeless attacking or defending type contributes nothing to the effectiveness
# product, matching the obs convention (a mono-type mon writes idx 0 into its 2nd type slot).
_T2I = TypeEncoder.TYPE_TO_IDX
N_TYPE_IDX = 19

# Gen 1-3 physical/special split is by the move's TYPE (the per-move category arrived in gen 4).
# These nine types are physical (Atk vs Def); the rest are special (SpA vs SpD). Mirrors
# `incoming_damage._PHYSICAL_TYPES` / `gen3_data.moves._GEN3_SPECIAL_TYPES` (the complement).
_PHYSICAL_TYPE_NAMES = (
    "NORMAL", "FIGHTING", "POISON", "GROUND", "FLYING", "BUG", "ROCK", "GHOST", "STEEL",
)

# Hidden Power: all variants share this num; gen3's max HP base power is 70 ("assume max damage").
HIDDEN_POWER_NUM = 237
HIDDEN_POWER_BP = 70.0
N_HP_TYPES = len(HIDDEN_POWER_TYPE_ORDER)   # 16


def _move_type_idx(md) -> int:
    """TypeEncoder index of a MoveData's type. Curse/typeless carry PokemonType.THREE_QUESTION_MARKS
    whose .name is 'THREE_QUESTION_MARKS', NOT '???' — map it explicitly (these are BP 0 anyway)."""
    if md.type is PokemonType.THREE_QUESTION_MARKS:
        return _T2I["???"]
    return _T2I.get(md.type.name, 0)


def build_damage_buffers(n_moves: int, n_species: int) -> Dict[str, torch.Tensor]:
    """Build the fixed lookup tensors, indexed 0..n-1 by national-dex ``num``.

    Returns a dict of float32 / long tensors for `DamageOperator` to register as buffers:
      MOVE_BP[n_moves]          base power (HP num 237 left 0 — expanded as typed candidates)
      MOVE_TYPE_IDX[n_moves]    TypeEncoder type index of each move
      MOVE_PHYS[n_moves]        1.0 physical / 0.0 special|status (from MoveData.category)
      BASE_STATS[n_species, 6]  [hp, atk, def, spa, spd, spe]
      CHART[19, 19]             effectiveness [def_idx, att_idx] on the TypeEncoder axis
      TYPE_IS_PHYS[19]          1.0 for the 9 gen3-physical type indices
      HP_TYPE_IDX[16]           TypeEncoder index of each Hidden Power type slot
      HP_IS_PHYS[16]            1.0 if that HP type is physical
    """
    move_bp = torch.zeros(n_moves, dtype=torch.float32)
    move_type_idx = torch.zeros(n_moves, dtype=torch.long)
    move_phys = torch.zeros(n_moves, dtype=torch.float32)
    for mid in gen3_data.moves.raw():
        md = gen3_data.moves.get(mid)
        num = md.num
        if num == HIDDEN_POWER_NUM:          # HP collision: handled via the typed-candidate path
            continue
        if not (0 <= num < n_moves):
            continue
        move_bp[num] = float(md.base_power)
        move_type_idx[num] = _move_type_idx(md)
        move_phys[num] = 1.0 if md.category == MoveCategory.PHYSICAL else 0.0

    # Base stats by species num. SpeciesData.base_stats is keyed atk/def/hp/spa/spd/spe — index by
    # KEY, not positional order, into our [hp, atk, def, spa, spd, spe] layout.
    stat_order = ("hp", "atk", "def", "spa", "spd", "spe")
    base_stats = torch.zeros(n_species, 6, dtype=torch.float32)
    for sid in gen3_data.species.raw():
        sd = gen3_data.species.get(sid)
        num = sd.num
        if not (0 <= num < n_species):
            continue
        for j, k in enumerate(stat_order):
            base_stats[num, j] = float(sd.base_stats.get(k, 0))

    # Effectiveness chart on the TypeEncoder axis. Neutral (1.0) everywhere by default; rows/cols
    # 0 (unknown) and 18 (???) stay neutral. Skip the dead FAIRY row/col (absent from TYPE_TO_IDX).
    chart = torch.ones(N_TYPE_IDX, N_TYPE_IDX, dtype=torch.float32)
    for def_name, row in gen3_data.type_chart.chart().items():
        if def_name not in _T2I:
            continue
        di = _T2I[def_name]
        for att_name, mult in row.items():
            if att_name not in _T2I:
                continue
            chart[di, _T2I[att_name]] = float(mult)

    type_is_phys = torch.zeros(N_TYPE_IDX, dtype=torch.float32)
    for tname in _PHYSICAL_TYPE_NAMES:
        type_is_phys[_T2I[tname]] = 1.0

    hp_type_idx = torch.tensor(
        [_T2I[t.name] for t in HIDDEN_POWER_TYPE_ORDER], dtype=torch.long
    )
    hp_is_phys = type_is_phys[hp_type_idx].clone()

    return {
        "MOVE_BP": move_bp,
        "MOVE_TYPE_IDX": move_type_idx,
        "MOVE_PHYS": move_phys,
        "BASE_STATS": base_stats,
        "CHART": chart,
        "TYPE_IS_PHYS": type_is_phys,
        "HP_TYPE_IDX": hp_type_idx,
        "HP_IS_PHYS": hp_is_phys,
    }
