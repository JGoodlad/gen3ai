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

# Defender ABILITY damage multipliers vs an attacking move TYPE (the gen3 immunity/resist abilities a
# DEFENDER carries — our mons' abilities are revealed, so this is known, not believed). A multiplier
# folded into the type-effectiveness product, same class as the chart. Wonder Guard (a piecewise
# only-super-effective gate, Shedinja-only) is deferred. Sourced by ability id so a num remap is safe.
_ABILITY_TYPE_MULTS = {
    "levitate": [("GROUND", 0.0)],
    "flashfire": [("FIRE", 0.0)],
    "waterabsorb": [("WATER", 0.0)],
    "voltabsorb": [("ELECTRIC", 0.0)],
    "thickfat": [("FIRE", 0.5), ("ICE", 0.5)],
}

# Hidden Power: all variants share this num; gen3's max HP base power is 70 ("assume max damage").
HIDDEN_POWER_NUM = 237
HIDDEN_POWER_BP = 70.0
N_HP_TYPES = len(HIDDEN_POWER_TYPE_ORDER)   # 16

# Columns of MOVE_EFFECT_FLAGS (the believed-move status/utility threat axes). Order is the contract
# the DamageOperator's effect-scalar block relies on — do not reorder without updating the op.
MOVE_EFFECT_COLS = ("recovery", "status", "phaze", "boost", "hazard", "protect")


def _move_type_idx(md) -> int:
    """TypeEncoder index of a MoveData's type. Curse/typeless carry PokemonType.THREE_QUESTION_MARKS
    whose .name is 'THREE_QUESTION_MARKS', NOT '???' — map it explicitly (these are BP 0 anyway)."""
    if md.type is PokemonType.THREE_QUESTION_MARKS:
        return _T2I["???"]
    return _T2I.get(md.type.name, 0)


def build_damage_buffers(n_moves: int, n_species: int, n_abilities: int) -> Dict[str, torch.Tensor]:
    """Build the fixed lookup tensors, indexed 0..n-1 by national-dex ``num``.

    Returns a dict of float32 / long tensors for `DamageOperator` to register as buffers:
      MOVE_BP[n_moves]          base power (HP num 237 left 0 — expanded as typed candidates)
      MOVE_TYPE_IDX[n_moves]    TypeEncoder type index of each move
      MOVE_PHYS[n_moves]        1.0 physical / 0.0 special|status (from MoveData.category)
      MOVE_ACCURACY[n_moves]    base hit probability (1.0 never-miss, else accuracy/100) — folds into P(KO)
      BASE_STATS[n_species, 6]  [hp, atk, def, spa, spd, spe]
      CHART[19, 19]             effectiveness [def_idx, att_idx] on the TypeEncoder axis
      TYPE_IS_PHYS[19]          1.0 for the 9 gen3-physical type indices
      HP_TYPE_IDX[16]           TypeEncoder index of each Hidden Power type slot
      HP_IS_PHYS[16]            1.0 if that HP type is physical
      MOVE_EFFECT_FLAGS[n_moves, 6]   per-move status/utility flags (MOVE_EFFECT_COLS)
      ABILITY_DAMAGE_MULT[n_abilities, 19]  defender-ability ×type multiplier (Levitate/Flash Fire/…)
    """
    move_bp = torch.zeros(n_moves, dtype=torch.float32)
    move_type_idx = torch.zeros(n_moves, dtype=torch.long)
    move_phys = torch.zeros(n_moves, dtype=torch.float32)
    # Per-move hit probability (base accuracy): 1.0 for never-miss moves (Swift/Aerial Ace/all
    # status), else accuracy/100. Folded into the op's P(KO) so an 85%-accurate Fire Blast reads a
    # lower KO-this-turn risk than a 100% move (the 3 damage rolls stay "damage IF it lands"). Dynamic
    # accuracy mods (evasion/Sand-Attack/Compound Eyes) are not modelled — base accuracy only.
    move_accuracy = torch.ones(n_moves, dtype=torch.float32)
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
        move_accuracy[num] = 1.0 if md.never_miss else float(md.accuracy) / 100.0

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

    # Per-move EFFECT flags (num axis), for the believed-move STATUS/UTILITY threat (noisy-OR over the
    # move belief): the unified op surfaces "P(opp active has a recovery / status / phaze / setup /
    # hazard / protect move)" — the strictly-more-capable axis the damage-only CPU block never had.
    # Sourced from MoveData (Showdown-derived flags; status_inflicted = the major status its PURPOSE is
    # to inflict). HP num-237 stays all-zero (it is a damaging move, no effect). Column order ==
    # MOVE_EFFECT_COLS.
    move_effect_flags = torch.zeros(n_moves, len(MOVE_EFFECT_COLS), dtype=torch.float32)
    for mid in gen3_data.moves.raw():
        md = gen3_data.moves.get(mid)
        num = md.num
        if num == HIDDEN_POWER_NUM or not (0 <= num < n_moves):
            continue
        flags = (md.is_heal, md.status_inflicted is not None, md.is_phaze,
                 md.is_boost, md.is_hazard, md.is_protect)
        for j, f in enumerate(flags):
            if f:
                move_effect_flags[num, j] = 1.0

    type_is_phys = torch.zeros(N_TYPE_IDX, dtype=torch.float32)
    for tname in _PHYSICAL_TYPE_NAMES:
        type_is_phys[_T2I[tname]] = 1.0

    # Defender-ability × move-type multiplier (1.0 = no effect). Gathered by the defender's revealed
    # ability num and multiplied into the type-effectiveness product — so Levitate reads 0× Ground,
    # Flash Fire 0× Fire, Thick Fat 0.5× Fire/Ice, etc., instead of a phantom super-effective KO.
    ability_damage_mult = torch.ones(n_abilities, N_TYPE_IDX, dtype=torch.float32)
    for aid, type_mults in _ABILITY_TYPE_MULTS.items():
        ad = gen3_data.abilities.get(aid)
        if ad is None or not (0 <= ad.num < n_abilities):
            continue
        for tname, m in type_mults:
            ability_damage_mult[ad.num, _T2I[tname]] = m

    hp_type_idx = torch.tensor(
        [_T2I[t.name] for t in HIDDEN_POWER_TYPE_ORDER], dtype=torch.long
    )
    hp_is_phys = type_is_phys[hp_type_idx].clone()

    return {
        "MOVE_BP": move_bp,
        "MOVE_TYPE_IDX": move_type_idx,
        "MOVE_PHYS": move_phys,
        "MOVE_ACCURACY": move_accuracy,
        "BASE_STATS": base_stats,
        "CHART": chart,
        "TYPE_IS_PHYS": type_is_phys,
        "HP_TYPE_IDX": hp_type_idx,
        "HP_IS_PHYS": hp_is_phys,
        "MOVE_EFFECT_FLAGS": move_effect_flags,
        "ABILITY_DAMAGE_MULT": ability_damage_mult,
    }


# Floor probability for a move a species is ~never seen to run (keeps an unseen move POSSIBLE — never
# logit(-inf) — so in-battle evidence can still lift it). Also the value for an unknown species (num 0).
_PRIOR_FLOOR = 0.02


def build_move_prior_logits(n_species: int, n_moves: int, floor: float = _PRIOR_FLOOR,
                            learnset_gate: bool = False) -> torch.Tensor:
    """``[n_species, n_moves]`` LOG-ODDS of the Smogon move-frequency prior, indexed by national-dex
    ``num`` on BOTH axes — the base rate ``P(move in set)`` for a species, ready to fuse additively into
    the move-belief logits (``posterior_logit = head_delta + prior_logit``).

    Sources `gen3_data.priors.moves(species)` -> ``{move_id: P(in set)}`` (un-normalized; a set runs
    ~4 moves). Probabilities for move_ids that collapse to one ``num`` are SUMMED (Hidden Power: all
    typed variants share num 237, and a mon runs at most one HP type, so ``P(has HP) = Σ typed usage``).

    Two modes (`learnset_gate` selects; the gate is a VERSION-CHECKED forward-behavior change, never a
    silent default flip — OFF reproduces the original buffer byte-for-byte):

    - **OFF (default, legacy):** clamp every ``(species, move)`` to ``[floor, 1-eps]`` then ``logit``.
      A species with no prior entry (or num 0 = unknown) gets ``logit(floor)`` everywhere — a low,
      non-zero base the learned delta can still move (keeps an unseen move POSSIBLE).
    - **ON (LEGALITY-ONLY gate):** the only thing pruned is the IMPOSSIBLE. A move a species **cannot
      learn** is driven to ``logit(eps)`` (≈ 0 — it removes the phantom "a special attacker might have
      Explosion" the flat floor invented). A move it CAN learn keeps its **true Smogon usage** — a rare
      tech stays rare-but-present (naturally negligible in the op's hard-max, yet liftable by the learned
      head, and pinned certain the moment it's revealed), NOT floored up to ``floor`` and NOT pruned. A
      legal move ABSENT from the usage data gets the small ``floor`` base (so in-battle evidence can still
      surface it). **No rarity cap** — a surprise move a mon legitimately runs is never zeroed out of the
      belief (the earlier ``<2%`` prune did that and crippled surprise-move anticipation). Because every
      move with recorded usage is necessarily legal, the legality mask only ever bites the ABSENT cells;
      Hidden Power's typed usages sum into ``num`` 237 (legal iff the bare ``'hiddenpower'`` is in the
      learnset). A hidden / unknown species (no learnset) keeps the legacy flat floor (no movepool known →
      nothing to prune; marginalising the learnset over a species belief is a later extension).

    Returned as a plain float32 tensor for `MoveBelief` to register as a NON-persistent buffer (pure
    data-derived physics, recomputable — never a saved weight)."""
    eps = 1e-6
    if not learnset_gate:
        # LEGACY (default): flat `floor` everywhere + accumulate usage. Byte-identical to the original.
        prob = torch.zeros(n_species, n_moves, dtype=torch.float64)
        for sid in gen3_data.species.raw():
            sd = gen3_data.species.get(sid)
            snum = sd.num
            if not (0 <= snum < n_species):
                continue
            for move_id, p in gen3_data.priors.moves(sid).items():
                md = gen3_data.moves.get(move_id)
                if md is not None and 0 <= md.num < n_moves:
                    prob[snum, md.num] += float(p)       # sum collisions (e.g. typed Hidden Power → 237)
        prob = prob.clamp(floor, 1.0 - eps)
        return torch.logit(prob).to(torch.float32)

    # LEGALITY-ONLY gate: illegal → eps (impossible); legal-observed → TRUE usage; legal-unobserved → floor.
    prob = torch.full((n_species, n_moves), eps, dtype=torch.float64)   # default = impossible
    for sid in gen3_data.species.raw():
        sd = gen3_data.species.get(sid)
        snum = sd.num
        if not (0 <= snum < n_species):
            continue
        legal = gen3_data.learnset.get_legal_moves(sid)
        if legal is None:
            prob[snum, :] = floor                        # unknown movepool → legacy flat floor (can't prune)
        else:
            for move_id in legal:                        # every LEGAL move → a small liftable base
                md = gen3_data.moves.get(move_id)
                if md is not None and 0 <= md.num < n_moves:
                    prob[snum, md.num] = floor
        # TRUE usage overrides the floor (an observed move is necessarily legal); HP variants sum into 237.
        usage: Dict[int, float] = {}
        for move_id, p in gen3_data.priors.moves(sid).items():
            md = gen3_data.moves.get(move_id)
            if md is None or not (0 <= md.num < n_moves):
                continue
            usage[md.num] = usage.get(md.num, 0.0) + float(p)
        for num, u in usage.items():
            if u > float(prob[snum, num]):
                prob[snum, num] = u                      # rare moves keep their real (small) rate
    prob = prob.clamp(eps, 1.0 - eps)
    return torch.logit(prob).to(torch.float32)           # log(p/(1-p)), the additive log-odds base rate
