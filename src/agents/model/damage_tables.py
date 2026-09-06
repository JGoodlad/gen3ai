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

Hidden Power is handled bidirectionally (gen3_typed_hidden_power_ids_v1):
  - OPPONENT side (type unrevealed): the protocol gives the bare ``hiddenpower`` → ``num=237``.
    That slot is left at BP 0 / type-idx 0; the operator instead expands HP into 16 **typed
    candidates** (one per type, BP 70 — gen3's max HP power), using `HP_TYPE_IDX` / `HP_IS_PHYS`
    here + the per-mon HP-type belief (`hp_probs`) from the obs. The move-belief PRIOR likewise
    folds all typed HP usage back onto 237 (see `_belief_num`).
  - OUR side (type always known): each typed variant has its OWN distinct num (355-370), so the
    num-indexed per-move buffers (BP/type/phys/accuracy/attr/secondary/latent) below populate those
    rows with the real typed values — the loops here iterate ``gen3_data.moves`` and skip ONLY the
    bare 237, so typed HP flow through naturally and our OUTGOING HP is priced with the right type.

🚨 **The BELIEF PRIORS live next door, in `belief_tables.py`, and the dex-IDENTITY facts under
that in `dex_ids.py`** (`gen3_belief_tables_split_v1` then `gen3_dex_ids_split_v1`, both
2026-09-06). `belief_tables` holds the opponent spread prior, its nature/EV decomposition, the
Hidden-Power TYPE prior, the ITEM prior, the per-species MOVE prior and the team-composition
SPECIES prior — i.e. the per-species Smogon distributions a belief HEAD predicts a delta on top of,
as opposed to the damage/type/stat buffers here. `dex_ids` holds what BOTH need: `HIDDEN_POWER_NUM`,
`_belief_num`, `_hp_typed_nums` and `build_species_usage_prior`. Every name in both is RE-EXPORTED
below, so `from agents.model.damage_tables import build_item_prior` (or `build_move_prior_logits`,
or `HIDDEN_POWER_NUM`) still resolves; edit them there. The layering runs ONE way —

    `damage_tables` → `belief_tables` → `dex_ids`,  and  `damage_tables` → `dex_ids`

— and neither of those two may import back, or they close a cycle Python resolves only for
whichever module is imported first (`belief_tables_test.py` AST-scans both edges). Four of the
moved names are genuine CONSUMERS here, which is what fixes the direction: `build_damage_buffers`
registers `build_opp_spread_prior` / `build_nature_mult` / `build_species_usage_prior` as the op's
`SPECIES_SPREAD_PRIOR` / `NATURE_MULT` / `SPECIES_USAGE_PRIOR`, and the typed-HP candidate
expansion reads `HIDDEN_POWER_NUM` + `_hp_typed_nums`.

Reference: `designs/ai_v6/design_differentiable_damage_op.md` (§3 the gen3 formula, §7 the
edge-case matrix). The math core mirrors `agents/observation/incoming_damage.py` (the live CPU
belief) and the proven torch port in the ai_v6 design.
"""
from __future__ import annotations

from typing import cast, Dict

import torch

from agents import gen3_data
from agents.gen3_data.items import ItemData
from agents.gen3_data.moves import MoveData
from agents.gen3_data.species import SpeciesData
from agents.enums import MoveCategory, PokemonType
from agents.gen3_mechanics import (ABILITY_STATUS_IMMUNITY, CURSE_NON_GHOST_BOOSTS,
                                   STATUS_MOVE_IMMUNITY)
from agents.observation.types import TypeEncoder
from agents.training.hidden_power_tracker import HIDDEN_POWER_TYPE_ORDER

# gen3_belief_tables_split_v1 + gen3_dex_ids_split_v1 (2026-09-06): the BELIEF-PRIOR constructors
# moved to `belief_tables.py` (the priors a belief HEAD fuses with — spread / nature+EV /
# Hidden-Power type / item / move / species co-occurrence), and the dex-IDENTITY facts the two halves
# SHARE moved to `dex_ids.py`. What stays here is the damage/type/stat buffers the op's physics reads.
# Four of the moved names are USED below — `build_damage_buffers` registers SPECIES_SPREAD_PRIOR,
# NATURE_MULT and SPECIES_USAGE_PRIOR for the op, and the typed-HP expansion reads HIDDEN_POWER_NUM
# and `_hp_typed_nums`; the rest are RE-EXPORTS so every historical
# `from agents.model.damage_tables import …` spelling still resolves. The layering is one-way by
# design — `damage_tables` → `belief_tables` → `dex_ids`, never back — so the three modules layer
# rather than cycle.
from agents.model.belief_tables import build_nature_mult, build_opp_spread_prior
from agents.model.dex_ids import HIDDEN_POWER_NUM, _hp_typed_nums, build_species_usage_prior
from agents.model.dex_ids import _belief_num                            # noqa: F401  (re-export)
from agents.model.dex_ids import _USAGE_PRIOR_FLOOR                     # noqa: F401  (re-export)
from agents.model.belief_tables import (                                # noqa: F401  (re-export)
    N_NATURES,                                                          # noqa: F401  (re-export)
    N_SPREAD_STATS,                                                     # noqa: F401  (re-export)
    SPECIES_CLAUSE_LOGIT,                                               # noqa: F401  (re-export)
    SPREAD_STAT_COLS,                                                   # noqa: F401  (re-export)
    _COOCCUR_LIFT_CLAMP,                                                # noqa: F401  (re-export)
    _ILLEGAL_PROB,                                                      # noqa: F401  (re-export)
    _MIN_PRIOR_FLOOR,                                                   # noqa: F401  (re-export)
    _NATURE_PRIOR_FLOOR,                                                # noqa: F401  (re-export)
    _PRIOR_FLOOR,                                                       # noqa: F401  (re-export)
    _SPECIES_CLAUSE_PROB,                                               # noqa: F401  (re-export)
    _SPECIES_PRIOR_FLOOR,                                               # noqa: F401  (re-export)
    _SPREAD_BASE_IDX,                                                   # noqa: F401  (re-export)
    build_hp_type_prior,                                                # noqa: F401  (re-export)
    build_item_prior,                                                   # noqa: F401  (re-export)
    build_move_prior_logits,                                            # noqa: F401  (re-export)
    build_species_base_stats,                                           # noqa: F401  (re-export)
    build_species_cooccur_prior,                                        # noqa: F401  (re-export)
    build_species_ev_prior,                                             # noqa: F401  (re-export)
    build_species_nature_prior,                                         # noqa: F401  (re-export)
    invert_nature_evs,                                                  # noqa: F401  (re-export)
    sanitize_historical_move_floor,                                     # noqa: F401  (re-export)
)

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

# Hidden Power: gen3's max HP base power is 70 ("assume max damage"). The typeless NUM the
# opponent's belief keys on lives in `dex_ids` (both this module's physics and `belief_tables` read it).
HIDDEN_POWER_BP = 70.0
N_HP_TYPES = len(HIDDEN_POWER_TYPE_ORDER)   # 16

# Columns of MOVE_EFFECT_FLAGS (the believed-move status/utility threat axes). Order is the contract
# the DamageOperator's effect-scalar block relies on — do not reorder without updating the op.
MOVE_EFFECT_COLS = ("recovery", "status", "phaze", "boost", "hazard", "protect")

# --- gen3_unified_move_system_v1: structured secondary effects + the move-attribute table ----------- #
# The 10 secondary-effect columns of MOVE_SECONDARY (the per-effect TRIGGER chance, 0..1, the op prices).
# Order is the contract; mirrors tools/…/sync._SECONDARY_COLS + the facade's secondary_chance() keys.
SECONDARY_COLS = (
    "par", "brn", "frz", "slp", "psn", "tox", "confusion", "flinch", "foe_statdrop", "self_boost",
)
N_SECONDARY = len(SECONDARY_COLS)            # 10
# The MAJOR-status prefix of that axis — par,brn,frz,slp,psn,tox — the six gen3 major statuses, in
# order, before the non-status effects (confusion/flinch/foe_statdrop/self_boost). `damage_op_layout`
# re-exports this as `_SECONDARY_MAJOR_N`; it was declared twice until gen3_pair_outcome_v1.
SECONDARY_MAJOR_N = 6
_SECONDARY_MAJOR_COLS_N = SECONDARY_MAJOR_N
assert SECONDARY_COLS[:SECONDARY_MAJOR_N] == ("par", "brn", "frz", "slp", "psn", "tox"), \
    "SECONDARY_COLS' major-status prefix moved — MOVE_STATUS_IDENT and the op's _SEC_CAT_IDX both " \
    "index it positionally."
SECONDARY_FLINCH_IDX = SECONDARY_COLS.index("flinch")

# Abilities that scale a move's SECONDARY-effect chance. Two ROLES (an attacker mult vs a defender
# negation — Serene-on-attacker and Shield-Dust-on-defender are NOT interchangeable, so two buffers):
#   ABILITY_SECONDARY_MULT  — the ATTACKER's own-secondary multiplier: Serene Grace DOUBLES every
#       secondary (gen3 Jirachi ≈ 25% OU usage — a real lever, not a footnote). Default 1.0.
#   ABILITY_SECONDARY_BLOCK — the DEFENDER's negation: Shield Dust NEGATES all incoming secondaries.
#       Default 1.0. (Rare in gen3 OU — completeness, not a lever.)
# Sourced by ability id (num-remap-safe).
_ABILITY_SECONDARY_MULTS = {"serenegrace": 2.0}
_ABILITY_SECONDARY_BLOCKS = {"shielddust": 0.0}

# The structured per-move ATTRIBUTE vector for the MoveLatentEncoder — the context-free "what does this
# move do" features (TYPE rides the shared type embedding, added by the encoder, so it is NOT duplicated
# here). Single source for the latent's structured input; order is a contract pinned by move_latent_test.
MOVE_ATTR_COLS = (
    "bp_norm", "is_phys", "is_spec", "is_status", "accuracy", "never_miss",
    "priority_norm", "drain", "recoil",
    *SECONDARY_COLS,                          # the 10 secondary-effect chances
    "is_heal", "is_boost", "is_protect", "is_phaze", "is_hazard", "cures_self", "cures_team",
)
N_MOVE_ATTR = len(MOVE_ATTR_COLS)            # 9 + 10 + 7 = 26
_PRIORITY_NORM = 6.0                          # gen3 priority spans ~ -6..+5 → normalize into ~[-1, 1]


def _move_type_idx(md: MoveData) -> int:
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
      MOVE_SECONDARY[n_moves, 10]     per-effect secondary chance 0..1 (SECONDARY_COLS) — gen3_unified_move_system_v1
      MOVE_PRIORITY[n_moves]          raw move priority (-6..+5); MOVE_DRAIN/MOVE_RECOIL[n_moves]  damage fractions
      ABILITY_SECONDARY_MULT[n_abilities]   attacker ×secondary-chance (Serene Grace 2×)
      ABILITY_SECONDARY_BLOCK[n_abilities]  defender ×secondary-negation (Shield Dust 0×)
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
        md = cast(MoveData, gen3_data.moves.get(mid))
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
    # BASE FORMS ONLY (`gen3_species_formes_v1`) — every table here is `table[species.num]`,
    # and an alternate forme SHARES its base's num (Deoxys-Speed / Unown-B / Castform-Sunny),
    # so iterating `raw()` would be last-write-wins and let a forme silently redefine the
    # base's stats/types at that num. The obs species channel is num-keyed too, so a forme is
    # observationally its base — the base's facts are the correct occupant of the row.
    for sid in gen3_data.species.base_form_ids():
        sd = cast(SpeciesData, gen3_data.species.get(sid))
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
        md = cast(MoveData, gen3_data.moves.get(mid))
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

    # gen3_typed_hp_belief_v1: the DISTINCT dex nums of the 16 TYPED Hidden Powers (355-370,
    # gen3_typed_hidden_power_ids_v1) in HIDDEN_POWER_TYPE_ORDER order. The opponent's Hidden Power is
    # 16 ORDINARY typed-move candidates at these nums (real BP/type, populated above) — the model NEVER
    # reasons over a typeless HP damage candidate. The composition now happens UPSTREAM, in
    # `HPTypeBelief.compose_typed_hp`, so by the time the DamageOperator sees the move-belief posterior
    # these cells already hold `P(HP present)·P(type)`; the op just consumes them like any other move.
    # `hp_cand_mask` therefore zeros ONLY the bare typeless num 237 (BP 0) — the PRESENCE channel, which
    # is a belief bookkeeping slot and never a damage candidate. (It used to zero the typed nums too,
    # because the op did the scatter itself.) Derived from data, never hardcoded.
    hp_typed_nums = torch.tensor(list(_hp_typed_nums()), dtype=torch.long)
    hp_cand_mask = torch.ones(n_moves, dtype=torch.float32)
    hp_cand_mask[HIDDEN_POWER_NUM] = 0.0                 # bare typeless HP = the presence channel (BP 0)
    # GIGO GUARD — fail LOUD if the data drifts, rather than scatter the HP-type belief onto the wrong move
    # or a 0-damage slot: each typed num MUST carry its HP_TYPE_ORDER type + BP 70, and the bare 237 BP 0.
    for j, t in enumerate(HIDDEN_POWER_TYPE_ORDER):
        n = int(hp_typed_nums[j])
        if not (0 <= n < n_moves):
            raise ValueError(
                f"typed HP '{t.name}' num {n} out of range [0,{n_moves}) — typed-HP move data "
                "(gen3_typed_hidden_power_ids_v1) missing from gen3_moves.json?")
        if int(move_type_idx[n]) != int(hp_type_idx[j]):
            raise ValueError(
                f"typed-HP num {n} ('{t.name}') type idx {int(move_type_idx[n])} != HP_TYPE_ORDER[{j}] "
                f"{int(hp_type_idx[j])} — the 355-370 ↔ HP_TYPE_ORDER alignment drifted (GIGO).")
        if float(move_bp[n]) != float(HIDDEN_POWER_BP):
            raise ValueError(f"typed-HP num {n} ('{t.name}') BP {float(move_bp[n])} != {HIDDEN_POWER_BP}")
    if float(move_bp[HIDDEN_POWER_NUM]) != 0.0:
        raise ValueError(
            f"bare HP num {HIDDEN_POWER_NUM} BP {float(move_bp[HIDDEN_POWER_NUM])} != 0 — it must stay the "
            "typeless PRESENCE token (BP 0), never a damage candidate.")

    # gen3_unified_move_system_v1: structured secondary / priority / drain / recoil (num axis). The op
    # prices the per-effect chance (× the active's ABILITY_SECONDARY_MULT), the outgoing-priority feature,
    # and the recovery magnitude (drain). HP num-237 stays all-zero (no secondary). Same skip-rule as the
    # damage buffers above.
    move_secondary = torch.zeros(n_moves, N_SECONDARY, dtype=torch.float32)
    move_priority = torch.zeros(n_moves, dtype=torch.float32)
    move_drain = torch.zeros(n_moves, dtype=torch.float32)
    move_recoil = torch.zeros(n_moves, dtype=torch.float32)
    move_fixed_damage = build_move_fixed_damage(n_moves)        # gen3_unified_op_physics_v1
    for mid in gen3_data.moves.raw():
        md = cast(MoveData, gen3_data.moves.get(mid))
        num = md.num
        if num == HIDDEN_POWER_NUM or not (0 <= num < n_moves):
            continue
        for j, col in enumerate(SECONDARY_COLS):
            move_secondary[num, j] = md.secondary_chance(col)
        move_priority[num] = float(md.priority)
        move_drain[num] = float(md.drain_fraction)
        move_recoil[num] = float(md.recoil_fraction)

    # Attacker secondary-chance multiplier (Serene Grace 2×) + defender negation (Shield Dust 0×),
    # 1.0 = no effect. Indexed by ability num, mirrors ability_damage_mult's gather.
    ability_secondary_mult = torch.ones(n_abilities, dtype=torch.float32)
    for aid, m in _ABILITY_SECONDARY_MULTS.items():
        ad = gen3_data.abilities.get(aid)
        if ad is not None and 0 <= ad.num < n_abilities:
            ability_secondary_mult[ad.num] = float(m)
    ability_secondary_block = torch.ones(n_abilities, dtype=torch.float32)
    for aid, m in _ABILITY_SECONDARY_BLOCKS.items():
        ad = gen3_data.abilities.get(aid)
        if ad is not None and 0 <= ad.num < n_abilities:
            ability_secondary_block[ad.num] = float(m)

    # gen3_bidir_threat_trunk_v1: the EXPECTED-LATENT-DEFENDER tables — for an UNREVEALED opp mon the op
    # can't read real types/ability, so it marginalizes over the move-belief's P(species). species_types
    # carries the (type1,type2) TypeEncoder ids; species_exp_mult folds the type chart × the per-species
    # expected ability immunity into ONE [n_species, N_TYPE_IDX] expected-multiplier table (matmul with
    # P(species)). Built here so they reuse the chart + ability_damage_mult already in scope.
    species_types = build_species_types(n_species)
    species_exp_mult = build_species_exp_mult(n_species, chart, ability_damage_mult, species_types)

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
        "HP_TYPED_NUMS": hp_typed_nums,          # gen3_opp_hp_typed_candidates_v1: the 16 typed-HP dex nums
        "HP_CAND_MASK": hp_cand_mask,            #   + the mask zeroing 237 + the typed nums from raw `w`
        "MOVE_EFFECT_FLAGS": move_effect_flags,
        "ABILITY_DAMAGE_MULT": ability_damage_mult,
        "SPECIES_TYPE": species_types,
        "SPECIES_IS_GHOST": build_species_is_ghost(species_types),
        "SPECIES_EXP_MULT": species_exp_mult,
        # full (mean,std) spread prior on the op too (SpreadBelief owns its own copy) — E[bulk] for an
        # unrevealed defender = P(species) @ means; the speed (mean,std) feeds the probabilistic outspeed.
        "SPECIES_SPREAD_PRIOR": build_opp_spread_prior(n_species),
        **build_trap_tables(n_species, n_abilities),
        **build_self_boost_tables(n_moves),
        **build_recovery_tables(n_moves),
        **build_sleep_tables(n_species, n_abilities),
        "MOVE_SECONDARY": move_secondary,
        "MOVE_PRIORITY": move_priority,
        "MOVE_DRAIN": move_drain,
        "MOVE_RECOIL": move_recoil,
        "MOVE_FIXED_DAMAGE": move_fixed_damage,
        "ABILITY_SECONDARY_MULT": ability_secondary_mult,
        "ABILITY_SECONDARY_BLOCK": ability_secondary_block,
        # gen3_unified_status_landing_v1: the status-MOVE landing tables (merged in so the op registers them
        # through the same single buffer loop). All non-persistent, recomputable-from-data.
        **build_status_landing(n_moves, n_species, n_abilities),
        # gen3_unified_choice_band_v1: P(CB | species) usage prior — the op's CB belief for an unrevealed opp.
        "SPECIES_CB_PRIOR": build_species_cb_prior(n_species),
        # gen3_unrevealed_outgoing_prior_v1: P(species) usage prior — the expected-latent defender for the
        # outgoing kernel's UNREVEALED columns (Species-Clause-filtered live in `unrevealed_species_probs`).
        "SPECIES_USAGE_PRIOR": build_species_usage_prior(n_species),
        # gen3_nature_ev_belief_v1: the [25,5] nature multiplier table — the op marginalises the nonlinear
        # P(KO) over the believed nature distribution (--spread-belief-nature-marginalize) using these.
        "NATURE_MULT": build_nature_mult(),
    }


# gen3_bidir_threat_trunk_v1: the EXPECTED-LATENT-DEFENDER buffers. For an UNREVEALED opp mon (no team
# preview in gen3) the op has no observed types/ability — so it marginalizes the move-belief's per-slot
# P(species) through the type chart + the per-species Smogon ability prior, with P(KO) NULLED (owner
# decision: a full-HP switch-in is ~never OHKO'd, so the threshold isn't a useful signal and the Jensen
# marginalization isn't worth the complexity). Only the expected MAGNITUDE survives.
def build_species_types(n_species: int) -> torch.Tensor:
    """``[n_species, 2]`` long — the TypeEncoder ids of each species' (type1, type2) from
    ``SpeciesData.types``; a mono-type species (and the unknown species num 0) gets idx 0 in slot 2,
    matching the obs mono-type convention (idx 0 = neutral in the chart). Non-persistent buffer."""
    types = torch.zeros(n_species, 2, dtype=torch.long)
    for sid in gen3_data.species.base_form_ids():
        sd = cast(SpeciesData, gen3_data.species.get(sid))
        if not (0 <= sd.num < n_species):
            continue
        for j, tname in enumerate(sd.types[:2]):
            types[sd.num, j] = _T2I.get(tname, 0)
    return types


def build_species_is_ghost(species_types: torch.Tensor) -> torch.Tensor:
    """``[n_species]`` float — 1.0 where the species is Ghost-typed (either slot).

    gen3_switch_branch_v1: `P(their slot j is Ghost)` = `P(species) @ this`, the leak-free
    marginal a Rapid Spin needs about an UNREVEALED arrival (gen-3 Rapid Spin fails outright
    against a Ghost — no damage and no hazard removal). Derived from ``SPECIES_TYPE`` rather than
    re-read from the facade, so the two can never disagree about a forme. Non-persistent buffer."""
    ghost = _T2I.get("GHOST")
    if ghost is None:
        raise ValueError("the TypeEncoder table has no GHOST id — SPECIES_IS_GHOST has no axis.")
    return ((species_types[:, 0] == ghost) | (species_types[:, 1] == ghost)).float()


def build_species_exp_mult(n_species: int, chart: torch.Tensor, ability_damage_mult: torch.Tensor,
                           species_types: torch.Tensor) -> torch.Tensor:
    """``[n_species, N_TYPE_IDX]`` the EXPECTED damage multiplier of each attacking type vs a species,
    folding BOTH marginalizations into one table so the op needs only ``E[mult vs att] = Σ_s P(s)·table``
    (a single matmul with the move-belief's P(species)):

      type effectiveness:  ``CHART[t1(s), att] · CHART[t2(s), att]``  (the species' own defensive types)
      expected ability:    ``1 - Σ_a P(a|s)·(1 - ABILITY_DAMAGE_MULT[a, att])``  — marginalize the
                            Smogon per-species ability prior over the chart's immunity/resist abilities
                            (Levitate→Ground 0×, Water/Volt Absorb→Water/Electric 0×, Flash Fire→Fire 0×,
                            Thick Fat→Fire/Ice 0.5×). Residual prior mass (and every non-immunity ability)
                            is neutral 1.0×, so the form is robust to an unnormalized prior — it can only
                            REDUCE the multiplier below the raw type effectiveness.

    Non-persistent buffer (pure data-derived, recomputable). The unknown species (num 0) stays NEUTRAL
    (types (0,0) → chart row 0 is all 1.0; no ability prior → no reduction)."""
    t1 = species_types[:, 0]
    t2 = species_types[:, 1]
    type_eff = chart[t1] * chart[t2]                       # [n_species, N_TYPE_IDX] (chart row = att axis)
    reduction = 1.0 - ability_damage_mult                  # [n_abilities, N_TYPE_IDX]; 0 for neutral abilities
    n_abilities = ability_damage_mult.shape[0]
    exp_ability = torch.ones(n_species, N_TYPE_IDX, dtype=torch.float32)
    for sid in gen3_data.species.base_form_ids():
        sd = cast(SpeciesData, gen3_data.species.get(sid))
        snum = sd.num
        if not (0 <= snum < n_species):
            continue
        acc = torch.zeros(N_TYPE_IDX, dtype=torch.float32)
        for aid, p in (gen3_data.priors.ability(sid) or {}).items():
            ad = gen3_data.abilities.get(aid)
            if ad is None or not (0 <= ad.num < n_abilities):
                continue
            acc = acc + float(p) * reduction[ad.num]
        exp_ability[snum] = (1.0 - acc).clamp(min=0.0)
    return type_eff * exp_ability


# gen3_unified_choice_band_v1: Choice Band is the dominant damage-relevant gen3 item — it ×1.5 the holder's
# PHYSICAL Attack (and move-locks it). The op prices it as a per-species BELIEF, not a baked multiplier: a
# usage prior P(CB | species) the op collapses to 0/1 once the item is revealed, then exposes the
# CB-CONDITIONAL physical damage tail + P(CB) decorrelated (the head weights them — the same provide-the-fact
# philosophy as the crit-split, since OHKO is a nonlinear threshold a mean-field ×(1+0.5·p_cb) would blur).
CHOICE_BAND_ITEM_NUM = int(cast(ItemData, gen3_data.items.get("choiceband")).num)   # 220
CHOICE_BAND_PHYS_MULT = 1.5
CURSE_MOVE_NUM = int(cast(MoveData, gen3_data.moves.get("curse")).num)              # 174 — the C1 runtime type branch
TOXIC_MOVE_NUM = int(cast(MoveData, gen3_data.moves.get("toxic")).num)              # 92 — C2 tox-vs-psn (shared cat 5)
REST_MOVE_NUM = int(cast(MoveData, gen3_data.moves.get("rest")).num)                # 156 — C3's self-sleep cost channel
BATON_PASS_MOVE_NUM = int(cast(MoveData, gen3_data.moves.get("batonpass")).num)     # 226 — C5's receiver-axis edge


def build_species_cb_prior(n_species: int) -> torch.Tensor:
    """``[n_species]`` P(holds Choice Band | species) from the Smogon item usage prior
    (`gen3_data.priors.items`) — Aerodactyl ≈0.76, Metagross ≈0.31, Snorlax ≈0.03. The op's PRIOR for an
    unrevealed opponent's CB belief (collapsed to 0/1 once the held/consumed item is revealed). Non-persistent
    buffer (recomputable from data/, zero params). A species with no usage data reads 0.0."""
    prior = torch.zeros(n_species, dtype=torch.float32)
    for sid in gen3_data.species.base_form_ids():
        snum = cast(SpeciesData, gen3_data.species.get(sid)).num
        if 0 <= snum < n_species:
            prior[snum] = float((gen3_data.priors.items(sid) or {}).get("choiceband", 0.0))
    return prior


# gen3_unified_op_physics_v1: FIXED-damage moves (constant damage at L100, ignoring Atk/Def/roll/crit but
# RESPECTING type immunity). They read BP 0 in the dex so the formula gives ~0 — the op overrides with this
# constant. Mirrors incoming_damage.FIXED_DAMAGE (the CPU block the GPU op must reach parity with so
# --unified-obs / --mask-incoming-damage-obs doesn't regress the model's damage understanding).
_FIXED_DAMAGE = {"seismictoss": 100, "nightshade": 100, "dragonrage": 40, "sonicboom": 20}


def build_move_fixed_damage(n_moves: int) -> torch.Tensor:
    """``[n_moves]`` L100 FIXED damage per move (Seismic Toss / Night Shade 100, Dragon Rage 40, Sonic
    Boom 20), 0 for every other move. Non-persistent buffer. The op multiplies by the type-immunity gate so
    Fighting Seismic Toss reads 0 vs Ghost (your named edge) and Ghost Night Shade 0 vs Normal."""
    fd = torch.zeros(n_moves, dtype=torch.float32)
    for mid, dmg in _FIXED_DAMAGE.items():
        md = cast(MoveData, gen3_data.moves.get(mid))
        if md is not None and 0 <= md.num < n_moves:
            fd[md.num] = float(dmg)
    return fd


# --- gen3_unified_status_landing_v1: "will my STATUS move land vs THIS opponent" tables ------------- #
# The op's GPU replacement for the masked move-effect block's `status_will_land`. The gen3 immunity RULES
# are imported from `gen3_mechanics` (STATUS_MOVE_IMMUNITY / ABILITY_STATUS_IMMUNITY) — the SINGLE,
# Showdown-fuzz-validated source — and lifted onto the op's num/type/ability axes here. We ADD Leech Seed
# (a volatile the CPU `status_will_land` never modeled): Grass-immune, not a major status (it can be applied
# to an already-statused mon), no ability blocks it. Sleep Clause + already-statused are computed live in
# the op from the obs (not data) since they depend on board state.
_STATUS_CAT = {"par": 1, "brn": 2, "frz": 3, "slp": 4, "psn": 5, "tox": 5}  # tox→psn (same Steel/Poison block)
_SLP_CAT = 4
LEECH_SEED_CAT = 6
N_STATUS_CAT = 7                              # index 0 = "not a status move"; 1..5 majors; 6 = Leech Seed
_LEECH_SEED_ID = "leechseed"
# gen3_status_economy_v1: the ability whose gen3 rule is "the status is shed on switch-out".
_NATURAL_CURE_ID = "naturalcure"
# SSOT guard: every status id the gen3_mechanics ability-immunity rules name MUST map to a known category,
# so a future gen3_mechanics edit (e.g. a new blocked status) fails LOUDLY at import here rather than
# silently dropping that ability's block from SPECIES_STATUS_BLOCK_PRIOR / ABILITY_STATUS_BLOCK.
assert {s for ss in ABILITY_STATUS_IMMUNITY.values() for s in ss} <= set(_STATUS_CAT), \
    "gen3_mechanics.ABILITY_STATUS_IMMUNITY names a status not in damage_tables._STATUS_CAT — add it."


def build_status_landing(n_moves: int, n_species: int, n_abilities: int) -> Dict[str, torch.Tensor]:
    """The per-move / per-ability / per-species status-landing tables (all on the gen3-data num axes):

      MOVE_STATUS_CAT[n_moves]                 long — 0 (not a status move) else the _STATUS_CAT / LEECH_SEED_CAT it inflicts
      MOVE_STATUS_IDENT[n_moves, 6]            one-hot over SECONDARY_COLS[:6] (par,brn,frz,slp,psn,tox) for a
                                               DEDICATED status move — the status IDENTITY, exact where
                                               MOVE_STATUS_CAT is not: cat 5 collapses psn and tox (they share the
                                               Steel/Poison immunity), and Toxic's escalating residual is a
                                               different outcome from Poison Powder's flat 1/8. Sourced from
                                               `MoveData.status_inflicted`, which distinguishes them. All-zero for
                                               a damaging move (its identity rides MOVE_SECONDARY) and for Leech
                                               Seed (no major-status column). gen3_pair_outcome_v1.
      MOVE_CURES_SELF_STATUS[n_moves]          1.0 if using the move leaves its USER statusless in ONE turn —
                                               `cures_self_status` (Refresh) OR `cures_team_status` (Heal Bell /
                                               Aromatherapy, which cure the user as part of the party). The
                                               facade keeps those two apart because they differ in SCOPE; here
                                               only "is this mon clean afterwards" matters, so they merge.
                                               REST is deliberately NOT here (the facade reports False for both):
                                               it undoes a status at a different price — 2 turns asleep — which
                                               the op prices from its own `rest_sleep_noeb`. The tempo_cost
                                               coordinate's source. gen3_pair_outcome_v1.
      MOVE_CURES_TEAM_STATUS[n_moves]          1.0 for a PARTY-WIDE cure (Heal Bell / Aromatherapy) — the
                                               CLERIC path's source (gen3_status_economy_v1). Kept APART from
                                               MOVE_CURES_SELF_STATUS, which merges the two because only "is
                                               this mon clean afterwards" mattered there. The scope is exactly
                                               what makes the cleric path exist: a party-wide cure reaches a
                                               mon sitting on the BENCH, so a teammate's Heal Bell is an undo
                                               path for mon j; Refresh is not.
      MOVE_INFLICTS_STATUS[n_moves]            1.0 if it is a dedicated status move (incl. Leech Seed)
      MOVE_IS_SLEEP[n_moves]                   1.0 if it inflicts sleep (the Sleep-Clause gate)
      MOVE_BLOCKED_IF_STATUSED[n_moves]        1.0 for a MAJOR status (can't double-apply); 0 for Leech Seed
      MOVE_STATUS_TYPE_IMMUNE[n_moves, N_TYPE_IDX]   1.0 where a DEFENDER type is immune to THIS move's status
      ABILITY_STATUS_BLOCK[n_abilities, N_STATUS_CAT]   1.0 if the (revealed) ability hard-blocks that category
      ABILITY_NATURAL_CURE[n_abilities]        1.0 at the Natural Cure row (gen3_status_economy_v1). NOT a
                                               status BLOCK — the status lands in full and every per-turn
                                               severity is paid while the mon stays in; what the ability buys
                                               is an UNDO PATH (the status is shed on switch-out), which is
                                               `tempo_cost`'s subject and not `ABILITY_STATUS_BLOCK`'s.
                                               Resolved FAIL-LOUD so a data rename cannot silently make every
                                               Natural Cure mon read as having no answer.
      SPECIES_STATUS_BLOCK_PRIOR[n_species, N_STATUS_CAT]   P(species' ability blocks) — Smogon-prior marginal

    Type immunity is keyed by MOVE id (the gen3 rule): Thunder Wave→Ground, Toxic/Poison Gas/Poison Powder
    →Steel/Poison, Will-O-Wisp→Fire (Stun Spore/Glare paralysis has NO type immunity), + Leech Seed→Grass."""
    cat = torch.zeros(n_moves, dtype=torch.long)
    # gen3_pair_outcome_v1: the DEDICATED status move's IDENTITY on the 6 major SECONDARY_COLS axis.
    # `_STATUS_CAT` cannot serve here — it folds tox into psn — so this reads `status_inflicted`
    # directly and keeps the two apart.
    ident = torch.zeros(n_moves, _SECONDARY_MAJOR_COLS_N, dtype=torch.float32)
    _ident_col = {c: i for i, c in enumerate(SECONDARY_COLS[:_SECONDARY_MAJOR_COLS_N])}
    cures_self = torch.zeros(n_moves, dtype=torch.float32)
    # gen3_status_economy_v1: the CLERIC table — party-wide cures only (Heal Bell / Aromatherapy).
    cures_team = torch.zeros(n_moves, dtype=torch.float32)
    inflicts = torch.zeros(n_moves, dtype=torch.float32)
    is_sleep = torch.zeros(n_moves, dtype=torch.float32)
    blocked_if_statused = torch.zeros(n_moves, dtype=torch.float32)
    type_immune = torch.zeros(n_moves, N_TYPE_IDX, dtype=torch.float32)
    for mid in gen3_data.moves.raw():
        md = cast(MoveData, gen3_data.moves.get(mid))
        num = md.num
        if not (0 <= num < n_moves) or num == HIDDEN_POWER_NUM:
            continue
        # gen3_pair_outcome_v1 — set BEFORE the `c == 0` skip below: a cure move need not itself
        # inflict a status (Refresh/Heal Bell do not), so populating it inside the status branch
        # would silently drop most of the table.
        cures_self[num] = 1.0 if (md.cures_self_status or md.cures_team_status) else 0.0
        cures_team[num] = 1.0 if md.cures_team_status else 0.0
        if mid == _LEECH_SEED_ID:
            c = LEECH_SEED_CAT
        elif md.status_inflicted is not None:
            c = _STATUS_CAT.get(md.status_inflicted, 0)
        else:
            c = 0
        if c == 0:
            continue
        cat[num] = c
        # gen3_pair_outcome_v1: the identity column, from the RAW status id (so tox != psn). Leech
        # Seed has no major-status column and correctly stays all-zero.
        _col = _ident_col.get(md.status_inflicted or "")
        if _col is not None:
            ident[num, _col] = 1.0
        inflicts[num] = 1.0
        is_sleep[num] = 1.0 if c == _SLP_CAT else 0.0
        blocked_if_statused[num] = 0.0 if c == LEECH_SEED_CAT else 1.0
        immune_types = ({PokemonType.GRASS} if c == LEECH_SEED_CAT
                        else set(STATUS_MOVE_IMMUNITY.get(mid, frozenset())))
        for pt in immune_types:
            ti = _T2I.get(pt.name)
            if ti is not None:
                type_immune[num, ti] = 1.0

    # gen3_status_economy_v1: the Natural Cure row. Fails loud rather than staying all-zero — an
    # all-zero table is indistinguishable from "no mon on either team has the ability", i.e. from a
    # null RESULT, and it would silently restore the exact defect this closes.
    ability_natural_cure = torch.zeros(n_abilities, dtype=torch.float32)
    _nc = gen3_data.abilities.get(_NATURAL_CURE_ID)
    if _nc is None or not (0 <= _nc.num < n_abilities):
        raise ValueError(
            f"gen3_data.abilities has no usable {_NATURAL_CURE_ID!r} row — the status-economy undo "
            "path would silently never fire, which reads exactly like a mon with no answer.")
    ability_natural_cure[_nc.num] = 1.0

    ability_block = torch.zeros(n_abilities, N_STATUS_CAT, dtype=torch.float32)
    for aid, statuses in ABILITY_STATUS_IMMUNITY.items():
        ad = gen3_data.abilities.get(aid)
        if ad is None or not (0 <= ad.num < n_abilities):
            continue
        for s in statuses:
            # `c` is bound as `int` by the status-MOVE loop above; this second, independent loop
            # reuses the name for an Optional lookup it immediately guards. Narrowing is correct at
            # runtime — mypy just keeps one binding per name in a function.
            c = _STATUS_CAT.get(s)  # type: ignore[assignment]
            if c is not None:
                ability_block[ad.num, c] = 1.0

    # Marginalize the per-species Smogon ability prior over the same ABILITY_STATUS_IMMUNITY rule → the
    # P(this species' ability blocks status c) used when the opp ability is UNREVEALED (priors-then-confirm).
    species_block = torch.zeros(n_species, N_STATUS_CAT, dtype=torch.float32)
    for sid in gen3_data.species.base_form_ids():
        sd = cast(SpeciesData, gen3_data.species.get(sid))
        snum = sd.num
        if not (0 <= snum < n_species):
            continue
        for aid, p in (gen3_data.priors.ability(sid) or {}).items():
            # DEDUPE categories per ability: Immunity blocks both psn & tox which fold to ONE category (5),
            # so add the prior mass ONCE per distinct category (else a 0.86 Immunity prior double-counts → 1.0).
            cats = {_STATUS_CAT[s] for s in ABILITY_STATUS_IMMUNITY.get(aid, ()) if s in _STATUS_CAT}
            for c in cats:
                species_block[snum, c] += float(p)
    species_block = species_block.clamp(max=1.0)

    return {
        "MOVE_STATUS_CAT": cat,
        "MOVE_STATUS_IDENT": ident,
        "MOVE_CURES_SELF_STATUS": cures_self,
        "MOVE_CURES_TEAM_STATUS": cures_team,
        "MOVE_INFLICTS_STATUS": inflicts,
        "MOVE_IS_SLEEP": is_sleep,
        "MOVE_BLOCKED_IF_STATUSED": blocked_if_statused,
        "MOVE_STATUS_TYPE_IMMUNE": type_immune,
        "ABILITY_STATUS_BLOCK": ability_block,
        "ABILITY_NATURAL_CURE": ability_natural_cure,
        "SPECIES_STATUS_BLOCK_PRIOR": species_block,
    }


# gen3_edge_bias_trunk_v1 (X family): Pursuit — the switch-out punisher (doubled BP on the
# switching mon). Resolved FAIL-LOUD at import like the trap ids.
def _pursuit_num() -> int:
    md = gen3_data.moves.get("pursuit")
    if md is None or md.num <= 0:
        raise ValueError("pursuit failed to resolve — the X-family exposure edge would be empty.")
    return md.num


# gen3_edge_bias_trunk_v1 (T family): the three gen3 TRAP abilities. Resolution is FAIL-LOUD — a
# rename/missing id would otherwise silently zero the whole family (the GIGO class).
_TRAP_ABILITY_IDS = ("shadowtag", "arenatrap", "magnetpull")


def build_trap_tables(n_species: int, n_abilities: int) -> Dict[str, torch.Tensor]:
    """Trapping tables (T-family edges):

      ABILITY_TRAP[n_abilities, 3]        1.0 at (ability, k) for shadowtag/arenatrap/magnetpull
      ABILITY_IS_LEVITATE[n_abilities]    1.0 for levitate (the arena-trap grounded check)
      SPECIES_TRAP_PRIOR[n_species, 4]    P(species runs [shadowtag, arenatrap, magnetpull, levitate])
                                          — the Smogon ability-prior marginal (unrevealed-ability read)
      TYPE_IS_STEEL / TYPE_IS_FLYING [N_TYPE_IDX]  victim-type masks (magnet-pull / grounded checks)
    """
    trap = torch.zeros(n_abilities, 3, dtype=torch.float32)
    trap_nums = []
    for k, aid in enumerate(_TRAP_ABILITY_IDS):
        ad = gen3_data.abilities.get(aid)
        if ad is None or not (0 <= ad.num < n_abilities):
            raise ValueError(f"trap ability {aid!r} failed to resolve (got {ad}) — the T-family "
                             "tables would be silently empty. Fix the id / regenerate gen3_abilities.")
        trap[ad.num, k] = 1.0
        trap_nums.append(ad.num)
    lev = gen3_data.abilities.get("levitate")
    if lev is None or not (0 <= lev.num < n_abilities):
        raise ValueError("levitate failed to resolve — the arena-trap grounded check would be wrong.")
    is_lev = torch.zeros(n_abilities, dtype=torch.float32)
    is_lev[lev.num] = 1.0

    prior = torch.zeros(n_species, 4, dtype=torch.float32)
    for sid in gen3_data.species.base_form_ids():
        sd = cast(SpeciesData, gen3_data.species.get(sid))
        if not (0 <= sd.num < n_species):
            continue
        for aid, pv in (gen3_data.priors.ability(sid) or {}).items():
            if aid in _TRAP_ABILITY_IDS:
                prior[sd.num, _TRAP_ABILITY_IDS.index(aid)] += float(pv)
            elif aid == "levitate":
                prior[sd.num, 3] += float(pv)
    prior = prior.clamp(max=1.0)

    steel = torch.zeros(N_TYPE_IDX, dtype=torch.float32)
    flying = torch.zeros(N_TYPE_IDX, dtype=torch.float32)
    steel[_T2I["STEEL"]] = 1.0
    flying[_T2I["FLYING"]] = 1.0
    return {
        "ABILITY_TRAP": trap,
        "ABILITY_IS_LEVITATE": is_lev,
        "SPECIES_TRAP_PRIOR": prior,
        "TYPE_IS_STEEL": steel,
        "TYPE_IS_FLYING": flying,
    }


_SELF_BOOST_STAT_ORDER = ("atk", "def", "spa", "spd", "spe")   # == DamageOperator._boost_stages order


def build_self_boost_tables(n_moves: int) -> Dict[str, torch.Tensor]:
    """Self-boost table (the C1 consequence-edge kernel):

      MOVE_SELF_BOOSTS[n_moves, 5]   the (atk,def,spa,spd,spe) stage deltas of the ~17 PURE setup
                                     moves (`MoveData.self_boosts`, gen3_setup_moves_v1 — Swords
                                     Dance +2 atk, Dragon Dance +1/+1, Calm Mind, Agility, …)

    Rows are the DECLARATIVE pure-setup classification only: Belly Drum (HP-cost callback), Curse
    (type-conditional, target normal), Defense Curl/Minimize (volatile / evasion) are all-zero —
    the same gates that keep those moves fail-loud in the rust engine. The C1 kernel prices only
    what this table asserts, so a zero row degrades to "no priced consequence", never a wrong one.
    Fail-loud: the canonical carrier (Swords Dance = +2 atk) is asserted so a facade/extractor
    regression can't silently empty the table."""
    t = torch.zeros(n_moves, 5, dtype=torch.float32)
    for mid in gen3_data.moves.raw():
        md = cast(MoveData, gen3_data.moves.get(mid))
        if md is None or not md.self_boosts or not (0 <= md.num < n_moves):
            continue
        for stat, stages in md.self_boosts:
            # SKIP any stat outside the five BATTLE stats this table has columns for.
            # `accuracy`/`evasion` are real gen-3 self-boost stats (Double Team,
            # `gen3_double_team_v1`) but the DamageOperator prices only the five, so an
            # evasion row has "no priced consequence" — exactly the degradation this
            # docstring already promises for Defense Curl / Minimize. Skipping keeps the
            # tensor BIT-IDENTICAL to before the data gained those rows.
            # ⚠️ This used to be a bare `.index(stat)`, which RAISED `ValueError: tuple.index(x):
            # x not in tuple` the moment the extractor admitted `evasion` — 321 test failures from
            # a one-row data change. The extractor's `_self_boosts` guard was load-bearing for TWO
            # consumers, not one: the rust engine (whose stated reason had gone stale) AND this
            # table (whose reason is LIVE). Neither the extractor-parity nor the obs-golden gate
            # covers it, because neither builds the damage tables.
            if stat not in _SELF_BOOST_STAT_ORDER:
                continue
            t[md.num, _SELF_BOOST_STAT_ORDER.index(stat)] = float(stages)
    sd = gen3_data.moves.get("swordsdance")
    if sd is None or float(t[sd.num, 0]) != 2.0:
        raise ValueError("MOVE_SELF_BOOSTS: Swords Dance did not resolve to +2 atk — the setup "
                         "table is empty/misaligned. Regenerate gen3_moves.json (selfBoosts).")
    # Curse, non-Ghost branch (owner-prioritized 2026-08-05 — CurseLax / Curse-Registeel are
    # gen3ou-defining): type-CONDITIONAL, so it stays OUT of the per-move table (which is
    # type-blind and doubles as the rust engine's draw-free contract) and rides its own [5]
    # buffer + a Ghost-type mask; the C1 kernel resolves the branch from the user's live types.
    curse = gen3_data.moves.get("curse")
    if curse is None:
        raise ValueError("curse failed to resolve — the C1 Curse branch would silently vanish.")
    if float(t[curse.num].abs().sum()) != 0.0:
        raise ValueError("MOVE_SELF_BOOSTS grew a Curse row — the type-blind table must NOT "
                         "price Curse (a Ghost user's Curse is a different move); the runtime "
                         "branch owns it.")
    curse_boosts = torch.zeros(5, dtype=torch.float32)
    for stat, stages in CURSE_NON_GHOST_BOOSTS.items():
        curse_boosts[_SELF_BOOST_STAT_ORDER.index(stat)] = float(stages)
    is_ghost = torch.zeros(N_TYPE_IDX, dtype=torch.float32)
    is_ghost[_T2I["GHOST"]] = 1.0
    # Belly Drum (the recorded TODO, now priced — MODEL-side ONLY, the selfBoosts JSON field
    # stays pure as the rust engine's draw-free contract): a +12 atk delta CLAMPS to the +6
    # "maximize" exactly (`_boost_mult` clamps stages to ±6 from any start), paid for by the
    # half-max-HP cost below + the fails-below-half gate in `_setup_deltas`.
    bd = gen3_data.moves.get("bellydrum")
    if bd is None or not (0 <= bd.num < n_moves):
        raise ValueError("bellydrum failed to resolve — its curated C1 row would silently vanish.")
    t[bd.num, 0] = 12.0
    hp_cost = torch.zeros(n_moves, dtype=torch.float32)
    hp_cost[bd.num] = 0.5
    return {"MOVE_SELF_BOOSTS": t, "CURSE_BOOSTS": curse_boosts, "TYPE_IS_GHOST": is_ghost,
            "MOVE_BOOST_HP_COST": hp_cost}


def build_recovery_tables(n_moves: int) -> Dict[str, torch.Tensor]:
    """Recovery table (the C3 consequence-edge kernel):

      MOVE_HEAL_FRACTION[n_moves]   the IMMEDIATE self-heal as a fraction of max HP — 0.5 for
                                    the plain heals (Recover/Softboiled/Milk Drink/Slack Off)
                                    AND the weather heals (Moonlight/Morning Sun/Synthesis — a
                                    documented v1 FLAT approximation of gen3's 2/3-sun /
                                    1/4-other-weather / 1/2-clear), 1.0 for Rest (its sleep
                                    cost is deliberately unpriced in v1).

    Wish is EXCLUDED — its heal is DELAYED and slot-keyed; the `gen3_wish_wired_v1` obs scalars
    own that fact. Rows come from `MoveData.is_heal` (flags.heal), so a non-heal move is an
    all-zero row and simply unpriced. Fail-loud on the canonical carrier (Recover = 0.5)."""
    t = torch.zeros(n_moves, dtype=torch.float32)
    wh = torch.zeros(n_moves, dtype=torch.float32)
    _WEATHER_HEALS = ("moonlight", "morningsun", "synthesis")
    for mid in gen3_data.moves.raw():
        md = cast(MoveData, gen3_data.moves.get(mid))
        if md is None or not md.is_heal or not (0 <= md.num < n_moves):
            continue
        if mid == "wish":
            continue
        t[md.num] = 1.0 if mid == "rest" else 0.5
        if mid in _WEATHER_HEALS:
            wh[md.num] = 1.0        # the C3 kernel folds LIVE weather: 2/3 sun, 1/4 other, 1/2 clear
    rec = gen3_data.moves.get("recover")
    if rec is None or float(t[rec.num]) != 0.5:
        raise ValueError("MOVE_HEAL_FRACTION: Recover did not resolve to 0.5 — the recovery "
                         "table is empty/misaligned. Regenerate gen3_moves.json (isHeal).")
    if float(wh.sum()) != 3.0:
        raise ValueError("MOVE_WEATHER_HEAL: the 3 gen3 weather heals did not all resolve.")
    return {"MOVE_HEAL_FRACTION": t, "MOVE_WEATHER_HEAL": wh}


def build_sleep_tables(n_species: int, n_abilities: int) -> Dict[str, torch.Tensor]:
    """Sleep-consequence tables (the C2 sleep channels):

      SPECIES_EARLYBIRD_PRIOR[n_species]   P(species runs Early Bird) — the Smogon ability-prior
                                           marginal (the unrevealed-ability read; ~0 for most)
      ABILITY_IS_EARLYBIRD[n_abilities]    1.0 at the earlybird row (revealed → exact)

    Fail-loud on earlybird resolving (the trap-tables convention) — a silent miss would make the
    EB marginalisation a no-op that always prices the full 2.5 free turns."""
    eb = gen3_data.abilities.get("earlybird")
    if eb is None or not (0 <= eb.num < n_abilities):
        raise ValueError("earlybird failed to resolve — the sleep-consequence EB fold would be "
                         "silently empty. Fix the id / regenerate gen3_abilities.")
    is_eb = torch.zeros(n_abilities, dtype=torch.float32)
    is_eb[eb.num] = 1.0
    prior = torch.zeros(n_species, dtype=torch.float32)
    for sid in gen3_data.species.base_form_ids():
        sd = cast(SpeciesData, gen3_data.species.get(sid))
        if not (0 <= sd.num < n_species):
            continue
        prior[sd.num] = float((gen3_data.priors.ability(sid) or {}).get("earlybird", 0.0))
    return {"SPECIES_EARLYBIRD_PRIOR": prior.clamp(max=1.0), "ABILITY_IS_EARLYBIRD": is_eb}


def build_move_type_idx(n_moves: int) -> torch.Tensor:
    """``[n_moves]`` TypeEncoder index of each move's canonical type (the `MoveLatentEncoder`'s type
    lookup for its context-free latent table). HP (num 237) stays idx 0 (unknown) — HP's type is
    per-candidate, resolved live elsewhere; the latent treats HP as one class. Mirrors the MOVE_TYPE_IDX
    built in build_damage_buffers (kept standalone so the encoder doesn't build the whole damage table)."""
    idx = torch.zeros(n_moves, dtype=torch.long)
    for mid in gen3_data.moves.raw():
        md = cast(MoveData, gen3_data.moves.get(mid))
        if 0 <= md.num < n_moves and md.num != HIDDEN_POWER_NUM:
            idx[md.num] = _move_type_idx(md)
    return idx


def build_move_attr(n_moves: int) -> torch.Tensor:
    """``[n_moves, N_MOVE_ATTR]`` structured per-move attribute table for the `MoveLatentEncoder` —
    the context-free "what does this move do" features (column order == `MOVE_ATTR_COLS`). TYPE is NOT
    here (it rides the shared type embedding the encoder concatenates). Registered as a NON-persistent
    buffer (pure data-derived, recomputable). HP (num 237) is SKIPPED → its row is left ALL-ZERO (same as
    build_move_type_idx leaving idx 0): 17 move-ids collide on num 237, so populating it would bake in
    whichever typed variant iterated last (order-dependent). The latent still distinguishes HP types via
    the slot's RESOLVED type embedding; the op's typed-candidate path handles HP's damage."""
    attr = torch.zeros(n_moves, N_MOVE_ATTR, dtype=torch.float32)
    idx = {c: i for i, c in enumerate(MOVE_ATTR_COLS)}
    for mid in gen3_data.moves.raw():
        md = cast(MoveData, gen3_data.moves.get(mid))
        num = md.num
        if num == HIDDEN_POWER_NUM or not (0 <= num < n_moves):
            continue
        is_status = md.base_power <= 0
        is_phys = md.category == MoveCategory.PHYSICAL
        attr[num, idx["bp_norm"]] = min(md.base_power / 200.0, 1.0)
        attr[num, idx["is_phys"]] = 1.0 if is_phys else 0.0
        attr[num, idx["is_spec"]] = 1.0 if (not is_status and not is_phys) else 0.0
        attr[num, idx["is_status"]] = 1.0 if is_status else 0.0
        attr[num, idx["accuracy"]] = 1.0 if md.never_miss else float(md.accuracy) / 100.0
        attr[num, idx["never_miss"]] = 1.0 if md.never_miss else 0.0
        attr[num, idx["priority_norm"]] = float(md.priority) / _PRIORITY_NORM
        attr[num, idx["drain"]] = float(md.drain_fraction)
        attr[num, idx["recoil"]] = float(md.recoil_fraction)
        for col in SECONDARY_COLS:
            attr[num, idx[col]] = md.secondary_chance(col)
        attr[num, idx["is_heal"]] = 1.0 if md.is_heal else 0.0
        attr[num, idx["is_boost"]] = 1.0 if md.is_boost else 0.0
        attr[num, idx["is_protect"]] = 1.0 if md.is_protect else 0.0
        attr[num, idx["is_phaze"]] = 1.0 if md.is_phaze else 0.0
        attr[num, idx["is_hazard"]] = 1.0 if md.is_hazard else 0.0
        attr[num, idx["cures_self"]] = 1.0 if md.cures_self_status else 0.0
        attr[num, idx["cures_team"]] = 1.0 if md.cures_team_status else 0.0
    return attr
