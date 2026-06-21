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

Reference: `designs/ai_v6/design_differentiable_damage_op.md` (§3 the gen3 formula, §7 the
edge-case matrix). The math core mirrors `agents/observation/incoming_damage.py` (the live CPU
belief) and the proven torch port in the ai_v6 design.
"""
from __future__ import annotations

from typing import Dict

import torch

from agents import gen3_data
from agents.enums import MoveCategory, PokemonType
from agents.gen3_mechanics import ABILITY_STATUS_IMMUNITY, STATUS_MOVE_IMMUNITY
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

# --- gen3_unified_move_system_v1: structured secondary effects + the move-attribute table ----------- #
# The 10 secondary-effect columns of MOVE_SECONDARY (the per-effect TRIGGER chance, 0..1, the op prices).
# Order is the contract; mirrors tools/…/sync._SECONDARY_COLS + the facade's secondary_chance() keys.
SECONDARY_COLS = (
    "par", "brn", "frz", "slp", "psn", "tox", "confusion", "flinch", "foe_statdrop", "self_boost",
)
N_SECONDARY = len(SECONDARY_COLS)            # 10
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


def _belief_num(move_id: str, md) -> int:
    """The move num the OPPONENT's move-belief PRIOR keys on. Gen 3 never reveals the opponent's
    Hidden Power TYPE, so every HP the opponent could run is observed BARE — they all aggregate to
    the typeless num ``237`` here, regardless of a typed variant's own distinct data num (355-370,
    which exist only for OUR known-HP per-move tables + obs). This keeps the opp-HP belief mass on
    237 so the operator's 237→16-typed-candidate expansion (weighted by ``hp_probs``) can read it;
    without it, the typed-HP Smogon usage would scatter to 355-370 and the opp-HP belief at 237 would
    collapse to the floor (the model would never anticipate the opponent's Hidden Power). Non-HP
    moves pass through unchanged. See designs/ai_v6/design_typed_hidden_power_ids.md."""
    return HIDDEN_POWER_NUM if move_id.startswith("hiddenpower") else md.num


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

    # gen3_opp_hp_typed_candidates_v1: the DISTINCT dex nums of the 16 TYPED Hidden Powers (355-370,
    # gen3_typed_hidden_power_ids_v1) in HIDDEN_POWER_TYPE_ORDER order — so the DamageOperator can treat the
    # opponent's HP as 16 ORDINARY typed-move candidates (real BP/type at these nums, populated above)
    # instead of a synthetic appended block. The op scatters the opp HP-type BELIEF onto these nums. The
    # bare typeless num 237 (BP 0) is the PRESENCE token, never a damage candidate. `hp_cand_mask` zeros
    # BOTH 237 and the 16 typed nums out of the raw move belief (the op writes the belief back into the
    # typed nums). Derived from the data (NOT hardcoded) so a num remap can't silently misalign.
    hp_typed_nums = torch.tensor(
        [gen3_data.moves.get("hiddenpower" + t.name.lower()).num for t in HIDDEN_POWER_TYPE_ORDER],
        dtype=torch.long,
    )
    hp_cand_mask = torch.ones(n_moves, dtype=torch.float32)
    hp_cand_mask[HIDDEN_POWER_NUM] = 0.0                 # bare typeless HP = the presence token (BP 0)
    hp_cand_mask[hp_typed_nums] = 0.0                    # typed HP slots → the op writes the belief here
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
        md = gen3_data.moves.get(mid)
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
        "SPECIES_EXP_MULT": species_exp_mult,
        # full (mean,std) spread prior on the op too (SpreadBelief owns its own copy) — E[bulk] for an
        # unrevealed defender = P(species) @ means; the speed (mean,std) feeds the probabilistic outspeed.
        "SPECIES_SPREAD_PRIOR": build_opp_spread_prior(n_species),
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
        # gen3_nature_ev_belief_v1: the [25,5] nature multiplier table — the op marginalises the nonlinear
        # P(KO) over the believed nature distribution (--spread-belief-nature-marginalize) using these.
        "NATURE_MULT": build_nature_mult(),
    }


# gen3_unified_spread_belief_v1: the 5 battle-relevant DERIVED stats the SpreadBelief predicts for the
# hidden opponent (HP is skipped — the op uses the obs HP fraction × a neutral maxhp estimate). Order is
# the contract the op + the belief head index by. Index into BASE_STATS' [hp,atk,def,spa,spd,spe] layout.
SPREAD_STAT_COLS = ("atk", "def", "spa", "spd", "spe")
N_SPREAD_STATS = len(SPREAD_STAT_COLS)                       # 5
_SPREAD_BASE_IDX = {"atk": 1, "def": 2, "spa": 3, "spd": 4, "spe": 5}


def build_opp_spread_prior(n_species: int) -> torch.Tensor:
    """``[n_species, 5, 2]`` usage-weighted ``(mean, std)`` of each species' realized L100/IV31 stat VALUE
    for {atk,def,spa,spd,spe}, derived from the Smogon spread priors (`gen3_data.priors.spreads`). This is
    the data-informed PRIOR the `SpreadBelief` head corrects — it REPLACES the DamageOperator's hand-coded
    de-timid (252/×1.1) / neutral-0-EV opp-spread constants with the real usage distribution per species
    (high for an invested sweeper, low for a wall — far better than one flat assumption). A species with no
    spread data falls back to the neutral-EV stat (mean) + a wide std spanning up to max investment.
    Registered as a NON-persistent buffer (pure data-derived, recomputable). Mirrors `priors.gen3_stat`
    (the same L100/IV31 formula the op uses for our revealed mons)."""
    prior = torch.zeros(n_species, N_SPREAD_STATS, 2, dtype=torch.float32)
    for sid in gen3_data.species.raw():
        sd = gen3_data.species.get(sid)
        snum = sd.num
        if not (0 <= snum < n_species):
            continue
        spr = gen3_data.priors.spreads(sid)
        for j, stat in enumerate(SPREAD_STAT_COLS):
            base = int(sd.base_stats.get(stat, 0))
            evi = _SPREAD_BASE_IDX[stat]                      # index into the 6-EV list [hp,atk,def,spa,spd,spe]
            m1 = m2 = wsum = 0.0
            for nature, evs, w in spr:
                nd = gen3_data.natures.get(str(nature).lower())
                mult = nd.multipliers.get(stat, 1.0) if nd is not None else 1.0
                val = float(gen3_data.priors.gen3_stat(base, int(evs[evi]), mult))
                m1 += w * val
                m2 += w * val * val
                wsum += float(w)
            if wsum <= 0.0:                                   # no usage data → neutral mean + wide std
                neutral = float(gen3_data.priors.gen3_stat(base, 0, 1.0))
                maxed = float(gen3_data.priors.gen3_stat(base, 252, 1.1))
                prior[snum, j, 0] = neutral
                prior[snum, j, 1] = max(1.0, (maxed - neutral) / 2.0)
            else:
                mean = m1 / wsum
                var = max(0.0, m2 / wsum - mean * mean)
                prior[snum, j, 0] = mean
                prior[snum, j, 1] = max(1.0, var ** 0.5)
    return prior


# ─────────────────────────────────────────────────────────────────────────────────────────────────────
# gen3_nature_ev_belief_v1 — the NATURE/EV-decomposed spread belief (data foundation).
# `build_opp_spread_prior` above predicts the DERIVED stat directly — a point estimate that sits between the
# nature ×1.1/×0.9 modes, hence the "over-estimates the largest EV" order-statistic bias. The generative head
# instead predicts (nature categorical ⊕ per-stat EV) ⊕ their Smogon priors and COMPUTES the derived stat, so
# the nature asymmetry + the EV budget are STRUCTURAL. These buffers are the prior-fusion bases (mirroring the
# move-belief / HP-type prior fusion) + the multiplier/base tables the head & op need to compute the derived
# stat. All non-persistent (data-derived, recomputable).
N_NATURES = 25                                              # gen3 has exactly 25 natures (num 0..24)
_NATURE_PRIOR_FLOOR = 0.02                                  # uniform mix so every nature stays liftable (no log 0)


def build_nature_mult() -> torch.Tensor:
    """``[N_NATURES, 5]`` the nature stat multiplier (0.9/1.0/1.1) for {atk,def,spa,spd,spe}, indexed by the
    nature ``num`` (0..24). The head marginalises ``E[mult] = P(nature) @ NATURE_MULT``; the op marginalises
    the nonlinear P(KO) over the top natures. GIGO-guarded: exactly 25 natures, each num in range."""
    raw = gen3_data.natures.raw()
    if len(raw) != N_NATURES:
        raise ValueError(f"build_nature_mult: expected {N_NATURES} natures, got {len(raw)}")
    mult = torch.ones(N_NATURES, N_SPREAD_STATS, dtype=torch.float32)
    for name, v in raw.items():
        num = int(v["num"])
        if not (0 <= num < N_NATURES):
            raise ValueError(f"build_nature_mult: nature {name} has out-of-range num {num}")
        for j, stat in enumerate(SPREAD_STAT_COLS):
            mult[num, j] = float(v.get(stat, 1.0))
    return mult


def build_species_nature_prior(n_species: int) -> torch.Tensor:
    """``[n_species, N_NATURES]`` per-species LOG-prior over natures (the prior-fusion base: the head adds a
    learned logit delta, softmax → posterior). From the Smogon usage spreads (`gen3_data.priors.spreads`):
    P(nature|species) ∝ Σ usage-weight, mixed with a small uniform floor so every nature stays liftable, then
    logged. A species with no usage data (and the unknown species 0) gets uniform log(1/25). Non-persistent."""
    logprior = torch.full((n_species, N_NATURES), 1.0 / N_NATURES, dtype=torch.float32).log()
    nat_raw = gen3_data.natures.raw()
    for sid in gen3_data.species.raw():
        snum = gen3_data.species.get(sid).num
        if not (0 <= snum < n_species):
            continue
        counts = torch.zeros(N_NATURES, dtype=torch.float32)
        for nature, _evs, w in gen3_data.priors.spreads(sid):
            nd = nat_raw.get(str(nature).lower())
            if nd is None:
                continue
            counts[int(nd["num"])] += float(w)
        tot = float(counts.sum())
        if tot <= 0.0:
            continue
        p = counts / tot
        p = (1.0 - _NATURE_PRIOR_FLOOR) * p + _NATURE_PRIOR_FLOOR / N_NATURES    # keep every nature liftable
        logprior[snum] = p.log()
    return logprior


def build_species_ev_prior(n_species: int) -> torch.Tensor:
    """``[n_species, 5]`` per-species usage-MEAN EV investment for {atk,def,spa,spd,spe} (the EV prior-fusion
    base; the head adds a learned delta). From the Smogon spreads' EV lists. No data → 0 EV (neutral). The
    head clamps the posterior EV to [0,252]. Non-persistent (data-derived)."""
    ev = torch.zeros(n_species, N_SPREAD_STATS, dtype=torch.float32)
    for sid in gen3_data.species.raw():
        snum = gen3_data.species.get(sid).num
        if not (0 <= snum < n_species):
            continue
        acc = torch.zeros(N_SPREAD_STATS, dtype=torch.float32)
        wsum = 0.0
        for _nature, evs, w in gen3_data.priors.spreads(sid):
            for j, stat in enumerate(SPREAD_STAT_COLS):
                acc[j] += float(w) * float(evs[_SPREAD_BASE_IDX[stat]])
            wsum += float(w)
        if wsum > 0.0:
            ev[snum] = acc / wsum
    return ev


def build_species_base_stats(n_species: int) -> torch.Tensor:
    """``[n_species, 5]`` the per-species BASE stat for {atk,def,spa,spd,spe} (NOT HP) — the SpreadBelief head
    needs it to compute the gen3 derived stat ``(2·base + 31 + EV/4 + 5)·mult``. Non-persistent."""
    base = torch.zeros(n_species, N_SPREAD_STATS, dtype=torch.float32)
    for sid in gen3_data.species.raw():
        sd = gen3_data.species.get(sid)
        if not (0 <= sd.num < n_species):
            continue
        for j, stat in enumerate(SPREAD_STAT_COLS):
            base[sd.num, j] = float(sd.base_stats.get(stat, 0))
    return base


def invert_nature_evs(derived, base, species_id=None):
    """Recover a ``(nature_num, [ev×5])`` generative decomposition that EXACTLY reproduces the gen3 DERIVED
    stats ``derived`` {atk,def,spa,spd,spe} for a mon with base stats ``base`` (same order), assuming IV 31 /
    L100. Used to build the privileged NATURE/EV supervision label from agent2's known ``mon.stats`` (gen3
    hides the opp's nature+EVs, so we INVERT the visible derived stats rather than need them in the obs).

    Returns ``None`` if no nature yields all-valid EVs (∈[0,252], multiple of 4, Σ≤510) — a GIGO guard (the
    slot is left unscored). The map is occasionally many-to-one (the ``×11//10`` / ``×9//10`` floor loses a few
    EV; the 5 all-neutral natures are degenerate), so among valid decompositions it prefers the one with the
    highest Smogon nature prior for ``species_id`` (the most plausible TRUE nature), then smallest num —
    deterministic and self-consistent (any returned pair reproduces ``derived`` by construction)."""
    nat_raw = gen3_data.natures.raw()
    weight = {}                                                          # nature usage hint for disambiguation
    if species_id is not None:
        for nature, _evs, w in gen3_data.priors.spreads(species_id):
            nd = nat_raw.get(str(nature).lower())
            if nd is not None:
                weight[int(nd["num"])] = weight.get(int(nd["num"]), 0.0) + float(w)
    candidates = []
    for _name, v in nat_raw.items():
        num = int(v["num"])
        evs, ok = [], True
        for j, stat in enumerate(SPREAD_STAT_COLS):
            m = float(v.get(stat, 1.0))
            D, b = int(round(float(derived[j]))), int(round(float(base[j])))
            found = next((ev for ev in range(0, 253, 4) if gen3_data.priors.gen3_stat(b, ev, m) == D), None)
            if found is None:
                ok = False
                break
            evs.append(found)
        if ok and sum(evs) <= 510:
            candidates.append((weight.get(num, 0.0), -num, num, evs))    # highest prior, then smallest num
    if not candidates:
        return None
    candidates.sort(reverse=True)
    _, _, num, evs = candidates[0]
    return num, evs


# gen3_opp_hp_type_belief_v1: the per-species Smogon Hidden-Power-TYPE usage prior. The DamageOperator's
# typed-HP candidate weight FLOOR (used when the obs `hp_probs` is still all-zero — it stays empty until
# the opp FIRES HP, the "opp HP reads immune" GIGO) AND the HPTypeBelief head's prior-fusion base (the
# learned head predicts a log-odds delta on top of this, mirroring MoveBelief's move-prior fusion).
def build_hp_type_prior(n_species: int) -> torch.Tensor:
    """``[n_species, 16]`` per-species P(Hidden Power type) over HIDDEN_POWER_TYPE_ORDER (the SAME 16-axis
    order the op's ``HP_TYPE_IDX`` / the obs ``hp_probs`` / ``belief_labels.HP_TYPE_NAMES`` use), from the
    Smogon HP-type usage prior (``gen3_data.priors.hidden_power_raw()``). Each row is normalized to sum 1; a
    species with no usage entry (and the unknown species num 0) gets a flat 1/16. Indexed by national-dex
    num (the move-belief / op / embedding axis). Non-persistent buffer (data-derived, recomputable)."""
    n_hp = len(HIDDEN_POWER_TYPE_ORDER)
    prior = torch.full((n_species, n_hp), 1.0 / n_hp, dtype=torch.float32)
    raw = gen3_data.priors.hidden_power_raw()
    names = [t.name.lower() for t in HIDDEN_POWER_TYPE_ORDER]
    for sid in gen3_data.species.raw():
        sd = gen3_data.species.get(sid)
        if not (0 <= sd.num < n_species):
            continue
        entry = raw.get(sid)
        if not entry:
            continue
        vec = torch.tensor([float(entry.get(name, 0.0)) for name in names], dtype=torch.float32)
        s = float(vec.sum())
        if s > 0.0:
            prior[sd.num] = vec / s     # else keep the flat 1/16 fallback
    return prior


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
    for sid in gen3_data.species.raw():
        sd = gen3_data.species.get(sid)
        if not (0 <= sd.num < n_species):
            continue
        for j, tname in enumerate(sd.types[:2]):
            types[sd.num, j] = _T2I.get(tname, 0)
    return types


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
    for sid in gen3_data.species.raw():
        sd = gen3_data.species.get(sid)
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
CHOICE_BAND_ITEM_NUM = int(gen3_data.items.get("choiceband").num)   # 220
CHOICE_BAND_PHYS_MULT = 1.5


def build_species_cb_prior(n_species: int) -> torch.Tensor:
    """``[n_species]`` P(holds Choice Band | species) from the Smogon item usage prior
    (`gen3_data.priors.items`) — Aerodactyl ≈0.76, Metagross ≈0.31, Snorlax ≈0.03. The op's PRIOR for an
    unrevealed opponent's CB belief (collapsed to 0/1 once the held/consumed item is revealed). Non-persistent
    buffer (recomputable from data/, zero params). A species with no usage data reads 0.0."""
    prior = torch.zeros(n_species, dtype=torch.float32)
    for sid in gen3_data.species.raw():
        snum = gen3_data.species.get(sid).num
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
        md = gen3_data.moves.get(mid)
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
# SSOT guard: every status id the gen3_mechanics ability-immunity rules name MUST map to a known category,
# so a future gen3_mechanics edit (e.g. a new blocked status) fails LOUDLY at import here rather than
# silently dropping that ability's block from SPECIES_STATUS_BLOCK_PRIOR / ABILITY_STATUS_BLOCK.
assert {s for ss in ABILITY_STATUS_IMMUNITY.values() for s in ss} <= set(_STATUS_CAT), \
    "gen3_mechanics.ABILITY_STATUS_IMMUNITY names a status not in damage_tables._STATUS_CAT — add it."


def build_status_landing(n_moves: int, n_species: int, n_abilities: int) -> Dict[str, torch.Tensor]:
    """The per-move / per-ability / per-species status-landing tables (all on the gen3-data num axes):

      MOVE_STATUS_CAT[n_moves]                 long — 0 (not a status move) else the _STATUS_CAT / LEECH_SEED_CAT it inflicts
      MOVE_INFLICTS_STATUS[n_moves]            1.0 if it is a dedicated status move (incl. Leech Seed)
      MOVE_IS_SLEEP[n_moves]                   1.0 if it inflicts sleep (the Sleep-Clause gate)
      MOVE_BLOCKED_IF_STATUSED[n_moves]        1.0 for a MAJOR status (can't double-apply); 0 for Leech Seed
      MOVE_STATUS_TYPE_IMMUNE[n_moves, N_TYPE_IDX]   1.0 where a DEFENDER type is immune to THIS move's status
      ABILITY_STATUS_BLOCK[n_abilities, N_STATUS_CAT]   1.0 if the (revealed) ability hard-blocks that category
      SPECIES_STATUS_BLOCK_PRIOR[n_species, N_STATUS_CAT]   P(species' ability blocks) — Smogon-prior marginal

    Type immunity is keyed by MOVE id (the gen3 rule): Thunder Wave→Ground, Toxic/Poison Gas/Poison Powder
    →Steel/Poison, Will-O-Wisp→Fire (Stun Spore/Glare paralysis has NO type immunity), + Leech Seed→Grass."""
    cat = torch.zeros(n_moves, dtype=torch.long)
    inflicts = torch.zeros(n_moves, dtype=torch.float32)
    is_sleep = torch.zeros(n_moves, dtype=torch.float32)
    blocked_if_statused = torch.zeros(n_moves, dtype=torch.float32)
    type_immune = torch.zeros(n_moves, N_TYPE_IDX, dtype=torch.float32)
    for mid in gen3_data.moves.raw():
        md = gen3_data.moves.get(mid)
        num = md.num
        if not (0 <= num < n_moves) or num == HIDDEN_POWER_NUM:
            continue
        if mid == _LEECH_SEED_ID:
            c = LEECH_SEED_CAT
        elif md.status_inflicted is not None:
            c = _STATUS_CAT.get(md.status_inflicted, 0)
        else:
            c = 0
        if c == 0:
            continue
        cat[num] = c
        inflicts[num] = 1.0
        is_sleep[num] = 1.0 if c == _SLP_CAT else 0.0
        blocked_if_statused[num] = 0.0 if c == LEECH_SEED_CAT else 1.0
        immune_types = ({PokemonType.GRASS} if c == LEECH_SEED_CAT
                        else set(STATUS_MOVE_IMMUNITY.get(mid, frozenset())))
        for pt in immune_types:
            ti = _T2I.get(pt.name)
            if ti is not None:
                type_immune[num, ti] = 1.0

    ability_block = torch.zeros(n_abilities, N_STATUS_CAT, dtype=torch.float32)
    for aid, statuses in ABILITY_STATUS_IMMUNITY.items():
        ad = gen3_data.abilities.get(aid)
        if ad is None or not (0 <= ad.num < n_abilities):
            continue
        for s in statuses:
            c = _STATUS_CAT.get(s)
            if c is not None:
                ability_block[ad.num, c] = 1.0

    # Marginalize the per-species Smogon ability prior over the same ABILITY_STATUS_IMMUNITY rule → the
    # P(this species' ability blocks status c) used when the opp ability is UNREVEALED (priors-then-confirm).
    species_block = torch.zeros(n_species, N_STATUS_CAT, dtype=torch.float32)
    for sid in gen3_data.species.raw():
        sd = gen3_data.species.get(sid)
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
        "MOVE_INFLICTS_STATUS": inflicts,
        "MOVE_IS_SLEEP": is_sleep,
        "MOVE_BLOCKED_IF_STATUSED": blocked_if_statused,
        "MOVE_STATUS_TYPE_IMMUNE": type_immune,
        "ABILITY_STATUS_BLOCK": ability_block,
        "SPECIES_STATUS_BLOCK_PRIOR": species_block,
    }


def build_move_type_idx(n_moves: int) -> torch.Tensor:
    """``[n_moves]`` TypeEncoder index of each move's canonical type (the `MoveLatentEncoder`'s type
    lookup for its context-free latent table). HP (num 237) stays idx 0 (unknown) — HP's type is
    per-candidate, resolved live elsewhere; the latent treats HP as one class. Mirrors the MOVE_TYPE_IDX
    built in build_damage_buffers (kept standalone so the encoder doesn't build the whole damage table)."""
    idx = torch.zeros(n_moves, dtype=torch.long)
    for mid in gen3_data.moves.raw():
        md = gen3_data.moves.get(mid)
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
        md = gen3_data.moves.get(mid)
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
                if md is not None:
                    bnum = _belief_num(move_id, md)      # typed Hidden Power → 237 (opp observed bare)
                    if 0 <= bnum < n_moves:
                        prob[snum, bnum] += float(p)     # sum collisions (e.g. typed Hidden Power → 237)
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
                if md is not None:
                    bnum = _belief_num(move_id, md)      # any HP (learnset carries bare 'hiddenpower') → 237
                    if 0 <= bnum < n_moves:
                        prob[snum, bnum] = floor
        # TRUE usage overrides the floor (an observed move is necessarily legal); HP variants sum into 237
        # (the opp is observed bare — see _belief_num).
        usage: Dict[int, float] = {}
        for move_id, p in gen3_data.priors.moves(sid).items():
            md = gen3_data.moves.get(move_id)
            if md is None:
                continue
            bnum = _belief_num(move_id, md)
            if not (0 <= bnum < n_moves):
                continue
            usage[bnum] = usage.get(bnum, 0.0) + float(p)
        for num, u in usage.items():
            if u > float(prob[snum, num]):
                prob[snum, num] = u                      # rare moves keep their real (small) rate
    prob = prob.clamp(eps, 1.0 - eps)
    return torch.logit(prob).to(torch.float32)           # log(p/(1-p)), the additive log-odds base rate
