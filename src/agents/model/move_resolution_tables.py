"""gen3_move_resolution_v1 — the move-resolution family's TABLES: per-move-num kinds and flags, per-species /
per-ability immunity marginals. Built ONCE at construction (never in the forward), from the data facade and the
source-verified rules of `move_resolution_rules`; `move_resolution` registers them as non-persistent buffers.
Kept out of `move_resolution.py` so the forward module's discrete-op inventory (`selection_sites`) holds only
what the forward runs."""
from __future__ import annotations

from typing import Dict, Tuple, cast

import torch

from agents.model.move_resolution_rules import (
    ABILITY_INNER_FOCUS, ABILITY_OWN_TEMPO, ABILITY_SLEEP_BLOCK, ABILITY_SOUNDPROOF, BYPASSSUB, DEFROST,
    FAILENCORE, FOE_NO_PROTECT, FOE_TARGETS, KINDS, REFLECTABLE, SEAT_KINDS, SOUND, STATUS_TYPE_IMMUNITY)

KIND_NAMES: Tuple[str, ...] = tuple(KINDS) + ("curse",)
KIND_IDX: Dict[str, int] = {n: i for i, n in enumerate(KIND_NAMES)}
SEAT_KIND_NAMES: Tuple[str, ...] = tuple(SEAT_KINDS)
SEAT_KIND_IDX: Dict[str, int] = {n: i for i, n in enumerate(SEAT_KIND_NAMES)}
#: Per-move-num flag columns.
FLAG_NAMES: Tuple[str, ...] = ("foe", "protectable", "bypasssub", "sound", "reflectable", "defrost",
                               "failencore", "status_type_imm", "status_cat", "damaging", "is_hp", "lowers_foe",
                               "cures_self")
FLAG_IDX: Dict[str, int] = {n: i for i, n in enumerate(FLAG_NAMES)}
#: Named abilities read per mon (revealed exact; the Smogon species prior otherwise).
NAMED_ABILITIES: Tuple[str, ...] = ((ABILITY_SOUNDPROOF, ABILITY_OWN_TEMPO, ABILITY_INNER_FOCUS) + ABILITY_SLEEP_BLOCK
                                     + ("naturalcure", "intimidate"))
#: Condition one-hot columns (`[None, BRN, PAR, SLP, FRZ, PSN, TOX]`).
C_BRN, C_PAR, C_SLP, C_FRZ, C_PSN, C_TOX = 1, 2, 3, 4, 5, 6
#: Screen columns (`global_env._SCREEN_CONDITIONS` × [ours, opp]).
S_REFLECT_OURS, S_LS_OURS, S_SG_OURS, S_SG_OPP, S_MIST_OURS = 0, 2, 4, 5, 6
#: Weather one-hot columns (`global_env._WEATHER_IDX`).
WEATHER_COL = {"sunnyday": 1, "raindance": 2, "sandstorm": 3, "hail": 4}

def build_move_tables(n_moves: int) -> Dict[str, torch.Tensor]:
    """``KIND [n_moves, n_kinds]``, ``FLAG [n_moves, n_flags]``, ``SEAT_KIND [n_moves, n_seat_kinds]`` from the data
    facade + `move_resolution_rules`. Every declared id must resolve (a missing one raises: a gate that never fires
    reads exactly like a null result)."""
    from agents import gen3_data
    from agents.model.dex_ids import HIDDEN_POWER_NUM, _hp_typed_nums

    def num(mid: str) -> int:
        md = gen3_data.moves.get(mid)
        if md is None or not (0 <= int(md.num) < n_moves):
            raise ValueError(f"move_resolution: move {mid!r} does not resolve in gen3_data.moves")
        return int(md.num)

    kind = torch.zeros(n_moves, len(KIND_NAMES))
    for k, ids in KINDS.items():
        for mid in ids:
            kind[num(mid), KIND_IDX[k]] = 1.0
    kind[num("curse"), KIND_IDX["curse"]] = 1.0
    seat = torch.zeros(n_moves, len(SEAT_KIND_NAMES))
    for k, ids in SEAT_KINDS.items():
        for mid in ids:
            seat[num(mid), SEAT_KIND_IDX[k]] = 1.0
    flag = torch.zeros(n_moves, len(FLAG_NAMES))
    for mid, rec in gen3_data.moves.raw().items():
        md = gen3_data.moves.get(mid)
        if md is None or not (0 <= int(md.num) < n_moves):
            continue
        n = int(md.num)
        foe = (md.target or rec.get("target")) in FOE_TARGETS
        f = flag[n]
        f[FLAG_IDX["foe"]] = float(foe)
        f[FLAG_IDX["protectable"]] = float(foe and mid not in FOE_NO_PROTECT)
        f[FLAG_IDX["bypasssub"]] = float(mid in BYPASSSUB)
        f[FLAG_IDX["sound"]] = float(mid in SOUND)
        f[FLAG_IDX["reflectable"]] = float(mid in REFLECTABLE)
        f[FLAG_IDX["defrost"]] = float(mid in DEFROST)
        f[FLAG_IDX["failencore"]] = float(mid in FAILENCORE)
        f[FLAG_IDX["status_type_imm"]] = float(mid in STATUS_TYPE_IMMUNITY)
        f[FLAG_IDX["status_cat"]] = float(str(rec.get("category")) == "Status")
        # a HIT: any non-Status move (so Seismic Toss / Night Shade / Super Fang count — they deal damage)
        f[FLAG_IDX["damaging"]] = float(str(rec.get("category")) != "Status")
        # a move that can LOWER the user's stats before our move (Haze resets them; a foe-targeting status move
        # with a negative `boosts` entry lowers them), which lifts a +6 boost cap that would otherwise fail ours
        # a move that cures its USER's major status (Refresh, Rest, Heal Bell, Aromatherapy) — a faster one lifts
        # the "already statused" block on our status move
        f[FLAG_IDX["cures_self"]] = float(bool(md.cures_self_status or md.cures_team_status) or mid == "rest")
        _b = rec.get("boosts") or {}
        f[FLAG_IDX["lowers_foe"]] = float(mid == "haze" or (str(rec.get("category")) == "Status" and foe
                                                             and any(float(v) < 0 for v in _b.values())))
    for n in (HIDDEN_POWER_NUM,) + tuple(_hp_typed_nums()):
        flag[n, FLAG_IDX["is_hp"]] = 1.0
    for mid in set().union(*(set(v) for v in (BYPASSSUB, SOUND, REFLECTABLE, DEFROST, FAILENCORE,
                                              STATUS_TYPE_IMMUNITY, FOE_NO_PROTECT))):
        if mid not in ("mirrormove", "struggle"):        # absent from the gen-3 vocabulary is fine for these
            num(mid)
    return {"KIND": kind, "FLAG": flag, "SEAT_KIND": seat}


def build_species_tables(chart: torch.Tensor, ability_damage_mult: torch.Tensor,
                         species_types: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Per-species / per-ability immunity tables over the 19-wide attacking-type axis:

    ``SPECIES_CHART0 [S,19]``  1.0 where the species' TYPES make it immune (the chart product is exactly 0);
    ``SPECIES_ABL_IMM [S,19]`` P(its ability makes it immune) — the Smogon ability prior over `ABILITY_IMM`;
    ``SPECIES_P_IMM [S,19]``   P(immune at all) = 1 − (1 − chart0)(1 − abl);
    ``SPECIES_HAS_TYPE [S,19]`` 1.0 at its two types;
    ``ABILITY_IMM [A,19]``     1.0 where an ability's damage multiplier is exactly 0 (Levitate, Flash Fire, …);
    ``ABILITY_NAMED [A,n]`` / ``SPECIES_NAMED_PRIOR [S,n]`` the named abilities' indicator / Smogon prior."""
    from agents import gen3_data
    from agents.gen3_data.species import SpeciesData
    n_species = species_types.shape[0]
    n_abil, n_type = ability_damage_mult.shape
    t1, t2 = species_types[:, 0].long(), species_types[:, 1].long()
    chart0 = ((chart[t1] * chart[t2]) == 0).float()                                   # [S,19]
    abl_imm = (ability_damage_mult == 0).float()                                      # [A,19]
    has_type = torch.zeros(n_species, n_type)
    has_type.scatter_(1, t1[:, None], 1.0)
    has_type.scatter_(1, t2[:, None], 1.0)
    has_type[:, 0] = 0.0                                                              # 0 = no second type
    named = torch.zeros(n_abil, len(NAMED_ABILITIES))
    for i, aid in enumerate(NAMED_ABILITIES):
        ad = gen3_data.abilities.get(aid)
        if ad is None or not (0 <= ad.num < n_abil):
            raise ValueError(f"move_resolution: ability {aid!r} does not resolve in gen3_data.abilities")
        named[ad.num, i] = 1.0
    sp_abl = torch.zeros(n_species, n_type)
    sp_named = torch.zeros(n_species, len(NAMED_ABILITIES))
    for sid in gen3_data.species.base_form_ids():
        sd = cast(SpeciesData, gen3_data.species.get(sid))
        if not (0 <= sd.num < n_species):
            continue
        for aid, p in (gen3_data.priors.ability(sid) or {}).items():
            ad = gen3_data.abilities.get(aid)
            if ad is None or not (0 <= ad.num < n_abil):
                continue
            sp_abl[sd.num] += float(p) * abl_imm[ad.num]
            sp_named[sd.num] += float(p) * named[ad.num]
    sp_abl = sp_abl.clamp(0.0, 1.0)
    sp_named = sp_named.clamp(0.0, 1.0)
    p_imm = 1.0 - (1.0 - chart0) * (1.0 - sp_abl)
    return {"SPECIES_CHART0": chart0, "SPECIES_ABL_IMM": sp_abl, "SPECIES_P_IMM": p_imm,
            "SPECIES_HAS_TYPE": has_type, "ABILITY_IMM": abl_imm, "ABILITY_NAMED": named,
            "SPECIES_NAMED_PRIOR": sp_named}


