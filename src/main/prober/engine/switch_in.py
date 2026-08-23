"""Switch-in OUTGOING damage: on a forced switch, what each candidate would DO to the opp active."""

from __future__ import annotations


from main.prober.engine.spread import _true_derived_spread
from main.prober.engine.util import _norm_species
from main.prober.engine.views import SwitchInOutgoingRow, SwitchInOutgoingView


# ---------------------------------------------------------------------------
# Switch-in outgoing damage (forced-switch panel — what each candidate would DO)
# ---------------------------------------------------------------------------
# On a forced switch the DamageOperator's OUTGOING block is all-zero (it prices the fainted
# active only), so the model picks a switch-in from INCOMING threat alone, with no estimate of
# what each candidate would then DO to the opp active. This CPU-side panel fills that view
# (prober-only, no model change): each ALIVE bench candidate → its best damaging move vs the opp
# active → [low–high %, →KO, ×mult, P(outspeed)], from the privileged true spreads (📋).


def _derived_hp(base: int, iv: int, ev: int) -> int:
    """Gen3 HP derived stat at level 100 (distinct from _derived_stat: +level+10, no nature)."""
    return 2 * int(base) + int(iv) + int(ev) // 4 + 110


def _hp_frac_from_str(hp: str) -> "float | None":
    """Board HP string ('31%' / '100%' / 'faint') → fraction in [0,1]; None if fainted/unknown."""
    s = (hp or "").strip().lower()
    if not s or s == "faint" or s.startswith("0%"):
        return None
    try:
        return max(0.0, min(1.0, float(s.rstrip("%")) / 100.0))
    except ValueError:
        return None


def _as_ptype(name: str):
    """A type string ('WATER') → poke-env PokemonType (species.types are UPPERCASE enum names)."""
    try:
        from poke_env.battle.pokemon_type import PokemonType
        return PokemonType[str(name).upper()]
    except (KeyError, AttributeError, ImportError):
        return None


def build_switch_in_outgoing(board, our_team_details, opp_team_details) -> "SwitchInOutgoingView | None":
    """Each ALIVE bench candidate's best damaging move's expected damage to the opp ACTIVE — the
    forced-switch panel. Model-free / privileged (true spreads). None when no reconstruction
    (no team_details), no opp active, or no candidate has a BP move."""
    if not our_team_details or not opp_team_details or board is None:
        return None
    from agents import gen3_data
    from agents.observation.incoming_damage import (
        gen3_damage_max, p_ko, p_outspeed, type_is_physical)
    from agents.gen3_mechanics import effective_multiplier_by_types

    opp_species = _norm_species(getattr(board.opp, "active_species", "") or "")
    if not opp_species:
        return None
    opp_detail = next((d for d in opp_team_details
                       if _norm_species(d.get("species", "")) == opp_species), None)
    opp_sp = gen3_data.species.get(opp_species)
    od = _true_derived_spread(opp_detail) if opp_detail is not None else None
    if opp_sp is None or od is None:
        return None
    opp_stats, _, _ = od                                   # (atk,def,spa,spd,spe)
    opp_def, opp_spd, opp_spe = opp_stats[1], opp_stats[3], opp_stats[4]
    ivs, evs = (opp_detail.get("ivs") or {}), (opp_detail.get("evs") or {})
    opp_max_hp = _derived_hp(opp_sp.base_stats.get("hp", 0), int(ivs.get("hp", 31)), int(evs.get("hp", 0)))
    opp_hp_frac = _hp_frac_from_str(getattr(board.opp, "active_hp", ""))
    if opp_max_hp <= 0 or opp_hp_frac is None:
        return None
    opp_remaining = max(1, int(round(opp_max_hp * opp_hp_frac)))
    opp_t = [t for t in (_as_ptype(x) for x in opp_sp.types) if t is not None]
    if not opp_t:
        return None
    opp_t1, opp_t2 = opp_t[0], (opp_t[1] if len(opp_t) > 1 else None)

    by_species = {_norm_species(d.get("species", "")): d for d in our_team_details}
    rows = []
    for cand in getattr(board.ours, "bench", ()):          # the switch-in candidates the board lists
        if _hp_frac_from_str(getattr(cand, "hp", "")) is None:
            continue                                        # fainted / unavailable
        sid = _norm_species(getattr(cand, "species", ""))
        d, sp = by_species.get(sid), gen3_data.species.get(sid)
        cd = _true_derived_spread(d) if d is not None else None
        if sp is None or cd is None:
            continue
        c_stats, _, _ = cd
        our_atk, our_spa, our_spe = c_stats[0], c_stats[2], c_stats[4]
        our_types = {t for t in (_as_ptype(x) for x in sp.types) if t is not None}
        best = None
        for mid in (d.get("moves") or ()):
            mv = gen3_data.moves.get(mid)
            if mv is None or int(getattr(mv, "base_power", 0)) <= 0:
                continue                                    # status / fixed-damage (v1: BP moves only)
            if mid in ("explosion", "selfdestruct"):
                continue                                    # KOs but self-KOs — not a switch-in's sustainable offense
            phys = type_is_physical(mv.type)
            eff = effective_multiplier_by_types(mv.type, opp_t1, opp_t2)
            dmax = gen3_damage_max(int(mv.base_power), int(our_atk if phys else our_spa),
                                   int(opp_def if phys else opp_spd),
                                   stab=(mv.type in our_types), type_eff=eff)
            high = 100.0 * dmax / opp_max_hp
            pko = p_ko(dmax, opp_remaining)
            if best is None or (pko, high) > (best[3], best[2]):
                best = (mid, eff, high, pko)
        if best is None:
            continue
        mid, eff, high, pko = best
        rows.append(SwitchInOutgoingRow(
            species=getattr(cand, "species", ""), hp=getattr(cand, "hp", ""), move=mid,
            low=high * 0.85, high=high, pko=pko, type_mult=eff,
            outspeed=p_outspeed(int(our_spe), [(int(opp_spe), 1.0)])))
    if not rows:
        return None
    rows.sort(key=lambda r: (r.pko, r.high), reverse=True)
    return SwitchInOutgoingView(opp_species=getattr(board.opp, "active_species", ""),
                                opp_hp=getattr(board.opp, "active_hp", ""), rows=tuple(rows))
