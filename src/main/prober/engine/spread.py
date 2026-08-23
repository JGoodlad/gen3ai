"""Believed vs TRUE derived spreads — the DamageOperator's stat input, checked against truth."""

from __future__ import annotations

import numpy as np

from main.prober.engine.util import _norm_species
from main.prober.engine.views import SpreadBeliefView, SpreadSlotBelief, SpreadStatRow


# ---------------------------------------------------------------------------
# Spread belief vs truth (the DamageOperator's stat input — gen3_unified_spread_belief_v1)
# ---------------------------------------------------------------------------

_SPREAD_COLS = ("atk", "def", "spa", "spd", "spe")   # the order of last_spread_belief's 5 columns
_SPREAD_BASE_KEY = {"atk": "atk", "def": "def", "spa": "spa", "spd": "spd", "spe": "spe"}
_SPREAD_PRIOR_CACHE: "dict[str, tuple[float, ...]] | None" = None


def _derived_stat(base: int, iv: int, ev: int, mult: float) -> int:
    """The gen3 non-HP derived stat at level 100 with the mon's ACTUAL IV (poke-env/team_details give the
    real IV — e.g. a Hidden-Power mon isn't IV31 everywhere), EV, and nature multiplier. Exact integer
    math, mirroring `gen3_data.priors.gen3_stat` but parameterized on IV (that helper hardcodes IV31)."""
    pre = 2 * int(base) + int(iv) + int(ev) // 4 + 5
    if mult > 1.0:
        return pre * 11 // 10
    if mult < 1.0:
        return pre * 9 // 10
    return pre


def _spread_prior_means(species_id: str) -> "tuple[float, ...] | None":
    """Usage-weighted mean DERIVED stat per {atk,def,spa,spd,spe} for ``species_id`` — the Smogon spread
    PRIOR the SpreadBelief head corrects (the same quantity `damage_tables.build_opp_spread_prior`'s mean
    column holds). `None` when the species has no spread data. Cached per species (cheap, data-only)."""
    from agents import gen3_data
    sp = gen3_data.species.get(species_id)
    if sp is None:
        return None
    spreads = gen3_data.priors.spreads(species_id)
    if not spreads:
        return None
    ev_idx = {"atk": 1, "def": 2, "spa": 3, "spd": 4, "spe": 5}   # index into [hp,atk,def,spa,spd,spe]
    out = []
    for stat in _SPREAD_COLS:
        base = int(sp.base_stats.get(stat, 0))
        m1 = wsum = 0.0
        for nature, evs, w in spreads:
            nd = gen3_data.natures.get(str(nature).lower())
            mult = nd.multipliers.get(stat, 1.0) if nd is not None else 1.0
            m1 += float(w) * float(gen3_data.priors.gen3_stat(base, int(evs[ev_idx[stat]]), mult))
            wsum += float(w)
        out.append(m1 / wsum if wsum > 0 else float(gen3_data.priors.gen3_stat(base, 0, 1.0)))
    return tuple(out)


def _true_derived_spread(detail: dict) -> "tuple[tuple[float, ...], str, str] | None":
    """From one `team_details()` entry → (the 5 true DERIVED stats {atk,def,spa,spd,spe}, nature, ev_note).
    Uses the mon's REAL base/IV/EV/nature (privileged). `None` if the species is unknown to the dex."""
    from agents import gen3_data
    sp = gen3_data.species.get(detail.get("species", ""))
    if sp is None:
        return None
    evs = detail.get("evs") or {}
    ivs = detail.get("ivs") or {}
    nature = str(detail.get("nature", "") or "").lower()
    nd = gen3_data.natures.get(nature)
    vals = []
    for stat in _SPREAD_COLS:
        base = int(sp.base_stats.get(stat, 0))
        iv = int(ivs.get(stat, 31))
        ev = int(evs.get(stat, 0))
        mult = nd.multipliers.get(stat, 1.0) if nd is not None else 1.0
        vals.append(float(_derived_stat(base, iv, ev, mult)))
    ev_note = "/".join(f"{s}{int(evs[s])}" for s in ("hp",) + _SPREAD_COLS
                       if int(evs.get(s, 0)) > 0) or "0 EVs"
    return tuple(vals), nature, ev_note


def build_spread_belief(raw, opp_team_details, top_revealed_only: bool = True) -> "SpreadBeliefView | None":
    """Match the model's believed opp spread (`ProbeModel.spread_belief_view` raw) to the TRUE mons from
    `reconstruction.json`'s `team_details()` and compare the believed vs true DERIVED stats per REVEALED opp
    slot (the head predicts spreads for seen mons — species known, EVs unknown — so revealed slots are the
    meaningful ones; hidden mons have no spread prediction). Match is by SPECIES id (exact — a revealed mon's
    species is known + unique). `opp_team_details` is the list of `{species, evs, ivs, nature, …}` dicts (or
    `None`/`()` → believed-only, no truth columns). Returns `None` when the run trained `--spread-belief` off
    (raw is `None`) or no revealed slot has a believed spread."""
    if not raw:
        return None
    spread = np.asarray(raw.get("spread"), dtype=np.float64)          # [6, 5]
    bmask = raw.get("believed_mask")                                  # [6] bool (True = HIDDEN) or None
    opp_species = list(raw.get("opp_species") or [])
    # Index the privileged truth by normalized species id (revealed species are unique under the clause).
    truth_by_species: "dict[str, dict]" = {}
    for d in (opp_team_details or ()):
        sid = _norm_species(d.get("species", ""))
        if sid:
            truth_by_species[sid] = d

    slots, abs_errs = [], []
    for i in range(min(6, spread.shape[0])):
        sid = (opp_species[i] if i < len(opp_species) else "").strip()
        hidden = bool(bmask[i]) if (bmask is not None and i < len(bmask)) else (not sid)
        if top_revealed_only and (hidden or not sid):
            continue                                                  # hidden slot → no spread prediction
        believed = spread[i]                                          # [5]
        prior = _spread_prior_means(sid)
        truth = truth_by_species.get(_norm_species(sid))
        td = _true_derived_spread(truth) if truth is not None else None
        true_vals, nature, ev_note = (td if td is not None else (None, "", ""))
        rows = []
        for j, stat in enumerate(_SPREAD_COLS):
            tv = float(true_vals[j]) if true_vals is not None else None
            pv = float(prior[j]) if prior is not None else None
            rows.append(SpreadStatRow(stat=stat, believed=float(believed[j]), true=tv, prior=pv))
            if tv is not None:
                abs_errs.append(abs(float(believed[j]) - tv))
        slots.append(SpreadSlotBelief(slot=i, species=sid, rows=tuple(rows),
                                      nature=nature, ev_note=ev_note, matched=td is not None))
    if not slots:
        return None
    mae = (sum(abs_errs) / len(abs_errs)) if abs_errs else None
    return SpreadBeliefView(slots=tuple(slots), n_slots=len(slots), mean_abs_err=mae)
