"""Data-derived BELIEF-PRIOR lookup tensors — the prior-fusion bases the belief heads correct.

Split out of `damage_tables.py` (`gen3_belief_tables_split_v1`, 2026-09-06), which had grown to
1,433 lines holding two unrelated subjects. The rule the cut follows is *what the buffer is about*:

  * **here** — the priors a BELIEF HEAD fuses with: the opponent's spread (derived stats, and its
    generative nature/EV decomposition), the Hidden-Power TYPE distribution, and the held ITEM.
    Every one is a per-species Smogon usage distribution the learned head predicts a DELTA on top
    of, so its zero-init cold start reproduces the prior exactly.
  * **`damage_tables.py`** — the damage/type/stat buffers the `DamageOperator` physics reads.

Every function here is a pure, independent constructor: it takes the num-axis sizes, reads
`agents.gen3_data`, and returns a fresh tensor. There is no module state, no `DamageOperator`
coupling, and no import back to `damage_tables` — the edge runs strictly ONE way (`damage_tables`
imports from here, because `build_damage_buffers` registers `SPECIES_SPREAD_PRIOR` and
`NATURE_MULT` for the op). Keeping that direction is what makes the two modules a layering rather
than a cycle; do not add a `from .damage_tables import …` to this file.

Every tensor these build is registered `persistent=False` by its owning head — they are derived
from `data/` and recomputable, never a saved weight — so **none of them appears in `state_dict`**
and this relocation moves no `state_dict` key.

`damage_tables` re-exports every name below, so `from agents.model.damage_tables import
build_item_prior` (and the ~10 other historical spellings across `belief_heads`, the prober, the
tests) still resolves.
"""
from __future__ import annotations

from typing import cast, Dict, List, Optional, Sequence, Tuple

import torch

from agents import gen3_data
from agents.gen3_data.species import SpeciesData
from agents.training.hidden_power_tracker import HIDDEN_POWER_TYPE_ORDER


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
    (the same L100/IV31 formula the op uses for our revealed mons).

    Relocated from `damage_tables.py` (`gen3_belief_tables_split_v1`, 2026-09-06)."""
    prior = torch.zeros(n_species, N_SPREAD_STATS, 2, dtype=torch.float32)
    for sid in gen3_data.species.base_form_ids():
        sd = cast(SpeciesData, gen3_data.species.get(sid))
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
    the nonlinear P(KO) over the top natures. GIGO-guarded: exactly 25 natures, each num in range.

    Relocated from `damage_tables.py` (`gen3_belief_tables_split_v1`, 2026-09-06)."""
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
    logged. A species with no usage data (and the unknown species 0) gets uniform log(1/25). Non-persistent.

    Relocated from `damage_tables.py` (`gen3_belief_tables_split_v1`, 2026-09-06)."""
    logprior = torch.full((n_species, N_NATURES), 1.0 / N_NATURES, dtype=torch.float32).log()
    nat_raw = gen3_data.natures.raw()
    for sid in gen3_data.species.base_form_ids():
        snum = cast(SpeciesData, gen3_data.species.get(sid)).num
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
    head clamps the posterior EV to [0,252]. Non-persistent (data-derived).

    Relocated from `damage_tables.py` (`gen3_belief_tables_split_v1`, 2026-09-06)."""
    ev = torch.zeros(n_species, N_SPREAD_STATS, dtype=torch.float32)
    for sid in gen3_data.species.base_form_ids():
        snum = cast(SpeciesData, gen3_data.species.get(sid)).num
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
    needs it to compute the gen3 derived stat ``(2·base + 31 + EV/4 + 5)·mult``. Non-persistent.

    Relocated from `damage_tables.py` (`gen3_belief_tables_split_v1`, 2026-09-06)."""
    base = torch.zeros(n_species, N_SPREAD_STATS, dtype=torch.float32)
    for sid in gen3_data.species.base_form_ids():
        sd = cast(SpeciesData, gen3_data.species.get(sid))
        if not (0 <= sd.num < n_species):
            continue
        for j, stat in enumerate(SPREAD_STAT_COLS):
            base[sd.num, j] = float(sd.base_stats.get(stat, 0))
    return base


def invert_nature_evs(derived: Sequence[float], base: Sequence[float],
                      species_id: Optional[str] = None) -> Optional[Tuple[int, List[int]]]:
    """Recover a ``(nature_num, [ev×5])`` generative decomposition that EXACTLY reproduces the gen3 DERIVED
    stats ``derived`` {atk,def,spa,spd,spe} for a mon with base stats ``base`` (same order), assuming IV 31 /
    L100. Used to build the privileged NATURE/EV supervision label from agent2's known ``mon.stats`` (gen3
    hides the opp's nature+EVs, so we INVERT the visible derived stats rather than need them in the obs).

    Returns ``None`` if no nature yields all-valid EVs (∈[0,252], multiple of 4, Σ≤510) — a GIGO guard (the
    slot is left unscored). The map is occasionally many-to-one (the ``×11//10`` / ``×9//10`` floor loses a few
    EV; the 5 all-neutral natures are degenerate), so among valid decompositions it prefers the one with the
    highest Smogon nature prior for ``species_id`` (the most plausible TRUE nature), then smallest num —
    deterministic and self-consistent (any returned pair reproduces ``derived`` by construction).

    Relocated from `damage_tables.py` (`gen3_belief_tables_split_v1`, 2026-09-06)."""
    nat_raw = gen3_data.natures.raw()
    weight: Dict[int, float] = {}                                        # nature usage hint for disambiguation
    if species_id is not None:
        for nature, _evs, w in gen3_data.priors.spreads(species_id):
            nd = nat_raw.get(str(nature).lower())
            if nd is not None:
                weight[int(nd["num"])] = weight.get(int(nd["num"]), 0.0) + float(w)
    candidates: List[Tuple[float, int, int, List[int]]] = []
    for _name, v in nat_raw.items():
        num = int(v["num"])
        evs, ok = [], True                       # type: List[int], bool
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
    num (the move-belief / op / embedding axis). Non-persistent buffer (data-derived, recomputable).

    Relocated from `damage_tables.py` (`gen3_belief_tables_split_v1`, 2026-09-06)."""
    n_hp = len(HIDDEN_POWER_TYPE_ORDER)
    prior = torch.full((n_species, n_hp), 1.0 / n_hp, dtype=torch.float32)
    raw = gen3_data.priors.hidden_power_raw()
    names = [t.name.lower() for t in HIDDEN_POWER_TYPE_ORDER]
    for sid in gen3_data.species.base_form_ids():
        sd = cast(SpeciesData, gen3_data.species.get(sid))
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


def build_item_prior(n_species: int, n_items: int) -> torch.Tensor:
    """``[n_species, n_items]`` per-species P(item) over ITEM NUMS, from the Smogon item prior
    (``gen3_data.priors.items`` — sum-1 over observed items, `nothing` included at num 0's...
    NOTE `nothing` maps to item num 0, which doubles as the UNREVEALED sentinel in the obs; here
    it is a legitimate class (no item). Each row keeps a small floor on every num so an
    off-usage item stays liftable by evidence, then renormalizes; a species with no usage entry
    (and the unknown-species num 0) is uniform. Indexed by national-dex num (the belief/embedding
    axis). Non-persistent (data-derived, recomputable). GIGO guard: Blissey's Leftovers is ~100%
    of its usage — if it does not resolve dominant the item-num axis has drifted and the whole
    prior silently flattened (`gen3_item_belief_v1`).

    Relocated from `damage_tables.py` (`gen3_belief_tables_split_v1`, 2026-09-06)."""
    _FLOOR = 1e-5   # floor mass total ≈0.6% of a row: cold-start CB column within ~0.6% of SPECIES_CB_PRIOR
    prior = torch.full((n_species, n_items), 1.0 / n_items, dtype=torch.float32)
    for sid in gen3_data.species.base_form_ids():
        sd = cast(SpeciesData, gen3_data.species.get(sid))
        if not (0 <= sd.num < n_species):
            continue
        entry = gen3_data.priors.items(sid)
        if not entry:
            continue
        vec = torch.full((n_items,), _FLOOR, dtype=torch.float32)
        for item_id, p in entry.items():
            it = gen3_data.items.get(item_id)
            num = int(it.num) if it is not None else (0 if item_id == "nothing" else None)
            if num is not None and 0 <= num < n_items:
                vec[num] += float(p)
        prior[sd.num] = vec / float(vec.sum())
    bl = gen3_data.species.get("blissey")
    lo = gen3_data.items.get("leftovers")
    if (bl is None or lo is None or not (0 < bl.num < n_species)
            or float(prior[bl.num, int(lo.num)]) < 0.5):
        raise ValueError(
            "build_item_prior: Blissey's Leftovers did not resolve dominant — the item prior is "
            "empty/misaligned (item-num axis drift?). GIGO guard.")
    return prior
