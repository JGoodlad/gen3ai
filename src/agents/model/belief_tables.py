"""Data-derived BELIEF-PRIOR lookup tensors — the prior-fusion bases the belief heads correct.

Split out of `damage_tables.py` in two rounds (`gen3_belief_tables_split_v1` then
`gen3_dex_ids_split_v1`, both 2026-09-06), which had grown to 1,433 lines holding unrelated
subjects. The rule the cut follows is *what the buffer is about*:

  * **here** — the priors a BELIEF HEAD fuses with: the opponent's spread (derived stats, and its
    generative nature/EV decomposition), the Hidden-Power TYPE distribution, the held ITEM, the
    per-species MOVE prior (the move-belief's legality-gated base rate) and the team-composition
    SPECIES prior (`t0_species` / `BeliefHead`'s naive-Bayes marginal + co-occurrence lift).
    Every one is a per-species Smogon usage distribution the learned head predicts a DELTA on top
    of, so its zero-init cold start reproduces the prior exactly.
  * **`damage_tables.py`** — the damage/type/stat buffers the `DamageOperator` physics reads.
  * **`dex_ids.py`** — the dex-IDENTITY facts BOTH halves key on (the Hidden-Power nums and their
    id→num fold, the species usage share), so neither has to reach into the other for them.

Every function here is a pure, independent constructor: it takes the num-axis sizes, reads
`agents.gen3_data` (and `dex_ids`), and returns a fresh tensor. There is no module state and no
`DamageOperator` coupling. The layering runs strictly ONE way —

    `damage_tables` → `belief_tables` → `dex_ids`,  and  `damage_tables` → `dex_ids`

— with `damage_tables` importing from here because `build_damage_buffers` registers
`SPECIES_SPREAD_PRIOR` and `NATURE_MULT` for the op. Keeping that direction is what makes the three
modules a layering rather than a cycle; do not add a `from .damage_tables import …` to this file.

Every tensor these build is registered `persistent=False` by its owning head — they are derived
from `data/` and recomputable, never a saved weight — so **none of them appears in `state_dict`**
and this relocation moves no `state_dict` key.

`damage_tables` re-exports every name below, so `from agents.model.damage_tables import
build_item_prior` / `build_move_prior_logits` / `SPECIES_CLAUSE_LOGIT` (and the ~20 other historical
spellings across `belief_heads`, `t0_species`, `extractor_build`, `snapshot`, `main.train.config`,
the prober and nine test modules) still resolves.
"""
from __future__ import annotations

import math
from typing import cast, Dict, List, Optional, Sequence, Tuple

import torch

from agents import gen3_data
from agents.gen3_data.species import SpeciesData
from agents.model.dex_ids import _belief_num, _hp_typed_nums, build_species_usage_prior
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


# Floor probability for a move a species CAN learn but is ~never seen to run (keeps an unseen-but-legal
# move POSSIBLE — never logit(-inf) — so in-battle evidence can still lift it). Also the value for a
# species with no known movepool (num 0 / no learnset entry), where there is nothing to prune.
_PRIOR_FLOOR = 0.02

# Probability assigned to the IMPOSSIBLE — a move the species physically cannot learn. Small enough to
# be ~0 in the belief, finite so `torch.logit` never produces -inf. logit(1e-6) = -13.8155, vs
# logit(_PRIOR_FLOOR=0.02) = -3.8918 — a 9.92-nat gap, so "illegal" and "legal but unobserved" are
# materially different states of the prior rather than the same number.
_ILLEGAL_PROB = 1e-6

# The legal-unobserved floor must sit MATERIALLY above `_ILLEGAL_PROB`, or the legality gate collapses:
# a floor of 0.0 (or anything <= _ILLEGAL_PROB) gets clamped straight back up to _ILLEGAL_PROB and every
# legal-but-unobserved move becomes indistinguishable from an impossible one. That is the silent-GIGO
# failure this bound exists to make loud. 1e-3 (logit = -6.907) still leaves a ~6.9-nat separation.
_MIN_PRIOR_FLOOR = 1e-3


def sanitize_historical_move_floor(kwargs: dict) -> dict:
    """Make a PRE-v65 config constructible, in place, without editing the config on disk.

    Every run before `gen3_unconditional_move_legality_v1` recorded ``move_candidate_floor: 0.0``,
    because that value used to double as the legality on/off SWITCH rather than name a probability.
    v65 gave the floor a validated range, so those configs now raise — which is correct for a
    training RESUME (a silently-changed prior is exactly what the version gate exists to catch) but
    wrong for the OFFLINE tooling that instantiates an extractor purely to read its structure:
    `delivery_graph`, the architecture viewer, and `extractor_compiles_test` all build from
    the committed `designs/production_config.json`.

    That file is a VERBATIM copy of a real run and must keep its 0.0 — editing it to satisfy a
    builder would falsify the historical record it exists to preserve, and would quietly break the
    reproducibility claim `ARCHITECTURE.md` makes about it. The prior floor changes no node, edge or
    graph shape, so the safe place to reconcile the two is at the point of CONSTRUCTION, once, here
    — rather than in `_migrate_config`, which would let a pre-v65 checkpoint resume by silently
    adopting a different prior.

    Relocated from `damage_tables.py` (`gen3_dex_ids_split_v1`, 2026-09-06).
    """
    if float(kwargs.get("move_candidate_floor", 0.0)) < _MIN_PRIOR_FLOOR:
        kwargs["move_candidate_floor"] = _PRIOR_FLOOR
    return kwargs


def build_move_prior_logits(n_species: int, n_moves: int, floor: float = _PRIOR_FLOOR) -> torch.Tensor:
    """``[n_species, n_moves]`` LOG-ODDS of the Smogon move-frequency prior, indexed by national-dex
    ``num`` on BOTH axes — the base rate ``P(move in set)`` for a species, ready to fuse additively into
    the move-belief logits (``posterior_logit = head_delta + prior_logit``).

    Sources `gen3_data.priors.moves(species)` -> ``{move_id: P(in set)}`` (un-normalized; a set runs
    ~4 moves). Probabilities for move_ids that collapse to one ``num`` are SUMMED (Hidden Power: all
    typed variants share num 237, and a mon runs at most one HP type, so ``P(has HP) = Σ typed usage``).

    **LEGALITY IS UNCONDITIONAL** — it is a correctness property, not a feature, and there is no flag to
    turn it off. A move a species physically **cannot learn** must carry ~zero belief mass; anything else
    invents phantom threats ("a special attacker might have Explosion") out of a flat floor.

    The rule, per ``(species, move)`` cell:

    - **Illegal** (not in the species' learnset) → ``logit(_ILLEGAL_PROB)`` ≈ 0 probability. This is the
      only thing pruned: the IMPOSSIBLE.
    - **Legal, with recorded usage** → its **true Smogon usage**, untouched. A rare tech stays
      rare-but-present (naturally negligible in the op's hard-max, yet liftable by the learned head, and
      pinned certain the moment it's revealed) — NOT floored up to ``floor`` and NOT pruned. **No rarity
      cap**: a surprise move a mon legitimately runs is never zeroed out of the belief (an earlier
      ``<2%`` prune did that and crippled surprise-move anticipation).
    - **Legal, absent from the usage data** → the small ``floor`` base, so in-battle evidence can still
      surface it.
    - **No learnset at all** (hidden / unknown species, num 0) → the flat ``floor`` everywhere. Nothing
      is known about the movepool, so there is nothing to prune; marginalising the learnset over a
      species belief is a later extension.

    Because every move with recorded usage is necessarily legal, the legality mask only ever bites the
    ABSENT cells. Hidden Power's typed usages sum into ``num`` 237 (legal iff the bare ``'hiddenpower'``
    is in the learnset).

    ``floor`` is the LEGAL-UNOBSERVED base only — it is not an on/off switch. It must be
    ``>= _MIN_PRIOR_FLOOR``; see that constant for why a 0.0 floor is a hard error rather than a silent
    collapse into "everything is impossible".

    Returned as a plain float32 tensor for `MoveBelief` to register as a NON-persistent buffer (pure
    data-derived physics, recomputable — never a saved weight).

    Relocated from `damage_tables.py` (`gen3_dex_ids_split_v1`, 2026-09-06)."""
    eps = _ILLEGAL_PROB
    if not (_MIN_PRIOR_FLOOR <= float(floor) < 1.0):
        # Fail LOUD. A floor at/below _ILLEGAL_PROB makes legal-unobserved == illegal (the gate becomes a
        # no-op in the wrong direction), and a floor of exactly 0.0 would additionally be logit(0) = -inf
        # on any code path that clamps from below — a NaN source, not a configuration.
        raise ValueError(
            f"build_move_prior_logits: floor={floor!r} is out of range. The move-prior floor is the "
            f"LEGAL-BUT-UNOBSERVED base probability and must satisfy "
            f"{_MIN_PRIOR_FLOOR} <= floor < 1.0 (default {_PRIOR_FLOOR}).\n"
            f"A floor <= {_ILLEGAL_PROB} would be indistinguishable from the ILLEGAL value, collapsing "
            f"the legality distinction; a floor of 0.0 is additionally logit(0) = -inf. "
            f"Pass --move-candidate-floor {_PRIOR_FLOOR} (or any value in range)."
        )

    # Illegal → eps (impossible); legal-observed → TRUE usage; legal-unobserved → floor.
    prob = torch.full((n_species, n_moves), eps, dtype=torch.float64)   # default = impossible
    # Rows this build never touches are NOT "a species that can learn nothing" — they are rows about
    # which nothing is known: national-dex num 0 (the UNKNOWN-SPECIES sentinel an unrevealed opponent
    # slot carries, and `MoveBelief.move_logits` indexes `move_prior_logits[opp_species_ids]` directly
    # with it) and any gap in the num range. Leaving them at the "impossible" default would tell the
    # model an unseen opponent has NO moves at all — strictly worse than the flat floor, and a claim
    # the data never made. They are flattened to `floor` below (same rule as a species whose learnset
    # is missing: no movepool known → nothing to prune).
    covered = torch.zeros(n_species, dtype=torch.bool)
    for sid in gen3_data.species.base_form_ids():
        sd = cast(SpeciesData, gen3_data.species.get(sid))
        snum = sd.num
        if not (0 <= snum < n_species):
            continue
        covered[snum] = True
        legal = gen3_data.learnset.get_legal_moves(sid)
        if legal is None:
            prob[snum, :] = floor                        # unknown movepool → flat floor (nothing to prune)
        else:
            for move_id in legal:                        # every LEGAL move → a small liftable base
                md = gen3_data.moves.get(move_id)
                if md is not None:
                    bnum = _belief_num(move_id, md)      # any HP (learnset carries bare 'hiddenpower') → 237
                    if 0 <= bnum < n_moves:
                        prob[snum, bnum] = floor
                    # gen3_typed_hp_belief_v1 — TYPED-HP LEGALITY. `gen3_learnset.json` carries only the
                    # bare `hiddenpower` (the type is an IV choice, not a learnset entry), so the 16 typed
                    # nums 355-370 fell through to the `eps` "impossible" default for EVERY species — the
                    # gate declared HP-Ice unlearnable by anything. Harmless only because the composition
                    # overwrites those cells; wrong data in a tensor is exactly the GIGO shape we don't
                    # leave lying around. A typed HP is legal iff the bare one is.
                    if move_id == "hiddenpower":
                        for tnum in _hp_typed_nums():
                            if 0 <= tnum < n_moves:
                                prob[snum, tnum] = floor
        # TRUE usage overrides the floor (an observed move is necessarily legal). HP usage sums into the
        # 237 PRESENCE channel (see `_belief_num`) AND is written per-type at 355-370, so the typed cells
        # carry their own real rate and are independently meaningful under inspection.
        usage: Dict[int, float] = {}
        for move_id, p in gen3_data.priors.moves(sid).items():
            md = gen3_data.moves.get(move_id)
            if md is None:
                continue
            bnum = _belief_num(move_id, md)
            if 0 <= bnum < n_moves:
                usage[bnum] = usage.get(bnum, 0.0) + float(p)
            if move_id.startswith("hiddenpower") and 0 <= md.num < n_moves and md.num != bnum:
                usage[md.num] = usage.get(md.num, 0.0) + float(p)   # the typed cell's own rate
        for num, u in usage.items():
            if u > float(prob[snum, num]):
                prob[snum, num] = u                      # rare moves keep their real (small) rate
    prob[~covered, :] = floor                            # unknown species (num 0) / dex gaps → flat floor
    prob = prob.clamp(eps, 1.0 - eps)
    return torch.logit(prob).to(torch.float32)           # log(p/(1-p)), the additive log-odds base rate


# ── gen3_species_prior_fusion_v1 (v68): the TEAM-COMPOSITION species prior ────────────────────────
#
# The base rate a species occupies a HIDDEN opponent slot, and how much each ALREADY-REVEALED
# teammate moves it. Sibling of `build_move_prior_logits` above — same job (a data-derived base rate
# the learned head becomes a DELTA on top of), same num axis, same "finite floors, never -inf"
# discipline; the only structural difference is that this prior is CONDITIONAL on the rest of the
# opponent's team, so it ships as TWO tensors the forward combines on-GPU rather than one lookup.

# A species absent from the training team pool is UNOBSERVED, not impossible — the exact distinction
# `_PRIOR_FLOOR` draws for a legal-but-unseen move. A small liftable base (log = -9.21), so an
# off-pool opponent (a ladder / random-battle team) is improbable rather than unrepresentable.
_SPECIES_PRIOR_FLOOR = 1e-4

# SPECIES CLAUSE is a RULE, not a frequency: a species already revealed on the opponent's team
# cannot ALSO be sitting in a hidden slot. This is the species-side `_ILLEGAL_PROB` — ~0 (log =
# -13.82), finite so it never poisons the gradient of the delta the head learns on top.
_SPECIES_CLAUSE_PROB = 1e-6
SPECIES_CLAUSE_LOGIT = math.log(_SPECIES_CLAUSE_PROB)


# Bound on a single teammate's log-lift evidence (the SMOGON source has no shrinkage step, so
# this clamp is what keeps one thin co-occurrence row from swinging a whole naive-Bayes read the
# way the old pool source's pseudo-count shrinkage did). e^4 ≈ 55x either way.
_COOCCUR_LIFT_CLAMP = 4.0


def build_species_cooccur_prior(n_species: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """The two num-indexed tensors of the team-composition species prior:

      * ``log_marginal`` ``[n_species]`` — ``log P(an unrevealed opp slot is species s)``
      * ``log_lift``     ``[n_species, n_species]`` — ``log[ P(s | t) / P(s) ]``, the PAIRWISE
        evidence a revealed teammate ``t`` contributes about ``s``. Zero means "carries no
        information" (an unobserved pair, or ``t == s``), so an empty revealed set makes the
        whole evidence term vanish and the prior degrades EXACTLY to the marginal.

    Combined by naive Bayes in the forward (`BeliefHead.species_prior_logits`):

        log P(s | R) ∝ log P(s) + Σ_{r ∈ R} log-lift(s, r)

    **Sourced from SMOGON, never the pool** (owner rule 2026-08-15 — priors are always
    Smogon-based; the 719-team pool measures structure but never ships as a prior). The marginal
    is the same normalized usage share `build_species_usage_prior` emits; the lift comes from the
    chaos ``Teammates`` field (``gen3_data.priors.teammates`` — the ONE species×species joint
    Smogon publishes, ~2.5M gen3ou battles): ``P(s | t)`` is the per-slot teammate conditional,
    and the independence baseline renormalizes the usage share to exclude ``t`` itself
    (a teammate of t is never t). Forme keys accumulate into their base num (the num-axis rule).
    An s absent from t's teammate row keeps lift 0 — UNOBSERVED at Smogon's truncation, not
    negative evidence. Until 2026-08-15 this was derived from ``data/teams/gen3_species_priors
    .json`` (the pool) — `agents.training.species_priors` remains as a pool-ANALYSIS tool only.

    Both are plain float32, for `BeliefHead` to register as NON-persistent buffers — data-derived
    and recomputable, never a saved weight (same contract as ``move_prior_logits``).

    Fail-loud GIGO guards, mirroring `build_species_usage_prior`: Tyranitar must resolve to a
    dominant usage marginal, and the sand core (Skarmory | Tyranitar) must carry POSITIVE lift —
    either failing means the id/num axis drifted and the prior silently flattened.

    Relocated from `damage_tables.py` (`gen3_dex_ids_split_v1`, 2026-09-06)."""
    usage_prior = build_species_usage_prior(n_species)                  # [S] slot shares, sum 1
    log_marginal = usage_prior.clamp_min(_SPECIES_PRIOR_FLOOR).log()
    log_marginal[0] = math.log(_SPECIES_PRIOR_FLOOR)                    # sentinel: never a candidate

    log_lift = torch.zeros((n_species, n_species), dtype=torch.float32)
    base_ids = set(gen3_data.species.base_form_ids())
    n_cols = 0
    for tid in base_ids:                                                # evidence species t
        td = gen3_data.species.get(tid)
        if td is None or not (0 < td.num < n_species):
            continue
        mates = gen3_data.priors.teammates(tid)
        if not mates:
            continue
        base_renorm = max(1.0 - float(usage_prior[td.num]), 1e-6)
        cond_by_num: dict = {}                                          # formes fold into base num
        for sid, p_cond in mates.items():
            sd = cast(SpeciesData, gen3_data.species.get(sid))
            if sd is None or not (0 < sd.num < n_species) or sd.num == td.num:
                continue
            cond_by_num[sd.num] = cond_by_num.get(sd.num, 0.0) + float(p_cond)
        for snum, p_cond in cond_by_num.items():
            expected = max(float(usage_prior[snum]), _SPECIES_PRIOR_FLOOR) / base_renorm
            lift = math.log(max(p_cond, 1e-9) / expected)
            log_lift[snum, td.num] = max(-_COOCCUR_LIFT_CLAMP, min(_COOCCUR_LIFT_CLAMP, lift))
        n_cols += 1
    if n_cols == 0:
        raise ValueError(
            "build_species_cooccur_prior: no species carried a Smogon teammate row — "
            "data/pokemon/gen3_teammate_priors.json is empty or its keys drifted.")

    tt = gen3_data.species.get("tyranitar")
    if tt is None or not (0 < tt.num < n_species) or float(log_marginal[tt.num]) < math.log(0.05):
        raise ValueError(
            "build_species_cooccur_prior: Tyranitar did not resolve to a dominant usage marginal — "
            "the team-composition species prior is empty/misaligned (id normalization drift?). "
            "GIGO guard.")
    sk = gen3_data.species.get("skarmory")
    if sk is None or float(log_lift[sk.num, tt.num]) <= 0.0:
        raise ValueError(
            "build_species_cooccur_prior: Skarmory|Tyranitar lift is not positive — the sand core "
            "co-occurs far above independence on every Smogon window, so the teammate table is "
            "empty/misaligned. GIGO guard.")
    return log_marginal, log_lift
