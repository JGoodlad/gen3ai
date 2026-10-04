"""Compute per-species prior probability distributions from aggregated Smogon stats.

Produces six files in one pass — all keyed by lowercase species name:

  data/pokemon/gen3_hidden_power_priors.json
    {species: {hp_type: probability}}   # 16 type buckets, omitted if usage=0

  data/pokemon/gen3_ability_priors.json
    {species: {ability_id: probability}}

  data/pokemon/gen3_move_priors.json     {species: {move_id: P(move in set)}}    # NOT sum-1 (~4)
  data/pokemon/gen3_spread_priors.json   {species: [[nature, [hp,atk,def,spa,spd,spe], weight], …]}
  data/pokemon/gen3_item_priors.json     {species: {item_id: P(item)}}           # sum-1
    # the latter three feed the incoming-damage / OHKO belief (gen3_incoming_damage_v1)

  data/pokemon/gen3_teammate_priors.json {species: {teammate_id: P(teammate)}}   # sum-1
    # the chaos Teammates field — the ONE joint (species x species co-occurrence) Smogon
    # publishes; the Smogon-based coupling prior for the hidden-TEAM belief. Priors are
    # ALWAYS Smogon-based, never pool-based (owner rule 2026-08-15): the 719-team pool may
    # MEASURE coupling (tmp/belief_coupling_lift.py) but never ship as a prior.

Ability priors are *anchored to the Showdown pokedex* as the ground truth for
what abilities each species CAN have in Gen 3. Smogon usage weights the
distribution with three branches based on coverage:

  1. **All covered** (every dex ability has non-zero Smogon counts):
     normalize Smogon usage directly. e.g. Snorlax → 86% Immunity / 14% Thick Fat.

  2. **Partial coverage** (some dex abilities observed, others not):
     keep the Smogon weights for observed abilities but assign a small floor
     (MIN_UNOBSERVED_PROB = 0.01) to each unobserved dex ability, then scale
     observed mass to fit. Preserves the strong Smogon signal where it exists
     while guaranteeing no dex-legal ability ever encodes as exactly 0.

  3. **No coverage** (no dex ability has Smogon data):
     uniform 1/N over dex abilities (50/50 for the usual two-ability case).

This three-tier rule protects against Smogon's 12-month window missing a rare
second ability without losing the strong-prior signal when one ability is
overwhelmingly favored.

Hidden Power priors stay Smogon-only (no dex anchor) — there's no "possible HP
type" constraint in the pokedex; usage data is authoritative.

WEIGHTED vs RAW — the denominator rule (`gen3_smogon_prior_denominator_v1`, F-X5-41, 2026-10-04).
A chaos species record carries ONE unweighted field and several RATING-WEIGHTED ones. Smogon's own
stats code (pkmn/stats `stats/src/stats.ts` @ 3321157c, `updateStats`; the legacy
Antar1011/Smogon-Usage-Stats `batchMovesetCounter.py` @ 59a9c1cf, L53-L134, is identical in this):

  * ``Raw count``   = ``p.raw.count++``  — the UNWEIGHTED number of sets (every rating, weight 1);
  * ``Abilities`` / ``Items`` / ``Spreads`` / ``Happiness`` / ``Moves`` each add ``weights.m`` (the
    rating weight at the file's cutoff, 1500 here) once per set — ``Moves`` once per move SLOT, an
    empty slot under the key ``""`` ("We're OK with triple counting 'nothing'");
  * ``Teammates`` adds ``weights.s`` (the player weight; = ``weights.m`` for every rated player).

So the four one-per-set fields share ONE weighted total W (= ``p.raw.weight``, which the chaos JSON
does not export), and ``Σ Moves = 4 W`` exactly. Smogon's moveset report prints every percentage as
``field[k] / p.raw.weight`` (pkmn/stats ``stats/src/reports.ts`` L233 / L246 / L259 / L271; the
legacy ``count = sum(abilities.values())``, L134 / L262). A weighted numerator over the UNWEIGHTED
``Raw count`` is a different population (W / Raw count ran 0.10–0.94 by species on the 2025-05 ..
2026-04 window), which deflated every move prior until 2026-10-04 (Skarmory Spikes 0.547 → 0.997).
:func:`weighted_count` is the ONE denominator, and it THROWS unless the four fields agree and
``Σ Moves == 4 W``; :func:`check_priors` THROWS on any output that breaks its own invariant
(a distribution summing to 1, a move prior summing with the empty-slot mass to exactly 4).

Run from repo root (after tools/smogon_stats_downloader/sync.py):
    python tools/smogon_stats_downloader/compute_priors.py
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from typing import Optional

STATS_PATH      = "data/pokemon/gen3_smogon_stats.json"
SPECIES_PATH    = "data/pokemon/gen3_species.json"
ABILITIES_PATH  = "data/pokemon/gen3_abilities.json"
POKEDEX_PATH    = "src/poke_env/data/static/pokedex/gen3pokedex.json"
HP_OUTPUT_PATH      = "data/pokemon/gen3_hidden_power_priors.json"
ABILITY_OUTPUT_PATH = "data/pokemon/gen3_ability_priors.json"
MOVE_OUTPUT_PATH    = "data/pokemon/gen3_move_priors.json"
SPREAD_OUTPUT_PATH  = "data/pokemon/gen3_spread_priors.json"
ITEM_OUTPUT_PATH    = "data/pokemon/gen3_item_priors.json"
TEAMMATE_OUTPUT_PATH = "data/pokemon/gen3_teammate_priors.json"

# Cap on spreads kept per species (sorted by usage). The Smogon tail is noise;
# the top ~25 cover the meaningful nature/EV modes (the facade derives Atk/SpA/Spe
# stat distributions from these for the incoming-damage / outspeed beliefs).
SPREAD_TOP_K = 25

# Canonical 16 hidden power types — alphabetical, matches HIDDEN_POWER_TYPE_ORDER
HIDDEN_POWER_TYPES = frozenset([
    "bug", "dark", "dragon", "electric", "fighting", "fire", "flying", "ghost",
    "grass", "ground", "ice", "poison", "psychic", "rock", "steel", "water",
])
HIDDEN_POWER_PREFIX = "hiddenpower"

# Floor probability for a dex-legal ability that Smogon never observed for a
# species. Chosen small enough to preserve a strong Smogon signal (Snorlax
# stays ~99% Immunity if Smogon never saw Thick Fat) while keeping the
# ability above the absolute-zero floor the embedding lookup uses for
# "unknown / no data" (ID 0). 1% is a soft "very unlikely but possible" prior.
MIN_UNOBSERVED_PROB = 0.01

# A set has exactly this many move SLOTS (the chaos `Moves` field counts every slot, an empty one
# under the key "" — see the module docstring).
MOVE_SLOTS = 4
# Relative tolerance for the weighted-total invariants. They are EXACT in real arithmetic; what is
# left is float summation order over 12 months x ~10^5-10^6 sets (measured max 1.2e-14 on the
# 2025-05 .. 2026-04 window). Any real defect is >= 6 % (the deflation it guards against was 6-90 %),
# eight orders of magnitude away.
WEIGHTED_TOTAL_RTOL = 1e-9
# The chaos fields counted once per set with the moveset weight — they must share one total.
_ONE_PER_SET_FIELDS = ("Abilities", "Items", "Spreads", "Happiness")


class PriorInvariantError(ValueError):
    """A Smogon prior (or its source record) broke an invariant that holds EXACTLY by construction."""


# ---------------------------------------------------------------------------
# Hidden Power
# ---------------------------------------------------------------------------

def _hp_type(move_key: str) -> Optional[str]:
    if not move_key.startswith(HIDDEN_POWER_PREFIX):
        return None
    t = move_key[len(HIDDEN_POWER_PREFIX):]
    return t if t in HIDDEN_POWER_TYPES else None


def compute_hidden_power_priors(chaos: dict, species_lookup: dict) -> dict:
    """Per-species HP type distributions normalized to sum to 1.0 over observed types."""
    type_counts_by_species: dict[str, dict[str, float]] = defaultdict(dict)
    for sp_name, sp_data in chaos["data"].items():
        moves = sp_data.get("Moves", {})
        for move_key, usage in moves.items():
            t = _hp_type(move_key.lower())
            if t is not None and usage > 0:
                type_counts_by_species[sp_name.lower()][t] = usage

    priors: dict[str, dict[str, float]] = {}
    total_obs = 0.0
    for sp_key, counts in type_counts_by_species.items():
        if sp_key not in species_lookup:
            continue
        total = sum(counts.values())
        if total == 0:
            continue
        priors[sp_key] = {t: c / total for t, c in counts.items()}
        total_obs += total
    return priors


# ---------------------------------------------------------------------------
# Abilities (dex-anchored)
# ---------------------------------------------------------------------------

def _to_id(name: str) -> str:
    return "".join(c for c in name.lower() if c.isalnum())


def _gen3_dex_abilities(species_id: str, pokedex: dict, valid_ability_ids: set) -> list[str]:
    """Return the species' Gen 3-valid ability IDs in dex order (slots 0 and 1)."""
    entry = pokedex.get(species_id)
    if entry is None:
        return []
    raw = entry.get("abilities", {})
    out = []
    for slot in ("0", "1"):
        name = raw.get(slot)
        if name is None:
            continue
        ab_id = _to_id(name)
        if ab_id in valid_ability_ids and ab_id not in out:
            out.append(ab_id)
    return out


def compute_ability_priors(
    chaos: dict,
    species_lookup: dict,
    pokedex: dict,
    valid_ability_ids: set,
) -> tuple[dict, dict]:
    """Per-species ability distributions, anchored to dex possibilities.

    Three-tier coverage rule (see module docstring for rationale):
      - smogon_full     — all dex abilities had Smogon data → pure Smogon weights
      - partial_floor   — some Smogon data + MIN_UNOBSERVED_PROB on unseen dex abilities
      - no_data_uniform — no Smogon data at all → uniform 1/N over dex abilities

    Returns (priors_dict, summary_dict). summary_dict tracks per-species counts
    plus "no_dex" for species not in the pokedex.
    """
    priors: dict[str, dict[str, float]] = {}
    summary = {
        "smogon_full": 0,
        "partial_floor": 0,
        "no_data_uniform": 0,
        "no_dex": 0,
        "no_dex_species": [],
    }
    rejected_post_gen3: dict[str, list[str]] = {}

    for sp_id in sorted(species_lookup.keys()):
        dex_abs = _gen3_dex_abilities(sp_id, pokedex, valid_ability_ids)
        if not dex_abs:
            summary["no_dex"] += 1
            summary["no_dex_species"].append(sp_id)
            continue

        # Look up Smogon usage for this species — chaos JSON uses capitalized names.
        chaos_entry = next(
            (v for k, v in chaos["data"].items() if k.lower() == sp_id),
            None,
        )
        raw_smogon = (chaos_entry or {}).get("Abilities", {})

        # Normalise Smogon keys and split into "in dex" vs "rejected" (post-Gen3 leakage)
        smogon_clean: dict[str, float] = {}
        for ab_raw, usage in raw_smogon.items():
            ab_id = _to_id(ab_raw)
            if ab_id in dex_abs:
                if usage > 0:
                    smogon_clean[ab_id] = float(usage)
            else:
                rejected_post_gen3.setdefault(sp_id, []).append(ab_id)

        observed = [ab for ab in dex_abs if smogon_clean.get(ab, 0) > 0]
        unobserved = [ab for ab in dex_abs if smogon_clean.get(ab, 0) == 0]

        if not observed:
            # Tier 3: no Smogon data → uniform across dex abilities (50/50 etc.)
            n = len(dex_abs)
            priors[sp_id] = {ab: 1.0 / n for ab in dex_abs}
            summary["no_data_uniform"] += 1
        elif not unobserved:
            # Tier 1: all dex abilities observed → pure Smogon weights
            total = sum(smogon_clean.values())
            priors[sp_id] = {ab: smogon_clean[ab] / total for ab in dex_abs}
            summary["smogon_full"] += 1
        else:
            # Tier 2: partial — keep Smogon weights for observed abilities but
            # reserve MIN_UNOBSERVED_PROB for each unobserved one. Total floor
            # mass scales linearly with the count of unseen abilities; observed
            # mass is scaled down to (1 - floor_mass) so probabilities still
            # sum to exactly 1.0.
            floor_mass = MIN_UNOBSERVED_PROB * len(unobserved)
            if floor_mass >= 1.0:
                # Degenerate case (can't happen with N≤2 and 1% floor, but
                # defensive). Fall back to uniform rather than emit negatives.
                n = len(dex_abs)
                priors[sp_id] = {ab: 1.0 / n for ab in dex_abs}
                summary["no_data_uniform"] += 1
                continue
            observed_total = sum(smogon_clean[ab] for ab in observed)
            scale = (1.0 - floor_mass) / observed_total
            sp_priors: dict[str, float] = {}
            for ab in dex_abs:
                if ab in observed:
                    sp_priors[ab] = smogon_clean[ab] * scale
                else:
                    sp_priors[ab] = MIN_UNOBSERVED_PROB
            priors[sp_id] = sp_priors
            summary["partial_floor"] += 1

    summary["rejected_post_gen3"] = rejected_post_gen3
    return priors, summary


# ---------------------------------------------------------------------------
# Move / Item / Spread priors (for the incoming-damage + outspeed beliefs)
# ---------------------------------------------------------------------------

def _close(a: float, b: float, rtol: float = WEIGHTED_TOTAL_RTOL) -> bool:
    return abs(a - b) <= rtol * max(abs(a), abs(b), 1e-300)


def weighted_count(sp_name: str, sp_data: dict) -> float:
    """W — the species' RATING-WEIGHTED set total, the ONE denominator for a weighted chaos field
    (module docstring: Smogon's ``p.raw.weight``). Read as ``Σ Abilities``; THROWS unless every other
    one-per-set field (``Items``, ``Spreads``, ``Happiness``) sums to the same W and ``Σ Moves`` to
    exactly ``MOVE_SLOTS · W``. Never ``Raw count`` (UNWEIGHTED — F-X5-41) and never ``usage`` (a
    latest-month share, not a count)."""
    w = float(sum(v for v in (sp_data.get("Abilities") or {}).values()))
    if not w > 0.0:
        raise PriorInvariantError(f"{sp_name}: no weighted Abilities total (W = {w!r}) — the chaos "
                                  f"record is not a moveset record; the denominator is undefined")
    for field in _ONE_PER_SET_FIELDS[1:]:
        if field in sp_data:
            t = float(sum((sp_data.get(field) or {}).values()))
            if not _close(t, w):
                raise PriorInvariantError(
                    f"{sp_name}: sum({field}) = {t!r} != sum(Abilities) = {w!r} — the one-per-set "
                    f"weighted fields must share ONE total (Smogon's p.raw.weight)")
    mv = float(sum((sp_data.get("Moves") or {}).values()))
    if not _close(mv, MOVE_SLOTS * w):
        raise PriorInvariantError(
            f"{sp_name}: sum(Moves) = {mv!r} != {MOVE_SLOTS} * W = {MOVE_SLOTS * w!r} — every set "
            f"contributes one weight per move SLOT (an empty slot under the key \"\")")
    return w


def empty_slot_mass(sp_name: str, sp_data: dict) -> float:
    """The expected number of EMPTY move slots per set, ``Moves[""] / W`` — what a species' move
    prior falls short of ``MOVE_SLOTS`` by (Ditto-like movepools; most sets run four)."""
    return float((sp_data.get("Moves") or {}).get("", 0.0)) / weighted_count(sp_name, sp_data)


def compute_move_priors(chaos: dict, species_lookup: dict) -> dict:
    """{species: {move_id: P(move in set)}} = Moves[m] / W (`weighted_count`, NOT ``Raw count``).

    NOT normalized to 1: ``Σ_m P(m) + empty_slot_mass == MOVE_SLOTS`` exactly (the values sum to
    ~4). This is P(the species' set contains move m), the quantity the §6.1 slot-accounting needs
    (revealed → 1, else this prior over the remaining slots). A value above 1 cannot happen (a set
    holds a move at most once) and THROWS rather than being clamped away."""
    out: dict[str, dict[str, float]] = {}
    for sp_name, sp_data in chaos["data"].items():
        sp_key = sp_name.lower()
        if sp_key not in species_lookup:
            continue
        moves = sp_data.get("Moves", {})
        if not moves:
            continue
        w = weighted_count(sp_name, sp_data)
        d: dict[str, float] = {}
        for mv, usage in moves.items():
            mid = _to_id(mv)
            if not mid or mid == "nomove" or usage <= 0:
                continue
            p = usage / w
            if p > 1.0 + WEIGHTED_TOTAL_RTOL:
                raise PriorInvariantError(f"{sp_name}: P({mid} in set) = {p!r} > 1")
            d[mid] = min(1.0, p)
        if d:
            out[sp_key] = d
    return out


def _check_distribution(name: str, sp: str, values: list, rtol: float = 1e-9) -> None:
    tot = float(sum(values))
    if not values or any(not (0.0 < v <= 1.0 + rtol) for v in values) or not _close(tot, 1.0, rtol):
        raise PriorInvariantError(f"{name}[{sp}]: not a distribution (sum {tot!r}, "
                                  f"min {min(values, default=None)!r}, max {max(values, default=None)!r})")


def check_move_priors(move_priors: dict, chaos: dict, rtol: float = WEIGHTED_TOTAL_RTOL) -> None:
    """THROWS unless every species' move prior obeys ``Σ_m P(m) + empty_slot_mass == MOVE_SLOTS``
    with each ``P(m)`` in ``(0, 1]`` — the exact invariant of a correct weighted denominator. A table
    divided by the unweighted ``Raw count`` fails it on EVERY species (its sums ran 0.41–3.75)."""
    by_key = {k.lower(): (k, v) for k, v in chaos["data"].items()}
    for sp, row in move_priors.items():
        if sp not in by_key:
            raise PriorInvariantError(f"move prior for {sp!r} has no chaos record")
        name, rec = by_key[sp]
        vals = list(row.values())
        if not vals or any(not (0.0 < v <= 1.0) for v in vals):
            raise PriorInvariantError(f"move prior[{sp}]: a P(in set) outside (0, 1]")
        tot = float(sum(vals)) + empty_slot_mass(name, rec)
        if not _close(tot, float(MOVE_SLOTS), rtol):
            raise PriorInvariantError(
                f"move prior[{sp}]: sum P(in set) + empty-slot mass = {tot!r} != {MOVE_SLOTS} — the "
                f"prior is not Moves / W (a deflated table divided by the UNWEIGHTED Raw count reads "
                f"0.41-3.75 here; F-X5-41)")


def check_priors(chaos: dict, *, hp: dict, ability: dict, move: dict, item: dict,
                 spread: dict, teammate: dict) -> None:
    """Every output's own invariant, checked BEFORE anything is written (THROWS
    :class:`PriorInvariantError`): HP types, abilities, items, spread weights and teammates are each a
    distribution per species (sum 1); moves obey :func:`check_move_priors`."""
    for name, table in (("hidden_power", hp), ("ability", ability), ("item", item),
                        ("teammate", teammate)):
        for sp, row in table.items():
            _check_distribution(name, sp, list(row.values()))
    for sp, rows in spread.items():
        _check_distribution("spread", sp, [float(r[2]) for r in rows])
    check_move_priors(move, chaos)


def compute_item_priors(chaos: dict, species_lookup: dict) -> dict:
    """{species: {item_id: P(item)}} normalized over observed items (sum→1).

    'nothing' (no item) is kept — it's informative (rules out Choice Band, the
    never-revealed item the §6.3 worst-case channel must infer)."""
    out: dict[str, dict[str, float]] = {}
    for sp_name, sp_data in chaos["data"].items():
        sp_key = sp_name.lower()
        if sp_key not in species_lookup:
            continue
        items = sp_data.get("Items", {})
        tot = sum(v for v in items.values() if v > 0)
        if tot <= 0:
            continue
        d = {_to_id(it): usage / tot for it, usage in items.items() if usage > 0 and _to_id(it)}
        if d:
            out[sp_key] = d
    return out


def compute_teammate_priors(chaos: dict, species_lookup: dict) -> dict:
    """{species: {teammate_species_id: P(teammate | species)}} normalized over kept teammates
    (sum→1). The chaos `Teammates` field is a rating-weighted co-occurrence count (verified
    all-positive on the 12-month merge; NOT the +/-delta the text-format tables print), so a
    per-species normalization is a proper conditional. Teammates outside our gen3 species
    table (illegal/OM spillover) are dropped BEFORE normalizing, so the kept mass is a
    distribution over teammates the belief can actually name."""
    out: dict[str, dict[str, float]] = {}
    for sp_name, sp_data in chaos["data"].items():
        sp_key = sp_name.lower()
        if sp_key not in species_lookup:
            continue
        mates = {_to_id(t): v for t, v in (sp_data.get("Teammates") or {}).items()
                 if v > 0 and _to_id(t) in species_lookup}
        tot = sum(mates.values())
        if tot <= 0:
            continue
        out[sp_key] = {t: v / tot for t, v in mates.items()}
    return out


def compute_spread_priors(chaos: dict, species_lookup: dict, top_k: int = SPREAD_TOP_K) -> dict:
    """{species: [[nature, [hp,atk,def,spa,spd,spe], weight], ...]} — top-K raw spreads,
    weights renormalized to sum→1. Raw nature/EV spreads (provenance-clean); the
    gen3_data facade derives the Atk/SpA/Spe **stat distributions** (L100, IV31, nature
    applied) used by the incoming-damage magnitude + P(outspeed) beliefs."""
    out: dict[str, list] = {}
    for sp_name, sp_data in chaos["data"].items():
        sp_key = sp_name.lower()
        if sp_key not in species_lookup:
            continue
        parsed = []
        for key, usage in sp_data.get("Spreads", {}).items():
            if usage <= 0 or ":" not in key:
                continue
            nature, evstr = key.split(":", 1)
            try:
                evs = [int(x) for x in evstr.split("/")]
            except ValueError:
                continue
            if len(evs) == 6:
                parsed.append((nature, evs, float(usage)))
        if not parsed:
            continue
        parsed.sort(key=lambda t: -t[2])
        parsed = parsed[:top_k]
        tot = sum(t[2] for t in parsed)
        out[sp_key] = [[nat, evs, w / tot] for nat, evs, w in parsed]
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _load_all() -> tuple[dict, dict, dict, set]:
    for path in (STATS_PATH, SPECIES_PATH, ABILITIES_PATH, POKEDEX_PATH):
        if not os.path.exists(path):
            print(f"ERROR: {path} missing. Run sync.py first if it's the stats file.",
                  file=sys.stderr)
            sys.exit(1)
    with open(STATS_PATH) as f:
        chaos = json.load(f)
    with open(SPECIES_PATH) as f:
        species = json.load(f)
    with open(POKEDEX_PATH) as f:
        pokedex = json.load(f)
    with open(ABILITIES_PATH) as f:
        gen3_abilities = json.load(f)
    return chaos, species, pokedex, set(gen3_abilities.keys())


def _write(path: str, data: dict) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")


def main() -> int:
    chaos, species, pokedex, valid_ability_ids = _load_all()

    # --- Hidden Power ---
    hp_priors = compute_hidden_power_priors(chaos, species)

    # --- Abilities (dex-anchored) ---
    ab_priors, summary = compute_ability_priors(chaos, species, pokedex, valid_ability_ids)
    print(
        f"Ability priors (dex-anchored):\n"
        f"  Tier 1 — Smogon-weighted (all dex abilities covered):    {summary['smogon_full']}\n"
        f"  Tier 2 — Partial + {MIN_UNOBSERVED_PROB:.0%} floor for unseen abilities: {summary['partial_floor']}\n"
        f"  Tier 3 — Uniform (no Smogon data for any dex ability):   {summary['no_data_uniform']}\n"
        f"  Skipped (species not in pokedex): {summary['no_dex']}"
    )
    if summary["rejected_post_gen3"]:
        n = sum(len(v) for v in summary["rejected_post_gen3"].values())
        print(
            f"  Rejected post-Gen3 ability entries: {n} across "
            f"{len(summary['rejected_post_gen3'])} species"
        )

    # Spot-check a few interesting species so a glance at output catches regressions
    print("\nSpot check:")
    for sp in ("snorlax", "shedinja", "salamence", "lanturn", "aerodactyl",
               "arcanine", "skarmory", "blissey"):
        p = ab_priors.get(sp, {})
        if not p:
            print(f"  {sp}: (no priors)")
            continue
        items = sorted(p.items(), key=lambda kv: -kv[1])
        line = ", ".join(f"{a}={v:.2%}" for a, v in items)
        print(f"  {sp:<12} {line}")

    # --- Move / Item / Spread priors (incoming-damage + outspeed beliefs) ---
    move_priors = compute_move_priors(chaos, species)
    item_priors = compute_item_priors(chaos, species)
    spread_priors = compute_spread_priors(chaos, species)
    teammate_priors = compute_teammate_priors(chaos, species)
    # Every output's invariant, BEFORE any write (a throw leaves data/ untouched).
    check_priors(chaos, hp=hp_priors, ability=ab_priors, move=move_priors, item=item_priors,
                 spread=spread_priors, teammate=teammate_priors)
    _write(HP_OUTPUT_PATH, hp_priors)
    print(f"Hidden Power priors: {len(hp_priors)} species → {HP_OUTPUT_PATH}")
    _write(ABILITY_OUTPUT_PATH, ab_priors)
    print(f"Ability priors: {len(ab_priors)} species → {ABILITY_OUTPUT_PATH}")

    _write(MOVE_OUTPUT_PATH, move_priors)
    print(f"\nMove priors: {len(move_priors)} species → {MOVE_OUTPUT_PATH}")
    _write(ITEM_OUTPUT_PATH, item_priors)
    print(f"Item priors: {len(item_priors)} species → {ITEM_OUTPUT_PATH}")
    _write(SPREAD_OUTPUT_PATH, spread_priors)
    print(f"Spread priors: {len(spread_priors)} species → {SPREAD_OUTPUT_PATH}")
    _write(TEAMMATE_OUTPUT_PATH, teammate_priors)
    print(f"Teammate priors: {len(teammate_priors)} species → {TEAMMATE_OUTPUT_PATH}")
    tm = sorted(teammate_priors.get("tyranitar", {}).items(), key=lambda kv: -kv[1])[:5]
    print("  tyranitar teammates: " + ", ".join(f"{t}={p:.0%}" for t, p in tm))

    print("\nSpot check (move/item/spread):")
    for sp in ("tyranitar", "salamence", "suicune", "blissey", "skarmory"):
        mv = sorted(move_priors.get(sp, {}).items(), key=lambda kv: -kv[1])[:5]
        it = sorted(item_priors.get(sp, {}).items(), key=lambda kv: -kv[1])[:3]
        spr = spread_priors.get(sp, [])
        print(f"  {sp}:")
        print("     moves: " + ", ".join(f"{m}={p:.0%}" for m, p in mv))
        print("     items: " + ", ".join(f"{i}={p:.0%}" for i, p in it)
              + f"   (choiceband={item_priors.get(sp, {}).get('choiceband', 0):.0%})")
        print("     top spreads: " + "; ".join(f"{n} {ev} w={w:.0%}" for n, ev, w in spr[:3]))

    return 0


if __name__ == "__main__":
    sys.exit(main())
