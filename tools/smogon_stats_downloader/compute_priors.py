"""Compute per-species prior probability distributions from aggregated Smogon stats.

Produces two files in one pass — both keyed by lowercase species name:

  data/pokemon/gen3_hidden_power_priors.json
    {species: {hp_type: probability}}   # 16 type buckets, omitted if usage=0

  data/pokemon/gen3_ability_priors.json
    {species: {ability_id: probability}}

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

def _raw_count(sp_data: dict) -> float:
    """Weighted total sets sampled for a species (denominator for P(in set))."""
    return float(sp_data.get("Raw count") or sp_data.get("usage") or 0.0)


def compute_move_priors(chaos: dict, species_lookup: dict) -> dict:
    """{species: {move_id: P(move in set)}} = Moves[m] / Raw count.

    NOT normalized to 1 — a set runs ~4 moves, so the values sum to ~4. This is
    exactly P(the species' set contains move m), the quantity the §6.1 slot-
    accounting needs (revealed → 1, else this prior over the remaining slots)."""
    out: dict[str, dict[str, float]] = {}
    for sp_name, sp_data in chaos["data"].items():
        sp_key = sp_name.lower()
        if sp_key not in species_lookup:
            continue
        raw = _raw_count(sp_data)
        moves = sp_data.get("Moves", {})
        if raw <= 0 or not moves:
            continue
        d: dict[str, float] = {}
        for mv, usage in moves.items():
            mid = _to_id(mv)
            if not mid or mid == "nomove" or usage <= 0:
                continue
            d[mid] = min(1.0, usage / raw)
        if d:
            out[sp_key] = d
    return out


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
    _write(HP_OUTPUT_PATH, hp_priors)
    print(f"Hidden Power priors: {len(hp_priors)} species → {HP_OUTPUT_PATH}")

    # --- Abilities (dex-anchored) ---
    ab_priors, summary = compute_ability_priors(chaos, species, pokedex, valid_ability_ids)
    _write(ABILITY_OUTPUT_PATH, ab_priors)
    print(
        f"Ability priors: {len(ab_priors)} species → {ABILITY_OUTPUT_PATH}\n"
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
    _write(MOVE_OUTPUT_PATH, move_priors)
    print(f"\nMove priors: {len(move_priors)} species → {MOVE_OUTPUT_PATH}")

    item_priors = compute_item_priors(chaos, species)
    _write(ITEM_OUTPUT_PATH, item_priors)
    print(f"Item priors: {len(item_priors)} species → {ITEM_OUTPUT_PATH}")

    spread_priors = compute_spread_priors(chaos, species)
    _write(SPREAD_OUTPUT_PATH, spread_priors)
    print(f"Spread priors: {len(spread_priors)} species → {SPREAD_OUTPUT_PATH}")

    print("\nSpot check (move/item/spread):")
    for sp in ("tyranitar", "salamence", "suicune", "blissey"):
        mv = sorted(move_priors.get(sp, {}).items(), key=lambda kv: -kv[1])[:5]
        it = sorted(item_priors.get(sp, {}).items(), key=lambda kv: -kv[1])[:3]
        spr = spread_priors.get(sp, [])
        print(f"  {sp}:")
        print(f"     moves: " + ", ".join(f"{m}={p:.0%}" for m, p in mv))
        print(f"     items: " + ", ".join(f"{i}={p:.0%}" for i, p in it)
              + f"   (choiceband={item_priors.get(sp, {}).get('choiceband', 0):.0%})")
        print(f"     top spreads: " + "; ".join(f"{n} {ev} w={w:.0%}" for n, ev, w in spr[:3]))

    return 0


if __name__ == "__main__":
    sys.exit(main())
