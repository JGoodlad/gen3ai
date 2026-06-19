"""Regenerate the gen-N Pokémon data mappings under data/pokemon/.

These mappings are the source of truth consumed by the observation encoders
(src/agents/observation/). They are *derived* from the poke-env static data
shipped in src/poke_env/data/static/ plus the Pokémon Showdown source tree in
deps/pokemon-showdown/. This tool rebuilds them so the derivation is
reproducible instead of a one-off hand edit.

Extracts (one --datasets entry each; `all` rebuilds every file):
  - abilities   -> data/pokemon/gen{N}_abilities.json   (pokedex + Showdown abilities.ts)
  - moves       -> data/pokemon/gen{N}_moves.json        (poke-env static moves; incl. `accuracy`)
  - species     -> data/pokemon/gen{N}_species.json      (poke-env pokedex; num + base stats)
  - items       -> data/pokemon/gen{N}_items.json        (Showdown items.ts; name + num)
  - type_chart  -> data/pokemon/gen{N}_type_chart.json   (GenData type chart; effectiveness)
  - natures     -> data/pokemon/gen{N}_natures.json      (poke-env natures; stat multipliers)

These are the runtime's source of truth (read via the `agents.gen3_data` facade); rebuilding
here keeps the derivation reproducible instead of a one-off hand edit.

Usage:
  python tools/pokemon_data_extractor/sync.py                 # all, gen 3, write files
  python tools/pokemon_data_extractor/sync.py --datasets moves species
  python tools/pokemon_data_extractor/sync.py --gen 3 --stdout
"""

import argparse
import json
import os
import re
import sys

# Repo root = two levels up from this file (tools/pokemon_data_extractor/sync.py).
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _static(*parts):
    return os.path.join(REPO_ROOT, "src", "poke_env", "data", "static", *parts)


def to_id_str(name):
    """Converts a string to a Showdown ID (lowercase alphanumeric only)."""
    return "".join(char for char in name.lower() if char.isalnum())


# --------------------------------------------------------------------------- #
# Abilities
# --------------------------------------------------------------------------- #

# Last ability `num` belonging to each generation (order of introduction in the
# games). Anything above the target gen's ceiling did not exist yet and is
# filtered out. Gen 3 ends at Air Lock (76); 77 is Tangled Feet, the first Gen 4
# ability — including it wrongly pulls Tangled Feet onto the Gen 3 Pidgey line.
_GEN_MAX_ABILITY_NUM = {
    1: 0, 2: 0, 3: 76, 4: 123, 5: 164, 6: 191, 7: 233, 8: 267, 9: 1000,
}


def get_ability_id_to_num(abilities_path):
    """Parse abilities.ts into a mapping of ability ID -> official index number."""
    if not os.path.exists(abilities_path):
        return {}

    with open(abilities_path, "r") as f:
        content = f.read()

    # Each ability block looks like:  \n\tabid: {\n ... num: 123, ... }
    matches = re.finditer(r"^\t([a-z0-9]+):\s*\{", content, re.MULTILINE)

    id_to_num = {}
    for match in matches:
        ab_id = match.group(1)
        block = content[match.start():match.start() + 5000]
        num_match = re.search(r"num:\s*(\d+)", block)
        if num_match:
            id_to_num[ab_id] = int(num_match.group(1))

    return id_to_num


def build_abilities(gen):
    """Build the gen-N ability map from the pokedex + Showdown abilities source."""
    pokedex_path = _static("pokedex", f"gen{gen}pokedex.json")
    abilities_path = os.path.join(REPO_ROOT, "deps", "pokemon-showdown", "data", "abilities.ts")

    if not os.path.exists(pokedex_path):
        raise FileNotFoundError(f"Pokedex file not found: {pokedex_path}")

    id_to_num = get_ability_id_to_num(abilities_path)
    if not id_to_num:
        raise FileNotFoundError(f"Could not parse abilities source: {abilities_path}")

    with open(pokedex_path, "r") as f:
        dex = json.load(f)

    max_num = _GEN_MAX_ABILITY_NUM.get(gen, 1000)
    abilities_map = {}

    for mon_id, mon_data in dex.items():
        num = mon_data.get("num", 0)
        if num <= 0:  # Skip CAP Pokémon
            continue

        mon_intro_gen = mon_data.get(
            "gen", 1 if num <= 151 else (2 if num <= 251 else (3 if num <= 386 else 4))
        )
        if mon_intro_gen > gen:
            continue

        # Skip non-base forms (Megas etc.); in Gen 3 only base abilities mattered.
        base_species = mon_data.get("baseSpecies", mon_id)
        if to_id_str(base_species) != mon_id:
            continue

        for slot, ability_name in mon_data.get("abilities", {}).items():
            if slot == "H" and gen < 5:
                continue
            if slot == "S" and gen < 7:
                continue

            ab_id = to_id_str(ability_name)
            ab_num = id_to_num.get(ab_id, 999)
            if ab_num > max_num:  # Did not exist in the target gen
                continue

            if ab_id not in abilities_map:
                abilities_map[ab_id] = {"name": ability_name, "num": ab_num}

    return {ab_id: abilities_map[ab_id] for ab_id in sorted(abilities_map)}


# --------------------------------------------------------------------------- #
# Moves
# --------------------------------------------------------------------------- #

# Major status conditions a status move can inflict (the move's PRIMARY `status`
# field). Damaging moves carry status only as a `secondary` chance, which we do
# NOT surface here — `inflicts_status` is about a move whose *purpose* is the status.
_MAJOR_STATUSES = frozenset({"par", "brn", "psn", "tox", "slp", "frz"})

# Entry-hazard side conditions (gen3 has only Spikes; later gens add Stealth Rock /
# Toxic Spikes / Sticky Web — keyed by Showdown's `sideCondition` id).
_HAZARD_SIDE_CONDITIONS = frozenset({"spikes", "stealthrock", "toxicspikes", "stickyweb"})

# Protect-family "stalling" moves set one of these volatiles (Protect/Detect → protect,
# Endure → endure). Detected by the `volatileStatus` field (declarative).
_PROTECT_VOLATILES = frozenset({"protect", "endure"})

# Moves whose self-boost is implemented ENTIRELY in a JavaScript `onHit` callback,
# so it is INVISIBLE in the declarative fields the poke-env static JSON carries.
# Determined by scanning deps/pokemon-showdown for gen3 Status moves that call
# `this.boost(...)` with no declarative `boosts`/`self.boosts` — the only one is
# Belly Drum (onHit: `this.boost({atk: 12}, target)` + directDamage half HP).
# Curse is deliberately NOT here: its self-boost (non-Ghost user → {atk,def,spe})
# is set conditionally on the user's type at onTryHit time, so it can only be
# resolved live in the encoder, not as a static flag. Memento is NOT a self-boost
# (its boosts target the FOE and the user faints) — excluded by the target=='self'
# gate below. See tools/CLAUDE.md.
_CALLBACK_SELF_BOOST = frozenset({"bellydrum"})

# Status-CURE moves (gen3_status_cure_moves_v1). Like the self-boost above, the cure
# lives ENTIRELY in a JavaScript `onHit` callback (`pokemon.cureStatus()` /
# `side.pokemon.forEach(... cureStatus)`), so it is INVISIBLE in the declarative fields
# the poke-env static JSON carries — hence a curated override, not a derived flag.
# Two scopes, mirroring the two new per-move obs bits:
#   - _CURES_SELF_STATUS: cures the USER'S OWN major status and leaves it statusless —
#     Refresh (par/psn/brn). Rest is deliberately EXCLUDED: it is already `is_heal`, and
#     it does not leave you statusless (it REPLACES the status with sleep), so flagging it
#     here would muddy the "this move clears my status" signal.
#   - _CURES_TEAM_STATUS: cures the WHOLE party's status — Heal Bell, Aromatherapy
#     (both `target: allyTeam` in gen3). Lets the model learn to value the move off the
#     BENCH status one-hots, not just the active's.
# Ability-based cures (Natural Cure / Shed Skin) are NOT moves and are out of scope here
# (surfaced via the per-mon ability block + the `ability_activated` volatile). See tools/CLAUDE.md.
_CURES_SELF_STATUS = frozenset({"refresh"})
_CURES_TEAM_STATUS = frozenset({"healbell", "aromatherapy"})

# --- gen3_typed_hidden_power_ids_v1: distinct nums for OUR-side typed Hidden Power ----------- #
# Showdown ships all 17 Hidden Power ids (bare 'hiddenpower' + 16 typed) at num 237 — the protocol
# never reveals the type, so the OPPONENT's HP is only ever observed bare. But WE always know our own
# HP type (IV-derived, declared in our team), so we give each typed variant its OWN distinct num: the
# move embedding row then IS the type, and our own-team obs / the damage-op per-move tables price it
# correctly without the type-blend workaround. The bare 'hiddenpower' KEEPS num 237 — the typeless
# form the opponent is observed as, AND the aggregation target for the opp move-belief prior/labels
# (damage_tables._belief_num / gen3_env._move_num fold every HP back to 237 on the opponent side).
# The 16 typed nums are assigned in FIXED alphabetical-by-type order (deterministic → the extractor
# parity test reproduces the committed file), starting at 355 — free below the move-embedding width
# (max real move num is 354; state_encoder max_moves=400). The order matches HIDDEN_POWER_TYPE_ORDER.
# Design: designs/ai_v6/design_typed_hidden_power_ids.md.
_HP_TYPE_ORDER = (
    "bug", "dark", "dragon", "electric", "fighting", "fire", "flying", "ghost",
    "grass", "ground", "ice", "poison", "psychic", "rock", "steel", "water",
)
_HP_FIRST_NUM = 355
_HP_TYPE_NUMS = {f"hiddenpower{t}": _HP_FIRST_NUM + i for i, t in enumerate(_HP_TYPE_ORDER)}

# --- gen3_unified_move_system_v1: structured secondary-effect extraction ---------------- #
# The 10 secondary-effect columns the model prices (single source of truth, mirrored by
# damage_tables.MOVE_SECONDARY). A damaging move's secondary is normalized into {column: percent}
# from the field Showdown keys it on: `secondary.status` (major status), `secondary.volatileStatus`
# (flinch/confusion), foe-targeting `secondary.boosts` (stat DROP), and `secondary.self.boosts`
# (self stat-RAISE). This REVERSES the old "secondary status is incidental" decision (the model now
# prices Body Slam's 30% para etc.) — see designs/ai_v6/design_unified_move_system.md.
_SECONDARY_COLS = (
    "par", "brn", "frz", "slp", "psn", "tox", "confusion", "flinch", "foe_statdrop", "self_boost",
)
_SECONDARY_STATUS_COLS = frozenset({"par", "brn", "frz", "slp", "psn", "tox"})

# Callback-only secondaries (invisible declaratively, like Belly Drum / Refresh). Tri Attack's
# 20% picks one of par/brn/frz at onHit time; we split the 20% across the three columns.
_SECONDARY_ONHIT = {"triattack": {"par": 7, "brn": 7, "frz": 6}}


def _accumulate_secondary(block, out):
    """Fold one Showdown `secondary`/`secondaries[i]` block into the column→percent dict `out`.
    A block has ONE trigger chance; it can carry a status, a flinch/confusion volatile, a foe
    stat-drop (`boosts`), and/or a self stat-raise (`self.boosts`)."""
    if not isinstance(block, dict):
        return
    chance = block.get("chance")
    chance = 100 if chance is None else int(chance)
    status = block.get("status")
    if status in _SECONDARY_STATUS_COLS:
        out[status] = out.get(status, 0) + chance
    vol = block.get("volatileStatus")
    if vol in ("flinch", "confusion"):
        out[vol] = out.get(vol, 0) + chance
    if isinstance(block.get("boosts"), dict):  # foe-targeting stat drop (Crunch -spd, Psychic -spd)
        out["foe_statdrop"] = out.get("foe_statdrop", 0) + chance
    self_block = block.get("self")
    if isinstance(self_block, dict) and isinstance(self_block.get("boosts"), dict):
        out["self_boost"] = out.get("self_boost", 0) + chance


def _secondary_effects(move_id, entry):
    """Normalize a move's secondary effect(s) into {column: percent} over `_SECONDARY_COLS`.
    Curated `onHit` override wins (Tri Attack); otherwise fold `secondary` + every `secondaries[i]`.
    Percent is capped at 100; empty dict when the move has no secondary."""
    if move_id in _SECONDARY_ONHIT:
        return dict(_SECONDARY_ONHIT[move_id])
    out = {}
    _accumulate_secondary(entry.get("secondary"), out)
    for block in (entry.get("secondaries") or []):
        _accumulate_secondary(block, out)
    return {k: min(100, v) for k, v in out.items() if v > 0}


def _fraction(pair):
    """Showdown `drain`/`recoil` are `[num, den]` fractions of damage dealt → a float in [0,1]."""
    if isinstance(pair, (list, tuple)) and len(pair) == 2 and pair[1]:
        return pair[0] / pair[1]
    return 0.0


def _has_self_positive_boost(entry):
    """True iff the move declaratively raises one of the USER'S OWN stats (a setup
    move). Covers the two declarative shapes — a top-level ``boosts`` on a
    ``target: self`` move (Swords Dance, Calm Mind, Dragon Dance, Tail Glow, …) and a
    ``self: {boosts: …}`` block (post-hit self-boosts) — requiring at least one
    POSITIVE stage so foe-targeting debuffs (Memento's ``boosts:{atk:-2,spa:-2}`` on
    ``target: normal``) and self-debuff drawbacks never count as setup."""
    if entry.get("target") == "self":
        boosts = entry.get("boosts")
        if isinstance(boosts, dict) and any(v > 0 for v in boosts.values()):
            return True
    self_block = entry.get("self")
    if isinstance(self_block, dict):
        sb = self_block.get("boosts")
        if isinstance(sb, dict) and any(v > 0 for v in sb.values()):
            return True
    return False


def build_moves(gen):
    """Build the gen-N move map from the poke-env static move data.

    Mirrors the fields the observation encoder relies on. Accuracy is split into
    two fields so `accuracy` is always numeric: the source stores either an int
    percentage (30-100) or the boolean `true` for never-miss moves (Swift, Aerial
    Ace, all status/self moves). We map never-miss to `accuracy: 100` and flag it
    via `never_miss: true`. A 100%-accuracy move can still miss into evasion
    (Double Team) or after Sand-Attack; a never-miss move bypasses the
    accuracy/evasion check entirely — hence the dedicated bit.

    Move-EFFECT classification (gen3_move_effects_v1) — the action-aligned per-move
    effect flags the observation's reactive block surfaces so the policy head can
    tell a setup move from a heal from a wasted status (all of which look identical
    at the head otherwise: base power 0, neutral type multiplier). Showdown implements
    effects through a mix of DECLARATIVE fields and JS CALLBACKS, so each flag is
    derived from the field Showdown actually keys the mechanic on — not guessed from
    the move name (garbage in, garbage out):
      - `is_heal`     ← `flags.heal == 1`. Showdown tags every HP-restoring move with
                        this flag (so Heal Block can key off it), INCLUDING the
                        callback-only ones with no declarative `heal` amount
                        (Moonlight/Synthesis/Morning Sun = weather-scaled, Rest, Wish,
                        Swallow). It correctly EXCLUDES drain attacks (Giga Drain has
                        `drain` but no `flags.heal`) and Leech Seed/Ingrain/Pain Split.
      - `is_protect`  ← `volatileStatus ∈ {protect, endure}` (Protect/Detect/Endure).
      - `is_phaze`    ← `forceSwitch` (Roar/Whirlwind).
      - `is_hazard`   ← `sideCondition` is an entry hazard (gen3: Spikes).
      - `inflicts_status` / `status` ← the move's PRIMARY `status` field, when it is a
                        major status (par/brn/psn/tox/slp/frz). This is the move whose
                        *purpose* is the status (Thunder Wave, Toxic).
      - `priority` / `secondaryEffects` / `drainFraction` / `recoilFraction`
                        (gen3_unified_move_system_v1) ← the structured secondary-effect
                        extraction. This REVERSES the old "secondary status is incidental"
                        decision: a damaging move's `secondary` is normalized into a
                        {column: percent} dict over `_SECONDARY_COLS` (Body Slam → {par:30},
                        Rock Slide → {flinch:30}, Crunch → {foe_statdrop:20}, Meteor Mash →
                        {self_boost:20}), so the model prices the secondary. These fields are
                        GPU-side only (DamageOperator + MoveLatentEncoder) — they do NOT enter
                        the obs vector, so the obs golden is unchanged.
      - `is_boost`    ← declarative self-positive boost (`_has_self_positive_boost`)
                        OR a curated callback override (`_CALLBACK_SELF_BOOST` = Belly
                        Drum, whose +6 Atk lives in an `onHit` callback). Curse is
                        resolved LIVE in the encoder (its boost depends on the user's
                        type), so it is NOT flagged here.
      - `curesSelfStatus` / `curesTeamStatus` (gen3_status_cure_moves_v1) ← curated
                        callback overrides (`_CURES_SELF_STATUS` = Refresh;
                        `_CURES_TEAM_STATUS` = Heal Bell / Aromatherapy). The cure lives in
                        an `onHit` callback (invisible declaratively), so these mirror the
                        Belly Drum treatment. They give the policy head a per-move signal it
                        previously lacked — that a move CLEARS status — so it can connect
                        Refresh/Heal Bell to the status one-hots it already sees (verified
                        gap: the head conditioned its own status onto Recover/switch but
                        never onto the cure move).
    """
    moves_path = _static("moves", f"gen{gen}moves.json")
    if not os.path.exists(moves_path):
        raise FileNotFoundError(f"Moves file not found: {moves_path}")

    with open(moves_path, "r") as f:
        src = json.load(f)

    moves_map = {}
    for move_id, entry in src.items():
        if entry.get("num", 0) <= 0:  # Skip fake/CAP moves (negative nums)
            continue

        raw_accuracy = entry.get("accuracy")
        never_miss = raw_accuracy is True

        flags = entry.get("flags") or {}
        primary_status = entry.get("status")
        if primary_status not in _MAJOR_STATUSES:
            primary_status = None
        side_condition = entry.get("sideCondition")

        moves_map[move_id] = {
            "name": entry.get("name"),
            # gen3_typed_hidden_power_ids_v1: OUR-side typed HP gets a distinct num (355-370); bare
            # 'hiddenpower' (and every other move) keeps its Showdown num. See _HP_TYPE_NUMS above.
            "num": _HP_TYPE_NUMS.get(move_id, entry.get("num")),
            "type": entry.get("type"),
            "basePower": entry.get("basePower"),
            "target": entry.get("target"),
            "hasSecondary": bool(entry.get("secondary") or entry.get("secondaries")),
            "hasRecoil": bool(entry.get("recoil")),
            "accuracy": 100 if never_miss else raw_accuracy,
            "never_miss": never_miss,
            # --- gen3_move_effects_v1: action-aligned effect classification ---
            "isBoost": _has_self_positive_boost(entry) or move_id in _CALLBACK_SELF_BOOST,
            "isHeal": bool(flags.get("heal")),
            "isProtect": entry.get("volatileStatus") in _PROTECT_VOLATILES,
            "isPhaze": bool(entry.get("forceSwitch")),
            "isHazard": side_condition in _HAZARD_SIDE_CONDITIONS,
            "status": primary_status,  # major status this move INFLICTS, else null
            # gen3_status_cure_moves_v1: curated callback overrides (onHit cure → no
            # declarative field). curesSelfStatus = Refresh; curesTeamStatus = Heal Bell /
            # Aromatherapy. Let the policy head connect the cure to the status one-hots.
            "curesSelfStatus": move_id in _CURES_SELF_STATUS,
            "curesTeamStatus": move_id in _CURES_TEAM_STATUS,
            # --- gen3_unified_move_system_v1: structured secondary / priority / drain / recoil ---
            # GPU-side only (the DamageOperator + MoveLatentEncoder read these); they do NOT enter
            # the obs vector, so the obs golden is unchanged. secondaryEffects is {column: percent}.
            "priority": int(entry.get("priority", 0) or 0),
            "secondaryEffects": _secondary_effects(move_id, entry),
            "drainFraction": _fraction(entry.get("drain")),
            "recoilFraction": _fraction(entry.get("recoil")),
        }

    return moves_map


# --------------------------------------------------------------------------- #
# Species
# --------------------------------------------------------------------------- #

# Last species `num` belonging to each generation (national-dex order). Anything
# above the target gen's ceiling did not exist yet and is filtered out.
_GEN_MAX_SPECIES_NUM = {
    1: 151, 2: 251, 3: 386, 4: 493, 5: 649, 6: 721, 7: 809, 8: 905, 9: 1025,
}


def build_species(gen):
    """Build the gen-N species map (num + base stats) from the poke-env pokedex.

    Only base forms are kept (one entry per species id); alternate forms (Megas,
    Deoxys-Attack, etc.) share the base num and are skipped, mirroring how the
    observation encoder keys species by their base id. Sorted by id so the file is
    stable across regenerations."""
    pokedex_path = _static("pokedex", f"gen{gen}pokedex.json")
    if not os.path.exists(pokedex_path):
        raise FileNotFoundError(f"Pokedex file not found: {pokedex_path}")

    with open(pokedex_path, "r") as f:
        dex = json.load(f)

    max_num = _GEN_MAX_SPECIES_NUM.get(gen, 100000)
    species_map = {}
    for mon_id, mon in dex.items():
        num = mon.get("num", 0)
        if num <= 0 or num > max_num:  # CAP (<=0) or a later-gen species
            continue
        # Skip non-base forms (Megas etc.); they share the base num/base stats key.
        base_species = mon.get("baseSpecies", mon_id)
        if to_id_str(base_species) != mon_id:
            continue

        bs = mon.get("baseStats", {})
        species_map[mon_id] = {
            "baseStats": {k: bs[k] for k in ("atk", "def", "hp", "spa", "spd", "spe")},
            "name": mon.get("name"),
            "num": num,
            # gen3_data.species.SpeciesData.types — the species' STAB/defensive types, UPPERCASED to the
            # TypeEncoder.TYPE_TO_IDX axis (the same axis gen3_type_chart.json + the obs type ids ride). Used
            # by the DamageOperator's expected-LATENT-defender read (marginalize a believed species' types
            # through the chart for an UNREVEALED opp mon) — the obs still reads revealed types live.
            "types": [t.upper() for t in mon.get("types", [])],
        }

    return {sid: species_map[sid] for sid in sorted(species_map)}


# --------------------------------------------------------------------------- #
# Items
# --------------------------------------------------------------------------- #

def build_items(gen):
    """Build the gen-N item map (name + num) from the Showdown items source.

    Parses `deps/pokemon-showdown/data/items.ts` the same way `build_abilities`
    parses `abilities.ts`: a per-item `\\titemid: {` block, then `name`, `num`, and
    the introduction `gen`. Items introduced after the target generation are
    filtered out. Sorted by id (like the other maps) for a stable file."""
    items_path = os.path.join(REPO_ROOT, "deps", "pokemon-showdown", "data", "items.ts")
    if not os.path.exists(items_path):
        raise FileNotFoundError(f"Items source not found: {items_path}")

    with open(items_path, "r") as f:
        content = f.read()

    items_map = {}
    for match in re.finditer(r"^\t([a-z0-9]+):\s*\{", content, re.MULTILINE):
        item_id = match.group(1)
        block = content[match.start():match.start() + 5000]
        name_match = re.search(r'name:\s*"([^"]+)"', block)
        # The obs encodes items by the item-dex `num`. `\bnum:` is required so we match the item
        # number and NOT `spritenum:` (the sprite index, which appears earlier in the block — e.g.
        # Leftovers spritenum=242 vs the true num=234). Items with no positive item-dex num (e.g.
        # Berserk Gene, a removed Gen-2 item) are dropped — they are not real gen-3 items. Cross-gen
        # aliases legitimately share a num (Cheri/PRZCureBerry, Sitrus/GoldBerry, Silk Scarf/Pink
        # Bow) — correct, they ARE the same item.
        num_match = re.search(r"\bnum:\s*(\d+)", block)
        gen_match = re.search(r"gen:\s*(\d+)", block)
        if not (name_match and num_match):
            continue
        num = int(num_match.group(1))
        if num <= 0:  # removed / non-item-dex entry — not a real gen-3 item
            continue
        item_gen = int(gen_match.group(1)) if gen_match else 0
        if item_gen > gen:  # introduced in a later generation
            continue
        items_map[item_id] = {"name": name_match.group(1), "num": num}

    return {iid: items_map[iid] for iid in sorted(items_map)}


# --------------------------------------------------------------------------- #
# Type chart  (effectiveness multipliers)
# --------------------------------------------------------------------------- #

def build_type_chart(gen):
    """Dump the gen-N type-effectiveness chart so the runtime reads it from data/
    instead of constructing it live from poke-env's `GenData` at import.

    Emitted byte-for-byte as `GenData.from_gen(gen).type_chart` produces it
    (`{DEFENDING_TYPE: {ATTACKING_TYPE: multiplier}}`, enum-name keys), so the
    consumer (`gen3_mechanics._CHART`) is identical whether it loads this file or
    the old `GenData` object — pinned by `gen3_mechanics_test.py`."""
    from poke_env.data import GenData
    return GenData.from_gen(gen).type_chart


# --------------------------------------------------------------------------- #
# Natures  (stat multipliers)
# --------------------------------------------------------------------------- #

def build_natures(gen):
    """Copy the nature table (num + the five stat multipliers) out of poke-env's
    static data so the runtime reads it from data/. Natures are gen-independent;
    `gen` only names the output file for directory consistency."""
    natures_path = _static("natures.json")
    if not os.path.exists(natures_path):
        raise FileNotFoundError(f"Natures source not found: {natures_path}")
    with open(natures_path, "r") as f:
        return json.load(f)


# --------------------------------------------------------------------------- #
# Learnset  (legal movepool)
# --------------------------------------------------------------------------- #

def build_learnset(gen):
    """Build the gen-N legal-movepool map ``{species_id: [move_id, ...]}`` (sorted).

    Derived from poke-env's static ``learnset.json``, whose per-move value is a list of
    ``<gen><method>`` source codes (e.g. ``3L`` level-up, ``3M`` TM/HM, ``3T`` tutor, ``3E`` egg,
    ``3S`` event). A move is gen-N-legal for a species iff ANY of its codes starts with the target
    generation's digit. We keep only species in the gen-N pokedex and only moves in the gen-N
    movedex (so a later-gen move id / species form can't leak in), then sort for a stable file.

    This is the *legality* primitive the move-belief prior uses to PRUNE impossible candidate moves
    — a species can't run a move it can't learn — distinct from the Smogon usage prior
    (``gen3_data.priors.moves``, how OFTEN a legal move is run). Deliberately OVER-inclusive: any
    ``"<gen>*"`` code counts (incl. ``3S`` events), so the legality gate never wrongly prunes a
    legal coverage move (the exact failure the candidate-bounding exists to avoid); the ``<2%``
    usage floor handles rarity separately."""
    learnset_path = _static("learnset.json")
    if not os.path.exists(learnset_path):
        raise FileNotFoundError(f"Learnset source not found: {learnset_path}")
    with open(learnset_path, "r") as f:
        raw = json.load(f)

    gen_prefix = str(gen)
    legal_species = set(build_species(gen))   # gen-N pokedex ids (base forms only)
    legal_moves = set(build_moves(gen))       # gen-N movedex ids (no later-gen leak)
    out = {}
    for species_id, entry in raw.items():
        if species_id not in legal_species:
            continue
        learnset = entry.get("learnset") or {}
        moves = sorted(
            mid for mid, codes in learnset.items()
            if mid in legal_moves and any(str(c).startswith(gen_prefix) for c in codes)
        )
        if moves:
            out[species_id] = moves
    return {sid: out[sid] for sid in sorted(out)}


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

_BUILDERS = {
    "abilities": ("gen{gen}_abilities.json", build_abilities),
    "moves": ("gen{gen}_moves.json", build_moves),
    "species": ("gen{gen}_species.json", build_species),
    "items": ("gen{gen}_items.json", build_items),
    "type_chart": ("gen{gen}_type_chart.json", build_type_chart),
    "natures": ("gen{gen}_natures.json", build_natures),
    "learnset": ("gen{gen}_learnset.json", build_learnset),
}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gen", type=int, default=3, help="Target generation (default: 3)")
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=list(_BUILDERS) + ["all"],
        default=["all"],
        help="Which mappings to regenerate (default: all)",
    )
    parser.add_argument("--stdout", action="store_true", help="Print JSON instead of writing files")
    args = parser.parse_args(argv)

    datasets = list(_BUILDERS) if "all" in args.datasets else args.datasets
    out_dir = os.path.join(REPO_ROOT, "data", "pokemon")

    for name in datasets:
        filename_tmpl, builder = _BUILDERS[name]
        data = builder(args.gen)

        if args.stdout:
            print(json.dumps(data, indent=4))
            continue

        out_path = os.path.join(out_dir, filename_tmpl.format(gen=args.gen))
        with open(out_path, "w") as f:
            json.dump(data, f, indent=4)
            f.write("\n")
        print(f"Wrote {len(data)} {name} -> {os.path.relpath(out_path, REPO_ROOT)}", file=sys.stderr)


if __name__ == "__main__":
    main()
