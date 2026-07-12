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


# --- gen3_item_mechanics_v1 (ability side): DMG_MOD + accMod params for src/rust_sim ---- #
# Same framework + mod-chain law as `_GEN3_ITEM_MECHANICS` (curated from the RESOLVED
# `Dex.mod('gen3')` via `src/rust_sim/harness/dump_gen3_mechanics.js`; drift-gated by its
# `--check`). The DMG_MOD family is WIRED (Phase 2); the accMod family is WIRED by the
# ACCURACY pipeline (`gen3_accuracy_pipeline_v1`). Obs-neutral (facade ignores unknown keys).
# Schema:
#   dmgMod: {mod: [num,den], fold: "basePower"|"atk"|"def"|"sourceBasePower",
#            type?/types?: the gating move type(s), pinch?: true (fires at hp <= maxhp/3),
#            whenStatused?: true, direct?: true (Hustle: an IMMEDIATE `this.modify` replace
#            — NOT chainModify — with its own rounding; ships WITH the accuracy pipeline)}
#   accMod: {op: "chain"|"multiply", mod: [num,den] (chain) | float (multiply),
#            side: "attacker" (onSourceModifyAccuracy) | "defender" (onModifyAccuracy),
#            weather?: "sandstorm" (Sand Veil), physicalTypesOnly?: true (Hustle — the
#            gen3-mod handler gates on move.type in the physical-type list, NOT category)}
#   statusImmune: {statuses: ["par"|"slp"|"psn"|"tox"|"brn"|"frz", ...], phase:
#            "setStatus"|"immunity"} (`gen3_status_immune_v1`) — the STATUS_IMMUNE class: an
#            ability that grants immunity to a specific MAJOR status. The `phase` is the
#            RESOLVED-dist handler that does the block (PROBE-settled by
#            `src/rust_sim/harness/probe_statusimmune_*.js`, drift-checked):
#              - "setStatus": an `onSetStatus` handler that RETURNS false INSIDE
#                `runEvent('SetStatus')` (Limber/Insomnia/Vital Spirit/Immunity/Water Veil) —
#                so in a clause format (gen3ou) the SetStatus event IS reached and its 2-clause
#                tie-shuffle STILL draws (the ability handler sorts into its OWN speed group by
#                its defined `speed`, so the 2 clauses stay a size-2 tie → 1 draw, UNCHANGED
#                from a normal status apply); the block itself is DRAW-FREE.
#              - "immunity": an `onImmunity(status)` handler that returns false at
#                `runStatusImmunity`, BEFORE `runEvent('SetStatus')` (Magma Armor → frz, like
#                the base Sun weather's frz `onImmunity`) — so the SetStatus event (hence its
#                clause shuffle) is NEVER reached; the block is DRAW-FREE and precedes any
#                clause draw. Immunity/tox: `runStatusImmunity('psn')` is checked for tox, but
#                Immunity's block is an `onSetStatus` (phase setStatus), NOT an onImmunity.
#            OWN TEMPO (confusion) + OBLIVIOUS (attract) block a VOLATILE via
#            `onTryAddVolatile`, NOT a major status via setStatus — so they are NOT in this
#            table (Own Tempo's confusion gate is modeled in the engine's confusion arm).
# The RESOLVED gen3 accuracy handlers (probe `harness/probe_accuracy_tohit.js`): Compound
# Eyes `chainModify(1.3)`, Sand Veil `chainModify(0.8)` in sand, Hustle `chainModify([3277,
# 4096])` for a physical-type move — NOT the base `.ts` shapes (the mod-chain law).
#
# --- gen3_ability_batch2_v1: the DRAW-BEARING "reactive" ability classes + block tail ---- #
#   contactProc: {statuses: [<id>...], chance: [num,den], sample: bool} (`gen3_ability_batch2_v1`)
#            — the CONTACT_PROC class: an `onDamagingHit` that, when the HOLDER is hit by a
#            CONTACT move, draws `randomChance(num,den)` and (on a pass) inflicts a status on the
#            ATTACKER. Static [par]/Poison Point [psn]/Flame Body [brn] = `{statuses:[<one>],
#            chance:[1,3], sample:false}` (ONE randomChance, then trySetStatus(the-one-status,
#            attacker)); Effect Spore = `{statuses:["slp","par","psn"], chance:[1,10],
#            sample:true}` (randomChance(1,10) gate → on a PASS a `sample(["slp","par","psn"])`
#            [one `random(3)`] → trySetStatus(sampled, attacker)). PROBE-settled draw model +
#            POSITION (`src/rust_sim/harness/probe_contact_proc_{rng,lands}.js` +
#            `probe_effectspore_sample.js`): the contact proc's randomChance draws INSIDE
#            runEvent('DamagingHit') (gen<5, battle-actions.ts:982) which fires AFTER the move's
#            OWN secondaries() (line 957) — so the ORDER is [move secondary random(100)] THEN
#            [contact-proc randomChance]. It draws on a contact hit that dealt damage (incl.
#            behind a SUBSTITUTE, and on a KO — the DamagingHit fires on the damaged/KO'd
#            target) but NOT on a non-contact move / a miss / an immune hit. The status lands on
#            the ATTACKER (source) with the gen-3 type/ability/already-statused gates; in gen3ou
#            trySetStatus draws the 2-clause SetStatus shuffle (draw-free in the e2e customgame).
#            CONSUMED by `turn.rs`'s `apply_contact_proc`. (Cute Charm draws the same randomChance
#            but adds the `attract` volatile — the separate `contactAttract` row, batch 4.)
#   contactRecoil: true (`gen3_ability_batch2_v1`) — Rough Skin: an `onDamagingHit` that, when
#            the holder is hit by a CONTACT move, deals `baseMaxhp/16` recoil to the ATTACKER —
#            DRAW-FREE (`this.damage`, no PRNG; probe-verified identical draw count to a no-op).
#   blocksSound: true (`gen3_ability_batch2_v1`) — Soundproof: an `onTryHit` that makes the
#            holder IMMUNE to a SOUND move (`move.flags.sound`). Of the MODELED moves the sound
#            ones are Sing / Grass Whistle (sleep) + Roar (phaze) + Perish Song (unmodeled). A
#            blocked sound move draws its ACCURACY then reports `-immune` (the same draw count +
#            short-circuit as a type-immune move; probe `probe_block_abilities_rng.js`).
#   blocksExplosion: true (`gen3_ability_batch2_v1`) — Damp: an `onAnyTryMove` that CANCELS
#            Explosion / Self-Destruct (any side's) at runEvent('TryMove') — BEFORE the self-KO
#            faint AND before the accuracy roll (battle-actions.ts:412 precedes the selfdestruct
#            faint at 422). The user does NOT self-KO; the move draws NOTHING (a big draw-count
#            drop vs a normal Explosion's acc+crit+dmg — probe-verified: only Quick Claw draws).
#   blocksPhazeDrag: true (`gen3_ability_batch2_v1`) — Suction Cups: an `onDragOut` that returns
#            null → a Roar/Whirlwind into the holder does NOT drag it (like Protect-blocks-phaze):
#            the phaze draws its accuracy, then the drag `sample` is NOT drawn (`-activate Suction
#            Cups`, the holder stays). Draw-free block past the accuracy already drawn.
#   synchronize: true (`gen3_ability_batch2_v1`) — Synchronize: an `onAfterSetStatus` that, when
#            the holder is inflicted a MAJOR status by a FOE source (a status MOVE or a damaging
#            move's SECONDARY), REFLECTS it back to that source. slp/frz are EXEMPT (no reflect);
#            tox reflects as psn. The reflected trySetStatus is DRAW-FREE in gen3customgame (the
#            e2e format — probe `probe_synchronize_rng.js`: identical draws to a no-op control);
#            in gen3ou it draws the reflected status's own 2-clause SetStatus shuffle. CONSUMED by
#            `turn.rs::try_set_status` (the single status choke point, source-threaded).
_GEN3_ABILITY_MECHANICS = {
    "blaze": {"dmgMod": {"mod": [3, 2], "fold": "basePower", "type": "Fire", "pinch": True}},
    "overgrow": {"dmgMod": {"mod": [3, 2], "fold": "basePower", "type": "Grass", "pinch": True}},
    "swarm": {"dmgMod": {"mod": [3, 2], "fold": "basePower", "type": "Bug", "pinch": True}},
    "torrent": {"dmgMod": {"mod": [3, 2], "fold": "basePower", "type": "Water", "pinch": True}},
    "guts": {"dmgMod": {"mod": [3, 2], "fold": "atk", "whenStatused": True}},
    "hugepower": {"dmgMod": {"mod": [2, 1], "fold": "atk"}},
    "purepower": {"dmgMod": {"mod": [2, 1], "fold": "atk"}},
    "hustle": {
        "dmgMod": {"mod": [3, 2], "fold": "atk", "direct": True},
        "accMod": {"op": "chain", "mod": [3277, 4096], "side": "attacker", "physicalTypesOnly": True},
    },
    "marvelscale": {"dmgMod": {"mod": [3, 2], "fold": "def", "whenStatused": True}},
    "thickfat": {"dmgMod": {"mod": [1, 2], "fold": "sourceBasePower", "types": ["Ice", "Fire"]}},
    # ACCURACY class (chainModify — accumulated into the ModifyAccuracy modifier).
    "compoundeyes": {"accMod": {"op": "chain", "mod": [13, 10], "side": "attacker"}},
    "sandveil": {"accMod": {"op": "chain", "mod": [8, 10], "side": "defender", "weather": "sandstorm"}},
    # STATUS_IMMUNE class (`gen3_status_immune_v1`) — immunity to a specific major status.
    # The 5 `setStatus`-phase members block via `onSetStatus` (inside runEvent('SetStatus'),
    # after the clause shuffle drew); Magma Armor blocks earlier via `onImmunity('frz')`.
    "limber": {"statusImmune": {"statuses": ["par"], "phase": "setStatus"}},
    "insomnia": {"statusImmune": {"statuses": ["slp"], "phase": "setStatus"}},
    "vitalspirit": {"statusImmune": {"statuses": ["slp"], "phase": "setStatus"}},
    "immunity": {"statusImmune": {"statuses": ["psn", "tox"], "phase": "setStatus"}},
    "waterveil": {"statusImmune": {"statuses": ["brn"], "phase": "setStatus"}},
    "magmaarmor": {"statusImmune": {"statuses": ["frz"], "phase": "immunity"}},
    # CRIT_IMMUNE class (`gen3_ability_batch1_v1`) — Battle Armor / Shell Armor block a
    # critical hit. The RESOLVED gen3 dist gives both `onCriticalHit = false` (a BOOLEAN, NOT
    # a fn). PROBE-settled draw model (`harness/probe_critimmune_rng.js`): the crit
    # `randomChance(1,critMult)` is DRAWN as normal (battle-actions.ts:1645), THEN — only if it
    # succeeded — `runEvent('CriticalHit')` (line 1650) reads the `onCriticalHit=false` handler
    # and OVERRIDES the crit boolean to false. So the roll count is UNCHANGED (identical to a
    # no-op control across every seed); only the resulting crit is forced false → DRAW-FREE.
    "battlearmor": {"critImmune": True},
    "shellarmor": {"critImmune": True},
    # WEATHER_SPEED class (`gen3_ability_batch1_v1`) — Chlorophyll (spe×2 in Sun) / Swift Swim
    # (spe×2 in Rain), the resolved `onModifySpe chainModify(2)` gated on effectiveWeather().
    # A SPEED MODIFIER feeding getActionSpeed → the action-order + eachEvent + residual
    # tie-shuffles (PROBE-settled `harness/probe_weather_speed_tie.js`: a Chlorophyll mon in
    # sun ties/orders on its ×2 speed, drawing the action-order tie-shuffle exactly like the
    # sim). DRAW-FREE itself (only the INPUT to an existing shuffle changes). The `weather` is
    # the RESOLVED effectiveWeather() id (Drought/Sunny Day → "sunnyday"; Drizzle/Rain Dance →
    # "raindance").
    "chlorophyll": {"weatherSpeed": {"weather": "sunnyday"}},
    "swiftswim": {"weatherSpeed": {"weather": "raindance"}},
    # WEATHER_NEGATE class (`gen3_ability_batch1_v1`) — Cloud Nine / Air Lock. While the holder
    # is active, `field.effectiveWeather()` returns '' (the `suppressWeather` gate on the
    # holder's abilityState.ending flag), so weather's damage/chip/speed effects are all
    # suppressed. DRAW-FREE (changes deterministic values — the sand-chip HP, the weather-speed
    # ×2 — never a roll; PROBE-settled `harness/probe_ability_batch_drawmodel.js`: identical
    # draw count to a no-op while under sand, only the STATE differs). A boolean flag (both
    # members share the identical mechanism).
    "cloudnine": {"weatherNegate": True},
    "airlock": {"weatherNegate": True},
    # CONTACT_PROC class (`gen3_ability_batch2_v1`) — an onDamagingHit that reacts to a CONTACT
    # move by drawing randomChance(chance) and (on a pass) statusing the ATTACKER. The single-
    # status members roll `randomChance(1,3)`; Effect Spore rolls `randomChance(1,10)` then (on a
    # pass) `sample(["slp","par","psn"])`. (Cute Charm rides the separate `contactAttract` row —
    # batch 4.)
    "static": {"contactProc": {"statuses": ["par"], "chance": [1, 3], "sample": False}},
    "poisonpoint": {"contactProc": {"statuses": ["psn"], "chance": [1, 3], "sample": False}},
    "flamebody": {"contactProc": {"statuses": ["brn"], "chance": [1, 3], "sample": False}},
    "effectspore": {"contactProc": {"statuses": ["slp", "par", "psn"], "chance": [1, 10], "sample": True}},
    # CONTACT_ATTRACT (`gen3_ability_batch4_v1`) — Cute Charm: the SAME DamagingHit-position
    # contact roll as the contactProc family (`randomChance(1,3)` drawn UNCONDITIONALLY on a
    # damaging contact hit — the gender gate lives INSIDE the attract volatile's onStart, which
    # fails DRAW-FREE for same-gender / genderless pairs), but on a pass it adds the ATTRACT
    # volatile to the ATTACKER instead of a status. PROBE-settled
    # (`src/rust_sim/harness/probe_cutecharm_attract_rng.js`): attract onBeforeMove priority 2
    # (confusion 3 > attract 2 > par 1) emits `-activate` ALWAYS then draws `randomChance(1,2)`
    # (cant on pass); the volatile clears when the SOURCE leaves the field (onUpdate) or the
    # HOLDER switches out; it sticks even when the attacker has a Substitute up.
    "cutecharm": {"contactAttract": {"chance": [1, 3]}},
    # CONTACT recoil (`gen3_ability_batch2_v1`) — Rough Skin: baseMaxhp/16 to the attacker on a
    # contact hit, DRAW-FREE.
    "roughskin": {"contactRecoil": True},
    # BLOCK abilities (`gen3_ability_batch2_v1`).
    "soundproof": {"blocksSound": True},   # onTryHit → -immune to a move.flags.sound move
    "damp": {"blocksExplosion": True},     # onAnyTryMove → cancels Explosion/Self-Destruct (no self-KO)
    "suctioncups": {"blocksPhazeDrag": True},  # onDragOut → null (a phaze into the holder draws no sample)
    # SYNCHRONIZE (`gen3_ability_batch2_v1`) — reflect a foe-inflicted major status back to the
    # source (slp/frz exempt; tox→psn). Draw-free in gen3customgame (the e2e format).
    "synchronize": {"synchronize": True},
    # SHED SKIN (`gen3_berry_trace_shedskin_v1` batch-3) — a residual (order 10 subOrder 3,
    # the Speed Boost / Rain Dish slot) that, while the holder has a MAJOR status, draws ONE
    # `randomChance(33,100)` and on a pass cures it (BEFORE the same-mon status DoT at
    # subOrder 6 — a cure turn takes NO chip; probe `probe_trace_shedskin_rng.js`).
    # Confusion is NOT cured; an unstatused holder draws NOTHING.
    "shedskin": {"shedSkin": True},
    # TRACE (`gen3_berry_trace_shedskin_v1` batch-3) — the gen3-RESOLVED onStart (mod-chain
    # law: the base/gen4 seek/notrace machinery is REPLACED): at switch-in it draws
    # `side.randomFoe()` (ONE `sample` — it DRAWS even with a single foe, the phaze-n=1
    # gotcha) and copies the foe's CURRENT ability with NO guard (No Ability / Wonder Guard /
    # a traced-through ability all copy). gen3 `setAbility` does NOT fire the copied onStart
    # (`gen > 3` gate: a Traced Intimidate/Drought does nothing on copy); the copied
    # ability's passive effects are LIVE; switch-out reverts to Trace (re-entry re-traces,
    # a new draw). Probe `probe_trace_shedskin_rng.js`.
    "trace": {"trace": True},
}


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
                entry = {"name": ability_name, "num": ab_num}
                if gen == 3 and ab_id in _GEN3_ABILITY_MECHANICS:
                    entry.update(_GEN3_ABILITY_MECHANICS[ab_id])
                abilities_map[ab_id] = entry

    if gen == 3:
        missing = [aid for aid in _GEN3_ABILITY_MECHANICS if aid not in abilities_map]
        if missing:  # a curated id that no longer derives = silent data rot — fail loud
            raise ValueError(f"curated gen3 ability mechanics not in the derived map: {missing}")

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


def _secondary_boosts_for_block(block):
    """Emit the STRUCTURED boost entries a single `secondary`/`secondaries[i]` block carries.

    The flattened `secondaryEffects` ({col: percent}) collapses every foe stat-drop to
    `foe_statdrop` and every self-raise to `self_boost`, LOSING which stat, how many stages,
    and the target — the `src/rust_sim` engine can't apply the real boost from it. This carries
    that lost spec verbatim (mirroring Showdown's own `secondary.boosts` / `secondary.self.boosts`):
      - a foe stat-drop  `secondary.boosts`      → {chance, target: "foe",  boosts}
      - a self stat-raise `secondary.self.boosts` → {chance, target: "self", boosts}
    A block can carry BOTH (none in gen3 do), so both are emitted when present. DRAW-FREE in the
    sim (each is the SAME one `random(100)` the secondary already draws); this records only the
    apply spec. Stat keys are the Showdown stat ids (atk/def/spa/spd/spe/accuracy/evasion)."""
    if not isinstance(block, dict):
        return []
    chance = block.get("chance")
    chance = 100 if chance is None else int(chance)
    out = []
    foe = block.get("boosts")
    if isinstance(foe, dict) and foe:
        out.append({"chance": chance, "target": "foe", "boosts": {k: int(v) for k, v in foe.items()}})
    self_block = block.get("self")
    if isinstance(self_block, dict) and isinstance(self_block.get("boosts"), dict) and self_block["boosts"]:
        out.append({"chance": chance, "target": "self", "boosts": {k: int(v) for k, v in self_block["boosts"].items()}})
    return out


def _secondary_boosts(entry):
    """Walk `secondary` + every `secondaries[i]`, returning the structured boost spec list (the
    foe stat-drop / self stat-raise the flat `secondaryEffects` discards). Empty list ⇒ the key
    is OMITTED (additive, only-when-present — the `critRatio` precedent), so the file diff is just
    the ~24 boost-bearing moves and the obs golden is unchanged (the facade ignores the key)."""
    out = []
    out.extend(_secondary_boosts_for_block(entry.get("secondary")))
    for block in (entry.get("secondaries") or []):
        out.extend(_secondary_boosts_for_block(block))
    return out


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


# A self-boost stat the `src/rust_sim` engine can apply with NO side effect on its
# draw model: the offensive/defensive battle stats. accuracy/evasion are DELIBERATELY
# EXCLUDED — the engine's accuracy roll (`random_chance(accuracy, 100)`) ignores the
# evasion/accuracy boost tables, so a +evasion (Double Team/Minimize) or +accuracy
# state would silently diverge the next time a move's accuracy is rolled against it.
# Those moves stay UNMODELED (fail-loud in the engine), so they are not emitted here.
_SELF_BOOST_STATS = frozenset({"atk", "def", "spa", "spd", "spe"})


def _self_boosts(entry):
    """The STRUCTURED PRIMARY self-boost spec for a PURE setup move (gen3_setup_moves_v1):
    a ``target: self`` Status move whose ENTIRE effect is its declarative top-level
    ``boosts`` map (Swords Dance ``{atk:2}``, Dragon Dance ``{atk:1,spe:1}``, Calm
    Mind ``{spa:1,spd:1}``, …). Returns the ``{stat: stages}`` map, ELSE ``None`` (the
    key is then OMITTED — additive, only-when-present, like ``secondaryBoosts``), so the
    file diff is just the ~17 pure setup moves and the obs facade ignores it.

    A move qualifies ONLY when it has NO other effect that would change the engine's
    draw model or state beyond the boost, so the `src/rust_sim` self-boost path is
    bit-for-bit DRAW-FREE:
      - ``target == "self"`` and ``category == "Status"`` (bp 0) — a pure setup move;
      - a declarative top-level ``boosts`` dict, every entry POSITIVE (a setup raise,
        never a self-debuff drawback);
      - every boosted stat is in ``_SELF_BOOST_STATS`` (no accuracy/evasion — see above);
      - NO ``volatileStatus`` (excludes Defense Curl's ``defensecurl`` flag + Minimize's
        ``minimize``), NO ``self``/``secondary`` block, NO ``onHit``/``onTryHit``/``heal``
        (excludes Belly Drum's HP-cost callback, which carries no declarative ``boosts``).
    Curse (``target: normal``, type-conditional onHit boost) and Belly Drum are excluded
    by these gates — same moves the engine's modeled set excludes (kept fail-loud)."""
    if entry.get("target") != "self":
        return None
    if entry.get("basePower"):  # a damaging move is not a pure setup move
        return None
    boosts = entry.get("boosts")
    if not isinstance(boosts, dict) or not boosts:
        return None
    if any(v <= 0 for v in boosts.values()):
        return None
    if any(stat not in _SELF_BOOST_STATS for stat in boosts):
        return None
    # Any other declarative effect disqualifies the pure-setup classification.
    if entry.get("volatileStatus") or entry.get("self") or entry.get("secondary"):
        return None
    if entry.get("onHit") or entry.get("onTryHit") or entry.get("heal"):
        return None
    return {stat: int(stages) for stat, stages in boosts.items()}


def _self_drops(entry):
    """The top-level ``move.self.boosts`` SELF STAT-DROP spec on a DAMAGING move
    (``gen3_move_coverage_batch1_v1`` — Overheat ``{spa:-2}``, Superpower ``{atk:-1,
    def:-1}``, Draco Meteor / Leaf Storm / Psycho Boost ``{spa:-2}``). Returns the
    ``{stat: stages}`` map (all stages NEGATIVE), ELSE ``None`` (the key is then
    OMITTED — additive, only-when-present, like ``selfBoosts``), so the file diff is
    just the handful of self-drop moves and the obs facade ignores it.

    Gated to a DAMAGING move (bp > 0) whose ``self`` block is EXACTLY a ``boosts`` map
    of self stat-DROPS in ``_SELF_BOOST_STATS`` — no accuracy/evasion, no other key on
    the ``self`` block. gen3 ``self.boosts`` are UNCONDITIONAL (no ``chance``/``selfDrops``
    ``random(100)`` — probe-verified draw-free), applied to the USER after the hit, so the
    ``src/rust_sim`` engine applies them via a draw-free ``boost()`` (±6 clamp). Rapid
    Spin's ``self`` block is ``{}`` (its clear is an ``onAfterHit``, not a self-boost) →
    excluded (no ``boosts`` key)."""
    if not entry.get("basePower"):  # a self-DROP rides a damaging move
        return None
    self_block = entry.get("self")
    if not isinstance(self_block, dict):
        return None
    boosts = self_block.get("boosts")
    if not isinstance(boosts, dict) or not boosts:
        return None
    # Only a pure self.boosts block (no onHit/volatileStatus/etc. on `self`).
    if any(k != "boosts" for k in self_block):
        return None
    if any(v >= 0 for v in boosts.values()):
        return None
    if any(stat not in _SELF_BOOST_STATS for stat in boosts):
        return None
    return {stat: int(stages) for stat, stages in boosts.items()}


def _stat_drop_boosts(entry):
    """The declarative FOE STAT-DROP spec for a standalone STAT-DROP STATUS MOVE
    (``gen3_move_coverage_batch2_v1`` — Screech ``{def:-2}``, Charm ``{atk:-2}``,
    Metal Sound ``{spd:-2}``, Feather Dance ``{atk:-2}``, Tickle ``{atk:-1,def:-1}``,
    Fake Tears ``{spd:-2}``). Returns the ``{stat: stages}`` map (all stages NEGATIVE),
    ELSE ``None`` (the key is then OMITTED — additive, only-when-present, like
    ``selfBoosts``/``selfDrops``), so the file diff is just the handful of stat-drop
    STATUS moves and the obs facade ignores it.

    Gated to a foe-targeting STATUS move (bp 0) whose ENTIRE effect is its declarative
    top-level ``boosts`` map of FOE stat-DROPS in ``_SELF_BOOST_STATS`` (no accuracy/
    evasion — the ``src/rust_sim`` engine's evasion is not folded into the accuracy roll,
    so an accuracy/evasion drop would silently desync) — mirroring ``_self_boosts`` but for
    a ``target: normal`` move with NEGATIVE stages and no other effect (NO ``status``/
    ``volatileStatus``/``self``/``secondary``/``onHit``/``onTryHit``/``heal``). The
    ``src/rust_sim`` engine draws the accuracy roll then applies these draw-free via
    ``boost()`` on the FOE with the Clear Body / White Smoke / Hyper Cutter / Keen Eye
    ``onTryBoost`` immunity gates (``apply_secondary_boost``)."""
    if entry.get("basePower"):  # a damaging move's stat-drop rides `secondary`, not here
        return None
    if entry.get("category") != "Status":
        return None
    if entry.get("target") not in ("normal", "adjacentFoe", "any"):
        return None
    boosts = entry.get("boosts")
    if not isinstance(boosts, dict) or not boosts:
        return None
    if any(v >= 0 for v in boosts.values()):  # a stat-DROP move (never a foe raise)
        return None
    if any(stat not in _SELF_BOOST_STATS for stat in boosts):
        return None
    # Any other declarative effect disqualifies the pure stat-drop classification.
    if entry.get("status") or entry.get("volatileStatus") or entry.get("self") or entry.get("secondary"):
        return None
    if entry.get("onHit") or entry.get("onTryHit") or entry.get("heal"):
        return None
    # A self-KO move (Memento's `selfdestruct: 'ifHit'` faints the user) is NOT a pure
    # stat-drop — its self-faint is a separate unmodeled mechanic → exclude (key omitted).
    if entry.get("selfdestruct"):
        return None
    return {stat: int(stages) for stat, stages in boosts.items()}


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

        move_dict = {
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
            # --- PP tracking (gen3_pp_tracking_v1) ---
            # The move's BASE PP (before PP-ups). Showdown computes a moveslot's
            # in-battle PP as calculatePP(move, ppUps) = pp * (5 + ppUps) / 5, and the
            # `Pokemon` constructor defaults ppUps to 3 (max) for every non-`noPPBoosts`
            # move — so a mon's in-battle PP is base_pp * 8 / 5 (gen3). `noPPBoosts` moves
            # (Struggle etc.) keep their raw pp and get 0 PP-ups. GPU-side / port-side only
            # (the obs facade ignores it, like critRatio); it exists so the src/rust_sim
            # Rust port can track per-move PP + force Struggle at 0 PP, bit-for-bit.
            "pp": int(entry.get("pp", 0) or 0),
            "noPPBoosts": bool(entry.get("noPPBoosts", False)),
            # --- gen3_ability_batch2_v1: move FLAGS the CONTACT_PROC / Soundproof classes gate on ---
            # `contact` — the move makes physical contact (Body Slam/Tackle/Crunch/…); the
            # CONTACT_PROC abilities (Static/Poison Point/Flame Body/Effect Spore/Rough Skin) +
            # Cute Charm/Color Change react ONLY to a contact move. `sound` — the move is
            # sound-based (Sing/Grass Whistle/Roar/Perish Song); Soundproof is IMMUNE to it.
            # GPU-side / port-side only (the obs facade ignores them, like `pp`/`critRatio`), so
            # obs-neutral; they exist for the `src/rust_sim` port's contact-proc + Soundproof gates.
            "contact": bool(flags.get("contact")),
            "sound": bool(flags.get("sound")),
        }
        # gen3 high-crit moves carry critRatio (2 in gen3: Slash, Crabhammer,
        # Aircutter, Blaze Kick, Leaf Blade, …); normal moves omit it and the dex
        # defaults to 1. Conditional, so the data-file diff is just the ~dozen
        # high-crit moves. Used by the turn engine's gen3 crit-ratio (1/16 vs 1/8).
        if entry.get("critRatio"):
            move_dict["critRatio"] = int(entry["critRatio"])
        # gen3 secondary STAT-boost spec (only-when-present, like critRatio): the
        # structured (stat, stages, target=foe|self) the flat secondaryEffects discards,
        # so the rust_sim engine can apply the real foe stat-drop / self stat-raise.
        # GPU-side neutral (the obs facade ignores it). Diff = just the ~24 boost moves.
        secondary_boosts = _secondary_boosts(entry)
        if secondary_boosts:
            move_dict["secondaryBoosts"] = secondary_boosts
        # gen3_setup_moves_v1: the PRIMARY self-boost spec for a PURE setup move (Swords
        # Dance / Dragon Dance / Calm Mind / Agility / …) — only-when-present, like
        # secondaryBoosts. The `{stat: stages}` the `src/rust_sim` engine applies on the
        # USER (draw-free, ±6 clamp). Obs-neutral (the facade ignores it); diff = the ~17
        # pure setup moves. Moves with an extra effect (Defense Curl/Minimize volatile,
        # evasion boost, Belly Drum HP cost, Curse) are excluded → key omitted.
        self_boosts = _self_boosts(entry)
        if self_boosts:
            move_dict["selfBoosts"] = self_boosts
        # gen3_move_coverage_batch1_v1: the top-level `move.self.boosts` SELF STAT-DROP
        # on a damaging move (Overheat -2 SpA, Superpower -1 Atk/-1 Def) — only-when-
        # present, like selfBoosts. The `{stat: stages}` (negative) the `src/rust_sim`
        # engine applies draw-free on the USER after the hit. Obs-neutral (the facade
        # ignores it); diff = the handful of self-drop moves.
        self_drops = _self_drops(entry)
        if self_drops:
            move_dict["selfDrops"] = self_drops
        # gen3_move_coverage_batch2_v1: the declarative FOE STAT-DROP for a standalone
        # stat-drop STATUS move (Screech -2 Def, Charm -2 Atk, Metal Sound -2 SpD,
        # Feather Dance -2 Atk, Tickle -1 Atk/-1 Def, Fake Tears -2 SpD) — only-when-
        # present, like selfDrops. The `{stat: stages}` (negative) the `src/rust_sim`
        # engine applies draw-free on the FOE (Clear Body/Hyper Cutter/etc. gated) after
        # its accuracy roll. Obs-neutral (the facade ignores it); diff = the ~6 stat-drop
        # status moves.
        stat_drop_boosts = _stat_drop_boosts(entry)
        if stat_drop_boosts:
            move_dict["statDropBoosts"] = stat_drop_boosts
        moves_map[move_id] = move_dict

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
        entry = {
            "baseStats": {k: bs[k] for k in ("atk", "def", "hp", "spa", "spd", "spe")},
            "name": mon.get("name"),
            "num": num,
            # gen3_data.species.SpeciesData.types — the species' STAB/defensive types, UPPERCASED to the
            # TypeEncoder.TYPE_TO_IDX axis (the same axis gen3_type_chart.json + the obs type ids ride). Used
            # by the DamageOperator's expected-LATENT-defender read (marginalize a believed species' types
            # through the chart for an UNREVEALED opp mon) — the obs still reads revealed types live.
            "types": [t.upper() for t in mon.get("types", [])],
        }
        # Showdown's pokedex carries a FIXED max-HP override for a handful of species
        # (Gen 3: only Shedinja, maxHP 1). Pokemon.setSpecies overwrites the computed
        # HP stat with this. Carry it through ONLY when present, so normal species stay
        # unchanged; the obs facade ignores the key (it reads base stats), while a
        # stat-computing consumer (e.g. the rust_sim port) applies it.
        if "maxHP" in mon:
            entry["maxHP"] = mon["maxHP"]
        species_map[mon_id] = entry

    return {sid: species_map[sid] for sid in sorted(species_map)}


# --------------------------------------------------------------------------- #
# Items
# --------------------------------------------------------------------------- #

# --- gen3_item_mechanics_v1: structured item mechanics for the src/rust_sim port -------- #
# The DATA-DRIVEN MECHANICS FRAMEWORK's per-item parameters (class TYPE_BOOST /
# SPECIES_STAT / CHOICE), consumed by the Rust engine's generic damage-path lookup
# (`turn.rs::resolve_atk_stat_mods` + `resolve_def_stat_mods`) instead of a hardcoded
# per-id match arm. All fields are ADDITIVE + obs-neutral (the `gen3_data` facade ignores
# unknown keys — the `critRatio`/`selfBoosts` precedent).
#
# THE MOD-CHAIN LAW (why this is a curated table, not a regex derivation): item handlers
# live in JavaScript callbacks resolved through the WHOLE mod chain (gen3 -> gen4 -> ... ->
# base), where later mods REPLACE and DELETE handlers. Light Ball is the cautionary tale:
# base = Atk+SpA x2, the gen4 mod REWRITES it to onBasePower x2, the gen3 mod REWRITES it
# again to SpA-ONLY x2. A single-file regex extraction is therefore GIGO. This table is
# derived from the RESOLVED `Dex.mod('gen3')` by
# `src/rust_sim/harness/dump_gen3_mechanics.js` (which also probes each handler's fold
# point + multiplier), and every regeneration is drift-gated by
# `node src/rust_sim/harness/dump_gen3_mechanics.js --check` (committed JSON vs the
# resolved dist) + the class-sweep golden `src/rust_sim/tests/item_mods_test.rs` (real sim
# battles — the final oracle).
#
# Schema (all multipliers are exact [num, den] rationals — bit-identical to Showdown's
# 4096 fixed-point chainModify; trunc(num*4096/den) == trunc(float*4096) for every entry):
#   typeBoost: {type, mod, fold} — an offensive boost on the matching MOVE TYPE.
#     fold="stat":            gen3-mod onModifyAtk/onModifySpA chainModify (x1.1 family +
#                             Sea Incense x1.05) — folds into the OFFENSIVE STAT chain.
#     fold="basePower":       base-data onBasePower chainModify (the 4 gen4-named incenses,
#                             x4915/4096 ~= 1.2 — NOT x1.1!) — folds into the BASE-POWER
#                             chain (one accumulated modifier, rounded once).
#     fold="basePowerDirect": base-data onBasePower `return basePower * 1.1` (Pink Bow /
#                             Polkadot Bow) — a DIRECT float return that REPLACES the
#                             event's relayVar; the non-integer result SKIPS runEvent's
#                             final chain-modifier application and is floored by
#                             clampIntRange.
#   statMods: {stat: [num, den], ...} — stat-event multipliers (onModifyAtk/SpA/Def/SpD).
#     Offensive stats fold into the attacker's stat chain; def/spd fold into the DEFENDER's
#     stat chain (the ModifyDef/ModifySpD event — after the boost table, before the
#     gen<=4 Explosion Def-halve).
#   onlySpecies: [species_id, ...] — the species gate (Thick Club etc.).
#   untransformedOnly: true — Metal Powder's `!pokemon.transformed` guard.
#   choice: true — Choice Band (the move-lock; its x1.5 Atk rides statMods).
_GEN3_ITEM_MECHANICS = {
    # CHOICE
    "choiceband": {"choice": True, "statMods": {"atk": [3, 2]}},
    # TYPE_BOOST — the gen3-mod stat fold (x1.1; Sea Incense x1.05)
    "blackbelt": {"typeBoost": {"type": "Fighting", "mod": [11, 10], "fold": "stat"}},
    "blackglasses": {"typeBoost": {"type": "Dark", "mod": [11, 10], "fold": "stat"}},
    "charcoal": {"typeBoost": {"type": "Fire", "mod": [11, 10], "fold": "stat"}},
    "dragonfang": {"typeBoost": {"type": "Dragon", "mod": [11, 10], "fold": "stat"}},
    "hardstone": {"typeBoost": {"type": "Rock", "mod": [11, 10], "fold": "stat"}},
    "magnet": {"typeBoost": {"type": "Electric", "mod": [11, 10], "fold": "stat"}},
    "metalcoat": {"typeBoost": {"type": "Steel", "mod": [11, 10], "fold": "stat"}},
    "miracleseed": {"typeBoost": {"type": "Grass", "mod": [11, 10], "fold": "stat"}},
    "mysticwater": {"typeBoost": {"type": "Water", "mod": [11, 10], "fold": "stat"}},
    "nevermeltice": {"typeBoost": {"type": "Ice", "mod": [11, 10], "fold": "stat"}},
    "poisonbarb": {"typeBoost": {"type": "Poison", "mod": [11, 10], "fold": "stat"}},
    "seaincense": {"typeBoost": {"type": "Water", "mod": [21, 20], "fold": "stat"}},
    "sharpbeak": {"typeBoost": {"type": "Flying", "mod": [11, 10], "fold": "stat"}},
    "silkscarf": {"typeBoost": {"type": "Normal", "mod": [11, 10], "fold": "stat"}},
    "silverpowder": {"typeBoost": {"type": "Bug", "mod": [11, 10], "fold": "stat"}},
    "softsand": {"typeBoost": {"type": "Ground", "mod": [11, 10], "fold": "stat"}},
    "spelltag": {"typeBoost": {"type": "Ghost", "mod": [11, 10], "fold": "stat"}},
    "twistedspoon": {"typeBoost": {"type": "Psychic", "mod": [11, 10], "fold": "stat"}},
    # TYPE_BOOST — the base-data DIRECT base-power fold (the gen2 bows, x1.1)
    "pinkbow": {"typeBoost": {"type": "Normal", "mod": [11, 10], "fold": "basePowerDirect"}},
    "polkadotbow": {"typeBoost": {"type": "Normal", "mod": [11, 10], "fold": "basePowerDirect"}},
    # TYPE_BOOST — the base-data base-power chain fold (the gen4-named incenses, x~1.2)
    "oddincense": {"typeBoost": {"type": "Psychic", "mod": [4915, 4096], "fold": "basePower"}},
    "rockincense": {"typeBoost": {"type": "Rock", "mod": [4915, 4096], "fold": "basePower"}},
    "roseincense": {"typeBoost": {"type": "Grass", "mod": [4915, 4096], "fold": "basePower"}},
    "waveincense": {"typeBoost": {"type": "Water", "mod": [4915, 4096], "fold": "basePower"}},
    # SPECIES_STAT (gen3-resolved semantics — e.g. gen3 Light Ball is SpA-ONLY x2)
    "deepseascale": {"statMods": {"spd": [2, 1]}, "onlySpecies": ["clamperl"]},
    "deepseatooth": {"statMods": {"spa": [2, 1]}, "onlySpecies": ["clamperl"]},
    "lightball": {"statMods": {"spa": [2, 1]}, "onlySpecies": ["pikachu"]},
    "metalpowder": {"statMods": {"def": [2, 1]}, "onlySpecies": ["ditto"], "untransformedOnly": True},
    "souldew": {"statMods": {"spa": [3, 2], "spd": [3, 2]}, "onlySpecies": ["latias", "latios"]},
    "thickclub": {"statMods": {"atk": [2, 1]}, "onlySpecies": ["cubone", "marowak"]},
    # ACCURACY_ITEM (gen3_accuracy_pipeline_v1) — DEFENDER-side onModifyAccuracy DIRECT
    # `accuracy * float`. The gen3 mod REWRITES both from the base `chainModify([3686,4096])`
    # to an exact float multiply (the mod-chain law: probe the RESOLVED dist). The float is
    # stored verbatim so Rust and JS parse identical f64 bits (a rational 9/10 differs in the
    # last bit for many accuracy values — proven).
    "brightpowder": {"accMod": {"op": "multiply", "mod": 0.9, "side": "defender"}},
    "laxincense": {"accMod": {"op": "multiply", "mod": 0.95, "side": "defender"}},
    # PROC_ITEM (`gen3_ability_batch4_v1`) — the two draw-bearing proc items, PROBE-settled
    # against the RESOLVED `Dex.mod('gen3')` (`src/rust_sim/harness/probe_kingsrock_rng.js` /
    # `probe_kingsrock_order_rng.js` / `probe_focusband_rng.js` / `probe_focusband_confusion_rng.js`):
    #   flinchSecondary — King's Rock: `onModifyMove` PUSHES `{chance: 10, volatileStatus:
    #       "flinch"}` onto `move.secondaries` for the moves in `moves` (the resolved literal
    #       list, EXECUTION-derived by dump_gen3_mechanics.js — deduped: the sim lists the 17
    #       Hidden Powers under the one id "hiddenpower"; the port canonicalizes its typed HP
    #       ids to the bare id for this lookup). It is an ORDINARY TRAILING secondary: rolled
    #       (ONE `random(100)`) AFTER the move's own secondary, BEFORE the foe's contact proc;
    #       Serene Grace doubles it to 20; Shield Dust filters it (NO draw); behind a Substitute
    #       it draws but does not apply; a fixed-damage listed move (Seismic Toss) still procs.
    #   surviveLethal — Focus Band: `onDamage` (priority -40) `randomChance(1,10) && damage >=
    #       target.hp && effect.effectType === 'Move'` — the && ORDER means the roll DRAWS
    #       FIRST, on EVERY Damage event into the holder (move hits, burn/sand/leech chips,
    #       Spikes, recoil, confusion self-hits — but NOT sub-absorbed hits, which never run
    #       the holder's Damage event); the survive-at-1-HP fires only for a lethal MOVE hit
    #       (a lethal chip still faints; the confusion self-hit counts as a Move).
    "kingsrock": {"flinchSecondary": {"chance": 10, "moves": [
        "aerialace", "aeroblast", "aircutter", "armthrust", "barrage", "beatup", "bide",
        "bind", "blastburn", "bonemerang", "bonerush", "bounce", "brickbreak", "bulletseed",
        "clamp", "cometpunch", "crabhammer", "crosschop", "cut", "dig", "dive", "doubleedge",
        "doublekick", "doubleslap", "dragonbreath", "dragonclaw", "dragonrage", "drillpeck",
        "earthquake", "eggbomb", "endeavor", "eruption", "explosion", "extremespeed",
        "falseswipe", "feintattack", "firespin", "flail", "fly", "frenzyplant", "frustration",
        "furyattack", "furycutter", "furyswipes", "gust", "hiddenpower", "highjumpkick",
        "hornattack", "hydrocannon", "hydropump", "hyperbeam", "iceball", "iciclespear",
        "jumpkick", "karatechop", "leafblade", "lowkick", "machpunch", "magicalleaf",
        "magnitude", "megahorn", "megakick", "megapunch", "meteormash", "muddywater",
        "mudshot", "nightshade", "outrage", "overheat", "payday", "peck", "petaldance",
        "pinmissile", "poisontail", "pound", "psychoboost", "psywave", "quickattack", "rage",
        "rapidspin", "razorleaf", "razorwind", "return", "revenge", "reversal", "rockblast",
        "rockthrow", "rollingkick", "rollout", "sandtomb", "scratch", "seismictoss",
        "selfdestruct", "shadowpunch", "shockwave", "signalbeam", "silverwind", "skullbash",
        "skyattack", "skyuppercut", "slam", "slash", "snore", "solarbeam", "sonicboom",
        "spikecannon", "spitup", "steelwing", "strength", "struggle", "submission", "surf",
        "swift", "tackle", "takedown", "thrash", "triplekick", "twister", "uproar", "vinewhip",
        "visegrip", "vitalthrow", "volttackle", "waterfall", "watergun", "waterpulse",
        "weatherball", "whirlpool", "wingattack", "wrap"]}},
    "focusband": {"surviveLethal": {"chance": [1, 10]}},
    # BERRY classes (`gen3_berry_trace_shedskin_v1` batch-3) — ONE consumption mechanism
    # (eatItem: the item becomes NONE for the battle) + per-berry parameter rows, all
    # PROBE-settled against the RESOLVED `Dex.mod('gen3')` (`harness/probe_berry_rng.js` /
    # `probe_berry_sub_tie_rng.js`; the mod-chain law — e.g. the resolved gen3 Figy family
    # heals baseMaxhp/8, NOT the base .ts's /3):
    #   class "cure"  — trigger at every `eachEvent('Update')` when the holder's major status
    #                   is in `statuses` (or it is confused, for `curesConfusion`); Lum
    #                   additionally eats IMMEDIATELY inside setStatus (`immediate`, its
    #                   onAfterSetStatus priority -1 — AFTER a Synchronize reflect). Draw-free.
    #   class "heal"  — trigger at the RESIDUAL (order 10 subOrder 4, the Leftovers slot) when
    #                   `hp <= maxhp/2` (exact: 2*hp <= maxhp): heal a fixed amount (Oran 10 /
    #                   Sitrus 30) or `floor(maxhp/8)` (`healFrac`, the Figy family — which
    #                   ALSO confuses if the holder's nature lowers `confuseIfMinus`, drawing
    #                   the confusion volatile's random(2,6)).
    #   class "pinch" — trigger at the RESIDUAL (order 10 subOrder 4) when `hp <= maxhp/4`
    #                   (exact: 4*hp <= maxhp): +1 stage on `boost` (draw-free), or Starf's
    #                   `random2` (ONE `sample` over the non-capped [atk,def,spa,spd,spe] in
    #                   that order — draws even for a single candidate — then +2), or Lansat's
    #                   `focusenergy` volatile (crit stage +2, draw-free).
    #   class "pp"    — Leppa: trigger at every `eachEvent('Update')` when a move slot hits
    #                   0 PP: restore `min(pp+10, maxpp)` on the first 0-PP slot. Draw-free.
    # Scope: the GEN-3 berries only (a gen-2 twin like PRZCureBerry is unobtainable under
    # gen3ou and stays unmodeled/off the e2e filter).
    "cheriberry": {"berryEffect": {"class": "cure", "statuses": ["par"], "curesConfusion": False, "immediate": False}},
    "chestoberry": {"berryEffect": {"class": "cure", "statuses": ["slp"], "curesConfusion": False, "immediate": False}},
    "pechaberry": {"berryEffect": {"class": "cure", "statuses": ["psn", "tox"], "curesConfusion": False, "immediate": False}},
    "rawstberry": {"berryEffect": {"class": "cure", "statuses": ["brn"], "curesConfusion": False, "immediate": False}},
    "aspearberry": {"berryEffect": {"class": "cure", "statuses": ["frz"], "curesConfusion": False, "immediate": False}},
    "persimberry": {"berryEffect": {"class": "cure", "statuses": [], "curesConfusion": True, "immediate": False}},
    "lumberry": {"berryEffect": {"class": "cure", "statuses": ["par", "slp", "psn", "tox", "brn", "frz"], "curesConfusion": True, "immediate": True}},
    "oranberry": {"berryEffect": {"class": "heal", "threshold": [1, 2], "heal": 10}},
    "sitrusberry": {"berryEffect": {"class": "heal", "threshold": [1, 2], "heal": 30}},
    "figyberry": {"berryEffect": {"class": "heal", "threshold": [1, 2], "healFrac": 8, "confuseIfMinus": "atk"}},
    "wikiberry": {"berryEffect": {"class": "heal", "threshold": [1, 2], "healFrac": 8, "confuseIfMinus": "spa"}},
    "magoberry": {"berryEffect": {"class": "heal", "threshold": [1, 2], "healFrac": 8, "confuseIfMinus": "spe"}},
    "aguavberry": {"berryEffect": {"class": "heal", "threshold": [1, 2], "healFrac": 8, "confuseIfMinus": "spd"}},
    "iapapaberry": {"berryEffect": {"class": "heal", "threshold": [1, 2], "healFrac": 8, "confuseIfMinus": "def"}},
    "liechiberry": {"berryEffect": {"class": "pinch", "threshold": [1, 4], "boost": "atk"}},
    "ganlonberry": {"berryEffect": {"class": "pinch", "threshold": [1, 4], "boost": "def"}},
    "salacberry": {"berryEffect": {"class": "pinch", "threshold": [1, 4], "boost": "spe"}},
    "petayaberry": {"berryEffect": {"class": "pinch", "threshold": [1, 4], "boost": "spa"}},
    "apicotberry": {"berryEffect": {"class": "pinch", "threshold": [1, 4], "boost": "spd"}},
    "lansatberry": {"berryEffect": {"class": "pinch", "threshold": [1, 4], "boost": "focusenergy"}},
    "starfberry": {"berryEffect": {"class": "pinch", "threshold": [1, 4], "boost": "random2"}},
    "leppaberry": {"berryEffect": {"class": "pp", "restore": 10}},
}

# gen4-NAMED items the sim still APPLIES under gen3 formats (gen3customgame carries no item
# validation; the resolved gen3 dex keeps their base onBasePower handlers). They sit in the
# e2e/fuzzer MODELED_ITEMS, so the port must know + price them — included as an explicit,
# documented exception to the `item_gen > gen` filter. Obs-neutral: the obs encodes items
# by per-id `num` lookup (no enumeration index), so ADDING entries changes no existing value.
_GEN4_ITEMS_APPLIED_IN_GEN3 = ("oddincense", "rockincense", "roseincense", "waveincense")


def build_items(gen):
    """Build the gen-N item map (name + num [+ mechanics]) from the Showdown items source.

    Parses `deps/pokemon-showdown/data/items.ts` the same way `build_abilities`
    parses `abilities.ts`: a per-item `\\titemid: {` block, then `name`, `num`, and
    the introduction `gen`. Items introduced after the target generation are
    filtered out (except `_GEN4_ITEMS_APPLIED_IN_GEN3`). Each block is bounded at the
    NEXT item's header (not a fixed window) so a per-block flag like `isBerry` can never
    be read from the following item. Gen-3 entries are then enriched with the curated
    `_GEN3_ITEM_MECHANICS` fields (see the mod-chain law above) + the declarative
    `isBerry` flag. Sorted by id (like the other maps) for a stable file."""
    items_path = os.path.join(REPO_ROOT, "deps", "pokemon-showdown", "data", "items.ts")
    if not os.path.exists(items_path):
        raise FileNotFoundError(f"Items source not found: {items_path}")

    with open(items_path, "r") as f:
        content = f.read()

    items_map = {}
    headers = list(re.finditer(r"^\t([a-z0-9]+):\s*\{", content, re.MULTILINE))
    for i, match in enumerate(headers):
        item_id = match.group(1)
        block_end = headers[i + 1].start() if i + 1 < len(headers) else len(content)
        block = content[match.start():block_end]
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
            if not (gen == 3 and item_id in _GEN4_ITEMS_APPLIED_IN_GEN3):
                continue
        entry = {"name": name_match.group(1), "num": num}
        # `isBerry: true` is declarative in the block (unlike the handler callbacks) — safe
        # to read textually now that the block is bounded. Only-when-present, like critRatio.
        if re.search(r"\bisBerry:\s*true", block):
            entry["isBerry"] = True
        if gen == 3 and item_id in _GEN3_ITEM_MECHANICS:
            entry.update(_GEN3_ITEM_MECHANICS[item_id])
        items_map[item_id] = entry

    if gen == 3:
        missing = [iid for iid in _GEN3_ITEM_MECHANICS if iid not in items_map]
        if missing:  # a curated id that no longer parses = silent data rot — fail loud
            raise ValueError(f"curated gen3 item mechanics not found in items.ts: {missing}")

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


def build_move_aliases(gen):
    """Build the gen-N MOVE-alias map ``{alias_id: canonical_move_id}`` from Showdown's
    ``data/aliases.ts``.

    Showdown resolves a packed-team move token through its alias table at
    ``dex.moves.get()`` time (e.g. ``wisp`` -> ``Will-O-Wisp`` -> ``willowisp``, ``sd`` ->
    ``swordsdance``, ``twave`` -> ``thunderwave``). The ``src/rust_sim`` port's dex reads
    only the CANONICAL ``gen3_moves.json`` keys, so a team carrying an alias (a real
    thing in the sample-team pool — Gengar's ``wisp``) makes the port's ``move_at``
    return ``None`` and NO-OP the move, drawing NOTHING while the sim runs the full move
    -> a bit-for-bit draw-count DESYNC. This file lets the port resolve aliases exactly
    like Showdown. **Obs-neutral**: the Python ``agents.gen3_data`` facade never loads it
    (it names its files explicitly, no glob), so the RL obs is unchanged; it exists ONLY
    for the port. The ``gen3_move_alias_resolution_v1`` fix (surfaced by e2e_86 once the
    DMG_MOD abilities admitted a ``wisp``-carrying Gengar team).

    Scope: aliases whose CANONICAL id is a gen-N move, EXCLUDING the ``hiddenpower*``
    aliases (``hp``/``hpice``/...): Showdown maps those to the bare ``hiddenpower``, but
    the port represents Hidden Power by its distinct TYPED move name
    (``hiddenpowerice``, ``gen3_typed_hidden_power_ids_v1``), so collapsing an ``hpice``
    token to bare ``hiddenpower`` would LOSE the type. No sample team uses the ``hp*``
    shorthand (they write the full typed form), so excluding them is both safe and
    correct for the port's model.
    """
    aliases_path = os.path.join(REPO_ROOT, "deps", "pokemon-showdown", "data", "aliases.ts")
    if not os.path.exists(aliases_path):
        raise FileNotFoundError(f"Aliases source not found: {aliases_path}")
    with open(aliases_path, "r") as f:
        content = f.read()

    legal_moves = set(build_moves(gen))  # canonical gen-N movedex ids (incl. typed HP)
    # Each alias entry is `\talias: "Display Name",` (values may contain spaces / hyphens /
    # apostrophes — normalized to an id by to_id_str, the same normalization the port +
    # the sim apply). Keys are already lowercase ids.
    out = {}
    for m in re.finditer(r'^\t([a-z0-9]+):\s*"([^"]+)",', content, re.MULTILINE):
        alias_id = m.group(1)
        canonical_id = to_id_str(m.group(2))
        if canonical_id not in legal_moves:
            continue            # alias for a non-gen-N move (or a species/item/format)
        if canonical_id == alias_id:
            continue            # not a real alias
        if canonical_id == "hiddenpower":
            continue            # port uses the distinct TYPED HP name, not bare
        out[alias_id] = canonical_id
    return {aid: out[aid] for aid in sorted(out)}


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
    "move_aliases": ("gen{gen}_move_aliases.json", build_move_aliases),
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
