// dump_gen3_mechanics.js — the DATA-DRIVEN MECHANICS FRAMEWORK's extraction +
// classification dump (Phase 1 foundation).
//
// WHY: the port used to hand-model items/abilities one id at a time; the A/B fuzzer
// caught the inevitable drift (Pink Bow / Polkadot Bow / the 4 gen4-named incenses sat
// in the e2e's MODELED_ITEMS while `resolve_atk_stat_mods` priced none of them). This
// harness extracts the gen3-RESOLVED item/ability tables ONCE (like the dex), assigns
// every entry a mechanic CLASS with its extractable PARAMETERS, and writes the CLASS
// MAP every future phase executes against.
//
// THE MOD-CHAIN LAW (the cautionary tale): gen3 resolves through the WHOLE mod chain
// (gen3 -> gen4 -> ... -> base), where later mods REPLACE and DELETE handlers (e.g.
// Light Ball: base = Atk+SpA x2, gen4 mod REWRITES it to onBasePower x2, gen3 mod
// REWRITES it again to SpA-only x2 + deletes onBasePower). NEVER extract from a raw
// data/*.ts file — this harness reads `Dex.mod('gen3')`, the RESOLVED dist, and the
// class-sweep golden (real sim battles) is the final oracle.
//
// Outputs:
//   node dump_gen3_mechanics.js            -> writes tests/vectors/gen3_mechanics_inventory.md
//   node dump_gen3_mechanics.js --json     -> prints the machine-readable extraction (per-id
//                                             class + params) — the curated-table source for
//                                             tools/pokemon_data_extractor/sync.py
//   node dump_gen3_mechanics.js --check    -> verifies data/pokemon/gen3_items.json's mechanics
//                                             fields (typeBoost/statMods/onlySpecies/choice/
//                                             isBerry) + gen3_abilities.json's + the FULL
//                                             gen3_species.json universe (base forms + the gen-3
//                                             alternate/cosmetic FORMES, gen3_species_formes_v1)
//                                             EXACTLY match the resolved dist — the
//                                             extractor-drift gate (exit 1 on any drift)
//
// (Needs the submodule dist/ + node_modules symlinks; see root CLAUDE.md.)

'use strict';

const path = require('path');
const fs = require('fs');

const PS = path.resolve(__dirname, '../../../deps/pokemon-showdown');
const { Dex } = require(path.join(PS, 'dist/sim/dex'));
const REPO = path.resolve(__dirname, '../../..');
const OUT_MD = path.resolve(__dirname, '../tests/vectors/gen3_mechanics_inventory.md');
const ITEMS_JSON = path.join(REPO, 'data/pokemon/gen3_items.json');
const ABILITIES_JSON = path.join(REPO, 'data/pokemon/gen3_abilities.json');

// The port's modeled sets (ONE source of truth — gen_e2e_fuzz.js module.exports).
const e2e = require('./gen_e2e_fuzz.js');

const d3 = Dex.mod('gen3');

// gen4-NAMED items the sim still applies under gen3 formats (gen3customgame has no item
// validation and the resolved dex carries them). They are in the e2e MODELED_ITEMS +
// (post-P1) in data/pokemon/gen3_items.json as an explicit, documented exception.
const GEN4_ITEMS_APPLIED_IN_GEN3 = new Set(['oddincense', 'rockincense', 'roseincense', 'waveincense']);

// Handler keys that are base-data cruft irrelevant to gen3 (Arceus plates / Genesect
// drives / Silvally memories type-override helpers) — excluded from the inventory.
const IGNORED_HANDLERS = new Set(['onDrive', 'onMemory', 'onPlate']);

function toId(s) { return ('' + (s || '')).toLowerCase().replace(/[^a-z0-9]/g, ''); }

function handlerInventory(entry) {
  const out = {};
  const keys = Object.keys(entry).filter((k) => k.startsWith('on') && entry[k] !== undefined && !IGNORED_HANDLERS.has(k));
  const hasFn = keys.some((k) => typeof entry[k] === 'function');
  for (const k of keys) {
    // A bare numeric onXPriority/onXOrder with NO function handler at all is base-data
    // residue (e.g. Pickup's out-of-battle onResidualOrder) — not a battle handler.
    if (!hasFn && typeof entry[k] !== 'function') continue;
    out[k] = typeof entry[k] === 'function' ? String(entry[k]).replace(/\s+/g, ' ') : entry[k];
  }
  return out;
}

function fnSrc(inv) {
  // All function-valued handler sources concatenated (for draw-bearing detection).
  return Object.values(inv).filter((v) => typeof v === 'string').join(' ');
}

// chainModify arg -> exact [num, den] the port's 4096 fixed-point chain uses.
// chainModify(1.1) computes trunc(1.1*4096)=4505 == trunc(11*4096/10) — the rational
// form is bit-identical, so we store rationals ([11,10], [21,20], [3,2], [2,1], [4915,4096]).
const CHAIN_RATIONALS = {
  '1.05': [21, 20], '1.1': [11, 10], '1.5': [3, 2], '2': [2, 1], '0.5': [1, 2],
  // ACCURACY chainModify args (gen3_accuracy_pipeline_v1): Compound Eyes / Sand Veil.
  '1.3': [13, 10], '0.8': [8, 10],
};
function chainArgToRational(arg) {
  const m = arg.match(/^\[\s*(\d+)\s*,\s*(\d+)\s*\]$/);
  if (m) return [Number(m[1]), Number(m[2])];
  if (CHAIN_RATIONALS[arg]) return CHAIN_RATIONALS[arg];
  throw new Error(`unmapped chainModify arg ${JSON.stringify(arg)} — add its exact rational`);
}

// --- ACCURACY mod extraction (gen3_accuracy_pipeline_v1): the accMod field from the
//     RESOLVED onModifyAccuracy (DEFENDER: Bright Powder / Lax Incense / Sand Veil) or
//     onSourceModifyAccuracy (ATTACKER: Compound Eyes / Hustle) handler source. Two op
//     shapes: `accuracy * FLOAT` (DIRECT multiply — the float verbatim) vs
//     `chainModify(ARG)`. Gates: `isWeather("sandstorm")` (Sand Veil), the
//     `physicalTypes.includes(move.type)` list (Hustle). Returns `undefined` for a
//     non-accuracy handler set. The mod-chain law: this reads the RESOLVED dist source.
function extractAccMod(inv) {
  const src = inv.onModifyAccuracy || inv.onSourceModifyAccuracy;
  if (typeof src !== 'string' || !src) return undefined;
  const side = inv.onSourceModifyAccuracy ? 'attacker' : 'defender';
  const am = { side };
  const chainM = src.match(/chainModify\(([^)]+)\)/);
  const multM = src.match(/accuracy \* ([\d.]+)/);
  if (chainM) {
    am.op = 'chain';
    am.mod = chainArgToRational(chainM[1].trim());
  } else if (multM) {
    am.op = 'multiply';
    am.mod = Number(multM[1]); // the EXACT float literal (0.9 / 0.95)
  } else {
    throw new Error(`accMod: unrecognized accuracy handler shape: ${src}`);
  }
  if (/isWeather\("sandstorm"\)/.test(src)) am.weather = 'sandstorm';
  if (/physicalTypes\.includes\(move\.type\)/.test(src)) am.physicalTypesOnly = true;
  // Emit keys in the SAME order the curated table / committed JSON uses so the
  // JSON.stringify equality in --check is order-stable: op, mod, side, [weather],
  // [physicalTypesOnly].
  const out = { op: am.op, mod: am.mod, side: am.side };
  if (am.weather) out.weather = am.weather;
  if (am.physicalTypesOnly) out.physicalTypesOnly = am.physicalTypesOnly;
  return out;
}

// ---------------------------------------------------------------------------
// ITEM extraction: per-id mechanics params from the RESOLVED handler sources.
// ---------------------------------------------------------------------------
function extractItem(id, it) {
  const inv = handlerInventory(it);
  const src = fnSrc(inv);
  const mech = {}; // the additive gen3_items.json fields
  const notes = [];

  // --- TYPE_BOOST: a type-gated offensive boost. Three resolved shapes:
  //   (a) gen3-mod stat fold:  onModifyAtk/onModifySpA `move.type === T -> chainModify(x)`
  //   (b) base BP chain fold:  onBasePower `move.type === T -> chainModify([n,d])`
  //   (c) base BP DIRECT fold: onBasePower `move.type === T -> basePower * x` (a float
  //       return that REPLACES relayVar — runEvent's final-modifier guard SKIPS a
  //       non-integer relayVar, and clampIntRange floors; the bows).
  for (const [key, fold] of [['onModifyAtk', 'stat'], ['onModifySpA', 'stat'], ['onBasePower', 'basePower']]) {
    const s = inv[key];
    if (typeof s !== 'string') continue;
    const typeM = s.match(/move\??\.type === "(\w+)"/);
    if (!typeM) continue;
    const chainM = s.match(/chainModify\(([^)]+)\)/);
    const directM = s.match(/return basePower \* ([\d.]+)/);
    if (chainM) {
      mech.typeBoost = { type: typeM[1], mod: chainArgToRational(chainM[1].trim()), fold };
    } else if (directM) {
      const rat = CHAIN_RATIONALS[directM[1]];
      if (!rat) throw new Error(`${id}: unmapped direct BP multiplier ${directM[1]}`);
      mech.typeBoost = { type: typeM[1], mod: rat, fold: 'basePowerDirect' };
    }
  }

  // --- SPECIES_STAT: a species-gated stat multiplier (Thick Club / Light Ball /
  //     DeepSea* / Metal Powder / Soul Dew). Species from `species.name === "X"` or
  //     `baseSpecies.num === N`; stat from the handler event; mult from chainModify.
  const statKeys = { onModifyAtk: 'atk', onModifySpA: 'spa', onModifyDef: 'def', onModifySpD: 'spd' };
  for (const [key, stat] of Object.entries(statKeys)) {
    const s = inv[key];
    if (typeof s !== 'string' || (mech.typeBoost && mech.typeBoost.fold === 'stat')) continue;
    const chainM = s.match(/chainModify\(([^)]+)\)/);
    if (!chainM) continue;
    const names = [...s.matchAll(/species\.name === "([^"]+)"/g)].map((m) => toId(m[1]));
    const nums = [...s.matchAll(/baseSpecies\.num === (\d+)/g)].map((m) => Number(m[1]));
    const numSpecies = nums.map((n) => {
      for (const sp of d3.species.all()) if (sp.num === n && !sp.forme) return toId(sp.name);
      throw new Error(`${id}: no species for num ${n}`);
    });
    const species = [...new Set([...names, ...numSpecies])].sort();
    if (species.length === 0) {
      // unconditional stat mod (Choice Band's onModifyAtk) — handled via `choice` below
      // unless something new appears.
      if (!it.isChoice) notes.push(`unconditional ${key} chainModify — NOT species-gated`);
      continue;
    }
    mech.statMods = mech.statMods || {};
    mech.statMods[stat] = chainArgToRational(chainM[1].trim());
    mech.onlySpecies = species;
    if (/!pokemon\.transformed/.test(s)) mech.untransformedOnly = true;
  }

  // --- CHOICE (gen3: Choice Band only) — the x1.5 Atk stat fold + the move lock.
  if (it.isChoice) {
    mech.choice = true;
    const s = inv.onModifyAtk || '';
    const chainM = s.match(/chainModify\(([^)]+)\)/);
    if (!chainM) throw new Error(`${id}: isChoice without an onModifyAtk chainModify`);
    mech.statMods = { atk: chainArgToRational(chainM[1].trim()) };
  }

  if (it.isBerry) mech.isBerry = true;
  // --- CRIT_ITEM (`gen3_crit_item_v1`): `onModifyCritRatio` critRatio + N, optionally
  //     species-gated. Scope Lens +1 (unconditional); Lucky Punch +2 (Chansey); Stick +2
  //     (Farfetch'd). A DRAW-FREE fold into the existing crit-ratio (the Focus Energy
  //     precedent) — the crit `randomChance(1, CRIT_MULT[ratio])` draw COUNT is unchanged;
  //     only the denominator index shifts (1→3 ⇒ 1/16→1/4). `leek` is the gen8 rename (same
  //     mechanic, NOT gen3-legal → not an entry).
  const cr = inv.onModifyCritRatio;
  if (typeof cr === 'string' && cr) {
    const boostM = cr.match(/critRatio \+ (\d+)/);
    if (!boostM) throw new Error(`${id}: onModifyCritRatio without a critRatio + N`);
    const boost = Number(boostM[1]);
    const ids = [...cr.matchAll(/species\.id === "(\w+)"/g)].map((m) => m[1]);
    const names = [...cr.matchAll(/species\.name === "([^"]+)"/g)].map((m) => toId(m[1]));
    const species = [...new Set([...ids, ...names])].sort();
    mech.critBoost = species.length ? { boost, onlySpecies: species } : { boost };
  }
  // --- BOOST_RESTORE (`gen3_white_herb_v1`): White Herb — restore all NEGATIVE boost stages
  //     to 0 (positives untouched) + consume, single-use. The resolved gen3 shape: an
  //     `onStart` that scans `boosts[i] < 0` → `useItem()`, and an `onUse` that does
  //     `setBoost(...)` + `-clearnegativeboost`; it fires from onAnyAfterMove / onAnySwitchIn
  //     / onResidual(29). DRAW-FREE — a boolean flag (no parameters). Detect on the unique
  //     onUse `setBoost` + `-clearnegativeboost` signature.
  if (typeof inv.onUse === 'string' && /-clearnegativeboost/.test(inv.onUse) && /setBoost/.test(inv.onUse)) {
    mech.boostRestore = true;
  }
  // ACCURACY_ITEM (Bright Powder / Lax Incense) — DEFENDER-side accMod.
  const accMod = extractAccMod(inv);
  if (accMod) mech.accMod = accMod;
  // BERRY classes (`gen3_berry_trace_shedskin_v1`) — derived for GEN-3 berries only (a
  // gen-2 twin like PRZCureBerry is gen3ou-unobtainable and stays unmodeled). Key order
  // matches sync.py's curated rows (the --check JSON.stringify equality is order-sensitive).
  if (it.isBerry && it.gen === 3) {
    const be = extractBerryEffect(id, inv);
    if (be) mech.berryEffect = be;
  }
  // PROC_ITEM (`gen3_ability_batch4_v1`) — King's Rock / Focus Band, EXECUTION-derived from
  // the RESOLVED handlers (the mod-chain law: never regex a single data file for the list):
  //   flinchSecondary — an onModifyMove that PUSHES a `{chance, volatileStatus: "flinch"}`
  //       secondary. The affected-move LIST is derived by EXECUTING the resolved handler
  //       against every gen<=3 move id (deduped — the 17 Hidden Powers share the sim id
  //       "hiddenpower") and the chance from the pushed literal.
  const mmSrc = typeof inv.onModifyMove === 'string' ? inv.onModifyMove : '';
  if (mmSrc && /volatileStatus: "flinch"/.test(mmSrc) && /move\.secondaries\.push/.test(mmSrc)) {
    const chanceM = mmSrc.match(/chance: (\d+)/);
    if (!chanceM) throw new Error(`${id}: flinch-push onModifyMove without a chance literal`);
    const moves = new Set();
    for (const mv of d3.moves.all()) {
      if (mv.gen > 3 || mv.isNonstandard) continue;
      const fake = { id: mv.id, secondaries: null };
      it.onModifyMove.call({}, fake);
      if (fake.secondaries && fake.secondaries.length) moves.add(mv.id);
    }
    mech.flinchSecondary = { chance: Number(chanceM[1]), moves: [...moves].sort() };
  }
  //   surviveLethal — an onDamage `randomChance(a,b) && damage >= target.hp &&
  //       effect.effectType === 'Move'` survive-at-1 (the roll draws FIRST, on every Damage
  //       event into the holder — probe_focusband_rng.js).
  const odSrc = typeof inv.onDamage === 'string' ? inv.onDamage : '';
  if (odSrc && /damage >= target\.hp/.test(odSrc)) {
    const rcM = odSrc.match(/randomChance\((\d+),\s*(\d+)\)/);
    if (!rcM) throw new Error(`${id}: survive-lethal onDamage without a randomChance`);
    mech.surviveLethal = { chance: [Number(rcM[1]), Number(rcM[2])] };
  }
  return { inv, mech, notes };
}

// Derive a gen-3 berry's `berryEffect` from its RESOLVED handlers (null = a flavor berry
// with no in-battle trigger). Shapes (probe-settled, `probe_berry_rng.js`):
//   cure  — base onUpdate `status === "x"` (+ lum's truthy `pokemon.status ||` = all six;
//           `volatiles["confusion"]` → curesConfusion; onAfterSetStatus → immediate).
//   heal  — gen3-mod onResidual `hp <= maxhp / 2` + onEat `this.heal(N)` or
//           `this.heal(pokemon.baseMaxhp / 8)` (+ `getNature().minus === "x"` → confusion).
//   pinch — gen3-mod onResidual `hp <= maxhp / 4` + onEat `boost({ x: 1 })` / the Starf
//           sample loop (`random2`) / `addVolatile("focusenergy")` (Lansat).
//   pp    — onUpdate `move.pp === 0` + onEat `+10` (Leppa; the ripen 20 is gen8+).
const _STATUS_ORDER_BERRY = ['par', 'slp', 'psn', 'tox', 'brn', 'frz'];
function extractBerryEffect(id, inv) {
  const upd = typeof inv.onUpdate === 'string' ? inv.onUpdate : '';
  const res = typeof inv.onResidual === 'string' ? inv.onResidual : '';
  const eat = typeof inv.onEat === 'string' ? inv.onEat : '';
  // PP (Leppa): onUpdate gated on a 0-PP move slot.
  if (upd && /move\.pp === 0|move\) => move\.pp === 0/.test(upd)) {
    return { class: 'pp', restore: 10 };
  }
  // CURE: an onUpdate that eats on a status / confusion condition.
  if (upd && /pokemon\.eatItem\(\)/.test(upd) && /status|confusion/.test(upd)) {
    const ids = new Set([...upd.matchAll(/status === "(\w+)"/g)].map((m) => m[1]));
    let statuses = _STATUS_ORDER_BERRY.filter((s) => ids.has(s));
    // lum: the bare-truthy `pokemon.status ||` guard = every major status.
    if (statuses.length === 0 && /pokemon\.status \|\|/.test(upd)) statuses = [..._STATUS_ORDER_BERRY];
    const curesConfusion = /volatiles\["confusion"\]/.test(upd) || /removeVolatile\("confusion"\)/.test(eat);
    const immediate = typeof inv.onAfterSetStatus === 'string';
    return { class: 'cure', statuses, curesConfusion, immediate };
  }
  // HEAL / PINCH: the gen3-mod residual thresholds.
  if (res && /pokemon\.eatItem\(\)/.test(res)) {
    if (/maxhp \/ 2/.test(res)) {
      const fixed = eat.match(/this\.heal\((\d+)\)/);
      if (fixed) return { class: 'heal', threshold: [1, 2], heal: Number(fixed[1]) };
      const frac = eat.match(/this\.heal\(pokemon\.baseMaxhp \/ (\d+)\)/);
      const minus = eat.match(/getNature\(\)\.minus === "(\w+)"/);
      if (frac && minus) {
        return { class: 'heal', threshold: [1, 2], healFrac: Number(frac[1]), confuseIfMinus: minus[1] };
      }
      throw new Error(`${id}: unrecognized HEAL_BERRY onEat shape`);
    }
    if (/maxhp \/ 4/.test(res)) {
      const boostM = eat.match(/this\.boost\(\{ (\w+): 1 \}\)/);
      if (boostM) return { class: 'pinch', threshold: [1, 4], boost: boostM[1] };
      if (/this\.sample\(stats\)/.test(eat)) return { class: 'pinch', threshold: [1, 4], boost: 'random2' };
      if (/addVolatile\("focusenergy"\)/.test(eat)) return { class: 'pinch', threshold: [1, 4], boost: 'focusenergy' };
      throw new Error(`${id}: unrecognized PINCH_BERRY onEat shape`);
    }
  }
  return null; // a flavor berry (no in-battle trigger)
}

// ---------------------------------------------------------------------------
// ITEM classification (the class map). Rule-based over the resolved handlers,
// with a curated override for the callback-shaped odd ones. Every entry MUST
// classify — an UNCLASSIFIED entry fails the dump loudly.
// ---------------------------------------------------------------------------
const ITEM_CLASS_OVERRIDES = {
  // The 3 proc items (draw-bearing when their class is built):
  quickclaw: 'PROC_ITEM',   // onFractionalPriority randomChance(1,5) — MODELED (the port draws it unconditionally end-of-turn)
  kingsrock: 'PROC_ITEM',   // onModifyMove adds a 10% flinch secondary -> an EXTRA secondary random(100)
  focusband: 'PROC_ITEM',   // onDamage randomChance(1,10) survive-at-1
  brightpowder: 'ACCURACY_ITEM', // onModifyAccuracy x0.9 (gen3 mod)
  laxincense: 'ACCURACY_ITEM',   // onModifyAccuracy x0.95
  scopelens: 'CRIT_ITEM',   // onModifyCritRatio +1
  luckypunch: 'CRIT_ITEM',  // Chansey-only +2
  stick: 'CRIT_ITEM',       // Farfetch'd-only +2
  leftovers: 'RESIDUAL_ITEM', // onResidual heal maxhp/16
  shellbell: 'DRAIN_ITEM',  // onAfterMoveSecondarySelf heal damage/8
  whiteherb: 'BOOST_RESTORE', // restore negative boosts + consume (onStart/useItem, fires onAnyAfterMove/onAnySwitchIn/onResidual(29))
  mentalherb: 'CURE_ITEM',  // cures Attract
  machobrace: 'SPEED_MOD',  // onModifySpe x0.5
  souldew: 'SPECIES_STAT',  // SpA+SpD x1.5 (Latias/Latios) — dual-stat
  berryjuice: 'HEAL_BERRY', // not isBerry but the same hp<=1/2 heal-20 residual shape
  leppaberry: 'PP_BERRY',   // restores 10 PP to a 0-PP move (onUpdate eatItem)
  mysteryberry: 'PP_BERRY', // the gen2 leppa (5 PP)
  mail: 'TAKE_ITEM_GUARD',  // onTakeItem false — blocks Thief/Knock Off/Trick item removal
};

function classifyItem(id, it, ex) {
  if (ITEM_CLASS_OVERRIDES[id]) return ITEM_CLASS_OVERRIDES[id];
  const inv = ex.inv;
  const src = fnSrc(inv);
  if (ex.mech.choice) return 'CHOICE';
  if (ex.mech.typeBoost) return 'TYPE_BOOST';
  if (ex.mech.statMods) return 'SPECIES_STAT';
  if (it.isBerry) {
    // gen3-mod berries are all onResidual eatItem shapes:
    //   pinch (hp <= maxhp/4, Liechi family + Starf/Lansat) | heal/confuse (hp <= maxhp/2,
    //   Oran/Sitrus/Figy family) | status-cure (onUpdate/onEat cureStatus, Cheri family).
    if (/cureStatus|removeVolatile\("confusion"\)/.test(src)) return 'CURE_BERRY';
    if (/maxhp \/ 4/.test(src)) return 'PINCH_BERRY';
    if (/maxhp \/ 2/.test(src)) return 'HEAL_BERRY';
    // Flavor berries (Bluk/Razz/...) resolve to a bare onEat with NO in-battle trigger
    // (no onUpdate/onResidual; no Fling/Bug Bite in gen3) — battle no-ops.
    if (!inv.onUpdate && !inv.onResidual && !inv.onTryEatItem) return 'NO_OP';
    return 'UNCLASSIFIED_BERRY';
  }
  if (Object.keys(inv).length === 0) return 'NO_OP';
  return 'UNCLASSIFIED';
}

// ---------------------------------------------------------------------------
// ABILITY extraction + classification.
// ---------------------------------------------------------------------------
const ABILITY_CLASS_OVERRIDES = {
  // DMG_MOD family (the stretch class): params extracted below.
  torrent: 'DMG_MOD', blaze: 'DMG_MOD', overgrow: 'DMG_MOD', swarm: 'DMG_MOD',
  hugepower: 'DMG_MOD', purepower: 'DMG_MOD', guts: 'DMG_MOD', marvelscale: 'DMG_MOD',
  hustle: 'DMG_MOD', // + an ACCURACY side (x0.8 physical acc) — must ship WITH the acc pipeline
  thickfat: 'DMG_MOD', // onSourceBasePower x0.5 Ice/Fire — MODELED (the port's defender_thick_fat)
  // Immunities / absorbs:
  levitate: 'TYPE_ABSORB', flashfire: 'TYPE_ABSORB', waterabsorb: 'TYPE_ABSORB', voltabsorb: 'TYPE_ABSORB',
  lightningrod: 'TYPE_ABSORB', // gen3: draws Electric moves in doubles; singles: immunity? (probe when built)
  // Status immunities:
  immunity: 'STATUS_IMMUNE', limber: 'STATUS_IMMUNE', insomnia: 'STATUS_IMMUNE',
  vitalspirit: 'STATUS_IMMUNE', waterveil: 'STATUS_IMMUNE', magmaarmor: 'STATUS_IMMUNE',
  owntempo: 'STATUS_IMMUNE', oblivious: 'STATUS_IMMUNE',
  // Boost-drop immunities (the Intimidate gate — MODELED):
  clearbody: 'BOOST_IMMUNE', whitesmoke: 'BOOST_IMMUNE', hypercutter: 'BOOST_IMMUNE', keeneye: 'BOOST_IMMUNE',
  battlearmor: 'CRIT_IMMUNE', shellarmor: 'CRIT_IMMUNE',
  swiftswim: 'WEATHER_SPEED', chlorophyll: 'WEATHER_SPEED',
  intimidate: 'SWITCH_IN', drizzle: 'SWITCH_IN', drought: 'SWITCH_IN', sandstream: 'SWITCH_IN',
  trace: 'SWITCH_IN', forecast: 'SWITCH_IN', naturalcure: 'SWITCH_OUT',
  static: 'CONTACT_PROC', poisonpoint: 'CONTACT_PROC', effectspore: 'CONTACT_PROC',
  flamebody: 'CONTACT_PROC', roughskin: 'CONTACT_PROC', cutecharm: 'CONTACT_PROC',
  colorchange: 'ON_HIT_MISC',
  speedboost: 'RESIDUAL', shedskin: 'RESIDUAL', raindish: 'RESIDUAL',
  arenatrap: 'TRAP', magnetpull: 'TRAP', shadowtag: 'TRAP',
  compoundeyes: 'ACCURACY', sandveil: 'ACCURACY',
  serenegrace: 'SECONDARY_MOD', shielddust: 'SECONDARY_MOD',
  // BLOCK abilities (`gen3_ability_batch2_v1`) — MODELED (data-driven, `turn.rs`):
  soundproof: 'BLOCK', damp: 'BLOCK', suctioncups: 'BLOCK',
  synchronize: 'SYNCHRONIZE', // reflect a foe status back to the source (`gen3_ability_batch2_v1`)
  // MISC mechanics (each its own future decision):
  stickyhold: 'TAKE_ITEM_GUARD', // onTakeItem false — the ability twin of Mail
  pressure: 'MISC', truant: 'MISC', rockhead: 'MISC', liquidooze: 'MISC',
  sturdy: 'MISC', airlock: 'MISC', cloudnine: 'MISC',
  wonderguard: 'DAMAGE_GATE', // the SE-only onTryHit damage gate (`gen3_wonder_guard_v1`)
  innerfocus: 'MISC', earlybird: 'MISC',
  voltabsorb2: 'MISC', minus: 'MISC', plus: 'MISC', overcoat: 'MISC',
  swiftswim2: 'MISC', anticipation: 'MISC',
};

function extractAbility(id, ab) {
  const inv = handlerInventory(ab);
  const src = fnSrc(inv);
  const mech = {};
  // DMG_MOD params (the stretch class): the pinch family / unconditional / status-gated /
  // the defender-side onSourceBasePower (Thick Fat).
  const s = inv.onBasePower || inv.onModifyAtk || inv.onModifyDef || inv.onSourceBasePower || '';
  if (typeof s === 'string' && s) {
    const chainM = s.match(/chainModify\(([^)]+)\)/);
    const modM = s.match(/this\.modify\(\w+, ([\d.]+)\)/);
    const types = [...s.matchAll(/move\.type === "(\w+)"/g)].map((m) => m[1]);
    const pinch = /hp <= \w+\.maxhp \/ 3/.test(s);
    const statusGated = /if \(pokemon\.status\)/.test(s);
    const mod = chainM ? chainArgToRational(chainM[1].trim()) : (modM ? CHAIN_RATIONALS[modM[1]] : null);
    if (mod) {
      const dm = { mod };
      if (inv.onBasePower) dm.fold = 'basePower';
      else if (inv.onModifyAtk) dm.fold = 'atk';
      else if (inv.onModifyDef) dm.fold = 'def';
      else if (inv.onSourceBasePower) dm.fold = 'sourceBasePower'; // defender-side (Thick Fat)
      if (modM) dm.direct = true; // Hustle: this.modify (an IMMEDIATE rounded replace, NOT chainModify)
      if (types.length === 1) dm.type = types[0];
      else if (types.length > 1) dm.types = types;
      if (pinch) dm.pinch = true;           // fires at hp <= maxhp/3
      if (statusGated) dm.whenStatused = true;
      mech.dmgMod = dm;
    }
  }
  // ACCURACY class (Compound Eyes / Sand Veil / Hustle acc-side) — the accMod field.
  const accMod = extractAccMod(inv);
  if (accMod) mech.accMod = accMod;
  // STATUS_IMMUNE class (`gen3_status_immune_v1`) — an ability that grants immunity to a
  // specific MAJOR status. DERIVED from the RESOLVED-dist handler that does the block:
  //   - `onSetStatus(status,...) { if (status.id !== "X") return; ...; return false; }`
  //     → a `setStatus`-phase immunity (blocks INSIDE runEvent('SetStatus'), after the
  //     clause shuffle drew). The gated status id(s) are the `status.id !== "X"` / `&&`
  //     literals in the guard.
  //   - `onImmunity(type,...) { if (type === "X") return false; }` for a MAJOR-status type
  //     → an `immunity`-phase block (blocks at runStatusImmunity, BEFORE the SetStatus
  //     event). (Sand Veil's onImmunity('sandstorm') / Oblivious's onImmunity('attract')
  //     are NOT major statuses → not a STATUS_IMMUNE member.)
  // Own Tempo / Oblivious block a VOLATILE via onTryAddVolatile, not a status via
  // setStatus → deliberately NOT emitted here (a different mechanism).
  const statusImmune = extractStatusImmune(inv);
  if (statusImmune) mech.statusImmune = statusImmune;
  // CRIT_IMMUNE class (`gen3_ability_batch1_v1`) — Battle Armor / Shell Armor resolve
  // `onCriticalHit = false` (a BOOLEAN, not a function — the base-data `onCriticalHit(...)`
  // fn is gen4-mod-DELETED / gen3-set-false). The crit roll is DRAWN then overridden to false
  // (draw-free — probe_critimmune_rng.js). Emitted as `critImmune: true`.
  if (ab.onCriticalHit === false) mech.critImmune = true;
  // WEATHER_SPEED class (`gen3_ability_batch1_v1`) — Chlorophyll / Swift Swim: an
  // `onModifySpe chainModify(2)` gated on effectiveWeather() being sun / rain. The gated
  // weather is the FIRST id in the `[...].includes(pokemon.effectiveWeather())` guard
  // (sunnyday/desolateland → "sunnyday"; raindance/primordialsea → "raindance"). A ×2 speed
  // modifier feeding getActionSpeed → the tie-shuffles (draw-free itself).
  const speSrc = typeof inv.onModifySpe === 'string' ? inv.onModifySpe : '';
  if (speSrc && /chainModify\(2\)/.test(speSrc)) {
    const wm = speSrc.match(/\["(\w+)"[^\]]*\]\.includes\(\s*pokemon\.effectiveWeather\(\)\s*\)/);
    if (wm) mech.weatherSpeed = { weather: wm[1] };
  }
  // WEATHER_NEGATE class (`gen3_ability_batch1_v1`) — Cloud Nine / Air Lock: an
  // `onStart`+`onEnd` pair whose onEnd runs `eachEvent("WeatherChange", ...)` and toggles
  // `abilityState.ending` — the resolved SHAPE of the suppress-weather ability (both are
  // identical). While active, `field.effectiveWeather()` returns '' so all weather effects are
  // suppressed (draw-free — changes deterministic chip/speed values, never a roll).
  const startSrc = typeof inv.onStart === 'string' ? inv.onStart : '';
  const endSrc = typeof inv.onEnd === 'string' ? inv.onEnd : '';
  if (/abilityState\.ending = false/.test(startSrc) && /abilityState\.ending = true/.test(endSrc)
    && /eachEvent\("WeatherChange"/.test(endSrc)) {
    mech.weatherNegate = true;
  }
  // --- gen3_ability_batch2_v1: the DRAW-BEARING "reactive" classes + block tail. ---
  // CONTACT_PROC / CONTACT recoil — an `onDamagingHit` gated on `move.flags["contact"]`.
  const dhSrc = typeof inv.onDamagingHit === 'string' ? inv.onDamagingHit : '';
  if (dhSrc && /move\.flags\["contact"\]/.test(dhSrc)) {
    // A status proc: randomChance(n,d) + trySetStatus. Static/PoisonPoint/FlameBody roll
    // randomChance(1,3) → one literal status; Effect Spore rolls randomChance(1,10) then
    // sample([...]) → trySetStatus(status). Cute Charm rolls randomChance(1,3) then
    // addVolatile("attract") — the CONTACT_ATTRACT row below (`gen3_ability_batch4_v1`).
    const rc = dhSrc.match(/randomChance\((\d+),\s*(\d+)\)/);
    // CONTACT_ATTRACT (`gen3_ability_batch4_v1`) — Cute Charm: the same DamagingHit-position
    // contact roll, but on a pass it adds the ATTRACT volatile to the attacker (the gender
    // gate lives inside attract.onStart — the roll itself is UNCONDITIONAL, probe-settled by
    // probe_cutecharm_attract_rng.js).
    if (rc && /addVolatile\("attract"/.test(dhSrc)) {
      mech.contactAttract = { chance: [Number(rc[1]), Number(rc[2])] };
    } else if (rc && /trySetStatus/.test(dhSrc)) {
      const sampleM = dhSrc.match(/sample\(\[([^\]]+)\]\)/);
      let statuses;
      let sample = false;
      if (sampleM) {
        // The nested sample list (Effect Spore): the quoted status ids in order.
        statuses = [...sampleM[1].matchAll(/"(\w+)"/g)].map((m) => m[1]);
        sample = true;
      } else {
        // A single literal: `trySetStatus("par", ...)`.
        const one = dhSrc.match(/trySetStatus\("(\w+)"/);
        statuses = one ? [one[1]] : [];
      }
      if (statuses.length) {
        mech.contactProc = { statuses, chance: [Number(rc[1]), Number(rc[2])], sample };
      }
    } else if (/this\.damage\(/.test(dhSrc) && /baseMaxhp/.test(dhSrc)) {
      // Rough Skin: DRAW-FREE contact recoil (`this.damage(source.baseMaxhp / 16, ...)`).
      mech.contactRecoil = true;
    }
    // The `onDamagingHitOrder` SORT KEY (`gen3_damaging_hit_order_v1`). `DamagingHit` is one
    // of the four events sorted by `compareLeftToRightOrder` (battle.ts:421) = ASCENDING
    // `order` (a MISSING order ⇒ 4294967296), then priority, then gather `index`. So an
    // ordered handler runs BEFORE every un-ordered one — including the DEFENDER's `frz` thaw
    // (conditions.ts, no order) — whereas the un-ordered abilities (Static / Poison Point /
    // Flame Body / Effect Spore / Cute Charm / Color Change) fall back to the gather order
    // (status → ability) and so run AFTER it. In the whole gen3 ability set (nums 1-76) the
    // ONLY carrier is roughskin (order 1); aftermath / electromorphosis / innards-out /
    // iron-barbs / wind-power are all later gens.
    if (typeof inv.onDamagingHitOrder === 'number') {
      mech.damagingHitOrder = inv.onDamagingHitOrder;
    }
  }
  // Soundproof — an `onTryHit` immune to a `move.flags["sound"]` move.
  const thSrc = typeof inv.onTryHit === 'string' ? inv.onTryHit : '';
  if (thSrc && /move\.flags\["sound"\]/.test(thSrc) && /return null/.test(thSrc)) {
    mech.blocksSound = true;
  }
  // Damp — an `onAnyTryMove` that cancels the explosion family (returns false, no self-KO).
  const atmSrc = typeof inv.onAnyTryMove === 'string' ? inv.onAnyTryMove : '';
  if (atmSrc && /"explosion"/.test(atmSrc) && /"selfdestruct"/.test(atmSrc) && /return false/.test(atmSrc)) {
    mech.blocksExplosion = true;
  }
  // Suction Cups — an `onDragOut` that returns null (blocks a phaze drag).
  const doSrc = typeof inv.onDragOut === 'string' ? inv.onDragOut : '';
  if (doSrc && /return null/.test(doSrc)) {
    mech.blocksPhazeDrag = true;
  }
  // Synchronize — an `onAfterSetStatus` that reflects the status to the source (trySetStatus).
  const assSrc = typeof inv.onAfterSetStatus === 'string' ? inv.onAfterSetStatus : '';
  if (assSrc && /trySetStatus/.test(assSrc) && /source/.test(assSrc)) {
    mech.synchronize = true;
  }
  // Shed Skin (`gen3_berry_trace_shedskin_v1`) — an onResidual `randomChance(33, 100)` +
  // cureStatus (order 10 subOrder 3 in the resolved gen4-mod override gen3 inherits).
  const resSrc = typeof inv.onResidual === 'string' ? inv.onResidual : '';
  if (resSrc && /randomChance\(33, 100\)/.test(resSrc) && /cureStatus/.test(resSrc)) {
    mech.shedSkin = true;
  }
  // Trace (`gen3_berry_trace_shedskin_v1`) — the gen3-mod onStart `side.randomFoe()` +
  // `setAbility` copy (the base/gen4 seek/onUpdate machinery is mod-REPLACED; Frisk's
  // randomFoe reports an ITEM, no setAbility → not a member).
  const stSrc = typeof inv.onStart === 'string' ? inv.onStart : '';
  if (stSrc && /randomFoe\(\)/.test(stSrc) && /setAbility/.test(stSrc)) {
    mech.trace = true;
  }
  // WONDER GUARD (`gen3_wonder_guard_v1`) — the SE-only damage gate: the gen4-override
  // (gen3-inherited) `onTryHit` blocks a damaging move unless `runEffectiveness(move) > 0`
  // AND not type-immune, emitting `-immune ... [from] ability: Wonder Guard`. Detected by the
  // resolved handler's `runEffectiveness(move) <= 0` guard.
  const wgSrc = typeof inv.onTryHit === 'string' ? inv.onTryHit : '';
  if (wgSrc && /runEffectiveness\(move\) <= 0/.test(wgSrc) && /Wonder Guard/.test(wgSrc)) {
    mech.wonderGuard = true;
  }
  return { inv, mech };
}

// The gen3 MAJOR statuses (the STATUS_IMMUNE class covers only these — a volatile like
// confusion/attract/flinch is a different mechanism).
const MAJOR_STATUSES = new Set(['par', 'slp', 'psn', 'tox', 'brn', 'frz']);

// Derive the STATUS_IMMUNE {statuses, phase} from an ability's resolved onSetStatus /
// onImmunity handler source. Returns null for a non-member. The statuses are emitted in a
// FIXED order (the MAJOR_STATUSES iteration order) so the JSON.stringify equality in --check
// is order-stable regardless of the source's literal order.
const _STATUS_ORDER = ['par', 'slp', 'psn', 'tox', 'brn', 'frz'];
function extractStatusImmune(inv) {
  // onSetStatus: gather every `status.id === "X"` / `status.id !== "X"` MAJOR-status literal.
  const setSrc = typeof inv.onSetStatus === 'string' ? inv.onSetStatus : '';
  if (setSrc) {
    const ids = new Set([...setSrc.matchAll(/status\.id (?:!==|===) "(\w+)"/g)].map((m) => m[1]));
    const majors = [..._STATUS_ORDER].filter((s) => ids.has(s) && MAJOR_STATUSES.has(s));
    if (majors.length) return { statuses: majors, phase: 'setStatus' };
  }
  // onImmunity: a MAJOR-status `type === "X" ... return false`.
  const immSrc = typeof inv.onImmunity === 'string' ? inv.onImmunity : '';
  if (immSrc) {
    const ids = new Set([...immSrc.matchAll(/type === "(\w+)"/g)].map((m) => m[1]));
    const majors = [..._STATUS_ORDER].filter((s) => ids.has(s) && MAJOR_STATUSES.has(s));
    if (majors.length) return { statuses: majors, phase: 'immunity' };
  }
  return null;
}

function classifyAbility(id, ab, ex) {
  if (ABILITY_CLASS_OVERRIDES[id]) return ABILITY_CLASS_OVERRIDES[id];
  if (Object.keys(ex.inv).length === 0) return 'NO_OP';
  return 'UNCLASSIFIED';
}

// ---------------------------------------------------------------------------
// Build the inventory.
// ---------------------------------------------------------------------------
function buildItems() {
  const rows = [];
  for (const it of d3.items.all()) {
    const id = it.id;
    if (!it.exists || !(it.num > 0)) continue;
    if (!(it.gen <= 3 || GEN4_ITEMS_APPLIED_IN_GEN3.has(id))) continue;
    const ex = extractItem(id, it);
    const cls = classifyItem(id, it, ex);
    // kingsrock: its onModifyMove ADDS a 10% flinch secondary -> an EXTRA secondary
    // random(100) at apply time (the draw is downstream of the handler source).
    // quickclaw: the gen3-resolved ITEM entry carries only a bare priority number —
    // the randomChance(1,5) lives in the gen3 battle scripts' action loop (the port
    // models exactly that draw, validated bit-for-bit) -> draw-bearing by fiat.
    const drawBearing = /randomChance|this\.random\(|this\.sample\(|randomFoe\(/.test(fnSrc(ex.inv)) ||
      id === 'kingsrock' || id === 'quickclaw';
    rows.push({
      id, name: it.name, num: it.num, gen: it.gen, cls,
      drawBearing, mech: ex.mech, handlers: Object.keys(ex.inv).sort(),
      modeled: e2e.MODELED_ITEMS.has(id), notes: ex.notes,
    });
  }
  rows.sort((a, b) => (a.cls === b.cls ? (a.id < b.id ? -1 : 1) : (a.cls < b.cls ? -1 : 1)));
  return rows;
}

function buildAbilities() {
  const rows = [];
  for (const ab of d3.abilities.all()) {
    const id = ab.id;
    if (!ab.exists || ab.num <= 0 || ab.num > 76) continue; // gen3 = ability nums 1..76
    const ex = extractAbility(id, ab);
    const cls = classifyAbility(id, ab, ex);
    const drawBearing = /randomChance|this\.random\(|this\.sample\(|randomFoe\(/.test(fnSrc(ex.inv));
    rows.push({
      id, name: ab.name, num: ab.num, cls, drawBearing, mech: ex.mech,
      handlers: Object.keys(ex.inv).sort(),
      modeled: e2e.MODELED_ABILITIES.has(id), noop: e2e.NOOP_ABILITIES.has(id),
    });
  }
  rows.sort((a, b) => (a.cls === b.cls ? (a.id < b.id ? -1 : 1) : (a.cls < b.cls ? -1 : 1)));
  return rows;
}

// ---------------------------------------------------------------------------
// --check: the committed data/pokemon/gen3_items.json mechanics fields must
// EXACTLY match the resolved-dist extraction (the extractor-drift gate).
// ---------------------------------------------------------------------------
function checkItemsJson(items) {
  const committed = JSON.parse(fs.readFileSync(ITEMS_JSON, 'utf8'));
  const errors = [];
  const mechanicsKeys = ['typeBoost', 'statMods', 'onlySpecies', 'untransformedOnly', 'choice', 'isBerry', 'critBoost', 'boostRestore', 'accMod', 'berryEffect',
    'flinchSecondary', 'surviveLethal'];
  for (const row of items) {
    const c = committed[row.id];
    if (!c) { errors.push(`${row.id}: MISSING from gen3_items.json`); continue; }
    for (const k of mechanicsKeys) {
      const want = row.mech[k];
      const got = c[k];
      if (JSON.stringify(want) !== JSON.stringify(got)) {
        errors.push(`${row.id}.${k}: committed ${JSON.stringify(got)} != resolved ${JSON.stringify(want)}`);
      }
    }
  }
  // No committed entry may carry mechanics fields the resolved dist doesn't have.
  const byId = new Map(items.map((r) => [r.id, r]));
  for (const [id, c] of Object.entries(committed)) {
    for (const k of mechanicsKeys) {
      if (c[k] !== undefined && !byId.has(id)) {
        errors.push(`${id}.${k}: committed mechanics on an id NOT in the resolved gen3 universe`);
      }
    }
  }
  return errors;
}

function checkAbilitiesJson(abilities) {
  const committed = JSON.parse(fs.readFileSync(ABILITIES_JSON, 'utf8'));
  const errors = [];
  const abilityKeys = ['dmgMod', 'accMod', 'statusImmune', 'critImmune', 'weatherSpeed', 'weatherNegate',
    'contactProc', 'contactRecoil', 'blocksSound', 'blocksExplosion', 'blocksPhazeDrag', 'synchronize', 'shedSkin', 'trace',
    'contactAttract', 'wonderGuard', 'damagingHitOrder'];
  const byId = new Map(abilities.map((r) => [r.id, r]));
  for (const row of abilities) {
    const c = committed[row.id];
    if (!c) { errors.push(`${row.id}: MISSING from gen3_abilities.json`); continue; }
    for (const k of abilityKeys) {
      const want = row.mech[k];
      const got = c[k];
      if (JSON.stringify(want) !== JSON.stringify(got)) {
        errors.push(`${row.id}.${k}: committed ${JSON.stringify(got)} != resolved ${JSON.stringify(want)}`);
      }
    }
  }
  // No committed entry may carry a mechanics field the resolved dist doesn't have.
  for (const [id, c] of Object.entries(committed)) {
    for (const k of abilityKeys) {
      if (c[k] !== undefined && !byId.has(id)) {
        errors.push(`${id}.${k}: committed mechanics on an id NOT in the resolved gen3 universe`);
      }
    }
  }
  return errors;
}

// ---------------------------------------------------------------------------
// --check (SPECIES COVERAGE, gen3_species_formes_v1): the committed
// data/pokemon/gen3_species.json must contain EXACTLY the gen-3 species universe
// the RESOLVED `Dex.mod('gen3')` has — every BASE form plus every gen-3-legal
// ALTERNATE forme (Deoxys-Attack/Defense/Speed, Castform-Sunny/Rainy/Snowy) plus
// every COSMETIC forme (the 27 Unown letters, which have no Pokedex entry of their
// own and are synthesized from the base).
//
// WHY this gate exists: the extractor used to drop EVERY non-base forme
// (`baseSpecies != id`) because poke-env's static pokedex is not gen-filtered by
// forme — it carries 135 formes with a gen-3 `num` (Megas / Gmax / Alolan /
// Galarian / Hisuian / Paldean / Pikachu cosmetics / Totems), all post-gen-3. That
// blanket filter also dropped the SIX real gen-3 formes + the Unown letters, so the
// port could not construct 6.6% of gen3 random-battle TEAMS (~14% of battles): the
// single largest non-pool coverage gap, and a DATA gap, not an engine one.
//
// The gen-3 legality predicate is the mod-chain-resolved one — `exists &&
// !isNonstandard && gen <= 3` (the gen3 mod marks every later forme
// `isNonstandard: 'Future'`); the static pokedex's own `gen` field is present on
// only 31 of 140 forme entries and cannot be used. Hence the curated
// `_GEN_ALT_FORMES` table in sync.py, gated here.
// ---------------------------------------------------------------------------
const SPECIES_JSON = path.join(REPO, 'data/pokemon/gen3_species.json');
const GEN3_MAX_SPECIES_NUM = 386;

function gen3SpeciesUniverse() {
  const universe = new Map();   // id -> the expected committed row
  const add = (id, s, baseId) => {
    const row = {
      baseStats: {
        atk: s.baseStats.atk, def: s.baseStats.def, hp: s.baseStats.hp,
        spa: s.baseStats.spa, spd: s.baseStats.spd, spe: s.baseStats.spe,
      },
      name: s.name,
      num: s.num,
      types: s.types.map((t) => t.toUpperCase()),
    };
    if (s.maxHP) row.maxHP = s.maxHP;
    if (s.weightkg !== undefined) row.weighthg = Math.round(s.weightkg * 10);
    if (s.gender) row.gender = s.gender;
    if (baseId) {
      row.baseSpecies = baseId;
      if (s.battleOnly) row.battleOnly = toId(s.battleOnly);
    }
    universe.set(id, row);
  };
  for (const id of Object.keys(Dex.data.Pokedex)) {
    const s = d3.species.get(id);
    if (!s || !s.exists || s.isNonstandard) continue;
    if (s.gen > 3) continue;
    if (!(s.num > 0 && s.num <= GEN3_MAX_SPECIES_NUM)) continue;
    const baseId = toId(s.baseSpecies) === id ? null : toId(s.baseSpecies);
    add(id, s, baseId);
    // COSMETIC formes carry no Pokedex entry — synthesize them off the base, exactly
    // as `dex-species.ts` does (a clone with only name/forme/baseSpecies changed).
    if (!baseId) {
      for (const formeName of s.cosmeticFormes || []) {
        const fid = toId(formeName);
        if (universe.has(fid)) continue;
        const f = d3.species.get(fid);
        add(fid, f, id);
      }
    }
  }
  return universe;
}

function checkSpeciesJson() {
  const committed = JSON.parse(fs.readFileSync(SPECIES_JSON, 'utf8'));
  const universe = gen3SpeciesUniverse();
  const errors = [];
  for (const [id, want] of universe) {
    const got = committed[id];
    if (!got) {
      errors.push(`${id}: MISSING from gen3_species.json (a gen-3 species the data layer cannot describe)`);
      continue;
    }
    for (const k of ['num', 'name', 'baseStats', 'types', 'maxHP', 'weighthg', 'gender', 'baseSpecies', 'battleOnly']) {
      if (JSON.stringify(want[k]) !== JSON.stringify(got[k])) {
        errors.push(`${id}.${k}: committed ${JSON.stringify(got[k])} != resolved ${JSON.stringify(want[k])}`);
      }
    }
  }
  for (const id of Object.keys(committed)) {
    if (!universe.has(id)) errors.push(`${id}: committed but NOT in the resolved gen-3 species universe`);
  }
  return errors;
}

// ---------------------------------------------------------------------------
// The markdown class map.
// ---------------------------------------------------------------------------
function writeMd(items, abilities) {
  const lines = [];
  lines.push('# gen3 mechanics inventory — the CLASS MAP (data-driven framework)');
  lines.push('');
  lines.push('GENERATED by `harness/dump_gen3_mechanics.js` from the RESOLVED `Dex.mod(\'gen3\')`');
  lines.push('(the whole mod chain applied — never raw data files; see the mod-chain law).');
  lines.push('Regenerate: `node src/rust_sim/harness/dump_gen3_mechanics.js`.');
  lines.push('Drift gate: `node src/rust_sim/harness/dump_gen3_mechanics.js --check` (verifies the');
  lines.push('committed `data/pokemon/gen3_items.json` / `gen3_abilities.json` mechanics fields —');
  lines.push('and the full `gen3_species.json` universe incl. the gen-3 FORMES — against the');
  lines.push('resolved dist; run it whenever any of them regenerates).');
  lines.push('');
  lines.push('Universe: every item with `num>0 && gen<=3` in the resolved gen3 dex, PLUS the 4');
  lines.push('gen4-named incenses the sim still applies under gen3 formats (odd/rock/rose/wave —');
  lines.push('documented exceptions, present in MODELED_ITEMS); every ability num 1..76.');
  lines.push('');
  lines.push('Legend: **modeled** = the port prices it today (engine, via the data-driven path or a');
  lines.push('dedicated mechanic); *noop-admitted* = provably no-op in the e2e\'s modeled-move fuzz');
  lines.push('(NOOP_ABILITIES); unmodeled = not yet priced (its class is a future phase). DRAW =');
  lines.push('draw-bearing (rolls the PRNG when it fires -> its class needs a draw-order probe');
  lines.push('before wiring); draw-free otherwise.');
  lines.push('');

  const fmtMech = (m) => {
    const parts = [];
    if (m.typeBoost) parts.push(`typeBoost ${m.typeBoost.type} x${m.typeBoost.mod[0]}/${m.typeBoost.mod[1]} @${m.typeBoost.fold}`);
    if (m.statMods) parts.push(`statMods ${Object.entries(m.statMods).map(([s, r]) => `${s} x${r[0]}/${r[1]}`).join(', ')}`);
    if (m.onlySpecies) parts.push(`only [${m.onlySpecies.join(', ')}]${m.untransformedOnly ? ' untransformed' : ''}`);
    if (m.choice) parts.push('choice-lock');
    if (m.isBerry) parts.push('berry');
    if (m.dmgMod) {
      const d = m.dmgMod;
      parts.push(`dmgMod x${d.mod[0]}/${d.mod[1]} @${d.fold}${d.type ? ` type=${d.type}` : ''}${d.pinch ? ' pinch<=1/3' : ''}${d.whenStatused ? ' whenStatused' : ''}${d.direct ? ' DIRECT-modify' : ''}`);
    }
    if (m.accMod) {
      const a = m.accMod;
      const factor = a.op === 'chain' ? `x${a.mod[0]}/${a.mod[1]} chain` : `x${a.mod} DIRECT`;
      parts.push(`accMod ${factor} @${a.side}${a.weather ? ` in-${a.weather}` : ''}${a.physicalTypesOnly ? ' physType' : ''}`);
    }
    if (m.statusImmune) {
      const si = m.statusImmune;
      parts.push(`statusImmune [${si.statuses.join(',')}] @${si.phase}`);
    }
    return parts.join('; ') || '-';
  };

  const section = (title, rows, isAbility) => {
    lines.push(`## ${title}`);
    lines.push('');
    const classes = [...new Set(rows.map((r) => r.cls))];
    for (const cls of classes) {
      const members = rows.filter((r) => r.cls === cls);
      const drawn = members.filter((r) => r.drawBearing).length;
      lines.push(`### ${cls} (${members.length} entries, ${drawn} draw-bearing)`);
      lines.push('');
      lines.push('| id | num | params | status | draw | resolved handlers |');
      lines.push('|---|---|---|---|---|---|');
      for (const r of members) {
        const status = r.modeled ? '**modeled**' : (isAbility && r.noop ? '*noop-admitted*' : 'unmodeled');
        lines.push(`| ${r.id} | ${r.num} | ${fmtMech(r.mech)} | ${status} | ${r.drawBearing ? 'DRAW' : 'free'} | ${r.handlers.join(', ') || '-'} |`);
      }
      lines.push('');
    }
  };

  section(`Items (${items.length})`, items, false);
  section(`Abilities (${abilities.length})`, abilities, true);

  lines.push('## Class wiring state (P1 items + P2 ability DMG_MOD + P3 accuracy WIRED)');
  lines.push('');
  lines.push('| class | state | engine path |');
  lines.push('|---|---|---|');
  lines.push('| TYPE_BOOST | **WIRED (data-driven)** | `turn.rs resolve_atk_stat_mods` reads `ItemData.type_boost` -> the stat chain (`fold=stat`), the BP chain (`fold=basePower`), or the BP float-direct (`fold=basePowerDirect`; the runEvent non-integer guard SKIPS the final chain modifier) |');
  lines.push('| SPECIES_STAT | **WIRED (data-driven)** | `ItemData.stat_mods` + `only_species` -> offensive mods into the atk chain, defensive (def/spd) into the NEW `resolve_def_stat_mods` chain (folded after the boost table, before the explosion Def-halve — the `ModifyDef/ModifySpD` event point) |');
  lines.push('| CHOICE | **WIRED (data-driven)** | `ItemData.choice` + `stat_mods.atk` (the lock was already modeled: `choice_locked_move`) |');
  lines.push('| PINCH_BERRY / HEAL_BERRY / CURE_BERRY | DATA-ONLY (`isBerry`) | one consumption mechanism (`eatItem` + the gen3-mod onResidual triggers), a future phase; PINCH members boost draws-free EXCEPT Starf (its onEat `this.sample` DRAWS the random stat — DRAW-tagged); consumption ORDER matters |');
  lines.push('| RESIDUAL_ITEM | **modeled (dedicated)** | Leftovers heal in `run_residuals` (predates the framework) |');
  lines.push('| PROC_ITEM | Quick Claw **modeled (dedicated)**; King\'s Rock / Focus Band unmodeled | draw-bearing — need draw-order probes when built |');
  lines.push('| ACCURACY_ITEM | **WIRED (data-driven)** | `ItemData.acc_mod` (Bright Powder x0.9 / Lax Incense x0.95 DIRECT) folded into `turn.rs::effective_accuracy` — the to-hit roll now = `move.accuracy x acc/eva stage table x accMod`, then `random(100) < effAcc` (DRAW-RELEVANT: a hit/miss flip desyncs the seed) |');
  lines.push('| CRIT_ITEM | **WIRED (data-driven)** | `ItemData.crit_boost` (`critBoost {boost, onlySpecies}`) folds +N crit stages into `effective_crit_ratio` (the Focus Energy precedent): Scope Lens +1 (unconditional), Lucky Punch +2 (Chansey), Stick +2 (Farfetch\'d) — DRAW-FREE (only the `CRIT_MULT` denominator index shifts) |');
  lines.push('| BOOST_RESTORE | **WIRED (data-driven)** | `ItemData.boost_restore` (`boostRestore: true`) — White Herb: restore all NEGATIVE boost stages to 0 + consume, at the after-move / switch-in stat-drop sites (`turn.rs::white_herb_restore`); DRAW-FREE |');
  lines.push('| DRAIN_ITEM / CURE_ITEM / SPEED_MOD | UNMAPPED | each a small dedicated hook |');
  lines.push('| ability DMG_MOD | **WIRED (data-driven)** | the pinch family (BP x1.5 @hp<=1/3) / Huge-Pure (Atk x2) / Guts (Atk x1.5 statused + burn-skip) / Marvel Scale (Def x1.5 statused) fold via `resolve_atk_stat_mods`/`resolve_def_stat_mods`/`resolve_bp_mods`; **Hustle** ships its Atk x1.5 (dmgMod) WITH its acc x0.8 (accMod) in the accuracy phase; Thick Fat keeps the dedicated `defender_thick_fat` |');
  lines.push('| ability ACCURACY | **WIRED (data-driven)** | `AbilityData.acc_mod` — Compound Eyes (x1.3 chain, attacker), Sand Veil (x0.8 chain in sand, defender), Hustle (x3277/4096 chain, attacker, physical-type moves) — folded into `turn.rs::effective_accuracy` (the runEvent integer-guard mirrored: a chain member is SKIPPED when the accuracy is a non-integer float) |');
  lines.push('| ability STATUS_IMMUNE | **WIRED (data-driven)** | `AbilityData.status_immune` — Limber (par) / Insomnia + Vital Spirit (slp) / Immunity (psn,tox) / Water Veil (brn) block via onSetStatus (phase=setStatus, INSIDE runEvent SetStatus, after the clause shuffle drew, DRAW-FREE block); Magma Armor (frz) blocks via onImmunity (phase=immunity, at runStatusImmunity BEFORE the SetStatus event, so NO clause shuffle) — read by `turn.rs::try_set_status` |');
  lines.push('| ability TYPE_ABSORB / BOOST_IMMUNE / SECONDARY_MOD / TRAP / SWITCH_IN (weather+Intimidate) / SWITCH_OUT (Natural Cure) | **modeled (dedicated)** | pre-framework engine paths (see CLAUDE.md module map) |');
  lines.push('| ability CRIT_IMMUNE / WEATHER_SPEED / CONTACT_PROC / RESIDUAL / ON_HIT_MISC / MISC | UNMAPPED (or noop-admitted) | per-class future phases; CONTACT_PROC members are draw-bearing |');
  lines.push('');
  return lines.join('\n') + '\n';
}

// ---------------------------------------------------------------------------
function main() {
  const items = buildItems();
  const abilities = buildAbilities();

  const unclassified = [
    ...items.filter((r) => r.cls.startsWith('UNCLASSIFIED')),
    ...abilities.filter((r) => r.cls.startsWith('UNCLASSIFIED')),
  ];
  if (unclassified.length) {
    console.error('UNCLASSIFIED entries (extend the rules/overrides):');
    for (const r of unclassified) console.error(`  ${r.id}: handlers=${r.handlers.join(',')}`);
    process.exit(1);
  }

  if (process.argv.includes('--json')) {
    console.log(JSON.stringify({ items, abilities }, null, 1));
    return;
  }
  if (process.argv.includes('--check')) {
    const speciesErrors = checkSpeciesJson();
    const errors = [...checkItemsJson(items), ...checkAbilitiesJson(abilities), ...speciesErrors];
    if (errors.length) {
      console.error(`MECHANICS DRIFT (${errors.length}):\n  ` + errors.join('\n  '));
      process.exit(1);
    }
    const nSpecies = Object.keys(JSON.parse(fs.readFileSync(SPECIES_JSON, 'utf8'))).length;
    console.error(`mechanics check OK: ${items.length} items + ${abilities.length} abilities + `
      + `${nSpecies} species match the resolved dist`);
    return;
  }

  fs.writeFileSync(OUT_MD, writeMd(items, abilities));
  const stat = (rows) => {
    const per = {};
    for (const r of rows) per[r.cls] = (per[r.cls] || 0) + 1;
    return Object.entries(per).map(([c, n]) => `${c}=${n}`).join(' ');
  };
  console.error(`items: ${items.length} [${stat(items)}]`);
  console.error(`abilities: ${abilities.length} [${stat(abilities)}]`);
  console.error(`draw-bearing: items=${items.filter((r) => r.drawBearing).length} abilities=${abilities.filter((r) => r.drawBearing).length}`);
  console.error(`-> ${OUT_MD}`);
}

main();
