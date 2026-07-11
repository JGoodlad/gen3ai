// gen_damage_golden.js — Gen-3 single-hit DAMAGE differential harness.
//
// Mirrors src/utils/bridge/damage_probe.js's OMNISCIENT-BattleStream pattern (no
// live server): per scenario, a fresh in-process BattleStream constructs a
// single-turn fight, p1 attacks p2, and we read EXACT both-side stats + the
// realized HP delta off the live battle object — the clean ground truth with zero
// measurement confounds (the sim's own storedStats, exact HP, one modifier each).
//
// EXACTNESS (the whole point): the realized damage is randomizer =
// tr(tr(baseDamage*(100-random(16)))/100), which equals the deterministic pre-roll
// baseDamage ONLY at the max roll (random(16)==0). We FORCE that max roll by
// replaying each scenario under MANY seeds and recording the MAXIMUM realized
// damage observed on the measured hit — that maximum IS baseDamage (r==0). So the
// emitted golden is the EXACT deterministic baseDamage, and tests/damage_test.rs
// asserts Rust's calc_damage().base == it (and rolls[15] == the MIN observed) with
// ZERO tolerance — not a range-membership band.
//
// Each scenario isolates exactly ONE modifier (neutral / STAB / SE / resist / 4x /
// type-immune / ability-immune / Thick Fat / Choice Band / type-item / burn /
// Reflect / Light Screen / rain / sun / +Atk / -Def / defender +Def UNDER crit /
// crit-through-Light-Screen / Explosion def-halve / min-vs-max base stats). Some
// scenarios stage the modifier in earlier turns (Will-O-Wisp, Reflect, boosts).
//
// Output: one self-contained TAB-delimited golden line per scenario the Rust test
// parses with std only — it carries the full reconstructable context (both sides'
// stats/types/boosts/status, the move's BP/type/category, the field, crit) plus
// the EXACT max-roll baseDamage and the min roll:
//
//   DMG <id> <fields...>   (see the header comment block written to the file)
//
// Run:  node src/rust_sim/harness/gen_damage_golden.js
// (Needs the submodule dist/ + node_modules symlinks; see root CLAUDE.md.)

'use strict';

const path = require('path');
const fs = require('fs');

const PS = path.resolve(__dirname, '../../../deps/pokemon-showdown');
const { BattleStream, getPlayerStreams } = require(path.join(PS, 'dist/sim/battle-stream'));
const { Teams } = require(path.join(PS, 'dist/sim'));

const OUT = path.resolve(__dirname, '../tests/vectors/damage_golden.txt');
const FORMAT = 'gen3customgame';
const IV31 = { hp: 31, atk: 31, def: 31, spa: 31, spd: 31, spe: 31 };
const EV0 = { hp: 0, atk: 0, def: 0, spa: 0, spd: 0, spe: 0 };

// Seeds to sweep per scenario. The randomizer draw lands one of r in 0..15; a few
// dozen distinct seeds reliably surface r==0 (the max roll) for a NON-crit hit
// (every seed hits). CRIT scenarios are harder: a crit is ~1/16 per seed, so we
// need a MUCH larger pool to collect enough crit samples that r==0 appears among
// them (verified by a stabilization check in `measure`). The gen5 LCG seed is 4
// 16-bit words; we build a large, well-spread pool deterministically.
function buildSeeds(n) {
  const out = [];
  // A simple deterministic LCG over the seed words → well-spread, reproducible.
  let x = 0x12345 >>> 0;
  const step = () => { x = (Math.imul(x, 1103515245) + 12345) >>> 0; return x & 0xffff; };
  for (let i = 0; i < n; i++) out.push([step() || 1, step() || 1, step() || 1, step() || 1]);
  return out;
}
// Non-crit scenarios converge in a few dozen; crit scenarios sweep the whole pool.
const SEEDS = buildSeeds(2400);
const NONCRIT_SEED_BUDGET = 80; // a non-crit max-roll appears well within this

function mon(species, moves, opts = {}) {
  return {
    species,
    item: opts.item || '',
    ability: opts.ability || 'No Ability',
    moves,
    evs: { ...EV0, ...(opts.evs || {}) },
    ivs: opts.ivs || IV31,
    nature: opts.nature || 'Serious',
    level: opts.level || 100,
    gender: opts.gender || 'N',
  };
}

function tick() { return new Promise((r) => setTimeout(r, 0)); }

function moveId(disp) { return String(disp).toLowerCase().replace(/[^a-z0-9]/g, ''); }

function parseHp(field, maxhp) {
  // "h/m" or "h/m status" or "0 fnt".
  const tok = String(field).trim().split(' ')[0];
  if (tok === '0' || tok.startsWith('0 ')) return 0;
  const [num, den] = tok.split('/');
  if (den === undefined) {
    // a percent form shouldn't happen in the omniscient log, but guard anyway
    return null;
  }
  const n = parseInt(num, 10);
  if (Number.isNaN(n)) return null;
  return n; // omniscient log carries exact HP (num/maxhp)
}

// Walk the omniscient log: return the LAST p1->p2 attacking hit's exact damage (in
// HP points) + whether it crit + whether it KO'd + the defender's HP right BEFORE
// that hit (so the caller can require a clean full-HP hit) + whether p1 was burned
// when it struck. None if no clean p1->p2 move is found.
//
// HP tracking is robust to interleaved heals/residuals (Rest, Leftovers, Toxic):
// `hp` follows every -damage/-heal/-sethp line, so the measured hit's damage is
// `hpBefore - hpAfter` at the moment of THAT move's -damage line.
function lastAttack(res) {
  const maxhp = res.p2.maxhp;
  let hp = maxhp;
  let last = null;
  let pending = null;   // moveId of an in-flight p1->p2 move awaiting its outcome
  let crit = false;
  let p1Burned = false; // is p1 burned at the time the pending move strikes?

  for (const line of res.log) {
    const parts = line.split('|');
    if (parts.length < 2) continue;
    const tag = parts[1];
    if (tag === 'move' && parts.length >= 5) {
      const actor = parts[2].trim();
      const target = parts[4].trim();
      if (actor.startsWith('p1a:') && target.startsWith('p2a:')) {
        pending = moveId(parts[3]);
        crit = false;
      } else {
        pending = null;
      }
    } else if (tag === '-status' && parts.length >= 4 && parts[2].trim().startsWith('p1a:')) {
      if (parts[3].trim() === 'brn') p1Burned = true;
    } else if (tag === '-curestatus' && parts.length >= 4 && parts[2].trim().startsWith('p1a:')) {
      if (parts[3].trim() === 'brn') p1Burned = false;
    } else if (tag === '-crit') {
      crit = true;
    } else if ((tag === '-immune' || tag === '-fail') && parts.length >= 3 && parts[2].trim().startsWith('p2a:')) {
      // An immune/failed p1->p2 move is a clean 0-damage "hit" (type/ability
      // immunity). Record it so the immunity scenarios resolve to base=0.
      if (pending !== null) {
        last = { move: pending, dmg: 0, crit: false, fainted: false, immune: true, hpBefore: hp, p1Burned };
        pending = null;
      }
    } else if (tag === '-start' && parts.length >= 4 && parts[2].trim().startsWith('p2a:')
               && parts[3].startsWith('ability:')) {
      // Flash Fire absorbs the Fire move via |-start|p2a:..|ability: Flash Fire|
      // (a 0-damage ability immunity for the pending p1->p2 hit).
      if (pending !== null) {
        last = { move: pending, dmg: 0, crit: false, fainted: false, immune: true, hpBefore: hp, p1Burned };
        pending = null;
      }
    } else if (tag === '-damage' && parts.length >= 4 && parts[2].trim().startsWith('p2a:')) {
      const isResidual = parts.slice(4).some((p) => p.startsWith('[from]'));
      const newHp = parseHp(parts[3], maxhp);
      if (newHp === null) continue;
      if (pending !== null && !isResidual) {
        last = { move: pending, dmg: hp - newHp, crit, fainted: newHp <= 0, immune: false, hpBefore: hp, p1Burned };
        pending = null;
      }
      hp = newHp;
    } else if ((tag === '-heal' || tag === '-sethp') && parts.length >= 4 && parts[2].trim().startsWith('p2a:')) {
      const nh = parseHp(parts[3], maxhp);
      if (nh !== null) hp = nh;
    }
  }
  return last;
}

function snapSide(side) {
  const a = side && side.active && side.active[0];
  if (!a) return null;
  let types;
  try { types = a.getTypes(); } catch (e) { types = a.types; }
  return {
    species: (a.species && a.species.name) || String(a.species),
    maxhp: a.maxhp,
    hp: a.hp,
    stats: a.storedStats, // {atk,def,spa,spd,spe}
    boosts: a.boosts,     // {atk,def,spa,spd,spe,accuracy,evasion}
    status: a.status || '',
    item: a.item || '',
    ability: a.ability || '',
    types,
    sideConditions: Object.keys(side.sideConditions || {}),
  };
}

async function runOnce(sc, seed) {
  const stream = new BattleStream();
  const streams = getPlayerStreams(stream);
  const log = [];
  (async () => { for await (const ch of streams.omniscient) { for (const l of ch.split('\n')) if (l) log.push(l); } })();

  streams.omniscient.write(`>start {"formatid":"${FORMAT}","seed":${JSON.stringify(seed)}}`);
  streams.omniscient.write(`>player p1 ${JSON.stringify({ name: 'P1', team: Teams.pack([sc.p1]) })}`);
  streams.omniscient.write(`>player p2 ${JSON.stringify({ name: 'P2', team: Teams.pack([sc.p2]) })}`);
  // A CONSTRUCTED single-turn hook (the `damage_op_probe.js` philosophy) for the ability
  // DMG_MOD scenarios: force the attacker's PINCH HP (`setHp` — the pinch family reads
  // `attacker.hp <= maxhp/3`) and/or a major status on either mon (Guts reads the
  // attacker's status, Marvel Scale the defender's) DIRECTLY on the live battle object,
  // BEFORE the measured turn resolves — deterministic, draw-free, no fragile staging.
  // The item scenarios never set these, so they are byte-identical to the old path.
  if (sc.atkHp != null || sc.atkStatus || sc.defStatus) {
    for (let i = 0; i < 4; i++) await tick(); // let construction settle (leads in)
    const b = stream.battle;
    const a1 = b.sides[0].active[0];
    const a2 = b.sides[1].active[0];
    if (sc.atkHp != null && a1) a1.hp = Math.max(1, Math.round(a1.maxhp * sc.atkHp));
    if (sc.atkStatus && a1) a1.setStatus(sc.atkStatus);
    if (sc.defStatus && a2) a2.setStatus(sc.defStatus);
  }
  for (const [side, choice] of sc.choices) streams.omniscient.write(`>${side} ${choice}`);

  for (let i = 0; i < 8; i++) await tick();

  const weather = (stream.battle.field && stream.battle.field.weather) || '';
  const out = {
    weather,
    p1: snapSide(stream.battle.sides[0]),
    p2: snapSide(stream.battle.sides[1]),
    log,
  };
  try { streams.omniscient.destroy(); } catch (e) { /* best effort */ }
  return out;
}

// ── Scenarios. attacker = p1, defender = p2. The measured hit is p1's LAST move
//    on p2. `wantCrit` selects which crit polarity the max-roll search keeps. ──
function scenarios() {
  const S = [];
  const add = (id, p1, p2, choices, opts = {}) =>
    S.push({
      id, p1, p2, choices,
      wantCrit: !!opts.wantCrit, requireBurn: !!opts.requireBurn,
      // gen3_item_mechanics_v1 (ability side) — constructed pinch-HP / status hooks.
      atkHp: opts.atkHp != null ? opts.atkHp : null,
      atkStatus: opts.atkStatus || null,
      defStatus: opts.defStatus || null,
    });

  const T1 = [['p1', 'move 1'], ['p2', 'move 1']];
  // p1 SETS UP on T1 (move 1 = a non-damaging setup), attacks on T2 (move 2). The
  // defender (p2 move 1) idles, so it is at FULL HP for the measured T2 hit.
  const T2_p1mv2 = [['p1', 'move 1'], ['p2', 'move 1'], ['p1', 'move 2'], ['p2', 'move 1']];
  // p2 SETS UP on T1 (move 1 = the modifier: screen/boost/burn) while p1 IDLES via
  // Protect (move 1, no stat effect, no hit). Then p1 attacks on T2 (move 2) into a
  // FULL-HP defender, which idles (p2 move 2 = a non-healing move). The defender
  // never takes damage before the measured hit, so hpBefore == maxhp holds.
  const STAGE_P2 = [['p1', 'move 1'], ['p2', 'move 1'], ['p1', 'move 2'], ['p2', 'move 2']];

  // 1. Neutral special, no STAB (Alakazam Shadow Ball -> Snorlax; Ghost vs Normal
  //    = 0x! use Psychic... Psychic vs Normal neutral, but Psychic is STAB on
  //    Alakazam). Use Magnemite Flash? Keep simple: Vaporeon Shadow Ball ->
  //    Blissey: Ghost vs Normal is 0x. Instead Glalie Shadow Ball -> ... messy.
  //    Cleanest non-STAB neutral: Snorlax Shadow Ball -> Suicune (Ghost vs Water =
  //    neutral, Snorlax isn't Ghost so no STAB).
  add('neutral_special_nostab',
    mon('Snorlax', ['shadowball'], { nature: 'Modest', evs: { spa: 252 } }),
    mon('Suicune', ['rest'], { nature: 'Calm', evs: { hp: 252, spd: 252 } }),
    T1);
  // 2. Neutral physical, no STAB (Snorlax Earthquake -> Suicune; Ground vs Water
  //    neutral, Snorlax not Ground -> no STAB).
  add('neutral_physical_nostab',
    mon('Snorlax', ['earthquake'], { nature: 'Adamant', evs: { atk: 252 } }),
    mon('Suicune', ['rest'], { nature: 'Bold', evs: { hp: 252, def: 252 } }),
    T1);
  // 3. STAB special (Suicune Surf -> Blissey, neutral + STAB).
  add('stab_special',
    mon('Suicune', ['surf'], { nature: 'Modest', evs: { spa: 252 } }),
    mon('Blissey', ['softboiled'], { nature: 'Calm', evs: { hp: 252, spd: 252 } }),
    T1);
  // 4. STAB physical (Snorlax Body Slam -> Suicune, neutral + STAB).
  add('stab_physical',
    mon('Snorlax', ['bodyslam'], { nature: 'Adamant', evs: { atk: 252 } }),
    mon('Suicune', ['rest'], { nature: 'Bold', evs: { hp: 252, def: 252 } }),
    T1);
  // 5. Super-effective 2x (Starmie Thunderbolt -> Suicune, Electric vs Water = 2x).
  add('super_effective_2x',
    mon('Starmie', ['thunderbolt'], { nature: 'Modest', evs: { spa: 252 } }),
    mon('Suicune', ['rest'], { nature: 'Calm', evs: { hp: 252, spd: 252 } }),
    T1);
  // 6. Resisted 0.5x (Jolteon Thunderbolt -> Zapdos, Electric vs Electric = 0.5x).
  add('resisted_half',
    mon('Jolteon', ['thunderbolt'], { nature: 'Modest', evs: { spa: 252 } }),
    mon('Zapdos', ['rest'], { nature: 'Calm', evs: { hp: 252, spd: 252 } }),
    T1);
  // 7. Quad effective 4x (Cloyster Ice Beam -> Salamence, Ice vs Dragon/Flying =4x).
  //    Use an UN-INVESTED, low-SpA Cloyster (no EVs, neutral) into a max-bulk
  //    Careful Salamence so the 4x hit does NOT KO (exact, not capped).
  add('quad_effective_4x',
    mon('Cloyster', ['icebeam'], { nature: 'Serious' }),
    mon('Salamence', ['rest'], { nature: 'Careful', evs: { hp: 252, spd: 252 } }),
    T1);
  // 8. Type immunity (Raikou Thunderbolt -> Swampert, Electric vs Ground = 0x).
  add('type_immune',
    mon('Raikou', ['thunderbolt'], { nature: 'Modest', evs: { spa: 252 } }),
    mon('Swampert', ['rest'], { nature: 'Bold', evs: { hp: 252 } }),
    T1);
  // 9. Levitate immunity (Tyranitar Earthquake -> Claydol via Levitate = 0x).
  add('ability_immune_levitate',
    mon('Tyranitar', ['earthquake'], { nature: 'Adamant', evs: { atk: 252 }, ability: 'Sand Stream' }),
    mon('Claydol', ['rest'], { nature: 'Bold', evs: { hp: 252 }, ability: 'Levitate' }),
    T1);
  // 10. Flash Fire immunity (Charizard Flamethrower -> Houndoom via Flash Fire 0x).
  add('ability_immune_flashfire',
    mon('Charizard', ['flamethrower'], { nature: 'Modest', evs: { spa: 252 } }),
    mon('Houndoom', ['rest'], { nature: 'Calm', evs: { hp: 252 }, ability: 'Flash Fire' }),
    T1);
  // 11. Water Absorb immunity (Suicune Surf -> Vaporeon via Water Absorb 0x).
  add('ability_immune_waterabsorb',
    mon('Suicune', ['surf'], { nature: 'Modest', evs: { spa: 252 } }),
    mon('Vaporeon', ['rest'], { nature: 'Calm', evs: { hp: 252 }, ability: 'Water Absorb' }),
    T1);
  // 12. Volt Absorb immunity (Jolteon Thunderbolt -> Jolteon via Volt Absorb 0x).
  add('ability_immune_voltabsorb',
    mon('Jolteon', ['thunderbolt'], { nature: 'Modest', evs: { spa: 252 } }),
    mon('Jolteon', ['rest'], { nature: 'Calm', evs: { hp: 252 }, ability: 'Volt Absorb' }),
    T1);
  // 13. Thick Fat 0.5x (Starmie Ice Beam -> Snorlax with Thick Fat halves Ice).
  add('thick_fat_ice',
    mon('Starmie', ['icebeam'], { nature: 'Modest', evs: { spa: 252 } }),
    mon('Snorlax', ['rest'], { nature: 'Careful', evs: { hp: 252, spd: 252 }, ability: 'Thick Fat' }),
    T1);
  // 14. Choice Band 1.5x physical (Tyranitar Rock Slide -> Skarmory).
  add('choice_band',
    mon('Tyranitar', ['rockslide'], { item: 'choiceband', nature: 'Adamant', evs: { atk: 252 }, ability: 'Sand Stream' }),
    mon('Skarmory', ['rest'], { nature: 'Impish', evs: { hp: 252, def: 252 } }),
    T1);
  // 15. Type-boost item 1.1x (Magnet -> Thunderbolt -> Blissey).
  add('type_boost_item',
    mon('Zapdos', ['thunderbolt'], { item: 'magnet', nature: 'Modest', evs: { spa: 252 } }),
    mon('Blissey', ['softboiled'], { nature: 'Calm', evs: { hp: 252, spd: 252 } }),
    T1);
  // 16. Sea Incense 1.05x special (Surf with Sea Incense -> Blissey).
  add('sea_incense',
    mon('Suicune', ['surf'], { item: 'seaincense', nature: 'Modest', evs: { spa: 252 } }),
    mon('Blissey', ['softboiled'], { nature: 'Calm', evs: { hp: 252, spd: 252 } }),
    T1);
  // 17. Burn 0.5x physical (p2 Will-O-Wisp burns p1 T1; p1 NON-secondary physical
  //     Earthquake T2). Earthquake has no self-boost (Meteor Mash's +1 Atk would
  //     break isolation); the defender uses a NON-healing move (Taunt) on T2 so the
  //     measured hit lands at full HP. `requireBurn` skips seeds where W-o-W missed.
  //     3-turn stage: T1 p2 Will-O-Wisp burns p1 (p1 idles via a harmless
  //     SELF-target move that Protect would otherwise block); T2 p2 Rest heals to
  //     full while p1 Protects; T3 p1 Rock Slide into the FULL-HP, now-burned hit.
  //     p1's idle is Curse on a NON-Ghost (Snorlax) — but that boosts, so instead
  //     p1 just attacks T1 (chipping the wall) and the wall Rests to full before T3.
  add('burn_half_physical',
    mon('Tyranitar', ['rockslide', 'protect'], { nature: 'Adamant', evs: { atk: 252 }, ability: 'Sand Stream' }),
    mon('Forretress', ['willowisp', 'rest'], { nature: 'Relaxed', evs: { hp: 252, def: 252 }, ability: 'Sturdy' }),
    [['p1', 'move 1'], ['p2', 'move 1'],   // T1: p1 Rock Slide (chip), p2 W-o-W (burn p1)
     ['p1', 'move 2'], ['p2', 'move 2'],   // T2: p1 Protect, p2 Rest (heal to full)
     ['p1', 'move 1'], ['p2', 'move 2']],  // T3: p1 Rock Slide (measured, full HP), p2 Rest
    { requireBurn: true });
  // 18. Reflect 0.5x physical (p2 Reflect T1 while p1 Protects; p1 physical T2).
  add('reflect_physical',
    mon('Tyranitar', ['protect', 'rockslide'], { nature: 'Adamant', evs: { atk: 252 }, ability: 'Sand Stream' }),
    mon('Skarmory', ['reflect', 'spikes'], { nature: 'Impish', evs: { hp: 252, def: 252 } }),
    STAGE_P2);
  // 19. Light Screen 0.5x special (p2 Light Screen T1 while p1 Protects; p1 special T2).
  add('lightscreen_special',
    mon('Starmie', ['protect', 'surf'], { nature: 'Modest', evs: { spa: 252 } }),
    mon('Blissey', ['lightscreen', 'thunderwave'], { nature: 'Calm', evs: { hp: 252, spd: 252 } }),
    STAGE_P2);
  // 20. Rain 1.5x Water (p1 Rain Dance T1; Surf T2).
  add('rain_water_boost',
    mon('Suicune', ['raindance', 'surf'], { nature: 'Modest', evs: { spa: 252 } }),
    mon('Snorlax', ['rest'], { nature: 'Careful', evs: { hp: 252, spd: 252 } }),
    T2_p1mv2);
  // 21. Sun 0.5x Water (p1 Sunny Day T1; Surf T2).
  add('sun_water_weaken',
    mon('Suicune', ['sunnyday', 'surf'], { nature: 'Modest', evs: { spa: 252 } }),
    mon('Snorlax', ['rest'], { nature: 'Careful', evs: { hp: 252, spd: 252 } }),
    T2_p1mv2);
  // 22. Sun 1.5x Fire (p1 Sunny Day T1; Flamethrower T2).
  add('sun_fire_boost',
    mon('Charizard', ['sunnyday', 'flamethrower'], { nature: 'Modest', evs: { spa: 252 } }),
    mon('Blissey', ['softboiled'], { nature: 'Calm', evs: { hp: 252, spd: 252 } }),
    T2_p1mv2);
  // 23. Rain 0.5x Fire (p1 Rain Dance T1; Flamethrower T2).
  add('rain_fire_weaken',
    mon('Charizard', ['raindance', 'flamethrower'], { nature: 'Modest', evs: { spa: 252 } }),
    mon('Blissey', ['softboiled'], { nature: 'Calm', evs: { hp: 252, spd: 252 } }),
    T2_p1mv2);
  // 24. +2 Atk boost (p1 Swords Dance T1; physical EQ T2).
  add('plus2_atk',
    mon('Salamence', ['swordsdance', 'earthquake'], { nature: 'Adamant', evs: { atk: 252 } }),
    mon('Swampert', ['rest'], { nature: 'Bold', evs: { hp: 252, def: 252 } }),
    T2_p1mv2);
  // 25. +2 SpA boost (p1 Calm Mind T1; special Surf T2).
  add('plus2_spa',
    mon('Suicune', ['calmmind', 'surf'], { nature: 'Modest', evs: { spa: 252 } }),
    mon('Snorlax', ['rest'], { nature: 'Careful', evs: { hp: 252, spd: 252 } }),
    T2_p1mv2);
  // 26. Defender +2 Def reduces a NON-crit physical hit (p2 Iron Defense T1 while
  //     p1 Protects; p1 EQ T2). The boost is the modifier under test.
  add('defender_plus2_def',
    mon('Tyranitar', ['protect', 'earthquake'], { nature: 'Adamant', evs: { atk: 252 }, ability: 'Sand Stream' }),
    mon('Forretress', ['irondefense', 'taunt'], { nature: 'Relaxed', evs: { hp: 252 }, ability: 'Sturdy' }),
    STAGE_P2);
  // 27. Defender +2 Def UNDER CRIT — the crit must IGNORE the positive Def boost,
  //     so realized == the UN-boosted-def crit damage (p2 Iron Defense T1, p1 crits
  //     T2). wantCrit selects the crit hit from the seed sweep; the Rust calc zeros
  //     the defender's +2 Def under crit, so its base must match.
  add('crit_ignores_defender_def',
    mon('Tyranitar', ['protect', 'earthquake'], { nature: 'Adamant', evs: { atk: 252 }, ability: 'Sand Stream' }),
    mon('Forretress', ['irondefense', 'taunt'], { nature: 'Relaxed', evs: { hp: 252 }, ability: 'Sturdy' }),
    STAGE_P2, { wantCrit: true });
  // 28. Crit THROUGH Light Screen — the crit must BYPASS the screen (p2 Light
  //     Screen T1 while p1 Protects; p1 special crit T2). Realized == crit damage
  //     with NO screen halving.
  add('crit_bypasses_lightscreen',
    mon('Starmie', ['protect', 'surf'], { nature: 'Modest', evs: { spa: 252 } }),
    mon('Blissey', ['lightscreen', 'thunderwave'], { nature: 'Calm', evs: { hp: 252, spd: 252 } }),
    STAGE_P2, { wantCrit: true });
  // 29. Explosion def-halve (gen3 halves defender Def for Explosion). Self-Destruct
  //     KOs the user; measure the boom on a bulky p2 (it survives). Use Explosion
  //     off a strong physical attacker into a wall.
  add('explosion_def_halve',
    mon('Snorlax', ['selfdestruct'], { nature: 'Adamant', evs: { atk: 252 } }),
    mon('Skarmory', ['rest'], { nature: 'Impish', evs: { hp: 252, def: 252 } }),
    T1);
  // 30. Min base-stat attacker into max base-stat wall (Magikarp Tackle ->
  //     Shuckle): tiny numbers, floor edges.
  add('min_vs_max_stats',
    mon('Magikarp', ['tackle'], { nature: 'Adamant', evs: { atk: 252 } }),
    mon('Shuckle', ['rest'], { nature: 'Bold', evs: { hp: 252, def: 252 } }),
    T1);
  // 31. -Def defender via Screech is hard one-shot; instead a low-level attacker
  //     (level 50) for level-scaling coverage.
  add('low_level_attacker',
    mon('Snorlax', ['bodyslam'], { nature: 'Adamant', evs: { atk: 252 }, level: 50 }),
    mon('Suicune', ['rest'], { nature: 'Bold', evs: { hp: 252, def: 252 } }),
    T1);

  // ── gen3_item_mechanics_v1: the data-driven item-modifier class members. Each
  //    newly-priced item gets an EXACT max-roll probe (+ wrong-type / wrong-species
  //    controls), pinning the PROBE-SETTLED fold math:
  //      bows     = onBasePower `return basePower * 1.1` (a DIRECT float replace —
  //                 skips the event chain, clampIntRange floors; BP 85 → 93)
  //      incenses = onBasePower chainModify([4915,4096]) ≈ ×1.2 (NOT ×1.1!)
  //      species  = onModifyAtk/SpA/Def/SpD chainModify (gen3-RESOLVED semantics:
  //                 Light Ball = SpA-ONLY ×2; Metal Powder = Def ×2 untransformed) ──
  // 32/33. The gen2 bows (Normal ×1.1 DIRECT at base power). Body Slam BP 85 →
  //        floor(85·1.1)=93 exercises the .5-fraction floor.
  add('pink_bow_normal',
    mon('Snorlax', ['bodyslam'], { item: 'pinkbow', nature: 'Adamant', evs: { atk: 252 } }),
    mon('Suicune', ['rest'], { nature: 'Bold', evs: { hp: 252, def: 252 } }),
    T1);
  add('polkadot_bow_normal',
    mon('Tauros', ['bodyslam'], { item: 'polkadotbow', nature: 'Adamant', evs: { atk: 252 } }),
    mon('Suicune', ['rest'], { nature: 'Bold', evs: { hp: 252, def: 252 } }),
    T1);
  // 34-37. The gen4-named incenses (×4915/4096 at the BASE-POWER chain). Wave
  //        Incense on Surf is the direct contrast with Sea Incense (16): the SAME
  //        move, ×1.2-at-BP vs ×1.05-at-stat.
  add('odd_incense_psychic',
    mon('Alakazam', ['psychic'], { item: 'oddincense', nature: 'Modest', evs: { spa: 252 } }),
    mon('Suicune', ['rest'], { nature: 'Calm', evs: { hp: 252, spd: 252 } }),
    T1);
  add('rock_incense_rock',
    mon('Tyranitar', ['rockslide'], { item: 'rockincense', nature: 'Adamant', evs: { atk: 252 }, ability: 'Sand Stream' }),
    mon('Suicune', ['rest'], { nature: 'Bold', evs: { hp: 252, def: 252 } }),
    T1);
  add('rose_incense_grass',
    mon('Sceptile', ['leafblade'], { item: 'roseincense', nature: 'Adamant', evs: { atk: 252 } }),
    mon('Suicune', ['rest'], { nature: 'Bold', evs: { hp: 252, def: 252 } }),
    T1);
  add('wave_incense_water',
    mon('Suicune', ['surf'], { item: 'waveincense', nature: 'Modest', evs: { spa: 252 } }),
    mon('Blissey', ['softboiled'], { nature: 'Calm', evs: { hp: 252, spd: 252 } }),
    T1);
  // 38. Wrong-type control: an incense holder using a NON-matching move gets NO
  //     boost (Odd Incense + Shadow Ball).
  add('incense_wrong_type_control',
    mon('Alakazam', ['shadowball'], { item: 'oddincense', nature: 'Modest', evs: { spa: 252 } }),
    mon('Suicune', ['rest'], { nature: 'Calm', evs: { hp: 252, spd: 252 } }),
    T1);
  // 39/40. Thick Club (Atk ×2, Cubone/Marowak only) + the wrong-species control.
  add('thick_club_marowak',
    mon('Marowak', ['earthquake'], { item: 'thickclub', nature: 'Adamant', evs: { atk: 252 } }),
    mon('Suicune', ['rest'], { nature: 'Bold', evs: { hp: 252, def: 252 } }),
    T1);
  add('thick_club_wrong_species_control',
    mon('Snorlax', ['earthquake'], { item: 'thickclub', nature: 'Adamant', evs: { atk: 252 } }),
    mon('Suicune', ['rest'], { nature: 'Bold', evs: { hp: 252, def: 252 } }),
    T1);
  // 41/42. gen3 Light Ball: SpA ×2 for Pikachu — and NO Atk boost (the gen4 Atk
  //        half must NOT exist; the physical control pins that).
  add('light_ball_pikachu_spa',
    mon('Pikachu', ['thunderbolt'], { item: 'lightball', nature: 'Modest', evs: { spa: 252 } }),
    mon('Blissey', ['softboiled'], { nature: 'Calm', evs: { hp: 252, spd: 252 } }),
    T1);
  add('light_ball_pikachu_atk_control',
    mon('Pikachu', ['strength'], { item: 'lightball', nature: 'Adamant', evs: { atk: 252 } }),
    mon('Suicune', ['rest'], { nature: 'Bold', evs: { hp: 252, def: 252 } }),
    T1);
  // 43/44. DeepSeaTooth (SpA ×2 Clamperl) + Soul Dew's SpA half (×1.5 Lati@s).
  add('deepseatooth_clamperl',
    mon('Clamperl', ['surf'], { item: 'deepseatooth', nature: 'Modest', evs: { spa: 252 } }),
    mon('Blissey', ['softboiled'], { nature: 'Calm', evs: { hp: 252, spd: 252 } }),
    T1);
  add('souldew_latios_spa',
    mon('Latios', ['psychic'], { item: 'souldew', nature: 'Modest', evs: { spa: 252 } }),
    mon('Blissey', ['softboiled'], { nature: 'Calm', evs: { hp: 252, spd: 252 } }),
    T1);
  // 45-47. The DEFENSE-side members (the ModifyDef/ModifySpD event, folded after
  //        the boost table + before the Explosion def-halve): DeepSeaScale (SpD ×2
  //        Clamperl), Metal Powder (Def ×2 untransformed Ditto — a level-50, no-EV
  //        Snorlax keeps the hit tiny so frail Ditto measures cleanly), Soul Dew's
  //        SpD half (×1.5 Latias).
  add('deepseascale_clamperl_def',
    mon('Suicune', ['surf'], { nature: 'Serious' }),
    mon('Clamperl', ['rest'], { item: 'deepseascale', nature: 'Calm', evs: { hp: 252 } }),
    T1);
  add('metalpowder_ditto_def',
    mon('Snorlax', ['bodyslam'], { nature: 'Serious', level: 50 }),
    mon('Ditto', ['rest'], { item: 'metalpowder', nature: 'Bold', evs: { hp: 252 } }),
    T1);
  add('souldew_latias_spd_def',
    mon('Suicune', ['surf'], { nature: 'Modest', evs: { spa: 252 } }),
    mon('Latias', ['rest'], { item: 'souldew', nature: 'Calm', evs: { hp: 252 } }),
    T1);
  // 48. Wrong-species DEFENSE control: DeepSeaScale on a non-Clamperl does nothing.
  add('deepseascale_wrong_species_def_control',
    mon('Suicune', ['surf'], { nature: 'Modest', evs: { spa: 252 } }),
    mon('Blissey', ['softboiled'], { item: 'deepseascale', nature: 'Calm', evs: { hp: 252, spd: 252 } }),
    T1);

  // ── gen3_item_mechanics_v1 (ABILITY side) — the DMG_MOD class exact-roll probes.
  //    Constructed single-turn hooks set the attacker's PINCH HP (atkHp) and/or a major
  //    status (atkStatus/defStatus) directly on the live battle before the measured hit.
  //    Each member: the boosted case + a control that MUST NOT boost (wrong type / full
  //    HP / no status), so a wrong multiplier/fold/condition diverges the exact damage. ──

  // 49-54. PINCH family (Torrent/Blaze/Overgrow/Swarm): BP ×1.5 for the ability's type at
  //   hp<=maxhp/3. Boosted (atkHp 0.2, on-type) + full-HP control (NOT boosted) + a
  //   wrong-type control (ability on, on-type-move absent → NOT boosted).
  add('torrent_pinch_water',
    mon('Blastoise', ['surf'], { ability: 'Torrent', nature: 'Modest', evs: { spa: 252 } }),
    mon('Snorlax', ['rest'], { nature: 'Calm', evs: { hp: 252, spd: 252 } }),
    T1, { atkHp: 0.2 });
  add('torrent_fullhp_control',
    mon('Blastoise', ['surf'], { ability: 'Torrent', nature: 'Modest', evs: { spa: 252 } }),
    mon('Snorlax', ['rest'], { nature: 'Calm', evs: { hp: 252, spd: 252 } }),
    T1); // full HP → no pinch boost
  add('torrent_wrongtype_pinch_control',
    // Torrent Blastoise using a NON-Water move (Ice Beam) at pinch → NO boost (type gate).
    mon('Blastoise', ['icebeam'], { ability: 'Torrent', nature: 'Modest', evs: { spa: 252 } }),
    mon('Snorlax', ['rest'], { nature: 'Calm', evs: { hp: 252, spd: 252 } }),
    T1, { atkHp: 0.2 });
  add('blaze_pinch_fire',
    mon('Charizard', ['flamethrower'], { ability: 'Blaze', nature: 'Modest', evs: { spa: 252 } }),
    mon('Snorlax', ['rest'], { nature: 'Calm', evs: { hp: 252, spd: 252 } }),
    T1, { atkHp: 0.2 });
  add('overgrow_pinch_grass',
    mon('Sceptile', ['leafblade'], { ability: 'Overgrow', nature: 'Adamant', evs: { atk: 252 } }),
    mon('Snorlax', ['rest'], { nature: 'Impish', evs: { hp: 252, def: 252 } }),
    T1, { atkHp: 0.2 });
  add('swarm_pinch_bug',
    mon('Heracross', ['megahorn'], { ability: 'Swarm', nature: 'Adamant', evs: { atk: 252 } }),
    mon('Snorlax', ['rest'], { nature: 'Impish', evs: { hp: 252, def: 252 } }),
    T1, { atkHp: 0.2 });

  // 55-57. HUGE / PURE POWER: Atk ×2, unconditional — a PHYSICAL boosted case + a SPECIAL
  //   control (ModifyAtk does NOT touch a special move).
  add('huge_power_physical',
    // A FIXED-BP physical (Body Slam BP 85, never happiness-dependent — Return/Frustration
    // have dex BP 0 which the golden's self-contained reconstruction can't recover).
    mon('Azumarill', ['bodyslam'], { ability: 'Huge Power', nature: 'Adamant', evs: { atk: 252 } }),
    mon('Snorlax', ['rest'], { nature: 'Impish', evs: { hp: 252, def: 252 } }),
    T1);
  add('huge_power_special_control',
    // Azumarill's Surf (special) must NOT get the Atk ×2 (it reads SpA, not Atk).
    mon('Azumarill', ['surf'], { ability: 'Huge Power', nature: 'Modest', evs: { spa: 252 } }),
    mon('Snorlax', ['rest'], { nature: 'Calm', evs: { hp: 252, spd: 252 } }),
    T1);
  add('pure_power_physical',
    mon('Medicham', ['bodyslam'], { ability: 'Pure Power', nature: 'Adamant', evs: { atk: 252 } }),
    mon('Snorlax', ['rest'], { nature: 'Impish', evs: { hp: 252, def: 252 } }),
    T1);

  // 58-60. GUTS: Atk ×1.5 when statused AND the burn-halve is SUPPRESSED (the key
  //   interaction). A BURNED Guts physical (×1.5, burn NOT halving) + a paralyzed Guts
  //   (×1.5, no burn interaction) + an UNSTATUSED control (no boost).
  add('guts_burned_physical',
    mon('Machamp', ['bodyslam'], { ability: 'Guts', nature: 'Adamant', evs: { atk: 252 } }),
    mon('Snorlax', ['rest'], { nature: 'Impish', evs: { hp: 252, def: 252 } }),
    T1, { atkStatus: 'brn' });
  add('guts_paralyzed_physical',
    mon('Machamp', ['bodyslam'], { ability: 'Guts', nature: 'Adamant', evs: { atk: 252 } }),
    mon('Snorlax', ['rest'], { nature: 'Impish', evs: { hp: 252, def: 252 } }),
    T1, { atkStatus: 'par' });
  add('guts_unstatused_control',
    mon('Machamp', ['bodyslam'], { ability: 'Guts', nature: 'Adamant', evs: { atk: 252 } }),
    mon('Snorlax', ['rest'], { nature: 'Impish', evs: { hp: 252, def: 252 } }),
    T1); // no status → no Guts boost
  // A BURNED NON-Guts control — the burn HALVES (proves the Guts case's suppression is
  // real, not a missing burn model). (The pre-existing `burn_physical` scenario at #17
  // already pins the plain burn-halve; this pairs it directly with the Guts species.)
  add('nonguts_burned_control',
    mon('Machamp', ['bodyslam'], { ability: 'No Ability', nature: 'Adamant', evs: { atk: 252 } }),
    mon('Snorlax', ['rest'], { nature: 'Impish', evs: { hp: 252, def: 252 } }),
    T1, { atkStatus: 'brn' });

  // 61-62. MARVEL SCALE: Def ×1.5 when the DEFENDER is statused (physical). A burned
  //   Marvel Scale defender (takes ×2/3) + an unstatused control (full damage).
  add('marvel_scale_burned_def',
    mon('Snorlax', ['bodyslam'], { nature: 'Adamant', evs: { atk: 252 } }),
    mon('Milotic', ['recover'], { ability: 'Marvel Scale', nature: 'Bold', evs: { hp: 252, def: 252 } }),
    T1, { defStatus: 'brn' });
  add('marvel_scale_unstatused_control',
    mon('Snorlax', ['bodyslam'], { nature: 'Adamant', evs: { atk: 252 } }),
    mon('Milotic', ['recover'], { ability: 'Marvel Scale', nature: 'Bold', evs: { hp: 252, def: 252 } }),
    T1); // defender not statused → no Def boost

  return S;
}

// Sweep seeds; collect the max realized damage on the measured hit for the desired
// crit polarity (the r==0 max roll == the deterministic baseDamage), the min
// realized (for rolls[15]), and the exact end-state snaps. ONLY a CLEAN hit counts:
// the defender at FULL HP (`hpBefore == maxhp`, no prior chip / heal confound) and,
// for `requireBurn`, the attacker actually burned. Stats/types/boosts are
// seed-independent (the modifiers are deterministic), so we snapshot from the first
// CLEAN hit; only the roll/crit differ across seeds.
//
// Stabilization: we stop once the max-roll value has not grown over the last
// `STABLE_AFTER` CLEAN samples — for a 16-value roll spread that makes a missed
// r==0 astronomically unlikely (non-crit converges in a handful; crit needs the big
// pool to collect enough crit samples first). For a crit scenario we sweep the FULL
// pool (capped by stabilization); for non-crit, a small budget suffices.
const STABLE_AFTER = 60;
async function measure(sc) {
  let maxDmg = -1;
  let minDmg = Infinity;
  let snapP1 = null;
  let snapP2 = null;
  let fainted = false;
  let sawAnyHit = false;
  let cleanSamples = 0;
  let sinceImprove = 0;
  let nonCritSeen = 0;
  const observed = new Set(); // every clean realized damage (for the r==0 verification)

  for (const seed of SEEDS) {
    // Non-crit scenarios don't need the whole pool.
    if (!sc.wantCrit && nonCritSeen >= NONCRIT_SEED_BUDGET && sinceImprove >= STABLE_AFTER) break;
    let res;
    try { res = await runOnce(sc, seed); } catch (e) { continue; }
    const hit = lastAttack(res);
    if (!hit) continue;
    sawAnyHit = true;
    const isImmuneHit = hit.immune;
    if (!isImmuneHit) {
      if (hit.crit !== sc.wantCrit) continue;       // wrong crit polarity
      if (hit.hpBefore !== res.p2.maxhp) continue;  // not a clean full-HP hit
      if (sc.requireBurn && !hit.p1Burned) continue; // burn didn't land this seed
    }
    cleanSamples++;
    nonCritSeen++;
    observed.add(hit.dmg);
    if (hit.dmg > maxDmg) { maxDmg = hit.dmg; sinceImprove = 0; } else { sinceImprove++; }
    if (hit.dmg < minDmg) minDmg = hit.dmg;
    fainted = fainted || hit.fainted;
    if (!snapP1) { snapP1 = res.p1; snapP2 = res.p2; }
    // For an immunity, one clean reading is enough.
    if (isImmuneHit) break;
    // Crit scenarios sweep the FULL pool (crits are sparse — ~1/16 per seed — so we
    // want every crit sample we can get; ~150 samples from the pool makes a missed
    // r==0 ~6e-5). No early stop for crit.
  }
  // r==0 VERIFICATION: if `maxDmg` is the r==0 roll then `maxDmg == baseDamage`, so
  // the r==1 roll `floor(maxDmg*99/100)` must ALSO have been observed (any decent
  // sample sees adjacent rolls). If `maxDmg` were actually r==1 (we missed r==0),
  // `maxDmg+1`'s r==1 roll could equal `maxDmg` and we'd be off by one — this check
  // catches that: a true r==0 has its neighbour present. (Skipped for tiny-damage
  // scenarios where the 85-100% band collapses to <2 distinct values, and for
  // immunities.) `rollVerified` is asserted in `main`.
  const r1 = Math.floor((maxDmg * 99) / 100);
  const bandCollapses = maxDmg - Math.floor((maxDmg * 85) / 100) < 2;
  const rollVerified = maxDmg <= 0 || bandCollapses || r1 === maxDmg || observed.has(r1);
  return {
    maxDmg, minDmg, snapP1, snapP2, fainted, sawAnyHit,
    sawCleanDesired: cleanSamples > 0, cleanSamples, rollVerified,
  };
}

function typesField(types) {
  return (types && types.length ? types : ['???']).join(',');
}
function boostsField(b) {
  // [atk,def,spa,spd,spe]
  return [b.atk || 0, b.def || 0, b.spa || 0, b.spd || 0, b.spe || 0].join(',');
}

async function main() {
  const lines = [];
  lines.push('# damage_golden.txt — Gen-3 single-hit damage golden (EXACT max-roll).');
  lines.push('# DMG <id> <bp> <move_type> <category> <crit> <weather> <reflect> <light_screen> <halves_def> \\');
  lines.push('#     <atk_level> <atk_stat> <spa_stat> <atk_types> <atk_boosts> <atk_status> <atk_ability> <atk_item> \\');
  lines.push('#     <def_stat> <spd_stat> <def_types> <def_boosts> <def_ability> \\');
  lines.push('#     <base_max_roll_dmg> <min_roll_dmg> <atk_species> <def_species> <def_item> \\');
  lines.push('#     <atk_hp> <atk_maxhp> <def_status> <def_ability>');
  lines.push('# All TAB-separated. boosts = atk,def,spa,spd,spe. types = comma list. immunity -> base=0.');
  lines.push('# crit=1 means the measured hit is a critical (x2, ignore-boosts/screens).');
  lines.push('# halves_def=1 marks Explosion/Self-Destruct (gen3 halves the defender Def).');
  lines.push('# atk_species/def_species/def_item (gen3_item_mechanics_v1, appended so pre-existing');
  lines.push('# indices are unchanged): the species-gated item mods + the DEFENDER-side item.');
  lines.push('# atk_hp/atk_maxhp/def_status/def_ability (gen3_item_mechanics_v1 ability side, appended):');
  lines.push('# the pinch HP (test recomputes 3*hp<=maxhp), the defender status+ability (Marvel Scale).');

  const S = scenarios();
  let n = 0;
  const failures = [];
  for (const sc of S) {
    const m = await measure(sc);
    if (!m.sawAnyHit) { failures.push(`${sc.id}: no measured p1->p2 hit found`); continue; }
    // Immunity scenarios: the measured move is absorbed (0x / ability) → base=0.
    const isImmune = m.maxDmg <= 0 && sc.id.includes('immune');
    if (!m.sawCleanDesired && !isImmune) {
      failures.push(`${sc.id}: no CLEAN full-HP hit of the desired polarity (wantCrit=${sc.wantCrit}, requireBurn=${sc.requireBurn}) across ${SEEDS.length} seeds`);
      continue;
    }
    if (m.fainted && !isImmune) {
      failures.push(`${sc.id}: defender FAINTED (damage KO-capped, not exact) — pick a bulkier defender`);
      continue;
    }
    if (!isImmune && !m.rollVerified) {
      failures.push(`${sc.id}: max-roll NOT verified as r==0 (neighbour roll absent; max=${m.maxDmg}, ${m.cleanSamples} clean samples) — widen the seed pool`);
      continue;
    }

    let base = 0;
    let minRoll = 0;
    let snapP1 = m.snapP1;
    let snapP2 = m.snapP2;
    if (!isImmune) {
      base = m.maxDmg;
      minRoll = m.minDmg;
    } else {
      // For an immunity we still need the stat/type context — take any snap.
      // Re-run one seed to get the end snaps (the hit may never have registered).
      const res = await runOnce(sc, SEEDS[0]);
      snapP1 = res.p1; snapP2 = res.p2;
    }

    const p1 = snapP1;
    const p2 = snapP2;
    // The actually-measured move id is whichever p1 used last; recover from a snap-
    // independent source: the LAST choice for p1 (T1: move 1; staged: move 2).
    const p1Choices = sc.choices.filter((c) => c[0] === 'p1');
    const lastP1Choice = p1Choices[p1Choices.length - 1][1]; // e.g. "move 2"
    const slot = parseInt(lastP1Choice.split(' ')[1], 10) - 1;
    const measuredMoveId = moveId(sc.p1.moves[slot]);

    // Look the move up in the real dex to emit BP/type/category (the Rust test also
    // looks it up, but we emit them so the golden is self-contained + auditable).
    const Dex = require(path.join(PS, 'dist/sim')).Dex;
    const dex3 = Dex.forGen(3);
    const md = dex3.moves.get(measuredMoveId);
    const bp = md.basePower;
    const mtype = md.type; // "Electric" etc.
    const category = md.category; // "Physical"/"Special"/"Status"

    const crit = sc.wantCrit ? 1 : 0;
    // Weather: read the LIVE field weather from the snap's run (re-run cheaply for a
    // definitive value rather than infer from the id).
    const wxRes = await runOnce(sc, SEEDS[0]);
    const wx = (wxRes.weather || '').toLowerCase();
    const weather = wx.includes('rain') ? 'rain' : (wx.includes('sun') ? 'sun' : 'none');
    const reflect = (p2.sideConditions || []).includes('reflect') ? 1 : 0;
    const lightScreen = (p2.sideConditions || []).includes('lightscreen') ? 1 : 0;
    // gen3 Explosion/Self-Destruct halve the defender's Def in the formula.
    const halvesDef = (measuredMoveId === 'explosion' || measuredMoveId === 'selfdestruct') ? 1 : 0;

    const fields = [
      'DMG', sc.id,
      bp, mtype, category, crit, weather, reflect, lightScreen, halvesDef,
      sc.p1.level,
      p1.stats.atk, p1.stats.spa,
      typesField(p1.types), boostsField(p1.boosts), (p1.status || 'none'),
      (p1.ability || 'none'), (sc.p1.item || 'none'),
      p2.stats.def, p2.stats.spd,
      typesField(p2.types), boostsField(p2.boosts), (p2.ability || 'none'),
      base, minRoll,
      // gen3_item_mechanics_v1 (appended): species for the species-gated item mods
      // + the DEFENDER-side item (DeepSeaScale / Metal Powder / Soul Dew SpD).
      moveId(p1.species), moveId(p2.species), (sc.p2.item || 'none'),
      // gen3_item_mechanics_v1 ABILITY side (appended so the pre-existing 28 indices are
      // unchanged): the attacker's HP + maxhp AT the measured hit (so the test recomputes
      // the pinch `3*hp<=maxhp`), the DEFENDER's status (Marvel Scale reads it — atk_status
      // is already emitted at f[15], covering Guts), and the DEFENDER's ability id (Marvel
      // Scale is a def-side ability the fold reads).
      p1.hp, p1.maxhp, (p2.status || 'none'), (p2.ability || 'none'),
    ];
    lines.push(fields.join('\t'));
    n++;
  }

  if (failures.length) {
    console.error('DAMAGE GOLDEN FAILURES:\n  ' + failures.join('\n  '));
    process.exit(1);
  }

  fs.writeFileSync(OUT, lines.join('\n') + '\n');
  console.error(`damage golden: ${n} scenarios -> ${OUT}`);
  process.exit(0);
}

main().catch((e) => { console.error(e && e.stack ? e.stack : String(e)); process.exit(1); });
