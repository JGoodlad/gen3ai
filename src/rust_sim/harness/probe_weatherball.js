// PROBE: gen-3 WEATHER BALL — runtime TYPE + BASE POWER + CATEGORY per weather,
//        the draw model, Air Lock / Cloud Nine suppression, and the read-time edge.
//
// Method: omniscient BattleStream (no server), fixed seed, gen3customgame.
//   * PRNG.prototype.random is wrapped (the SOLE path to rng.next()) -> the draw log.
//   * BattleActions.prototype.getDamage is wrapped at ENTRY -> the resolved activeMove's
//     type / basePower / category, plus source.hasType(type) (STAB) and the type-chart
//     effectiveness, plus the returned damage.
//   * chunks starting 'sideupdate' are DROPPED -> the chronological omniscient log.
//
// The measurement board is a MEW MIRROR on purpose: base 100 across the board means
// Atk == SpA on the attacker and Def == SpD on the defender, and Psychic is NEUTRAL to
// all five candidate types with NO STAB — so realized damage is a PURE function of base
// power and the physical/special flip cannot confound it.
const path = require('path');
const PS = '/home/goodlad/dev/gen3ai/deps/pokemon-showdown';
const { BattleStream } = require(path.join(PS, 'dist/sim/battle-stream.js'));
const { Dex } = require(path.join(PS, 'dist/sim/dex.js'));
const { PRNG } = require(path.join(PS, 'dist/sim/prng'));
const { BattleActions } = require(path.join(PS, 'dist/sim/battle-actions.js'));

// ---- dex facts (resolved gen3, the whole mod chain applied) ----
const d3 = Dex.mod('gen3').moves.get('weatherball');
console.log('== RESOLVED gen3 dex row');
console.log('   num', d3.num, 'accuracy', d3.accuracy, 'basePower', d3.basePower,
  'category', d3.category, 'type', d3.type, 'target', d3.target,
  'priority', d3.priority, 'flags', JSON.stringify(d3.flags));
console.log('   onModifyType?', !!d3.onModifyType, 'onModifyMove?', !!d3.onModifyMove,
  'secondaries', JSON.stringify(d3.secondaries || d3.secondary || null));

// ---- instrumentation ----
let draws = [];
const origRandom = PRNG.prototype.random;
PRNG.prototype.random = function (...a) {
  const r = origRandom.apply(this, a);
  draws.push(`random(${a})->${r}`);
  return r;
};

let dmgCalls = [];
const origGetDamage = BattleActions.prototype.getDamage;
BattleActions.prototype.getDamage = function (source, target, move, suppress) {
  let rec = null;
  if (move && typeof move === 'object' && move.id) {
    let eff = null, imm = null, stab = null;
    try {
      eff = this.battle.dex.getEffectiveness(move.type, target);
      imm = !this.battle.dex.getImmunity(move.type, target);
      stab = source.hasType(move.type);
    } catch (e) { /* best effort */ }
    rec = {
      id: move.id, type: move.type, bp: move.basePower, cat: move.category,
      stab, eff, imm,
      atk: source.getStat(move.category === 'Physical' ? 'atk' : 'spa'),
      def: target.getStat(move.category === 'Physical' ? 'def' : 'spd'),
      weather: this.battle.field.effectiveWeather(), rawWeather: this.battle.field.weather,
    };
  }
  const r = origGetDamage.apply(this, arguments);
  if (rec) { rec.damage = r; dmgCalls.push(rec); }
  return r;
};

const LINE_RE = /^\|(move|-damage|-crit|-supereffective|-resisted|-immune|-miss|-fail|-activate|-start|-weather|-formechange|-heal|turn|switch|faint|-enditem)\|/;

async function run(label, p1, p2, script) {
  draws = []; dmgCalls = [];
  const s = new BattleStream();
  const ch = [];
  (async () => { for await (const c of s) ch.push(c); })();
  s.write(`>start {"formatid":"gen3customgame","seed":[9,9,9,9]}\n>player p1 {"name":"P1","team":"${p1}"}\n>player p2 {"name":"P2","team":"${p2}"}`);
  await new Promise(r => setTimeout(r, 220));
  const marks = [];
  for (const c of script) {
    const before = draws.length;
    s.write(c);
    await new Promise(r => setTimeout(r, 220));
    marks.push({ cmd: c.replace(/\n/g, ' ; '), drew: draws.slice(before) });
  }
  await new Promise(r => setTimeout(r, 220));
  const omni = ch.filter(c => !c.startsWith('sideupdate')).join('\n').split('\n').filter(l => LINE_RE.test(l));
  console.log(`\n===== ${label}`);
  console.log('  LOG:\n    ' + omni.join('\n    '));
  for (const m of marks) console.log(`  DRAWS [${m.cmd}] (${m.drew.length}): ${m.drew.join('  ') || '(none)'}`);
  for (const c of dmgCalls) {
    console.log(`  getDamage(${c.id}): type=${c.type} bp=${c.bp} cat=${c.cat} stab=${c.stab} eff=${c.eff} imm=${c.imm} atk=${c.atk} def=${c.def} effWeather=${c.weather} rawWeather=${c.rawWeather} -> damage=${c.damage}`);
  }
  return { omni, dmgCalls: dmgCalls.slice(), marks };
}

// ---- teams ----
// The measurement attacker: Mew, every base stat 100, Psychic (no STAB on any candidate
// type), Synchronize (inert here), max Spe EVs so it always moves first.
const MEW_USER = 'Mew||Leftovers|Synchronize|weatherball,raindance,sunnyday,sandstorm,hail,surf,tackle,splash|Hardy|0,0,0,0,0,252|||||';
// The measurement target: Mew, 0 EVs -> Def == SpD == 236, slower, only Splash.
const MEW_TARGET = 'Mew||Leftovers|Synchronize|splash|Hardy|0,0,0,0,0,0|||||';

const SPLASH2 = '>p2 move 1';
const WB = '>p1 move 1', RAIN = '>p1 move 2', SUN = '>p1 move 3', SAND = '>p1 move 4',
  HAIL = '>p1 move 5', SURF = '>p1 move 6', TACKLE = '>p1 move 7', SPLASH1 = '>p1 move 8';

(async () => {
  // ---------- Q1: TYPE + BP + CATEGORY per weather (same attacker, same target) ----------
  await run('Q1-a NO WEATHER  (weatherball, then the BP-50 control Tackle)',
    MEW_USER, MEW_TARGET, [`${WB}\n${SPLASH2}`, `${TACKLE}\n${SPLASH2}`]);

  await run('Q1-b RAIN DANCE  (set turn 1, weatherball turn 2)',
    MEW_USER, MEW_TARGET, [`${RAIN}\n${SPLASH2}`, `${WB}\n${SPLASH2}`]);

  await run('Q1-c SUNNY DAY',
    MEW_USER, MEW_TARGET, [`${SUN}\n${SPLASH2}`, `${WB}\n${SPLASH2}`]);

  await run('Q1-d SANDSTORM',
    MEW_USER, MEW_TARGET, [`${SAND}\n${SPLASH2}`, `${WB}\n${SPLASH2}`]);

  await run('Q1-e HAIL',
    MEW_USER, MEW_TARGET, [`${HAIL}\n${SPLASH2}`, `${WB}\n${SPLASH2}`]);

  // ---------- Q2: STAB + type chart ----------
  // Water-type user in rain -> does the RUNTIME type earn STAB?
  await run('Q2-a STAB: VAPOREON (Water) weatherball in rain vs Mew',
    'Vaporeon||Leftovers|WaterAbsorb|weatherball,raindance,splash|Hardy|0,0,0,0,0,252|M||||',
    MEW_TARGET, [`>p1 move 2\n${SPLASH2}`, `>p1 move 1\n${SPLASH2}`]);
  // Water-type TARGET under rain -> Water resists Water.
  await run('Q2-b RESIST: rain weatherball (Water) into VAPOREON (Water resist)',
    MEW_USER,
    'Vaporeon||Leftovers|WaterAbsorb|splash|Hardy|0,0,0,0,0,0|M||||',
    [`${RAIN}\n${SPLASH2}`, `${WB}\n${SPLASH2}`]);
  // No weather -> Normal -> a GHOST is IMMUNE.
  await run('Q2-c IMMUNE: no-weather weatherball (Normal) into GENGAR (Ghost)',
    MEW_USER,
    'Gengar||Leftovers|Levitate|splash|Hardy|0,0,0,0,0,0|M||||',
    [`${WB}\n${SPLASH2}`]);
  // Hail -> Ice -> 4x into Dragon/Flying.
  await run('Q2-d 4x: hail weatherball (Ice) into a Dragon/Flying',
    MEW_USER,
    'Salamence||Leftovers|Intimidate|splash|Hardy|0,0,0,0,0,0|M||||',
    [`${HAIL}\n${SPLASH2}`, `${WB}\n${SPLASH2}`]);
  // Sun -> Fire -> FLASH FIRE absorbs it (an ability immunity on the RUNTIME type).
  await run('Q2-e FLASH FIRE: sun weatherball (Fire) into HOUNDOOM',
    MEW_USER,
    'Houndoom||Leftovers|FlashFire|splash|Hardy|0,0,0,0,0,0|M||||',
    [`${SUN}\n${SPLASH2}`, `${WB}\n${SPLASH2}`]);

  // ---------- Q3: the draw model (control = Surf, acc 100, special, no secondary) ----------
  await run('Q3-a DRAW CONTROL: SURF (bp 95, acc 100, no secondary) in rain',
    MEW_USER, MEW_TARGET, [`${RAIN}\n${SPLASH2}`, `${SURF}\n${SPLASH2}`]);
  // (Q1-b's turn-2 draws are the weatherball arm of the same comparison.)

  // ---------- Q4: AIR LOCK / CLOUD NINE ----------
  await run('Q4-a AIR LOCK: rain up, weatherball into RAYQUAZA (Air Lock)',
    MEW_USER,
    'Rayquaza||Leftovers|AirLock|splash|Hardy|0,0,0,0,0,0|||||',
    [`${RAIN}\n${SPLASH2}`, `${WB}\n${SPLASH2}`]);
  await run('Q4-b CLOUD NINE: rain up, weatherball into GOLDUCK (Cloud Nine)',
    MEW_USER,
    'Golduck||Leftovers|CloudNine|splash|Hardy|0,0,0,0,0,0|M||||',
    [`${RAIN}\n${SPLASH2}`, `${WB}\n${SPLASH2}`]);
  // Control: the SAME Rayquaza target but a normal ability, to isolate the suppression.
  await run('Q4-c CONTROL: rain up, weatherball into RAYQUAZA with a plain ability',
    MEW_USER,
    'Rayquaza||Leftovers|Levitate|splash|Hardy|0,0,0,0,0,0|||||',
    [`${RAIN}\n${SPLASH2}`, `${WB}\n${SPLASH2}`]);

  // ---------- Q5: WHEN is the weather read? ----------
  // (a) a FASTER foe sets the weather on the SAME turn the slower user Weather Balls.
  await run('Q5-a READ TIME: fast ELECTRODE sets Sunny Day, slow Mew weatherballs same turn',
    'Mew||Leftovers|Synchronize|weatherball,splash|Hardy|0,0,0,0,0,0|||||',
    'Electrode||Leftovers|Static|sunnyday,splash|Hardy|0,0,0,0,0,252|||||',
    ['>p1 move 1\n>p2 move 1']);
  // (b) the reverse: SLOWER foe sets it AFTER the user's weatherball resolves.
  await run('Q5-b READ TIME: fast Mew weatherballs, SLOW foe sets Sunny Day after',
    'Mew||Leftovers|Synchronize|weatherball,splash|Hardy|0,0,0,0,0,252|||||',
    'Snorlax||Leftovers|Immunity|sunnyday,splash|Hardy|0,0,0,0,0,0|M||||',
    ['>p1 move 1\n>p2 move 1']);
  // (c) the 5-turn timer EXPIRY: rain turn 1, weatherball on the LAST rain turn and again after.
  await run('Q5-c EXPIRY: rain turn 1, weatherball on turns 2..7 (spans the 5-turn expiry)',
    MEW_USER,
    'Blissey||Leftovers|NaturalCure|splash|Hardy|252,0,0,0,0,0|F||||',
    [`${RAIN}\n${SPLASH2}`, `${WB}\n${SPLASH2}`, `${WB}\n${SPLASH2}`, `${WB}\n${SPLASH2}`,
     `${WB}\n${SPLASH2}`, `${WB}\n${SPLASH2}`, `${WB}\n${SPLASH2}`]);
  // (d) an AIR LOCK holder SWITCHES IN mid-turn, then the slower user's weatherball.
  await run('Q5-d SUPPRESSOR SWITCH-IN: rain up, foe switches to Rayquaza (Air Lock) the same turn',
    'Mew||Leftovers|Synchronize|weatherball,raindance,splash|Hardy|0,0,0,0,0,0|||||',
    'Snorlax||Leftovers|Immunity|splash|Hardy|0,0,0,0,0,0|M||||]Rayquaza||Leftovers|AirLock|splash|Hardy|0,0,0,0,0,252|||||',
    ['>p1 move 2\n>p2 move 1', '>p1 move 1\n>p2 switch 2']);
  // (e) FROZEN defender: does a sun (Fire) weatherball thaw it? gen3 frz.onDamagingHit
  //     keys on the BASE-dex type, and weatherball's base type is Normal.
  await run('Q5-e FREEZE THAW: sun weatherball (runtime Fire) into a FROZEN target',
    'Mew||Leftovers|Synchronize|weatherball,sunnyday,icebeam,splash|Hardy|0,0,0,0,0,252|||||',
    'Snorlax||Leftovers|Immunity|splash|Hardy|0,0,0,0,0,0|M||||',
    ['>p1 move 2\n>p2 move 1', '>p1 move 3\n>p2 move 1', '>p1 move 3\n>p2 move 1',
     '>p1 move 3\n>p2 move 1', '>p1 move 1\n>p2 move 1']);
})();
