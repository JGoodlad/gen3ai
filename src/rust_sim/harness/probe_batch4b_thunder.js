// probe_batch4b_thunder.js — ground-truth THUNDER (id `thunder`) bit-for-bit vs the
// OMNISCIENT in-process BattleStream (no server). Thunder is the WEATHER-ACCURACY move:
//   - base acc 70, 120-BP Special Electric, 30% paralysis secondary
//   - onModifyMove (gen3, RESOLVED) mutates move.accuracy by the DEFENDER's effectiveWeather():
//       rain (raindance/primordialsea) -> accuracy = true (never-miss)
//       sun  (sunnyday/desolateland)   -> accuracy = 50
//       else (none/sand/hail)          -> unchanged (70)
//
// THE CRUX (draw-count): gen3 tryMoveHit computes
//     let accuracy = move.accuracy;           // <- already mutated by onModifyMove
//     if (accuracy !== true) { ...acc/eva stages + runEvent('ModifyAccuracy')... }
//     if (accuracy !== true && !randomChance(accuracy,100)) accPass = false;
//   so accuracy === true (rain) SKIPS the WHOLE acc pipeline AND the randomChance ->
//   ZERO accuracy draw (one fewer draw than base). sun (50) STILL draws (base swapped 70->50).
//
// This probe COUNTS the per-turn PRNG draws under each weather + confirms:
//   1. rain => Thunder draws ZERO accuracy random() (never-miss, draw-count -1 vs base)
//   2. sun  => Thunder draws accuracy at effAcc 50 (Bright Powder folds AFTER -> 45)
//   3. none/sand/hail => base 70 draw
//   4. Cloud Nine / Air Lock (effectiveWeather suppressed) => base 70 even under rain
//   5. the 30% paralysis secondary draws its own random(100), unchanged by weather
//
// Run:  node src/rust_sim/harness/probe_batch4b_thunder.js
'use strict';
const path = require('path');
const PS = path.resolve(__dirname, '../../../deps/pokemon-showdown');
const { BattleStream, getPlayerStreams } = require(path.join(PS, 'dist/sim/battle-stream'));
const { Teams, Dex } = require(path.join(PS, 'dist/sim'));

const FORMAT = 'gen3customgame';
const IV31 = { hp: 31, atk: 31, def: 31, spa: 31, spd: 31, spe: 31 };
const EV0 = { hp: 0, atk: 0, def: 0, spa: 0, spd: 0, spe: 0 };
function mon(species, moves, opts = {}) {
  return {
    species, item: opts.item || '', ability: opts.ability || 'No Ability', moves,
    evs: { ...EV0, ...(opts.evs || {}) }, ivs: opts.ivs || IV31,
    nature: opts.nature || 'Serious', level: opts.level || 100, gender: opts.gender || 'N',
  };
}
function tick() { return new Promise((r) => setTimeout(r, 0)); }

function dumpResolved() {
  const d = Dex.forFormat(FORMAT);
  const m = d.moves.get('thunder');
  console.log('=== resolved gen3 thunder ===');
  console.log(`  cat=${m.category} bp=${m.basePower} acc=${m.accuracy} type=${m.type} ` +
    `never_miss=${m.accuracy === true} secondary=${JSON.stringify(m.secondary)} flags=${JSON.stringify(m.flags)}`);
  console.log(`  onModifyMove src: ${m.onModifyMove.toString().replace(/\s+/g, ' ')}`);
}

// Instrument the PRNG with call-site tags so we can see WHICH draws fire.
function instrument(battle) {
  const rng = battle.prng.rng;
  const realNext = rng.next.bind(rng);
  const draws = [];
  rng.next = function (...a) {
    const st = new Error().stack.split('\n').slice(2, 8)
      .map((l) => l.trim().replace(/^at\s+/, ''))
      .filter((l) => /battle|actions|pokemon|field|dex/.test(l))
      .map((l) => l.replace(/\s*\(.*$/, '').replace(/^Battle(Actions)?\./, ''))
      .slice(0, 2).join(' < ');
    const v = realNext(...a);
    draws.push(st);
    return v;
  };
  return draws;
}

async function run(label, p1team, p2team, plan, inject) {
  const stream = new BattleStream();
  const streams = getPlayerStreams(stream);
  const log = [];
  (async () => { for await (const ch of streams.omniscient) { for (const l of ch.split('\n')) if (l) log.push(l); } })();
  const seed = (inject && inject.seed) || [7, 11, 13, 17];
  streams.omniscient.write(`>start {"formatid":"${FORMAT}","seed":${JSON.stringify(seed)}}`);
  streams.omniscient.write(`>player p1 ${JSON.stringify({ name: 'P1', team: Teams.pack(p1team) })}`);
  streams.omniscient.write(`>player p2 ${JSON.stringify({ name: 'P2', team: Teams.pack(p2team) })}`);
  for (let i = 0; i < 12; i++) await tick();

  const battle = stream.battle;
  for (const inj of ((inject && inject.acts) || [])) {
    const m = inj.side === undefined ? null : battle.sides[inj.side].active[0];
    if (!m) continue;
    if (inj.status) m.setStatus(inj.status, m, null, true);
    if (inj.hp !== undefined) m.hp = inj.hp;
    if (inj.item !== undefined) m.item = inj.item;
  }

  const draws = instrument(battle);
  console.log(`\n=== ${label} ===  initSeed=${battle.prng.getSeed()}  weather=${battle.field.weather || '-'}`);
  let i = 0, safety = 0;
  while (!battle.ended && safety < 8) {
    safety++;
    const rs = battle.requestState;
    if (rs !== 'move' && rs !== 'switch') { await tick(); continue; }
    const dc0 = draws.length;
    const logLen0 = log.length;
    const before = battle.prng.getSeed();
    const entry = plan[Math.min(i, plan.length - 1)]; i++;
    if (entry.p1) streams.omniscient.write(`>p1 ${entry.p1}`);
    if (entry.p2) streams.omniscient.write(`>p2 ${entry.p2}`);
    for (let k = 0; k < 18; k++) await tick();
    const after = battle.prng.getSeed();
    const a0 = battle.sides[0].active[0], a1 = battle.sides[1].active[0];
    const fmt = (m) => m ? `${m.species.name} ${m.hp}/${m.maxhp} ${m.status || '-'}` : '-';
    const myDraws = draws.slice(dc0);
    console.log(`  [${rs}] ${JSON.stringify(entry)} weather=${battle.field.weather || '-'} draws=${myDraws.length}  seed ${before}->${after}`);
    console.log(`        p1=${fmt(a0)}  p2=${fmt(a1)}`);
    myDraws.forEach((dl, k) => console.log(`        DRAW[${k}] ${dl}`));
    const newLines = log.slice(logLen0).filter((l) =>
      /\|move\||-damage|-status|-miss|-immune|-crit|-supereffective|-weather|-activate|cant/.test(l));
    for (const l of newLines) console.log(`        LINE ${l}`);
    if (entry.stop) break;
  }
  try { streams.omniscient.destroy(); } catch (e) {}
}

async function main() {
  dumpResolved();

  // p1 Zapdos (fast) Thunders; p2 Blissey (slow) Soft-Boiled (heal, never-miss, DRAW-FREE).
  // The p2 ability SETS the permanent weather (customgame lets us pick any ability), so the
  // weather is up from turn 1 with no Rain-Dance turn confounding the per-turn draw count.
  // A draw-free p2 move + distinct speeds => the ONLY draws are Thunder's + the end-of-turn
  // Quick Claw random(1,5). So: no-weather Thunder-lands = acc + crit + dmg + secondary + QC.
  const thund = (ability) => [mon('Zapdos', ['thunder'], { ability: 'Pressure', evs: { spa: 252, spe: 252 } })];
  const blissey = (ability, item) => [mon('Blissey', ['softboiled'], { ability, item: item || '', evs: { hp: 252 } })];

  // 1) NO WEATHER — base acc 70. Thunder draws acc + crit + dmg + secondary + QC.
  await run('NO WEATHER: base acc 70 (acc draw present)',
    thund(), blissey('No Ability'),
    [{ p1: 'move 1', p2: 'move 1', stop: true }]);

  // 2) RAIN (Blissey Drizzle) — never-miss. THE CRUX: NO accuracy draw (one fewer than base).
  await run('RAIN (Drizzle): never-miss -> ZERO accuracy draw (draw-count -1)',
    thund(), blissey('Drizzle'),
    [{ p1: 'move 1', p2: 'move 1', stop: true }]);

  // 3) SUN (Blissey Drought) — accuracy = 50. Draw PRESENT at effAcc 50 (base swapped 70->50).
  await run('SUN (Drought): effAcc 50 (acc draw present, lower)',
    thund(), blissey('Drought'),
    [{ p1: 'move 1', p2: 'move 1', stop: true }]);

  // 4) SAND (Blissey Sand Stream) — weather does NOT touch Thunder. Base 70 draw.
  await run('SAND (Sand Stream): base acc 70 (acc draw present)',
    thund(), blissey('Sand Stream'),
    [{ p1: 'move 1', p2: 'move 1', stop: true }]);

  // 5) HAIL (Blissey Snow Warning? gen3 has none; set via move? use Air Lock instead below).
  //    gen3 permanent hail ability doesn't exist; skip hail as a weather-set-ability probe
  //    (SAND already proves "non-rain/sun weather = base 70").

  // 6) CLOUD NINE suppresses the weather -> effectiveWeather() returns '' -> base 70 EVEN in rain.
  //    Zapdos carries Cloud Nine while Blissey Drizzle sets rain; Thunder sees NO effective rain.
  await run('CLOUD NINE under Drizzle: effectiveWeather suppressed -> base 70 (acc draw present)',
    [mon('Zapdos', ['thunder'], { ability: 'Cloud Nine', evs: { spa: 252, spe: 252 } })],
    blissey('Drizzle'),
    [{ p1: 'move 1', p2: 'move 1', stop: true }]);

  // 6b) AIR LOCK under Drizzle — same suppression, the other negater.
  await run('AIR LOCK under Drizzle: effectiveWeather suppressed -> base 70',
    [mon('Zapdos', ['thunder'], { ability: 'Air Lock', evs: { spa: 252, spe: 252 } })],
    blissey('Drizzle'),
    [{ p1: 'move 1', p2: 'move 1', stop: true }]);

  // 7) BRIGHT POWDER in SUN — the accMod chain folds AFTER the weather base: 50 * 0.9 = 45.
  //    Confirms the weather mutation is the BASE that the accMod pipeline reads (weather first).
  await run('SUN + Bright Powder: effAcc = 50 * 0.9 = 45 (weather base, THEN accMod)',
    thund(), blissey('Drought', 'brightpowder'),
    [{ p1: 'move 1', p2: 'move 1', stop: true }]);

  // 8) BRIGHT POWDER in RAIN — never-miss WINS over Bright Powder (accuracy===true short-circuits
  //    the whole ModifyAccuracy chain). NO accuracy draw; the item is inert.
  await run('RAIN + Bright Powder: never-miss beats Bright Powder -> STILL zero accuracy draw',
    thund(), blissey('Drizzle', 'brightpowder'),
    [{ p1: 'move 1', p2: 'move 1', stop: true }]);

  // 9) The 30% PARALYSIS secondary — unchanged by weather. In rain (never-miss) the secondary
  //    random(100) STILL draws AFTER the (skipped-acc) hit. Sweep a few seeds to see it land.
  await run('RAIN para-secondary: hit is never-miss, secondary random(100) still draws',
    thund(), blissey('Drizzle'),
    [{ p1: 'move 1', p2: 'move 1' }, { p1: 'move 1', p2: 'move 1', stop: true }],
    { seed: [1, 2, 3, 99] });
}
main().catch((e) => { console.error(e.stack || String(e)); process.exit(1); });
