// probe_substitute_status.js — exact draw args for STATUS moves + a damaging move
// (Rock Slide flinch / Body Slam par) into a SUB, plus how the sub-block reports.
'use strict';
const path = require('path');
const PS = path.resolve(__dirname, '../../../deps/pokemon-showdown');
const { BattleStream, getPlayerStreams } = require(path.join(PS, 'dist/sim/battle-stream'));
const { Teams } = require(path.join(PS, 'dist/sim'));

const FORMAT = 'gen3customgame';
const IV31 = { hp: 31, atk: 31, def: 31, spa: 31, spd: 31, spe: 31 };
const EV0 = { hp: 0, atk: 0, def: 0, spa: 0, spd: 0, spe: 0 };
function mon(species, moves, opts = {}) {
  return {
    species, item: opts.item || '', ability: opts.ability || 'No Ability', moves,
    evs: { ...EV0, ...(opts.evs || {}) }, ivs: opts.ivs || IV31,
    nature: opts.nature || 'Serious', level: opts.level || 100, gender: 'N',
  };
}
function tick() { return new Promise((r) => setTimeout(r, 0)); }

async function run(label, p1team, p2team, plan, seed) {
  const stream = new BattleStream();
  const streams = getPlayerStreams(stream);
  const protoLog = [];
  (async () => { for await (const ch of streams.omniscient) { protoLog.push(ch); } })();
  streams.omniscient.write(`>start {"formatid":"${FORMAT}","seed":${JSON.stringify(seed)}}`);
  streams.omniscient.write(`>player p1 ${JSON.stringify({ name: 'P1', team: Teams.pack(p1team) })}`);
  streams.omniscient.write(`>player p2 ${JSON.stringify({ name: 'P2', team: Teams.pack(p2team) })}`);
  for (let i = 0; i < 12; i++) await tick();
  const battle = stream.battle;

  let log = [];
  const realRandom = battle.random.bind(battle);
  battle.random = function (from, to) { const v = realRandom(from, to); log.push(`random(${from},${to})=${v}`); return v; };
  const realRC = battle.randomChance.bind(battle);
  battle.randomChance = function (num, den) { const v = realRC(num, den); log.push(`randomChance(${num},${den})=${v}`); return v; };

  console.log(`\n=== ${label} (seed ${seed}) ===`);
  let i = 0, safety = 0;
  while (!battle.ended && safety < 50) {
    safety++;
    const rs = battle.requestState;
    if (rs !== 'move' && rs !== 'switch') { await tick(); continue; }
    if (i >= plan.length) break;
    const entry = plan[i]; i++;
    log = [];
    protoLog.length = 0;
    if (entry.p1) streams.omniscient.write(`>p1 ${entry.p1}`);
    if (entry.p2) streams.omniscient.write(`>p2 ${entry.p2}`);
    for (let k = 0; k < 18; k++) await tick();
    const a0 = battle.sides[0].active[0], a1 = battle.sides[1].active[0];
    const subOf = (m) => (m && m.volatiles && m.volatiles['substitute']) ? `SUB(${m.volatiles['substitute'].hp})` : '';
    const fmt = (m) => m ? `${m.species.name} ${m.hp}/${m.maxhp} ${m.status || '-'} b=[${m.boosts.atk},${m.boosts.spd}] ${subOf(m)}` : '-';
    console.log(`  ${JSON.stringify(entry)} -> p1=${fmt(a0)} | p2=${fmt(a1)}`);
    console.log(`      draws: ${log.join('  ')}`);
    const activates = protoLog.join('').split('\n').filter((l) => /Substitute|-activate|-miss|-immune|-status|-fail/.test(l));
    if (activates.length) console.log(`      proto: ${activates.join(' | ')}`);
  }
  try { streams.omniscient.destroy(); } catch (e) {}
}

async function main() {
  // STATUS move (Thunder Wave) into a sub: exact draws (acc only? + QC).
  await run('Thunder Wave into SUBBED Blissey',
    [mon('Jolteon', ['thunderwave', 'splash'], { evs: { spe: 252 } })],
    [mon('Blissey', ['substitute', 'softboiled'], { ability: 'Natural Cure', evs: { hp: 252 } })],
    [
      { p1: 'move 2', p2: 'move 1' }, // Jolteon Splash, Blissey Substitute
      { p1: 'move 1', p2: 'move 2' }, // Jolteon Thunder Wave INTO sub ; Blissey Soft-Boiled
    ], [7, 11, 13, 17]);

  // Toxic into a sub (never-miss status — does it draw at all?).
  await run('Toxic into SUBBED Blissey',
    [mon('Gengar', ['toxic', 'splash'], { evs: { spe: 252 } })],
    [mon('Blissey', ['substitute', 'softboiled'], { ability: 'Natural Cure', evs: { hp: 252 } })],
    [
      { p1: 'move 2', p2: 'move 1' }, // Gengar Splash, Blissey Substitute
      { p1: 'move 1', p2: 'move 2' }, // Gengar Toxic INTO sub ; Blissey Soft-Boiled
    ], [7, 11, 13, 17]);

  // Rock Slide (flinch 30) into a sub: secondary random(100) drawn? flinch applied?
  await run('Rock Slide flinch into SUBBED Blissey (slower foe)',
    [mon('Aerodactyl', ['rockslide', 'splash'], { evs: { spe: 252, atk: 252 } })],
    [mon('Blissey', ['substitute', 'softboiled'], { ability: 'Natural Cure', evs: { hp: 252 } })],
    [
      { p1: 'move 2', p2: 'move 1' }, // Aero Splash, Blissey Substitute
      { p1: 'move 1', p2: 'move 2' }, // Aero Rock Slide INTO sub (faster) ; Blissey Soft-Boiled
    ], [7, 11, 13, 17]);
}
main().catch((e) => { console.error(e.stack || String(e)); process.exit(1); });
