// probe_substitute_confusion.js — does a CONFUSION self-hit hit the SUB or the MON
// (and what does the secondary confusion / flinch draw vs a sub)?
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
  (async () => { for await (const ch of streams.omniscient) { void ch; } })();
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
    if (entry.injectBefore) {
      for (const inj of entry.injectBefore) {
        const m = battle.sides[inj.side].active[0];
        if (inj.hp !== undefined) m.hp = inj.hp;
        if (inj.confusion) m.addVolatile('confusion');
      }
    }
    log = [];
    if (entry.p1) streams.omniscient.write(`>p1 ${entry.p1}`);
    if (entry.p2) streams.omniscient.write(`>p2 ${entry.p2}`);
    for (let k = 0; k < 18; k++) await tick();
    const a0 = battle.sides[0].active[0], a1 = battle.sides[1].active[0];
    const subOf = (m) => (m && m.volatiles && m.volatiles['substitute']) ? `SUB(${m.volatiles['substitute'].hp})` : '';
    const confOf = (m) => (m && m.volatiles && m.volatiles['confusion']) ? `CONF(${m.volatiles['confusion'].time})` : '';
    const fmt = (m) => m ? `${m.species.name} ${m.hp}/${m.maxhp} ${m.status || '-'} ${subOf(m)}${confOf(m)}` : '-';
    console.log(`  ${JSON.stringify(entry)} -> p1=${fmt(a0)} | p2=${fmt(a1)}`);
    console.log(`      draws: ${log.join('  ')}`);
  }
  try { streams.omniscient.destroy(); } catch (e) {}
}

async function main() {
  // Sub up, confuse, then keep choosing Splash and watch whether a confusion self-hit
  // hits the SUB or the MON. Sweep seeds to force a self-hit. Snorlax maxhp 524, sub 131.
  for (const seed of [[1, 2, 3, 4], [9, 9, 9, 9], [5, 4, 3, 2], [42, 17, 8, 3]]) {
    await run('CONFUSION self-hit: SUB or MON?',
      [mon('Snorlax', ['substitute', 'splash'], { evs: { hp: 252, atk: 252 } })],
      [mon('Blissey', ['splash'], { ability: 'Natural Cure', evs: { hp: 252 } })],
      [
        { p1: 'move 1', p2: 'move 1' }, // Substitute (sub up)
        { p1: 'move 2', p2: 'move 1', injectBefore: [{ side: 0, confusion: true }] },
        { p1: 'move 2', p2: 'move 1' },
        { p1: 'move 2', p2: 'move 1' },
        { p1: 'move 2', p2: 'move 1' },
        { p1: 'move 2', p2: 'move 1' },
      ], seed);
  }

  // A CONFUSION SECONDARY (Water Pulse 20% conf) into a SUBBED mon: drawn? does the
  // random(2,6) duration draw fire if blocked? (The conf would land on the SUB-owner's mon.)
  await run('Water Pulse confusion secondary into SUBBED mon',
    [mon('Suicune', ['waterpulse', 'splash'], { evs: { spa: 252 } })],
    [mon('Snorlax', ['substitute', 'splash'], { evs: { hp: 252 } })],
    [
      { p1: 'move 2', p2: 'move 1' }, // Suicune Splash, Snorlax Substitute
      { p1: 'move 1', p2: 'move 2' }, // Suicune Water Pulse INTO sub ; Snorlax Splash
      { p1: 'move 1', p2: 'move 2' },
    ], [7, 11, 13, 17]);
}
main().catch((e) => { console.error(e.stack || String(e)); process.exit(1); });
