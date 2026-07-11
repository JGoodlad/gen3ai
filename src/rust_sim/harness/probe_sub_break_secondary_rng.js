// probe_sub_break_secondary_rng.js — settle TWO sub×secondary interleavings vs the resolved gen3 sim:
//  (1) does the per-move secondary random(100) still draw when the hit BREAKS the sub?
//  (2) does a Shield Dust defender still FILTER the secondary when the hit lands on its SUB?
// Plus the exact A/B minimal repro board (Magcargo Flamethrower vs Venomoth Shield Dust sub).
'use strict';
const path = require('path');
const PS = path.resolve(__dirname, '../../../deps/pokemon-showdown');
const { BattleStream, getPlayerStreams } = require(path.join(PS, 'dist/sim/battle-stream'));
const { Teams } = require(path.join(PS, 'dist/sim'));

const FORMAT = 'gen3customgame';
const IV31 = { hp: 31, atk: 31, def: 31, spa: 31, spd: 31, spe: 31 };
const EV0 = { hp: 0, atk: 0, def: 0, spa: 0, spd: 0, spe: 0 };
function mon(species, moves, opts = {}) {
  return { species, item: opts.item || '', ability: opts.ability || 'No Ability', moves,
    evs: { ...EV0, ...(opts.evs || {}) }, ivs: opts.ivs || IV31,
    nature: opts.nature || 'Serious', level: opts.level || 100, gender: opts.gender || 'N' };
}
function tick() { return new Promise((r) => setTimeout(r, 0)); }

async function run(label, p1team, p2team, plan, seed) {
  const stream = new BattleStream();
  const streams = getPlayerStreams(stream);
  (async () => { for await (const ch of streams.omniscient) { void ch; } })();
  streams.omniscient.write(`>start {"formatid":"${FORMAT}","seed":${JSON.stringify(seed || [7,11,13,17])}}`);
  streams.omniscient.write(`>player p1 ${JSON.stringify({ name: 'P1', team: Teams.pack(p1team) })}`);
  streams.omniscient.write(`>player p2 ${JSON.stringify({ name: 'P2', team: Teams.pack(p2team) })}`);
  for (let i = 0; i < 12; i++) await tick();
  const battle = stream.battle;
  let log = [];
  const realRandom = battle.random.bind(battle);
  battle.random = function (from, to) { const v = realRandom(from, to); log.push(`random(${from},${to})=${v}`); return v; };
  const realRC = battle.randomChance.bind(battle);
  battle.randomChance = function (num, den) { const v = realRC(num, den); log.push(`randomChance(${num},${den})=${v}`); return v; };
  const realSample = battle.sample.bind(battle);
  battle.sample = function (arr) { const v = realSample(arr); log.push(`sample(${arr.length})=${JSON.stringify(v)}`); return v; };
  console.log(`\n=== ${label} ===`);
  let i = 0, safety = 0;
  while (!battle.ended && safety < 60) {
    safety++;
    const rs = battle.requestState;
    if (rs !== 'move' && rs !== 'switch') { await tick(); continue; }
    if (i >= plan.length) break;
    const entry = plan[i]; i++;
    log = [];
    if (entry.p1) streams.omniscient.write(`>p1 ${entry.p1}`);
    if (entry.p2) streams.omniscient.write(`>p2 ${entry.p2}`);
    for (let k = 0; k < 18; k++) await tick();
    const a0 = battle.sides[0].active[0], a1 = battle.sides[1].active[0];
    const subOf = (m) => (m && m.volatiles && m.volatiles['substitute']) ? `SUB(${m.volatiles['substitute'].hp})` : 'nosub';
    const fmt = (m) => m ? `${m.species.name} ${m.hp}/${m.maxhp} ${m.status || '-'} ${subOf(m)}` : '-';
    console.log(`  dec${i-1} ${JSON.stringify(entry)} -> p1=${fmt(a0)} | p2=${fmt(a1)} | seed=${battle.prng.getSeed ? battle.prng.getSeed() : battle.prng.seed}`);
    console.log(`      draws(${log.length}): ${log.join('  ')}`);
  }
  try { streams.omniscient.destroy(); } catch (e) {}
}

async function main() {
  // (1a) HELD sub + secondary (validated baseline): Body Slam into Blissey's big sub.
  await run('1a HELD: Body Slam par30 into SUBBED Blissey (sub survives)',
    [mon('Snorlax', ['bodyslam', 'splash'], { evs: { atk: 252 } })],
    [mon('Blissey', ['substitute', 'softboiled'], { evs: { hp: 252, def: 252 } })],
    [{ p1: 'move 2', p2: 'move 1' }, { p1: 'move 1', p2: 'move 2' }]);

  // (1b) BREAKING hit + secondary: Body Slam into Gengar's small sub (damage > sub hp).
  await run('1b BREAK: Body Slam par30 BREAKS Gengar sub',
    [mon('Snorlax', ['bodyslam', 'splash'], { evs: { atk: 252 } })],
    [mon('Gengar', ['substitute', 'splash'], { evs: { hp: 252 } })],
    [{ p1: 'move 2', p2: 'move 1' }, { p1: 'move 1', p2: 'move 2' }]);
  // NOTE Body Slam is Normal -> Gengar Ghost IMMUNE. Use Sludge Bomb instead for the break case:
  await run('1b2 BREAK: Sludge Bomb psn30 BREAKS Celebi sub',
    [mon('Gengar', ['sludgebomb', 'splash'], { evs: { spa: 252 } })],
    [mon('Celebi', ['substitute', 'splash'], { evs: {} })],
    [{ p1: 'move 2', p2: 'move 1' }, { p1: 'move 1', p2: 'move 2' }]);
  // control: same board, sub HELD (bulk the sub via hp evs? sub hp=maxhp/4) — use weaker attacker:
  await run('1b3 HELD control: Sludge Bomb psn30 into HELD Snorlax sub',
    [mon('Gengar', ['sludgebomb', 'splash'], { evs: {} })],
    [mon('Snorlax', ['substitute', 'splash'], { evs: { hp: 252, spd: 252 } })],
    [{ p1: 'move 2', p2: 'move 1' }, { p1: 'move 1', p2: 'move 2' }]);

  // (2) Shield Dust behind a sub: does the filter still apply (no draw) or does the draw fire?
  await run('2a Shield Dust BARE control: Flamethrower brn10 into bare Venomoth (no draw expected)',
    [mon('Magcargo', ['flamethrower', 'splash'], { ability: 'Flame Body', evs: { spa: 252 } })],
    [mon('Venomoth', ['splash'], { ability: 'Shield Dust', evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 1' }]);
  await run('2b Shield Dust SUB: Flamethrower brn10 into SUBBED Venomoth',
    [mon('Magcargo', ['flamethrower', 'splash'], { ability: 'Flame Body', evs: {} })],
    [mon('Venomoth', ['substitute', 'splash'], { ability: 'Shield Dust', evs: { hp: 252 } })],
    [{ p1: 'move 2', p2: 'move 1' }, { p1: 'move 1', p2: 'move 2' }]);
  // non-ShieldDust twin of 2b (same stats board) to compare draw counts directly:
  await run('2c twin: Flamethrower brn10 into SUBBED Venomoth WITHOUT Shield Dust',
    [mon('Magcargo', ['flamethrower', 'splash'], { ability: 'Flame Body', evs: {} })],
    [mon('Venomoth', ['substitute', 'splash'], { ability: 'Swarm', evs: { hp: 252 } })],
    [{ p1: 'move 2', p2: 'move 1' }, { p1: 'move 1', p2: 'move 2' }]);
}
if (!process.env.PROBE_PART2) { main().catch((e) => { console.error(e.stack || String(e)); process.exit(1); }); }

// APPENDED (same session): Tri Attack + King's Rock vs a Shield Dust defender behind a SUB.
async function main2() {
  // Tri Attack into a bare Shield Dust defender (control: filter -> no random(100)).
  await run('3a TriAttack into bare Shield Dust Venomoth',
    [mon('Porygon2', ['triattack', 'splash'], { ability: 'Trace', evs: {} , level: 50})],
    [mon('Venomoth', ['splash'], { ability: 'Shield Dust', evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 1' }]);
  // Tri Attack into a SUBBED Shield Dust defender: does the random(100) draw?
  await run('3b TriAttack into SUBBED Shield Dust Venomoth',
    [mon('Porygon2', ['triattack', 'splash'], { ability: 'Trace', evs: {}, level: 50 })],
    [mon('Venomoth', ['substitute', 'splash'], { ability: 'Shield Dust', evs: { hp: 252 } })],
    [{ p1: 'move 2', p2: 'move 1' }, { p1: 'move 1', p2: 'move 2' }]);
  // King's Rock (listed move) into a bare Shield Dust defender (control: no KR draw).
  await run('4a KR Hidden Power into bare Shield Dust Venomoth',
    [mon('Sceptile', ['hiddenpowerdark', 'splash'], { item: "King's Rock", evs: {}, level: 50 })],
    [mon('Venomoth', ['splash'], { ability: 'Shield Dust', evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 1' }]);
  // King's Rock into a SUBBED Shield Dust defender: does the KR random(100) draw?
  await run('4b KR Hidden Power into SUBBED Shield Dust Venomoth',
    [mon('Sceptile', ['hiddenpowerdark', 'splash'], { item: "King's Rock", evs: {}, level: 50 })],
    [mon('Venomoth', ['substitute', 'splash'], { ability: 'Shield Dust', evs: { hp: 252 } })],
    [{ p1: 'move 2', p2: 'move 1' }, { p1: 'move 1', p2: 'move 2' }]);
}
if (process.env.PROBE_PART2) { main2().catch((e) => { console.error(e.stack || String(e)); process.exit(1); }); }
