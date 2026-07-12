// probe_batch1_order.js — nail the PROTOCOL-LINE ORDER of the batch-1 post-hit
// effects (recoil / drain / self-drop / item-removal / rapid-spin) relative to the
// move's own secondary + effectiveness + damage lines, by capturing the omniscient
// `|...|` stream for a single hit of each move. All effects are draw-free (settled by
// probe_batch1_movecoverage.js); this only fixes the emitted line order.
//
// Run:  node src/rust_sim/harness/probe_batch1_order.js
'use strict';
const path = require('path');
const PS = path.resolve(__dirname, '../../../deps/pokemon-showdown');
const { BattleStream, getPlayerStreams } = require(path.join(PS, 'dist/sim/battle-stream'));
const { Teams } = require(path.join(PS, 'dist/sim'));

const IV31 = { hp: 31, atk: 31, def: 31, spa: 31, spd: 31, spe: 31 };
const EV0 = { hp: 0, atk: 0, def: 0, spa: 0, spd: 0, spe: 0 };
function mon(species, moves, opts = {}) {
  return {
    species, item: opts.item || '', ability: opts.ability || 'No Ability', moves,
    evs: { ...EV0, ...(opts.evs || {}) }, ivs: IV31,
    nature: opts.nature || 'Serious', level: 100, gender: 'N',
  };
}
function tick() { return new Promise((r) => setTimeout(r, 0)); }

async function run(label, p1team, p2team, inject, seed) {
  const stream = new BattleStream();
  const lines = [];
  const streams = getPlayerStreams(stream);
  (async () => { for await (const ch of streams.omniscient) { for (const l of String(ch).split('\n')) if (l) lines.push(l); } })();
  streams.omniscient.write(`>start {"formatid":"gen3customgame","seed":${JSON.stringify(seed || [7, 11, 13, 17])}}`);
  streams.omniscient.write(`>player p1 ${JSON.stringify({ name: 'P1', team: Teams.pack(p1team) })}`);
  streams.omniscient.write(`>player p2 ${JSON.stringify({ name: 'P2', team: Teams.pack(p2team) })}`);
  for (let i = 0; i < 12; i++) await tick();
  const battle = stream.battle;
  for (const inj of (inject || [])) {
    const m = battle.sides[inj.side].active[0];
    if (inj.spikes) for (let k = 0; k < inj.spikes; k++) battle.sides[inj.side].addSideCondition('spikes', battle.sides[1 - inj.side].active[0]);
    if (inj.leechseed) m.addVolatile('leechseed', battle.sides[1 - inj.side].active[0]);
    if (inj.hp !== undefined) m.hp = inj.hp;
  }
  const start = lines.length;
  streams.omniscient.write('>p1 move 1');
  streams.omniscient.write('>p2 move 1');
  for (let k = 0; k < 18; k++) await tick();
  console.log(`\n=== ${label} ===`);
  for (const l of lines.slice(start).filter((l) => l.startsWith('|') && !l.startsWith('|t:|') && l !== '|upkeep')) {
    console.log('  ' + l);
  }
  try { streams.omniscient.destroy(); } catch (e) {}
}

async function main() {
  await run('RECOIL: Double-Edge',
    [mon('Tauros', ['doubleedge'], { evs: { atk: 252, spe: 252 } })],
    [mon('Snorlax', ['splash'], { ability: 'Immunity', evs: { hp: 252 } })]);
  await run('DRAIN: Giga Drain (user injured)',
    [mon('Sceptile', ['gigadrain'], { evs: { spa: 252, spe: 252 } })],
    [mon('Snorlax', ['splash'], { ability: 'Immunity', evs: { hp: 252 } })],
    [{ side: 0, hp: 100 }]);
  await run('SELF-DROP: Overheat',
    [mon('Charizard', ['overheat'], { evs: { spa: 252, spe: 252 } })],
    [mon('Snorlax', ['splash'], { ability: 'Immunity', evs: { hp: 252 } })]);
  await run('SELF-DROP: Superpower',
    [mon('Machamp', ['superpower'], { evs: { atk: 252, spe: 252 } })],
    [mon('Snorlax', ['splash'], { ability: 'Immunity', evs: { hp: 252 } })]);
  await run('ITEM: Knock Off',
    [mon('Tyranitar', ['knockoff'], { evs: { atk: 252, spe: 252 } })],
    [mon('Snorlax', ['splash'], { item: 'Leftovers', ability: 'Immunity', evs: { hp: 252 } })]);
  await run('ITEM: Thief (itemless attacker)',
    [mon('Gengar', ['thief'], { item: '', evs: { spa: 252, spe: 252 } })],
    [mon('Snorlax', ['splash'], { item: 'Leftovers', ability: 'Immunity', evs: { hp: 252 } })]);
  await run('ITEM: Covet (itemless attacker)',
    [mon('Gengar', ['covet'], { item: '', evs: { atk: 252, spe: 252 } })],
    [mon('Snorlax', ['splash'], { item: 'Leftovers', ability: 'Immunity', evs: { hp: 252 } })]);
  await run('RAPID SPIN: clears user spikes + leech',
    [mon('Forretress', ['rapidspin'], { evs: { hp: 252, spe: 252 } })],
    [mon('Snorlax', ['splash'], { ability: 'Immunity', evs: { hp: 252 } })],
    [{ side: 0, spikes: 2, leechseed: true }]);
  // Recoil + secondary composition (Take Down has no secondary; try a move with recoil AND secondary?
  // gen3 recoil moves don't carry a status secondary except contact procs).
  await run('KNOCKOFF into Sticky Hold (blocked)',
    [mon('Tyranitar', ['knockoff'], { evs: { atk: 252, spe: 252 } })],
    [mon('Gastrodon', ['splash'], { item: 'Leftovers', ability: 'Sticky Hold', evs: { hp: 252 } })]);
}
main().catch((e) => { console.error(e.stack || String(e)); process.exit(1); });
