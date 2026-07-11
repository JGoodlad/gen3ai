// probe_taunt_disable_duration.js — nail the EXACT duration arithmetic + residual tick +
// the raw random(2,6) draw for Disable, and the residual duration-handler for BOTH, vs the
// omniscient sim. This is the crux the port must match bit-for-bit.
//
// Instruments:
//   - battle.random / battle.randomChance calls (label + args + result) so we SEE the exact
//     Disable duration draw and that Taunt draws NONE.
//   - the disable/taunt volatile duration BEFORE and AFTER each residual (the tick-down).
//   - the residual handler ORDER (does taunt at onResidualOrder 10 sub 15 participate in the
//     speed-sort tie-shuffle? does disable — which has NO onResidualOrder — tick at all in the
//     residual, or is its duration decremented by the generic addVolatile duration machinery?).
//
// Run:  node src/rust_sim/harness/probe_taunt_disable_duration.js
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

function dur(m, key) {
  return m && m.volatiles[key] ? m.volatiles[key].duration : '-';
}

async function run(label, p1team, p2team, plan) {
  const stream = new BattleStream();
  const streams = getPlayerStreams(stream);
  (async () => { for await (const ch of streams.omniscient) { void ch; } })();
  const seed = [7, 11, 13, 17];
  streams.omniscient.write(`>start {"formatid":"${FORMAT}","seed":${JSON.stringify(seed)}}`);
  streams.omniscient.write(`>player p1 ${JSON.stringify({ name: 'P1', team: Teams.pack(p1team) })}`);
  streams.omniscient.write(`>player p2 ${JSON.stringify({ name: 'P2', team: Teams.pack(p2team) })}`);
  for (let i = 0; i < 12; i++) await tick();

  const battle = stream.battle;
  // Instrument random / randomChance so we SEE the labeled draws in a decision window.
  const log = [];
  const realRandom = battle.random.bind(battle);
  battle.random = function (from, to) {
    const r = realRandom(from, to);
    log.push(`random(${from},${to})=>${r}`);
    return r;
  };
  const realChance = battle.randomChance.bind(battle);
  battle.randomChance = function (num, den) {
    const r = realChance(num, den);
    log.push(`randomChance(${num},${den})=>${r}`);
    return r;
  };

  console.log(`\n=== ${label} ===  initSeed=${battle.prng.getSeed()}`);

  let i = 0, safety = 0;
  while (!battle.ended && safety < 40) {
    safety++;
    const rs = battle.requestState;
    if (rs !== 'move' && rs !== 'switch') { await tick(); continue; }
    const before = battle.prng.getSeed();
    log.length = 0;
    const entry = plan[Math.min(i, plan.length - 1)]; i++;
    try { if (entry.p1) streams.omniscient.write(`>p1 ${entry.p1}`); } catch (e) {}
    try { if (entry.p2) streams.omniscient.write(`>p2 ${entry.p2}`); } catch (e) {}
    for (let k = 0; k < 18; k++) await tick();
    const after = battle.prng.getSeed();
    const a0 = battle.sides[0].active[0], a1 = battle.sides[1].active[0];
    console.log(`  ${JSON.stringify(entry)}  draws=[${log.join(' | ')}]`);
    console.log(`     p1 taunt=${dur(a0, 'taunt')} disable=${dur(a0, 'disable')} | ` +
      `p2 taunt=${dur(a1, 'taunt')} disable=${dur(a1, 'disable')}`);
    if (entry.stop) break;
  }
  try { streams.omniscient.destroy(); } catch (e) {}
}

async function main() {
  // TAUNT: caster (Gengar, fast) taunts slow Snorlax. Watch: taunt turn draws ONLY the
  // accuracy roll (NO random for duration). Taunt duration ticks each residual.
  await run('TAUNT duration + residual tick (caster fast → residual ticks same turn)',
    [mon('Gengar', ['taunt', 'shadowball'], { evs: { spa: 252, spe: 252 } })],
    [mon('Snorlax', ['bodyslam', 'toxic'], { evs: { hp: 252, atk: 252 }, nature: 'Brave', ivs: { ...IV31, spe: 0 } })],
    [
      { p1: 'move 1', p2: 'move 1' }, // Taunt — draws only accuracy? watch duration
      { p1: 'move 2', p2: 'move 1' },
      { p1: 'move 2', p2: 'move 1' },
      { p1: 'move 2', p2: 'move 1' },
    ]);

  // TAUNT caster SLOWER — taunts AFTER the target already moved. Does duration behave the
  // same? (No willMove++ for taunt — taunt has no such onStart clause.)
  await run('TAUNT caster SLOWER (target moved first)',
    [mon('Snorlax', ['taunt', 'bodyslam'], { evs: { hp: 252, atk: 252 }, nature: 'Brave', ivs: { ...IV31, spe: 0 } })],
    [mon('Gengar', ['shadowball', 'toxic'], { evs: { spa: 252, spe: 252 } })],
    [
      { p1: 'move 1', p2: 'move 2' }, // Gengar Toxic first, then Snorlax Taunt
      { p1: 'move 2', p2: 'move 1' },
      { p1: 'move 2', p2: 'move 1' },
      { p1: 'move 2', p2: 'move 1' },
    ]);

  // DISABLE caster SLOWER (target moved first → willMove false → duration++). Raw random(2,6)
  // + the ++ + the residual tick. p2 (Snorlax) faster moves first, then p1 (Blissey) Disables.
  await run('DISABLE caster SLOWER: random(2,6) + duration++ (willMove false) + residual tick',
    [mon('Blissey', ['disable', 'softboiled'], { evs: { hp: 252, def: 252 }, nature: 'Brave', ivs: { ...IV31, spe: 0 } })],
    [mon('Snorlax', ['bodyslam', 'rest'], { evs: { hp: 252, atk: 252 } })], // faster
    [
      { p1: 'move 2', p2: 'move 1' }, // p2 Body Slam (lastMove); p1 Soft-Boiled
      { p1: 'move 1', p2: 'move 2' }, // p1 Disable (p2 Rest) — p2 faster moved first → willMove++?
      { p1: 'move 2', p2: 'move 2' },
      { p1: 'move 2', p2: 'move 2' },
      { p1: 'move 2', p2: 'move 2' },
      { p1: 'move 2', p2: 'move 2' },
    ]);

  // DISABLE caster FASTER (target moves AFTER disable → willMove TRUE → NO duration++). p1
  // (Aerodactyl, fast) disables. But target must have a lastMove — so give it a prior turn.
  await run('DISABLE caster FASTER: random(2,6) + NO duration++ (willMove true) + residual tick',
    [mon('Aerodactyl', ['disable', 'rockslide'], { evs: { hp: 252, spe: 252 } })], // fast
    [mon('Snorlax', ['bodyslam', 'rest'], { evs: { hp: 252, atk: 252 }, nature: 'Brave', ivs: { ...IV31, spe: 0 } })], // slow
    [
      { p1: 'move 2', p2: 'move 1' }, // p2 Body Slam (lastMove); p1 Rock Slide
      { p1: 'move 1', p2: 'move 1' }, // p1 Disable (faster → p2 moves AFTER → willMove TRUE, no ++)
      { p1: 'move 2', p2: 'move 2' },
      { p1: 'move 2', p2: 'move 2' },
      { p1: 'move 2', p2: 'move 2' },
      { p1: 'move 2', p2: 'move 2' },
    ]);
}
main().catch((e) => { console.error(e.stack || String(e)); process.exit(1); });
