// probe_multi_phaze_sample.js — diagnose the multi-phaze `sample` draw-POSITION desync.
//
// ROOT CAUSE FOUND (2026-07-01, FIXED): the desync was NOT an eligible-list ORDER bug (the
// port's `possibleSwitches`/array-swap order matches the sim exactly). It was that gen-3
// Roar/Whirlwind carry the **`protect: 1`** flag, so a Protect/Detect on the target BLOCKS the
// phaze at `runEvent('TryHit')` (AFTER the accuracy roll) → NO forceSwitchFlag → NO drag → NO
// `sample` draw. The port's phaze arm was MISSING that block, so it dragged a mon (an EXTRA
// `sample`) into a protected foe the sim left in place — shifting every LATER phaze's `sample`
// PRNG position ("same total draw COUNT, wrong `sample` INDEX", compensated elsewhere → the
// post-turn seed matched while the dragged mon differed). Fixed by a `protect_blocks` check in
// the phaze arm (`turn.rs`), pinned by `regression_test.rs::phaze_blocked_by_protect_…`, and the
// PHAZE-PROTECT case in `probe_phaze_regression_rng.js`. This probe stays as the general
// multi-phaze differential (it drives constructed multi-phaze battles + prints the sim's ground
// truth per drag). It captures, at EACH phaze:
//   - the sim's `possibleSwitches` array (species IN ORDER — the list `sample` indexes)
//   - the sim's FULL `side.pokemon` array order (to see the position swaps)
//   - the `random(n)` value + sampled index + dragged mon
//   - the running seed
// It instruments `battle.sample` (the array-order source of truth) directly, so it captures the
// EXACT list the sim indexed + the index it chose — the sim ground truth the port must reproduce.
//
// Run:  node src/rust_sim/harness/probe_multi_phaze_sample.js
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

function arrayOrder(side) {
  return side.pokemon.map((p) => `${p.species.name}${p.fainted ? '(FNT)' : ''}${p.isActive ? '*' : ''}`);
}

async function run(label, seed, p1team, p2team, plan) {
  const stream = new BattleStream();
  const streams = getPlayerStreams(stream);
  const log = [];
  (async () => { for await (const ch of streams.omniscient) for (const l of ch.split('\n')) if (l) log.push(l); })();
  streams.omniscient.write(`>start {"formatid":"${FORMAT}","seed":${JSON.stringify(seed)}}`);
  streams.omniscient.write(`>player p1 ${JSON.stringify({ name: 'P1', team: Teams.pack(p1team) })}`);
  streams.omniscient.write(`>player p2 ${JSON.stringify({ name: 'P2', team: Teams.pack(p2team) })}`);
  for (let i = 0; i < 10; i++) await tick();

  const battle = stream.battle;
  let drawCount = 0;
  const rng = battle.prng.rng;
  const realNext = rng.next.bind(rng);
  rng.next = function (...a) { drawCount++; return realNext(...a); };

  // Instrument `battle.sample` — the array-order source of truth. Capture the list it
  // was given (species in order) + the index it returned.
  const sampleEvents = [];
  const realSample = battle.sample.bind(battle);
  battle.sample = function (items) {
    const seedBefore = battle.prng.getSeed();
    const result = realSample(items);
    const idx = items.indexOf(result);
    const species = items.map((p) => (p && p.species ? p.species.name : String(p)));
    sampleEvents.push({ turn: battle.turn, list: species, idx, picked: result && result.species ? result.species.name : String(result), seedBefore });
    return result;
  };

  console.log(`\n=== ${label} (seed ${JSON.stringify(seed)}) ===`);
  let i = 0, safety = 0;
  while (!battle.ended && safety < 120) {
    safety++;
    const rs = battle.requestState;
    if (rs !== 'move' && rs !== 'switch') { await tick(); continue; }
    const before = battle.prng.getSeed();
    const dc0 = drawCount;
    const sc0 = sampleEvents.length;
    const entry = plan[Math.min(i, plan.length - 1)];
    i++;
    if (entry.p1) streams.omniscient.write(`>p1 ${entry.p1}`);
    if (entry.p2) streams.omniscient.write(`>p2 ${entry.p2}`);
    for (let k = 0; k < 22; k++) await tick();
    const after = battle.prng.getSeed();
    const drew = drawCount - dc0;
    const a0 = battle.sides[0].active[0], a1 = battle.sides[1].active[0];
    console.log(
      `  [${rs}] ${JSON.stringify(entry)} draws=${drew}\n` +
      `        seedBefore=${before}\n` +
      `        seedAfter =${after}\n` +
      `        p1.pokemon=${JSON.stringify(arrayOrder(battle.sides[0]))}\n` +
      `        p2.pokemon=${JSON.stringify(arrayOrder(battle.sides[1]))}\n` +
      `        active p1=${a0 ? a0.species.name : '-'} p2=${a1 ? a1.species.name : '-'}`);
    // Print any sample events fired this decision window (the phaze drag).
    for (let s = sc0; s < sampleEvents.length; s++) {
      const ev = sampleEvents[s];
      console.log(`        >>> SAMPLE turn=${ev.turn} list=${JSON.stringify(ev.list)} random(${ev.list.length})=${ev.idx} PICKED=${ev.picked} seedBefore=${ev.seedBefore}`);
    }
    if (entry.stop) break;
  }
  console.log(`  ended=${battle.ended} winner=${battle.winner}`);
  const drags = log.filter((l) => l.startsWith('|drag|'));
  console.log(`  DRAGS: ${JSON.stringify(drags)}`);
  try { streams.omniscient.destroy(); } catch (e) {}
  return { drags, log, sampleEvents };
}

async function main() {
  // SCENARIO M: two phazes in one battle separated by an intervening VOLUNTARY SWITCH
  // that reorders p2's bench. p1 is a slow Roar phazer; p2 has 4 mons.
  //   Turn 1: p1 Roar (last) drags a random p2 bench mon. This SWAPS p2's array.
  //   Turn 2: p2 voluntarily switches (another array swap on p2).
  //   Turn 3: p1 Roar again — the SECOND phaze. Does the port's eligible list match?
  await run('M: two phazes with an intervening voluntary switch', [1, 2, 3, 4],
    [mon('Suicune', ['roar', 'surf'], { ability: 'Pressure', evs: { hp: 252, def: 252 }, nature: 'Relaxed' })],
    [mon('Blissey', ['softboiled'], { ability: 'Pressure', evs: { hp: 252 } }),
     mon('Snorlax', ['bodyslam'], { ability: 'Pressure', evs: { hp: 252 } }),
     mon('Skarmory', ['steelwing'], { ability: 'Keen Eye', evs: { hp: 252 } }),
     mon('Starmie', ['recover'], { ability: 'Natural Cure', evs: { hp: 252 } })],
    [
      { p1: 'move 1', p2: 'move 1' }, // T1: Roar drags a p2 bench
      { p1: 'move 2', p2: 'switch 3' }, // T2: p1 Surf; p2 voluntarily switches (array swap)
      { p1: 'move 1', p2: 'move 1' }, // T3: Roar again — SECOND phaze
      { p1: 'move 1', p2: 'move 1' }, // T4: Roar again — THIRD phaze
      { p1: 'move 1', p2: 'move 1' }, // T5: Roar again — FOURTH phaze
      { stop: true },
    ]);

  // SCENARIO N: BOTH sides phaze across a long history + a phaze shares a turn with a
  // secondary-move random(100) (the exact e2e repro shape). p1 Roars, p2 Ice Beams.
  await run('N: mutual phaze + a secondary-move random(100) in the same turn', [7, 11, 13, 17],
    [mon('Suicune', ['roar', 'surf'], { ability: 'Pressure', evs: { hp: 252, def: 252 }, nature: 'Relaxed' }),
     mon('Skarmory', ['whirlwind', 'steelwing'], { ability: 'Keen Eye', evs: { hp: 252 } }),
     mon('Milotic', ['recover'], { ability: 'Marvel Scale', evs: { hp: 252 } })],
    [mon('Blissey', ['icebeam', 'softboiled'], { ability: 'Natural Cure', evs: { hp: 252 } }),
     mon('Snorlax', ['bodyslam', 'whirlwind'], { ability: 'Pressure', evs: { hp: 252 } }),
     mon('Regice', ['icebeam'], { ability: 'Clear Body', evs: { hp: 252 } }),
     mon('Salamence', ['roar', 'dragonclaw'], { ability: 'Intimidate', evs: { hp: 252 } })],
    [
      { p1: 'move 1', p2: 'move 1' }, // T1: p1 Roar (last, drags p2 bench); p2 Ice Beam (secondary random(100))
      { p1: 'switch 2', p2: 'move 1' }, // T2: p1 switch to Skarmory; p2 Ice Beam
      { p1: 'move 1', p2: 'switch 4' }, // T3: p1 Whirlwind (drags p2 bench); p2 voluntary switch
      { p1: 'move 1', p2: 'move 1' }, // T4: p1 Whirlwind again; p2 move
      { p1: 'switch 3', p2: 'move 2' }, // T5: p1 switch; p2 Whirlwind (p2 drags p1 bench!)
      { p1: 'move 1', p2: 'move 1' }, // T6
      { stop: true },
    ]);
}
main().catch((e) => { console.error(e.stack || String(e)); process.exit(1); });
