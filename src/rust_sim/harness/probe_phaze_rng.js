// probe_phaze_rng.js — instrument the gen3 Roar/Whirlwind phaze draw model bit-for-bit.
//
// Verifies, against the OMNISCIENT in-process BattleStream (no server):
//   1. PRIORITY: Roar/Whirlwind are priority -6 → the phazer moves LAST.
//   2. ACCURACY: gen3 Roar/Whirlwind resolve to `accuracy: 100` (NOT `true` — the base
//      Showdown data says `true`, but the resolved gen-3 dex value is 100), so a phaze is
//      NOT never-miss → it DRAWS `randomChance(100,100)` (always passes but CONSUMES a draw).
//   3. THE RANDOM TARGET DRAW: on a successful phaze, `dragIn` →
//      `getRandomSwitchable(side)` → `sample(canSwitchIn)` → `this.random(n)` —
//      ONE draw, EVEN when n==1 (`random(1)` always returns 0 but STILL consumes a draw).
//   4. THE FAIL CASE: a phaze with NO eligible target (foe's last mon alive) →
//      `forceSwitch` checks `canSwitch(target.side)` (false) → NO forceSwitchFlag, NO drag,
//      NO draw (the move just `-fail`s — but it's never-miss, so the WHOLE turn for the
//      phaze move is draw-free).
//   5. PHAZE-INTO-SPIKES: the dragged-in mon takes Spikes via the runSwitch EntryHazard.
//   6. PHAZE INTO A SPIKES-KO: the dragged-in mon is KO'd on entry → forces a NORMAL replace.
//
// We wrap `battle.prng.next` to count raw draws per decision window, then run a few
// constructed turns and print per-window draw count + realized state. This is the manual
// oracle that pins the Rust engine's phaze draw count + target selection.
//
// Run:  node src/rust_sim/harness/probe_phaze_rng.js
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

  console.log(`\n=== ${label} (seed ${JSON.stringify(seed)}) ===`);
  let i = 0, safety = 0;
  while (!battle.ended && safety < 80) {
    safety++;
    const rs = battle.requestState;
    if (rs !== 'move' && rs !== 'switch') { await tick(); continue; }
    const before = battle.prng.getSeed();
    const dc0 = drawCount;
    const entry = plan[Math.min(i, plan.length - 1)];
    i++;
    if (entry.p1) streams.omniscient.write(`>p1 ${entry.p1}`);
    if (entry.p2) streams.omniscient.write(`>p2 ${entry.p2}`);
    for (let k = 0; k < 18; k++) await tick();
    const after = battle.prng.getSeed();
    const drew = drawCount - dc0;
    const spk = [battle.sides[0].sideConditions.spikes, battle.sides[1].sideConditions.spikes]
      .map((s) => (s ? s.layers : 0));
    const a0 = battle.sides[0].active[0], a1 = battle.sides[1].active[0];
    console.log(
      `  [${rs}] ${JSON.stringify(entry)} draws=${drew} seedBefore=${before} seedAfter=${after}\n` +
      `        spikes[p1,p2]=${JSON.stringify(spk)} ` +
      `p1=${a0 ? a0.species.name + ' ' + a0.hp + '/' + a0.maxhp + (a0.fainted ? ' FNT' : '') : '-'} ` +
      `p2=${a1 ? a1.species.name + ' ' + a1.hp + '/' + a1.maxhp + (a1.fainted ? ' FNT' : '') : '-'}`);
    if (entry.stop) break;
  }
  console.log(`  ended=${battle.ended} winner=${battle.winner}`);
  // Print which mons got dragged in across the whole battle (the random-target proof).
  const drags = log.filter((l) => l.startsWith('|drag|'));
  console.log(`  DRAGS: ${JSON.stringify(drags)}`);
  try { streams.omniscient.destroy(); } catch (e) {}
  return { drags, log };
}

async function main() {
  // (A) Roar drags in a RANDOM mon — sweep seeds so DIFFERENT mons get dragged.
  //     p1 Roar phazer (slow → moves last); p2 active Blissey + 2 bench (Snorlax, Skarmory).
  //     Across seeds, the sampled bench should vary, proving the random-target draw.
  for (const seed of [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [2, 22, 222, 2222]]) {
    await run('A: Roar drags a random bench mon', seed,
      [mon('Suicune', ['roar', 'surf'], { ability: 'Pressure', evs: { hp: 252, def: 252 }, nature: 'Relaxed' })],
      [mon('Blissey', ['softboiled', 'icebeam'], { ability: 'Pressure', evs: { hp: 252 } }),
       mon('Snorlax', ['bodyslam'], { ability: 'Pressure', evs: { hp: 252 } }),
       mon('Skarmory', ['steelwing'], { ability: 'Keen Eye', evs: { hp: 252 } })],
      [
        { p1: 'move 1', p2: 'move 1' }, // Suicune Roar (slow, last) drags a p2 bench
        { stop: true },
      ]);
  }

  // (B) Roar with n==1 eligible (one bench mon) — STILL draws the sample (random(1)).
  await run('B: Roar with ONE eligible bench (n=1 still draws)', [3, 5, 7, 9],
    [mon('Suicune', ['roar', 'surf'], { ability: 'Pressure', evs: { hp: 252, def: 252 }, nature: 'Relaxed' })],
    [mon('Blissey', ['softboiled'], { ability: 'Pressure', evs: { hp: 252 } }),
     mon('Snorlax', ['bodyslam'], { ability: 'Pressure', evs: { hp: 252 } })],
    [
      { p1: 'move 1', p2: 'move 1' }, // Roar → only Snorlax eligible → sample([Snorlax]) DRAWS
      { stop: true },
    ]);

  // (C) Roar FAIL — foe has NO eligible switch-in (last mon alive). NO drag, NO draw.
  await run('C: Roar FAILS (foe last mon, no eligible target)', [3, 5, 7, 9],
    [mon('Suicune', ['roar', 'surf'], { ability: 'Pressure', evs: { hp: 252, def: 252 }, nature: 'Relaxed' })],
    [mon('Blissey', ['softboiled'], { ability: 'Pressure', evs: { hp: 252 } })],
    [
      { p1: 'move 1', p2: 'move 1' }, // Roar → p2 has no bench → -fail, draw-free
      { stop: true },
    ]);

  // (D) Roar INTO SPIKES — p1 lays spikes on p2 side, then Roars; the dragged grounded mon
  //     takes the hazard chip on entry (EntryHazard in runSwitch).
  await run('D: Roar into Spikes (dragged mon takes hazard)', [1, 2, 3, 4],
    [mon('Skarmory', ['spikes', 'roar', 'steelwing'], { ability: 'Keen Eye', evs: { hp: 252, def: 252 }, nature: 'Relaxed' })],
    [mon('Blissey', ['softboiled'], { ability: 'Pressure', evs: { hp: 252 } }),
     mon('Snorlax', ['bodyslam'], { ability: 'Pressure', evs: { hp: 252 } }),
     mon('Tyranitar', ['crunch'], { ability: 'Sand Stream', evs: { hp: 252 } })],
    [
      { p1: 'move 1', p2: 'move 1' }, // Skarmory Spikes (1)
      { p1: 'move 1', p2: 'move 1' }, // Spikes (2)
      { p1: 'move 2', p2: 'move 1' }, // Roar → drags a grounded p2 bench → spikes chip
      { stop: true },
    ]);

  // (E) Roar into a SPIKES-KO — 3 layers, the dragged grounded lvl-1 mon is KO'd on entry,
  //     forcing a NORMAL replacement.
  await run('E: Roar into a Spikes-KO (chains a normal replacement)', [1, 2, 3, 4],
    [mon('Skarmory', ['spikes', 'roar', 'drillpeck'], { ability: 'Keen Eye', evs: { hp: 252, atk: 252 } })],
    [mon('Blissey', ['softboiled'], { ability: 'Pressure', evs: { hp: 252 } }),
     mon('Diglett', ['scratch'], { level: 1, ability: 'Sand Veil' }),    // grounded lvl1 → spikes KO on entry
     mon('Sandshrew', ['scratch'], { level: 1, ability: 'Sand Veil' })], // grounded lvl1
    [
      { p1: 'move 1', p2: 'move 1' }, // Spikes (1)
      { p1: 'move 1', p2: 'move 1' }, // Spikes (2)
      { p1: 'move 1', p2: 'move 1' }, // Spikes (3)
      { p1: 'move 2', p2: 'move 1' }, // Roar → drags Diglett/Sandshrew → spikes KO on entry → forced replace
      { stop: true },
    ]);

  // (F) Whirlwind — same as Roar (different move id, identical mechanic).
  for (const seed of [[1, 2, 3, 4], [9, 10, 11, 12]]) {
    await run('F: Whirlwind drags a random bench mon', seed,
      [mon('Skarmory', ['whirlwind', 'steelwing'], { ability: 'Keen Eye', evs: { hp: 252, def: 252 }, nature: 'Relaxed' })],
      [mon('Blissey', ['softboiled'], { ability: 'Pressure', evs: { hp: 252 } }),
       mon('Snorlax', ['bodyslam'], { ability: 'Pressure', evs: { hp: 252 } }),
       mon('Starmie', ['surf'], { ability: 'Natural Cure', evs: { hp: 252 } })],
      [
        { p1: 'move 1', p2: 'move 1' }, // Whirlwind → drags a p2 bench
        { stop: true },
      ]);
  }
}
main().catch((e) => { console.error(e.stack || String(e)); process.exit(1); });
