// probe_spikes_rng.js — instrument the gen3 Spikes draw model bit-for-bit.
//
// Verifies, against the OMNISCIENT in-process BattleStream (no server):
//   1. The Spikes MOVE draws NOTHING (never-miss, sets the foe-side condition; a
//      Spikes-at-max FAILS draw-free).
//   2. A grounded switch-in to a spiked side takes `max(floor([_,3,4,6][layers]*maxhp/24),1)`
//      damage and the runSwitch (EntryHazard → SwitchIn → ability Start) draws NOTHING
//      beyond the existing action-order/eachEvent shuffles.
//   3. A Flying / Levitate switch-in takes ZERO.
//   4. A Spikes hit that KOs a low-HP switch-in faints it + forces ANOTHER replacement.
//
// We wrap `battle.prng.next` to log every draw with a label window, then run a few
// constructed turns and print the per-window draw count + the realized HP deltas. This
// is the manual oracle that pins the Rust engine's Spikes draw count + damage.
//
// Run:  node src/rust_sim/harness/probe_spikes_rng.js
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

async function run(label, p1team, p2team, plan) {
  const stream = new BattleStream();
  const streams = getPlayerStreams(stream);
  const log = [];
  (async () => { for await (const ch of streams.omniscient) for (const l of ch.split('\n')) if (l) log.push(l); })();
  const seed = [7, 11, 13, 17];
  streams.omniscient.write(`>start {"formatid":"${FORMAT}","seed":${JSON.stringify(seed)}}`);
  streams.omniscient.write(`>player p1 ${JSON.stringify({ name: 'P1', team: Teams.pack(p1team) })}`);
  streams.omniscient.write(`>player p2 ${JSON.stringify({ name: 'P2', team: Teams.pack(p2team) })}`);
  for (let i = 0; i < 10; i++) await tick();

  const battle = stream.battle;
  // Wrap the underlying rng.next to count raw draws (battle.prng.rng is the backend).
  let drawCount = 0;
  const rng = battle.prng.rng;
  const realNext = rng.next.bind(rng);
  rng.next = function (...a) { drawCount++; return realNext(...a); };

  console.log(`\n=== ${label} ===`);
  let i = 0, safety = 0;
  while (!battle.ended && safety < 60) {
    safety++;
    const rs = battle.requestState;
    if (rs !== 'move' && rs !== 'switch') { await tick(); continue; }
    const before = battle.prng.getSeed();
    const dc0 = drawCount;
    const entry = plan[Math.min(i, plan.length - 1)];
    i++;
    if (entry.p1) streams.omniscient.write(`>p1 ${entry.p1}`);
    if (entry.p2) streams.omniscient.write(`>p2 ${entry.p2}`);
    for (let k = 0; k < 16; k++) await tick();
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
  try { streams.omniscient.destroy(); } catch (e) {}
}

async function main() {
  // (A) Spikes move draws nothing; grounded switch-in takes maxhp/8 (1 layer).
  //     p1: Skarmory (Spikes) + Snorlax bench. p2: Forretress + Snorlax bench (grounded).
  await run('1-layer grounded switch-in',
    [mon('Skarmory', ['spikes', 'steelwing'], { ability: 'Keen Eye', evs: { hp: 252, def: 252 } }),
     mon('Snorlax', ['bodyslam'], { ability: 'Immunity', evs: { hp: 252 } })],
    [mon('Blissey', ['softboiled', 'icebeam'], { ability: 'Natural Cure', evs: { hp: 252, def: 252 } }),
     mon('Snorlax', ['bodyslam'], { ability: 'Immunity', evs: { hp: 252 } })],
    [
      { p1: 'move 1', p2: 'move 1' }, // Skarmory Spikes (lay 1 on p2 side) ; Blissey Soft-Boiled
      { p1: 'move 2', p2: 'switch 2' }, // p2 switches Snorlax IN → takes maxhp/8 spikes
    ]);

  // (B) 2 + 3 layers — stack, then a grounded switch-in takes maxhp/6, maxhp/4.
  await run('stack to 3 layers + grounded switch-in',
    [mon('Skarmory', ['spikes', 'steelwing'], { ability: 'Keen Eye', evs: { hp: 252, def: 252 } })],
    [mon('Blissey', ['softboiled'], { ability: 'Natural Cure', evs: { hp: 252, def: 252 } }),
     mon('Snorlax', ['bodyslam'], { ability: 'Immunity', evs: { hp: 252 } })],
    [
      { p1: 'move 1', p2: 'move 1' }, // Spikes (1)
      { p1: 'move 1', p2: 'move 1' }, // Spikes (2)
      { p1: 'move 1', p2: 'move 1' }, // Spikes (3)
      { p1: 'move 1', p2: 'move 1' }, // Spikes (already 3 → FAIL, draw-free)
      { p1: 'move 2', p2: 'switch 2' }, // Snorlax IN → maxhp/4 (3 layers)
    ]);

  // (C) Flying + Levitate switch-in takes ZERO.
  await run('flying + levitate immune',
    [mon('Skarmory', ['spikes', 'steelwing'], { ability: 'Keen Eye', evs: { hp: 252, def: 252 } })],
    [mon('Blissey', ['softboiled'], { ability: 'Natural Cure', evs: { hp: 252, def: 252 } }),
     mon('Salamence', ['dragonclaw'], { ability: 'Intimidate', evs: { hp: 252 } }),  // Flying
     mon('Claydol', ['psychic'], { ability: 'Levitate', evs: { hp: 252 } })],         // Levitate
    [
      { p1: 'move 1', p2: 'move 1' }, // Spikes (1)
      { p1: 'move 1', p2: 'move 1' }, // Spikes (2)
      { p1: 'move 2', p2: 'switch 2' }, // Salamence (Flying) IN → ZERO
      { p1: 'move 2', p2: 'switch 3' }, // Claydol (Levitate) IN → ZERO
    ]);

  // (D) Spikes KO on a FORCED replacement → ANOTHER forced replacement (which also
  //     takes spikes). p1 has 3 layers of spikes on the p2 side; p2's lead faints to an
  //     attack, and its low-HP grounded replacement is KO'd by the spikes ON ENTRY, which
  //     forces yet ANOTHER replacement (also grounded → also takes spikes). p1 never
  //     attacks the replacement (Splash) so the spikes hit is the SOLE damage — a pure
  //     spikes-KO-on-entry chain. The replacements are pre-chipped (Final Gambit-free: use
  //     low maxhp lvl-1 mons so 3-layer spikes [floor(maxhp/4)] ≥ their HP).
  await run('spikes KO on forced replacement → ANOTHER replacement',
    [mon('Skarmory', ['spikes', 'drillpeck', 'splash'], { ability: 'Keen Eye', evs: { hp: 252, atk: 252 } })],
    [mon('Magikarp', ['splash'], { level: 1, ability: 'Swift Swim' }),       // lead, KO'd by Drill Peck
     mon('Diglett', ['scratch'], { level: 1, ability: 'Sand Veil' }),         // grounded, lvl1 → spikes KO on entry
     mon('Sandshrew', ['scratch'], { level: 1, ability: 'Sand Veil' })],      // grounded, lvl1 → spikes KO on entry too
    [
      { p1: 'move 1', p2: 'move 1' }, // Spikes (1)
      { p1: 'move 1', p2: 'move 1' }, // Spikes (2)
      { p1: 'move 1', p2: 'move 1' }, // Spikes (3)
      { p1: 'move 2', p2: 'move 1' }, // Drill Peck KOs Magikarp (lvl1) → p2 forced to replace
      { p1: 'move 3', p2: 'switch 2' }, // p1 Splash ; Diglett IN → spikes KO on entry → ANOTHER replace
      { p1: 'move 3', p2: 'switch 3' }, // Sandshrew IN → spikes (also KO?) ; p1 Splash
    ]);
}
main().catch((e) => { console.error(e.stack || String(e)); process.exit(1); });
