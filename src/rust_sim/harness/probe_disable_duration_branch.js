// probe_disable_duration_branch.js — nail the EXACT stored Disable duration for BOTH the
// FASTER-disabler (willMove(target) TRUE → onStart duration--) and SLOWER-disabler
// (willMove(target) FALSE → no decrement) branches, and the per-turn residual tick, against
// the OMNISCIENT in-process BattleStream (no server). The port stores the sim's POST-onStart
// duration; its residual DisableDuration handler ticks it −1 each residual (including the
// disable turn's own end-of-turn residual), matching the sim's generic duration loop.
//
// We SWEEP seeds until the acc-55 Disable LANDS on the intended turn, then print the volatile
// `duration` after EACH following turn's residual (the tick-down → the free-up turn) plus the
// raw random(2,6) roll and whether the disabler moved first (willMove(target)).
//
// Run:  node src/rust_sim/harness/probe_disable_duration_branch.js
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
function dur(m, key) { return m && m.volatiles[key] ? m.volatiles[key].duration : null; }

// disablerSide uses Disable (slot 0) on turn 2; the target (1-disablerSide) sets its lastMove
// on turn 1 (its slot 0), then just uses slot 1 (a non-disabled filler) each following turn.
async function runOnce(seed, p1team, p2team, disablerSide) {
  const stream = new BattleStream();
  const streams = getPlayerStreams(stream);
  (async () => { for await (const ch of streams.omniscient) { void ch; } })();
  streams.omniscient.write(`>start {"formatid":"${FORMAT}","seed":${JSON.stringify(seed)}}`);
  streams.omniscient.write(`>player p1 ${JSON.stringify({ name: 'P1', team: Teams.pack(p1team) })}`);
  streams.omniscient.write(`>player p2 ${JSON.stringify({ name: 'P2', team: Teams.pack(p2team) })}`);
  for (let i = 0; i < 12; i++) await tick();
  const battle = stream.battle;
  const targetSide = 1 - disablerSide;

  let rolled = null;
  const realRandom = battle.random.bind(battle);
  battle.random = function (from, to) {
    const r = realRandom(from, to);
    if (from === 2 && to === 6) rolled = r;
    return r;
  };

  const disMove = disablerSide === 0 ? 'move 1' : 'move 2';
  const disDis = disablerSide === 0 ? 'move 1' : 'move 1'; // Disable is slot 0 on the disabler
  // Plans: turn 1 both attack (target's slot 0), turn 2 disabler uses Disable (its slot 0),
  // target uses its slot 1 filler. Then filler forever.
  const p1turn1 = disablerSide === 0 ? 'move 2' : 'move 1';
  const p2turn1 = disablerSide === 0 ? 'move 1' : 'move 2';
  const p1turn2 = disablerSide === 0 ? 'move 1' : 'move 2';
  const p2turn2 = disablerSide === 0 ? 'move 2' : 'move 1';
  const plan = [
    { p1: p1turn1, p2: p2turn1 },
    { p1: p1turn2, p2: p2turn2 },
    ...Array(8).fill({ p1: 'move 2', p2: 'move 2' }),
  ];

  const seq = [];
  let willMove = null;
  let i = 0, safety = 0, sawDisable = false;
  while (!battle.ended && safety < 20) {
    safety++;
    const rs = battle.requestState;
    if (rs !== 'move' && rs !== 'switch') { await tick(); continue; }
    const entry = plan[Math.min(i, plan.length - 1)];
    const isDisableTurn = i === 1;
    if (isDisableTurn) rolled = null;
    try { if (entry.p1) streams.omniscient.write(`>p1 ${entry.p1}`); } catch (e) {}
    try { if (entry.p2) streams.omniscient.write(`>p2 ${entry.p2}`); } catch (e) {}
    for (let k = 0; k < 18; k++) await tick();
    i++;
    const tgt = battle.sides[targetSide].active[0];
    const d = dur(tgt, 'disable');
    if (isDisableTurn) {
      const dis = battle.sides[disablerSide].active[0];
      willMove = dis.speed > tgt.speed; // disabler faster ⇒ target still to move ⇒ willMove TRUE
      if (d === null) { try { streams.omniscient.destroy(); } catch (e) {} return null; } // missed → retry
      sawDisable = true;
      seq.push(d);
    } else if (sawDisable) {
      seq.push(d);
      if (d === null) break;
    }
  }
  try { streams.omniscient.destroy(); } catch (e) {}
  return { rolled, willMove, seq };
}

async function findLanding(label, p1team, p2team, disablerSide) {
  for (let s = 0; s < 40; s++) {
    const seed = [s * 7 + 1, s * 11 + 3, s * 13 + 5, s * 17 + 7];
    const r = await runOnce(seed, p1team, p2team, disablerSide);
    if (r && r.seq.length && r.seq[0] !== null) {
      const postOnStart = r.seq[0] + 1; // the disable turn's residual already ticked once
      const expected = r.willMove ? r.rolled - 1 : r.rolled;
      const freeTurn = r.seq.indexOf(null); // #turns after the disable turn that it frees
      console.log(`\n=== ${label} ===  seed=${JSON.stringify(seed)}`);
      console.log(`  rolled(random(2,6)) = ${r.rolled}`);
      console.log(`  willMove(target) at onStart = ${r.willMove} (disabler ${r.willMove ? 'FASTER' : 'SLOWER'})`);
      console.log(`  post-onStart duration (reconstructed) = ${postOnStart}`);
      console.log(`  duration after each following residual (disable turn first) = [${r.seq.join(', ')}]`);
      console.log(`  → the move frees up ${freeTurn} turns after the disable turn (inclusive of the disable turn's own residual)`);
      console.log(`  → invariant post-onStart == (willMove ? rolled-1 : rolled) = ${expected}: ${postOnStart === expected ? 'MATCH' : 'MISMATCH'}`);
      return r;
    }
  }
  console.log(`\n=== ${label} ===  NO landing found in the seed sweep`);
  return null;
}

async function main() {
  // FASTER disabler: fast Aerodactyl disables slow Snorlax (target moves AFTER → willMove TRUE).
  await findLanding('FASTER disabler (willMove TRUE → onStart duration--)',
    [mon('Aerodactyl', ['disable', 'rockslide'], { evs: { spe: 252 } })],
    [mon('Snorlax', ['bodyslam', 'blizzard'], { evs: { hp: 252, atk: 128, spd: 128 }, nature: 'Brave', ivs: { ...IV31, spe: 0 } })],
    0);

  // SLOWER disabler: slow Blissey disables fast Snorlax (target moves FIRST → willMove FALSE).
  await findLanding('SLOWER disabler (willMove FALSE → NO decrement)',
    [mon('Blissey', ['disable', 'softboiled'], { evs: { hp: 252, def: 252 }, nature: 'Brave', ivs: { ...IV31, spe: 0 } })],
    [mon('Aerodactyl', ['rockslide', 'earthquake'], { evs: { spe: 252 } })],
    0);
}
main().catch((e) => { console.error(e.stack || String(e)); process.exit(1); });
