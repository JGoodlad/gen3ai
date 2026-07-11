// probe_disable_duration_direct.js — DIRECTLY instrument the disable volatile's duration
// lifecycle by hooking the sim's own mutation points, removing ALL polling ambiguity:
//   - addVolatile → durationCallback sets effectState.duration (the raw random(2,6) roll)
//   - the condition onStart's `this.effectState.duration--` (willMove gate)
//   - the fieldEvent('Residual') generic loop's `handler.state.duration--` + the end() removal
// We print the disable volatile's duration at every transition WITH its cause, for BOTH a
// faster-disabler and a slower-disabler landing. Ground truth for the port's stored value.
//
// Run:  node src/rust_sim/harness/probe_disable_duration_direct.js
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

  const events = [];
  let curTurn = () => battle.turn;

  // Hook fieldEvent to catch the Residual generic duration decrement + removal on the disable
  // volatile. We wrap it and, around the call, snapshot the disable volatile's duration.
  const tgtMon = () => battle.sides[targetSide].active[0];
  const realFieldEvent = battle.fieldEvent.bind(battle);
  battle.fieldEvent = function (eventid, targets) {
    if (eventid === 'Residual') {
      const before = tgtMon() && tgtMon().volatiles.disable ? tgtMon().volatiles.disable.duration : null;
      const r = realFieldEvent(eventid, targets);
      const after = tgtMon() && tgtMon().volatiles.disable ? tgtMon().volatiles.disable.duration : null;
      if (before !== null || after !== null) {
        events.push(`  [turn ${battle.turn}] RESIDUAL: disable.duration ${before} → ${after}${after === null ? '  (ENDED)' : ''}`);
      }
      return r;
    }
    return realFieldEvent(eventid, targets);
  };

  // Hook random to capture the durationCallback roll.
  let rolled = null;
  const realRandom = battle.random.bind(battle);
  battle.random = function (from, to) {
    const r = realRandom(from, to);
    if (from === 2 && to === 6) { rolled = r; events.push(`  [turn ${battle.turn}] durationCallback random(2,6) = ${r}`); }
    return r;
  };

  const disMove = 'move 1'; // Disable is slot 0 on the disabler
  const p1turn1 = disablerSide === 0 ? 'move 2' : 'move 1';
  const p2turn1 = disablerSide === 0 ? 'move 1' : 'move 2';
  const p1turn2 = disablerSide === 0 ? 'move 1' : 'move 2';
  const p2turn2 = disablerSide === 0 ? 'move 2' : 'move 1';
  const plan = [
    { p1: p1turn1, p2: p2turn1 },
    { p1: p1turn2, p2: p2turn2 },
    ...Array(8).fill({ p1: 'move 2', p2: 'move 2' }),
  ];

  let willMove = null, landed = false;
  let i = 0, safety = 0;
  while (!battle.ended && safety < 20) {
    safety++;
    const rs = battle.requestState;
    if (rs !== 'move' && rs !== 'switch') { await tick(); continue; }
    const entry = plan[Math.min(i, plan.length - 1)];
    const isDisableTurn = i === 1;
    try { if (entry.p1) streams.omniscient.write(`>p1 ${entry.p1}`); } catch (e) {}
    try { if (entry.p2) streams.omniscient.write(`>p2 ${entry.p2}`); } catch (e) {}
    for (let k = 0; k < 18; k++) await tick();
    i++;
    if (isDisableTurn) {
      const dis = battle.sides[disablerSide].active[0], tgt = tgtMon();
      willMove = dis.speed > tgt.speed;
      landed = !!(tgt && tgt.volatiles.disable);
      if (!landed) { try { streams.omniscient.destroy(); } catch (e) {} return null; }
    }
    if (i > 1 && !(tgtMon() && tgtMon().volatiles.disable) && landed) break;
  }
  try { streams.omniscient.destroy(); } catch (e) {}
  return { rolled, willMove, events };
}

async function find(label, p1team, p2team, disablerSide) {
  for (let s = 0; s < 60; s++) {
    const seed = [s * 7 + 1, s * 11 + 3, s * 13 + 5, s * 17 + 7];
    const r = await runOnce(seed, p1team, p2team, disablerSide);
    if (r) {
      console.log(`\n======== ${label} ========  seed=${JSON.stringify(seed)}`);
      console.log(`  rolled=${r.rolled}  willMove(target)=${r.willMove}  (disabler ${r.willMove ? 'FASTER' : 'SLOWER'})`);
      console.log(r.events.join('\n'));
      const expected = r.willMove ? r.rolled - 1 : r.rolled;
      console.log(`  → sim post-onStart stored duration SHOULD be (willMove?rolled-1:rolled) = ${expected}`);
      return r;
    }
  }
  console.log(`\n======== ${label} ========  NO landing found`);
}

async function main() {
  await find('FASTER disabler',
    [mon('Aerodactyl', ['disable', 'rockslide'], { evs: { spe: 252 } })],
    [mon('Snorlax', ['bodyslam', 'blizzard'], { evs: { hp: 252, atk: 128, spd: 128 }, nature: 'Brave', ivs: { ...IV31, spe: 0 } })],
    0);
  await find('SLOWER disabler',
    [mon('Blissey', ['disable', 'softboiled'], { evs: { hp: 252, def: 252 }, nature: 'Brave', ivs: { ...IV31, spe: 0 } })],
    [mon('Aerodactyl', ['rockslide', 'earthquake'], { evs: { spe: 252 } })],
    0);
}
main().catch((e) => { console.error(e.stack || String(e)); process.exit(1); });
