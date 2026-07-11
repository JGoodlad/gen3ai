// probe_disable_full_lifecycle.js — the UNIFIED ground truth: for a FASTER and a SLOWER
// disabler landing, print (a) willMove(target), (b) the raw random(2,6) roll, (c) the
// post-onStart stored duration (hooked at addVolatile return), and (d) the FULL residual
// tick-down sequence to the free-up turn (hooked at each Residual). One probe, no cross-probe
// inference. This is what the port's stored value + its residual handler must jointly match.
'use strict';
const path = require('path');
const PS = path.resolve(__dirname, '../../../deps/pokemon-showdown');
const { BattleStream, getPlayerStreams } = require(path.join(PS, 'dist/sim/battle-stream'));
const { Teams } = require(path.join(PS, 'dist/sim'));
const { Pokemon } = require(path.join(PS, 'dist/sim/pokemon'));

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
  const tgtMon = () => battle.sides[targetSide].active[0];

  let rolled = null, willMove = null, postOnStart = null;
  const realRandom = battle.random.bind(battle);
  battle.random = function (from, to) {
    const r = realRandom(from, to);
    if (from === 2 && to === 6) rolled = r;
    return r;
  };
  const realAdd = Pokemon.prototype.addVolatile;
  Pokemon.prototype.addVolatile = function (status, source, sourceEffect, linkedStatus) {
    const id = (typeof status === 'string') ? this.battle.dex.conditions.get(status).id : status.id;
    if (id === 'disable') {
      willMove = !!this.battle.queue.willMove(this);
      rolled = null;
      const ret = realAdd.call(this, status, source, sourceEffect, linkedStatus);
      postOnStart = this.volatiles.disable ? this.volatiles.disable.duration : null;
      return ret;
    }
    return realAdd.call(this, status, source, sourceEffect, linkedStatus);
  };
  const residualSeq = [];
  const realFieldEvent = battle.fieldEvent.bind(battle);
  battle.fieldEvent = function (eventid, targets) {
    if (eventid === 'Residual') {
      const r = realFieldEvent(eventid, targets);
      const t = tgtMon();
      residualSeq.push(t && t.volatiles.disable ? t.volatiles.disable.duration : null);
      return r;
    }
    return realFieldEvent(eventid, targets);
  };

  const p1turn1 = disablerSide === 0 ? 'move 2' : 'move 1';
  const p2turn1 = disablerSide === 0 ? 'move 1' : 'move 2';
  const p1turn2 = disablerSide === 0 ? 'move 1' : 'move 2';
  const p2turn2 = disablerSide === 0 ? 'move 2' : 'move 1';
  const plan = [{ p1: p1turn1, p2: p2turn1 }, { p1: p1turn2, p2: p2turn2 }, ...Array(8).fill({ p1: 'move 2', p2: 'move 2' })];

  let i = 0, safety = 0, landed = false;
  while (!battle.ended && safety < 20) {
    safety++;
    const rs = battle.requestState;
    if (rs !== 'move' && rs !== 'switch') { await tick(); continue; }
    const entry = plan[Math.min(i, plan.length - 1)];
    try { if (entry.p1) streams.omniscient.write(`>p1 ${entry.p1}`); } catch (e) {}
    try { if (entry.p2) streams.omniscient.write(`>p2 ${entry.p2}`); } catch (e) {}
    for (let k = 0; k < 18; k++) await tick();
    i++;
    if (i === 2 && postOnStart === null) { Pokemon.prototype.addVolatile = realAdd; try { streams.omniscient.destroy(); } catch (e) {} return null; }
    if (i >= 2 && residualSeq.length && residualSeq[residualSeq.length - 1] === null && postOnStart !== null) break;
  }
  Pokemon.prototype.addVolatile = realAdd;
  try { streams.omniscient.destroy(); } catch (e) {}
  // residualSeq includes turn-1's residual (disable not yet present → null); trim to from the
  // disable turn onward. The disable turn is turn 2 → its residual is residualSeq[1].
  const fromDisable = residualSeq.slice(1);
  return { rolled, willMove, postOnStart, fromDisable };
}

async function find(label, p1team, p2team, disablerSide) {
  for (let s = 0; s < 60; s++) {
    const seed = [s * 7 + 1, s * 11 + 3, s * 13 + 5, s * 17 + 7];
    const r = await runOnce(seed, p1team, p2team, disablerSide);
    if (r && r.postOnStart !== null) {
      const freeIdx = r.fromDisable.indexOf(null); // #residuals after the disable turn's own
      console.log(`\n======== ${label} ========  seed=${JSON.stringify(seed)}`);
      console.log(`  willMove(target)=${r.willMove}  rolled(random(2,6))=${r.rolled}`);
      console.log(`  post-onStart stored duration = ${r.postOnStart}  (== rolled${r.postOnStart === r.rolled ? '' : (r.postOnStart === r.rolled - 1 ? '-1' : (r.postOnStart === r.rolled + 1 ? '+1' : '???'))})`);
      console.log(`  residual duration each turn from the disable turn on = [${r.fromDisable.join(', ')}]`);
      console.log(`  → disabled move frees up on residual #${freeIdx} counting the disable turn's own residual as #0`);
      return r;
    }
  }
  console.log(`\n======== ${label} ========  NO landing`);
}

async function main() {
  await find('FASTER disabler (Aerodactyl fast → disables slow Snorlax)',
    [mon('Aerodactyl', ['disable', 'rockslide'], { evs: { spe: 252 } })],
    [mon('Snorlax', ['bodyslam', 'blizzard'], { evs: { hp: 252, atk: 128, spd: 128 }, nature: 'Brave', ivs: { ...IV31, spe: 0 } })],
    0);
  await find('SLOWER disabler (Blissey slow → disables fast Aerodactyl)',
    [mon('Blissey', ['disable', 'softboiled'], { evs: { hp: 252, def: 252 }, nature: 'Brave', ivs: { ...IV31, spe: 0 } })],
    [mon('Aerodactyl', ['rockslide', 'earthquake'], { evs: { spe: 252 } })],
    0);
}
main().catch((e) => { console.error(e.stack || String(e)); process.exit(1); });
