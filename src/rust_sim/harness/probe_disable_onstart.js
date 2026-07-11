// probe_disable_onstart.js — DEFINITIVELY nail the disable volatile's post-onStart stored
// duration for a FASTER and a SLOWER disabler, by hooking Pokemon.prototype.addVolatile: read
// queue.willMove(target) right before it runs, then read the resulting volatiles.disable.duration
// right after (which is post-durationCallback + post-onStart). No inference, no interleave.
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

  const events = [];
  let rolled = null;
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
      const willMove = !!this.battle.queue.willMove(this); // `this` is the TARGET being disabled
      rolled = null;
      const ret = realAdd.call(this, status, source, sourceEffect, linkedStatus);
      const durAfter = this.volatiles.disable ? this.volatiles.disable.duration : null;
      events.push(`  [turn ${this.battle.turn}] addVolatile(disable) on target: willMove(target)=${willMove}  rolled=${rolled}  → duration AFTER onStart = ${durAfter}  (post-onStart == rolled${durAfter === rolled ? '' : (durAfter === rolled - 1 ? '-1' : (durAfter === rolled + 1 ? '+1' : '???'))})`);
      return ret;
    }
    return realAdd.call(this, status, source, sourceEffect, linkedStatus);
  };

  const p1turn1 = disablerSide === 0 ? 'move 2' : 'move 1';
  const p2turn1 = disablerSide === 0 ? 'move 1' : 'move 2';
  const p1turn2 = disablerSide === 0 ? 'move 1' : 'move 2';
  const p2turn2 = disablerSide === 0 ? 'move 2' : 'move 1';
  const plan = [{ p1: p1turn1, p2: p2turn1 }, { p1: p1turn2, p2: p2turn2 }, ...Array(6).fill({ p1: 'move 2', p2: 'move 2' })];

  let i = 0, safety = 0, landed = false;
  while (!battle.ended && safety < 16) {
    safety++;
    const rs = battle.requestState;
    if (rs !== 'move' && rs !== 'switch') { await tick(); continue; }
    const entry = plan[Math.min(i, plan.length - 1)];
    try { if (entry.p1) streams.omniscient.write(`>p1 ${entry.p1}`); } catch (e) {}
    try { if (entry.p2) streams.omniscient.write(`>p2 ${entry.p2}`); } catch (e) {}
    for (let k = 0; k < 18; k++) await tick();
    i++;
    if (i === 2) {
      const tgt = battle.sides[targetSide].active[0];
      landed = !!(tgt && tgt.volatiles.disable);
      Pokemon.prototype.addVolatile = realAdd;
      if (!landed) { try { streams.omniscient.destroy(); } catch (e) {} return null; }
      break;
    }
  }
  Pokemon.prototype.addVolatile = realAdd;
  try { streams.omniscient.destroy(); } catch (e) {}
  return { rolled, events };
}

async function find(label, p1team, p2team, disablerSide) {
  for (let s = 0; s < 60; s++) {
    const seed = [s * 7 + 1, s * 11 + 3, s * 13 + 5, s * 17 + 7];
    const r = await runOnce(seed, p1team, p2team, disablerSide);
    if (r && r.events.length) {
      console.log(`\n======== ${label} ========  seed=${JSON.stringify(seed)}`);
      console.log(r.events.join('\n'));
      return r;
    }
  }
  console.log(`\n======== ${label} ========  NO landing`);
}

async function main() {
  await find('FASTER disabler (Aerodactyl vs slow Snorlax)',
    [mon('Aerodactyl', ['disable', 'rockslide'], { evs: { spe: 252 } })],
    [mon('Snorlax', ['bodyslam', 'blizzard'], { evs: { hp: 252, atk: 128, spd: 128 }, nature: 'Brave', ivs: { ...IV31, spe: 0 } })],
    0);
  await find('SLOWER disabler (Blissey vs fast Aerodactyl)',
    [mon('Blissey', ['disable', 'softboiled'], { evs: { hp: 252, def: 252 }, nature: 'Brave', ivs: { ...IV31, spe: 0 } })],
    [mon('Aerodactyl', ['rockslide', 'earthquake'], { evs: { spe: 252 } })],
    0);
}
main().catch((e) => { console.error(e.stack || String(e)); process.exit(1); });
