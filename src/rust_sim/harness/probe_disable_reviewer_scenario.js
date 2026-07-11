// probe_disable_reviewer_scenario.js — reproduce the REVIEWERS' exact scenario (seed
// [7,11,13,17], SLOWER disabler Blissey vs Snorlax, rolled=4) and dump the FULL disable
// duration lifecycle: willMove, rolled, post-onStart (addVolatile-return), and every residual
// tick. This settles whether the port's stored value should be rolled+1 (my probes) or rolled
// (the reviewers' claim). The sim is the sole source of truth.
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

async function main() {
  const stream = new BattleStream();
  const streams = getPlayerStreams(stream);
  (async () => { for await (const ch of streams.omniscient) { void ch; } })();
  const seed = [7, 11, 13, 17];
  // The reviewers' scenario: p1 Blissey (slow) uses Disable on p2 Snorlax (faster).
  const p1team = [mon('Blissey', ['disable', 'softboiled'], { evs: { hp: 252, def: 252 }, nature: 'Brave', ivs: { ...IV31, spe: 0 } })];
  const p2team = [mon('Snorlax', ['bodyslam', 'rest'], { evs: { hp: 252, atk: 252 } })];
  streams.omniscient.write(`>start {"formatid":"${FORMAT}","seed":${JSON.stringify(seed)}}`);
  streams.omniscient.write(`>player p1 ${JSON.stringify({ name: 'P1', team: Teams.pack(p1team) })}`);
  streams.omniscient.write(`>player p2 ${JSON.stringify({ name: 'P2', team: Teams.pack(p2team) })}`);
  for (let i = 0; i < 12; i++) await tick();
  const battle = stream.battle;
  const tgt = () => battle.sides[1].active[0]; // Snorlax is the disable target

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
      const willMove = !!this.battle.queue.willMove(this);
      rolled = null;
      const ret = realAdd.call(this, status, source, sourceEffect, linkedStatus);
      const post = this.volatiles.disable ? this.volatiles.disable.duration : null;
      console.log(`  addVolatile(disable): willMove(target)=${willMove}  rolled=${rolled}  post-onStart duration=${post}  (== rolled${post === rolled ? '' : (post === rolled + 1 ? '+1' : (post === rolled - 1 ? '-1' : '???'))})`);
      return ret;
    }
    return realAdd.call(this, status, source, sourceEffect, linkedStatus);
  };
  const realFieldEvent = battle.fieldEvent.bind(battle);
  battle.fieldEvent = function (eventid, targets) {
    if (eventid === 'Residual') {
      const b = tgt() && tgt().volatiles.disable ? tgt().volatiles.disable.duration : null;
      const r = realFieldEvent(eventid, targets);
      const a = tgt() && tgt().volatiles.disable ? tgt().volatiles.disable.duration : null;
      if (b !== null || a !== null) console.log(`  [turn ${battle.turn}] residual: disable ${b} → ${a}${a === null ? '  (FREED — move usable next selection)' : ''}`);
      return r;
    }
    return realFieldEvent(eventid, targets);
  };

  // Plan: turn 1 both attack (Snorlax Body Slam → lastMove slot 0); turn 2 Blissey Disable
  // (slower → willMove(Snorlax) FALSE), Snorlax Rest; then filler.
  const plan = [
    { p1: 'move 2', p2: 'move 1' }, // Blissey Soft-Boiled, Snorlax Body Slam (lastMove=slot0)
    { p1: 'move 1', p2: 'move 2' }, // Blissey Disable (slower), Snorlax Rest
    ...Array(8).fill({ p1: 'move 2', p2: 'move 2' }),
  ];
  let i = 0, safety = 0;
  while (!battle.ended && safety < 16) {
    safety++;
    const rs = battle.requestState;
    if (rs !== 'move' && rs !== 'switch') { await tick(); continue; }
    const entry = plan[Math.min(i, plan.length - 1)];
    console.log(`turn ${i + 1}: p1=${entry.p1} p2=${entry.p2}`);
    try { if (entry.p1) streams.omniscient.write(`>p1 ${entry.p1}`); } catch (e) {}
    try { if (entry.p2) streams.omniscient.write(`>p2 ${entry.p2}`); } catch (e) {}
    for (let k = 0; k < 18; k++) await tick();
    i++;
    const d = tgt() && tgt().volatiles.disable ? tgt().volatiles.disable.duration : null;
    console.log(`     → after turn: Snorlax disable duration = ${d}`);
    if (i >= 3 && d === null) break;
  }
  Pokemon.prototype.addVolatile = realAdd;
  try { streams.omniscient.destroy(); } catch (e) {}
}
main().catch((e) => { console.error(e.stack || String(e)); process.exit(1); });
